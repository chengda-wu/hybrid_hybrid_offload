"""SGLang KV cache backend — wraps the real SGLang RadixCache."""

from __future__ import annotations

from array import array
from typing import Any

from simulator.config.model_config import KVBackendConfig
from simulator.config.sglang_config import SGLangConfig
from simulator.kv_cache.base import KVBackend


# ---------------------------------------------------------------------------
# Mock allocator — satisfies the BaseTokenToKVPoolAllocator protocol
# ---------------------------------------------------------------------------


class MockTokenToKVPoolAllocator:
    """Minimal mock allocator for standalone RadixCache usage.

    Allocates integer token indices from a flat pool.  Supports free()
    for correct cache eviction behavior.  Optional offset for multi-pool
    setups to avoid index namespace collisions.
    """

    def __init__(self, total_tokens: int, offset: int = 0,
                 on_free: "callable | None" = None):
        self._total = total_tokens
        self._offset = offset
        self._next_idx = offset
        self._free_list: list[int] = []
        self._on_free = on_free

    def allocate(self, num_tokens: int):
        """Allocate *num_tokens* token indices, reusing freed ones first.

        Returns None on OOM, matching real TokenToKVPoolAllocator.alloc
        (allocator/token.py:60).  Caller (allocate_slots) handles evict+retry.
        """
        import torch

        if num_tokens <= 0:
            return torch.tensor([], dtype=torch.int64)
        if num_tokens <= len(self._free_list):
            indices = self._free_list[-num_tokens:]
            del self._free_list[-num_tokens:]
            return torch.tensor(indices, dtype=torch.int64)
        start = self._next_idx
        self._next_idx += num_tokens
        if self._next_idx > self._offset + self._total:
            self._next_idx = start  # rollback — matches real alloc behavior
            return None
        return torch.arange(start, start + num_tokens, dtype=torch.int64)

    def free(self, indices) -> None:
        """Return token indices to the free pool."""
        if hasattr(indices, "tolist"):
            vals = indices.tolist()
        elif isinstance(indices, list):
            vals = list(indices)
        else:
            return
        self._free_list.extend(vals)
        if self._on_free is not None:
            self._on_free(len(vals))

    @property
    def total_tokens(self) -> int:
        return self._total

    def available_size(self) -> int:
        return self._total - self._next_idx + len(self._free_list)

    def evictable_size(self) -> int:
        return 0

    def get_physical_pool_id(self) -> str:
        return "mock"

    @property
    def device(self):
        import torch

        return torch.device("cpu")


# ---------------------------------------------------------------------------
# SGLang backend
# ---------------------------------------------------------------------------


class SGLangBackend(KVBackend):
    """Wraps the real SGLang RadixCache for token-level simulation."""

    def __init__(self, backend_config: KVBackendConfig):
        sglang_cfg = SGLangConfig.from_backend_config(backend_config)

        self._backend_config = backend_config
        self._page_size = sglang_cfg.page_size
        # Base per-position caps from real SGLang's DSV4PoolConfigurator (one
        # full slot / one swa slot per token position).  Stored for diagnostics
        # and tests; the layer-slot caps below are derived from these.
        self._full_token = sglang_cfg.full_token
        self._swa_token = sglang_cfg.swa_token

        # Single RadixCache allocator (one index per token position).  Real
        # SGLang has two allocator index spaces (full_attn_allocator size=full,
        # swa_attn_allocator size=swa); we collapse to one flat dispenser sized
        # to their sum.  This is a generous, NON-binding cap — the per-pool
        # check in allocate_slots is the real gate; the flat allocator only
        # hands out indices and never independently OOMs.
        total_slots = sglang_cfg.full_token + sglang_cfg.swa_token
        # on_free fires for every flat-index free (rejected tail in
        # free_rejected_slots, unaligned tail in free(), tree nodes in evict()).
        # It deducts the FULL pool ONLY — NOT SWA.  SWA is decoupled: out-of-
        # window SWA is returned by _reclaim_swa_out_of_window, in-window SWA
        # at request finish (free) / rejection (free_rejected_slots).  This
        # avoids the double-deduction where evict() of a tree node whose SWA
        # was already reclaimed would otherwise subtract SWA a second time
        # (real SWARadixCache prevents this via tombstoning reclaimed nodes
        # before evict, swa_radix_cache.py:615 — we use plain RadixCache with
        # no tombstones, so SWA must not flow through on_free).
        self._mock_allocator = MockTokenToKVPoolAllocator(
            total_slots,
            on_free=lambda n: self._deduct_pool_used(n),
        )

        # Two-pool model [swa, full] matching real SGLang's get_max_pool_usage
        # (pool_stats_observer.py:64-71: max(full, swa)).  c4/c128 are NOT
        # independent pools — they are sub-allocated in lockstep from the
        # unified full pool (allocator/swa.py:20-78 SWATokenToKVPoolAllocator
        # has exactly two sub-allocators: full + swa).  Charging c4+c128
        # together against one ``full`` cap makes the full pool bind at
        # ``full_token`` positions, matching real SGLang; the previous
        # 3-independent-pool model bound c128 at full/128 (128× too early).
        arch = backend_config.model_arch
        self._swa_per_tok = arch.num_layers
        self._c4_per_tok = (sum(1 for cr in (arch.compress_ratios or []) if cr == 4) * 2)
        self._c128_per_tok = sum(1 for cr in (arch.compress_ratios or []) if cr == 128)
        self._full_per_tok = self._c4_per_tok + self._c128_per_tok
        # Caps in per-layer slot equivalents (cap = base_positions × per_tok).
        # Ratio (used·per_tok)/cap is unit-invariant, so usage = max(swa, full)
        # mirrors real SGLang's token-usage ratios.
        self._pool_caps = [
            sglang_cfg.swa_token * self._swa_per_tok,
            sglang_cfg.full_token * self._full_per_tok,
        ]
        self._pool_used = [0, 0]  # [swa, full] slots used
        # Peak per-pool utilization observed over the run (high-water mark of
        # _pool_used[i] / _pool_caps[i]).  End-state usage is ~0 (requests free
        # on finish), so the peak is the informative number for diagnosing
        # which pool bottlenecks first (SWA ring vs full KV).
        self._pool_peak: list[int] = [0, 0]
        self._pool_names = ["swa", "full"]
        self._pool_per_tok = [
            self._swa_per_tok, self._full_per_tok,
        ]
        # SWA sliding window (HF sliding_window; 128 for DSV4).  Real SGLang
        # frees SWA slots once they leave the window (see
        # _reclaim_swa_out_of_window).  Derive from arch, not a literal.
        sw = arch.sliding_window
        self._sliding_window = sw if sw and sw > 0 else 128

        # Diagnostic from the last failed allocate_slots (None when last call
        # succeeded).  SGLang fails on per-pool capacity, not total-pool free,
        # so this records which pool(s) were over budget for a clearer OOM
        # message.  Used only by the scheduler's error path.
        self.last_alloc_failure: dict | None = None

        from sglang.srt.mem_cache.radix_cache import RadixCache

        self._cache = RadixCache.create_simulated(
            disable=False,
            mock_allocator=self._mock_allocator,
            page_size=self._page_size,
            enable_kv_cache_events=False,
        )

    # ---- KVBackend interface ----

    def create_request(
        self, request_id: str, prompt_token_ids: list[int], max_tokens: int
    ) -> "SGLangSimRequest":
        return SGLangSimRequest(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            max_tokens=max_tokens,
        )

    def register_request(self, sim_req: "SGLangSimRequest") -> None:
        """No-op for SGLang — insertion happens in allocate_slots."""
        pass


    def get_computed_blocks(self, sim_req: "SGLangSimRequest") -> tuple[Any, int]:
        """Match prefix via RadixCache.match_prefix.

        Returns (device_indices_tensor, num_matched_tokens).
        """
        # Only called during prefill (output is empty), matching real SGLang
        # where match_prefix uses prompt-only tokens.
        all_tokens = array("q", sim_req.prompt_token_ids)
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        result = self._cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=all_tokens))
        )
        num_matched = len(result.device_indices)
        return result.device_indices, num_matched

    def allocate_slots(
        self,
        sim_req: "SGLangSimRequest",
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Any | None = None,
    ) -> Any | None:
        """Allocate token slots from the mock pool.  No radix tree insert.

        Insert happens in sync_state() after accepted tokens are known,
        mirroring real SGLang's cache_unfinished_req (radix_cache.py:494).
        """
        import torch

        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        # Prepend matched-prefix indices so sync_state can build
        # prefix + new = full sequence (real SGLang req_to_token_pool
        # holds the entire row).  Order must be prefix first, new second.
        if new_computed_blocks is not None and len(new_computed_blocks) > 0:
            sim_req._allocated_indices.append(new_computed_blocks)

        to_alloc = num_new_tokens
        if to_alloc <= 0:
            return torch.tensor([], dtype=torch.int64)

        # Two-pool capacity check (swa, full).  c4+c128 are charged together
        # against the unified full pool (they are sub-allocated in lockstep in
        # real SGLang and never independently bind).  Each token consumes
        # swa_layers SWA slots + (c4_layers*2 + c128_layers) full slots.
        swa_need = to_alloc * self._swa_per_tok
        full_need = to_alloc * self._full_per_tok
        needs = [swa_need, full_need]

        def _over_budget() -> list[int]:
            return [
                i for i in range(2)
                if self._pool_used[i] + needs[i] > self._pool_caps[i]
            ]

        # If any pool is over budget, trigger RadixCache eviction.
        # evict() expects token count, not slot count.
        if _over_budget():
            # evict() frees whole tree nodes → on_free deducts the FULL pool
            # only (see _deduct_pool_used invariant).  SWA is NOT deducted
            # here on purpose: evicted nodes' out-of-window SWA was already
            # reclaimed by _reclaim_swa_out_of_window, and deduplicating that
            # via on_free would double-count (no tombstones, unlike real
            # SWARadixCache).
            self._cache.evict(EvictParams(num_tokens=to_alloc))
            # Re-check: eviction may have freed enough
            over = _over_budget()
            if over:
                self.last_alloc_failure = {
                    "over_budget_pools": [
                        {
                            "name": self._pool_names[i],
                            "used_slots": self._pool_used[i],
                            "need_slots": needs[i],
                            "cap_slots": self._pool_caps[i],
                            "per_token_slots": self._pool_per_tok[i],
                        }
                        for i in over
                    ],
                    "alloc_tokens": to_alloc,
                }
                return None

        # RadixCache token-level allocation (one index per token).
        # Flat pool check is redundant with per-pool cap check above;
        # the flat pool has ample capacity (15.6M slots).
        new_indices = self._mock_allocator.allocate(to_alloc)
        if new_indices is None:
            self.last_alloc_failure = {
                "flat_pool_oom": True,
                "alloc_tokens": to_alloc,
                "available_tokens": self._mock_allocator.available_size(),
            }
            return None

        # Track per-pool slot usage
        self._pool_used[0] += swa_need
        self._pool_used[1] += full_need
        # Update peak (allocs are the only place usage rises; frees/reclaims
        # only lower it, so checking here captures the high-water mark).
        for i in range(2):
            if self._pool_used[i] > self._pool_peak[i]:
                self._pool_peak[i] = self._pool_used[i]
        # Bill this request's SWA charged-token count (prefix-hit tokens are
        # not billed — they were a cache hit, not a fresh allocation).
        sim_req.swa_charged_tokens += to_alloc

        sim_req._allocated_indices.append(new_indices)
        self.last_alloc_failure = None
        return new_indices

    def _deduct_pool_used(self, num_tokens: int) -> None:
        """Decrement the full pool when flat indices are freed.

        SWA is intentionally NOT deducted here.  The full pool (c4+c128) is
        full-retention (no sliding-window reclamation), so a flat-index free
        corresponds one-to-one to a full-slot return — whether from rejected-
        spec tail, unaligned finish tail, or evict() of a cached tree node.
        SWA differs: out-of-window SWA is returned by
        _reclaim_swa_out_of_window and in-window SWA at finish/rejection (see
        free / free_rejected_slots).  Routing SWA through this callback would
        double-deduct the reclaimed portion when evict() later frees the same
        tree node (real SWARadixCache avoids this via tombstones; we have none,
        so SWA stays out of on_free).
        """
        if num_tokens <= 0:
            return
        self._pool_used[1] = max(0, self._pool_used[1] - num_tokens * self._full_per_tok)

    def _reclaim_swa_out_of_window(self, sim_req: "SGLangSimRequest") -> None:
        """Return SWA slots outside the sliding window to the SWA sub-pool.

        Mirrors real SGLang ``free_swa_out_of_window_slots`` (common.py:68),
        called every decode step via ``ScheduleBatch._evict_swa``
        (schedule_batch.py:2924) on DSV4's SWARadixCache + separate SWA
        sub-pool.  Without this, ``_pool_used[0]`` grows with total sequence
        length and OOMs far too early: a single 1100-token request bills
        47300 SWA slots (1100×43) vs the real ~window-bounded footprint.

        Cursor formula (common.py:96-105, drop_page_margin=False for radix):
            evict_threshold = pre_len - sliding_window - page_size
            new_cursor = max(old_cursor, page_align_floor(threshold))
        where ``pre_len = charged - 1`` matches ``req.seqlen - 1`` on the
        request's own positions.  Only ``_pool_used[0]`` (SWA) is decremented
        — real ``free_swa`` returns to the SWA sub-pool only, NOT the
        full/C4/C128 pools, so the mock flat allocator is untouched (those
        token-indices stay allocated to the other pools).

        The cursor is on the request's *charged* token positions (cumulative
        new tokens billed: prefill num_new + decode 1+K − rejected/beyond),
        not full-sequence positions.  A cache-hit prefill is not billed for
        the shared prefix, so basing the cursor on charged tokens avoids
        over-deducting the (unbilled) prefix and driving the request's SWA
        contribution negative.

        Simplifications vs real SGLang (documented):
        - Reclaims every ``sync_state``; real gates on ``eviction_interval``
          + ``decode_batch_idx >= 1``.  Steady-state bound is identical.
        - No ``cache_protected_len`` floor: the charged-token cursor already
          keeps the live window untouched (threshold < charged − window).
          Real SGLang floors ``new_cursor`` at ``cache_protected_len``
          (common.py:84) to avoid reclaiming tree-held prefix SWA per-step;
          the charged-token cursor achieves the same effect on the request's
          own (post-prefix) region, because real's effective reclaim range
          ``[cache_protected_len, seqlen)`` == the charged region.  So both
          reclaim the request's out-of-window *new* tokens at the same point
          (charged > sliding_window + page_size + 1).

        Tree-held prefix SWA is intentionally NOT modeled.  Real SGLang's SWA
        pool also holds the in-window SWA of cached radix prefix nodes
        (~sliding_window×43 per node, reclaimed only via SWA-LRU tombstoning,
        not per-step).  The simulator omits this term because: (a) it's
        immaterial — ~0.1% of the SWA cap per cached prefix, and SWA is a tight
        ring (0.1·full) that binds only under high concurrency with long
        in-window tails, not from the omitted prefix term; (b) modeling
        it correctly requires deduplicating shared prefixes across requests
        (real SGLang's SWARadixCache tombstone + lock_ref machinery), which
        the plain RadixCache used here has no analogue for.  Naively billing
        each request's cache-hit prefix into ``_pool_used[0]`` (as a
        full-sequence cursor would) DOUBLE-COUNTS: N requests sharing one
        prefix would add N×prefix×43 instead of the real 1×prefix×43,
        producing false OOM.  The charged-token design is self-consistent:
        bill only what a request newly allocates, reclaim only what it billed.
        """
        charged = sim_req.swa_charged_tokens
        pre_len = charged - 1
        threshold = pre_len - self._sliding_window - self._page_size
        if threshold <= 0:
            return
        threshold = (threshold // self._page_size) * self._page_size
        new_cursor = max(sim_req.swa_evicted_charged, threshold)
        delta = new_cursor - sim_req.swa_evicted_charged
        if delta > 0:
            self._pool_used[0] = max(
                0, self._pool_used[0] - delta * self._swa_per_tok
            )
            sim_req.swa_evicted_charged = new_cursor

    def set_spec_tokens(
        self, sim_req: "SGLangSimRequest", tokens: list[int]
    ) -> None:
        sim_req.spec_token_ids = tokens

    def sync_state(
        self, sim_req: "SGLangSimRequest", output_token_ids: list[int]
    ) -> None:
        """Insert page-aligned prefix into the radix tree.

        Mirrors real SGLang's cache_unfinished_req (radix_cache.py:494-515):
        - key = RadixKey(prompt+output).page_aligned(page_size)
        - values = kv_indices[:len(key)]  (insert only when value >= key)
        - Unaligned tail is NOT freed — kept for next step (matches
          real SGLang prefix_indices, radix_cache.py:542-544).
        """
        import torch

        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        sim_req.output_token_ids = list(output_token_ids)

        all_tokens = array(
            "q", sim_req.prompt_token_ids + sim_req.output_token_ids
        )
        key = RadixKey(token_ids=all_tokens).page_aligned(self._page_size)
        key_len = len(key)

        flat_indices = torch.cat(
            [t for t in sim_req._allocated_indices if len(t) > 0]
        ) if sim_req._allocated_indices else torch.tensor([], dtype=torch.int64)

        # Collapse the append-only log into one tensor so the next decode step's
        # torch.cat is O(1) instead of O(n) over a list that grows by one entry
        # per decode step.  Without this, long decodes (~1K+ steps) pay an O(n²)
        # cat cost in sync_state — the only compaction site was free_rejected_slots,
        # which fires on rejection and rarely at high acceptance rates.  The
        # values tensor below already clones the prefix we insert, so reusing
        # flat_indices here as the single cached entry is safe and exact.
        if len(sim_req._allocated_indices) > 1:
            sim_req._allocated_indices = (
                [flat_indices] if len(flat_indices) > 0 else []
            )

        # flat >= key_len always holds: flat == total_tokens and
        # key_len = floor(total_tokens/page_size)*page_size <= total_tokens.
        assert len(flat_indices) >= key_len, (
            f"flat({len(flat_indices)}) < key_len({key_len}); "
            f"_allocated_indices out of sync with output"
        )
        values = flat_indices[:key_len].clone()
        result = self._cache.insert(InsertParams(key=key, value=values))
        # Lock the new prefix so evict() won't free it while the request
        # is active (matches real SGLang cache_unfinished_req, radix_cache.py:536-537).
        if sim_req._last_node is not None:
            self._cache.dec_lock_ref(sim_req._last_node)
        if result.last_device_node is not None:
            self._cache.inc_lock_ref(result.last_device_node)
            sim_req._last_node = result.last_device_node

        # Reclaim SWA slots that left the sliding window this step (mirrors
        # real SGLang _evict_swa → free_swa_out_of_window_slots).  Done after
        # the radix insert so charged_tokens reflects this step's new tokens.
        self._reclaim_swa_out_of_window(sim_req)

    def free_rejected_slots(
        self, sim_req: "SGLangSimRequest", num_rejected: int
    ) -> None:
        """Free rejected spec token slots from the tail of allocated indices.

        Requires: rejected draft slots are the LAST num_rejected entries
        in _allocated_indices.  This holds because each decode step calls
        allocate_slots exactly once with 1+K tokens appended at the tail.
        If future code adds multiple allocs per step, this must be changed
        to track per-step segment bounds.
        """
        import torch

        if num_rejected <= 0:
            return
        flat = torch.cat(
            [t for t in sim_req._allocated_indices if len(t) > 0]
        ) if sim_req._allocated_indices else torch.tensor([], dtype=torch.int64)
        if len(flat) < num_rejected:
            # Should be unreachable: each decode step appends 1+K tokens and
            # rejects ≤ K, so the tail always has enough to free.  If this
            # fires, _allocated_indices is out of sync with the scheduler's
            # rejection count (e.g. a future multi-segment-per-step change) and
            # the rejected slots would silently leak — fail loudly instead of
            # leaking both the flat indices and their full/SWA pool slots.
            raise RuntimeError(
                f"free_rejected_slots: only {len(flat)} allocated indices but "
                f"asked to free {num_rejected} for request {sim_req.request_id}. "
                f"The tail-assumption contract is broken (see docstring) — "
                f"rejecting now would leak slots."
            )
        # Reaching here guarantees len(flat) >= num_rejected (the guard above
        # raised otherwise), so the tail free + keep_len are safe.  Contract:
        # rejected slots are the global tail of the flattened allocation log,
        # which holds only because each decode step calls allocate_slots once
        # (1+K tokens appended at the tail) and accepted tokens are never freed
        # mid-request.  If a future change adds multi-segment per-step
        # allocation, the guard above must switch to per-step segment bounds.
        self._mock_allocator.free(flat[-num_rejected:])
        # on_free deducts the FULL pool only (see _deduct_pool_used invariant);
        # SWA for these rejected in-window tokens is deducted explicitly below.
        # Remove freed indices from _allocated_indices
        keep_len = len(flat) - num_rejected
        if keep_len > 0:
            sim_req._allocated_indices = [flat[:keep_len].clone()]
        else:
            sim_req._allocated_indices = []
        # Rejected/beyond tokens were billed to swa_charged_tokens (and
        # _pool_used[0]) in allocate_slots; drop them so the SWA reclaim
        # cursor stays consistent with actually-held tokens.  on_free now
        # deducts only the full pool, so deduct SWA explicitly here — the
        # rejected tail is in-window (just allocated this step, reclaim
        # hasn't touched it), so its full SWA slots are returned.
        sim_req.swa_charged_tokens = max(
            0, sim_req.swa_charged_tokens - num_rejected
        )
        self._pool_used[0] = max(
            0, self._pool_used[0] - num_rejected * self._swa_per_tok
        )

    def free(self, sim_req: "SGLangSimRequest") -> None:
        """Free unaligned tail indices and release lock_ref.

        Page-aligned indices are freed lazily by evict() — tail indices
        are NOT in the tree, so they must be freed explicitly here.
        Lock is released so evict() can reclaim the request's tree nodes
        (matches real SGLang cache_finished_req, radix_cache.py:483).
        """
        import torch
        from array import array
        from sglang.srt.mem_cache.radix_cache import RadixKey

        # Release lock on prefix node
        if sim_req._last_node is not None:
            self._cache.dec_lock_ref(sim_req._last_node)
            sim_req._last_node = None

        # Free unaligned tail (on_free deducts the full pool only — NOT SWA).
        all_tokens = array(
            "q", sim_req.prompt_token_ids + sim_req.output_token_ids
        )
        key_len = len(RadixKey(token_ids=all_tokens).page_aligned(self._page_size))

        flat = torch.cat(
            [t for t in sim_req._allocated_indices if len(t) > 0]
        ) if sim_req._allocated_indices else torch.tensor([], dtype=torch.int64)

        if len(flat) > key_len:
            tail = flat[key_len:]
            # on_free deducts the FULL pool only (see _deduct_pool_used
            # invariant); the in-window SWA for this finish is returned
            # explicitly below.
            self._mock_allocator.free(tail)
        # Return the request's remaining (in-window) SWA slots.  Out-of-window
        # SWA was already returned by _reclaim_swa_out_of_window (tracked by
        # swa_evicted_charged); the in-window remainder is returned here, once.
        # This is the counterpart to routing SWA out of on_free: without it the
        # in-window SWA would leak.  (Real SGLang returns in-window SWA at
        # evict() of the cached node, not at finish — we finish-early here
        # because plain RadixCache has no tombstone to prevent evict
        # double-counting.  Conservative: under-counts SWA between finish and
        # evict for finished-but-still-cached requests, which the SWA pool's
        # large headroom absorbs.)
        in_window = sim_req.swa_charged_tokens - sim_req.swa_evicted_charged
        if in_window > 0:
            self._pool_used[0] = max(
                0, self._pool_used[0] - in_window * self._swa_per_tok
            )
        sim_req.swa_charged_tokens = 0
        sim_req.swa_evicted_charged = 0
        sim_req._allocated_indices = []

    @property
    def usage(self) -> float:
        # Bottleneck-pool utilization = max across pools, matching real SGLang's
        # get_max_pool_usage() (pool_stats_observer.py:64-71: max(full, swa,
        # mamba)).  An average would mask the binding pool: SWA 95% / full 30%
        # averages to 62% (looks fine) but the SWA pool is about to OOM.  Max
        # surfaces that pressure — it is what real SGLang reports to Prometheus
        # and uses for admission throttling.  Per-pool detail remains available
        # via pool_usage_detail() / pool_peak_detail().
        ratios = [r for _, r in self.pool_usage_detail()]
        return max(ratios) if ratios else 0.0

    def pool_usage_detail(self) -> list[tuple[str, float]]:
        # Per-pool utilization (swa, full).  Caps differ by orders of magnitude
        # (SWA ring-bounded at 0.1·full vs full KV), so the per-pool numbers are
        # more informative than the aggregate ``usage`` — e.g. SWA nearing 1.0
        # while full is near 0 signals an SWA-pool bottleneck.
        detail = []
        for name, used, cap in zip(
            self._pool_names, self._pool_used, self._pool_caps
        ):
            detail.append((name, used / cap if cap > 0 else 0.0))
        return detail

    def pool_peak_detail(self) -> list[tuple[str, float]] | None:
        # Peak per-pool utilization over the run.  More useful than the
        # instantaneous pool_usage_detail at end-of-run (which is ~0 after
        # requests free) for diagnosing which pool nearly OOM'd.
        detail = []
        for name, peak, cap in zip(
            self._pool_names, self._pool_peak, self._pool_caps
        ):
            detail.append((name, peak / cap if cap > 0 else 0.0))
        return detail

    @property
    def num_free_blocks(self) -> int:
        return self._mock_allocator.available_size()

    @property
    def total_blocks(self) -> int:
        return self._mock_allocator.total_tokens

    @property
    def total_bytes(self) -> int:
        """Total KV cache bytes — matches real SGLang DSV4PoolConfigurator.

        Uses real SGLang DSV4PoolConfigurator formulas:
          swa_tokens  = align(full_tokens * swa_ratio, page_size)
          swa_slots   = swa_tokens // window_size
          c4/c128 KV  = full_tokens // compress_ratio
          c4/c128 state = swa_slots * ring_size * state_bytes_per_token
          indexer KV  = c4 KV sized, 132 B/token
          indexer state = swa_slots * ring * state_bytes_per_token(2048)

        ``full_tokens`` / ``swa_tokens`` reuse the configurator-derived base caps
        computed in ``__init__`` (``self._full_token`` / ``self._swa_token``) so
        this byte ledger stays consistent with the token-pool caps that drive
        OOM — no second, hand-rolled ratio derivation.
        """
        arch = self._backend_config.model_arch
        page_size = self._page_size
        swa_page_size = self._sliding_window  # cfg.window_size (pool_configurator.py:512)

        # Ring sizes — import from SGLang (module-level, no server_args needed).
        # Spec mode doubles ring sizes (pool_configurator.py:514).  Online c128
        # collapses ring_size to 1 (get_compress_state_ring_size, memory_pool.py:34).
        is_spec = self._backend_config.num_spec_tokens > 0
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            get_compress_state_ring_size,
        )
        c4_ring = get_compress_state_ring_size(4, is_speculative=is_spec)
        c128_ring = get_compress_state_ring_size(128, is_speculative=is_spec)

        # Per-token byte costs from SGLang (pool_configurator.py:578-594).
        # kv_bytes = qk_nope + qk_rope*2 + 8 (fp8_ds_mla UE8M0 layout) =
        # head_size + qk_rope + 8 (584 for DSV4).  Derived from the arch so a
        # future MLA model with a different RoPE head dim is priced correctly.
        kv_bytes = arch.kv_bytes_per_token
        # dtype sizes from _get_dsv4_compress_state_dtype_sizes()
        # (pool_configurator.py:74-88, default float32→4)
        from sglang.srt.model_executor.pool_configurator import (
            _get_dsv4_compress_state_dtype_sizes,
        )
        c4_dt, c128_dt = _get_dsv4_compress_state_dtype_sizes()
        # last_dim = 2*(1+overlap)*head_dim (deepseek_v4_compress_state.py:125),
        # attn_head_dim = qk_nope + qk_rope = arch.head_size (pool_configurator.py:585).
        # c4 overlap=True (ring>1), c128 overlap=False (ring>1).  Both use
        # attn_head_dim, NOT the state_dim in KVGroupInfo (2048/1024) — a
        # different quantity.  Matches pool_configurator.py:589-596.
        # Online compress (SGLANG_OPT_USE_ONLINE_COMPRESS) changes c128 from
        # (kv, score) = 2*head_dim to (max, sum, kv) = 3*head_dim per slot
        # (pool_configurator.py:593-596).  Read the real flag so total_bytes
        # tracks SGLang when the user enables it (otherwise c128 state is
        # undercounted by ~50%).
        from sglang.srt.environ import envs
        c128_online = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        head_dim = arch.head_size
        indexer_hd = arch.indexer_head_dim
        c4_state_bytes = 2 * 2 * head_dim * c4_dt      # last_dim = 2*2*head_dim
        c128_state_bytes = (
            (3 if c128_online else 2 * 1) * head_dim * c128_dt
        )
        idx_state_bytes = 2 * 2 * indexer_hd * c4_dt   # indexer uses c4 dtype

        # Base token caps come from the real configurator (self._full_token /
        # self._swa_token), already spec-(T+D)/T-scaled and page-aligned in
        # __init__ — reuse them verbatim so the byte ledger matches the OOM-
        # driving pool caps exactly (no second ratio derivation to drift).
        full_tokens = self._full_token
        swa_tokens = self._swa_token
        swa_slots = swa_tokens // swa_page_size

        def _sglang_page_count(size: int, page: int) -> int:
            # SGLang's page-count idiom (deepseek_v4_memory_pool._create_buffer,
            # pool_configurator): ``(size + page + 1) // page``.  This is NOT
            # a standard ceil_div — the extra ``+1`` over ``ceil_div`` adds one
            # guard page when size is page-aligned (e.g. _sglang_page_count(256,
            # 256) = 2, not 1).  Named ``_sglang_page_count`` (not ``ceil_div``)
            # so callers know the guard page is included.
            return (size + page + 1) // page

        def _pad576(raw: int) -> int:
            return -(-raw // 576) * 576  # ceil to 576 for create_buffer

        total = 0
        for info in self._backend_config.build_kv_cache_groups():
            if info.name == "swa":
                # SWA: pad per-page (swa_page_size * kv_bytes), then × slots × layers
                padded_page = _pad576(swa_page_size * kv_bytes)
                swa_pages = _sglang_page_count(swa_tokens, swa_page_size)
                group_bytes = swa_pages * padded_page * info.layer_count
            elif info.name == "c4_compressor":
                # CompressStatePool._size = ceil(size + ring + 1, ratio) * ratio
                # (deepseek_v4_compress_state.py:117-123)
                raw = swa_slots * c4_ring
                state_tokens = -(-(raw + c4_ring + 1) // 4) * 4
                group_bytes = state_tokens * c4_state_bytes * info.layer_count
            elif info.name == "c128_compressor":
                # CompressStatePool sizing (deepseek_v4_compress_state.py:106-124):
                #   offline: _size = ceil(size + ring + 1, ratio) * ratio  (pad to ratio)
                #   online:  _logical_size = size + ring + 1  (NO ratio padding);
                #            _size = _logical_size * (1 + online_mtp_max_draft_tokens)
                #            (online_mtp only when SGLANG_EXPERIMENTAL_ONLINE_C128_MTP,
                #             pool_configurator.py:601-602,515-517).  c4 is always
                #             offline (online flag is ratio==128 only, memory_pool.py:813).
                raw = swa_slots * c128_ring
                if c128_online:
                    state_tokens = raw + c128_ring + 1
                    if envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
                        # max_speculative_num_draft_tokens (configurator:515-517);
                        # the mock server_args passes None → 0 in the configurator,
                        # so this is 0 unless a draft-token count is configured.
                        online_mtp = self._backend_config.num_spec_tokens or 0
                        state_tokens *= (1 + online_mtp)
                else:
                    state_tokens = -(-(raw + c128_ring + 1) // 128) * 128
                group_bytes = state_tokens * c128_state_bytes * info.layer_count
            elif info.name == "c4_indexer":
                # DeepSeekV4IndexerPool._create_buffer: no 576 padding.
                # pages = ceil_div(size, page_size)
                # c4_tok = full_tokens // 4.  Real SGLang divides by
                # (4 * c4_shrink_factor) when HiSparse is configured
                # (pool_configurator.py:604,627), but c4_shrink_factor defaults
                # to 1 (pool_configurator.py:525) and the simulator does not
                # expose a HiSparse config, so //4 is correct for the default
                # DSV4 path.  Add a c4_shrink_factor knob here if HiSparse
                # support is needed.
                c4_tok = full_tokens // 4
                c4_page_tokens = info.block_size // 4  # 64
                kv_pages = _sglang_page_count(c4_tok, c4_page_tokens)
                kv = info.layer_count * kv_pages * info.page_bytes
                # Indexer state: same size+ring+1 rounding as compressor state
                raw = swa_slots * c4_ring
                state_tok = -(-(raw + c4_ring + 1) // 4) * 4
                state = state_tok * idx_state_bytes * info.layer_count
                group_bytes = kv + state
            else:
                # Main KV (c4_mla / c128_mla / full): ceil_div(size, page) pages.
                # storage_bs = page_bytes // per_token_bytes (tokens per storage
                # block).  For DSV4 MLA groups per_token_bytes = kv_bytes
                # (head_size+qk_rope+8 = 584), so page_bytes (64*kv_bytes or
                # 2*kv_bytes) divides evenly.  Derive per_token_bytes from the
                # architecture rather than hardcoding so a future non-MLA "full"
                # group (page_bytes = 2*bs*kv_heads*head_size*dtype, not a
                # multiple of kv_bytes) does not silently yield storage_bs=0
                # and a wrong size_tok.
                if self._backend_config.model_arch.is_mla:
                    per_token_bytes = kv_bytes  # arch.kv_bytes_per_token
                else:
                    # _build_kv_cache_groups: full page_bytes =
                    # 2 * block_size * kv_heads * head_size * dtype_size(=2)
                    # → per_token_bytes = page_bytes / block_size.
                    per_token_bytes = info.page_bytes // info.block_size
                # Fall back to the DSV4 constant when the derived value would
                # not divide page_bytes (guards against partial non-MLA paths
                # in this otherwise DSV4-specific method).
                if per_token_bytes <= 0 or info.page_bytes % per_token_bytes:
                    per_token_bytes = kv_bytes
                storage_bs = info.page_bytes // per_token_bytes
                size_tok = full_tokens // (info.block_size // storage_bs) if storage_bs else full_tokens
                kv_pages = _sglang_page_count(size_tok, storage_bs)
                group_bytes = info.layer_count * kv_pages * _pad576(info.page_bytes)
            total += group_bytes
        return total

    @property
    def name(self) -> str:
        return "sglang"


# ---------------------------------------------------------------------------
# Sim-side request wrapper
# ---------------------------------------------------------------------------


class SGLangSimRequest:
    """Simulator-side request handle for SGLang backend."""

    __slots__ = (
        "request_id",
        "prompt_token_ids",
        "max_tokens",
        "output_token_ids",
        "spec_token_ids",
        "_allocated_indices",
        "_last_node",
        # SWA sliding-window reclamation cursor (see
        # SGLangBackend._reclaim_swa_out_of_window).  Tracks this request's
        # own charged-token positions whose SWA slots have been returned to
        # the SWA sub-pool, mirroring real SGLang req.swa_evicted_seqlen.
        "swa_charged_tokens",
        "swa_evicted_charged",
    )

    def __init__(
        self, request_id: str, prompt_token_ids: list[int], max_tokens: int
    ):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.max_tokens = max_tokens
        self.output_token_ids: list[int] = []
        self.spec_token_ids: list[int] = []
        self._allocated_indices: list[Any] = []
        self._last_node: Any = None
        # Net tokens this request is billed for in the SWA pool
        # (prefill num_new + decode 1+K − rejected/beyond).  Cache-hit prefix
        # is NOT billed (it lives in the shared SWARadixCache, not this
        # request's allocation) — so the cursor is on charged positions,
        # not full-sequence positions, to avoid over-deducting on hits.
        self.swa_charged_tokens: int = 0
        self.swa_evicted_charged: int = 0

    @property
    def num_tokens(self) -> int:
        """Total tokens on the sim-side handle, INCLUDING pending spec tokens.

        Mirrors ``vLLMSimRequest.num_tokens``.  Currently unused by the
        scheduler (which reads ``SimRequestState.num_tokens``); kept for
        parity and potential diagnostics.
        """
        return len(self.prompt_token_ids) + len(self.output_token_ids) + len(self.spec_token_ids)
