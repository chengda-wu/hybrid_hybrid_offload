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

    def __init__(self, backend_config: KVBackendConfig,
                 num_spec_tokens: int = 0):
        sglang_cfg = SGLangConfig.from_backend_config(backend_config)

        self._backend_config = backend_config
        self._num_spec_tokens = num_spec_tokens
        self._page_size = sglang_cfg.page_size

        # Single RadixCache allocator (one index per token).
        total_slots = sglang_cfg.swa_tokens + sglang_cfg.c4_tokens + sglang_cfg.c128_tokens
        self._mock_allocator = MockTokenToKVPoolAllocator(
            total_slots,
            on_free=lambda n: self._deduct_pool_used(n),
        )

        # Three-pool capacities in per-layer slot equivalents.
        # Each token consumes swa_layers + c4_layers*2 + c128_layers slots.
        self._pool_caps = [
            sglang_cfg.swa_tokens,
            sglang_cfg.c4_tokens,
            sglang_cfg.c128_tokens,
        ]
        self._pool_used = [0, 0, 0]  # [swa, c4, c128] slots used
        arch = backend_config.model_arch
        self._swa_per_tok = arch.num_layers
        self._c4_per_tok = (sum(1 for cr in (arch.compress_ratios or []) if cr == 4) * 2)
        self._c128_per_tok = sum(1 for cr in (arch.compress_ratios or []) if cr == 128)

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

        # Three-pool capacity check: each token consumes per-layer slots.
        swa_need = to_alloc * self._swa_per_tok
        c4_need = to_alloc * self._c4_per_tok
        c128_need = to_alloc * self._c128_per_tok

        # If any pool is over budget, trigger RadixCache eviction.
        # evict() expects token count, not slot count.
        if (self._pool_used[0] + swa_need > self._pool_caps[0] or
                self._pool_used[1] + c4_need > self._pool_caps[1] or
                self._pool_used[2] + c128_need > self._pool_caps[2]):
            self._cache.evict(EvictParams(num_tokens=to_alloc))
            # Re-check: eviction may have freed enough
            if (self._pool_used[0] + swa_need > self._pool_caps[0] or
                    self._pool_used[1] + c4_need > self._pool_caps[1] or
                    self._pool_used[2] + c128_need > self._pool_caps[2]):
                return None

        # RadixCache token-level allocation (one index per token).
        # Flat pool check is redundant with per-pool cap check above;
        # the flat pool has ample capacity (15.6M slots).
        new_indices = self._mock_allocator.allocate(to_alloc)
        if new_indices is None:
            return None

        # Track per-pool slot usage
        self._pool_used[0] += swa_need
        self._pool_used[1] += c4_need
        self._pool_used[2] += c128_need

        sim_req._allocated_indices.append(new_indices)
        return new_indices

    def _deduct_pool_used(self, num_tokens: int) -> None:
        """Decrement per-pool slot usage when tokens are freed."""
        if num_tokens <= 0:
            return
        self._pool_used[0] = max(0, self._pool_used[0] - num_tokens * self._swa_per_tok)
        self._pool_used[1] = max(0, self._pool_used[1] - num_tokens * self._c4_per_tok)
        self._pool_used[2] = max(0, self._pool_used[2] - num_tokens * self._c128_per_tok)

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
        if len(flat) >= num_rejected:
            self._mock_allocator.free(flat[-num_rejected:])
            # Remove freed indices from _allocated_indices
            keep_len = len(flat) - num_rejected
            if keep_len > 0:
                sim_req._allocated_indices = [flat[:keep_len].clone()]
            else:
                sim_req._allocated_indices = []

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

        # Free unaligned tail
        all_tokens = array(
            "q", sim_req.prompt_token_ids + sim_req.output_token_ids
        )
        key_len = len(RadixKey(token_ids=all_tokens).page_aligned(self._page_size))

        flat = torch.cat(
            [t for t in sim_req._allocated_indices if len(t) > 0]
        ) if sim_req._allocated_indices else torch.tensor([], dtype=torch.int64)

        if len(flat) > key_len:
            tail = flat[key_len:]
            self._mock_allocator.free(tail)  # on_free handles _deduct_pool_used
        sim_req._allocated_indices = []

    def reset(self) -> None:
        self._cache.reset()
        self._pool_used = [0, 0, 0]
        total_slots = sum(self._pool_caps)
        self._mock_allocator = MockTokenToKVPoolAllocator(
            total_slots,
            on_free=lambda n: self._deduct_pool_used(n),
        )

    @property
    def usage(self) -> float:
        # Average of three-pool utilization
        ratios = []
        for used, cap in zip(self._pool_used, self._pool_caps):
            ratios.append(used / cap if cap > 0 else 0.0)
        return sum(ratios) / len(ratios) if ratios else 0.0

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
          swa_tokens  = align(full_tokens * 0.1, page_size)
          swa_slots   = swa_tokens // window_size(128)
          c4/c128 KV  = full_tokens // compress_ratio
          c4/c128 state = swa_slots * ring_size * state_bytes_per_token
          indexer KV  = c4 KV sized, 132 B/token
          indexer state = swa_slots * ring * state_bytes_per_token(2048)
        """
        blocks = self._backend_config.num_kv_cache_blocks
        page_size = self._page_size
        scheduler_bs = self._backend_config.scheduler_block_size
        swa_page_size = 128  # cfg.window_size; pool_configurator.py:470, HF sliding_window=128

        # Ring sizes — import from SGLang (module-level, no server_args needed).
        # Spec mode doubles ring sizes (pool_configurator.py:514).
        is_spec = self._num_spec_tokens > 0
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            get_compress_state_ring_size,
        )
        c4_ring = get_compress_state_ring_size(4, is_speculative=is_spec)
        c128_ring = get_compress_state_ring_size(128, is_speculative=is_spec)

        # Per-token byte costs from SGLang (pool_configurator.py:578-594).
        # kv_bytes = qk_nope + qk_rope*2 + 8 (fp8_ds_mla UE8M0 layout)
        kv_bytes = 584
        # dtype sizes from _get_dsv4_compress_state_dtype_sizes()
        # (pool_configurator.py:74-88, default float32→4)
        from sglang.srt.model_executor.pool_configurator import (
            _get_dsv4_compress_state_dtype_sizes,
        )
        c4_dt, c128_dt = _get_dsv4_compress_state_dtype_sizes()
        c4_state_bytes = 2 * 2 * 512 * c4_dt    # last_dim=2048, overlap=True
        c128_state_bytes = 2 * 1 * 512 * c128_dt # last_dim=1024, overlap=False
        idx_state_bytes = 2 * 2 * 128 * c4_dt    # indexer uses c4 dtype

        # spec mode: draft worker scaling (pool_configurator.py:538-545),
        # applied consistently to both pool_caps (SGLangConfig) and total_bytes here.
        full_tokens = blocks * scheduler_bs
        if self._num_spec_tokens > 0:
            full_tokens = (full_tokens * 43 // 44 // page_size) * page_size
        swa_tokens = (int(full_tokens * 0.1) // page_size) * page_size
        swa_slots = swa_tokens // swa_page_size

        def _ceil_div(a: int, b: int) -> int:
            return -(-a // b)

        def _pad576(raw: int) -> int:
            return _ceil_div(raw, 576) * 576  # matches SGLang create_buffer

        total = 0
        for info in self._backend_config.build_kv_cache_groups():
            if info.name == "swa":
                # SWA: pad per-page (swa_page_size * kv_bytes), then × slots × layers
                padded_page = _pad576(swa_page_size * kv_bytes)
                group_bytes = swa_slots * padded_page * info.layer_count
            elif info.name == "c4_compressor":
                # CompressStatePool._size = ceil(size + ring + 1, ratio) * ratio
                # (deepseek_v4_compress_state.py:117-123)
                raw = swa_slots * c4_ring
                state_tokens = -(-(raw + c4_ring + 1) // c4_ring) * c4_ring
                group_bytes = state_tokens * c4_state_bytes * info.layer_count
            elif info.name == "c128_compressor":
                raw = swa_slots * c128_ring
                state_tokens = -(-(raw + c128_ring + 1) // c128_ring) * c128_ring
                group_bytes = state_tokens * c128_state_bytes * info.layer_count
            elif info.name == "c4_indexer":
                # DeepSeekV4IndexerPool._create_buffer: no 576 padding.
                # pages = ceil_div(size + page_size + 1, page_size)
                c4_tok = full_tokens // 4
                c4_page_tokens = info.block_size // 4  # 64
                kv_pages = _ceil_div(c4_tok + c4_page_tokens + 1, c4_page_tokens)
                kv = info.layer_count * kv_pages * info.page_bytes
                # Indexer state: same size+ring+1 rounding as compressor state
                raw = swa_slots * c4_ring
                state_tok = -(-(raw + c4_ring + 1) // c4_ring) * c4_ring
                state = state_tok * idx_state_bytes * info.layer_count
                group_bytes = kv + state
            else:
                # Main KV: ceil_div(size + page_size + 1, page_size) pages.
                # storage_bs = page_bytes // 584 (unpadded bytes / per-token bytes)
                storage_bs = info.page_bytes // 584
                size_tok = full_tokens // (info.block_size // storage_bs) if storage_bs else full_tokens
                kv_pages = _ceil_div(size_tok + storage_bs + 1, storage_bs)
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

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids) + len(self.spec_token_ids)
