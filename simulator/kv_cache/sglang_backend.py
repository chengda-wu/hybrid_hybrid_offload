"""SGLang KV cache backend — wraps the real SGLang RadixCache."""

from __future__ import annotations

from array import array
from typing import Any

from simulator.config.model_config import KVBackendConfig, SGLangConfig
from simulator.kv_cache.base import KVBackend


# ---------------------------------------------------------------------------
# Mock allocator — satisfies the BaseTokenToKVPoolAllocator protocol
# ---------------------------------------------------------------------------


class MockTokenToKVPoolAllocator:
    """Minimal mock allocator for standalone RadixCache usage.

    Allocates integer token indices from a flat pool.  Supports free()
    for correct cache eviction behavior.
    """

    def __init__(self, total_tokens: int):
        self._total = total_tokens
        self._next_idx = 0
        self._free_list: list[int] = []

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
        if self._next_idx > self._total:
            self._next_idx = start  # rollback — matches real alloc behavior
            return None
        return torch.arange(start, start + num_tokens, dtype=torch.int64)

    def free(self, indices) -> None:
        """Return token indices to the free pool."""
        if hasattr(indices, "tolist"):
            self._free_list.extend(indices.tolist())
        elif isinstance(indices, list):
            self._free_list.extend(indices)

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

        # Single flat token pool for simulation.  The real SGLang uses three
        # separate pools (SWA ring buffer, C4 pages, C128 pages) but RadixCache
        # prefix matching is content-addressed and independent of pool layout.
        # The total token count accounts for the combined space.
        self._mock_allocator = MockTokenToKVPoolAllocator(sglang_cfg.total_tokens)

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
        all_tokens = array("q", sim_req.prompt_token_ids + sim_req.output_token_ids)
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

        if self._mock_allocator.available_size() < to_alloc:
            deficit = to_alloc - self._mock_allocator.available_size()
            self._cache.evict(EvictParams(num_tokens=deficit))

        if self._mock_allocator.available_size() < to_alloc:
            return None

        new_indices = self._mock_allocator.allocate(to_alloc)
        sim_req._allocated_indices.append(new_indices)
        return new_indices

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
        self._cache.insert(InsertParams(key=key, value=values))

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
        """Free unaligned tail indices (radix_cache.py:478-479).

        Page-aligned indices are freed lazily by evict() — tail indices
        are NOT in the tree, so they must be freed explicitly here.
        """
        import torch
        from array import array
        from sglang.srt.mem_cache.radix_cache import RadixKey

        all_tokens = array(
            "q", sim_req.prompt_token_ids + sim_req.output_token_ids
        )
        key_len = len(RadixKey(token_ids=all_tokens).page_aligned(self._page_size))

        flat = torch.cat(
            [t for t in sim_req._allocated_indices if len(t) > 0]
        ) if sim_req._allocated_indices else torch.tensor([], dtype=torch.int64)

        if len(flat) > key_len:
            self._mock_allocator.free(flat[key_len:])
        sim_req._allocated_indices = []

    def reset(self) -> None:
        self._cache.reset()

    @property
    def usage(self) -> float:
        total = self._mock_allocator._total
        num_free = self._mock_allocator.available_size()
        return (total - num_free) / total if total > 0 else 0.0

    @property
    def num_free_blocks(self) -> int:
        return self._mock_allocator.available_size()  # token slots, not blocks

    @property
    def total_blocks(self) -> int:
        return self._mock_allocator._total  # token slots, not blocks

    @property
    def total_bytes(self) -> int:
        """Total KV cache bytes.

        SWA portion uses the real SGLang pool_configurator logic:
          swa_tokens = align(full_tokens * swa_ratio, page_size)
        where full_tokens = blocks * scheduler_block_size (the shared
        token budget) and swa_ratio=0.1 (deepseek_v4_hook.py:57).
        """
        blocks = self._backend_config.num_kv_cache_blocks
        page_size = self._page_size
        scheduler_bs = self._backend_config.scheduler_block_size
        total = 0
        for info in self._backend_config.build_kv_cache_groups():
            if info.name == "swa":
                # pool_configurator.py:313-314,407:
                # swa_tokens = align(int(full_tokens * ratio), page_size)
                full_tokens = blocks * scheduler_bs
                swa_tokens = (int(full_tokens * 0.1) // page_size) * page_size
                per_token = info.page_bytes // info.block_size
                group_bytes = swa_tokens * per_token * info.layer_count
            else:
                group_bytes = info.layer_count * blocks * info.page_bytes
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
    )

    def __init__(
        self, request_id: str, prompt_token_ids: list[int], max_tokens: int
    ):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.max_tokens = max_tokens
        self.output_token_ids: list[int] = []
        self.spec_token_ids: list[int] = []
        self._allocated_indices: list[Any] = []  # track all allocs for free()

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids) + len(self.spec_token_ids)
