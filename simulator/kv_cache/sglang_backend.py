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
        """Allocate *num_tokens* token indices, reusing freed ones first."""
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
            raise RuntimeError(
                f"No free tokens: need {num_tokens}, "
                f"only {self._total - start} remaining"
            )
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
        self._total_tokens = sglang_cfg.total_tokens
        self._mock_allocator = MockTokenToKVPoolAllocator(self._total_tokens)

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

    def add_request(self, sim_req: "SGLangSimRequest") -> None:
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
        """Allocate token slots and insert into the radix tree.

        Returns newly allocated indices tensor or None on failure.
        """
        import torch

        from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        # Total tokens after this step
        needed = sim_req.num_tokens + num_new_tokens
        current = num_new_computed_tokens
        to_alloc = needed - current
        if to_alloc <= 0:
            return torch.tensor([], dtype=torch.int64)

        # Evict if needed
        if self._mock_allocator.available_size() < to_alloc:
            deficit = to_alloc - self._mock_allocator.available_size()
            self._cache.evict(EvictParams(num_tokens=deficit))

        if self._mock_allocator.available_size() < to_alloc:
            return None

        new_indices = self._mock_allocator.allocate(to_alloc)

        # Insert the full token sequence into the radix tree
        all_tokens = array(
            "q",
            sim_req.prompt_token_ids
            + sim_req.output_token_ids
            + sim_req.spec_token_ids,
        )
        key = RadixKey(token_ids=all_tokens[:needed])
        self._cache.insert(InsertParams(key=key, value=new_indices))

        return new_indices

    def free(self, sim_req: "SGLangSimRequest") -> None:
        """Free all token slots for this request."""
        all_tokens = array("q", sim_req.prompt_token_ids + sim_req.output_token_ids)
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        result = self._cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=all_tokens))
        )
        if len(result.device_indices) > 0:
            self._mock_allocator.free(result.device_indices)

    def reset(self) -> None:
        self._cache.reset()

    @property
    def usage(self) -> float:
        num_used = self._total_tokens - self._mock_allocator.available_size()
        return num_used / self._total_tokens if self._total_tokens > 0 else 0.0

    @property
    def num_free_blocks(self) -> int:
        return self._mock_allocator.available_size()

    @property
    def total_blocks(self) -> int:
        return self._total_tokens

    @property
    def total_bytes(self) -> int:
        """Estimate from model architecture and total token slots."""
        arch = self._backend_config.model_arch
        if arch.is_mla:
            per_token = 584  # fp8_ds_mla
        else:
            dtype_size = 2  # bf16
            per_token = 2 * arch.num_kv_heads * arch.head_size * dtype_size
        return self._total_tokens * per_token * arch.num_layers

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
    )

    def __init__(
        self, request_id: str, prompt_token_ids: list[int], max_tokens: int
    ):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.max_tokens = max_tokens
        self.output_token_ids: list[int] = []
        self.spec_token_ids: list[int] = []

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids) + len(self.spec_token_ids)
