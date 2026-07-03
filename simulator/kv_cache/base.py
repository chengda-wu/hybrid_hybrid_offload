"""Abstract interface for KV cache backends (vLLM / SGLang)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class KVBackend(ABC):
    """Abstract interface for KV cache backends.

    vLLM uses block-level granularity; SGLang uses token-level.
    This interface normalizes both into ``num_computed_tokens`` semantics.
    """

    @abstractmethod
    def create_request(
        self, request_id: str, prompt_token_ids: list[int], max_tokens: int
    ) -> Any:
        """Create a backend-specific request handle.

        Returns a lightweight wrapper that the scheduler will pass
        to other backend methods.
        """
        ...

    @abstractmethod
    def register_request(self, sim_req: Any) -> None:
        """Register a new request with the backend.  Idempotent — safe
        to call multiple times (e.g. on prefill retry).
        """
        ...

    @abstractmethod
    def get_computed_blocks(self, sim_req: Any) -> tuple[Any, int]:
        """Match prefix cache.

        Returns:
            (blocks, num_computed_tokens) where blocks is backend-specific
            (KVCacheBlocks for vLLM, torch.Tensor for SGLang) and
            num_computed_tokens is the number of prefix-cache-hit tokens.
        """
        ...

    @abstractmethod
    def allocate_slots(
        self,
        sim_req: Any,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Any | None = None,
    ) -> Any | None:
        """Allocate KV cache slots for new tokens.

        Returns backend-specific allocation or None on failure.
        """
        ...

    @abstractmethod
    def set_spec_tokens(self, sim_req: Any, tokens: list[int]) -> None:
        """Push speculative draft token IDs to the backend handle."""
        ...

    @abstractmethod
    def sync_state(self, sim_req: Any, output_token_ids: list[int]) -> None:
        """Sync accepted output tokens to the backend after a decode step."""
        ...

    def free_rejected_slots(self, sim_req: Any, num_rejected: int) -> None:
        """Free rejected spec token slots.  vLLM: no-op (position rollback).
        SGLang: explicit free from mock pool."""
        pass

    @abstractmethod
    def free(self, sim_req: Any) -> None:
        """Free all blocks for a request."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the entire cache (for warmup boundaries)."""
        ...

    @property
    @abstractmethod
    def usage(self) -> float:
        """Current cache utilization ratio [0, 1]."""
        ...

    @property
    @abstractmethod
    def total_bytes(self) -> int:
        """Total KV cache size in bytes.

        vLLM delegates to ``_bucket_layers_by_page_size``.
        SGLang imports ``get_compress_state_ring_size`` and
        ``_get_dsv4_compress_state_dtype_sizes`` from SGLang source;
        remaining per-token formulas match ``pool_configurator.py``.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name ('vllm' or 'sglang')."""
        ...
