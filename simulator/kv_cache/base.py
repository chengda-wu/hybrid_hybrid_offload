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
        SGLang: explicit free from mock pool.

        vLLM no-op is *accurate*, not an approximation: real vLLM's
        ``update_from_output`` only rolls back ``num_computed_tokens`` by
        ``num_rejected`` — it does NOT return the rejected drafts' blocks to
        the free pool.  Those blocks stay in ``req_to_blocks`` and are reused
        on the next decode step (because the rolled-back ``num_computed`` means
        fewer blocks are needed, so ``allocate_new_blocks`` allocates nothing
        and the existing trailing blocks are overwritten in place).  Hence
        ``get_num_free_blocks`` does not rise after a rejection, matching this
        no-op.  The scheduler still calls ``subtract_rejected_tokens`` to mirror
        the ``num_computed_tokens`` rollback.

        SGLang differs: its ``req_to_token_pool`` rows are append-only with no
        in-place reuse, so rejected tail slots must be explicitly freed (else
        they leak until ``free()``).

        SWA sliding-window reclamation is NOT handled here, and does not need
        to be: each backend models it directly.
        - vLLM: ``remove_skipped_blocks`` is called automatically at the start
          of every ``allocate_slots`` (kv_cache_manager.py:400-404), so SWA
          head blocks outside the window are freed every step.
        - SGLang: ``SGLangBackend._reclaim_swa_out_of_window`` (called at the
          end of ``sync_state``) mirrors real SGLang
          ``free_swa_out_of_window_slots`` (common.py:68, driven by
          ``ScheduleBatch._evict_swa`` every decode step), returning
          out-of-window SWA slots to the SWA sub-pool.  Without it the SWA
          pool would grow with total sequence length and OOM far too early.
        So SWA occupancy is modeled faithfully on both backends, not approximated.

        Per-step call chain in real vLLM (and in this simulator, which
        delegates ``allocate_slots`` straight through):
            scheduler.schedule()                          # each step
              └─ per running request:
                   kv_cache_manager.allocate_slots()      # sched L527
                     └─ coordinator.remove_skipped_blocks()  # kv_cache_manager L400
                          └─ per KV group:
                               SWA group (SlidingWindowManager):
                                 get_num_skipped_tokens =
                                   max(0, num_computed - sliding_window + 1)
                                 → free head blocks outside the window, replace
                                   with null_block in req_to_blocks
                               other groups (Full/MLA/Compressor/Indexer):
                                 get_num_skipped_tokens == 0 → no-op
        So every decode step, every running request, triggers one SWA head-block
        reclamation — matching real vLLM's timing and frequency.  (Two other
        ``remove_skipped_blocks`` call sites exist — kv_cache_manager L487 and
        scheduler L2342, the latter KV-connector-only at request finish — but
        the per-step reclamation is entirely driven by the L400 site above.)

        Verified empirically: 8000-token single-request decode gives
        avg_cache_usage 0.018 with reclamation vs 0.41 when
        ``remove_skipped_blocks`` is patched to a no-op (22x difference),
        confirming it is active and materially correct.
        """
        pass

    @abstractmethod
    def free(self, sim_req: Any) -> None:
        """Free all blocks for a request."""
        ...

    @property
    @abstractmethod
    def usage(self) -> float:
        """Current cache utilization ratio [0, 1]."""
        ...

    def pool_usage_detail(self) -> list[tuple[str, float]] | None:
        """Per-pool utilization for backends with multiple physical pools.

        Returns a list of (pool_name, utilization_ratio) for backends that
        split KV cache across independent pools (SGLang: swa/full), or
        None for single-pool backends (vLLM shared block pool).  Used for
        end-of-run diagnostics; ``usage`` is the single aggregated number.
        """
        return None

    def pool_peak_detail(self) -> list[tuple[str, float]] | None:
        """Peak per-pool utilization over the run, or None for single-pool.

        More informative than ``pool_usage_detail`` at end-of-run (which is
        ~0 after requests free): shows which pool nearly OOM'd.  Backends
        without per-pool tracking return None.
        """
        return None

    @property
    def num_free_blocks(self) -> int:
        """Free block/token count, for OOM diagnostics only.

        vLLM reports free pool blocks; SGLang reports free token slots.  Not
        part of the scheduling control flow (only ``usage`` is) — the scheduler
        reads it solely to build a clearer prefill-OOM message.  Declared here
        (non-abstract, default 0) rather than left implicit so a third backend
        that omits it degrades to an unhelpful "0 free" diagnostic instead of
        crashing the error-reporting path with AttributeError.  Concrete
        backends override.
        """
        return 0

    @property
    def total_blocks(self) -> int:
        """Total block/token capacity, for diagnostics only.  See
        ``num_free_blocks``.  Concrete backends override.
        """
        return 0

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
