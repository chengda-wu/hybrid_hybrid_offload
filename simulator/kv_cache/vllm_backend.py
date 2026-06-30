"""vLLM KV cache backend — wraps the real vllm KVCacheManager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from simulator.config.model_config import KVBackendConfig, VLLMConfig
from simulator.kv_cache.base import KVBackend

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks


class vLLMBackend(KVBackend):
    """Wraps the real vllm KVCacheManager for token-level simulation."""

    def __init__(self, backend_config: KVBackendConfig):
        vllm_cfg = VLLMConfig.from_backend_config(backend_config)

        # Lazy imports so this module can be loaded when vllm is not installed
        from vllm.utils.hashing import sha256
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.kv_cache_utils import (
            get_request_block_hasher,
            init_none_hash,
        )

        self._block_size = backend_config.block_size
        self._hash_block_size = backend_config.hash_block_size
        self._scheduler_block_size = backend_config.scheduler_block_size

        # Must be called before any block hashing
        init_none_hash(sha256)

        self._manager = KVCacheManager(
            kv_cache_config=vllm_cfg.kv_cache_config,
            max_model_len=backend_config.max_model_len,
            scheduler_block_size=self._scheduler_block_size,
            hash_block_size=self._hash_block_size,
            enable_caching=True,
            log_stats=False,
            enable_kv_cache_events=False,
        )
        self._block_hasher = get_request_block_hasher(
            self._hash_block_size, sha256
        )

    # ---- KVBackend interface ----

    def create_request(
        self, request_id: str, prompt_token_ids: list[int], max_tokens: int
    ) -> "vLLMSimRequest":
        return vLLMSimRequest(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            max_tokens=max_tokens,
        )

    def add_request(self, sim_req: "vLLMSimRequest") -> None:
        """Build the real vllm Request and register it."""
        from vllm.sampling_params import SamplingParams

        sampling_params = SamplingParams(max_tokens=sim_req.max_tokens)
        sampling_params.update_from_generation_config({}, eos_token_id=-1)

        from vllm.v1.request import Request

        sim_req._vllm_request = Request(
            request_id=sim_req.request_id,
            prompt_token_ids=list(sim_req.prompt_token_ids),
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=self._block_hasher,
        )

    def get_computed_blocks(
        self, sim_req: "vLLMSimRequest"
    ) -> tuple["KVCacheBlocks", int]:
        assert sim_req._vllm_request is not None
        return self._manager.get_computed_blocks(sim_req._vllm_request)

    def allocate_slots(
        self,
        sim_req: "vLLMSimRequest",
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: "KVCacheBlocks | None" = None,
    ) -> "KVCacheBlocks | None":
        assert sim_req._vllm_request is not None
        return self._manager.allocate_slots(
            request=sim_req._vllm_request,
            num_new_tokens=num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
        )

    def free(self, sim_req: "vLLMSimRequest") -> None:
        if sim_req._vllm_request is not None:
            self._manager.free(sim_req._vllm_request)
            sim_req._vllm_request = None

    def reset(self) -> None:
        self._manager.reset_prefix_cache()

    @property
    def usage(self) -> float:
        num_free = self.num_free_blocks
        total = self.total_blocks
        return 1.0 - (num_free / total) if total > 0 else 0.0

    @property
    def num_free_blocks(self) -> int:
        return self._manager.block_pool.get_num_free_blocks()

    @property
    def total_blocks(self) -> int:
        return self._manager.kv_cache_config.num_blocks

    @property
    def total_bytes(self) -> int:
        """Sum of all kv_cache_tensor sizes — the real allocation including
        vLLM's block padding and hybrid group layout."""
        return sum(t.size for t in self._manager.kv_cache_config.kv_cache_tensors)

    @property
    def name(self) -> str:
        return "vllm"


# ---------------------------------------------------------------------------
# Sim-side request wrapper
# ---------------------------------------------------------------------------


class vLLMSimRequest:
    """Simulator-side request handle backed by a vLLM Request.

    Separates simulator state from backend state.  The vllm Request is
    created lazily when ``add_request`` is called (at prefill admission).
    """

    __slots__ = (
        "request_id",
        "prompt_token_ids",
        "max_tokens",
        "output_token_ids",
        "spec_token_ids",
        "_vllm_request",
    )

    def __init__(
        self, request_id: str, prompt_token_ids: list[int], max_tokens: int
    ):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.max_tokens = max_tokens
        self.output_token_ids: list[int] = []
        self.spec_token_ids: list[int] = []
        self._vllm_request: Any | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids) + len(self.spec_token_ids)

    def sync_to_vllm(self) -> None:
        """Push simulator token state into the vllm Request.

        Only sync prompt + accepted output tokens; spec tokens are excluded
        since they may be rejected.
        """
        if self._vllm_request is None:
            return
        self._vllm_request.num_computed_tokens = (
            len(self.prompt_token_ids) + len(self.output_token_ids)
        )
