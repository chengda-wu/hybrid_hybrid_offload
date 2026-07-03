"""SGLang-specific KV cache config builder.

No vllm imports — only loaded when backend='sglang'.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulator.config.model_config import KVBackendConfig


@dataclass
class SGLangConfig:
    """SGLang-specific config built from KVBackendConfig."""

    page_size: int
    total_tokens: int  # total token slots in the mock allocator

    @classmethod
    def from_backend_config(cls, bc: KVBackendConfig) -> "SGLangConfig":
        """Build SGLang config from the common backend config.

        For DeepSeek V4 in SGLang:
        - System page_size = block_size (256), used by RadixCache for prefix
          matching alignment.
        - Mock pool size = num_blocks × scheduler_block_size — models a
          single shared block budget (matching vLLM's single BlockPool).
          Groups don't each get their own full quota; the coordinator
          splits the shared budget internally.
        """
        arch = bc.model_arch
        page_size = bc.block_size
        total_tokens = bc.num_kv_cache_blocks * bc.scheduler_block_size

        return cls(page_size=page_size, total_tokens=total_tokens)
