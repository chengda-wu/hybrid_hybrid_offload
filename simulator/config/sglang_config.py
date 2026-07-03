"""SGLang-specific KV cache config builder.

No vllm imports — only loaded when backend='sglang'.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulator.config.model_config import KVBackendConfig


@dataclass
class SGLangConfig:
    """SGLang-specific config built from KVBackendConfig.

    Models three independent physical KV pools (SWA ring, C4, C128)
    matching real SGLang's DSV4PoolConfigurator.  Compressor state
    pools are not part of the KV pool budget — they are ring buffers
    sized separately and tracked in total_bytes only.
    """

    page_size: int  # system page_size for RadixCache alignment (256)

    # Three KV pool capacities in token-equivalents (per-layer × total layers)
    swa_tokens: int   # SWA ring budget × 43 layers
    c4_tokens: int    # C4 KV budget × 21 layers (main + indexer)
    c128_tokens: int  # C128 KV budget × 20 layers

    @classmethod
    def from_backend_config(cls, bc: KVBackendConfig) -> "SGLangConfig":
        """Build SGLang config from the common backend config."""
        blocks = bc.num_kv_cache_blocks
        sbs = bc.scheduler_block_size
        ps = bc.block_size  # system page_size

        full_tokens = blocks * sbs

        arch = bc.model_arch
        c4_layers = sum(1 for cr in arch.compress_ratios if cr == 4) if arch.compress_ratios else 0
        c128_layers = sum(1 for cr in arch.compress_ratios if cr == 128) if arch.compress_ratios else 0

        # spec mode: draft worker scaling (pool_configurator.py:538-545),
        # then page-align (pool_configurator.py:622).  All three pool caps
        # derive from the scaled+aligned full_tokens.
        num_spec = getattr(bc, "num_spec_tokens", 0) or 0
        if num_spec > 0:
            full_tokens = (full_tokens * arch.num_layers // (arch.num_layers + 1))
            full_tokens = (full_tokens // ps) * ps  # page-align

        swa_tok = (int(full_tokens * 0.1) // ps) * ps

        return cls(
            page_size=ps,
            swa_tokens=swa_tok * arch.num_layers,
            c4_tokens=(full_tokens // 4) * c4_layers * 2,  # main + indexer
            c128_tokens=(full_tokens // 128) * c128_layers,
        )
