"""Convert HuggingFace config.json into KV cache backend configurations.

Supports both vLLM and SGLang backends. DeepSeek V4 Flash is the primary
target with hardcoded defaults; generic models are supported via config.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


# ---------------------------------------------------------------------------
# ModelArchitecture — canonical representation parsed from HF config.json
# ---------------------------------------------------------------------------


@dataclass
class ModelArchitecture:
    """Parsed from a HuggingFace config.json."""

    model_type: str
    num_layers: int
    num_kv_heads: int
    num_attention_heads: int
    head_size: int  # KV head dimension (512 for DeepSeek V4 MLA)
    max_position_embeddings: int
    hidden_size: int
    dtype: str = "bfloat16"

    # MLA-specific (DeepSeek V2/V3/V4)
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    compress_ratio: int = 1  # 1=no compression, 4=C4A, 128=C128A
    sliding_window: int | None = None

    # Derived
    vocab_size: int = 129280
    is_mla: bool = False

    def __post_init__(self) -> None:
        if self.kv_lora_rank is not None and self.qk_rope_head_dim is not None:
            self.is_mla = True

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelArchitecture":
        """Load from a HuggingFace config.json file."""
        with open(path) as f:
            cfg = json.load(f)

        model_type = cfg.get("model_type", cfg.get("architectures", ["unknown"])[0])

        # Head size detection
        head_size = cfg.get("head_dim", 0)
        if not head_size:
            nope = cfg.get("qk_nope_head_dim", 0)
            rope = cfg.get("qk_rope_head_dim", 0)
            if nope and rope:
                head_size = nope + rope
        if not head_size:
            head_size = cfg.get("hidden_size", 4096) // cfg.get(
                "num_attention_heads", 32
            )

        kv_lora_rank = cfg.get("kv_lora_rank")
        qk_rope_head_dim = cfg.get("qk_rope_head_dim")

        # Per-layer compress ratios (DeepSeek V4); take the most common non-zero
        compress_ratios = cfg.get("compress_ratios")
        if compress_ratios:
            from collections import Counter

            nonzero = [r for r in compress_ratios if r > 0]
            compress_ratio = Counter(nonzero).most_common(1)[0][0] if nonzero else 1
        else:
            compress_ratio = 1

        return cls(
            model_type=model_type,
            num_layers=cfg.get("num_hidden_layers", 32),
            num_kv_heads=cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 32)),
            num_attention_heads=cfg.get("num_attention_heads", 32),
            head_size=head_size,
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            hidden_size=cfg.get("hidden_size", 4096),
            dtype=cfg.get("torch_dtype", "bfloat16").replace("torch.", ""),
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            compress_ratio=compress_ratio,
            sliding_window=cfg.get("sliding_window"),
            vocab_size=cfg.get("vocab_size", 129280),
        )

    @classmethod
    def deepseek_v4_flash(cls) -> "ModelArchitecture":
        """Hardcoded defaults for DeepSeek V4 Flash (FP8)."""
        return cls(
            model_type="deepseek_v4",
            num_layers=61,
            num_kv_heads=1,
            num_attention_heads=128,
            head_size=512,  # qk_nope_head_dim=448 + qk_rope_head_dim=64 for V4 Flash
            max_position_embeddings=131072,
            hidden_size=7168,
            dtype="bfloat16",
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            compress_ratio=8,  # most common in V4 Flash
            sliding_window=128,
            vocab_size=129280,
            is_mla=True,
        )


# ---------------------------------------------------------------------------
# KVBackendConfig — shared configuration for both vLLM and SGLang
# ---------------------------------------------------------------------------


@dataclass
class KVBackendConfig:
    """Common config produced for either backend."""

    model_arch: ModelArchitecture
    block_size: int = 16
    hash_block_size: int = 16
    max_model_len: int = 8192
    num_kv_cache_blocks: int = 4096
    scheduler_block_size: int = 16
    page_size: int = 1  # SGLang: tokens per page (1 = token-level)
    kv_cache_dtype: str = "auto"
    model_version: str = "deepseek_v4"

    @property
    def num_kv_cache_groups(self) -> int:
        """Number of KV cache groups (1 for uniform models)."""
        return 1


# ---------------------------------------------------------------------------
# vLLM-specific config builder
# ---------------------------------------------------------------------------


@dataclass
class VLLMConfig:
    """vLLM-specific config built from KVBackendConfig."""

    kv_cache_config: "KVCacheConfig"

    @classmethod
    def from_backend_config(cls, bc: KVBackendConfig) -> "VLLMConfig":
        """Build KVCacheConfig for a vLLM KVCacheManager."""

        # Lazy import so the module is importable without vllm installed
        import torch

        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
            MLAAttentionSpec,
        )

        arch = bc.model_arch

        # Determine dtype for KV cache
        if bc.kv_cache_dtype == "fp8_ds_mla":
            kv_dtype = torch.uint8
            cache_dtype_str = "fp8_ds_mla"
        elif bc.kv_cache_dtype == "auto" and arch.is_mla:
            kv_dtype = torch.uint8
            cache_dtype_str = "fp8_ds_mla"
        else:
            kv_dtype = getattr(torch, arch.dtype)
            cache_dtype_str = None

        # Build KV cache spec
        if arch.is_mla:
            kv_cache_spec = MLAAttentionSpec(
                block_size=bc.block_size,
                num_kv_heads=arch.num_kv_heads,
                head_size=arch.head_size,
                dtype=kv_dtype,
                compress_ratio=arch.compress_ratio,
                cache_dtype_str=cache_dtype_str,
                model_version=bc.model_version,
            )
        else:
            kv_cache_spec = FullAttentionSpec(
                block_size=bc.block_size,
                num_kv_heads=arch.num_kv_heads,
                head_size=arch.head_size,
                dtype=kv_dtype,
                sliding_window=arch.sliding_window,
            )

        # All layers share the same spec group
        layer_names = [f"model.layers.{i}.self_attn" for i in range(arch.num_layers)]
        kv_cache_groups = [KVCacheGroupSpec(layer_names, kv_cache_spec)]

        # Build tensor layout: one tensor per layer (simplest allocation)
        page_bytes = kv_cache_spec.page_size_bytes
        kv_cache_tensors = [
            KVCacheTensor(size=page_bytes * bc.num_kv_cache_blocks, shared_by=[name])
            for name in layer_names
        ]

        return cls(
            kv_cache_config=KVCacheConfig(
                num_blocks=bc.num_kv_cache_blocks,
                kv_cache_tensors=kv_cache_tensors,
                kv_cache_groups=kv_cache_groups,
            )
        )


# ---------------------------------------------------------------------------
# SGLang-specific config builder
# ---------------------------------------------------------------------------


@dataclass
class SGLangConfig:
    """SGLang-specific config built from KVBackendConfig."""

    page_size: int
    total_tokens: int  # total token slots in the mock allocator

    @classmethod
    def from_backend_config(cls, bc: KVBackendConfig) -> "SGLangConfig":
        """Build SGLang config from the common backend config."""
        # SGLang uses token-level granularity: total_tokens = blocks * block_size
        total_tokens = bc.num_kv_cache_blocks * bc.block_size
        return cls(page_size=bc.page_size, total_tokens=total_tokens)
