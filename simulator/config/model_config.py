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
    compress_ratio: int = 1  # representative (most common non-zero, for backward compat)
    compress_ratios: list[int] | None = None  # per-layer list from config.json
    sliding_window: int | None = None

    # Derived
    vocab_size: int = 129280
    is_mla: bool = False

    def __post_init__(self) -> None:
        if self.kv_lora_rank is not None and self.qk_rope_head_dim is not None:
            self.is_mla = True
        if self.compress_ratios is None:
            self.compress_ratios = [self.compress_ratio] * self.num_layers

    @property
    def layer_groups(self) -> list[tuple[str, int, int]]:
        """Return KV cache layer groups for hybrid models.

        Each element is (group_name, block_size, compress_ratio, num_layers).
        DeepSeek V4 has three groups covering ALL layers:
          - SWA:  ALL layers have sliding window attention (SlidingWindowMLASpec,
                  block_size=64).  This is a per-layer ring buffer, not just
                  for layers with compress_ratio <= 1.
          - C4:   layers with compress_ratio == 4 additionally use compressed
                  long-context attention (MLAAttentionSpec, block_size=256, cr=4).
          - C128: layers with compress_ratio == 128 (MLAAttentionSpec,
                  block_size=256, cr=128).

        For non-hybrid models, returns a single group.
        """
        if not self.is_mla or self.compress_ratios is None:
            return [("full", self.head_size, 1, self.num_layers)]

        ratios = self.compress_ratios
        groups: list[tuple[str, int, int, int]] = []

        # SWA: all layers have sliding window attention
        swa_count = self.num_layers
        # Remove SWA-only count from compress_ratio grouping; mark separately
        c4_count = sum(1 for cr in ratios if cr == 4)
        c128_count = sum(1 for cr in ratios if cr == 128)

        groups.append(("swa", 64, 1, swa_count))
        if c4_count:
            groups.append(("c4", 256, 4, c4_count))
        if c128_count:
            groups.append(("c128", 256, 128, c128_count))
        if not groups:
            groups.append(("full", 256, 1, self.num_layers))
        return groups

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

        # Per-layer compress ratios (DeepSeek V4); keep the full list
        compress_ratios_raw = cfg.get("compress_ratios")
        if compress_ratios_raw:
            compress_ratios = list(compress_ratios_raw)
            from collections import Counter

            nonzero = [r for r in compress_ratios_raw if r > 0]
            compress_ratio = Counter(nonzero).most_common(1)[0][0] if nonzero else 1
        else:
            compress_ratios = None
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
            compress_ratios=compress_ratios,
            sliding_window=cfg.get("sliding_window"),
            vocab_size=cfg.get("vocab_size", 129280),
        )

    @classmethod
    def deepseek_v4_flash(cls) -> "ModelArchitecture":
        """Hardcoded defaults for DeepSeek V4 Flash (FP8).

        KV cache layout (three groups from vLLM):
          - SWA (compress_ratio=0):  SlidingWindowMLASpec, block_size=64, page=37376 B
          - C4  (compress_ratio=4):  MLAAttentionSpec,    block_size=256, page=37376 B (256/4*584)
          - C128 (compress_ratio=128): MLAAttentionSpec,   block_size=256, page=1168 B  (256/128*584)

        Typical 43-layer distribution: 5 SWA + 19 C4 + 19 C128.
        """
        # Build representative per-layer pattern
        compress_ratios = (
            [0] * 5 + [4] * 19 + [128] * 19
        )[:43]  # ensure exactly 43
        return cls(
            model_type="deepseek_v4",
            num_layers=43,
            num_kv_heads=1,
            num_attention_heads=128,
            head_size=512,  # qk_nope_head_dim=448 + qk_rope_head_dim=64 for V4 Flash
            max_position_embeddings=131072,
            hidden_size=7168,
            dtype="bfloat16",
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            compress_ratio=8,  # representative (most common non-zero is 4 or 128)
            compress_ratios=compress_ratios,
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
        """Build KVCacheConfig for a vLLM KVCacheManager.

        For hybrid models (e.g. DeepSeek V4), creates separate
        KVCacheGroupSpecs for each attention type (SWA, C4, C128),
        matching vLLM's real layout.
        """

        import torch

        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
            MLAAttentionSpec,
            SlidingWindowMLASpec,
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

        if arch.is_mla and arch.compress_ratios is not None:
            # Hybrid model with multiple KV cache types
            kv_cache_groups, kv_cache_tensors = cls._build_hybrid_groups(
                arch, bc, kv_dtype, cache_dtype_str
            )
        else:
            # Uniform model — single group
            kv_cache_spec = FullAttentionSpec(
                block_size=bc.block_size,
                num_kv_heads=arch.num_kv_heads,
                head_size=arch.head_size,
                dtype=kv_dtype,
                sliding_window=arch.sliding_window,
            )
            layer_names = [f"model.layers.{i}.self_attn" for i in range(arch.num_layers)]
            kv_cache_groups = [KVCacheGroupSpec(layer_names, kv_cache_spec)]
            page_bytes = kv_cache_spec.page_size_bytes
            kv_cache_tensors = [
                KVCacheTensor(
                    size=page_bytes * bc.num_kv_cache_blocks, shared_by=[name]
                )
                for name in layer_names
            ]

        return cls(
            kv_cache_config=KVCacheConfig(
                num_blocks=bc.num_kv_cache_blocks,
                kv_cache_tensors=kv_cache_tensors,
                kv_cache_groups=kv_cache_groups,
            )
        )

    @staticmethod
    def _build_hybrid_groups(
        arch: "ModelArchitecture",
        bc: "KVBackendConfig",
        kv_dtype: torch.dtype,
        cache_dtype_str: str | None,
    ) -> tuple[list["KVCacheGroupSpec"], list["KVCacheTensor"]]:
        """Build per-type KV cache groups for hybrid MLA models.

        DeepSeek V4 has three types:
          - SWA  (compress_ratio <= 1): SlidingWindowMLASpec, block_size=64
          - C4   (compress_ratio == 4): MLAAttentionSpec,    block_size=256, cr=4
          - C128 (compress_ratio == 128): MLAAttentionSpec,  block_size=256, cr=128
        """
        from vllm.v1.kv_cache_interface import (
            KVCacheGroupSpec,
            KVCacheTensor,
            MLAAttentionSpec,
            SlidingWindowMLASpec,
        )

        assert arch.compress_ratios is not None
        ratios = arch.compress_ratios

        # Partition layer indices by type.
        # SWA applies to ALL layers (sliding window attention).
        # C4/C128 layers additionally have compressed long-context attention.
        all_layers = list(range(arch.num_layers))
        c4_layers: list[int] = []
        c128_layers: list[int] = []

        for i, cr in enumerate(ratios):
            if cr == 4:
                c4_layers.append(i)
            elif cr == 128:
                c128_layers.append(i)
            # cr <= 1: SWA-only, already covered by all_layers
            # unknown: treat as C4

        groups: list[KVCacheGroupSpec] = []
        tensors: list[KVCacheTensor] = []

        per_group_blocks = bc.num_kv_cache_blocks

        # SWA group: SlidingWindowMLASpec for ALL layers, block_size=64
        swa_spec = SlidingWindowMLASpec(
            block_size=64,  # hardcoded in DeepseekV4SWACache
            num_kv_heads=arch.num_kv_heads,
            head_size=arch.head_size,
            dtype=kv_dtype,
            sliding_window=arch.sliding_window or 128,
            cache_dtype_str=cache_dtype_str,
            alignment=576 if cache_dtype_str == "fp8_ds_mla" else None,
            model_version=bc.model_version,
        )
        swa_names = [f"model.layers.{i}.self_attn" for i in all_layers]
        groups.append(KVCacheGroupSpec(swa_names, swa_spec))
        for name in swa_names:
            tensors.append(
                KVCacheTensor(
                    size=swa_spec.page_size_bytes * per_group_blocks,
                    shared_by=[name],
                )
            )

        # C4 group: MLAAttentionSpec, block_size=256, compress_ratio=4
        if c4_layers:
            c4_spec = MLAAttentionSpec(
                block_size=256,
                num_kv_heads=arch.num_kv_heads,
                head_size=arch.head_size,
                dtype=kv_dtype,
                compress_ratio=4,
                cache_dtype_str=cache_dtype_str,
                alignment=576 if cache_dtype_str == "fp8_ds_mla" else None,
                model_version=bc.model_version,
            )
            c4_names = [f"model.layers.{i}.self_attn" for i in c4_layers]
            groups.append(KVCacheGroupSpec(c4_names, c4_spec))
            for name in c4_names:
                tensors.append(
                    KVCacheTensor(
                        size=c4_spec.page_size_bytes * per_group_blocks,
                        shared_by=[name],
                    )
                )

        # C128 group: MLAAttentionSpec, block_size=256, compress_ratio=128
        if c128_layers:
            c128_spec = MLAAttentionSpec(
                block_size=256,
                num_kv_heads=arch.num_kv_heads,
                head_size=arch.head_size,
                dtype=kv_dtype,
                compress_ratio=128,
                cache_dtype_str=cache_dtype_str,
                alignment=576 if cache_dtype_str == "fp8_ds_mla" else None,
                model_version=bc.model_version,
            )
            c128_names = [f"model.layers.{i}.self_attn" for i in c128_layers]
            groups.append(KVCacheGroupSpec(c128_names, c128_spec))
            for name in c128_names:
                tensors.append(
                    KVCacheTensor(
                        size=c128_spec.page_size_bytes * per_group_blocks,
                        shared_by=[name],
                    )
                )

        return groups, tensors


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
        """Build SGLang config from the common backend config.

        For DeepSeek V4 in SGLang:
        - System page_size = block_size (256), used by RadixCache for prefix
          matching alignment (matches rounded down to page boundaries).
        - Three internal pools: SWA (page=64), C4 (page=64), C128 (page=2).
        - Total token slots = blocks * block_size across all pools.
        """
        arch = bc.model_arch

        # SGLang RadixCache page_size = system page size
        page_size = bc.block_size

        # Compute total tokens slots across hybrid groups
        if arch.is_mla and arch.compress_ratios is not None:
            # For hybrid models, each group contributes its own pages
            total_tokens = 0
            for _name, grp_block_size, compress_ratio, n_layers in arch.layer_groups:
                storage = grp_block_size // max(compress_ratio, 1)
                total_tokens += n_layers * bc.num_kv_cache_blocks * storage
        else:
            total_tokens = bc.num_kv_cache_blocks * bc.block_size * arch.num_layers

        return cls(page_size=page_size, total_tokens=total_tokens)
