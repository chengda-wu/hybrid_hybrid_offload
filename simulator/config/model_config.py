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
    def layer_groups(self) -> list[tuple[str, int, int, int]]:
        """Return KV cache layer groups for hybrid MLA models.

        Each element is (group_name, block_size, compress_ratio, num_layers).
        Real vLLM DeepSeek V4 has 6 groups (5 types, C4/C128 MLA split):
          - SWA:  DeepseekV4SWACache → SlidingWindowMLASpec(bs=64),
                  ALL 43 layers (attention.py:290)
          - C4 Compressor:  CompressorStateCache → SlidingWindowMLASpec(bs=4,
                  sw=8, head_size=state_dim, dtype=float32) — 21 layers,
                  only for cr=4 (compressor.py:150,157-169)
          - C128 Compressor: CompressorStateCache → SlidingWindowMLASpec(bs=8,
                  sw=128, head_size=state_dim, dtype=float32) — 20 layers,
                  only for cr=128 (compressor.py:152,157-169)
          - Main MLA: DeepseekV4Attention → MLAAttentionSpec(bs=256, cr=4 or 128)
                  — 41 layers with cr>1 (attention.py:601-619)
          - C4 Indexer: DeepseekV4IndexerCache → MLAAttentionSpec(alignment=576)
                  — 21 layers, only for cr=4 (attention.py:643-655)

        For non-hybrid models, returns a single group.
        """
        if not self.is_mla or self.compress_ratios is None:
            return [("full", 0, 1, self.num_layers)]

        ratios = self.compress_ratios
        groups: list[tuple[str, int, int, int]] = []

        # 1. SWA main KV: DeepseekV4SWACache (bs=64), ALL 43 layers
        groups.append(("swa", 64, 1, self.num_layers))

        # 2. C4 Compressor: CompressorStateCache (bs=4, dtype=float32).
        # cr=0 is a sentinel — callers use max(cr,1) so storage = bs//1 = bs.
        c4_count = sum(1 for cr in ratios if cr == 4)
        if c4_count:
            groups.append(("c4_compressor", 4, 0, c4_count))

        # 3. C128 Compressor: CompressorStateCache (bs=8, dtype=float32)
        c128_count = sum(1 for cr in ratios if cr == 128)
        if c128_count:
            groups.append(("c128_compressor", 8, 0, c128_count))

        # 4. Main MLA: DeepseekV4Attention (bs=256, cr=4/128)
        compressed_count = c4_count + c128_count
        if compressed_count:
            # Use separate groups for C4 and C128 since they have different
            # compress_ratios and therefore different page sizes
            if c4_count:
                groups.append(("c4_mla", 256, 4, c4_count))
            if c128_count:
                groups.append(("c128_mla", 256, 128, c128_count))

        # 5. C4 Indexer: DeepseekV4IndexerCache (alignment=576)
        if c4_count:
            groups.append(("c4_indexer", 256, 4, c4_count))

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

        Real compress_ratios from HuggingFace config.json
        (deepseek-ai/DeepSeek-V4-Flash):
          layers 0-1:  SWA-only (compress_ratio=0)
          layers 2-42: alternating C4 (ratio=4) and C128 (ratio=128)
          → SWA=2, C4=21, C128=20
        Plus 1 MTP layer at index 43 (compress_ratio=0, not counted).

        KV cache groups:
          - SWA:  SlidingWindowMLASpec, block_size=64, all 43 layers
          - C4:   MLAAttentionSpec,  block_size=256, cr=4,  21 layers
          - C128: MLAAttentionSpec,  block_size=256, cr=128, 20 layers
        """
        # Real per-layer pattern from HuggingFace config.json
        compress_ratios = [0, 0]
        for i in range(41):
            compress_ratios.append(4 if i % 2 == 0 else 128)
        # → 43 layers: [0, 0, 4, 128, 4, 128, ..., 4]
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
            compress_ratio=4,  # representative (21 C4 layers vs 20 C128)
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

        DeepSeek V4 has 6 groups (5 attention types):
          - SWA (bs=64, all layers)
          - C4/C128 Compressor (bs=4/8, float32 state)
          - C4/C128 Main MLA (bs=256, cr=4/128)
          - C4 Indexer (alignment=576)
        """
        import torch as _torch

        from vllm.v1.kv_cache_interface import (
            KVCacheGroupSpec,
            KVCacheTensor,
            MLAAttentionSpec,
            SlidingWindowMLASpec,
        )

        assert arch.compress_ratios is not None
        ratios = arch.compress_ratios

        # Partition layers by compress_ratio
        all_layers = list(range(arch.num_layers))
        c4_layers = [i for i, cr in enumerate(ratios) if cr == 4]
        c128_layers = [i for i, cr in enumerate(ratios) if cr == 128]
        compressed_layers = c4_layers + c128_layers

        groups: list[KVCacheGroupSpec] = []
        tensors: list[KVCacheTensor] = []

        blocks = bc.num_kv_cache_blocks

        # 1. SWA main KV — DeepseekV4SWACache (attention.py:290)
        #    SlidingWindowMLASpec(bs=64, head_size=512, sw=window_size)
        #    Created for ALL 43 layers unconditionally.
        swa_spec = SlidingWindowMLASpec(
            block_size=64,
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
            tensors.append(KVCacheTensor(
                size=swa_spec.page_size_bytes * blocks, shared_by=[name]))

        # 2. C4 Compressor — CompressorStateCache (compressor.py:150,157-169)
        #    SlidingWindowMLASpec(bs=4, sw=8, head_size=state_dim, dtype=float32)
        if c4_layers:
            coff = 2  # compressor.py:141  coff = 1 + (compress_ratio == 4)
            c4_cmpr_state_dim = 2 * coff * arch.head_size  # 2*2*512 = 2048
            c4_cmpr_spec = SlidingWindowMLASpec(
                block_size=4,
                num_kv_heads=1,
                head_size=c4_cmpr_state_dim,
                dtype=_torch.float32,                sliding_window=coff * 4,  # 8
                alignment=576 if cache_dtype_str == "fp8_ds_mla" else None,
            )
            c4_cmpr_names = [f"model.layers.{i}.self_attn.compressor" for i in c4_layers]
            groups.append(KVCacheGroupSpec(c4_cmpr_names, c4_cmpr_spec))
            for name in c4_cmpr_names:
                tensors.append(KVCacheTensor(
                    size=c4_cmpr_spec.page_size_bytes * blocks, shared_by=[name]))

        # 3. C128 Compressor — CompressorStateCache (compressor.py:152,157-169)
        #    SlidingWindowMLASpec(bs=8, sw=128, head_size=state_dim, dtype=float32)
        if c128_layers:
            coff = 1  # compressor.py:141  coff = 1 + (compress_ratio == 4)
            c128_cmpr_state_dim = 2 * coff * arch.head_size  # 2*1*512 = 1024
            c128_cmpr_spec = SlidingWindowMLASpec(
                block_size=8,
                num_kv_heads=1,
                head_size=c128_cmpr_state_dim,
                dtype=_torch.float32,                sliding_window=coff * 128,  # 128
                alignment=576 if cache_dtype_str == "fp8_ds_mla" else None,
            )
            c128_cmpr_names = [f"model.layers.{i}.self_attn.compressor" for i in c128_layers]
            groups.append(KVCacheGroupSpec(c128_cmpr_names, c128_cmpr_spec))
            for name in c128_cmpr_names:
                tensors.append(KVCacheTensor(
                    size=c128_cmpr_spec.page_size_bytes * blocks, shared_by=[name]))

        # 4. Main MLA — DeepseekV4Attention (attention.py:601-619)
        #    MLAAttentionSpec(bs=256, cr=4 or 128), only for cr>1
        if c4_layers:
            c4_mla_spec = MLAAttentionSpec(
                block_size=256, num_kv_heads=arch.num_kv_heads,
                head_size=arch.head_size, dtype=kv_dtype,
                compress_ratio=4, cache_dtype_str=cache_dtype_str,
                alignment=576 if cache_dtype_str == "fp8_ds_mla" else None,
                model_version=bc.model_version,
            )
            c4_mla_names = [f"model.layers.{i}.self_attn" for i in c4_layers]
            groups.append(KVCacheGroupSpec(c4_mla_names, c4_mla_spec))
            for name in c4_mla_names:
                tensors.append(KVCacheTensor(
                    size=c4_mla_spec.page_size_bytes * blocks, shared_by=[name]))

        if c128_layers:
            c128_mla_spec = MLAAttentionSpec(
                block_size=256, num_kv_heads=arch.num_kv_heads,
                head_size=arch.head_size, dtype=kv_dtype,
                compress_ratio=128, cache_dtype_str=cache_dtype_str,
                alignment=576 if cache_dtype_str == "fp8_ds_mla" else None,
                model_version=bc.model_version,
            )
            c128_mla_names = [f"model.layers.{i}.self_attn" for i in c128_layers]
            groups.append(KVCacheGroupSpec(c128_mla_names, c128_mla_spec))
            for name in c128_mla_names:
                tensors.append(KVCacheTensor(
                    size=c128_mla_spec.page_size_bytes * blocks, shared_by=[name]))

        # 5. C4 Indexer — DeepseekV4IndexerCache (attention.py:643-655)
        #    MLAAttentionSpec(alignment=576), only for cr=4
        if c4_layers:
            # indexer_head_dim = 128; k_cache_head_dim = 128 + 4 = 132 bytes/token
            indexer_head_dim = 128 + 128 // 128 * 4  # ≈ 132
            # DeepseekV4IndexerCache.get_kv_cache_spec (attention.py:643-655)
            # does NOT set model_version or cache_dtype_str
            c4_idx_spec = MLAAttentionSpec(
                block_size=256, num_kv_heads=1,
                head_size=indexer_head_dim, dtype=_torch.uint8,
                compress_ratio=4, cache_dtype_str=None,
                alignment=576,
            )
            c4_idx_names = [f"model.layers.{i}.self_attn.k_cache" for i in c4_layers]
            groups.append(KVCacheGroupSpec(c4_idx_names, c4_idx_spec))
            for name in c4_idx_names:
                tensors.append(KVCacheTensor(
                    size=c4_idx_spec.page_size_bytes * blocks, shared_by=[name]))

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
          matching alignment.
        - Mock pool size = num_blocks × scheduler_block_size — models a
          single shared block budget (matching vLLM's single BlockPool).
          Groups don't each get their own full quota; the coordinator
          splits the shared budget internally.
        """
        arch = bc.model_arch
        page_size = bc.block_size

        # Use the scheduler-level block budget as the mock pool capacity.
        # Each scheduler_block_size of tokens counts as one "block equivalent".
        # The real vLLM HybridKVCacheCoordinator splits num_blocks across
        # groups — we model that as one flat pool of num_blocks *
        # scheduler_block_size token slots.
        total_tokens = bc.num_kv_cache_blocks * bc.scheduler_block_size

        return cls(page_size=page_size, total_tokens=total_tokens)
