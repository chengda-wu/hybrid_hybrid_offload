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
    # NOTE: SGLang's model_config.py:783 labels DeepSeek V4 as AttentionArch.MHA,
    # but the actual memory layout (584 B/token, compress_ratios, FlashMLA backend)
    # is MLA.  Our simulation uses is_mla=True regardless of backend — the per-token
    # byte cost and page sizes are the same.

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
        """Number of KV cache groups (derived from layer_groups)."""
        return len(self.model_arch.layer_groups)

    def build_kv_cache_groups(self) -> list["KVCacheGroupSpec"]:
        """Build KVCacheGroupSpecs from the model architecture.

        Shared by both vLLM and SGLang backends — this is a model description,
        not framework-specific logic.
        """
        return _build_kv_cache_groups(self)


# ---------------------------------------------------------------------------
# vLLM-specific config builder
# ---------------------------------------------------------------------------


@dataclass
class VLLMConfig:
    """vLLM-specific config built from KVBackendConfig."""

    kv_cache_config: "KVCacheConfig"

    @classmethod
    def from_backend_config(cls, bc: KVBackendConfig) -> "VLLMConfig":
        """Build KVCacheConfig by calling vLLM's own tensor layout logic.

        Delegates to vLLM's ``_get_kv_cache_config_packed`` for hybrid models,
        or ``get_kv_cache_config_from_groups`` for uniform models, so that
        block counts and tensor sizes are computed correctly by vLLM itself.
        """

        from vllm.v1.core.kv_cache_utils import (
            _bucket_layers_by_page_size,
            _get_kv_cache_config_packed,
        )
        from vllm.v1.kv_cache_interface import KVCacheConfig

        # Build group specs (same as before — spec construction is our
        # responsibility since we're simulating a model, not loading one).
        kv_cache_groups = bc.build_kv_cache_groups()

        # Use vLLM's _bucket_layers_by_page_size + _get_kv_cache_config_packed
        # to compute correct tensor sizes for the shared block pool.
        buckets = _bucket_layers_by_page_size(kv_cache_groups)
        bytes_per_block = sum(ps * len(slots) for ps, slots in buckets.items())
        available_memory = bc.num_kv_cache_blocks * bytes_per_block

        # Build the KVCacheConfig.  _get_kv_cache_config_packed needs a
        # VllmConfig for may_override_num_blocks; we pass a minimal namespace
        # whose cache_config.num_gpu_blocks_override forces our block count.
        from types import SimpleNamespace
        vllm_ns = SimpleNamespace(
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=bc.num_kv_cache_blocks,
            ),
        )
        num_blocks, kv_cache_tensors = _get_kv_cache_config_packed(
            vllm_ns, kv_cache_groups, available_memory
        )

        return cls(
            kv_cache_config=KVCacheConfig(
                num_blocks=num_blocks,
                kv_cache_tensors=kv_cache_tensors,
                kv_cache_groups=kv_cache_groups,
            )
        )

def _build_kv_cache_groups(bc: KVBackendConfig) -> list["KVCacheGroupSpec"]:
    """Build KVCacheGroupSpecs from the model architecture.

    Shared by both vLLM and SGLang backends.  Tensor sizing is handled
    separately by each backend (vLLM delegates to _get_kv_cache_config_packed,
    SGLang uses page_size_bytes directly).
    """
    import torch as _torch

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
    )

    arch = bc.model_arch

    if not arch.is_mla or arch.compress_ratios is None:
        layer_names = [f"model.layers.{i}.self_attn" for i in range(arch.num_layers)]
        return [KVCacheGroupSpec(layer_names, FullAttentionSpec(
            block_size=bc.block_size,
            num_kv_heads=arch.num_kv_heads,
            head_size=arch.head_size,
            dtype=getattr(_torch, arch.dtype),
            sliding_window=arch.sliding_window,
        ))]

    kv_dtype = _torch.uint8
    cache_dtype_str = "fp8_ds_mla"
    ratios = arch.compress_ratios
    all_layers = list(range(arch.num_layers))
    c4_layers = [i for i, cr in enumerate(ratios) if cr == 4]
    c128_layers = [i for i, cr in enumerate(ratios) if cr == 128]
    groups: list[KVCacheGroupSpec] = []

    # 1. SWA (attention.py:290), all 43 layers
    groups.append(KVCacheGroupSpec(
        [f"model.layers.{i}.self_attn" for i in all_layers],
        SlidingWindowMLASpec(
            block_size=64, num_kv_heads=arch.num_kv_heads,
            head_size=arch.head_size, dtype=kv_dtype,
            sliding_window=arch.sliding_window or 128,
            cache_dtype_str=cache_dtype_str, alignment=576,
            model_version=bc.model_version,
        )))

    # 2. C4 Compressor (compressor.py:150,157-169)
    if c4_layers:
        coff = 2
        groups.append(KVCacheGroupSpec(
            [f"model.layers.{i}.self_attn.compressor" for i in c4_layers],
            SlidingWindowMLASpec(
                block_size=4, num_kv_heads=1,
                head_size=2 * coff * arch.head_size,
                dtype=_torch.float32,
                sliding_window=coff * 4, alignment=576,
            )))

    # 3. C128 Compressor (compressor.py:152,157-169)
    if c128_layers:
        coff = 1
        groups.append(KVCacheGroupSpec(
            [f"model.layers.{i}.self_attn.compressor" for i in c128_layers],
            SlidingWindowMLASpec(
                block_size=8, num_kv_heads=1,
                head_size=2 * coff * arch.head_size,
                dtype=_torch.float32,
                sliding_window=coff * 128, alignment=576,
            )))

    # 4. Main MLA C4/C128 (attention.py:601-619)
    if c4_layers:
        groups.append(KVCacheGroupSpec(
            [f"model.layers.{i}.self_attn" for i in c4_layers],
            MLAAttentionSpec(
                block_size=256, num_kv_heads=arch.num_kv_heads,
                head_size=arch.head_size, dtype=kv_dtype,
                compress_ratio=4, cache_dtype_str=cache_dtype_str,
                alignment=576, model_version=bc.model_version,
            )))
    if c128_layers:
        groups.append(KVCacheGroupSpec(
            [f"model.layers.{i}.self_attn" for i in c128_layers],
            MLAAttentionSpec(
                block_size=256, num_kv_heads=arch.num_kv_heads,
                head_size=arch.head_size, dtype=kv_dtype,
                compress_ratio=128, cache_dtype_str=cache_dtype_str,
                alignment=576, model_version=bc.model_version,
            )))

    # 5. C4 Indexer (attention.py:643-655, 729-732)
    if c4_layers:
        index_head_dim = 128
        quant_block_size = 128
        k_cache_head_dim = index_head_dim + index_head_dim // quant_block_size * 4
        groups.append(KVCacheGroupSpec(
            [f"model.layers.{i}.self_attn.k_cache" for i in c4_layers],
            MLAAttentionSpec(
                block_size=256, num_kv_heads=1,
                head_size=k_cache_head_dim, dtype=_torch.uint8,
                compress_ratio=4, cache_dtype_str=None,
                alignment=576,
            )))

    return groups


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
