"""Convert HuggingFace config.json into KV cache backend configurations.

Supports both vLLM and SGLang backends. DeepSeek V4 Flash is the primary
target with hardcoded defaults; generic models are supported via config.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

    # MTP (Multi-Token Prediction) draft layers.
    # DeepSeek V4 Flash has 1 (config.num_nextn_predict_layers).  Only the
    # vLLM backend uses this — each MTP draft layer adds one SWA-cache layer
    # to the shared block pool (compress_ratio=1 → no MLA/compressor/indexer).
    # The SGLang backend ignores it: real SGLang hardcodes draft_layers=1 in
    # its (T+D)/T inflation (pool_configurator.py:543), so the simulator
    # matches that regardless of this field.
    num_mtp_layers: int = 1

    # KV options
    use_fp4_indexer: bool = False  # deepseek_v4_memory_pool.py:279-282
    # DSV4 indexer head dim (HF config ``index_head_dim``; 128 for V4 Flash).
    # Real SGLang reads it at pool_configurator.py:503; vLLM derives 132 B/token
    # from it (128 + 128//128*4).  Defaults to 128 so DSV4 works without HF cfg.
    indexer_head_dim: int = 128

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
        # NOTE: compress_ratios is NOT filled here when None.  A non-DSV4 MLA
        # config (e.g. DSV2/V3) has kv_lora_rank but no compress_ratios —
        # filling [compress_ratio]*N (= [1]*N) would set is_mla=True yet
        # layer_groups would produce only an SWA group (no c4/c128/main MLA),
        # a contradictory hybrid layout.  Leaving it None lets layer_groups /
        # build_kv_cache_groups / engine all take the uniform "full" fallback
        # (they guard on ``compress_ratios is None``), which is the safe
        # behavior for an unsupported non-DSV4 model.  All readers also use
        # ``compress_ratios or []`` so None is handled everywhere.

    @property
    def kv_bytes_per_token(self) -> int:
        """Per-token KV bytes for the MLA fp8_ds_mla (UE8M0) layout.

        = ``qk_nope + qk_rope*2 + 8`` (SGLang ``pool_configurator.py:578``;
        vLLM uses the same cost).  Since ``qk_nope = head_size - qk_rope``,
        this simplifies to ``head_size + qk_rope + 8``.  For DSV4 Flash:
        512 + 64 + 8 = 584.  Deriving (not hardcoding 584) keeps the cost
        correct for any future MLA model with a different RoPE head dim.
        """
        # ``is not None`` (not ``or``) so a hypothetical qk_rope_head_dim=0 is
        # respected rather than falling back to 64.  In practice this property
        # is only called on the MLA path, where __post_init__ guarantees
        # qk_rope_head_dim is not None, so the fallback never fires — but the
        # null-coalescing matches the kv_lora_rank handling below.
        qk_rope = self.qk_rope_head_dim if self.qk_rope_head_dim is not None else 64
        return self.head_size + qk_rope + 8

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

        # Unreachable: the SWA group (step 1) is appended unconditionally for
        # MLA models, and non-MLA models return early above.  Assert so a
        # future change that makes the SWA append conditional surfaces an
        # empty-groups bug loudly.
        assert groups, "layer_groups must not be empty for an MLA model"
        return groups

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelArchitecture":
        """Load from a HuggingFace config.json file."""
        with open(path) as f:
            cfg = json.load(f)

        # ``cfg.get("architectures", ["unknown"])[0]`` would IndexError on an
        # empty architectures list (key present, default not used); guard with
        # ``or`` so a missing model_type falls back to the first architecture
        # or "unknown".
        model_type = (
            cfg.get("model_type")
            or (cfg.get("architectures") or ["unknown"])[0]
        )

        # Head size detection
        head_size = cfg.get("head_dim", 0)
        if not head_size:
            nope = cfg.get("qk_nope_head_dim", 0)
            rope = cfg.get("qk_rope_head_dim", 0)
            # ``or`` (not ``and``): a config with qk_nope_head_dim but a
            # missing/zero qk_rope_head_dim should still use nope (+0) rather
            # than falling through to hidden_size//num_attention_heads.
            if nope or rope:
                head_size = nope + rope
        if not head_size:
            head_size = cfg.get("hidden_size", 4096) // cfg.get(
                "num_attention_heads", 32
            )

        # MLA flag: DSV2/V3 use ``kv_lora_rank``; DSV4 renamed it to
        # ``q_lora_rank`` (the V4 config has no ``kv_lora_rank`` key, so reading
        # only that key left kv_lora_rank=None → is_mla stayed False → the
        # hybrid layer_groups collapsed to a single "full" group).  Accept
        # either key; the value is only used as a non-None MLA sentinel (it
        # never enters the per-token byte math).
        # ``or`` would treat a legitimate kv_lora_rank=0 as falsy and fall
        # through to q_lora_rank; use ``is not None`` null-coalescing.  (The
        # value is only an MLA sentinel — it never enters byte math — so a 0
        # has no real-world impact, but the coalescing is still more correct.)
        kv = cfg.get("kv_lora_rank")
        kv_lora_rank = kv if kv is not None else cfg.get("q_lora_rank")
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
            use_fp4_indexer=cfg.get("enable_deepseek_v4_fp4_indexer", False),
            indexer_head_dim=cfg.get("index_head_dim", 128),
            # Default 0: num_nextn_predict_layers is DSV4-specific (DSV4=1).
            # A non-DSV4 config (e.g. DSV2/V3) lacks the key — falling back to
            # 1 would add a phantom MTP SWA layer when spec is on
            # (vllm_config.py: num_mtp_layers applies when num_spec_tokens>0).
            # deepseek_v4_flash() sets this explicitly to 1, and a real DSV4
            # config.json carries num_nextn_predict_layers=1, so DSV4 is
            # unaffected; only non-DSV4 models rely on this default.
            num_mtp_layers=cfg.get("num_nextn_predict_layers", 0),
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
    model_version: str = "deepseek_v4"
    num_spec_tokens: int = 0
    # SGLang DSV4 SWA/full token ratio (deepseek_v4_hook.py:57 overrides the
    # SGLang default 0.0 to 0.1).  Exposed so the SWA pool share is tunable
    # without editing source; the configurator reads it via the mock below.
    swa_full_tokens_ratio: float = 0.1

    @property
    def num_kv_cache_groups(self) -> int:
        """Number of KV cache groups (derived from layer_groups)."""
        return len(self.model_arch.layer_groups)

    def build_kv_cache_groups(self) -> list["KVGroupInfo"]:
        """Build framework-agnostic KV group descriptions.

        Returns KVGroupInfo with unpadded page_bytes — no vllm or sglang types.
        vLLM backend converts to KVCacheGroupSpecs; SGLang reads page_bytes directly.
        """
        return _build_kv_cache_groups(self)


# ---------------------------------------------------------------------------
# Framework-agnostic KV group info (shared by vLLM and SGLang)
# ---------------------------------------------------------------------------


@dataclass
class KVGroupInfo:
    """Framework-agnostic description of one KV cache group.

    Contains just the information both backends need — no vllm or sglang types.
    vLLM converts these to KVCacheGroupSpecs; SGLang reads page_bytes directly.
    """

    name: str           # "swa", "c4_compressor", etc.
    block_size: int     # tokens per block
    page_bytes: int     # unpadded bytes per page (no vLLM alignment)
    layer_count: int    # number of layers in this group


def _build_kv_cache_groups(bc: KVBackendConfig) -> list[KVGroupInfo]:
    """Build framework-agnostic KV group descriptions.

    Returns KVGroupInfo with unpadded page_bytes — no vllm or sglang types.
    Each backend converts these to its own representation.
    """
    arch = bc.model_arch

    if not arch.is_mla or arch.compress_ratios is None:
        # Uniform model: page = 2 * block_size * num_kv_heads * head_size * dtype
        dtype_size = 2  # bf16
        page = 2 * bc.block_size * arch.num_kv_heads * arch.head_size * dtype_size
        return [KVGroupInfo("full", bc.block_size, page, arch.num_layers)]

    ratios = arch.compress_ratios
    c4_layers = [i for i, cr in enumerate(ratios) if cr == 4]
    c128_layers = [i for i, cr in enumerate(ratios) if cr == 128]
    groups: list[KVGroupInfo] = []

    # 1. SWA (attention.py:290), all layers
    kv_bytes = arch.kv_bytes_per_token  # qk_nope + qk_rope*2 + 8 (584 for DSV4)
    groups.append(KVGroupInfo("swa", 64, 64 * kv_bytes, arch.num_layers))

    # 2. C4 Compressor (compressor.py:150): bs=4, float32, state_dim=2048
    if c4_layers:
        groups.append(KVGroupInfo(
            "c4_compressor", 4, 4 * 2048 * 4, len(c4_layers)))

    # 3. C128 Compressor (compressor.py:152): bs=8, float32, state_dim=1024
    if c128_layers:
        groups.append(KVGroupInfo(
            "c128_compressor", 8, 8 * 1024 * 4, len(c128_layers)))

    # 4. C4 Main MLA (attention.py:601): bs=256, cr=4 -> 64 effective tok/block
    if c4_layers:
        groups.append(KVGroupInfo("c4_mla", 256, 64 * kv_bytes, len(c4_layers)))

    # 5. C128 Main MLA: bs=256, cr=128 -> 2 effective tok/block
    if c128_layers:
        groups.append(KVGroupInfo("c128_mla", 256, 2 * kv_bytes, len(c128_layers)))

    # 6. C4 Indexer (attention.py:643-655): 132 B/token
    if c4_layers:
        # Indexer per-token bytes.  SGLang: 132 (fp8) or 68 (fp4) — fp4 uses
        # index_head_dim//2 + 4 (deepseek_v4_memory_pool.py:279-282).  vLLM:
        # ALWAYS 132 even in fp4 mode (attention.py:725-729 — "FP4 indexer
        # cache still allocates the same amount of memory as FP8, but only
        # uses the first half").  This KVGroupInfo is the shared model
        # description; its page_bytes is accurate for SGLang's total_bytes.
        # The vLLM backend ignores it (VLLMConfig._build_vllm_specs hardcodes
        # idx_hd=132, see vllm_config.py:101-103), so vLLM is unaffected —
        # but do NOT use KVGroupInfo.page_bytes for cross-backend comparison
        # of the indexer in fp4 mode.
        idx_bytes = 68 if arch.use_fp4_indexer else 132
        groups.append(KVGroupInfo("c4_indexer", 256, 64 * idx_bytes, len(c4_layers)))

    return groups


