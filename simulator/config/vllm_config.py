"""vLLM-specific KV cache config builder.

Imports from vllm.* — only loaded when backend='vllm'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from simulator.config.model_config import KVBackendConfig

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec


@dataclass
class VLLMConfig:
    """vLLM-specific config built from KVBackendConfig."""

    kv_cache_config: "KVCacheConfig"

    @staticmethod
    def _build_vllm_specs(bc: KVBackendConfig) -> list["KVCacheGroupSpec"]:
        """Convert framework-agnostic KVGroupInfo to vLLM KVCacheGroupSpecs."""
        import torch as _torch

        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheGroupSpec,
            MLAAttentionSpec,
            SlidingWindowMLASpec,
        )

        arch = bc.model_arch
        cache_dtype_str = "fp8_ds_mla" if arch.is_mla else None
        kv_dtype = _torch.uint8 if arch.is_mla else getattr(_torch, arch.dtype)

        # MTP draft layer count, from the model architecture
        # (config.num_nextn_predict_layers; DeepSeek V4 Flash = 1).  In real
        # vLLM each MTP draft layer is a DeepseekV4Attention with
        # compress_ratio=1, so it has no MLA / compressor / indexer — only a
        # DeepseekV4SWACache (sparse_swa.py:81) whose SlidingWindowMLASpec is
        # identical to the target SWA spec (same block_size/head_size/
        # sliding_window/alignment → same page_size).  Each MTP layer therefore
        # joins the SWA bucket, growing it by num_mtp_layers.  Draft layers
        # share the target block pool (llm_base_proposer.py asserts all draft
        # layers belong to one kv_cache_group), so no new group or pool is
        # added — only the SWA bucket's layer count rises.
        num_mtp_layers = arch.num_mtp_layers if bc.num_spec_tokens > 0 else 0

        # Actual layer indices per compress-ratio bucket, matching the HF
        # config's per-layer compress_ratios (e.g. DSV4: [0,0,4,128,4,128,...]).
        # Real vLLM names modules by their real layer index
        # (model.layers.{N}.attn, N from extract_layer_index(prefix),
        # model.py:558,804) — NOT a sequential 0..nlayers-1 bucket index.
        # _bucket_layers_by_page_size only consumes len(layer_names), so the
        # counts are what matter for sizing; but using the real indices and the
        # real ``.attn`` prefix (not ``.self_attn``) keeps these names valid as
        # dict keys should a future code path bind them (init_attn_backend /
        # _reshape_kv_cache / bind_kv_cache look up by name).
        ratios = arch.compress_ratios or []
        c4_idx = [i for i, cr in enumerate(ratios) if cr == 4]
        c128_idx = [i for i, cr in enumerate(ratios) if cr == 128]
        all_idx = list(range(arch.num_layers))

        groups: list[KVCacheGroupSpec] = []

        for info in bc.build_kv_cache_groups():
            name, bs, _page, nlayers = info.name, info.block_size, info.page_bytes, info.layer_count

            if name == "swa":
                # Target SWA layers (0..num_layers-1) + MTP draft SWA layer(s)
                # (num_layers..num_layers+num_mtp_layers-1) when spec is on.
                swa_layer_names = [
                    f"model.layers.{i}.attn.swa_cache"
                    for i in range(arch.num_layers + num_mtp_layers)
                ]
                groups.append(KVCacheGroupSpec(
                    swa_layer_names,
                    SlidingWindowMLASpec(
                        block_size=bs, num_kv_heads=arch.num_kv_heads,
                        head_size=arch.head_size, dtype=kv_dtype,
                        sliding_window=arch.sliding_window or 128,
                        cache_dtype_str=cache_dtype_str, alignment=576,
                        model_version=bc.model_version,
                    )))
            elif name == "c4_compressor":
                groups.append(KVCacheGroupSpec(
                    [f"model.layers.{i}.attn.compressor" for i in c4_idx],
                    SlidingWindowMLASpec(
                        block_size=bs, num_kv_heads=1,
                        head_size=2048, dtype=_torch.float32,
                        sliding_window=8, alignment=576,
                    )))
            elif name == "c128_compressor":
                groups.append(KVCacheGroupSpec(
                    [f"model.layers.{i}.attn.compressor" for i in c128_idx],
                    SlidingWindowMLASpec(
                        block_size=bs, num_kv_heads=1,
                        head_size=1024, dtype=_torch.float32,
                        sliding_window=128, alignment=576,
                    )))
            elif name in ("c4_mla", "c128_mla"):
                cr = 4 if name == "c4_mla" else 128
                idx = c4_idx if name == "c4_mla" else c128_idx
                groups.append(KVCacheGroupSpec(
                    [f"model.layers.{i}.attn" for i in idx],
                    MLAAttentionSpec(
                        block_size=bs, num_kv_heads=arch.num_kv_heads,
                        head_size=arch.head_size, dtype=kv_dtype,
                        compress_ratio=cr, cache_dtype_str=cache_dtype_str,
                        alignment=576, model_version=bc.model_version,
                    )))
            elif name == "c4_indexer":
                # vLLM indexer always allocates 132 B/token regardless of fp4/fp8
                # (attention.py:725-729 — fp4 uses same allocation, half unused)
                idx_hd = 132
                groups.append(KVCacheGroupSpec(
                    [f"model.layers.{i}.attn.indexer" for i in c4_idx],
                    MLAAttentionSpec(
                        block_size=bs, num_kv_heads=1,
                        head_size=idx_hd, dtype=_torch.uint8,
                        compress_ratio=4, cache_dtype_str=None,
                        alignment=576,
                    )))
            elif name == "full":
                groups.append(KVCacheGroupSpec(
                    [f"model.layers.{i}.self_attn" for i in all_idx],
                    FullAttentionSpec(
                        block_size=bs,
                        num_kv_heads=arch.num_kv_heads,
                        head_size=arch.head_size,
                        dtype=kv_dtype,
                        sliding_window=arch.sliding_window,
                    )))

        return groups

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

        # Build KVCacheGroupSpecs from framework-agnostic KVGroupInfo.
        # vLLM needs full spec objects for _get_kv_cache_config_packed.
        kv_cache_groups = cls._build_vllm_specs(bc)

        # Use vLLM's _bucket_layers_by_page_size + _get_kv_cache_config_packed
        # to compute correct tensor sizes for the shared block pool.
        buckets = _bucket_layers_by_page_size(kv_cache_groups)
        bytes_per_block = sum(ps * len(slots) for ps, slots in buckets.items())
        available_memory = bc.num_kv_cache_blocks * bytes_per_block

        # Build KVCacheConfig.  _vllm_config_ns provides a minimal
        # VllmConfig-like object for may_override_num_blocks (which only
        # reads cache_config.num_gpu_blocks_override).  See function docstring.
        num_blocks, kv_cache_tensors = _get_kv_cache_config_packed(
            _vllm_config_ns(bc.num_kv_cache_blocks),
            kv_cache_groups,
            available_memory,
        )

        return cls(
            kv_cache_config=KVCacheConfig(
                num_blocks=num_blocks,
                kv_cache_tensors=kv_cache_tensors,
                kv_cache_groups=kv_cache_groups,
            )
        )


def _vllm_config_ns(num_blocks: int):
    """Build a real, minimal VllmConfig for ``may_override_num_blocks``.

    We construct an actual ``vllm.config.VllmConfig`` (only
    cache_config.num_gpu_blocks_override set; model_config left at its default
    None) rather than a SimpleNamespace stand-in.  ``__post_init__`` early-
    returns when model_config is None (vllm.py:1893), so no model/parallel
    cross-validation runs; CacheConfig has no ``__post_init__``.  This is
    robust to vLLM upgrades: if a future version makes the call chain read
    additional vllm_config fields, the real object has them (with defaults)
    instead of raising AttributeError or silently returning None and
    miscounting blocks.

    Verified call path: ``from_backend_config`` calls
    ``_get_kv_cache_config_packed`` directly (vllm_config.py:153), which only
    touches ``cache_config.num_gpu_blocks_override`` via
    ``may_override_num_blocks`` (kv_cache_utils.py:940-947).  It does NOT route
    through ``get_kv_cache_config_from_groups`` / ``_use_packed_kv_cache_config``
    (which read parallel_config / kv_transfer_config / model_config) — but the
    real VllmConfig carries those too, so even that path change is safe.

    Built once per backend construction (not per request), so the cost of
    initializing the default sub-configs is negligible.
    """
    from vllm.config import CacheConfig, VllmConfig

    # model_config is omitted (NOT passed as None): VllmConfig is a pydantic
    # dataclass whose model_config field defaults to None, but explicitly
    # passing None trips pydantic's type validation (ModelConfig required).
    # Omitting it takes the default, which __post_init__ handles via its
    # `if self.model_config is None` early-return (vllm.py:1893).
    return VllmConfig(
        cache_config=CacheConfig(num_gpu_blocks_override=num_blocks),
    )
