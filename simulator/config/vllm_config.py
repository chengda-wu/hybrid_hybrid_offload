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

        groups: list[KVCacheGroupSpec] = []

        for info in bc.build_kv_cache_groups():
            name, bs, _page, nlayers = info.name, info.block_size, info.page_bytes, info.layer_count
            layer_names = [f"model.layers.{i}.self_attn" for i in range(nlayers)]

            if name == "swa":
                groups.append(KVCacheGroupSpec(
                    [f"model.layers.{i}.self_attn" for i in range(arch.num_layers)],
                    SlidingWindowMLASpec(
                        block_size=bs, num_kv_heads=arch.num_kv_heads,
                        head_size=arch.head_size, dtype=kv_dtype,
                        sliding_window=arch.sliding_window or 128,
                        cache_dtype_str=cache_dtype_str, alignment=576,
                        model_version=bc.model_version,
                    )))
            elif name == "c4_compressor":
                groups.append(KVCacheGroupSpec(
                    [f"model.layers.{i}.self_attn.compressor" for i in range(nlayers)],
                    SlidingWindowMLASpec(
                        block_size=bs, num_kv_heads=1,
                        head_size=2048, dtype=_torch.float32,
                        sliding_window=8, alignment=576,
                    )))
            elif name == "c128_compressor":
                groups.append(KVCacheGroupSpec(
                    [f"model.layers.{i}.self_attn.compressor" for i in range(nlayers)],
                    SlidingWindowMLASpec(
                        block_size=bs, num_kv_heads=1,
                        head_size=1024, dtype=_torch.float32,
                        sliding_window=128, alignment=576,
                    )))
            elif name in ("c4_mla", "c128_mla"):
                cr = 4 if name == "c4_mla" else 128
                groups.append(KVCacheGroupSpec(
                    [f"model.layers.{i}.self_attn" for i in range(nlayers)],
                    MLAAttentionSpec(
                        block_size=bs, num_kv_heads=arch.num_kv_heads,
                        head_size=arch.head_size, dtype=kv_dtype,
                        compress_ratio=cr, cache_dtype_str=cache_dtype_str,
                        alignment=576, model_version=bc.model_version,
                    )))
            elif name == "c4_indexer":
                groups.append(KVCacheGroupSpec(
                    [f"model.layers.{i}.self_attn.k_cache" for i in range(nlayers)],
                    MLAAttentionSpec(
                        block_size=bs, num_kv_heads=1,
                        head_size=132, dtype=_torch.uint8,
                        compress_ratio=4, cache_dtype_str=None,
                        alignment=576,
                    )))
            elif name == "full":
                groups.append(KVCacheGroupSpec(
                    layer_names,
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

        # Build the KVCacheConfig.  We control num_blocks and tensor layout
        # is handled by _get_kv_cache_config_packed — no need for a real
        # VllmConfig or SimpleNamespace workaround.
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
    """Minimal VllmConfig-like object for may_override_num_blocks.

    Isolated helper — replace with a real VllmConfig if may_override_num_blocks
    ever reads beyond cache_config.num_gpu_blocks_override.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=num_blocks),
    )
