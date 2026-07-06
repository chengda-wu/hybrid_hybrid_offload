"""DSV4 KV cache layout probe — reproduce the numbers in
DSV4_KV_CACHE_MANAGEMENT.md using vLLM's real grouping functions.

This is NOT a simulator. It builds the same KVCacheSpec dict that DSV4's
attention/compressor/indexer modules produce, then feeds it through vLLM's
actual get_kv_cache_groups + _bucket_layers_by_page_size +
_get_kv_cache_config_packed. All group/bucket/slot/bytes values in the doc
come from running this script.

Usage:
    cd /home/witcher/hybrid_hybrid_offload
    .venv/bin/python dsv4_layout.py

No GPU needed — only constructs spec dataclasses and runs the layout planner.

DSV4-Flash config (from HF config.json):
  num_hidden_layers=43, head_dim=512, qk_rope_head_dim=64, sliding_window=128
  compress_ratios=[0,0,4,128,4,128,...,4]  (2 SWA-only, 21 C4, 20 C128)
  block_size=256, cache_dtype=fp8_ds_mla
"""
from collections import Counter
from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_utils import (
    _bucket_layers_by_page_size,
    _get_kv_cache_config_packed,
    get_kv_cache_groups,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

# ---- DSV4-Flash real config -------------------------------------------------
NUM_LAYERS = 43
COMPRESS_RATIOS = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(41)]
SLIDING_WINDOW = 128
HEAD_DIM = 512
INDEX_HEAD_DIM = 128  # -> k_cache_head_dim = 128 + 4 = 132
BLOCK_SIZE = 256  # cache_config.block_size
CACHE_DTYPE = "fp8_ds_mla"

assert len(COMPRESS_RATIOS) == NUM_LAYERS
C4_LAYERS = [i for i, cr in enumerate(COMPRESS_RATIOS) if cr == 4]
C128_LAYERS = [i for i, cr in enumerate(COMPRESS_RATIOS) if cr == 128]
SWA_ONLY = [i for i, cr in enumerate(COMPRESS_RATIOS) if cr <= 1]


def _swa(block_size, sw, head_size, dtype, model_version="deepseek_v4"):
    """SlidingWindowMLASpec — SWA cache and compressor state caches.

    SWA cache sets model_version="deepseek_v4" (sparse_swa.py:93); compressor
    state caches leave it at the default None (compressor.py:157-169). Both
    set alignment=576. cache_dtype_str is fp8_ds_mla for the SWA uint8 cache
    and None for the fp32 compressor states.
    """
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        sliding_window=sw,
        cache_dtype_str=CACHE_DTYPE if dtype == torch.uint8 else None,
        alignment=576,
        model_version=model_version,
    )


def _mla(cr, head_size, dtype, cache_dtype_str=CACHE_DTYPE, model_version="deepseek_v4"):
    """MLAAttentionSpec — main MLA and indexer K caches.

    Main MLA sets model_version="deepseek_v4" + cache_dtype_str=fp8_ds_mla
    (attention.py:601-619). Indexer K cache leaves both at default None
    (attention.py:643-655) — it uses the element-size page formula, not the
    fp8_ds_mla 584B path, but head_size=132 already encodes the scale bytes.
    """
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        compress_ratio=cr,
        cache_dtype_str=cache_dtype_str,
        alignment=576,
        model_version=model_version,
    )


def build_specs() -> dict:
    """Build the full KVCacheSpec dict the way DSV4 modules would produce.

    Mirrors get_kv_cache_spec collection in gpu_model_runner.py:7482 — every
    AttentionLayerBase submodule (swa_cache_layer, compressor.state_cache,
    indexer.k_cache, indexer.compressor.state_cache, attention itself)
    contributes one spec.
    """
    specs = {}
    # 1. SWA cache on ALL 43 layers (sparse_swa.py:81): model_version=deepseek_v4
    for i in range(NUM_LAYERS):
        specs[f"L{i}.swa"] = _swa(64, SLIDING_WINDOW, HEAD_DIM, torch.uint8)
    # 2. C4 main compressor (state_dim=2*coff*512=2048) + indexer compressor
    #    (state_dim=2*coff*128=512); both bs=4, sw=8 (compressor.py:142).
    #    Compressor specs leave model_version=None (compressor.py:157-169).
    for i in C4_LAYERS:
        specs[f"L{i}.c4_comp"] = _swa(4, 8, 2048, torch.float32, model_version=None)
        specs[f"L{i}.c4_idx_comp"] = _swa(4, 8, 512, torch.float32, model_version=None)
    # 3. C128 compressor (state_dim=1024, bs=8, sw=128)
    for i in C128_LAYERS:
        specs[f"L{i}.c128_comp"] = _swa(8, SLIDING_WINDOW, 1024, torch.float32, model_version=None)
    # 4. C4 main MLA (cr=4, attention.py:601) + indexer k_cache (head_dim=132,
    #    attention.py:643-655). Indexer leaves model_version/cache_dtype_str=None.
    for i in C4_LAYERS:
        specs[f"L{i}.c4_mla"] = _mla(4, HEAD_DIM, torch.uint8)
        specs[f"L{i}.c4_idx"] = _mla(4, 132, torch.uint8, cache_dtype_str=None, model_version=None)
    # 5. C128 main MLA (cr=128)
    for i in C128_LAYERS:
        specs[f"L{i}.c128_mla"] = _mla(128, HEAD_DIM, torch.uint8)
    return specs


def main() -> None:
    print(
        f"layers: SWA-only={len(SWA_ONLY)} C4={len(C4_LAYERS)} "
        f"C128={len(C128_LAYERS)} total={NUM_LAYERS}"
    )

    specs = build_specs()
    print(f"\n#total specs collected: {len(specs)}")
    print("spec type counts:", Counter(type(s).__name__ for s in specs.values()))

    print("\n--- per-spec page_size_bytes (after 576 alignment padding) ---")
    by_ps = Counter()
    for s in specs.values():
        by_ps[s.page_size_bytes] += 1
    for ps, n in sorted(by_ps.items()):
        print(f"  page_size={ps:>7}  count={n}")

    # Minimal VllmConfig-like object for the layout functions.
    vllm_cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(
            block_size=BLOCK_SIZE,
            cache_dtype=CACHE_DTYPE,
            hash_block_size=None,
            num_gpu_blocks_override=None,
            enable_prefix_caching=True,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )

    # --- Step 1: group_and_unify + _get_kv_cache_groups_uniform_groups ---
    groups = get_kv_cache_groups(vllm_cfg, specs)
    print(f"\n--- get_kv_cache_groups -> {len(groups)} KVCacheGroupSpec ---")
    for gi, g in enumerate(groups):
        sp = g.kv_cache_spec
        if isinstance(sp, UniformTypeKVCacheSpecs):
            pss = sorted({s.page_size_bytes for s in sp.kv_cache_specs.values()})
            print(
                f"  group {gi}: UniformType block_size={sp.block_size} "
                f"nlayers={len(g.layer_names)} page_sizes={pss}"
            )
        else:
            print(
                f"  group {gi}: single block_size={sp.block_size} "
                f"nlayers={len(g.layer_names)} page_size={sp.page_size_bytes}"
            )

    # --- Step 2: bucket by page_size ---
    buckets = _bucket_layers_by_page_size(groups)
    print(f"\n--- _bucket_layers_by_page_size -> {len(buckets)} buckets ---")
    total_per_block = 0
    for ps, slots in sorted(buckets.items()):
        print(
            f"  ps={ps:>7}  slot_count={len(slots)}  "
            f"layers_per_slot={[len(s) for s in slots]}"
        )
        total_per_block += ps * len(slots)
    print(f"  bytes_per_block = {total_per_block}")

    # --- Step 2b: per-group fill rate when it owns one block ---
    # A group owns a block -> all its layers write 1 slot each into that block
    # (same-group layers share one block_table, gpu_model_runner.py:2466).
    # Fill bytes = sum over the group's layers of page_size_bytes.
    print(f"\n--- per-group fill rate (block owned = all group layers write 1 slot) ---")
    from collections import Counter as _Counter

    for gi, g in enumerate(groups):
        sp = g.kv_cache_spec
        if isinstance(sp, UniformTypeKVCacheSpecs):
            ps_count = _Counter(s.page_size_bytes for s in sp.kv_cache_specs.values())
            filled = sum(c * ps for ps, c in ps_count.items())
            detail = " + ".join(f"{c}×{ps}" for ps, c in sorted(ps_count.items()))
        else:
            filled = sp.page_size_bytes
            detail = f"1×{sp.page_size_bytes}"
        fill_pct = filled / total_per_block * 100
        print(
            f"  G{gi}: {filled:>8,} B / {total_per_block:,} = {fill_pct:5.1f}%  ({detail})"
        )

    # --- Step 3: scheduler / hash block size ---
    kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    sched_bs, hash_bs = resolve_kv_cache_block_sizes(kv_cache_config, vllm_cfg)
    print(f"\n  scheduler_block_size={sched_bs}  hash_block_size={hash_bs}")

    # --- Step 4: packed tensor plan ---
    available = 1000 * total_per_block
    num_blocks, tensors = _get_kv_cache_config_packed(
        vllm_cfg, groups, available
    )
    print(f"\n--- _get_kv_cache_config_packed ---")
    print(f"  num_blocks={num_blocks}  num_tensors={len(tensors)}")
    print(f"  tensor[0]: offset={tensors[0].offset} "
          f"stride={tensors[0].block_stride} shared_by={len(tensors[0].shared_by)} layers")
    print(f"  all {len(tensors)} tensors share one backing (block_stride > 0)")


if __name__ == "__main__":
    main()
