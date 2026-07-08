"""vLLM KVCacheGroupSpec layer-name contract.

Real vLLM names DSV4 attention modules ``model.layers.{N}.attn`` (NOT
``.self_attn``) and keys sub-caches by their real layer index N (from
``extract_layer_index(prefix)``, model.py:558,804), not a sequential 0..k-1
bucket index.  ``_bucket_layers_by_page_size`` only consumes ``len(layer_names)``
so sizing is unaffected, but the names must be correct so a future code path
that binds them as dict keys (init_attn_backend / _reshape_kv_cache /
bind_kv_cache) does not KeyError.
"""

import importlib.util
import math
import unittest

_HAS_VLLM = importlib.util.find_spec("vllm") is not None
requires_vllm = unittest.skipUnless(_HAS_VLLM, "requires vllm")

from simulator.config.model_config import KVBackendConfig, ModelArchitecture
from simulator.config.vllm_config import VLLMConfig


def _make_bc(num_spec=0):
    arch = ModelArchitecture.deepseek_v4_flash()
    bsizes = [g[1] for g in arch.layer_groups]
    lcm = bsizes[0]
    for bs in bsizes[1:]:
        lcm = lcm * bs // math.gcd(lcm, bs)
    return KVBackendConfig(
        model_arch=arch,
        block_size=max(bsizes),
        hash_block_size=bsizes[0],
        max_model_len=8192,
        num_kv_cache_blocks=4096,
        scheduler_block_size=lcm,
        num_spec_tokens=num_spec,
    )


@requires_vllm
class TestLayerNames(unittest.TestCase):
    def setUp(self):
        self.arch = ModelArchitecture.deepseek_v4_flash()
        self.ratios = self.arch.compress_ratios
        self.c4_idx = [i for i, cr in enumerate(self.ratios) if cr == 4]
        self.c128_idx = [i for i, cr in enumerate(self.ratios) if cr == 128]

    def _groups(self, num_spec=0):
        bc = _make_bc(num_spec=num_spec)
        return VLLMConfig._build_vllm_specs(bc)

    def _by_spec_type(self, groups):
        # group by the spec class name suffix
        out = {}
        for g in groups:
            key = type(g.kv_cache_spec).__name__
            out.setdefault(key, []).append(g)
        return out

    def test_swa_uses_attn_prefix_and_all_layer_indices(self):
        groups = self._groups()
        # SWA is the SlidingWindowMLASpec with the model's full head_size
        # (compressors use head_size 2048/1024, SWA uses arch.head_size=512).
        swa_groups = [
            g for g in groups
            if type(g.kv_cache_spec).__name__ == "SlidingWindowMLASpec"
            and g.kv_cache_spec.head_size == self.arch.head_size
        ]
        self.assertEqual(len(swa_groups), 1)
        names = swa_groups[0].layer_names
        # All 43 layers + 0 MTP (spec off here).
        self.assertEqual(len(names), self.arch.num_layers)
        for i, n in enumerate(names):
            self.assertEqual(n, f"model.layers.{i}.attn.swa_cache",
                             f"SWA name {n!r} at {i} must use .attn.swa_cache")
        # Real layer indices, not bucket indices.
        self.assertEqual([int(n.split(".")[2]) for n in names],
                         list(range(self.arch.num_layers)))

    def test_swa_includes_mtp_layer_when_spec_on(self):
        groups = self._groups(num_spec=2)
        swa_groups = [
            g for g in groups
            if type(g.kv_cache_spec).__name__ == "SlidingWindowMLASpec"
            and g.kv_cache_spec.head_size == self.arch.head_size
        ]
        names = swa_groups[0].layer_names
        # 43 target + 1 MTP draft layer (num_mtp_layers=1 regardless of K).
        self.assertEqual(len(names), self.arch.num_layers + 1)
        self.assertTrue(names[-1].startswith(f"model.layers.{self.arch.num_layers}."))

    def test_compressor_uses_real_c4_c128_indices(self):
        groups = self._groups()
        comp_groups = [
            g for g in groups
            if type(g.kv_cache_spec).__name__ == "SlidingWindowMLASpec"
            and g.kv_cache_spec.sliding_window in (8, 128)
            and g.kv_cache_spec.head_size in (2048, 1024)
        ]
        by_head = {g.kv_cache_spec.head_size: g for g in comp_groups}
        # C4 compressor (head_size=2048) keyed by real c4 layer indices.
        c4_names = by_head[2048].layer_names
        self.assertEqual(len(c4_names), len(self.c4_idx))
        for i, n in zip(self.c4_idx, c4_names):
            self.assertEqual(n, f"model.layers.{i}.attn.compressor")
        # C128 compressor (head_size=1024) keyed by real c128 layer indices.
        c128_names = by_head[1024].layer_names
        self.assertEqual(len(c128_names), len(self.c128_idx))
        for i, n in zip(self.c128_idx, c128_names):
            self.assertEqual(n, f"model.layers.{i}.attn.compressor")

    def test_main_mla_uses_real_indices_and_attn_prefix(self):
        groups = self._groups()
        mla_groups = [
            g for g in groups
            if type(g.kv_cache_spec).__name__ == "MLAAttentionSpec"
            and g.kv_cache_spec.head_size == self.arch.head_size
        ]
        by_cr = {g.kv_cache_spec.compress_ratio: g for g in mla_groups}
        c4 = by_cr[4].layer_names
        c128 = by_cr[128].layer_names
        for i, n in zip(self.c4_idx, c4):
            self.assertEqual(n, f"model.layers.{i}.attn")
        for i, n in zip(self.c128_idx, c128):
            self.assertEqual(n, f"model.layers.{i}.attn")
        # No .self_attn anywhere in MLA names.
        self.assertFalse(any(".self_attn" in n for n in c4 + c128))

    def test_indexer_uses_attn_indexer_and_real_c4_indices(self):
        groups = self._groups()
        indexer_groups = [
            g for g in groups
            if type(g.kv_cache_spec).__name__ == "MLAAttentionSpec"
            and g.kv_cache_spec.head_size == 132
        ]
        self.assertEqual(len(indexer_groups), 1)
        names = indexer_groups[0].layer_names
        self.assertEqual(len(names), len(self.c4_idx))
        for i, n in zip(self.c4_idx, names):
            self.assertEqual(n, f"model.layers.{i}.attn.indexer")

    def test_no_self_attn_in_dsv4_names(self):
        """DSV4 uses .attn, not .self_attn — guard against regression."""
        groups = self._groups()
        all_names = [n for g in groups for n in g.layer_names]
        # The 'full' group (non-DSV4) legitimately uses .self_attn, but DSV4
        # has no full group, so none should appear here.
        self.assertFalse(any(".self_attn" in n for n in all_names),
                         f"DSV4 layer names must use .attn, found .self_attn: "
                         f"{[n for n in all_names if '.self_attn' in n]}")

    def test_counts_unchanged_so_sizing_is_stable(self):
        """The fix must not change layer counts (only the name strings)."""
        groups = self._groups()
        total = sum(len(g.layer_names) for g in groups)
        # SWA(43) + c4_comp(21) + c128_comp(20) + c4_mla(21) + c128_mla(20)
        # + c4_indexer(21) = 146
        self.assertEqual(total, 43 + 21 + 20 + 21 + 20 + 21)


if __name__ == "__main__":
    unittest.main()
