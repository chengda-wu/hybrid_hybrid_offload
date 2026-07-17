"""Round-10 F1/F2 regression: config-surface fixes.

F1 — ``swa_full_tokens_ratio`` (was hardcoded 0.1 in the mock server_args) is
now exposed on ``SimulatorConfig``, parsed from JSON, overridable via the
``--swa-full-tokens-ratio`` CLI flag, and threaded into ``KVBackendConfig``.

F2 — ``from_json`` now cross-validates ``speculative.enabled`` vs
``num_spec_tokens``.  Pre-fix an inconsistent JSON like
``{"enabled": false, "num_spec_tokens": 2}`` inflated the KV pool (engine
reads num_spec_tokens) while never running speculation — a silent ~2.3%
over-allocation.  Post-fix ``enabled=False`` forces ``num_spec_tokens=0``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simulator.config.simulator_config import SimulatorConfig


def _write_json(obj: dict) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


class TestSwaRatioConfigurable(unittest.TestCase):
    """F1: swa_full_tokens_ratio flows through all three config surfaces."""

    def test_default_is_0_1(self):
        cfg = SimulatorConfig()
        self.assertEqual(cfg.swa_full_tokens_ratio, 0.1)

    def test_from_json_parses_swa_ratio(self):
        path = _write_json({"swa_full_tokens_ratio": 0.25})
        cfg = SimulatorConfig.from_json(path)
        self.assertAlmostEqual(cfg.swa_full_tokens_ratio, 0.25)
        Path(path).unlink()

    def test_from_json_omitted_defaults_to_0_1(self):
        path = _write_json({})
        cfg = SimulatorConfig.from_json(path)
        self.assertAlmostEqual(cfg.swa_full_tokens_ratio, 0.1)
        Path(path).unlink()


class TestSpecReconciliation(unittest.TestCase):
    """F2: enabled vs num_spec_tokens cross-validation in from_json."""

    def test_disabled_forces_num_spec_tokens_zero(self):
        # The bug case: enabled=false but num_spec_tokens=2.  Pre-fix this
        # inflated the KV pool while never running speculation.
        path = _write_json({
            "speculative": {"enabled": False, "num_spec_tokens": 2},
        })
        cfg = SimulatorConfig.from_json(path)
        self.assertFalse(cfg.speculative.enabled)
        self.assertEqual(cfg.speculative.num_spec_tokens, 0)
        Path(path).unlink()

    def test_enabled_with_tokens_stays_enabled(self):
        path = _write_json({
            "speculative": {"enabled": True, "num_spec_tokens": 3},
        })
        cfg = SimulatorConfig.from_json(path)
        self.assertTrue(cfg.speculative.enabled)
        self.assertEqual(cfg.speculative.num_spec_tokens, 3)
        Path(path).unlink()

    def test_enabled_with_zero_tokens_is_inert(self):
        # enabled=true, num_spec_tokens=0: no inflation, no drafts.  The
        # reconciler leaves this as-is (enabled true, 0 tokens), which the
        # spec engine treats as off (generate_draft_tokens returns [] when
        # num_spec_tokens==0) — consistent with the CLI's enabled=num>0.
        path = _write_json({
            "speculative": {"enabled": True, "num_spec_tokens": 0},
        })
        cfg = SimulatorConfig.from_json(path)
        self.assertEqual(cfg.speculative.num_spec_tokens, 0)
        Path(path).unlink()


class TestCliFlagNamesMatchJsonKeys(unittest.TestCase):
    """Round-12 F1: every CLI flag is the kebab-case of its JSON field name.

    Pre-fix the flags were ad-hoc abbreviations (--kv-block-size vs
    kv_cache_block_size, --seed vs random_seed, --fp4-indexer vs
    use_fp4_indexer, --swa-ratio vs swa_full_tokens_ratio, --model-config vs
    model_config_path), so users couldn't predict the JSON key from the flag.
    Now they map 1:1.  Old flag names are NOT accepted (no aliases).
    """

    def test_new_flag_names_parse(self):
        from simulator.run import _build_parser
        p = _build_parser()
        ns = p.parse_args([
            "--kv-cache-block-size", "8",
            "--num-kv-cache-blocks", "1024",
            "--model-config-path", "cfg.json",
            "--use-fp4-indexer",
            "--swa-full-tokens-ratio", "0.25",
            "--random-seed", "7",
            "--max-model-len", "4096",
            "--stall-limit", "50",
            "--verbose",
        ])
        self.assertEqual(ns.kv_cache_block_size, 8)
        self.assertEqual(ns.num_kv_cache_blocks, 1024)
        self.assertEqual(ns.model_config_path, "cfg.json")
        self.assertTrue(ns.use_fp4_indexer)
        self.assertAlmostEqual(ns.swa_full_tokens_ratio, 0.25)
        self.assertEqual(ns.random_seed, 7)
        self.assertTrue(ns.verbose)

    def test_old_flag_names_rejected(self):
        """No backward-compat aliases — old names must error out."""
        from simulator.run import _build_parser
        p = _build_parser()
        for old in [
            "--kv-block-size 16", "--num-kv-blocks 16", "--model-config x",
            "--fp4-indexer", "--swa-ratio 0.1", "--seed 1",
        ]:
            with self.assertRaises(SystemExit):
                p.parse_args(old.split())


class TestConfigModeCliOverrides(unittest.TestCase):
    """Round-12 F2: in --config mode, a non-default CLI flag overrides the
    JSON value; a default CLI flag preserves the JSON value.

    Pre-fix --config mode silently dropped every CLI flag except --use-fp4-
    indexer, so ``--config x.json --swa-full-tokens-ratio 0.25`` was ignored.
    """

    def _cfg_with_swa(self, ratio: float) -> str:
        return _write_json({
            "backend": "vllm",
            "swa_full_tokens_ratio": ratio,
            "num_kv_cache_blocks": 4096,
        })

    def test_non_default_cli_flag_overrides_json(self):
        import os
        from simulator.run import _build_parser, _override
        path = self._cfg_with_swa(0.1)
        try:
            p = _build_parser()
            ns = p.parse_args(["--config", path, "--swa-full-tokens-ratio", "0.3"])
            cfg = SimulatorConfig.from_json(path)
            _override(cfg, "swa_full_tokens_ratio", ns.swa_full_tokens_ratio, 0.1)
            self.assertAlmostEqual(cfg.swa_full_tokens_ratio, 0.3)
        finally:
            Path(path).unlink()

    def test_default_cli_flag_preserves_json_value(self):
        # JSON sets 0.2; the CLI flag is at its default 0.1, so the JSON value
        # must NOT be clobbered.
        from simulator.run import _build_parser, _override
        path = self._cfg_with_swa(0.2)
        try:
            p = _build_parser()
            ns = p.parse_args(["--config", path])
            cfg = SimulatorConfig.from_json(path)
            _override(cfg, "swa_full_tokens_ratio", ns.swa_full_tokens_ratio, 0.1)
            self.assertAlmostEqual(cfg.swa_full_tokens_ratio, 0.2)
        finally:
            Path(path).unlink()


class TestDatasetFieldDefaultsAligned(unittest.TestCase):
    """The CLI default, the SyntheticConfig dataclass default, and the
    from_json default must agree for every dataset flag — otherwise the same
    "default config" produces different behavior on the CLI path vs the
    --config-with-omitted-key path.

    This regression was found on output_length_fixed (CLI/README=256 vs
    dataclass/from_json=512).  Sibling fields (prompt_length_fixed,
    num_requests, shared_prefix_ratio) were already aligned; this test pins
    all of them so the drift can't silently return.
    """

    def test_dataset_defaults_agree_across_cli_dataclass_and_from_json(self):
        from simulator.run import _build_parser
        from simulator.config.simulator_config import SyntheticConfig

        p = _build_parser()
        ns = p.parse_args([])  # all defaults
        empty = _write_json({})
        try:
            cfg = SimulatorConfig.from_json(empty)
        finally:
            Path(empty).unlink()

        cases = [
            ("num_requests", ns.num_requests,
             SyntheticConfig().num_requests, cfg.dataset.synthetic.num_requests),
            ("prompt_length_fixed", ns.prompt_length,
             SyntheticConfig().prompt_length_fixed,
             cfg.dataset.synthetic.prompt_length_fixed),
            ("output_length_fixed", ns.output_length,
             SyntheticConfig().output_length_fixed,
             cfg.dataset.synthetic.output_length_fixed),
            ("shared_prefix_ratio", ns.shared_prefix_ratio,
             SyntheticConfig().shared_prefix_ratio,
             cfg.dataset.synthetic.shared_prefix_ratio),
        ]
        for name, cli, dataclass, from_json in cases:
            with self.subTest(field=name):
                self.assertEqual(cli, dataclass,
                                 f"{name}: CLI default {cli} != dataclass default {dataclass}")
                self.assertEqual(cli, from_json,
                                 f"{name}: CLI default {cli} != from_json default {from_json}")


class TestDeadFieldsAndAbcDefaults(unittest.TestCase):
    """Pin two cleanups: the dead SimulatorConfig.hash_block_size field is
    gone (engine.py never read it — it derives hash_block_size from
    layer_groups for hybrid models, or reuses kv_cache_block_size otherwise),
    and KVBackend declares num_free_blocks/total_blocks with default
    implementations so a backend omitting them degrades to a "0 free"
    diagnostic instead of crashing the OOM error path.
    """

    def test_hash_block_size_removed_from_simulator_config(self):
        import dataclasses
        from simulator.config.simulator_config import SimulatorConfig

        field_names = {f.name for f in dataclasses.fields(SimulatorConfig)}
        self.assertNotIn("hash_block_size", field_names)

    def test_from_json_ignores_stray_hash_block_size_key(self):
        # A JSON config that still carries the removed key must not error —
        # from_json simply drops it (no consumer ever read it).
        path = _write_json({"hash_block_size": 4})
        try:
            cfg = SimulatorConfig.from_json(path)
            self.assertFalse(hasattr(cfg, "hash_block_size"))
        finally:
            Path(path).unlink()

    def test_kvbackend_abc_provides_default_num_free_blocks(self):
        # A minimal backend that omits num_free_blocks/total_blocks inherits
        # the ABC default (0) rather than raising AttributeError on the
        # scheduler's OOM-diagnostic path.
        from simulator.kv_cache.base import KVBackend

        class _Bare(KVBackend):
            create_request = register_request = get_computed_blocks = \
                allocate_slots = set_spec_tokens = sync_state = free = \
                lambda *a, **k: None

            @property
            def usage(self):
                return 0.0

            @property
            def total_bytes(self):
                return 0

            @property
            def name(self):
                return "bare"

        self.assertEqual(_Bare().num_free_blocks, 0)
        self.assertEqual(_Bare().total_blocks, 0)


class TestFromJsonMlaDetection(unittest.TestCase):
    """from_json must detect MLA on a real DSV4 config.

    DSV4's HF config.json has no ``kv_lora_rank`` key — it was renamed to
    ``q_lora_rank``.  Pre-fix, from_json read only ``kv_lora_rank`` → None →
    is_mla stayed False → layer_groups collapsed to a single "full" group,
    silently producing the wrong (single-pool) KV layout whenever a user
    passed ``--model-config-path <real DSV4 config.json>``.  The default
    (hardcoded) path was unaffected, so the bug was latent.

    This synthesizes a minimal DSV4-style config (using q_lora_rank, no
    kv_lora_rank) so the test is self-contained and portable.
    """

    def _dsv4_style_config(self) -> dict:
        # Real DSV4 per-layer pattern: 2 SWA(0), then alternating 4/128.
        # The trailing MTP layer (ratio 0) is omitted here — layer_groups
        # only counts cr==4 / cr==128, so it doesn't affect the assertion.
        compress_ratios = [0, 0]
        for i in range(41):
            compress_ratios.append(4 if i % 2 == 0 else 128)
        return {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "hidden_size": 4096,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1024,  # DSV4 key (NOT kv_lora_rank)
            "compress_ratios": compress_ratios,
            "sliding_window": 128,
            "torch_dtype": "bfloat16",
            "vocab_size": 129280,
            "num_nextn_predict_layers": 1,
            "index_head_dim": 128,
        }

    def test_q_lora_rank_triggers_mla_detection(self):
        from simulator.config.model_config import ModelArchitecture

        path = _write_json(self._dsv4_style_config())
        try:
            arch = ModelArchitecture.from_json(path)
        finally:
            Path(path).unlink()

        self.assertTrue(arch.is_mla, "q_lora_rank must set is_mla=True")
        self.assertEqual(arch.kv_lora_rank, 1024)

    def test_from_json_produces_six_hybrid_groups(self):
        from simulator.config.model_config import ModelArchitecture

        path = _write_json(self._dsv4_style_config())
        try:
            arch = ModelArchitecture.from_json(path)
        finally:
            Path(path).unlink()

        groups = arch.layer_groups
        # Must NOT be the single-group fallback [("full", 0, 1, 43)].
        self.assertNotEqual(groups, [("full", 0, 1, 43)])
        names = [g[0] for g in groups]
        self.assertEqual(names, [
            "swa", "c4_compressor", "c128_compressor",
            "c4_mla", "c128_mla", "c4_indexer",
        ])
        # Counts must match DSV4: SWA=43, C4=21, C128=20.
        by_name = {g[0]: g[3] for g in groups}
        self.assertEqual(by_name["swa"], 43)
        self.assertEqual(by_name["c4_mla"], 21)
        self.assertEqual(by_name["c128_mla"], 20)
        # And the per-token byte cost is the DSV4 584.
        self.assertEqual(arch.kv_bytes_per_token, 584)

    def test_legacy_kv_lora_rank_still_detected(self):
        # DSV2/V3 use kv_lora_rank; that path must not regress.
        from simulator.config.model_config import ModelArchitecture

        cfg = self._dsv4_style_config()
        cfg.pop("q_lora_rank")
        cfg["kv_lora_rank"] = 512
        path = _write_json(cfg)
        try:
            arch = ModelArchitecture.from_json(path)
        finally:
            Path(path).unlink()
        self.assertTrue(arch.is_mla)
        self.assertEqual(arch.kv_lora_rank, 512)


class TestFromJsonNumMtpDefault(unittest.TestCase):
    """from_json must default num_mtp_layers to 0, not 1.

    ``num_nextn_predict_layers`` is DSV4-specific (DSV4 = 1).  A non-DSV4
    config (e.g. DSV2/V3) lacks the key; pre-fix from_json fell back to 1,
    which — with spec on — added a phantom MTP SWA layer to the vLLM block
    pool (vllm_config.py: ``num_mtp_layers = arch.num_mtp_layers if
    num_spec_tokens > 0 else 0``).  DSV4 configs carry the key explicitly, so
    they are unaffected by the default.
    """

    _BASE = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "hidden_size": 4096,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1024,
        "compress_ratios": [0, 0] + [4 if i % 2 == 0 else 128 for i in range(41)],
        "sliding_window": 128,
        "torch_dtype": "bfloat16",
        "vocab_size": 129280,
        "index_head_dim": 128,
    }

    def test_omitted_key_defaults_to_zero(self):
        from simulator.config.model_config import ModelArchitecture

        path = _write_json(dict(self._BASE))  # no num_nextn_predict_layers
        try:
            arch = ModelArchitecture.from_json(path)
        finally:
            Path(path).unlink()
        self.assertEqual(arch.num_mtp_layers, 0)

    def test_explicit_one_still_parsed(self):
        # DSV4 carries the key; the default change must not regress DSV4.
        from simulator.config.model_config import ModelArchitecture

        cfg = dict(self._BASE)
        cfg["num_nextn_predict_layers"] = 1
        path = _write_json(cfg)
        try:
            arch = ModelArchitecture.from_json(path)
        finally:
            Path(path).unlink()
        self.assertEqual(arch.num_mtp_layers, 1)


class TestFromJsonEdgeCases(unittest.TestCase):
    """from_json robustness on malformed/edge configs."""

    _BASE = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "hidden_size": 4096,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1024,
        "compress_ratios": [0, 0] + [4 if i % 2 == 0 else 128 for i in range(41)],
        "sliding_window": 128,
        "torch_dtype": "bfloat16",
        "vocab_size": 129280,
        "index_head_dim": 128,
    }

    def test_empty_architectures_no_model_type_no_crash(self):
        # ``cfg.get("architectures", ["unknown"])[0]`` would IndexError when
        # architectures is an empty list (key present → default unused).  The
        # guard must fall back to "unknown" instead of crashing.
        from simulator.config.model_config import ModelArchitecture

        cfg = dict(self._BASE)
        del cfg["model_type"]
        cfg["architectures"] = []
        path = _write_json(cfg)
        try:
            arch = ModelArchitecture.from_json(path)
        finally:
            Path(path).unlink()
        self.assertEqual(arch.model_type, "unknown")

    def test_kv_lora_rank_zero_not_treated_as_falsy(self):
        # ``or`` coalescing would treat kv_lora_rank=0 as falsy and fall
        # through to q_lora_rank.  ``is not None`` keeps 0.  (The value is
        # only an MLA sentinel, so 0 still sets is_mla=True via the qk_rope
        # pair — but the stored field must reflect the real key.)
        from simulator.config.model_config import ModelArchitecture

        cfg = dict(self._BASE)
        cfg.pop("q_lora_rank", None)
        cfg["kv_lora_rank"] = 0
        path = _write_json(cfg)
        try:
            arch = ModelArchitecture.from_json(path)
        finally:
            Path(path).unlink()
        self.assertEqual(arch.kv_lora_rank, 0)
        self.assertTrue(arch.is_mla)


class TestGpuDataPointsCliParsing(unittest.TestCase):
    """--gpu-data-points is a JSON string on the CLI; run.py must json.loads it
    before handing it to GPUPerfConfig.  Pre-fix the raw string was passed
    through, and _fit()'s ``PerfDataPoint(*p) for p in data_points`` iterated
    the string char-by-char → TypeError crash.
    """

    def test_gpu_data_points_json_string_parses_to_triples(self):
        import json
        from simulator.run import _build_parser
        from simulator.config.simulator_config import GPUPerfConfig
        from simulator.metrics.gpu_perf_model import GPUPerfModel

        p = _build_parser()
        ns = p.parse_args([
            "--gpu-data-points", "[[0,1,0.5],[1000,1,1.5],[0,500,20.0]]",
        ])
        parsed = json.loads(ns.gpu_data_points)
        cfg = GPUPerfConfig(data_points=parsed)
        # GPUPerfModel.__init__ calls _fit(), which unpacks each triple into
        # PerfDataPoint — this is the line that crashed pre-fix on a raw str.
        model = GPUPerfModel(cfg)
        self.assertGreater(model.predict(500, 200), 0.0)

    def test_gpu_data_points_overrides_json_in_config_mode(self):
        # --gpu-data-points applies in --config mode too (GPU perf tuning is
        # independent of the dataset).  Pre-fix it was silently ignored in
        # --config mode.  Mirror run.py's --config override: load a JSON with
        # gpu_perf.data_points set, then apply the CLI flag and confirm the
        # CLI value wins (parsed to triples, GPUPerfModel fits it).
        import json
        from simulator.config.simulator_config import SimulatorConfig, GPUPerfConfig
        from simulator.metrics.gpu_perf_model import GPUPerfModel

        cfg_path = _write_json({
            "gpu_perf": {"data_points": [[0, 1, 9.0], [1000, 1, 9.0]]},
        })
        try:
            cfg = SimulatorConfig.from_json(cfg_path)
            # JSON value present before override.
            self.assertIsNotNone(cfg.gpu_perf.data_points)
            # run.py applies this when args.gpu_data_points is not None:
            cfg.gpu_perf = GPUPerfConfig(
                data_points=json.loads("[[0,1,0.5],[1000,1,1.5],[0,500,20.0]]")
            )
            model = GPUPerfModel(cfg.gpu_perf)
            self.assertGreater(model.predict(500, 200), 0.0)
        finally:
            Path(cfg_path).unlink()

    def test_gpu_data_points_preserves_coeff_overrides(self):
        # --gpu-data-points must override ONLY data_points, preserving JSON-set
        # coefficient overrides.  Pre-fix run.py rebuilt GPUPerfConfig, dropping
        # the coeffs and throwing a coeffs-configured model back into fit mode.
        # Mirror run.py's new field-level override (cfg.gpu_perf.data_points =
        # json.loads(...)) and confirm coeffs survive and drive the prediction.
        import json
        from simulator.config.simulator_config import SimulatorConfig
        from simulator.metrics.gpu_perf_model import GPUPerfModel

        cfg_path = _write_json({
            "gpu_perf": {
                "loaded_coeff": 0.001,
                "computed_coeff": 0.01,
                "interaction_coeff": 0.0,
                "base_latency_ms": 0.5,
                "data_points": [[0, 1, 9.0], [1000, 1, 9.0]],
            },
        })
        try:
            cfg = SimulatorConfig.from_json(cfg_path)
            # run.py's --gpu-data-points override (field-level, not rebuild):
            cfg.gpu_perf.data_points = json.loads(
                "[[0,1,0.5],[1000,1,1.5],[0,500,20.0]]"
            )
            # Coeffs preserved → explicit-coeff mode (bypasses fit).
            self.assertEqual(cfg.gpu_perf.loaded_coeff, 0.001)
            self.assertEqual(cfg.gpu_perf.base_latency_ms, 0.5)
            model = GPUPerfModel(cfg.gpu_perf)
            # a*loaded + b*computed + c*loaded*computed + d
            expected = 0.001 * 1000 + 0.01 * 1 + 0.0 + 0.5
            self.assertAlmostEqual(model.predict(1000, 1), expected, places=3)
        finally:
            Path(cfg_path).unlink()


class TestDeadConfigFieldsRemoved(unittest.TestCase):
    """Pin removal of dead SimulatorConfig fields: model_name,
    RequestArrivalConfig.num_requests, and the backend-handle num_tokens
    properties (all written/declared but never read).
    """

    def test_model_name_removed(self):
        import dataclasses
        from simulator.config.simulator_config import SimulatorConfig

        names = {f.name for f in dataclasses.fields(SimulatorConfig)}
        self.assertNotIn("model_name", names)

    def test_request_arrival_num_requests_removed(self):
        import dataclasses
        from simulator.config.simulator_config import RequestArrivalConfig

        names = {f.name for f in dataclasses.fields(RequestArrivalConfig)}
        self.assertNotIn("num_requests", names)

    def test_from_json_tolerates_removed_arrival_num_requests(self):
        # A JSON still carrying the removed key must not error.
        path = _write_json({"arrival": {"num_requests": 50, "poisson_rate": 2.0}})
        try:
            cfg = SimulatorConfig.from_json(path)
            self.assertNotIn("num_requests",
                             {f.name for f in __import__("dataclasses").fields(cfg.arrival)})
            self.assertEqual(cfg.arrival.poisson_rate, 2.0)
        finally:
            Path(path).unlink()

    def test_backend_handles_have_no_num_tokens(self):
        # The unused num_tokens properties were removed from both handles;
        # SimRequestState.num_tokens is the single source of truth.
        from simulator.kv_cache.vllm_backend import vLLMSimRequest
        from simulator.kv_cache.sglang_backend import SGLangSimRequest

        for cls in (vLLMSimRequest, SGLangSimRequest):
            self.assertNotIn("num_tokens", cls.__dict__,
                             f"{cls.__name__} should not define num_tokens")

    def test_sim_request_state_has_no_allocated_blocks(self):
        # SimRequestState.allocated_blocks was written at prefill and decode
        # (scheduler.py) but never read anywhere — a dead field.  Pinned gone so
        # a future "let's stash the allocation here for diagnostics" doesn't
        # silently resurrect it; the backend handle (backend_req) already owns
        # backend block state.
        import dataclasses
        from simulator.core.request_state import SimRequestState

        names = {f.name for f in dataclasses.fields(SimRequestState)}
        self.assertNotIn("allocated_blocks", names)

    def test_sim_request_state_dead_props_removed(self):
        # is_prefill_needed / is_admitted had no callers anywhere in the
        # codebase (is_admitted also returned True for FINISHED, a semantic
        # bug).  is_finished IS used (scheduler.py step loop) and stays.
        from simulator.core.request_state import SimRequestState

        attrs = dir(SimRequestState)
        self.assertNotIn("is_prefill_needed", attrs)
        self.assertNotIn("is_admitted", attrs)
        self.assertIn("is_finished", attrs)

    def test_kv_backend_config_has_no_kv_cache_dtype(self):
        # KVBackendConfig.kv_cache_dtype was never read: vLLM derives
        # cache_dtype_str from arch.is_mla (vllm_config.py:36), SGLang ignores
        # it, and engine.py never passes it.  Pinned gone.
        import dataclasses
        from simulator.config.model_config import KVBackendConfig

        names = {f.name for f in dataclasses.fields(KVBackendConfig)}
        self.assertNotIn("kv_cache_dtype", names)

    def test_vllm_backend_has_no_bare_block_size_attr(self):
        # vLLMBackend stored self._block_size in __init__ but never read it
        # (only _hash_block_size / _scheduler_block_size are read).  The bare
        # attr is an instance attribute set in __init__, so inspect __init__'s
        # source to pin its absence (can't observe via __dict__ without
        # constructing a real vLLMBackend, which needs the vllm package).
        import inspect
        from simulator.kv_cache.vllm_backend import vLLMBackend

        src = inspect.getsource(vLLMBackend.__init__)
        self.assertNotIn("self._block_size = ", src)
        # The read variants must remain.
        self.assertIn("self._hash_block_size = ", src)
        self.assertIn("self._scheduler_block_size = ", src)


if __name__ == "__main__":
    unittest.main()
