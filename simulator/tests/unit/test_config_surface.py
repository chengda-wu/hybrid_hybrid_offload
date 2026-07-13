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


if __name__ == "__main__":
    unittest.main()
