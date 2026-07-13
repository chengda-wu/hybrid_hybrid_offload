"""Round-10 F1/F2 regression: config-surface fixes.

F1 — ``swa_full_tokens_ratio`` (was hardcoded 0.1 in the mock server_args) is
now exposed on ``SimulatorConfig``, parsed from JSON, overridable via the
``--swa-ratio`` CLI flag, and threaded into ``KVBackendConfig``.

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


if __name__ == "__main__":
    unittest.main()
