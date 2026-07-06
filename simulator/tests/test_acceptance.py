"""Acceptance model tests: ValueError on rates < K, chain-breaking, etc.

Drafts are always ground truth (SpeculativeDecodeEngine); acceptance is
purely the per-position rate. rate 1.0 ⇒ always accept, 0.0 ⇒ always reject,
so chain-breaking is tested deterministically.
"""

import unittest

from simulator.config.simulator_config import SpeculativeDecodeConfig
from simulator.core.request_state import SimRequestState
from simulator.speculative.acceptance import AcceptanceModel


class TestAcceptanceRatesValidation(unittest.TestCase):
    """acceptance_rates < K must raise ValueError."""

    def test_raises_when_rates_shorter_than_k(self):
        cfg = SpeculativeDecodeConfig(
            accept_mode="per_position",
            acceptance_rates=[0.9, 0.8],   # only 2 rates
            num_spec_tokens=4,               # but K=4
        )
        with self.assertRaises(ValueError):
            AcceptanceModel(cfg, seed=42)

    def test_ok_when_rates_equal_k(self):
        cfg = SpeculativeDecodeConfig(
            accept_mode="per_position",
            acceptance_rates=[0.9, 0.8, 0.7, 0.6],
            num_spec_tokens=4,
        )
        AcceptanceModel(cfg, seed=42)  # should not raise

    def test_ok_when_rates_longer_than_k(self):
        cfg = SpeculativeDecodeConfig(
            accept_mode="per_position",
            acceptance_rates=[0.9, 0.8, 0.7, 0.6, 0.5],
            num_spec_tokens=3,
        )
        AcceptanceModel(cfg, seed=42)  # should not raise

    def test_fixed_mode_no_validation_needed(self):
        cfg = SpeculativeDecodeConfig(
            accept_mode="fixed",
            acceptance_rate=0.85,
            num_spec_tokens=4,
        )
        AcceptanceModel(cfg, seed=42)  # fixed mode doesn't use acceptance_rates


class TestAcceptanceLogic(unittest.TestCase):
    """Core acceptance behavior — rate-driven, deterministic at 0.0/1.0."""

    def setUp(self):
        # Ground truth: tokens 10-19 (indices 0-9)
        self.req = SimRequestState("t1", [1, 2], list(range(10, 20)), 10)

    def _eval(self, rates, output_token_ids, drafts):
        cfg = SpeculativeDecodeConfig(
            accept_mode="per_position",
            acceptance_rates=rates,
            num_spec_tokens=len(drafts),
        )
        model = AcceptanceModel(cfg, seed=42)
        self.req.output_token_ids = output_token_ids
        return model.evaluate(self.req, drafts)

    def test_full_rate_accepts_all(self):
        # output=[10], first draft at ground_truth[2]=12
        accepted, rejected, beyond = self._eval([1.0, 1.0, 1.0], [10], [12, 13, 14])
        self.assertEqual((accepted, rejected, beyond), (3, 0, 0))

    def test_first_rejection_breaks_chain(self):
        accepted, rejected, beyond = self._eval([0.0, 1.0, 1.0], [10], [12, 13, 14])
        self.assertEqual((accepted, rejected, beyond), (0, 3, 0))

    def test_mid_chain_rejection(self):
        accepted, rejected, beyond = self._eval([1.0, 0.0, 1.0], [10], [12, 13, 14])
        self.assertEqual((accepted, rejected, beyond), (1, 2, 0))

    def test_bonus_offset_correct(self):
        """Draft[i] checks ground_truth[output_position + 1 + i]."""
        # output_pos=2, first draft at 2+1+0=3 → ground_truth[3]=13
        accepted, _, _ = self._eval([1.0, 1.0, 1.0], [10, 11], [13, 14, 15])
        self.assertEqual(accepted, 3)

    def test_beyond_ground_truth_breaks(self):
        # output_pos=9, first draft at 9+1+0=10 → out of range
        accepted, rejected, beyond = self._eval(
            [1.0, 1.0, 1.0],
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
            [19, 99, 99],
        )
        # All 3 fall beyond ground truth — not real rejections.
        self.assertEqual((accepted, rejected, beyond), (0, 0, 3))


if __name__ == "__main__":
    unittest.main()
