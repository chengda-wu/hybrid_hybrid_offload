"""Acceptance model tests: ValueError on rates < K, chain-breaking, etc."""

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
    """Core acceptance behavior."""

    def setUp(self):
        cfg = SpeculativeDecodeConfig(
            accept_mode="per_position",
            acceptance_rates=[0.99, 0.99, 0.99],  # nearly always accept
            num_spec_tokens=3,
        )
        self.model = AcceptanceModel(cfg, seed=42)
        # Ground truth: tokens 10-19
        self.req = SimRequestState("t1", [1, 2], list(range(10, 20)), 10)

    def test_all_matching_drafts_accepted_with_high_rate(self):
        # output=[10], first draft after bonus checks ground_truth[2]=12
        self.req.output_token_ids = [10]
        accepted, rejected, beyond = self.model.evaluate(self.req, [12, 13, 14])
        self.assertEqual(accepted + rejected + beyond, 3)
        self.assertEqual(beyond, 0)
        self.assertGreaterEqual(accepted, 2)  # high rate => most accepted

    def test_first_mismatch_breaks_chain(self):
        self.req.output_token_ids = [10]
        accepted, rejected, beyond = self.model.evaluate(self.req, [999, 12, 13])
        self.assertEqual(accepted, 0, "first draft wrong, should reject all")
        self.assertEqual(rejected, 3)
        self.assertEqual(beyond, 0)

    def test_mid_chain_mismatch(self):
        self.req.output_token_ids = [10]
        accepted, rejected, beyond = self.model.evaluate(self.req, [12, 999, 13])
        self.assertEqual(accepted, 1, "only first draft correct")
        self.assertEqual(rejected, 2)
        self.assertEqual(beyond, 0)

    def test_bonus_offset_correct(self):
        """Draft[i] checks ground_truth[output_position + 1 + i]."""
        self.req.output_token_ids = [10, 11]
        # output_pos=2, first draft at 2+1+0=3 → ground_truth[3]=13
        accepted, _, _ = self.model.evaluate(self.req, [13, 14, 15])
        self.assertGreaterEqual(accepted, 2)

    def test_beyond_ground_truth_breaks(self):
        self.req.output_token_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18]
        # output_pos=9, first draft at 9+1+0=10 → ground_truth[10] out of range
        accepted, rejected, beyond = self.model.evaluate(self.req, [19, 99, 99])
        self.assertEqual(accepted, 0, "only one token left at pos 10=out of range")
        # All 3 fall beyond ground truth — not real rejections.
        self.assertEqual(rejected, 0)
        self.assertEqual(beyond, 3)


if __name__ == "__main__":
    unittest.main()
