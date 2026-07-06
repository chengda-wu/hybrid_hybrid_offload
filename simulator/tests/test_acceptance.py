"""Acceptance model tests: marginal-rate validation, chain-breaking, reproduction.

Input acceptance_rates are MARGINAL (P(draft_i accepted)). Internally
converted to conditional rates (cond[i] = marginal[i]/marginal[i-1]) so that
chain-breaking sampling reproduces the marginals. rate 1.0 ⇒ always accept,
0.0 ⇒ always reject → chain-breaking tested deterministically.
"""

import unittest

from simulator.config.simulator_config import SpeculativeDecodeConfig
from simulator.core.request_state import SimRequestState
from simulator.speculative.acceptance import AcceptanceModel


class TestAcceptanceRatesValidation(unittest.TestCase):
    """Marginal-rate input validation."""

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

    def test_raises_on_non_monotonic_marginal(self):
        # Marginal rates must be non-increasing: 0.6 then 0.8 is invalid.
        cfg = SpeculativeDecodeConfig(
            accept_mode="per_position",
            acceptance_rates=[0.6, 0.8],
            num_spec_tokens=2,
        )
        with self.assertRaises(ValueError):
            AcceptanceModel(cfg, seed=42)

    def test_ok_on_equal_marginal(self):
        # Equal marginals ⇒ cond[i>0] = 1.0 (accept all after first). Valid.
        cfg = SpeculativeDecodeConfig(
            accept_mode="per_position",
            acceptance_rates=[0.5, 0.5, 0.5],
            num_spec_tokens=3,
        )
        AcceptanceModel(cfg, seed=42)

    def test_fixed_mode_no_validation_needed(self):
        cfg = SpeculativeDecodeConfig(
            accept_mode="fixed",
            acceptance_rate=0.85,
            num_spec_tokens=4,
        )
        AcceptanceModel(cfg, seed=42)  # fixed mode doesn't use acceptance_rates

    def test_per_position_empty_rates_falls_back_to_half(self):
        # per_position with no acceptance_rates must not silently guess; it
        # falls back to a flat 0.5 conditional rate (and logs a warning).
        # Verified by observing the conditional rate the model uses.
        cfg = SpeculativeDecodeConfig(
            accept_mode="per_position",
            acceptance_rates=None,
            num_spec_tokens=3,
        )
        model = AcceptanceModel(cfg, seed=42)
        self.assertEqual(model._cond_rates, [0.5, 0.5, 0.5])


class TestAcceptanceLogic(unittest.TestCase):
    """Core acceptance behavior — chain-breaking, deterministic at 0.0/1.0."""

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
        # marginal [1,1,1] ⇒ cond [1,1,1] ⇒ accept all
        accepted, rejected, beyond = self._eval([1.0, 1.0, 1.0], [10], [12, 13, 14])
        self.assertEqual((accepted, rejected, beyond), (3, 0, 0))

    def test_first_rejection_breaks_chain(self):
        # marginal [0,0,0] ⇒ cond[0]=0 ⇒ reject at pos 0, chain breaks
        accepted, rejected, beyond = self._eval([0.0, 0.0, 0.0], [10], [12, 13, 14])
        self.assertEqual((accepted, rejected, beyond), (0, 3, 0))

    def test_mid_chain_rejection(self):
        # marginal [1,0,1] ⇒ cond [1,0,inf→guard]. pos0=1 accept, pos1=0 reject
        accepted, rejected, beyond = self._eval([1.0, 0.0, 0.0], [10], [12, 13, 14])
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


class TestMarginalReproduction(unittest.TestCase):
    """Reported per-position marginal rates reproduce the input (large sample)."""

    def test_per_position_rates_converge_to_input(self):
        K = 4
        marginal_in = [0.9, 0.8, 0.7, 0.6]
        cfg = SpeculativeDecodeConfig(
            accept_mode="per_position",
            acceptance_rates=marginal_in,
            num_spec_tokens=K,
        )
        model = AcceptanceModel(cfg, seed=7)
        from simulator.speculative.engine import SpeculativeDecodeEngine
        from simulator.core.request_state import RequestStatus
        engine = SpeculativeDecodeEngine(cfg)
        gt = list(range(1000000, 2000000))  # 1M tokens
        req = SimRequestState("r", [1], gt, len(gt))
        req.status = RequestStatus.DECODING

        while req.output_length < len(gt):
            drafts = engine.generate_draft_tokens(req)
            if not drafts:
                break
            output_pos = len(req.output_token_ids)
            bonus = gt[output_pos] if output_pos < len(gt) else None
            req.spec_token_ids = drafts
            na, nr, nb = model.evaluate(req, drafts)
            req.subtract_rejected_tokens(nr + nb)
            out = ([bonus] if bonus is not None else []) + drafts[:na]
            req.append_output_tokens(out)
            req.clear_spec_tokens()

        measured = model.per_position_acceptance_rates
        self.assertEqual(len(measured), K)
        for got, want in zip(measured, marginal_in):
            self.assertAlmostEqual(got, want, delta=0.01)


if __name__ == "__main__":
    unittest.main()
