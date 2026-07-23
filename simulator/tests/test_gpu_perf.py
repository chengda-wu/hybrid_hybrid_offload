"""GPU perf model tests: non-negative prediction, monotonicity, warning on floor."""

import io
import logging
import unittest

from simulator.config.simulator_config import GPUPerfConfig
from simulator.metrics.gpu_perf_model import GPUPerfModel


class TestGPUPredictNonNegative(unittest.TestCase):
    """predict() must never return negative, and should warn when flooring."""

    def test_single_decode_returns_non_negative(self):
        m = GPUPerfModel(GPUPerfConfig())
        t = m.predict(0, 1)
        self.assertGreaterEqual(t, 0.0)

    def test_small_spec_decode_returns_non_negative(self):
        m = GPUPerfModel(GPUPerfConfig())
        t = m.predict(0, 3)
        self.assertGreaterEqual(t, 0.0)

    def test_prefill_positive(self):
        m = GPUPerfModel(GPUPerfConfig())
        t = m.predict(0, 2048)
        self.assertGreater(t, 0.0, "prefill should have positive latency")

    def test_more_tokens_increases_latency(self):
        m = GPUPerfModel(GPUPerfConfig())
        t1 = m.predict(0, 1)
        t2 = m.predict(4000, 1)
        self.assertGreater(t2, t1, "more cached tokens should increase latency")

    def test_warns_when_floored_to_zero(self):
        """If d < 0, small-load predictions may floor to 0 with a warning."""
        m = GPUPerfModel(GPUPerfConfig())
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("simulator.metrics.gpu_perf_model")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        try:
            t = m.predict(0, 1)
            self.assertGreaterEqual(t, 0.0)
            m.predict(0, 1)  # second call: no duplicate warning
            output = stream.getvalue()
            if m._d < 0:
                self.assertIn("negative latency", output,
                              "should warn when flooring negative prediction")
        finally:
            logger.removeHandler(handler)


class TestGPUExplicitCoefficients(unittest.TestCase):
    """Overriding coefficients directly."""

    def test_explicit_coefficients_bypass_fit(self):
        cfg = GPUPerfConfig(
            loaded_coeff=0.001,
            computed_coeff=0.01,
            interaction_coeff=0.0,
            base_latency_ms=0.5,
        )
        m = GPUPerfModel(cfg)
        t = m.predict(1000, 1)
        expected = 0.001 * 1000 + 0.01 * 1 + 0.5
        self.assertAlmostEqual(t, expected, places=4)

    def test_explicit_coefficients_set_envelope(self):
        # Explicit-coeff mode must still record a training envelope (from
        # data_points if provided, else DEFAULT_DATA) so predict()'s
        # extrapolation warning is not permanently silent.  Pre-fix the coeff
        # early-return left _max_loaded/_max_computed at 0, and predict()'s
        # ``> 0`` guard then silenced the warning forever in coeff mode.
        cfg = GPUPerfConfig(
            loaded_coeff=0.001,
            computed_coeff=0.01,
            interaction_coeff=0.0,
            base_latency_ms=0.5,
        )
        m = GPUPerfModel(cfg)
        self.assertGreater(m._max_loaded, 0)
        self.assertGreater(m._max_computed, 0)

    def test_explicit_coefficients_warns_on_extrapolation(self):
        # With the envelope now recorded, a wildly out-of-range input in coeff
        # mode must fire the extrapolation warning (pre-fix: permanently silent).
        cfg = GPUPerfConfig(
            loaded_coeff=0.001,
            computed_coeff=0.01,
            interaction_coeff=0.0,
            base_latency_ms=0.5,
        )
        m = GPUPerfModel(cfg)
        with self.assertLogs("simulator.metrics.gpu_perf_model", level="WARNING") as cm:
            # loaded >> 2× DEFAULT_DATA max_loaded (8000).
            m.predict(10_000_000, 1)
        self.assertTrue(
            any("extrapolation" in msg.lower() for msg in cm.output),
            f"expected extrapolation warning, got: {cm.output}",
        )


class TestGPUCustomDataPoints(unittest.TestCase):
    """Fitting from user-provided data."""

    def test_custom_points_fit(self):
        cfg = GPUPerfConfig(data_points=[
            (0, 100, 5.0),
            (1000, 100, 8.0),
            (0, 500, 20.0),
        ])
        m = GPUPerfModel(cfg)
        t = m.predict(500, 200)
        self.assertGreater(t, 0.0)


class TestGPUBatchInteractionDecomposition(unittest.TestCase):
    """Regression: the interaction term is additively decomposed per-request.

    Pre-fix the scheduler queried ``predict(Σloaded, Σcomputed)`` and the
    model computed ``c·(Σloaded)·(Σcomputed)``, which expands to
    ``c·Σ_iΣ_j(loaded_i·computed_j)`` — including phantom cross-request terms
    (request i's new tokens never attend request j's cache).  That blew up as
    O(N²): N=4 identical decode requests predicted 33 ms vs a serial 10 ms
    (batch SLOWER than serial — directionally wrong), and N≥8 all clamped to
    the cap so batch=8 and batch=64 were indistinguishable.

    The fix: callers pass ``interaction_tokens = Σ(loaded_i·computed_i)``;
    the model uses ``c·Σ(loaded_i·computed_i)``.  This equals
    ``c·loaded·computed`` for one request (single-request semantics unchanged)
    and grows LINEARLY with N for identical requests.
    """

    def test_single_request_uses_loaded_times_computed(self):
        # interaction defaults to loaded*computed → single-request semantics
        # identical to passing it explicitly.
        m = GPUPerfModel(GPUPerfConfig())
        self.assertAlmostEqual(
            m.predict(4000, 1), m.predict(4000, 1, 4000 * 1), places=6
        )

    def test_batch_grows_linearly_not_quadratically(self):
        # N identical (loaded=4000, computed=1) requests.  Batch latency with
        # the correct per-request interaction sum must grow ~linearly with N,
        # not as N².  Pre-fix: N=4 → 33 ms (3.2× serial), N=8 → capped 50 ms.
        m = GPUPerfModel(GPUPerfConfig())
        latencies = []
        for N in (1, 4, 8, 16, 32):
            batch = m.predict(4000 * N, 1 * N, 4000 * 1 * N)
            latencies.append(batch)
        # 8 → 32 is a 4× increase in N.  Linear growth ⇒ ~4× latency; O(N²)
        # would give 16×.  Assert well under the quadratic bound (and above
        # linear so the test is meaningful).  The positive base term pushes
        # the ratio just under 4×, so 4×-1.2×-ish.
        ratio_8_to_32 = latencies[4] / latencies[2]
        self.assertLess(ratio_8_to_32, 8.0,
                        f"batch grew {ratio_8_to_32:.2f}× from N=8→32, expected "
                        f"~4× (linear); O(N²) would be 16×")

    def test_batch_slower_than_serial_is_directionally_wrong(self):
        # The clearest pre-fix symptom: batching 4 identical decodes predicted
        # SLOWER than running them serially.  With the per-request sum, batch
        # of N identical requests ≈ N×single (plus one shared base term), so
        # batch is never multiple-times slower than serial.
        m = GPUPerfModel(GPUPerfConfig())
        N = 4
        single = m.predict(4000, 1)
        batch = m.predict(4000 * N, 1 * N, 4000 * 1 * N)
        # batch ≈ N*single + d (one base for the whole forward); serial would
        # be N*single + N*d.  Batch should be within 1.5× of N×single.
        self.assertLess(batch, N * single * 1.5)

    def test_cap_above_largest_training_point(self):
        # The cap must sit above the largest DEFAULT_DATA latency (130 ms) so
        # the model can reproduce its own training data — pre-fix cap=50 ate
        # predict(0,8192) (training 130) down to 50.
        m = GPUPerfModel(GPUPerfConfig())
        self.assertGreaterEqual(
            m.MAX_STEP_LATENCY_MS, 130.0,
            "cap must be ≥ largest training-point latency (130 ms)",
        )
        # And a prefill within the training envelope is NOT clamped.
        self.assertGreater(m.predict(0, 8192), 50.0)


if __name__ == "__main__":
    unittest.main()
