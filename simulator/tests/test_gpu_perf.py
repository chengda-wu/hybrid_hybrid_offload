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


if __name__ == "__main__":
    unittest.main()
