"""End-to-end scheduler integration tests.

Drives the full SimulationEngine.run() path (admit → prefill → decode →
spec accept/reject → finish → report) on a small synthetic workload, on
both backends and both spec-on / spec-off. Guards against regressions
that previously hung the loop (spec-off bonus-token bug, prefill-retry
OOM) via a per-test wall-clock timeout.
"""

import importlib.util
import io
import signal
import unittest
from contextlib import redirect_stdout

from simulator.config.simulator_config import (
    DatasetConfig,
    SimulatorConfig,
    SpeculativeDecodeConfig,
    SyntheticConfig,
)
from simulator.core.engine import SimulationEngine

_HAS_VLLM = importlib.util.find_spec("vllm") is not None
_HAS_SGLANG = importlib.util.find_spec("sglang") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None

requires_vllm = unittest.skipUnless(_HAS_VLLM and _HAS_TORCH, "requires vllm+torch")
requires_sglang = unittest.skipUnless(
    _HAS_SGLANG and _HAS_TORCH, "requires sglang+torch"
)


class _HangTimeout:
    """Raise AssertionError if the wrapped call exceeds `seconds`.

    Uses SIGALRM so a hung main loop fails the test instead of stalling
    the whole suite (the exact failure mode of the spec-off / prefill-retry
    bugs this suite guards against).
    """

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)

    @staticmethod
    def _handler(signum, frame):
        raise AssertionError(f"Simulation hung: exceeded wall-clock timeout")


def _config(backend: str, num_spec_tokens: int) -> SimulatorConfig:
    return SimulatorConfig(
        backend=backend,
        dataset=DatasetConfig(
            synthetic=SyntheticConfig(
                num_requests=3,
                prompt_length_fixed=256,
                output_length_fixed=64,
                shared_prefix_ratio=0.5,
            )
        ),
        speculative=SpeculativeDecodeConfig(
            enabled=num_spec_tokens > 0,
            num_spec_tokens=num_spec_tokens,
            acceptance_rates=[0.8, 0.7] if num_spec_tokens > 0 else None,
        ),
        warmup_steps=2,
        random_seed=42,
        num_kv_cache_blocks=4096,
    )


def _run(backend: str, num_spec_tokens: int):
    """Run a small simulation, returning the report. Fails on hang."""
    cfg = _config(backend, num_spec_tokens)
    engine = SimulationEngine(cfg)
    buf = io.StringIO()
    # run() prints a config summary; swallow it so test output stays clean.
    with _HangTimeout(120):
        with redirect_stdout(buf):
            report = engine.run()
    return report


class TestSchedulerE2E(unittest.TestCase):
    """Full-pipeline smoke tests on both backends, spec on/off."""

    # ---- spec-on: every request finishes, acceptance is reported ----

    @requires_vllm
    def test_vllm_spec_on_completes(self):
        report = _run("vllm", num_spec_tokens=2)
        self.assertEqual(report.total_requests, 3)
        self.assertGreater(report.total_tokens_generated, 0)
        self.assertEqual(report.backend, "vllm")
        # spec on → acceptance rate is a real number in [0, 1]
        self.assertIsNotNone(report.avg_acceptance_rate)
        self.assertGreaterEqual(report.avg_acceptance_rate, 0.0)
        self.assertLessEqual(report.avg_acceptance_rate, 1.0)
        # throughput and per-position rates are well-formed
        self.assertGreater(report.tokens_per_second, 0.0)
        self.assertEqual(len(report.per_position_acceptance_rates), 2)
        for r in report.per_position_acceptance_rates:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
        # wall-clock field present (renamed from total_sim_time_ms)
        self.assertGreaterEqual(report.wall_clock_sim_time_ms, 0.0)

    @requires_sglang
    def test_sglang_spec_on_completes(self):
        report = _run("sglang", num_spec_tokens=2)
        self.assertEqual(report.total_requests, 3)
        self.assertGreater(report.total_tokens_generated, 0)
        self.assertEqual(report.backend, "sglang")
        self.assertIsNotNone(report.avg_acceptance_rate)
        self.assertGreaterEqual(report.avg_acceptance_rate, 0.0)
        self.assertLessEqual(report.avg_acceptance_rate, 1.0)
        self.assertGreater(report.tokens_per_second, 0.0)
        self.assertEqual(len(report.per_position_acceptance_rates), 2)
        for r in report.per_position_acceptance_rates:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
        self.assertGreaterEqual(report.wall_clock_sim_time_ms, 0.0)

    # ---- spec-off: must NOT hang (regression: bonus-token bug) ----

    @requires_vllm
    def test_vllm_spec_off_completes_without_hang(self):
        report = _run("vllm", num_spec_tokens=0)
        self.assertEqual(report.total_requests, 3)
        self.assertGreater(report.total_tokens_generated, 0)
        self.assertGreater(report.tokens_per_second, 0.0)
        # spec off → no spec tokens evaluated → None (not 0.0)
        self.assertIsNone(report.avg_acceptance_rate)
        self.assertEqual(report.per_position_acceptance_rates, [])

    @requires_sglang
    def test_sglang_spec_off_completes_without_hang(self):
        report = _run("sglang", num_spec_tokens=0)
        self.assertEqual(report.total_requests, 3)
        self.assertGreater(report.total_tokens_generated, 0)
        self.assertGreater(report.tokens_per_second, 0.0)
        self.assertIsNone(report.avg_acceptance_rate)
        self.assertEqual(report.per_position_acceptance_rates, [])

    # ---- prefill OOM: must raise, not hang (regression) ----

    @requires_sglang
    def test_prefill_oom_raises_instead_of_hanging(self):
        cfg = _config("sglang", num_spec_tokens=2)
        cfg.num_kv_cache_blocks = 1  # far too small for a 256-token prompt
        cfg.dataset.synthetic.prompt_length_fixed = 2048
        # The too-small config may be rejected either at backend construction
        # (real SGLang's DSV4PoolConfigurator raises "Not enough memory" when
        # the spec-scaled full_token collapses to 0) or at first prefill (pool
        # over budget).  Both are RuntimeError; both satisfy "raises, doesn't
        # hang", so wrap construction inside the assert.
        with self.assertRaises(RuntimeError):
            with _HangTimeout(60):
                with redirect_stdout(io.StringIO()):
                    engine = SimulationEngine(cfg)
                    engine.run()


if __name__ == "__main__":
    unittest.main()
