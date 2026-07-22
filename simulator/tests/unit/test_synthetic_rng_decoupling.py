"""Regression: token content is decoupled from arrival pattern.

Pre-fix, ``SyntheticDataGenerator`` used one shared RNG for both token
content/length sampling and poisson arrival-time sampling.  The poisson
``expovariate`` draws shifted every downstream request's token stream, so
switching ``arrival_pattern`` (poisson↔staggered↔burst) silently changed the
workload — confounding cache-hit-rate comparisons, which depend on the shared
prefix tokens.  Arrival times now draw from an independent RNG, so the workload
stays fixed across arrival patterns for a given seed.
"""

from __future__ import annotations

import unittest

from simulator.config.simulator_config import (
    DatasetConfig,
    RequestArrivalConfig,
    SyntheticConfig,
)
from simulator.data.synthetic import SyntheticDataGenerator


def _gen(pattern: str, seed: int = 42):
    sc = SyntheticConfig(
        num_requests=6,
        prompt_length_dist="uniform",
        prompt_length_min=64,
        prompt_length_max=256,
        output_length_dist="uniform",
        output_length_min=32,
        output_length_max=128,
        shared_prefix_ratio=0.0,  # isolate token-content invariance from prefix reuse
    )
    arr = RequestArrivalConfig(
        arrival_pattern=pattern, poisson_rate=1.0, stagger_delay_steps=5
    )
    return SyntheticDataGenerator(sc, seed=seed, arrival_config=arr).generate()


class TestArrivalRngDecoupled(unittest.TestCase):
    def test_token_content_identical_across_arrival_patterns(self):
        # Same seed → identical prompts and ground-truth outputs regardless of
        # arrival pattern.  Cache-hit-rate experiments compare runs that differ
        # only in arrival pattern, so the workload (tokens) must not move.
        by_pattern = {p: _gen(p) for p in ("burst", "staggered", "poisson")}
        base = by_pattern["burst"]
        for p, reqs in by_pattern.items():
            for a, b in zip(base, reqs):
                self.assertEqual(
                    a.prompt_token_ids, b.prompt_token_ids,
                    f"prompt tokens differ under {p}",
                )
                self.assertEqual(
                    a.ground_truth_output, b.ground_truth_output,
                    f"ground truth differs under {p}",
                )

    def test_arrival_times_differ_across_patterns(self):
        # Decoupling must not freeze arrival times — the patterns still produce
        # distinct schedules (burst all-zero, staggered linear, poisson random).
        burst = _gen("burst")
        staggered = _gen("staggered")
        poisson = _gen("poisson")
        self.assertTrue(all(r.arrival_time == 0.0 for r in burst))
        self.assertNotEqual(
            [r.arrival_time for r in staggered],
            [r.arrival_time for r in poisson],
        )

    def test_arrival_rng_independent_of_token_rng(self):
        # The two RNGs are distinct objects (not the same stream), so drawing
        # from one does not advance the other.
        g = SyntheticDataGenerator(
            SyntheticConfig(num_requests=1), seed=42,
            arrival_config=RequestArrivalConfig(arrival_pattern="poisson"),
        )
        self.assertIsNot(g._rng, g._arrival_rng)


if __name__ == "__main__":
    unittest.main()
