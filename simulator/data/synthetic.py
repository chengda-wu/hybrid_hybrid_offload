"""Synthetic request data generation with controllable prefix overlap.

This is the primary data source for cache-hit-rate experiments.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from simulator.config.simulator_config import SyntheticConfig


@dataclass
class RequestData:
    """A single request with prompt and ground-truth completion."""

    request_id: str
    prompt_token_ids: list[int]
    ground_truth_output: list[int]  # full output for acceptance verification
    arrival_time: float = 0.0  # simulation time in ms


class SyntheticDataGenerator:
    """Generates synthetic prompt/completion pairs.

    Key feature: ``shared_prefix_ratio`` determines how much of each
    request's prompt reuses tokens from preceding requests, giving
    controllable KV cache hit rates.

    The first request establishes a shared prefix. Subsequent requests
    reuse ``shared_prefix_ratio * prompt_len`` tokens from it.
    """

    # Token ID range for synthetic data (roughly LLaMA vocabulary size).
    VOCAB_SIZE = 128256

    def __init__(self, config: SyntheticConfig, seed: int = 42,
                 arrival_config=None):
        self._config = config
        self._arrival = arrival_config
        self._rng = random.Random(seed)
        self._shared_prefix: list[int] | None = None
        self._next_arrival: float = 0.0

    def generate(self) -> list[RequestData]:
        """Generate all synthetic requests."""
        results: list[RequestData] = []
        for i in range(self._config.num_requests):
            rid = f"req-{i:06d}"
            prompt_len = self._sample_prompt_length()
            output_len = self._sample_output_length()

            # Build prompt
            shared_len = 0
            if i == 0 or self._shared_prefix is None:
                # First request: build from scratch
                prompt = self._random_tokens(prompt_len)
                shared_len = self._compute_shared_len(prompt_len)
                if shared_len > 0:
                    self._shared_prefix = prompt[:shared_len]
            else:
                # Reuse shared prefix, fill remainder randomly
                assert self._shared_prefix is not None
                shared_len = min(len(self._shared_prefix), prompt_len)
                unique_len = prompt_len - shared_len
                prompt = list(self._shared_prefix[:shared_len])
                if unique_len > 0:
                    prompt += self._random_tokens(unique_len)

            ground_truth = self._random_tokens(output_len)
            arrival_time = self._compute_arrival_time(i)

            results.append(
                RequestData(
                    request_id=rid,
                    prompt_token_ids=prompt,
                    ground_truth_output=ground_truth,
                    arrival_time=arrival_time,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_tokens(self, n: int) -> list[int]:
        if n <= 0:
            return []
        return [self._rng.randint(0, self.VOCAB_SIZE - 1) for _ in range(n)]

    def _sample_prompt_length(self) -> int:
        dist = self._config.prompt_length_dist
        if dist == "fixed":
            return self._config.prompt_length_fixed
        elif dist == "uniform":
            return self._rng.randint(
                self._config.prompt_length_min,
                self._config.prompt_length_max,
            )
        elif dist == "normal":
            mu = self._config.prompt_length_fixed
            sigma = (self._config.prompt_length_max - self._config.prompt_length_min) / 4
            val = self._rng.gauss(mu, sigma)
            return max(self._config.prompt_length_min, min(int(val), self._config.prompt_length_max))
        return self._config.prompt_length_fixed

    def _sample_output_length(self) -> int:
        dist = self._config.output_length_dist
        if dist == "fixed":
            return self._config.output_length_fixed
        elif dist == "uniform":
            return self._rng.randint(
                self._config.output_length_min,
                self._config.output_length_max,
            )
        return self._config.output_length_fixed

    def _compute_shared_len(self, prompt_len: int) -> int:
        if self._config.shared_prefix_length is not None:
            return min(self._config.shared_prefix_length, prompt_len)
        return int(prompt_len * self._config.shared_prefix_ratio)

    def _compute_arrival_time(self, index: int) -> float:
        """Arrival time in ms (matching _sim_time unit)."""
        if index == 0:
            return 0.0
        if self._arrival is None:
            return index * 10.0  # default 10ms stagger

        pattern = self._arrival.arrival_pattern
        if pattern == "burst":
            return 0.0
        elif pattern == "poisson":
            # Exponential inter-arrival: mean = 1000/rate ms.
            # _next_arrival starts at 0.0 (set in __init__), so the first
            # request (index 0) returns 0.0 above; subsequent ones accumulate.
            gap = self._rng.expovariate(self._arrival.poisson_rate)
            self._next_arrival += gap * 1000.0  # seconds → ms
            return self._next_arrival
        else:  # staggered
            return index * self._arrival.stagger_delay_steps * 10.0
