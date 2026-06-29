"""Acceptance model for speculative decoding.

Two conditions must BOTH be satisfied for a draft token to be accepted:

1. Draft token must match the ground-truth token at that output position.
2. A random draw from the per-position acceptance rate must pass.

First failure breaks the chain — all subsequent draft tokens are rejected.
"""

from __future__ import annotations

import random

from simulator.config.simulator_config import SpeculativeDecodeConfig
from simulator.core.request_state import SimRequestState


class AcceptanceModel:
    """Models token acceptance in speculative decoding.

    Usage::

        model = AcceptanceModel(config, seed=42)
        num_accepted, num_rejected = model.evaluate(request, draft_tokens)
    """

    def __init__(self, config: SpeculativeDecodeConfig, seed: int = 42):
        self._config = config
        self._rng = random.Random(seed)

    def evaluate(
        self, req: SimRequestState, draft_tokens: list[int]
    ) -> tuple[int, int]:
        """Evaluate K draft tokens against ground truth and acceptance rates.

        Args:
            req: Request state with ``ground_truth_output``.
            draft_tokens: K draft tokens proposed for this decode step.

        Returns:
            (num_accepted, num_rejected).  The bonus (position 0) is NOT
            counted here — the caller handles the bonus separately.
        """
        K = len(draft_tokens)
        if K == 0:
            return 0, 0

        output_position = len(req.output_token_ids)
        num_accepted = 0

        for i in range(K):
            abs_position = output_position + i

            # Beyond ground truth — cannot verify
            if abs_position >= len(req.ground_truth_output):
                break

            ground_truth_token = req.ground_truth_output[abs_position]

            # Condition 1: token must match ground truth
            if draft_tokens[i] != ground_truth_token:
                break

            # Condition 2: per-position acceptance rate sampling
            accept_rate = self._get_accept_rate(i)
            if self._rng.random() >= accept_rate:
                break

            num_accepted += 1

        num_rejected = max(0, K - num_accepted)
        return num_accepted, num_rejected

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_accept_rate(self, position: int) -> float:
        if self._config.accept_mode == "fixed":
            return self._config.acceptance_rate
        elif self._config.accept_mode == "per_position":
            rates = self._config.acceptance_rates
            if rates:
                return rates[min(position, len(rates) - 1)]
            return 0.5
        return 0.8
