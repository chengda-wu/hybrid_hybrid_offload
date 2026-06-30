"""Draft token generation for speculative decoding simulation.

Draft tokens are generated probabilistically: they match the ground-truth
token with ``draft_accuracy`` probability (configurable per-position quality),
simulating a speculator of a given quality level without implementing an
actual model (EAGLE, Medusa, etc.).
"""

from __future__ import annotations

import random

from simulator.config.simulator_config import SpeculativeDecodeConfig
from simulator.core.request_state import SimRequestState


class SpeculativeDecodeEngine:
    """Generates draft tokens for the simulation.

    The first token in the returned list is the *bonus* token (model's
    autoregressive prediction).  Remaining tokens are speculative (draft).

    For simulation we probabilistically generate the ground-truth token
    (correct draft) or a random token (incorrect draft), letting us
    control acceptance behavior for experiments.
    """

    # Random token range for incorrect drafts
    RANDOM_TOKEN_MAX = 128256

    def __init__(self, config: SpeculativeDecodeConfig, seed: int = 42):
        self._config = config
        self._rng = random.Random(seed)

    def generate_draft_tokens(self, req: SimRequestState) -> list[int]:
        """Generate 1+K draft tokens for the current decode step.

        Returns:
            list[int]: [bonus_token, draft_0, draft_1, ..., draft_{K-1}]
            with length 1 + K (or shorter if we run past ground truth).
        """
        K = self._config.num_spec_tokens
        if K == 0 or not self._config.enabled:
            return []

        output_pos = len(req.output_token_ids)
        max_needed = 1 + K  # bonus + spec tokens
        drafts: list[int] = []

        for i in range(max_needed):
            abs_pos = output_pos + i
            if abs_pos >= len(req.ground_truth_output):
                break

            if self._rng.random() < self._config.draft_accuracy:
                # Correct draft
                drafts.append(req.ground_truth_output[abs_pos])
            else:
                # Incorrect — random token
                drafts.append(self._rng.randint(0, self.RANDOM_TOKEN_MAX - 1))

        return drafts
