"""Draft token generation for speculative decoding simulation.

Drafts are the ground-truth tokens at the upcoming positions. Draft quality
is NOT modeled here — the per-position acceptance rate (measured end-to-end
on a real speculator) is applied downstream by AcceptanceModel and already
encodes how often a draft is correct *and* verified. There is no separate
"draft accuracy" knob.
"""

from __future__ import annotations

from simulator.config.simulator_config import SpeculativeDecodeConfig
from simulator.core.request_state import SimRequestState


class SpeculativeDecodeEngine:
    """Generates draft tokens for the simulation.

    Returns the K spec-draft tokens (ground truth at the upcoming output
    positions). Acceptance is decided downstream by AcceptanceModel.
    """

    def __init__(self, config: SpeculativeDecodeConfig):
        self._config = config

    def generate_draft_tokens(self, req: SimRequestState) -> list[int]:
        """Generate up to K draft tokens for the current decode step.

        Drafts equal ground truth at positions ``output_pos+1 .. output_pos+K``
        (the +1 skips the bonus token, which the scheduler takes from ground
        truth directly). Truncated at end of ground truth.
        """
        K = self._config.num_spec_tokens
        if K == 0 or not self._config.enabled:
            return []

        output_pos = len(req.output_token_ids)
        drafts: list[int] = []
        for i in range(K):
            abs_pos = output_pos + 1 + i
            if abs_pos >= len(req.ground_truth_output):
                break
            drafts.append(req.ground_truth_output[abs_pos])
        return drafts
