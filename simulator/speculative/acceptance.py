"""Acceptance model for speculative decoding.

Input ``acceptance_rates`` are **marginal** per-position acceptance rates —
the real, end-to-end measured fraction of decode steps where draft position
``i`` was accepted (``P(draft_i accepted)``), already encoding both draft
correctness and target verification. They must be non-increasing (drafts
farther down the chain are accepted no more often than earlier ones).

Because real spec decode chain-breaks (the verifier stops at the first
rejection, so position ``i`` can only be accepted if all of ``0..i-1`` were),
the marginal rates are reproduced by sampling a **conditional** rate at each
position derived from the marginals::

    cond[i] = marginal[i] / marginal[i-1]      (marginal[-1] = 1.0)

which gives ``P(accept_i) = prod_{j<=i} cond[j] = marginal[i]``.

First sampled failure breaks the chain; all subsequent drafts in the step are
rejected.
"""

from __future__ import annotations

import random

from simulator.config.simulator_config import SpeculativeDecodeConfig
from simulator.core.request_state import SimRequestState


class AcceptanceModel:
    """Models token acceptance in speculative decoding.

    Usage::

        model = AcceptanceModel(config, seed=42)
        num_accepted, num_rejected, num_beyond = model.evaluate(request, draft_tokens)

    A single RNG is shared across all requests and consumed in scheduling
    order, so results are reproducible for a fixed schedule but
    inter-request-coupled and order-dependent.
    """

    def __init__(self, config: SpeculativeDecodeConfig, seed: int = 42):
        self._config = config
        self._rng = random.Random(seed)
        self._K = config.num_spec_tokens

        # Per-position accumulators for the report (marginal rates, caliber A:
        # denominator = all spec-decode steps).  A single global step count is
        # used for every position so that rates[i] = accepted[i] / total_steps
        # reproduces the marginal P(draft_i accepted).  Later positions are
        # naturally lower for short outputs whose sequences end mid-chain (those
        # steps still count in the denominator but the trailing positions had
        # no draft to accept) — this matches how a real speculator's
        # end-to-end per-position rate behaves near EOS, and converges to the
        # input marginal for long outputs.
        self._per_pos_accepted: list[int] = [0] * self._K
        self._total_spec_steps: int = 0

        # Precompute conditional rates from the marginal input.
        # per_position mode: acceptance_rates are marginal; convert to conditional.
        # fixed mode: acceptance_rate is a single conditional rate (legacy simple
        # knob); marginals then decay geometrically and are reported as-is.
        self._cond_rates: list[float] = []
        if config.accept_mode == "per_position":
            rates = config.acceptance_rates or []
            if rates:
                if len(rates) < self._K:
                    raise ValueError(
                        f"acceptance_rates has {len(rates)} entries "
                        f"but num_spec_tokens={self._K}. "
                        f"Provide at least {self._K} marginal rates "
                        f"(one per draft position)."
                    )
                self._cond_rates = self._marginal_to_conditional(rates[: self._K])
            elif self._K > 0:
                # No rates provided in per_position mode — fall back to a flat
                # 0.5 conditional rate so the simulation still runs, but warn:
                # this is a guess, not a measurement, and the reported
                # per_position_acceptance_rates will reflect it.
                import logging

                logging.getLogger(__name__).warning(
                    "accept_mode='per_position' but acceptance_rates is empty "
                    "— falling back to a flat 0.5 conditional rate for all %d "
                    "draft positions.  Pass --acceptance-rates (real measured "
                    "marginal rates) for a faithful simulation.  (This warning "
                    "is printed once.)",
                    self._K,
                )
                self._cond_rates = [0.5] * self._K

    @staticmethod
    def _marginal_to_conditional(marginal: list[float]) -> list[float]:
        """Convert non-increasing marginal rates to chain-conditional rates.

        ``cond[i] = marginal[i] / marginal[i-1]`` (marginal[-1] = 1.0).
        Requires ``marginal`` to be non-increasing; otherwise the input is
        physically inconsistent with chain-breaking spec decode.
        """
        cond: list[float] = []
        prev = 1.0
        for i, m in enumerate(marginal):
            if not 0.0 <= m <= 1.0:
                raise ValueError(
                    f"acceptance_rates must be in [0,1]; got {m} at position {i}"
                )
            if m > prev + 1e-9:
                raise ValueError(
                    f"acceptance_rates must be non-increasing (marginal rates): "
                    f"position {i} rate {m} > position {i-1} rate {prev}. "
                    f"Chain-breaking spec decode cannot accept a later draft "
                    f"more often than an earlier one."
                )
            cond.append(0.0 if prev == 0.0 else m / prev)
            prev = m
        return cond

    def evaluate(
        self, req: SimRequestState, draft_tokens: list[int]
    ) -> tuple[int, int, int]:
        """Evaluate K draft tokens against the conditional acceptance rates.

        Args:
            req: Request state with ``ground_truth_output``.
            draft_tokens: K draft tokens proposed for this decode step. Only
                the length is used (drafts are assumed to be ground truth,
                generated by SpeculativeDecodeEngine); acceptance is decided
                purely by the rate.

        Returns:
            (num_accepted, num_rejected, num_beyond_ground_truth).
            - num_accepted: drafts that passed the rate draw.
            - num_rejected: drafts that failed the rate draw before the chain
              broke. These are the meaningful rejections for acceptance-rate
              metrics.
            - num_beyond_ground_truth: drafts that could not be evaluated
              because they fall past the end of ground truth. NOT real
              rejections — the sequence simply ended — so excluded from
              acceptance-rate metrics. The caller still frees their slots
              (num_accepted + num_rejected + num_beyond == K).

            The bonus (position 0) is NOT counted here — the caller handles
            the bonus separately.
        """
        K = len(draft_tokens)
        if K == 0:
            return 0, 0, 0

        output_position = len(req.output_token_ids)
        num_accepted = 0
        beyond_ground_truth = 0

        # One spec-decode step evaluated (caliber-A marginal denominator:
        # every step with ≥1 draft counts, including those whose drafts fall
        # beyond ground truth — they are still spec-decode steps).
        if K <= self._K:
            self._total_spec_steps += 1

        for i in range(K):
            # +1 because the bonus token occupies output_position;
            # the first draft token is at output_position + 1
            abs_position = output_position + 1 + i

            # Beyond ground truth — cannot verify.  Not a rejection.
            if abs_position >= len(req.ground_truth_output):
                beyond_ground_truth = K - i
                break

            # Conditional-rate sampling (derived from the marginal input).
            # First miss breaks the chain.
            accept_rate = self._get_cond_rate(i)
            if self._rng.random() >= accept_rate:
                break

            num_accepted += 1
            if i < self._K:
                self._per_pos_accepted[i] += 1

        num_rejected = max(0, K - num_accepted - beyond_ground_truth)
        return num_accepted, num_rejected, beyond_ground_truth

    @property
    def per_position_acceptance_rates(self) -> list[float]:
        """Measured marginal per-position acceptance rates (caliber A).

        ``rates[i] = accepted[i] / total_spec_steps`` where the denominator is
        every spec-decode step (steps with ≥1 draft), matching the end-to-end
        per-position rate a real speculator reports.  Converges to the input
        marginal ``acceptance_rates[i]`` for long outputs.  Empty when no spec
        steps ran.
        """
        if self._total_spec_steps == 0:
            return []
        return [a / self._total_spec_steps for a in self._per_pos_accepted]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_cond_rate(self, position: int) -> float:
        if self._config.accept_mode == "fixed":
            return self._config.acceptance_rate
        # per_position: precomputed conditional rates.
        if self._cond_rates:
            return self._cond_rates[min(position, len(self._cond_rates) - 1)]
        return 0.5
