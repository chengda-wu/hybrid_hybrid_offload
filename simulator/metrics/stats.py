"""Final statistical summary of a simulation run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from simulator.metrics.recorder import MetricsRecorder, RequestRecord, StepRecord


@dataclass
class SimulationReport:
    """Final statistical summary."""

    # ---- Per-step averages ----
    avg_loaded_tokens_per_step: float
    avg_computed_tokens_per_step: float
    avg_accepted_tokens_per_step: float

    # ---- Latency ----
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p99_ms: float
    avg_step_latency_ms: float

    # ---- Queue ----
    avg_waiting_queue_length: float
    max_waiting_queue_length: int

    # ---- Cache ----
    cache_hit_rate: float
    avg_cache_usage: float

    # ---- Spec decode ----
    avg_acceptance_rate: float | None   # None when no spec tokens were ever
                                        # evaluated (spec off / no decode) —
                                        # distinct from 0.0 (spec on, all rejected)
    per_position_acceptance_rates: list[float]  # marginal rate per draft position
                                                # (accepted / spec-decode steps);
                                                # should reproduce the input rates

    # ---- Throughput ----
    total_requests: int
    total_tokens_generated: int
    wall_clock_sim_time_ms: float   # Wall-clock sim_time at the last recorded
                                    # (post-warmup) step.  Accumulates from step 1,
                                    # so it INCLUDES warmup duration and idle
                                    # fast-forward gaps.  NOT the throughput
                                    # denominator — tokens_per_second divides
                                    # post-warmup generated tokens by post-warmup
                                    # GPU-busy time (sum of step_latency).
    tokens_per_second: float

    # ---- Config ----
    backend: str = ""
    kv_cache_size_gb: float = 0.0

    def to_json(self, path: str | Path | None = None) -> str:
        data = asdict(self)
        text = json.dumps(data, indent=2)
        if path is not None:
            Path(path).write_text(text)
        return text


class StatisticsComputer:
    """Computes summary statistics from recorded metrics."""

    def compute(
        self,
        recorder: MetricsRecorder,
        backend: str = "",
        kv_cache_size_gb: float = 0.0,
        acceptance_model=None,
    ) -> SimulationReport:
        steps = recorder.steps
        reqs = recorder.requests

        # Per-step averages
        avg_loaded = _mean([s.total_loaded_tokens for s in steps])
        avg_computed = _mean([s.total_computed_tokens for s in steps])
        avg_accepted = _mean([s.total_accepted_tokens for s in steps])

        # TTFT
        ttfts = [r.ttft_ms for r in reqs if r.ttft_ms is not None]
        ttft_p50 = _percentile(ttfts, 50)
        ttft_p99 = _percentile(ttfts, 99)

        # TPOT: time per output token (excludes first token / TTFT).
        # For requests with >1 output token: (total - ttft) / (output - 1)
        tpots = []
        for r in reqs:
            if r.output_length > 1 and r.ttft_ms is not None:
                tpot = (r.total_latency_ms - r.ttft_ms) / (r.output_length - 1)
                tpots.append(tpot)
        tpot_p50 = _percentile(tpots, 50)
        tpot_p99 = _percentile(tpots, 99)

        # Step latency
        avg_step_latency = _mean([s.step_latency_ms for s in steps])

        # Queue
        avg_queue = _mean([s.num_waiting for s in steps])
        max_queue = max((s.num_waiting for s in steps), default=0)

        # Cache hit rate
        total_prompt = sum(r.prompt_length for r in reqs)
        total_hits = sum(r.cache_hit_tokens_prefill for r in reqs)
        cache_hit_rate = total_hits / total_prompt if total_prompt > 0 else 0.0

        # Cache usage
        avg_cache_usage = _mean([s.cache_usage for s in steps])

        # Spec accept rate.  None when no spec tokens were ever evaluated
        # (spec disabled, or no decode steps ran) — distinguishes "spec off"
        # from "spec on but 0% accepted".
        total_accept = sum(r.num_accepted_spec_tokens for r in reqs)
        total_reject = sum(r.num_rejected_spec_tokens for r in reqs)
        total_spec = total_accept + total_reject
        avg_accept_rate = (
            total_accept / total_spec if total_spec > 0 else None
        )

        # Throughput.  Both numerator and denominator are summed over the
        # recorded (post-warmup) steps, so they are self-consistent:
        #   - numerator   = sum of tokens actually *produced* in post-warmup
        #                   steps (bonus + accepted spec), NOT the full
        #                   output_length of requests that happened to finish
        #                   post-warmup (which would double-count tokens
        #                   generated during warmup);
        #   - denominator = sum of step latencies over those same steps
        #                   (GPU-busy time, excluding warmup and idle
        #                   fast-forward gaps).
        total_generated_throughput = sum(s.total_generated_tokens for s in steps)
        total_busy_time = sum(s.step_latency_ms for s in steps)
        wall_clock_sim_time = max(s.sim_time_ms for s in steps) if steps else 0.0
        tokens_per_sec = (
            1000.0 * total_generated_throughput / total_busy_time
            if total_busy_time > 0 else 0.0
        )

        # total_tokens_generated reports the full output of requests that
        # completed post-warmup (a count of work done, not a throughput
        # numerator).
        total_generated = sum(r.output_length for r in reqs)

        per_pos_rates = (
            list(acceptance_model.per_position_acceptance_rates)
            if acceptance_model is not None else []
        )

        return SimulationReport(
            avg_loaded_tokens_per_step=round(avg_loaded, 2),
            avg_computed_tokens_per_step=round(avg_computed, 2),
            avg_accepted_tokens_per_step=round(avg_accepted, 2),
            ttft_p50_ms=round(ttft_p50, 2),
            ttft_p99_ms=round(ttft_p99, 2),
            tpot_p50_ms=round(tpot_p50, 2),
            tpot_p99_ms=round(tpot_p99, 2),
            avg_step_latency_ms=round(avg_step_latency, 2),
            avg_waiting_queue_length=round(avg_queue, 2),
            max_waiting_queue_length=max_queue,
            cache_hit_rate=round(cache_hit_rate, 4),
            avg_cache_usage=round(avg_cache_usage, 4),
            avg_acceptance_rate=(
                round(avg_accept_rate, 4) if avg_accept_rate is not None else None
            ),
            per_position_acceptance_rates=[round(r, 4) for r in per_pos_rates],
            total_requests=len(reqs),
            total_tokens_generated=total_generated,
            wall_clock_sim_time_ms=round(wall_clock_sim_time, 2),
            tokens_per_second=round(tokens_per_sec, 1),
            backend=backend,
            kv_cache_size_gb=round(kv_cache_size_gb, 2),
        )


# ---- helpers (stdlib, no numpy) ----


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile (linear interpolation)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
    return sorted_vals[f]
