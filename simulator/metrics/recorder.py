"""Per-step and per-request metrics collection."""

from __future__ import annotations

from dataclasses import dataclass, field

from simulator.core.request_state import SimRequestState


@dataclass
class StepRecord:
    """Metrics for one scheduler step."""

    step: int
    sim_time_ms: float
    step_latency_ms: float
    num_running: int            # active requests in this step
    num_waiting: int            # waiting queue length
    cache_usage: float          # KV cache utilization [0, 1]
    total_loaded_tokens: int    # sum of cache-hit tokens across all requests
    total_computed_tokens: int  # sum of new tokens computed
    total_accepted_tokens: int  # sum of accepted spec tokens


@dataclass
class RequestRecord:
    """Metrics for one completed request."""

    request_id: str
    prompt_length: int
    output_length: int

    # Latency
    ttft_ms: float | None          # time to first token
    total_latency_ms: float        # arrival → finish

    # Token counts
    cache_hit_tokens_prefill: int  # prefix-cache hit at prefill
    num_decode_steps: int
    num_accepted_spec_tokens: int
    num_rejected_spec_tokens: int


class MetricsRecorder:
    """Collects per-step and per-request metrics during simulation."""

    def __init__(self):
        self.steps: list[StepRecord] = []
        self.requests: list[RequestRecord] = []
        self._step_accepted: int = 0

    def record_step(
        self,
        step: int,
        sim_time: float,
        step_latency: float,
        num_running: int,
        num_waiting: int,
        cache_usage: float,
        loaded_tokens: int,
        computed_tokens: int,
        accepted_tokens: int,
    ) -> None:
        self.steps.append(
            StepRecord(
                step=step,
                sim_time_ms=sim_time,
                step_latency_ms=step_latency,
                num_running=num_running,
                num_waiting=num_waiting,
                cache_usage=cache_usage,
                total_loaded_tokens=loaded_tokens,
                total_computed_tokens=computed_tokens,
                total_accepted_tokens=accepted_tokens,
            )
        )

    def record_request_done(self, req: SimRequestState, sim_time: float) -> None:
        """Record final metrics for a finished request."""
        ttft = None
        if req.first_token_time is not None:
            ttft = req.first_token_time - req.arrival_time

        self.requests.append(
            RequestRecord(
                request_id=req.request_id,
                prompt_length=len(req.prompt_token_ids),
                output_length=len(req.output_token_ids),
                ttft_ms=ttft,
                total_latency_ms=sim_time - req.arrival_time,
                cache_hit_tokens_prefill=req.num_cache_hits_on_prefill,
                num_decode_steps=req.num_decode_steps,
                num_accepted_spec_tokens=req.num_accepted_spec_tokens,
                num_rejected_spec_tokens=req.num_rejected_spec_tokens,
            )
        )
