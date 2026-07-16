"""SimRequestState — canonical request state for the simulation scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class RequestStatus(Enum):
    QUEUED = auto()    # Not yet admitted to the scheduler
    PRE_FILL = auto()  # Initial full prefill step pending
    DECODING = auto()  # Active decode loop (including spec decode)
    FINISHED = auto()  # Output complete or stopped


@dataclass
class SimRequestState:
    """Unified request state for the simulation scheduler.

    This is the canonical state that the scheduler operates on.
    It wraps backend-specific request objects through ``backend_req``.
    """

    # ---- Identity & data (set at creation) ----

    request_id: str
    prompt_token_ids: list[int]
    ground_truth_output: list[int]  # full ground truth for acceptance sampling
    max_output_tokens: int          # max_tokens from sampling params

    # ---- Running state ----

    status: RequestStatus = RequestStatus.QUEUED
    output_token_ids: list[int] = field(default_factory=list)
    spec_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    # ---- Timing ----

    arrival_time: float = 0.0               # sim time when request arrived
    first_token_time: float | None = None   # TTFT timestamp (ms)
    finish_time: float | None = None        # when FINISHED

    # ---- Backend handle ----

    backend_req: Any | None = None  # vLLMSimRequest | SGLangSimRequest

    # ---- Counters for metrics ----

    num_prefill_tokens: int = 0
    num_decode_steps: int = 0
    num_accepted_spec_tokens: int = 0
    num_rejected_spec_tokens: int = 0
    num_cache_hits_on_prefill: int = 0

    # ---- Derived ----

    @property
    def num_tokens(self) -> int:
        """Total tokens currently in the sequence (including spec)."""
        return len(self.prompt_token_ids) + len(self.output_token_ids) + len(self.spec_token_ids)

    @property
    def is_prefill_needed(self) -> bool:
        return self.status == RequestStatus.PRE_FILL

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    @property
    def is_admitted(self) -> bool:
        return self.status not in (RequestStatus.QUEUED, RequestStatus.PRE_FILL)

    @property
    def output_length(self) -> int:
        return len(self.output_token_ids)

    def advance_computed_tokens(self, n: int) -> None:
        """Advance num_computed_tokens (mimics _update_after_schedule)."""
        self.num_computed_tokens += n

    def subtract_rejected_tokens(self, n: int) -> None:
        """Subtract rejected spec tokens (mimics update_from_output)."""
        self.num_computed_tokens = max(0, self.num_computed_tokens - n)

    def append_output_tokens(self, token_ids: list[int]) -> None:
        self.output_token_ids.extend(token_ids)

    def clear_spec_tokens(self) -> None:
        self.spec_token_ids = []
