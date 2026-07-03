"""SimulatorScheduler — the main event loop.

Token-level simulation of a continuous-batching scheduler.
Each step processes all active requests, simulating one forward pass.
"""

from __future__ import annotations

from collections import deque

from simulator.config.simulator_config import SimulatorConfig
from simulator.core.request_state import RequestStatus, SimRequestState
from simulator.kv_cache.base import KVBackend
from simulator.metrics.gpu_perf_model import GPUPerfModel
from simulator.metrics.recorder import MetricsRecorder
from simulator.speculative.acceptance import AcceptanceModel
from simulator.speculative.engine import SpeculativeDecodeEngine


class SimulatorScheduler:
    """Main simulation scheduler loop."""

    def __init__(
        self,
        config: SimulatorConfig,
        kv_backend: KVBackend,
        acceptance_model: AcceptanceModel,
        gpu_perf_model: GPUPerfModel,
        recorder: MetricsRecorder,
    ):
        self._config = config
        self._backend = kv_backend
        self._spec_engine = SpeculativeDecodeEngine(
            config.speculative, seed=config.random_seed
        )
        self._acceptance = acceptance_model
        self._gpu_perf = gpu_perf_model
        self._recorder = recorder

        # Queues
        self._waiting: deque[SimRequestState] = deque()
        self._running: dict[str, SimRequestState] = {}
        self._all: dict[str, SimRequestState] = {}

        # State
        self._step: int = 0
        self._sim_time: float = 0.0  # ms
        self._warmup = config.warmup_steps
        self._warmup_reset_done: bool = False
        self._verbose = config.verbose

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self, requests: list[SimRequestState]) -> None:
        """Load all requests, sorted by arrival time."""
        for r in sorted(requests, key=lambda r: r.arrival_time):
            self._all[r.request_id] = r
            self._waiting.append(r)

    def step(self) -> bool:
        """Run one scheduling step. Returns False when all requests finished."""
        self._step += 1

        # 1. Admit waiting requests
        self._admit()

        # 2. Process each active request; accumulate step-level counters
        total_loaded = 0
        total_computed = 0
        total_accepted = 0
        for req in list(self._running.values()):
            if req.is_finished:
                continue

            if req.status == RequestStatus.PRE_FILL:
                loaded, computed, accepted = self._handle_prefill(req)
            elif req.status == RequestStatus.DECODING:
                loaded, computed, accepted = self._handle_decode(req)
            else:
                loaded = computed = accepted = 0

            total_loaded += loaded
            total_computed += computed
            total_accepted += accepted

        # 3. Simulate GPU latency for this step (aggregate over batch)
        step_latency = 0.0
        if total_computed > 0:
            step_latency = self._gpu_perf.predict(total_loaded, total_computed)
        self._sim_time += step_latency

        # 4. Record per-step metrics (only after warmup cache has been cleared).
        if self._step > self._warmup and self._warmup_reset_done:
            active_count = sum(1 for r in self._running.values() if not r.is_finished)
            self._recorder.record_step(
                step=self._step,
                sim_time=self._sim_time,
                step_latency=step_latency,
                num_running=active_count,
                num_waiting=len(self._waiting),
                cache_usage=self._backend.usage,
                loaded_tokens=total_loaded,
                computed_tokens=total_computed,
                accepted_tokens=total_accepted,
            )

        # 5. Free finished
        self._cleanup()

        # 5a. Reset cache at warmup boundary (first idle step after warmup).
        #     Warmup allocations are cleared before measurement phase begins.
        if (self._step > self._warmup and not self._warmup_reset_done
                and not self._running):
            self._backend.reset()
            self._warmup_reset_done = True

        # 6. If no running requests but waiting queue has arrivals in the
        #    future, fast-forward sim_time to the next arrival.
        if not self._running and self._waiting:
            next_arrival = self._waiting[0].arrival_time
            if next_arrival > self._sim_time:
                self._sim_time = next_arrival

        remaining = sum(1 for r in self._running.values() if not r.is_finished)
        return remaining > 0 or len(self._waiting) > 0

    # ------------------------------------------------------------------
    # Per-request handlers
    # ------------------------------------------------------------------

    def _handle_prefill(self, req: SimRequestState) -> tuple[int, int, int]:
        """Full prefill in one step (no chunked prefill).

        Returns (loaded_tokens, computed_tokens, accepted_tokens) for this step.
        """
        backend = self._backend

        # Register with backend (idempotent — safe on retry)
        backend.register_request(req.backend_req)

        # Find prefix cache hits
        blocks, num_computed = backend.get_computed_blocks(req.backend_req)
        loaded = num_computed  # cache-hit tokens are "loaded" from cache
        req.num_computed_tokens = num_computed
        req.num_cache_hits_on_prefill = num_computed

        # Allocate remaining prompt tokens
        num_new_tokens = req.num_tokens - num_computed
        allocated = backend.allocate_slots(
            req.backend_req,
            num_new_tokens=num_new_tokens,
            num_new_computed_tokens=num_computed,
            new_computed_blocks=blocks,
        )
        if allocated is None:
            # Cannot allocate — leave in PRE_FILL to retry next step
            if self._verbose:
                print(f"  [{req.request_id}] prefill allocation failed, retrying")
            return 0, 0, 0

        req.allocated_blocks = allocated
        req.num_computed_tokens = req.num_tokens
        req.num_prefill_tokens = num_new_tokens

        # Insert prefill tokens into radix tree (SGLang: cache_unfinished_req)
        backend.sync_state(req.backend_req, [])

        # Transition to decode
        req.status = RequestStatus.DECODING
        if self._verbose:
            print(
                f"  [{req.request_id}] prefill done: "
                f"cache_hit={num_computed}, new_tokens={num_new_tokens}"
            )

        return loaded, num_new_tokens, 0

    def _handle_decode(self, req: SimRequestState) -> tuple[int, int, int]:
        """One decode step.

        Returns (loaded_tokens, computed_tokens, accepted_tokens).
        """
        # 1. Generate draft tokens from spec engine
        drafts = self._spec_engine.generate_draft_tokens(req)

        # Bonus is position 0 (always from ground truth for correctness)
        bonus_token: int | None = None
        spec_tokens: list[int] = []
        if drafts:
            output_pos = len(req.output_token_ids)
            if output_pos < len(req.ground_truth_output):
                bonus_token = req.ground_truth_output[output_pos]
            spec_tokens = drafts[1:]  # drafts[1:] are the K spec tokens

        K = len(spec_tokens)

        # 2. Set spec tokens on backend handle
        req.spec_token_ids = spec_tokens
        self._backend.set_spec_tokens(req.backend_req, spec_tokens)

        # 3. Loaded = what's already in cache; Computed = 1+K for this step
        loaded = req.num_computed_tokens
        computed = 1 + K

        # 4. Allocate slots for 1+K tokens
        allocated = self._backend.allocate_slots(
            req.backend_req,
            num_new_tokens=computed,
        )
        if allocated is None:
            if self._verbose:
                print(f"  [{req.request_id}] decode alloc failed, skipping step")
            req.clear_spec_tokens()
            return loaded, 0, 0

        req.allocated_blocks = allocated

        # 5. _update_after_schedule: advance by all tokens
        req.advance_computed_tokens(computed)

        # 6. Evaluate acceptance on spec tokens
        num_accepted, num_rejected = self._acceptance.evaluate(req, spec_tokens)

        # 7. update_from_output: subtract rejected, free rejected slots
        req.subtract_rejected_tokens(num_rejected)
        self._backend.free_rejected_slots(req.backend_req, num_rejected)

        # 8. Append accepted output tokens
        accepted_tokens: list[int] = []
        if bonus_token is not None:
            accepted_tokens.append(bonus_token)
        accepted_tokens.extend(spec_tokens[:num_accepted])

        req.append_output_tokens(accepted_tokens)

        # Clear spec tokens BEFORE syncing to backend (sync should reflect
        # only accepted tokens, not pending draft tokens)
        req.clear_spec_tokens()

        self._backend.sync_state(req.backend_req, req.output_token_ids)

        req.num_accepted_spec_tokens += num_accepted
        req.num_rejected_spec_tokens += num_rejected
        req.num_decode_steps += 1

        # 9. Timing
        if req.first_token_time is None and req.output_length > 0:
            req.first_token_time = self._sim_time

        # 10. Check stop
        if req.output_length >= req.max_output_tokens:
            req.status = RequestStatus.FINISHED
            req.finish_time = self._sim_time

        if self._verbose:
            print(
                f"  [{req.request_id}] decode: bonus={bonus_token}, "
                f"K={K}, accepted={num_accepted}, rejected={num_rejected}, "
                f"output={req.output_length}"
            )

        return loaded, computed, num_accepted

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _admit(self) -> None:
        """Admit waiting requests whose arrival time has passed."""
        while self._waiting:
            req = self._waiting[0]
            if req.arrival_time > self._sim_time:
                break
            self._waiting.popleft()
            req.status = RequestStatus.PRE_FILL
            self._running[req.request_id] = req
            if self._verbose:
                print(f"  [{req.request_id}] admitted at step {self._step}")

    def _cleanup(self) -> None:
        """Free finished requests."""
        for req_id, req in list(self._running.items()):
            if req.is_finished:
                self._backend.free(req.backend_req)
                if self._step > self._warmup and self._warmup_reset_done:
                    self._recorder.record_request_done(req, self._sim_time)
                del self._running[req_id]
                if self._verbose:
                    print(f"  [{req_id}] finished, freed")
