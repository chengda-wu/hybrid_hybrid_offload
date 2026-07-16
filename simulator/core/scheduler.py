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
        self._spec_engine = SpeculativeDecodeEngine(config.speculative)
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
        self._warmup = config.warmup_steps  # first N steps skipped from metrics; cache NOT cleared
        self._verbose = config.verbose
        # Stall detection: counts consecutive steps where every active request
        # made zero progress (no prefill completed, no decode token generated).
        # Decode alloc failure with a full KV pool and no evictable entries can
        # otherwise loop forever — see _check_stall.  Real SGLang prevents this
        # at admission (bounding running req count); we have no such bound, so
        # guard the loop instead.
        self._stall_count: int = 0
        self._STALL_LIMIT: int = config.stall_limit

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
        total_generated = 0  # output tokens actually produced this step
                              # (bonus + accepted spec); for throughput, this
                              # is the faithful numerator — it counts only
                              # tokens generated in THIS step, so summing over
                              # post-warmup steps stays self-consistent with the
                              # post-warmup busy-time denominator.
        first_token_this_step: list[SimRequestState] = []
        for req in list(self._running.values()):
            if req.is_finished:
                continue

            had_output = req.output_length > 0
            if req.status == RequestStatus.PRE_FILL:
                loaded, computed, accepted, generated = self._handle_prefill(req)
            elif req.status == RequestStatus.DECODING:
                loaded, computed, accepted, generated = self._handle_decode(req)
            else:
                loaded = computed = accepted = generated = 0

            total_loaded += loaded
            total_computed += computed
            total_accepted += accepted
            total_generated += generated

            # A request produces its first output token in the decode step
            # where output_length goes 0 → >0.  Stamp TTFT after step_latency
            # is known (below) so it includes that step's decode latency.
            if not had_output and req.output_length > 0:
                first_token_this_step.append(req)

        # 3. Simulate GPU latency for this step (aggregate over batch)
        step_latency = 0.0
        if total_computed > 0:
            step_latency = self._gpu_perf.predict(total_loaded, total_computed)
        self._sim_time += step_latency

        # TTFT includes this step's decode latency (the cost of producing
        # the first token).
        for req in first_token_this_step:
            req.first_token_time = self._sim_time

        # Stall detection: if active requests exist but NONE advanced this
        # step, the loop is making no progress — typically all requests
        # failing decode alloc against a full KV pool with nothing evictable.
        # A step progresses if it computed any tokens (a prefill completed:
        # _handle_prefill returns num_new_tokens as computed) or generated any
        # output (a decode produced tokens).  Bail out loudly instead of
        # looping forever.  (Prefill alloc failure already raises RuntimeError
        # in _handle_prefill; this catches the decode-side equivalent that
        # silently returns zeros.)
        active = [r for r in self._running.values() if not r.is_finished]
        if active and total_computed == 0 and total_generated == 0:
            self._stall_count += 1
        else:
            self._stall_count = 0
        if self._stall_count >= self._STALL_LIMIT:
            raise RuntimeError(
                f"Scheduler stalled: {len(active)} active request(s) made no "
                f"progress for {self._stall_count} consecutive steps (KV pool "
                f"full, nothing evictable).  Increase --num-kv-cache-blocks or reduce "
                f"concurrency.  (limit={self._STALL_LIMIT})"
            )

        # 4. Record per-step metrics (skip warmup steps).
        if self._step > self._warmup:
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
                generated_tokens=total_generated,
            )

        # 5. Free finished
        self._cleanup()

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

    def _handle_prefill(self, req: SimRequestState) -> tuple[int, int, int, int]:
        """Full prefill in one step (no chunked prefill).

        Returns (loaded_tokens, computed_tokens, accepted_tokens, generated_tokens)
        for this step.  Prefill produces no output tokens, so generated=0.
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
            # Whole-prompt prefill must fit in one step (no chunked prefill by
            # design). If it never fits, retrying forever would hang the loop.
            # Fail loudly so the user increases num_kv_cache_blocks.
            raise RuntimeError(
                self._prefill_oom_message(req, num_new_tokens, num_computed)
            )

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

        return loaded, num_new_tokens, 0, 0

    def _prefill_oom_message(
        self, req: SimRequestState, num_new_tokens: int, num_computed: int
    ) -> str:
        """Build a backend-specific OOM message for a failed prefill alloc.

        vLLM packs 6 KV groups (block sizes 4..256) into one shared pool, so
        "free blocks" is a pool-block count and demand far exceeds
        tokens / scheduler_block_size — we report the real cross-group demand.
        SGLang fails on per-pool capacity (swa/full), not total free, so we
        report which pool(s) are over budget.
        """
        free = self._backend.num_free_blocks
        hint = "Increase --num-kv-cache-blocks or reduce prompt length."

        # vLLM: shared pool, real demand in pool blocks across all groups.
        required = getattr(self._backend, "last_alloc_required_blocks", None)
        if required is not None:
            return (
                f"Cannot allocate KV cache for prefill of request "
                f"{req.request_id}: needs {num_new_tokens} new tokens "
                f"(prompt={req.num_tokens}, cache_hit={num_computed}) "
                f"with only {free} free pool blocks, needs {required} pool "
                f"blocks across all KV groups.  Note: pool blocks are shared "
                f"across all KV groups, so demand is much larger than tokens / "
                f"scheduler_block_size.  {hint}"
            )

        # SGLang: per-pool capacity.  Show which pool(s) are over budget.
        failure = getattr(self._backend, "last_alloc_failure", None)
        if isinstance(failure, dict) and failure.get("over_budget_pools"):
            pool_lines = "; ".join(
                f"{p['name']}: used {p['used_slots']} + need {p['need_slots']} "
                f"> cap {p['cap_slots']} slots ({p['per_token_slots']} slots/token)"
                for p in failure["over_budget_pools"]
            )
            return (
                f"Cannot allocate KV cache for prefill of request "
                f"{req.request_id}: needs {num_new_tokens} new tokens "
                f"(prompt={req.num_tokens}, cache_hit={num_computed}).  "
                f"SGLang KV pool(s) over budget — {pool_lines}.  "
                f"Note: SGLang fails on per-pool capacity (swa/full), "
                f"not total free ({free} token slots free).  {hint}"
            )

        # Fallback (no diagnostic available).
        return (
            f"Cannot allocate KV cache for prefill of request {req.request_id}: "
            f"needs {num_new_tokens} new tokens "
            f"(prompt={req.num_tokens}, cache_hit={num_computed}) "
            f"with only {free} free blocks.  {hint}"
        )

    def _handle_decode(self, req: SimRequestState) -> tuple[int, int, int, int]:
        """One decode step.

        Returns (loaded_tokens, computed_tokens, accepted_tokens, generated_tokens).
        """
        # 1. Generate draft tokens from spec engine
        drafts = self._spec_engine.generate_draft_tokens(req)

        # Bonus token (position 0) is the model's own autoregressive prediction —
        # always produced, even when speculation is disabled (drafts == []).
        # It is taken from ground truth for simulation correctness.
        output_pos = len(req.output_token_ids)
        bonus_token: int | None = None
        if output_pos < len(req.ground_truth_output):
            bonus_token = req.ground_truth_output[output_pos]

        # drafts are the K spec tokens (ground truth at upcoming positions).
        # When speculation is off, drafts is empty → K=0.
        spec_tokens = drafts

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
            # Return loaded=0, not the request's cached-token count: this
            # request did NOT run a forward pass this step, so the GPU never
            # read its cached KV.  Counting its loaded tokens would inflate
            # predict(total_loaded, total_computed) and over-estimate step
            # latency under memory pressure.  computed=0 already excludes it
            # from the >0 gate (step_latency only added when total_computed>0),
            # but total_loaded is summed unconditionally, so it must be 0 here.
            return 0, 0, 0, 0

        # 5. _update_after_schedule: advance by all tokens
        req.advance_computed_tokens(computed)

        # 6. Evaluate acceptance on spec tokens
        num_accepted, num_rejected, num_beyond = self._acceptance.evaluate(
            req, spec_tokens
        )

        # 7. update_from_output: subtract rejected + beyond-ground-truth,
        # free their slots.  Both classes of non-accepted draft must be rolled
        # back (their slots were pre-allocated), even though only num_rejected
        # counts toward acceptance-rate metrics.
        num_to_rollback = num_rejected + num_beyond
        req.subtract_rejected_tokens(num_to_rollback)
        self._backend.free_rejected_slots(req.backend_req, num_to_rollback)

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
        # Only real rejections count toward the acceptance-rate metric;
        # drafts truncated by end-of-ground-truth are not "rejected".
        req.num_rejected_spec_tokens += num_rejected
        req.num_decode_steps += 1

        # TTFT is stamped by step() after step_latency is added to sim_time,
        # so it includes the decode cost of producing the first token.

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

        return loaded, computed, num_accepted, len(accepted_tokens)

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
                # Release the finished request's cached acceptance RNG so
                # _req_rngs doesn't grow unbounded over long runs.
                self._acceptance.forget_request(req.request_id)
                if self._step > self._warmup:
                    self._recorder.record_request_done(req, self._sim_time)
                del self._running[req_id]
                if self._verbose:
                    print(f"  [{req_id}] finished, freed")
