"""Round-10 F4 regression: failed decode allocation must not contribute its
cached tokens to the GPU performance model.

Pre-fix, ``_handle_decode`` returned ``(loaded, 0, 0, 0)`` when allocation
failed — so the request's cached-token count (``loaded``) was summed into
``total_loaded`` and fed to ``predict(total_loaded, total_computed)``, even
though the request ran NO forward pass that step and the GPU never read its
cached KV.  Post-fix it returns ``(0, 0, 0, 0)``.

This drives the scheduler with a fake backend whose ``allocate_slots`` fails
on the first decode of one request but succeeds for another, then asserts the
failed request's cached tokens are absent from the perf-model input.
"""

from __future__ import annotations

import unittest
from typing import Any

from simulator.config.simulator_config import (
    GPUPerfConfig,
    SimulatorConfig,
    SpeculativeDecodeConfig,
)
from simulator.core.request_state import RequestStatus, SimRequestState
from simulator.core.scheduler import SimulatorScheduler
from simulator.kv_cache.base import KVBackend
from simulator.metrics.gpu_perf_model import GPUPerfModel
from simulator.metrics.recorder import MetricsRecorder
from simulator.speculative.acceptance import AcceptanceModel


class _FakeReq:
    """Minimal backend handle; the scheduler only stores it on the state."""

    def __init__(self, rid: str):
        self.request_id = rid


class _FailDecodeBackend(KVBackend):
    """A fake KVBackend where one request's first decode allocation fails.

    ``_fail_req`` is the request id whose decode allocate_slots returns None
    (simulating a full KV pool); every other call succeeds.  All other methods
    are no-ops/stubs — the scheduler only needs allocate_slots + lifecycle
    plumbing to reach the perf-model call.
    """

    def __init__(self, fail_req: str, prompt_len: int):
        self._fail_req = fail_req
        self._prompt_len = prompt_len
        self._decode_seen: set[str] = set()

    # -- lifecycle (mostly unused) --
    def create_request(self, request_id, prompt_token_ids, max_tokens):
        return _FakeReq(request_id)

    def register_request(self, sim_req):
        pass

    def get_computed_blocks(self, sim_req):
        # No prefix cache hits — full prompt must be allocated.
        return (None, 0)

    def allocate_slots(self, sim_req, num_new_tokens, num_new_computed_tokens=0,
                       new_computed_blocks=None):
        # First alloc is the prefill (num_new_tokens == prompt length) → OK.
        # Subsequent allocs are decodes (num_new_tokens == 1+K).
        if num_new_tokens <= self._prompt_len and num_new_tokens > 2:
            return object()  # prefill success sentinel
        # Decode path: fail only the target request, once.
        if sim_req.request_id == self._fail_req and sim_req.request_id not in self._decode_seen:
            self._decode_seen.add(sim_req.request_id)
            return None
        return object()

    def set_spec_tokens(self, sim_req, tokens):
        pass

    def sync_state(self, sim_req, output_token_ids):
        pass

    def free(self, sim_req):
        pass

    @property
    def usage(self):
        return 0.0

    @property
    def total_bytes(self):
        return 0

    @property
    def name(self):
        return "fake"


class _CapturingPerfModel(GPUPerfModel):
    """Records the (loaded, computed) args passed to predict()."""

    def __init__(self):
        super().__init__(GPUPerfConfig())
        self.calls: list[tuple[int, int]] = []

    def predict(self, loaded_tokens, computed_tokens):  # type: ignore[override]
        self.calls.append((loaded_tokens, computed_tokens))
        return 1.0  # constant latency so the loop progresses


def _make_state(rid: str, prompt_len: int, out_len: int) -> SimRequestState:
    s = SimRequestState(
        request_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        ground_truth_output=list(range(10000, 10000 + out_len)),
        max_output_tokens=out_len,
        arrival_time=0.0,
    )
    s.num_computed_tokens = prompt_len  # already prefilled
    s.status = RequestStatus.DECODING
    s.backend_req = _FakeReq(rid)
    return s


class TestFailedDecodeExcludedFromPerfModel(unittest.TestCase):
    def test_failed_decode_loaded_not_counted(self):
        # Two decoding requests: 1000 and 4000 cached tokens.  The 4000-token
        # request fails its first decode alloc.  Pre-fix the perf model would
        # see loaded = 1000 + 4000 = 5000 (the failed req's 4000 wrongly
        # included).  Post-fix it sees only 1000 (the successful req).
        prompt_len = 100
        ok_state = _make_state("ok", prompt_len, out_len=50)
        fail_state = _make_state("fail", prompt_len, out_len=50)
        # Give the fail request a large cached-token count so its inclusion
        # would be unmistakable in the captured loaded total.
        fail_state.num_computed_tokens = 4000

        backend = _FailDecodeBackend(fail_req="fail", prompt_len=prompt_len)
        config = SimulatorConfig(
            speculative=SpeculativeDecodeConfig(enabled=False, num_spec_tokens=0),
            warmup_steps=0, stall_limit=50,
        )
        perf = _CapturingPerfModel()
        sched = SimulatorScheduler(
            config=config, kv_backend=backend,
            acceptance_model=AcceptanceModel(config.speculative, seed=0),
            gpu_perf_model=perf, recorder=MetricsRecorder(),
        )
        sched._running = {"ok": ok_state, "fail": fail_state}

        sched.step()

        # At least one predict call captured; the failed req (4000 cached)
        # must NOT appear in loaded.  The ok req contributes ~prompt_len
        # (num_computed_tokens) tokens, well under 4000.
        self.assertTrue(perf.calls, "perf model was never called")
        max_loaded = max(loaded for loaded, _ in perf.calls)
        self.assertLess(max_loaded, 4000,
                        f"failed decode's cached tokens leaked into perf model: "
                        f"max_loaded={max_loaded}, calls={perf.calls}")


if __name__ == "__main__":
    unittest.main()
