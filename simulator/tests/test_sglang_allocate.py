"""P1 regression test: SGLang allocate_slots must not double-allocate."""

import importlib.util
import math
import unittest

_HAS_SGLANG = importlib.util.find_spec("sglang") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_sglang = unittest.skipUnless(
    _HAS_SGLANG and _HAS_TORCH, "requires sglang+torch"
)

from simulator.config.model_config import (
    KVBackendConfig,
    ModelArchitecture,
)
from simulator.kv_cache.sglang_backend import SGLangBackend


def _make_backend(num_blocks=4096):
    arch = ModelArchitecture.deepseek_v4_flash()
    bsizes = [g[1] for g in arch.layer_groups]
    lcm = bsizes[0]
    gcd = bsizes[0]
    for bs in bsizes[1:]:
        lcm = lcm * bs // math.gcd(lcm, bs)
        gcd = math.gcd(gcd, bs)
    bc = KVBackendConfig(
        model_arch=arch,
        block_size=max(bsizes),
        hash_block_size=gcd,
        max_model_len=8192,
        num_kv_cache_blocks=num_blocks,
        scheduler_block_size=lcm,
    )
    return SGLangBackend(bc)


@requires_sglang
class TestAllocateSlots(unittest.TestCase):
    """P1: allocate_slots must allocate exactly num_new_tokens."""

    def test_prefill_no_cache_hit(self):
        backend = _make_backend()
        req = backend.create_request("r1", list(range(64)), max_tokens=10)
        backend.register_request(req)
        indices = backend.allocate_slots(req, num_new_tokens=64)
        self.assertEqual(len(indices), 64,
                         f"expected 64 tokens, got {len(indices)} (2x bug?)")

    def test_decode_does_not_reallocate_entire_sequence(self):
        backend = _make_backend()
        req = backend.create_request("r1", list(range(128)), max_tokens=20)
        backend.register_request(req)
        # Prefill
        backend.allocate_slots(req, num_new_tokens=128)
        backend.sync_state(req, [])
        # Decode: K=2, should allocate 3 (1+K), not 131
        indices = backend.allocate_slots(req, num_new_tokens=3)
        self.assertEqual(len(indices), 3,
                         f"expected 3, got {len(indices)} (sequence re-allocation?)")

    def test_prefill_with_cache_hit(self):
        backend = _make_backend()
        # Insert a prefix — need >= page_size(256) tokens for RadixTree match
        prompt = list(range(512))
        req1 = backend.create_request("r1", prompt, max_tokens=10)
        backend.register_request(req1)
        backend.allocate_slots(req1, num_new_tokens=512)
        backend.sync_state(req1, [])

        # Second request shares the first 256 tokens of prefix
        req2 = backend.create_request("r2", prompt, max_tokens=10)
        backend.register_request(req2)
        _, num_computed = backend.get_computed_blocks(req2)
        # Should hit at least 256 (page-aligned)
        self.assertGreaterEqual(num_computed, 256)
        # Only allocate remaining
        remaining = 512 - num_computed
        indices = backend.allocate_slots(
            req2, num_new_tokens=remaining,
            num_new_computed_tokens=num_computed,
        )
        self.assertEqual(len(indices), remaining,
                         f"expected {remaining}, got {len(indices)}")

    def test_cache_hit_sync_state_value_equals_key_length(self):
        """After cache-hit prefill, sync_state must build value == key length."""
        backend = _make_backend()
        prompt = list(range(512))
        # First request: populate the cache
        req1 = backend.create_request("r1", prompt, max_tokens=10)
        backend.register_request(req1)
        backend.allocate_slots(req1, num_new_tokens=512)
        backend.sync_state(req1, [])

        # Second request: shares prefix, cache hit
        req2 = backend.create_request("r2", prompt, max_tokens=10)
        backend.register_request(req2)
        blocks, num_computed = backend.get_computed_blocks(req2)
        self.assertGreaterEqual(num_computed, 256)

        remaining = 512 - num_computed
        backend.allocate_slots(
            req2, num_new_tokens=remaining,
            num_new_computed_tokens=num_computed,
            new_computed_blocks=blocks,
        )
        # This is the critical path — sync_state after cache-hit prefill.
        # Before the fix, value was shorter than key (prefix indices missing).
        # After the fix, _allocated_indices = [prefix, new] → value == key.
        backend.sync_state(req2, [])
        # No exception raised → value length matched key length.

    def test_sync_state_key_includes_bonus_token(self):
        backend = _make_backend()
        req = backend.create_request("r1", list(range(512)), max_tokens=10)
        backend.register_request(req)
        # Prefill + sync
        backend.allocate_slots(req, num_new_tokens=512)
        backend.sync_state(req, [])
        # Decode: allocate 1 token, then sync with bonus
        backend.allocate_slots(req, num_new_tokens=1)
        backend.sync_state(req, [100])  # bonus=100
        # New request: should match prefix including the bonus
        req2 = backend.create_request("r2", list(range(512)) + [100], max_tokens=10)
        backend.register_request(req2)
        _, num_matched = backend.get_computed_blocks(req2)
        self.assertGreaterEqual(num_matched, 256,
                                f"bonus token may not be in tree: matched {num_matched}")

    def test_free_rejected_slots_frees_only_tail(self):
        """free_rejected_slots must free exactly the last N allocated tokens.

        Locks the tail-assumption contract: a decode step allocates 1+K tokens
        appended at the tail; rejecting N of them frees the global last N
        indices and leaves the prefix (512 + 1 accepted) intact.
        """
        import torch

        backend = _make_backend()
        req = backend.create_request("r1", list(range(512)), max_tokens=20)
        backend.register_request(req)
        # Prefill: 512 tokens
        backend.allocate_slots(req, num_new_tokens=512)
        backend.sync_state(req, [])

        n_before = sum(len(t) for t in req._allocated_indices)
        self.assertEqual(n_before, 512)

        # Decode step: allocate 1+K = 3 tokens (bonus + 2 spec)
        backend.allocate_slots(req, num_new_tokens=3)
        n_after_alloc = sum(len(t) for t in req._allocated_indices)
        self.assertEqual(n_after_alloc, 515)

        # Reject 1 spec token: should free exactly the last 1 index.
        backend.free_rejected_slots(req, num_rejected=1)
        n_after_free = sum(len(t) for t in req._allocated_indices)
        self.assertEqual(n_after_free, 514,
                         "free_rejected_slots should free exactly num_rejected "
                         "from the tail, leaving 512 + 1 bonus + 1 accepted")

        # The surviving indices must still include the original prefill prefix:
        # the first 512 indices are unchanged.
        flat = torch.cat([t for t in req._allocated_indices if len(t) > 0])
        self.assertEqual(len(flat), 514)
        # Allocator should have reclaimed exactly 1 index.
        self.assertEqual(backend._mock_allocator.available_size(),
                         backend._mock_allocator.total_tokens - 514)


@requires_sglang
class TestSwaNoDoubleDeduction(unittest.TestCase):
    """NEW-L regression: SWA must not be double-deducted.

    _reclaim_swa_out_of_window returns out-of-window SWA slots during decode.
    Later, when the request finishes and its tree nodes are evicted, the
    on_free callback must NOT deduct SWA again for those reclaimed tokens
    (real SWARadixCache prevents this via tombstones; we use plain RadixCache,
    so SWA is decoupled from on_free).  Before the fix, evict() fired
    _deduct_pool_used on already-reclaimed tokens, under-counting SWA by the
    reclaimed amount.
    """

    def test_swa_returns_to_zero_after_free_and_evict(self):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        backend = _make_backend()
        swa_per_tok = backend._swa_per_tok
        # Long enough to trigger SWA reclamation (threshold = charged-1-128-256
        # > 0  ⇒  charged > 385).
        req = backend.create_request("r1", list(range(900)), max_tokens=10)
        backend.register_request(req)
        backend.allocate_slots(req, num_new_tokens=900)
        backend.sync_state(req, [])
        # Reclaim must have advanced the cursor (charged 900 > 385).
        self.assertGreater(req.swa_evicted_charged, 0,
                           "SWA reclamation did not fire — test preconditions unmet")

        # Finish the request, then evict its (now unlocked) tree nodes.
        backend.free(req)
        for _ in range(40):
            backend._cache.evict(EvictParams(num_tokens=4096))

        # All three pools must be fully drained — no SWA double-deduction
        # residue (clamped negatives) and no c4/c128 leak.
        self.assertEqual(backend._pool_used, [0, 0, 0],
                         f"pools not drained after free+evict: {backend._pool_used}")

    def test_swa_not_undercounted_when_other_request_still_live(self):
        """The original NEW-L symptom: after evicting a finished request's
        tree nodes, SWA must still reflect the still-live request's footprint
        (not be driven below it by the reclaimed-portion double-deduction)."""
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        backend = _make_backend()
        swa_per_tok = backend._swa_per_tok
        r1 = backend.create_request("r1", list(range(900)), max_tokens=10)
        backend.register_request(r1)
        backend.allocate_slots(r1, num_new_tokens=900)
        backend.sync_state(r1, [])

        r2 = backend.create_request("r2", list(range(900, 1800)), max_tokens=10)
        backend.register_request(r2)
        backend.allocate_slots(r2, num_new_tokens=900)
        backend.sync_state(r2, [])

        r2_in_window = (r2.swa_charged_tokens - r2.swa_evicted_charged) * swa_per_tok

        # Finish + evict r1 while r2 is still live.
        backend.free(r1)
        for _ in range(40):
            backend._cache.evict(EvictParams(num_tokens=4096))

        # SWA must equal r2's live in-window footprint (r1 fully drained).
        self.assertEqual(backend._pool_used[0], r2_in_window,
                         f"SWA under-counted after r1 evict: got "
                         f"{backend._pool_used[0]}, expected {r2_in_window} "
                         f"(double-deduction of r1's reclaimed SWA)")


    def test_free_rejected_slots_raises_when_tail_too_short(self):
        """NEW-N: free_rejected_slots must not silently no-op when the tail
        is shorter than num_rejected.  Previously the ``len(flat) <
        num_rejected`` case fell through with no else branch, leaking the
        rejected slots (and, after the SWA-decoupling fix, leaking their
        c4/c128/SWA pool slots too).  It now raises RuntimeError so a future
        contract break surfaces loudly instead of corrupting pool accounting."""
        backend = _make_backend()
        req = backend.create_request("r1", list(range(512)), max_tokens=20)
        backend.register_request(req)
        backend.allocate_slots(req, num_new_tokens=512)
        backend.sync_state(req, [])
        # Only 515 allocated, but ask to free 999 — far more than the tail.
        with self.assertRaises(RuntimeError):
            backend.free_rejected_slots(req, num_rejected=999)


if __name__ == "__main__":
    unittest.main()