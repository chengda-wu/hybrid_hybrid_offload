"""P1 regression test: SGLang allocate_slots must not double-allocate."""

import math
import unittest

from simulator.config.model_config import (
    KVBackendConfig,
    ModelArchitecture,
)
from simulator.kv_cache.sglang_backend import SGLangBackend


def _make_backend(num_blocks=128):
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


if __name__ == "__main__":
    unittest.main()
