"""SGLang-specific KV cache config builder.

No vllm imports — only loaded when backend='sglang'.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from simulator.config.model_config import KVBackendConfig


@dataclass
class SGLangConfig:
    """SGLang-specific config built from KVBackendConfig.

    Real SGLang DSV4 has exactly TWO independently-tracked allocatable pools —
    ``full`` and ``swa`` (``pool_stats_observer.py::get_max_pool_usage`` reads
    only ``full_token_usage`` and ``swa_token_usage``).  c4 and c128 are NOT
    independent pools: they are sub-allocated in lockstep from the unified
    ``full`` pool by ``SWATokenToKVPoolAllocator`` (``allocator/swa.py:20-78``,
    which has exactly two sub-allocators: ``full_attn_allocator`` +
    ``swa_attn_allocator``).  ``c4_logical_size = c128_size * 32``
    (``deepseek_v4_memory_pool.py:477``) — c4/c128 are proportioned to fill
    simultaneously with ``full`` and can never independently OOM.

    This config therefore carries only the two real base position caps
    (``full_token``, ``swa_token``) as produced by SGLang's own
    ``DSV4PoolConfigurator``.  The backend turns them into layer-slot caps
    (cap = base × per-token layer-slots) so SWA's existing charged-token reclaim
    math is undisturbed; the ratio (used·per_tok)/cap is unit-invariant, so
    ``usage = max(swa_ratio, full_ratio)`` mirrors ``get_max_pool_usage``.

    Compressor state pools are not part of the KV pool budget — they are ring
    buffers sized separately and tracked in ``total_bytes`` only.
    """

    page_size: int  # system page_size for RadixCache alignment (256)

    # Base per-position capacities from real SGLang's DSV4PoolConfigurator
    # (full_max_total_num_tokens / swa_max_total_num_tokens).  One full slot
    # serves one token position across c4+c128 (sub-allocated); one swa slot
    # serves one position in the SWA ring (reclaimed out of window).
    full_token: int
    swa_token: int

    @classmethod
    def from_backend_config(cls, bc: KVBackendConfig) -> "SGLangConfig":
        """Build SGLang config from the common backend config.

        Delegates pool sizing to SGLang's own ``DSV4PoolConfigurator`` via its
        token-constrained entry point ``calculate_pool_sizes_from_max_tokens``
        — the same call SGLang's runtime makes when a user token cap applies
        (``model_runner_kv_cache_mixin.py:1116``).  This replaces the previous
        hand-rolled ``swa=0.1·full / c4=full//4 / c128=full//128`` ratios with
        genuine reuse (project rule: always import from vllm/sglang, never
        reimplement KV cache sizing).
        """
        blocks = bc.num_kv_cache_blocks
        sbs = bc.scheduler_block_size
        ps = bc.block_size  # set to main_block_size (256 for DSV4) by engine.py;
                            # KVBackendConfig.block_size itself defaults to 16

        full_tokens = blocks * sbs

        arch = bc.model_arch

        # Spec-mode (T+D)/T pre-scaling.  Real SGLang applies this inflation
        # inside ``bytes_per_full_token`` (pool_configurator.py:538-545), which
        # the BYTES path uses — but ``calculate_pool_sizes_from_max_tokens``
        # (the TOKEN path) calls ``_compute_dsv4_sizes`` directly and does NOT
        # re-apply it.  We therefore pre-scale ``full_tokens`` ourselves so the
        # token-path caps reflect the draft-worker reservation, matching the
        # bytes-path intent.  Then page-align (pool_configurator.py:622).
        num_spec = getattr(bc, "num_spec_tokens", 0) or 0
        if num_spec > 0:
            full_tokens = (full_tokens * arch.num_layers // (arch.num_layers + 1))
            full_tokens = (full_tokens // ps) * ps  # page-align

        # Build a minimal mock ModelRunner carrying exactly the fields
        # ``DSV4PoolConfigurator.__init__`` reads (pool_configurator.py:499-571).
        # Use SimpleNamespace (not MagicMock): if SGLang later adds a field read,
        # this raises AttributeError loudly instead of silently returning a
        # MagicMock that corrupts sizing.
        qk_rope = arch.qk_rope_head_dim or 64
        qk_nope = arch.head_size - qk_rope  # 512-64=448 for DSV4 Flash
        mc = SimpleNamespace(
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            index_head_dim=arch.indexer_head_dim,  # HF config.index_head_dim (128 for DSV4); reproduces 132 B/token via 128 + 128//128*4
            compress_ratios=arch.compress_ratios,
            window_size=arch.sliding_window,  # 128 → swa_page_size
        )
        is_spec = num_spec > 0

        # Online c128 + experimental online-c128-MTP path.  Real SGLang gates
        # this on spec_algorithm.is_eagle() (pool_configurator.py:556-565
        # asserts; MTP spec is rejected).  The simulator models MTP, not EAGLE,
        # but the online-c128-MTP *byte cost* (extra draft-state banks in
        # DeepSeekV4CompressStatePool) depends only on the draft-token count,
        # not the spec algorithm.  When the user enables BOTH experimental env
        # flags with spec on, report is_eagle()=True and pass the draft count
        # so (a) the configurator's assert passes instead of crashing at
        # construction, and (b) total_bytes' MTP multiplier (which uses
        # num_spec_tokens) stays consistent with the configurator's sizing.
        # Online compress + spec WITHOUT the MTP flag still crashes — faithful
        # to real SGLang, which also rejects that combination.
        from sglang.srt.environ import envs
        online_mtp_path = (
            is_spec
            and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
            and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get()
        )
        sa = SimpleNamespace(
            swa_full_tokens_ratio=bc.swa_full_tokens_ratio,  # DSV4 hook override
            speculative_algorithm="MTP" if is_spec else None,
            max_speculative_num_draft_tokens=(
                num_spec if online_mtp_path else None
            ),
        )
        # NOTE: the configurator (DSV4PoolConfigurator) only tests
        # ``speculative_algorithm is None/is not None`` to set is_speculative —
        # it never compares the *string value*.  The ``== "EAGLE"`` assertion
        # lives in deepseek_v4_hook.py, which the simulator bypasses (it builds
        # DSV4PoolConfigurator directly).  So "MTP" here is a faithful label of
        # what the simulator models; it is NOT a bug to be "fixed" by changing
        # it to "EAGLE" (that would mislabel the MTP model as EAGLE).  The
        # is_eagle() gate on the experimental online path is faked via the
        # separate ``spec_algorithm`` namespace below, not this string.
        # spec_algorithm is only read when SGLANG_OPT_USE_ONLINE_COMPRESS env is
        # set (default off); is_eagle() is forced True only on the experimental
        # online-c128-MTP path (see above).
        spec_algorithm = SimpleNamespace(
            is_none=lambda: not is_spec,
            is_eagle=lambda: online_mtp_path,
        )
        mr = SimpleNamespace(
            model_config=mc,
            server_args=sa,
            start_layer=0,
            end_layer=len(arch.compress_ratios or []),
            pp_size=1,  # pp_group is guarded by `if pp_size > 1` — never read
            enable_hisparse=False,  # → c4_shrink_factor = 1 (no HiSparse modeled)
            spec_algorithm=spec_algorithm,
        )

        from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

        cfg = DSV4PoolConfigurator(mr)
        sizes = cfg.calculate_pool_sizes_from_max_tokens(full_tokens, ps)

        return cls(
            page_size=ps,
            full_token=sizes.full_max_total_num_tokens,
            swa_token=sizes.swa_max_total_num_tokens,
        )
