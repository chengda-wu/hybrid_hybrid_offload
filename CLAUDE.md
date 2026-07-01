# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation & running

The environment may or may not be set up.  Check first:

```bash
# Check what's available
python3 -c "import vllm" 2>/dev/null && echo "vllm: OK" || echo "vllm: NOT INSTALLED"
python3 -c "import sglang" 2>/dev/null && echo "sglang: OK" || echo "sglang: NOT INSTALLED"
ls 3rdparty/vllm/vllm/ 2>/dev/null && echo "submodule vllm: OK" || echo "submodule vllm: EMPTY — run: git submodule update --init"
```

If not set up:

```bash
git submodule update --init --recursive
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e 3rdparty/vllm
uv pip install -e 3rdparty/sglang/python
```

Run the simulation:

```bash
.venv/bin/python -m simulator.run --backend vllm --num-requests 20
.venv/bin/python -m simulator.run --backend sglang --num-requests 20
```

## Architecture

```
SimulatorConfig → SimulationEngine
                    ├── KVBackend (vLLM / SGLang) ── real vllm KVCacheManager or sglang RadixCache
                    ├── SimulatorScheduler ── main step loop
                    │     ├── _handle_prefill  ── get_computed_blocks → allocate_slots
                    │     └── _handle_decode   ── draft tokens → allocate → accept → reject
                    ├── AcceptanceModel ── dual-condition spec decode acceptance
                    ├── SpeculativeDecodeEngine ── draft token generation
                    ├── GPUPerfModel ── latency = a·m + b·n + c·m·n + d
                    └── MetricsRecorder → StatisticsComputer → SimulationReport (JSON)
```

**Every step**: admit waiting requests → prefix cache lookup → allocate → simulate forward → acceptance → adjust num_computed_tokens → free finished. Spec decode mirrors vLLM's `_update_after_schedule(advance all)` / `update_from_output(subtract rejected)` semantics.

## Key files

| File | Role |
|------|------|
| `simulator/config/model_config.py` | ModelArchitecture (HF config.json → KV specs), `VLLMConfig.from_backend_config` (builds KVCacheGroupSpecs, delegates to vLLM's `_get_kv_cache_config_packed` for tensor sizing), `SGLangConfig` |
| `simulator/config/simulator_config.py` | All simulation parameter dataclasses |
| `simulator/core/scheduler.py` | `SimulatorScheduler.step()` — main loop. `_handle_prefill` / `_handle_decode` |
| `simulator/core/engine.py` | `SimulationEngine.run()` — wires everything, prints KV cache size |
| `simulator/kv_cache/vllm_backend.py` | Wraps real vLLM `KVCacheManager`. `sync_to_vllm()` excludes spec tokens. `total_bytes` via `_bucket_layers_by_page_size` |
| `simulator/kv_cache/sglang_backend.py` | Wraps real SGLang `RadixCache.create_simulated()` + `MockTokenToKVPoolAllocator`. `free()` is a no-op (matches real SGLang: only `evict()` frees). `total_bytes` per-group with correct per-type byte costs |
| `simulator/speculative/acceptance.py` | `AcceptanceModel.evaluate()` — draft must match ground truth AND pass per-position rate. Raises `ValueError` if `acceptance_rates` < K |
| `simulator/speculative/engine.py` | `SpeculativeDecodeEngine.generate_draft_tokens()` — bonus + K drafts |
| `simulator/metrics/gpu_perf_model.py` | `GPUPerfModel` — 4×4 Gaussian elimination fit. Predict floors negative to 0 with one-time warning |
| `simulator/metrics/stats.py` | `SimulationReport` — TTFT p50/p99, TPOT p50/p99 (excl. first token), queue length, cache hit rate |

## DeepSeek V4 Flash KV cache layout

Real HF config: 43 layers, `compress_ratios = [0,0,4,128,4,128,...,4]` → SWA=2, C4=21, C128=20.

vLLM packs 6 groups into a **shared** block pool (one physical allocation via offset+block_stride):
1. SWA (bs=64, all 43L)
2. C4 Compressor (bs=4, float32, state_dim=2048, 21L)
3. C128 Compressor (bs=8, float32, state_dim=1024, 20L)
4. C4 Main MLA (bs=256, cr=4, 21L)
5. C128 Main MLA (bs=256, cr=128, 20L)
6. C4 Indexer (bs=256, head_dim=132, 21L)

SGLang SWA ring uses `swa_full_tokens_ratio=0.1` (deepseek_v4_hook.py:57) — only 10% density.

## Critical rules

### Always import from vllm/sglang — never reimplement

The simulation must reflect real engine behavior exactly. Any logic related to KV cache management — block splitting, tensor sizing, page alignment, prefix matching, eviction — must be done by calling the real vllm or sglang function, not by hand-coding our own version.

Examples of what we **correctly** delegate:
- vLLM block/tensor layout → `_get_kv_cache_config_packed`, `_bucket_layers_by_page_size`
- SGLang prefix matching → `RadixCache.create_simulated()`, `match_prefix()`, `insert()`, `evict()`
- vLLM `num_computed_tokens` advancement → mirrors `_update_after_schedule` / `update_from_output` semantics

Examples of what we should **never** do:
- Hand-code block splitting across groups (use vLLM's `_get_kv_cache_config_packed`)
- Hand-code page size computation (use vLLM's `page_size_bytes` property on the spec)
- Hand-code radix tree logic (use SGLang's `RadixCache` directly)

### KVBackend is the abstraction layer

```
SimulatorScheduler ── KVBackend (ABC)
                        ├── vLLMBackend  ── vllm KVCacheManager  ── 0 sglang imports
                        └── SGLangBackend ── sglang RadixCache   ── 0 vllm imports
                             │                    │
                             └──── KVGroupInfo ───┘  (framework-agnostic, in model_config.py)
```

The scheduler only speaks `KVBackend`. It never imports vllm or sglang directly. All framework differences (block-level vs token-level, packed vs flat layout, lock_ref semantics) are hidden behind the backend interface. When adding a feature that differs between vllm and sglang, push the difference DOWN into the backend implementations, not UP into the scheduler.

### vLLM and SGLang must be mutually independent

- **vLLM backend** imports only from `vllm.*` — never from `sglang.*` or `SGLangConfig`
- **SGLang backend** imports only from `sglang.*` — never from `vllm.*` or `VLLMConfig`
- **Shared model description** lives in `KVGroupInfo` (framework-agnostic dataclass in `model_config.py`): name, block_size, page_bytes (unpadded), layer_count.  Both backends consume this; neither backend's types leak into it.
- If a conversion is needed (e.g. KVGroupInfo → KVCacheGroupSpec), it lives in the backend that needs it (`VLLMConfig._build_vllm_specs`).
- **Don't hardcode model parameters.** DeepSeek V4 defaults come from the actual HF `config.json` (verified against `deepseek-ai/DeepSeek-V4-Flash`). Any new model should be configurable via `--model-config`.
- **Spec token lifecycle mirrors vLLM scheduler.** `_update_after_schedule` adds all (1+K), `update_from_output` subtracts rejected. Bonus token is always from ground truth, never counted in acceptance.
- **No chunked prefill.** By design — documented limitation.
- **SGLang free() is intentionally a no-op.** Real SGLang frees via `evict()` only; `dec_lock_ref` just marks evictable. Our simulation matches this.
- **GPU perf model** fits `latency = a·m + b·n + c·m·n + d` with proper 4×4 Gaussian elimination. Adding data points near origin (e.g. `[0,1,0.5]`) prevents negative predictions.
- **`total_computed_tokens_ever` was removed** — dead field, only written never read.

## Real source references

- `3rdparty/vllm/vllm/v1/core/kv_cache_utils.py` — `_get_kv_cache_config_packed`, `_bucket_layers_by_page_size`, `get_kv_cache_config_from_groups`, `_pool_bytes_per_block`
- `3rdparty/vllm/vllm/v1/core/kv_cache_manager.py` — `KVCacheManager`
- `3rdparty/vllm/vllm/v1/core/sched/scheduler.py` — spec decode flow: `_update_after_schedule` (L1154), `update_from_output` (L1488)
- `3rdparty/vllm/vllm/models/deepseek_v4/attention.py` — `DeepseekV4SWACache`, `get_kv_cache_spec()` (L601), `DeepseekV4IndexerCache` (L622)
- `3rdparty/vllm/vllm/models/deepseek_v4/compressor.py` — `CompressorStateCache.get_kv_cache_spec()` (L157)
- `3rdparty/sglang/python/sglang/srt/mem_cache/radix_cache.py` — `RadixCache.create_simulated()`, `evict()`, `lock_ref` mechanism
- `3rdparty/sglang/python/sglang/srt/arg_groups/deepseek_v4_hook.py` — `swa_full_tokens_ratio=0.1` override
