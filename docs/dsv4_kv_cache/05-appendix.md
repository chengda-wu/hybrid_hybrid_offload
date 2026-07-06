# Part 5 · 附录

> 对应原文档章节，完整目录见 [README.md](README.md)。

[← Part 4](04-runtime-and-apc.md) · [目录](README.md)

---

## 8. 源码索引

| 主题 | 文件:行 |
|------|--------|
| SWA cache spec | `vllm/v1/attention/backends/mla/sparse_swa.py:50,81` |
| Compressor state cache spec | `vllm/models/deepseek_v4/compressor.py:121,157` |
| Indexer cache spec | `vllm/models/deepseek_v4/attention.py:622,643` |
| Indexer 内部 compressor | `vllm/models/deepseek_v4/attention.py:737` |
| 主 MLA spec | `vllm/models/deepseek_v4/attention.py:601` |
| spec 收集 (get_kv_cache_spec) | `vllm/v1/worker/gpu_model_runner.py:7482` |
| per-token 字节 / page_size | `vllm/v1/kv_cache_interface.py:380,607` |
| 576 对齐 padding (第一层) | `vllm/v1/kv_cache_interface.py:327` |
| 分组 (group_and_unify) | `vllm/v1/core/kv_cache_utils.py:1499` |
| page_size 第二层 padding + SWA 分裂 | `vllm/v1/core/kv_cache_utils.py:1572,1623` |
| _approximate_gcd | `vllm/v1/core/kv_cache_utils.py:1537` |
| packed 布局 (_get_kv_cache_config_packed) | `vllm/v1/core/kv_cache_utils.py:1277` |
| bucketing (_bucket_layers_by_page_size) | `vllm/v1/core/kv_cache_utils.py:1230` |
| scheduler/hash block size | `vllm/v1/core/kv_cache_utils.py:607` |
| KVCacheManager | `vllm/v1/core/kv_cache_manager.py:110` |
| HybridKVCacheCoordinator | `vllm/v1/core/kv_cache_coordinator.py:514` |
| find_longest_cache_hit (APC) | `vllm/v1/core/kv_cache_coordinator.py:630` |
| SingleTypeKVCacheManager.allocate | `vllm/v1/core/single_type_kv_cache_manager.py:279` |
| 物理显存物化 (as_strided) | `vllm/v1/worker/gpu_model_runner.py:7046,7191` |
| spec decode block 调整 | `vllm/v1/core/sched/scheduler.py:1154,1488` |

---

## 9. 数值实测验证

本文档数值由两类实测得出（非手算）：
- **§1-§5 的布局数值**（groups/buckets/bytes_per_block/scheduler_bs 等）：由 `dsv4_layout.py` 调真实 vLLM 布局函数得出。
- **§7 的 block id 与 ref_cnt**：由真实 `KVCacheManager`（`generate_scheduler_kv_cache_config` + `KVCacheManager`）跑 A/B 请求得出。这需要 `UniformTypeKVCacheSpecs` 先经 `generate_scheduler_kv_cache_config` 拆成具体 spec（`kv_cache_utils.py:1766`），否则 manager 无法创建。

### 9.1 脚本位置与作用

`docs/dsv4_kv_cache/dsv4_layout.py`（相对项目根 `/home/witcher/hybrid_hybrid_offload`）—— 按 §1 的真实 spec 类构造 DSV4 全部 167 个 `KVCacheSpec`（与 `gpu_model_runner.py:7482` 的收集逻辑一致），然后调用 vLLM 的真实布局函数：
- `get_kv_cache_groups` → 5 groups
- `_bucket_layers_by_page_size` → 3 buckets
- `resolve_kv_cache_block_sizes` → scheduler/hash block size
- `_get_kv_cache_config_packed` → 63 tensors

**不需要 GPU**，只构造 spec dataclass 并跑布局规划器。

### 9.2 使用方式

```bash
cd /home/witcher/hybrid_hybrid_offload
.venv/bin/python docs/dsv4_kv_cache/dsv4_layout.py
```

> 依赖：`.venv` 里已安装 vllm（`VLLM_USE_PRECOMPILED=1 uv pip install -e 3rdparty/vllm`，见 `CLAUDE.md`）。
> 无需改任何源码，脚本直接 `import vllm.v1.*`。

### 9.3 预期输出

```
layers: SWA-only=2 C4=21 C128=20 total=43

#total specs collected: 167
spec type counts: Counter({'SlidingWindowMLASpec': 105, 'MLAAttentionSpec': 62})

--- per-spec page_size_bytes (after 576 alignment padding) ---
  page_size=   1728  count=20
  page_size=   8640  count=42
  page_size=  32832  count=41
  page_size=  37440  count=64

--- get_kv_cache_groups -> 5 KVCacheGroupSpec ---
  group 0: UniformType block_size=256 nlayers=62 page_sizes=[1728, 8640, 37440]
  group 1: UniformType block_size=64 nlayers=22 page_sizes=[37440]
  group 2: UniformType block_size=64 nlayers=21 page_sizes=[37440]
  group 3: UniformType block_size=4  nlayers=42 page_sizes=[8640, 37440]
  group 4: UniformType block_size=8  nlayers=20 page_sizes=[37440]

--- _bucket_layers_by_page_size -> 3 buckets ---
  ps=   1728  slot_count=20  layers_per_slot=[1×20]
  ps=   8640  slot_count=21  layers_per_slot=[2×21]
  ps=  37440  slot_count=22  layers_per_slot=[5×20, 4, 1]
  bytes_per_block = 1039680

--- per-group fill rate (block owned = all group layers write 1 slot) ---
  G0: 1,002,240 B / 1,039,680 =  96.4%  (20×1728 + 21×8640 + 21×37440)
  G1:   823,680 B / 1,039,680 =  79.2%  (22×37440)
  G2:   786,240 B / 1,039,680 =  75.6%  (21×37440)
  G3:   967,680 B / 1,039,680 =  93.1%  (21×8640 + 21×37440)
  G4:   748,800 B / 1,039,680 =  72.0%  (20×37440)

  scheduler_block_size=256  hash_block_size=4

--- _get_kv_cache_config_packed ---
  num_blocks=1000  num_tensors=63
  tensor[0]: offset=0 stride=1039680 shared_by=5 layers
  all 63 tensors share one backing (block_stride > 0)
```

### 9.4 数值汇总

- spec 总数 167（SlidingWindowMLASpec 105 + MLAAttentionSpec 62）
- 5 groups：G0(62层,bs256) G1(22层,bs64) G2(21层,bs64) G3(42层,bs4) G4(20层,bs8)
- 3 buckets：37440(22 slots) / 8640(21 slots) / 1728(20 slots)
- bytes_per_block = 1,039,680；63 tensors 共享 1 backing
- scheduler_block_size=256，hash_block_size=4
- 单 block 填充率（group 持有时）：G0 96.4% / G1 79.2% / G2 75.6% / G3 93.1% / G4 72.0%（见 §4.6）

> 若 vLLM submodule 升级后分组逻辑变化，重跑此脚本即可更新文档数值。

---

*文档生成于 2026-07-06，基于 vLLM submodule HEAD（ab132ee98）。*
*所有 group/bucket/slot/bytes 数值由真实 vLLM 分组函数实测验证。*


---

[← Part 4](04-runtime-and-apc.md) · [目录](README.md)
