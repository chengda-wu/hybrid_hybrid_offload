# Hybrid Attention KV Offload

探索混合注意力（Hybrid Attention）场景下 KV Cache 管理方式及多级卸载策略。包含一个**KV Cache 仿真系统**，用于模拟调度流程、prefix cache 命中、投机解码等行为。

## KV Cache 仿真系统

纯 CPU 仿真系统，直接调用 vLLM/SGLang 的真实 KV cache 代码（`KVCacheManager` / `RadixCache`），保证内存分配、前缀匹配、驱逐策略 100% 准确。

### 快速开始

```bash
# 1. 初始化 submodule + 创建 venv
git submodule update --init --recursive
uv venv --python 3.12
source .venv/bin/activate

# 2. 安装 vllm / sglang（editable install from submodule）
VLLM_USE_PRECOMPILED=1 uv pip install -e 3rdparty/vllm
uv pip install -e 3rdparty/sglang/python

# 3. 运行仿真
python -m simulator.run --backend vllm --num-requests 20
```

### 命令行参数

```
python -m simulator.run [OPTIONS]
```

#### 模型与后端

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backend` | `vllm` | KV cache 后端，`vllm` 或 `sglang` |
| `--model-config` | — | HuggingFace `config.json` 路径。省略则使用 **DeepSeek V4 Flash** 硬编码默认值（43 层 MLA，512 维 head，SWA=2/C4=21/C128=20） |
| `--max-model-len` | `8192` | 模型最大上下文长度（token 数） |
| `--kv-block-size` | `16` | KV cache 块大小（每个 block 包含的 token 数）。vLLM 按此粒度分配/匹配 |
| `--num-kv-blocks` | `4096` | KV cache 块池总数。决定总可用缓存空间：`blocks × block_size × per_token_bytes` |

#### 数据集

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-requests` | `100` | 请求总数 |
| `--prompt-length` | `512` | 每个请求的 prompt token 数（仅 `fixed` 模式） |
| `--output-length` | `256` | 每个请求期望生成的 output token 数 |
| `--shared-prefix-ratio` | `0.5` | 请求间共享前缀比例，范围 `[0, 1]`。第 1 个请求的 prompt 作为基准；后续请求的前 `ratio × prompt_length` 个 token 与第 1 个请求相同。设为 1.0 表示完全共享，设为 0.0 表示无共享 |

#### 投机解码

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-spec-tokens` | `2` | 每步 draft token 数量 **K**。设为 `0` 关闭投机解码 |
| `--accept-mode` | `per_position` | `fixed`：所有位置使用同一接受率；`per_position`：每个位置使用 `--acceptance-rates` 中对应索引的值 |
| `--acceptance-rate` | `0.85` | 固定接受率，仅 `--accept-mode fixed` 时生效。表示 draft 匹配 ground truth 后，额外通过采样的概率 |
| `--acceptance-rates` | — | 逐位置接受率，空格分隔 **K 个浮点数**。如 `0.9 0.7 0.5 0.3` 表示 draft token 1 接受率 0.9、draft token 2 接受率 0.7……以此类推。仅 `--accept-mode per_position` 时生效。**长度必须 ≥ K** |
| `--draft-accuracy` | `0.7` | draft token 本身匹配 ground truth 的概率，范围 `[0, 1]`。模拟投机模型的"推测质量"：1.0 表示 draft 永远正确，0.0 表示永远错误 |

#### GPU 性能模型

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu-data-points` | — | GPU 延迟拟合数据点，格式为 JSON 数组。每个元素是 `[loaded_tokens, computed_tokens, latency_ms]` 三元组。详见下方说明 |

**GPU 数据点说明：**

仿真器用公式 `latency = a×loaded + b×computed + c×loaded×computed + d` 拟合 GPU 延迟，系数通过最小二乘法从数据点自动计算。

> **绝对延迟不可信**：未提供 `--gpu-data-points` 时，使用内置的 H100-like 估值默认值（`GPUPerfModel.DEFAULT_DATA`），**非实测**。因此报告中的绝对 ms 值（TTFT/TPOT/step_latency）仅供相对趋势参考，不能当作真实 DSV4 性能基准。需要可信绝对值时，请用真实 benchmark 数据喂 `--gpu-data-points` 重新拟合。

每条数据点包含三个值：

| 位置 | 名称 | 含义 |
|------|------|------|
| `[0]` | `loaded_tokens` | 本轮 forward 中**从 KV cache 直接读取**的 token 数（已缓存的 prefix） |
| `[1]` | `computed_tokens` | 本轮 forward 中**实际需要计算**的 token 数（新 prefill token 或 1+K decode token） |
| `[2]` | `latency_ms` | 该次 forward 的端到端延迟（毫秒） |

典型数据点示例：

```
[0,     1,    0.5]    ← 单 token decode，无缓存 prefix
[0,     512,  8.0]    ← prefill 512 token，无缓存
[0,     4096,  60.0]  ← prefill 4096 token
[1000,  1,    1.2]    ← 单 token decode，有 1K 缓存 prefix
[4000,  1,    3.5]    ← 单 token decode，有 4K 缓存 prefix
[1000,  4,    3.5]    ← 投机 decode (K=3, 即 1+3=4) 有 1K 缓存
```

省略此参数时，使用内置的 H100 级别默认数据点。

#### 其他

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | `42` | 随机种子，保证合成数据和接受率采样的可复现性 |
| `--output` / `-o` | — | 输出 JSON 报告路径。省略则打印到 stdout |
| `--verbose` / `-v` | — | 打印每步调度日志（请求状态变化、prefill/decode 详情） |
| `--config` | — | JSON 配置文件路径，可替代上述所有 CLI 参数。示例 6 展示了完整的配置文件格式 |

### 输出指标

```json
{
  "avg_loaded_tokens_per_step": 4730.7,   // 平均每步从 cache 读取的 token
  "avg_computed_tokens_per_step": 140.0,   // 平均每步实际计算的 token
  "avg_accepted_tokens_per_step": 20.3,    // 平均每步投机接受的 token
  "ttft_p50_ms": 48.9,                     // 首 token 延迟 p50
  "ttft_p99_ms": 49.8,                     // 首 token 延迟 p99
  "tpot_p50_ms": 4.5,                      // 每输出 token 时间 p50
  "tpot_p99_ms": 4.8,                      // 每输出 token 时间 p99
  "avg_step_latency_ms": 8.8,              // 平均每步延迟
  "avg_waiting_queue_length": 0.5,         // 平均等待队列长度
  "max_waiting_queue_length": 19,          // 最大等待队列长度
  "cache_hit_rate": 0.475,                 // prefix cache 命中率
  "avg_cache_usage": 0.092,                // 平均 cache 利用率
  "avg_acceptance_rate": 0.433,            // 投机解码平均接受率
  "total_requests": 20,                    // 请求总数
  "total_tokens_generated": 1280,          // 生成的总 token 数
  "total_sim_time_ms": 309.0,              // 仿真总耗时
  "tokens_per_second": 4142.3,             // 吞吐量
  "backend": "vllm"                        // 后端标识
}
```

### DeepSeek V4 Flash KV Cache 布局（4096 blocks）

仿真器分别为 vLLM 和 SGLang 建模了各自的真实 KV 管理方式。

#### vLLM — 9.60 GB

vLLM 使用 **packed layout**（`_get_kv_cache_config_packed`），6 组共享单一 BlockPool 并通过 `offset`+`block_stride` 共享物理分配。下表为各组独立 page 容量（十进制 GB），不可加和（共享池）：

| Group | 类型 | 层数 | block_size | page 字节 | 总容量 |
|-------|------|------|------------|-----------|--------|
| SWA | SlidingWindowMLASpec | 43L | 64 | 37,440 B | 6.59 GB |
| C4 Compressor | SlidingWindowMLASpec | 21L | 4 | 32,832 B | 2.82 GB |
| C128 Compressor | SlidingWindowMLASpec | 20L | 8 | 32,832 B | 2.69 GB |
| C4 Main MLA | MLAAttentionSpec | 21L | 256 (cr=4) | 37,440 B | 3.22 GB |
| C128 Main MLA | MLAAttentionSpec | 20L | 256 (cr=128) | 1,728 B | 0.14 GB |
| C4 Indexer | MLAAttentionSpec | 21L | 256 (alignment=576) | 8,640 B | 0.74 GB |

- 所有 43 层都有 SWA cache（`DeepseekV4SWACache`，attention.py:290 无条件创建）
- Compressor state 用 float32 + state_dim（C4=2048, C128=1024）
- Page 字节已含 vLLM 的 576 字节对齐 padding
- Packed bucket 合并：SWA+C4 main 同为 ps=37440，按 max(43,21)=43 slots；C4+C128 comp 同为 ps=32832，按 max(21,20)=21 slots。总量 = 4096 × (1728×20 + 8640×21 + 32832×21 + 37440×43) / 1024³ ≈ 9.60 GiB。表内"总容量"为各组独立 page × 4096（非 packed 实际），不可加和
- **spec-on（K>0）≈ 9.74 GiB**：MTP draft layer（compress_ratio=1）只有 SWA cache（`DeepseekV4SWACache`，spec 与 target SWA 相同 → 同 bucket），加入 SWA bucket 使其从 43→44 层。MTP draft layer 共享 target block pool（`llm_base_proposer.py` 断言所有 draft layer 属同一 kv_cache_group），不新增 group/pool。其它 5 组（compressor / main MLA / indexer）不受影响（MTP layer 无 compress_ratio>1）。增量 = 4096 × 37440 / 1024³ ≈ 0.14 GiB
- 来源：`vllm/models/deepseek_v4/attention.py`、`compressor.py`、`v1/core/kv_cache_utils.py`、`v1/spec_decode/llm_base_proposer.py`、`v1/attention/backends/mla/sparse_swa.py`

#### SGLang — 24.35 GB（spec 默认 K=2, ring 16/256, 43/44 scaled）／ 15.58 GB（nonspec, ring 8/128）

SGLang 使用 **DSV4PoolConfigurator**（`pool_configurator.py:449`），ring buffer + 共享 token budget。
KV 池页面有 576 字节对齐 padding（`DeepSeekV4SingleKVPool.create_buffer`），state 池无 padding。

| 池 | 公式 | 层数 | 总容量 |
|----|------|------|--------|
| SWA ring | `swa_slots × pad(128×584) × 43L` | 43L | 2.46 GB |
| C4 Compressor ring | `swa_slots × ring(16) × 8192B × 21L`（spec）／ ring(8)×21L（nonspec） | 21L | 2.10 GB / 1.05 GB |
| C128 Compressor ring | `swa_slots × ring(256) × 4096B × 20L`（spec）／ ring(128)×20L（nonspec） | 20L | 15.98 GB / 7.99 GB |
| C4 Main KV | `blocks × pad(64×584) × 21L` | 21L | 3.00 GB |
| C128 Main KV | `blocks × pad(2×584) × 20L` | 20L | 0.13 GB |
| C4 Indexer | KV: `pages × (64×132) × 21L`（无 pad）+ State: `swa_slots × ring(16) × 2048B × 21L`（spec）／ ring(8)（nonspec） | 21L | 1.18 GB / 0.94 GB |

- `full_token = blocks × scheduler_block_size = 4096 × 256 = 1,048,576`
- `swa_tokens = 104,704`（full_token × 0.1, page-aligned）；`swa_page_size = 128`（cfg.window_size）；`swa_slots = 818`
- `pad(raw) = ceil(raw / 576) * 576` — 576 字节页对齐（`deepseek_v4_memory_pool.py:106-107`）
- SWA ring 密度 10%（`deepseek_v4_hook.py:57`：`swa_full_tokens_ratio=0.1`）
- Compressor state: float32, C4 `last_dim=2048`（8192 B/token）、C128 `last_dim=1024`（4096 B/token）
- Indexer state: float32, `last_dim=512`（2048 B/token）
- 来源：`sglang/srt/model_executor/pool_configurator.py`、`mem_cache/deepseek_v4_memory_pool.py`

SGLang（spec 默认 K=2）24.35 GB vs vLLM 9.74 GB（spec-on），差值 +14.61 GB（总盘相减）。主要来源：C128 compressor state spec ring(256) 远大于 vLLM packed。nonspec 下 vLLM 9.60 GB vs SGLang 15.58 GB（差值 +5.98 GB）。vLLM packed 布局下各组容量不可简单加减（共享 bucket），差值以总盘为准。

两端差异来自框架本身的架构选择，非模拟器偏差。三个根因：

1. **物理内存共享方式不同**：vLLM 把 6 组打包进单一物理 buffer（`_get_kv_cache_config_packed`），同 page_size 的组复用 slot——`c4_mla`(21L) 藏进 `swa`(43L) 的 slot、`c128_compressor`(20L) 藏进 `c4_compressor`(21L) 的 slot，物理上 0 额外字节。SGLang 每个 group 独立分配，无跨组共享。
2. **compressor state sizing 模型不同**：vLLM 把 compressor 建模为共享 BlockPool 里的 sliding-window KV（且因根因 1 被藏进 c4_compressor 的 slot，几乎不占额外内存）；SGLang 把 compressor 建成独立 ring buffer（`swa_slots × ring × last_dim × dt × layers`），c128 单独就 8.0 GiB（nonspec）。这是 nonspec 差距（+5.98 GB）的主因。
3. **spec 模式 draft 内存机制不同（真实架构差异）**：两端都建模了 spec-on 的 draft 内存，但机制不同——
   - **SGLang**：draft worker 是独立进程，按 `(T+D)/T = 44/43` 膨胀整个 `bytes_per_full_token`（`pool_configurator.py:538-545`），等效 `full_tokens × 43/44`；同时 compressor ring 翻倍（c4 8→16、c128 128→256）。total 15.58→24.35 GB（+8.77 GB）。
   - **vLLM**：MTP draft layer 共享 target block pool（`llm_base_proposer.py` 断言所有 draft layer 属同一 kv_cache_group），只多 1 层 SWA cache（compress_ratio=1，无 MLA/compressor/indexer）加入 SWA bucket（43→44 层）。total 9.60→9.74 GB（+0.14 GB）。
   - spec-on 差距从 nonspec 的 +5.98 GB 扩大到 +14.61 GB，几乎全部来自 SGLang 的 ring 翻倍 + (T+D)/T 膨胀，而 vLLM 共享 pool 下 draft 几乎免费——这是两个框架真实的 draft 内存架构差异，非建模偏差。

> **可比性说明**：`num_kv_cache_blocks=4096` 在两端含义不同（vLLM 是共享 pool 的块数，每块含所有组的一个切片；SGLang 是 `full_tokens` 的来源，再按比例切给各池）。两端数字各自与其真实框架的物理分配逐字节吻合，spec-on/off 两端均已建模 draft 内存。spec-on 的 +14.61 GB 差距是真实的框架架构差异（SGLang 独立 draft 预算 vs vLLM draft 共享 pool），不宜直接解读为"vLLM 比 SGLang 省 2.5×"——它反映的是 draft 内存策略不同，而非 target KV 效率不同。

### 使用示例

**示例 1：基础对比 — vLLM vs SGLang，相同配置**

```bash
# vLLM: block 级缓存匹配，LRU 驱逐
python -m simulator.run --backend vllm \
  --num-requests 50 --prompt-length 512 --output-length 256 \
  --shared-prefix-ratio 0.5 --seed 42 -o report_vllm.json

# SGLang: Radix Tree 匹配（page_size=256 页对齐），可配置驱逐策略
python -m simulator.run --backend sglang \
  --num-requests 50 --prompt-length 512 --output-length 256 \
  --shared-prefix-ratio 0.5 --seed 42 -o report_sglang.json

# 对比两个 JSON 文件中的 cache_hit_rate、avg_cache_usage 等字段
```

**示例 2：Prefix cache 命中率实验**

```bash
# 从无共享到完全共享，观察 cache_hit_rate 单调变化
for ratio in 0.0 0.25 0.5 0.75 1.0; do
  echo "=== shared_prefix_ratio = $ratio ==="
  python -m simulator.run --backend vllm --num-requests 100 \
    --prompt-length 512 --output-length 64 \
    --shared-prefix-ratio $ratio --num-spec-tokens 0 \
    --seed 42 -o "hit_rate_${ratio}.json"
  # 从输出 JSON 提取 cache_hit_rate
  python3 -c "import json; print(json.load(open('hit_rate_${ratio}.json'))['cache_hit_rate'])"
done
```

**示例 3：投机解码效率**

```bash
# 关闭投机解码（K=0）：每个 decode step 只计算 1 个 token
python -m simulator.run --backend vllm --num-spec-tokens 0 \
  --num-requests 50 --prompt-length 256 --output-length 128 \
  --seed 42 -o no_spec.json

# K=3，高接受率：模拟强投机模型
#   draft 位置 0 接受率 0.9, 位置 1 接受率 0.8, 位置 2 接受率 0.7
python -m simulator.run --backend vllm --num-spec-tokens 3 \
  --accept-mode per_position --acceptance-rates 0.9 0.8 0.7 \
  --draft-accuracy 0.9 \
  --num-requests 50 --prompt-length 256 --output-length 128 \
  --seed 42 -o spec_high.json

# K=3，低接受率：模拟弱投机模型
python -m simulator.run --backend vllm --num-spec-tokens 3 \
  --accept-mode per_position --acceptance-rates 0.5 0.3 0.1 \
  --draft-accuracy 0.5 \
  --num-requests 50 --prompt-length 256 --output-length 128 \
  --seed 42 -o spec_low.json

# 对比 avg_accepted_tokens_per_step：高接受率应该显著高于低接受率
```

**示例 4：自定义 GPU 性能模型**

```bash
# 用实测数据点拟合延迟公式 latency = a×loaded + b×computed + c×loaded×computed + d
# 数据点格式: [loaded_tokens, computed_tokens, latency_ms]
python -m simulator.run --backend vllm --num-requests 50 \
  --gpu-data-points '[
    [0,     1,    0.5],
    [1000,  1,    1.2],
    [4000,  1,    3.5],
    [8000,  1,    5.8],
    [0,     512,  8.0],
    [0,     2048, 30.0],
    [0,     4096, 62.0],
    [1000,  4,    4.2]
  ]' \
  --seed 42 -o custom_gpu.json

# 也可以只设置系数（覆盖拟合）通过 JSON 配置文件
```

**示例 5：使用真实 HuggingFace 模型配置**

```bash
# 从本地 config.json 读取层数、head 数、MLA 参数等
python -m simulator.run --backend vllm \
  --model-config /path/to/Meta-Llama-3-8B/config.json \
  --num-requests 100 --seed 42 -o llama.json

# 省略 --model-config 时自动使用 DeepSeek V4 Flash 默认配置
python -m simulator.run --backend vllm \
  --num-requests 100 --seed 42 -o deepseek_v4_flash.json
```

**示例 6：完整 JSON 配置文件**

所有 CLI 参数都可以通过 `--config` JSON 文件传入，适合批量实验和版本控制：

```bash
# 创建配置文件
cat > batch_experiment.json << 'EOF'
{
  "backend": "vllm",
  "max_model_len": 16384,
  "num_kv_cache_blocks": 8192,
  "kv_cache_block_size": 16,
  "random_seed": 42,
  "speculative": {
    "num_spec_tokens": 4,
    "accept_mode": "per_position",
    "acceptance_rates": [0.9, 0.8, 0.6, 0.4],
    "draft_accuracy": 0.85
  },
  "dataset": {
    "source": "synthetic",
    "synthetic": {
      "num_requests": 200,
      "prompt_length_dist": "fixed",
      "prompt_length_fixed": 1024,
      "output_length_dist": "fixed",
      "output_length_fixed": 512,
      "shared_prefix_ratio": 0.7
    }
  },
  "gpu_perf": {
    "data_points": [
      [0, 1, 0.5],
      [4000, 1, 3.5],
      [0, 2048, 30.0]
    ]
  }
}
EOF

python -m simulator.run --config batch_experiment.json -o result.json
```

### 已知简化

- **无测试覆盖**：当前无单元/集成测试（21 个 unittest 覆盖 allocate/acceptance/perf model，但缺 scheduler 和 E2E 集成测试），所有改动依赖人工 review。
- **无 chunked prefill**：每个请求的 prompt 在一步内完成 prefill，不分块。真实引擎会将长 prompt 分成多个 chunk 与 decode 交替执行。
- **无抢占**：分配失败时请求留在 PRE_FILL 状态重试，不会被换出。
- **FP4 indexer**：通过 `--fp4-indexer` CLI 或 config.json `use_fp4_indexer` 开关控制（SGLang fp4→68 B/token, vLLM 永 132）
- **vLLM packed layout**：DSV4 的 tensor 布局和 block 计数由 vLLM 的 `_get_kv_cache_config_packed` 计算，正确反映共享 block pool。
- **SGLang SWA ring**：`deepseek_v4_hook.py` 设置 `swa_full_tokens_ratio=0.1`，仿真按此比例计算 SWA 容量（ring buffer 只占满密度的 10%）。
- **SGLang 三池建模**：容量上限已按 SWA/C4/C128 三池独立校验（_pool_caps/_pool_used），C128 为最小池先满触发驱逐，无跨池借用。底层索引空间共享单一平坦 allocator（RadixCache 要求）。prefix 匹配不受影响，但内存压力模型是近似的——真实 SGLang 各池独立触发 OOM（如 C128 仅 8192 token 即满），sim 单池把三池容量混在一起，C128 用尽时可从 SWA/C4 配额"借"，cache_usage 和驱逐时机会偏离真实。常规负载（<10% 利用率）下影响可忽略，高压场景需注意。

### 调度逻辑说明

仿真调度器每步执行以下流程（模拟 vLLM `_update_after_schedule` / `update_from_output` 语义）：

```
Step N:
  1. 从等待队列注入到达时间的请求
  2. 对每个活跃请求:
     Prefill:  get_computed_blocks → allocate_slots(完整 prompt)
     Decode:   生成 draft tokens [bonus, draft_0, ..., draft_{K-1}]
               allocate_slots(1+K)
               num_computed_tokens += (1+K)           ← _update_after_schedule
               接受判定: draft 匹配 ground truth + 逐位置采样
               num_computed_tokens -= rejected         ← update_from_output
               净推进: bonus(1) + accepted
  3. GPU 延迟模拟: predict(total_loaded, total_computed)
  4. 记录 per-step 指标
  5. 释放完成的请求
```

投机判定采用双条件：draft token 必须**同时**满足：
1. 与 ground truth output token 匹配
2. 通过该位置的 accept_rate 随机采样

首个失败的 draft 立即断链，后续全部 reject。

### 目录结构

```
hybrid_hybrid_offload/
├── 3rdparty/                  # 第三方推理引擎 (submodule)
│   ├── vllm/
│   └── sglang/
├── simulator/                 # KV Cache 仿真系统
│   ├── config/                # 配置（模型、仿真参数）
│   ├── core/                  # 调度器、请求状态机、引擎
│   ├── kv_cache/              # KV 后端适配器 (vllm/sglang)
│   ├── speculative/           # 投机解码引擎
│   ├── metrics/               # 指标收集与统计
│   ├── data/                  # 数据加载（合成/真实）
│   └── run.py                 # CLI 入口
├── docs/                      # 设计文档
└── README.md
```

### GitHub Submodule 初始化

```bash
git clone --recurse-submodules <this-repo-url>
# 或
git submodule update --init --recursive
```

## License

TBD
