# Hybrid Attention KV Offload

探索混合注意力（Hybrid Attention）场景下 KV Cache 管理方式及多级卸载策略。包含一个**KV Cache 仿真系统**，用于模拟调度流程、prefix cache 命中、投机解码等行为。

## KV Cache 仿真系统

纯 CPU 仿真系统，直接调用 vLLM/SGLang 的真实 KV cache 代码（`KVCacheManager` / `RadixCache`），保证内存分配、前缀匹配、驱逐策略 100% 准确。

### 快速开始

```bash
# 安装依赖（需要 Python 3.12+）
uv sync
source .venv/bin/activate

# 运行仿真
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

### 使用示例

**示例 1：基础对比 — vLLM vs SGLang，相同配置**

```bash
# vLLM: block 级缓存匹配，LRU 驱逐
python -m simulator.run --backend vllm \
  --num-requests 50 --prompt-length 512 --output-length 256 \
  --shared-prefix-ratio 0.5 --seed 42 -o report_vllm.json

# SGLang: token 级 Radix Tree 匹配，可配置驱逐策略
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
