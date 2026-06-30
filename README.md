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

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backend` | `vllm` | 后端：`vllm` 或 `sglang` |
| `--num-requests` | `100` | 请求总数 |
| `--prompt-length` | `512` | 每个请求的 prompt token 数 |
| `--output-length` | `256` | 每个请求的输出 token 数 |
| `--shared-prefix-ratio` | `0.5` | 请求间共享前缀比例 [0, 1] |
| `--num-spec-tokens` | `2` | 投机解码每步 draft token 数 K |
| `--accept-mode` | `per_position` | 接受率模式：`fixed` 或 `per_position` |
| `--acceptance-rate` | `0.85` | 固定接受率（accept_mode=fixed 时） |
| `--acceptance-rates` | — | 逐位置接受率，如 `0.9 0.7 0.5 0.3` |
| `--draft-accuracy` | `0.7` | Draft token 匹配 ground truth 的概率 |
| `--kv-block-size` | `16` | KV cache 块大小 |
| `--max-model-len` | `8192` | 最大模型长度 |
| `--num-kv-blocks` | `4096` | KV cache 块池大小 |
| `--model-config` | — | HuggingFace config.json 路径 |
| `--gpu-data-points` | — | GPU 性能数据点 JSON 字符串 |
| `--seed` | `42` | 随机种子 |
| `--output` / `-o` | — | 输出 JSON 文件路径 |
| `--verbose` / `-v` | — | 打印每步详细日志 |
| `--config` | — | 上述全部参数的 JSON 配置文件 |

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

**示例 1：基础对比 — vllm vs sglang，相同配置**

```bash
# vLLM
python -m simulator.run --backend vllm --num-requests 50 \
  --prompt-length 512 --output-length 256 --shared-prefix-ratio 0.5 \
  --seed 42 -o report_vllm.json

# SGLang
python -m simulator.run --backend sglang --num-requests 50 \
  --prompt-length 512 --output-length 256 --shared-prefix-ratio 0.5 \
  --seed 42 -o report_sglang.json
```

**示例 2：Prefix cache 命中率实验**

```bash
for ratio in 0.0 0.25 0.5 0.75 1.0; do
  python -m simulator.run --backend vllm --num-requests 100 \
    --shared-prefix-ratio $ratio --num-spec-tokens 0 \
    --seed 42 -o "hit_rate_${ratio}.json"
done
```

**示例 3：投机解码效率分析**

```bash
# 关闭投机
python -m simulator.run --backend vllm --num-spec-tokens 0 \
  --num-requests 50 --seed 42 -o no_spec.json

# K=3，高接受率
python -m simulator.run --backend vllm --num-spec-tokens 3 \
  --acceptance-rates 0.9 0.8 0.7 --draft-accuracy 0.9 \
  --num-requests 50 --seed 42 -o spec_high.json

# K=3，低接受率
python -m simulator.run --backend vllm --num-spec-tokens 3 \
  --acceptance-rates 0.5 0.3 0.1 --draft-accuracy 0.5 \
  --num-requests 50 --seed 42 -o spec_low.json
```

**示例 4：GPU 性能拟合**

```bash
# 用自定义数据点拟合 GPU 延迟模型
python -m simulator.run --backend vllm --num-requests 50 \
  --gpu-data-points '[[0,1,0.5],[1000,1,1.2],[0,2048,25.0],[4000,1,3.5]]' \
  --seed 42 -o custom_gpu.json
```

**示例 5：使用真实模型 config.json**

```bash
python -m simulator.run --backend vllm \
  --model-config /path/to/DeepSeek-V4-Flash/config.json \
  --num-requests 100 --shared-prefix-ratio 0.5 \
  --seed 42 -o ds_v4.json
```

**示例 6：JSON 配置文件**

```bash
# 创建 my_config.json
cat > my_config.json << 'EOF'
{
  "backend": "vllm",
  "max_model_len": 16384,
  "num_kv_cache_blocks": 8192,
  "speculative": {
    "num_spec_tokens": 4,
    "acceptance_rates": [0.9, 0.8, 0.6, 0.4],
    "draft_accuracy": 0.85
  },
  "dataset": {
    "synthetic": {
      "num_requests": 200,
      "prompt_length_fixed": 1024,
      "output_length_fixed": 512,
      "shared_prefix_ratio": 0.7
    }
  }
}
EOF

python -m simulator.run --config my_config.json
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
