# Hybrid Attention KV Offload

探索混合注意力（Hybrid Attention）场景下，多种注意力状态的 KV Cache 管理方式及多级卸载策略。

## 背景

在大模型推理中，不同注意力机制（如 SWA、CSA/HCA、dLLM 等）会产生不同生命周期和访问频率的 KV Cache。本仓库旨在探索一套统一的 KV 管理框架，支持根据注意力状态特征将 KV 数据分级存储与卸载，在保证推理精度的前提下最大化显存利用效率。

## 关注的注意力状态

| 状态 | 全称 | 特点 |
|------|------|------|
| **SWA** | Sliding Window Attention | 局部窗口，高频访问，数据生命周期短 |
| **CSA** | Context Sparse Attention | 稀疏全局上下文，中频访问 |
| **HCA** | Hierarchical Context Attention | 层次化上下文，金字塔式访问模式 |
| **dLLM** | Deep LLM Attention | 深层注意力，长距离依赖，数据生命周期长 |

## 多级卸载层级

```
+------+
| HBM  |  GPU 高带宽显存   热点 KV，极低延迟
+------+
| DRAM |  CPU 内存          温数据，低延迟
+------+
| SSD  |  NVMe 固态存储     冷数据，中延迟
+------+
| HDD  |  机械硬盘           归档数据，高延迟
+------+
| S3   |  对象存储           历史数据，离线场景
+------+
```

## 研究目标

1. **统一 KV 抽象**：为不同注意力状态提供统一的 KV Cache 接口
2. **智能调度**：基于访问模式（频率、语义相似度、时间局部性）自动决策 KV 数据驻留层级
3. **分级卸载**：支持 HBM → DRAM → SSD → HDD → S3 的逐级卸载与按需换入
4. **混合精度**：探索不同存储层级下的 KV 量化/压缩策略
5. **与推理引擎集成**：基于 vllm、sglang 等主流推理框架实现上述能力

## 第三方依赖

本仓库通过 git submodule 引入以下推理引擎：

- [vllm](https://github.com/vllm-project/vllm) — `3rdparty/vllm`
- [sglang](https://github.com/sgl-project/sglang) — `3rdparty/sglang`

```bash
# 克隆时同时拉取 submodule
git clone --recurse-submodules <this-repo-url>

# 或克隆后初始化
git submodule update --init --recursive
```

## 目录结构

```
hybrid_hybrid_offload/
├── 3rdparty/           # 第三方推理引擎 (submodule)
│   ├── vllm/
│   └── sglang/
├── docs/               # 设计文档
├── src/                # 核心代码
├── tests/              # 测试
└── README.md
```

## License

TBD
