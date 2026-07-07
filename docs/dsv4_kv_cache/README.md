# DeepSeek V4 Flash KV Cache 管理深度解析

> 纯基于 `3rdparty/vllm` 源码。所有数值由真实 vLLM 分组函数实测得出
> （`get_kv_cache_groups` + `_bucket_layers_by_page_size` + `_get_kv_cache_config_packed`），
> 非手算。每节标注源文件:行号。
>
> 默认配置：`block_size=256`、`cache_dtype=fp8_ds_mla`、`enable_prefix_caching=True`。

文档按主题拆分为 6 个部分，便于逐节阅读。章节编号 §0–§14 保留，跨文件引用（如"见 §4.5"）依旧有效。

## 目录

| 部分 | 内容 | 章节 |
|------|------|------|
| [Part 1 · 模型参数与 KVCacheSpec](01-model-and-specs.md) | DSV4 模型参数、7 类 spec、dtype（uint8 vs fp8）、per-token 字节、page_size 推导 | §0–§1 |
| [Part 2 · 核心概念与分组](02-concepts-and-grouping.md) | group/bucket/slot/block 语义、slot 与 block_id/slot_mapping 区分、7 类 spec → 5 个 group 的分组过程 | §2–§3 |
| [Part 3 · Packed 布局](03-packed-layout.md) | 3 个 bucket 的物理块排布、bytes_per_block、填充率分析、跨组共享 slot 的安全性 | §4 |
| [Part 4 · 运行时与 APC 例子](04-runtime-and-apc.md) | scheduler/hash block size、block 生命周期、分配 / APC 命中 / 释放完整例子 | §5–§7 |
| [Part 5 · 附录](05-appendix.md) | 源码索引、数值实测验证、`dsv4_layout.py` probe 脚本说明 | §8–§9 |
| [Part 6 · Delta 段三角形计算可行性](06-staircase-delta-feasibility.md) | APC 命中分化、delta 段冗余重算、staircase 三角形论证、省量分析、前提与替代方案 | §10–§14 |

## 快速验证

```bash
cd /home/witcher/hybrid_hybrid_offload
.venv/bin/python docs/dsv4_kv_cache/dsv4_layout.py
```

无需 GPU，只构造 spec dataclass 并跑布局规划器。预期输出见 [Part 5 §9.3](05-appendix.md)。

---

*基于 vLLM submodule HEAD（ab132ee98）。所有 group/bucket/slot/bytes 数值由真实 vLLM 分组函数实测验证。*
