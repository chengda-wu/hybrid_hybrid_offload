# Part 1 · 模型参数与 KVCacheSpec

> 对应原文档章节，完整目录见 [README.md](README.md)。

[目录](README.md) · [Part 2 →](02-concepts-and-grouping.md)

---

## 0. DSV4-Flash 模型参数

来自 HF `config.json`（DSV4-Flash 实测）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `num_hidden_layers` | 43 | 主模型 decoder 层数。MTP 是独立的 proposer 模型（`DeepSeekV4MTP`，registry `DeepSeekV4MTPModel`），仅 spec decode 开启时加载，不进主模型 KV group，本文档不涉及 |
| `head_dim` | 512 | = `qk_nope_head_dim(448) + qk_rope_head_dim(64)` |
| `qk_rope_head_dim` | 64 | RoPE 部分头维 |
| `sliding_window` | 128 | SWA 窗口（`config.sliding_window`） |
| `compress_ratios` | `[0,0,4,128,4,128,...,4]` | 43 层：前 2 层 SWA-only(cr≤1)，后 41 层 4/128 交替 |
| 层分布 | SWA-only=2, C4=21, C128=20 | SWA cache 每层都有(43)；C4/C128 按 cr |
| `index_head_dim` | 128 | indexer 的 head_dim |
| `cache_dtype` | `fp8_ds_mla` | UE8M0 block-scaled fp8，packed as uint8 |

> **SWA cache 存在于全部 43 层**（包括 C4/C128 层）；C4=21、C128=20 按 `compress_ratio` 区分。

---

## 1. 每层产生的 KVCacheSpec（7 类）

每层 `DeepseekV4Attention.__init__`（`attention.py:153`）+ `DeepseekCompressor`（`compressor.py:188`）创建多个 `AttentionLayerBase` 子模块，`get_kv_cache_spec`（`gpu_model_runner.py:7482`）从 `static_forward_context` 收集它们的 spec：

| # | 代号 | cache 对象 | spec 类 | 所属层 | block_size | sliding_window | head_size/state_dim | 存储 dtype | 数值精度 |
|---|------|-----------|---------|--------|-----------|----------------|---------------------|-----------|---------|
| 1 | `SWA_sw` | `swa_cache_layer` | `SlidingWindowMLASpec` | 全43层 | 64 | 128 | 512 | uint8 | fp8 (UE8M0) |
| 2 | `C4comp_sw` | C4 `compressor.state_cache` | `SlidingWindowMLASpec` | 21层 | 4 | 8 | 2048 | fp32 | fp32 |
| 3 | `C128comp_sw` | C128 `compressor.state_cache` | `SlidingWindowMLASpec` | 20层 | 8 | 128 | 1024 | fp32 | fp32 |
| 4 | `C4mla` | C4 主 `kv_cache` (MLA) | `MLAAttentionSpec` | 21层 | 256 (cr=4) | — | 512 | uint8 | fp8 (UE8M0) |
| 5 | `C128mla` | C128 主 `kv_cache` (MLA) | `MLAAttentionSpec` | 20层 | 256 (cr=128) | — | 512 | uint8 | fp8 (UE8M0) |
| 6 | `C4idx` | C4 `indexer.k_cache` | `MLAAttentionSpec` | 21层 | 256 (cr=4) | — | 132 | uint8 | fp8 (UE8M0) |
| 7 | `C4idxcomp_sw` | C4 `indexer.compressor.state_cache` | `SlidingWindowMLASpec` | 21层 | 4 | 8 | 512 | fp32 | fp32 |

> **命名约定**：本文档统一用"代号"列的简称指代各 KV 类型。**`_sw` 后缀表示 `SlidingWindowMLASpec`**（运行时用 `SlidingWindowManager`，跳过窗口外 token，见 §7.3）；不带 `_sw` 的是 `MLAAttentionSpec`（`FullAttentionManager`，不跳过）。group 命名同理：`G3(C4comp_sw)`、`G4(C128comp_sw)`。

> **存储 dtype vs 数值精度**：#1/#4/#5/#6 的物理 tensor 是 `torch.uint8`，但数值精度是 **fp8**。
> 详见下方"### dtype：uint8 与 fp8 的关系"。

> 第 7 项是 indexer 内部的 compressor（`attention.py:737`），head_dim=128 → state_dim=`2×coff×128 = 2×2×128 = 512`。
> 它与 C4 主 compressor 同为 `(block_size=4, sliding_window=8)`，故后续分到同一组。

### dtype：uint8 与 fp8 的关系

DSV4 默认 `cache_dtype=fp8_ds_mla`，由 `_resolve_dsv4_kv_cache_dtype`（`attention.py:64-94`）解析为 `(cache_dtype_str, torch_dtype)`：

| `--kv-cache-dtype` | 布局 | `torch_dtype`（物理存储） | 数值精度 |
|--------------------|------|-------------------------|---------|
| `fp8_ds_mla`（DSV4 默认） | UE8M0 block-scaled，paged | **`torch.uint8`** | fp8（带 UE8M0 block scale） |
| `fp8`（普通 per-tensor） | plain-row（FlashInfer） | `torch.float8_e4m3fn` | fp8 E4M3 |
| `auto` / `bfloat16` | plain-row | `torch.bfloat16` | bf16 |

**为什么 fp8 却存成 uint8？** `fp8_ds_mla` 不是简单的 fp8 元素数组，而是 DeepSeek 自定义的**打包字节格式**：每 token `448B fp8 NoPE + 128B fp8 RoPE + 8B scale = 584B`（`sparse_swa.py:139`）。这种混合了 fp8 数据与 scale 字节的布局无法用 `torch.float8_e4m3fn` 单一 dtype 表示，因此底层 tensor 用 `uint8`（原始字节流），由专门 kernel（`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`，`attention.py:550`）按 UE8M0 格式读写。

**对 page_size 的影响**：`fp8_ds_mla` 路径下，`real_page_size_bytes` **硬编码**为 `storage_block_size × 584`（`kv_cache_interface.py:380-385`），不经过 `dtype × head_dim` 公式，故 dtype 字段不影响字节数。普通 fp8/bf16 路径才走 `storage_block_size × num_kv_heads × head_dim × sizeof(dtype)`（`kv_cache_interface.py:393-398`）。

**compressor state 为何是 fp32**：compressor 的 `state_cache` 存的是递归压缩状态（kv_state + score_state），`CompressorStateCache` 显式 `assert self.dtype == torch.float32`（`compressor.py:139`），是真正的 fp32 存储，非打包格式。

### per-token 字节数的含义

- **584**（`kv_cache_interface.py:383,610`，`test_fused_deepseek_v4_qnorm_rope_kv_insert.py:45`）：fp8_ds_mla 布局 = `448B NoPE + 128B RoPE + 8B fp8 scale`。这是**混合精度打包**：NoPE 部分 448 element × 1B (fp8) = 448B；RoPE 部分 64 element × 2B (bf16) = 128B（RoPE 需更高精度）；+ 8B scale。故 `head_dim=512 = nope(448)+rope(64)` 与 584B 自洽。`SWA_sw`(#1) 和 `C4mla`(#4)/`C128mla`(#5) 用此。
- **132**（`attention.py:729`）：indexer K = `head_dim = 128 + 128//128×4 = 128 fp8 数据 + 4B fp32 scale`。`C4idx`(#6) 用此。
- **8,192 / 4,096 / 2,048**（`compressor.py:241`）：compressor state = `state_dim × fp32`，`state_dim = 2 × coff × head_dim`，`coff = 1 + (cr==4)`：
  - `C4comp_sw`(#2)：`coff=2`，`state_dim=2×2×512=2048` → 8,192B
  - `C128comp_sw`(#3)：`coff=1`，`state_dim=2×1×512=1024` → 4,096B
  - `C4idxcomp_sw`(#7)：`coff=2`，`state_dim=2×2×128=512` → 2,048B

### page_size 推导：storage_block_size → 未 pad page → 576 对齐后 page

每层的 page_size 经三步推导。`storage_block_size = block_size // compress_ratio`（`kv_cache_interface.py:376,603`）：主MLA 实际存**压缩后**的 token，故每物理 block 只装 `block_size/cr` 个压缩 token；compressor 的 cr=1，block_size 即 storage_block_size。`alignment=576` 在 spec `__post_init__` 经 `_apply_alignment_padding`（`kv_cache_interface.py:327`）把 page 向上 pad 到 576 倍数（FlashMLA UE8M0 分块要求），`page_size_bytes` 属性返回 padded 值（`kv_cache_interface.py:182`）。

| spec | spec 类型 | block_size | cr | storage_block_size | per-token (B) | 未 pad page | 576 对齐后 page | 计算 |
|------|----------|-----------|-----|--------------------|--------------|------------|----------------|------|
| #1 `SWA_sw` | SlidingWindowMLA | 64 | 1 | 64 | 584 | 37,376 | **37,440** | 65×576 |
| #2 `C4comp_sw` | SlidingWindowMLA | 4 | 1 | 4 | 8,192 | 32,768 | **32,832** | 57×576 |
| #3 `C128comp_sw` | SlidingWindowMLA | 8 | 1 | 8 | 4,096 | 32,768 | **32,832** | 57×576 |
| #4 `C4mla` | MLA (FullAttention) | 256 | 4 | 64 | 584 | 37,376 | **37,440** | 65×576 |
| #5 `C128mla` | MLA (FullAttention) | 256 | 128 | 2 | 584 | 1,168 | **1,728** | 3×576 |
| #6 `C4idx` | MLA (FullAttention) | 256 | 4 | 64 | 132 | 8,448 | **8,640** | 15×576 |
| #7 `C4idxcomp_sw` | SlidingWindowMLA | 4 | 1 | 4 | 2,048 | 8,192 | **8,640** | 15×576 |

> 未 pad page = `storage_block_size × per-token`。
> `SWA_sw`(#1) 与 `C4mla`(#4) 的 page 巧合相等（都 37,376→37,440）：SWA 是 `64 token × 584B`，C4mla 是 `256/4=64 压缩 token × 584B`。
> **§3 还有第二层 padding**：分组时 SWA-MLA 组的 page 会再 pad 到 full-MLA 组已有的 page_size。
>
> **spec 类型与运行时行为**：`SlidingWindowMLA`（#1,#2,#3,#7）运行时用 `SlidingWindowManager`，会跳过窗口外 token（用 null_block 占位，见 §7.3）；`MLA (FullAttention)`（#4,#5,#6）用 `FullAttentionManager`，不跳过，所有 token 都分配真实 block。这个差异直接影响 APC 命中（§7）和填充率（§4.6）。

---

[目录](README.md) · [Part 2 →](02-concepts-and-grouping.md)
