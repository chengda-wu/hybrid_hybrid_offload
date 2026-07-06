# DeepSeek V4 Flash KV Cache 管理深度解析

> 纯基于 `3rdparty/vllm` 源码。所有数值由真实 vLLM 分组函数实测得出
> （`get_kv_cache_groups` + `_bucket_layers_by_page_size` + `_get_kv_cache_config_packed`），
> 非手算。每节标注源文件:行号。
>
> 默认配置：`block_size=256`、`cache_dtype=fp8_ds_mla`、`enable_prefix_caching=True`。

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

## 1. 每层产生的 KVCacheSpec（6 类）

每层 `DeepseekV4Attention.__init__`（`attention.py:153`）+ `DeepseekCompressor`（`compressor.py:188`）创建多个 `AttentionLayerBase` 子模块，`get_kv_cache_spec`（`gpu_model_runner.py:7482`）从 `static_forward_context` 收集它们的 spec：

| # | cache 对象 | spec 类 | 所属层 | block_size | sliding_window | head_size/state_dim | 存储 dtype | 数值精度 |
|---|-----------|---------|--------|-----------|----------------|---------------------|-----------|---------|
| 1 | `swa_cache_layer` | `SlidingWindowMLASpec` | 全43层 | 64 | 128 | 512 | uint8 | fp8 (UE8M0) |
| 2 | C4 `compressor.state_cache` | `SlidingWindowMLASpec` | 21层 | 4 | 8 | 2048 | fp32 | fp32 |
| 3 | C128 `compressor.state_cache` | `SlidingWindowMLASpec` | 20层 | 8 | 128 | 1024 | fp32 | fp32 |
| 4 | C4 主 `kv_cache` (MLA) | `MLAAttentionSpec` | 21层 | 256 (cr=4) | — | 512 | uint8 | fp8 (UE8M0) |
| 5 | C128 主 `kv_cache` (MLA) | `MLAAttentionSpec` | 20层 | 256 (cr=128) | — | 512 | uint8 | fp8 (UE8M0) |
| 6 | C4 `indexer.k_cache` | `MLAAttentionSpec` | 21层 | 256 (cr=4) | — | 132 | uint8 | fp8 (UE8M0) |
| 7 | C4 `indexer.compressor.state_cache` | `SlidingWindowMLASpec` | 21层 | 4 | 8 | 512 | fp32 | fp32 |

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

- **584**（`kv_cache_interface.py:383,610`）：fp8_ds_mla 布局 = `448B NoPE + 128B RoPE + 8B fp8 scale`。SWA(#1) 和主MLA(#4,#5) 用此。
- **132**（`attention.py:729`）：indexer K = `head_dim = 128 + 128//128×4 = 128 fp8 数据 + 4B fp32 scale`。
- **8,192 / 4,096 / 2,048**（`compressor.py:241`）：compressor state = `state_dim × fp32`，`state_dim = 2 × coff × head_dim`，`coff = 1 + (cr==4)`：
  - C4 主(#2)：`coff=2`，`state_dim=2×2×512=2048` → 8,192B
  - C128(#3)：`coff=1`，`state_dim=2×1×512=1024` → 4,096B
  - C4 indexer(#7)：`coff=2`，`state_dim=2×2×128=512` → 2,048B

### page_size 推导：storage_block_size → 未 pad page → 576 对齐后 page

每层的 page_size 经三步推导。`storage_block_size = block_size // compress_ratio`（`kv_cache_interface.py:376,603`）：主MLA 实际存**压缩后**的 token，故每物理 block 只装 `block_size/cr` 个压缩 token；compressor 的 cr=1，block_size 即 storage_block_size。`alignment=576` 在 spec `__post_init__` 经 `_apply_alignment_padding`（`kv_cache_interface.py:327`）把 page 向上 pad 到 576 倍数（FlashMLA UE8M0 分块要求），`page_size_bytes` 属性返回 padded 值（`kv_cache_interface.py:182`）。

| spec | block_size | cr | storage_block_size | per-token (B) | 未 pad page | 576 对齐后 page | 计算 |
|------|-----------|-----|--------------------|--------------|------------|----------------|------|
| #1 SWA | 64 | 1 | 64 | 584 | 37,376 | **37,440** | 65×576 |
| #2 C4 主 comp | 4 | 1 | 4 | 8,192 | 32,768 | **32,832** | 57×576 |
| #3 C128 comp | 8 | 1 | 8 | 4,096 | 32,768 | **32,832** | 57×576 |
| #4 C4 主MLA | 256 | 4 | 64 | 584 | 37,376 | **37,440** | 65×576 |
| #5 C128 主MLA | 256 | 128 | 2 | 584 | 1,168 | **1,728** | 3×576 |
| #6 C4 indexer | 256 | 4 | 64 | 132 | 8,448 | **8,640** | 15×576 |
| #7 C4 indexer comp | 4 | 1 | 4 | 2,048 | 8,192 | **8,640** | 15×576 |

> 未 pad page = `storage_block_size × per-token`。
> SWA(#1) 与 C4 主MLA(#4) 的 page 巧合相等（都 37,376→37,440）：SWA 是 `64 token × 584B`，C4 主MLA 是 `256/4=64 压缩 token × 584B`。
> **§3 还有第二层 padding**：分组时 SWA-MLA 组的 page 会再 pad 到 full-MLA 组已有的 page_size。

---

## 2. 四个核心概念的语义

| 概念 | 定义 | DSV4 实例 |
|------|------|----------|
| **Group(组)** | 一组**同 spec 类型**的层；拥有**独立 block table + 独立 block-id 命名空间**，由一个 `SingleTypeKVCacheManager` 管理。 | **5 个 group**（见 §3） |
| **Bucket(桶)** | packed 布局里把**所有组的所有层**按 `page_size_bytes`(padded) 重新分桶。`buckets[ps]` = 该 page size 下的所有 slot。 | **3 个桶**（见 §4） |
| **Slot(槽)** | 桶内下标 `slot_idx`。bucketing **逐组**扫描，组内第 i 个 page_size=ps 的层拿 slot i；**跨组**处于相同 `(ps, slot_idx)` 的层共享**同一 `KVCacheTensor`**（即同一 offset+block_stride 视图）。**slot 数 = 该 ps 下各组的最大 layer_count**；若某 ps 涉及多个 group（如 SWA 分裂成 G1/G2），各 group 独立递增 slot_idx，会在同 slot 重叠。 | 见 §4 |
| **Block(物理块)** | pool 里的一个单元，大小 = `bytes_per_block = Σ ps × #(slots at ps)`。id ∈ [0, num_blocks)。任意时刻**只归一个 group**。 | 1,039,680 B |

**关系链**：
```
Group  ──拥有──▶ 独立 block table ──指向──▶ 物理块(全 pool 共享 id 空间)
Layer  ──属于──▶ 某个 Group
Layer  ──按 page_size 归入──▶ Bucket ──内含──▶ 多个 Slot
Slot   ──对应──▶ 一个 KVCacheTensor(offset, block_stride) = 物理块内一段字节区
```

---

## 3. 分组：6 类 spec → 5 个 group

分组经两步：先按 spec 类型归并成 4 个 `UniformTypeKVCacheSpecs`，再按 layer-tuple 数对齐分裂成 5 个 `KVCacheGroupSpec`。下面每步用 probe 实测的中间结果说明。

### 3.1 第一步：`group_and_unify_kv_cache_specs`（`kv_cache_utils.py:1499`）→ 4 个 uniform spec

- 所有 `MLAAttentionSpec`（#4 C4主MLA 21 + #5 C128主MLA 20 + #6 C4 indexer 21 = **62 层**）→ 1 个 uniform spec（block_size=256）。它们 page_size 不同（37440/1728/8640）但同属 `MLAAttentionSpec`，`is_uniform_type` 只看 block_size 相同（都是 256），故合一。
- 所有 `SlidingWindowMLASpec` 按 `(block_size, sliding_window)` 分组：
  - `(64, 128)` → #1 SWA，**43 层**
  - `(4, 8)` → #2 C4主comp 21 + #7 C4indexer comp 21 = **42 层**
  - `(8, 128)` → #3 C128 comp，**20 层**

实测得到 4 个 uniform spec（注意此时 page_size 还含 32832）：

| uniform spec | block_size | nlayers | page_sizes（576 对齐后，分组前） | (bs,sw) |
|-------------|-----------|---------|--------------------------------|---------|
| U0 (MLA) | 256 | 62 | {1728, 8640, 37440} | (256, —) |
| U1 (SWA) | 64 | 43 | {37440} | (64, 128) |
| U2 (C4comp) | 4 | 42 | {8640, 32832} | (4, 8) |
| U3 (C128comp) | 8 | 20 | {32832} | (8, 128) |

### 3.2 第二步：`_get_kv_cache_groups_uniform_groups`（`kv_cache_utils.py:1572`）→ 5 个 group

这一步做两件事：(a) 把 SWA-MLA 组的 page_size pad 到 full-MLA 组已有的 page_size；(b) 按 layer-tuple 数对齐并分裂。

**(a) 第二层 page_size padding**（`kv_cache_utils.py:1623-1641`）：把 SWA-MLA 组（U1/U2/U3）每层的 page_size pad 到 full-MLA 组（U0）已有的 page_size 集合 `{1728, 8640, 37440}` 中 nearest larger 的值：

| uniform spec | 层 | pad 前 page | nearest ≥ in {1728,8640,37440} | pad 后 page |
|-------------|-----|------------|------------------------------|------------|
| U0 (MLA, 不 pad) | C4主MLA | 37440 | — | **37,440** |
| U0 | C128主MLA | 1728 | — | **1,728** |
| U0 | C4 indexer | 8640 | — | **8,640** |
| U1 (SWA) | SWA | 37440 | 37440（已在集合） | **37,440** |
| U2 (C4comp) | C4主comp | 32832 | 37440 | **37,440** |
| U2 | C4indexer comp | 8640 | 8640（已在集合） | **8,640** |
| U3 (C128comp) | C128 comp | 32832 | 37440 | **37,440** |

> 关键效果：**32832 桶消失**（被 pad 到 37440），最终只剩 3 个 distinct page_size：{37440, 8640, 1728}。
> 代价：C4/C128 compressor 每 slot 实际只用 32832B 却占 37440B → 每 slot 浪费 4,608B（`real_page_size` vs `page_size_padded`）。这一步是为了让 SWA-MLA 组能和 full-MLA 组共享同一个 bucket/tensor（§4 跨组共享 slot）。

**(b) layer-tuple 对齐 + 分裂**（`_approximate_gcd`，`kv_cache_utils.py:1537`）：每个 uniform spec 的 `num_layer_tuples` = 按 page_size 分组后**最多的那个 page_size 的层数**（`get_num_layer_tuples`，`kv_cache_interface.py:836`）：

| uniform spec | num_layer_tuples（实测） | 含义 |
|-------------|------------------------|------|
| U0 (MLA) | **21** | 3 种 page_size 各 20/21/21 层，取最多 = 21 |
| U1 (SWA) | **43** | 只有 1 种 page_size，43 层 |
| U2 (C4comp) | **21** | 2 种 page_size 各 21 层，取 21 |
| U3 (C128comp) | **20** | 只有 1 种 page_size，20 层 |

`_approximate_gcd([21,43,21,20], lower_bound=21)`（`kv_cache_utils.py:1604`）选 **22**——它暴力遍历 d∈[21,43]，选总 padding 最小的 d（`Σ ceil(x/d)×d - x`）：

| d | 总 padding | 各组 round_up 到 d 的倍数 |
|---|----------|------------------------|
| 21 | 21 | [21, 63, 21, 21] |
| **22** | **5** | [22, 44, 22, 22] |
| 43 | 67 | [43, 43, 43, 43] |

d=22 时 padding 最小（5）。于是每组 round_up 到 22 的倍数，**层数超过 22 的组分裂成多个子组**：U1(SWA) 43 层 → `ceil(43/22)=2` 个子组（22+21）；其余组 ≤22 不分裂。

### 3.3 最终 5 个 group（实测）

| group | spec 类型 | block_size | layer_count | 内含 cache | page_sizes |
|-------|-----------|-----------|-------------|-----------|------------|
| **G0** | U0(MLA) | 256 | 62 | #4 C4主MLA + #5 C128主MLA + #6 C4 indexer | {1728, 8640, 37440} |
| **G1** | U1(SWA) 分裂A | 64 | 22 | #1 SWA（前 22 层） | {37440} |
| **G2** | U1(SWA) 分裂B | 64 | 21 | #1 SWA（后 21 层） | {37440} |
| **G3** | U2(C4comp) | 4 | 42 | #2 C4主comp + #7 C4 indexer comp | {8640, 37440} |
| **G4** | U3(C128comp) | 8 | 20 | #3 C128 comp | {37440} |

> **为什么 SWA 43 层要分裂？** packed 布局要求所有 group 的 layer-tuple 数一致（round_up 到 22）。SWA 有 43 层远超 22，必须分裂成 2 个子组（22+21）才能与其它组对齐。这是 packed 布局统一 block_stride 的代价：组间 layer-tuple 数必须相同，否则无法共享物理块。

---

## 4. Packed 布局：3 个 bucket 的物理块排布

### 4.1 bucketing（实测，`_bucket_layers_by_page_size` `kv_cache_utils.py:1230`）

`_bucket_layers_by_page_size` **逐组**扫描，组内每层按其 page_size 递增 slot_idx；**跨组**处于相同 `(ps, slot_idx)` 的层落进同一 slot。注意 SWA 分裂成的 G1/G2 是两个独立 group，bucketing 时各自独立递增 slot_idx，故两个 group 的 SWA 层会**重叠占用相同 slot**：

| bucket (ps) | slot 数 | 每 slot 内含层（实测 layer name） |
|-------------|---------|--------------------------------|
| **37,440** | 22 | slot 0-19: 5 层 = [c4_mla, c4_comp, c128_comp, swa(G1), swa(G2)]；slot 20: 4 层 = [c4_mla, c4_comp, swa(G1), swa(G2)]（无 c128_comp，C128 只 20 层）；slot 21: 1 层 = [swa(G1)]（G1 第 22 层，G2 只 21 层） |
| **8,640** | 21 | 每 slot: 2 层 = [c4_idx(G0), c4_idx_comp(G3)] |
| **1,728** | 20 | 每 slot: 1 层 = [c128_mla(G0)] |

> **slot 数的含义**：slot 数 = 该 ps 下**各组的最大 layer_count**（`len(buckets[ps])` 取所有组里层数最多的）。37440 桶由 G1(SWA) 的 22 层决定；8640 桶由 G0/G3 的 21 层决定；1728 桶由 G0 的 20 层决定。
>
> **SWA 在 slot 里出现两次**：因为 G1 和 G2 是两个独立 group，bucketing 各自独立扫，slot 0 同时容纳 G1 的第 0 个 SWA 层和 G2 的第 0 个 SWA 层。它们虽在同一 slot（共享 `KVCacheTensor` 视图公式），但运行时写入不同物理 block（G1/G2 各有独立 block table），不冲突（§4.5）。

### 4.2 bytes_per_block 计算（实测）

```
bytes_per_block = Σ (ps × slot_count)
              = 37440×22 + 8640×21 + 1728×20
              = 823,680 + 181,440 + 34,560
              = 1,039,680 B  ≈ 0.99 MiB / 物理块
```

`num_blocks = available_memory // 1,039,680`。共产生 **63 个 `KVCacheTensor`**（= 22+21+20 slots），全部 alias 同一块 backing。

### 4.3 一个 packed block 的内部排布

物理块内按 page_size 分 3 个 region，每 region 内按 slot 切片：

```
字节偏移       ps       slot数  归属层(每 slot 内含层，跨组)                   字节
┌──────────────────────────────────────────────────────────────────────────┐
│ 0            37,440   22     C4主MLA(21), C4主comp(21), C128comp(20),    │ 823,680
│              │                SWA(43, 分属G1/G2两组) — comp pad 到 37440   │
│   ┌────────┬────────┬────────┬ ... ┬────────┬────────┬────────┐          │
│   │ slot 0 │ slot 1 │ slot 2 │     │slot 19 │slot 20 │slot 21 │          │
│   │ 5层    │ 5层    │ 5层    │     │ 5层    │ 4层    │ 1层    │          │
│   │c4mla,  │ ...    │ ...    │     │ ...    │无c128c │仅swaG1│          │
│   │c4comp, │        │        │     │        │        │        │          │
│   │c128c,  │        │        │     │        │        │        │          │
│   │swaG1,  │        │        │     │        │        │        │          │
│   │swaG2   │        │        │     │        │        │        │          │
│   └────────┴────────┴────────┴ ... ┴────────┴────────┴────────┘          │
├──────────────────────────────────────────────────────────────────────────┤
│ 823,680      8,640    21     C4 indexer(G0,21), C4 indexer comp(G3,21)  │ 181,440
│   ┌────────┬────────┬ ... ┬────────┐                                    │
│   │ slot 0 │ slot 1 │     │slot 20 │  每 slot 2 层跨组共享              │
│   └────────┴────────┴ ... ┴────────┘                                    │
├──────────────────────────────────────────────────────────────────────────┤
│ 1,005,120    1,728    20     C128 主MLA(G0,20)                         │ 34,560
│   ┌────────┬────────┬ ... ┬────────┐                                    │
│   │ slot 0 │ slot 1 │     │slot 19 │  每 slot 1 层                      │
│   └────────┴────────┴ ... ┴────────┘                                    │
└──────────────────────────────────────────────────────────────────────────┘
                                                       block_stride = 1,039,680
```

> 37440 区每 slot 的 5 层 = `c4_mla + c4_comp + c128_comp + 2×swa`（2 个 SWA 分别来自 G1、G2 两个 group）。slot 20 无 c128_comp（C128 只 20 层），slot 21 只有 G1 的 SWA（G2 只 21 层）。

**层 L 的视图**（`gpu_model_runner.py:7191`，`torch.as_strided`）：
```
backing[ offset(L_region) + slot_idx(L)×ps(L) + block_id × 1,039,680
        : ... + real_page_size(L) ]   # 只写 real_page_size，padding 部分空闲
```
- `offset(L_region)` = 该 region 起始偏移（0 / 823,680 / 1,005,120）
- `slot_idx(L)` = 组内该 ps 的序号
- `block_id` = 该层 group 的 block table 里指向的物理块 id

### 4.4 整个 pool（竖向 = block_id，横向 = region）

```
                  region→   37,440区              8,640区        1,728区
                            (C4主MLA/C4comp/C128comp/SWA)  (indexer+idxcomp)  (C128主MLA)
 blk #0  (G0拥有)           │C4主MLA    │        │C4idx█│       │C128mla█│  ← G0 跨3个region
 blk #1  (G0拥有)           │C4主MLA    │        │C4idx█│       │C128mla█│     填自己 slot
 blk #2  (G1拥有)           │ SWA(G1)   │        │ (空) │       │ (空)   │  ← 只填本组 slot
 blk #22 (G1拥有)           │ SWA(G1)   │        │ (空) │       │ (空)   │
 blk #23 (G2拥有)           │ SWA(G2)   │        │ (空) │       │ (空)   │  ← G2 的 SWA 也
 blk #44 (G3拥有)           │C4comp +   │        │C4idxc│       │ (空)   │     在 37440 区,
                            │ idxcomp   │        │(8640)│       │        │     但不同物理块
 blk #65 (G4拥有)           │ C128comp  │        │ (空) │       │ (空)   │
        ...
 block_id 命名空间 0..num_blocks-1 共享；同一时刻每个 id 只归一个 group
```

> **关键**：G1(SWA-A) 和 G2(SWA-B) 的 SWA 层都在 37440 区的**同一 slot 位置**（视图 offset 相同），但写入**不同物理 block**——G1 拥有的 block 和 G2 拥有的 block 是不同 id。这正是跨组共享 slot 安全的核心（§4.5）。
>
> G0(MLA) 同时占用 37440 区（C4主MLA）、8640 区（C4 indexer）、1728 区（C128主MLA）三个 region 的对应 slot——因为 MLA 组内含 3 种 page_size 的层。G0 拥有的一个物理块，在三个 region 都填自己 slot 的数据。

### 4.5 跨组共享 slot 为何安全（核心 invariant）

1. 所有 63 个 `KVCacheTensor` alias 同一块 `packed_backing`（`gpu_model_runner.py:7060-7076`，所有 `block_stride>0` 的 tensor 共享同一 backing）。
2. 每个 group **独立**从 `block_pool.get_new_blocks()` 捞 block id（`single_type_kv_cache_manager.py:302`）。
3. prefix-cache 的 hash key 含 `group_id`（`BlockHashWithGroupId`，`kv_cache_utils.py:57`）。
4. ⇒ **任意时刻一个物理 block id 只被一个 group 持有**。于是 `(ps, slot)` 相同但属于不同组的层，虽 `offset` 相同，却写入不同物理块 → 永不写冲突。

> 这正是 `_bucket_layers_by_page_size` 注释"they have independent block tables so block-id namespaces never collide"的含义：不是 id 数值不同，而是**不同组永不同时持有同一物理块**，所以同 offset 的视图安全复用。
>
> **代价**：每个物理块只被拥有它的那个 group 部分填充。如 G4(C128comp, 20 层) 拥有一个物理块时，只在 37440 区填入它组内某 1 个 c128_comp 层的 slot（37440B 区里用 32832B，8640/1728 区全空）。同理 G1/G2(SWA) 拥有的块只在 37440 区填 1 个 SWA slot。这是 packed 布局换取消除碎片化的代价——牺牲单块填充率，换来所有组共享同一 block pool 无碎片。

---

## 5. scheduler / hash block size（实测）

`resolve_kv_cache_block_sizes`（`kv_cache_utils.py:607`）：

```
group_block_sizes = [256, 64, 64, 4, 8]  (5 个 group)
scheduler_block_size = LCM(...) = 256   # num_computed_tokens 对齐粒度
hash_block_size      = GCD(...) = 4      # block hash 计算粒度（prefix caching 开启时）
```

- **block hash 每 4 token 算一次**，链式累计（`parent_hash + tokens + extra_keys`，`kv_cache_utils.py:577`）。
- 每个 group 用 `BlockHashListWithBlockSize`（`kv_cache_utils.py:2156`）把 4-token hash 缩放到自己的 block_size（都必须是 4 的倍数：256/4=64, 64/4=16, 4/4=1, 8/4=2 ✓）。
- `num_computed_tokens` 按 `scheduler_block_size=256` 对齐。

---

## 6. 运行时 block 生命周期

`KVCacheManager`（`kv_cache_manager.py:110`）通过 `HybridKVCacheCoordinator`（`kv_cache_coordinator.py:514`）给每个 group 维护独立 block table，底层共享 pool：

| 方法 | 作用 | 源码 |
|------|------|------|
| `get_computed_blocks(req)` | APC 查找；`find_longest_cache_hit` 跨组定长收敛 | `kv_cache_manager.py:202`, `coordinator.py:630` |
| `allocate_slots(...)` | 为新 token 分配 block，每组各自填 block table | `kv_cache_manager.py:244` |
| `free(req)` | 每组 manager 减 ref_cnt；归 0 才还 free list | `kv_cache_manager.py:462`, `coordinator.py:285` |
| spec decode | `_update_after_schedule` 预分配 1+K block；`update_from_output` 释放被拒的 | `v1/core/sched/scheduler.py:1154,1488` |

**SWA 的"窗口外丢弃"**：DSV4 不真正 evict 窗口外的 block，而是靠 `decode_swa_indices` 只索引窗口内 slot（`sparse_swa.py:612` triton kernel）——block 还在，只是不参与 attention。

---

## 7. 完整例子：分配 / APC 命中 / 释放

### 7.1 设定

- `hash_block_size = 4`，`scheduler_block_size = 256`
- pool 初始全空闲，free list 按 id 升序发放
- **Request A**：512 token prompt，全新（无命中）
- **Request B**：512 token prompt，**前 256 token 与 A 完全相同**（共享前缀），后 256 token 不同

每组对 N 个 token 需要的 block 数（`ceil(N / block_size)`）：

| Group | block_size | 256 token | 512 token |
|-------|-----------|-----------|-----------|
| G0 (MLA) | 256 | 1 | 2 |
| G1 (SWA-A) | 64 | 4 | 8 |
| G2 (SWA-B) | 64 | 4 | 8 |
| G3 (C4comp) | 4 | 64 | 128 |
| G4 (C128comp) | 8 | 32 | 64 |
| **合计** | | **105** | **210** |

> 注：C4comp 的 block_size=4，故 512 token 需要 128 个 block——这是 DSV4 需要大 block pool 的主因。

### 7.2 Phase 1：A 到达，全分配（无命中）

A 的 `get_computed_blocks` → hashmap 空，`num_computed = 0`。

`allocate_slots(num_new_tokens=512)`：5 个 group 各自从共享 free list 顺序捞**互不重叠**的 block id（`single_type_kv_cache_manager.py:302`，每组独立 `get_new_blocks`）：

| Group | A 拿到的 block id | 数量 |
|-------|------------------|------|
| G0 (MLA) | 0, 1 | 2 |
| G1 (SWA-A) | 2..9 | 8 |
| G2 (SWA-B) | 10..17 | 8 |
| G3 (C4comp) | 18..145 | 128 |
| G4 (C128comp) | 146..209 | 64 |

A forward。每跑完一个 `hash_block_size`(4 token) 边界且该 group 的 block 满，`cache_blocks` 把 `(block_hash, group_id) → block_id` 插入全局 hashmap。A 跑完后，A 的所有满 block 都进了 hashmap（key 含 group_id，故 5 组互不干扰）。

### 7.3 Phase 2：B 到达，APC 命中共享前缀

B 前 256 token 与 A 相同 → 算出的 block hash 链前 `256/4 = 64` 个 hash 与 A 完全一致。

`get_computed_blocks` → `find_longest_cache_hit`（`coordinator.py:630`）逐组查 hashmap，用 `BlockHashListWithBlockSize` 把 4-token hash 缩放到各组 block_size：

| Group | block_size | 命中 block 数 | 命中 token | 共享 A 的 block id |
|-------|-----------|-------------|-----------|-------------------|
| G0 (MLA) | 256 | 256/256 = 1 | 256 | 0 |
| G1 (SWA-A) | 64 | 256/64 = 4 | 256 | 2,3,4,5 |
| G2 (SWA-B) | 64 | 4 | 256 | 10,11,12,13 |
| G3 (C4comp) | 4 | 256/4 = 64 | 256 | 18..81 |
| G4 (C128comp) | 8 | 256/8 = 32 | 256 | 146..177 |

定长收敛后 `num_computed = 256`（各组一致，取最小）。

**命中 block 不新分配，直接共享**：ref_cnt `1 → 2`，**零拷贝**。

随后 B 还需 `512 - 256 = 256` 个新 token 的 block，`allocate_slots(num_new_tokens=256, num_new_computed_tokens=256)`，从 free list 续捞（下一个空闲 id = 210）：

| Group | B 新拿到 block id | 数量 |
|-------|------------------|------|
| G0 (MLA) | 210 | 1 |
| G1 (SWA-A) | 211..214 | 4 |
| G2 (SWA-B) | 215..218 | 4 |
| G3 (C4comp) | 219..282 | 64 |
| G4 (C128comp) | 283..314 | 32 |

（G4 新拿 32 个：283..314）

B 的 block table（前半共享、后半独占），以 G1(SWA-A) 为例：
```
G1: [2,3,4,5 (shared, ref_cnt=2), 211,212,213,214 (独占, ref_cnt=1),
     6,7,8,9 (shared), ...]   # 8 blocks = 4 shared + 4 独占
```
B forward。新满 block 进 hashmap。

### 7.4 Phase 3：B 完成 → free(B)

每组 manager 对 B 的 block 减 ref_cnt（`coordinator.py:285`）：

- **共享 block**（如 G0 的 0、G1 的 2-5、G3 的 18-81、G4 的 146-177）：ref_cnt `2 → 1`，**不释放**（A 仍引用）。
- **B 独占 block**（如 G0 的 210、G1 的 211-214、G3 的 219-282、G4 的 283-314）：ref_cnt `1 → 0`，归还 free list，标记 evictable；hash 暂留（evict 时才删，给后续请求复用机会）。

### 7.5 Phase 4：A 完成 → free(A)

- **共享 block**：ref_cnt `1 → 0`，归还 free list，evict 时删 hash。
- **A 独占 block**（如 G0 的 1、G1 的 6-9、G3 的 82-145、G4 的 178-209）：ref_cnt `1 → 0`，归还。
- pool 恢复初始全空闲状态。

### 7.6 关键观察

1. **跨组 block id 永不重叠**：A 的 5 组分别拿到 `0-1 / 2-9 / 10-17 / 18-145 / 146-209`，因为同一 free list 顺序消费。这是 packed 布局安全性的运行时保障（§4.5 invariant #2）。
2. **APC 按 group 独立命中**：`(hash, group_id)` 复合 key 让 5 组各自查各自的命中，互不干扰；`find_longest_cache_hit` 用定长收敛保证 5 组命中长度一致（取最小）。
3. **命中粒度 = hash_block_size = 4 token**，但每组实际命中 block 数按各自 block_size 折算（G0 1 个 256-block，G3 64 个 4-block，G4 32 个 8-block）。
4. **共享 = ref_cnt++，非拷贝**：B 命中 A 的 block 时零拷贝，仅 bump 引用计数。
5. **释放按引用计数**：`free` 不立即删 hash，evict 时才删——这就是为何 B 完成后其独占 block 仍可能被后续 C 请求命中。

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

本文档所有数值由 `dsv4_layout.py` 用真实 vLLM 函数实测（非手算）。

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

> 若 vLLM submodule 升级后分组逻辑变化，重跑此脚本即可更新文档数值。

---

*文档生成于 2026-07-06，基于 vLLM submodule HEAD（ab132ee98）。*
*所有 group/bucket/slot/bytes 数值由真实 vLLM 分组函数实测验证。*
