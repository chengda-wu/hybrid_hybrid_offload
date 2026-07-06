# Part 2 · 核心概念与分组

> 对应原文档章节，完整目录见 [README.md](README.md)。

[← Part 1](01-model-and-specs.md) · [目录](README.md) · [Part 3 →](03-packed-layout.md)

---

## 2. 四个核心概念的语义

| 概念 | 定义 | DSV4 实例 |
|------|------|----------|
| **Group(组)** | 一组**同 spec 类型**的层；拥有**独立 block table + 独立 block-id 命名空间**，由一个 `SingleTypeKVCacheManager` 管理。 | **5 个 group**（见 §3） |
| **Bucket(桶)** | packed 布局里把**所有组的所有层**按 `page_size_bytes`(padded) 重新分桶。`buckets[ps]` = 该 page size 下的所有 slot。 | **3 个桶**（见 §4） |
| **Slot(槽)** | **层在物理 block 内的"座位号"**，与 token 位置无关。bucketing **逐组**扫描，组内第 i 个 page_size=ps 的层拿 slot_idx=i（`kv_cache_utils.py:1247` `slot_idx = slot_count[ps]; slot_count[ps] += 1`）。层 L 在 block N 内的字节偏移 = `offset(ps, slot_idx) + N × block_stride`（`gpu_model_runner.py:7117`）。**slot 数 = 该 ps 下各组的最大 layer_count**；跨组同 `(ps, slot_idx)` 的层共享同一 `KVCacheTensor`（同一 offset+block_stride 视图），但写入不同物理 block（block_id 不同）。 | 见 §4 |
| **Block(物理块)** | pool 里的一个单元，大小 = `bytes_per_block = Σ ps × #(slots at ps)`。id ∈ [0, num_blocks)。任意时刻**只归一个 group**。 | 1,039,680 B |

**关系链**：
```
Group  ──拥有──▶ 独立 block table ──指向──▶ 物理块(全 pool 共享 id 空间)
Layer  ──属于──▶ 某个 Group
Layer  ──按 page_size 归入──▶ Bucket ──内含──▶ 多个 Slot
Slot   ──对应──▶ 一个 KVCacheTensor(offset, block_stride) = 物理块内一段字节区
```

### slot / block_id / slot_mapping 三者区分（易混淆）

文档里的 "slot" 是 **packed 布局规划层面的 `slot_idx`**，不是运行时的 `slot_mapping`。三者必须分开：

| 概念 | 决定什么 | 由什么决定 | 与 token 关系 |
|------|---------|-----------|-------------|
| **block_id** | 写入哪个物理块 | 请求的 token 段经 group 的 block_table 映射 | **直接绑定**（token 段 → block） |
| **slot_idx**（本文档的 slot） | 层在 block 内的字节偏移 | 该 group 内同 page_size 的**层序号** | **无关**（层的属性，不随 token 变） |
| **slot_mapping**（运行时） | 每个 token 写入 block 内的具体字节位置 | token 位置 + block_table（`block_id×block_size + 块内偏移`） | **直接绑定**（每 token 一个） |

**一句话**：block_id 回答"哪个块"，slot_idx 回答"层在块内的座位"，slot_mapping 回答"token 在块内的座位"。slot_idx 是**层的属性**（同层在所有 block 里座位号相同），不是 token 的属性。

**层的写入地址公式**（`gpu_model_runner.py:7191`，`torch.as_strided`）：
```
层 L 的 KV[block_id=N, token=t] 位于 backing[
    offset(L 的 ps, L 的 slot_idx) + N × block_stride + (t 在块内的偏移)
]
```
- `offset(ps, slot_idx)`：固定，由 packed 规划决定（与 token、block_id 都无关）
- `N × block_stride`：由 block_table 定位到哪个块；
- `t 在块内偏移`：由 token 位置决定（运行时 slot_mapping 算的）。

> 注意：vLLM 源码里 `slot_mapping`（运行时，token 级）和 `slot_idx`（packed 规划，层级）都叫 "slot"，但完全是两个概念。本文档除本节外，"slot" 一律指规划层面的 `slot_idx`。

---

## 3. 分组：7 类 spec → 5 个 group

分组经两步：先按 spec 类型归并成 4 个 `UniformTypeKVCacheSpecs`，再按 layer-tuple 数对齐分裂成 5 个 `KVCacheGroupSpec`。下面每步用 probe 实测的中间结果说明。

### 3.1 第一步：`group_and_unify_kv_cache_specs`（`kv_cache_utils.py:1499`）→ 4 个 uniform spec

- 所有 `MLAAttentionSpec`（#4 `C4mla` 21 + #5 `C128mla` 20 + #6 `C4idx` 21 = **62 层**）→ 1 个 uniform spec（block_size=256）。它们 page_size 不同（37440/1728/8640）但同属 `MLAAttentionSpec`，`is_uniform_type` 只看 block_size 相同（都是 256），故合一。
- 所有 `SlidingWindowMLASpec` 按 `(block_size, sliding_window)` 分组：
  - `(64, 128)` → #1 `SWA_sw`，**43 层**
  - `(4, 8)` → #2 `C4comp_sw` 21 + #7 `C4idxcomp_sw` 21 = **42 层**
  - `(8, 128)` → #3 `C128comp_sw`，**20 层**

实测得到 4 个 uniform spec（注意此时 page_size 还含 32832）：

| uniform spec | block_size | nlayers | page_sizes（576 对齐后，分组前） | (bs,sw) |
|-------------|-----------|---------|--------------------------------|---------|
| U0 (MLA) | 256 | 62 | {1728, 8640, 37440} | (256, —) |
| U1 (`SWA_sw`) | 64 | 43 | {37440} | (64, 128) |
| U2 (`C4comp_sw`+`C4idxcomp_sw`) | 4 | 42 | {8640, 32832} | (4, 8) |
| U3 (`C128comp_sw`) | 8 | 20 | {32832} | (8, 128) |

### 3.2 第二步：`_get_kv_cache_groups_uniform_groups`（`kv_cache_utils.py:1572`）→ 5 个 group

这一步做两件事：(a) 把 SWA-MLA 组的 page_size pad 到 full-MLA 组已有的 page_size；(b) 按 layer-tuple 数对齐并分裂。

**(a) 第二层 page_size padding**（`kv_cache_utils.py:1623-1641`）：把 SWA-MLA 组（U1/U2/U3）每层的 page_size pad 到 full-MLA 组（U0）已有的 page_size 集合 `{1728, 8640, 37440}` 中 nearest larger 的值：

| uniform spec | 层 | pad 前 page | nearest ≥ in {1728,8640,37440} | pad 后 page |
|-------------|-----|------------|------------------------------|------------|
| U0 (MLA, 不 pad) | `C4mla` | 37440 | — | **37,440** |
| U0 | `C128mla` | 1728 | — | **1,728** |
| U0 | `C4idx` | 8640 | — | **8,640** |
| U1 (`SWA_sw`) | `SWA_sw` | 37440 | 37440（已在集合） | **37,440** |
| U2 (`C4comp_sw`+`C4idxcomp_sw`) | `C4comp_sw` | 32832 | 37440 | **37,440** |
| U2 | `C4idxcomp_sw` | 8640 | 8640（已在集合） | **8,640** |
| U3 (`C128comp_sw`) | `C128comp_sw` | 32832 | 37440 | **37,440** |

> 关键效果：**32832 桶消失**（被 pad 到 37440），最终只剩 3 个 distinct page_size：{37440, 8640, 1728}。
> 代价：`C4comp_sw`/`C128comp_sw` 每 slot 实际只用 32832B 却占 37440B → 每 slot 浪费 4,608B（`real_page_size` vs `page_size_padded`）。这一步是为了让 SWA-MLA 组能和 full-MLA 组共享同一个 bucket/tensor（§4 跨组共享 slot）。

**(b) layer-tuple 对齐 + 分裂**（`_approximate_gcd`，`kv_cache_utils.py:1537`）：每个 uniform spec 的 `num_layer_tuples` = 按 page_size 分组后**最多的那个 page_size 的层数**（`get_num_layer_tuples`，`kv_cache_interface.py:836`）：

| uniform spec | num_layer_tuples（实测） | 含义 |
|-------------|------------------------|------|
| U0 (MLA) | **21** | 3 种 page_size 各 20/21/21 层，取最多 = 21 |
| U1 (`SWA_sw`) | **43** | 只有 1 种 page_size，43 层 |
| U2 (`C4comp_sw`+`C4idxcomp_sw`) | **21** | 2 种 page_size 各 21 层，取 21 |
| U3 (`C128comp_sw`) | **20** | 只有 1 种 page_size，20 层 |

`_approximate_gcd([21,43,21,20], lower_bound=21)`（`kv_cache_utils.py:1604`）选 **22**——它暴力遍历 d∈[21,43]，选总 padding 最小的 d（`Σ ceil(x/d)×d - x`）：

| d | 总 padding | 各组 round_up 到 d 的倍数 |
|---|----------|------------------------|
| 21 | 21 | [21, 63, 21, 21] |
| **22** | **5** | [22, 44, 22, 22] |
| 43 | 67 | [43, 43, 43, 43] |

d=22 时 padding 最小（5）。于是每组 round_up 到 22 的倍数，**层数超过 22 的组分裂成多个子组**：U1(`SWA_sw`) 43 层 → `ceil(43/22)=2` 个子组（22+21）；其余组 ≤22 不分裂。

**为什么 tuple 数必须相同（分裂的根因）**：packed 布局里所有 group 共用同一个 block pool、同一个 `block_stride`（`bytes_per_block`）。一个物理 block 被 group G 持有时，G 的所有层都往里写（§4.6），即 G 在每个涉及 page_size 上各填"层数"个 slot。要让所有 group 的 `bytes_per_block = Σ ps × slot_count` 相同（否则无法共享 pool），各组在每个 ps 上的 slot 数必须统一——而 slot 数 = 该组该 ps 的层数 = tuple 数。**tuple 数不同 → slot 数不同 → block 大小不同 → 无法共享 pool**。故必须把所有组 round_up 到同一个 `num_layer_tuples`，超出的组（如 `SWA_sw` 43 > 22）就分裂成多个子组。

**为什么选 d=22 而不是 d=43（不分裂）**：这是"分裂 vs padding"的权衡。若选 d=43（`SWA_sw` 不分裂），其余组 round_up 到 43：G0 21→43、`C4comp_sw` 21→43、`C128comp_sw` 20→43，各 pad 20+ 个空 tuple，总 padding=67。选 d=22 时 `SWA_sw` 分裂成 22+21（只 pad 1 层），其余组各 pad 1-2 层，总 padding=5。`_approximate_gcd` 选最小化总 padding 的 d，故选 22 并让 `SWA_sw` 分裂——比让所有组 pad 到 43 更省内存。

> **分裂 ≠ 层被拆散**：`SWA_sw` 43 层分裂成 G1(22)+G2(21) 后，每层仍完整属于一个子组，只是分到不同 block table。运行时 G1 和 G2 是两个独立 `SingleTypeKVCacheManager`，各自从共享 free list 捞 block（§4.4）。

### 3.3 最终 5 个 group（实测）

| group | spec 类型 | block_size | layer_count | 内含 cache | page_sizes |
|-------|-----------|-----------|-------------|-----------|------------|
| **G0** | U0(MLA) | 256 | 62 | #4 `C4mla` + #5 `C128mla` + #6 `C4idx` | {1728, 8640, 37440} |
| **G1** | U1(`SWA_sw`) 分裂A | 64 | 22 | #1 `SWA_sw`（前 22 层） | {37440} |
| **G2** | U1(`SWA_sw`) 分裂B | 64 | 21 | #1 `SWA_sw`（后 21 层） | {37440} |
| **G3** | U2(`C4comp_sw`+`C4idxcomp_sw`) | 4 | 42 | #2 `C4comp_sw` + #7 `C4idxcomp_sw` | {8640, 37440} |
| **G4** | U3(`C128comp_sw`) | 8 | 20 | #3 `C128comp_sw` | {37440} |

> **为什么 `SWA_sw` 43 层要分裂？** packed 布局要求所有 group 的 layer-tuple 数一致（round_up 到 22）。`SWA_sw` 有 43 层远超 22，必须分裂成 2 个子组（22+21）才能与其它组对齐。这是 packed 布局统一 block_stride 的代价：组间 layer-tuple 数必须相同，否则无法共享物理块。

---

[← Part 1](01-model-and-specs.md) · [目录](README.md) · [Part 3 →](03-packed-layout.md)
