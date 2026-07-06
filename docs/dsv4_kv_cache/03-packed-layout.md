# Part 3 · Packed 布局

> 对应原文档章节，完整目录见 [README.md](README.md)。

[← Part 2](02-concepts-and-grouping.md) · [目录](README.md) · [Part 4 →](04-runtime-and-apc.md)

---

## 4. Packed 布局：3 个 bucket 的物理块排布

### 4.1 bucketing（实测，`_bucket_layers_by_page_size` `kv_cache_utils.py:1230`）

`_bucket_layers_by_page_size` **逐组**扫描，组内每层按其 page_size 递增 slot_idx；**跨组**处于相同 `(ps, slot_idx)` 的层落进同一 slot。注意 `SWA_sw` 分裂成的 G1/G2 是两个独立 group，bucketing 时各自独立递增 slot_idx，故两个 group 的 `SWA_sw` 层会**重叠占用相同 slot**：

| bucket (ps) | slot 数 | 每 slot 内含层（实测 layer name） |
|-------------|---------|--------------------------------|
| **37,440** | 22 | slot 0-19: 5 层 = [`C4mla`, `C4comp_sw`, `C128comp_sw`, `SWA_sw`(G1), `SWA_sw`(G2)]；slot 20: 4 层 = [`C4mla`, `C4comp_sw`, `SWA_sw`(G1), `SWA_sw`(G2)]（无 `C128comp_sw`，C128 只 20 层）；slot 21: 1 层 = [`SWA_sw`(G1)]（G1 第 22 层，G2 只 21 层） |
| **8,640** | 21 | 每 slot: 2 层 = [`C4idx`(G0), `C4idxcomp_sw`(G3)] |
| **1,728** | 20 | 每 slot: 1 层 = [`C128mla`(G0)] |

> **slot 数的含义**：slot 数 = 该 ps 下**各组的最大 layer_count**（`len(buckets[ps])` 取所有组里层数最多的）。37440 桶由 G1(`SWA_sw`) 的 22 层决定；8640 桶由 G0/G3 的 21 层决定；1728 桶由 G0 的 20 层决定。
>
> **`SWA_sw` 在 slot 里出现两次**：因为 G1 和 G2 是两个独立 group，bucketing 各自独立扫，slot 0 同时容纳 G1 的第 0 个 `SWA_sw` 层和 G2 的第 0 个 `SWA_sw` 层。它们虽在同一 slot（共享 `KVCacheTensor` 视图公式），但运行时写入不同物理 block（G1/G2 各有独立 block table），不冲突（§4.5）。

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
│ 0            37,440   22     C4mla(21), C4comp_sw(21), C128comp_sw(20), │ 823,680
│              │                SWA_sw(43, 分属G1/G2两组) — comp pad 到37440│
│   ┌────────┬────────┬────────┬ ... ┬────────┬────────┬────────┐          │
│   │ slot 0 │ slot 1 │ slot 2 │     │slot 19 │slot 20 │slot 21 │          │
│   │ 5层    │ 5层    │ 5层    │     │ 5层    │ 4层    │ 1层    │          │
│   │C4mla   │ ...    │ ...    │     │ ...    │无C128c │仅SWA  │          │
│   │C4comp_sw│       │        │     │        │omp_sw  │sw G1  │          │
│   │C128c   │        │        │     │        │        │        │          │
│   │omp_sw  │        │        │     │        │        │        │          │
│   │SWA_sw G1│       │        │     │        │        │        │          │
│   │SWA_sw G2│       │        │     │        │        │        │          │
│   └────────┴────────┴────────┴ ... ┴────────┴────────┴────────┘          │
├──────────────────────────────────────────────────────────────────────────┤
│ 823,680      8,640    21     C4idx(G0,21), C4idxcomp_sw(G3,21)          │ 181,440
│   ┌────────┬────────┬ ... ┬────────┐                                    │
│   │ slot 0 │ slot 1 │     │slot 20 │  每 slot 2 层跨组共享              │
│   └────────┴────────┴ ... ┴────────┘                                    │
├──────────────────────────────────────────────────────────────────────────┤
│ 1,005,120    1,728    20     C128mla(G0,20)                            │ 34,560
│   ┌────────┬────────┬ ... ┬────────┐                                    │
│   │ slot 0 │ slot 1 │     │slot 19 │  每 slot 1 层                      │
│   └────────┴────────┴ ... ┴────────┘                                    │
└──────────────────────────────────────────────────────────────────────────┘
                                                       block_stride = 1,039,680
```

> 37440 区每 slot 的 5 层 = `C4mla + C4comp_sw + C128comp_sw + 2×SWA_sw`（2 个 `SWA_sw` 分别来自 G1、G2 两个 group）。slot 20 无 `C128comp_sw`（C128 只 20 层），slot 21 只有 G1 的 `SWA_sw`（G2 只 21 层）。

**层 L 的视图**（`gpu_model_runner.py:7191`，`torch.as_strided`）：
```
backing[ offset(L_region) + slot_idx(L)×ps(L) + block_id × 1,039,680
        : ... + real_page_size(L) ]   # 只写 real_page_size，padding 部分空闲
```
- `offset(L_region)` = 该 region 起始偏移（0 / 823,680 / 1,005,120）
- `slot_idx(L)` = 组内该 ps 的序号
- `block_id` = 该层 group 的 block table 里指向的物理块 id

### 4.4 整个 pool（竖向 = block_id，横向 = region）

以 §7.2 中 A 请求的分配为例（block 0 是 `null_block`，A 从 block 1 起）：

```
                  region→   37,440区              8,640区        1,728区
                            (C4mla/C4comp_sw/C128comp_sw/SWA_sw)  (C4idx+C4idxcomp_sw)  (C128mla)
 blk #0  null_block         │ (占位)    │        │ (空) │       │ (空)   │  ← 保留，不归任何 group
 blk #1  (G0拥有)           │C4mla×21   │        │C4idx×21│     │C128mla×20│  ← G0 跨3个region
 blk #2  (G0拥有)           │C4mla×21   │        │C4idx×21│     │C128mla×20│     填全部层 slot
 blk #3  (G1拥有)           │SWA_sw×22  │        │ (空) │       │ (空)   │  ← 只填 37440 区
 blk #10 (G1拥有)           │SWA_sw×22  │        │ (空) │       │ (空)   │
 blk #11 (G2拥有)           │SWA_sw×21  │        │ (空) │       │ (空)   │  ← G2 的 SWA_sw 也在
 blk #19 (G3拥有)           │C4comp_sw×21│       │C4idxcomp_sw×21│ (空)   │     37440 区, 但不同
 blk #147(G4拥有)           │C128comp_sw×20│     │ (空) │       │ (空)   │     物理 block
        ...
 block_id 命名空间 0..num_blocks-1 共享；同一时刻每个 id 只归一个 group
```

> 上图按 §7.2 A 的分配顺序标注 id（G0:1-2, G1:3-10, G2:11-18, G3:19-146, G4:147-210）。多请求并发时各 group 的 id 会交错，但"同一时刻一个 block 只归一个 group"不变。
>
> **关键**：G1 和 G2 的 `SWA_sw` 层都在 37440 区的**同一 slot 位置**（视图 offset 相同），但写入**不同物理 block**——G1 拥有的 block 和 G2 拥有的 block 是不同 id。这正是跨组共享 slot 安全的核心（§4.5）。
>
> G0(MLA) 同时占用 37440 区（`C4mla`×21）、8640 区（`C4idx`×21）、1728 区（`C128mla`×20）三个 region——因为 MLA 组内含 3 种 page_size 的层。G0 拥有的一个物理块，在三个 region 都填自己全部层的 slot（共 62 层）。
>
> **block 0 是 `null_block`**（`block_pool.py:191`），保留给 `SWA_sw` 窗口外占位等用途，不归任何 group，不计 ref_cnt。

### 4.5 跨组共享 slot 为何安全（核心 invariant）

1. 所有 63 个 `KVCacheTensor` alias 同一块 `packed_backing`（`gpu_model_runner.py:7060-7076`，所有 `block_stride>0` 的 tensor 共享同一 backing）。
2. 每个 group **独立**从 `block_pool.get_new_blocks()` 捞 block id（`single_type_kv_cache_manager.py:302`）。
3. prefix-cache 的 hash key 含 `group_id`（`BlockHashWithGroupId`，`kv_cache_utils.py:57`）。
4. ⇒ **任意时刻一个物理 block id 只被一个 group 持有**。于是 `(ps, slot)` 相同但属于不同组的层，虽 `offset` 相同，却写入不同物理块 → 永不写冲突。

> 这正是 `_bucket_layers_by_page_size` 注释"they have independent block tables so block-id namespaces never collide"的含义：不是 id 数值不同，而是**不同组永不同时持有同一物理块**，所以同 offset 的视图安全复用。
>
> 这里有个关键推论常被误解：**一个 group 持有一个 block 时，组内所有层都往这个 block 写数据**（同 group 的所有层共享同一个 block_table，`gpu_model_runner.py:2466` "make layers in the same group share the same metadata"）。所以"一个 block 被某 group 持有"≠"只填 1 个 slot"，而是填该 group 全部层各自的 slot。填充率分析见 §4.6。

### 4.6 所有 group 共用一个 block pool + 填充率分析

**共用一个 pool**：`KVCacheCoordinator.__init__` 只创建**一个** `BlockPool`（`kv_cache_coordinator.py:91`），5 个 group 的 `SingleTypeKVCacheManager` 全部拿到**同一个** `block_pool` 引用（`kv_cache_coordinator.py:112`）。`BlockPool` 内部只有一个 `free_block_queue`（`block_pool.py:182`），block id 空间统一 `0..num_blocks-1`，不按 group 分区。任意 group 调 `get_new_blocks` 都从同一队首捞，block id 不与 group 绑定。

**一个 block 被某 group 持有时填多少字节**：因为同 group 所有层共享 block_table，group G 持有 block N 时，G 的每一层都在 block N 的对应 region 写 1 个 slot。故 G 的填充字节数 = `Σ (G 在各 ps 的层数 × ps)`。

每个 region 的 slot 数 = 各 group 在该 ps 下层数的**最大值**（§4.1：37440 区 22 slot、8640 区 21 slot、1728 区 20 slot）。所以：

| group | 持有 1 block 时填写 | 计算（各 ps 层数 × ps） | 填充率 | 浪费 |
|-------|------------------|---------------------|--------|------|
| **G0** (62层: 21 `C4mla`+20 `C128mla`+21 `C4idx`) | 1,002,240 B | 21×37440 + 20×1728 + 21×8640 | **96.4%** | 3.6% |
| **G1** (22 `SWA_sw`) | 823,680 B | 22×37440 | **79.2%** | 20.8% |
| **G2** (21 `SWA_sw`) | 786,240 B | 21×37440 | **75.6%** | 24.4% |
| **G3** (42层: 21 `C4comp_sw`+21 `C4idxcomp_sw`) | 967,680 B | 21×37440 + 21×8640 | **93.1%** | 6.9% |
| **G4** (20 `C128comp_sw`) | 748,800 B | 20×37440 | **72.0%** | 28.0% |

> `bytes_per_block = 1,039,680 B`。填充率 = 填写字节 / 1,039,680。
>
> **G0 填充率最高（96.4%）**：它含 3 种 page_size 的层，持有 block 时在 3 个 region 都填 slot，几乎填满（37440 区用 21/22 slot，8640 区 21/21 填满，1728 区 20/20 填满）。
>
> **G4 填充率最低（72.0%）**：它只有 20 个 `C128comp_sw` 层，全在 37440 区（用 20/22 slot），8640 区和 1728 区**全空**——这部分就是浪费。

**浪费的来源**：每个 region 的 slot 数按"层数最多的 group"预留（37440 区按 G1 的 22 层预留），但持有 block 的 group 可能层数更少（如 G4 只有 20 层），多出的 slot 就空着。此外，持有 block 的 group 若不涉及某 region（如 G1/G2/G4 不涉及 8640、1728 区），整个 region 都空。

**为什么仍值得**：packed 布局用"单 block 部分填充"换"统一 pool 零碎片"。若每个 group 独立 pool，用量小的 group pool 闲置、用量大的 group 不够用——内部碎片。共用 pool 让总利用率 = 总需求 / 总容量，无内部碎片。对 DSV4 这种 6 类异构 cache 的模型，统一 pool 的简化（无需为每类 cache 单独规划容量）被认为值得单 block 的填充率损失。

**vLLM 的缓解措施**：
- `_approximate_gcd`（§3.2）选 d=22 让 `SWA_sw` 43 层分裂成 22+21，而非更大 padding——直接缩小 37440 区的 slot 数（若选 d=43，37440 区要 43 slot，浪费更大）。
- 第二层 page_size padding（§3.2a）把 compressor 的 32832 pad 到 37440，每 slot 额外浪费 4,608 B，但换来 compressor 能与 `SWA_sw`/mainMLA 共 37440 桶。
- 非 packed 路径（`get_kv_cache_config_from_groups` general case，`kv_cache_utils.py:1368`）要求所有 group page_size 相同——DSV4 有 3 种 page_size 走不了，故 packed 是唯一选择（除非 `--disable-hybrid-kv-cache-manager` 退化成全 full-attention，丢失 SWA/compressor 的 KV 节省）。


---

[← Part 2](02-concepts-and-grouping.md) · [目录](README.md) · [Part 4 →](04-runtime-and-apc.md)
