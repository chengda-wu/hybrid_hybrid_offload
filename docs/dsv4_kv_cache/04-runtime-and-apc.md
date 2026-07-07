# Part 4 · 运行时与 APC 例子

> 对应原文档章节，完整目录见 [README.md](README.md)。

[← Part 3](03-packed-layout.md) · [目录](README.md) · [Part 5 →](05-appendix.md)

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

**`_sw` 类型的窗口外跳过（三层机制）**：影响所有 `_sw` 类型（`SWA_sw`/`C4comp_sw`/`C128comp_sw`/`C4idxcomp_sw`）。G0（MLA, `C4mla`/`C128mla`/`C4idx` 为 FullAttention）不跳过，所有 token 都分配真实 block。详见 §6.1。

### 6.1 SWA skip 机制：skip token 与 skip block

`SlidingWindowMLASpec`（`_sw` 类型）的窗口外 KV 不需保留。vLLM 用 `get_num_skipped_tokens` 决定跳过多少，**而它是否触发，取决于"已经算完多少 token"，与 prompt 长度无关**——这正是 prefill 与 decode 行为分化的根因。

**核心公式**（`SlidingWindowManager.get_num_skipped_tokens`，`single_type_kv_cache_manager.py:838,864`）：

```
n = total_computed_tokens                          # 本步调度前已算完的 token 数
num_skipped_tokens(n) = max(0, n - sliding_window + 1)
num_skipped_blocks    = num_skipped_tokens // block_size
```

`n` 的来源（`allocate_slots` L355-361, L400-404）：`n = request.num_computed_tokens + num_new_computed_tokens(prefix命中) + num_external_computed_tokens`。**关键是它只算"本步之前已算完的 token"，本步要新算的 `num_new_tokens` 不计入**——所以 prefill（还没算过任何 token）和 decode（已经算完整个 prompt）取到完全不同的 `n`。

#### prefill vs decode：skip 何时触发

| 阶段 | `n`（已算完） | `num_skipped_tokens` | 后果 |
|------|--------------|---------------------|------|
| **prefill 无命中**（请求刚到） | 0 | `max(0, 0-sw+1) = 0` | **不 skip**：整个 prompt 全分配真实 block，0 个 null。即使 prompt=512 > sw=128 也不释放——"还没算过任何 token"，skip 公式无从触发 |
| **prefill 有命中 H**（APC 命中 H token） | H | `max(0, H-sw+1)` | `n=H`（**不是 0**），分配侧算出 skip>0，但此时 block table 为空（命中块尚未写入）→ `remove_skipped_blocks` 早退不释放。窗口外 null 由**命中侧**预填（见下表及 §6.1(2)） |
| **decode step 1**（prefill 算完后） | 512 | `max(0,512-128+1) = 385` | **开始 skip**：窗口外的 block 在 `remove_skipped_blocks` 里被释放成 null（block table 非空，真正释放） |
| **decode step k** | 512 + (k-1) | 每步 +1 | skip_tokens 单调增；skip_blocks 只在跨 block 边界时才新增 null |

> **prefill + 命中的时序**（决定性）：`allocate_slots` 内部先 L400 `remove_skipped_blocks(n=H)`，**后** L430 `allocate_new_computed_blocks` 才把命中块写入 `req_to_blocks`。故 `remove_skipped_blocks` 面对的 block table 为空，`_remove_blocks_in_range`（`single_type_kv_cache_manager.py:491`）`if request_id not in self.req_to_blocks: return` 早退——分配侧此步**一个 block 都不释放**。窗口外的 null 占位全部来自命中侧 `find_longest_cache_hit` 的预填。§7.3 的 B 即此场景（H=256），其 G1 table `[0,0,5,6,212-215]` 前 2 个 null 是命中侧预填，非分配侧释放。

实测对比（A = 512 token prompt，无 prefix 命中）：

| 阶段 | `n` | skip_tokens | G1 (bs64,sw128) | G3 (bs4,sw8) | G4 (bs8,sw128) |
|------|-----|-------------|-----------------|--------------|----------------|
| prefill 无命中 | 0 | 0 | real=8, null=0 | real=128, null=0 | real=64, null=0 |
| decode step 1 | 512 | 385 | real=3, null=6 | real=3, null=126 | real=17, null=48 |

> decode step 1 的 real 数 = 窗口内残留 block + 1 个新 decode block：G1 `cdiv(128,64)=2` 残留 +1 = 3；G3 `cdiv(8,4)=2` +1 = 3；G4 `cdiv(128,8)=16` +1 = 17。null 数 = prompt block 数 − 残留 block 数（G1 `8−2=6`、G3 `128−2=126`、G4 `64−16=48`）。
>
> **结论：SWA 的 KV 内存节省是 decode 稳态属性，不是 prefill 属性。** prefill 阶段窗口外的 KV 暂时全量保留——无命中时全分配真实 block（n=0 不释放），有命中时分配侧也释放不了（block table 空，n=H 仅驱动命中侧预填 null）。要等到 decode 第一步 `remove_skipped_blocks` 才真正释放窗口外 block。

下面三层都围绕"`n` = 已算完 token 数"展开，但每层在 prefill/decode 下的表现不同。

#### (1) 分配侧 —— skip token 决定 skip block（运行时持续释放）

每次 `allocate_slots`（`kv_cache_manager.py:400`）开头先调 `remove_skipped_blocks`（`single_type_kv_cache_manager.py:507`）：
- 算出 `num_skipped_blocks`，把 block table **前** `num_skipped_blocks` 个真实 block 替换成 `null_block`（id=0）并归还 free list（`_remove_blocks_in_range:480`）。
- block table **长度不变**：被释放的 block 用 `null_block` 占位，保持 token→block 索引关系；attention kernel 靠位置算窗口，不读 null 标记。
- `allocate_new_blocks`（base class，`:279`）随后只为新 token 追加 block：`num_new = cdiv(num_tokens, bs) - len(req_blocks)`，因 `len(req_blocks)` 含 null 占位，不会因 skip 多分配。

**分配侧真正释放 block 的前提是 block table 非空**：`_remove_blocks_in_range`（`:491`）`if request_id not in self.req_to_blocks: return` 早退。故 prefill 无命中时 `n=0 → skip_blocks=0`（不释放）；prefill 有命中时 `n=H` 算出 skip>0 但 block table 为空（命中块尚未写入）也早退（不释放）；只有 decode 时 block table 非空且 `n>sw` 才真正释放。这是"分配侧只有 decode 才 skip"的来源。

#### (2) 命中侧 —— 连续 block 命中（APC 查找）

`SlidingWindowManager.find_longest_cache_hit`（`single_type_kv_cache_manager.py:688`）**不调用 `get_num_skipped_tokens`**，而是用独立的连续命中逻辑：

1. 先把整个候选命中区间 `[0, max_num_blocks)` **全部预填 `null_block`**（`:718`）。
2. 需连续 `cdiv(sliding_window - 1, block_size)` 个真实 block 命中才算 hit（`_contiguous_blocks_for_hit:675,678`，即 `cdiv(sw-1, bs)`）——窗口必须完整覆盖这么多 block 才能复用。
3. **从右往左**扫描 hashmap（`:725`），找到连续命中后 trim 掉右侧多余 null，**左侧的 null 全部保留**作为窗口外占位。

命中块序列 = 若干 `null_block`（窗口外）+ 窗口内真实 block。null 数 = `命中块数 − cdiv(sw-1, bs)`，详见 §7.3 推导表。

**此侧只对"有 prefix 命中"的请求生效**（如 §7.3 的 B，命中 256 token）。全新 prefill 无命中走不到这里。

> **命中侧 vs 分配侧：公式不同，但 block 对齐时数值相等。** 命中侧用 `cdiv(sw-1, bs)` 连续命中（不调 `get_num_skipped_tokens`），分配侧用 `get_num_skipped_tokens(n)//bs = max(0,n-sw+1)//bs`。两者是不同代码路径，但命中总按 block 对齐（`H = k·bs`），此时 `H//bs − cdiv(sw-1,bs) = (H−sw+1)//bs`——命中侧 null 数恰等于分配侧 skip_blocks。故 §7.3 的 null 数推导两种算法都自洽。两侧的 null 语义一致：分配侧"释放窗口外 block 成 null"，命中侧"预填窗口外 null"，都是"窗口外用 `null_block` 占位"。prefill+命中时只有命中侧的 null 生效（分配侧因 block table 空而 no-op）。

#### (3) 计算侧 —— triton kernel 只 gather 窗口内 KV（prefill 与 decode 都生效）

attention forward 时，`_compute_swa_indices_and_lens_kernel`（`sparse_swa.py:612`）对每个 query position `pos` 算：

```
start_pos = max(pos - window_size + 1, 0)     # sparse_swa.py:644
```

只在该 `[start_pos, pos]` 窗口内 gather KV。`decode_swa_indices`（`sparse_swa.py:339,412`）预计算每个 decode token 的窗口索引下标。

**这里 prefill 与 decode 都用窗口**——计算侧的 skip 按 query **位置** `pos` 算，与 `num_computed_tokens` 无关：prefill 第 200 个 token 只 attend `[73,200]`，decode 第 512 个 token 只 attend `[385,512]`。block table 里 `null_block` 占位的位置即便被间接索引到，也落在窗口外（mask 掉），不参与计算。

> **易混点：计算侧与分配侧的 skip 触发条件不同**。
> - 计算侧（triton kernel）按 query **位置** `pos` 算窗口，**prefill 和 decode 都 skip**（只 gather 窗口内 KV）。
> - 分配侧（`remove_skipped_blocks`）按 `n=num_computed_tokens` 算 skip，**只有 decode 才 skip**（释放窗口外 block）。
>
> 所以 prefill 时计算上已不用窗口外 KV，但内存上仍全量保留——直到 decode 才释放。这就是"SWA 省 KV 是 decode 属性、不是 prefill 属性"的根因。

#### 三层的关系

| 层面 | 作用对象 | skip 触发条件 | 公式 | prefill 无命中 | prefill 有命中 | decode |
|------|---------|--------------|------|---------------|---------------|--------|
| 分配侧 `remove_skipped_blocks` | block table（释放窗口外 block） | `n>sw` **且 block table 非空** | `max(0,n-sw+1)//bs` | 不 skip（n=0） | 算出 skip 但 table 空→早退不释放 | **真正释放** |
| 命中侧 `find_longest_cache_hit` | APC 命中块序列 | 有 prefix 命中 | `cdiv(sw-1,bs)` 连续命中 | 不触发（无命中） | **预填窗口外 null** | 不触发（decode 不查 APC） |
| 计算侧 triton kernel | attention 的 KV gather | 按 query 位置 `pos`，恒生效 | `max(pos-sw+1,0)` | **skip**（按位置） | **skip**（按位置） | **skip**（按位置） |

三层分别管**内存释放**、**prefix 命中**、**计算 gather**，用各自独立的公式（分配侧 `max(0,n-sw+1)//bs`、命中侧 `cdiv(sw-1,bs)`、计算侧 `max(pos-sw+1,0)`），block 对齐时分配侧与命中侧数值相等。prefill 无命中时三层中只有计算侧 skip；prefill 有命中时计算侧 skip + 命中侧预填 null（分配侧 no-op）；decode 时计算侧 skip + 分配侧真正释放。null_block 是分配侧/命中侧的 block table 占位手段，计算侧不依赖它。

---

## 7. 完整例子：分配 / APC 命中 / 释放

> 本节所有 block id 由真实 `KVCacheManager` 实测得出（用 `dsv4_layout.py` 构造 spec + `generate_scheduler_kv_cache_config` + 真实 `KVCacheManager` 跑 A/B 两个请求），非手算。

### 7.1 设定

- `hash_block_size = 4`，`scheduler_block_size = 256`，`num_blocks = 1000`
- pool 初始全空闲；**block 0 保留为 `null_block`**（`block_pool.py:191`），free list 从 block 1 开始发放
- **Request A**：512 token prompt（token ids 100..611），全新（无命中）
- **Request B**：512 token prompt，**前 256 token 与 A 完全相同**（100..355），后 256 token 不同（2000..2255）

每组对 N 个 token 需要的 block 数（`ceil(N / block_size)`）：

| Group | block_size | sliding_window | 256 token | 512 token |
|-------|-----------|----------------|-----------|-----------|
| G0 (`C4mla`+`C128mla`+`C4idx`, FullAttention) | 256 | — | 1 | 2 |
| G1 (`SWA_sw`-A) | 64 | 128 | 4 | 8 |
| G2 (`SWA_sw`-B) | 64 | 128 | 4 | 8 |
| G3 (`C4comp_sw`+`C4idxcomp_sw`) | 4 | 8 | 64 | 128 |
| G4 (`C128comp_sw`) | 8 | 128 | 32 | 64 |
| **合计** | | | **105** | **210** |

> 注：G1-G4 都是 `SlidingWindowMLASpec`（`_sw`），用 `SlidingWindowManager`，会跳过窗口外的 token（见 §7.3）。G0 是 `FullAttentionManager`，不跳过。

> **上表是 prefill 视角的 block 数**（`ceil(N / block_size)`，纯按 block_size 切分）。prefill 时 `num_computed_tokens = 0` → `get_num_skipped_tokens(0) = 0`，窗口外跳过尚未生效，512 token 全分配真实 block（0 个 null）。窗口外 block 在 decode 阶段窗口移动后才被释放成 null。完整机制与 prefill/decode 实测对比见 **§6.1**。

### 7.2 Phase 1：A 到达，全分配（无命中）

A 的 `get_computed_blocks` → hashmap 空，`num_computed = 0`。

`allocate_slots(num_new_tokens=512)`：5 个 group 各自从共享 free list 顺序捞**互不重叠**的 block id（`single_type_kv_cache_manager.py:302`，每组独立 `get_new_blocks`）。实测 A 拿到：

| Group | A 拿到的 block id | 数量 |
|-------|------------------|------|
| G0 (`C4mla`+`C128mla`+`C4idx`) | 1, 2 | 2 |
| G1 (`SWA_sw`-A) | 3..10 | 8 |
| G2 (`SWA_sw`-B) | 11..18 | 8 |
| G3 (`C4comp_sw`+`C4idxcomp_sw`) | 19..146 | 128 |
| G4 (`C128comp_sw`) | 147..210 | 64 |

> block id 从 1 起（block 0 是 `null_block`）。A 共用 210 个 block，free list 剩 789 个（1000 − 1 null − 210）。

A forward 后 `cache_blocks` 把满 block 的 `(block_hash, group_id) → block_id` 插入全局 hashmap（key 含 group_id，5 组互不干扰）。

### 7.3 Phase 2：B 到达，APC 命中共享前缀

B 前 256 token 与 A 相同 → block hash 链前 `256/4 = 64` 个 hash 与 A 一致。`get_computed_blocks` → `find_longest_cache_hit`（`coordinator.py:630`）逐组查 hashmap，命中长度按 `scheduler_block_size=256` 对齐（`single_type_kv_cache_manager.py:606`）。

**关键：`_sw` group 的命中含 null_block**。命中侧（`find_longest_cache_hit`）预填 null 后从右往左找连续 `cdiv(sw-1, bs)` 个真实命中，左侧 null 保留作窗口外占位（机制详见 §6.1(2)）。故 `_sw` group 的命中块序列 = 若干 `null_block` + 窗口内真实 block；G0 是 FullAttention，全真实。

实测 B 命中（`num_computed = 256`）：

| Group | sw | 命中块数 | null_block 数 | 真实命中 block id | 说明 |
|-------|-----|---------|-------------|-----------------|------|
| G0 (`C4mla`+`C128mla`+`C4idx`) | — | 1 | 0 | **1** | FullAttention，全真实 |
| G1 (`SWA_sw`-A) | 128 | 4 | 2 | **5, 6** | 前 2 个 null（窗口外），后 2 个真实 |
| G2 (`SWA_sw`-B) | 128 | 4 | 2 | **13, 14** | 同 G1 |
| G3 (`C4comp_sw`+`C4idxcomp_sw`) | 8 | 64 | 62 | **81, 82** | sw=8 极小，62 个 null，仅 2 个真实 |
| G4 (`C128comp_sw`) | 128 | 32 | 16 | **163..178** | 16 个 null + 16 个真实 |

> **null 数怎么来的**：命中区间 256 token，命中块数 = `256 / block_size`；命中侧需连续 `cdiv(sw-1, block_size)` 个真实 block，故真实块数 = 该值，null 数 = 命中块数 − 真实块数。
> - G1/G2：`cdiv(127, 64)=2` → 真实 2，null = 4−2 = 2 ✓
> - G3：`cdiv(7, 4)=2` → 真实 2，null = 64−2 = 62 ✓
> - G4：`cdiv(127, 8)=16` → 真实 16，null = 32−16 = 16 ✓
> - G0（FullAttention）：无窗口，全真实。

> **null_block 不占容量、不被共享**。真实命中的 block 才 ref_cnt++。所以 G3 虽"命中 64 块"，实际只共享 A 的 block 81、82 两个；前 62 个是 null 占位。`find_longest_cache_hit` 的定长收敛保证 5 组命中 token 数一致（256）。

`allocate_slots(num_new_tokens=256, num_new_computed_tokens=256, new_computed_blocks=bB)` 把命中块加入 B 的 block table（ref_cnt `1 → 2`，零拷贝），并为新 token 分配 block。B 的完整 block table（实测）：

| Group | B 的 block table（null + 共享 + 新分配） | 新分配 block id | 新分配数 |
|-------|----------------------------------------|----------------|---------|
| G0 | [1 (shared), 211 (new)] | 211 | 1 |
| G1 | [0,0 (null), 5,6 (shared), 212,213,214,215 (new)] | 212..215 | 4 |
| G2 | [0,0, 13,14, 216,217,218,219] | 216..219 | 4 |
| G3 | [0×62, 81,82, 220..283] | 220..283 | 64 |
| G4 | [0×16, 163..178, 284..315] | 284..315 | 32 |

> B 新分配共 105 个 block（1+4+4+64+32），free list 从 789 → 684。注意 B 的 block table 里 null_block 占位不计入新分配——`_sw` group 的窗口跳过让 B 不必为前 256 token 的窗口外部分分配真实 block，这是 `_sw` 类型省 KV 内存的关键。

### 7.4 Phase 3：B 完成 → free(B)

每组 manager 对 B 的 block 减 ref_cnt（`coordinator.py:285`），**null_block 不计入**：

- **共享 block**（G0 的 1、G1 的 5-6、G2 的 13-14、G3 的 81-82、G4 的 163-178）：ref_cnt `2 → 1`，**不释放**（A 仍引用）。
- **B 独占 block**（G0 的 211、G1 的 212-215、G2 的 216-219、G3 的 220-283、G4 的 284-315）：ref_cnt `1 → 0`，归还 free list，标记 evictable；hash 暂留（evict 时才删）。
- free list: 684 → 789。

### 7.5 Phase 4：A 完成 → free(A)

- **共享 block**：ref_cnt `1 → 0`，归还 free list，evict 时删 hash。
- **A 独占 block**（G0 的 2、G1 的 3-4,7-10、G2 的 11-12,15-18、G3 的 19-80,83-146、G4 的 147-162,179-210）：ref_cnt `1 → 0`，归还。
- free list: 789 → 999（仅 null_block 占 1）。

### 7.6 关键观察

1. **block 0 是 null_block**：`BlockPool` 保留 block 0 作 null_block（`block_pool.py:191`），真实分配从 block 1 起。null_block 用于 `_sw` 窗口外占位等，不计 ref_cnt、不占容量。
2. **跨组 block id 永不重叠**：A 的 5 组分别拿到 `1-2 / 3-10 / 11-18 / 19-146 / 147-210`，因为同一 free list 顺序消费。这是 packed 布局安全性的运行时保障（§4.5 invariant #2）。
3. **`_sw` group 命中含 null_block**：`SlidingWindowManager` 跳过窗口外 token（`get_num_skipped_tokens`，`single_type_kv_cache_manager.py:864`），命中块序列 = null 占位 + 窗口内真实 block。sw 越小（G3 sw=8），null 占比越高。FullAttention（G0）不跳过，全真实。
4. **APC 按 group 独立命中**：`(hash, group_id)` 复合 key 让 5 组各自查 hashmap；`find_longest_cache_hit` 定长收敛保证 5 组命中 token 数一致（256，按 `scheduler_block_size` 对齐）。
5. **共享 = ref_cnt++，非拷贝**：B 命中 A 的真实 block 时零拷贝，仅 bump 引用计数（实测 G0 block 1、G3 block 81/82 ref_cnt 均 `1→2`）。
6. **释放按引用计数**：`free` 不立即删 hash，evict 时才删——这就是为何 B 完成后其独占 block 仍可能被后续 C 请求命中。

---

[← Part 3](03-packed-layout.md) · [目录](README.md) · [Part 5 →](05-appendix.md)
