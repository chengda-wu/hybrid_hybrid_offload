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

**`_sw` 类型的窗口外跳过（两层机制）**：
- **分配侧**：`SlidingWindowManager.get_num_skipped_tokens(n) = max(0, n - sliding_window + 1)`（`single_type_kv_cache_manager.py:864`）。窗口外的 token 不分配真实 block，block table 里用 `null_block`（id=0）占位。这省下窗口外的 KV 内存（见 §7.3 实测）。影响所有 `_sw` 类型（`SWA_sw`/`C4comp_sw`/`C128comp_sw`/`C4idxcomp_sw`）。
- **计算侧**：attention 时靠 `decode_swa_indices` 只索引窗口内 slot（`sparse_swa.py:612` triton kernel），null_block 位置不参与计算。
- G0(MLA, `C4mla`/`C128mla`/`C4idx` 为 FullAttention) 不跳过，所有 token 都分配真实 block。

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

> **上表是 prefill 视角的 block 数**（`ceil(N / block_size)`，纯按 block_size 切分）。prefill 时 `num_computed_tokens = 0`，`get_num_skipped_tokens(0) = 0`（`single_type_kv_cache_manager.py:864`），窗口外跳过尚未生效，故 512 token 全分配——G1/G2 各 8 个真实 block、G3 128 个、G4 64 个，**0 个 null**。窗口外 block 的释放在 **decode 阶段窗口移动后**才发生：每步 decode `remove_skipped_blocks`（`kv_cache_manager.py:400`）把 `get_num_skipped_tokens(computed) // block_size` 个窗口外 block 替换成 `null_block` 并归还 free list。实测 A prefill 512 后第 1 步 decode（computed=513）：G1 `real 8→3`、G3 `real 128→3`、G4 `real 64→17`，其余变 null。故 SWA 的 KV 内存节省体现在 decode 稳态，不在 prefill 当下。

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

**关键：`_sw` group 的窗口跳过机制**。`SlidingWindowManager.get_num_skipped_tokens(n) = max(0, n - sliding_window + 1)`（`single_type_kv_cache_manager.py:864`）是**分配侧**公式（`allocate_slots` 时窗口外 token 不分配真实 block）。**命中侧**（`find_longest_cache_hit`，`single_type_kv_cache_manager.py:688`）逻辑等价但实现不同：先把整个命中区间预填 `null_block`，再**从右往左**找连续 `cdiv(sliding_window - 1, block_size)` 个命中的真实 block，找到后 trim 掉右侧多余的 null，**左侧的 null 全部保留**作为窗口外占位。两种机制殊途同归：命中块序列 = 若干 `null_block` + 窗口内的真实 block。G0 是 FullAttention，不跳过，全部真实。

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
