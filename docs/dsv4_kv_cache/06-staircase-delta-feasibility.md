# Part 6 · APC 命中分化下的 Delta 段三角形计算可行性论证

> 对应新增章节 §10。完整目录见 [README.md](README.md)。
> 纯基于 `3rdparty/vllm` 源码，数值取自真实 HF `config.json` 与 vLLM 分组函数。

[← Part 5](05-appendix.md) · [目录](README.md)

---

## 10. 问题设定：APC 命中分化与 Delta 段

### 10.1 场景

DSV4 的 5 个 KV cache group（§3）block_size 与窗口各异：

| Group | 类型 | block_size | sliding_window | 覆盖语义 |
|-------|------|-----------|----------------|----------|
| G0 | `C4mla`+`C128mla`+`C4idx`（FullAttention） | 256 | — | **整条序列**（每 `compress_ratio` token 一条压缩条目） |
| G1/G2 | `SWA_sw` | 64 | 128 | 末尾 ring（window=128） |
| G3 | `C4comp_sw`+`C4idxcomp_sw` | 4 | **8** | 末尾 compressor state（window=8） |
| G4 | `C128comp_sw` | 8 | **128** | 末尾 compressor state（window=128） |

> 窗口来源：SWA = `config.sliding_window = 128`（`attention.py:179`）；compressor = `coff * compress_ratio`，`coff = 1 + (compress_ratio==4)`（`compressor.py:141-142`）→ C4 window=8、C128 window=128。

APC 查找时，`find_longest_cache_hit`（`coordinator.py:630`）对 5 个 group 做**定长收敛**：任一 group 命中更短 → 全局 `hit_length` 收缩 → 重新检查所有 group，迭代到 fixed point。最终 `hit_length = min over all groups`，对齐到 `scheduler_block_size = 256`。

> **DSV4 当前走全局 min，不是 per-group。** `find_longest_cache_hit_per_group`（`coordinator.py:742`）只在 `has_mamba_layers` 分支被调用（`scheduler.py:678-691`）；DSV4 非 Mamba hybrid，走 `else` → `get_computed_blocks` → 全局 `find_longest_cache_hit`（`scheduler.py:712-715`、`kv_cache_manager.py:228`）。

### 10.2 命中分化 → Delta 段

由于各 group 的写入/驱逐历史不同，命中率会分化。设：

- **G0（main MLA 压缩 KV）命中到 A**：压缩条目覆盖 `[0, A)`，整条前缀都在。
- **G1–G4（SWA ring + compressor state）只命中到 B < A**：被全局 min 拽下来。
- 全局 `hit_length = B`，scheduler 据此 `num_computed_tokens = B`（`scheduler.py:799`）。
- 尾部 `num_new_tokens = num_tokens − B` 进入 forward，其中 `[B, A)` 这段就是 **delta 段**：`D = A − B`。

```
token 位置   0 ─────────── B ─────────── A ─────── N
              ◄── 命中前缀 ──►◄── delta ──►◄新尾部►
              [0, B)          [B, A)       [A, N)
              全 group 命中    G0 命中、     无任何命中
                               G1-G4 未命中
              APC 复用         被迫回退重算   正常 prefill
              (不进 forward)   (进 forward)   (进 forward)
```

### 10.3 Delta 段当前被全量重算（含冗余）

`num_new_tokens` 包含 delta，故 delta 的每个 token 都进入这一步的 `slot_mapping`。compressor forward（`compressor.py:274-399`）对 `num_actual = slot_mapping.shape[0]` 个 token **无差别执行** `save_partial_states` + `compress_norm_rope_store`——**没有"该 token 的压缩条目已缓存就 skip"的逻辑**。

| Delta token 的产物 | 是否已缓存 | 当前行为 | 是否冗余 |
|---|---|---|---|
| main MLA 压缩 KV 条目（G0） | **已缓存**（G0 命中到 A） | **重算写回** | **冗余** |
| compressor state（G3/G4） | 未缓存（命中仅到 B） | 从 B 累积到 A | 必需 |
| SWA ring 末端（G1/G2） | 未缓存 | 写末端 | 必需 |
| Q/K/V + FFN GEMM | — | 全层全 token 算 | 见 §10.4 |

---

## 11. 核心论证：Delta 段可做三角形（staircase）计算

### 11.1 为什么普通 prefill 不能做三角形

普通 prefill 的目标是**把每个 token 在每一层的 KV/state 全部填进 cache**，供后续 decode 复用。这要求每个 token 的隐藏态逐层穿透到最深层——"深层不算早期 token"会破坏 cache 完整性。故普通 prefill 是**矩形**（每层算同样多 token），三角形不成立（见 §11.6 对照）。

### 11.2 Delta 段满足三角形的前提

Delta 段重算的**唯一目的**是修复 cache，让 decode 在位置 A 能接得上。decode 在位置 A 实际需要的 delta 产物：

| 产物 | decode 需要的范围 | 窗口 |
|---|---|---|
| main MLA 压缩 KV | **不需要重算**（已缓存，skip 写回） | — |
| SWA ring | 仅末尾 window=128 个 token 的 SWA KV | 128 |
| compressor state（C4） | 仅末尾 window=8 的 state | 8 |
| compressor state（C128） | 仅末尾 window=128 的 state | 128 |

→ **只有靠近 A 的尾部 ~128 个 delta token 需要正确隐藏态**（写 SWA ring + C128 compressor 边界 state）。delta 靠前的 token，其 SWA KV 会被 ring 覆盖、compressor state 会被覆盖、main MLA 条目已缓存——**没有任何下游路径会读它们**。

### 11.3 三条读路径都是局部的（关键）

delta token `p` 在第 `l` 层的隐藏态，是否被更深层/后续读到？

| 读路径 | 是否读靠前 delta token | 依据 |
|---|---|---|
| main MLA 压缩 attention（更深层） | **不读** | 读的是**已缓存的压缩条目**，不依赖重算的隐藏态（前提：skip 写回） |
| SWA attention（同层/更深层） | 仅 window=128 内 | `start_pos = max(pos−128+1, 0)`（`sparse_swa.py:644`） |
| compressor state 累积 | 仅 window 内（C4=8、C128=128） | `sliding_window = coff*compress_ratio`（`compressor.py:142`） |

三条路径**最大窗口 = 128**（SWA 与 C128 compressor）。故 delta token `p` 的隐藏态，只影响位置 `p..p+127` 在同层的计算；逐层向上，影响范围以每层 ~127 token 收缩。**靠前 delta token 在深层无任何读路径 → 不需逐层穿透 → 隐藏态可截断。**

### 11.4 有效穿透宽度 W_eff

为在**最深层（L=43）**产出末尾 128 个 delta token 的正确隐藏态，需在第 0 层算多少 token？感受野随深度增长：

```
W_eff ≈ 128 + (L − 1) · (window − 1)
      ≈ 128 + 42 · 127
      ≈ 5462  ≈ 5500
```

- `L = num_hidden_layers = 43`
- `window − 1 = 127`（SWA / C128 compressor 的最大单层感受野扩展）

> DSV4 每层都有 SWA（window=128），故每层感受野扩展恒为 127，与该层是 SWA-only / C4 / C128 无关。C4 层的 compressor window=8 更小，不构成约束上限。

### 11.5 层间 token 递减图（三角形）

设 delta `D = A − B`。纵轴层数（0=输入，43=输出），横轴 delta 内位置。深色 `█` = 该层实际计算的 token：

```
                 ◄──────────── delta [B, A)，D = A−B ────────────►
                 B                                              A
  layer 43 (出)                                  │████████████████│  算 128  (SWA ring 尾 + C128 comp 边界)
  layer 42                                       │██████████████████│  算 255
  layer 41                                       │████████████████████│  算 382
  layer 40                                       │██████████████████████│  算 509
    ⋮        (靠前 delta 在深层无读路径，不计算)   ⋮                        ⋮
  layer k                                        │██████████████████████████│  算 128+(43−k)·127
    ⋮                                            ⋮                        ⋮
  layer 1                                        │████████████████████████████│  算 ~5335
  layer 0 (入)         │████████████████████████████████████████████████████████│  算 ~5462  (≤ D)
                      ◄── W_eff ≈ 5500 ──►
                      (layer 0 算 min(D, W_eff) 个；左缘随层数下降每层左移 ~127)
```

- **倒三角形**：layer 0 算最多（≤ D），每升一层左缘右移 ~127，layer 43 只算末尾 128。
- **三角形之外（delta 左缘 `[B, B+D−W_eff)`）**：当 `D > W_eff` 时，这部分 token 在**所有层都不算**——它们既不在任何感受野内，产物也都不需要（main MLA 已缓存、SWA/compressor 会被覆盖）。

### 11.6 与普通 prefill 矩形的对照

```
普通 prefill（要填满每层 cache，矩形）          Delta 段三角形（只修复尾部 cache）

  layer 43  ████████████████████████  D          layer 43                  ████  128
  layer 42  ████████████████████████  D          layer 42                ████████  255
  layer 41  ████████████████████████  D          layer 41              ████████████  382
    ⋮       ████████████████████████              ⋮                  ████████████████
  layer 1   ████████████████████████  D          layer 1      ████████████████████████████
  layer 0   ████████████████████████  D          layer 0  ████████████████████████████████████
            ◄──────── D ────────►                          ◄── W_eff ──►(≤D)

  总量 = D · L（矩形面积）                        总量 ≈ W_eff · L / 2（三角形面积，D≥W_eff 时）
```

普通 prefill 每层算 D 个（矩形 `D·L`）；delta 段每层递减（三角形 `≈ W_eff·L/2`）。差异的根因：普通 prefill 每个 token 每层都要写 KV（cache 完整性），delta 段只需写尾部 window（局部修复）。

---

## 12. 省的比例

设 `L = 43`，`W_eff ≈ 5500`，delta `D = A − B`。

| D 的规模 | layer 0 算 | 总 token·层 | vs 矩形 `D·L` | 省比例 |
|---|---|---|---|---|
| `D ≪ W_eff`（如 D=128） | 全 D | `≈ D·L`（矩形，cascade 被 B 截断） | `D·L` | **≈ 0**（无三角形空间） |
| `D ≈ W_eff`（≈5500） | 全 D | `≈ W_eff·L/2` | `D·L` | **≈ 50%** |
| `D ≫ W_eff`（如 11000） | 仅 W_eff | `≈ W_eff·L/2`（固定） | `D·L` | **≈ 75%** |
| `D → ∞` | 仅 W_eff | `≈ W_eff·L/2`（固定） | `D·L` | **→ 100%**（超额部分全免） |

> 数值（`L=43`，`W_eff=5500`）：
> - 矩形 `D·L`：D=5500 → 236 500；D=11000 → 473 000
> - 三角形 `W_eff·L/2`：≈ 118 000（固定）
> - D=5500 省 (236500−118000)/236500 ≈ 50%；D=11000 省 ≈ 75%。

**结论**：省量随 `D/W_eff` 增长。`D ≈ 5500`（约 `43×128`）时省约一半；`D` 更大省更多；`D` 很小（<几百）几乎不省。命分化越严重（D 越大），三角形收益越高。

---

## 13. 前提条件与边界

### 13.1 必须满足的前提

| 前提 | 当前状态 | 说明 |
|---|---|---|
| **main MLA 压缩条目 skip 写回** | ❌ 未实现 | compressor 对 delta 无差别重算（§10.3）。**三角形的前置条件**——否则每个 delta token 每层都要跑 compressor 产出压缩条目，隐藏态必须逐层穿透，三角形退化为矩形 |
| compressor state 滑窗语义 | ✅ 已是 | `sliding_window = coff*compress_ratio`（`compressor.py:142`），靠前 state 可弃 |
| SWA 滑窗语义 | ✅ 已是 | window=128，`start_pos=max(pos−128+1,0)`（`sparse_swa.py:644`） |
| delta 段 main MLA 条目确实已缓存 | ✅（场景内） | G0 命中到 A，`[B,A)` 条目存在且正确（hash 匹配） |
| `D ≳ W_eff ≈ 5500` 才有可观收益 | 取决于命中分化 | D 小则无三角形空间 |

> **三角形 ≠ 免实现**。当前代码即便 delta 的 main MLA 条目已缓存，仍重算写回（冗余）。三角形要成立，**先要实现 skip-writeback**：让 compressor 对"该 token 压缩条目已缓存"的 delta token 跳过 `save_partial_states` + `compress_norm_rope_store`。这一步本身就能省掉 delta 段的 main MLA 投影冗余（即使不做三角形也有收益）；加上三角形再省隐藏态逐层 GEMM。

### 13.2 替代/更优方案：松绑全局 min 协议

三角形的收益依赖 `D` 大（命中分化严重）。更根治的办法是**让 D 不要发生**：

- DSV4 当前用全局 `find_longest_cache_hit`（强制 5 group 取 min，`coordinator.py:630`）。
- G1–G4 是**末尾窗口型** cache（SWA ring / compressor state），其"命中"语义本应是"末尾 window 已在 ring 里"，而非"前缀块对齐命中"。把它们的命中从全局 min 协议摘出，或改用 `find_longest_cache_hit_per_group`（`coordinator.py:742`，当前仅 Mamba hybrid 用），让全局 `hit_length` 只由 G0（main MLA，整条覆盖）决定 → `hit_length = A`，delta 消失，无需重算。

| 方案 | 作用 | 收益 | 实现难度 |
|---|---|---|---|
| **松绑全局 min**（让 SWA/compressor 不拖低 hit_length） | 消除 delta（D→0） | 100% 省 delta 段 | 中（改命中协议/对齐语义） |
| **skip-writeback**（已缓存压缩条目不重算） | 省 delta 的 main MLA 投影冗余 | 部分（即使无三角形） | 低-中（compressor 加 skip 判断） |
| **三角形 staircase**（在 skip-writeback 基础上） | 省 delta 隐藏态逐层 GEMM | D≈5500 省 ~50%，D 更大更多 | 高（改调度+forward 计算图） |

> 三者不互斥：松绑 min 消除大部分 delta；残余 delta（松绑后仍可能存在的、因 G0 自身部分未命中产生的回退）用 skip-writeback + 三角形兜底。

### 13.3 不适用场景

- **普通 prefill（无 APC 命中分化）**：每层每 token 都要写 cache，矩形，三角形不成立（§11.6）。
- **`D < window`**：cascade 被 B 截断，整段 delta 在所有层都要算，无三角形空间。
- **decode**：本就是单 token，不涉及。

---

## 15. 更优变体：利用 C4/C128 周期结构的差异化三角形

§11–§13 的 staircase 把 43 层一视同仁。但 DSV4 的层结构本身就有天然的"周期存储"，可让三角形更优。本节分析"浅层/深层差异化存储"这类方向，并给出最有 promise 的变体。

### 15.1 DSV4 已是天然周期存储

实测 HF `config.json`（`deepseek-ai/DeepSeek-V4-Flash`）：43 层的 `compress_ratios` 为

```
[0, 0, 4, 128, 4, 128, 4, 128, ..., 4, 128, 4]   (43 层)
   SW SW  C4  C128 C4  C128               C128 C4
```

- 2 层 SWA-only（cr=0，无 main MLA 压缩条目，仅 SWA ring）
- 21 层 C4（cr=4，每 4 token 一条压缩条目）
- 20 层 C128（cr=128，每 128 token 一条压缩条目）
- **C4 与 C128 严格交替**，周期 = 2 层

这不是可调策略，而是模型架构自带。但它决定了各层写回 main MLA 压缩条目的成本天差地别：

| 层类型 | 层数 | 每 token 条目 | D=5500 delta 总条目 |
|---|---|---|---|
| C4 (cr=4) | 21 | 1/4 | **28 875** |
| C128 (cr=128) | 20 | 1/128 | **840** |
| SWA-only | 2 | 0 | 0 |

→ **C128 层写回成本比 C4 层低 34×**（28875 / 840）。这是差异化策略的物理基础。

```
layer 0  1  2  3  4  5  ... 40 41 42
       SW SW C4 C128 C4 C128    C4 C128 C4
              └─┬─┘ └─┬─┘
              密集    稀疏
              cr=4    cr=128
              每4tok  每128tok
              一条     一条
```

### 15.2 硬约束：层间隐藏态连续性

`layer l` 的输入 = `layer l−1` 的输出隐藏态。三角形一旦在某层跳过 token p（不算其隐藏态），**所有更深层都没有 p 的输入**，无法计算 p。故三角形必须**整体连续**——左缘从 layer 0 到 layer 42 单调右移，不能"这层三角形、下层又完整"。

```
可行的三角形（整体连续）          不可行（层间断裂）

layer 0  ████████                  layer 0  ████████
layer 1   ███████                  layer 1   ███████      ← 砍了左缘
layer 2    ██████                  layer 2  ████████      ← 想恢复完整？没有左缘 token 的输入隐藏态
layer 3     █████                   layer 3    █████
  (左缘单调右移 ✓)                   (层间不连续 ✗，layer 2 缺 layer 1 砍掉的 token 输入)
```

→ "逐层独立选密集/稀疏"**不可行**。能选的只是"从第几层开始三角形"或"按周期对整体处理"。

### 15.3 方向 A：浅层完整 + 深层三角形

前 K 层矩形（完整 cache），后 43−K 层三角形。

```
                 ◄──────── delta [B, A), D ────────►
layer 42 (深)                              ████      ← 三角形顶部
layer 41                                 ███████
  ⋮                                      ███████
layer K+1                          ████████████
layer K (完整)    ████████████████████████████████    ← 矩形，存完整 cache
  ⋮              ████████████████████████████████
layer 1 (完整)    ████████████████████████████████
layer 0 (完整)    ████████████████████████████████
                 ◄── 完整 K 层 ──►◄── 三角形 43−K 层 ──►
```

| 项 | 全三角形（K=0） | 方向 A（K>0） |
|---|---|---|
| 单次计算 | `(43)²·64 ≈ 118k` | `K·D + (43−K)²·64` |
| cache 复用 | ❌ 深层 cache 缺失 | ✅ 浅层 K 层完整 |
| 后续命中重算 | 重算全 43 层三角形 | 只重算深层 43−K 层 |

权衡：A 的单次计算略高于全三角形，但浅层 cache 完整可被后续命中复用。命中频率高时 A 更优，低时全三角形更优。

### 15.4 方向 B：周期 checkpoint（省内存，不省计算）

每 P 层存一次完整 cache，中间层不存。但层间连续性要求 checkpoint 之间必须连续 forward 重算——等同激活重计算，**不省计算，只省 cache 内存**，对 delta 段的三角形收益无直接帮助。❌ 不推荐作为"更优"方向。

### 15.5 方向 C：C4 三角 + C128 免费（★ 最优变体）

利用 §15.1 的 34× 差异：C128 层 cr=128，delta 段只需写 D/128 条压缩条目（极少），而 C4 层要写 D/4 条（34 倍）。

**策略：对 C4 层做 skip-writeback + 三角形，对 C128 层直接全量写回（成本本就极低）。**

```
                 ◄──── delta [B, A), D ────────►
layer 42  C4     ░░░░                ████        ← C4 三角形 + skip-writeback
layer 41  C128   ████████████████████████████    ← C128 全量写回（只 D/128 条，几乎免费）
layer 40  C4     ░░░                ███████      ← C4 三角形 + skip
layer 39  C128   ████████████████████████████    ← C128 全量
layer 38  C4     ░░                ████████
  ⋮
layer 2   C4     ░░░░░░░░░░░░░░░░█████████████    ← C4 三角形 + skip
layer 1   SW     (无 main MLA 条目，仅 SWA ring)
layer 0   SW     (同上)

  ░ = skip-writeback（条目已缓存，不重算写回）   █ = 实际计算
  C4 层：三角形 + skip   C128 层：矩形全量（成本 ≈ 0）
```

**层间连续性的微妙处**：C128 层（layer 41）全量算，需要 layer 40（C4）的全部 delta token 隐藏态作输入；而 layer 40 是三角形，砍了左缘 → **矛盾**。故方向 C 不能逐层做，必须按 **(C4, C128) 周期对**整体处理：每两层一个周期，周期内 C4+C128 一起算，周期之间三角形收敛。

```
真正可行的方向 C：按周期对 (C4, C128) 一起做三角形
layer 42-41 (C4,C128)                    ████      ← 顶层周期，只算尾部
layer 40-39 (C4,C128)                  ███████    ← 上一层周期，左缘右移 2·127
  ⋮
layer 2-1   (C4,C128+SW) ████████████████████████████    ← 底层周期，算最多
```

收益：C128 层即便全量算，写回成本 = D/128 ≈ 可忽略；C4 层是写回 + 计算主体，三角形 + skip 集中在 C4。**等效于"只对 21 个 C4 层做三角形，C128 层近似免费"**——三角形高度从 43 降到 21，`W_eff` 减半：

| 方案 | 三角形层数 | W_eff | D=5500 省比例 |
|---|---|---|---|
| 原全三角形（§11，全 43 层） | 43 | ~5500 | ~50% |
| 方向 C（C4 三角 + C128 免费） | 21 | ~2700 | **~75%** |

> `W_eff` 减半后，D=5500 变成 2×W_eff，省比例从 50% 跳到 ~75%。C128 层 cache 保持完整（部分复用性）。

### 15.6 方向对比

| 方向 | 单次计算 | cache 复用 | 实现难度 | 推荐度 |
|---|---|---|---|---|
| 原 staircase（全 43 层三角，§11） | 省 50%@D=5500 | ❌ 破坏 | 高 | 基线 |
| A 浅层完整 + 深层三角 | 略高于基线 | ✅ 浅层 | 中 | 命中频繁时 |
| B 周期 checkpoint | 不省计算 | 省 cache 内存 | 中 | ❌ 不解决计算 |
| **C C4 三角 + C128 免费** | **省 75%@D=5500** | 部分（C128 完整） | 高 | **★ 最优变体** |
| 松绑全局 min（§13.2） | 消除 delta | ✅ 完整 | 中 | ★ 根治 |

### 15.7 小结

1. **DSV4 已是天然周期存储**（C4/C128 交替，§15.1），C128 层写回成本比 C4 低 34×——原 staircase 没利用的结构优势。
2. **层间连续性**（§15.2）限制了"逐层独立选密集/稀疏"，但按 (C4, C128) 周期对处理可行。
3. **方向 C 是真正更优的 staircase 变体**：`W_eff` 从 5500 降到 2700（三角形只作用于 21 个 C4 层），D=5500 省比例 50% → 75%，且 C128 层 cache 保持完整。
4. **仍不如松绑全局 min**（直接消除 delta）。最优组合：**松绑 min 消除大部分 delta + 残余 delta 用方向 C 兜底**。

---

## 16. 结论

1. **场景真实存在**：DSV4 的全局 min APC 协议（`find_longest_cache_hit`，`coordinator.py:630`）会让末尾窗口型 group（SWA/compressor）把 main MLA 的更长命中拽低，产生 delta 段 `[B, A)`（§10）。
2. **delta 段当前被全量重算，含冗余**：main MLA 压缩条目已缓存却被重算写回（compressor 无 skip，`compressor.py:274-399`）（§10.3）。
3. **三角形在 delta 段成立**：因 main MLA 条目可 skip 写回 + SWA/compressor 都是滑窗（max window=128），靠前 delta token 在深层无任何读路径，隐藏态不需逐层穿透。有效穿透宽度 `W_eff ≈ L·128 ≈ 5500`（§11.3-11.5）。
4. **省量随 D 增长**：`D ≈ 5500` 省约一半，`D` 更大省更多，`D` 很小不省（§12）。
5. **前置条件**：须先实现 main MLA 压缩条目 skip-writeback（当前未实现）；否则三角形退化为矩形（§13.1）。
6. **更优解**：松绑全局 min 协议（让 SWA/compressor 不拖低命中）可直接消除 delta，收益更彻底、实现更简单（§13.2）。
7. **更优 staircase 变体（方向 C）**：DSV4 的 C4/C128 周期结构使 C128 层写回成本比 C4 低 34×。按 (C4, C128) 周期对做三角形——C4 层三角 + skip、C128 层近似免费全量——`W_eff` 从 5500 降到 2700，D=5500 省比例从 50% 提升到 ~75%，且 C128 层 cache 保持完整。最优组合：松绑 min 消除大部分 delta + 残余 delta 用方向 C 兜底（§15）。

---

[← Part 5](05-appendix.md) · [目录](README.md)
