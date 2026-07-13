# Part 6 · APC 命中分化下的 Delta 段三角形计算可行性论证

> 对应新增章节 §10。完整目录见 [README.md](README.md)。
> 纯基于 `3rdparty/vllm` 源码，数值取自真实 HF `config.json` 与 vLLM 分组函数。

> **与 [swa-kv-offloading-analysis.md](swa-kv-offloading-analysis.md) 的关系**：那份文档给出 SWA 依赖锥的一般形式化（$R_\ell$ 集合递推、DRAM/HBM/计算三轴开销），用 DSV4 Pro（$L=61$）做量级估算；本文档把它落到 DSV4 Flash（$L=43$）的 APC delta 段场景。本文 §11.4 的 staircase 三角形在几何上就是那份文档 §3 的依赖锥（见 §11.4 等价说明），本文 §15.3 方向 A 的浅层边界带对应那份文档 §4 的 $|R_m|$ 公式。

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

> **DSV4 当前走全局 min，不是 per-group。** `find_longest_cache_hit_per_group`（`coordinator.py:742`）只在 `connector is not None and has_mamba_layers and HybridKVCacheCoordinator` 分支被调用（`scheduler.py:678-700`，调用在 L687）；DSV4 非 Mamba hybrid，走 `else` → `get_computed_blocks`（`scheduler.py:714`）→ 内部调全局 `find_longest_cache_hit`（`kv_cache_manager.py:202`、调用在 L229）。

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

`num_new_tokens` 包含 delta，故 delta 的每个 token 都进入这一步的 `slot_mapping`。compressor forward（`compressor.py:274-399`）对 `num_actual = slot_mapping.shape[0]` 个 token **无差别执行** `save_partial_states` + `compress_norm_rope_store`——**没有"该 token 的压缩条目已缓存就 skip"的逻辑**。结果：整个 `[B, A)` 被全量重算，但其中大部分并非必需。

关键区分——**compressor state 是滑窗的**（`sliding_window = coff·compress_ratio`，`compressor.py:142`；C4=8、C128=128），不是从 B 整段累积。state 累积只需末尾 window，**不需从 B 累积到 A**。SWA ring 同理（window=128）。

| Delta token 的产物 | 是否已缓存 | 当前行为 | 真正必需范围 | 冗余部分 |
|---|---|---|---|---|
| main MLA 压缩 KV 条目（G0） | **已缓存**（G0 命中到 A） | 整段 `[B,A)` 重算写回 | **0**（已缓存，skip 写回即可） | 全部冗余 |
| compressor state（G3/G4） | 末尾 window 外不保留 | 整段 `[B,A)` 跑 GEMM + 累积 | 仅 `[A−window, A)` 累积（C4=8、C128=128） | `[B, A−window)` 的 GEMM 冗余 |
| SWA ring 末端（G1/G2） | 末尾 window 外被覆盖 | 整段 `[B,A)` 做 KV insert | 仅 `[A−128, A)` 写 ring | `[B, A−128)` 冗余 |
| Q/K/V + FFN GEMM | — | 全层全 token 算 | 见 §11（三角形：只 `W_eff≈5500` 范围） | delta 左缘可三角形剪枝 |

> **"从 B 累积到 A"是当前全量重算的行为，不是必需**。因为 compressor state / SWA ring 都是滑窗，靠前 delta token 的 state 会被末尾覆盖、ring 会被覆盖、main MLA 条目已缓存——没有任何下游路径读它们（§11.3）。真正必需的只是靠近 A 的末尾 window，以及为产出该 window 正确隐藏态而向上游回溯的 `W_eff` 宽度（§11.4）。这正是 §11 三角形论证的立足点。

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
| SWA attention（同层/更深层） | 仅 window=128 内 | 每层都有 SWA（`swa_cache_layer` 无条件创建，`attention.py:290`），`window_size = config.sliding_window = 128`（`attention.py:179`）；窗口通过 top-k index gather + `clamp(min=0, max=window−1)`（`sparse_swa.py:230`）限定，只取末尾 128 个位置 |
| compressor state 累积 | 仅 window 内（C4=8、C128=128） | `sliding_window = coff*compress_ratio`（`compressor.py:142`） |

三条路径**最大窗口 = 128**（SWA 与 C128 compressor）。故 delta token `p` 的隐藏态，只影响位置 `p..p+127` 在同层的计算；逐层向上，影响范围以每层 ~127 token 收缩。**靠前 delta token 在深层无任何读路径 → 不需逐层穿透 → 隐藏态可截断。**

### 11.4 有效穿透宽度 W_eff

为在**最深层（L=43）**产出末尾 128 个 delta token 的正确隐藏态，需在第 0 层算多少 token？这是一个 SWA 依赖锥。形式化地，令 $R_\ell$ 为"为产出最深层末尾 window 的正确隐藏态，第 $\ell$ 层必须计算的位置集合"（记号与 `swa-kv-offloading-analysis.md` §3 一致），则递推关系为

$$
R_L = \{\text{末尾 } W \text{ 个位置}\}, \qquad R_{\ell-1} = \bigcup_{t \in R_\ell} \{t-W+1,\dots,t\}
$$

忽略序列起点截断，每向低层一层，集合宽度增加 $W-1$：

$$
|R_\ell| \approx W + (L-\ell)(W-1)
$$

第 0 层（输入侧）的宽度即有效穿透宽度：

$$
W_{eff} = |R_0| \approx W + (L-1)(W-1) = 128 + 42 \times 127 = 5462 \approx 5500
$$

- $L = num\_hidden\_layers = 43$，$W = sliding\_window = 128$
- $W-1 = 127$（SWA / C128 compressor 的最大单层感受野扩展）

> DSV4 每层都有 SWA（window=128），故每层感受野扩展恒为 127，与该层是 SWA-only / C4 / C128 无关。C4 层的 compressor window=8 更小，不构成约束上限。
>
> **与依赖锥的等价**：本文的 staircase 三角形在几何上就是 `swa-kv-offloading-analysis.md` §3 的 SWA 依赖锥——二者都是"顶层只需末尾 window、逐层向低层向更早 token 扩张"的同一形状。那份文档用"顶层需 1 个 token"的口径得到 $|R_0| = 1 + L(W-1)$；本文用"顶层需 $W$ 个 token"（修复尾部 window）的口径得到 $|R_0| = W + (L-1)(W-1)$。两者代数恒等（$1+L(W-1) \equiv W+(L-1)(W-1)$），对 $L=43,W=128$ 都等于 5462。区别在场景：分析文档的锥是"从零恢复单个 token 输出"的全重算；本文的锥是"修复 delta 段尾部 window"且 main MLA 已缓存 + SWA/compressor 滑窗，故锥只需覆盖 delta 段尾部、靠前 delta token 可整体剪枝（§11.5）。

### 11.5 层间 token 递减图（三角形）

设 delta `D = A − B`。纵轴层数（0=输入，43=输出），横轴 delta 内位置。深色 `█` = 该层实际计算的 token：

```
        ◄────────────── delta [B, A)，D = A−B ──────────────►
        B                                              A
   layer │ 0      10     20     30     40     50    55 (k token，相对 B)
   ──────┼──────────────────────────────────────────────────
   43 出 │                                       ░░░░░░░░░░░░░░░  算 128   (SWA ring 尾 + C128 comp 边界)
   42    │                                   ░░░░░░░░░░░░░░░░░░  算 255
   41    │                                 ░░░░░░░░░░░░░░░░░░░░  算 382
   40    │                               ░░░░░░░░░░░░░░░░░░░░░░  算 509
    ⋮    │                          (靠前 delta 深层无读路径，不计算)
   k     │              ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  算 128+(43-k)·127
    ⋮    │
    1    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  算 ~5335
    0 入 │  ██████████████████████████████████████████████████  算 ~5462  (≤ D)
   ──────┴──────────────────────────────────────────────────
          ◄────────── W_eff ≈ 5500 ──────────►(对齐到 A)
          (layer 0 算 min(D, W_eff) 个；每升一层左缘左移 ~127，layer 43 只剩末尾 128)

   ░ = 本层实际计算的 delta token；delta 左缘 [B, B+D−W_eff) 在所有层都不算
```

- **倒三角形**：layer 0 算最多（≤ D），每升一层左缘右移 ~127，layer 43 只算末尾 128。
- **三角形之外（delta 左缘 `[B, B+D−W_eff)`）**：当 `D > W_eff` 时，这部分 token 在**所有层都不算**——它们既不在任何感受野内，产物也都不需要（main MLA 已缓存、SWA/compressor 会被覆盖）。

### 11.6 与普通 prefill 矩形的对照

```
普通 prefill（要填满每层 cache，矩形）          Delta 段三角形（只修复尾部 cache）

  layer 43  ████████████████████████  D          layer 43                  ░░░░░  128
  layer 42  ████████████████████████  D          layer 42                ░░░░░░░  255
  layer 41  ████████████████████████  D          layer 41              ░░░░░░░░░  382
    ⋮       ████████████████████████              ⋮                  ░░░░░░░░░░░
  layer 1   ████████████████████████  D          layer 1   ░░░░░░░░░░░░░░░░░░░░░░  ~5335
  layer 0   ████████████████████████  D          layer 0  ██████████████████████████  ~5462
            ◄──────── D ────────►                          ◄──── W_eff ────►(≤D)

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

> **成本口径说明**：本节的"省比例"只计**计算轴**（token·层 hidden-state GEMM），对应 `swa-kv-offloading-analysis.md` 的"计算开销"轴。该文档把成本分三轴——DRAM 长期 KV 存储 / HBM 搬运 / 计算——本文的三角形只直接影响计算轴与 HBM 轴（少算 → 少读写），**不改变 DRAM 轴**（main MLA 压缩条目是否常驻由 APC 命中决定，与三角形无关）。本文 §13.1 的 skip-writeback 与 §15.5 方向 C 的收益主要落在 DRAM/写回轴（不重复写已缓存条目）而非计算轴——这也是方向 C"计算省量与基线持平、但写回更优"的口径依据（§15.5.2）。

---

## 13. 前提条件与边界

### 13.1 必须满足的前提

| 前提 | 当前状态 | 说明 |
|---|---|---|
| **main MLA 压缩条目 skip 写回** | ❌ 未实现 | compressor 对 delta 无差别重算（§10.3）。**三角形的前置条件**——否则每个 delta token 每层都要跑 compressor 产出压缩条目，隐藏态必须逐层穿透，三角形退化为矩形 |
| compressor state 滑窗语义 | ✅ 已是 | `sliding_window = coff*compress_ratio`（`compressor.py:142`），靠前 state 可弃 |
| SWA 滑窗语义 | ✅ 已是 | window=128，每层都有 SWA（`attention.py:290`），窗口经 top-k gather + `clamp` 限定（`sparse_swa.py:230`） |
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
| 单次计算（为复用算满浅层） | `(43)²·64 ≈ 118k` | `K·D + (43−K)²·64` |
| 单次计算（只算必需） | 同上 | `K·|R_K| + (43−K)²·64` |
| cache 复用 | ❌ 深层 cache 缺失 | ✅ 浅层 K 层完整（仅"算满"变体） |
| 后续命中重算 | 重算全 43 层三角形 | 只重算深层 43−K 层 |

> **浅层边界带宽度 $|R_K|$**（对应 `swa-kv-offloading-analysis.md` §4 的浅层 KV 边界带）：深层 $(43-K)$ 层三角形的底（在分界层 $K$）所需宽度为
>
> $$|R_K| \approx 1 + (43-K)(W-1) = 1 + (43-K)\times 127$$
>
> 浅层 $K$ 层若**只为喂给深层三角形**，只需算这条宽度为 $|R_K|$ 的边界带（`K·|R_K|`），不必算满全 delta 宽度 $D$。若要**保留完整浅层 cache 供后续命中复用**，才需算满 `K·D`。以 $K=21$、$D=5500$ 为例：$|R_K|=2795$，"只算必需" $=21\times2795+22^2\times64\approx 89.7k$（**低于**全三角形 118k），"为复用算满" $=21\times5500+22^2\times64\approx 146.5k$（高于全三角形）。

权衡：A 的"只算必需"变体单次计算低于全三角形，但浅层 cache 不完整、无复用；"算满"变体浅层 cache 完整可复用，但单次计算高于全三角形。命中频率高时"算满"更优，低时"只算必需"或全三角形更优。

### 15.4 方向 B：周期 checkpoint（省内存，不省计算）

每 P 层存一次完整 cache，中间层不存。但层间连续性要求 checkpoint 之间必须连续 forward 重算——等同激活重计算，**不省计算，只省 cache 内存**，对 delta 段的三角形收益无直接帮助。❌ 不推荐作为"更优"方向。

### 15.5 方向 C：C4 三角 + C128 全量保留（写回/内存收益，非计算省量）

利用 §15.1 的 34× 差异：C128 层 cr=128，delta 段只需写 D/128 条压缩条目（极少），而 C4 层要写 D/4 条（34 倍）。

**策略：对 C4 层做 skip-writeback + 三角形，对 C128 层全量写回（写回成本本就可忽略），让 C128 层 cache 保持完整以利后续命中复用。**

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
  C4 层：三角形 + skip   C128 层：矩形全量（写回 ≈ 免费）
```

**层间连续性的硬约束**：C128 层（layer 41）全量算，需要 layer 40（C4）的全部 delta token 隐藏态作输入；而 layer 40 若独立做三角形砍了左缘 → **矛盾**。故方向 C **不能逐层独立**做，必须按 **(C4, C128) 周期对**整体处理：每两层一个周期，周期内 C4+C128 一起算同样多 token，周期之间三角形收敛。

```
真正可行的方向 C：按周期对 (C4, C128) 一起做三角形
layer 42-41 (C4,C128)                    ████      ← 顶层周期对，只算尾部
layer 40-39 (C4,C128)                  ███████    ← 下一周期对，左缘右移 2·127=254
  ⋮
layer 2-1   (C4,C128+SW) ████████████████████████████    ← 底层周期对，算最多
```

#### 15.5.1 计算省量：与基线相当，并非"减半到 75%"

> ⚠️ **勘误**：早期版本曾认为"三角形只作用于 21 个 C4 层，`W_eff` 从 5500 减半到 2700，省比例 50%→75%"。该结论**不成立**——它假设 C4 层可独立三角化，违反 §15.2 的层间连续性（C128 在 C4 之上，需 C4 全部隐藏态）。

按周期对正确计算：每个周期对向上扩展 `2·127 = 254`（两层各扩 127），21 个周期对的有效穿透宽度为

```
W_eff(周期对) ≈ 128 + (21 − 1) · 254 ≈ 5208
```

与全三角形 `W_eff ≈ 5462` **几乎相同**（仅少一层），三角形面积也相近：

| 方案 | 三角形单元 | W_eff | 三角形面积(token·层) | vs 矩形 `D·L` 省比例 |
|---|---|---|---|---|
| 原全三角形（§11，全 43 层逐层） | 43 层 | ~5462 | `5462·43/2 ≈ 117k` | **~50%** |
| 方向 C（21 个周期对一起三角） | 21 对 | ~5208 | `5208·21 ≈ 109k`（每对 2 层同宽） | **~54%** |

→ **方向 C 的计算省量与原全三角形基本持平（~50% vs ~54%）**，并没有"75%"。周期对把高度从 43 层压到 21 对，但每对的扩展翻倍（254 vs 127），两者抵消，`W_eff` 不变。

#### 15.5.2 真正的收益：写回/内存，不是计算

方向 C 的价值不在计算省量，而在 **main MLA 压缩条目的写回成本与 cache 完整性**：

- **C128 层全量保留**：cr=128，全 delta 写回也只 `D/128·20 = 840` 条（D=5500），可忽略。于是 C128 层 cache **完整**，后续命中可复用——而原全三角形会破坏所有层的 cache 完整性。
- **C4 层才是写回主体**：`D/4·21 = 28875` 条（D=5500）。三角形 + skip-writeback 把这部分省下来（仅写三角形内的 ~`W_eff/4·21` 条）。
- 即便不做三角形，仅 **C4 层 skip-writeback**（已缓存条目不重算写回）就能省掉 C4 层的 main MLA 投影冗余——这才是 DSV4 周期结构带来的、可独立于三角形落地的收益。

| 方案 | 计算省量 | 写回成本（main MLA 条目） | C128 cache 完整性 |
|---|---|---|---|
| 原全三角形（§11） | ~50% | 全部省（所有层三角） | ❌ 破坏 |
| 方向 C（周期对三角 + C4 skip） | ~54% | C4 省、C128 本就可忽略 | ✅ 完整（可复用） |

> **方向 C 的正确定位**：在计算省量上与原全三角形持平，但在 **cache 完整性/写回** 上更优——C128 层近乎免费地保持完整，后续 APC 命中可复用；C4 层的 skip-writeback 是独立于三角形也成立的省写回手段。它不是"省 75% 的更优变体"，而是"同等计算省量 + 更好 cache 复用性"的变体。

### 15.6 方向对比

| 方向 | 单次计算 | cache 复用 | 实现难度 | 推荐度 |
|---|---|---|---|---|
| 原 staircase（全 43 层三角，§11） | 省 50%@D=5500 | ❌ 全破坏 | 高 | 基线 |
| A 浅层 + 深层三角 | 算满：略高于基线；只算必需：低于基线 | 算满 ✅ 浅层 / 只算必需 ❌ | 中 | 命中频繁用算满 |
| B 周期 checkpoint | 不省计算 | 省 cache 内存 | 中 | ❌ 不解决计算 |
| C 周期对三角 + C4 skip-writeback | 省 ~54%@D=5500（与基线持平） | ✅ C128 完整 | 高 | 计算持平、cache 更优 |
| 松绑全局 min（§13.2） | 消除 delta | ✅ 完整 | 中 | ★ 根治 |

### 15.7 小结

1. **DSV4 已是天然周期存储**（C4/C128 交替，§15.1），C128 层写回成本比 C4 低 34×——原 staircase 没利用的结构差异。
2. **层间连续性**（§15.2）限制了"逐层独立选密集/稀疏"，只能按 (C4, C128) 周期对整体处理。
3. **方向 C 的计算省量并非"75%"**：按周期对正确计算 `W_eff ≈ 5208`（非 2700），与原全三角形（5462）几乎相同，省量 ~54% vs ~50%，基本持平。周期对把高度从 43 层压到 21 对，但每对扩展翻倍（254 vs 127），两者抵消。
4. **方向 C 的真正价值在写回/内存与 cache 复用**：C128 层 cr=128，全量写回也仅 840 条，近乎免费地保持 cache 完整以供后续命中复用；C4 层（28875 条主体）用三角形 + skip-writeback 省下。其中 **C4 层 skip-writeback 独立于三角形也成立**——即便不做三角形，让 compressor 对"压缩条目已缓存"的 delta token 跳过写回，就能省掉 C4 层 main MLA 投影冗余。
5. **仍不如松绑全局 min**（直接消除 delta）。最优组合：**松绑 min 消除大部分 delta + 残余 delta 用 skip-writeback（+ 可选三角形）兜底**。

---

## 16. 结论

1. **场景真实存在**：DSV4 的全局 min APC 协议（`find_longest_cache_hit`，`coordinator.py:630`）会让末尾窗口型 group（SWA/compressor）把 main MLA 的更长命中拽低，产生 delta 段 `[B, A)`（§10）。
2. **delta 段当前被全量重算，含冗余**：main MLA 压缩条目已缓存却被重算写回（compressor 无 skip，`compressor.py:274-399`）（§10.3）。
3. **三角形在 delta 段成立**：因 main MLA 条目可 skip 写回 + SWA/compressor 都是滑窗（max window=128），靠前 delta token 在深层无任何读路径，隐藏态不需逐层穿透。有效穿透宽度 `W_eff ≈ L·128 ≈ 5500`（§11.3-11.5）。
4. **省量随 D 增长**：`D ≈ 5500` 省约一半，`D` 更大省更多，`D` 很小不省（§12）。
5. **前置条件**：须先实现 main MLA 压缩条目 skip-writeback（当前未实现）；否则三角形退化为矩形（§13.1）。
6. **更优解**：松绑全局 min 协议（让 SWA/compressor 不拖低命中）可直接消除 delta，收益更彻底、实现更简单（§13.2）。
7. **更优 staircase 变体（方向 C）的再评估**：DSV4 的 C4/C128 周期结构使 C128 层写回成本比 C4 低 34×（28875 vs 840 条）。但按 (C4, C128) 周期对正确计算，`W_eff ≈ 5208`（非早期误判的 2700），计算省量 ~54%——与原全三角形（~50%）基本持平，**并非"省 75%"**。方向 C 的真正价值在写回/内存与 cache 复用：C128 层近乎免费地保持 cache 完整供后续命中复用；C4 层的 skip-writeback（已缓存条目不重算写回）独立于三角形也成立。最优组合：松绑 min 消除大部分 delta + 残余 delta 用 skip-writeback（+ 可选周期对三角形）兜底（§15）。

---

[← Part 5](05-appendix.md) · [目录](README.md)
