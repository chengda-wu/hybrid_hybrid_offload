# 计划：在 vLLM 中实现 DSV4 skip-writeback 与 staircase 重算

> 承接 §10–§16（`06-staircase-delta-feasibility.md`）。两个技术点工程分层、有硬依赖：
>
> - **Feature 1 skip-writeback**：闸掉 `[B,A)` 冗余的 G0 写回。小改动、默认关、逐位一致，**先行**。
> - **Feature 2 staircase**：把 delta 段重算从全矩形 `D×L` 改成逐层递减三角形，省 ~50%
>   hidden-state GEMM。**硬依赖 Feature 1**：不 skip-writeback 时 main MLA 全序列读路径使 delta
>   hidden state 全层穿透、三角形退化为矩形；skip-writeback 消除该路径后依赖锥 W_eff 成立
>   （见 [可行性与前置条件](#staircase-可行性与前置条件)）。主要工程障碍在 MegaMoE 残差融合
>   kernel，**第二阶段**。
>
> Feature 2 复用 Feature 1 的 per-request A 透传，且 `enable_staircase` 隐含
> `enable_skip_writeback`。

## 背景

DSV4 有 5 个 KV cache group，APC 下**命中分化**：主 MLA 压缩组（G0，全序列）命中到 `A`，
SWA 与 compressor 组（滑窗）只命中到 `B < A`。全局最小共识（`find_longest_cache_hit`，
`kv_cache_coordinator.py:630`）取 `num_computed_tokens = B`，故 delta 段 `[B,A)` 的 G0 条目
**已缓存**却仍进 forward。记 `D = A - B`。

关键事实（均已核验，附行号）：

- `A`、`B` 均 **block 对齐**（G0 block_size=256）。G0 的 spec 是 `MLAAttentionSpec`
  （`attention.py:610`），它是 `FullAttentionSpec` 的**子类**（`kv_cache_interface.py:363`），
  故 `find_longest_cache_hit` 的 Full attention 路径对 G0 适用：block 对齐 trim
  （`kv_cache_coordinator.py:687-689`）、downward-closed（L683-690）、排第一（L592-593）。返回
  `hit_length = B`；迭代中的 `longest_hit_length = A`（L719、L737）**未返回**，需暴露。
- compressor `forward`（`compressor.py:274-399`）对 delta token 无条件跑两 kernel，**二者数据流不同**：
  - `save_partial_states`（L318）：**per-token scatter 写**——token `p` 把自己的 `kv/score` 写进
    `state_cache[slot_p]` 一个槽（`save_partial_states.py:85-89`，一个 program 一个 token，零跨
    token 读）。它**本身不是滑窗**；滑窗在它的读者那里。
  - `compress_norm_rope_store`（L375）：`state_cache` 的**唯一读者**，gather 一个窗口
    `[p-W, p]`，`W=(1+overlap)·cr`（C4=8、C128=128，`fused_compress_quant_cache.py:169-171`），
    并把压缩结果写回 G0（`[B,A)` 冗余）。
  - 故 `state_cache[p]@L` 活不活，取决于有没有某个**跑 compress 的边界 token** `q∈[p, p+W)` 的窗口
    覆盖 `p`。这是 skip-writeback 能在 compressor-state 侧促成三角的根因（见前置条件）。
- `compress_norm_rope_store` 第一参数 `slot_mapping_ptr` **只**用于 `slot_id<0` 早退
  （`fused_compress_quant_cache.py:159`）。
- `CommonAttentionMetadata` 已有 `positions` 字段（`backend.py:436`），gate 构建可直接用。

---

## Feature 1：skip-writeback

### 机制

引入独立 gate 张量 `compress_gate_slot_mapping`（不改 kernel、不改 attention）：

- = 未压缩 slot 副本，把每个 `position < A` 的 token 置 `-1`。
- 作为 `compress_norm_rope_store` 的第一参数 `slot_mapping` 传入（`compressor.py:380`）。
- `save_partial_states`（`compressor.py:324`）仍用原 slot，**不变**。

`compress_norm_rope_store` 的第一参数 `slot_mapping_ptr` **只**用于 `slot_id<0` 早退（核验：
`fused_compress_quant_cache.py` sparse_attn kernel 内 `slot_mapping_ptr` 仅出现于 L46-47 早退；
state gather 走 `block_table` L174-179；G0 store 走**独立第二门** `kv_slot_mapping_ptr`
L108-112 = `k_cache_metadata.slot_mapping`）。故 gate 在第一门早退会**跳过整条
`compress_norm_rope_store` 流水**（compress 聚合 + RMSNorm + RoPE + 量化 + G0 store），不只是
G0 写——多省了 compress 聚合 GEMM。

结果：`compress_norm_rope_store` 对 `[B,A)` token 整条早退；`save_partial_states` 照跑（独立
kernel，吃原 slot）；attention 及其 compressed slot 不动。

> PAD 所有 `pos<A` 而非仅边界：kernel 本就对非边界早退（`fused_compress_quant_cache.py:163`），
> PAD 非边界是 no-op，但省去重算 `compress_ratio`。真正受影响的只有 `[B,A)` 边界 token。
>
> 用独立 gate 而非共享 compressed slot：后者被 sparse-MLA attention 消费
> （`sparse_mla.py:383` `is_valid_token = slot_mapping>=0`），PAD 它会破坏 attention。

### 步骤

**S1 配置 flag** — `vllm/config/cache.py` 加 `enable_skip_writeback: bool = False`（靠近
`mamba_cache_mode`，约 L134）。

**S2 暴露 A** — `vllm/v1/core/kv_cache_coordinator.py::find_longest_cache_hit`（L630-740）：
`longest_hit_length`（A）已在 L719 算出。改返回签名为
`(blocks_per_group, hit_length, longest_hit_length)`，或在 `get_computed_blocks` 侧经
`self.num_uncached_common_prefix_tokens`（L737 = A−B）反推 A = B + 该值。前者更直白，取前者。

**S3 挂到 Request** — `vllm/v1/core/kv_cache_manager.py::get_computed_blocks`（L202-242）：
捕获 A，flag 开启时设 `request.g0_hit_length = A`。**未发生 preemption 时，A 在本次运行周期内
固定**：`get_computed_blocks` 只在 `request.num_computed_tokens == 0` 时调用一次（
`scheduler.py:676` 守卫、L714 调用），running 请求与后续 prefill chunk 都不重算。首 chunk 调用时
`max_cache_hit_length = request.num_tokens - 1`（全 prompt 长度，`kv_cache_manager.py:227`），故一次
即得全 prompt 的 G0 命中 A，跨 chunk 不变。scheduler.py 无逻辑改动（DSV4 走常规路径，
`scheduler.py:712`）。

> **待澄清：A 的“request 生命周期内固定”是否跨 preemption。** `_preempt_request` 会释放请求
> blocks、把 `num_computed_tokens` 清零并将同一逻辑请求放回 waiting queue；恢复调度时会重新执行
> prefix-cache lookup。这里的“request 生命周期”可能仅指一次 admission/block-residency 周期，也
> 可能指包含 preemption/resume 的完整逻辑请求。实现前须明确：preemption 是否结束当前 A 的有效
> 周期，以及 resume lookup 是否无条件用新结果覆盖 `g0_hit_length`（包括新 A=0）。若能保证覆盖，
> 无需规定 preemption 时单独清零；若不能，则须补充失效处理，避免沿用已释放 blocks 对应的旧 A。

**S4 per-request A 上 GPU** —
- `vllm/v1/worker/input_batch.py`：加 `g0_cached_prefix_len_cpu` 数组（仿
  `num_computed_tokens_cpu`），默认 0。
- `vllm/v1/worker/gpu_model_runner.py::_prepare_inputs`（约 L1982 建 per-request CPU 数组、
  L2345 建 `CommonAttentionMetadata`）：读 `request.g0_hit_length` 入 CPU 数组 → 拷 GPU 张量
  `self.g0_cached_prefix_len[:num_reqs]` → 挂 `cm_base.g0_cached_prefix_len`（per-request，
  长 `num_reqs_padded`，0=不跳）。
- `vllm/v1/attention/backend.py::CommonAttentionMetadata`（L394）：加字段
  `g0_cached_prefix_len: torch.Tensor | None = None`。

**S5 compressor 构建 gate** — `vllm/models/deepseek_v4/compressor.py`：
- `CompressorMetadata`（L77-83）：加 `compress_gate_slot_mapping: torch.Tensor | None = None`。
- `CompressorMetadataBuilder.build`（L101-118）：flag 开且有 A 时构建
  ```python
  gate = cm.slot_mapping.clone()
  A_per_token = cm.g0_cached_prefix_len[cm.token_to_req_indices]
  gate[cm.positions < A_per_token] = -1
  metadata.compress_gate_slot_mapping = gate
  ```
  （`token_to_req_indices` 已在 L110-112 构建；`positions` 用 `cm.positions`。）flag 关 / 无 A
  → 留 `None`。
- `Compressor.forward`（L375-399）：`compress_norm_rope_store_fn` 的 `slot_mapping` 实参
  （L380）改为 `state_metadata.compress_gate_slot_mapping or state_metadata.slot_mapping`。
  `save_partial_states`（L324）不变。

**待改文件**：`vllm/config/cache.py`、`vllm/v1/core/kv_cache_coordinator.py`、
`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/worker/input_batch.py`、
`vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/attention/backend.py`、
`vllm/models/deepseek_v4/compressor.py`。

### 验证（Feature 1）

- **一致性（主门槛）**：flag 关无路径变化（gate `None`→原 slot）。构造前缀缓存分化 prefill
  （第一段填满 G0，再发分化第二请求），断言 flag 开/关 logits 逐位一致。
- **单测**：`A>B` 小模型；断言 `[B,A)` 边界 token 的 G0 slot 字节不变（写计数 0），而
  `[B,A)` state cache 仍被 `save_partial_states` 写入。
- **待补 preemption 用例**：待上述生命周期语义明确后，覆盖 resume lookup 得到不同 A（尤其 A=0）
  的情况，验证实现符合选定语义且不会误用旧命中边界。
- **省量**：Nsight 数 `compress_norm_rope_store` launch；flag 开时 `[B,A)` 边界 launch 消失。
  D=5500、cr=128 测 delta 步 GPU 时间。

---

## staircase 可行性与前置条件

### 前置条件：skip-writeback 消除 main MLA 全序列读路径

staircase 的三角形成立**依赖 Feature 1 已生效**。原因在 main MLA 这条读路径（核验
`flashmla.py:237-304`）：

- 单层 `DeepseekV4DecoderLayer.forward`（`model.py:909`）内，先跑 compressor
  `compress_norm_rope_store` **用本层 delta hidden state 写回 `[B,A)` 的 G0 压缩条目**，紧接着
  main MLA attention `dequantize_and_gather_k_cache`（`flashmla.py:296-304`）按
  `seq_lens // compress_ratio`（L299，seq_len=A）gather G0 中 `[0, A/cr)` **全部压缩条目，含
  本步刚写回的 `[B,A)` 条目**。
- 故层 l+1 的 main MLA attention 读层 l compressor 写回的 delta G0 → 层 l+1 依赖层 l 的 delta
  hidden state → **每个 delta token 每层都要算，全层穿透，三角形退化为矩形**（与 §11.3 第一条
  读路径一致，该条明确标注"前提：skip 写回"）。

**skip-writeback 生效后**：compressor 不重写 `[B,A)` G0，更深层 main MLA 读**旧缓存**（前缀缓存
命中到 A 的条目）→ 不依赖本步 delta hidden state → 该全序列读路径消失。这正是不 skip-writeback
时 staircase 不成立、skip 后才成立的根因。

> **唯一的矩形制造者是 main MLA**。compressor-state 侧（`save_partial_states` + compress 读者）
> 本身是**窗口**依赖（W=128），不是矩形源——见下条 liveness 论证。

### compressor-state 侧的 liveness 论证（锥界自洽）

`save_partial_states` 是 per-token scatter（写自己一格），**不是滑窗**；`state_cache[p]@L` 活不活
取决于有没有**跑 compress 的边界 token** `q∈[p, p+W)` 覆盖它。开 skip-writeback 后，`position<A`
的 token 的 compress 被闸，故读者只剩 `q≥A` 的 compress：

- `p ∈ [B, A-W)`：读者 `q ∈ [p, p+W) ⊂ [B,A)`，**全被闸** → `state_cache[p]@L` 死 →
  `save_partial_states[p]@L` 不必跑 → `p` 不必算。这些 token 落在顶层锥外，staircase 物理丢弃它们
  与"无读者"自洽。
- `p ∈ [A-W, A)`：读者 `q ∈ [A, p+W) ⊂ [A, A+W)`，`q≥A` **未被闸** → `state_cache[p]@L` 活 →
  `save_partial_states[p]@L` 要跑 → 需要 `hidden_states[p]@L`。这正好是顶层锥内的 `[A-W, A)`
  （宽 W=128）。

故顶层只需算 `[A-W, A)`，与依赖推导的锥顶一致——skip-writeback 在 compressor-state 侧**不是**
矩形源，而是把 `[B,A-W)` 变死、让锥顶收窄到 `[A-W,A)` 的配合条件。不开 skip 时 `[B,A)` 边界
compress 照跑 → `state_cache[B,A)@L` 全被读 → `save_partial_states[B,A)@L` 每层都跑（但仍是窗口
依赖，矩形仍只来自 main MLA）。

### 剩余读路径均为滑窗 → 依赖锥 W_eff 成立

skip-writeback 后，delta token `p` 的 hidden state 只剩两条读路径（核验 `sparse_swa.py:228-230`、
`fused_compress_quant_cache.py:169-172`）：

| 读路径 | 窗口 | 依据 |
|---|---|---|
| SWA attention（同层/更深层） | 128 | `window_size = config.sliding_window = 128`（`attention.py:179`），gather_lens `= D + min(B, W-1)`（`sparse_swa.py:228-230`） |
| compressor state 聚合 | C4=8、C128=128 | `start = position - (1+OVERLAP)·cr + 1`（`fused_compress_quant_cache.py:169`），`overlap = cr==4`（`compressor.py:216`） |

两路径最大窗口 W=128，构成依赖锥的**驱动源**：
- **SWA**（窗口 128，分化组）：`[B,A)` 的 SWA K/V 未缓存，须从 `hidden_states@L-1` 重算 →
  `hidden_states[p]@L` 依赖 `hidden_states[p-128, p]@L-1`，**每层扩张 128**。这是锥扩张的主因。
- **compress 读者**（窗口 W=128）：`compress@L` 读 `state_cache[p-W,p]@L`（同层，由
  `save_partial_states@L` 从 `hidden_states[p-W,p]@L` 写）→ 同样每层扩张 W。

**main MLA 不再驱动锥**：skip-writeback 后它读**缓存 G0**（命中到 A），token `p`@L 仅依赖
`hidden_states[p]@L-1`（作 Q），K/V 来自 cache，**无扩张**——这正是 skip-writeback 把矩形降回锥的
落点。

`save_partial_states`（`save_partial_states.py:68-90`）one program per token，只写自己那格，零跨
token 读，不制造额外依赖；其 liveness 由 compress 读者决定（见上节 liveness 论证）。

### 依赖推导

锥由最宽窗口 W=128 决定。令 R_l 为"为在顶层（L=42，output 端）产出末尾 W 个 delta token 的正确
hidden state，第 l 层必须计算的位置集合"，递推 `R_{l-1} = ∪_{t∈R_l} {t-W+1,…,t}`
（与 §11.4 / `swa-kv-offloading-analysis.md` §3 一致）。层 l（input 端 l=0，output 端 l=42）的
token 下界 `p >= end - W·(42 - l + 1)`：

- input 端 l=0：`[max(B, end-W_eff), end)`，宽 ≈ W_eff = 128+42·127 = 5462。
- output 端 l=42：`[end-128, end)`，宽 128。

> **锥右边界 = `end`（设计决策，2026-07）**：锥覆盖本步调度的全部 token
> `[B, end)`（`end = B + num_scheduled`），即 delta `[B, A)` **与**新 token `[A, end)` 一起并入
> 锥。理由：SWA 跨层依赖对所有 token 均匀（新 token `p@L` 同样只依赖 `[p-128,p]@L-1`），故新 token
> 也服从锥，无需单独全量算。这比原 plan 的"锥仅 `[lo_l, A)` + 新 token 全宽"形态简单——单一残差流、
> 单一 token 集，无需两套 token 拼接残差。**注**：delta 的写回语义（skip-writeback gate、
> `save_partial_states`）按 token 位置判定，锥内 delta token 与锥内新 token 各自走原写回逻辑，锥右边界
> 扩到 `end` 不改变写回判定。

**input 端算最多（W_eff）、output 端算最少（128）**，三角成立，省 ~50% hidden-state GEMM
（D≥W_eff 时，与 §15.3.1 一致）。staircase 跳过锥外 token 的 hidden-state 计算；锥内 token 的
`save_partial_states` 照常写（`[B,A)` state 本次要写，因 state cache 跨请求只命中到 B）。两者自洽。

### 分析纠错记录（避免再翻）

讨论中出现过两次错误翻转，记录结论以免重蹈：

1. **误判 A**："skip-writeback 不是 staircase 的前提、二者无关" —— 错。main MLA 同层读刚写回的
   delta G0（`flashmla.py:296-304` 已核验）造成全序列穿透、三角形塌成矩形；skip-writeback 让
   main MLA 改读缓存 G0，矩形才降回锥。**skip-writeback 是硬前提**。
2. **误判 B**："`save_partial_states` 强制每层穿透 → 矩形 blocker，skip-writeback 解不了" —— 错。
   `save_partial_states` 是 per-token scatter（`save_partial_states.py:85-89`），**不是滑窗**，其
   liveness 由 compress 读者窗口决定。开 skip 后 `[B,A-W)` 无读者→死→不必每层跑。矩形**只**来自
   main MLA，不来自 `save_partial_states`。

**正确结论**：skip-writeback 是 staircase 的硬前提（消除 main MLA 全序列读路径）；剩余依赖锥由
SWA + compress 读者窗口（均 W=128）驱动，三角成立。`save_partial_states` 既非矩形源、也非
skip-writeback 的必需理由，但 skip-writeback 通过死读者机制使其在锥外 `[B,A-W)` 不必跑，与锥界
自洽。

---

## Feature 2：staircase

### 目标

delta `[B,A)` 重算从全矩形 `D×L` 改成逐层递减三角形：input 端（layer 0）算 ~W_eff token，
output 端（layer 42）算 ~128，省 ~50% hidden-state GEMM（D≥W_eff 时）。**仅在
`enable_skip_writeback` 开启时生效**（见前置条件）。

### 机制：逐层物理收窄 token 集

正确原语是**物理移除锥外 token 行**（按层重排 `hidden_states`/`positions`/`slot_mapping`/
`block_table`/`query_start_loc`），**不是** `slot_mapping=-1`。核验过：`-1` 在 SWA 路径只跳
KV 写（`attention.py:550/569/583`）并把该 token gather 输出置 0（`sparse_swa.py:629-632`），
对真实 token 是破坏，且不省 GEMM（Q norm/RoPE/attn/MLP 按稠密张量整张跑）。`-1` 只对 padding
token 安全。

收窄策略（请求 r）：output 端（顶层）最窄 `[end-W, end)`，input 端（底层）最宽
`[max(B, end-W_eff), end)`（`end = B + num_scheduled`，见"锥右边界"决策）。距顶层 k 层只算
`[max(B, end-W·(k+1)), end)`。锥方向：input 端最宽、output 端最窄，所有锥同右边界 `end`、左边界
随层右移——故每层 token 集合是**更靠 output 层的后缀**（更靠 input 层是其超集，向 pos 更小方向
逐层扩张）。层间传递时更靠 input 层覆盖更靠 output 层的全部 token 并向左扩张——hidden state 层间
传递连续（更靠 output 层算过的 token 在更靠 input 层继续算，多算的左侧 token 为其提供窗口上下文）。
底层（input 端）= 最宽锥，与既有全矩形在底层对齐。

### MegaMoE 残差融合 kernel：已验证可吃子集（原 T4 风险点，已降级）

初版计划把 `mhc_*_tilelang` 当作"主要风险点，需先做可行性 spike"。**经核验三 kernel 后推翻此定性**：
它们都能直接吃更短的 token 子集，无需写 mask 变体。

核验对象（`vllm/model_executor/kernels/mhc/tilelang.py` + `tilelang_kernels.py`）：
`mhc_pre_tilelang`（L90）、`mhc_post_tilelang`（L303 / kernel L482）、
`mhc_fused_post_pre_tilelang`（L326）。结论依据：

1. **grid 按 token 维动态**：`num_tokens = T.dynamic("num_tokens")`，grid 由输入 token 维推出
   （`tilelang.py:162-163/402-403`，`tilelang_kernels.py:88/234/396/503/560`）。传更短张量 →
   自动更少 block，**非硬编码全序列长**。
2. **零跨 token reduction**：每个 token 是独立 grid block。Sinkhorn 只在 `hc_mult` 轴归约
   （`tilelang_kernels.py:135-153/276-291`），RMS sqrsum 与 GEMM 均 per-token
   （L97-99/244-245/576-582）。**没有任何跨 token 的求和/归一**。
3. **hc 参数是全局学习参数**：`hc_attn_fn`/`hc_ffn_fn`（`hc_mult3, hc_mult*hidden`）、`hc_scale`
   （`(3,)`）、`hc_base`（`(hc_mult3,)`）都不依赖 token 数（`tilelang.py:148-151/388-390`）；
   per-token 混合权重在 kernel 内逐 token 重算，token 数逐层变化**不破坏**它们。
4. **残差状态 per-token、跨层逐字传递**：`residual`/`post_mix`/`res_mix` 形状 `(num_tokens,…)`，
   无 batch 全局状态，无 order/count-sensitive per-token 状态（`model.py:872-933`）。token `t` 的
   残差只依赖 token `t` 自己的历史。

**结论**：只要 caller 传**一致切片**的 `x`/`residual`/`post_mix`/`res_mix`（同子集、同长度），
三 kernel 直接正确。T4 从"需写 mask 变体 / spike"降级为"保证子集连续性 + 一致切片"。

### 承重约束（核验得出，T3/T4 落实时必须遵守）

- **连续性（最硬约束）**：三 launcher 都做 `residual.view(-1, hc_mult, hidden_size)`
  （`tilelang.py:162/402`）与 `x.view(num_tokens, hidden_size)`（L404）。boolean-mask gather
  产出**非连续**张量，`.view()` 会 raise。**连续切片 `x[:k]` 安全**；mask 子集须先
  `.contiguous()`。staircase 子集是 `[lo_L, end)`（尾部对齐 `end`）：
  - **单请求**：是 token 段内尾部连续切片 → `x[offset:]` 连续 ✓。
  - **多请求混批**：各请求尾部子段在 batch 维拼接，需按请求 gather + `cat` + `.contiguous()`
    （非连续，必须拷贝）。这是多请求混批的实质开销。
- **`n_splits`/`use_small_fma` 随 token 数变**：`compute_num_split`（`tilelang_kernels.py:30-40`，
  调于 `tilelang.py:172/421`）与 `use_small_fma`（≤16 token，L411-415）按 token 数选不同
  `n_splits`/kernel 变体。**仅影响性能与核选择，非正确性**（split-K 累加对任意 n_splits 正确）。
  顶层锥宽 128 远大于 16，但深层 chunk 边界可能落入小 token 路径——需测性能而非正确性。
- **零 token 无显式 guard**：`hc_head_fused_kernel_tilelang` 有 `if num_tokens==0: return`
  （`tilelang.py:626-627`），但**这三 kernel 没有**。空张量 launch 空 grid、返回零大小输出——
  可能 OK 但未测。锥内永远 ≥128 token，不会空；但**chunk 边界**可能产生空子集，须显式跳过。
- **真·跨 token 依赖在 `self.attn`（L909），不在 mhc**：mhc 收窄安全，但 `self.attn`/`self.ffn`
  在收窄的 `x` 上跑。锥内 attention 只算锥内 token 的 Q，K/V 从 cache 读——这正是 staircase 要的
  （锥外 token 本就不该影响锥内输出，因依赖只向 `pos` 更小方向；锥内 token 的 SWA 窗口 `[p-128,p]`
  可能滑入锥外，但那些锥外 token 的 SWA K/V **本次必须写**——见下"锥边界 SWA 写入"）。须保证
  收窄后 attention 的 `block_table`/`slot_mapping`/`seq_lens` 对锥内 token 正确。
- **锥边界 SWA 写入语义**：锥内 token `p` 的 SWA K/V 未缓存（SWA 命中到 B），须按层写入
  （`save_partial_states` 同理对 compressor state）。staircase **只跳锥外 token 的 hidden-state
  计算**，锥内 token 的 SWA/state 写入照常。锥内 `p∈[lo_L,A)` 的 SWA 窗口 `[p-128,p]` 在
  `p<lo_L+128` 时会滑入锥外 `[lo_L-128, lo_L)`——这部分锥外 token 的 SWA K/V 是否已由更早的
  prefill chunk 写入？**若 `[lo_L-128, lo_L)` 已在前序 forward 中写入 cache，则锥内 attention
  读 cache 即可**；否则锥内 SWA 读到未写条目（错误）。这是 staircase 正确性的**关键校验点**：
  须确认锥内 SWA 窗口覆盖的锥外 token，其 K/V 在更早的层/chunk 已落 cache。

### 实现路径（分步）

**T1 复用 A 透传** — 三角边界用每请求 A（=G0 命中长度），Feature 1 的 S2-S4 已建立
`request.g0_hit_length` → `CommonAttentionMetadata.g0_cached_prefix_len` 通道，直接复用；跨
preemption 的取值规则取决于 S3 待澄清的生命周期语义。

**T2 三角边界计算** — runner 侧（`gpu_model_runner.py` prefill eager 路径）按请求算每层 token
下界 `lo_L = max(B, end - W·(L_top-L+1))`（`end = B + num_scheduled`），产出 per-request per-layer
的 `[lo_L, end)` 区间。W=128。

**T3 层循环按层收窄** — `DeepseekV4Model.forward`（`model.py`）层循环内：依据当前层
`lo_L` 切片 `hidden_states`/`positions`，把该层各 prefix 的 metadata **收窄**到锥
`[lo_L, end)`，经 `forward_context`（`get_forward_context().attn_metadata[prefix]`，每层独立
prefix，`compressor.py:137`）注入该层。**不改 dataclass、不重 build**——收窄 = 对 runner 正常
`build()` 产出的完整 metadata 做**尾切片 + 标量重算**（见下"收窄方式"）。SWA/compressor/indexer
每层共享同一子集（锥由 W=128 统一）。

**T3 注入机制（已核验可行）**：每层 attn/compressor/swa/indexer 有独立 `self.prefix`
（`attention.py:167/634/718`，如 `f"{layer_prefix}.compressor"`），forward 时各自
`get_forward_context().attn_metadata[self.prefix]` 取自己 metadata。`attn_metadata` 是
`dict[str, AttentionMetadata]`，**forward 期动态设置**（`forward_context.py:132/139`）。故 T3
注入点 = 层循环内、调 `layer()` 前，把 `get_forward_context().attn_metadata[prefix]` 替换为该层
收窄版。

> **承重风险——缓存复用污染**：runner 侧 attn_metadata 按 `(KVCacheSpec, builder type)` 缓存
> 复用，**组内多层共享同一 metadata 对象**（`gpu_model_runner.py:2454-2469` `cached_attn_metadata`，
> L2478-2479 `attn_metadata_dict[layer_name] = attn_metadata_i` 同对象赋给组内所有层）。T3 **绝不能
> mutate 该共享对象**（会污染同组其他层）。必须为每层构造**独立**的收窄 metadata
> （或独立张量引用），在层循环内临时替换、层后还原。这是 T3 实现的首要正确性约束。

#### T3 收窄方式：切片真实 build，而非逐层重 build / triton（2026-07 决策）

**核心难点**：收窄 hidden_states 不够——attention/compressor/swa/indexer 各自从
`forward_context.attn_metadata[prefix]` 读 `slot_mapping`/`block_table`/`seq_lens`/
`query_start_loc`/`positions`（SWA builder L385-391 全从 `common_attn_metadata` 取）。这些必须与
收窄后的 token 子集**一致**，否则 slot 错位或形状不匹配。

**方案选择（核验 DSV4 四个 builder 的 `build()` 后定）**：逐层调用 `builder.build()` 太慢；曾考虑
triton 算子重算派生量。但逐 kernel 核验（SWA `_compute_swa_indices_and_lens_kernel`、
`_compute_prefill_metadata_kernel`；MLA `_build_c128a_topk_metadata_kernel`；indexer
`_build_prefill_chunk_metadata_kernel`）后确认：**单请求 prefill 下，每个 per-token 派生量只依赖
该 token 自己的 position `p` + 不变的 `block_table`/`seq_lens=end`，与 batch 里其它 token 无关**。
具体：
- SWA `prefill_swa_indices[i]`/`prefill_swa_lens[i]`：每 token 环窗口只由 `p`+block_table 决定。
- MLA `c128a_prefill_topk_indices` 行：prefill 分支写 `local_indices[j]=j if j<(p+1)//cr else -1`，
  内容只依赖 `p`。
- indexer `cu_seq_len_ke[out] = (p+1)//cr`（`start_pos + offset = p` 恒成立，`start_pos` 逐层虽变但
  `start_pos+offset` 恒等于 `p`）；`cu_seq_len_ks[out]=0`（单请求）；`cu_seq_lens`/`token_to_seq`
  是 KV 侧、由 `seq_lens=end` 构建，与 `lo` 无关。

故收窄 = **对完整 build 输出做尾切片 `full[lo-B:]`（per-token 张量，view 免拷贝）+ Python 重算少数
per-request 标量**（`query_start_loc=[0,end-lo]`、`num_actual_tokens=end-lo`、
`prefill_gather_lens=(end-lo)+min(lo,W-1)`、`token_start=lo`/`token_end=end` 等）。**无 kernel、无
逐层 `build()`**。实现见 `vllm/models/deepseek_v4/nvidia/staircase.py`（四个 narrow 函数 +
`narrow_metadata` 分发）。

> **为什么不用 triton 算子重算派生量**：① 不必要——慢的是逐层完整 `build()`，而完整步本来就已 build
> 一次，锥层只需切片那一次 build 的输出；切片是 view，比 kernel launch+写还便宜。② 违反父仓 CLAUDE.md
> 硬规则"KV cache 相关逻辑必须调真实 vllm 函数、不得手搓"——一个重算 `build()` 派生量（压缩 slot
> mapping、c128a topk、indexer `cu_seq_len_ke`）的 triton kernel 就是手搓 build() 逻辑。切片真实
> `build()` 的*输出*是消费、不是重实现，合规。

> **为什么限单请求（正确性前提，非优化）**：锥是 per-request 概念（不同请求 B/A/seq_len 不同，锥边界
> 不同）；token 轴按请求**连续排布**。多请求下对整条 token 轴做尾切片 `full[N-k:]` 切到的是**最后一个
> 请求**的尾部，而非"每请求各自收窄成自己的锥"——没有对应物。要 per-request 不同 `lo` 得对每请求段
> 分别切再重排（gather，非切片），并重建 `query_start_loc`/`token_to_req_indices`/`block_table`。
> 且 indexer `cu_seq_lens` 跨请求累积，收窄某请求的 query 段要重算前后所有请求的 `token_to_seq`。
> 故 staircase **只单请求 prefill 启用**（runner gate `num_reqs==1 and prefill and A>B`），narrow
> helper 内 `assert` 单请求 prefill，fail-closed。多请求 staircase 是另一量级改动，需先解 compressor
> state-cache 跨层缓存前置，不在本计划。

**单请求语义澄清（承重）**：prefill 时 `num_computed_tokens=B`，本步调度的 token 覆盖位置
`[B, end)`（`end = B + num_scheduled`）。锥是 `[lo_L, end)`（右边界 = `end`，见"锥右边界"决策）。
本步实际算的 token = 调度 token ∩ 锥 = `[max(B, lo_L), end)`。单请求 spike 假设 **整段 delta 在一次
prefill 内**（即 `end >= A`，无 chunked prefill 切分——见工程约束 4 的 chunk 化语义，spike 阶段先
禁用 chunked prefill 或保证单 chunk）。

> 位置 vs token 索引：prefill 调度的 token 在 hidden_states 里是**连续**的，位置从 B 起递增。
> 位置 `p` 对应 hidden_states 行 `p - B`。锥 `[lo, end)` → 行切片 `[lo - B, end - B)`，**连续尾部对齐
> end**（mhc `.view()` 要求连续，单请求天然满足）。

**runner 侧（`gpu_model_runner.py`）**：只发激活信号 `staircase_ab`（已实现，L4350 一带：当
`enable_staircase and num_reqs==1 and prefill and A>B` 时建 `tensor([[A,B]])`，经
`set_forward_context(staircase_ab=...)` 传入）。**不预建收窄 metadata**——收窄由 model lazy 完成
（方案 B，见下）。

**model 侧（`DeepseekV4Model.forward` 层循环，`model.py`）**：`stair_md` 是 model 自有的空 dict
（不再由 runner 预填）。层循环里，对当前层各 prefix，首次用到时从 `fc.attn_metadata[prefix]`（runner
正常 `build()` 的完整对象）调 `narrow_metadata(...)` 切片收窄、缓存进 `stair_md`，再 swap 进
`attn_metadata[prefix]`、层后还原。

```python
# model, DeepseekV4Model.forward 层循环（已实现）：
fc = get_forward_context()
stair_ab = getattr(fc, "staircase_ab", None)
staircase_active = stair_ab is not None   # runner 唯一激活信号
if staircase_active:
    stair_md: dict[str, Any] = {}         # model 自有缓存，lazy 填充
    b = int(stair_ab[0, 1].item())        # B (num_computed)
    end = b + hidden_states.shape[0]      # 锥右边界（见"锥右边界"决策）
    W = self.staircase_window             # 128
    L_top = self.num_layers_total - 1
    cur_lo = b                            # 当前 hidden_states 左边界（绝对位置）

for i, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    l = self.start_layer + i  # 全局层号
    if staircase_active:
        lo = max(b, end - W * (L_top - l + 1))
        rel = lo - cur_lo              # 本层锥相对当前 hidden_states 左端的偏移（≥0）
        hs_narrow = hidden_states[rel:]            # 后缀切片（见下"层顺序与切片方向"）
        pos_narrow = positions[lo - b : end - b]   # positions 全宽，按绝对位置切
        ids_narrow = input_ids[lo - b : end - b] if input_ids is not None else None
        if residual is not None: residual = residual[rel:]
        if post_mix is not None: post_mix = post_mix[rel:]
        if res_mix is not None: res_mix = res_mix[rel:]
        # 首次用到时 lazy 收窄该层各 prefix 的 metadata，缓存进 stair_md
        self._ensure_layer_narrowed(fc, layer, stair_md, lo, end, b, W)
        saved = self._swap_layer_metadata(fc, layer, stair_md)  # 注入收窄版、保存原引用
        hidden_states, residual, post_mix, res_mix = layer(
            hs_narrow, pos_narrow, ids_narrow, post_mix, res_mix, residual,
        )
        self._restore_layer_metadata(fc, saved)     # 层后还原（下一层有自己收窄版）
        cur_lo = lo  # 输出收缩到 [lo, end)
    else:
        hidden_states, residual, post_mix, res_mix = layer(
            hidden_states, positions, input_ids, post_mix, res_mix, residual,
        )
```

`_ensure_layer_narrowed` 对层各 prefix 调 `narrow_metadata(attn_md[prefix], lo, end, b, W)`（
`staircase.py`），按 metadata 类型分发到四个 narrow 函数：per-token 张量尾切片 `full[lo-b:]`、
per-request 标量重算。`_swap_layer_metadata`/`_restore_layer_metadata` 只重绑 dict 条目，绝不 mutate
runner 的共享缓存对象。

**关键设计决策**：
- **seq_len 不收窄，只收窄 query**：锥内 token 的 SWA/MLA K/V 仍从 cache 读 `[0, seq_len)`（`seq_len`
  = 该请求本步上下文 `end`），只是把 query（本步要算的 token）限制在锥内 `[lo, end)`。这与全矩形路径
  语义一致——全矩形也是 query = 调度 token `[B, end)`，K/V 读 cache。区别只是 query 范围从
  `[B, end)` 缩到 `[lo, end)`。
- **slot_mapping/block_table 收窄**：锥内 token 的 slot 必须正确指向其 cache 槽（KV 写入用）。连续切片
  `slot_mapping[lo-b:]` 保持映射一致（slot 按 token 顺序排）。
- **query_start_loc 重置为 `[0, n_cone]`**：单请求，n_cone 个 query token。
- **切片而非重 build**：避开缓存复用污染——narrow 产出新对象（尾切片 view + 新标量），不碰 runner 的
  `cached_attn_metadata`；且免去逐层 `build()` 的开销与 triton 重算（见"收窄方式"）。
- **`stair_md` 是 model 自有 lazy 缓存**（`DeepseekV4Model.forward` 局部空 dict，激活时建、层循环里
  填）。runner 只传 `staircase_ab` 激活信号——把"怎么收窄某层 metadata"这件深度依赖 model 结构
  的事留在 model，不上漏到 runner。

> **层顺序与切片方向（承重，2026-07 修正）**：层循环 input→output（层 0 最宽 → 层 L_top 最窄），
> 与 `islice(self.layers, start, end)` 的自然顺序一致。每层锥 `[lo_l, end)` 是上一层锥
> `[lo_{l-1}, end)` 的**后缀**（`lo_l ≥ lo_{l-1}`，向 output 端左边界右移）。故：
> - `hidden_states` 每层**收缩**到本层锥宽，下一层取其**尾部切片** `hidden_states[rel:]`（`rel =
>   lo_l - cur_lo ≥ 0`），**不是**按绝对行号 `hidden_states[lo-b:end-b]`（那会切错，因 hidden_states
>   已不是全宽）。`cur_lo` 跟踪当前左边界。
> - `positions`/`input_ids` 保持全宽，按绝对位置切（它们不被层改写）。
> - `residual`/`post_mix`/`res_mix` 同 `hidden_states`，按 `rel` 尾部切片。
>
> 旧 plan 伪代码用绝对行号切片 `hidden_states[row_lo:row_hi]` 是**错的**——层输出收缩后绝对行号失效。
> 已修正为相对 `cur_lo` 的尾部切片（见 model.py 实现与 `test_dsv4_staircase_cone.py`）。

**残差跨层对齐（后缀切片，承重）**：锥方向已修正为 **input 端最宽、output 端最窄，所有锥同右边界
`end`、左边界随层右移**。故每层锥是上一层锥的**后缀**（尾部对齐 `end`，左端逐层右移）：

```
层 0  (input端): [lo_0 ==============================> end)
层 1:              [lo_1 =========================> end)   ← lo_1 > lo_0，是层0的后缀
  ...
层 42 (output端):                [lo_42 = end-128 ===> end)
```

每层 `residual`/`post_mix`/`res_mix` 长度 = 该层锥宽。下一层（`lo_{l+1} > lo_l`，更窄）要的残差 =
本层残差**砍掉左端 `(lo_{l+1} - lo_l)` 个**的尾部后缀（即 `residual[rel:]`，`rel = lo_{l+1} - lo_l`）。
跨层传递只需一个 slice，**无需全宽缓冲、无需 scatter**。

> 旧 plan 的"全宽残差缓冲 + 每层 scatter 写回 (β)" 是基于**错误的锥方向**（层向 output 变宽，下一层
> 更宽需左端补新 token）。方向修正后下一层更窄，纯后缀切片即可。**(α)/(β) 全宽缓冲方案作废。**
>
> **首层（input 端，最宽）的 `residual=None` 路径**：`DeepseekV4DecoderLayer.forward` 在 `residual is
> None` 时走 `mhc_pre_tilelang`（`model.py:872-887`）初始化残差。staircase 首层锥 = 最宽锥
> `[lo_0, end)`，传 `residual=None` 触发 `mhc_pre` 对该子集初始化，正确。后续层传上一层的后缀切片
> 残差，走 `mhc_fused_post_pre_tilelang`。锥宽变化不影响 mhc（grid 动态、per-token、已验证可吃子集）。
>
> **末层 `mhc_post_tilelang`**（`model.py:1097`）：层循环后对最后一层输出收尾。staircase 下最后一层
> 输出 = 最窄锥 `[end-128, end)`，`mhc_post` 在该子集上跑，输出 128 行 hidden state——这是锥内（output
> 端）token 的最终 hidden state，正是 staircase 要产出的。锥外 token 的 hidden state 不产出（它们在
> 更早的、更宽的层算过，但其最终态对 output 端无影响，因依赖只向 pos 更小方向）。

**待 spike 验证项（T3/T4 合并）**：
1. 收窄（切片真实 build）出的 SWA/compressor metadata，能否让锥内 attention 正确读 `[0,A)` 的 K/V 并
   只写锥内 token 的 K/V。**CPU 单测已覆盖切片等价性**（`test_dsv4_staircase_narrow.py`：per-token
   张量 == `full[lo-B:]` 尾切片、标量 == 闭式），但"锥内 attention 正确读 K/V"需真实 forward。
2. 残差后缀切片的跨层对齐正确性（每层锥是上层后缀，slice 对齐 `end`）。**CPU 单测已覆盖**
   （`test_dsv4_staircase_cone.py`：mock 层循环验证残差按 `rel` 尾切片 hand-off）。
3. 锥边界 SWA 自洽（前述"锥边界 SWA 写入语义"）。
4. flag 开/关（skip-writeback 两边开）锥内 token 输出逐位一致。**E2E 测试已写**
   （`test_dsv4_staircase_e2e.py`，`@skipif(not SM90+)`，本机 skip、目标 GPU 机跑）。

**T4 MegaMoE 残差处理** — mhc 三 kernel **本身**已验证可吃子集（grid 动态、零跨 token reduction、
hc 全局参数）。锥方向修正后（input 端宽、output 端窄、同右边界 `end`），每层锥是上层后缀，残差跨层
只需后缀切片（见 T3"残差跨层对齐（后缀切片）"），**无全宽缓冲、无 scatter、(α)/(β) 作废**。

**正确形态**：每层整条流水（mhc+attn+ffn）在锥宽 `[lo_l, end)` 上跑，输出残差是 `[lo_l, end)` 子集；
下一层取本层残差的尾部后缀（砍左端 `lo_l - lo_{l+1}` 个）作初值。首层 `residual=None` 触发 `mhc_pre`
对最宽锥初始化。mhc kernel **不改 tilelang 源**——子集调用已验证可行，T4 工作量 = 层循环内的切片 +
后缀传递，非改 kernel。

**T4 代码已落在 T3a**：`DeepseekV4Model.forward` 层循环里 `if residual is not None: residual =
residual[rel:]`（post_mix/res_mix 同理），首层 `residual=None` 走 `mhc_pre`——即 T3a 的相对尾切片
narrowing 路径同时实现了 T4 的残差跨层后缀传递。故 **T4 无独立代码**，只剩 GPU 验证项（见下"锥内
一致性验证"）。

**待 spike 验证**（与 T3 合并，见上"待 spike 验证项"）：残差后缀切片跨层对齐正确性、
锥边界 SWA 自洽、flag 开/关锥内 token 逐位一致。

**T5 配置 flag** — `vllm/config/cache.py` 加 `enable_staircase: bool = False`，gate 在
`DeepseekV4Model.forward`。**硬依赖 Feature 1**：`enable_staircase=True` 隐含强制
`enable_skip_writeback=True`（否则 main MLA 全序列读路径使三角形退化为矩形，见前置条件）。flag
检查里若 skip-writeback 未开则报错或自动连带开启。默认关 → 全矩形 → 逐位一致。

### 工程约束

1. **CUDA graph 静态形状**：三角 token 数依赖 D=A−B（每请求不同）→ 不能静态 capture。但
   staircase 是 prefill 期优化，默认 `CUDAGraphMode.FULL_DECODE_ONLY=(FULL,NONE)`
   （`compilation.py:62`）下 **prefill 走 NONE（eager）**，无约束。**限制**：与 `FULL`
   （prefill 也 capture）不兼容，flag 检查里拒绝 `FULL` 或回退全矩形。
2. **多请求混批**：边界 per-request，收窄后 token 集合需在 batch 维重拼 `query_start_loc`
  （批内 D 不同 → 总 token 数动态）。eager prefill 下可接受。**且子集非连续**：各请求尾部子段
  `[lo_L_r, A_r)` 拼接须 `gather`+`cat`+`.contiguous()`（mhc launcher 的 `.view()` 要求连续，见
  承重约束）。单请求则是连续切片，无此开销。
3. **C4 窗口(8)被 W=128 拉平**：统一用 W=128 定锥，C4 本可更窄但被拉平，省量略低于上界。
4. **chunked prefill（真实 vLLM 默认开）**：`enable_chunked_prefill=True`（`scheduler.py:84`），
   DSV4 不禁用。仓库 CLAUDE.md 的"no chunked prefill"指 **simulator**，不是真实引擎。故 delta
   `[B,A)` **可能跨多个 prefill chunk**，staircase 三角边界须按 chunk 起止切分，不能假设
   "delta 在单次 prefill 内"。未发生 preemption 时，首次 lookup 用全 prompt 长度
   `request.num_tokens-1` 得到 A，后续 chunk 沿用；跨 preemption 的 A 是否延续或由 resume lookup
   覆盖，取决于 S3 待澄清的生命周期语义。每 chunk 内可参与三角的 token 范围 = 该 chunk 与当时
   有效 `[B,A)` 的交集，三角边界需按 chunk 逐段定。这是 staircase 的实质复杂度，设计阶段需明确
   chunk 化三角语义。
5. **锥边界 SWA 写入正确性（关键校验点）**：见"MegaMoE 残差融合 kernel"节的"锥边界 SWA 写入语义"。
   锥内 token 的 SWA 窗口可能滑入锥外，须确认那些锥外 K/V 已在更早层/chunk 落 cache，否则锥内
   attention 读到未写条目。**这是 staircase 正确性的首要验证项**，优先于性能。

### 待改文件（Feature 2）

- `vllm/config/cache.py` — `enable_staircase` flag。
- `vllm/models/deepseek_v4/nvidia/model.py` — `DeepseekV4Model.forward` 层循环按层收窄 + 注入；
  `DeepseekV4DecoderLayer.forward` 内 `mhc_*` 调用透传收窄的 `x`/`residual`/`post_mix`/`res_mix`
  （**不改 tilelang 源**，三 kernel 已支持子集，仅保证一致切片 + 连续性）。
- `vllm/models/deepseek_v4/attention.py`、`compressor.py` — 接受按层收窄 metadata（复用 prefix
  通道，不改 dataclass；注意不得 mutate runner 缓存的共享 metadata 对象，见 T3 承重风险）。
- `vllm/models/deepseek_v4/attention.py`、`compressor.py` — 接受按层收窄 metadata（复用 prefix
  通道，不改 dataclass）。
- `vllm/v1/worker/gpu_model_runner.py` — prefill eager 路径准备 per-request per-layer 边界。

### 验证（Feature 2）

- **一致性（主门槛）**：flag 关走全矩形（与主线逐位一致）。flag 开用**底层对齐**验证：对请求 r，
  层 L 锥内 token 的 hidden state 不依赖层 L 锥外 token（依赖只向 `pos` 更小方向），故三角路径
  锥内 token 输出应与全矩形路径逐位一致。逐层断言锥内 token hidden state 一致。
  **前提**：比较基线两边都须 `enable_skip_writeback=True`（staircase 硬依赖它），仅
  `enable_staircase` 开/关对比。
- **单测**：`D>W_eff` 小模型；断言每层进入 forward 的 token 数 = 期望三角宽度，锥内输出与全矩形
  一致。
- **省量**：Nsight 数各层 GEMM FLOPs/kernel 时间，总 ≈ 全矩形 ~50%。D=5500 测 delta 步 GPU 时间。
- **cudagraph 兼容**：`enable_staircase=True` 且 `FULL` 模式时回退全矩形或报错。

---

## 通用验证

- `pre-commit run ruff-check --all-files`；`pre-commit run mypy-3.12 --all-files --hook-stage manual`
  （行宽 88，Google 风格 docstring）。
- `.venv/bin/python -m pytest <test_file> -v`（绝不用系统 python3）。

### 硬件验证约束（承重，已核验）

T3/T4 的"锥内逐位一致"验证需要跑真实 DSV4 forward，但 DSV4 的 attention/MoE kernel 有硬架构要求：

- **FlashMLA sparse**（DSV4 主 MLA + SWA decode 路径都走它）要求 `capability.major ∈ [9, 10]`
  （`vllm/v1/attention/backends/mla/flashmla.py:83`）。
- **MegaMoE** 要求 SM100（`vllm/models/deepseek_v4/nvidia/model.py:303-304`）。

当前开发机是 **NVIDIA T1200 Laptop GPU（SM7.5）** → **无法跑通任何 DSV4 attention 路径**，B 阶段
（最小 forward 基线）在本机不可行。editable 安装已就绪（`.venv` 直指 `3rdparty/vllm` 源码树），但
forward 验证须在 **SM90+（H100）或 SM100（B200）** 上进行。本机能做的仅限：import 冒烟、字段/类型
检查、纯 CPU 逻辑（如锥边界 `lo_L` 计算、`staircase_ab` 构造）。

** implication for T3/T4**：收窄逻辑的代码已本机编写 + 静态检查 + CPU 单测（narrow 切片等价、
cone 边界、残差后缀 hand-off），但"锥内逐位一致"必须留到目标 GPU 机器验证。已写 E2E 测试脚手架
`tests/v1/attention/test_dsv4_staircase_e2e.py`（`@skipif(not SM90+)`），本机 skip、目标机跑。

### 锥内一致性验证（E2E 测试设计，已写）

`test_dsv4_staircase_e2e.py::test_staircase_cone_matches_full_rectangle`：flag 开/关各跑同一请求，
比较 output-end 锥 `[end-W, end)` token 的 logits 逐位一致（staircase 只省算、不改锥内结果）。

**APC 命中长度的可控构造（不依赖自然分化）**：真实请求跑出命中 `[B, A)` 后，monkeypatch
`KVCacheCoordinator.find_longest_cache_hit` 把报告的命中**收缩**到更小的 `[B', A')`，其中
`B <= B' < A' <= A`、`A' > B'`：

- `B'` 截断调度左边界（vLLM 调度 `[B', end)` 而非 `[B, end)`，少算几个 delta token，仍全缓存）。
- `A'` 设 G0 命中长度（staircase gate 见 `A' > B'` 激活）。`[B', A')` ⊂ 真实缓存 `[B, A)`，故 main MLA
  全序列读 G0 读到真缓存条目（skip-writeback 前提成立）。
- `B'`、`A'` 向下取整到 `scheduler_block_size`，截断的 hit blocks 保持 block 对齐（vLLM 要求
  `num_computed_tokens` block 对齐）。收缩只**少用**已有真实命中，KV 状态自洽、锥计算正确。

**为何不直接 mock `staircase_ab`**：`staircase_ab` 的 `b` 同时是 hidden_states 左边界
（`model.py: end = b + hidden_states.shape[0]`），b 必须等于真实 `num_computed_tokens`，否则 end 错位。
故只能从 APC 命中源头（`find_longest_cache_hit`）改 A、B，让 vLLM 全程自洽用 `[B', A')`，而非在 runner
构造 `staircase_ab` 处改。这也回应了"真实请求跑出 a、b 取 min 后在此基础上改 a、b 跑两次"——改的点是
APC 计算，不是 staircase_ab。

**已知未验证风险（本机 SM7.5 跑不了 forward，目标机可能需调 1-3 轮）**：
1. coordinator 访问路径 `engine_core.engine_core.scheduler.kv_cache_manager.coordinator`（inproc 模式，
   `VLLM_ENABLE_V1_MULTIPROCESSING=0`）——静态推断，未实跑。
2. patch 的 block 对齐假设 `scheduler_block_size // spec.block_size` 为整数（DSV4 各组 block_size 是
   scheduler_block_size 的因数）——若某组不是因数会截错。
3. `load_format="random"` + 完整 43 层 DSV4 的显存 / 数值稳定性——未实跑。
4. logprobs 比较是间接的（锥内逐位一致更严格应比 hidden states）——可能需改用 hidden-states 钩子。

## 阶段划分

1. **阶段一**：Feature 1（skip-writeback），S1-S5，独立可落地，先合入。建立 A 透传基础设施。
   此阶段本身即有收益（省 `[B,A)` 边界 token 的 compress 聚合 + G0 写），且是阶段二的硬前置。
2. **阶段二**：Feature 2（staircase），**硬依赖阶段一**（`enable_staircase` 强制
   `enable_skip_writeback`）。T5a flag + T2 (A,B) 透传骨架**已实现**（commit `91c13e415`）。
   T3（层循环收窄，切片真实 build + lazy narrow）+ T4（残差跨层后缀切片，代码落在 T3a）
   **已实现**（commit `903e87944` T3a、`fe040f6a2` T3b）。CPU 单测已覆盖 narrow 切片等价、cone 边界、
   残差后缀 hand-off（`test_dsv4_staircase_narrow.py` / `test_dsv4_staircase_cone.py`）。E2E 锥内一致性
   测试已写（`test_dsv4_staircase_e2e.py`，`@skipif(not SM90+)`，本机 skip、目标 GPU 机跑，见"锥内一致性
   验证"）。**剩余仅 GPU 验证**：在 SM90+/SM100 目标机跑 E2E 测试，确认 flag 开/关锥内 token 逐位一致
   （先单请求、eager prefill；多请求混批需 gather+cat+contiguous，future work）。

## 当前状态与遗留问题（2026-07-15）

### 已完成

**Feature 1（skip-writeback）**：S1-S5 全部落地。
- `compress_gate_slot_mapping` 独立 gate 张量（`position < A` 置 -1），`compress_norm_rope_store` 早退，
  `save_partial_states` 不变。默认关 → 逐位一致。
- per-request A 透传：`request.g0_hit_length` → `input_batch.g0_cached_prefix_len_cpu` →
  `CommonAttentionMetadata.g0_cached_prefix_len` → compressor builder。
- CPU 单测：`test_dsv4_skip_writeback_gate.py`（builder gate 构造 + mock-kernel dispatch，6 测试）。

**Feature 2（staircase）**：T2-T5 全部落地。
- T5a flag：`enable_staircase`（`cache.py`），强制 `enable_skip_writeback`。
- T2：runner `staircase_ab = [[A,B]]`（单请求 prefill + A>B 时）。
- T3a：model 层循环相对尾切片 narrowing + `_swap/_restore_layer_metadata`（commit `903e87944`）。
- T3b：`staircase.py` 四个 narrow helper（切片真实 build 输出，无逐层 rebuild / 无 triton）+ model lazy
  收窄接入（commit `fe040f6a2`）。
- T4：残差跨层后缀切片，代码落在 T3a（无独立代码）。
- CPU 单测：`test_dsv4_staircase_cone.py`（8 测试，cone 边界 + 层循环 narrowing + 残差 hand-off）、
  `test_dsv4_staircase_narrow.py`（14 测试，narrow 切片等价 + 闭式标量 + fail-closed）。
- E2E 测试：`test_dsv4_staircase_e2e.py`（`@skipif(not SM90+)`，本机 skip）。

### 遗留问题

1. **【阻塞，本机不可解】E2E 锥内一致性未实跑**：开发机 SM7.5 跑不了 DSV4 forward。需在 SM90+（H100）
   或 SM100（B200）目标机跑 `test_dsv4_staircase_e2e.py`，确认 flag 开/关锥内 token 逐位一致。测试脚手架
   已写但 forward 路径未实跑，目标机可能需调 1-3 轮（已知风险见"锥内一致性验证"节：coordinator 访问
   路径、block 对齐、random-weight 稳定性、logprobs 间接性）。
2. **【设计限制，非 bug】单请求 prefill only**：staircase 只在 `num_reqs==1 and prefill and A>B` 激活。
   多请求下锥是 per-request、token 轴按请求连续排布，尾切片无对应物——这是正确性前提，narrow helper
   内 `assert` 单请求 fail-closed。多请求 staircase 是另一量级改动，需先解 compressor state-cache 跨层
   缓存前置，future work。
3. **【设计限制】eager prefill only / 无 CUDA graph**：锥宽逐层变化（非静态形状），不能 cudagraph
   capture。staircase 激活时强制 eager。
4. **【设计限制】无 chunked prefill**：单请求 spike 假设整段 delta 在一次 prefill 内（`end >= A`）。
   chunked prefill 切分语义未处理，spike 阶段禁用。
5. **【待观测】收益未实测**：~50% hidden-state GEMM 节省是设计估算，未在目标机 Nsight/trace 实测。
   skip-writeback 的 `[B,A)` 边界 token compress 聚合 + G0 写节省同样未实测。
6. **【代码质量】model.py 有 pre-existing I001 ruff import-sort**（非本工作引入，surgical-changes 不动）。

### 下一步（按优先级）

1. 拿到 SM90+/SM100 目标机，跑 `test_dsv4_staircase_e2e.py`，按已知风险调通，确认锥内逐位一致。
2. 目标机实测收益（Nsight/Chrome trace：`compress_norm_rope_store` launch 数、delta 步 GPU 时间
   flag 开/关对比）。
3. （可选，future work）多请求混批 staircase：先做 compressor state-cache 跨层前缀缓存前置，再设计
   per-request 锥的 gather+cat+contiguous 收窄。
