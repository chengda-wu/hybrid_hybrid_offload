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
捕获 A，flag 开启时设 `request.g0_hit_length = A`。**A 在 request 生命周期内固定**：`get_computed_blocks`
只在 `request.num_computed_tokens == 0` 时调用一次（`scheduler.py:676` 守卫、L714 调用），running
请求与后续 prefill chunk 都不重算。首 chunk 调用时 `max_cache_hit_length = request.num_tokens - 1`
（全 prompt 长度，`kv_cache_manager.py:227`），故一次即得全 prompt 的 G0 命中 A，跨 chunk 不变。
scheduler.py 无逻辑改动（DSV4 走常规路径，scheduler.py:712）。

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
token 下界 `p >= A - W·(42 - l + 1)`：

- input 端 l=0：`[max(B, A-W_eff), A)`，宽 ≈ W_eff = 128+42·127 = 5462。
- output 端 l=42：`[A-128, A)`，宽 128。

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

收窄策略（请求 r）：output 端（顶层）最窄 `[A-W, A)`，input 端（底层）最宽
`[max(B, A-W_eff), A)`。距顶层 k 层只算 `[max(B, A-W·(k+1)), A)`。每层 token 集合是**更靠 output
层的超集**（向 pos 更小方向逐层扩张），故层间传递时下层（更靠 input）覆盖上层全部 token、并向
左扩张——hidden state 层间传递连续（上层算过的 token 在下层继续算，下层多算的左侧 token 为上层
提供窗口上下文）。底层 = 全 `[B,A)`，与既有全矩形在底层对齐。

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
  `.contiguous()`。staircase 子集是 `[lo_L, A)`（尾部对齐 A）：
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
`request.g0_hit_length` → `CommonAttentionMetadata.g0_cached_prefix_len` 通道，直接复用。

**T2 三角边界计算** — runner 侧（`gpu_model_runner.py` prefill eager 路径）按请求算每层 token
下界 `lo_L = max(B, A - W·(L_top-L+1))`，产出 per-request per-layer 的 `[lo_L, A)` 区间。W=128。

**T3 层循环按层收窄** — `DeepseekV4Model.forward`（`model.py:1077-1085`）层循环内：依据当前层
`lo_L` 切片 `hidden_states`/`positions`，构造收窄的 `slot_mapping`/`block_table`/
`token_to_req_indices`/`query_start_loc`，经 `forward_context`（`get_forward_context()
.attn_metadata[prefix]`，每层独立 prefix，`compressor.py:137`）注入该层。**不改 dataclass**，
只按层替换张量引用。SWA/compressor/indexer 每层共享同一子集（锥由 W=128 统一）。

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

#### T3 详细设计（伪代码级，已对照真实代码）

**核心难点**：收窄 hidden_states 不够——attention/compressor/swa/indexer 各自从
`forward_context.attn_metadata[prefix]` 读 `slot_mapping`/`block_table`/`seq_lens`/
`query_start_loc`/`positions`（SWA builder L385-391 全从 `common_attn_metadata` 取）。这些必须与
收窄后的 token 子集**一致**，否则 slot 错位或形状不匹配。且 builder 在 runner 侧（model 侧拿不到），
故选 **方案 2：runner 预建 per-layer 收窄 metadata，model 层循环只读取替换**。

**单请求语义澄清（承重）**：prefill 时 `num_computed_tokens=B`，本步调度的 token 覆盖位置
`[B, B+num_scheduled)`。锥是 `[lo_L, A)`。**但本步实际算的 token = 调度 token ∩ 锥 =
`[max(B, lo_L), min(A, B+num_scheduled))`**。单请求 spike 假设 **整段 delta 在一次 prefill 内**
（即 `B+num_scheduled >= A`，无 chunked prefill 切分——见工程约束 4 的 chunk 化语义，spike 阶段先
禁用 chunked prefill 或保证单 chunk）。在此假设下，本步锥内 token = `[max(B, lo_L), A)`。

> 位置 vs token 索引：prefill 调度的 token 在 hidden_states 里是**连续**的，位置从 B 起递增。
> 位置 `p` 对应 hidden_states 行 `p - B`。锥 `[lo, A)` → 行切片 `[lo - B, A - B)`，**连续尾部对齐
> A**（mhc `.view()` 要求连续，单请求天然满足）。

**runner 侧（`gpu_model_runner.py`，`_build_attention_metadata` 之后、`set_forward_context` 之前）**：

当 `staircase_ab` 非空（单请求 prefill + A>B）时，为每个 cone 层 `l ∈ [start_layer, end_layer)` 预建
收窄 metadata，存入 `forward_context`（新字段 `staircase_layer_metadata: dict[str, AttentionMetadata]`，
key = 层各 prefix）。

```python
# runner, after building full attn_metadata, when staircase active:
a, b = staircase_ab[0].tolist()  # A, B for the single request
L_top = num_layers_total - 1
W = sliding_window  # 128
# 该请求本步调度 token 的位置区间 [b, b + num_scheduled_tokens)
sched_hi = b + num_scheduled_tokens  # == A 在 spike 假设下
staircase_layer_metadata = {}
for l in range(start_layer, end_layer):
    lo = max(b, a - W * (L_top - l + 1))
    if lo >= a:  # 锥空（不会发生，lo<=a 恒成立因 L_top-l+1>=1 → a-W*...<=a）
        continue
    # 行切片 [lo - b, a - b)，连续
    row_lo, row_hi = lo - b, a - b
    n_cone = row_hi - row_lo
    # 为每个 kv_cache_group 用收窄的 cm 重新 build
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
        cm_narrow = copy(cm_base)
        cm_narrow.positions = cm_base.positions[row_lo:row_hi]          # 连续切片
        cm_narrow.slot_mapping = slot_mappings[kv_cache_gid][row_lo:row_hi]
        cm_narrow.block_table_tensor = _get_block_table(kv_cache_gid)   # 不变（per-req）
        cm_narrow.query_start_loc = torch.tensor([0, n_cone], dtype=int32, dev)
        cm_narrow.query_start_loc_cpu = ...同上 cpu
        cm_narrow.seq_lens = <A>  # seq_len 不变（KV 仍读 [0,A)），只收窄 query
        cm_narrow.num_actual_tokens = n_cone
        cm_narrow.num_reqs = 1
        builder = attn_groups[kv_cache_gid][...].get_metadata_builder(0)
        md_narrow = builder.build(common_prefix_len=..., common_attn_metadata=cm_narrow)
        for layer_name in kv_cache_group.cone_layer_names(l):  # 该组在层 l 的 prefix
            staircase_layer_metadata[layer_name] = md_narrow
forward_context.staircase_layer_metadata = staircase_layer_metadata
```

**关键设计决策**：
- **seq_len 不收窄，只收窄 query**：锥内 token 的 SWA/MLA K/V 仍从 cache 读 `[0, A)`（A 是 seq_len），
  只是把 query（本步要算的 token）限制在锥内 `[lo, A)`。这与全矩形路径语义一致——全矩形也是 query
  = 调度 token，K/V 读 cache。区别只是 query 范围从 `[B, A)` 缩到 `[lo, A)`。
- **slot_mapping/block_table 收窄**：锥内 token 的 slot 必须正确指向其 cache 槽（KV 写入用）。连续切片
  `slot_mapping[row_lo:row_hi]` 保持映射一致（slot 按 token 顺序排）。
- **query_start_loc 重置为 `[0, n_cone]`**：单请求，n_cone 个 query token。
- **每层独立 build**：避开缓存复用污染——`md_narrow` 是新对象，不碰 runner 的 `cached_attn_metadata`。
- **`staircase_layer_metadata` 经 ForwardContext 传到 model**（新字段，仿 `staircase_ab`）。

**model 侧（`DeepseekV4Model.forward` 层循环，`model.py:1077-1085`）**：

```python
fc = get_forward_context()
stair_md = getattr(fc, "staircase_layer_metadata", None)  # None 当 flag 关
a, b = (fc.staircase_ab[0].tolist() if fc.staircase_ab is not None else (None, None))
W = self.staircase_window
L_top = self.num_layers_total - 1
saved = {}  # 层前保存原 metadata 引用，层后还原

for i, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    l = self.start_layer + i  # 全局层号
    if stair_md is not None:
        lo = max(b, a - W * (L_top - l + 1))
        row_lo, row_hi = lo - b, a - b
        # 收窄 hidden_states/positions（连续切片）
        hs_narrow = hidden_states[row_lo:row_hi]
        pos_narrow = positions[row_lo:row_hi]
        # 替换该层所有 prefix 的 metadata（attn/swa/compressor/indexer/k_cache）
        for prefix in layer.all_prefixes:  # 层拥有的全部 forward_context key
            if prefix in stair_md:
                saved[prefix] = fc.attn_metadata[prefix]      # 保存原引用
                fc.attn_metadata[prefix] = stair_md[prefix]   # 注入收窄版
        hidden_states, residual, post_mix, res_mix = layer(
            hs_narrow, pos_narrow, input_ids[row_lo:row_hi] if input_ids is not None else None,
            post_mix, res_mix, residual,
        )
        # 还原（避免泄漏到下一层——下一层有自己收窄版，但 input_ids 等共享张量要还原引用）
        for prefix in saved:
            fc.attn_metadata[prefix] = saved[prefix]
        saved.clear()
        # NOTE: residual/post_mix/res_mix 现在是收窄子集长度，下一层会用更宽子集——
        # 见下"残差跨层对齐"问题
    else:
        hidden_states, residual, post_mix, res_mix = layer(
            hidden_states, positions, input_ids, post_mix, res_mix, residual,
        )
```

**残差跨层对齐（T4 的真实形态，承重）**：每层锥宽不同（input 端宽、output 端窄），但
`residual`/`post_mix`/`res_mix` 跨层传递。下一层（更靠 input，更宽）的子集是本层子集的**超集**
（向左扩张）。故跨层传递时：
- 本层算出 `[lo_l, A)` 的残差（长 `A - lo_l`）。
- 下一层要 `[lo_{l+1}, A)`（`lo_{l+1} < lo_l`，更宽）。下一层**左端新增** `[lo_{l+1}, lo_l)` 的 token，
  这些是锥外但下一层要算的——它们的 residual/post_mix/res_mix **本层没算过**。
- **解法**：下一层首层对这些新 token 走 `mhc_pre`（`residual=None` 路径，`model.py:872-887`）？
  **不行**——`mhc_pre` 只在整批 residual=None 时触发，且会重置全批。实际应：**残差张量始终按最宽
  （input 端）锥分配**，每层只在 `[lo_l, A)` 子集上写，左端 `[lo_{l+1}, lo_l)` 的残差由下一层自己算
  并填入同一全宽张量的对应行。

  ⇒ **正确的残差传递**：维护一个**全宽残差缓冲** `res_full`（宽 = input 端锥 = `[B, A)` 或
  `[lo_{end}, A)`），每层只计算并写入 `[lo_l, A)` 行；锥外行（`[lo_{end}, lo_l)`）**冻结沿用更靠 input
  层已算的值**。但 mhc kernel 是 per-token、grid 动态——可以对**全宽**张量调用 mhc，但只对锥内行传入
  有效 `x`（锥外行传 0 或上轮值），grid 仍跑全宽（不省 GEMM）。

  ⇒ **这暴露 T4 的真实约束**：mhc 能吃子集（已验证），但**残差跨层需要全宽缓冲 + 每层部分写入**，
  否则锥宽变化使残差长度不匹配。两种实现：
  - **(α) 每层全宽 mhc，锥外行冻结**：mhc 跑全宽 `[lo_end, A)`（不省 mhc 的 GEMM，只省 attn/ffn 的
    GEMM）。残差连续全宽，跨层对齐天然成立。**简单但省量打折**（mhc 的 GEMM 不省）。
  - **(β) 每层锥宽 mhc + scatter 回全宽残差**：mhc 跑 `[lo_l, A)`（省 mhc GEMM），结果 scatter 写回
    全宽 `res_full` 的 `[lo_l, A)` 行。**省量满但需 scatter + 全宽缓冲**。
  spike 阶段先 **(α)**（正确性优先），验证锥内输出一致后再优化到 (β)。

> **(α) 的隐含简化**：若每层 mhc 跑全宽，则 `hidden_states`/`residual` 其实不必收窄——只有
> `attn`/`ffn` 的 GEMM 收窄。但 `attn`/`ffn` 吃 `x`（mhc 输出），`x` 收窄则 mhc 输出也收窄……
> 矛盾。故 (α) 实际是：**mhc 全宽**，`x` 全宽，但 `attn`/`ffn` 只对锥内行算（锥外行 `x` 不参与
> attn/ffn 的 GEMM，但其 residual 仍由 mhc 更新）。这要求 attn/ffn 能跳过锥外行——回到 mask 问题。
> ⇒ **结论：纯 (α) 仍需 attn/ffn 按行 mask 或子集**。最干净是 **(β)**：每层整条流水（mhc+attn+ffn）
> 全部在锥宽 `[lo_l, A)` 上跑，残差用全宽缓冲 + scatter。这是 spike 要验证的核心机制。

**待 spike 验证项（T3/T4 合并）**：
1. 收窄 cm 重新 build 出的 SWA/compressor metadata，能否让锥内 attention 正确读 `[0,A)` 的 K/V 并只
   写锥内 token 的 K/V。
2. 残差全宽缓冲 + scatter 方案 (β) 的跨层对齐正确性。
3. 锥边界 SWA 自洽（前述"锥边界 SWA 写入语义"）。
4. flag 开/关（skip-writeback 两边开）锥内 token 输出逐位一致。

**T4 MegaMoE 残差处理** — mhc 三 kernel **本身**已验证可吃子集（grid 动态、零跨 token reduction、
hc 全局参数）。但 T3 详细设计暴露了 T4 的**真实难点不在 kernel 单层调用，在残差跨层对齐**：每层锥宽
不同（input 端宽、output 端窄），`residual`/`post_mix`/`res_mix` 跨层传递时长度不匹配。

**正确形态（见 T3 详细设计"残差跨层对齐"）**：维护**全宽残差缓冲**（宽 = input 端锥 `[lo_end, A)`），
每层只在锥宽 `[lo_l, A)` 上跑整条流水（mhc+attn+ffn），结果 scatter 写回全宽缓冲的对应行；锥外行
（`[lo_end, lo_l)`）冻结沿用更靠 input 层已算的值。两种实现：
- **(α) mhc 全宽、attn/ffn 收窄**：简单但 attn/ffn 仍需按行 mask 或子集，省量打折。**纯 (α) 退回
  mask 问题，不推荐**。
- **(β) 每层整条流水在锥宽跑 + scatter 回全宽残差**：省量满，需全宽缓冲 + scatter。**spike 采用 (β)**。

> mhc kernel 仍**不改 tilelang 源**——(β) 里 mhc 跑锥宽子集（已验证可行），scatter 是 caller 侧
> `res_full[lo_l-b : A-b] = mhc_out` 的张量拷贝。T4 工作量 = 全宽残差缓冲分配 + scatter 写回，
> 非改 kernel。

**待 spike 验证**（与 T3 合并，见上"待 spike 验证项"）：残差全宽缓冲 + scatter 的跨层对齐正确性、
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
   "delta 在单次 prefill 内"。注：A 仍只在首 chunk（`num_computed_tokens==0`）算一次（
   `get_computed_blocks` 用全 prompt 长度 `request.num_tokens-1`，首 chunk 即得全 prompt 的 A），
   跨 chunk 不变；但每 chunk 内可参与三角的 token 范围 = 该 chunk 与 `[B,A)` 的交集，三角边界
   需按 chunk 逐段定。这是 staircase 的实质复杂度，设计阶段需明确 chunk 化三角语义。
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

## 阶段划分

1. **阶段一**：Feature 1（skip-writeback），S1-S5，独立可落地，先合入。建立 A 透传基础设施。
   此阶段本身即有收益（省 `[B,A)` 边界 token 的 compress 聚合 + G0 写），且是阶段二的硬前置。
2. **阶段二**：Feature 2（staircase），**硬依赖阶段一**（`enable_staircase` 强制
   `enable_skip_writeback`）。T5a flag + T2 (A,B) 透传骨架**已实现**（commit `91c13e415`）。
   剩余 T3（层循环收窄）+ T4（残差跨层对齐）是**结构性难点**，先做 **B 阶段：搭可验证的最小 DSV4
   forward 基线**（本地 `DeepseekV4Config` + 随机权重，editable 安装已就绪），确认能跑通全矩形 forward
   并能开/关 flag，**再写 T3/T4 收窄逻辑**——否则 T3/T4 盲写无法验证"锥内逐位一致"。spike 验证项见
   T3 详细设计"待 spike 验证项"：收窄 metadata 正确性、残差全宽缓冲+scatter (β) 跨层对齐、锥边界 SWA
   自洽、flag 开/关锥内 token 逐位一致。先单请求（连续切片）、eager prefill、`FULL_DECODE_ONLY` 下
   验证，再扩多请求混批（需 gather+cat+contiguous）。
