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

收窄策略（自顶向下，请求 r）：距顶层 k 层只算 `[max(B, A-W·(k+1)), A)`，每层 token 集合是下层
的前缀，保证 hidden state 层间传递连续；底层 = 全 `[B,A)`，与既有全矩形在底层对齐。

### 主要工程障碍：MegaMoE 残差融合 kernel（非绝墙）

核验 `DeepseekV4DecoderLayer.forward`（`model.py:861-933`）发现真正难点**不在 attention 层**，
在残差流：

- 层状态 `residual`/`post_mix`/`res_mix` 跨层传递，由 `mhc_fused_post_pre_tilelang`
  （L889、L913）**按整张 tile 融合更新**。
- `x = self.attn(positions, x)`（L909）→ 残差融合（L913）→ `self.ffn(x)`（L932）共用同一 token
  集。若某层只对锥内 token 算 attn，输出 `x` 变短，但 `residual` 是全张量 →
  `mhc_fused_post_pre_tilelang(x_short, residual_full, ...)` 形状不匹配。
- 故"按层收窄"须**同时收窄 `residual`/`post_mix`/`res_mix`** 并保持与 `x` 同子集，即锥外 token
  的残差流要"冻结"沿用上层值。而 `mhc_*_tilelang` 按整张 tile 操作，无法 cheaply 跳过锥外 token。

**结论**：staircase 需要 (a) 一个支持 token 子集 / mask 的 `mhc_*_tilelang` 变体，或 (b) 对锥内
子集单独算 mhc 再 scatter 写回全张量残差。这是主要工作量所在，但**不改变数学可行性**。

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

**T4 MegaMoE 残差处理** — 解决 T3 的残差形状问题：锥内子集算 `mhc_*` 后 scatter 写回全张量
`residual`/`post_mix`/`res_mix`（锥外 token 残差冻结）。需评估是写 mhc mask 变体还是子集重算 +
scatter（前者性能好但改 tilelang kernel，后者实现快但有 scatter 开销）。**T4 是主要风险点**，
设计阶段需先做可行性 spike。

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
  （批内 D 不同 → 总 token 数动态）。eager prefill 下可接受。
3. **C4 窗口(8)被 W=128 拉平**：统一用 W=128 定锥，C4 本可更窄但被拉平，省量略低于上界。
4. **chunked prefill（真实 vLLM 默认开）**：`enable_chunked_prefill=True`（`scheduler.py:84`），
   DSV4 不禁用。仓库 CLAUDE.md 的"no chunked prefill"指 **simulator**，不是真实引擎。故 delta
   `[B,A)` **可能跨多个 prefill chunk**，staircase 三角边界须按 chunk 起止切分，不能假设
   "delta 在单次 prefill 内"。注：A 仍只在首 chunk（`num_computed_tokens==0`）算一次（
   `get_computed_blocks` 用全 prompt 长度 `request.num_tokens-1`，首 chunk 即得全 prompt 的 A），
   跨 chunk 不变；但每 chunk 内可参与三角的 token 范围 = 该 chunk 与 `[B,A)` 的交集，三角边界
   需按 chunk 逐段定。这是 staircase 的实质复杂度，设计阶段需明确 chunk 化三角语义。

### 待改文件（Feature 2）

- `vllm/config/cache.py` — `enable_staircase` flag。
- `vllm/models/deepseek_v4/nvidia/model.py` — `DeepseekV4Model.forward` 层循环按层收窄 + 注入。
- `vllm/models/deepseek_v4/nvidia/model.py`（mhc kernel 调用处）/ 对应 tilelang 源 — T4 残差
  子集/scatter 支持。
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
   `enable_skip_writeback`）。**先做 T4 可行性 spike**（mhc 残差子集/scatter），spike 通过再推进
   T1-T5。先单请求、eager prefill、`FULL_DECODE_ONLY` 下验证，再扩多请求混批。
