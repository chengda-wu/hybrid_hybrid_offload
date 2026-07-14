# 计划：在 vLLM 中实现 DSV4 skip-writeback 与 staircase 重算

> 承接 §10–§16（`06-staircase-delta-feasibility.md`）。两个技术点数学独立、工程分层：
>
> - **Feature 1 skip-writeback**：闸掉 `[B,A)` 冗余的 G0 写回。小改动、默认关、逐位一致，**先行**。
> - **Feature 2 staircase**：把 delta 段重算从全矩形 `D×L` 改成逐层递减三角形，省 ~50%
>   hidden-state GEMM。**数学可行**（compress 是固定窗口聚合，见
>   [核验](#staircase-数学可行性核验)），主要工程障碍在 MegaMoE 残差融合 kernel，**第二阶段**。
>
> Feature 2 复用 Feature 1 建立的 per-request A 透传。

## 背景

DSV4 有 5 个 KV cache group，APC 下**命中分化**：主 MLA 压缩组（G0，全序列）命中到 `A`，
SWA 与 compressor 组（滑窗）只命中到 `B < A`。全局最小共识（`find_longest_cache_hit`，
`kv_cache_coordinator.py:630`）取 `num_computed_tokens = B`，故 delta 段 `[B,A)` 的 G0 条目
**已缓存**却仍进 forward。记 `D = A - B`。

关键事实（均已核验，附行号）：

- `A`、`B` 均 **block 对齐**（G0 block_size=256）。`find_longest_cache_hit` 对 Full attention
  做 `curr_hit_length // block_size * block_size`（`kv_cache_coordinator.py:687-689`），返回
  `hit_length = B`；迭代中的 `longest_hit_length = A`（L719、L737）**未返回**，需暴露。
- compressor `forward`（`compressor.py:274-399`）对 delta token 无条件跑两 kernel：
  `save_partial_states`（L318，写 state cache，滑窗，`[B,A)` 确实缺失**必跑**）+
  `compress_norm_rope_store`（L375，G0 写回，`[B,A)` **冗余**）。
- `compress_norm_rope_store` 第一参数 `slot_mapping_ptr` **只**用于 `slot_id<0` 早退
  （`fused_compress_quant_cache.py:159`）。
- `CommonAttentionMetadata` 已有 `positions` 字段（`backend.py:436`），gate 构建可直接用。

---

## Feature 1：skip-writeback

### 机制

引入独立 gate 张量 `compress_gate_slot_mapping`（不改 kernel、不改 attention）：

- = 未压缩 slot 副本，把每个 `position < A` 的 token 置 `-1`。
- 作为 `compress_norm_rope_store` 的 `slot_mapping` 传入（`compressor.py:380`）。
- `save_partial_states`（`compressor.py:324`）仍用原 slot，**不变**。

结果：`compress_norm_rope_store` 对 `[B,A)` token 早退；`save_partial_states` 照跑；attention
及其 compressed slot 不动。

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
捕获 A，flag 开启时设 `request.g0_hit_length = A`。**A 在 request 生命周期内固定**：首次调度
时 `get_computed_blocks` 设定，后续 chunk / running 步不再变（G0 是全序列缓存，命中长度不随
请求推进改变）。scheduler.py 无逻辑改动（DSV4 走常规路径，scheduler.py:712）。

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

## staircase 数学可行性核验

早先据 `save_partial_states` 判 staircase 不可行（旧 Blocker 2）。核验 compress kernel 后**推翻**。

### 证据：compress 是固定窗口聚合

`fused_compress_quant_cache.py` 三个 kernel 对边界 token `position` 的状态聚合窗口**固定大小**
（L169-172，三 kernel 一致）：

```python
start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)   # 固定窗口宽
```

C4 窗口 ≈8，C128 ≈256（与 SWA 128 同量级）。`save_partial_states`（`save_partial_states.py:68-90`）
one program per token，只写自己那格，零跨 token 读。

### 依赖推导

compressor 状态窗口（≤256）≤ SWA 窗口（128 量级），不拓宽锥。锥由最宽窗口 W=128 决定。自顶向
下（顶层 L=42，底层 L=0），层 L 需要的 token 下界 `p >= A - W·(L_top - L + 1)`：

- 底层 L=0：`[max(B, A-W_eff), A)`，宽 ≈ W_eff = 128+42·127 ≈ 5462。
- 顶层 L=42：`[A-128, A)`，宽 128。

**越浅算越多、越深算越少**，三角成立，省 ~50% hidden-state GEMM（与 §15.3.1 一致）。

### 旧 Blocker 2 为何不成立

旧文混淆了"state cache 跨请求命中到 B"（前缀缓存属性，只意味 `[B,A)` state 本次要写）与"本次
forward 内全层穿透"（依赖主张，**错**）。本次 forward 内依赖是滑窗的，锥逐层收窄。staircase
跳过的是锥外 token 的 hidden-state 计算；锥内 token 的 `save_partial_states` 照常写。两者自洽。

---

## Feature 2：staircase

### 目标

delta `[B,A)` 重算从全矩形 `D×L` 改成逐层递减三角形：底层算 ~W_eff token，顶层算 ~128，省
~50% hidden-state GEMM。

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
`DeepseekV4Model.forward`。仅 `enable_skip_writeback` 也开时生效。默认关 → 全矩形 → 逐位一致。

### 工程约束

1. **CUDA graph 静态形状**：三角 token 数依赖 D=A−B（每请求不同）→ 不能静态 capture。但
   staircase 是 prefill 期优化，默认 `CUDAGraphMode.FULL_DECODE_ONLY=(FULL,NONE)`
   （`compilation.py:62`）下 **prefill 走 NONE（eager）**，无约束。**限制**：与 `FULL`
   （prefill 也 capture）不兼容，flag 检查里拒绝 `FULL` 或回退全矩形。
2. **多请求混批**：边界 per-request，收窄后 token 集合需在 batch 维重拼 `query_start_loc`
  （批内 D 不同 → 总 token 数动态）。eager prefill 下可接受。
3. **C4 窗口(8)被 W=128 拉平**：统一用 W=128 定锥，C4 本可更窄但被拉平，省量略低于上界。
4. **chunked prefill**：DSV4 当前 no chunked prefill（仓库 CLAUDE.md）。先限定 delta 在单次
   prefill 内。

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
2. **阶段二**：Feature 2（staircase）。**先做 T4 可行性 spike**（mhc 残差子集/scatter），spike
   通过再推进 T1-T5。先单请求、eager prefill、`FULL_DECODE_ONLY` 下验证，再扩多请求混批。
