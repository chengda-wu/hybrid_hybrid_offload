# DSV4 Flash $g_1/g_2$ Token Checkpoint Pareto 实验

## 1. 实验口径

DeepSeek V4 Flash 使用

$$
L=43,
\qquad
W=128,
\qquad
d=W-1=127.
$$

Layer $1,\dots,l$ 使用 token checkpoint gap $g_1$，Layer $l+1,\dots,L$ 使用 gap $g_2$，并要求

$$
1\le l\le42,
\qquad
1\le g_1\le g_2.
$$

所有候选使用同一个 prefix workload：整数 $N$ 在

$$
[1{,}000{,}000,2{,}000{,}000]
$$

上均匀分布。Pareto 判定只使用两个 metric：

$$
(\overline B,\overline C)
=
(\text{平均 DRAM checkpoint 存储量},
\text{平均恢复 token-layer 数}).
$$

不使用 P95、P99 或其它尾部指标。

## 2. 单个 $N$ 的恢复计算量

定义

$$
T_h(r)=\sum_{j=1}^{h}(r-jd)_+,
$$

以及两个相位和第一段结束边界

$$
r_1=N\bmod g_1,
$$

$$
b_1=N-\min(r_1,ld),
$$

$$
r_2=b_1\bmod g_2.
$$

实验使用精确离散公式

$$
\boxed{
C(N,l,g_1,g_2)
=lr_1-T_l(r_1)
+(L-l)g_2-T_{L-l}(r_2).
}
$$

该公式不要求任一段在给定 $N$ 下命中 checkpoint，也不要求 $g_2$ 是 $g_1$ 的整数倍。

恢复量均值使用固定随机种子，在相同的 $N$ 集合上计算。三个搜索阶段使用嵌套样本：

| 阶段 | 共同 $N$ 样本数 |
|---|---:|
| 粗搜索 | 1024 |
| 中等精化 | 8192 |
| 最终精化 | 32768 |

样本从完整 workload 中无放回均匀抽取，并固定包含两个端点。所有候选共享同一批样本，避免候选特有的 LCM 区间改变 workload。

## 3. 平均存储量

给定 $N$ 时

$$
B(N,l,g_1,g_2)
=\left\lceil\frac{N}{g_1}\right\rceil P_l
+\left\lceil\frac{N}{g_2}\right\rceil(P-P_l),
$$

其中 $P_l$ 是前 $l$ 层单个 checkpoint 的总字节数，全部 43 层为

$$
P=15467904\ \text{B}.
$$

平均存储量没有采样，而是在全部 $1{,}000{,}001$ 个整数 $N$ 上精确计算。定义

$$
F_g(x)=\sum_{N=1}^{x}\left\lceil\frac Ng\right\rceil.
$$

若 $x=qg+r$ 且 $0\le r<g$，则

$$
F_g(x)=g\frac{q(q+1)}2+(q+1)r.
$$

因此每个 gap 的平均 checkpoint 数可以在 $O(1)$ 时间内得到，再按 $P_l$ 和 $P-P_l$ 组合成 $\overline B$。

## 4. 参数搜索

本次运行设置最大 gap 为 8192，并限制平均存储量不超过 1024 GiB。为了避免对数亿个整数 gap pair 分别枚举完整 workload，使用三级搜索：

1. $1\le g\le256$ 逐整数枚举；之后使用步长 16 的粗网格，并额外加入所有层段命中边界 $hd+1$ 和 2 的幂。
2. 保留各 $l$ 的局部 Pareto 点，并在全局前沿 2% 计算量范围内的点周围做半径 8 的完整整数精化。
3. 在中等精化前沿 0.5% 范围内的点周围做半径 3 的完整整数精化，最终用 32768 个共同 $N$ 样本重新计算。
4. 最终候选额外包含上一版已发布代表点的半径 8 整数邻域，避免提高搜索分辨率后因筛选路径变化丢失已有较优点。

每个 $l$ 先做局部 Pareto 剪枝，再合并求全局前沿。最终 HTML 只包含全局 Pareto 点，避免把数百万个候选嵌入浏览器。

## 5. 运行统计

| 指标 | 数值 |
|---|---:|
| 粗 gap 网格大小 | 790 |
| 粗搜索、存储过滤后的参数点 | 12,791,827 |
| 粗搜索局部 Pareto 点 | 91,130 |
| 中等精化参数 | 1,607,836 |
| 最终精化参数 | 499,453 |
| 上一版代表点邻域回归保护参数 | 1,995 |
| 最终 Pareto 点 | 21,213 |
| HTML 候选云采样点 | 12,600 |
| 总运行时间 | 228.71 s |

本次运行的随机种子为 `20260721`。详细配置和各阶段耗时见 [`dsv4_checkpoint_g1_g2_metadata.json`](./dsv4_checkpoint_g1_g2_metadata.json)。

## 6. 代表性 Pareto 点

下表是在给定平均存储上限下，最终前沿中平均恢复计算量最低的点：

| 平均存储上限 | $g_1$ | $g_2$ | $l$ | 平均存储 | 平均恢复计算量 |
|---:|---:|---:|---:|---:|---:|
| 4 GiB | 6115 | 6118 | 37 | 3.541 GiB | 75989.92 |
| 8 GiB | 2746 | 2747 | 39 | 7.876 GiB | 47684.81 |
| 16 GiB | 1351 | 1396 | 42 | 15.996 GiB | 27001.92 |
| 32 GiB | 677 | 678 | 42 | 31.925 GiB | 14168.01 |
| 64 GiB | 338 | 341 | 42 | 63.932 GiB | 7295.35 |
| 128 GiB | 169 | 170 | 42 | 127.860 GiB | 3687.57 |
| 256 GiB | 85 | 85 | 42 | 254.224 GiB | 1855.11 |
| 512 GiB | 43 | 43 | 42 | 502.528 GiB | 925.40 |
| 1024 GiB | 21 | 24 | 41 | 1022.676 GiB | 458.65 |

## 7. 输出

- Pareto 数据：[`dsv4_checkpoint_g1_g2_pareto.csv`](./dsv4_checkpoint_g1_g2_pareto.csv)
- 静态图：[`imgs/dsv4-checkpoint-g1-g2-pareto.png`](./imgs/dsv4-checkpoint-g1-g2-pareto.png)
- 交互图：[`dsv4_checkpoint_g1_g2_pareto_interactive.html`](./dsv4_checkpoint_g1_g2_pareto_interactive.html)
- 生成程序：[`plot_checkpoint_g1_g2_pareto.py`](./plot_checkpoint_g1_g2_pareto.py)

静态图和交互图均延续旧图口径：横轴为平均存储量 $\overline B$，纵轴为平均恢复计算量 $\overline C$，两轴均使用对数刻度。交互图同时显示下采样候选云、Pareto 点和当前选择，并提供 $l,g_1,g_2$ 控件。
