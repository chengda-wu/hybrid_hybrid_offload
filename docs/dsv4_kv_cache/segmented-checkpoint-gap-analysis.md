# 分段 Token Checkpoint Gap 分析

本文独立记录分段 token checkpoint gap 的 Pareto 几何模型。基础方案将网络前 $l$ 层的 gap 设为 $g$，后 $L-l$ 层设为 $2g$；计算量按 hidden-state token-layer 单元计。

更一般的独立 $g_1/g_2$ 数值采样实验见 [`g1-g2-checkpoint-pareto.md`](./g1-g2-checkpoint-pareto.md)，三段 gap 的初步探测实现见 [`probe_checkpoint_g1_g2_g3.py`](./probe_checkpoint_g1_g2_g3.py)。

SWA 依赖每向浅层经过一层向前扩张

$$
d=W-1
$$

个 token。

## 1. 浅层计算量

令当前位置 $N$ 相对最近 $g$ checkpoint 的相位为

$$
g'=N\bmod g,
\qquad
g'\in\{0,1,\dots,g-1\}.
$$

在均匀相位假设下：

$$
\mathbb E[g']=\frac{g-1}{2},
\qquad
\mathbb E[(g')^2]=\frac{(g-1)(2g-1)}{6}.
$$

浅层计算区域近似为高 $l$、宽 $g'$ 的矩形，减去高 $g'/d$、底 $g'$ 的三角形：

$$
C_{\mathrm{low}}(g')
\approx lg'-\frac{(g')^2}{2d}.
$$

因此浅层平均计算量为

$$
\boxed{
\overline C_{\mathrm{low}}
=\frac{l(g-1)}{2}
-\frac{(g-1)(2g-1)}{12(W-1)}.
}
$$

## 2. 深层计算量

浅层 checkpoint 是 $g$ 的整数倍，因此进入第 $l+1$ 层时，相对 $2g$ checkpoint 网格的相位只有 $0$ 和 $g$，在一个 $2g$ 周期内各占一半。记 $k=L-l$：

$$
C_{\mathrm{high}}^{(0)}=2gk,
$$

$$
C_{\mathrm{high}}^{(g)}
\approx 2gk-\frac{g^2}{2(W-1)}.
$$

所以深层平均计算量为

$$
\boxed{
\overline C_{\mathrm{high}}
=2g(L-l)-\frac{g^2}{4(W-1)}.
}
$$

## 3. 总计算量与可行域

两部分相加得到绘图程序使用的计算量：

$$
\boxed{
\overline C(L,W,l,g)
=\frac{l(g-1)}{2}
+2g(L-l)
-\frac{5g^2-3g+1}{12(W-1)}.
}
$$

为了让两个被减去的三角形完整落在各自层段中，需要

$$
g-1\le l(W-1),
\qquad
g\le(L-l)(W-1).
$$

因此整数参数的可行域为

$$
\boxed{
1\le l\le L-1,
\qquad
1\le g\le
\min\{l(W-1)+1,(L-l)(W-1)\}.
}
$$

## 4. DRAM 存储量

设第 $i$ 层一个完整可续算 checkpoint 的大小为 $p_i$。对 checkpoint 相位平均后，前缀长度为 $N$ 时的 DRAM 存储量为

$$
\boxed{
\overline B(N,l,g)
=\frac{N}{g}\sum_{i=1}^{l}p_i
+\frac{N}{2g}\sum_{i=l+1}^{L}p_i.
}
$$

若所有层的 checkpoint 大小相同且均为 $p$，则简化为

$$
\overline B(N,l,g)=\frac{Np}{2g}(L+l).
$$

## 5. DeepSeek V4 Flash 参数代入

Pareto 曲线使用：

$$
L=43,
\qquad
W=128,
\qquad
d=127,
\qquad
N=1{,}000{,}000.
$$

完整可续算 checkpoint 的逐层字节数为：

| 层类型 | 层位置 | 层数 | $p_i$ |
|---|---|---:|---:|
| SWA-only | 1–2 | 2 | 74880 B |
| C4 | 3, 5, ..., 43 | 21 | 157824 B |
| C128 | 4, 6, ..., 42 | 20 | 600192 B |

全部 43 层在同一个 token 位置上的 checkpoint 总大小为

$$
P=15467904\ \text{B}.
$$

记前 $l$ 层 checkpoint 字节数之和为 $P_l$。当 $l\ge2$ 时：

$$
P_l
=149760
+157824\left\lceil\frac{l-2}{2}\right\rceil
+600192\left\lfloor\frac{l-2}{2}\right\rfloor,
$$

而 $P_1=74880$。于是 DSV4 Flash 的存储量可以写成

$$
\boxed{
\overline B_{\mathrm{DSV4}}(l,g)
=\frac{10^6}{2g}(P+P_l)\ \text{B}.
}
$$

换算为 GiB：

$$
\boxed{
\overline B_{\mathrm{DSV4,GiB}}(l,g)
=\frac{10^6(P+P_l)}{2g\cdot2^{30}}.
}
$$

计算量为

$$
\boxed{
\overline C_{\mathrm{DSV4}}(l,g)
=\frac{l(g-1)}{2}
+2g(43-l)
-\frac{5g^2-3g+1}{1524}.
}
$$

参数扫描范围为

$$
1\le l\le42,
\qquad
1\le g\le\min\{127l+1,127(43-l)\}.
$$

$N=10^6$ 只进入存储公式；对相位平均后的计算量不依赖绝对位置 $N$。

## 6. Pareto 判定

对每个可行整数参数对 $(l,g)$ 计算

$$
(B,C)=
(\overline B_{\mathrm{DSV4,GiB}},
\overline C_{\mathrm{DSV4}}).
$$

若不存在另一个可行参数对 $(l',g')$ 同时满足

$$
B(l',g')\le B(l,g),
\qquad
C(l',g')\le C(l,g),
$$

且至少一个不等式严格成立，则 $(l,g)$ 是 Pareto 最优点。上述可行域共生成 58695 个采样点，其中 3164 个为 Pareto 点。数据和绘图实现见 [`dsv4_checkpoint_points.csv`](./dsv4_checkpoint_points.csv) 与 [`plot_checkpoint_pareto.py`](./plot_checkpoint_pareto.py)。
