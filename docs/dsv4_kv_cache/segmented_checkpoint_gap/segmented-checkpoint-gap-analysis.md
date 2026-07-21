# 分段 Token Checkpoint Gap 分析

考虑两段 token checkpoint gap。层号从输入侧向输出侧递增：Layer $1$ 是最浅层，Layer $L$ 是最深层。分割位置为 $l$；远离输出端的 Layer $1,\dots,l$ 使用较大的 gap $g_2$，靠近输出端的 Layer $l+1,\dots,L$ 使用较小的 gap $g_1$，其中 $g_1\le g_2$，但不要求 $g_2$ 是 $g_1$ 的整数倍。因此 $l$ 越大，分割位置越深、越靠近输出侧；输出侧 $g_1$ 段的层数为 $L-l$。计算量按 hidden-state token-layer 单元计。

基于本文固定 $N$ 离散公式的参数搜索、Pareto 数据和图表见 [`g1-g2-checkpoint-pareto.md`](./g1-g2-checkpoint-pareto.md)。

SWA 依赖每向浅层经过一层向前扩张

$$
d=W-1
$$

个 token。

## 1. 给定 $N$ 的恢复计算量

所有计算按 layer-token 格点计数。定义高度为 $h$、相位为 $r$ 时被减去的离散三角形或梯形：

$$
T_h(r)=\sum_{j=1}^{h}(r-jd)_+,
\qquad(x)_+=\max\{x,0\}.
$$

令

$$
m_h(r)=\min\left\{h,\max\left(0,\left\lfloor\frac{r-1}{d}\right\rfloor\right)\right\},
$$

则

$$
\boxed{
T_h(r)=m_h(r)r-d\frac{m_h(r)(m_h(r)+1)}2.
}
$$

### 1.1 输出侧 $g_1$ 段（恢复时先经过）

恢复依赖从输出侧向输入侧回溯，所以首先经过使用 $g_1$ 的深层段。记该段高度

$$
h=L-l,
$$

并令当前位置 $N$ 相对 $g_1$ checkpoint 网格的相位为

$$
r_1=N\bmod g_1.
$$

从最深层向浅层回溯 $j$ 层后，边界最多扩展 $jd$，但到达最近的 $g_1$ checkpoint 后停止，所以该层的恢复宽度是 $\min(r_1,jd)$。输出侧段计算量为

$$
\boxed{
C_1(N)
=\sum_{j=1}^{h}\min(r_1,jd)
=hr_1-T_h(r_1).
}
$$

输出侧段结束、进入浅层段时的边界位置为

$$
\boxed{
b_1=N-\min(r_1,hd).
}
$$

其中 $r_1\le hd$ 表示边界命中 $g_1$ checkpoint；若 $r_1>hd$，则边界只到达 $N-hd$，并且

$$
C_1(N)=d\frac{h(h+1)}2.
$$

### 1.2 输入侧 $g_2$ 段（恢复时后经过）

输入侧段的进入相位必须由输出侧段的结束边界计算：

$$
\boxed{
r_2=b_1\bmod g_2
=\left[N-\min(N\bmod g_1,(L-l)d)\right]\bmod g_2.
}
$$

输入侧段相对最近 $g_2$ checkpoint 的基准恢复长度为

$$
\boxed{
a_2
=N-\left\lfloor\frac{b_1}{g_2}\right\rfloor g_2
=\min(r_1,(L-l)d)+r_2.
}
$$

输入侧段第 $j$ 层的恢复宽度为

$$
a_2-(r_2-jd)_+.
$$

因此输入侧段计算量为

$$
\boxed{
C_2(N)
=\sum_{j=1}^{l}\left[a_2-(r_2-jd)_+\right]
=la_2-T_l(r_2).
}
$$

输入侧段命中 checkpoint 当且仅当 $r_2\le ld$。若 $r_2>ld$，则

$$
C_2(N)=l\min(r_1,(L-l)d)+d\frac{l(l+1)}2.
$$

特别地，$r_2=0$ 时 $C_2=l\min(r_1,(L-l)d)$。

### 1.3 总计算量

给定 $N$ 后按恢复方向依次计算 $r_1,b_1,r_2$，总恢复计算量为

$$
\boxed{
\begin{aligned}
r_1&=N\bmod g_1,\\
b_1&=N-\min(r_1,(L-l)d),\\
r_2&=b_1\bmod g_2,\\
C(N,l,g_1,g_2)
&=(L-l)r_1-T_{L-l}(r_1)\\
&\quad +l\left(N-\left\lfloor\frac{b_1}{g_2}\right\rfloor g_2\right)
-T_l(r_2).
\end{aligned}
}
$$

这个公式同时覆盖两段命中和未命中的情况，不再需要完整三角形约束。输入侧相位 $r_2$ 与 $N$、输出侧相位 $r_1$ 以及输出侧段是否命中共同相关，因此不能分别对两个相位取期望后再代入；对于给定 $N$ 应直接使用上式。

## 2. $g_1=g,\ g_2=2g$ 的相位平均特例

旧绘图程序额外要求两段都能命中 checkpoint。输出侧 $g_1$ 段的相位 $r=N\bmod g$ 在 $\{0,1,\dots,g-1\}$ 上均匀分布，因此

$$
\boxed{
\overline C_{\mathrm{near}}
=\frac{(L-l)(g-1)}2
-\frac1g\sum_{r=0}^{g-1}T_{L-l}(r).
}
$$

输出侧段命中后，$b_1$ 是 $g$ 的整数倍，其相对 $2g$ 网格的相位只有 $0$ 和 $g$，在一个 $2g$ 周期内各占一半。对给定的 $r$，输入侧段的基准恢复长度分别为 $r$ 和 $r+g$，因此

$$
C_{\mathrm{far}}^{(0)}(r)=lr,
\qquad
C_{\mathrm{far}}^{(g)}(r)=l(r+g)-T_l(g),
$$

从而

$$
\boxed{
\overline C(L,W,l,g)
=\frac{(L-l)(g-1)}2
+\frac{l(2g-1)}2
-\frac1g\sum_{r=0}^{g-1}T_{L-l}(r)
-\frac{T_l(g)}2.
}
$$

旧扫描为了让两个被减去的三角形完整落在各自层段中，限制

$$
g-1\le (L-l)(W-1),
\qquad
g\le l(W-1).
$$

其整数参数域为

$$
\boxed{
1\le l\le L-1,
\qquad
1\le g\le
\min\{(L-l)(W-1)+1,l(W-1)\}.
}
$$

## 3. DRAM 存储量

设第 $i$ 层一个完整可续算 checkpoint 的大小为 $p_i$。前缀长度为整数 $N$ 时，checkpoint 数量也按整数计数：

$$
\boxed{
B(N,l,g_1,g_2)
=\left\lceil\frac{N}{g_2}\right\rceil\sum_{i=1}^{l}p_i
+\left\lceil\frac{N}{g_1}\right\rceil\sum_{i=l+1}^{L}p_i.
}
$$

若所有层的 checkpoint 大小相同且均为 $p$，则简化为

$$
B(N,l,g_1,g_2)
=p\left[
l\left\lceil\frac N{g_2}\right\rceil
+(L-l)\left\lceil\frac N{g_1}\right\rceil
\right].
$$

取 $g_1=g,\ g_2=2g$ 即退化为旧扫描使用的存储公式。

## 4. DeepSeek V4 Flash 参数代入

固定模型与前缀参数为：

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
| C4 | 1, 3, ..., 41 | 21 | 157824 B |
| C128 | 2, 4, ..., 40 | 20 | 600192 B |
| SWA-only | 42–43（输出侧最深两层） | 2 | 74880 B |

全部 43 层在同一个 token 位置上的 checkpoint 总大小为

$$
P=15467904\ \text{B}.
$$

记输入侧前 $l$ 层 checkpoint 字节数之和为 $P_l$。当 $1\le l\le41$ 时：

$$
P_l
=157824\left\lceil\frac l2\right\rceil
+600192\left\lfloor\frac l2\right\rfloor.
$$

此外 $P_{42}=P_{41}+74880$。于是给定 $N,g_1,g_2$ 时，DSV4 Flash 的离散存储量为

$$
\boxed{
B_{\mathrm{DSV4}}(N,l,g_1,g_2)
=\left\lceil\frac{N}{g_2}\right\rceil P_l
+\left\lceil\frac{N}{g_1}\right\rceil(P-P_l)\ \text{B}.
}
$$

换算为 GiB：

$$
\boxed{
B_{\mathrm{DSV4,GiB}}(N,l,g_1,g_2)
=\frac{
\left\lceil N/g_2\right\rceil P_l
+\left\lceil N/g_1\right\rceil(P-P_l)
}{2^{30}}.
}
$$

给定 $N$ 的计算量为

$$
\boxed{
\begin{aligned}
r_1&=N\bmod g_1,\\
b_1&=N-\min(r_1,127(43-l)),\\
r_2&=b_1\bmod g_2,\\
C_{\mathrm{DSV4}}(N,l,g_1,g_2)
&=(43-l)r_1-T_{43-l}(r_1)\\
&\quad +l\left(N-\left\lfloor\frac{b_1}{g_2}\right\rfloor g_2\right)
-T_l(r_2).
\end{aligned}
}
$$

### 4.1 旧 $g,2g$ Pareto 扫描

现有数据和绘图使用 $N=10^6$、$g_1=g$、$g_2=2g$，并对相位平均。此时存储量为

$$
\boxed{
B_{\mathrm{DSV4}}(l,g)
=\left\lceil\frac{10^6}{2g}\right\rceil P_l
+\left\lceil\frac{10^6}{g}\right\rceil(P-P_l)\ \text{B},
}
$$

计算量为

$$
\boxed{
\overline C_{\mathrm{DSV4}}(l,g)
=\frac{(43-l)(g-1)}2
+\frac{l(2g-1)}2
-\frac1g\sum_{r=0}^{g-1}T_{43-l}(r)
-\frac{T_l(g)}2.
}
$$

参数扫描范围为

$$
1\le l\le42,
\qquad
1\le g\le\min\{127(43-l)+1,127l\}.
$$

$N=10^6$ 只进入该旧扫描的存储公式；相位平均后的计算量不依赖绝对位置 $N$。对于一般的 $g_1,g_2$，应使用前述给定 $N$ 的公式直接计算。

## 5. Pareto 判定

对每个可行整数参数对 $(l,g)$ 计算

$$
(B,C)=
(B_{\mathrm{DSV4,GiB}},
\overline C_{\mathrm{DSV4}}).
$$

若不存在另一个可行参数对 $(l',g')$ 同时满足

$$
B(l',g')\le B(l,g),
\qquad
C(l',g')\le C(l,g),
$$

且至少一个不等式严格成立，则 $(l,g)$ 是 Pareto 最优点。上述可行域共生成 58695 个参数点，其中 1882 个为 Pareto 点。数据见 [`dsv4_checkpoint_points.csv`](./dsv4_checkpoint_points.csv)，交互图见 [`dsv4_checkpoint_pareto_interactive.html`](./dsv4_checkpoint_pareto_interactive.html)，静态图见 [`imgs/dsv4-checkpoint-pareto-discrete.png`](./imgs/dsv4-checkpoint-pareto-discrete.png)，生成程序见 [`plot_checkpoint_pareto.py`](./plot_checkpoint_pareto.py)。
