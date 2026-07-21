# 分段 Token Checkpoint Gap 分析

考虑两段 token checkpoint gap：前 $l$ 层使用 $g_1$，后 $L-l$ 层使用 $g_2$，其中 $g_1\le g_2$，但不要求 $g_2$ 是 $g_1$ 的整数倍。计算量按 hidden-state token-layer 单元计。

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

### 1.1 第一段

令当前位置 $N$ 相对 $g_1$ checkpoint 网格的相位为

$$
r_1=N\bmod g_1.
$$

经过第 $j$ 层后，边界最多扩展 $jd$，但到达最近的 $g_1$ checkpoint 后停止，所以该层的恢复宽度是 $\min(r_1,jd)$。第一段计算量为

$$
\boxed{
C_1(N)
=\sum_{j=1}^{l}\min(r_1,jd)
=lr_1-T_l(r_1).
}
$$

第一段结束时的边界位置为

$$
\boxed{
b_1=N-\min(r_1,ld).
}
$$

其中 $r_1\le ld$ 表示边界命中 $g_1$ checkpoint；若 $r_1>ld$，则边界只到达 $N-ld$，并且

$$
C_1(N)=d\frac{l(l+1)}2.
$$

### 1.2 第二段

第二段的进入相位必须由第一段结束边界计算：

$$
\boxed{
r_2=b_1\bmod g_2
=\left[N-\min(N\bmod g_1,ld)\right]\bmod g_2.
}
$$

记 $k=L-l$。沿用本文原有的深层 checkpoint 单元计数约定，第二段第 $j$ 层的恢复宽度为

$$
g_2-r_2+\min(r_2,jd)
=g_2-(r_2-jd)_+.
$$

因此第二段计算量为

$$
\boxed{
C_2(N)
=\sum_{j=1}^{k}\left[g_2-(r_2-jd)_+\right]
=kg_2-T_k(r_2).
}
$$

第二段命中 checkpoint 当且仅当 $r_2\le kd$。若 $r_2>kd$，则

$$
C_2(N)=k(g_2-r_2)+d\frac{k(k+1)}2.
$$

特别地，$r_2=0$ 时 $C_2=kg_2$，这与后文旧特例中的 $C_{\mathrm{high}}^{(0)}=2gk$ 一致。

### 1.3 总计算量

给定 $N$ 后依次计算 $r_1,b_1,r_2$，总恢复计算量为

$$
\boxed{
\begin{aligned}
r_1&=N\bmod g_1,\\
b_1&=N-\min(r_1,ld),\\
r_2&=b_1\bmod g_2,\\
C(N,l,g_1,g_2)
&=lr_1-T_l(r_1)+(L-l)g_2-T_{L-l}(r_2).
\end{aligned}
}
$$

这个公式同时覆盖两段命中和未命中的情况，不再需要完整三角形约束。第二段相位 $r_2$ 与 $N$、第一段相位和第一段是否命中共同相关，因此不能分别对两个相位取期望后再代入；对于给定 $N$ 应直接使用上式。

## 2. $g_1=g,\ g_2=2g$ 的相位平均特例

旧绘图程序额外要求两段都能命中 checkpoint。第一段相位 $r=N\bmod g$ 在 $\{0,1,\dots,g-1\}$ 上均匀分布，因此

$$
\boxed{
\overline C_{\mathrm{low}}
=\frac{l(g-1)}2
-\frac1g\sum_{r=0}^{g-1}T_l(r).
}
$$

第一段命中后，$b_1$ 是 $g$ 的整数倍，其相对 $2g$ 网格的相位只有 $0$ 和 $g$，在一个 $2g$ 周期内各占一半。因此

$$
C_{\mathrm{high}}^{(0)}=2g(L-l),
\qquad
C_{\mathrm{high}}^{(g)}=2g(L-l)-T_{L-l}(g),
$$

从而

$$
\boxed{
\overline C(L,W,l,g)
=\frac{l(g-1)}2
+2g(L-l)
-\frac1g\sum_{r=0}^{g-1}T_l(r)
-\frac{T_{L-l}(g)}2.
}
$$

旧扫描为了让两个被减去的三角形完整落在各自层段中，限制

$$
g-1\le l(W-1),
\qquad
g\le(L-l)(W-1).
$$

其整数参数域为

$$
\boxed{
1\le l\le L-1,
\qquad
1\le g\le
\min\{l(W-1)+1,(L-l)(W-1)\}.
}
$$

## 3. DRAM 存储量

设第 $i$ 层一个完整可续算 checkpoint 的大小为 $p_i$。前缀长度为整数 $N$ 时，checkpoint 数量也按整数计数：

$$
\boxed{
B(N,l,g_1,g_2)
=\left\lceil\frac{N}{g_1}\right\rceil\sum_{i=1}^{l}p_i
+\left\lceil\frac{N}{g_2}\right\rceil\sum_{i=l+1}^{L}p_i.
}
$$

若所有层的 checkpoint 大小相同且均为 $p$，则简化为

$$
B(N,l,g_1,g_2)
=p\left[
l\left\lceil\frac N{g_1}\right\rceil
+(L-l)\left\lceil\frac N{g_2}\right\rceil
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

而 $P_1=74880$。于是给定 $N,g_1,g_2$ 时，DSV4 Flash 的离散存储量为

$$
\boxed{
B_{\mathrm{DSV4}}(N,l,g_1,g_2)
=\left\lceil\frac{N}{g_1}\right\rceil P_l
+\left\lceil\frac{N}{g_2}\right\rceil(P-P_l)\ \text{B}.
}
$$

换算为 GiB：

$$
\boxed{
B_{\mathrm{DSV4,GiB}}(N,l,g_1,g_2)
=\frac{
\left\lceil N/g_1\right\rceil P_l
+\left\lceil N/g_2\right\rceil(P-P_l)
}{2^{30}}.
}
$$

给定 $N$ 的计算量为

$$
\boxed{
\begin{aligned}
r_1&=N\bmod g_1,\\
b_1&=N-\min(r_1,127l),\\
r_2&=b_1\bmod g_2,\\
C_{\mathrm{DSV4}}(N,l,g_1,g_2)
&=lr_1-T_l(r_1)+(43-l)g_2-T_{43-l}(r_2).
\end{aligned}
}
$$

### 4.1 旧 $g,2g$ Pareto 扫描

现有数据和绘图使用 $N=10^6$、$g_1=g$、$g_2=2g$，并对相位平均。此时存储量为

$$
\boxed{
B_{\mathrm{DSV4}}(l,g)
=\left\lceil\frac{10^6}{g}\right\rceil P_l
+\left\lceil\frac{10^6}{2g}\right\rceil(P-P_l)\ \text{B},
}
$$

计算量为

$$
\boxed{
\overline C_{\mathrm{DSV4}}(l,g)
=\frac{l(g-1)}{2}
+2g(43-l)
-\frac1g\sum_{r=0}^{g-1}T_l(r)
-\frac{T_{43-l}(g)}2.
}
$$

参数扫描范围为

$$
1\le l\le42,
\qquad
1\le g\le\min\{127l+1,127(43-l)\}.
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

且至少一个不等式严格成立，则 $(l,g)$ 是 Pareto 最优点。上述可行域共生成 58695 个参数点，其中 1843 个为 Pareto 点。数据和绘图实现见 [`dsv4_checkpoint_points.csv`](./dsv4_checkpoint_points.csv) 与 [`plot_checkpoint_pareto.py`](./plot_checkpoint_pareto.py)，静态图见 [`imgs/dsv4-checkpoint-pareto-discrete.png`](./imgs/dsv4-checkpoint-pareto-discrete.png)。
