> **中文专版｜功能指标目录（中文版）** · [原 README](./README_METRICS.md) · [Toy 模型中文版](./README_TOY_MODELS_ZH.md)
>
> 本文保留原说明的数学公式、编号、数值算法与限制条件；原文件当前也以中文撰写。

# Functional Metrics：M001–M024 的数学公式、意义和数值实现

这里所有指标都作用于**连续标量函数**，不是二分类似然，也不区分 choice、pricing 等任务。T 模型决定生成公式，输出后缀决定连续 $Y$，而 **M 编号唯一标识指标**。

实际注册表位于 [`src/rottoy/metrics/specs.py`](../src/rottoy/metrics/specs.py)，其中每项都有 `key`、唯一编号、英文名称、公式、数值实现、意义和可否用来判别旋转的标记。计算均走 [`src/rottoy/metrics/`](../src/rottoy/metrics/) 的统一接口。未来添加 M025、M026，不为任何 toy 创建专用指标文件。

## 1. 统一数学对象：解析函数与神经网络完全成对

每个版本为 `Tn-w`（见 [模型及 Y 范围](README_TOY_MODELS.md)），其连续目标为

$$
h_R^\eta(X,u)
=T\big[g_w(\eta_T(X,R^\top u))\big],
\qquad
h_R^F(X,u)=T\big[F_{T,w}(X,R^\top u)\big].
$$

$T$ 是 `--analysis-scale raw|log|inverse` 对应的分析尺度：直接响应、裁剪后对数、可逆包装的逆函数（平方版本不可逆）。**对任意指标 $M$，解析指标 $M[h_R^\eta]$ 与网络指标 $M[h_R^F]$ 使用同一公式、同一组 X/U 数值积分节点与同一旋转矩阵。** 不需定义“被试”和“试次”。

设 $K=\dim u$、$D=\dim X$，并定义

$$
g_k=\partial_{u_k}h,\qquad
G_{jk}=\partial_{X_j}\partial_{u_k}h,\qquad
H_{k\ell}=\partial_{u_k}\partial_{u_\ell}h,\qquad
Q_{j\ell}=\partial_{X_j}\partial_{X_\ell}h.
$$

函数层次的期望与数值求积：

$$
\mathbb E_{X,U} f(X,U)
=\int f(x,u)\,dQ_X(x)dQ_U(u)
\ \approx\ \frac1{N_XN_U}\sum_{t=1}^{N_X}\sum_{m=1}^{N_U} f(x_t,u_m).
$$

$X_t$、$\theta_m$ 在**固定原始物理区间** $[0,1]^{D}$、$[0,1]^K$ 取样，$u_m=R\theta_m$。当旋转改变时仍使用同一批物理节点，从而比较相同函数区域。`--x-points`、`--u-points` 是数值积分点数，不是 500 次试验或若干被试。

记 $\mathrm{off}(A)=A-\mathrm{diag}(\mathrm{diag}A)$。数值上，导数主要通过 PyTorch autodiff 生成一次公共 derivative bank，再用旋转链式法则获得新的 U 轴梯度；积分用等权 Sobol 求积（没有假设精确解析积分）。

## 2. 全部已实现指标速查表

| ID | 唯一 key / 清晰名称 | 主要数学对象 | 含义和解释 | 作用 |
|---|---|---|---|---|
| **M001** | `xu_group_l1` · XU 组 L1 | 跨 X/U 的各 U 轴 XU 梯度 RMS | 某些 U 轴是否独占 X 敏感性 | 备选 |
| **M002** | `xu_energy_concentration` · XU 能量集中 | 组能量比例的 $1-\sum p_k^2$ | 高维影响是否集中在少数 U 轴 | 备选 |
| **M003** | `local_xu_l1` · 局部 XU 稀疏 | 每点每个 X 梯度行的归一化 L1 | 局部是否单轴，T5/T9 有失败反例 | 负对照 |
| **M004** | `grad_xcov_offdiag` · 梯度 X 协方差 JD | $\mathbb E_U\mathrm{Cov}_X(g)$ 的非对角比例 | 梯度随 X 改变的相关性 | 备选 |
| **M005** | `grad_gram_offdiag` · 未中心化 Gram JD | $\mathbb E_{X,U}gg^\top$ 的非对角比例 | 总梯度方向间的混合 | 备选 |
| **M006** | `uu_offdiag` · UU 交叉曲率比例 | UU Hessian 的非对角能量 | T3 中 45° 可能错误最优 | 负对照 |
| **M007** | `uu_diagonal` · UU 自身曲率比例 | UU Hessian 的对角能量 | 与 M006 互补，非独立普适目标 | 诊断 |
| **M008** | `u_grad_energy` · U 总梯度能量 | $\mathbb E\lVert \nabla_uh\rVert^2$ | 正交旋转不变 | 不变量 |
| **M009** | `xu_total_energy` · XU 总能量 | $\mathbb E\lVert G\rVert_F^2$ | 正交旋转不变 | 不变量 |
| **M010** | `x_gradient_energy` · X 总梯度能量 | $\mathbb E\lVert \nabla_Xh\rVert^2$ | 固定物理节点时旋转不变 | 不变量 |
| **M011** | `x_curvature_energy` · XX Hessian 总能量 | $\mathbb E\lVert Q\rVert_F^2$ | 固定物理节点时旋转不变 | 不变量 |
| **M012** | `sweep_affine` · U 轴仿射残差 | 单轴 sweep 对直线拟合的残差 | 每个坐标是否线性/仿射影响响应 | 备选 |
| **M013** | `sweep_shape_min` · 最简单响应形状轴 | 归一化响应向量的变化方差 | 是否**存在**一条纯比例轴 | 诊断 |
| **M014** | `sweep_rank1_min` · 最小秩一残差 | $1-\sigma_1(S)^2/\lVert S\rVert_F^2$ | 是否**存在**一条近秩一的轴 | 诊断 |
| **M015** | `sweep_curvature` · U 扫描二阶粗糙度 | sweep 曲线的二阶差分平方 | 单轴非线性 | 备选 |
| **M016** | `rectangle_uu` · UU 矩形交互 | 四角交叉差分平方 | 全局非局部交互；T3 可错选 | 备选/负对照 |
| **M017** | `operator_v1_jd` · v1 功能算子联合对角化 | Level + First-X 的矩阵 JD | 是否存在共同功能轴 | 备选 |
| **M018** | `operator_v2_jd` · v2 功能算子联合对角化 | Level + First-X + Second-X 的矩阵 JD | 丰富 X 作用算子的共同轴 | 备选 |
| **M019** | `operator_v1_level` · Level 分项 JD | Level 矩阵的 JD | X 积分后的 U 平均作用 | 分项 |
| **M020** | `operator_v1_first` · First-X 分项 JD | 一阶 X 算子 JD | U 对 X 效应的调节 | 分项 |
| **M021** | `operator_v2_second_diagonal` · XX 自身二阶 JD | X 单维二阶算子 JD | X 弯曲结构 | 分项 |
| **M022** | `operator_v2_second_mixed` · XX 混合二阶 JD | X 交互二阶算子 JD | X 之间的交互结构 | 分项 |
| **M023** | `operator_fisher_offdiag` · sigmoid 加权梯度 Gram JD | 加权外积矩阵的非对角能量 | 一种功能信息量矩阵；**不是真实连续 Y 的 Fisher likelihood** | 备选 |
| **M024** | `operator_fisher_trace` · sigmoid 加权梯度 Gram 迹 | 同一加权矩阵的迹 | 正交旋转不变量 | 不变量 |

“备选”只表示可尝试最小化；不代表已证明在所有 toy 的 0° 达到最小。**0° 是否最小必须用解析函数和数值扫描验证**。

## 3. M001–M011：局部微分与 X/U 整体统计

令

$$
e_k=\sqrt{\mathbb E_{X,U}\sum_jG_{jk}^2},
\quad p_k=\frac{e_k^2}{\sum_l e_l^2},
\quad C=\mathbb E_U[\mathrm{Cov}_X(g(X,U))],
\quad B=\mathbb E_{X,U}[gg^\top].
$$

各指标的**精确实现式**：

$$
\begin{aligned}
\mathrm{M001}&=\frac{\sum_ke_k}{\sqrt K\sqrt{\sum_ke_k^2}},\\
\mathrm{M002}&=1-\sum_kp_k^2,\\
\mathrm{M003}&=\mathbb E_{\lVert G_j\rVert_2>0}\frac{\lVert G_j\rVert_1}{\lVert G_j\rVert_2},\\
\mathrm{M004}&=\frac{\lVert \mathrm{off}(C)\rVert_F^2}{\lVert C\rVert_F^2},\\
\mathrm{M005}&=\frac{\lVert \mathrm{off}(B)\rVert_F^2}{\lVert B\rVert_F^2},\\
\mathrm{M006}&=\frac{\mathbb E_{X,U}\sum_{k\ne l}H_{kl}^2}{\mathbb E_{X,U}\lVert H\rVert_F^2},\\
\mathrm{M007}&=\frac{\mathbb E_{X,U}\sum_{k}H_{kk}^2}{\mathbb E_{X,U}\lVert H\rVert_F^2},\\
\mathrm{M008}&=\mathbb E_{X,U}\lVert \nabla_uh\rVert_2^2,\\
\mathrm{M009}&=\mathbb E_{X,U}\lVert G\rVert_F^2,\\
\mathrm{M010}&=\mathbb E_{X,U}\lVert \nabla_Xh\rVert_2^2,\\
\mathrm{M011}&=\mathbb E_{X,U}\lVert Q\rVert_F^2.
\end{aligned}
$$

计算细节：M003 对局部梯度范数低于数值阈值的行不计入平均；M004 **先在每个 U 位置对 X 中心化**再对 U 积分，不能误写成 $\mathrm{Cov}_{X,U}$；M005 是未中心化外积。M006+M007 在 Hessian 非零时为 1。零分母的指标代码使用保护值/返回 0，不能把数学上的不可判定解释为“找到了正确轴”。

M008–M011 都是**旋转不变量或与旋转无关的检验量**：横轴旋转角度时它们应该形成近似水平线。它们帮助检验实现是否正确，不能作为区分角度的旋转损失。

## 4. M012–M016：有限范围的 U-sweep

设 $\theta_0=(0.5,\ldots,0.5)$；对旋转矩阵 $R$ 的第 $k$ 行方向定义

$$
\theta_k(s)=\theta_0+sR_{k,:},\quad s\in[-r,r],\qquad
S_k(s,x)=h(x,\theta_k(s)).
$$

默认 $r=0.45$、`--sweep-steps 101`，X 使用同一数值积分网格。令 $S_k$ 表示“扫描点 × X 点”的离散函数矩阵。所有扫描点都验证反变换后的物理参数在 $[0,1]$ 内。

$$
\begin{aligned}
\mathrm{M012}&=\frac1K\sum_k
\frac{\min_{\alpha_x,\beta_x}\mathbb E_{s,X}
 (S_k(s,X)-\alpha_X-\beta_Xs)^2}{\mathrm{Var}_{s,X}(S_k)},\\
\mathrm{M013}&=\min_k\mathbb E_X\mathrm{Var}_s\left[
\frac{S_k(s,X)}{\lVert (S_k(s,x_t))_t\rVert_2}\right],\\
\mathrm{M014}&=\min_k\left[1-\frac{\sigma_1(S_k)^2}{\lVert S_k\rVert_F^2}\right],\\
\mathrm{M015}&=\frac1K\sum_k\mathbb E_{s,X}
\left[\frac{S_k(s+\Delta s,X)-2S_k(s,X)+S_k(s-\Delta s,X)}
{(\Delta s)^2}\right]^2,\\
\mathrm{M016}&=\mathbb E_{k<l,X}\left[
\frac{S_{++}-S_{+-}-S_{-+}+S_{--}}{4\delta^2}\right]^2.
\end{aligned}
$$

M013 的 $S_k$ 向量和 M014 的整个矩阵使用有限样本范数；$\mathrm{Var}$ 用等权节点经验方差。M016 的四角函数值是在中心 $\theta_0\pm\delta R_{k,:}\pm\delta R_{l,:}$ 评价，默认 $\delta=0.2$。代码对于非常小的方差或范数设置数值阈值。

**注意：** M013、M014 对 $k$ 取最小值，只检验是否存在某个简单方向，而非所有轴都与真实参数一一对应。T2 的基础函数对任何线性坐标变换都沿单轴仿射，因此 M012 不能唯一识别 T2。

## 5. M017–M024：通用函数算子矩阵与联合对角化

此组源于旧版 v1/v2 的函数特征构造，现在是**与任意 toy 输出通用的一组指标**。公共实现位于 [`operators/functionals.py`](../src/rottoy/operators/functionals.py)。

### 5.1 连续函数尺度和数值有限差分

此组使用独立超参数 `--operator-scale raw|inverse`，默认 `raw`。令

$$
q(X,\theta)=
\begin{cases}
g_w(\eta_T(X,\theta)), & \text{raw},\\
\eta_T(X,\theta),& \text{inverse（仅可逆变换）}.
\end{cases}
\quad
g_q(X,\theta)=\nabla_\theta q(X,\theta).
$$

神经网络版本将 $\eta_T$ 换成 $F_{T,w}$ 预测及其可逆变换；均使用相同的实际数值算子。平方 `-sq` 没有全局逆，不能选择 inverse。

**这里的 q 永远是连续值**，与该函数是不是 Sigmoid 映射无关。M023/M024 中的 sigmoid 加权**不是** Bernoulli likelihood，不能称为连续输出的真实 Fisher 信息。

### 5.2 先对 X 积分，后外积，最后对 θ/U 积分

令 $s_j=\mathrm{SD}_{Q_X}(X_j)$；分别定义

$$
\begin{aligned}
\psi_0(\theta)&=\mathbb E_X[g_q(X,\theta)],\\
\psi_j(\theta)&=s_j\mathbb E_X[\partial_{X_j}g_q(X,\theta)],\\
\psi_{jj}(\theta)&=s_j^2\mathbb E_X[\partial_{X_j}^2g_q(X,\theta)],\\
\psi_{jl}(\theta)&=s_js_l\mathbb E_X[\partial_{X_j}\partial_{X_l}g_q(X,\theta)]
\quad(j<l).
\end{aligned}
$$

每个 $\psi_\alpha$ 是一个 $K$ 维函数向量；形成算子矩阵

$$
\boxed{K_\alpha=\mathbb E_\Theta[\psi_\alpha(\Theta)\psi_\alpha(\Theta)^\top].}
$$

这个聚合顺序不可随意交换，尤其 $\mathbb E_\theta[(\mathbb E_Xg_q)(\mathbb E_Xg_q)^\top]\neq \mathbb E_{\theta,X}[g_qg_q^\top]$。

解析定义使用导数；当前两类后端**都**使用匹配的有限差分近似 X 的一、二阶导数，默认步长

$$
h_j^{(1)}=\max(0.02s_j,10^{-5}),\qquad
h_j^{(2)}=\max(0.10s_j,10^{-4}).
$$

靠近 X 的 0 或 1 边界使用单侧差分，避免越界评价。

### 5.3 单算子归一化、能量筛选与联合对角损失

定义每个算子原始能量 $e_\alpha=\lVert K_\alpha\rVert_F$。v1 仅包含 Level 和 First-X；v2 包含 Level、First-X 和两种 Second-X。两个集合分别以各自的最大算子能量筛选：

$$
\mathcal A_r=\{\alpha:e_\alpha>\varepsilon,\ e_\alpha\ge\rho_r\max_{\beta\in r}e_\beta\},
\qquad
\rho_1=10^{-4},\quad\rho_2=0.01.
$$

对保留算子定义 $\widetilde K_\alpha=K_\alpha/\lVert K_\alpha\rVert_F$，最终

$$
J_{\mathcal A}(R)=
\frac{1}{|\mathcal A|}\sum_{\alpha\in\mathcal A}
\lVert \mathrm{off}(R\widetilde K_\alpha R^\top)\rVert_F^2.
$$

当算子集合为空时，实现返回 0；该值表示**没有算子信息**，不是“完美恢复”。程序把原算子、保留矩阵和能量保存为 `functional_operators.npz`，可单独核对。

### 5.4 八个指标各自的公式

$$
\begin{aligned}
\mathrm{M017}&=J_{\mathcal A_{v1}}(R) &&\text{Level + First-X},\\
\mathrm{M018}&=J_{\mathcal A_{v2}}(R) &&\text{Level + First + Second-X},\\
\mathrm{M019}&=J_{\mathcal A_{v1}\cap\{\text{level}\}}(R),\\
\mathrm{M020}&=J_{\mathcal A_{v1}\cap\{\text{first-X}\}}(R),\\
\mathrm{M021}&=J_{\mathcal A_{v2}\cap\{\text{diagonal XX}\}}(R),\\
\mathrm{M022}&=J_{\mathcal A_{v2}\cap\{\text{mixed XX}\}}(R).
\end{aligned}
$$

**M017 和 M018 不一定相等**：即便在 T2 等线性基础模型的 inverse 尺度下二阶 X 算子都为零，两者的阈值不同，保留的非零算子仍可能有差别。

最后定义**sigmoid 加权的函数梯度 Gram 矩阵**（本项目不将它当成针对连续观测的 Fisher likelihood）：

$$
I_w=\mathbb E_{\Theta,X}\left[
\sigma(q)(1-\sigma(q))g_qg_q^\top
\right],\qquad I_w(R)=RI_wR^\top.
$$

$$
\boxed{\mathrm{M023}=\lVert \mathrm{off}(I_w(R))\rVert_F^2,\qquad
\mathrm{M024}=\mathrm{tr}(I_w(R)).}
$$

M024 对正交旋转**严格不变**；M023 **未除以** $\lVert I_w\rVert_F^2$，因此不可直接与归一化的 M004/M005 比绝对数值大小。

## 6. 数学、数值和神经逼近的解释原则

1. **输出尺度必须一致。** `-id`、`-sig`、`-sq`、`-tanh`、`-sp`、`-exp` 可能具有完全不同的导数、曲率与算子排序。相同数学公式作用于 $\eta'$ 和 $F'$，不等于不同变换之间结果相同。
2. **函数值近似不足以推导导数近似。** 即使 $F'\approx\eta'$，其 $\partial_XF'$、$\partial_UF'$、二阶导数仍可能有显著偏差，尤其在 Sigmoid 饱和区域。
3. **反例需要保留。** T3 的 M006 可以在错误的 45° 方向最小；T5/T9 的真实 XU 局部梯度本来涉及不止一个参数。失败不应被强行调参掩盖。
4. **比较 0° 不等于规定 0° 必须最小。** 模型真参数坐标对应 0°，但每项指标能否识别由分析与数据决定。
5. **数值求积不是模拟被试。** 当前研究分析可微函数本身；将来加入 MP/NP 训练是另一个独立阶段。

## 7. 查看指标及运行

```bash
python run.py --list-metrics
python run.py --describe-metric M018
python run.py --models T1-id T3-sq T9-sig --metrics M001 M006 M017 M018 --pairs all
python run.py --models T9-sig --metrics M017 M018 M020 --operator-scale inverse --pairs all --angles 181
python run.py --models T3-id --backend both --train-neural --metrics M004 M006 M017 M018
```

在 `metrics/specs.py` 末尾注册新指标时必须写出 **名称、数学公式、数值方法、含义和性质**，再于公共实现中支持统一 `compute_metrics` 调用，并在本文件加入新编号。已有 M 编号不能重排。
