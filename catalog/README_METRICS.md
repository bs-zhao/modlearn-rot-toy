# Functional Metric Catalog — M001, M002, …

> **代码对应版本：** \`src/rottoy/metrics.py\`、\`src/rottoy/operators.py\` 和 \`src/rottoy/plots.py\`。本文逐项记录**当前实际实现的全部 M001–M024 指标**：精确计算式、用途、默认聚合方式、旋转不变量和局限。新指标始终继续使用 M 编号，不设 choice / Fisher / 旧版本的单独指标入口。
>
> Toy 模型定义和 T1–T9 公式请看 [README_TOY_MODELS.md](README_TOY_MODELS.md)。

## 1. 数学约定及解析函数—神经网络成对关系

\[
u=R\theta,\quad \theta=R^\top u,\qquad
\eta'_R(x,u)=\eta(x,R^\top u),\qquad
F'_R(x,u)=F(x,R^\top u).
\]

指标在相同的数值探测点上，分别计算于解析函数 \(\eta'_R\) 与冻结神经网络 \(F'_R\)。以下统一记作 \(h_R(x,u)\)，对应的实际分析尺度 \(T(Y)\) 是 \`--analysis-scale raw|log|inverse-link\`：

\[
h_R^\eta=T\!\left[g_w\big(\eta(x,R^\top u)\big)\right],
\qquad
h_R^F=T\!\left[F(x,R^\top u)\right].
\]

默认是 \`raw\`；Sigmoid 包装的 \`inverse-link\` 是 clip 后 logit。**M017–M024 另有独立的函数算子尺度规则**，见后文，它们对于 sigmoid wrapper 总是在 logit 尺度计算，不能误认为和其他指标使用相同输出尺度。

令 \(K=\dim(u),D=\dim(x)\)，定义局部导数张量

\[
\begin{aligned}
g_k(x,u)&=\partial_{u_k}h_R(x,u),\\
G_{jk}(x,u)&=\partial_{x_j}\partial_{u_k}h_R(x,u),\\
H_{k\ell}(x,u)&=\partial_{u_k}\partial_{u_\ell}h_R(x,u),\\
Q_{j\ell}(x,u)&=\partial_{x_j}\partial_{x_\ell}h_R(x,u).
\end{aligned}
\]

当输入为解析模型，\(h_R=h_R^\eta\)；输入为神经网络时，所有上述导数直接通过对 \(F'_R\) 自动微分得到（代码还允许从原坐标导数应用旋转链式法则）。

### 数值求积，不是“被试均值”和“试次均值”

\[
\langle f\rangle_{X,U}=\int f(x,u)\,dQ_X(x)\,dQ_U(u)
\approx\frac{1}{N_uN_x}\sum_{n=1}^{N_u}\sum_{t=1}^{N_x}f(x_t,u_n).
\]

其中 \(u_n=R\theta_n\)，\(\theta_n\) 来自固定 \([0,1]^K\) 内 Sobol 节点。所有旋转角度使用**相同的原参数 θ 和 X 节点**，只是坐标改变。\(N_u,N_x\) 都是数值积分点数。

记

\[
\mathrm{off}(A)=A-\operatorname{diag}(\operatorname{diag}A),\qquad
\|\mathrm{off}(A)\|_F^2=\sum_{k\ne\ell}A_{k\ell}^2.
\]

零能量/零分母时程序通常返回零，或使用 \(10^{-12}\) 量级的分母保护；这些结果需要与真正的“找到了结构零点”区分。

## 2. 全部指标速查表

| ID | 注册名称 | 计算对象 | 核心含义 | 能否直接用“越小越好”判定正确旋转 |
|---|---|---|---|---|
| **M001** | \`xu_group_l1\` | XU 梯度 RMS 组 L1 | XU 作用集中在少数 U 轴 | 可作候选，非普适 |
| **M002** | \`xu_energy_concentration\` | XU 能量 Herfindahl/Gini-Simpson 型 | 组能量集中度 | 可作候选，非普适 |
| **M003** | \`local_xu_l1\` | 逐 X 维、逐局部点的标准化 L1 | 局部梯度是否单轴 | **可能错选**，T5/T9 负对照 |
| **M004** | \`grad_xcov_offdiag\` | X 上 U 梯度协方差非对角比例 | 同一函数作用随 X 变化的共变 | 可作候选，有简并 |
| **M005** | \`grad_gram_offdiag\` | U 梯度 Gram 非对角比例 | 梯度作用方向的正交程度 | 可作候选，未中心化 |
| **M006** | \`uu_offdiag\` | UU Hessian 非对角能量比例 | 参数交叉曲率 | **T3 的 45° 会更好（错误）** |
| **M007** | \`uu_diagonal\` | UU Hessian 自身平方能量比例 | 每坐标自身曲率 | 与 M006 互补，不能单独普适 |
| **M008** | \`u_grad_energy\` | 全 U 梯度平方和 | 总潜变量敏感性 | **正交旋转不变量** |
| **M009** | \`xu_total_energy\` | 全 XU 混合梯度平方和 | 总 XU 作用量 | **正交旋转不变量** |
| **M010** | \`x_gradient_energy\` | X 梯度平方和 | 总 X 敏感性 | **旋转不变量** |
| **M011** | \`x_curvature_energy\` | XX Hessian 平方和 | 输入曲率 | **旋转不变量** |
| **M012** | \`sweep_affine\` | 单轴 U sweep 的最佳直线残差 | 改变 U 是否引入自身非线性 | 有条件；T2 通常为 0 |
| **M013** | \`sweep_shape_min\` | 归一化响应向量形状方差的最小轴 | 是否存在纯幅度方向 | 只检测“存在某个简单轴” |
| **M014** | \`sweep_rank1_min\` | 单轴响应矩阵最佳秩一残差的最小轴 | 是否存在函数形状固定的单轴 | 只检测“存在某个简单轴” |
| **M015** | \`sweep_curvature\` | 单轴扫掠二阶差分平方 | U-sweep 弯曲/粗糙度 | 有条件 |
| **M016** | \`rectangle_uu\` | U 两轴四角交叉差分平方 | 非局部 UU 交互 | **T3 可能错选** |
| **M017** | \`operator_v1_jd\` | level + first-X 算子 JD | 旧 v1 类型功能矩阵联合对角化 | 需解析验证 |
| **M018** | \`operator_v2_jd\` | level + first/second-X 算子 JD | 旧 v2 类型功能矩阵联合对角化 | 需解析验证，**0° 不保证最低** |
| **M019** | \`operator_v1_level\` | v1 的 level 子集 JD | 对 X 平均的 U 梯度功能 | 有条件 |
| **M020** | \`operator_v1_first\` | v1 的 first-X 子集 JD | U 梯度随 X 的一阶改变 | 有条件 |
| **M021** | \`operator_v2_second_diagonal\` | v2 的纯二阶 X 子集 JD | U 梯度对各 X 的二阶响应 | 线性 toy 通常为零 |
| **M022** | \`operator_v2_second_mixed\` | v2 的混合二阶 X 子集 JD | U 梯度的跨 X 曲率 | 线性 toy 通常为零 |
| **M023** | \`operator_fisher_offdiag\` | logistic-weighted gradient Gram 的非对角 | Fisher 型信息方向混合 | 有条件 |
| **M024** | \`operator_fisher_trace\` | 上述 Fisher 型矩阵的迹 | 信息总量 | **正交旋转不变量** |

**重要：** 上表记录的是**实际实现的数值准则**，并非数学上已经证明的通用可解释性损失。尤其 T2 只是原 C1 的线性类比，旧 NP 上 v2 曾成功，**不构成 M018 在 T2 的 0° 必定全局最优的证明**。

## 3. M001–M011：梯度、协方差和 Hessian 指标

以下 \(\mathbb E\) 均按当前交叉积分网格计算（被积函数在 R 旋转后由链式法则转换）。

### M001 — XU RMS group L1

先计算每个潜轴的 XU 能量

\[
e_k^2=\mathbb E_{X,U}\left[\sum_{j=1}^{D}G_{jk}^2\right],\quad
e_k=\sqrt{e_k^2}.
\]

代码返回

\[
\boxed{M001=\frac{\sum_ke_k}{\sqrt K\,\sqrt{\sum_ke_k^2}}.}
\]

当仅一个轴有非零作用时是 \(1/\sqrt K\)，所有轴等能量时是 1。**较小代表更集中**，但不能要求每个局部 XU 梯度严格单轴。

### M002 — XU energy concentration

令 \(p_k=e_k^2/\sum_\ell e_\ell^2\)，则

\[
\boxed{M002=1-\sum_kp_k^2.}
\]

值越小代表作用集中于较少的 U 轴；若能量全零，代码返回 0（不可据此声称识别成功）。

### M003 — Local normalized XU L1

对每个积分点和 X 维度形成一行 \(G_j(x,u)\in\mathbb R^K\)：

\[
\boxed{
M003=
\mathbb E_{\{(x,u,j):\|G_j\|>0\}}
\left[\frac{\|G_j\|_1}{\|G_j\|_2}\right].
}
\]

程序会跳过近零行；越小越接近局部单轴。然而 T5/T9 的真实 X1 线性与二次参数会同时出现在同一个局部梯度中，因此局部 L1 可能**稳定选择错误旋转**。

### M004 — Stimulus covariance of U gradients

固定每个 U 积分节点，先对 X 中心化：

\[
\bar g(u)=\mathbb E_X g(X,u),\qquad
C=\mathbb E_U\left[\operatorname{Cov}_X(g(X,u))\right].
\]

\[
\boxed{M004=\frac{\|\mathrm{off}(C)\|_F^2}{\|C\|_F^2}.}
\]

描述潜坐标梯度随刺激变化时，彼此的功能共变。先在 X 内中心化、再在 U 上平均；**不是** \(\operatorname{Cov}_{X,U}(g)\)。若 C 为零矩阵，程序返回 0。

### M005 — Uncentered U-gradient Gram offdiagonal

\[
B=\mathbb E_{X,U}\big[g(X,U)g(X,U)^\top\big],\qquad
\boxed{M005=\frac{\|\mathrm{off}(B)\|_F^2}{\|B\|_F^2}.}
\]

与 M004 的区别是**不在 X 上中心化**，因而会受到梯度均值和参数分布的影响。

### M006 — UU cross curvature fraction

\[
H=\nabla_u^2h,\qquad
\boxed{M006=
\frac{\mathbb E_{X,U}\sum_{k\ne\ell}H_{k\ell}^2}
{\mathbb E_{X,U}\|H\|_F^2}.}
\]

零 Hessian 时返回 0。T3 的真参数坐标只有混合 \(ab\) 项，45° 后交叉项消失，因此**M006 越小不一定越接近真实坐标**。

### M007 — UU diagonal curvature fraction

\[
\boxed{M007=
\frac{\mathbb E_{X,U}\sum_{k}H_{kk}^2}
{\mathbb E_{X,U}\|H\|_F^2}.}
\]

非零分母时 \(M006+M007=1\)。注意 M006 与 M007 的推荐优化方向往往互相矛盾；它们是结构诊断，不是可以同时简单最小化的两个独立目标。

### M008 — U gradient total energy

\[
\boxed{M008=\mathbb E_{X,U}\sum_k g_k^2.}
\]

全空间正交旋转下保持不变：**用于链式法则数值对照，不用于寻找最优角度**。

### M009 — XU total energy

\[
\boxed{M009=\mathbb E_{X,U}\sum_{j,k}G_{jk}^2.}
\]

对全 U 坐标正交旋转不变，不应期待 0° 特殊最优。

### M010 — X gradient energy

\[
\boxed{M010=\mathbb E_{X,U}\sum_j(\partial_{x_j}h)^2.}
\]

在同一组物理 \((x,\theta)\) 节点下，纯 U 坐标旋转不改变 X 梯度，整条角度曲线应基本恒定。

### M011 — XX curvature energy

\[
\boxed{M011=\mathbb E_{X,U}\sum_{j,\ell}
(\partial_{x_jx_\ell}h)^2.}
\]

同样对 U 正交旋转保持不变；可以用于检查公式、自动微分与求积一致性。

## 4. M012–M016：全局 U sweep 与非局部交互

实现不是逐“被试” sweep，而是在群体统一定义的原参数中心附近，选定方向 \(R_{k\cdot}\)：

\[
\theta_0=(0.5,\ldots,0.5),\qquad
\theta_{k,s}=\theta_0+sR_{k\cdot},\qquad
s\in[-r,r].
\]

默认 \(r=0.45\)、\`--sweep-steps 101\`。输入为固定的 X 数值网格 \(\{x_t\}\)。

每轴 sweep 定义 \(S_k(s,x)=h(x,\theta_{k,s})\)，矩阵
\(M_k=[S_k(s_m,x_t)]_{m,t}\in\mathbb R^{N_s\times N_x}\)。

### M012 — Per-axis best affine fit, mean across axes

\[
\boxed{
M012=\frac1K\sum_k
\frac{\min_{\alpha_x,\beta_x}\mathbb E_{s,x}[S_k(s,x)-\alpha_x-\beta_xs]^2}
{\operatorname{Var}_{s,x}(S_k(s,x))}.
}
\]

数值实现使用最小二乘投影到 \(\operatorname{span}\{1,s\}\)；方差分母有最小阈值 \(10^{-12}\)。较小意味着沿每个 U 轴的函数更接近仿射。**T2 在原始 logit/identity 尺度下任何线性旋转都保持仿射，因此这一指标不能唯一识别 T2。**

### M013 — Minimum normalized response-shape variation

对每个扫描值，把**所有 X 的响应向量**标准化：

\[
v_k(s)=\frac{(S_k(s,x_t))_t}
{\max(\|(S_k(s,x_t))_t\|_2,10^{-12})}.
\]

计算每个向量分量沿 \(s\) 的方差再平均，最后取最佳轴：

\[
\boxed{M013=\min_k\mathbb E_t[\operatorname{Var}_s(v_{k,t}(s))].}
\]

值为零意味着存在至少一个 sweep 轴，其响应向量方向恒定（只发生整体缩放）；**不说明剩余所有参数轴都被正确恢复**。

### M014 — Minimum rank-one sweep approximation error

对每个 \(M_k\)，计算最大的奇异值：

\[
E_k=1-\frac{\sigma_1(M_k)^2}{\|M_k\|_F^2},
\qquad\boxed{M014=\min_k E_k}.
\]

越低表示至少存在一条 U 轴，其“扫描点 × X 响应”矩阵接近秩一。与 M013 相关但并不相同：该指标作用于原始响应矩阵而不是每行单独标准化后的形状。

### M015 — Squared second finite difference curvature

对均匀 sweep 步长 \(\Delta s\)：

\[
\delta_s^2 S_k(s_m,x)=
\frac{S_k(s_{m+1},x)-2S_k(s_m,x)+S_k(s_{m-1},x)}{(\Delta s)^2},
\]

\[
\boxed{M015=\frac1K\sum_k\mathbb E_{m,t}
[(\delta_s^2S_k(s_m,x_t))^2].}
\]

衡量 U 轴的整体非线性或曲率；依赖扫掠尺度和输出表示，不是通用恢复判据。

### M016 — Finite UU rectangle mixed effect

对每一对坐标轴 \(k<\ell\)，在中心 \(\theta_0\) 的四个点计算

\[
S_{\pm\pm}(x)=
h\big(x,\theta_0\pm\delta R_{k\cdot}\pm\delta R_{\ell\cdot}\big)
\]

（每个 \(\pm\) 独立取号），然后

\[
D_{k\ell}(x)=
\frac{S_{++}(x)-S_{+-}(x)-S_{-+}(x)+S_{--}(x)}{4\delta^2},
\quad
\boxed{M016=\mathbb E_{k<\ell,X}[D_{k\ell}(X)^2].}
\]

默认 \(\delta=0.2\)。该非局部差分近似 UU 混合导数；如果生成模型本来含 \(ab\) 交互（T3），简单地将它最小化可能错误偏好混合坐标。

## 5. M017–M024：通用 X/U 函数算子库

这组来自原项目 \`decoder_fisher_jointdiag\` 的**数学构造**，但现在统一为普通 M 指标，没有任何特殊 toy 接口。它们位于公共的 [operators.py](../src/rottoy/operators.py) 的 \`build_operator_bank\` 与 \`operator_family_metrics\`。

### 5.1 与 M001–M016 不同的分析尺度

令

\[
q(x,u)=
\begin{cases}
\operatorname{logit}\bigl(P(Y=1\mid x,u)\bigr),&
\text{wrapper 为 sigmoid}\\
Y(x,u),&\text{wrapper 为 identity / square}
\end{cases}
,\qquad
g_q(x,u)=\nabla_uq(x,u).
\]

函数算子 M017–M024 **不使用** \`--analysis-scale raw/log/inverse-link\` 来选择 q。对于非 sigmoid 包装，即使原模型是连续 \(Y\)，代码也将 Y **形式上当作 logit** 进行 logistic 权重计算；因此 M023/M024 是“信息矩阵型诊断”，**不是已验证的该连续生成模型的真实 Fisher likelihood**。

在原参数物理网格上直接算 \(g_\theta=\nabla_\theta q(x,\theta)\)，构造下面的参考矩阵 \(K_\alpha\)。对角度 R 做 \(R K_\alpha R^\top\) 完成相同的 U 旋转。

### 5.2 X 积分后才形成外积（关键顺序）

固定某个 \(\theta\) 节点，对 X 积分，并以各 X 维参考标准差 \(s_j=\mathrm{SD}_{Q_X}(X_j)\) 作尺度规范化：

\[
\begin{aligned}
\psi_0(\theta)&=\mathbb E_X[g_\theta(X,\theta)]
&&\text{Level},\\
\psi_j(\theta)&=
s_j\mathbb E_X[\partial_{x_j}g_\theta(X,\theta)]
&&\text{First X},\\
\psi_{jj}(\theta)&=
s_j^2\mathbb E_X[\partial_{x_j}^2g_\theta(X,\theta)]
&&\text{Diagonal second X},\\
\psi_{j\ell}(\theta)&=
s_js_\ell\mathbb E_X[\partial_{x_j}\partial_{x_\ell}g_\theta(X,\theta)]
&&\text{Mixed second X}.
\end{aligned}
\]

\[
\boxed{K_\alpha=\mathbb E_\Theta[\psi_\alpha(\Theta)\psi_\alpha(\Theta)^\top].}
\]

**必须先做 X 积分，再做外积，最后对 Θ/U 做积分**：
\(\mathbb E_\theta[(\mathbb E_Xg)(\mathbb E_Xg)^\top]\)；
这**不等于**
\(\mathbb E_{\theta,X}[gg^\top]\)。

实现第一、二阶 X 导数时使用 \(\nabla_\theta q\) 的数值有限差分；靠近 X∈[0,1] 边界采用单侧差分，默认

\[
h_j^{(1)}=\max(0.02\,s_j,10^{-5}),\qquad
h_j^{(2)}=\max(0.10\,s_j,10^{-4}).
\]

数学定义是连续导数，但当前网络和解析函数的这些指标**均采用同一有限差分数值实现**，用于保证对照公平。

### 5.3 过滤、规范化与 JD 损失

对每组原始矩阵 \(K_\alpha\)，计算 \(w_\alpha=\|K_\alpha\|_F\)，令 \(w_{\max}=\max_\alpha w_\alpha\)。保留满足

\[
w_\alpha\ge \rho\,w_{\max},
\quad w_\alpha>10^{-14}
\]

的算子，并进行

\[
\widetilde K_\alpha=K_\alpha/\|K_\alpha\|_F.
\]

v1 的 \(\rho=10^{-4}\)，v2 的 \(\rho=0.01\)，且是各自算子集合内的最大值。设选中的算子下标集合为 \(\mathcal A\)：

\[
\boxed{J_{\mathcal A}(R)=
\frac{1}{|\mathcal A|}\sum_{\alpha\in\mathcal A}
\|\mathrm{off}(R\widetilde K_\alpha R^\top)\|_F^2.}
\]

如果没有保留的算子，该组 \`_jd\` 当前返回 **0**；**这代表没有可用证据，而非证明该旋转正确**。不同模型的保留集合与 v1/v2 的不同相对阈值会导致指标不完全可比。

### M017 — v1 operator JD

\[
\boxed{M017=J_{\mathcal A_1}(R),\quad
\mathcal A_1=\{\text{Level, all First-X}\}_{\rm filtered}.}
\]

更小意味着这些函数作用算子在同一潜坐标系中更接近共同对角。历史来源是原 decoder Fisher JD 的 v1 family，但它本身不是普适的真实参数轴定理。

### M018 — v2 operator JD

\[
\boxed{M018=J_{\mathcal A_2}(R),\quad
\mathcal A_2=\{\text{Level, First-X, Diagonal-XX, Mixed-XX}\}_{\rm filtered}.}
\]

增加 X 的二阶纯/混合导数功能算子。**即使基础函数对 X 线性，二阶项为空，由于 v1/v2 的阈值不同，M018 也不一定逐点等于 M017。** 在 T2 上 0° 是否为全局最小仍需检验。

### M019 — v1 level component

\[
\boxed{M019=J_{\{\text{v1 保留的 Level 算子}\}}(R).}
\]

该项强调对所有 X 平均后的整体 U 效应。对原偏置或 U 均值结构可能敏感。

### M020 — v1 first-X component

\[
\boxed{M020=J_{\{\text{v1 保留的 First-X 算子}\}}(R).}
\]

刻画各 X 坐标的一阶变化如何由不同潜轴承担；对于 T2/T5/T9 特别值得与 M017/M018 对比。

### M021 — v2 diagonal-second-X component

\[
\boxed{M021=J_{\{\text{v2 保留的 Diagonal-XX 算子}\}}(R).}
\]

二阶 X 自身曲率对各 U 的方向约束。T2 原始 logit 对 X 线性，理论值零，此项可能因过滤后空集合直接返回零。

### M022 — v2 mixed-second-X component

\[
\boxed{M022=J_{\{\text{v2 保留的 Mixed-XX 算子}\}}(R).}
\]

测量不同 X 维度之间混合曲率的潜轴结构。对不存在 \(x_jx_\ell\) 交互的函数理论上为零；数值零不代表恢复了全部参数。

### M023 — logistic-weighted gradient offdiagonal

代码对 \(q\) 构造

\[
p=\sigma(q),\qquad
I_\theta=\mathbb E_{X,\Theta}[p(1-p)\,g_\theta g_\theta^\top],
\quad I_R=RI_\theta R^\top,
\]

\[
\boxed{M023=\|\mathrm{off}(I_R)\|_F^2.}
\]

注意**没有除以** \(\|I_R\|_F^2\)，与 M004、M005 的比例化目标不同。Sigmoid wrapper 下这是基于 logit 的 Bernoulli Fisher 型对象；identity/square 下是将输出形式上代入 logistic 的诊断。

### M024 — logistic-weighted Fisher trace

\[
\boxed{M024=\operatorname{tr}(RI_\theta R^\top)=\operatorname{tr}(I_\theta).}
\]

全空间正交旋转不变量，应该随角度保持常数。用于测试数值稳定性和矩阵旋转正确性，**不是**找 0° 最优的损失。

## 6. 旋转与数学解释的关键限制

1. **固定比较尺度与参考测度。** 所有角度使用相同物理 \(\theta,X\) 网格；平方、Sigmoid、logit 输出不可混为一谈。
2. **区分局部与全局。** M003 追求局部 XU 稀疏；T5/T9 的真实参数下局部 XU 梯度本来就会混合多个参数。
3. **区分“某一轴简单”和“全部轴恢复”。** M013、M014 对所有 U 轴取最小，不是衡量完整 K 维轴对齐。
4. **区分“算子没有能量”和“旋转真正简单”。** M006/M007 的零分母和 M019–M022 的空算子集合均可能返回 0。
5. **已知对照。** M006 在 T3 的真实坐标中有交叉项，45° 混合使交叉项消失；不能解释为更接近真参数。
6. **正交不变量。** M008–M011、M024 的角度曲线应基本水平，适合计算一致性检查。
7. **历史成功不构成通用最优性证明。** M018 对简化线性 toy T2 的 0° 取最小是**待检验的假设**，不是预置目标或确保发生的事实。

## 7. 实际运行、输出和追加编号

\`\`\`bash
python run.py --list-metrics
python run.py --models T1 T3 T9 --metrics M001 M004 M017 M018 --pairs all --wrappers sigmoid
python run.py --models T2 --metrics M017 M018 M019 M020 M021 M022 M023 M024 --wrappers sigmoid --pairs all
\`\`\`

CSV 表头使用稳定的 \`M001\` 等编号；同角度的解析/神经网络对比输出 \`M001__eta\`、\`M001__F\`、\`M001__difference\`。

当前代码在 [metrics.py](../src/rottoy/metrics.py) 用 \`register_metric("name","description")\` 按注册顺序自动分配 \`M001, M002, ...\`。增加新指标时务必：

- **只在现有注册表末尾追加，不能插到前面**（否则会使已有 M 号码漂移）；
- 在公共指标/算子实现中增加对应函数并通过统一 \`evaluate_metrics\` 输出；
- 将新 M 编号、具体公式、数值计算和局限同步添加到**本 README**；
- 保证相同指标能分别对 \(\eta'\) 与 \(F'\) 计算，并在相同网格上核验数值一致性。

运行验证：

\`\`\`bash
python tests/test_registry.py
python tests/test_toys.py
python tests/test_functional_operators.py
\`\`\`

**本文件是现有 24 项实现的数学说明，不代表 72 项更广泛候选库均已编码。**
