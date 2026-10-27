# Toy Model Catalog — T1, T2, …

> **代码对应版本：** 当前仓库的 \`src/rottoy/toy_models/\` 和 \`src/rottoy/models.py\`。本文是**全部已注册 toy 生成函数的公式总表**；新模型只按 T 编号递增，不另设 Choice/C2 类别或模型专用运行入口。
>
> 指标的定义、求积和数学解释请看同目录 [README_METRICS.md](README_METRICS.md)。

## 1. 所有模型共用的数学定义

设第 \(m\) 个 toy 为

\[
\eta_m:\mathcal X_m\times\Theta_m\longrightarrow\mathbb R,\qquad
\eta_m(x,\theta),
\]

其中 \(D_X=\dim(x)\)、\(K=\dim(\theta)\)。目前全部**原始物理变量**满足

\[
x_j\in[0,1],\qquad \theta_k\in[0,1].
\]

实际 Sobol 数值采样在开区间内进行（默认将节点缩到 \([10^{-4},1-10^{-4}]\)），以避免 \(x=0\) 时非整数幂的导数奇点。每一维由相同的统一采样器生成，不表示“被试”或“试次”。

### 输出包装器（所有 T 模型统一）

基础函数 \(\eta_m(x,\theta)\) 的返回值**不带** sigmoid：

\[
\begin{aligned}
g_{\rm identity}(t)&=t,\\
g_{\rm sigmoid}(t)&=\sigma(t)=\frac{1}{1+e^{-t}},\\
g_{\rm square}(t)&=t^2,\\
Y_{m,w}(x,\theta)&=g_w\!\left(\eta_m(x,\theta)\right).
\end{aligned}
\]

分析尺度 \`--analysis-scale\` 进一步作用于 \(Y\)：

| 设置 | 计算的函数 \(h=T(Y)\) | 备注 |
|---|---|---|
| \`raw\` | \(h=Y\) | 默认；\`sigmoid\` 时是概率 |
| \`log\` | \(h=\log\max(Y,10^{-20})\) | 仅在正值响应上解释为 log；实现包含 clamp |
| \`inverse-link\` | \(h=\operatorname{logit}\{\operatorname{clip}(Y,10^{-12},1-10^{-12})\}\) | 只允许 \`sigmoid\` 包装器 |

**注意：** logit / log / raw 下的梯度和 Hessian 不是相同指标；平方包装器可能带来符号不可识别性。

### 旋转定义

目前的解析测试取理想可达潜空间 \(z=\theta\)。在任意两个坐标 \((i,j)\) 中给定平面正交旋转 \(R_{ij}(\phi)\)，其 \(2\times2\) 非平凡子块为

\[
R_{ij}(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi\end{pmatrix},\qquad
u=R\theta,\quad\theta=R^\top u.
\]

所以解析和网络的旋转函数分别为

\[
\boxed{\eta'_R(x,u)=\eta_m(x,R^\top u)},\qquad
\boxed{F'_R(x,u)=F(x,R^\top u)}.
\]

\(\phi=0^\circ\) 对应原始参数轴；旋转后的 \(u\) **不要求**仍属于 \([0,1]^K\)，但评价点逆变换后的 \(\theta\) 必须位于合法原始定义域。旋转只更改坐标，不改变同一 \((x,\theta)\) 的函数值。

## 2. 已实现模型总表（T1–T9）

| ID | 真实参数顺序 \(\theta\) | \(K\) | X 顺序 | \(D_X\) | 生成函数及用途 | 代码 |
|---|---|---:|---|---:|---|---|
| **T1** | \((a,b)\) | 2 | \((x)\) | 1 | 幂函数；幅度与指数作用 | [builtins.py](../src/rottoy/toy_models/builtins.py) |
| **T2** | \((a,b,c)\) | 3 | \((x_1,x_2)\) | 2 | 线性项与偏置；C1 型简化类比 | [builtins.py](../src/rottoy/toy_models/builtins.py) |
| **T3** | \((a,b)\) | 2 | \((x_1,x_2)\) | 2 | 真正的 \(ab\) 参数交互；45° 负对照 | [builtins.py](../src/rottoy/toy_models/builtins.py) |
| **T4** | \((a,b,c,d)\) | 4 | \((x_1,x_2)\) | 2 | 共享幂指数与 \(cd\)；不可识别负对照 | [builtins.py](../src/rottoy/toy_models/builtins.py) |
| **T5** | \((a,b,c,d)\) | 4 | \((x_1,x_2)\) | 2 | 同一 X 上的线性/二次效应 | [builtins.py](../src/rottoy/toy_models/builtins.py) |
| **T6** | \((a,b,c,d)\) | 4 | \((x_1,x_2)\) | 2 | \(x_1x_2\) 刺激交互 | [builtins.py](../src/rottoy/toy_models/builtins.py) |
| **T7** | \((a,b)\) | 2 | \((x)\) | 1 | 指数函数；独立插件范例 | [t7.py](../src/rottoy/toy_models/t7.py) |
| **T8** | \((a,b,c,d)\) | 4 | \((x_1,x_2,x_3)\) | 3 | T4 加入 \(cx_3\) 的可识别性对照 | [builtins.py](../src/rottoy/toy_models/builtins.py) |
| **T9** | \((b_1,b_2,b_3,b_4)\) | 4 | \((X_{1A},X_{1B},X_{2A},X_{2B})\) | 4 | 原项目 C2 的完整 logit 生成公式 | [t9.py](../src/rottoy/toy_models/t9.py) |

模型 ID 是唯一正式接口；先前的 \`T4ID\` 现在是 **T8**，先前单列的 C2 公式现在是 **T9**。

## 3. 各模型具体公式、可辨识结构

### T1 — 幂函数：幅度与指数

\[
\boxed{\eta_1(x;a,b)=a x^b.}
\]

\(a\) 同比例缩放响应，\(b\) 控制形状；\(x>0\) 的区域可取

\[
\log\eta_1=\log a+b\log x.
\]

例如 \(\partial_{xb}\log\eta_1=1/x\)、\(\partial_{xa}\log\eta_1=0\)。适用于研究纯尺度方向、形状变化、U-sweep、log 表示对函数简单性的影响。

### T2 — 两项线性加偏置

\[
\boxed{\eta_2(x_1,x_2;a,b,c)=a x_1+b x_2+c.}
\]

\[
\nabla_\theta\eta_2=(x_1,x_2,1)^\top,\qquad
H_{\theta\theta}=0.
\]

参数在 logit/identity 基础函数尺度上全部线性，任何**线性旋转**后 \(H_{UU}\) 仍然为零。它只是 **C1 型简化 toy**，**不是**原项目 C1 的逐项原始公式或参数分布。

### T3 — 刺激作用中的参数乘积

\[
\boxed{\eta_3(x_1,x_2;a,b)=a x_1+ab x_2.}
\]

\[
H_{ab}=\begin{pmatrix}0&x_2\\x_2&0\end{pmatrix}.
\]

按 \(u_1=(a+b)/\sqrt2,u_2=(a-b)/\sqrt2\) 进行 45° 变换后，

\[
\eta_3(x,u)=\frac{u_1+u_2}{\sqrt2}x_1+
\frac{u_1^2-u_2^2}{2}x_2.
\]

正确轴下非对角 UU 曲率非零；45° 下非对角消失、出现自身平方项。因此**只最小化 UU 非对角曲率会选错**。将 \((a,b)\) 变成 \((a,ab)\) 是非线性重参数化，不允许在当前线性旋转空间内直接使用。

### T4 — 共享指数 + 不可识别乘积

\[
\boxed{\eta_4(x_1,x_2;a,b,c,d)=a x_1^b+cd x_2^b.}
\]

\[
\partial_c\eta_4=d x_2^b,\qquad
\partial_d\eta_4=c x_2^b.
\]

两列成比例；\((c,d)\mapsto(kc,d/k)\) 保持输出不变，因此在通常的非退化区域内可识别的独立函数方向最多只有 3 个。**不能要求仅从函数恢复全部 4 个真参数。**

### T5 — 同一刺激的线性/二次效应

\[
\boxed{\eta_5(x_1,x_2;a,b,c,d)=a x_1+b x_1^2+c x_2+d.}
\]

\[
\partial_{x_1}\nabla_\theta\eta_5=(1,\,2x_1,\,0,\,0)^\top,\qquad
\partial_{x_1x_1}\nabla_\theta\eta_5=(0,\,2,\,0,\,0)^\top.
\]

同一个局部 XU 导数涉及多个真参数，不能强行要求每个局部导数只作用于单一参数。对 X 求均值/协方差、二阶导数等整体算子可能分离其作用。

### T6 — 刺激交互

\[
\boxed{\eta_6(x_1,x_2;a,b,c,d)=a x_1+b x_2+c x_1x_2+d.}
\]

\[
\partial_{x_1x_2}\eta_6=c,\qquad
\partial_c\partial_{x_1x_2}\eta_6=1.
\]

测试 XXU 高阶算子与跨 X 交互参数的可分离性。

### T7 — 指数幅度与指数斜率

\[
\boxed{\eta_7(x;a,b)=a e^{bx}.}
\]

对数尺度下 \(\log\eta_7=\log a+bx\)，与 T1 的幅度/形状分解进行对比。这也是自动发现新 toy 插件的单文件示例。

### T8 — 为 T4 增加独立识别通道

\[
\boxed{\eta_8(x_1,x_2,x_3;a,b,c,d)
=a x_1^b+cd x_2^b+c x_3.}
\]

当 \(c>0\)，且 X 的变化具有足够丰富的独立信息时，\(cx_3\) 为 \(c\) 提供一个独立的函数作用，进而可能把 \(d\) 与 \(c\) 分离。与 T4 成对构成可识别性正/负对照。

### T9 — 原项目 C2 的四参数 logit 函数

物理 X 坐标顺序必须严格保持为
\((X_{1A},X_{1B},X_{2A},X_{2B})\)。

\[
\boxed{\begin{aligned}
\eta_9(X,b)={}&b_1+
\sqrt3\,b_2(X_{1A}-X_{1B})\\
&+\frac{\sqrt5}{2}b_3
\left[(3X_{1A}^2-1)-(3X_{1B}^2-1)\right]\\
&+\sqrt3\,b_4(X_{2A}-X_{2B}).
\end{aligned}}
\]

使用 \`--wrappers sigmoid\` 时，
\[
\boxed{P(Y=1\mid X,b)=\sigma(\eta_9(X,b)).}
\]

该公式来自原项目的 C2 生成器，但本 toy 仓库改用每个 \(b_k\) 与 X 维度均在 0–1 范围的参考分布：**公式相同，不代表原数据设计、参数先验或模型恢复性能相同**。

在原始 logit 函数尺度，
\[
\partial_{X_{1A}}\nabla_b\eta_9
=(0,\sqrt3,3\sqrt5 X_{1A},0)^\top.
\]
因此真实参数下的局部梯度本来也可以涉及 \(b_2,b_3\) 两个坐标；这应被用于检验错误的局部稀疏假设。

## 4. 生成、积分与 U-sweep 设置

- **X / θ 联合参考积分：** Sobol 节点 \(\theta^{(n)}\in(0,1)^K\)、\(x^{(t)}\in(0,1)^{D_X}\)。调用 \`paired_grid\` 做完整交叉组合；\`--u-points\` 是 θ/U 数值求积节点数，\`--x-points\` 是 X 节点数，**不是被试和试次**。
- **X 各维单轴扫描：** \`--x-mode axis_sweeps\`；每个 X 维度使用 \`--x-points\` 个中点，其他 X 坐标固定为 0.5。
- **角度：** \`--pairs first|all|i,j\`，\`--angles\`；0° 是参数对齐的原始坐标。
- **U 单轴 sweep：** 默认在原参数中心 \(\theta_0=(0.5,\ldots,0.5)\) 沿第 \(k\) 个旋转坐标方向扫描：\(\theta(s)=\theta_0+sR_{k\cdot}\)、\(s\in[-0.45,0.45]\)；\`--sweep-steps\` 默认 101。
- **U 双轴矩形扰动：** \(\theta=\theta_0\pm hR_{k\cdot}\pm hR_{l\cdot}\)，默认 \`--rectangle-step 0.2\`。
- **网络近似：** \`--backend analytic|neural|both\`；当前 neural 为输入 \((\theta,X)\) 预测包装后 \(Y\) 的 MLP，尚不是 NP/MP，训练好后在旋转间共享同一个 checkpoint。

**结论的检验边界：** 此处的 toy 只在规定的正交线性旋转集合和参考积分测度下比较函数性质；一个指标在某个模型的 0° 最小，并不意味着对所有模型都能恢复真参数。

## 5. 运行命令与以后新增模型的维护规则

\`\`\`bash
python run.py --list-models
python run.py --models T1 T3 T9 --wrappers sigmoid --backend analytic
python run.py --models T9 --wrappers sigmoid --pairs all --u-points 100 --x-points 100
\`\`\`

以后新增 T10 时，在 \`src/rottoy/toy_models/t10.py\` 中用 \`@register_model("T10",k=...,dx=...,description=...)\` 注册原始可微生成函数，然后**在本文件增加 T10 表格行及一段准确数学公式**；不要改动已有 T 编号，也不要建立 T10 专用的指标/绘图接口。

验证当前自动注册结果：

\`\`\`bash
python tests/test_registry.py
\`\`\`
