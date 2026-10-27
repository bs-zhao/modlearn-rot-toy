# modlearn-rot-toy

独立的**连续函数旋转数学测试台**。基础模型统一编号 `T1、T2、…`，输出变换统一后缀 `-id/-sig/-sq/-tanh/-sp/-exp`；指标统一编号 `M001、M002、…`。全部输出都是连续数值，不采样二分类标签。每项指标在解析函数 $\eta'$ 和神经网络近似 $F'$ 上使用**同一**数学定义、数值节点和绘图程序。

## 详细参考文档

- **[全部 T 模型：生成公式、每个参数/X 的范围、基础 η 和六种 Y 版本的理论上下界](catalog/README_TOY_MODELS.md)**
- **[全部 24 项 M 指标：数学公式、意义、近似算法、局限与正交不变量](catalog/README_METRICS.md)**

新增 T/M 编号时，文档总表与实际代码一一对应；测试会检查覆盖情况。

## 新项目代码布局

```text
src/rottoy/
  toy_models/           # T1、T2、...：只有基础 η 的生成公式
  models.py             # toy 注册、T编号-后缀解析、eta_bounds、平面旋转
  transforms.py         # 所有 toy 共享的六种连续输出变换及精确范围映射
  backends.py           # analytic eta 或神经近似 F；连续 Y 的 MSE
  sampling.py           # X 与 θ/U 的合法数值积分节点
  operators/
    derivatives.py      # X/U 导数 bank、梯度与 Hessian
    functionals.py      # X积分 → 外积 → U积分 → 算子矩阵
  metrics/
    specs.py            # 稳定的 M001–M024 名称、数学式、实现方法、解释
    pointwise.py        # M001–M011
    sweeps.py           # M012–M016
    __init__.py         # M017–M024 与统一 compute_metrics
  plotting.py           # 指标曲线、角度热力图、解析 η 对比网络 F
  runner.py             # 唯一实验运行器和参数解析
  __main__.py           # 包入口
catalog/
  README_TOY_MODELS.md
  README_METRICS.md
tests/
  test_core.py          # 数学检验、参数/X/Y 范围、算子一致性
  test_e2e.py           # 解析/神经双后端完整小规模测试
run.py                  # 无需 pip install -e . 的独立入口
outputs/                # 所有生成文件；gitignore，不推送 GitHub
```

## 快速运行

```bash
python -m pip install -r requirements.txt
python run.py --list-models
python run.py --list-metrics
python run.py --describe-metric M018
python tests/test_core.py
python tests/test_e2e.py
```

比较多个 toy 变体（裸 `T3` 表示 `T3-id`）：

```bash
python run.py --models T1-id T3-sq T9-sig --metrics M001 M006 M017 M018 --pairs all --u-points 32 --x-points 100 --angles 37
```

新增 **T10-sig**（连续 Y = sigmoid(a·X1^b + a·c·X2^b + d)），四参数联合旋转：

```bash
python run.py --models T10-sig --metrics all --rotation-mode joint-path --joint-path ramped --backend analytic --u-points 24 --x-points 100 --angles 73 --sweep-steps 61
```

需要搜索全部六个独立旋转角度时，可以用 `--rotation-mode joint-search` 并优先只选择少数 M 指标。

在旧 v2 的**基础 logit 数学尺度**下探测 T9：

```bash
python run.py --models T9-sig --metrics M017 M018 M020 --operator-scale inverse --pairs all --u-points 100 --x-points 100 --angles 181
```

比较解析函数与神经逼近（神经端目前是小型连续输出 MLP，为未来 MP/NP 留好共同接口）：

```bash
python run.py --models T3-id T9-sig --backend both --train-neural --nn-epochs 120 --metrics M001 M004 M017 M018
```

结果全部保存在 `outputs/toy_rotation/` 下。每个 T/后缀/坐标对均产生 `metrics_by_angle.csv`、`metric_heatmap.png`、`metric_curves.png`、`metric_optima.csv`；`--backend both` 还输出解析/网络曲线与差异热力图。

## 全部参数一起旋转：SO(K) 联合实验

先前的 `--pairs all` 仅是**两两平面旋转的多组独立实验**。它不是多维联合旋转。现在用统一命令 `--rotation-mode`：

| 模式 | 数学操作 | 输出 | 是否覆盖完整旋转空间 |
|---|---|---|---|
| `pair`（默认，兼容旧版） | 单个平面 `G_ij(phi)`；`--pairs all` 遍历每对 | `u1_u2/` 等角度曲线/热力图 | 否 |
| `joint-path` | `R(phi)=G_K(c_K phi)···G_1(c_1 phi)`，**全部平面同时变化** | `joint_ramped/` 等角度曲线/热力图 | 否：只是 SO(K) 中的一条一维路径 |
| `joint-search` | `R(phi_1,...,phi_D)`，`D=K(K-1)/2` 个角度独立变化 | `joint_search/` 中各指标优化结果/旋转矩阵图 | 搜索全部自由度；局部数值优化不保证全局最优 |
| `all` | 上述三类全部运行 | 三类输出 | 包含完整自由度搜索 |

对于 T8 (`K=4`)，联合旋转有**六个独立角度**（12、13、14、23、24、34）。它们采用按字典序固定的 Givens 乘积，确保任何评价矩阵都有 `R.T @ R = I` 且 `det R=+1`。

**联合旋转路径（直接画与过去相同格式的图）：**

```bash
python run.py --models T8-id --metrics all --rotation-mode joint-path --joint-path ramped --u-points 24 --x-points 100 --angles 73 --sweep-steps 61
```

这条路径横轴仍是 `phi`，但每个平面实际旋转 `c_ij * phi` 度。内置路径权重：`balanced`（全相同）、`alternating`（正负交替）、`ramped`（不同幅度，默认）、`random`（固定随机种子）。所有路径在 **phi=0** 时都等于真实坐标系 `R=I`。`rotation_matrices.npz` 保存每个画图角度对应的**完整 K×K 矩阵**及平面顺序，不能将一条路径得到的最小值当作全 SO(K) 最小值。

**真正的六角度独立优化（推荐先选少量 M 指标）：**

```bash
python run.py --models T8-id --metrics M004 M006 M017 M018 --rotation-mode joint-search --u-points 24 --x-points 100 --joint-samples 64 --joint-restarts 3 --joint-maxiter 100
```

该模式先对统一的物理 X/θ 数值节点构建指标所需算子（只计算一次），再对每个选中的 M 指标**单独**执行带随机起点的 L-BFGS-B 数值优化；不使用监督真参数方向作为损失。每项记录 identity 基准指标、找到的最小指标、全部六个角度、完整 R、它与最近的带符号置换轴的距离；`flat` 或正交不变量不进行无意义的寻优。

输出目录：

```text
outputs/toy_rotation/T8/id/raw/
  joint_ramped/
    metric_heatmap.png
    metric_curves.png
    metric_optima.csv
    metrics_by_angle.csv
    rotation_matrices.npz
  joint_search/
    joint_candidates.csv         # 多个完整四维旋转的同组指标
    joint_candidates_heatmap.png # 横轴不同六维候选旋转，纵轴 M 指标
    joint_search_summary.csv     # 每项指标在完整自由度空间发现的最小值
    best_rotations.npz           # 每项最优的六角度及 4x4 矩阵
    best_rotation_M018.png       # 按指标编号生成的系数矩阵热力图
    joint_search_improvements.png
    settings.json
```

`--backend both` 可以在解析 η 和网络 F 上分别执行相同流程，另有 `paired_joint_search_summary.csv`。**找到的最优值只是多起点局部搜索的结果，不是数学证明的全局最优。** 最优矩阵还应考虑坐标符号/置换等价，而非要求恰好 R=I。

## 扩展原则

1. **新增 T10：** 新建 `toy_models/t10.py`；使用 `@register_model("T10",k=...,dx=...,description=...,eta_bounds=(L,H))` 注册**基础连续函数与完整定义域的解析范围**。六种变体的 Y 范围自动推导，不需要给每个版本写一套公式。
2. **新增输出变换：** 仅在 `transforms.py` 登记后缀、数学前向、逆函数可用性、神经输出头和区间映射；对所有 T 模型自动生效。
3. **新增 M025：** 在 `metrics/specs.py` 末尾注册名称、公式、计算方式与含义，再向公共算子/指标实现添加方法；不为特定 T 编写指标。
4. **数学范围不等于样本范围：** 各参数/X 的 $[0,1]$ 是物理参考定义域；toy 文档给出严格 $\eta,Y$ 理论区间。神经近似只自动满足包装器固有值域，可能偏离某个 T 模型更窄的边界。
5. **共享数学算子：** 神经端的函数值、梯度、高阶导数与解析端分别验证；不因旧版方法曾在 NP 任务中成功就断言 0° 必然最小。
6. **迁移旧命令：** `--models T9 --wrappers sigmoid` 改为 `--models T9-sig`；`--analysis-scale inverse-link` 改为 `--analysis-scale inverse`。

**一切实验输出仅在 gitignored 的 `outputs/` 内；源码、模型定义、指标定义和文档放在 GitHub。**
