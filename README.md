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

在旧 v2 的**基础 logit 数学尺度**下探测 T9：

```bash
python run.py --models T9-sig --metrics M017 M018 M020 --operator-scale inverse --pairs all --u-points 100 --x-points 100 --angles 181
```

比较解析函数与神经逼近（神经端目前是小型连续输出 MLP，为未来 MP/NP 留好共同接口）：

```bash
python run.py --models T3-id T9-sig --backend both --train-neural --nn-epochs 120 --metrics M001 M004 M017 M018
```

结果全部保存在 `outputs/toy_rotation/` 下。每个 T/后缀/坐标对均产生 `metrics_by_angle.csv`、`metric_heatmap.png`、`metric_curves.png`、`metric_optima.csv`；`--backend both` 还输出解析/网络曲线与差异热力图。

## 扩展原则

1. **新增 T10：** 新建 `toy_models/t10.py`；使用 `@register_model("T10",k=...,dx=...,description=...,eta_bounds=(L,H))` 注册**基础连续函数与完整定义域的解析范围**。六种变体的 Y 范围自动推导，不需要给每个版本写一套公式。
2. **新增输出变换：** 仅在 `transforms.py` 登记后缀、数学前向、逆函数可用性、神经输出头和区间映射；对所有 T 模型自动生效。
3. **新增 M025：** 在 `metrics/specs.py` 末尾注册名称、公式、计算方式与含义，再向公共算子/指标实现添加方法；不为特定 T 编写指标。
4. **数学范围不等于样本范围：** 各参数/X 的 $[0,1]$ 是物理参考定义域；toy 文档给出严格 $\eta,Y$ 理论区间。神经近似只自动满足包装器固有值域，可能偏离某个 T 模型更窄的边界。
5. **共享数学算子：** 神经端的函数值、梯度、高阶导数与解析端分别验证；不因旧版方法曾在 NP 任务中成功就断言 0° 必然最小。
6. **迁移旧命令：** `--models T9 --wrappers sigmoid` 改为 `--models T9-sig`；`--analysis-scale inverse-link` 改为 `--analysis-scale inverse`。

**一切实验输出仅在 gitignored 的 `outputs/` 内；源码、模型定义、指标定义和文档放在 GitHub。**
