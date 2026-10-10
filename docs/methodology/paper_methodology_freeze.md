# BUL 论文方法冻结说明（Priority 0）

## 1. 适用范围

本说明同步现有 BUL 工程与论文方法表述，不重新设计实验，不修改任何核心指标、
BUL 聚合或 Laplace 机制，也不改变已经完成的 3150 条正式实验结果。

本说明覆盖四个必须冻结的方法口径：

1. 固定 Top-k 统计域及其覆盖质量报告；
2. 二阶时间统计的 DP 发布与截断高斯后处理；
3. conditional DP 的声明边界。
4. DP-full 与 DP-edge-only 的结构损失定义及比较边界。

## 2. 固定 Top-k 与 OTHER

### 2.1 主实验参数

主实验固定使用：

```text
Top-k = 100
```

该值在正式实验前预先冻结，并在三个数据集、全部 epsilon、全部方法和全部 seed
之间保持不变。主实验不使用 95% coverage 阈值选择 `k`，也不根据每个 noisy
结果重新选取 Top-k。

Top-k 之外的具体 variant 聚合为 `OTHER`。完整 variant 计数仍被保留；Top-k
视图仅定义对齐后的统计空间。

### 2.2 覆盖质量

对数据集中的 variant 计数 `C(v)`，固定 `k` 的覆盖率定义为：

```text
Coverage(k) = [sum_{v in Top-k} C(v)] / [sum_v C(v)]
```

OTHER 的计数和质量定义为：

```text
C(OTHER)  = sum_{v not in Top-k} C(v)
OTHER mass = C(OTHER) / sum_v C(v)
           = 1 - Coverage(k)
```

`Coverage(k)` 与 `OTHER mass` 是固定 `k=100` 的描述性报告量，不是选参目标、
阈值约束或通过/失败判据。任何早期产物中的 `meets_threshold` 字段均不用于论文
选参结论；为保持既有结果不可变，不删除或重写已有 CSV，但论文级表格不把该字段
作为方法依据。

### 2.3 OTHER 的用途边界

`OTHER` 仅用于：

- Variant 分布统计；
- Variant JS 和后续 BUL 计算中的统一统计空间。

`OTHER` 不表示一条可执行的具体活动序列，因此不参与 Synthetic Event Log 生成。
合成日志只使用 Top-k 中的具体 variant，并在排除 `OTHER` 后重新归一化 DP variant
计数。

## 3. 时间 DP 发布

### 3.1 查询

对冻结 edge domain 中的每条边 `e`，裁剪后的相邻事件间隔为
`t in [0, Tmax]`。发布以下时间统计：

```text
S_T(e) = sum t
Q_T(e) = sum t^2
```

同时使用 DP edge count `C'_E(e)` 作为重构时的样本数量依据。

### 3.2 Case-level 敏感度

邻接关系为 add/remove-one case。单个 case 经 `Lmax` 裁剪后最多贡献
`Lmax-1` 条 transition，因此：

```text
Delta_ST = (Lmax - 1) * Tmax
Delta_QT = (Lmax - 1) * Tmax^2
```

其余查询的敏感度保持：

```text
Delta_activity = Lmax
Delta_edge     = Lmax - 1
Delta_variant  = 1
```

### 3.3 预算组合

四类统计按顺序组合：

```text
epsilon_A + epsilon_E + epsilon_V + epsilon_T = epsilon_total
```

时间预算内部进一步按顺序组合平均拆分：

```text
epsilon_ST = epsilon_T / 2
epsilon_QT = epsilon_T / 2
```

该过程不得描述为并行组合。

### 3.4 DP 后处理和时间重构

令 `delta=1e-8`。仅基于 DP 发布结果计算：

```text
mu'_e = S'_T(e) / max(C'_E(e), delta)

sigma'^2_e =
    max(
        Q'_T(e) / max(C'_E(e), delta) - (mu'_e)^2,
        delta
    )
```

每条 edge 的生成样本数为：

```text
N'_e = max(round(C'_E(e)), 1)
```

时间样本按以下截断高斯分布生成：

```text
T'_e ~ TruncatedNormal(mu'_e, sigma'^2_e; lower=0, upper=Tmax)
```

时间重构是 DP 后处理。它不得访问原始 trace、原始时间样本或未经 DP 保护的
variant 分布；不得平移原始样本、重复 DP 均值代替分布，或用原始样本修正负方差。
随机数生成器统一使用实验 seed。

## 4. DP-full 与 DP-edge-only 冻结定义

### 4.1 结构损失定义

DP-full 发布 Edge 和 Variant 统计，其完整结构损失定义为：

```text
L_structure_full
  = mean(
      Edge JS,
      DFG Cosine loss,
      Variant JS
    )
```

DP-edge-only 只发布 Edge count，其结构损失定义为：

```text
L_structure_edge_only
  = mean(
      Edge JS,
      DFG Cosine loss
    )
```

`L_structure_full` 与 `L_structure_edge_only` 的组成不同，不得将二者视为同一
结构指标进行直接等价比较，也不得将二者差值解释为 Variant 发布带来的净效应。
允许在各自方法族内部比较 epsilon、模式或 seed。

### 4.2 不适用维度

DP-edge-only 不发布 Variant 或时间统计，因此以下字段必须记为 NA：

```text
L_temporal = NA
L_information = NA
BUL = NA
```

不得通过假设、插补或其他未经 DP 保护的数据补充这些维度。

### 4.3 隐私预算模式

DP-edge-only 的两种模式分别定义为：

```text
component mode:
epsilon_edge = epsilon_total / 4

mechanism mode:
epsilon_edge = epsilon_total
```

`component` 模式用于主实验的 component-level 参照；`mechanism` 模式用于附录
的 mechanism-level 参照。两种模式必须独立记录、独立制表和独立解释，禁止混用。

## 5. Conditional DP 声明

### 5.1 允许的声明

论文允许声明：给定冻结的裁剪边界、统计域和评价归一化因子，Activity、Edge、
Variant、`S_T` 与 `Q_T` 的 Laplace 发布在 case-level add/remove-one 邻接关系下
满足相应预算的 conditional DP，并按上述顺序组合进行隐私会计。

### 5.2 条件中冻结的对象

以下对象在每个数据集上于正式实验前固定，并在 epsilon、方法和 seed 之间保持
不变：

- `Lmax`；
- `Tmax`；
- `Top-k=100`；
- activity、edge 和 variant 统计域，尤其是 edge domain；
- nEMD 的评价归一化因子 `Z_e`。

其中，首先计算原始评价尺度：

```text
Z_e = P95(T_e) - P5(T_e)

if Z_e == 0:
    Z_e = 1
```

正的非零尺度保持原值，即使 `0 < Z_e < 1` 也不替换为 1。`Z_e` 只用于 nEMD
评价归一化，不是 DP 发布统计量，也不作为 Synthetic Event Log 的输入。

### 5.3 禁止的声明

公共基准实验中的 `Lmax`、`Tmax`、统计域和 `Z_e` 由原始基准日志校准后冻结，
该校准过程未纳入隐私预算。因此不得声称整个数据读取、校准、统计域选择、DP
发布、评价和合成日志流程具有全流程差分隐私保证。

若面向真实私有部署，需要预先公开/声明这些边界与统计域，或将其私有选择纳入
额外的隐私预算和完整隐私会计。这一部署要求不改变本文公共基准上的数值实验。

## 6. 不变项

本次方法同步不修改：

- JS divergence；
- Cosine loss；
- EMD/nEMD；
- Entropy；
- BUL aggregation；
- Laplace DP 机制；
- 既有 `experiment_result.csv`；
- 既有 3150 条正式条件结果。
