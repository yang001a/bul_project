# BUL 最终实验协议

## 1. 协议状态与研究问题

本协议对应 BUL SCI 论文可复现冻结版本。其目标是量化 case-level DP 事件日志
统计发布造成的行为效用损失，并区分：

- 结构行为损失；
- 时间行为损失；
- 信息分布损失；
- 三维等权聚合的总体 BUL。

BUL 是已有指标的融合评价框架，不是新的基础距离或新的隐私机制。本协议不改变
JS、Cosine loss、nEMD、Entropy、BUL aggregation 或 Laplace 机制。

## 2. 数据与规范化

数据集为 BPIC2012、BPIC2017 和 Sepsis Cases。XES 事件统一映射为：

```text
case_id
activity
timestamp
```

同一 case 内按 timestamp 稳定排序。缺失 activity 的事件删除并审计；缺失或
非法 timestamp 的事件保留在活动序列中，但不参与时间间隔计算。负间隔作为数据
质量问题报告，禁止取绝对值。

Phase 1 输出事件表、trace、清洗报告、数据集报告和数据源哈希。

## 3. 裁剪与统计空间

### 3.1 Case-level 裁剪

每个数据集的 `Lmax` 由 trace 长度 P95 校准后冻结。每个 case 最多保留
`Lmax` 个事件，因此最多贡献 `Lmax-1` 条相邻 edge。

`Tmax` 由全部有效 transition interval 的 P95 校准后冻结；每个非负时间间隔
裁剪到 `[0,Tmax]`。

### 3.2 Variant 与 edge domain

主实验固定：

```text
Top-k = 100
edge support tau = 10
```

Top-k 之外的 variant 聚合为 `OTHER`。Coverage 与 OTHER mass 仅用于诊断固定
统计域的覆盖质量，不用于选择 `k`，也不作为实验通过条件。

Raw、ClippedRaw 与 DP 结果投影到相同的冻结 domain。不得根据 epsilon、方法、
seed 或 noisy 输出重新选择 Top-k 或 edge domain。

## 4. Case-level DP 发布

邻接关系采用 add/remove-one case。敏感度为：

```text
Delta_activity = Lmax
Delta_edge     = Lmax - 1
Delta_variant  = 1
Delta_ST       = (Lmax - 1) * Tmax
Delta_QT       = (Lmax - 1) * Tmax^2
```

DP-full 发布 Activity、Edge、Variant、时间和 `S_T` 与时间平方和 `Q_T`。
四类统计按顺序组合：

```text
epsilon_A + epsilon_E + epsilon_V + epsilon_T = epsilon_total
```

时间预算继续按顺序组合：

```text
epsilon_ST = epsilon_T / 2
epsilon_QT = epsilon_T / 2
```

所有统计使用既有 Laplace 机制，并记录 dataset、epsilon、预算策略、seed、
敏感度、预算与冻结 domain 元数据。

## 5. DP 时间重构

令 `delta=1e-8`，仅由 DP 输出计算：

```text
mu'_e = S'_T(e) / max(C'_E(e), delta)

sigma'^2_e =
    max(
        Q'_T(e) / max(C'_E(e), delta) - (mu'_e)^2,
        delta
    )
```

生成样本数量：

```text
N'_e = max(round(C'_E(e)), 1)
```

随后生成：

```text
T'_e ~ TruncatedNormal(mu'_e, sigma'^2_e; 0, Tmax)
```

时间重构与合成日志生成只能访问 DP 输出，统一使用实验 seed。禁止访问原始
trace、原始时间样本或未经 DP 保护的 variant 分布。

时间维度比较的是原始经验时间分布与“DP 矩统计 + 截断高斯后处理”所得分布。
因此观测到的时间损失同时包含 DP 扰动影响与矩重构/分布族近似造成的误差底线，
不能将其全部表述为 Laplace 噪声损失。高 epsilon 下的时间损失平台应解释为
DP transformation（含重构后处理）的剩余失真。

## 6. 指标与 BUL

全部进入 BUL 的量统一为“数值越大，损失越大”：

- JS divergence；
- `1 - cosine_similarity`；
- 固定 `Z_e` 归一化后的 nEMD；
- 有界 Entropy deviation。

DP-full 的维度为：

```text
L_structure_full
  = mean(Edge JS, DFG Cosine loss, Variant JS)

L_temporal
  = fixed-edge weighted nEMD

L_information
  = mean(Activity entropy loss, Edge entropy loss, Activity JS)

BUL_equal
  = mean(L_structure_full, L_temporal, L_information)
```

三个维度必须分别报告。不得将裁剪、DP 噪声和总损失解释为线性可加关系。

## 7. DP-edge-only 解释协议

DP-edge-only 只发布 edge count，因此只定义：

```text
L_structure_edge_only
  = mean(Edge JS, DFG Cosine loss)
```

其 `L_temporal`、`L_information` 和 BUL 必须为 NA。

正式 CSV 为保持统一字段名，DP-full 与 DP-edge-only 都使用 `L_structure` 列。
论文制表和解释必须按 `method` 映射为：

| CSV 条件 | 论文名称 |
|---|---|
| `method=DP-full` | `L_structure_full` |
| `method=DP-edge-only` | `L_structure_edge_only` |

二者的组成不同，不能直接作为同一个结构指标比较绝对水平，也不能把两者差值解释
为 Variant 发布带来的净效应。允许在各自方法族内部比较 epsilon、模式或 seed。

`component` 模式使用 `epsilon_total/4`；`mechanism` 模式使用
`epsilon_total`。前者用于主实验参照，后者用于附录机制参照。

## 8. 正式实验矩阵

### 8.1 主矩阵

| 组成 | 条件数 |
|---|---:|
| DP-full：3 数据集 × 5 epsilon × 5 策略 × 30 seed | 2250 |
| DP-edge-only：3 数据集 × 5 epsilon × 2 模式 × 30 seed | 900 |
| DP 条件总计 | 3150 |
| Raw 与 ClippedRaw 基线 | 6 |
| `experiment_result.csv` 总行数 | 3156 |

epsilon 为 `{0.5,1,2,4,8}`，seed 为 `0–29`。

### 8.2 参数敏感性

参数敏感性只在 BPIC2017、DP-full、uniform 上执行：

- `Top-k ∈ {50,100,200}`；
- `tau ∈ {5,10,20}`；
- `epsilon ∈ {0.5,2,8}`；
- seed `0–29`。

共 810 条结果。该分析检查固定参考设置 `k=100,tau=10` 的结论稳健性，不重新
选择主参数。

### 8.3 统计分析

论文统计分析包括描述统计、bootstrap CI、Wilcoxon、Friedman、Holm 校正、
配对效应量和 epsilon 的 Spearman 趋势。Spearman 点估计基于 5 个 epsilon
水平的均值，p 值通过穷举 `5!=120` 种排列得到精确双侧检验结果，不使用小样本
渐近 p 值；置信区间以 seed 为 cluster 执行 bootstrap。所有配对严格使用相同
dataset、method、策略/模式和 seed。epsilon 趋势不被强制为严格单调。

统计显著性必须与绝对差值、置信区间和效应量共同解释。基于 DP seed 的检验描述
算法随机扰动稳定性，不代表从事件日志总体重新抽样得到的总体推断。

## 9. Representative external feasibility validation

PM4Py 仅作为 representative external feasibility validation，验证以下链路能够
在代表性条件下完成：

```text
DP statistics
→ DP-consistent Synthetic Event Log
→ Inductive Miner
→ Petri Net
→ fitness / precision / generalization
```

固定条件为 BPIC2017、DP-full、uniform、seed 27，以及
`epsilon ∈ {0.5,2,8}`，共 3 条。

过程质量指标不进入 BUL。该小规模验证只支持“外部流程可执行”的可行性结论，
不执行或报告 BUL 与 fitness、precision、generalization 的相关性检验，也不用于
证明 BUL 的构念有效性、预测有效性或跨数据集普适性。

现有结果表中的 `external_validation_role` 等早期字段名按
`feasibility_only` 解释，不扩大证据强度。

## 10. Conditional DP 声明

隐私保证以每个数据集的 `Lmax`、`Tmax`、`Top-k=100`、冻结统计域和 `Z_e`
已经固定为条件。公共基准上的参数校准没有计入隐私预算。

因此论文只作 conditional DP 声明。`Z_e` 仅为 nEMD 的评价 Normalization
factor，不是发布量，不进入合成日志生成。面向私有部署时，应预先公开这些参数，
或将其私有选择纳入独立的隐私会计。

## 11. 结果解释边界

论文可以声称：

- BUL 分别量化结构、时间和信息行为损失；
- 固定实验条件下不同 epsilon 与预算策略产生不同效用损失；
- 参数敏感性和维度消融用于检查结论稳健性；
- 跨数据集只比较 epsilon 响应趋势和相对变化，不进行绝对 BUL 排名；
- 三个代表性 PM4Py 条件证明外部生成与过程发现链路可执行。

论文不得声称：

- BUL 是新的距离度量或新的 DP 算法；
- Coverage 用于选择 `Top-k=100`；
- 三个数据集使用可交换的绝对 BUL 尺度，或据此建立数据集优劣排名；
- 时间维度的全部损失均由 Laplace 噪声直接造成；
- DP-full 与 DP-edge-only 的结构损失定义相同；
- 裁剪损失与 DP 噪声损失线性相加；
- 三个 PM4Py 条件建立了 BUL 与过程质量指标的统计关系；
- 公共基准的参数校准过程本身已纳入当前隐私预算。
