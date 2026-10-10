# BUL 论文实验参数冻结说明（Priority 0）

## 1. 冻结原则

本文件是论文参数说明的 Priority 0 口径。它记录已经用于正式数值实验的参考参数，
不触发重新选参，不修改既有配置或结果。

## 2. 共同参数

| 参数 | 冻结值 | 论文用途 |
|---|---:|---|
| `Top-k` | 100 | 主实验固定 variant 统计域 |
| edge support `tau` | 10 | 主实验固定 edge 统计域参考阈值 |
| epsilon | 0.5, 1, 2, 4, 8 | 隐私预算水平 |
| seed | 0–29 | 30 次独立且可复现的随机运行 |
| `delta` | `1e-8` | DP 时间均值/方差后处理的数值下界 |
| `epsilon_ST` | `epsilon_T/2` | 时间和发布预算 |
| `epsilon_QT` | `epsilon_T/2` | 时间平方和发布预算 |
| `Z_e` 零尺度回退 | 1 | 仅用于 nEMD 评价归一化 |

预算策略保持 `uniform`、`activity_focus`、`edge_focus`、
`variant_focus` 和 `time_focus`。时间预算的二次拆分属于顺序组合。

## 3. 数据集级冻结参数

以下数值来自发布版已冻结的 `configs/experiment_config_final.json`：

| 数据集 | `Lmax` | `Tmax`（秒） | `Top-k` | `tau` | 冻结 `Z_e` edge 数 |
|---|---:|---:|---:|---:|---:|
| BPIC2012 | 58 | 258995.1866 | 100 | 10 | 112 |
| BPIC2017 | 70 | 326449.4304 | 100 | 10 | 138 |
| SEPSIS | 30 | 259200.0 | 100 | 10 | 77 |

`Lmax` 为 trace 长度的 P95 校准值；`Tmax` 为所有有效 transition interval
样本的 P95 校准值，不是 case duration。每条 edge 的 `Z_e` 按以下规则冻结：

```text
Z_e = P95(T_e) - P5(T_e)

if Z_e == 0:
    Z_e = 1
```

正的非零尺度保持原值，不执行下界为 1 的截断。`Z_e` 仅作为 nEMD 的评价
Normalization factor，不是 DP 发布统计量，也不进入 Synthetic Event Log
生成。

## 4. Top-k 报告字段

论文或补充材料中，每个数据集的固定 `k=100` 至少报告以下字段：

| 字段 | 含义 |
|---|---|
| `dataset` | 数据集标识 |
| `k` | 固定候选值，主实验为 100 |
| `effective_k` | 实际存在的具体 Top-k variant 数 |
| `total_cases` | 完整 variant 分布中的 case 总数 |
| `top_k_case_count` | Top-k 具体 variant 覆盖的 case 数 |
| `coverage` | `top_k_case_count / total_cases` |
| `other_count` | Top-k 外聚合到 OTHER 的 case 数 |
| `other_mass` | `other_count / total_cases = 1 - coverage` |

`coverage` 和 `other_mass` 仅用于说明固定统计域的信息保留程度。主实验不设置
95% coverage 选择规则，不依据 coverage 改变 `k`。已有
`variant_coverage.csv` 中若保留早期 `meets_threshold` 列，该列不进入论文级
选参逻辑，也不得据此声称 `k=100` 满足 95% 覆盖率。

## 5. Conditional DP 参数边界

本实验的隐私声明以以下参数和统计域已冻结为条件：

- 数据集级 `Lmax`；
- 数据集级 `Tmax`；
- 全部数据集固定的 `Top-k=100`；
- 数据集级固定 activity、edge、variant domain；
- 数据集级固定 `Z_e`。

上述公共基准校准未计入隐私预算，因此论文使用“conditional DP”，不得使用
全流程差分隐私措辞。对于私有部署，需改用公开/预声明参数，或对私有选参另行进行
隐私会计。

## 6. 结果影响

本文件只同步论文表述和复现配置说明。它不改变参数数值、不改写结果表，也不要求
重跑 3150 条正式实验。
