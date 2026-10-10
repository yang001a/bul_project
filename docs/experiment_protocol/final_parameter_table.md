# BUL 最终参数表

## 1. 主实验共同参数

| 参数 | 冻结值 | 角色 |
|---|---:|---|
| 数据集 | BPIC2012、BPIC2017、SEPSIS | 跨数据集主实验 |
| `Top-k` | 100 | 固定 Variant 统计域 |
| edge support `tau` | 10 | 固定 edge domain 参考阈值 |
| epsilon | 0.5、1、2、4、8 | 实验级隐私预算 |
| seed | 0–29 | 30 次可复现运行 |
| `Lmax` 分位数 | P95 | case 贡献裁剪 |
| `Tmax` 分位数 | transition interval P95 | 时间贡献裁剪 |
| `Z_e` | `P95(T_e)-P5(T_e)`；仅当精确为0时置1 | nEMD 评价归一化，不是DP发布统计量 |
| DP 后处理 `delta` | `1e-8` | 分母与方差下界 |
| 主 BUL 权重 | 1/3、1/3、1/3 | 结构、时间、信息等权 |
| 时间预算内部权重 | 1/2、1/2 | `S_T` 与 `Q_T` 顺序组合 |

## 2. 数据集级冻结参数与 Coverage

| 数据集 | case 数 | `Lmax` | `Tmax`（秒） | `k` | Top-k case 数 | Coverage(100) | OTHER 数 | OTHER mass | `Z_e` edge 数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BPIC2012 | 13087 | 58 | 258995.1866 | 100 | 7899 | 0.603576 | 5188 | 0.396424 | 112 |
| BPIC2017 | 31509 | 70 | 326449.4304 | 100 | 9689 | 0.307499 | 21820 | 0.692501 | 138 |
| SEPSIS | 1050 | 30 | 259200.0 | 100 | 304 | 0.289524 | 746 | 0.710476 | 77 |

Coverage 和 OTHER mass 来源于冻结的 `variant_coverage.csv`。它们只描述固定
`k=100` 的覆盖质量，不是选择规则。较大的 OTHER mass 必须在论文限制部分透明
报告，但不得据此事后改变主实验 `k`。

`Z_e` 的完整冻结规则为：

```text
Z_e = P95(T_e) - P5(T_e)

if Z_e == 0:
    Z_e = 1
```

正的非零尺度保持原值。`Z_e` 仅用于 nEMD 评价归一化，不是 DP 发布统计量，
也不进入 Synthetic Event Log 生成。

## 3. DP-full 预算策略

下表中的值是 `epsilon_total` 的比例：

| 策略 | Activity | Edge | Variant | Time |
|---|---:|---:|---:|---:|
| `uniform` | 0.25 | 0.25 | 0.25 | 0.25 |
| `activity_focus` | 0.40 | 0.20 | 0.20 | 0.20 |
| `edge_focus` | 0.20 | 0.40 | 0.20 | 0.20 |
| `variant_focus` | 0.20 | 0.20 | 0.40 | 0.20 |
| `time_focus` | 0.20 | 0.20 | 0.20 | 0.40 |

每行均按同一 case 总体进行顺序组合且总和为 1。Time 预算内部继续等分为
`epsilon_ST` 和 `epsilon_QT`。

## 4. 敏感度与噪声尺度

| 查询 | 敏感度 | Laplace scale |
|---|---|---|
| Activity count | `Lmax` | `Lmax/epsilon_A` |
| Edge count | `Lmax-1` | `(Lmax-1)/epsilon_E` |
| Variant count | `1` | `1/epsilon_V` |
| `S_T(e)=sum(t)` | `(Lmax-1)Tmax` | `Delta_ST/epsilon_ST` |
| `Q_T(e)=sum(t^2)` | `(Lmax-1)Tmax^2` | `Delta_QT/epsilon_QT` |

## 5. DP-edge-only 参数与指标名称

| 模式 | Edge 预算 | 论文角色 |
|---|---:|---|
| `component` | `epsilon_total/4` | 主实验 component-level 参照 |
| `mechanism` | `epsilon_total` | 附录 mechanism-level 参照 |

结构损失名称必须区分：

| 方法 | 论文报告名 | 组成 | 其他维度 |
|---|---|---|---|
| DP-full | `L_structure_full` | Edge JS、DFG Cosine loss、Variant JS | 正常计算 |
| DP-edge-only | `L_structure_edge_only` | Edge JS、DFG Cosine loss | Temporal、Information、BUL 为 NA |

正式 CSV 的物理列名仍是 `L_structure`；不修改该文件。论文层根据 `method`
映射报告名。两个结构损失的组成不同，禁止直接等价比较。

## 6. 参数敏感性范围

| 参数 | 值 |
|---|---|
| 数据集 | BPIC2017 |
| 方法 | DP-full |
| 预算策略 | uniform |
| Top-k | 50、100、200 |
| edge support `tau` | 5、10、20 |
| epsilon | 0.5、2、8 |
| seed | 0–29 |
| 条件数 | 810 |

参考设置始终是 `Top-k=100,tau=10`。敏感性结果用于稳健性解释，不用于重新选择
参考参数。

## 7. Representative external feasibility validation 参数

| 参数 | 冻结值 |
|---|---|
| 数据集 | BPIC2017 |
| 方法 | DP-full |
| budget strategy | uniform |
| seed | 27 |
| epsilon | 0.5、2、8 |
| 条件数 | 3 |
| Inductive Miner noise threshold | 0.0 |
| conformance variant | token-based |
| 输出 | Synthetic XES、Petri Net、fitness、precision、generalization |
| 证据角色 | representative external feasibility validation |

过程质量指标不进入 BUL，不进行 BUL 与这些指标的相关性检验。

## 8. 正式软件环境

| 软件 | 版本 |
|---|---:|
| CPython | 3.13.9 |
| NumPy | 2.3.5 |
| Pandas | 2.3.3 |
| SciPy | 1.16.3 |
| Matplotlib | 3.10.6 |
| Seaborn | 0.13.2 |
| PM4Py | 2.7.23.1 |
| lxml | 5.3.0 |
| pytest | 8.4.2 |

完整传递依赖见 `requirements-lock.txt`。
