# GRACE 实验指标说明

本文档按实验梳理仓库中结果图和 CSV 涉及的主要指标，并记录其计算方法。说明依据当前代码实现整理，核心来源包括 `train.py`、`eval.py`、`experiments/method_comparison/run_full_pipeline.py` 以及各扩展实验脚本。

> 术语说明：用户写法 `margin_mean` 在当前代码和 CSV 中对应字段为 `mean_margin`。下文以代码字段 `mean_margin` 为准，并在首次出现时注明别名。

## 通用评估流程

除采样偏差实验外，性能对比、消融、效率、统计显著性和超参数影响分析都复用同一套节点分类评估流程。

### 节点分类评估

训练结束后，`train.py::test` 调用 `eval.py::label_classification`：

1. 使用训练好的编码器得到节点嵌入 `z`。
2. 将标签转为 one-hot。
3. 对嵌入做 L2 归一化。
4. 按 `ratio=0.1` 划分训练集与测试集，即 10% 节点用于训练线性分类器，90% 节点用于测试。
5. 使用 `OneVsRestClassifier(LogisticRegression(solver="liblinear"))` 做分类。
6. 通过 `GridSearchCV` 在 `C = 2^[-10], ..., 2^9` 上做 5 折交叉验证选择逻辑回归正则强度。
7. 在测试集上预测类别概率，并取最大概率类别转为 one-hot 预测。
8. 计算 Micro-F1 与 Macro-F1。
9. 上述评估重复 `eval_repeats` 次，若设置 `eval_seed`，第 `i` 次使用 `eval_seed + i` 作为划分随机种子。

### Micro-F1

CSV 字段：`F1Mi_mean`、`F1Mi_std`

单次评估中，Micro-F1 使用 `sklearn.metrics.f1_score(..., average="micro")` 计算：

```text
Micro-F1 = 2 * TP_all / (2 * TP_all + FP_all + FN_all)
```

其中 `TP_all`、`FP_all`、`FN_all` 是把所有类别的真阳性、假阳性、假阴性先汇总后再计算得到的整体指标。对于本项目这种单标签多分类任务，预测时每个节点只取一个最大概率类别，因此 Micro-F1 与整体分类准确率高度一致。

`label_classification` 返回：

```text
F1Mi_mean = 多次 eval_repeats 中 Micro-F1 的均值
F1Mi_std  = 多次 eval_repeats 中 Micro-F1 的标准差
```

注意：在后续实验脚本中，`F1Mi_std` 有两层含义：

- 单个训练 run 内：来自 `eval_repeats` 的 Micro-F1 标准差。
- 多个训练 run 的 summary 行内：通常表示多个 run 的 `F1Mi_mean` 的总体标准差；部分脚本还保留 `within_run_F1Mi_std_mean` 表示单个 run 内评估标准差的平均值。

### Macro-F1

CSV 字段：`F1Ma_mean`、`F1Ma_std`

单次评估中，Macro-F1 使用 `sklearn.metrics.f1_score(..., average="macro")` 计算：

```text
F1_k = 2 * TP_k / (2 * TP_k + FP_k + FN_k)
Macro-F1 = (1 / C) * sum_k F1_k
```

其中 `C` 是类别数，`F1_k` 是第 `k` 个类别的 F1。Macro-F1 对每个类别等权平均，因此更能反映类别间表现是否均衡。

`label_classification` 返回：

```text
F1Ma_mean = 多次 eval_repeats 中 Macro-F1 的均值
F1Ma_std  = 多次 eval_repeats 中 Macro-F1 的标准差
```

### robust_score

CSV 字段：`robust_score`、`robust_score_std`

`robust_score` 在 `experiments/method_comparison/run_full_pipeline.py::robust_score` 中定义：

```text
robust_score = F1Mi_mean - std_weight * F1Mi_std
```

当前各主实验脚本的 `std_weight` 默认值为 `0.5`。因此：

- `F1Mi_mean` 越高，`robust_score` 越高。
- `F1Mi_std` 越大，说明同一训练结果在多次线性评估划分中波动越大，`robust_score` 会被扣减。
- 该指标偏向选择“平均分类性能高且评估稳定”的配置。

在多次训练 run 的 summary 行中：

```text
robust_score     = 多个 run 的 robust_score 均值
robust_score_std = 多个 run 的 robust_score 总体标准差
```

## 采样偏差实验

相关代码：`train.py::experiment1_metrics`、`experiments/sampling_bias_validation/plot_sampling_bias_curves.py`、`plot/plot.py`

本实验不使用 Micro-F1、Macro-F1 或 robust_score，而是在每个训练 epoch 中记录对比学习正负样本相似度关系，用于观察采样偏差。

设两次图增强得到的节点嵌入为 `z1` 和 `z2`，先按行做 L2 归一化：

```text
z1n_i = normalize(z1_i)
z2n_j = normalize(z2_j)
```

对第 `i` 个节点：

```text
pos_sim_i = z1n_i · z2n_i
sim_ij = z1n_i · z2n_j
max_neg_sim_i = max_{j != i} sim_ij
margin_i = pos_sim_i - max_neg_sim_i
```

### violation_rate

图中名称：`violation_rate`

计算方法：

```text
violation_rate = (1 / N) * sum_i 1[max_neg_sim_i > pos_sim_i]
```

含义：最大负样本相似度超过正样本相似度的节点比例。值越高，表示越多节点的“最相似负样本”比真实跨视图正样本还相似，采样偏差越严重。

### mean_margin / margin_mean

图中名称：`mean_margin`

别名：`margin_mean`

计算方法：

```text
mean_margin = (1 / N) * sum_i margin_i
            = (1 / N) * sum_i (pos_sim_i - max_neg_sim_i)
```

含义：正样本相似度相对最大负样本相似度的平均边界距。值越大，说明正样本整体更容易从负样本中区分；值接近 0 或为负，说明最强负样本与正样本混淆严重。

### 其他记录字段

采样偏差 CSV 还记录：

- `p10_margin`：`margin_i` 的 10% 分位数，用于观察较困难节点的边界距。
- `mean_pos_sim`：`pos_sim_i` 的平均值。
- `mean_max_neg_sim`：`max_neg_sim_i` 的平均值。

## 性能对比实验

相关代码：`experiments/method_comparison/run_full_pipeline.py`、`experiments/method_comparison/plot_method_comparison_results.py`

本实验比较 `GRACE`、`GCA`、`SG-GR`、`SG-GC`。核心图展示：

- `robust_score`
- `Micro-F1`，对应 CSV 字段 `F1Mi_mean`
- `Macro-F1`，对应 CSV 字段 `F1Ma_mean`

### run 级指标

每次训练完成后，脚本从 `train.py` 输出中解析：

```text
F1Mi_mean, F1Mi_std, F1Ma_mean, F1Ma_std
```

随后计算：

```text
robust_score = F1Mi_mean - std_weight * F1Mi_std
```

### GRACE 基线与 delta_vs_grace

GRACE 会运行 `baseline_runs` 次。每次 run 都有自己的 `robust_score`。脚本计算基线参考值：

```text
baseline_robust = mean(robust_score of GRACE baseline runs)
```

其他方法的相对变化：

```text
delta_vs_grace = robust_score_method_run - baseline_robust
```

绘图汇总时，会对每个方法的候选配置做重复验证，并按以下优先级选择每个数据集上的代表配置：

```text
1. robust_score_mean 更高
2. F1Mi_mean 更高
3. robust_score_std 更低
```

其中：

```text
robust_score_mean = 同一方法/候选配置多个验证 run 的 robust_score 均值
robust_score_std  = 同一方法/候选配置多个验证 run 的 robust_score 总体标准差
```

## 消融实验

相关代码：`experiments/component_ablation/run_component_ablation.py`、`experiments/component_ablation/plot_component_ablation.py`

本实验只针对 `SG-GR` 与 `SG-GC`，比较完整方法与关闭某个组件后的变化：

- `full`：完整 SG-GCL。
- `no_warmup` / `M-off`：`warmup_epochs=0`。
- `single_mining` / `K-off`：`update_interval=num_epochs+1`，即训练期间不做动态更新。
- `uniform_weight` / `w-off`：`beta=0.0`，即关闭语义权重调节。

### 基础指标

每个 variant 的每个 run 仍然先计算：

```text
F1Mi_mean, F1Mi_std, F1Ma_mean, F1Ma_std
robust_score = F1Mi_mean - std_weight * F1Mi_std
```

summary 行中：

```text
F1Mi_mean = 多个 run 的 F1Mi_mean 均值
F1Mi_std  = 多个 run 的 F1Mi_mean 总体标准差
F1Ma_mean = 多个 run 的 F1Ma_mean 均值
F1Ma_std  = 多个 run 的 F1Ma_mean 总体标准差
robust_score = 多个 run 的 robust_score 均值
robust_score_std = 多个 run 的 robust_score 总体标准差
```

同时保留：

```text
within_run_F1Mi_std_mean = 多个 run 内部 F1Mi_std 的平均值
within_run_F1Ma_std_mean = 多个 run 内部 F1Ma_std 的平均值
```

### 相对完整方法的变化

设完整方法的 summary 指标为：

```text
full_robust = robust_score(full)
variant_robust = robust_score(某个 ablation variant)
```

则：

```text
delta_vs_full = variant_robust - full_robust
drop_vs_full = full_robust - variant_robust
relative_drop_vs_full = drop_vs_full / full_robust
```

解释：

- `delta_vs_full < 0` 表示关闭该模块后稳健性评分下降。
- `drop_vs_full > 0` 表示完整方法优于该消融配置。
- `relative_drop_vs_full` 是相对下降比例，便于跨数据集比较。

## 效率分析实验

相关代码：`experiments/efficiency/run_efficiency_experiment.py`、`experiments/efficiency/plot_efficiency_experiment.py`

效率实验同时记录时间指标和分类质量指标。分类质量部分仍然使用：

```text
F1Mi_mean, F1Mi_std, F1Ma_mean, F1Ma_std
robust_score = F1Mi_mean - std_weight * F1Mi_std
```

### 时间指标

脚本从训练日志中解析每个 epoch 的耗时和累计训练耗时：

```text
train_total_sec = 最后一个 epoch 日志中的 total 时间
epoch_time_mean_sec = epoch_time 的平均值
epoch_time_std_sec = epoch_time 的总体标准差
epoch_time_median_sec = epoch_time 的中位数
throughput_epoch_per_sec = epochs_observed / train_total_sec
```

端到端时间：

```text
wall_time_sec = run_train 调用前后 perf_counter 的差值
```

评估及脚本调度开销：

```text
eval_overhead_sec = wall_time_sec - train_total_sec
```

### 相对耗时

在 summary 行中，脚本还计算：

```text
time_vs_grace_sec = method_time - grace_time
time_ratio_vs_grace = method_time / grace_time
```

其中 `method_time` 优先使用 `train_total_sec`，若缺失则使用 `wall_time_sec`。

对于有直接基线的方法，还计算：

```text
overhead_vs_base_sec = method_time - base_time
overhead_ratio_vs_base = overhead_vs_base_sec / base_time
```

其中：

- `GCA` 和 `SG-GR` 的 base method 为 `GRACE`。
- `SG-GC` 的 base method 为 `GCA`。

## 统计显著性实验

相关代码：`experiments/statistical_significance/run_significance_experiment.py`、`experiments/statistical_significance/analyze_significance_results.py`、`experiments/statistical_significance/plot_significance_results.py`

统计显著性实验关注方法提升是否稳定。它对每个方法用相同的 `run_idx`、`seed` 和 `eval_seed` 生成配对结果，然后比较：

- `SG-GR` vs `GRACE`
- `SG-GC` vs `GCA`

分析脚本也会计算补充比较：

- `SG-GC` vs `GRACE`
- `GCA` vs `GRACE`

### 被检验指标

显著性分析中进入统计检验的指标为：

```text
robust_score
F1Mi_mean
F1Ma_mean
```

### 配对差值

对任一指标 `metric`，在共享 run 上计算：

```text
delta_r = metric_target,r - metric_baseline,r
mean_delta = mean(delta_r)
median_delta = median(delta_r)
```

结果图中的“相对基线稳健性评分robust_score变化”就是针对 `robust_score` 的 `mean_delta`。

### 置信区间和显著性

脚本计算：

```text
ci95_low, ci95_high = 对 delta_r 均值做 bootstrap 95% 置信区间
p_value = Wilcoxon signed-rank test 的 p 值
p_value_paired_ttest = paired t-test 的 p 值
effect_size = rank_biserial_effect(delta_r)
cohen_dz = mean(delta_r) / sample_std(delta_r)
```

多重比较校正使用 Holm-Bonferroni：

```text
p_value_holm = Holm 校正后的 p 值
```

显著优于基线的判断规则：

```text
significant = (p_value_holm < alpha) and (mean_delta > 0)
```

默认 `alpha=0.05`。

## 超参数影响分析实验

相关代码：`experiments/hyperparameter_sensitivity/run_ifl_param_sensitivity.py`、`experiments/hyperparameter_sensitivity/plot_ifl_param_sensitivity.py`

本实验只针对 `SG-GR` 与 `SG-GC`，分析三个论文参数的影响：

```text
t_s -> similarity_threshold
M   -> warmup_epochs
K   -> update_interval
```

### 每个观测点的指标

对每个参数取值 `sweep_value`，运行若干次训练并记录：

```text
F1Mi_mean, F1Mi_std, F1Ma_mean, F1Ma_std
robust_score = F1Mi_mean - std_weight * F1Mi_std
```

summary 行中：

```text
F1Mi_mean = 多个 run 的 F1Mi_mean 均值
F1Mi_std  = 多个 run 的 F1Mi_mean 总体标准差
F1Ma_mean = 多个 run 的 F1Ma_mean 均值
F1Ma_std  = 多个 run 的 F1Ma_mean 总体标准差
robust_score = 多个 run 的 robust_score 均值
robust_score_std = 多个 run 的 robust_score 总体标准差
```

### 观测点和相对变化

`anchor_value` 是从最佳网格搜索配置中读取的参数值；若 `t_s` 没有显式记录，则脚本通过训练 trace 中的 `trace_ts_mean` 估计锚点阈值。

图例中的“观测点”对应：

```text
is_anchor = True
```

相对观测点变化：

```text
delta_vs_anchor = robust_score(sweep_value) - robust_score(anchor_value)
```

### trace 指标

超参数实验还记录语义正样本挖掘过程中的 trace 指标：

- `trace_ts_mean`：训练日志中活跃阈值 `ts` 的平均值。
- `trace_ts_last`：最后一次记录的活跃阈值。
- `trace_mined_pairs_mean`：每次记录中挖掘到的正样本对数平均值。
- `trace_mined_pairs_last`：最后一次记录的正样本对数。
- `trace_avg_pairs_mean`：每个节点平均正样本对数的平均值。
- `trace_avg_pairs_last`：最后一次记录的每节点平均正样本对数。

## 指标到实验的对应关系

| 实验 | 主要图中指标 | 主要 CSV 字段 | 说明 |
| --- | --- | --- | --- |
| 采样偏差实验 | `violation_rate`, `mean_margin` | `violation_rate`, `mean_margin`, `p10_margin`, `mean_pos_sim`, `mean_max_neg_sim` | 每个 epoch 的正负样本相似度关系 |
| 性能对比实验 | `robust_score`, `Micro-F1`, `Macro-F1` | `robust_score`, `F1Mi_mean`, `F1Ma_mean`, `delta_vs_grace` | 比较 GRACE/GCA/SG-GR/SG-GC |
| 消融实验 | `robust_score` | `robust_score`, `robust_score_std`, `delta_vs_full`, `drop_vs_full`, `relative_drop_vs_full` | 比较 Full 与 M-off/K-off/w-off |
| 效率分析实验 | `train_total_time`, `wall_time` | `train_total_sec`, `wall_time_sec`, `robust_score`, `F1Mi_mean`, `F1Ma_mean` | 同时记录效率与分类质量 |
| 统计显著性实验 | `robust_score` 相对基线变化 | `mean_delta`, `p_value_holm`, `significant`, `robust_score` | 对配对 run 做 Wilcoxon 和 Holm 校正 |
| 超参数影响分析 | `robust_score` | `robust_score`, `robust_score_std`, `delta_vs_anchor`, `is_anchor` | 比较 `t_s`、`M`、`K` 不同取值 |

