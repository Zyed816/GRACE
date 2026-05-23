# SG-GCL 补充实验设计与执行记录

本文档记录论文补充实验，包括组件级消融实验、效率实验和统计显著性实验。三个实验均复用当前项目已有的统一训练入口 `train.py`、已有性能对比实验的最佳参数结果，以及四个论文数据集：

- `Cora`
- `CiteSeer`
- `PubMed`
- `DBLP`

其中 `PubMed` 和 `DBLP` 沿用 `config.yaml` 中当前的子图设置，以保证补充实验与现有性能对比章节处于同一实验设定。

## 1. 组件级消融实验

### 1.1 实验目标

已有性能对比实验已经覆盖 `GRACE`、`GCA`、`SG-GR` 和 `SG-GC` 四种方法，因此组件级消融实验不再重复四方法整体对比，而是验证 SG-GCL 内部机制的有效性。

消融实验关注三个组件：

- `w/o warmup`：去掉预热训练，验证在挖掘隐藏正样本之前是否需要稳定的初始表示。
- `w/o dynamic update`：只挖掘一次隐藏正样本，之后固定不变，验证周期性更新机制是否必要。
- `w/o semantic weight`：保留隐藏正样本挖掘，但将隐藏正样本权重退化为统一权重，验证相似度加权机制是否有效。

实验对象为两个 SG 方法：

- `SG-GR`：代码方法名 `ifl-gr`
- `SG-GC`：代码方法名 `ifl-gc`

### 1.2 实验设计

每个 `(dataset, method)` 组合运行四个版本：

| 版本 | 含义 | 配置修改 |
|---|---|---|
| `full` | 完整 SG-GCL 方法 | 复用性能对比实验中该方法的最佳验证参数 |
| `no_warmup` | 去掉预热阶段 | `warmup_epochs = 0` |
| `single_mining` | 去掉动态更新 | `update_interval = num_epochs + 1` |
| `uniform_weight` | 去掉语义权重 | `beta = 0.0` |

默认重复运行次数为 `runs=3`。同一个 `(dataset, method)` 下的 `full` 和三个消融版本共享同一组 seed：

```text
run_seed = dataset_base_seed + run_idx
```

其中 `dataset_base_seed` 来自 `config.yaml`。

### 1.3 参数来源与公平性

完整方法不重新做网格搜索，而是复用已有性能对比实验中的最佳参数：

1. 优先读取 `results/<dataset>_full_pipeline_results.csv`。
2. 筛选 `stage=top_verify` 且 `method` 为目标方法的记录。
3. 按 `candidate_rank` 分组。
4. 选择平均 `robust_score` 最高的候选参数。
5. 从该候选任意一行的 `params_json` 中恢复完整参数。
6. 如果 full-pipeline 结果不存在，则回退到 `results/grid_search_<method_slug>_<dataset>_results.csv` 的第一行。

每个消融版本只改变目标组件对应的参数，模型结构、数据集、增强策略、基础超参数和评估流程保持不变。

### 1.4 指标与结果位置

主要指标沿用现有体系：

- `F1Mi_mean`
- `F1Mi_std`
- `F1Ma_mean`
- `F1Ma_std`
- `robust_score = F1Mi_mean - std_weight * F1Mi_std`

消融对比指标包括：

- `delta_vs_full = robust_score_variant - robust_score_full`
- `drop_vs_full = robust_score_full - robust_score_variant`
- `relative_drop_vs_full = drop_vs_full / robust_score_full`

训练过程指标从 `train.py` 日志中解析：

- `trace_ts_mean`
- `trace_ts_last`
- `trace_mined_pairs_mean`
- `trace_mined_pairs_last`
- `trace_avg_pairs_mean`
- `trace_avg_pairs_last`

默认结果存放位置：

```text
results/extra_ablation_<dataset>_results.csv
results/plots/extra_ablation_overview.png
results/plots/extra_ablation_overview.pdf
results/plots/extra_ablation_drop_vs_full.png
results/plots/extra_ablation_drop_vs_full.pdf
results/plots/extra_ablation_analysis.md
```

### 1.5 运行命令

轻量验证命令：

```bash
python experiments/component_ablation/run_component_ablation.py --datasets Cora --methods ifl-gr --runs 1 --gpu_id 0
python experiments/component_ablation/plot_component_ablation.py --inputs results/extra_ablation_cora_results.csv
```

完整消融实验命令：

```bash
python experiments/component_ablation/run_component_ablation.py --datasets Cora CiteSeer PubMed DBLP --methods ifl-gr ifl-gc --runs 3 --gpu_id 0
python experiments/component_ablation/plot_component_ablation.py
```

### 1.6 结果解释规则

- 如果移除某个组件后 `robust_score` 在多数数据集和方法上下降，则说明该组件对 SG-GCL 有正向贡献。
- 如果 `no_warmup` 明显下降，说明预热训练有助于避免早期表示不稳定导致的错误隐藏正样本挖掘。
- 如果 `single_mining` 明显下降，说明随着模型表示演化，周期性更新隐藏正样本集合是必要的。
- 如果 `uniform_weight` 明显下降，说明相似度加权有助于区分隐藏正样本可靠性，降低误选样本影响。
- 如果某个消融版本在个别数据集上不下降或略有提升，应作为数据集差异或组件适用性边界讨论，不作为普遍规律。

## 2. 效率实验

### 2.1 实验目标

效率实验用于回答 SG-GCL 的时间开销是否可接受。该实验不重新比较分类性能结论，而是在复用已有性能对比实验最佳参数的前提下，记录不同方法的训练耗时。

默认比较四种方法：

- `GRACE`：代码方法名 `grace`
- `GCA`：代码方法名 `gca`
- `SG-GR`：代码方法名 `ifl-gr`
- `SG-GC`：代码方法名 `ifl-gc`

其中：

- `SG-GR` 主要与 `GRACE` 比较，观察隐藏正样本挖掘与修正损失带来的额外耗时。
- `SG-GC` 主要与 `GCA` 比较，观察在结构感知增强基础上加入 SG-GCL 机制后的额外耗时。
- 所有方法也记录相对 `GRACE` 的 `time_ratio_vs_grace`，用于展示整体时间量级。

### 2.2 参数来源

效率实验不重新搜索参数：

1. `grace` 使用 `results/<dataset>_full_pipeline_results.csv` 中 `stage=baseline, method=grace` 的参数。
2. `gca/ifl-gr/ifl-gc` 使用 `stage=top_verify` 中平均 `robust_score` 最高的 `candidate_rank`。
3. 如果对应 full-pipeline CSV 不存在，则 `gca/ifl-gr/ifl-gc` 回退到对应 `grid_search_*_results.csv` 的第一行。
4. 所有方法共享同一 seed 策略：第 `i` 次运行使用 `dataset_base_seed + i`。

### 2.3 记录指标与结果位置

效率实验输出单次运行行 `stage=run` 和汇总行 `stage=summary`。主要时间指标包括：

- `wall_time_sec`：从启动 `train.py` 到解析最终结果完成的端到端耗时。
- `train_total_sec`：从 `train.py` 最后一个 epoch 日志的 `total` 字段解析出的训练循环耗时。
- `eval_overhead_sec = wall_time_sec - train_total_sec`：主要包含子进程启动、数据准备、最终评估和输出解析等耗时。
- `epoch_time_mean_sec`
- `epoch_time_std_sec`
- `epoch_time_median_sec`
- `throughput_epoch_per_sec`
- `refresh_count`
- `refresh_epoch_time_mean_sec`
- `warmup_epoch_time_mean_sec`
- `corrected_epoch_time_mean_sec`
- `time_vs_grace_sec`
- `time_ratio_vs_grace`
- `overhead_vs_base_sec`
- `overhead_ratio_vs_base`

相对时间指标优先基于 `train_total_sec` 计算，以减少子进程启动、首次 CUDA 初始化和最终评估对算法训练时间比较的干扰；`wall_time_sec` 保留作为端到端运行成本参考。

默认结果存放位置：

```text
results/efficiency_<dataset>_results.csv
results/plots/efficiency_train_total_time.png
results/plots/efficiency_train_total_time.pdf
results/plots/efficiency_wall_time.png
results/plots/efficiency_wall_time.pdf
results/plots/efficiency_time_ratio.png
results/plots/efficiency_time_ratio.pdf
results/plots/efficiency_analysis.md
```

### 2.4 运行命令

轻量验证命令：

```bash
python experiments/efficiency/run_efficiency_experiment.py --datasets Cora --methods grace ifl-gr --runs 1 --gpu_id 0
python experiments/efficiency/plot_efficiency_experiment.py --inputs results/efficiency_cora_results.csv
```

完整效率实验命令：

```bash
python experiments/efficiency/run_efficiency_experiment.py --datasets Cora CiteSeer PubMed DBLP --methods grace gca ifl-gr ifl-gc --runs 3 --gpu_id 0
python experiments/efficiency/plot_efficiency_experiment.py
```

### 2.5 结果解释规则

论文中建议重点报告：

- 每个数据集上四种方法的 `train_total_sec`，并可补充 `wall_time_sec` 作为端到端参考。
- `SG-GR` 相对 `GRACE` 的 `overhead_ratio_vs_base`。
- `SG-GC` 相对 `GCA` 的 `overhead_ratio_vs_base`。
- 若 SG 方法的时间比值在可接受范围内，同时性能章节已经显示其性能提升，则可说明该方法在时间效率上具有可行性。

## 3. 统计显著性实验

### 3.1 实验目标

统计显著性实验用于验证性能对比章节中的主要提升是否具有统计可靠性。现有 `*_full_pipeline_results.csv` 虽然包含 `run_idx`，但方法对比流程原本没有为每次重复显式改变训练 seed，因此不能直接作为主结论级显著性检验依据。

本实验采用配对 seed 设计：固定每个数据集上各方法的最终最佳参数后，对四种方法使用同一组训练 seed 重复运行。统计单元为一次完整训练 seed 的最终评估结果。

### 3.2 实验设计

比较方法：

- `GRACE`：代码方法名 `grace`
- `GCA`：代码方法名 `gca`
- `SG-GR`：代码方法名 `ifl-gr`
- `SG-GC`：代码方法名 `ifl-gc`

默认每个 `(dataset, method)` 运行 `runs=10` 次。第 `i` 次运行的训练 seed 和评估 seed 均设置为：

```text
seed = eval_seed = dataset_base_seed + i
```

其中 `dataset_base_seed` 来自 `config.yaml`。`eval.py` 已支持 `eval_repeats` 和 `eval_seed`，默认仍保持 3 次线性评估；显著性实验中通过临时 YAML 写入 `eval_seed`，使最终分类评估划分可复现。

### 3.3 参数来源

统计显著性实验不重新做网格搜索：

1. `grace` 使用 `results/<dataset>_full_pipeline_results.csv` 中 `stage=baseline, method=grace` 的参数。
2. `gca/ifl-gr/ifl-gc` 使用 `stage=top_verify` 中平均 `robust_score` 最高的 `candidate_rank`。
3. 如果 full-pipeline CSV 不存在，则非 `grace` 方法回退到对应 `results/grid_search_<method_slug>_<dataset>_results.csv` 第一行。
4. 所有方法在同一数据集上使用完全相同的 seed 列表，以支持配对检验。

### 3.4 统计检验方案

主比较：

- `SG-GR` vs `GRACE`：`ifl-gr` 对比 `grace`
- `SG-GC` vs `GCA`：`ifl-gc` 对比 `gca`

补充比较：

- `SG-GC` vs `GRACE`：`ifl-gc` 对比 `grace`
- `GCA` vs `GRACE`：`gca` 对比 `grace`

主指标：

- `robust_score`

辅助指标：

- `F1Mi_mean`
- `F1Ma_mean`

检验与报告内容：

- 主检验使用配对 Wilcoxon signed-rank test。
- 辅助报告 paired t-test 的 p 值。
- 报告平均差值、中位数差值、bootstrap 95% 置信区间、Cohen's dz 和 rank-biserial effect size。
- 对每个指标内的所有比较执行 Holm-Bonferroni 多重比较校正。
- 只有当校正后 `p_value_holm < 0.05` 且 `mean_delta > 0` 时，才写作“显著优于”；否则写作“有提升趋势”或“未达到显著”。

### 3.5 记录指标与结果位置

运行脚本输出单次运行行 `stage=run` 和汇总行 `stage=summary`。分析脚本会将显著性检验行 `stage=test` 追加回同一 CSV。

默认结果存放位置：

```text
results/significance_<dataset>_results.csv
results/plots/significance_tests_summary.csv
results/plots/significance_analysis.md
results/plots/significance_mean_std.png
results/plots/significance_mean_std.pdf
results/plots/significance_paired_delta.png
results/plots/significance_paired_delta.pdf
```

其中：

- `results/significance_<dataset>_results.csv` 保存该数据集的单次运行、方法汇总和配对检验结果。
- `results/plots/significance_tests_summary.csv` 汇总所有数据集的显著性检验结果。
- `results/plots/significance_analysis.md` 是文字分析报告。
- `significance_mean_std.*` 展示四种方法在各数据集上的均值与标准差。
- `significance_paired_delta.*` 展示配对 seed 下的 `robust_score` 差值，并用星号标记达到显著性的比较。

### 3.6 运行命令

如果希望一行命令依次执行完整效率实验和完整统计显著性实验，可运行：

```bash
python experiments/run_efficiency_and_significance.py --gpu_id 0
```

该脚本默认等价于依次执行：

```bash
python experiments/efficiency/run_efficiency_experiment.py --datasets Cora CiteSeer PubMed DBLP --methods grace gca ifl-gr ifl-gc --runs 3 --gpu_id 0
python experiments/statistical_significance/run_significance_experiment.py --datasets Cora CiteSeer PubMed DBLP --methods grace gca ifl-gr ifl-gc --runs 10 --gpu_id 0
```

如果需要调整重复次数，可使用：

```bash
python experiments/run_efficiency_and_significance.py --efficiency_runs 3 --significance_runs 10 --gpu_id 0
```

轻量验证命令：

```bash
python experiments/statistical_significance/run_significance_experiment.py --datasets Cora --methods grace ifl-gr --runs 2 --gpu_id 0
python experiments/statistical_significance/analyze_significance_results.py --inputs results/significance_cora_results.csv
python experiments/statistical_significance/plot_significance_results.py --inputs results/significance_cora_results.csv
```

完整统计显著性实验命令：

```bash
python experiments/statistical_significance/run_significance_experiment.py --datasets Cora CiteSeer PubMed DBLP --methods grace gca ifl-gr ifl-gc --runs 10 --gpu_id 0
python experiments/statistical_significance/analyze_significance_results.py
python experiments/statistical_significance/plot_significance_results.py
```

如果使用当前项目 Conda 环境，也可以显式指定解释器：

```bash
D:\SoftWare\anaconda\envs\GCL\python.exe experiments\statistical_significance\run_significance_experiment.py --datasets Cora CiteSeer PubMed DBLP --methods grace gca ifl-gr ifl-gc --runs 10 --gpu_id 0
D:\SoftWare\anaconda\envs\GCL\python.exe experiments\statistical_significance\analyze_significance_results.py
D:\SoftWare\anaconda\envs\GCL\python.exe experiments\statistical_significance\plot_significance_results.py
```

### 3.7 结果解释规则

- 优先围绕 `robust_score` 解释主结论。
- `F1Mi_mean` 和 `F1Ma_mean` 用作补充证据，说明提升是否同时体现在微平均和宏平均分类性能上。
- 如果 `mean_delta > 0` 但 `p_value_holm >= 0.05`，应写作“性能有提升趋势，但未达到统计显著”。
- 如果 `p_value_holm < 0.05` 但 `mean_delta <= 0`，不能写作目标方法显著优于基线。
- 对 `PubMed` 和 `DBLP` 的结论需说明其基于当前配置中的子图实验设置。
