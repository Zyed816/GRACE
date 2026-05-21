# SG-GCL Component Ablation Experiments

## 1. Experiment Goal

The existing performance-comparison experiment already compares `GRACE`, `GCA`, `SG-GR`, and `SG-GC`. The additional ablation experiment therefore focuses on internal SG-GCL components instead of repeating the four-method comparison.

The component ablation experiment evaluates three variants:

- `w/o warmup`: remove the warmup phase to test whether stable initial representations are necessary before hidden-positive mining.
- `w/o dynamic update`: mine hidden positives only once, then keep the mined set fixed to test the value of periodic updates.
- `w/o semantic weight`: keep hidden-positive mining, but use uniform hidden-positive weights to test the value of similarity-based weighting.

The evaluated SG methods are:

- `SG-GR`, implemented as `ifl-gr`
- `SG-GC`, implemented as `ifl-gc`

## 2. Experiment Design

Datasets follow the current thesis setup:

- `Cora`
- `CiteSeer`
- `PubMed`
- `DBLP`

Each `(dataset, method)` pair runs four variants:

| Variant | Meaning | Config change |
|---|---|---|
| `full` | Complete SG-GCL method | Reuse the best verified parameters from the performance-comparison experiment |
| `no_warmup` | w/o warmup | `warmup_epochs = 0` |
| `single_mining` | w/o dynamic update | `update_interval = num_epochs + 1` |
| `uniform_weight` | w/o semantic weight | `beta = 0.0` |

The default number of repeated runs is `runs=3`, matching the thesis performance and sensitivity experiments. The same run seed list is shared by all variants under the same `(dataset, method)` pair.

## 3. Parameter Source and Fairness

The full method does not rerun grid search. Its base parameters are selected as follows:

1. Prefer `results/<dataset>_full_pipeline_results.csv`.
2. Filter rows with `stage=top_verify` and the target `method`.
3. Group by `candidate_rank`.
4. Select the candidate with the highest mean `robust_score`.
5. Recover the full parameter row from `params_json`.
6. If no full-pipeline candidate is available, fall back to the first row of `results/grid_search_<method_slug>_<dataset>_results.csv`.

Only the target component changes in an ablation variant. Model structure, augmentations, base hyperparameters, datasets, and evaluation remain unchanged.

For run `i`, the implementation sets:

```text
run_seed = dataset_base_seed + i
```

where `dataset_base_seed` comes from `config.yaml`.

## 4. Metrics and Outputs

The experiment records the existing metrics:

- `F1Mi_mean`
- `F1Mi_std`
- `F1Ma_mean`
- `F1Ma_std`
- `robust_score = F1Mi_mean - std_weight * F1Mi_std`

It also records ablation-specific comparisons:

- `delta_vs_full = robust_score_variant - robust_score_full`
- `drop_vs_full = robust_score_full - robust_score_variant`
- `relative_drop_vs_full = drop_vs_full / robust_score_full`

Training-trace metrics are parsed from `train.py` logs:

- `trace_ts_mean`
- `trace_ts_last`
- `trace_mined_pairs_mean`
- `trace_mined_pairs_last`
- `trace_avg_pairs_mean`
- `trace_avg_pairs_last`

Default outputs:

```text
results/extra_ablation_<dataset>_results.csv
results/plots/extra_ablation_overview.png
results/plots/extra_ablation_overview.pdf
results/plots/extra_ablation_drop_vs_full.png
results/plots/extra_ablation_drop_vs_full.pdf
results/plots/extra_ablation_analysis.md
```

## 5. Implementation Entry Points

Run ablation experiments:

```bash
python experiments/component_ablation/run_component_ablation.py --datasets Cora --methods ifl-gr --runs 1 --gpu_id 0
```

Run the full planned experiment:

```bash
python experiments/component_ablation/run_component_ablation.py --datasets Cora CiteSeer PubMed DBLP --methods ifl-gr ifl-gc --runs 3 --gpu_id 0
```

Generate plots and the Markdown report:

```bash
python experiments/component_ablation/plot_component_ablation.py
```

## 6. Interpretation Rules

- If removing a component lowers `robust_score` on most datasets and methods, the component is considered useful.
- If `no_warmup` drops clearly, warmup helps avoid unreliable early hidden-positive mining.
- If `single_mining` drops clearly, periodically updating hidden positives is necessary as representations evolve.
- If `uniform_weight` drops clearly, semantic weighting helps distinguish more reliable hidden positives from noisier ones.
- If an ablation improves on a specific dataset, treat it as dataset-dependent behavior or a boundary case, not as a universal conclusion.

## 7. Efficiency Experiment

效率实验用于回答 SG-GCL 的时间开销是否可接受。该实验不重新比较分类性能结论，而是在复用已有性能对比实验最佳参数的前提下，记录不同方法的训练耗时。

### 7.1 Compared Methods

默认比较四种方法：

- `GRACE`：代码方法名 `grace`
- `GCA`：代码方法名 `gca`
- `SG-GR`：代码方法名 `ifl-gr`
- `SG-GC`：代码方法名 `ifl-gc`

其中 SG 方法的时间开销解释规则为：

- `SG-GR` 主要与 `GRACE` 比较，观察隐藏正样本挖掘与修正损失带来的额外耗时。
- `SG-GC` 主要与 `GCA` 比较，观察在结构感知增强基础上加入 SG-GCL 机制后的额外耗时。
- 所有方法也都会记录相对 `GRACE` 的 `time_ratio_vs_grace`，方便展示整体时间量级。

### 7.2 Parameter Source

效率实验复用已有性能对比实验中的参数，不重新搜索：

1. `grace` 使用 `results/<dataset>_full_pipeline_results.csv` 中 `stage=baseline, method=grace` 的参数。
2. `gca/ifl-gr/ifl-gc` 使用 `stage=top_verify` 中平均 `robust_score` 最高的 `candidate_rank`。
3. 如果对应 full-pipeline CSV 不存在，则 `gca/ifl-gr/ifl-gc` 回退到对应 `grid_search_*_results.csv` 第一行。
4. 所有方法共享同一 seed 策略：第 `i` 次运行使用 `config.yaml` 中的 `dataset_base_seed + i`。

### 7.3 Recorded Metrics

效率实验输出单次运行行 `stage=run` 和汇总行 `stage=summary`。主要时间指标包括：

- `wall_time_sec`：从启动 `train.py` 到解析最终结果完成的端到端耗时。
- `train_total_sec`：从 `train.py` 日志最后一个 epoch 的 `total` 字段解析出的训练循环耗时。
- `eval_overhead_sec`：`wall_time_sec - train_total_sec`，主要包含子进程启动、数据准备、最终评估和输出解析等耗时。
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

其中相对时间指标 `time_vs_grace_sec/time_ratio_vs_grace/overhead_vs_base_sec/overhead_ratio_vs_base` 优先基于 `train_total_sec` 计算，以减少子进程启动、首次 CUDA 初始化和最终评估对算法训练时间比较的干扰；`wall_time_sec` 仍保留作为端到端可运行成本参考。

CSV 中也保留 `F1Mi/F1Ma/robust_score`，用于确认效率实验运行的是有效训练配置。

### 7.4 Implementation Entry Points

效率实验代码位于：

```text
experiments/efficiency/
  __init__.py
  run_efficiency_experiment.py
  plot_efficiency_experiment.py
```

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

### 7.5 Output Files

默认结果位置：

```text
results/efficiency_<dataset>_results.csv
results/plots/efficiency_wall_time.png
results/plots/efficiency_wall_time.pdf
results/plots/efficiency_time_ratio.png
results/plots/efficiency_time_ratio.pdf
results/plots/efficiency_analysis.md
```

论文中建议重点报告：

- 每个数据集上四种方法的 `train_total_sec`，并可补充 `wall_time_sec` 作为端到端参考。
- `SG-GR` 相对 `GRACE` 的 `overhead_ratio_vs_base`。
- `SG-GC` 相对 `GCA` 的 `overhead_ratio_vs_base`。
- 若 SG 方法的时间比值在可接受范围内，同时性能章节已经显示其性能提升，则可说明该方法在时间效率上具有可行性。
