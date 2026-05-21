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
