# GRACE Code Structure

This repository is now organized around one core layer and three experiment blocks.

## 1. Top-Level Layout

```text
GRACE/
  config.yaml
  train.py
  model.py
  eval.py
  README.md
  requirements.txt
  experiments/
    comparison/
    hyperparameter_analysis/
    sampling_bias/
  docs/
    CODE_STRUCTURE.md
    GRID_SEARCH_GUIDE.md
  tools/
    ... legacy compatibility wrappers ...
  results/
  logs/
  datasets/
```

## 2. Core Layer

- `train.py`
  Unified training entry for `grace`, `gca`, `ifl-gr`, and `ifl-gc`.
- `model.py`
  Encoder, projection head, contrastive losses, and corrected IFL variants.
- `eval.py`
  Linear-evaluation based node classification reporting `F1Mi` and `F1Ma`.
- `config.yaml`
  Dataset-specific default hyper-parameters and large-graph settings.

These files stay at the repository root because they are shared by every experiment type.

## 3. Experiment Blocks

### 3.1 Comparison

Path: `experiments/comparison/`

Purpose:
- Compare `GRACE`, `GCA`, `IFL-GR`, and `IFL-GC`
- Run grid search
- Verify top-ranked settings
- Produce unified full-pipeline summaries

Key scripts:
- `grid_search_iflgr_*.py`
- `grid_search_gca_*.py`
- `grid_search_iflgc_*.py`
- `verify_top_params.py`
- `run_cora_full_pipeline.py`
- `run_citeseer_full_pipeline.py`
- `run_pubmed_full_pipeline.py`
- `run_dblp_full_pipeline.py`
- `run_selected_full_pipelines.py`

### 3.2 Hyper-Parameter Analysis

Path: `experiments/hyperparameter_analysis/`

Purpose:
- Analyze the impact of paper-level hyper-parameters for `IFL-GR` / `IFL-GC`
- Reuse top-ranked search results as anchors
- Generate plots and markdown summaries

Key scripts:
- `run_ifl_param_sensitivity.py`
- `plot_ifl_param_sensitivity.py`

### 3.3 Sampling Bias

Path: `experiments/sampling_bias/`

Purpose:
- Validate sampling-bias behavior during training
- Plot `violation_rate` and `mean_margin` from logged CSV files

Key scripts:
- `plot_exp1_curves.py`

## 4. Outputs and Artifacts

To avoid changing experimental behavior, artifact directories remain unchanged:

- `results/`
  Comparison outputs, grid-search CSVs, sensitivity CSVs, and generated plots.
- `logs/`
  Sampling-bias CSV logs and curve figures.
- `datasets/`
  Project-local dataset cache used by PyG.

## 5. Compatibility Layer

Path: `tools/`

The original `tools/` command paths are still available, but they are now thin wrappers that forward to the new `experiments/` locations.

Examples:

```bash
# Recommended
python experiments/comparison/run_cora_full_pipeline.py --gpu_id 0

# Still supported
python tools/run_cora_full_pipeline.py --gpu_id 0
```

## 6. Recommended Entry Points

### Core training

```bash
python train.py --dataset Cora --method grace
python train.py --dataset Cora --method ifl-gr
python train.py --dataset Cora --method gca
python train.py --dataset Cora --method ifl-gc
```

### Method comparison

```bash
python experiments/comparison/grid_search_iflgr_cora.py --gpu_id 0 --topk 10
python experiments/comparison/verify_top_params.py --dataset Cora --method ifl-gr --top_params results/grid_search_iflgr_cora_results.csv --topk 3 --runs 3 --gpu_id 0
python experiments/comparison/run_cora_full_pipeline.py --gpu_id 0
```

### Hyper-parameter analysis

```bash
python experiments/hyperparameter_analysis/run_ifl_param_sensitivity.py --datasets Cora --methods ifl-gr ifl-gc --gpu_id 0
python experiments/hyperparameter_analysis/plot_ifl_param_sensitivity.py --dataset Cora
```

### Sampling-bias validation

```bash
python train.py --dataset Cora --method grace --exp1_metrics --exp1_log_csv logs/exp1_cora.csv
python experiments/sampling_bias/plot_exp1_curves.py --csv logs/exp1_cora.csv --out logs/exp1_cora_curves.png
```

## 7. Reading Order

1. `train.py`
2. `model.py`
3. `eval.py`
4. `experiments/comparison/run_cora_full_pipeline.py`
5. `experiments/comparison/grid_search_*.py`
6. `experiments/hyperparameter_analysis/run_ifl_param_sensitivity.py`
7. `experiments/sampling_bias/plot_exp1_curves.py`
