# GRACE

<img src="grace.png" alt="model" style="zoom: 50%;" />

This repository is based on the paper [deep **GRA**ph **C**ontrastive r**E**presentation learning (GRACE)](https://arxiv.org/pdf/2006.04131v2.pdf), and extends it with a unified training/evaluation/search pipeline.

本仓库基于 GRACE 论文实现，并在同一代码框架中扩展了统一训练、评估、自动寻参与对比实验流程。

For a broad collection of graph SSL methods, see [awesome-self-supervised-learning-for-graphs](https://github.com/SXKDZ/awesome-self-supervised-learning-for-graphs).

## What This Project Can Do | 项目能力总览

### Methods | 支持方法

- `grace`: original GRACE
- `gca`: structure-aware Graph Contrastive Augmentation
- `ifl-gr`: IFL-enhanced GRACE (corrected InfoNCE with unlabeled semantic positives)
- `ifl-gc`: IFL + GCA hybrid

### Datasets | 支持数据集

- `Cora`
- `CiteSeer`
- `PubMed`
- `DBLP`

### Experiment Capabilities | 实验能力

- Single-run training and evaluation for all dataset-method combinations
  所有数据集与方法组合都可单独训练评估
- Automatic hyper-parameter search (1-stage grid search)
  自动超参数搜索（一阶段网格搜索）
- Top-K parameter verification with repeated runs
  Top-K 参数多次复验
- Unified comparison pipeline (baseline + search + verification + summary CSV)
  统一对比实验流水线（基线 + 搜索 + 复验 + 汇总）

## Quick Start | 快速开始

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Train one model

```bash
python train.py --dataset Cora --method grace
```

### 3) Run full comparison on one dataset

```bash
python tools/run_cora_full_pipeline.py --dataset Cora --gpu_id 0
```

Default output:
- `results/cora_full_pipeline_results.csv`

默认输出：
- `results/cora_full_pipeline_results.csv`

## Core Usage | 核心用法

### Training Entry | 训练入口

The unified training entry is:

```bash
python train.py --dataset <DATASET> --method <METHOD>
```

Where:
- `<DATASET>` in `{Cora, CiteSeer, PubMed, DBLP}`
- `<METHOD>` in `{grace, gca, ifl-gr, ifl-gc}`

示例：

```bash
python train.py --dataset Cora --method grace
python train.py --dataset Cora --method gca
python train.py --dataset Cora --method ifl-gr
python train.py --dataset Cora --method ifl-gc
```

### Dataset Cache Behavior | 数据集缓存行为

- Default dataset root: `GRACE/datasets/`
- If data exists, it is reused directly
- If data is missing, PyG downloads/processes it under `GRACE/datasets/`
- You can override with `--dataset_root <path>`

默认数据根目录是 `GRACE/datasets/`；若已存在则直接读取，不存在则自动下载并处理。

## Sampling-Bias Validation | 采样偏差验证

可以。当前项目已经支持采样偏差验证，核心是记录每个 epoch 的对比学习偏差指标：

- `violation_rate`
- `mean_margin`
- `p10_margin`

这些指标由 `train.py` 的 `--exp1_log_csv` 输出为 CSV。

### 1) 单次验证（单数据集 + 单方法）

以 Cora + GRACE 为例：

```bash
python train.py --dataset Cora --method grace --gpu_id 0 --exp1_log_csv logs/exp1_cora_grace.csv
```

绘制曲线图：

```bash
python tools/plot_exp1_curves.py --csv logs/exp1_cora_grace.csv --out logs/exp1_cora_grace_curves.png --title "Cora GRACE: Sampling Bias Curves"
```

### 2) 同一数据集比较不同方法的采样偏差

```bash
python train.py --dataset Cora --method grace  --gpu_id 0 --exp1_log_csv logs/exp1_cora_grace.csv
python train.py --dataset Cora --method gca    --gpu_id 0 --exp1_log_csv logs/exp1_cora_gca.csv
python train.py --dataset Cora --method ifl-gr --gpu_id 0 --exp1_log_csv logs/exp1_cora_iflgr.csv
python train.py --dataset Cora --method ifl-gc --gpu_id 0 --exp1_log_csv logs/exp1_cora_iflgc.csv
```

然后分别画图：

```bash
python tools/plot_exp1_curves.py --csv logs/exp1_cora_gca.csv --out logs/exp1_cora_gca_curves.png --title "Cora GCA: Sampling Bias Curves"
python tools/plot_exp1_curves.py --csv logs/exp1_cora_iflgr.csv --out logs/exp1_cora_iflgr_curves.png --title "Cora IFL-GR: Sampling Bias Curves"
python tools/plot_exp1_curves.py --csv logs/exp1_cora_iflgc.csv --out logs/exp1_cora_iflgc_curves.png --title "Cora IFL-GC: Sampling Bias Curves"
```

## Method Comparison Across Datasets | 各数据集上不同方法比较

可以。项目已经提供 baseline + grid search + top-k verify + summary 的统一对比流水线。

### 1) 单数据集完整比较

```bash
python tools/run_cora_full_pipeline.py --dataset Cora --gpu_id 0
python tools/run_citeseer_full_pipeline.py --gpu_id 0
python tools/run_pubmed_full_pipeline.py --gpu_id 0
python tools/run_dblp_full_pipeline.py --gpu_id 0
```

输出示例：

- `results/cora_full_pipeline_results.csv`
- `results/citeseer_full_pipeline_results.csv`
- `results/pubmed_full_pipeline_results.csv`
- `results/dblp_full_pipeline_results.csv`

### 2) 多数据集批量比较（推荐）

```bash
python tools/run_selected_full_pipelines.py --datasets Cora CiteSeer PubMed DBLP --gpu_id 0
```

失败后继续跑后续数据集：

```bash
python tools/run_selected_full_pipelines.py --datasets Cora CiteSeer PubMed DBLP --gpu_id 0 --continue_on_error
```

只跑大图数据集：

```bash
python tools/run_selected_full_pipelines.py --datasets PubMed DBLP --gpu_id 0
```

## Automatic Hyper-Parameter Search (1-Stage) | 自动参数搜索（一阶段）

### Script Matrix | 脚本矩阵

#### IFL-GR

```bash
python tools/grid_search_iflgr_cora.py --dataset Cora --gpu_id 0 --topk 10
python tools/grid_search_iflgr_citeseer.py --gpu_id 0 --topk 10
python tools/grid_search_iflgr_pubmed.py --gpu_id 0 --topk 10
python tools/grid_search_iflgr_dblp.py --gpu_id 0 --topk 10
```

#### GCA

```bash
python tools/grid_search_gca_cora.py --dataset Cora --gpu_id 0 --topk 10
python tools/grid_search_gca_citeseer.py --gpu_id 0 --topk 10
python tools/grid_search_gca_pubmed.py --gpu_id 0 --topk 10
python tools/grid_search_gca_dblp.py --gpu_id 0 --topk 10
```

#### IFL-GC

```bash
python tools/grid_search_iflgc_cora.py --dataset Cora --gpu_id 0 --topk 10
python tools/grid_search_iflgc_citeseer.py --gpu_id 0 --topk 10
python tools/grid_search_iflgc_pubmed.py --gpu_id 0 --topk 10
python tools/grid_search_iflgc_dblp.py --gpu_id 0 --topk 10
```

### Typical Outputs | 典型输出文件

- `results/grid_search_iflgr_<dataset>_results.csv`
- `results/grid_search_gca_<dataset>_results.csv`
- `results/grid_search_iflgc_<dataset>_results.csv`

示例：
- `results/grid_search_iflgr_cora_results.csv`
- `results/grid_search_gca_pubmed_results.csv`

### Score Definition | 评分定义

Grid search ranks candidates by:

`robust_score = F1Mi_mean - std_weight * F1Mi_std`

默认 `std_weight=0.5`。

## Top-K Verification | Top-K 参数复验

After grid search, verify top candidates with repeated runs:

```bash
python tools/verify_top_params.py --dataset Cora --method ifl-gr --top_params results/grid_search_iflgr_cora_results.csv --topk 3 --runs 3 --gpu_id 0
python tools/verify_top_params.py --dataset Cora --method gca --top_params results/grid_search_gca_cora_results.csv --topk 3 --runs 3 --gpu_id 0
python tools/verify_top_params.py --dataset Cora --method ifl-gc --top_params results/grid_search_iflgc_cora_results.csv --topk 3 --runs 3 --gpu_id 0
```

复验目的：降低单次随机波动影响，得到更稳定的最终参数推荐。

## Automated Comparison Pipelines | 自动化对比实验

### Single Dataset Full Pipeline | 单数据集完整流水线

```bash
python tools/run_cora_full_pipeline.py --dataset Cora --gpu_id 0
python tools/run_citeseer_full_pipeline.py --gpu_id 0
python tools/run_pubmed_full_pipeline.py --gpu_id 0
python tools/run_dblp_full_pipeline.py --gpu_id 0
```

The core implementation is in `tools/run_cora_full_pipeline.py`.
Other dataset scripts are wrappers that call the same core logic with different dataset names.

核心逻辑在 `tools/run_cora_full_pipeline.py`，其他三个脚本是按数据集封装的入口。

### Multi-Dataset Batch Dispatch | 多数据集批量调度

```bash
python tools/run_selected_full_pipelines.py --datasets Cora CiteSeer PubMed DBLP --gpu_id 0
```

Continue even if one dataset fails:

```bash
python tools/run_selected_full_pipelines.py --datasets Cora CiteSeer PubMed DBLP --gpu_id 0 --continue_on_error
```

If you want to run only PubMed and DBLP:

```bash
python tools/run_selected_full_pipelines.py --datasets PubMed DBLP --gpu_id 0
```

### Full Pipeline Output | 完整流水线输出

By default:
- `results/cora_full_pipeline_results.csv`
- `results/citeseer_full_pipeline_results.csv`
- `results/pubmed_full_pipeline_results.csv`
- `results/dblp_full_pipeline_results.csv`

Each unified CSV includes:
- baseline rows (`stage=baseline`)
- top candidate verification rows (`stage=top_verify`)
- summary rows (`stage=summary`)

## Result Interpretation | 结果解读

Common columns in result CSV:

- `F1Mi_mean`, `F1Mi_std`
- `F1Ma_mean`, `F1Ma_std`
- `robust_score`
- `delta_vs_grace`
- `params_json`

Interpretation tips:
- Prefer settings with high `robust_score`
- Check `delta_vs_grace > 0` for gains over GRACE baseline
- Use verification means rather than a single grid trial when choosing final parameters

建议优先参考复验后的均值结果，而不是单次网格 trial。

## Recommended Validation Path | 推荐验证路径

To verify that your environment can run the full chain:

1. Run one baseline training:

```bash
python train.py --dataset Cora --method grace
```

2. Run one grid search:

```bash
python tools/grid_search_iflgr_cora.py --dataset Cora --gpu_id 0 --topk 3
```

3. Run one full pipeline:

```bash
python tools/run_cora_full_pipeline.py --dataset Cora --gpu_id 0 --topk_verify 3 --runs_per_top 3
```

Success signals:
- terminal prints final evaluation line with `F1Mi=...+-...`
- corresponding CSV files are created under `results/`
 
## Notes for PubMed / DBLP | PubMed / DBLP 注意事项

- PubMed and DBLP are larger; runs are significantly longer than Cora/CiteSeer
- Chunked computation is used to reduce OOM risk
- You can adjust chunk sizes in `config.yaml`:
  - `contrastive_batch_size`
  - `corrected_batch_size`
  - `mining_batch_size`
- PubMed/DBLP now use the same live terminal output style as Cora/CiteSeer, and still keep 30-second heartbeat messages during long baseline or grid-search steps

PubMed/DBLP 训练耗时更长，建议优先在 Cora/CiteSeer 验证流程后再跑全量实验。

## File Guide | 文档导航

- `CODE_STRUCTURE.md`: architecture and call flow details
- `tools/GRID_SEARCH_GUIDE.md`: practical search and verification guide

## Django Web Experiment System | Django 实验管理与可视化系统

项目已新增 Django 系统目录：

- `gcl_experiment_system/`

该系统在不修改核心算法实现（`train.py` / `model.py` / `eval.py`）的前提下，提供：

- Dashboard（方法数、数据集数、实验数、最近实验）
- Datasets 页面（Cora/CiteSeer/PubMed/DBLP 统计与简介）
- Models 页面（GRACE/GCA/IFL-GR/IFL-GC 说明）
- Run Experiment（参数配置 + 异步启动）
- Training Monitor（Loss + 采样偏差 + 语义相似度曲线，ECharts 实时轮询）
- Results（数据集-方法对比柱状图 + 历史 CSV 汇总，主指标为 F1Mi/F1Ma）
- Experiment History（实验历史与详情）

### 1) 进入 Django 项目目录

```bash
cd gcl_experiment_system
```

### 2) 初始化数据库

```bash
python manage.py migrate
```

### 3) 启动 Web 服务

```bash
python manage.py runserver 0.0.0.0:8000
```

浏览器访问：

- `http://127.0.0.1:8000/`

### 4) 启动实验后台处理（可选但推荐）

当你在页面上创建实验但不立即运行，或希望统一队列执行时：

```bash
python manage.py process_experiments --poll-interval 5
```

单次处理一个 pending 实验：

```bash
python manage.py process_experiments --once
```

### 5) 同步已有结果 CSV（可选）

如果你已经有 `results/` 目录里的历史 CSV，可以先同步进数据库：

```bash
python manage.py sync_pipeline_results
```

### 6) 命令行实验封装（可选）

提供了一个 Django 外围脚本来调用现有训练入口：

```bash
python scripts/run_experiment.py --dataset Cora --model grace --epochs 200
```

它会临时覆盖 `config.yaml` 对应数据集参数并调用 `train.py`。

## Requirements | 依赖

- torch 1.4.0
- torch-geometric 1.5.0
- sklearn 0.21.3
- numpy 1.18.1
- pyyaml 5.3.1

If `torch-geometric` installation fails, refer to the official docs:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

## Citation

Please cite our paper if you use this code:

```bibtex
@inproceedings{Zhu:2020vf,
  author = {Zhu, Yanqiao and Xu, Yichen and Yu, Feng and Liu, Qiang and Wu, Shu and Wang, Liang},
  title = {{Deep Graph Contrastive Representation Learning}},
  booktitle = {ICML Workshop on Graph Representation Learning and Beyond},
  year = {2020},
  url = {http://arxiv.org/abs/2006.04131}
}
```

## Django Task System v2

The Django UI now supports one unified task queue for:

- `train.py` single training runs
- `grid_search_*` scripts
- `verify_top_params.py`
- `run_*_full_pipeline.py`
- `run_selected_full_pipelines.py`

### Worker model

Web only creates tasks in DB; a single worker process executes pending tasks sequentially:

```bash
cd gcl_experiment_system
python manage.py process_experiments --poll-interval 1
```

### New UI behaviors

- Task form is tab-based (`Train`, `Grid Search`, `Top Verify`, `Full Pipeline Single`, `Full Pipeline Multi`)
- All tasks support JSON-array passthrough CLI args
- Task detail page shows live terminal output (HTTP polling)
- `Stop Task` supports:
  - `pending -> cancelled` immediately
  - `running -> cancel_requested -> cancelled` via cooperative process termination
