# GRACE

<img src="grace.png" alt="model" style="zoom: 50%;" />

This is the code for the Paper: [deep **GRA**ph **C**ontrastive r**E**presentation learning (GRACE)](https://arxiv.org/pdf/2006.04131v2.pdf).

For a thorough resource collection of self-supervised learning methods on graphs, you may refer to [this](https://github.com/SXKDZ/awesome-self-supervised-learning-for-graphs) awesome list.

## Usage

Train and evaluate the model by executing
```
python train.py --dataset Cora
```
The `--dataset` argument should be one of [ Cora, CiteSeer, PubMed, DBLP ].

Method selection is supported via `--method`:
- `grace`: original GRACE training
- `ifl-gr`: corrected IFL-GR training
- `gca`: GCA-style structure-aware augmentation training
- `ifl-gc`: IFL-GC (GCA sampling + semantically guided corrected InfoNCE)

Code navigation:
- `CODE_STRUCTURE.md`: file-by-file responsibilities, method training/testing flow, and hyper-parameter search flow.
- `tools/GRID_SEARCH_GUIDE.md`: practical commands for grid search and top-parameter verification.
- `TWOSTAGE_GUIDE.md`: **new** — two-stage parameter search strategy (Stage 1: method parameters, Stage 2: learning_rate optimization; reduces total trials while maintaining robustness).

Examples:
```
python train.py --dataset Cora --method grace
python train.py --dataset Cora --method ifl-gr
python train.py --dataset Cora --method gca
python train.py --dataset Cora --method ifl-gc
```

### Automated Grid Search & Parameter Tuning

**Quick start — two-stage search** (recommended for comprehensive experiments):
```bash
cd tools
python run_cora_full_pipeline_2stage.py --gpu_id 0 --topk_verify 3 --runs_per_top 3
# Output: ../results/cora_full_pipeline_2stage_results.csv
```
This runs:
1. GRACE baseline (3 runs)
2. IFL-GR two-stage search (Stage 1 grid → top-3 Stage 2 LR scan, each verified 3x)
3. IFL-GC two-stage search (Stage 1 grid → top-3 Stage 2 LR scan, each verified 3x)
4. Summary statistics and method comparison

For full documentation, see `TWOSTAGE_GUIDE.md`.

**One-stage search** (alternative, original method):
```bash
cd tools
python run_cora_full_pipeline.py --gpu_id 0 --topk_verify 3 --runs_per_top 3
# Output: ../results/cora_full_pipeline_results.csv
```
Covers all four methods (GRACE, IFL-GR, GCA, IFL-GC) with single-stage grid search.

See `tools/GRID_SEARCH_GUIDE.md` for more detailed usage and per-method scripts.

Dataset location behavior:
- By default, datasets are cached under `GRACE/datasets/`.
- If files already exist there, training reads them directly.
- If files are missing, PyG will download/process and store them under `GRACE/datasets/`.
- For Cora/CiteSeer/PubMed, the typical layout is `GRACE/datasets/<DatasetName>/{raw,processed}`.
- You can override with `--dataset_root <path>`.

## Requirements

- torch 1.4.0
- torch-geometric 1.5.0
- sklearn 0.21.3
- numpy 1.18.1
- pyyaml 5.3.1

Install all dependencies using
```
pip install -r requirements.txt
```

If you encounter some problems during installing `torch-geometric`, please refer to the installation manual on its [official website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).


## Citation

Please cite our paper if you use the code:

```
@inproceedings{Zhu:2020vf,
  author = {Zhu, Yanqiao and Xu, Yichen and Yu, Feng and Liu, Qiang and Wu, Shu and Wang, Liang},
  title = {{Deep Graph Contrastive Representation Learning}},
  booktitle = {ICML Workshop on Graph Representation Learning and Beyond},
  year = {2020},
  url = {http://arxiv.org/abs/2006.04131}
}
```
