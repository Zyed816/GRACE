# `run_selected_full_pipelines.py` 参数搜索空间与 Trial 统计

本文档记录当前项目中，运行 `tools/run_selected_full_pipelines.py` 时，各数据集在各方法下的参数搜索空间，以及对应的 trial 数。

## 1. 运行入口与实际生效脚本

- 总入口：`tools/run_selected_full_pipelines.py`
- 各数据集完整 pipeline 入口：
  - `tools/run_cora_full_pipeline.py`
  - `tools/run_citeseer_full_pipeline.py`
  - `tools/run_pubmed_full_pipeline.py`
  - `tools/run_dblp_full_pipeline.py`
- 实际定义搜索空间的脚本只有 3 份：
  - `tools/grid_search_iflgr_cora.py`
  - `tools/grid_search_gca_cora.py`
  - `tools/grid_search_iflgc_cora.py`

说明：

- `CiteSeer / PubMed / DBLP` 的 `grid_search_*_{dataset}.py` 都只是转发到上述 `*_cora.py`，并通过 `--dataset` 切换到对应分支。
- 因此，真正生效的搜索空间都写在这 3 个 `*_cora.py` 文件的 `if dataset_key == ...` 分支中。

## 2. 当前仓库状态下，直接运行时会不会重新搜索

当前 `results/` 目录下已经存在以下 12 个候选结果文件：

- `grid_search_iflgr_cora_results.csv`
- `grid_search_gca_cora_results.csv`
- `grid_search_iflgc_cora_results.csv`
- `grid_search_iflgr_citeseer_results.csv`
- `grid_search_gca_citeseer_results.csv`
- `grid_search_iflgc_citeseer_results.csv`
- `grid_search_iflgr_pubmed_results.csv`
- `grid_search_gca_pubmed_results.csv`
- `grid_search_iflgc_pubmed_results.csv`
- `grid_search_iflgr_dblp_results.csv`
- `grid_search_gca_dblp_results.csv`
- `grid_search_iflgc_dblp_results.csv`

因此，在默认情况下：

- 如果直接运行 `python tools/run_selected_full_pipelines.py --datasets Cora CiteSeer PubMed DBLP --gpu_id 0`
- 且不加 `--force_grid`
- 则完整 pipeline 会跳过 grid search，直接读取已有 CSV 的 Top-K 参数做复验

默认参数为：

- `baseline_runs = 3`
- `topk_verify = 3`
- `runs_per_top = 3`

于是，当前仓库状态下每个数据集的实际训练次数为：

- `grace` baseline：3 次
- `ifl-gr` Top-3 复验：`3 x 3 = 9` 次
- `gca` Top-3 复验：`3 x 3 = 9` 次
- `ifl-gc` Top-3 复验：`3 x 3 = 9` 次
- 合计：`30` 次训练 / 数据集

如果 4 个数据集都跑，则合计：

- `30 x 4 = 120` 次训练
- 此时 grid search 的 trial 数为 `0`

## 3. Trial 计算公式

- `IFL-GR`：`total_trials = len(values_product)`
- `GCA`：`total_trials = len(values_product) * len(feature_profiles)`
- `IFL-GC`：`total_trials = len(values_product) * len(edge_profiles) * len(feature_profiles)`

## 4. 各数据集搜索空间与 trial 数

### 4.1 Cora

#### `grace`

- 无参数搜索空间
- search trial 数：`0`
- baseline 默认训练次数：`3`

#### `ifl-gr`

搜索空间：

- `similarity_percentile = [99.5, 99.7]`
- `max_du_per_node = [12, 14, 16]`
- `unlabeled_weight = [0.2, 0.3]`
- `warmup_epochs = [80, 100]`
- `tau = [0.3, 0.4]`

固定参数：

- `update_interval = 3`
- `similarity_threshold = None`
- `use_mutual_topk = True`
- `beta = 2.2`
- `corrected_ramp_epochs = 20`

trial 数：

- `2 x 3 x 2 x 2 x 2 = 48`

#### `gca`

搜索空间：

- `gca_drop_scheme = ['uniform']`
- `drop_edge_rate_1 = [0.5, 0.6, 0.7]`
- `drop_edge_rate_2 = [0.6, 0.7]`
- `tau = [0.8, 1.0]`

`feature_profiles`：

- `(drop_feature_rate_1=0.4, drop_feature_rate_2=0.5)`
- `(drop_feature_rate_1=0.5, drop_feature_rate_2=0.6)`

固定参数：

- `gca_pr_k = 200`

trial 数：

- `(1 x 3 x 2 x 2) x 2 = 24`

#### `ifl-gc`

搜索空间：

- `gca_drop_scheme = ['degree', 'pr']`
- `similarity_percentile = [99.5, 99.7]`
- `max_du_per_node = [12, 14]`
- `unlabeled_weight = [0.2, 0.3]`
- `warmup_epochs = [80]`
- `iflgc_refl_du_weight = [0.4, 0.5, 0.6]`
- `tau = [0.3, 0.4]`

`edge_profiles`：

- `(drop_edge_rate_1=0.3, drop_edge_rate_2=0.5)`

`feature_profiles`：

- `(drop_feature_rate_1=0.3, drop_feature_rate_2=0.4)`

固定参数：

- `similarity_threshold = None`
- `update_interval = 3`
- `use_mutual_topk = True`
- `beta = 2.2`
- `corrected_ramp_epochs = 20`
- `gca_pr_k = 200`

trial 数：

- `(2 x 2 x 2 x 2 x 1 x 3 x 2) x 1 x 1 = 96`

#### Cora 小结

- `ifl-gr = 48`
- `gca = 24`
- `ifl-gc = 96`
- 搜索 trial 合计：`168`
- fresh full pipeline 总训练次数：`168 + 30 = 198`

### 4.2 CiteSeer

#### `grace`

- 无参数搜索空间
- search trial 数：`0`
- baseline 默认训练次数：`3`

#### `ifl-gr`

搜索空间：

- `similarity_percentile = [99.3, 99.5]`
- `max_du_per_node = [8, 10, 12]`
- `unlabeled_weight = [0.3, 0.4, 0.5]`
- `warmup_epochs = [60, 80]`
- `tau = [0.7, 0.9]`

固定参数：

- `update_interval = 3`
- `similarity_threshold = None`
- `use_mutual_topk = True`
- `beta = 2.5`
- `corrected_ramp_epochs = 30`

trial 数：

- `2 x 3 x 3 x 2 x 2 = 72`

#### `gca`

搜索空间：

- `gca_drop_scheme = ['uniform']`
- `drop_edge_rate_1 = [0.6, 0.7]`
- `drop_edge_rate_2 = [0.5, 0.6]`
- `tau = [1.0]`

`feature_profiles`：

- `(drop_feature_rate_1=0.4, drop_feature_rate_2=0.5)`
- `(drop_feature_rate_1=0.5, drop_feature_rate_2=0.6)`

固定参数：

- `gca_pr_k = 200`

trial 数：

- `(1 x 2 x 2 x 1) x 2 = 8`

#### `ifl-gc`

搜索空间：

- `gca_drop_scheme = ['degree', 'pr']`
- `similarity_percentile = [99.3, 99.5]`
- `max_du_per_node = [8, 10]`
- `unlabeled_weight = [0.3, 0.4]`
- `warmup_epochs = [80]`
- `iflgc_refl_du_weight = [0.5, 0.6]`
- `tau = [0.7, 0.9]`

`edge_profiles`：

- `(drop_edge_rate_1=0.2, drop_edge_rate_2=0.0)`

`feature_profiles`：

- `(drop_feature_rate_1=0.3, drop_feature_rate_2=0.2)`

固定参数：

- `similarity_threshold = None`
- `update_interval = 3`
- `use_mutual_topk = True`
- `beta = 2.5`
- `corrected_ramp_epochs = 30`
- `gca_pr_k = 200`

trial 数：

- `(2 x 2 x 2 x 2 x 1 x 2 x 2) x 1 x 1 = 64`

#### CiteSeer 小结

- `ifl-gr = 72`
- `gca = 8`
- `ifl-gc = 64`
- 搜索 trial 合计：`144`
- fresh full pipeline 总训练次数：`144 + 30 = 174`

### 4.3 PubMed

#### `grace`

- 无参数搜索空间
- search trial 数：`0`
- baseline 默认训练次数：`3`

#### `ifl-gr`

搜索空间：

- `similarity_percentile = [99.5, 99.7]`
- `max_du_per_node = [12, 14]`
- `unlabeled_weight = [0.2, 0.3]`
- `warmup_epochs = [100]`
- `tau = [0.3]`

固定参数：

- `update_interval = 3`
- `similarity_threshold = None`
- `use_mutual_topk = True`
- `beta = 2.2`
- `corrected_ramp_epochs = 20`

trial 数：

- `2 x 2 x 2 x 1 x 1 = 8`

#### `gca`

搜索空间：

- `gca_drop_scheme = ['uniform']`
- `drop_edge_rate_1 = [0.5, 0.6, 0.7]`
- `drop_edge_rate_2 = [0.6, 0.7]`
- `tau = [0.8]`

`feature_profiles`：

- `(drop_feature_rate_1=0.5, drop_feature_rate_2=0.6)`

固定参数：

- `gca_pr_k = 200`

trial 数：

- `(1 x 3 x 2 x 1) x 1 = 6`

#### `ifl-gc`

搜索空间：

- `gca_drop_scheme = ['degree']`
- `similarity_percentile = [99.5, 99.7]`
- `max_du_per_node = [12, 14]`
- `unlabeled_weight = [0.2, 0.3]`
- `warmup_epochs = [100]`
- `iflgc_refl_du_weight = [0.4, 0.5]`
- `tau = [0.3]`

`edge_profiles`：

- `(drop_edge_rate_1=0.3, drop_edge_rate_2=0.5)`

`feature_profiles`：

- `(drop_feature_rate_1=0.3, drop_feature_rate_2=0.4)`

固定参数：

- `similarity_threshold = None`
- `update_interval = 3`
- `use_mutual_topk = True`
- `beta = 2.2`
- `corrected_ramp_epochs = 20`
- `gca_pr_k = 200`

trial 数：

- `(1 x 2 x 2 x 2 x 1 x 2 x 1) x 1 x 1 = 16`

#### PubMed 小结

- `ifl-gr = 8`
- `gca = 6`
- `ifl-gc = 16`
- 搜索 trial 合计：`30`
- fresh full pipeline 总训练次数：`30 + 30 = 60`

### 4.4 DBLP

#### `grace`

- 无参数搜索空间
- search trial 数：`0`
- baseline 默认训练次数：`3`

#### `ifl-gr`

搜索空间：

- `similarity_percentile = [99.5, 99.7]`
- `max_du_per_node = [12, 14]`
- `unlabeled_weight = [0.2, 0.3]`
- `warmup_epochs = [100]`
- `tau = [0.7, 0.8]`

固定参数：

- `update_interval = 3`
- `similarity_threshold = None`
- `use_mutual_topk = True`
- `beta = 2.2`
- `corrected_ramp_epochs = 20`

trial 数：

- `2 x 2 x 2 x 1 x 2 = 16`

#### `gca`

搜索空间：

- `gca_drop_scheme = ['uniform']`
- `drop_edge_rate_1 = [0.6, 0.7]`
- `drop_edge_rate_2 = [0.7, 0.8]`
- `tau = [0.3]`

`feature_profiles`：

- `(drop_feature_rate_1=0.5, drop_feature_rate_2=0.6)`

固定参数：

- `gca_pr_k = 200`

trial 数：

- `(1 x 2 x 2 x 1) x 1 = 4`

#### `ifl-gc`

搜索空间：

- `gca_drop_scheme = ['degree']`
- `similarity_percentile = [99.5]`
- `max_du_per_node = [12]`
- `unlabeled_weight = [0.2, 0.3]`
- `warmup_epochs = [100]`
- `iflgc_refl_du_weight = [0.4, 0.5]`
- `tau = [0.7, 0.8]`

`edge_profiles`：

- `(drop_edge_rate_1=0.3, drop_edge_rate_2=0.5)`

`feature_profiles`：

- `(drop_feature_rate_1=0.3, drop_feature_rate_2=0.4)`

固定参数：

- `similarity_threshold = None`
- `update_interval = 3`
- `use_mutual_topk = True`
- `beta = 2.2`
- `corrected_ramp_epochs = 40`
- `gca_pr_k = 200`

trial 数：

- `(1 x 1 x 2 x 1 x 2 x 2) x 1 x 1 = 8`

说明：

- 上式中省略了 `gca_drop_scheme=['degree']` 与 `warmup_epochs=[100]` 这两个大小为 1 的维度。

#### DBLP 小结

- `ifl-gr = 16`
- `gca = 4`
- `ifl-gc = 8`
- 搜索 trial 合计：`28`
- fresh full pipeline 总训练次数：`28 + 30 = 58`

## 5. 四个数据集汇总

如果强制重新搜索，即运行时加 `--force_grid`，则四个数据集合计：

- Cora：`168` search trials
- CiteSeer：`144` search trials
- PubMed：`30` search trials
- DBLP：`28` search trials
- 搜索 trial 总数：`370`

再加上默认完整 pipeline 的固定训练部分：

- 每个数据集 `30` 次
- 四个数据集共 `120` 次

则 fresh full pipeline 总训练次数为：

- `370 + 120 = 490`

如果不加 `--force_grid`，且当前已有 CSV 可复用，则：

- 搜索 trial 总数：`0`
- 总训练次数：`120`

## 6. 对应结果文件中的实际 trial 数

当前 `results/` 目录中的 CSV 行数也与上述 trial 统计一致：

- `grid_search_iflgr_cora_results.csv`：`48`
- `grid_search_gca_cora_results.csv`：`24`
- `grid_search_iflgc_cora_results.csv`：`96`
- `grid_search_iflgr_citeseer_results.csv`：`72`
- `grid_search_gca_citeseer_results.csv`：`8`
- `grid_search_iflgc_citeseer_results.csv`：`64`
- `grid_search_iflgr_pubmed_results.csv`：`8`
- `grid_search_gca_pubmed_results.csv`：`6`
- `grid_search_iflgc_pubmed_results.csv`：`16`
- `grid_search_iflgr_dblp_results.csv`：`16`
- `grid_search_gca_dblp_results.csv`：`4`
- `grid_search_iflgc_dblp_results.csv`：`8`

## 7. 备注

- 本文档基于当前仓库代码与当前 `results/` 目录状态整理。
- 如果后续修改了 `grid_search_iflgr_cora.py`、`grid_search_gca_cora.py`、`grid_search_iflgc_cora.py` 中的搜索空间分支，本文档需要同步更新。
- 如果删除了已有 `results/grid_search_*_results.csv`，或运行时显式加上 `--force_grid`，则完整 pipeline 会重新执行上述搜索。
