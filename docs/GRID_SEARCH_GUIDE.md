# 方法比较与网格搜索说明

本文档补充说明 `experiments/method_comparison/` 模块中的常用流程。

## 1. 为什么网格搜索结果和单次训练结果可能不完全一致

常见原因有三类：

1. 训练过程本身包含随机性，即使固定种子，不同 trial 的增强顺序和过程状态也可能不同。
2. `eval.py` 中的线性评估本身会有波动，因此 `F1Mi`、`F1Ma` 不是完全静态的数字。
3. 网格搜索阶段通常是“每组参数跑一次并排序”，而复验阶段会对前几组参数做多次重复运行。

因此，建议不要只看网格搜索的 Top-1 单次结果，而是继续做 Top-K 复验。

## 2. 推荐流程

### 第一步：先跑基线

```bash
python train.py --dataset Cora --method grace --gpu_id 0
```

### 第二步：做网格搜索

```bash
python experiments/method_comparison/grid_search_iflgr.py --dataset Cora --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_gca.py --dataset Cora --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_iflgc.py --dataset Cora --gpu_id 0 --topk 10
```

### 第三步：复验前几组参数

```bash
python experiments/method_comparison/verify_top_params.py --dataset Cora --method ifl-gr --top_params results/grid_search_iflgr_cora_results.csv --topk 3 --runs 3 --gpu_id 0
python experiments/method_comparison/verify_top_params.py --dataset Cora --method gca --top_params results/grid_search_gca_cora_results.csv --topk 3 --runs 3 --gpu_id 0
python experiments/method_comparison/verify_top_params.py --dataset Cora --method ifl-gc --top_params results/grid_search_iflgc_cora_results.csv --topk 3 --runs 3 --gpu_id 0
```

### 第四步：直接跑完整流程

如果你不想手动分步骤执行，也可以直接运行：

```bash
python experiments/method_comparison/run_full_pipeline.py --dataset Cora --gpu_id 0
```

这个脚本会自动完成：

1. `grace` 基线重复实验
2. `ifl-gr` 网格搜索与复验
3. `gca` 网格搜索与复验
4. `ifl-gc` 网格搜索与复验
5. 最终汇总输出

## 3. 多数据集批量运行

```bash
python experiments/method_comparison/run_full_pipeline_batch.py --datasets Cora CiteSeer PubMed DBLP --gpu_id 0
```

如需失败后继续运行后续数据集：

```bash
python experiments/method_comparison/run_full_pipeline_batch.py --datasets Cora CiteSeer PubMed DBLP --gpu_id 0 --continue_on_error
```

## 4. 结果文件说明

### 4.1 网格搜索输出

- `results/grid_search_iflgr_<dataset>_results.csv`
- `results/grid_search_gca_<dataset>_results.csv`
- `results/grid_search_iflgc_<dataset>_results.csv`

常见字段：

- `F1Mi_mean`
- `F1Mi_std`
- `F1Ma_mean`
- `F1Ma_std`
- `robust_score`
- `delta_vs_grace`

其中：

```text
robust_score = F1Mi_mean - 0.5 * F1Mi_std
```

### 4.2 完整流程输出

- `results/cora_full_pipeline_results.csv`
- `results/citeseer_full_pipeline_results.csv`
- `results/pubmed_full_pipeline_results.csv`
- `results/dblp_full_pipeline_results.csv`

这些文件里会包含：

- `stage=baseline`
- `stage=top_verify`
- `stage=summary`

## 5. 大图数据集建议

对于 `PubMed` 和 `DBLP`：

- 默认配置中已经启用分块计算参数，优先直接使用当前 `config.yaml`
- 默认配置中启用了子图实验设置，便于快速跑通实验流程
- 建议先在 `Cora` 或 `CiteSeer` 上确认环境和流程，再跑大图

## 6. 与旧命令的关系

历史 `tools/` 目录已经迁出主路径，不再作为正式入口。

当前推荐命令全部使用：

- `experiments/method_comparison/...`
- `experiments/hyperparameter_sensitivity/...`
- `experiments/sampling_bias_validation/...`
