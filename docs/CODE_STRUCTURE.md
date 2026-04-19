# 代码结构说明

## 1. 顶层目录

```text
GRACE/
  train.py
  model.py
  eval.py
  config.yaml
  experiments/
  docs/
  legacy/
  results/
  logs/
  datasets/
```

### 顶层各文件职责

- `train.py`
  统一训练入口，负责加载数据、构建模型、训练并输出评估结果。
- `model.py`
  编码器、投影头、对比损失以及 IFL 相关逻辑的核心实现。
- `eval.py`
  节点分类线性评估逻辑，输出 `F1Mi` 和 `F1Ma`。
- `config.yaml`
  各数据集的默认超参数配置，以及大图数据集的分块/子图设置。

## 2. experiments 目录

当前仓库把实验脚本划分为三个正式模块。

### 2.1 `experiments/method_comparison/`

用途：

- 比较 `grace`、`gca`、`ifl-gr`、`ifl-gc`
- 进行网格搜索
- 对 Top-K 参数做复验
- 自动生成单数据集或多数据集完整对比流程

关键脚本：

- `grid_search_iflgr.py`
- `grid_search_gca.py`
- `grid_search_iflgc.py`
- `verify_top_params.py`
- `run_full_pipeline.py`
- `run_full_pipeline_batch.py`

### 2.2 `experiments/hyperparameter_sensitivity/`

用途：

- 对 `ifl-gr`、`ifl-gc` 做单因素超参数敏感性分析
- 输出 CSV、图片和简短文字报告

关键脚本：

- `run_ifl_param_sensitivity.py`
- `plot_ifl_param_sensitivity.py`

### 2.3 `experiments/sampling_bias_validation/`

用途：

- 将训练阶段输出的采样偏差日志画成曲线

关键脚本：

- `plot_sampling_bias_curves.py`

## 3. outputs 目录

为保证实验结果路径不受重构影响，以下目录仍保留原命名。

- `results/`
  保存网格搜索结果、完整流程结果、敏感性分析结果及其图表。
- `logs/`
  保存采样偏差验证的 CSV 日志和曲线图。
- `datasets/`
  保存 PyG 处理后的数据集缓存。

## 4. legacy 目录

`legacy/` 仅保留旧 `tools/` 目录的迁移说明，不再作为主入口。

如果你在旧命令里看到 `tools/...`，请直接切换到 `experiments/...` 下的新入口脚本。

## 5. 入口脚本建议

### 统一训练

```bash
python train.py --dataset Cora --method grace
python train.py --dataset Cora --method ifl-gr
python train.py --dataset Cora --method gca
python train.py --dataset Cora --method ifl-gc
```

### 方法比较

```bash
python experiments/method_comparison/grid_search_iflgr.py --dataset Cora --gpu_id 0 --topk 10
python experiments/method_comparison/verify_top_params.py --dataset Cora --method ifl-gr --top_params results/grid_search_iflgr_cora_results.csv --topk 3 --runs 3 --gpu_id 0
python experiments/method_comparison/run_full_pipeline.py --dataset Cora --gpu_id 0
```

### 超参数敏感性分析

```bash
python experiments/hyperparameter_sensitivity/run_ifl_param_sensitivity.py --datasets Cora --methods ifl-gr ifl-gc --gpu_id 0
python experiments/hyperparameter_sensitivity/plot_ifl_param_sensitivity.py --dataset Cora
```

### 采样偏差验证

```bash
python train.py --dataset Cora --method grace --exp1_metrics --exp1_log_csv logs/exp1_cora.csv
python experiments/sampling_bias_validation/plot_sampling_bias_curves.py --csv logs/exp1_cora.csv --out logs/exp1_cora_curves.png
```

## 6. 推荐阅读顺序

1. `train.py`
2. `model.py`
3. `eval.py`
4. `experiments/method_comparison/run_full_pipeline.py`
5. `experiments/method_comparison/grid_search_*.py`
6. `experiments/hyperparameter_sensitivity/run_ifl_param_sensitivity.py`
7. `experiments/sampling_bias_validation/plot_sampling_bias_curves.py`
