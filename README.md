# GRACE

<img src="grace.png" alt="GRACE framework" style="zoom: 50%;" />

## 项目简介

本项目基于论文 [Deep Graph Contrastive Representation Learning (GRACE)](https://arxiv.org/abs/2006.04131)，在原始 GRACE 训练代码的基础上，扩展出一套面向实验复现的统一框架。当前仓库围绕三类实验任务组织：

1. 采样偏差验证
2. 图对比学习方法比较
3. 超参数敏感性分析

代码层面支持四种方法：

- `grace`：原始 GRACE
- `gca`：Graph Contrastive Augmentation
- `sg-gr`：基于 IFL 的 GRACE 变体
- `sg-gc`：IFL 与 GCA 的混合方法

支持四个数据集：

- `Cora`
- `CiteSeer`
- `PubMed`
- `DBLP`

## 当前目录结构

```text
GRACE/
  train.py
  model.py
  eval.py
  config.yaml
  README.md
  requirements.txt
  experiments/
    method_comparison/
      grid_search_sggr.py
      grid_search_gca.py
      grid_search_sggc.py
      verify_top_params.py
      run_full_pipeline.py
      run_full_pipeline_batch.py
    hyperparameter_sensitivity/
      run_sg_param_sensitivity.py
      plot_sg_param_sensitivity.py
    sampling_bias_validation/
      plot_sampling_bias_curves.py
  docs/
    CODE_STRUCTURE.md
    GRID_SEARCH_GUIDE.md
  legacy/
    README.md
  results/
  logs/
  datasets/
```

目录设计原则如下：

- 根目录只保留所有实验共用的核心训练代码与配置。
- `experiments/` 只保留三大模块对应的正式入口脚本。
- `results/`、`logs/`、`datasets/` 的默认位置保持不变，以免影响已有实验结果与脚本行为。
- 历史 `tools/` 兼容层已经移出主路径，迁移说明放在 `legacy/README.md`。

## 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

数据集缓存说明：

- 默认数据缓存目录为 `datasets/`
- 若本地已有处理后的数据，会直接复用
- 若本地缺失数据，PyG 会自动下载并处理到 `datasets/`
- 如需自定义缓存目录，可使用 `--dataset_root <path>`

大图数据集说明：

- `PubMed` 和 `DBLP` 的默认配置中已经启用了分块计算相关参数，以降低 OOM 风险
- `PubMed` 和 `DBLP` 的默认配置还启用了子图实验开关 `use_subset: true`，这样可以更快完成大图实验流程

## 快速开始

### 1. 运行一次基础训练

```bash
python train.py --dataset Cora --method grace
```

### 2. 在单个数据集上跑完整方法比较流程

```bash
python experiments/method_comparison/run_full_pipeline.py --dataset Cora --gpu_id 0
```

默认输出文件：

- `results/cora_full_pipeline_results.csv`

## 核心训练入口

统一训练入口如下：

```bash
python train.py --dataset <DATASET> --method <METHOD>
```

其中：

- `<DATASET>` 可选：`Cora`、`CiteSeer`、`PubMed`、`DBLP`
- `<METHOD>` 可选：`grace`、`gca`、`sg-gr`、`sg-gc`

常用示例：

```bash
python train.py --dataset Cora --method grace
python train.py --dataset Cora --method gca
python train.py --dataset Cora --method sg-gr
python train.py --dataset Cora --method sg-gc
```

## 实验模块一：采样偏差验证

该模块用于记录训练过程中与采样偏差相关的统计量，并将其画成曲线。当前正式入口为：

- `train.py`
- `experiments/sampling_bias_validation/plot_sampling_bias_curves.py`

### 实验流程

第一步：训练时记录偏差指标到 CSV

```bash
python train.py --dataset Cora --method grace --gpu_id 0 --exp1_metrics --exp1_log_csv logs/exp1_cora.csv
```

如果你想比较其他方法，也可以把 `grace` 换成 `gca`、`sg-gr` 或 `sg-gc`。

第二步：将日志绘制为曲线

```bash
python experiments/sampling_bias_validation/plot_sampling_bias_curves.py --csv logs/exp1_cora.csv --out logs/exp1_cora_curves.png
```

常用参数：

- `--csv`：输入日志文件
- `--out`：输出图片路径
- `--title`：图标题

输出结果：

- 原始日志：`logs/exp1_<dataset>.csv`
- 曲线图片：`logs/exp1_<dataset>_curves.png`

## 实验模块二：图对比学习方法比较

该模块用于比较 `grace`、`gca`、`sg-gr`、`sg-gc` 四种方法的表现，是当前项目最核心的一组实验。正式入口位于 `experiments/method_comparison/`：

- `grid_search_sggr.py`
- `grid_search_gca.py`
- `grid_search_sggc.py`
- `verify_top_params.py`
- `run_full_pipeline.py`
- `run_full_pipeline_batch.py`

### 推荐实验顺序

建议按下面顺序复现实验：

1. 先运行单次基础训练，确认环境正常
2. 再对 `sg-gr`、`gca`、`sg-gc` 做网格搜索
3. 对每种方法的 Top-K 参数做重复实验复验
4. 最后使用完整流程脚本统一生成对比结果

### 2.1 单次训练

如果只想快速看某个方法能否正常运行：

```bash
python train.py --dataset Cora --method sg-gr --gpu_id 0
```

### 2.2 网格搜索

#### SG-GR

```bash
python experiments/method_comparison/grid_search_sggr.py --dataset Cora --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_sggr.py --dataset CiteSeer --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_sggr.py --dataset PubMed --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_sggr.py --dataset DBLP --gpu_id 0 --topk 10
```

#### GCA

```bash
python experiments/method_comparison/grid_search_gca.py --dataset Cora --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_gca.py --dataset CiteSeer --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_gca.py --dataset PubMed --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_gca.py --dataset DBLP --gpu_id 0 --topk 10
```

#### SG-GC

```bash
python experiments/method_comparison/grid_search_sggc.py --dataset Cora --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_sggc.py --dataset CiteSeer --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_sggc.py --dataset PubMed --gpu_id 0 --topk 10
python experiments/method_comparison/grid_search_sggc.py --dataset DBLP --gpu_id 0 --topk 10
```

网格搜索结果默认输出到：

- `results/grid_search_sggr_<dataset>_results.csv`
- `results/grid_search_gca_<dataset>_results.csv`
- `results/grid_search_sggc_<dataset>_results.csv`

排序指标为：

```text
robust_score = F1Mi_mean - std_weight * F1Mi_std
```

默认 `std_weight=0.5`。

### 2.3 Top-K 参数复验

网格搜索结束后，建议对前几组参数做重复实验，以降低单次随机性的影响。

```bash
python experiments/method_comparison/verify_top_params.py --dataset Cora --method sg-gr --top_params results/grid_search_sggr_cora_results.csv --topk 3 --runs 3 --gpu_id 0
python experiments/method_comparison/verify_top_params.py --dataset Cora --method gca --top_params results/grid_search_gca_cora_results.csv --topk 3 --runs 3 --gpu_id 0
python experiments/method_comparison/verify_top_params.py --dataset Cora --method sg-gc --top_params results/grid_search_sggc_cora_results.csv --topk 3 --runs 3 --gpu_id 0
```

关键参数：

- `--top_params`：网格搜索输出的 CSV
- `--topk`：复验前多少组参数
- `--runs`：每组参数重复运行次数
- `--method`：需要与 CSV 对应的方法一致

### 2.4 单数据集完整流程

如果想自动执行“GRACE 基线 + 三种方法搜索 + Top-K 复验 + 汇总输出”，直接运行：

```bash
python experiments/method_comparison/run_full_pipeline.py --dataset Cora --gpu_id 0
python experiments/method_comparison/run_full_pipeline.py --dataset CiteSeer --gpu_id 0
python experiments/method_comparison/run_full_pipeline.py --dataset PubMed --gpu_id 0
python experiments/method_comparison/run_full_pipeline.py --dataset DBLP --gpu_id 0
```

常用可调参数：

- `--baseline_runs`：GRACE 基线重复次数，默认 `3`
- `--topk_verify`：每种方法取前多少组参数复验，默认 `3`
- `--runs_per_top`：每组候选参数重复运行次数，默认 `3`
- `--force_grid`：即使已有历史搜索 CSV，也强制重新搜索
- `--out`：自定义输出 CSV 路径

默认输出：

- `results/cora_full_pipeline_results.csv`
- `results/citeseer_full_pipeline_results.csv`
- `results/pubmed_full_pipeline_results.csv`
- `results/dblp_full_pipeline_results.csv`

每个完整流程 CSV 中会包含：

- `stage=baseline`：GRACE 基线
- `stage=top_verify`：候选参数复验
- `stage=summary`：汇总统计

### 2.5 多数据集批量调度

按顺序批量跑多个数据集：

```bash
python experiments/method_comparison/run_full_pipeline_batch.py --datasets Cora CiteSeer PubMed DBLP --gpu_id 0
```

如果希望某个数据集失败后继续跑后面的数据集：

```bash
python experiments/method_comparison/run_full_pipeline_batch.py --datasets Cora CiteSeer PubMed DBLP --gpu_id 0 --continue_on_error
```

只跑部分数据集：

```bash
python experiments/method_comparison/run_full_pipeline_batch.py --datasets PubMed DBLP --gpu_id 0
```

## 实验模块三：超参数敏感性分析

该模块基于方法比较阶段生成的最优参数 CSV，固定其余参数，只改变一个论文超参数，观察性能变化。正式入口位于 `experiments/hyperparameter_sensitivity/`：

- `run_sg_param_sensitivity.py`
- `plot_sg_param_sensitivity.py`

当前支持的方法：

- `sg-gr`
- `sg-gc`

当前支持的论文超参数映射：

- `t_s -> similarity_threshold`
- `M -> warmup_epochs`
- `K -> update_interval`

### 3.1 运行敏感性实验

先确保已经有对应的网格搜索结果文件，例如：

- `results/grid_search_sggr_cora_results.csv`
- `results/grid_search_sggc_cora_results.csv`

然后运行：

```bash
python experiments/hyperparameter_sensitivity/run_sg_param_sensitivity.py --datasets Cora --methods sg-gr sg-gc --gpu_id 0
```

默认行为：

- 读取 `base_rank=1` 的最优参数组
- 在锚点附近对 `t_s`、`M`、`K` 分别做单因素扰动
- 将原始运行结果和汇总结果写入 `results/`

输出文件默认为：

- `results/sensitivity_sggr_<dataset>_results.csv`
- `results/sensitivity_sggc_<dataset>_results.csv`

自定义 sweep 范围示例：

```bash
python experiments/hyperparameter_sensitivity/run_sg_param_sensitivity.py --datasets Cora --methods sg-gr --ts_values 99.5 99.7 99.9 --m_values 10 12 14 --k_values 80 100 120 --runs 3 --gpu_id 0
```

使用非 Top-1 结果作为锚点示例：

```bash
python experiments/hyperparameter_sensitivity/run_sg_param_sensitivity.py --datasets PubMed --methods sg-gc --base_rank 2 --runs 3 --gpu_id 0
```

### 3.2 绘制敏感性分析图与报告

```bash
python experiments/hyperparameter_sensitivity/plot_sg_param_sensitivity.py --dataset Cora
```

默认输出：

- 图片：`results/plots/cora_sg_sensitivity_overview.png`
- 文字报告：`results/plots/cora_sg_sensitivity_analysis.md`

## 结果文件命名规范

为了保证重构前后的实验结果路径不变，当前仍沿用以下输出命名规则：

- 方法搜索：`results/grid_search_<method_slug>_<dataset_slug>_results.csv`
- 完整流程：`results/<dataset_slug>_full_pipeline_results.csv`
- 敏感性分析：`results/sensitivity_<method_slug>_<dataset_slug>_results.csv`
- 敏感性图表：`results/plots/<dataset_slug>_sg_sensitivity_overview.png`
- 采样偏差日志：`logs/exp1_<dataset_slug>.csv`
- 采样偏差曲线：`logs/exp1_<dataset_slug>_curves.png`

其中：

- `sg-gr -> sggr`
- `sg-gc -> sggc`
- `Cora -> cora`
- `CiteSeer -> citeseer`
- `PubMed -> pubmed`
- `DBLP -> dblp`

## 推荐复现实验顺序

如果你第一次使用本仓库，建议按下列顺序执行：

1. `python train.py --dataset Cora --method grace --gpu_id 0`
2. `python experiments/method_comparison/grid_search_sggr.py --dataset Cora --gpu_id 0 --topk 3`
3. `python experiments/method_comparison/verify_top_params.py --dataset Cora --method sg-gr --top_params results/grid_search_sggr_cora_results.csv --topk 3 --runs 3 --gpu_id 0`
4. `python experiments/method_comparison/run_full_pipeline.py --dataset Cora --gpu_id 0`
5. `python experiments/hyperparameter_sensitivity/run_sg_param_sensitivity.py --datasets Cora --methods sg-gr sg-gc --gpu_id 0`
6. `python train.py --dataset Cora --method grace --gpu_id 0 --exp1_metrics --exp1_log_csv logs/exp1_cora.csv`
7. `python experiments/sampling_bias_validation/plot_sampling_bias_curves.py --csv logs/exp1_cora.csv --out logs/exp1_cora_curves.png`

## 补充文档

- `docs/CODE_STRUCTURE.md`：代码结构说明
- `docs/GRID_SEARCH_GUIDE.md`：方法比较与网格搜索的补充说明
- `experiments/README.md`：三大实验模块简介
- `legacy/README.md`：旧目录迁移说明

## 引用

如果本项目对你的研究有帮助，请引用原始 GRACE 论文：

```bibtex
@inproceedings{Zhu:2020vf,
  author = {Zhu, Yanqiao and Xu, Yichen and Yu, Feng and Liu, Qiang and Wu, Shu and Wang, Liang},
  title = {{Deep Graph Contrastive Representation Learning}},
  booktitle = {ICML Workshop on Graph Representation Learning and Beyond},
  year = {2020},
  url = {http://arxiv.org/abs/2006.04131}
}
```
