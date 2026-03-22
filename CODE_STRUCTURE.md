# GRACE 代码结构与流程速览

本文档帮助你快速理解以下三件事：
- 每个文件做什么
- 四种方法（GRACE / IFL-GR / GCA / IFL-GC）的训练与测试如何串联
- 自动寻参脚本如何运行、输出、复验

## 1. 目录结构（当前）

```text
GRACE/
  config.yaml                         # 全部方法的超参数入口（按数据集分组）
  train.py                            # 统一训练入口：方法分发 + 训练循环 + 最终评估
  model.py                            # 编码器、投影头、对比损失（含 IFL-GR / IFL-GC 修正损失）
  eval.py                             # 线性分类评估（F1Mi / F1Ma）
  README.md                           # 快速使用说明
  requirements.txt                    # 依赖
  results/
    grid_search_iflgr_cora_results.csv
  tools/
    grid_search_iflgr_cora.py         # IFL-GR 网格搜索
    grid_search_gca_cora.py           # GCA 网格搜索
    grid_search_iflgc_cora.py         # IFL-GC 网格搜索
    grid_search_iflgr_citeseer.py     # IFL-GR(CiteSeer) 网格搜索入口
    grid_search_gca_citeseer.py       # GCA(CiteSeer) 网格搜索入口
    grid_search_iflgc_citeseer.py     # IFL-GC(CiteSeer) 网格搜索入口
    verify_top_params.py              # Top-K 参数复验（支持 ifl-gr / gca / ifl-gc）
    run_cora_full_pipeline.py         # 一键自动化：基线+三方法寻参+Top复验+统一结果文件
    run_citeseer_full_pipeline.py     # CiteSeer 一键自动化完整流程
    GRID_SEARCH_GUIDE.md              # 搜索与复验说明
```

## 2. 统一训练入口怎么读（train.py）

### 2.1 主路径

1. 读取参数：从 `config.yaml` 的 `Cora` 段（或其他数据集段）读取基础超参。
2. 构建数据：Planetoid/CitationFull + NormalizeFeatures。
3. 构建模型：Encoder + Model。
4. 根据 `--method` 进入分支训练：
   - `grace`
   - `ifl-gr`
   - `gca`
   - `ifl-gc`
5. 训练结束后调用统一 `test()`。
6. `test()` 固定调用 `label_classification()`，输出 F1Mi/F1Ma。

### 2.2 四种方法的差异

#### A. GRACE
- 增强：随机 drop edge + drop feature
- 损失：标准 InfoNCE（对角正样本）
- 函数路径：`train_grace()` -> `model.loss(corrected=False)`

#### B. IFL-GR
- 增强：与 GRACE 相同
- 额外步骤：从当前表示挖掘语义正样本 `D_U^+`
- 损失：标准正样本 + 无标签语义正样本（跨视图）
- 函数路径：
  - 挖掘：`mine_unlabeled_positives()`
  - 训练：`train_iflgr()` -> `model.loss(corrected=True, corrected_variant='ifl-gr')`

#### C. GCA
- 增强：结构感知增强（degree/pr/uniform）
- 损失：标准 InfoNCE
- 函数路径：`train_gca()` -> `model.loss(corrected=False)`

#### D. IFL-GC
- 增强：使用 GCA 的结构感知增强
- 额外步骤：与 IFL-GR 一样挖掘 `D_U^+`
- 损失：标准正样本 + 语义正样本（跨视图 + 同视图混合）
- 关键系数：`iflgc_refl_du_weight`（同视图语义项占比）
- 函数路径：
  - 挖掘：`mine_unlabeled_positives()`
  - 训练：`train_iflgc()` -> `model.loss(corrected=True, corrected_variant='ifl-gc')`

## 3. 测试流程（四种方法一致）

统一在 `train.py` 末尾：
1. 使用当前训练好的 encoder 得到节点表示 `z`
2. 调用 `eval.py:label_classification(z, y, ratio=0.1)`
3. 输出格式：
   - `F1Mi=mean+-std`
   - `F1Ma=mean+-std`

说明：网格搜索脚本就是通过正则提取这行输出指标。

## 4. 自动寻参流程

### 4.1 网格搜索脚本

- IFL-GR：`tools/grid_search_iflgr_cora.py`
- GCA：`tools/grid_search_gca_cora.py`
- IFL-GC：`tools/grid_search_iflgc_cora.py`

它们的共同行为：
1. 先跑一次 `grace` 基线
2. 遍历方法特有搜索空间
3. 每组参数生成临时 config，调用 `train.py`
4. 解析 F1 指标，计算：
   - `robust_score = F1Mi_mean - std_weight * F1Mi_std`
   - `delta_vs_grace = robust_score - baseline_robust`
5. 按 `robust_score` 排序输出 CSV

### 4.2 Top 参数复验

脚本：`tools/verify_top_params.py`

流程：
1. 读取搜索结果 CSV 的前 `topk` 行
2. 每组参数重复运行 `runs` 次
3. 汇总跨次均值与波动，输出更稳定的推荐结果

支持方法：
- `--method ifl-gr`
- `--method gca`
- `--method ifl-gc`

## 5. 快速命令清单

```bash
# 训练
python train.py --dataset Cora --method grace
python train.py --dataset Cora --method ifl-gr
python train.py --dataset Cora --method gca
python train.py --dataset Cora --method ifl-gc

# 网格搜索
python tools/grid_search_iflgr_cora.py --gpu_id 0 --topk 10
python tools/grid_search_gca_cora.py --gpu_id 0 --topk 10
python tools/grid_search_iflgc_cora.py --gpu_id 0 --topk 10

# 复验
python tools/verify_top_params.py --method ifl-gr --top_params results/grid_search_iflgr_cora_results.csv --topk 3 --runs 3 --gpu_id 0
python tools/verify_top_params.py --method gca --top_params results/grid_search_gca_cora_results.csv --topk 3 --runs 3 --gpu_id 0
python tools/verify_top_params.py --method ifl-gc --top_params results/grid_search_iflgc_cora_results.csv --topk 3 --runs 3 --gpu_id 0

# 一键自动化完整流程（基线3次 + 三方法自动寻参 + Top3各3次复验）
python tools/run_cora_full_pipeline.py --gpu_id 0
```

## 6. 推荐阅读顺序

1. `train.py`（看方法分发和整体训练循环）
2. `model.py`（看损失差异，特别是 corrected 变体）
3. `eval.py`（确认测试流程）
4. `tools/grid_search_*.py`（看自动寻参）
5. `tools/verify_top_params.py`（看稳定性复验）
