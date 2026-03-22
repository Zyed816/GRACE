# IFL-GR 网格搜索快速指南

## 为什么网格搜索结果和单独训练结果会不同？

### 原因
1. **随机性累积**：虽然种子固定，但数据增强的随机序列在网格搜索（串联多个 trial）和独立训练（新进程）中是不同的。
2. **评估方差**：`eval.py` 中的 `@repeat(3)` 装饰器对逻辑回归分类器重复 3 次，所以不同运行的 F1 指标本身就有方差（通常 ±0.01）。
3. **网格搜索的单次估计**：网格搜索中每个参数组合只运行一次评估，而你独立运行时是新的随机重复。

**预期差异范围**：Top-1 参数的 F1 指标在单独运行时可能波动 ±0.005~0.015，这是正常的。

## 如何进行可靠的参数验证？

### 方法 1：快速验证（推荐）
对网格搜索输出的 Top-3 参数各运行 **3 次**，取平均值作为最终评估指标。
```bash
python tools/verify_top_params.py --top_params results/grid_search_iflgr_cora_results.csv --topk 3 --gpu_id 0
```

### 方法 2：直接使用 Top-1 参数
如果你的时间有限，可以直接：
1. 将网格搜索输出的 Top-1 参数填入 `config.yaml` 的 Cora 配置。
2. 运行 `python train.py --dataset Cora --method ifl-gr` 三遍，手动取平均和标准差。
3. 与 `python train.py --dataset Cora --method grace` 的结果对比。

### 方法 3：完整二阶段搜索
在时间充裕的情况下，用 `fine_grid_search.py` 进行更精细的搜索（围绕 Top-3 做 3x3x3 网格）。

## CSV 文件释义

`results/grid_search_iflgr_cora_results.csv` 中的列名：
- `similarity_percentile`: 自适应阈值的分位数（较高 = 更严格的伪正样本筛选）
- `max_du_per_node`: 每个节点最多保留多少个 DU+
- `unlabeled_weight`: 无标签项的权重系数
- `warmup_epochs`: 预热轮数
- `F1Mi_mean / F1Mi_std`: 微 F1 的均值和标准差
- `F1Ma_mean / F1Ma_std`: 宏 F1 的均值和标准差
- `robust_score`: 稳健评分（F1Mi_mean - 0.5 * F1Mi_std）
- `delta_vs_grace`: 相对 GRACE 基线的提升

## 典型情况

### 情况 A：参数已验证且稳定超过 GRACE
- 网格搜索的 Top-1 参数 F1Mi 为 0.850+，独立运行的结果为 0.833+
- **原因**：网格搜索在其他参数配置下找到的最优值，在新的随机序列中可能有衰减。
- **解决**：用 verify_top_params.py 验证，或对该参数再运行 3 遍确认平均值。

### 情况 B：参数不稳定或下降
- 独立运行多次出现 > 0.015 的标准差
- **可能原因**：该参数组合对初始化无序列敏感，或伪正样本质量不稳定。
- **解决**：选择 Top-2 或 Top-3 的参数再试，或参考 CSV 中 delta_vs_grace 相对靠前的其他参数。

## 快速启动

1. **运行网格搜索**（约 20-30 分钟，取决于硬件）
   ```bash
   python tools/grid_search_iflgr_cora.py --gpu_id 0 --topk 10
   ```

2. **验证 Top-3 参数**（约 40-60 分钟，3 遍完整训练）
   ```bash
   python tools/verify_top_params.py --top_params results/grid_search_iflgr_cora_results.csv --topk 3 --gpu_id 0
   ```

3. **更新配置** 或 **进行精搜**（可选）

## GCA 参数搜索

GCA 的搜索脚本与 IFL-GR 保持一致的输出与对比方式（同样使用 robust_score 和 delta_vs_grace）：

1. **运行 GCA 网格搜索**（默认 108 个 trial）
   ```bash
   python tools/grid_search_gca_cora.py --gpu_id 0 --topk 10
   ```

2. **验证 GCA Top 参数**（复用 verify 工具）
   ```bash
   python tools/verify_top_params.py --method gca --top_params results/grid_search_gca_cora_results.csv --topk 3 --runs 3 --gpu_id 0
   ```

`results/grid_search_gca_cora_results.csv` 关键列：
- `gca_drop_scheme`: 增强策略（degree / pr / uniform）
- `drop_edge_rate_1`, `drop_edge_rate_2`: 两个视图的边丢弃率
- `drop_feature_rate_1`, `drop_feature_rate_2`: 两个视图的特征丢弃率
- `tau`: 对比温度
- `robust_score`, `delta_vs_grace`: 与 IFL-GR 搜索同定义

## IFL-GC 参数搜索

IFL-GC 继承 GCA 的采样增强，并加入语义引导的修正损失，搜索脚本如下：

1. **运行 IFL-GC 网格搜索**
   ```bash
   python tools/grid_search_iflgc_cora.py --gpu_id 0 --topk 10
   ```

2. **验证 IFL-GC Top 参数**
   ```bash
   python tools/verify_top_params.py --method ifl-gc --top_params results/grid_search_iflgc_cora_results.csv --topk 3 --runs 3 --gpu_id 0
   ```

`results/grid_search_iflgc_cora_results.csv` 关键列：
- `similarity_percentile`, `max_du_per_node`: 语义正样本筛选强度
- `unlabeled_weight`: 无标签语义正样本损失权重
- `iflgc_refl_du_weight`: 同视图语义正样本项权重
- `gca_drop_scheme`, `drop_edge_rate_*`, `drop_feature_rate_*`: 采样增强策略
- `robust_score`, `delta_vs_grace`: 与其他搜索脚本同定义

## 一键自动化完整实验流程

如果你希望按固定流程自动执行并将结果写到同一个文件，可使用：

```bash
python tools/run_cora_full_pipeline.py --gpu_id 0
```

默认流程：
1. `grace` 基线运行 3 次
2. `ifl-gr` 自动寻参 + Top3 各 3 次复验
3. `gca` 自动寻参 + Top3 各 3 次复验
4. `ifl-gc` 自动寻参 + Top3 各 3 次复验

默认统一结果文件：
- `results/cora_full_pipeline_results.csv`

可调参数示例：
```bash
python tools/run_cora_full_pipeline.py --gpu_id 0 --baseline_runs 3 --topk_verify 3 --runs_per_top 3 --out results/cora_compare.csv
```
