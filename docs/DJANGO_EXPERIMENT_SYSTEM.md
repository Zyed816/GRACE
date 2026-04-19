# Django 实验系统说明

当前仓库已经新增一个基于 Django 的实验系统，用于在保留原有脚本能力的基础上，通过网页进行实验配置、提交、查看结果与基础可视化。

## 已实现功能

1. 图对比学习方法比较
   支持从网页选择数据集并发起完整流水线实验。
   实验后台会复用 `experiments/method_comparison/run_full_pipeline.py`。
   结果页会展示统一 CSV 预览和方法稳健分数柱状图。

2. 采样偏差验证
   支持配置数据集、方法、GPU 和图像标题。
   实验后台会先运行 `train.py --exp1_metrics --exp1_log_csv ...`，再调用绘图脚本。
   结果页会展示折线图、日志 CSV 和生成的 PNG。

3. 超参数敏感性分析
   支持选择数据集、方法集合、论文超参数、锚点排名和运行次数。
   实验后台会逐方法复用 `run_ifl_param_sensitivity.py`，随后调用 `plot_ifl_param_sensitivity.py` 生成总览图与报告。
   结果页会展示最优稳健分数柱状图、CSV 预览、PNG 和 Markdown 报告。

## 目录结构

新增的主要目录如下：

```text
manage.py
grace_web/
lab/
docs/DJANGO_EXPERIMENT_SYSTEM.md
```

其中：

- `grace_web/`：Django 项目配置
- `lab/`：实验系统应用，包含模型、表单、视图、任务调度、模板和样式
- `lab/management/commands/process_experiment.py`：后台 worker 命令

## 运行方式

建议在你当前能正常运行原始训练脚本的同一个 Python / Conda 环境中执行：

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

启动后访问：

```text
http://127.0.0.1:8000/
```

## 结果输出策略

为了避免覆盖你现有的结果文件，网页实验默认把新产物写入独立目录：

- `results/webapp/run_<id>/...`
- `logs/webapp/run_<id>/...`

原有命令行默认输出路径保持不变。

此外，`experiments/method_comparison/run_full_pipeline.py` 新增了一个可选参数：

```bash
--grid_dir <dir>
```

这个参数只影响网页系统使用时的网格搜索输出目录；如果你继续沿用原命令行方式，不传该参数即可，行为与原先一致。

## 结果页面内容

每个实验详情页目前包含：

- 实验状态、时间、配置摘要
- 后台执行命令
- 完整运行日志
- 结果图像
- CSV 预览
- 基础 SVG 可视化图表
- Markdown 报告内容

## 后续可扩展方向

如果你后面还想继续扩展，这个 Django 系统可以很自然地继续增加：

- 用户认证与实验权限
- 任务队列和更稳定的异步执行器
- 更丰富的交互式图表
- 多实验对比视图
- 结果导出与报告生成
