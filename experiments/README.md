# 实验模块总览

`experiments/` 目录按当前项目的三大实验模块组织：

- `method_comparison/`
  图对比学习方法比较，包括网格搜索、Top-K 参数复验、单数据集完整流程和多数据集批量调度。
- `hyperparameter_sensitivity/`
  超参数敏感性分析，包括单因素敏感性实验和结果可视化。
- `sampling_bias_validation/`
  采样偏差验证相关脚本，用于将训练阶段记录的偏差指标绘制成曲线。

推荐优先阅读根目录 [README.md](../README.md)；其中已经给出了当前目录结构、实验流程和命令示例。
