DATASET_CHOICES = [
    ("Cora", "Cora"),
    ("CiteSeer", "CiteSeer"),
    ("PubMed", "PubMed"),
    ("DBLP", "DBLP"),
]

METHOD_CHOICES = [
    ("grace", "GRACE"),
    ("gca", "GCA"),
    ("ifl-gr", "SG-GR"),
    ("ifl-gc", "SG-GC"),
]

SENSITIVITY_METHOD_CHOICES = [
    ("ifl-gr", "SG-GR"),
    ("ifl-gc", "SG-GC"),
]

SG_METHOD_CHOICES = [
    ("ifl-gr", "SG-GR"),
    ("ifl-gc", "SG-GC"),
]

SIGNIFICANCE_COMPARISON_CHOICES = [
    ("sg_gr_vs_grace", "SG-GR vs GRACE"),
    ("sg_gc_vs_gca", "SG-GC vs GCA"),
]

SENSITIVITY_PARAM_CHOICES = [
    ("t_s", "t_s / 相似度阈值"),
    ("M", "M / 预热轮数"),
    ("K", "K / 更新间隔"),
]

METHOD_LABELS = {
    "grace": "GRACE",
    "gca": "GCA",
    "ifl-gr": "SG-GR",
    "ifl-gc": "SG-GC",
}

METHOD_DISPLAY_ORDER = [
    "grace",
    "gca",
    "ifl-gr",
    "ifl-gc",
]

EXPERIMENT_TYPE_LABELS = {
    "method_comparison": "方法比较流水线",
    "sampling_bias": "采样偏差验证",
    "sensitivity": "超参数敏感性分析",
    "component_ablation": "组件级消融实验",
    "efficiency": "效率实验",
    "significance": "统计显著性实验",
}

SIGNIFICANCE_COMPARISON_METHODS = {
    "sg_gr_vs_grace": ("grace", "ifl-gr"),
    "sg_gc_vs_gca": ("gca", "ifl-gc"),
}

SENSITIVITY_PARAM_LABELS = {
    "t_s": "t_s / 相似度阈值",
    "M": "M / 预热轮数",
    "K": "K / 更新间隔",
}

METHOD_FILE_SLUG = {
    "grace": "grace",
    "gca": "gca",
    "ifl-gr": "iflgr",
    "ifl-gc": "iflgc",
}
