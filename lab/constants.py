DATASET_CHOICES = [
    ("Cora", "Cora"),
    ("CiteSeer", "CiteSeer"),
    ("PubMed", "PubMed"),
    ("DBLP", "DBLP"),
]

METHOD_CHOICES = [
    ("grace", "GRACE"),
    ("gca", "GCA"),
    ("ifl-gr", "IFL-GR"),
    ("ifl-gc", "IFL-GC"),
]

SENSITIVITY_METHOD_CHOICES = [
    ("ifl-gr", "IFL-GR"),
    ("ifl-gc", "IFL-GC"),
]

SENSITIVITY_PARAM_CHOICES = [
    ("t_s", "t_s / 相似度阈值"),
    ("M", "M / 预热轮数"),
    ("K", "K / 更新间隔"),
]

METHOD_LABELS = {
    "grace": "GRACE",
    "gca": "GCA",
    "ifl-gr": "IFL-GR",
    "ifl-gc": "IFL-GC",
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
