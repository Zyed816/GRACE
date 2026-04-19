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
    ("t_s", "t_s / similarity threshold"),
    ("M", "M / warmup epochs"),
    ("K", "K / update interval"),
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
    "method_comparison": "Method Comparison Pipeline",
    "sampling_bias": "Sampling Bias Validation",
    "sensitivity": "Sensitivity Analysis",
}

SENSITIVITY_PARAM_LABELS = {
    "t_s": "t_s / similarity threshold",
    "M": "M / warmup epochs",
    "K": "K / update interval",
}

METHOD_FILE_SLUG = {
    "grace": "grace",
    "gca": "gca",
    "ifl-gr": "iflgr",
    "ifl-gc": "iflgc",
}
