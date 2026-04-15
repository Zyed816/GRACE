METHOD_CATALOG = {
    "grace": {
        "key": "grace",
        "display_name": "GRACE",
        "description": "Original graph contrastive learning baseline with random edge and feature dropout.",
        "architecture": "Two-view augmentation + shared GCN encoder + standard InfoNCE.",
        "key_parameters": ["learning_rate", "hidden_dim", "epochs", "temperature", "drop_edge_rate", "drop_feature_rate"],
    },
    "gca": {
        "key": "gca",
        "display_name": "GCA",
        "description": "Structure-aware augmentation based on degree or PageRank guided dropping.",
        "architecture": "Weighted edge/feature dropout + shared encoder + InfoNCE.",
        "key_parameters": ["gca_drop_scheme", "gca_pr_k", "drop_edge_rate", "drop_feature_rate", "temperature"],
    },
    "ifl-gr": {
        "key": "ifl-gr",
        "display_name": "IFL-GR",
        "description": "GRACE with semantically mined unlabeled positives and corrected contrastive loss.",
        "architecture": "Warmup GRACE + semantic positive mining + corrected InfoNCE.",
        "key_parameters": ["similarity_percentile", "max_du_per_node", "unlabeled_weight", "warmup_epochs", "update_interval", "beta"],
    },
    "ifl-gc": {
        "key": "ifl-gc",
        "display_name": "IFL-GC",
        "description": "GCA augmentation combined with semantic positive mining.",
        "architecture": "GCA-style views + corrected InfoNCE with same-view semantics.",
        "key_parameters": ["gca_drop_scheme", "similarity_percentile", "max_du_per_node", "iflgc_refl_du_weight", "unlabeled_weight"],
    },
}
