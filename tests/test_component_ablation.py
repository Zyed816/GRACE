import unittest
from pathlib import Path

import yaml

from experiments.component_ablation.run_component_ablation import (
    build_trial_updates,
    params_to_dataset_updates,
    select_base_params,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class ComponentAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (REPO_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
            cls.base_config = yaml.safe_load(f)

    def test_selects_best_verified_cora_iflgr_candidate(self):
        selected = select_base_params(str(REPO_ROOT), "Cora", "ifl-gr")

        self.assertEqual(selected["source"], "full_pipeline")
        self.assertEqual(selected["candidate_rank"], "3")
        self.assertIn("similarity_percentile", selected["params"])
        self.assertIn("warmup_epochs", selected["params"])

    def test_variant_updates_change_only_target_controls(self):
        base_params = {
            "similarity_percentile": "99.5",
            "max_du_per_node": "12",
            "unlabeled_weight": "0.2",
            "warmup_epochs": "80",
            "update_interval": "3",
            "beta": "2.2",
            "use_mutual_topk": "True",
            "corrected_ramp_epochs": "20",
            "tau": "0.3",
        }

        full = build_trial_updates(self.base_config, "Cora", "ifl-gr", base_params, "full", run_idx=1)
        no_warmup = build_trial_updates(
            self.base_config, "Cora", "ifl-gr", base_params, "no_warmup", run_idx=1
        )
        single_mining = build_trial_updates(
            self.base_config, "Cora", "ifl-gr", base_params, "single_mining", run_idx=1
        )
        uniform_weight = build_trial_updates(
            self.base_config, "Cora", "ifl-gr", base_params, "uniform_weight", run_idx=1
        )

        self.assertEqual(full["seed"], self.base_config["Cora"]["seed"] + 1)
        self.assertEqual(no_warmup["warmup_epochs"], 0)
        self.assertEqual(single_mining["update_interval"], self.base_config["Cora"]["num_epochs"] + 1)
        self.assertEqual(uniform_weight["beta"], 0.0)
        self.assertEqual(single_mining["warmup_epochs"], full["warmup_epochs"])
        self.assertEqual(uniform_weight["warmup_epochs"], full["warmup_epochs"])

    def test_iflgc_params_are_mapped_to_dataset_updates(self):
        base_params = {
            "gca_drop_scheme": "degree",
            "similarity_percentile": "99.7",
            "max_du_per_node": "14",
            "unlabeled_weight": "0.3",
            "iflgc_refl_du_weight": "0.5",
            "warmup_epochs": "80",
            "tau": "0.4",
            "drop_edge_rate_1": "0.3",
            "drop_edge_rate_2": "0.5",
            "drop_feature_rate_1": "0.3",
            "drop_feature_rate_2": "0.4",
            "update_interval": "3",
            "beta": "2.2",
            "use_mutual_topk": "True",
            "corrected_ramp_epochs": "20",
            "gca_pr_k": "200",
        }

        updates = params_to_dataset_updates(base_params, "ifl-gc")

        self.assertEqual(updates["gca_drop_scheme"], "degree")
        self.assertEqual(updates["max_du_per_node"], 14)
        self.assertEqual(updates["iflgc_refl_du_weight"], 0.5)
        self.assertEqual(updates["gca_pr_k"], 200)
        self.assertIsNone(updates["similarity_threshold"])


if __name__ == "__main__":
    unittest.main()
