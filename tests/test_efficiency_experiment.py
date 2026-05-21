import unittest
from pathlib import Path

import yaml

from experiments.efficiency.run_efficiency_experiment import (
    build_trial_updates,
    params_to_dataset_updates,
    parse_timing_stats,
    select_base_params,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class EfficiencyExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (REPO_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
            cls.base_config = yaml.safe_load(f)

    def test_parse_timing_stats_counts_refresh_and_phase_times(self):
        output = "\n".join(
            [
                "(T) | Epoch=001, phase=warmup, loss=1.0000, this epoch 0.1000, total 0.1000",
                "(T) | Epoch=002, loss=0.9000, phase=corrected, refresh_du=1, "
                "lambda_u=0.1000, ts=0.9000, mined_pairs=10, avg_pairs_per_node=1.00, "
                "mean_w=1.2000, this epoch 0.3000, total 0.4000",
                "(T) | Epoch=003, loss=0.8000, phase=corrected, refresh_du=0, "
                "lambda_u=0.1000, ts=0.9000, mined_pairs=10, avg_pairs_per_node=1.00, "
                "mean_w=1.2000, this epoch 0.2000, total 0.6000",
            ]
        )

        stats = parse_timing_stats(output)

        self.assertEqual(stats["epochs_observed"], 3)
        self.assertAlmostEqual(stats["train_total_sec"], 0.6)
        self.assertAlmostEqual(stats["epoch_time_mean_sec"], 0.2)
        self.assertAlmostEqual(stats["epoch_time_median_sec"], 0.2)
        self.assertAlmostEqual(stats["throughput_epoch_per_sec"], 5.0)
        self.assertEqual(stats["refresh_count"], 1)
        self.assertAlmostEqual(stats["refresh_epoch_time_mean_sec"], 0.3)
        self.assertAlmostEqual(stats["warmup_epoch_time_mean_sec"], 0.1)
        self.assertAlmostEqual(stats["corrected_epoch_time_mean_sec"], 0.25)

    def test_method_params_are_mapped_to_config_updates(self):
        grace_updates = params_to_dataset_updates(
            {
                "drop_edge_rate_1": "0.5",
                "drop_edge_rate_2": "0.6",
                "drop_feature_rate_1": "0.4",
                "drop_feature_rate_2": "0.5",
                "tau": "0.8",
            },
            "grace",
        )
        gca_updates = params_to_dataset_updates(
            {
                "gca_drop_scheme": "pr",
                "gca_pr_k": "200",
                "drop_edge_rate_1": "0.3",
                "drop_edge_rate_2": "0.4",
                "drop_feature_rate_1": "0.2",
                "drop_feature_rate_2": "0.3",
                "tau": "0.4",
            },
            "gca",
        )

        self.assertEqual(grace_updates["tau"], 0.8)
        self.assertEqual(gca_updates["gca_drop_scheme"], "pr")
        self.assertEqual(gca_updates["gca_pr_k"], 200)

    def test_selects_existing_full_pipeline_grace_params(self):
        selected = select_base_params(str(REPO_ROOT), "Cora", "grace")

        self.assertEqual(selected["source"], "full_pipeline")
        self.assertEqual(selected["candidate_rank"], "0")
        self.assertIn("tau", selected["params"])

    def test_trial_seed_is_shared_policy(self):
        updates = build_trial_updates(
            self.base_config,
            "Cora",
            "grace",
            {"tau": "0.8"},
            run_idx=1,
        )

        self.assertEqual(updates["seed"], self.base_config["Cora"]["seed"] + 1)


if __name__ == "__main__":
    unittest.main()
