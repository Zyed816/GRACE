import unittest
from pathlib import Path

import yaml

from experiments.statistical_significance.analyze_significance_results import (
    bootstrap_ci,
    cohen_dz,
    holm_bonferroni,
    rank_biserial_effect,
    wilcoxon_pvalue,
)
from experiments.statistical_significance.run_significance_experiment import (
    CSV_HEADERS,
    METHOD_CHOICES,
    build_trial_updates,
    select_base_params,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class StatisticalSignificanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (REPO_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
            cls.base_config = yaml.safe_load(f)

    def test_selects_existing_best_verified_params(self):
        selected = select_base_params(str(REPO_ROOT), "Cora", "sg-gr")

        self.assertEqual(selected["source"], "full_pipeline")
        self.assertIn("similarity_percentile", selected["params"])
        self.assertIn("warmup_epochs", selected["params"])

    def test_seed_policy_is_shared_across_methods(self):
        seeds = []
        eval_seeds = []
        for method in METHOD_CHOICES:
            selected = select_base_params(str(REPO_ROOT), "Cora", method)
            updates = build_trial_updates(
                self.base_config,
                "Cora",
                method,
                selected["params"],
                run_idx=3,
                eval_repeats=3,
            )
            seeds.append(updates["seed"])
            eval_seeds.append(updates["eval_seed"])

        expected = self.base_config["Cora"]["seed"] + 3
        self.assertEqual(seeds, [expected] * len(METHOD_CHOICES))
        self.assertEqual(eval_seeds, [expected] * len(METHOD_CHOICES))

    def test_statistical_helpers_are_stable_on_small_sample(self):
        deltas = [0.02, 0.01, 0.03, -0.01]

        p_value = wilcoxon_pvalue(deltas)
        ci_low, ci_high = bootstrap_ci(deltas, n_boot=1000, seed=7)
        adjusted = holm_bonferroni([0.01, 0.04, 0.20], alpha=0.05)

        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
        self.assertLess(ci_low, ci_high)
        self.assertEqual(adjusted, [0.03, 0.08, 0.20])
        self.assertGreater(cohen_dz(deltas), 0.0)
        self.assertGreater(rank_biserial_effect(deltas), 0.0)

    def test_csv_header_contains_run_summary_and_test_fields(self):
        required = {
            "stage",
            "dataset",
            "method",
            "run_idx",
            "seed",
            "eval_seed",
            "robust_score",
            "metric",
            "baseline_method",
            "target_method",
            "p_value_holm",
            "significant",
        }

        self.assertTrue(required.issubset(set(CSV_HEADERS)))


if __name__ == "__main__":
    unittest.main()
