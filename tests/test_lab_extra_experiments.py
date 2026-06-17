import csv
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import django
from django.test import SimpleTestCase
from django.urls import reverse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "grace_web.settings")
django.setup()

from lab.forms import ComponentAblationForm, EfficiencyForm, SignificanceForm
from lab import official_results
from lab.parsers import (
    build_component_ablation_summary,
    build_efficiency_summary,
    build_significance_summary,
)
from lab.services import _run_component_ablation, _run_efficiency, _run_significance
from lab.services import _run_sampling_bias


class TrainingProcessDemoTests(SimpleTestCase):
    def test_training_process_page_renders(self):
        response = self.client.get(reverse("lab:training_process_demo"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "SG-GCL 训练过程可视化")
        self.assertContains(response, "training_process.js")


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ExtraExperimentFormTests(SimpleTestCase):
    def test_new_forms_have_expected_defaults(self):
        ablation = ComponentAblationForm(prefix="ablation")
        efficiency = EfficiencyForm(prefix="efficiency")
        significance = SignificanceForm(prefix="significance")

        self.assertEqual(ablation.fields["methods"].initial, ["sg-gr", "sg-gc"])
        self.assertEqual(efficiency.fields["methods"].initial, ["grace", "gca", "sg-gr", "sg-gc"])
        self.assertEqual(significance.fields["comparison_pairs"].initial, ["sg_gr_vs_grace", "sg_gc_vs_gca"])
        self.assertEqual(significance.fields["runs"].min_value, 2)

    def test_significance_requires_at_least_two_runs(self):
        form = SignificanceForm(
            data={
                "significance-name": "Sig",
                "significance-dataset": "Cora",
                "significance-comparison_pairs": ["sg_gr_vs_grace"],
                "significance-gpu_id": "0",
                "significance-runs": "1",
                "significance-eval_repeats": "3",
                "significance-std_weight": "0.5",
                "significance-alpha": "0.05",
            },
            prefix="significance",
        )

        self.assertFalse(form.is_valid())
        self.assertIn("runs", form.errors)


class ExtraExperimentParserTests(unittest.TestCase):
    def test_component_ablation_summary_reads_drop_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "extra_ablation_cora_results.csv"
            write_csv(
                csv_path,
                ["stage", "dataset", "method", "variant", "robust_score", "drop_vs_full"],
                [
                    {"stage": "summary", "dataset": "Cora", "method": "sg-gr", "variant": "full", "robust_score": "0.82", "drop_vs_full": ""},
                    {"stage": "summary", "dataset": "Cora", "method": "sg-gr", "variant": "no_warmup", "robust_score": "0.81", "drop_vs_full": "0.01"},
                ],
            )

            summary = build_component_ablation_summary([csv_path])

        self.assertEqual(summary["summary_rows"], 2)
        self.assertEqual(summary["max_drop"]["variant"], "no_warmup")
        self.assertAlmostEqual(summary["max_drop"]["drop_vs_full"], 0.01)

    def test_efficiency_summary_finds_fastest_and_ratios(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "efficiency_cora_results.csv"
            write_csv(
                csv_path,
                [
                    "stage",
                    "dataset",
                    "method",
                    "train_total_sec",
                    "wall_time_sec",
                    "time_ratio_vs_grace",
                    "overhead_ratio_vs_base",
                ],
                [
                    {"stage": "summary", "dataset": "Cora", "method": "grace", "train_total_sec": "3.0", "wall_time_sec": "5.0", "time_ratio_vs_grace": "1.0", "overhead_ratio_vs_base": ""},
                    {"stage": "summary", "dataset": "Cora", "method": "sg-gr", "train_total_sec": "4.5", "wall_time_sec": "6.0", "time_ratio_vs_grace": "1.5", "overhead_ratio_vs_base": "1.5"},
                    {"stage": "summary", "dataset": "Cora", "method": "sg-gc", "train_total_sec": "6.0", "wall_time_sec": "7.0", "time_ratio_vs_grace": "2.0", "overhead_ratio_vs_base": "1.25"},
                ],
            )

            summary = build_efficiency_summary([csv_path])

        self.assertEqual(summary["fastest_method"]["method"], "grace")
        self.assertAlmostEqual(summary["sggr_ratio"], 1.5)
        self.assertAlmostEqual(summary["sggc_ratio"], 1.25)

    def test_significance_summary_filters_primary_robust_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "significance_tests_summary.csv"
            fieldnames = [
                "stage",
                "dataset",
                "metric",
                "baseline_method",
                "target_method",
                "mean_delta",
                "p_value_holm",
                "significant",
                "notes",
            ]
            write_csv(
                csv_path,
                fieldnames,
                [
                    {"stage": "test", "dataset": "Cora", "metric": "robust_score", "baseline_method": "grace", "target_method": "sg-gr", "mean_delta": "0.01", "p_value_holm": "0.01", "significant": "True", "notes": "primary"},
                    {"stage": "test", "dataset": "Cora", "metric": "F1Mi_mean", "baseline_method": "grace", "target_method": "sg-gr", "mean_delta": "0.02", "p_value_holm": "0.01", "significant": "True", "notes": "primary"},
                    {"stage": "test", "dataset": "Cora", "metric": "robust_score", "baseline_method": "grace", "target_method": "gca", "mean_delta": "0.03", "p_value_holm": "0.01", "significant": "True", "notes": "supplementary"},
                ],
            )

            summary = build_significance_summary([], summary_csv=csv_path)

        self.assertEqual(summary["primary_tests"], 1)
        self.assertEqual(summary["significant_tests"], 1)
        self.assertEqual(summary["best_delta"]["comparison"], "SG-GR vs GRACE")


class ExtraExperimentServiceCommandTests(SimpleTestCase):
    def collect_commands(self, runner, run, summary_patch, summary_value):
        commands = []

        def fake_run_command(_run, command, label):
            commands.append((label, command))
            return ""

        with patch("lab.services._run_command", side_effect=fake_run_command), patch(
            "lab.services._register_artifact"
        ), patch(summary_patch, return_value=summary_value):
            runner(run)
        return commands

    def collect_commands_and_artifacts(self, runner, run, summary_patch, summary_value):
        commands = []
        artifacts = []

        def fake_run_command(_run, command, label):
            commands.append((label, command))
            return ""

        def fake_register_artifact(_run, label, artifact_type, relative_path, metadata=None):
            artifacts.append((label, artifact_type, Path(relative_path).as_posix()))
            return SimpleNamespace(label=label, artifact_type=artifact_type, relative_path=relative_path)

        with patch("lab.services._run_command", side_effect=fake_run_command), patch(
            "lab.services._register_artifact", side_effect=fake_register_artifact
        ), patch(summary_patch, return_value=summary_value):
            runner(run)
        return commands, artifacts

    def test_component_ablation_uses_run_scoped_paths(self):
        run = SimpleNamespace(
            pk=123,
            config={
                "dataset": "Cora",
                "methods": ["sg-gr"],
                "gpu_id": 0,
                "runs": 1,
                "std_weight": 0.5,
                "continue_on_error": False,
            },
        )

        commands = self.collect_commands(
            _run_component_ablation,
            run,
            "lab.services.build_component_ablation_summary",
            {"summary_rows": 0},
        )

        self.assertEqual(commands[0][0], "component-ablation")
        self.assertIn("experiments/component_ablation/run_component_ablation.py", commands[0][1])
        self.assertIn("results/webapp/run_123/extra_ablation_cora_results.csv", commands[0][1])
        self.assertEqual(commands[1][0], "component-ablation-plot")
        self.assertIn("results/webapp/run_123/plots", commands[1][1])

    def test_sampling_bias_registers_svg_curve_when_plotting(self):
        run = SimpleNamespace(
            pk=129,
            config={
                "dataset": "Cora",
                "method": "grace",
                "gpu_id": 0,
                "title": "Sampling",
            },
        )

        commands, artifacts = self.collect_commands_and_artifacts(
            _run_sampling_bias,
            run,
            "lab.services.build_sampling_bias_summary",
            {"epochs": 0},
        )
        paths = [item[2] for item in artifacts]

        self.assertEqual(commands[1][0], "sampling-plot")
        self.assertIn("--formats", commands[1][1])
        self.assertIn("logs/webapp/run_129/exp1_cora_grace_curves.png", paths)
        self.assertIn("logs/webapp/run_129/exp1_cora_grace_curves.svg", paths)

    def test_component_ablation_registers_split_effect_plots(self):
        run = SimpleNamespace(
            pk=126,
            config={
                "dataset": "Cora",
                "methods": ["sg-gr"],
                "gpu_id": 0,
                "runs": 1,
                "std_weight": 0.5,
                "continue_on_error": False,
            },
        )

        _commands, artifacts = self.collect_commands_and_artifacts(
            _run_component_ablation,
            run,
            "lab.services.build_component_ablation_summary",
            {"summary_rows": 0},
        )
        paths = [item[2] for item in artifacts]

        for stem in [
            "extra_ablation_warmup_M_effect",
            "extra_ablation_update_K_effect",
            "extra_ablation_weight_w_effect",
        ]:
            self.assertIn(f"results/webapp/run_126/plots/{stem}.png", paths)
            self.assertIn(f"results/webapp/run_126/plots/{stem}.svg", paths)
        self.assertNotIn("results/webapp/run_126/plots/extra_ablation_overview.png", paths)

    def test_efficiency_uses_run_scoped_paths(self):
        run = SimpleNamespace(
            pk=124,
            config={
                "dataset": "Cora",
                "methods": ["grace", "sg-gr"],
                "gpu_id": 0,
                "runs": 1,
                "std_weight": 0.5,
                "continue_on_error": False,
            },
        )

        commands = self.collect_commands(
            _run_efficiency,
            run,
            "lab.services.build_efficiency_summary",
            {"summary_rows": 0},
        )

        self.assertEqual(commands[0][0], "efficiency")
        self.assertIn("experiments/efficiency/run_efficiency_experiment.py", commands[0][1])
        self.assertIn("results/webapp/run_124/efficiency_cora_results.csv", commands[0][1])
        self.assertEqual(commands[1][0], "efficiency-plot")

    def test_efficiency_registers_only_train_and_wall_time_plots(self):
        run = SimpleNamespace(
            pk=127,
            config={
                "dataset": "Cora",
                "methods": ["grace", "sg-gr"],
                "gpu_id": 0,
                "runs": 1,
                "std_weight": 0.5,
                "continue_on_error": False,
            },
        )

        _commands, artifacts = self.collect_commands_and_artifacts(
            _run_efficiency,
            run,
            "lab.services.build_efficiency_summary",
            {"summary_rows": 0},
        )
        paths = [item[2] for item in artifacts]

        self.assertIn("results/webapp/run_127/plots/efficiency_train_total_time.png", paths)
        self.assertIn("results/webapp/run_127/plots/efficiency_train_total_time.svg", paths)
        self.assertIn("results/webapp/run_127/plots/efficiency_wall_time.png", paths)
        self.assertIn("results/webapp/run_127/plots/efficiency_wall_time.svg", paths)
        self.assertNotIn("results/webapp/run_127/plots/efficiency_time_ratio.png", paths)

    def test_significance_maps_pairs_to_methods_and_run_scoped_paths(self):
        run = SimpleNamespace(
            pk=125,
            config={
                "dataset": "Cora",
                "comparison_pairs": ["sg_gr_vs_grace"],
                "gpu_id": 0,
                "runs": 2,
                "eval_repeats": 3,
                "std_weight": 0.5,
                "alpha": 0.05,
                "continue_on_error": False,
            },
        )

        commands = self.collect_commands(
            _run_significance,
            run,
            "lab.services.build_significance_summary",
            {"primary_tests": 0},
        )

        self.assertEqual(commands[0][0], "significance")
        self.assertIn("experiments/statistical_significance/run_significance_experiment.py", commands[0][1])
        self.assertIn("grace", commands[0][1])
        self.assertIn("sg-gr", commands[0][1])
        self.assertNotIn("gca", commands[0][1])
        self.assertIn("results/webapp/run_125/significance_cora_results.csv", commands[0][1])
        self.assertEqual(commands[1][0], "significance-analyze")
        self.assertEqual(commands[2][0], "significance-plot")

    def test_significance_registers_paired_delta_svg(self):
        run = SimpleNamespace(
            pk=128,
            config={
                "dataset": "Cora",
                "comparison_pairs": ["sg_gr_vs_grace"],
                "gpu_id": 0,
                "runs": 2,
                "eval_repeats": 3,
                "std_weight": 0.5,
                "alpha": 0.05,
                "continue_on_error": False,
            },
        )

        _commands, artifacts = self.collect_commands_and_artifacts(
            _run_significance,
            run,
            "lab.services.build_significance_summary",
            {"primary_tests": 0},
        )
        paths = [item[2] for item in artifacts]

        self.assertIn("results/webapp/run_128/plots/significance_paired_delta.png", paths)
        self.assertIn("results/webapp/run_128/plots/significance_paired_delta.svg", paths)


class OfficialResultArtifactTests(SimpleTestCase):
    def fake_artifact(self, label, artifact_type, path, metadata=None):
        return {
            "label": label,
            "artifact_type": artifact_type,
            "relative_path": Path(path).as_posix(),
            "metadata": metadata or {},
        }

    def test_component_ablation_official_artifacts_use_split_effect_plots(self):
        with patch("lab.official_results._artifact", side_effect=self.fake_artifact):
            entries = official_results._component_ablation_entries("en")

        paths = [artifact["relative_path"] for artifact in entries[0]["artifacts"]]
        self.assertTrue(any(path.endswith("extra_ablation_warmup_M_effect.png") for path in paths))
        self.assertTrue(any(path.endswith("extra_ablation_update_K_effect.svg") for path in paths))
        self.assertFalse(any(path.endswith("extra_ablation_overview.png") for path in paths))

    def test_sampling_bias_official_artifacts_include_svg_curve(self):
        with patch("lab.official_results._artifact", side_effect=self.fake_artifact):
            entries = official_results._sampling_bias_entries("en")

        paths = [artifact["relative_path"] for artifact in entries[0]["artifacts"]]
        self.assertTrue(any(path.endswith("_curves.svg") for path in paths))

    def test_efficiency_official_artifacts_include_svg_without_ratio_plot(self):
        with patch("lab.official_results._artifact", side_effect=self.fake_artifact):
            entries = official_results._efficiency_entries("en")

        paths = [artifact["relative_path"] for artifact in entries[0]["artifacts"]]
        self.assertTrue(any(path.endswith("efficiency_train_total_time.svg") for path in paths))
        self.assertTrue(any(path.endswith("efficiency_wall_time.svg") for path in paths))
        self.assertFalse(any(path.endswith("efficiency_time_ratio.png") for path in paths))

    def test_significance_official_artifacts_include_paired_delta_svg(self):
        with patch("lab.official_results._artifact", side_effect=self.fake_artifact):
            entries = official_results._significance_entries("en")

        paths = [artifact["relative_path"] for artifact in entries[0]["artifacts"]]
        self.assertTrue(any(path.endswith("significance_paired_delta.svg") for path in paths))

    def test_sensitivity_official_artifacts_include_combined_split_plots(self):
        with patch("lab.official_results._artifact", side_effect=self.fake_artifact):
            entries = official_results._sensitivity_entries("en")

        combined = [entry for entry in entries if entry["slug"] == "sensitivity-all"]
        self.assertEqual(len(combined), 1)
        paths = [artifact["relative_path"] for artifact in combined[0]["artifacts"]]
        for stem in ["sg_sensitivity_ts_effect", "sg_sensitivity_M_effect", "sg_sensitivity_K_effect"]:
            self.assertTrue(any(path.endswith(f"{stem}.png") for path in paths), stem)
            self.assertTrue(any(path.endswith(f"{stem}.svg") for path in paths), stem)
