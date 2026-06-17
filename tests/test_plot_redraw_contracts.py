import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


class PlotRedrawContractTests(unittest.TestCase):
    def test_shared_helpers_save_png_pdf_svg_without_rasterizing_svg(self):
        from experiments.plotting_common import normalize_formats, save_figure_formats

        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            paths = save_figure_formats(
                fig,
                Path(tmp) / "demo_plot",
                normalize_formats([".PNG", "pdf", "svg", "svg"]),
                dpi=120,
            )
            plt.close(fig)

            suffixes = [path.suffix for path in paths]
            self.assertEqual(suffixes, [".png", ".pdf", ".svg"])
            for path in paths:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0)

            svg_text = paths[-1].read_text(encoding="utf-8")
            self.assertNotIn("<image", svg_text.lower())

    def test_panel_label_uses_chinese_parentheses_and_dataset_name(self):
        from experiments.plotting_common import panel_label

        self.assertEqual(panel_label(0, "Cora"), "（a）Cora")
        self.assertEqual(panel_label(3, "DBLP"), "（d）DBLP")

    def test_component_ablation_effect_specs_use_requested_file_names(self):
        from experiments.component_ablation.plot_component_ablation import EFFECT_SPECS

        stems = [spec["file_stem"] for spec in EFFECT_SPECS]

        self.assertEqual(
            stems,
            [
                "extra_ablation_warmup_M_effect",
                "extra_ablation_update_K_effect",
                "extra_ablation_weight_w_effect",
            ],
        )

    def test_sensitivity_effect_specs_use_requested_file_names(self):
        from experiments.hyperparameter_sensitivity.plot_sg_param_sensitivity import PARAM_EFFECT_SPECS

        stems = [spec["file_stem"] for spec in PARAM_EFFECT_SPECS]

        self.assertEqual(
            stems,
            [
                "sg_sensitivity_ts_effect",
                "sg_sensitivity_M_effect",
                "sg_sensitivity_K_effect",
            ],
        )

    def test_sampling_bias_legend_uses_raw_metric_names(self):
        from plot.plot import LEGEND_LABELS

        self.assertEqual(LEGEND_LABELS["violation"], "violation_rate")
        self.assertEqual(LEGEND_LABELS["margin"], "mean_margin")

    def test_sampling_bias_curves_have_print_friendly_line_styles(self):
        from plot.plot import CURVE_STYLES

        violation = CURVE_STYLES["violation"]
        margin = CURVE_STYLES["margin"]

        self.assertNotEqual(violation["linestyle"], margin["linestyle"])
        self.assertNotEqual(violation["marker"], margin["marker"])
        self.assertGreater(violation["markevery"], 0)
        self.assertGreater(margin["markevery"], 0)

    def test_method_overview_uses_robust_score_metric_label(self):
        from experiments.method_comparison.plot_method_comparison_results import (
            OVERVIEW_METRIC_LABELS,
            REQUESTED_METRIC_NAMES,
        )

        self.assertEqual(OVERVIEW_METRIC_LABELS[0], "robust_score")
        self.assertEqual(REQUESTED_METRIC_NAMES, ["robust_score", "F1Mi_mean", "F1Ma_mean"])

    def test_ablation_effect_uses_stable_score_label_and_local_y_limits(self):
        from experiments.component_ablation.plot_component_ablation import (
            ROBUST_SCORE_YLABEL,
            effect_axis_limits,
        )

        summary_df = pd.DataFrame(
            [
                {"dataset": "Cora", "method": "sg-gr", "variant": "full", "robust_score": 0.790, "robust_score_std": 0.001},
                {"dataset": "Cora", "method": "sg-gr", "variant": "uniform_weight", "robust_score": 0.775, "robust_score_std": 0.001},
                {"dataset": "Cora", "method": "sg-gc", "variant": "full", "robust_score": 0.787, "robust_score_std": 0.001},
                {"dataset": "Cora", "method": "sg-gc", "variant": "uniform_weight", "robust_score": 0.772, "robust_score_std": 0.001},
            ]
        )

        lower, upper = effect_axis_limits(summary_df, "uniform_weight")

        self.assertEqual(ROBUST_SCORE_YLABEL, "稳健性评分robust_score（%）")
        self.assertGreaterEqual(lower, 75.0)
        self.assertLessEqual(upper, 81.0)

    def test_ablation_effect_plot_scales_each_dataset_panel_locally(self):
        from experiments.component_ablation import plot_component_ablation as ablation_plot

        rows = []
        for dataset, base in [("Cora", 0.790), ("DBLP", 0.710)]:
            for method in ["sg-gr", "sg-gc"]:
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "variant": "full",
                        "robust_score": base,
                        "robust_score_std": 0.001,
                    }
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "variant": "uniform_weight",
                        "robust_score": base - 0.012,
                        "robust_score_std": 0.001,
                    }
                )
        summary_df = pd.DataFrame(rows)
        observed_limits = []
        original_draw = ablation_plot.draw_effect_axis
        ablation_plot.configure_plot_style()

        def capture_limits(ax, subset, variant, y_limits):
            observed_limits.append(y_limits)
            return original_draw(ax, subset, variant, y_limits)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ablation_plot, "draw_effect_axis", side_effect=capture_limits):
                ablation_plot.make_effect_plot(
                    summary_df,
                    ablation_plot.EFFECT_SPECS[-1],
                    Path(tmp),
                    dpi=80,
                    formats=["png"],
                )

        self.assertEqual(len(observed_limits), 2)
        self.assertNotEqual(observed_limits[0], observed_limits[1])
        self.assertLess(observed_limits[0][0] - observed_limits[1][0], 10.0)

    def test_ablation_method_split_specs_cover_each_method_and_module(self):
        from experiments.component_ablation.plot_component_ablation import METHOD_EFFECT_SPECS

        stems = [spec["file_stem"] for spec in METHOD_EFFECT_SPECS]

        self.assertEqual(
            stems,
            [
                "extra_ablation_sggr_warmup_M_effect",
                "extra_ablation_sggc_warmup_M_effect",
                "extra_ablation_sggr_update_K_effect",
                "extra_ablation_sggc_update_K_effect",
                "extra_ablation_sggr_weight_w_effect",
                "extra_ablation_sggc_weight_w_effect",
            ],
        )

    def test_ablation_variants_have_print_friendly_hatches(self):
        from experiments.component_ablation.plot_component_ablation import VARIANT_HATCHES

        self.assertEqual(VARIANT_HATCHES["full"], "//")
        self.assertEqual(VARIANT_HATCHES["no_warmup"], "\\\\")
        self.assertEqual(VARIANT_HATCHES["single_mining"], "xx")
        self.assertEqual(VARIANT_HATCHES["uniform_weight"], "--")

    def test_significance_split_specs_cover_primary_comparisons(self):
        from experiments.statistical_significance.plot_significance_results import COMPARISON_EFFECT_SPECS

        stems = [spec["file_stem"] for spec in COMPARISON_EFFECT_SPECS]

        self.assertEqual(
            stems,
            [
                "significance_sggr_vs_grace",
                "significance_sggc_vs_gca",
            ],
        )

    def test_significance_delta_ylabel_uses_stable_score_wording(self):
        from experiments.statistical_significance.plot_significance_results import DELTA_YLABEL

        self.assertEqual(DELTA_YLABEL, "相对基线稳健性评分\nrobust_score变化")

    def test_sensitivity_labels_use_stable_score_and_observation_point(self):
        from experiments.hyperparameter_sensitivity.plot_sg_param_sensitivity import (
            ANCHOR_LABEL,
            ROBUST_SCORE_YLABEL,
        )

        self.assertEqual(ROBUST_SCORE_YLABEL, "稳健性评分robust_score（%）")
        self.assertEqual(ANCHOR_LABEL, "观测点")


if __name__ == "__main__":
    unittest.main()
