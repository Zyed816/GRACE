import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.plotting_common import (
    DEFAULT_FIGURE_FORMATS,
    add_panel_label_below,
    apply_common_vector_settings,
    normalize_formats,
    panel_label,
    save_figure_formats,
)


from experiments.statistical_significance.run_significance_experiment import DATASET_CHOICES, METHOD_CHOICES


METHOD_LABELS = {
    "grace": "GRACE",
    "gca": "GCA",
    "ifl-gr": "SG-GR",
    "ifl-gc": "SG-GC",
}
METHOD_COLORS = {
    "grace": "#42567A",
    "gca": "#348380",
    "ifl-gr": "#73B86E",
    "ifl-gc": "#7A6E9F",
}
METRIC_LABELS = {
    "robust_score": "robust_score",
    "F1Mi_mean": "Micro-F1",
    "F1Ma_mean": "Macro-F1",
}
DELTA_YLABEL = "相对基线稳健性评分\nrobust_score变化"
PRIMARY_COMPARISONS = [
    ("grace", "ifl-gr", "SG-GR\nvs GRACE"),
    ("gca", "ifl-gc", "SG-GC\nvs GCA"),
]
COMPARISON_EFFECT_SPECS = [
    {
        "baseline": "grace",
        "target": "ifl-gr",
        "label": "SG-GR VS GRACE",
        "file_stem": "significance_sggr_vs_grace",
    },
    {
        "baseline": "gca",
        "target": "ifl-gc",
        "label": "SG-GC VS GCA",
        "file_stem": "significance_sggc_vs_gca",
    },
]


def resolve_input_paths(repo_root, explicit_inputs):
    if explicit_inputs:
        paths = []
        for raw in explicit_inputs:
            path = Path(raw)
            if not path.is_absolute():
                path = repo_root / path
            paths.append(path.resolve())
        return paths
    return sorted((repo_root / "results").glob("significance_*_results.csv"))


def load_rows(input_paths):
    frames = []
    for path in input_paths:
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No significance rows were found.")
    return pd.concat(frames, ignore_index=True)


def configure_plot_style():
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.edgecolor": "#303030",
            "axes.linewidth": 0.7,
            "legend.fontsize": 8.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    apply_common_vector_settings(plt)


def make_mean_std_plot(df, out_dir, dpi, formats):
    run_df = df[df["stage"] == "run"].copy()
    if run_df.empty:
        raise RuntimeError("No run rows available for mean/std plot.")

    for col in ["robust_score", "F1Mi_mean", "F1Ma_mean"]:
        run_df[col] = pd.to_numeric(run_df[col], errors="coerce")

    metrics = ["robust_score", "F1Mi_mean", "F1Ma_mean"]
    datasets = [item for item in DATASET_CHOICES if item in set(run_df["dataset"].astype(str))]
    x = np.arange(len(datasets), dtype=float)
    width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharex=False)
    legend_handles = []
    legend_labels = []
    for ax, metric in zip(axes, metrics):
        for idx, method in enumerate(METHOD_CHOICES):
            values = []
            errors = []
            for dataset in datasets:
                subset = run_df[
                    (run_df["dataset"].astype(str) == dataset)
                    & (run_df["method"].astype(str) == method)
                ][metric].dropna()
                values.append(float(subset.mean()) * 100.0 if not subset.empty else np.nan)
                errors.append(float(subset.std(ddof=0)) * 100.0 if len(subset) > 1 else 0.0)

            bars = ax.bar(
                x + (idx - 1.5) * width,
                values,
                yerr=errors,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.4,
                capsize=2.0,
                label=METHOD_LABELS[method],
            )
            if len(legend_handles) < len(METHOD_CHOICES):
                legend_handles.append(bars[0])
                legend_labels.append(METHOD_LABELS[method])

        ax.text(
            0.5,
            1.03,
            METRIC_LABELS[metric],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_handles), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    saved_paths = save_figure_formats(fig, out_dir / "significance_mean_std", formats, dpi=dpi)
    plt.close(fig)
    return saved_paths


def make_delta_plot(df, out_dir, dpi, formats):
    run_df = df[df["stage"] == "run"].copy()
    if run_df.empty:
        raise RuntimeError("No run rows available for delta plot.")
    run_df["robust_score"] = pd.to_numeric(run_df["robust_score"], errors="coerce")

    datasets = [item for item in DATASET_CHOICES if item in set(run_df["dataset"].astype(str))]
    comparison_labels = [label for _base, _target, label in PRIMARY_COMPARISONS]
    x = np.arange(len(comparison_labels), dtype=float)

    panel_stats = {}
    y_values = [0.0]
    for dataset in datasets:
        dataset_df = run_df[run_df["dataset"].astype(str) == dataset]
        means = []
        errors = []
        significant_flags = []
        for base, target, _label in PRIMARY_COMPARISONS:
            base_df = dataset_df[dataset_df["method"].astype(str) == base].set_index("run_idx")
            target_df = dataset_df[dataset_df["method"].astype(str) == target].set_index("run_idx")
            shared = sorted(set(base_df.index) & set(target_df.index), key=lambda item: int(float(item)))
            deltas = []
            for run_idx in shared:
                b = pd.to_numeric(pd.Series([base_df.loc[run_idx, "robust_score"]]), errors="coerce").iloc[0]
                t = pd.to_numeric(pd.Series([target_df.loc[run_idx, "robust_score"]]), errors="coerce").iloc[0]
                if pd.notna(b) and pd.notna(t):
                    deltas.append((float(t) - float(b)) * 100.0)

            mean = float(np.mean(deltas)) if deltas else np.nan
            error = float(np.std(deltas, ddof=0)) if len(deltas) > 1 else 0.0
            means.append(mean)
            errors.append(error)
            if not np.isnan(mean):
                y_values.extend([mean - error, mean + error])

            test_df = df[
                (df["stage"] == "test")
                & (df["dataset"].astype(str) == dataset)
                & (df["metric"].astype(str) == "robust_score")
                & (df["baseline_method"].astype(str) == base)
                & (df["target_method"].astype(str) == target)
            ]
            is_significant = (
                not test_df.empty
                and str(test_df.iloc[0].get("significant", "")).lower() == "true"
                and mean > 0
            )
            significant_flags.append(is_significant)
        panel_stats[dataset] = {
            "means": means,
            "errors": errors,
            "significant": significant_flags,
        }

    finite_y = [value for value in y_values if np.isfinite(value)]
    y_low = min(finite_y)
    y_high = max(finite_y)
    y_span = max(y_high - y_low, 1.0)
    y_lim = (y_low - y_span * 0.22, y_high + y_span * 0.28)
    star_offset = y_span * 0.06

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.2), sharey=True)
    axes = axes.ravel()
    for ax_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        stats = panel_stats[dataset]
        means = stats["means"]
        errors = stats["errors"]
        colors = [METHOD_COLORS[target] for _base, target, _label in PRIMARY_COMPARISONS]
        bars = ax.bar(
            x,
            means,
            yerr=errors,
            color=colors,
            edgecolor="white",
            linewidth=0.45,
            capsize=2.2,
        )
        ax.axhline(0.0, color="black", linewidth=0.75)
        add_panel_label_below(ax, panel_label(ax_idx, dataset), y=-0.25, fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_labels, rotation=0, ha="center")
        ax.set_ylabel(DELTA_YLABEL if ax_idx % 2 == 0 else "")
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(*y_lim)

        for bar_idx, is_significant in enumerate(stats["significant"]):
            if not is_significant:
                continue
            height = means[bar_idx]
            if np.isnan(height):
                continue
            y_pos = height + errors[bar_idx] + star_offset
            ax.text(bar_idx, y_pos, "*", ha="center", va="center", fontsize=12)

    for ax in axes[len(datasets) :]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0.04, 0.04, 1.0, 1.0], w_pad=1.2, h_pad=2.5)
    saved_paths = save_figure_formats(fig, out_dir / "significance_paired_delta", formats, dpi=dpi)
    plt.close(fig)
    return saved_paths


def comparison_delta_stats(df, spec):
    run_df = df[df["stage"] == "run"].copy()
    if run_df.empty:
        raise RuntimeError("No run rows available for comparison delta plot.")
    run_df["robust_score"] = pd.to_numeric(run_df["robust_score"], errors="coerce")

    datasets = [item for item in DATASET_CHOICES if item in set(run_df["dataset"].astype(str))]
    means = []
    errors = []
    significant_flags = []

    for dataset in datasets:
        dataset_df = run_df[run_df["dataset"].astype(str) == dataset]
        base_df = dataset_df[dataset_df["method"].astype(str) == spec["baseline"]].set_index("run_idx")
        target_df = dataset_df[dataset_df["method"].astype(str) == spec["target"]].set_index("run_idx")
        shared = sorted(set(base_df.index) & set(target_df.index), key=lambda item: int(float(item)))
        deltas = []
        for run_idx in shared:
            base = pd.to_numeric(pd.Series([base_df.loc[run_idx, "robust_score"]]), errors="coerce").iloc[0]
            target = pd.to_numeric(pd.Series([target_df.loc[run_idx, "robust_score"]]), errors="coerce").iloc[0]
            if pd.notna(base) and pd.notna(target):
                deltas.append((float(target) - float(base)) * 100.0)

        mean = float(np.mean(deltas)) if deltas else np.nan
        error = float(np.std(deltas, ddof=0)) if len(deltas) > 1 else 0.0
        means.append(mean)
        errors.append(error)

        test_df = df[
            (df["stage"] == "test")
            & (df["dataset"].astype(str) == dataset)
            & (df["metric"].astype(str) == "robust_score")
            & (df["baseline_method"].astype(str) == spec["baseline"])
            & (df["target_method"].astype(str) == spec["target"])
        ]
        significant_flags.append(
            not test_df.empty
            and str(test_df.iloc[0].get("significant", "")).lower() == "true"
            and mean > 0
        )

    return datasets, means, errors, significant_flags


def make_comparison_delta_plot(df, spec, out_dir, dpi, formats):
    datasets, means, errors, significant_flags = comparison_delta_stats(df, spec)
    if not datasets:
        raise RuntimeError(f"No datasets available for {spec['label']} comparison plot.")

    finite_values = [0.0]
    for mean, error in zip(means, errors):
        if np.isfinite(mean):
            finite_values.extend([mean - error, mean + error])
    y_low = min(finite_values)
    y_high = max(finite_values)
    y_span = max(y_high - y_low, 1.0)
    y_lim = (y_low - y_span * 0.22, y_high + y_span * 0.28)
    star_offset = y_span * 0.06

    x = np.arange(len(datasets), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(
        x,
        means,
        yerr=errors,
        color=METHOD_COLORS[spec["target"]],
        edgecolor="white",
        linewidth=0.45,
        capsize=2.4,
    )
    ax.axhline(0.0, color="black", linewidth=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("相对基线稳健性评分robust_score变化")
    ax.set_ylim(*y_lim)
    ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors="#303030", pad=2)

    for bar, is_significant, mean, error in zip(bars, significant_flags, means, errors):
        if not is_significant or not np.isfinite(mean):
            continue
        x_pos = bar.get_x() + bar.get_width() * 0.5
        y_pos = mean + error + star_offset
        ax.text(x_pos, y_pos, "*", ha="center", va="center", fontsize=12)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    saved_paths = save_figure_formats(fig, out_dir / spec["file_stem"], formats, dpi=dpi)
    plt.close(fig)
    return saved_paths


def make_comparison_delta_plots(df, out_dir, dpi, formats):
    generated = []
    for spec in COMPARISON_EFFECT_SPECS:
        generated.extend(make_comparison_delta_plot(df, spec, out_dir, dpi, formats))
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Plot statistical significance experiment summaries.")
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default=os.path.join("results", "plots"))
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=DEFAULT_FIGURE_FORMATS,
        help="Figure formats to save. Default: png pdf svg.",
    )
    parser.add_argument(
        "--split-primary-comparisons",
        action="store_true",
        help="Save one robust_score delta figure for each primary significance comparison.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.formats = normalize_formats(args.formats)
    repo_root = Path(PROJECT_ROOT)
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    input_paths = resolve_input_paths(repo_root, args.inputs)
    if not input_paths:
        raise RuntimeError("No significance CSV files were found.")
    df = load_rows(input_paths)

    generated = []
    if args.split_primary_comparisons:
        generated.extend(make_comparison_delta_plots(df, out_dir, args.dpi, args.formats))
    else:
        generated.extend(make_mean_std_plot(df, out_dir, args.dpi, args.formats))
        generated.extend(make_delta_plot(df, out_dir, args.dpi, args.formats))
    for path in generated:
        print(f"[significance-plot] saved: {path}")


if __name__ == "__main__":
    main()
