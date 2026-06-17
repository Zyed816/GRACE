import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from experiments.plotting_common import (
    DEFAULT_FIGURE_FORMATS,
    add_panel_label_below,
    apply_common_vector_settings,
    normalize_formats,
    panel_label,
    save_figure_formats,
)


METHOD_ORDER = ["grace", "gca", "sg-gr", "sg-gc"]
METHOD_LABELS = {
    "grace": "GRACE",
    "gca": "GCA",
    "sg-gr": "SG-GR",
    "sg-gc": "SG-GC",
}
METHOD_COLORS = {
    "grace": "#42567A",
    "gca": "#348380",
    "sg-gr": "#73B86E",
    "sg-gc": "#7A6E9F",
}
METHOD_HATCHES = {
    "grace": "//",
    "gca": "\\" * 2,
    "sg-gr": "xx",
    "sg-gc": "--",
}
METHOD_HATCH_COLOR = "#4A4A4A"
METHOD_HATCH_LINEWIDTH = 0.35
METHOD_BAR_EDGE_COLOR = "white"

DATASET_ORDER = ["Cora", "CiteSeer", "PubMed", "DBLP"]
OVERVIEW_DATASET_ORDER = DATASET_ORDER
OVERVIEW_METRIC_LABELS = ["robust_score", "Micro-F1", "Macro-F1"]
REQUESTED_METRIC_NAMES = ["robust_score", "F1Mi_mean", "F1Ma_mean"]
DATASET_SLUG_TO_LABEL = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "dblp": "DBLP",
}

RAW_STAGE_SET = {"baseline", "top_verify"}
NUMERIC_COLUMNS = [
    "candidate_rank",
    "run_idx",
    "F1Mi_mean",
    "F1Mi_std",
    "F1Ma_mean",
    "F1Ma_std",
    "robust_score",
    "delta_vs_grace",
]

METRIC_SPECS = {
    "robust_score": {
        "label": "robust_score",
        "err_col": "robust_score_std",
        "file_stem": "method_comparison_robust_score",
        "report_note": (
            "robust_score combines mean performance and stability, and is therefore "
            "well suited to summarize the overall effectiveness of each method."
        ),
    },
    "F1Mi_mean": {
        "label": "Micro-F1",
        "err_col": "F1Mi_std",
        "file_stem": "method_comparison_micro_f1",
        "report_note": (
            "Micro-F1 reflects the overall classification accuracy across all nodes "
            "and is the most direct indicator of downstream classification quality."
        ),
    },
    "F1Ma_mean": {
        "label": "Macro-F1",
        "err_col": "F1Ma_std",
        "file_stem": "method_comparison_macro_f1",
        "report_note": (
            "Macro-F1 gives equal weight to each class and is helpful for observing "
            "whether the method also improves balanced performance across categories."
        ),
    },
    "delta_vs_grace": {
        "label": "Delta vs GRACE",
        "err_col": "delta_vs_grace_std",
        "file_stem": "method_comparison_delta_vs_grace",
        "report_note": (
            "Delta vs GRACE is a supplementary metric that directly reveals the gain "
            "brought by each method relative to the vanilla contrastive baseline."
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped bar charts for method-comparison results. "
            "By default, the script selects the best verified candidate for each "
            "method on each dataset using mean robust_score."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Optional explicit full-pipeline CSV paths.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("results", "plots"),
        help="Directory used to save figures and summary files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=REQUESTED_METRIC_NAMES,
        choices=list(METRIC_SPECS.keys()),
        help="Metrics to visualize.",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Skip the 2x2 overview figure and only save the requested metric figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Raster output dpi.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=DEFAULT_FIGURE_FORMATS,
        help="Figure formats to save. Default: png pdf svg.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate each bar with its numeric value.",
    )
    return parser.parse_args()


def resolve_input_paths(repo_root, explicit_inputs):
    if explicit_inputs:
        paths = []
        for raw_path in explicit_inputs:
            path = Path(raw_path)
            if not path.is_absolute():
                path = repo_root / path
            paths.append(path.resolve())
        return paths

    return sorted((repo_root / "results").glob("*_full_pipeline_results.csv"))


def dataset_label_from_path(csv_path):
    stem = csv_path.stem
    suffix = "_full_pipeline_results"
    if stem.endswith(suffix):
        slug = stem[: -len(suffix)]
    else:
        slug = stem
    return DATASET_SLUG_TO_LABEL.get(slug.lower(), slug)


def load_raw_rows(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["stage"].isin(RAW_STAGE_SET)].copy()
    return df


def summarize_candidate(dataset, method, candidate_rank, group_df):
    return {
        "dataset": dataset,
        "method": method,
        "candidate_rank": int(candidate_rank),
        "n_runs": int(len(group_df)),
        "robust_score_mean": float(group_df["robust_score"].mean()),
        "robust_score_std": float(group_df["robust_score"].std(ddof=0)) if len(group_df) > 1 else 0.0,
        "F1Mi_mean": float(group_df["F1Mi_mean"].mean()),
        "F1Mi_std": float(group_df["F1Mi_mean"].std(ddof=0)) if len(group_df) > 1 else 0.0,
        "F1Ma_mean": float(group_df["F1Ma_mean"].mean()),
        "F1Ma_std": float(group_df["F1Ma_mean"].std(ddof=0)) if len(group_df) > 1 else 0.0,
    }


def select_best_method_rows(dataset, raw_df):
    selected_rows = []

    baseline_df = raw_df[(raw_df["method"] == "grace") & (raw_df["stage"] == "baseline")].copy()
    if baseline_df.empty:
        raise RuntimeError(f"Missing baseline rows for dataset={dataset}.")

    baseline_summary = summarize_candidate(dataset, "grace", 0, baseline_df)
    baseline_summary["selection_source"] = "baseline_runs"
    selected_rows.append(baseline_summary)

    for method in [m for m in METHOD_ORDER if m != "grace"]:
        method_df = raw_df[(raw_df["method"] == method) & (raw_df["stage"] == "top_verify")].copy()
        if method_df.empty:
            continue

        candidate_summaries = []
        for candidate_rank, group_df in method_df.groupby("candidate_rank", dropna=True):
            candidate_summaries.append(summarize_candidate(dataset, method, candidate_rank, group_df))

        if not candidate_summaries:
            continue

        candidate_summaries.sort(
            key=lambda item: (
                item["robust_score_mean"],
                item["F1Mi_mean"],
                -item["robust_score_std"],
            ),
            reverse=True,
        )
        best = candidate_summaries[0]
        best["selection_source"] = "best_verified_candidate"
        selected_rows.append(best)

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return selected_df

    baseline_row = selected_df[selected_df["method"] == "grace"].iloc[0]
    baseline_mean = float(baseline_row["robust_score_mean"])
    baseline_std = float(baseline_row["robust_score_std"])

    selected_df["delta_vs_grace"] = selected_df["robust_score_mean"] - baseline_mean
    selected_df["delta_vs_grace_std"] = np.sqrt(selected_df["robust_score_std"] ** 2 + baseline_std ** 2)
    selected_df.loc[selected_df["method"] == "grace", "delta_vs_grace"] = 0.0
    selected_df.loc[selected_df["method"] == "grace", "delta_vs_grace_std"] = baseline_std

    return selected_df


def build_plot_summary(input_paths):
    summary_frames = []

    for csv_path in input_paths:
        dataset = dataset_label_from_path(csv_path)
        raw_df = load_raw_rows(csv_path)
        if raw_df.empty:
            continue
        summary_frames.append(select_best_method_rows(dataset, raw_df))

    if not summary_frames:
        raise RuntimeError("No valid method-comparison rows were found in the provided CSV files.")

    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df["dataset"] = pd.Categorical(summary_df["dataset"], categories=DATASET_ORDER, ordered=True)
    summary_df["method"] = pd.Categorical(summary_df["method"], categories=METHOD_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["dataset", "method"]).reset_index(drop=True)
    return summary_df


def configure_plot_style():
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#303030",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    apply_common_vector_settings(plt)
    if "hatch.linewidth" in plt.rcParams:
        plt.rcParams["hatch.linewidth"] = METHOD_HATCH_LINEWIDTH


def metric_value_columns(metric_name):
    if metric_name == "robust_score":
        return "robust_score_mean", "robust_score_std"
    if metric_name == "delta_vs_grace":
        return "delta_vs_grace", "delta_vs_grace_std"
    return metric_name, METRIC_SPECS[metric_name]["err_col"]


def to_plot_units(values):
    return np.asarray(values, dtype=float) * 100.0


def metric_axis_label(metric_name):
    if metric_name == "delta_vs_grace":
        return "相对GRACE变化（百分点）"
    return f"{METRIC_SPECS[metric_name]['label']}（%）"


def compute_axis_limits(value_matrix, err_matrix, metric_name):
    values = np.asarray(value_matrix, dtype=float).reshape(-1)
    errors = np.asarray(err_matrix, dtype=float).reshape(-1)
    valid = ~(np.isnan(values) | np.isnan(errors))
    values = values[valid]
    errors = errors[valid]

    if values.size == 0:
        return 0.0, 1.0

    low = float(np.min(values - errors))
    high = float(np.max(values + errors))
    span = max(high - low, 1e-6)

    if metric_name == "delta_vs_grace":
        lower = min(-0.2, low - span * 0.18)
        upper = max(0.2, high + span * 0.22)
        return lower, upper

    upper = high + max(span * 0.18, 1.0)
    return 0.0, upper


def annotate_bars(ax, bars, values):
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.012
    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        x = bar.get_x() + bar.get_width() * 0.5
        y = value
        ax.text(
            x,
            y + offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
        )


def method_legend_handle(method):
    return Patch(
        facecolor=METHOD_COLORS[method],
        edgecolor=METHOD_HATCH_COLOR,
        linewidth=0.45,
        hatch=METHOD_HATCHES[method],
        label=METHOD_LABELS[method],
    )


def add_method_hatch_overlay(ax, positions, values, width, method, zorder):
    ax.bar(
        positions,
        values,
        width=width,
        facecolor="none",
        edgecolor=METHOD_HATCH_COLOR,
        linewidth=0.0,
        hatch=METHOD_HATCHES[method],
        label="_nolegend_",
        zorder=zorder,
    )


def make_metric_plot(summary_df, metric_name, out_dir, dpi, annotate, formats):
    value_col, err_col = metric_value_columns(metric_name)
    metric_spec = METRIC_SPECS[metric_name]

    datasets = [d for d in DATASET_ORDER if d in set(summary_df["dataset"].astype(str))]
    x = np.arange(len(datasets), dtype=float)
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    all_values = []
    all_errors = []
    legend_handles = []
    legend_labels = []

    for idx, method in enumerate(METHOD_ORDER):
        subset = summary_df[summary_df["method"].astype(str) == method].copy()
        subset = subset.set_index(subset["dataset"].astype(str)).reindex(datasets)

        values = to_plot_units(subset[value_col].to_numpy(dtype=float))
        errors = to_plot_units(subset[err_col].fillna(0.0).to_numpy(dtype=float)) if err_col in subset else None
        positions = x + (idx - 1.5) * width

        bars = ax.bar(
            positions,
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor=METHOD_BAR_EDGE_COLOR,
            linewidth=0.45,
            yerr=errors,
            capsize=2.2 if errors is not None else 0,
            error_kw={
                "elinewidth": 0.65,
                "capthick": 0.65,
                "ecolor": "#3b3b3b",
            },
        )
        add_method_hatch_overlay(ax, positions, values, width, method, zorder=bars[0].get_zorder() + 0.1)

        if annotate:
            annotate_bars(ax, bars, values)

        all_values.append(values)
        all_errors.append(errors if errors is not None else np.zeros_like(values))
        legend_handles.append(method_legend_handle(method))
        legend_labels.append(METHOD_LABELS[method])

    lower, upper = compute_axis_limits(np.array(all_values), np.array(all_errors), metric_name)
    ax.set_ylim(lower, upper)
    if metric_name == "delta_vs_grace":
        ax.axhline(0.0, color="#303030", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(metric_axis_label(metric_name))
    ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors="#303030", pad=2)
    ax.margins(x=0.04)
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(legend_handles),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    saved_paths = save_figure_formats(fig, out_dir / metric_spec["file_stem"], formats, dpi=dpi)
    plt.close(fig)

    return saved_paths


def make_overview_plot(summary_df, out_dir, dpi, formats):
    overview_metrics = ["robust_score", "F1Mi_mean", "F1Ma_mean"]
    datasets = [d for d in OVERVIEW_DATASET_ORDER if d in set(summary_df["dataset"].astype(str))]
    metric_labels = OVERVIEW_METRIC_LABELS
    x = np.arange(len(overview_metrics), dtype=float)
    width = 0.15

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.25), sharex=False)
    axes = axes.ravel()
    legend_handles = []
    legend_labels = []

    for panel_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        all_values = []
        all_errors = []

        for idx, method in enumerate(METHOD_ORDER):
            method_df = summary_df[
                (summary_df["dataset"].astype(str) == dataset)
                & (summary_df["method"].astype(str) == method)
            ]
            values = []
            errors = []
            for metric_name in overview_metrics:
                value_col, err_col = metric_value_columns(metric_name)
                if method_df.empty:
                    values.append(np.nan)
                    errors.append(np.nan)
                else:
                    values.append(float(method_df.iloc[0][value_col]) * 100.0)
                    errors.append(float(method_df.iloc[0][err_col]) * 100.0)
            values = np.asarray(values, dtype=float)
            errors = np.asarray(errors, dtype=float)
            positions = x + (idx - 1.5) * width

            bars = ax.bar(
                positions,
                values,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor=METHOD_BAR_EDGE_COLOR,
                linewidth=0.45,
                alpha=0.94,
                yerr=errors,
                capsize=2.2,
                error_kw={
                    "elinewidth": 0.65,
                    "capthick": 0.65,
                    "ecolor": "#3b3b3b",
                },
                label=METHOD_LABELS[method],
            )
            add_method_hatch_overlay(ax, positions, values, width, method, zorder=bars[0].get_zorder() + 0.1)
            if len(legend_handles) < len(METHOD_ORDER):
                legend_handles.append(method_legend_handle(method))
                legend_labels.append(METHOD_LABELS[method])

            all_values.append(values)
            all_errors.append(errors)

        valid_values = np.asarray(all_values, dtype=float)
        valid_errors = np.asarray(all_errors, dtype=float)
        lower, upper = compute_axis_limits(valid_values, valid_errors, "F1Mi_mean")
        ax.set_ylim(lower, upper)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        add_panel_label_below(ax, panel_label(panel_idx, dataset), y=-0.27, fontsize=9.5)
        ax.set_ylabel("分数（%）")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.margins(x=0.04)
        ax.tick_params(axis="both", colors="#303030", pad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(datasets) :]:
        ax.set_visible(False)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.992),
            ncol=len(legend_handles),
            frameon=False,
            handlelength=1.6,
            columnspacing=1.6,
        )

    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.90], w_pad=1.3, h_pad=3.0)

    saved_paths = save_figure_formats(
        fig,
        out_dir / "method_comparison_overview",
        formats,
        dpi=dpi,
        pad_inches=0.04,
    )
    plt.close(fig)
    return saved_paths


def build_report(summary_df, generated_files):
    lines = [
        "# Method Comparison Visualization Notes",
        "",
        "## Metric Choice",
        f"- `robust_score`: {METRIC_SPECS['robust_score']['report_note']}",
        f"- `F1Mi_mean`: {METRIC_SPECS['F1Mi_mean']['report_note']}",
        f"- `F1Ma_mean`: {METRIC_SPECS['F1Ma_mean']['report_note']}",
        f"- `delta_vs_grace`: {METRIC_SPECS['delta_vs_grace']['report_note']}",
        "",
        "## Statistical Rule",
        "- For each dataset and each method, the figure uses the best verified candidate selected by the highest mean robust score.",
        "- The error bars are computed from repeated verification runs of the selected candidate.",
        "- GRACE uses its baseline repeated runs directly.",
        "",
        "## Winner Snapshot",
    ]

    for dataset in DATASET_ORDER:
        dataset_df = summary_df[summary_df["dataset"].astype(str) == dataset]
        if dataset_df.empty:
            continue

        best_robust = dataset_df.sort_values("robust_score_mean", ascending=False).iloc[0]
        best_micro = dataset_df.sort_values("F1Mi_mean", ascending=False).iloc[0]
        best_macro = dataset_df.sort_values("F1Ma_mean", ascending=False).iloc[0]
        lines.append(
            "- "
            f"{dataset}: best robust_score = {METHOD_LABELS[str(best_robust['method'])]} "
            f"({best_robust['robust_score_mean']:.4f}); "
            f"best Micro-F1 = {METHOD_LABELS[str(best_micro['method'])]} "
            f"({best_micro['F1Mi_mean']:.4f}); "
            f"best Macro-F1 = {METHOD_LABELS[str(best_macro['method'])]} "
            f"({best_macro['F1Ma_mean']:.4f})."
        )

    lines.extend(
        [
            "",
            "## Generated Files",
        ]
    )
    for path in generated_files:
        lines.append(f"- `{path.as_posix()}`")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    args.formats = normalize_formats(args.formats)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    input_paths = resolve_input_paths(repo_root, args.inputs)
    if not input_paths:
        raise RuntimeError("No method-comparison CSV files were found.")

    summary_df = build_plot_summary(input_paths)

    generated_files = []
    if not args.skip_overview:
        generated_files.extend(make_overview_plot(summary_df, out_dir, args.dpi, args.formats))

    for metric_name in args.metrics:
        generated_files.extend(
            make_metric_plot(
            summary_df=summary_df,
            metric_name=metric_name,
            out_dir=out_dir,
            dpi=args.dpi,
            annotate=args.annotate,
            formats=args.formats,
            )
        )

    for path in generated_files:
        print(f"[plot] saved figure/file: {path}")


if __name__ == "__main__":
    main()
