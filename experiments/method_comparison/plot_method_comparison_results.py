import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["grace", "gca", "ifl-gr", "ifl-gc"]
METHOD_LABELS = {
    "grace": "GRACE",
    "gca": "GCA",
    "ifl-gr": "IFL-GR",
    "ifl-gc": "IFL-GC",
}
METHOD_COLORS = {
    "grace": "#6B7280",
    "gca": "#2563EB",
    "ifl-gr": "#D97706",
    "ifl-gc": "#059669",
}

DATASET_ORDER = ["Cora", "CiteSeer", "PubMed", "DBLP"]
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
        "label": "Robust Score",
        "err_col": "robust_score_std",
        "file_stem": "method_comparison_robust_score",
        "report_note": (
            "Robust Score combines mean performance and stability, and is therefore "
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
        default=["robust_score", "F1Mi_mean", "F1Ma_mean", "delta_vs_grace"],
        choices=list(METRIC_SPECS.keys()),
        help="Metrics to visualize.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Raster output dpi.",
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
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STSong"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#111827",
            "axes.linewidth": 1.1,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def metric_value_columns(metric_name):
    if metric_name == "robust_score":
        return "robust_score_mean", "robust_score_std"
    if metric_name == "delta_vs_grace":
        return "delta_vs_grace", "delta_vs_grace_std"
    return metric_name, METRIC_SPECS[metric_name]["err_col"]


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
        lower = min(-0.002, low - span * 0.18)
        upper = max(0.002, high + span * 0.22)
        return lower, upper

    lower = max(0.0, low - span * 0.25)
    upper = min(1.0, high + span * 0.22)

    if upper - lower < 0.04:
        pad = (0.04 - (upper - lower)) * 0.5
        lower = max(0.0, lower - pad)
        upper = min(1.0, upper + pad)

    return lower, upper


def annotate_bars(ax, bars, values, errors, metric_name):
    for bar, value, error in zip(bars, values, errors):
        if np.isnan(value):
            continue
        x = bar.get_x() + bar.get_width() * 0.5
        y = value + (error if not np.isnan(error) else 0.0)
        offset = 0.006 if metric_name != "delta_vs_grace" else 0.0018
        ax.text(
            x,
            y + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#374151",
        )


def make_metric_plot(summary_df, metric_name, out_dir, dpi, annotate):
    value_col, err_col = metric_value_columns(metric_name)
    metric_spec = METRIC_SPECS[metric_name]

    datasets = [d for d in DATASET_ORDER if d in set(summary_df["dataset"].astype(str))]
    x = np.arange(len(datasets), dtype=float)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.8, 6.0))

    all_values = []
    all_errors = []
    legend_handles = []
    legend_labels = []

    for idx, method in enumerate(METHOD_ORDER):
        subset = summary_df[summary_df["method"].astype(str) == method].copy()
        subset = subset.set_index(subset["dataset"].astype(str)).reindex(datasets)

        values = subset[value_col].to_numpy(dtype=float)
        errors = subset[err_col].to_numpy(dtype=float)
        positions = x + (idx - 1.5) * width

        bars = ax.bar(
            positions,
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="#1F2937",
            linewidth=0.8,
            zorder=3,
        )

        ax.errorbar(
            positions,
            values,
            yerr=errors,
            fmt="none",
            ecolor="#111827",
            elinewidth=0.9,
            capsize=3.2,
            capthick=0.9,
            zorder=4,
        )

        if annotate:
            annotate_bars(ax, bars, values, errors, metric_name)

        all_values.append(values)
        all_errors.append(errors)
        legend_handles.append(bars[0])
        legend_labels.append(METHOD_LABELS[method])

    lower, upper = compute_axis_limits(np.array(all_values), np.array(all_errors), metric_name)
    ax.set_ylim(lower, upper)
    if metric_name == "delta_vs_grace":
        ax.axhline(0.0, color="#6B7280", linewidth=1.0, linestyle="--", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric_spec["label"])
    ax.set_title(f"Performance Comparison by {metric_spec['label']}")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.22, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111827")
    ax.spines["bottom"].set_color("#111827")

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.6,
    )

    png_path = out_dir / f"{metric_spec['file_stem']}.png"
    pdf_path = out_dir / f"{metric_spec['file_stem']}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


def make_overview_plot(summary_df, out_dir, dpi):
    overview_metrics = ["robust_score", "F1Mi_mean", "F1Ma_mean"]
    datasets = [d for d in DATASET_ORDER if d in set(summary_df["dataset"].astype(str))]
    x = np.arange(len(datasets), dtype=float)
    width = 0.18

    fig, axes = plt.subplots(1, len(overview_metrics), figsize=(18.0, 5.6), sharex=False)
    if len(overview_metrics) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    for ax, metric_name in zip(axes, overview_metrics):
        value_col, err_col = metric_value_columns(metric_name)
        metric_spec = METRIC_SPECS[metric_name]
        all_values = []
        all_errors = []

        for idx, method in enumerate(METHOD_ORDER):
            subset = summary_df[summary_df["method"].astype(str) == method].copy()
            subset = subset.set_index(subset["dataset"].astype(str)).reindex(datasets)

            values = subset[value_col].to_numpy(dtype=float)
            errors = subset[err_col].to_numpy(dtype=float)
            positions = x + (idx - 1.5) * width

            bars = ax.bar(
                positions,
                values,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor="#1F2937",
                linewidth=0.8,
                zorder=3,
            )
            ax.errorbar(
                positions,
                values,
                yerr=errors,
                fmt="none",
                ecolor="#111827",
                elinewidth=0.9,
                capsize=2.8,
                capthick=0.9,
                zorder=4,
            )

            if not legend_handles:
                legend_handles.append(bars[0])
                legend_labels.append(METHOD_LABELS[method])
            elif len(legend_handles) < len(METHOD_ORDER):
                legend_handles.append(bars[0])
                legend_labels.append(METHOD_LABELS[method])

            all_values.append(values)
            all_errors.append(errors)

        lower, upper = compute_axis_limits(np.array(all_values), np.array(all_errors), metric_name)
        ax.set_ylim(lower, upper)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_title(metric_spec["label"])
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.22, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Dataset")
        if metric_name == "robust_score":
            ax.set_ylabel("Score")

    fig.suptitle("Method Comparison Overview", fontsize=15, fontweight="bold", y=1.03)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
    )

    png_path = out_dir / "method_comparison_overview.png"
    pdf_path = out_dir / "method_comparison_overview.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


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
            f"{dataset}: best Robust Score = {METHOD_LABELS[str(best_robust['method'])]} "
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
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    input_paths = resolve_input_paths(repo_root, args.inputs)
    if not input_paths:
        raise RuntimeError("No method-comparison CSV files were found.")

    summary_df = build_plot_summary(input_paths)

    summary_csv = out_dir / "method_comparison_best_verified_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    generated_files = [summary_csv]
    overview_png, overview_pdf = make_overview_plot(summary_df, out_dir, args.dpi)
    generated_files.extend([overview_png, overview_pdf])

    for metric_name in args.metrics:
        png_path, pdf_path = make_metric_plot(
            summary_df=summary_df,
            metric_name=metric_name,
            out_dir=out_dir,
            dpi=args.dpi,
            annotate=args.annotate,
        )
        generated_files.extend([png_path, pdf_path])

    report_path = out_dir / "method_comparison_visualization_notes.md"
    report_path.write_text(build_report(summary_df, generated_files), encoding="utf-8")
    generated_files.append(report_path)

    print(f"[plot] saved summary: {summary_csv}")
    print(f"[plot] saved report: {report_path}")
    for path in generated_files[1:]:
        print(f"[plot] saved figure/file: {path}")


if __name__ == "__main__":
    main()
