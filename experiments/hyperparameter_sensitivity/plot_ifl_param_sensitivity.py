import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METHOD_FILE_SLUG = {
    "ifl-gr": "iflgr",
    "ifl-gc": "iflgc",
}

METHOD_LABELS = {
    "ifl-gr": "IFL-GR",
    "ifl-gc": "IFL-GC",
}

METHOD_COLORS = {
    "ifl-gr": "#1f77b4",
    "ifl-gc": "#d62728",
}

PARAM_ORDER = ["t_s", "M", "K"]
PARAM_LABELS = {
    "t_s": r"$t_s$",
    "M": "M",
    "K": "K",
}

METRIC_SPECS = {
    "F1Mi_mean": {
        "label": "F1Mi",
        "err_col": "F1Mi_std",
    },
    "robust_score": {
        "label": "Robust Score",
        "err_col": "robust_score_std",
    },
}

NUMERIC_COLUMNS = [
    "anchor_value",
    "anchor_grid_robust",
    "sweep_value",
    "num_runs",
    "F1Mi_mean",
    "F1Mi_std",
    "F1Ma_mean",
    "F1Ma_std",
    "within_run_F1Mi_std_mean",
    "within_run_F1Ma_std_mean",
    "robust_score",
    "robust_score_std",
    "trace_ts_mean",
    "trace_ts_last",
    "trace_mined_pairs_mean",
    "trace_mined_pairs_last",
    "trace_avg_pairs_mean",
    "trace_avg_pairs_last",
    "delta_vs_anchor",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot IFL-GR / IFL-GC sensitivity-analysis CSV files and generate a short report."
    )
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--methods", nargs="+", default=["ifl-gr", "ifl-gc"], choices=list(METHOD_FILE_SLUG))
    parser.add_argument("--inputs", nargs="+", default=None, help="Optional explicit CSV paths.")
    parser.add_argument("--out_dir", type=str, default=os.path.join("results", "plots"))
    return parser.parse_args()


def resolve_input_paths(grace_dir, dataset, methods, explicit_inputs):
    if explicit_inputs:
        return [os.path.join(grace_dir, path) if not os.path.isabs(path) else path for path in explicit_inputs]

    paths = []
    dataset_slug = dataset.lower()
    for method in methods:
        method_slug = METHOD_FILE_SLUG[method]
        paths.append(
            os.path.join(grace_dir, "results", f"sensitivity_{method_slug}_{dataset_slug}_results.csv")
        )
    return paths


def load_summary_rows(csv_path):
    if not os.path.exists(csv_path):
        return {
            "path": csv_path,
            "status": "missing",
            "summary_rows": 0,
            "all_rows": 0,
        }, pd.DataFrame()

    df = pd.read_csv(csv_path)
    if df.empty:
        return {
            "path": csv_path,
            "status": "empty",
            "summary_rows": 0,
            "all_rows": 0,
        }, df

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_anchor" in df.columns:
        df["is_anchor"] = df["is_anchor"].astype(str).str.lower() == "true"

    summary_df = df[df["stage"] == "summary"].copy() if "stage" in df.columns else pd.DataFrame(columns=df.columns)
    status = "ok" if not summary_df.empty else "no_summary"
    return {
        "path": csv_path,
        "status": status,
        "summary_rows": int(len(summary_df)),
        "all_rows": int(len(df)),
    }, summary_df


def format_value(param_name, value):
    if pd.isna(value):
        return "NA"
    if param_name == "t_s":
        return f"{float(value):.4f}"
    return str(int(round(float(value))))


def make_plot(dataset, methods, summary_df, out_path):
    fig, axes = plt.subplots(
        nrows=len(METRIC_SPECS),
        ncols=len(PARAM_ORDER),
        figsize=(15, 7),
        squeeze=False,
    )

    legend_map = {}

    has_required_columns = {"paper_param", "method", "sweep_value", "is_anchor"}.issubset(summary_df.columns)

    for row_idx, (metric_col, metric_spec) in enumerate(METRIC_SPECS.items()):
        for col_idx, param_name in enumerate(PARAM_ORDER):
            ax = axes[row_idx][col_idx]
            ax.set_title(PARAM_LABELS[param_name], fontsize=12, fontweight="bold")
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)

            has_any_data = False

            if not has_required_columns or metric_col not in summary_df.columns:
                ax.text(
                    0.5,
                    0.5,
                    "No summary data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="#666666",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == len(METRIC_SPECS) - 1:
                    ax.set_xlabel(PARAM_LABELS[param_name])
                if col_idx == 0:
                    ax.set_ylabel(metric_spec["label"])
                continue

            for method in methods:
                subset = summary_df[
                    (summary_df["paper_param"] == param_name) & (summary_df["method"] == method)
                ].copy()
                if subset.empty:
                    continue

                subset = subset.sort_values("sweep_value")
                subset = subset.dropna(subset=["sweep_value", metric_col])
                if subset.empty:
                    continue

                has_any_data = True
                color = METHOD_COLORS[method]
                label = METHOD_LABELS[method]
                legend_map[label] = color
                err_col = metric_spec["err_col"]
                yerr = subset[err_col] if err_col in subset.columns else None
                if yerr is not None:
                    yerr = yerr.fillna(0.0)

                if len(subset) > 1:
                    ax.errorbar(
                        subset["sweep_value"],
                        subset[metric_col],
                        yerr=yerr,
                        color=color,
                        marker="o",
                        linewidth=1.8,
                        capsize=3,
                        label=label,
                    )
                else:
                    ax.scatter(
                        subset["sweep_value"],
                        subset[metric_col],
                        color=color,
                        s=45,
                        label=label,
                        zorder=3,
                    )

                anchor_subset = subset[subset["is_anchor"]]
                if not anchor_subset.empty:
                    ax.scatter(
                        anchor_subset["sweep_value"],
                        anchor_subset[metric_col],
                        color=color,
                        marker="*",
                        s=180,
                        edgecolors="black",
                        linewidths=0.7,
                        zorder=4,
                    )

            if not has_any_data:
                ax.text(
                    0.5,
                    0.5,
                    "No summary data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="#666666",
                )
                ax.set_xticks([])
                ax.set_yticks([])

            if row_idx == len(METRIC_SPECS) - 1:
                ax.set_xlabel(PARAM_LABELS[param_name])
            if col_idx == 0:
                ax.set_ylabel(metric_spec["label"])

    if legend_map:
        handles = [
            plt.Line2D([], [], color=color, marker="o", linestyle="", markersize=7, label=label)
            for label, color in legend_map.items()
        ]
        fig.legend(
            handles,
            list(legend_map),
            loc="upper center",
            ncol=len(legend_map),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(f"{dataset} Sensitivity Analysis Overview", fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_one_group(group_df):
    group_df = group_df.sort_values("sweep_value")
    param_name = str(group_df["paper_param"].iloc[0])
    method = str(group_df["method"].iloc[0])
    method_label = METHOD_LABELS.get(method, method)

    if group_df.empty:
        return f"- {method_label} / {param_name}: no summary rows."

    anchor_df = group_df[group_df["is_anchor"]]
    anchor_row = anchor_df.iloc[0] if not anchor_df.empty else None
    best_idx = group_df["robust_score"].idxmax()
    best_row = group_df.loc[best_idx]

    if len(group_df) == 1:
        line = (
            f"- {method_label} / {param_name}: only 1 point "
            f"({format_value(param_name, best_row['sweep_value'])}), "
            f"F1Mi={best_row['F1Mi_mean']:.4f}, robust={best_row['robust_score']:.4f}; "
            "insufficient to judge the trend."
        )
        if anchor_row is not None and pd.notna(anchor_row.get("anchor_grid_robust")):
            line += (
                f" Grid-search anchor robust={anchor_row['anchor_grid_robust']:.4f}."
            )
        return line

    robust_span = group_df["robust_score"].max() - group_df["robust_score"].min()
    line = (
        f"- {method_label} / {param_name}: best robust at "
        f"{format_value(param_name, best_row['sweep_value'])} "
        f"(robust={best_row['robust_score']:.4f}, F1Mi={best_row['F1Mi_mean']:.4f}); "
        f"robust range={robust_span:.4f}."
    )

    if anchor_row is not None and pd.notna(anchor_row.get("robust_score")):
        delta = float(best_row["robust_score"]) - float(anchor_row["robust_score"])
        line += f" Delta vs anchor={delta:+.4f}."

    diffs = group_df["robust_score"].diff().dropna()
    if not diffs.empty:
        if (diffs >= 0).all():
            line += " Trend is monotonic increasing in the tested range."
        elif (diffs <= 0).all():
            line += " Trend is monotonic decreasing in the tested range."
        else:
            line += " Trend is non-monotonic, suggesting a local optimum."

    return line


def build_report(dataset, input_infos, summary_df, out_png):
    lines = [
        f"# {dataset} Sensitivity Analysis",
        "",
        f"- Plot: `{os.path.relpath(out_png)}`",
        "",
        "## Data Status",
    ]

    for info in input_infos:
        lines.append(
            f"- `{os.path.relpath(info['path'])}`: status={info['status']}, "
            f"rows={info['all_rows']}, summary_rows={info['summary_rows']}"
        )

    lines.extend(["", "## Findings"])

    if summary_df.empty:
        lines.append("- No summary rows were found, so no trend analysis is possible yet.")
        return "\n".join(lines) + "\n"

    for method in summary_df["method"].dropna().unique():
        method_df = summary_df[summary_df["method"] == method]
        for param_name in PARAM_ORDER:
            group_df = method_df[method_df["paper_param"] == param_name]
            if group_df.empty:
                lines.append(
                    f"- {METHOD_LABELS.get(method, method)} / {param_name}: no summary rows."
                )
                continue
            lines.append(summarize_one_group(group_df))

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grace_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    out_dir = os.path.join(grace_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    input_paths = resolve_input_paths(grace_dir, args.dataset, args.methods, args.inputs)
    input_infos = []
    summary_frames = []
    for path in input_paths:
        info, summary_df = load_summary_rows(path)
        input_infos.append(info)
        if not summary_df.empty:
            summary_frames.append(summary_df)

    merged_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    dataset_slug = args.dataset.lower()
    png_path = os.path.join(out_dir, f"{dataset_slug}_ifl_sensitivity_overview.png")
    report_path = os.path.join(out_dir, f"{dataset_slug}_ifl_sensitivity_analysis.md")

    make_plot(args.dataset, args.methods, merged_summary, png_path)
    report = build_report(args.dataset, input_infos, merged_summary, png_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[plot] saved figure: {png_path}")
    print(f"[plot] saved report: {report_path}")


if __name__ == "__main__":
    main()
