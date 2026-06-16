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
import pandas as pd
from matplotlib.lines import Line2D

from experiments.plotting_common import (
    DEFAULT_FIGURE_FORMATS,
    add_panel_label_below,
    apply_common_vector_settings,
    normalize_formats,
    panel_label,
    save_figure_formats,
)


METHOD_FILE_SLUG = {
    "ifl-gr": "iflgr",
    "ifl-gc": "iflgc",
}

METHOD_LABELS = {
    "ifl-gr": "SG-GR",
    "ifl-gc": "SG-GC",
}

METHOD_COLORS = {
    "ifl-gr": "#73B86E",
    "ifl-gc": "#7A6E9F",
}

METHOD_MARKERS = {
    "ifl-gr": "o",
    "ifl-gc": "s",
}
ROBUST_SCORE_YLABEL = "robust_score（%）"
ANCHOR_LABEL = "观测点"

PARAM_ORDER = ["t_s", "M", "K"]
PARAM_LABELS = {
    "t_s": r"$t_s$",
    "M": "M",
    "K": "K",
}
PARAM_EFFECT_SPECS = [
    {"param": "t_s", "file_stem": "ifl_sensitivity_ts_effect"},
    {"param": "M", "file_stem": "ifl_sensitivity_M_effect"},
    {"param": "K", "file_stem": "ifl_sensitivity_K_effect"},
]

DATASET_ORDER = ["Cora", "CiteSeer", "DBLP", "PubMed"]
DATASET_SLUG_TO_LABEL = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "dblp": "DBLP",
    "pubmed": "PubMed",
}

METRIC_SPECS = {
    "robust_score": {
        "label": "robust_score",
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
        description="Plot SG-GR / SG-GC sensitivity-analysis CSV files and generate a short report."
    )
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_ORDER,
        help="Datasets used by --combined. Default: Cora CiteSeer DBLP PubMed.",
    )
    parser.add_argument("--methods", nargs="+", default=["ifl-gr", "ifl-gc"], choices=list(METHOD_FILE_SLUG))
    parser.add_argument("--inputs", nargs="+", default=None, help="Optional explicit CSV paths.")
    parser.add_argument("--out_dir", type=str, default=os.path.join("results", "plots"))
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create 2x2 robust_score overview figures for all selected datasets.",
    )
    parser.add_argument("--dpi", type=int, default=320, help="Raster output dpi.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=DEFAULT_FIGURE_FORMATS,
        help="Figure formats to save. Default: png pdf svg.",
    )
    return parser.parse_args()


def normalize_dataset_name(dataset):
    slug = dataset.lower().replace("-", "").replace("_", "").replace(" ", "")
    return DATASET_SLUG_TO_LABEL.get(slug, dataset)


def dataset_slug(dataset):
    return normalize_dataset_name(dataset).lower()


def resolve_input_paths(grace_dir, dataset, methods, explicit_inputs):
    if explicit_inputs:
        return [os.path.join(grace_dir, path) if not os.path.isabs(path) else path for path in explicit_inputs]

    paths = []
    slug = dataset_slug(dataset)
    for method in methods:
        method_slug = METHOD_FILE_SLUG[method]
        paths.append(
            os.path.join(grace_dir, "results", f"sensitivity_{method_slug}_{slug}_results.csv")
        )
    return paths


def project_relpath(path):
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return os.fspath(path)


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


def configure_plot_style():
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#303030",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
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


def robust_score_limits(summary_df):
    if summary_df.empty or "robust_score" not in summary_df.columns:
        return 0.0, 1.0

    subset = summary_df[summary_df["paper_param"].isin(PARAM_ORDER)].copy()
    values = pd.to_numeric(subset["robust_score"], errors="coerce") * 100.0
    errors = (
        pd.to_numeric(subset["robust_score_std"], errors="coerce").fillna(0.0) * 100.0
        if "robust_score_std" in subset.columns
        else 0.0
    )

    low = float((values - errors).min())
    high = float((values + errors).max())
    if pd.isna(low) or pd.isna(high):
        return 0.0, 1.0

    span = max(high - low, 1e-6)
    lower = max(0.0, low - span * 0.18)
    upper = high + span * 0.20
    if upper - lower < 3.0:
        pad = (3.0 - (upper - lower)) * 0.5
        lower = max(0.0, lower - pad)
        upper += pad
    return lower, upper


def style_axis(ax, show_ylabel=False):
    ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", direction="out", length=3.2, width=0.8, colors="#303030")
    if show_ylabel:
        ax.set_ylabel(ROBUST_SCORE_YLABEL)


def draw_robust_param_axis(ax, summary_df, methods, param_name, y_limits=None, show_ylabel=False):
    # For the t_s figure, move the x-axis label slightly upward so that
    # it has more visual separation from the panel label, e.g., "(a) Cora".
    if param_name == "t_s":
        ax.set_xlabel(PARAM_LABELS[param_name], labelpad=-3)
    else:
        ax.set_xlabel(PARAM_LABELS[param_name])
    style_axis(ax, show_ylabel=show_ylabel)

    has_any_data = False
    required_cols = {"paper_param", "method", "sweep_value", "robust_score"}.issubset(summary_df.columns)
    if not required_cols:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    for method in methods:
        subset = summary_df[
            (summary_df["paper_param"] == param_name) & (summary_df["method"] == method)
        ].copy()
        subset = subset.dropna(subset=["sweep_value", "robust_score"]).sort_values("sweep_value")
        if subset.empty:
            continue

        has_any_data = True
        x = subset["sweep_value"].to_numpy(dtype=float)
        y = subset["robust_score"].to_numpy(dtype=float) * 100.0

        color = METHOD_COLORS[method]
        ax.plot(
            x,
            y,
            color=color,
            marker=METHOD_MARKERS[method],
            markersize=4.5,
            linewidth=1.7,
            label=METHOD_LABELS[method],
            zorder=3,
        )

        if "is_anchor" in subset.columns:
            anchor_subset = subset[subset["is_anchor"]]
            if not anchor_subset.empty:
                ax.scatter(
                    anchor_subset["sweep_value"],
                    anchor_subset["robust_score"] * 100.0,
                    color=color,
                    marker="*",
                    s=95,
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=5,
                )

    if not has_any_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
        ax.set_xticks([])
        ax.set_yticks([])

    if param_name == "t_s":
        ax.tick_params(axis="x", labelrotation=32)

    if y_limits:
        ax.set_ylim(*y_limits)


def make_plot(dataset, methods, summary_df, out_base, formats, dpi):
    fig, axes = plt.subplots(
        nrows=len(METRIC_SPECS),
        ncols=len(PARAM_ORDER),
        figsize=(12.6, 3.7),
        squeeze=False,
    )

    legend_map = {}

    has_required_columns = {"paper_param", "method", "sweep_value", "is_anchor"}.issubset(summary_df.columns)

    for row_idx, (metric_col, metric_spec) in enumerate(METRIC_SPECS.items()):
        for col_idx, param_name in enumerate(PARAM_ORDER):
            ax = axes[row_idx][col_idx]
            style_axis(ax, show_ylabel=col_idx == 0)

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
                    ax.set_ylabel(ROBUST_SCORE_YLABEL)
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

                if len(subset) > 1:
                    ax.plot(
                        subset["sweep_value"],
                        subset[metric_col] * 100.0,
                        color=color,
                        marker=METHOD_MARKERS[method],
                        linewidth=1.8,
                        label=label,
                    )
                else:
                    ax.scatter(
                        subset["sweep_value"],
                        subset[metric_col] * 100.0,
                        color=color,
                        s=45,
                        label=label,
                        zorder=3,
                    )

                anchor_subset = subset[subset["is_anchor"]]
                if not anchor_subset.empty:
                    ax.scatter(
                        anchor_subset["sweep_value"],
                        anchor_subset[metric_col] * 100.0,
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
                ax.set_ylabel(ROBUST_SCORE_YLABEL)

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
            bbox_to_anchor=(0.5, 1.01),
        )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    saved_paths = save_figure_formats(fig, out_base, formats, dpi=dpi)
    plt.close(fig)
    return saved_paths


def load_dataset_summary(grace_dir, dataset, methods):
    input_paths = resolve_input_paths(grace_dir, dataset, methods, explicit_inputs=None)
    input_infos = []
    summary_frames = []
    for path in input_paths:
        info, summary_df = load_summary_rows(path)
        input_infos.append(info)
        if not summary_df.empty:
            summary_frames.append(summary_df)
    merged_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return input_infos, merged_summary


def make_combined_robust_plot(grace_dir, datasets, methods, out_dir, dpi):
    datasets = [normalize_dataset_name(dataset) for dataset in datasets]
    fig = plt.figure(figsize=(14.2, 9.0))
    outer_grid = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.895,
        wspace=0.16,
        hspace=0.34,
    )

    for index, dataset in enumerate(datasets[:4]):
        _, summary_df = load_dataset_summary(grace_dir, dataset, methods)
        row = index // 2
        col = index % 2
        inner_grid = outer_grid[row, col].subgridspec(1, len(PARAM_ORDER), wspace=0.12)
        axes = [fig.add_subplot(inner_grid[0, param_idx]) for param_idx in range(len(PARAM_ORDER))]
        y_limits = robust_score_limits(summary_df)

        for param_idx, (ax, param_name) in enumerate(zip(axes, PARAM_ORDER)):
            draw_robust_param_axis(
                ax,
                summary_df,
                methods,
                param_name,
                y_limits=y_limits,
                show_ylabel=param_idx == 0,
            )
            if param_idx > 0:
                ax.tick_params(axis="y", labelleft=False)

        left = axes[0].get_position().x0
        right = axes[-1].get_position().x1
        top = axes[0].get_position().y1
        fig.text(
            (left + right) * 0.5,
            top + 0.026,
            f"({chr(ord('a') + index)}) {dataset}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
        )

    legend_handles = [
        Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.8,
            markersize=5.5,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    legend_handles.append(
        Line2D(
            [],
            [],
            marker="*",
            color="#374151",
            markerfacecolor="#374151",
            markeredgecolor="white",
            linestyle="",
            markersize=10,
            label=ANCHOR_LABEL,
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=len(legend_handles),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.6,
    )

    png_path = os.path.join(out_dir, "ifl_sensitivity_robust_overview.png")
    pdf_path = os.path.join(out_dir, "ifl_sensitivity_robust_overview.pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def make_combined_param_effect_plot(grace_dir, datasets, methods, out_dir, dpi, formats, spec):
    datasets = [normalize_dataset_name(dataset) for dataset in datasets]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), squeeze=False)
    axes = axes.ravel()

    for index, dataset in enumerate(datasets[:4]):
        ax = axes[index]
        _, summary_df = load_dataset_summary(grace_dir, dataset, methods)
        draw_robust_param_axis(
            ax,
            summary_df,
            methods,
            spec["param"],
            y_limits=robust_score_limits(summary_df),
            show_ylabel=True,
        )
        # Keep the t_s panel label only slightly lower. Do not use a very
        # large negative y value together with a larger bottom margin, because
        # tight_layout may compress the 2x2 axes and make the whole figure messy.
        panel_label_y = -0.52 if spec["param"] == "t_s" else -0.40
        add_panel_label_below(ax, panel_label(index, dataset), y=panel_label_y, fontsize=10.5)

    for ax in axes[len(datasets[:4]) :]:
        ax.set_visible(False)

    legend_handles = [
        Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.8,
            markersize=5.5,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    legend_handles.append(
        Line2D(
            [],
            [],
            marker="*",
            color="#374151",
            markerfacecolor="#374151",
            markeredgecolor="white",
            linestyle="",
            markersize=10,
            label=ANCHOR_LABEL,
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(legend_handles),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.6,
    )

    # Use a moderate vertical padding. A very large h_pad or bottom margin
    # can make the first and second rows too far apart or compress the axes.
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.93], w_pad=1.3, h_pad=2.0)
    svg_paths = []
    if "svg" in formats:
        svg_dir = Path(out_dir).parent / "plot"
        svg_paths = save_figure_formats(fig, svg_dir / spec["file_stem"], ["svg"], dpi=dpi)
    other_formats = [f for f in formats if f != "svg"]
    other_paths = []
    if other_formats:
        other_paths = save_figure_formats(fig, Path(out_dir) / spec["file_stem"], other_formats, dpi=dpi)
    saved_paths = svg_paths + other_paths
    plt.close(fig)
    return saved_paths


def make_combined_param_effect_plots(grace_dir, datasets, methods, out_dir, dpi, formats):
    generated = []
    for spec in PARAM_EFFECT_SPECS:
        generated.extend(make_combined_param_effect_plot(grace_dir, datasets, methods, out_dir, dpi, formats, spec))
    return generated


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
        f"- Plot: `{project_relpath(out_png)}`",
        "",
        "## Data Status",
    ]

    for info in input_infos:
        lines.append(
            f"- `{project_relpath(info['path'])}`: status={info['status']}, "
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
    args.formats = normalize_formats(args.formats)
    grace_dir = PROJECT_ROOT
    out_dir = os.path.join(grace_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    configure_plot_style()

    if args.combined:
        generated_paths = make_combined_param_effect_plots(
            grace_dir=grace_dir,
            datasets=args.datasets,
            methods=args.methods,
            out_dir=out_dir,
            dpi=args.dpi,
            formats=args.formats,
        )
        for path in generated_paths:
            print(f"[plot] saved combined figure: {path}")
        return

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
    out_base = os.path.join(out_dir, f"{dataset_slug}_ifl_sensitivity_overview")

    saved_paths = make_plot(args.dataset, args.methods, merged_summary, out_base, args.formats, args.dpi)

    for path in saved_paths:
        print(f"[plot] saved figure: {path}")


if __name__ == "__main__":
    main()
