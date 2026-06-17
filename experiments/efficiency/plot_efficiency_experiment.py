import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from experiments.plotting_common import (
    DEFAULT_FIGURE_FORMATS,
    add_panel_label_below,
    apply_common_vector_settings,
    normalize_formats,
    panel_label,
    save_figure_formats,
)


DATASET_ORDER = ["Cora", "CiteSeer", "PubMed", "DBLP"]
METHOD_ORDER = ["grace", "gca", "ifl-gr", "ifl-gc"]
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
NUMERIC_COLUMNS = [
    "num_runs",
    "num_epochs_config",
    "epochs_observed",
    "wall_time_sec",
    "wall_time_std_sec",
    "train_total_sec",
    "train_total_std_sec",
    "eval_overhead_sec",
    "epoch_time_mean_sec",
    "epoch_time_std_sec",
    "epoch_time_last_sec",
    "epoch_time_median_sec",
    "throughput_epoch_per_sec",
    "refresh_count",
    "refresh_epoch_time_mean_sec",
    "non_refresh_epoch_time_mean_sec",
    "warmup_epoch_time_mean_sec",
    "corrected_epoch_time_mean_sec",
    "time_vs_grace_sec",
    "time_ratio_vs_grace",
    "overhead_vs_base_sec",
    "overhead_ratio_vs_base",
    "F1Mi_mean",
    "F1Mi_std",
    "F1Ma_mean",
    "F1Ma_std",
    "robust_score",
    "robust_score_std",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SG-GCL efficiency experiment results and generate a short report."
    )
    parser.add_argument("--inputs", nargs="+", default=None, help="Optional explicit efficiency CSV files.")
    parser.add_argument("--out_dir", type=str, default=os.path.join("results", "plots"))
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=DEFAULT_FIGURE_FORMATS,
        help="Figure formats to save. Default: png pdf svg.",
    )
    return parser.parse_args()


def resolve_input_paths(explicit_inputs):
    if explicit_inputs:
        paths = []
        for raw_path in explicit_inputs:
            path = Path(raw_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            paths.append(path.resolve())
        return paths
    return sorted((PROJECT_ROOT / "results").glob("efficiency_*_results.csv"))


def load_summary_rows(input_paths):
    frames = []
    input_infos = []

    for path in input_paths:
        info = {
            "path": path,
            "status": "missing",
            "rows": 0,
            "summary_rows": 0,
        }
        if not path.exists():
            input_infos.append(info)
            continue

        df = pd.read_csv(path)
        info["rows"] = int(len(df))
        if df.empty:
            info["status"] = "empty"
            input_infos.append(info)
            continue

        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        summary_df = df[df["stage"] == "summary"].copy()
        info["summary_rows"] = int(len(summary_df))
        info["status"] = "ok" if not summary_df.empty else "no_summary"
        input_infos.append(info)
        if not summary_df.empty:
            frames.append(summary_df)

    if not frames:
        return input_infos, pd.DataFrame()
    return input_infos, pd.concat(frames, ignore_index=True)


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#303030",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
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


def available_ordered(values, preferred_order):
    observed = [value for value in preferred_order if value in set(values)]
    observed.extend(sorted(value for value in set(values) if value not in preferred_order))
    return observed


def axis_limits(values, errors=None, baseline_zero=True):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return (0.0, 1.0)

    if errors is not None:
        errors = pd.to_numeric(pd.Series(errors), errors="coerce").fillna(0.0)
        low = float((values.reset_index(drop=True) - errors.reset_index(drop=True)).min())
        high = float((values.reset_index(drop=True) + errors.reset_index(drop=True)).max())
    else:
        low = float(values.min())
        high = float(values.max())

    span = max(high - low, 1e-6)
    if baseline_zero:
        return 0.0, high + max(span * 0.20, high * 0.08, 1e-3)
    return low - span * 0.25, high + span * 0.25


def draw_method_bars(ax, subset, metric, err_metric=None, baseline_zero=True):
    ordered_methods = [method for method in METHOD_ORDER if method in set(subset["method"])]
    if not ordered_methods:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    subset = subset.set_index("method").reindex(ordered_methods)
    values = subset[metric].astype(float).to_numpy()
    errors = None
    if err_metric and err_metric in subset.columns:
        errors = subset[err_metric].astype(float).fillna(0.0).to_numpy()

    x = range(len(ordered_methods))
    colors = [METHOD_COLORS.get(method, "#777777") for method in ordered_methods]
    ax.bar(x, values, yerr=errors, capsize=2.5 if errors is not None else 0, color=colors, edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in ordered_methods], rotation=20, ha="center")
    ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(*axis_limits(values, errors=errors, baseline_zero=baseline_zero))


def make_panel_plot(summary_df, metric, err_metric, ylabel, title, out_base, dpi, formats, baseline_zero=True, svg_out_dir=None):
    datasets = available_ordered(summary_df["dataset"].dropna().astype(str), DATASET_ORDER)
    if not datasets:
        raise RuntimeError("No dataset summary rows available for plotting.")

    ncols = 2 if len(datasets) > 1 else 1
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 2.9 * nrows),
        squeeze=False,
    )

    for idx, dataset in enumerate(datasets):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]
        subset = summary_df[summary_df["dataset"] == dataset].copy()
        draw_method_bars(
            ax,
            subset,
            metric=metric,
            err_metric=err_metric,
            baseline_zero=baseline_zero,
        )
        add_panel_label_below(ax, panel_label(idx, dataset), y=-0.27, fontsize=11)
        ax.set_ylabel(ylabel)
        if metric == "time_ratio_vs_grace":
            ax.axhline(1.0, color="#303030", linewidth=0.8)

    for idx in range(len(datasets), nrows * ncols):
        row_idx, col_idx = divmod(idx, ncols)
        axes[row_idx][col_idx].set_visible(False)

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0], w_pad=1.2, h_pad=2.5)

    raster_formats = [fmt for fmt in formats if fmt != "svg"]
    saved_paths = save_figure_formats(fig, out_base, raster_formats, dpi=dpi)
    if "svg" in formats and svg_out_dir is not None:
        svg_out_dir = Path(svg_out_dir)
        svg_out_dir.mkdir(parents=True, exist_ok=True)
        svg_path = svg_out_dir / f"{Path(out_base).name}.svg"
        fig.savefig(svg_path, facecolor="white", pad_inches=0.04)
        saved_paths.append(svg_path)
    elif "svg" in formats:
        saved_paths.extend(save_figure_formats(fig, out_base, ["svg"], dpi=dpi))

    plt.close(fig)
    return saved_paths


def summarize_efficiency(summary_df):
    lines = []
    if summary_df.empty:
        return ["- No summary rows were available."]

    for dataset in available_ordered(summary_df["dataset"].dropna().astype(str), DATASET_ORDER):
        dataset_df = summary_df[summary_df["dataset"] == dataset]
        if dataset_df.empty:
            continue
        fastest = dataset_df.dropna(subset=["train_total_sec"]).sort_values("train_total_sec").head(1)
        if fastest.empty:
            lines.append(f"- {dataset}: train_total_sec is unavailable.")
            continue
        fastest_row = fastest.iloc[0]
        parts = [
            f"- {dataset}: fastest={METHOD_LABELS.get(str(fastest_row['method']), fastest_row['method'])} "
            f"(train={fastest_row['train_total_sec']:.2f}s)"
        ]
        for method in [method for method in METHOD_ORDER if method in set(dataset_df["method"])]:
            row = dataset_df[dataset_df["method"] == method].iloc[0]
            ratio = row.get("time_ratio_vs_grace")
            ratio_text = "n/a" if pd.isna(ratio) else f"{ratio:.2f}x vs GRACE"
            parts.append(
                f"{METHOD_LABELS.get(method, method)} train={row['train_total_sec']:.2f}s, "
                f"wall={row['wall_time_sec']:.2f}s, {ratio_text}"
            )
        lines.append("; ".join(parts) + ".")
    return lines


def build_report(input_infos, summary_df, generated_files):
    lines = [
        "# SG-GCL Efficiency Analysis",
        "",
        "## Data Status",
    ]

    for info in input_infos:
        rel = os.path.relpath(info["path"], PROJECT_ROOT)
        lines.append(
            f"- `{rel}`: status={info['status']}, rows={info['rows']}, "
            f"summary_rows={info['summary_rows']}"
        )

    lines.extend(["", "## Time Summary"])
    lines.extend(summarize_efficiency(summary_df))

    lines.extend(["", "## Generated Files"])
    for path in generated_files:
        lines.append(f"- `{os.path.relpath(path, PROJECT_ROOT)}`")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    args.formats = normalize_formats(args.formats)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_out_dir = PROJECT_ROOT / "results" / "plot"
    svg_out_dir.mkdir(parents=True, exist_ok=True)

    input_paths = resolve_input_paths(args.inputs)
    if not input_paths:
        raise RuntimeError("No efficiency_*_results.csv files were found.")

    input_infos, summary_df = load_summary_rows(input_paths)
    if summary_df.empty:
        raise RuntimeError("No summary rows were found in the provided efficiency CSV files.")

    summary_df["dataset"] = pd.Categorical(summary_df["dataset"], categories=DATASET_ORDER, ordered=True)
    summary_df["method"] = pd.Categorical(summary_df["method"], categories=METHOD_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["dataset", "method"]).reset_index(drop=True)

    configure_plot_style()
    generated = []

    generated.extend(
        make_panel_plot(
            summary_df=summary_df,
            metric="train_total_sec",
            err_metric="train_total_std_sec",
            ylabel="训练时间（s）",
            title="",
            out_base=out_dir / "efficiency_train_total_time",
            dpi=args.dpi,
            formats=args.formats,
            baseline_zero=True,
            svg_out_dir=svg_out_dir,
        )
    )

    generated.extend(
        make_panel_plot(
            summary_df=summary_df,
            metric="wall_time_sec",
            err_metric="wall_time_std_sec",
            ylabel="端到端时间（s）",
            title="",
            out_base=out_dir / "efficiency_wall_time",
            dpi=args.dpi,
            formats=args.formats,
            baseline_zero=True,
            svg_out_dir=svg_out_dir,
        )
    )

    for path in generated:
        print(f"[plot] saved: {path}")


if __name__ == "__main__":
    main()
