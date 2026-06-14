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
METHOD_ORDER = ["ifl-gr", "ifl-gc"]
METHOD_LABELS = {
    "ifl-gr": "SG-GR",
    "ifl-gc": "SG-GC",
}
VARIANT_ORDER = ["full", "no_warmup", "single_mining", "uniform_weight"]
VARIANT_LABELS = {
    "full": "Full",
    "no_warmup": "M-off",
    "single_mining": "K-off",
    "uniform_weight": "w-off",
}
VARIANT_COLORS = {
    "full": "#4E79A7",
    "no_warmup": "#F28E2B",
    "single_mining": "#59A14F",
    "uniform_weight": "#E15759",
}
ROBUST_SCORE_YLABEL = "稳健性评分robust_score（%）"
EFFECT_SPECS = [
    {
        "variant": "no_warmup",
        "file_stem": "extra_ablation_warmup_M_effect",
        "label": "M-off",
        "legend_label": "M-off",
    },
    {
        "variant": "single_mining",
        "file_stem": "extra_ablation_update_K_effect",
        "label": "K-off",
        "legend_label": "K-off",
    },
    {
        "variant": "uniform_weight",
        "file_stem": "extra_ablation_weight_w_effect",
        "label": "w-off",
        "legend_label": "w-off",
    },
]

NUMERIC_COLUMNS = [
    "num_runs",
    "F1Mi_mean",
    "F1Mi_std",
    "F1Ma_mean",
    "F1Ma_std",
    "robust_score",
    "robust_score_std",
    "delta_vs_full",
    "drop_vs_full",
    "relative_drop_vs_full",
    "trace_ts_mean",
    "trace_ts_last",
    "trace_mined_pairs_mean",
    "trace_mined_pairs_last",
    "trace_avg_pairs_mean",
    "trace_avg_pairs_last",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SG-GCL component ablation results and generate a short report."
    )
    parser.add_argument("--inputs", nargs="+", default=None, help="Optional explicit ablation CSV files.")
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
    return sorted((PROJECT_ROOT / "results").glob("extra_ablation_*_results.csv"))


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
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#303030",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
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


def axis_score_limits(values, errors=None, baseline_zero=True):
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
        return 0.0, high + max(span * 0.18, 1.0)
    return low - span * 0.25, high + span * 0.25


def local_score_limits(values, errors=None, min_span=4.0):
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
    pad = max(span * 0.35, 0.5)
    lower = max(0.0, low - pad)
    upper = min(100.0, high + pad)

    if upper - lower < min_span:
        extra = (min_span - (upper - lower)) * 0.5
        lower = max(0.0, lower - extra)
        upper = min(100.0, upper + extra)

    return lower, upper


def draw_variant_bars(ax, subset, metric, err_metric=None, baseline_zero=True):
    ordered_variants = [variant for variant in VARIANT_ORDER if variant in set(subset["variant"])]
    if not ordered_variants:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    subset = subset.set_index("variant").reindex(ordered_variants)
    values = subset[metric].astype(float).to_numpy() * 100.0
    errors = None
    if err_metric and err_metric in subset.columns:
        errors = subset[err_metric].astype(float).fillna(0.0).to_numpy() * 100.0

    x = range(len(ordered_variants))
    colors = [VARIANT_COLORS[variant] for variant in ordered_variants]
    ax.bar(x, values, yerr=errors, capsize=2.5 if errors is not None else 0, color=colors, edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels([VARIANT_LABELS[variant] for variant in ordered_variants], rotation=28, ha="right")
    ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(*axis_score_limits(values, errors=errors, baseline_zero=baseline_zero))


def draw_effect_axis(ax, subset, variant, y_limits):
    ordered_methods = [method for method in METHOD_ORDER if method in set(subset["method"])]
    if not ordered_methods:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return []

    variants = ["full", variant]
    width = 0.34
    x = list(range(len(ordered_methods)))
    handles = []
    for idx, item in enumerate(variants):
        item_df = subset[subset["variant"] == item].set_index("method").reindex(ordered_methods)
        values = item_df["robust_score"].astype(float).to_numpy() * 100.0
        errors = (
            item_df["robust_score_std"].astype(float).fillna(0.0).to_numpy() * 100.0
            if "robust_score_std" in item_df.columns
            else None
        )
        positions = [pos + (idx - 0.5) * width for pos in x]
        bars = ax.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=2.4 if errors is not None else 0,
            color=VARIANT_COLORS[item],
            edgecolor="white",
            linewidth=0.45,
            label=VARIANT_LABELS[item],
        )
        handles.append(bars[0])

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in ordered_methods])
    ax.set_ylabel(ROBUST_SCORE_YLABEL)
    ax.set_ylim(*y_limits)
    ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors="#303030", pad=2)
    return handles


def effect_axis_limits(summary_df, variant):
    subset = summary_df[summary_df["variant"].isin(["full", variant])].copy()
    values = pd.to_numeric(subset["robust_score"], errors="coerce") * 100.0
    errors = (
        pd.to_numeric(subset["robust_score_std"], errors="coerce").fillna(0.0) * 100.0
        if "robust_score_std" in subset.columns
        else None
    )
    return local_score_limits(values, errors=errors)


def make_effect_plot(summary_df, spec, out_dir, dpi, formats):
    datasets = available_ordered(summary_df["dataset"].dropna().astype(str), DATASET_ORDER)
    if not datasets:
        raise RuntimeError("No dataset summary rows available for plotting.")

    ncols = 2 if len(datasets) > 1 else 1
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.4 * ncols, 3.15 * nrows),
        squeeze=False,
    )
    legend_handles = []
    legend_labels = []

    for idx, dataset in enumerate(datasets):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]
        subset = summary_df[
            (summary_df["dataset"] == dataset)
            & (summary_df["variant"].isin(["full", spec["variant"]]))
        ].copy()
        y_limits = effect_axis_limits(subset, spec["variant"])
        handles = draw_effect_axis(ax, subset, spec["variant"], y_limits)
        add_panel_label_below(ax, panel_label(idx, dataset), y=-0.24, fontsize=10)
        if handles and not legend_handles:
            legend_handles = handles
            legend_labels = ["Full", spec["legend_label"]]

    for idx in range(len(datasets), nrows * ncols):
        row_idx, col_idx = divmod(idx, ncols)
        axes[row_idx][col_idx].set_visible(False)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(legend_handles),
            frameon=False,
            handlelength=1.5,
            columnspacing=1.4,
        )

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.93], w_pad=1.2, h_pad=2.7)
    saved_paths = save_figure_formats(fig, out_dir / spec["file_stem"], formats, dpi=dpi)
    plt.close(fig)
    return saved_paths


def make_panel_plot(summary_df, metric, err_metric, ylabel, title, out_base, dpi, baseline_zero=True):
    datasets = available_ordered(summary_df["dataset"].dropna().astype(str), DATASET_ORDER)
    methods = available_ordered(summary_df["method"].dropna().astype(str), METHOD_ORDER)
    if not datasets or not methods:
        raise RuntimeError("No dataset/method summary rows available for plotting.")

    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=len(datasets),
        figsize=(3.2 * len(datasets), 2.85 * len(methods)),
        squeeze=False,
    )

    for row_idx, method in enumerate(methods):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            subset = summary_df[(summary_df["dataset"] == dataset) & (summary_df["method"] == method)].copy()
            draw_variant_bars(
                ax,
                subset,
                metric=metric,
                err_metric=err_metric,
                baseline_zero=baseline_zero,
            )
            if col_idx == 0:
                ax.set_ylabel(f"{METHOD_LABELS.get(method, method)}\n{ylabel}")
            if row_idx == len(methods) - 1:
                ax.set_xlabel(dataset)
            if metric in {"drop_vs_full", "delta_vs_full"}:
                ax.axhline(0.0, color="#303030", linewidth=0.8)

    fig.tight_layout()
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def summarize_component_impact(summary_df):
    lines = []
    ablations = summary_df[summary_df["variant"] != "full"].copy()
    if ablations.empty:
        return ["- No ablation summary rows were found."]

    for variant in [variant for variant in VARIANT_ORDER if variant != "full"]:
        variant_df = ablations[ablations["variant"] == variant]
        if variant_df.empty:
            lines.append(f"- {VARIANT_LABELS[variant]}: no summary rows.")
            continue
        drop = variant_df["drop_vs_full"].dropna()
        if drop.empty:
            lines.append(f"- {VARIANT_LABELS[variant]}: drop_vs_full is unavailable.")
            continue
        positive = int((drop > 0).sum())
        total = int(len(drop))
        lines.append(
            f"- {VARIANT_LABELS[variant]}: mean drop={drop.mean():.4f}, "
            f"max drop={drop.max():.4f}, positive drops={positive}/{total}."
        )
    return lines


def summarize_dataset_method(group):
    dataset = str(group["dataset"].iloc[0])
    method = str(group["method"].iloc[0])
    full = group[group["variant"] == "full"]
    full_score = float(full["robust_score"].iloc[0]) if not full.empty else None
    prefix = f"- {dataset} / {METHOD_LABELS.get(method, method)}"
    if full_score is None:
        return f"{prefix}: missing full variant."

    parts = [f"{prefix}: full robust={full_score:.4f}"]
    for variant in [item for item in VARIANT_ORDER if item != "full"]:
        row = group[group["variant"] == variant]
        if row.empty:
            continue
        robust = row["robust_score"].iloc[0]
        drop = row["drop_vs_full"].iloc[0]
        parts.append(f"{VARIANT_LABELS[variant]} robust={robust:.4f}, drop={drop:+.4f}")
    return "; ".join(parts) + "."


def build_report(input_infos, summary_df, generated_files):
    lines = [
        "# SG-GCL Component Ablation Analysis",
        "",
        "## Data Status",
    ]

    for info in input_infos:
        rel = os.path.relpath(info["path"], PROJECT_ROOT)
        lines.append(
            f"- `{rel}`: status={info['status']}, rows={info['rows']}, "
            f"summary_rows={info['summary_rows']}"
        )

    lines.extend(["", "## Component Impact"])
    if summary_df.empty:
        lines.append("- No summary rows were available, so no conclusions can be generated yet.")
    else:
        lines.extend(summarize_component_impact(summary_df))

    lines.extend(["", "## Dataset And Method Details"])
    if summary_df.empty:
        lines.append("- No details available.")
    else:
        for (_dataset, _method), group in summary_df.groupby(["dataset", "method"], sort=False):
            lines.append(summarize_dataset_method(group))

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

    input_paths = resolve_input_paths(args.inputs)
    if not input_paths:
        raise RuntimeError("No extra_ablation_*_results.csv files were found.")

    input_infos, summary_df = load_summary_rows(input_paths)
    if summary_df.empty:
        raise RuntimeError("No summary rows were found in the provided component-ablation CSV files.")

    summary_df["dataset"] = pd.Categorical(summary_df["dataset"], categories=DATASET_ORDER, ordered=True)
    summary_df["method"] = pd.Categorical(summary_df["method"], categories=METHOD_ORDER, ordered=True)
    summary_df["variant"] = pd.Categorical(summary_df["variant"], categories=VARIANT_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["dataset", "method", "variant"]).reset_index(drop=True)

    configure_plot_style()
    generated = []

    for spec in EFFECT_SPECS:
        generated.extend(make_effect_plot(summary_df, spec, out_dir, args.dpi, args.formats))

    report_path = out_dir / "extra_ablation_analysis.md"
    report_path.write_text(
        build_report(input_infos, summary_df, generated + [report_path]),
        encoding="utf-8",
    )
    generated.append(report_path)

    for path in generated:
        print(f"[plot] saved: {path}")


if __name__ == "__main__":
    main()
