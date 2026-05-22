import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.statistical_significance.analyze_significance_results import COMPARISONS
from experiments.statistical_significance.run_significance_experiment import DATASET_CHOICES, METHOD_CHOICES


METHOD_LABELS = {
    "grace": "GRACE",
    "gca": "GCA",
    "ifl-gr": "IFL-GR",
    "ifl-gc": "IFL-GC",
}
METHOD_COLORS = {
    "grace": "#4E79A7",
    "gca": "#F28E2B",
    "ifl-gr": "#59A14F",
    "ifl-gc": "#E15759",
}
METRIC_LABELS = {
    "robust_score": "Robust Score",
    "F1Mi_mean": "Micro-F1",
    "F1Ma_mean": "Macro-F1",
}


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


def make_mean_std_plot(df, out_dir, dpi):
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

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_handles), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    png = out_dir / "significance_mean_std.png"
    pdf = out_dir / "significance_mean_std.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def make_delta_plot(df, out_dir, dpi):
    run_df = df[df["stage"] == "run"].copy()
    if run_df.empty:
        raise RuntimeError("No run rows available for delta plot.")
    run_df["robust_score"] = pd.to_numeric(run_df["robust_score"], errors="coerce")

    datasets = [item for item in DATASET_CHOICES if item in set(run_df["dataset"].astype(str))]
    comparison_labels = [
        f"{METHOD_LABELS[target]}-{METHOD_LABELS[base]}"
        for base, target, _kind in COMPARISONS
    ]
    x = np.arange(len(comparison_labels), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.2), sharey=True)
    axes = axes.ravel()
    for ax_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        dataset_df = run_df[run_df["dataset"].astype(str) == dataset]
        means = []
        errors = []
        for base, target, _kind in COMPARISONS:
            base_df = dataset_df[dataset_df["method"].astype(str) == base].set_index("run_idx")
            target_df = dataset_df[dataset_df["method"].astype(str) == target].set_index("run_idx")
            shared = sorted(set(base_df.index) & set(target_df.index), key=lambda item: int(float(item)))
            deltas = []
            for run_idx in shared:
                b = pd.to_numeric(pd.Series([base_df.loc[run_idx, "robust_score"]]), errors="coerce").iloc[0]
                t = pd.to_numeric(pd.Series([target_df.loc[run_idx, "robust_score"]]), errors="coerce").iloc[0]
                if pd.notna(b) and pd.notna(t):
                    deltas.append((float(t) - float(b)) * 100.0)
            means.append(float(np.mean(deltas)) if deltas else np.nan)
            errors.append(float(np.std(deltas, ddof=0)) if len(deltas) > 1 else 0.0)

        bars = ax.bar(x, means, yerr=errors, color="#5B8C85", edgecolor="white", linewidth=0.45, capsize=2.2)
        ax.axhline(0.0, color="black", linewidth=0.75)
        ax.set_title(f"({chr(ord('a') + ax_idx)}) {dataset}", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_labels, rotation=25, ha="right")
        ax.set_ylabel("Robust Score Delta (pp)" if ax_idx % 2 == 0 else "")
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        test_df = df[
            (df["stage"] == "test")
            & (df["dataset"].astype(str) == dataset)
            & (df["metric"].astype(str) == "robust_score")
        ]
        for bar_idx, (base, target, _kind) in enumerate(COMPARISONS):
            match = test_df[
                (test_df["baseline_method"].astype(str) == base)
                & (test_df["target_method"].astype(str) == target)
            ]
            if match.empty or str(match.iloc[0].get("significant", "")) != "True":
                continue
            height = means[bar_idx]
            if np.isnan(height):
                continue
            ax.text(bar_idx, height + (1.0 if height >= 0 else -1.0), "*", ha="center", va="center", fontsize=12)

    for ax in axes[len(datasets) :]:
        ax.set_visible(False)

    fig.tight_layout()
    png = out_dir / "significance_paired_delta.png"
    pdf = out_dir / "significance_paired_delta.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def parse_args():
    parser = argparse.ArgumentParser(description="Plot statistical significance experiment summaries.")
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default=os.path.join("results", "plots"))
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(PROJECT_ROOT)
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    input_paths = resolve_input_paths(repo_root, args.inputs)
    if not input_paths:
        raise RuntimeError("No significance CSV files were found.")
    df = load_rows(input_paths)

    generated = []
    generated.extend(make_mean_std_plot(df, out_dir, args.dpi))
    generated.extend(make_delta_plot(df, out_dir, args.dpi))
    for path in generated:
        print(f"[significance-plot] saved: {path}")


if __name__ == "__main__":
    main()
