import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.ticker import MultipleLocator

try:
    from scipy.signal import savgol_filter
except ImportError:  # Keep plotting usable in lighter Python environments.
    savgol_filter = None


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "dblp": "DBLP",
}
COMBINED_ORDER = ["cora", "citeseer", "pubmed", "dblp"]
REQUIRED_COLUMNS = ["epoch", "violation_rate", "mean_margin"]
DEFAULT_FORMATS = ["png", "pdf", "tiff"]
VIOLATION_COLOR = "#C2185B"
MARGIN_COLOR = "#1976D2"


def dataset_slug(dataset):
    return dataset.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def dataset_label(slug):
    return DATASET_LABELS.get(slug, slug.upper() if slug == "dblp" else slug.title())


def infer_slug_from_csv(csv_path):
    stem = csv_path.stem
    if stem.startswith("exp1_"):
        return stem[len("exp1_") :]
    return stem


def resolve_user_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def smooth_series(values, window_length=11, polyorder=2):
    if savgol_filter is None:
        return values

    n = len(values)
    if n < 5:
        return values

    window = min(window_length, n if n % 2 == 1 else n - 1)
    if window < 5:
        return values
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return values

    return savgol_filter(values, window_length=window, polyorder=polyorder)


def load_metrics(csv_path):
    df = pd.read_csv(csv_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing_cols}. "
            f"Current columns: {df.columns.tolist()}"
        )

    data = df.loc[:, REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=REQUIRED_COLUMNS).sort_values("epoch")
    if data.empty:
        raise ValueError(f"No valid metric rows found in CSV: {csv_path}")

    return data


def output_paths_for(csv_path, slug, args):
    if args.out:
        out_path = resolve_user_path(args.out)
        if args.all:
            if out_path.suffix:
                out_path = out_path.with_name(f"{out_path.stem}_{slug}{out_path.suffix}")
                return [out_path]
            out_path = out_path.parent / f"{out_path.name}_{slug}"
        if out_path.suffix:
            return [out_path]
        return [out_path.with_suffix(f".{fmt}") for fmt in args.formats]

    output_dir = resolve_user_path(args.output_dir) if args.output_dir else csv_path.parent
    output_base = output_dir / f"exp1_{slug}_academic"
    return [output_base.with_suffix(f".{fmt}") for fmt in args.formats]


def combined_output_paths(args):
    if args.out:
        out_path = resolve_user_path(args.out)
        if out_path.suffix:
            return [out_path]
        return [out_path.with_suffix(f".{fmt}") for fmt in args.formats]

    output_dir = resolve_user_path(args.output_dir) if args.output_dir else SCRIPT_DIR
    output_base = output_dir / "exp1_combined_academic"
    return [output_base.with_suffix(f".{fmt}") for fmt in args.formats]


def configure_style():
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    matplotlib.rcParams["font.size"] = 14
    matplotlib.rcParams["axes.linewidth"] = 1.4
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"


def choose_epoch_locator(epoch_min, epoch_max):
    span = int(epoch_max - epoch_min)
    if span <= 50:
        return MultipleLocator(10)
    if span <= 150:
        return MultipleLocator(20)
    return MultipleLocator(25)


def padded_limits(values, min_pad=0.005):
    y_min = float(values.min())
    y_max = float(values.max())
    pad = max(min_pad, (y_max - y_min) * 0.08)
    return y_min - pad, y_max + pad


def draw_metric_panel(
    ax1,
    data,
    title,
    args,
    compact=False,
    show_legend=True,
    show_xlabel=True,
):
    x = data["epoch"]
    y1 = data["violation_rate"]
    y2 = data["mean_margin"]

    y1_smooth = smooth_series(
        y1,
        window_length=args.smooth_window,
        polyorder=args.smooth_polyorder,
    )
    y2_smooth = smooth_series(
        y2,
        window_length=args.smooth_window,
        polyorder=args.smooth_polyorder,
    )

    ax2 = ax1.twinx()

    title_size = 18 if compact else 20
    label_size = 14 if compact else 18
    tick_size = 12 if compact else 15
    raw_width = 1.0 if compact else 1.2
    smooth_width = 2.2 if compact else 2.6

    ax1.plot(x, y1, color=VIOLATION_COLOR, linewidth=raw_width, alpha=0.22, zorder=1)
    ax2.plot(x, y2, color=MARGIN_COLOR, linewidth=raw_width, alpha=0.22, zorder=1)

    line1, = ax1.plot(
        x,
        y1_smooth,
        color=VIOLATION_COLOR,
        linewidth=smooth_width,
        label="Violation Rate",
        zorder=3,
    )
    line2, = ax2.plot(
        x,
        y2_smooth,
        color=MARGIN_COLOR,
        linewidth=smooth_width,
        label="Mean Margin",
        zorder=3,
    )

    ax1.set_title(title, fontsize=title_size, pad=10 if compact else 14)
    if show_xlabel:
        ax1.set_xlabel("Epoch", fontsize=label_size, labelpad=7)
    ax1.set_ylabel("Violation Rate", fontsize=label_size, color=VIOLATION_COLOR, labelpad=8)
    ax2.set_ylabel("Mean Margin", fontsize=label_size, color=MARGIN_COLOR, labelpad=9)

    ax1.set_xlim(float(x.min()), float(x.max()))
    ax1.set_ylim(*padded_limits(y1))
    ax2.set_ylim(*padded_limits(y2))
    ax1.xaxis.set_major_locator(choose_epoch_locator(x.min(), x.max()))

    ax1.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.28)
    ax1.grid(axis="x", visible=False)

    ax1.tick_params(axis="x", direction="out", length=6, width=1.2, labelsize=tick_size)
    ax1.tick_params(
        axis="y",
        direction="out",
        length=6,
        width=1.2,
        labelsize=tick_size,
        colors=VIOLATION_COLOR,
    )
    ax2.tick_params(
        axis="y",
        direction="out",
        length=6,
        width=1.2,
        labelsize=tick_size,
        colors=MARGIN_COLOR,
    )

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.6)
    ax1.spines["bottom"].set_linewidth(1.6)
    ax2.spines["right"].set_linewidth(1.6)

    lines = [line1, line2]
    if show_legend:
        labels = [line.get_label() for line in lines]
        legend = ax1.legend(
            lines,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            fontsize=14,
            borderaxespad=0.0,
        )
        legend.get_frame().set_edgecolor("0.75")
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_alpha(0.92)

    return lines


def plot_one(csv_path, slug, title, args, plt):
    data = load_metrics(csv_path)

    configure_style()
    fig, ax1 = plt.subplots(figsize=(10, 6.2), dpi=300)
    draw_metric_panel(ax1, data, title, args)

    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=0.88)

    saved_paths = output_paths_for(csv_path, slug, args)
    for output_path in saved_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, facecolor="white")

    if args.show:
        plt.show()
    plt.close(fig)

    return saved_paths


def plot_combined(args, plt):
    jobs = []
    missing_paths = []
    for slug in COMBINED_ORDER:
        csv_path = SCRIPT_DIR / f"exp1_{slug}.csv"
        if csv_path.exists():
            jobs.append((csv_path, slug, dataset_label(slug)))
        else:
            missing_paths.append(csv_path)

    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing CSV file(s) for combined plot: {missing}")

    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    legend_lines = None

    for index, (ax1, (csv_path, _slug, title)) in enumerate(zip(axes.flat, jobs)):
        data = load_metrics(csv_path)
        show_xlabel = index >= 2
        lines = draw_metric_panel(
            ax1,
            data,
            title,
            args,
            compact=True,
            show_legend=False,
            show_xlabel=show_xlabel,
        )
        if legend_lines is None:
            legend_lines = lines

    if legend_lines:
        labels = [line.get_label() for line in legend_lines]
        legend = fig.legend(
            legend_lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=2,
            frameon=True,
            fontsize=15,
        )
        legend.get_frame().set_edgecolor("0.75")
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_alpha(0.94)

    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.075, top=0.92, wspace=0.36, hspace=0.34)

    saved_paths = combined_output_paths(args)
    for output_path in saved_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, facecolor="white")

    if args.show:
        plt.show()
    plt.close(fig)

    return saved_paths


def discover_csvs():
    return sorted(SCRIPT_DIR.glob("exp1_*.csv"))


def build_plot_jobs(args):
    if args.all:
        csv_paths = discover_csvs()
        if not csv_paths:
            raise FileNotFoundError(f"No exp1_*.csv files found in: {SCRIPT_DIR}")
        return [
            (csv_path, infer_slug_from_csv(csv_path), dataset_label(infer_slug_from_csv(csv_path)))
            for csv_path in csv_paths
        ]

    if args.csv:
        csv_path = resolve_user_path(args.csv)
        slug = dataset_slug(args.dataset) if args.dataset else infer_slug_from_csv(csv_path)
    else:
        if not args.dataset:
            raise ValueError("Please provide --dataset, --csv, or --all.")
        slug = dataset_slug(args.dataset)
        csv_path = SCRIPT_DIR / f"exp1_{slug}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    return [(csv_path, slug, dataset_label(slug))]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot academic-style curves for Experiment 1 CSV files."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name, e.g. Cora / CiteSeer / PubMed / DBLP.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional explicit CSV path. Relative paths are resolved from the current directory.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every exp1_*.csv file in the same directory as this script.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Plot Cora, CiteSeer, PubMed, and DBLP as one 2x2 combined figure.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Optional output file or output base path. If no suffix is provided, "
            "--formats are appended."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=DEFAULT_FORMATS,
        help="Output formats used when --out has no suffix. Default: png pdf tiff.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Savitzky-Golay smoothing window length.",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay smoothing polynomial order.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving. Disabled by default for headless runs.",
    )
    return parser.parse_args()


def normalize_formats(formats):
    normalized = []
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if fmt and fmt not in normalized:
            normalized.append(fmt)
    if not normalized:
        raise ValueError("At least one output format is required.")
    return normalized


def main():
    args = parse_args()
    args.formats = normalize_formats(args.formats)

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    generated_paths = []
    if args.combined:
        generated_paths = plot_combined(args, plt)
        for output_path in generated_paths:
            print(f"[plot] saved combined figure: {output_path}")
        return generated_paths

    for csv_path, slug, title in build_plot_jobs(args):
        saved_paths = plot_one(csv_path, slug, title, args, plt)
        generated_paths.extend(saved_paths)
        print(f"[plot] {title}: {csv_path}")
        for output_path in saved_paths:
            print(f"[plot] saved: {output_path}")

    return generated_paths


if __name__ == "__main__":
    main()
