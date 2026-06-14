import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.plotting_common import (
    apply_common_vector_settings,
    normalize_formats,
    save_figure_formats,
    save_figure_paths,
)


def read_metrics(csv_path: str):
    epochs: List[int] = []
    violation_rate: List[float] = []
    mean_margin: List[float] = []

    with open(csv_path, mode='r', encoding='utf-8', newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            epochs.append(int(row['epoch']))
            violation_rate.append(float(row['violation_rate']))
            mean_margin.append(float(row['mean_margin']))

    if not epochs:
        raise ValueError(f'No rows found in csv: {csv_path}')

    return epochs, violation_rate, mean_margin


def build_curve_figure(csv_path: str, title: str):
    epochs, violation_rate, mean_margin = read_metrics(csv_path)

    apply_common_vector_settings(plt)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        epochs,
        violation_rate,
        color='#d81b60',
        linewidth=2.0,
        label='violation_rate',
    )
    line2 = ax2.plot(
        epochs,
        mean_margin,
        color='#1e88e5',
        linewidth=2.0,
        label='mean_margin',
    )

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('违例率violation_rate', color='#d81b60')
    ax2.set_ylabel('边界距mean_margin', color='#1e88e5')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    fig.tight_layout()
    return fig


def plot_curves(csv_path: str, output_path: str, title: str):
    fig = build_curve_figure(csv_path, title)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_figure_paths(fig, [output_path], dpi=320)
    plt.close(fig)


def plot_curves_formats(csv_path: str, output_base: str, title: str, formats):
    output_base = os.path.splitext(output_base)[0]
    fig = build_curve_figure(csv_path, title)
    saved_paths = save_figure_formats(fig, output_base, formats, dpi=320)
    plt.close(fig)
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description='Plot Experiment 1 curves from CSV')
    parser.add_argument('--csv', type=str, default='logs/exp1_cora.csv')
    parser.add_argument('--out', type=str, default='logs/exp1_cora_curves.png')
    parser.add_argument('--title', type=str, default='采样偏差实验')
    parser.add_argument(
        '--formats',
        nargs='+',
        default=None,
        help='Optional formats used when --out is treated as an output base, e.g. png pdf svg.',
    )
    args = parser.parse_args()

    if args.formats:
        saved_paths = plot_curves_formats(args.csv, args.out, args.title, normalize_formats(args.formats))
        for path in saved_paths:
            print(f'Saved figure to: {path}')
    else:
        plot_curves(args.csv, args.out, args.title)
        print(f'Saved figure to: {args.out}')


if __name__ == '__main__':
    main()
