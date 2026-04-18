import argparse
import csv
import os
from typing import List

import matplotlib.pyplot as plt


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


def plot_curves(csv_path: str, output_path: str, title: str):
    epochs, violation_rate, mean_margin = read_metrics(csv_path)

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
    ax1.set_ylabel('Violation Rate', color='#d81b60')
    ax2.set_ylabel('Mean Margin (s+ - max s-)', color='#1e88e5')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    fig.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot Experiment 1 curves from CSV')
    parser.add_argument('--csv', type=str, default='logs/exp1_cora.csv')
    parser.add_argument('--out', type=str, default='logs/exp1_cora_curves.png')
    parser.add_argument('--title', type=str, default='Experiment 1: Violation Rate and Margin')
    args = parser.parse_args()

    plot_curves(args.csv, args.out, args.title)
    print(f'Saved figure to: {args.out}')


if __name__ == '__main__':
    main()
