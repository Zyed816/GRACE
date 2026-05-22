import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.component_ablation.run_component_ablation import fmt_float
from experiments.statistical_significance.run_significance_experiment import (
    CSV_HEADERS,
    DATASET_CHOICES,
)


COMPARISONS = [
    ("grace", "ifl-gr", "primary"),
    ("gca", "ifl-gc", "primary"),
    ("grace", "ifl-gc", "supplementary"),
    ("grace", "gca", "supplementary"),
]
METRICS = ["robust_score", "F1Mi_mean", "F1Ma_mean"]
METHOD_LABELS = {
    "grace": "GRACE",
    "gca": "GCA",
    "ifl-gr": "IFL-GR",
    "ifl-gc": "IFL-GC",
}


def read_rows(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def dataset_from_path(path):
    stem = Path(path).stem
    prefix = "significance_"
    suffix = "_results"
    slug = stem
    if stem.startswith(prefix):
        slug = stem[len(prefix) :]
    if slug.endswith(suffix):
        slug = slug[: -len(suffix)]
    lookup = {item.lower(): item for item in DATASET_CHOICES}
    return lookup.get(slug.lower(), slug)


def paired_values(rows, baseline_method, target_method, metric):
    by_method_run = {}
    for row in rows:
        if row.get("stage") != "run":
            continue
        method = row.get("method")
        if method not in {baseline_method, target_method}:
            continue
        run_idx = row.get("run_idx")
        value = safe_float(row.get(metric))
        if run_idx == "" or value is None:
            continue
        by_method_run[(method, str(run_idx))] = value

    baseline_runs = {run for method, run in by_method_run if method == baseline_method}
    target_runs = {run for method, run in by_method_run if method == target_method}
    shared_runs = sorted(baseline_runs & target_runs, key=lambda item: int(float(item)))

    baseline = np.array([by_method_run[(baseline_method, run)] for run in shared_runs], dtype=float)
    target = np.array([by_method_run[(target_method, run)] for run in shared_runs], dtype=float)
    return shared_runs, baseline, target


def bootstrap_ci(values, n_boot=10000, seed=12345):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None, None
    if values.size == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def cohen_dz(deltas):
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size < 2:
        return 0.0
    std = float(np.std(deltas, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(deltas) / std)


def rank_biserial_effect(deltas):
    deltas = np.asarray(deltas, dtype=float)
    nonzero = deltas[np.abs(deltas) > 1e-12]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    pos = float(ranks[nonzero > 0].sum())
    neg = float(ranks[nonzero < 0].sum())
    total = float(ranks.sum())
    if total <= 1e-12:
        return 0.0
    return (pos - neg) / total


def wilcoxon_pvalue(deltas):
    deltas = np.asarray(deltas, dtype=float)
    nonzero = deltas[np.abs(deltas) > 1e-12]
    if nonzero.size == 0:
        return 1.0
    try:
        return float(stats.wilcoxon(nonzero, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def paired_ttest_pvalue(deltas):
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size < 2 or float(np.std(deltas, ddof=1)) <= 1e-12:
        return 1.0
    return float(stats.ttest_1samp(deltas, popmean=0.0).pvalue)


def holm_bonferroni(p_values, alpha):
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(indexed, start=1):
        raw_adj = (m - rank + 1) * float(p_value)
        running_max = max(running_max, raw_adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def compute_test_rows(dataset, rows, alpha):
    test_rows = []
    for baseline_method, target_method, comparison_type in COMPARISONS:
        for metric in METRICS:
            shared_runs, baseline, target = paired_values(rows, baseline_method, target_method, metric)
            if len(shared_runs) < 2:
                continue

            deltas = target - baseline
            ci_low, ci_high = bootstrap_ci(deltas)
            p_wilcoxon = wilcoxon_pvalue(deltas)
            p_ttest = paired_ttest_pvalue(deltas)
            dz = cohen_dz(deltas)
            rb = rank_biserial_effect(deltas)

            test_rows.append(
                {
                    "timestamp": "",
                    "stage": "test",
                    "dataset": dataset,
                    "method": "",
                    "run_idx": "",
                    "seed": "",
                    "eval_seed": "",
                    "num_runs": "",
                    "F1Mi_mean": "",
                    "F1Mi_std": "",
                    "F1Ma_mean": "",
                    "F1Ma_std": "",
                    "robust_score": "",
                    "robust_score_std": "",
                    "params_json": "",
                    "metric": metric,
                    "baseline_method": baseline_method,
                    "target_method": target_method,
                    "test_name": "wilcoxon_signed_rank",
                    "n_pairs": len(shared_runs),
                    "mean_delta": fmt_float(float(np.mean(deltas))),
                    "median_delta": fmt_float(float(np.median(deltas))),
                    "ci95_low": fmt_float(ci_low),
                    "ci95_high": fmt_float(ci_high),
                    "p_value": fmt_float(p_wilcoxon),
                    "p_value_holm": "",
                    "p_value_paired_ttest": fmt_float(p_ttest),
                    "effect_size": fmt_float(rb),
                    "effect_size_name": "rank_biserial",
                    "cohen_dz": fmt_float(dz),
                    "rank_biserial": fmt_float(rb),
                    "significant": "",
                    "notes": comparison_type,
                }
            )
    apply_holm(test_rows, alpha)
    return test_rows


def apply_holm(test_rows, alpha):
    by_metric = defaultdict(list)
    for idx, row in enumerate(test_rows):
        by_metric[row["metric"]].append((idx, safe_float(row["p_value"]) or 1.0))

    for _metric, indexed_pvals in by_metric.items():
        indices = [idx for idx, _p in indexed_pvals]
        pvals = [p for _idx, p in indexed_pvals]
        adjusted = holm_bonferroni(pvals, alpha)
        for idx, p_adj in zip(indices, adjusted):
            row = test_rows[idx]
            mean_delta = safe_float(row["mean_delta"]) or 0.0
            row["p_value_holm"] = fmt_float(p_adj)
            row["significant"] = str(bool(p_adj < alpha and mean_delta > 0.0))


def build_report(all_test_rows, output_summary):
    lines = [
        "# Statistical Significance Analysis",
        "",
        "## Rule",
        "- Primary test: paired Wilcoxon signed-rank test over shared training seeds.",
        "- Holm-Bonferroni correction is applied within each metric.",
        "- A method is reported as significantly better only when adjusted p < 0.05 and mean delta > 0.",
        "",
        "## Results",
    ]

    if not all_test_rows:
        lines.append("- No valid paired test rows were generated.")
    else:
        ordered = sorted(
            all_test_rows,
            key=lambda row: (
                DATASET_CHOICES.index(row["dataset"]) if row["dataset"] in DATASET_CHOICES else 99,
                METRICS.index(row["metric"]) if row["metric"] in METRICS else 99,
                row["baseline_method"],
                row["target_method"],
            ),
        )
        for row in ordered:
            sig_text = "significant" if row["significant"] == "True" else "not significant"
            lines.append(
                "- "
                f"{row['dataset']} {row['metric']} "
                f"{METHOD_LABELS.get(row['target_method'], row['target_method'])} vs "
                f"{METHOD_LABELS.get(row['baseline_method'], row['baseline_method'])}: "
                f"delta={float(row['mean_delta']):+.4f}, "
                f"p_holm={float(row['p_value_holm']):.4f}, "
                f"{sig_text}."
            )

    lines.extend(["", "## Generated Files", f"- `{output_summary.as_posix()}`"])
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze paired-seed significance CSV files.")
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default=os.path.join("results", "plots"))
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(PROJECT_ROOT)
    input_paths = resolve_input_paths(repo_root, args.inputs)
    if not input_paths:
        raise RuntimeError("No significance CSV files were found.")

    all_test_rows = []
    for path in input_paths:
        dataset = dataset_from_path(path)
        rows = read_rows(path)
        base_rows = [row for row in rows if row.get("stage") != "test"]
        test_rows = compute_test_rows(dataset, base_rows, args.alpha)
        write_rows(path, base_rows + test_rows)
        all_test_rows.extend(test_rows)
        print(f"[significance-analysis] updated {path} with {len(test_rows)} test rows")

    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "significance_tests_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(all_test_rows)

    report_path = out_dir / "significance_analysis.md"
    report_path.write_text(build_report(all_test_rows, summary_csv), encoding="utf-8")
    print(f"[significance-analysis] saved summary: {summary_csv}")
    print(f"[significance-analysis] saved report: {report_path}")


if __name__ == "__main__":
    main()
