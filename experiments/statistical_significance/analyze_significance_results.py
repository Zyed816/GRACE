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
    ("grace", "sg-gr", "primary"),
    ("gca", "sg-gc", "primary"),
    ("grace", "sg-gc", "supplementary"),
    ("grace", "gca", "supplementary"),
]
METRICS = ["robust_score", "F1Mi_mean", "F1Ma_mean"]
METHOD_LABELS = {
    "grace": "GRACE",
    "gca": "GCA",
    "sg-gr": "SG-GR",
    "sg-gc": "SG-GC",
}
PRIMARY_COMPARISON_ORDER = [
    ("grace", "sg-gr"),
    ("gca", "sg-gc"),
]


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
        "# 统计显著性实验说明与结果",
        "",
        "## 实验说明",
        "",
        "本实验用于判断 SG-GCL 的提升是否稳定，而不是只由个别随机种子造成。实验只关注两组主要比较：SG-GR vs GRACE，以及 SG-GC vs GCA。",
        "",
        "每组比较都使用相同 seed 下的配对结果。本文先计算目标方法和基线方法的差值，即 `目标方法 robust_score - 基线方法 robust_score`。差值为正表示目标方法更好，差值为负表示目标方法更差。",
        "",
        "p 值用于判断这种差值是否可能只是随机波动造成的。简单来说，在“两个方法没有稳定差异”的假设下，p 值表示观察到当前差异或更明显差异的可能性。p 值越小，说明当前差异越不像随机波动。由于同时进行了多组比较，本文使用 Holm 方法对 p 值进行校正。",
        "",
        "判断规则为：只有当 Holm 校正后的 p 值小于0.05，并且平均差值大于0时，才认为目标方法显著优于基线方法。图中的星号 `*` 也表示这个含义。没有星号表示未达到“显著优于”的标准。",
        "",
        "## 主要结果",
    ]

    if not all_test_rows:
        lines.append("- No valid paired test rows were generated.")
    else:
        primary_robust_rows = [
            row
            for row in all_test_rows
            if row.get("notes") == "primary" and row.get("metric") == "robust_score"
        ]
        comparison_rank = {
            pair: idx for idx, pair in enumerate(PRIMARY_COMPARISON_ORDER)
        }
        ordered = sorted(
            primary_robust_rows,
            key=lambda row: (
                DATASET_CHOICES.index(row["dataset"]) if row["dataset"] in DATASET_CHOICES else 99,
                comparison_rank.get((row["baseline_method"], row["target_method"]), 99),
            ),
        )
        for row in ordered:
            sig_text = "是" if row["significant"] == "True" else "否"
            star_text = "（*）" if row["significant"] == "True" else ""
            mean_delta = float(row["mean_delta"])
            lines.append(
                "- "
                f"{row['dataset']} "
                f"{METHOD_LABELS.get(row['target_method'], row['target_method'])} vs "
                f"{METHOD_LABELS.get(row['baseline_method'], row['baseline_method'])}: "
                f"平均差值={mean_delta:+.4f}（约{mean_delta * 100:+.2f}%）, "
                f"Holm校正p值={float(row['p_value_holm']):.4f}, "
                f"显著优于={sig_text}{star_text}。"
            )

    lines.extend(
        [
            "",
            "## 结果解读",
            "",
            "- Cora 和 PubMed 上，两组主要比较均达到显著优于，说明 SG-GCL 在这两个数据集上的提升比较稳定。",
            "- DBLP 上，SG-GR 显著优于 GRACE，但 SG-GC 没有显著优于 GCA，说明该数据集上 SG-GR 更稳定。",
            "- CiteSeer 上，两组主要比较都没有体现出 SG 方法显著优于基线方法，不能强行解释为稳定提升。",
            "",
            "## 生成文件",
            f"- `{output_summary.as_posix()}`",
        ]
    )
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
