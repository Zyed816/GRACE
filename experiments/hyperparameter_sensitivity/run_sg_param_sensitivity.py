import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.method_comparison.run_full_pipeline import (
    make_temp_config_for_method,
    robust_score,
    run_train,
)


DATASET_CHOICES = ["Cora", "CiteSeer", "PubMed", "DBLP"]
METHOD_CHOICES = ["sg-gr", "sg-gc"]
METHOD_FILE_SLUG = {
    "sg-gr": "sggr",
    "sg-gc": "sggc",
}
PARAM_SPECS = {
    "t_s": {
        "config_key": "similarity_threshold",
        "default_step": 0.01,
        "cli_values_attr": "ts_values",
        "cli_step_attr": "ts_step",
        "paper_mapping": "t_s -> similarity_threshold",
    },
    "M": {
        "config_key": "warmup_epochs",
        "default_step": 20,
        "cli_values_attr": "m_values",
        "cli_step_attr": "m_step",
        "paper_mapping": "M -> warmup_epochs",
    },
    "K": {
        "config_key": "update_interval",
        "default_step": 1,
        "cli_values_attr": "k_values",
        "cli_step_attr": "k_step",
        "paper_mapping": "K -> update_interval",
    },
}
TRACE_PATTERN = re.compile(
    r"ts=(?P<ts>-?\d+\.\d+),\s+"
    r"mined_pairs=(?P<mined_pairs>\d+),\s+"
    r"avg_pairs_per_node=(?P<avg_pairs_per_node>\d+\.\d+)"
)
CSV_HEADERS = [
    "timestamp",
    "stage",
    "dataset",
    "method",
    "base_rank",
    "paper_param",
    "config_key",
    "anchor_value",
    "anchor_grid_robust",
    "sweep_value",
    "is_anchor",
    "run_idx",
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
    "top_params_csv",
    "base_params_json",
    "trial_params_json",
    "notes",
]


def safe_mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def safe_pstdev(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean_val = safe_mean(values)
    return (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5


def fmt_float(value, digits=6):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def normalize_method_slug(method):
    return METHOD_FILE_SLUG[method]


def read_ranked_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No ranked rows found in: {csv_path}")
    return rows


def parse_numeric_from_csv(raw_value, config_key):
    if config_key in ["max_du_per_node", "warmup_epochs", "update_interval"]:
        return int(float(raw_value))
    return float(raw_value)


def coerce_value(config_key, value, dataset_cfg):
    if config_key == "similarity_threshold":
        return round(min(max(float(value), -1.0), 1.0), 4)
    if config_key == "similarity_percentile":
        return round(min(max(float(value), 0.0), 100.0), 4)
    if config_key == "max_du_per_node":
        return max(1, int(round(float(value))))
    if config_key == "warmup_epochs":
        max_warmup = max(0, int(dataset_cfg["num_epochs"]) - 10)
        return min(max(0, int(round(float(value)))), max_warmup)
    if config_key == "update_interval":
        return max(1, int(round(float(value))))
    raise ValueError(f"Unsupported config key: {config_key}")


def infer_step(sorted_values, explicit_step, default_step):
    if explicit_step is not None:
        return explicit_step

    diffs = []
    for left, right in zip(sorted_values, sorted_values[1:]):
        diff = float(right) - float(left)
        if diff > 1e-12:
            diffs.append(diff)

    if diffs:
        return min(diffs)
    return default_step


def build_sweep_values(base_value, observed_values, dataset_cfg, spec, span, explicit_values, explicit_step):
    config_key = spec["config_key"]
    values = {coerce_value(config_key, base_value, dataset_cfg)}

    if explicit_values:
        for raw_value in explicit_values:
            values.add(coerce_value(config_key, raw_value, dataset_cfg))
        return sorted(values)

    sorted_observed = sorted(set(observed_values))
    step = infer_step(sorted_observed, explicit_step, spec["default_step"])

    low = float(base_value) - float(step) * max(0, span)
    high = float(base_value) + float(step) * max(0, span)

    for observed in sorted_observed:
        if low - 1e-12 <= float(observed) <= high + 1e-12:
            values.add(coerce_value(config_key, observed, dataset_cfg))

    for delta in range(-max(0, span), max(0, span) + 1):
        candidate = float(base_value) + float(delta) * float(step)
        values.add(coerce_value(config_key, candidate, dataset_cfg))

    return sorted(values)


def has_explicit_csv_value(row, config_key):
    raw_value = row.get(config_key, "")
    if raw_value is None:
        return False
    text = str(raw_value).strip().lower()
    return text not in ["", "none", "null"]


def parse_trace_stats(output_text):
    ts_values = []
    mined_pairs = []
    avg_pairs = []

    for match in TRACE_PATTERN.finditer(output_text):
        ts_values.append(float(match.group("ts")))
        mined_pairs.append(float(match.group("mined_pairs")))
        avg_pairs.append(float(match.group("avg_pairs_per_node")))

    return {
        "trace_ts_mean": safe_mean(ts_values),
        "trace_ts_last": ts_values[-1] if ts_values else None,
        "trace_mined_pairs_mean": safe_mean(mined_pairs),
        "trace_mined_pairs_last": mined_pairs[-1] if mined_pairs else None,
        "trace_avg_pairs_mean": safe_mean(avg_pairs),
        "trace_avg_pairs_last": avg_pairs[-1] if avg_pairs else None,
    }


def append_result_row(csv_path, row, write_header=False):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        if row is not None:
            writer.writerow(row)


def make_notes(param_name):
    return PARAM_SPECS[param_name]["paper_mapping"]


def aggregate_successful_runs(records):
    f1mi_mean_values = [item["metrics"]["F1Mi_mean"] for item in records]
    f1mi_std_values = [item["metrics"]["F1Mi_std"] for item in records]
    f1ma_mean_values = [item["metrics"]["F1Ma_mean"] for item in records]
    f1ma_std_values = [item["metrics"]["F1Ma_std"] for item in records]
    robust_values = [item["robust"] for item in records]

    return {
        "num_runs": len(records),
        "F1Mi_mean": safe_mean(f1mi_mean_values),
        "F1Mi_std": safe_pstdev(f1mi_mean_values),
        "F1Ma_mean": safe_mean(f1ma_mean_values),
        "F1Ma_std": safe_pstdev(f1ma_mean_values),
        "within_run_F1Mi_std_mean": safe_mean(f1mi_std_values),
        "within_run_F1Ma_std_mean": safe_mean(f1ma_std_values),
        "robust_score": safe_mean(robust_values),
        "robust_score_std": safe_pstdev(robust_values),
        "trace_ts_mean": safe_mean(
            [item["trace"]["trace_ts_mean"] for item in records if item["trace"]["trace_ts_mean"] is not None]
        ),
        "trace_ts_last": safe_mean(
            [item["trace"]["trace_ts_last"] for item in records if item["trace"]["trace_ts_last"] is not None]
        ),
        "trace_mined_pairs_mean": safe_mean(
            [item["trace"]["trace_mined_pairs_mean"] for item in records if item["trace"]["trace_mined_pairs_mean"] is not None]
        ),
        "trace_mined_pairs_last": safe_mean(
            [item["trace"]["trace_mined_pairs_last"] for item in records if item["trace"]["trace_mined_pairs_last"] is not None]
        ),
        "trace_avg_pairs_mean": safe_mean(
            [item["trace"]["trace_avg_pairs_mean"] for item in records if item["trace"]["trace_avg_pairs_mean"] is not None]
        ),
        "trace_avg_pairs_last": safe_mean(
            [item["trace"]["trace_avg_pairs_last"] for item in records if item["trace"]["trace_avg_pairs_last"] is not None]
        ),
    }


def estimate_anchor_threshold(grace_dir, base_config, dataset, method, base_row, gpu_id, std_weight):
    anchor_row = dict(base_row)
    anchor_row["similarity_threshold"] = "None"
    temp_cfg = make_temp_config_for_method(base_config, dataset, anchor_row, method)

    try:
        metrics, combined = run_train(
            grace_dir,
            temp_cfg,
            dataset=dataset,
            method=method,
            gpu_id=gpu_id,
        )
    finally:
        if os.path.exists(temp_cfg):
            os.remove(temp_cfg)

    trace = parse_trace_stats(combined)
    estimated = trace["trace_ts_mean"]
    if estimated is None:
        raise RuntimeError(
            f"Failed to infer anchor t_s from training trace for {dataset}/{method}. "
            "The base grid-search row does not contain similarity_threshold, so an "
            "anchor threshold must be inferred from the logged active threshold."
        )

    return {
        "anchor_threshold": estimated,
        "metrics": metrics,
        "trace": trace,
        "robust": robust_score(metrics, std_weight),
    }


def default_grid_csv_path(grace_dir, dataset, method):
    dataset_slug = dataset.lower()
    method_slug = normalize_method_slug(method)
    return os.path.join(grace_dir, "results", f"grid_search_{method_slug}_{dataset_slug}_results.csv")


def default_out_csv_path(grace_dir, dataset, method):
    dataset_slug = dataset.lower()
    method_slug = normalize_method_slug(method)
    return os.path.join(grace_dir, "results", f"sensitivity_{method_slug}_{dataset_slug}_results.csv")


def summarize_error(exc):
    text = str(exc).replace("\r", " ").replace("\n", " | ")
    return text[:1000]


def run_one_combo(grace_dir, base_config, dataset, method, args, top_params_override=None, out_override=None):
    dataset_cfg = base_config.get(dataset, {})
    if not dataset_cfg:
        raise RuntimeError(f"Dataset config not found: {dataset}")

    top_params_csv = top_params_override if top_params_override else default_grid_csv_path(grace_dir, dataset, method)
    if not os.path.isabs(top_params_csv):
        top_params_csv = os.path.join(grace_dir, top_params_csv)
    if not os.path.exists(top_params_csv):
        raise RuntimeError(
            f"Grid-search result not found for {dataset}/{method}: {top_params_csv}. "
            "Please run the corresponding grid-search script first."
        )

    ranked_rows = read_ranked_rows(top_params_csv)
    if args.base_rank <= 0 or args.base_rank > len(ranked_rows):
        raise RuntimeError(
            f"base_rank={args.base_rank} is out of range for {top_params_csv}; "
            f"available rows: 1..{len(ranked_rows)}"
        )

    base_row = dict(ranked_rows[args.base_rank - 1])
    out_csv = out_override if out_override else default_out_csv_path(grace_dir, dataset, method)
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(grace_dir, out_csv)

    if os.path.exists(out_csv):
        os.remove(out_csv)
    append_result_row(out_csv, row=None, write_header=True)

    base_params_json = json.dumps(base_row, ensure_ascii=True)
    top_params_rel = os.path.relpath(top_params_csv, grace_dir).replace("\\", "/")
    estimated_anchor = None

    print("=" * 90)
    print(f"[sensitivity] dataset={dataset} | method={method} | base_rank={args.base_rank}")
    print(f"[sensitivity] top params csv: {top_params_csv}")
    print(f"[sensitivity] output csv: {out_csv}")

    for param_name in args.params:
        spec = PARAM_SPECS[param_name]
        config_key = spec["config_key"]
        if has_explicit_csv_value(base_row, config_key):
            base_value = coerce_value(
                config_key,
                parse_numeric_from_csv(base_row[config_key], config_key),
                dataset_cfg,
            )
        elif config_key == "similarity_threshold":
            if estimated_anchor is None:
                print(
                    f"[sensitivity] {dataset}/{method} | estimating anchor t_s from "
                    "the active threshold trace of the best grid-search configuration"
                )
                estimated_anchor = estimate_anchor_threshold(
                    grace_dir=grace_dir,
                    base_config=base_config,
                    dataset=dataset,
                    method=method,
                    base_row=base_row,
                    gpu_id=args.gpu_id,
                    std_weight=args.std_weight,
                )
            base_value = coerce_value(config_key, estimated_anchor["anchor_threshold"], dataset_cfg)
        else:
            raise RuntimeError(
                f"Cannot resolve anchor value for {param_name} ({config_key}) from top params CSV: {top_params_csv}"
            )
        observed_values = [
            coerce_value(config_key, parse_numeric_from_csv(row[config_key], config_key), dataset_cfg)
            for row in ranked_rows
            if has_explicit_csv_value(row, config_key)
        ]
        explicit_values = getattr(args, spec["cli_values_attr"])
        explicit_step = getattr(args, spec["cli_step_attr"])
        sweep_values = build_sweep_values(
            base_value=base_value,
            observed_values=observed_values,
            dataset_cfg=dataset_cfg,
            spec=spec,
            span=args.neighbor_span,
            explicit_values=explicit_values,
            explicit_step=explicit_step,
        )
        anchor_grid_robust = float(base_row.get("robust_score", 0.0))

        print(
            f"[sensitivity] {dataset}/{method} | vary {param_name} ({config_key}) | "
            f"anchor={base_value} | sweep={sweep_values}"
        )

        records_by_value = {}

        for sweep_value in sweep_values:
            trial_row = dict(base_row)
            trial_row[config_key] = sweep_value
            if config_key == "similarity_threshold":
                trial_row["similarity_threshold"] = str(sweep_value)

            records_by_value[sweep_value] = []
            for run_idx in range(1, args.runs + 1):
                print(
                    f"  [{dataset}/{method}] {param_name}={sweep_value} "
                    f"run {run_idx}/{args.runs} start"
                )
                temp_cfg = make_temp_config_for_method(base_config, dataset, trial_row, method)

                try:
                    metrics, combined = run_train(
                        grace_dir,
                        temp_cfg,
                        dataset=dataset,
                        method=method,
                        gpu_id=args.gpu_id,
                    )
                    trace = parse_trace_stats(combined)
                    score = robust_score(metrics, args.std_weight)
                    records_by_value[sweep_value].append({
                        "metrics": metrics,
                        "trace": trace,
                        "robust": score,
                    })

                    append_result_row(
                        out_csv,
                        row={
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "stage": "run",
                            "dataset": dataset,
                            "method": method,
                            "base_rank": args.base_rank,
                            "paper_param": param_name,
                            "config_key": config_key,
                            "anchor_value": base_value,
                            "anchor_grid_robust": fmt_float(anchor_grid_robust),
                            "sweep_value": sweep_value,
                            "is_anchor": str(sweep_value == base_value),
                            "run_idx": run_idx,
                            "num_runs": "",
                            "F1Mi_mean": fmt_float(metrics["F1Mi_mean"]),
                            "F1Mi_std": fmt_float(metrics["F1Mi_std"]),
                            "F1Ma_mean": fmt_float(metrics["F1Ma_mean"]),
                            "F1Ma_std": fmt_float(metrics["F1Ma_std"]),
                            "within_run_F1Mi_std_mean": "",
                            "within_run_F1Ma_std_mean": "",
                            "robust_score": fmt_float(score),
                            "robust_score_std": "",
                            "trace_ts_mean": fmt_float(trace["trace_ts_mean"]),
                            "trace_ts_last": fmt_float(trace["trace_ts_last"]),
                            "trace_mined_pairs_mean": fmt_float(trace["trace_mined_pairs_mean"]),
                            "trace_mined_pairs_last": fmt_float(trace["trace_mined_pairs_last"]),
                            "trace_avg_pairs_mean": fmt_float(trace["trace_avg_pairs_mean"]),
                            "trace_avg_pairs_last": fmt_float(trace["trace_avg_pairs_last"]),
                            "delta_vs_anchor": "",
                            "top_params_csv": top_params_rel,
                            "base_params_json": base_params_json,
                            "trial_params_json": json.dumps(trial_row, ensure_ascii=True),
                            "notes": (
                                f"{make_notes(param_name)}"
                                if config_key != "similarity_threshold"
                                else f"{make_notes(param_name)} | anchor inferred from trace_ts_mean"
                            ),
                        },
                    )

                    print(
                        f"    success: F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}, "
                        f"robust={score:.4f}"
                    )
                except Exception as exc:
                    append_result_row(
                        out_csv,
                        row={
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "stage": "run_failed",
                            "dataset": dataset,
                            "method": method,
                            "base_rank": args.base_rank,
                            "paper_param": param_name,
                            "config_key": config_key,
                            "anchor_value": base_value,
                            "anchor_grid_robust": fmt_float(anchor_grid_robust),
                            "sweep_value": sweep_value,
                            "is_anchor": str(sweep_value == base_value),
                            "run_idx": run_idx,
                            "num_runs": "",
                            "F1Mi_mean": "",
                            "F1Mi_std": "",
                            "F1Ma_mean": "",
                            "F1Ma_std": "",
                            "within_run_F1Mi_std_mean": "",
                            "within_run_F1Ma_std_mean": "",
                            "robust_score": "",
                            "robust_score_std": "",
                            "trace_ts_mean": "",
                            "trace_ts_last": "",
                            "trace_mined_pairs_mean": "",
                            "trace_mined_pairs_last": "",
                            "trace_avg_pairs_mean": "",
                            "trace_avg_pairs_last": "",
                            "delta_vs_anchor": "",
                            "top_params_csv": top_params_rel,
                            "base_params_json": base_params_json,
                            "trial_params_json": json.dumps(trial_row, ensure_ascii=True),
                            "notes": summarize_error(exc),
                        },
                    )
                    print(f"    failed: {summarize_error(exc)}")
                    if not args.continue_on_error:
                        raise
                finally:
                    if os.path.exists(temp_cfg):
                        os.remove(temp_cfg)

        summary_by_value = {}
        for sweep_value, records in records_by_value.items():
            if records:
                summary_by_value[sweep_value] = aggregate_successful_runs(records)

        anchor_summary = summary_by_value.get(base_value)
        anchor_robust = anchor_summary["robust_score"] if anchor_summary else None

        for sweep_value in sweep_values:
            summary = summary_by_value.get(sweep_value)
            if summary is None:
                continue

            delta_vs_anchor = None
            if anchor_robust is not None and summary["robust_score"] is not None:
                delta_vs_anchor = summary["robust_score"] - anchor_robust

            append_result_row(
                out_csv,
                row={
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stage": "summary",
                    "dataset": dataset,
                    "method": method,
                    "base_rank": args.base_rank,
                    "paper_param": param_name,
                    "config_key": config_key,
                    "anchor_value": base_value,
                    "anchor_grid_robust": fmt_float(anchor_grid_robust),
                    "sweep_value": sweep_value,
                    "is_anchor": str(sweep_value == base_value),
                    "run_idx": "",
                    "num_runs": summary["num_runs"],
                    "F1Mi_mean": fmt_float(summary["F1Mi_mean"]),
                    "F1Mi_std": fmt_float(summary["F1Mi_std"]),
                    "F1Ma_mean": fmt_float(summary["F1Ma_mean"]),
                    "F1Ma_std": fmt_float(summary["F1Ma_std"]),
                    "within_run_F1Mi_std_mean": fmt_float(summary["within_run_F1Mi_std_mean"]),
                    "within_run_F1Ma_std_mean": fmt_float(summary["within_run_F1Ma_std_mean"]),
                    "robust_score": fmt_float(summary["robust_score"]),
                    "robust_score_std": fmt_float(summary["robust_score_std"]),
                    "trace_ts_mean": fmt_float(summary["trace_ts_mean"]),
                    "trace_ts_last": fmt_float(summary["trace_ts_last"]),
                    "trace_mined_pairs_mean": fmt_float(summary["trace_mined_pairs_mean"]),
                    "trace_mined_pairs_last": fmt_float(summary["trace_mined_pairs_last"]),
                    "trace_avg_pairs_mean": fmt_float(summary["trace_avg_pairs_mean"]),
                    "trace_avg_pairs_last": fmt_float(summary["trace_avg_pairs_last"]),
                    "delta_vs_anchor": fmt_float(delta_vs_anchor),
                    "top_params_csv": top_params_rel,
                    "base_params_json": base_params_json,
                    "trial_params_json": json.dumps({config_key: sweep_value}, ensure_ascii=True),
                    "notes": (
                        f"{make_notes(param_name)}"
                        if config_key != "similarity_threshold"
                        else f"{make_notes(param_name)} | anchor inferred from trace_ts_mean"
                    ),
                },
            )

        if anchor_robust is None:
            print(f"[sensitivity] warning: no successful anchor runs for {dataset}/{method}/{param_name}")
        else:
            print(
                f"[sensitivity] {dataset}/{method}/{param_name} anchor mean robust="
                f"{anchor_robust:.4f}"
            )

    print(f"[sensitivity] saved: {out_csv}")
    return out_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis for SG-GR / SG-GC. "
            "Starting from a ranked grid-search CSV, keep the best parameter set fixed "
            "and vary one paper hyper-parameter at a time."
        )
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=DATASET_CHOICES, choices=DATASET_CHOICES)
    parser.add_argument("--methods", nargs="+", default=METHOD_CHOICES, choices=METHOD_CHOICES)
    parser.add_argument("--params", nargs="+", default=["t_s", "M", "K"], choices=["t_s", "M", "K"])
    parser.add_argument("--base_rank", type=int, default=1, help="Which ranked grid-search row to use as anchor.")
    parser.add_argument("--runs", type=int, default=3, help="Number of reruns for each sweep value.")
    parser.add_argument("--std_weight", type=float, default=0.5)
    parser.add_argument("--neighbor_span", type=int, default=1, help="Sweep anchor +/- span * inferred_step.")
    parser.add_argument("--top_params", type=str, default=None, help="Custom grid CSV. Only valid for one dataset + one method.")
    parser.add_argument("--out", type=str, default=None, help="Custom output CSV. Only valid for one dataset + one method.")
    parser.add_argument("--ts_values", nargs="+", type=float, default=None, help="Explicit sweep values for t_s.")
    parser.add_argument("--m_values", nargs="+", type=int, default=None, help="Explicit sweep values for M.")
    parser.add_argument("--k_values", nargs="+", type=int, default=None, help="Explicit sweep values for K.")
    parser.add_argument("--ts_step", type=float, default=None, help="Override inferred step for t_s.")
    parser.add_argument("--m_step", type=int, default=None, help="Override inferred step for M.")
    parser.add_argument("--k_step", type=int, default=None, help="Override inferred step for K.")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to the next run / dataset / method after a training failure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.runs <= 0:
        raise RuntimeError("--runs must be >= 1")
    if args.neighbor_span < 0:
        raise RuntimeError("--neighbor_span must be >= 0")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    grace_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    config_path = os.path.join(grace_dir, args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    datasets = list(dict.fromkeys(args.datasets))
    methods = list(dict.fromkeys(args.methods))
    params = list(dict.fromkeys(args.params))
    args.params = params

    single_combo = len(datasets) == 1 and len(methods) == 1
    if (args.top_params or args.out) and not single_combo:
        raise RuntimeError("--top_params and --out can only be used with one dataset and one method.")

    failures = []
    for dataset in datasets:
        for method in methods:
            try:
                run_one_combo(
                    grace_dir=grace_dir,
                    base_config=base_config,
                    dataset=dataset,
                    method=method,
                    args=args,
                    top_params_override=args.top_params if single_combo else None,
                    out_override=os.path.join(grace_dir, args.out) if (single_combo and args.out) else None,
                )
            except Exception as exc:
                failures.append((dataset, method, exc))
                print(f"[sensitivity] FAILED dataset={dataset} method={method}: {summarize_error(exc)}")
                if not args.continue_on_error:
                    raise

    if failures:
        print("=" * 90)
        for dataset, method, exc in failures:
            print(f"[sensitivity] failed: dataset={dataset}, method={method}, error={summarize_error(exc)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
