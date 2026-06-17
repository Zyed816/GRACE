import argparse
import copy
import csv
import json
import os
import sys
import tempfile
from datetime import datetime

import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.hyperparameter_sensitivity.run_sg_param_sensitivity import parse_trace_stats
from experiments.method_comparison.run_full_pipeline import robust_score, run_train


DATASET_CHOICES = ["Cora", "CiteSeer", "PubMed", "DBLP"]
METHOD_CHOICES = ["sg-gr", "sg-gc"]
METHOD_FILE_SLUG = {
    "sg-gr": "sggr",
    "sg-gc": "sggc",
}

VARIANT_SPECS = [
    {
        "variant": "full",
        "component": "complete SG-GCL",
        "notes": "complete method with best verified parameters",
    },
    {
        "variant": "no_warmup",
        "component": "w/o warmup",
        "notes": "warmup_epochs=0",
    },
    {
        "variant": "single_mining",
        "component": "w/o dynamic update",
        "notes": "update_interval=num_epochs+1",
    },
    {
        "variant": "uniform_weight",
        "component": "w/o semantic weight",
        "notes": "beta=0.0",
    },
]
VARIANT_ORDER = [spec["variant"] for spec in VARIANT_SPECS]
VARIANT_BY_NAME = {spec["variant"]: spec for spec in VARIANT_SPECS}

CSV_HEADERS = [
    "timestamp",
    "stage",
    "dataset",
    "method",
    "variant",
    "component",
    "run_idx",
    "seed",
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
    "base_params_json",
    "trial_params_json",
    "notes",
]


def dataset_slug(dataset):
    return dataset.lower()


def method_slug(method):
    return METHOD_FILE_SLUG[method]


def has_value(row, key):
    value = row.get(key)
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "none", "null", "nan"}


def to_float(value):
    return float(value)


def to_int(value):
    return int(float(value))


def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_mean(values):
    return sum(values) / len(values) if values else None


def safe_pstdev(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean_val = safe_mean(values)
    return (sum((value - mean_val) ** 2 for value in values) / len(values)) ** 0.5


def fmt_float(value, digits=6):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def fmt_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def summarize_error(exc):
    return str(exc).replace("\r", " ").replace("\n", " | ")[:1000]


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_result_row(csv_path, row=None, write_header=False):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        if row is not None:
            writer.writerow(row)


def parse_params_json(raw_value, source):
    if not raw_value:
        raise RuntimeError(f"Missing params_json in {source}")
    try:
        params = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid params_json in {source}: {exc}") from exc
    if not isinstance(params, dict):
        raise RuntimeError(f"params_json in {source} is not a JSON object")
    return params


def robust_from_row(row):
    return float(row["robust_score"])


def select_base_params_from_full_pipeline(grace_dir, dataset, method):
    csv_path = os.path.join(grace_dir, "results", f"{dataset_slug(dataset)}_full_pipeline_results.csv")
    if not os.path.exists(csv_path):
        return None

    rows = [
        row
        for row in read_csv_rows(csv_path)
        if row.get("stage") == "top_verify" and row.get("method") == method
    ]
    if not rows:
        return None

    grouped = {}
    for row in rows:
        rank = str(row.get("candidate_rank", "")).strip()
        if not rank:
            continue
        try:
            score = robust_from_row(row)
        except (KeyError, ValueError):
            continue
        grouped.setdefault(rank, []).append((score, row))

    if not grouped:
        return None

    def rank_key(item):
        rank, scored_rows = item
        scores = [score for score, _row in scored_rows]
        f1_values = []
        for _score, row in scored_rows:
            try:
                f1_values.append(float(row["F1Mi_mean"]))
            except (KeyError, ValueError):
                pass
        rank_num = int(rank) if rank.isdigit() else 9999
        return (safe_mean(scores) or float("-inf"), safe_mean(f1_values) or float("-inf"), -rank_num)

    best_rank, scored_rows = max(grouped.items(), key=rank_key)
    best_score = safe_mean([score for score, _row in scored_rows])
    best_row = scored_rows[0][1]
    params = parse_params_json(best_row.get("params_json", ""), f"{csv_path} candidate_rank={best_rank}")

    return {
        "params": params,
        "source": "full_pipeline",
        "source_path": os.path.relpath(csv_path, grace_dir).replace("\\", "/"),
        "candidate_rank": best_rank,
        "source_robust_score": best_score,
    }


def select_base_params_from_grid(grace_dir, dataset, method):
    csv_path = os.path.join(
        grace_dir,
        "results",
        f"grid_search_{method_slug(method)}_{dataset_slug(dataset)}_results.csv",
    )
    if not os.path.exists(csv_path):
        return None

    rows = read_csv_rows(csv_path)
    if not rows:
        return None

    params = dict(rows[0])
    return {
        "params": params,
        "source": "grid_search",
        "source_path": os.path.relpath(csv_path, grace_dir).replace("\\", "/"),
        "candidate_rank": "1",
        "source_robust_score": float(params["robust_score"]) if has_value(params, "robust_score") else None,
    }


def select_base_params(grace_dir, dataset, method):
    selected = select_base_params_from_full_pipeline(grace_dir, dataset, method)
    if selected is not None:
        return selected

    selected = select_base_params_from_grid(grace_dir, dataset, method)
    if selected is not None:
        return selected

    raise RuntimeError(
        f"No base parameters found for {dataset}/{method}. "
        "Run the method-comparison pipeline or grid search first."
    )


def params_to_dataset_updates(params, method):
    updates = {}

    def add_float(key):
        if has_value(params, key):
            updates[key] = to_float(params[key])

    def add_int(key):
        if has_value(params, key):
            updates[key] = to_int(params[key])

    def add_bool(key):
        if has_value(params, key):
            updates[key] = to_bool(params[key])

    add_float("similarity_percentile")
    add_int("max_du_per_node")
    add_float("unlabeled_weight")
    add_int("warmup_epochs")
    add_int("update_interval")
    add_float("beta")
    add_bool("use_mutual_topk")
    add_int("corrected_ramp_epochs")
    add_float("tau")

    if has_value(params, "similarity_threshold"):
        updates["similarity_threshold"] = to_float(params["similarity_threshold"])
    else:
        updates["similarity_threshold"] = None

    if method == "sg-gc":
        if has_value(params, "gca_drop_scheme"):
            updates["gca_drop_scheme"] = params["gca_drop_scheme"]
        add_float("sggc_refl_du_weight")
        add_float("drop_edge_rate_1")
        add_float("drop_edge_rate_2")
        add_float("drop_feature_rate_1")
        add_float("drop_feature_rate_2")
        add_int("gca_pr_k")
    elif method != "sg-gr":
        raise ValueError(f"Unsupported component-ablation method: {method}")

    return updates


def variant_overrides(base_config, dataset, variant):
    dataset_cfg = base_config[dataset]
    if variant == "full":
        return {}
    if variant == "no_warmup":
        return {"warmup_epochs": 0}
    if variant == "single_mining":
        return {"update_interval": int(dataset_cfg["num_epochs"]) + 1}
    if variant == "uniform_weight":
        return {"beta": 0.0}
    raise ValueError(f"Unsupported variant: {variant}")


def build_trial_updates(base_config, dataset, method, base_params, variant, run_idx):
    updates = params_to_dataset_updates(base_params, method)
    updates.update(variant_overrides(base_config, dataset, variant))
    base_seed = int(base_config[dataset].get("seed", 0))
    updates["seed"] = base_seed + int(run_idx)
    return updates


def make_temp_config(base_config, dataset, dataset_updates, temp_dir):
    cfg = copy.deepcopy(base_config)
    cfg[dataset].update(dataset_updates)

    os.makedirs(temp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".yaml",
        prefix=f"{dataset_slug(dataset)}_",
        dir=temp_dir,
        delete=False,
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        return f.name


def aggregate_records(records):
    f1mi_mean_values = [record["metrics"]["F1Mi_mean"] for record in records]
    f1mi_std_values = [record["metrics"]["F1Mi_std"] for record in records]
    f1ma_mean_values = [record["metrics"]["F1Ma_mean"] for record in records]
    f1ma_std_values = [record["metrics"]["F1Ma_std"] for record in records]
    robust_values = [record["robust"] for record in records]

    def trace_mean(key):
        return safe_mean([record["trace"][key] for record in records if record["trace"][key] is not None])

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
        "trace_ts_mean": trace_mean("trace_ts_mean"),
        "trace_ts_last": trace_mean("trace_ts_last"),
        "trace_mined_pairs_mean": trace_mean("trace_mined_pairs_mean"),
        "trace_mined_pairs_last": trace_mean("trace_mined_pairs_last"),
        "trace_avg_pairs_mean": trace_mean("trace_avg_pairs_mean"),
        "trace_avg_pairs_last": trace_mean("trace_avg_pairs_last"),
    }


def make_output_path(grace_dir, args, dataset):
    if args.out:
        return args.out if os.path.isabs(args.out) else os.path.join(grace_dir, args.out)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(grace_dir, args.out_dir)
    return os.path.join(out_dir, f"extra_ablation_{dataset_slug(dataset)}_results.csv")


def build_base_note(selected):
    robust_text = fmt_float(selected.get("source_robust_score"))
    return (
        f"base_source={selected['source']}; "
        f"base_path={selected['source_path']}; "
        f"base_rank={selected['candidate_rank']}; "
        f"base_robust={robust_text}"
    )


def write_run_row(csv_path, dataset, method, variant, run_idx, seed, metrics, trace, score, base_params, trial_updates, notes):
    spec = VARIANT_BY_NAME[variant]
    append_result_row(
        csv_path,
        row={
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": "run",
            "dataset": dataset,
            "method": method,
            "variant": variant,
            "component": spec["component"],
            "run_idx": run_idx,
            "seed": seed,
            "num_runs": "",
            "F1Mi_mean": fmt_float(metrics["F1Mi_mean"]),
            "F1Mi_std": fmt_float(metrics["F1Mi_std"]),
            "F1Ma_mean": fmt_float(metrics["F1Ma_mean"]),
            "F1Ma_std": fmt_float(metrics["F1Ma_std"]),
            "robust_score": fmt_float(score),
            "robust_score_std": "",
            "delta_vs_full": "",
            "drop_vs_full": "",
            "relative_drop_vs_full": "",
            "trace_ts_mean": fmt_float(trace["trace_ts_mean"]),
            "trace_ts_last": fmt_float(trace["trace_ts_last"]),
            "trace_mined_pairs_mean": fmt_float(trace["trace_mined_pairs_mean"]),
            "trace_mined_pairs_last": fmt_float(trace["trace_mined_pairs_last"]),
            "trace_avg_pairs_mean": fmt_float(trace["trace_avg_pairs_mean"]),
            "trace_avg_pairs_last": fmt_float(trace["trace_avg_pairs_last"]),
            "base_params_json": fmt_json(base_params),
            "trial_params_json": fmt_json(trial_updates),
            "notes": notes,
        },
    )


def write_failed_row(csv_path, dataset, method, variant, run_idx, seed, base_params, trial_updates, notes):
    spec = VARIANT_BY_NAME[variant]
    append_result_row(
        csv_path,
        row={
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": "run_failed",
            "dataset": dataset,
            "method": method,
            "variant": variant,
            "component": spec["component"],
            "run_idx": run_idx,
            "seed": seed,
            "num_runs": "",
            "F1Mi_mean": "",
            "F1Mi_std": "",
            "F1Ma_mean": "",
            "F1Ma_std": "",
            "robust_score": "",
            "robust_score_std": "",
            "delta_vs_full": "",
            "drop_vs_full": "",
            "relative_drop_vs_full": "",
            "trace_ts_mean": "",
            "trace_ts_last": "",
            "trace_mined_pairs_mean": "",
            "trace_mined_pairs_last": "",
            "trace_avg_pairs_mean": "",
            "trace_avg_pairs_last": "",
            "base_params_json": fmt_json(base_params),
            "trial_params_json": fmt_json(trial_updates),
            "notes": notes,
        },
    )


def write_summary_rows(csv_path, dataset, method, summaries, base_params, trial_updates_by_variant):
    full_robust = None
    if "full" in summaries:
        full_robust = summaries["full"]["robust_score"]

    for variant in VARIANT_ORDER:
        if variant not in summaries:
            continue
        spec = VARIANT_BY_NAME[variant]
        summary = summaries[variant]
        robust_value = summary["robust_score"]

        delta_vs_full = None
        drop_vs_full = None
        relative_drop_vs_full = None
        if full_robust is not None and robust_value is not None:
            delta_vs_full = robust_value - full_robust
            drop_vs_full = full_robust - robust_value
            if abs(full_robust) > 1e-12:
                relative_drop_vs_full = drop_vs_full / full_robust

        append_result_row(
            csv_path,
            row={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stage": "summary",
                "dataset": dataset,
                "method": method,
                "variant": variant,
                "component": spec["component"],
                "run_idx": "",
                "seed": "",
                "num_runs": summary["num_runs"],
                "F1Mi_mean": fmt_float(summary["F1Mi_mean"]),
                "F1Mi_std": fmt_float(summary["F1Mi_std"]),
                "F1Ma_mean": fmt_float(summary["F1Ma_mean"]),
                "F1Ma_std": fmt_float(summary["F1Ma_std"]),
                "robust_score": fmt_float(summary["robust_score"]),
                "robust_score_std": fmt_float(summary["robust_score_std"]),
                "delta_vs_full": fmt_float(delta_vs_full),
                "drop_vs_full": fmt_float(drop_vs_full),
                "relative_drop_vs_full": fmt_float(relative_drop_vs_full),
                "trace_ts_mean": fmt_float(summary["trace_ts_mean"]),
                "trace_ts_last": fmt_float(summary["trace_ts_last"]),
                "trace_mined_pairs_mean": fmt_float(summary["trace_mined_pairs_mean"]),
                "trace_mined_pairs_last": fmt_float(summary["trace_mined_pairs_last"]),
                "trace_avg_pairs_mean": fmt_float(summary["trace_avg_pairs_mean"]),
                "trace_avg_pairs_last": fmt_float(summary["trace_avg_pairs_last"]),
                "base_params_json": fmt_json(base_params),
                "trial_params_json": fmt_json(trial_updates_by_variant.get(variant, {})),
                "notes": spec["notes"],
            },
        )


def run_one_combo(grace_dir, base_config, dataset, method, args, out_csv):
    selected = select_base_params(grace_dir, dataset, method)
    base_params = selected["params"]
    base_note = build_base_note(selected)
    temp_dir = os.path.join(grace_dir, "results", "_component_ablation_tmp")

    print("=" * 90)
    print(f"[ablation] dataset={dataset} | method={method}")
    print(f"[ablation] {base_note}")

    records_by_variant = {variant: [] for variant in VARIANT_ORDER}
    trial_updates_by_variant = {}

    for variant in VARIANT_ORDER:
        spec = VARIANT_BY_NAME[variant]
        print(f"[ablation] variant={variant} | {spec['component']}")

        for run_idx in range(1, args.runs + 1):
            trial_updates = build_trial_updates(
                base_config=base_config,
                dataset=dataset,
                method=method,
                base_params=base_params,
                variant=variant,
                run_idx=run_idx,
            )
            summary_updates = dict(trial_updates)
            summary_updates.pop("seed", None)
            summary_updates["seed_list"] = []
            trial_updates_by_variant.setdefault(variant, summary_updates)
            seed = trial_updates["seed"]
            temp_cfg = None

            try:
                print(
                    f"  [{dataset}/{method}/{variant}] "
                    f"run {run_idx}/{args.runs} start | seed={seed}"
                )
                temp_cfg = make_temp_config(base_config, dataset, trial_updates, temp_dir)
                metrics, combined = run_train(
                    grace_dir,
                    temp_cfg,
                    dataset=dataset,
                    method=method,
                    gpu_id=args.gpu_id,
                    verbose_output=args.verbose_train_output,
                )
                trace = parse_trace_stats(combined)
                score = robust_score(metrics, args.std_weight)

                records_by_variant[variant].append(
                    {
                        "metrics": metrics,
                        "trace": trace,
                        "robust": score,
                    }
                )
                trial_updates_by_variant[variant]["seed_list"].append(seed)

                write_run_row(
                    csv_path=out_csv,
                    dataset=dataset,
                    method=method,
                    variant=variant,
                    run_idx=run_idx,
                    seed=seed,
                    metrics=metrics,
                    trace=trace,
                    score=score,
                    base_params=base_params,
                    trial_updates=trial_updates,
                    notes=base_note,
                )

                print(
                    f"    success: F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}, "
                    f"robust={score:.4f}"
                )
            except Exception as exc:
                error = summarize_error(exc)
                write_failed_row(
                    csv_path=out_csv,
                    dataset=dataset,
                    method=method,
                    variant=variant,
                    run_idx=run_idx,
                    seed=seed,
                    base_params=base_params,
                    trial_updates=trial_updates,
                    notes=f"{base_note}; error={error}",
                )
                print(f"    failed: {error}")
                if not args.continue_on_error:
                    raise
            finally:
                if temp_cfg and os.path.exists(temp_cfg):
                    os.remove(temp_cfg)

    summaries = {
        variant: aggregate_records(records)
        for variant, records in records_by_variant.items()
        if records
    }
    write_summary_rows(out_csv, dataset, method, summaries, base_params, trial_updates_by_variant)
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
    print(f"[ablation] saved rows for {dataset}/{method}: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SG-GCL component ablation experiments for SG-GR and SG-GC."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=DATASET_CHOICES, choices=DATASET_CHOICES)
    parser.add_argument("--methods", nargs="+", default=METHOD_CHOICES, choices=METHOD_CHOICES)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--std_weight", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--out", type=str, default=None, help="Custom output CSV. Only valid for one dataset.")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue after a failed training run and write run_failed rows.",
    )
    parser.add_argument(
        "--verbose_train_output",
        action="store_true",
        help="Stream raw train.py output for each ablation run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs <= 0:
        raise RuntimeError("--runs must be >= 1")

    datasets = list(dict.fromkeys(args.datasets))
    methods = list(dict.fromkeys(args.methods))
    if args.out and len(datasets) != 1:
        raise RuntimeError("--out can only be used with exactly one dataset.")

    grace_dir = PROJECT_ROOT
    config_path = args.config if os.path.isabs(args.config) else os.path.join(grace_dir, args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    for dataset in datasets:
        out_csv = make_output_path(grace_dir, args, dataset)
        if os.path.exists(out_csv):
            os.remove(out_csv)
        append_result_row(out_csv, row=None, write_header=True)

        for method in methods:
            run_one_combo(
                grace_dir=grace_dir,
                base_config=base_config,
                dataset=dataset,
                method=method,
                args=args,
                out_csv=out_csv,
            )

    print("[ablation] all requested component ablations finished")


if __name__ == "__main__":
    main()
