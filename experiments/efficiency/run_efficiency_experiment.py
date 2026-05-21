import argparse
import csv
import os
import re
import sys
from datetime import datetime
from time import perf_counter as t

import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.component_ablation.run_component_ablation import (
    dataset_slug,
    fmt_float,
    fmt_json,
    has_value,
    make_temp_config,
    parse_params_json,
    params_to_dataset_updates as ifl_params_to_dataset_updates,
    read_csv_rows,
    safe_mean,
    safe_pstdev,
    to_float,
    to_int,
)
from experiments.method_comparison.run_full_pipeline import robust_score, run_train


DATASET_CHOICES = ["Cora", "CiteSeer", "PubMed", "DBLP"]
METHOD_CHOICES = ["grace", "gca", "ifl-gr", "ifl-gc"]
METHOD_FILE_SLUG = {
    "gca": "gca",
    "ifl-gr": "iflgr",
    "ifl-gc": "iflgc",
}
BASE_METHOD = {
    "grace": "",
    "gca": "grace",
    "ifl-gr": "grace",
    "ifl-gc": "gca",
}

EPOCH_TIMING_PATTERN = re.compile(
    r"\(T\) \| Epoch=(?P<epoch>\d+).*?"
    r"phase=(?P<phase>[^,]+).*?"
    r"this epoch (?P<epoch_time>\d+(?:\.\d+)?), "
    r"total (?P<total_time>\d+(?:\.\d+)?)"
)
REFRESH_PATTERN = re.compile(r"refresh_du=(?P<refresh>[01])")

CSV_HEADERS = [
    "timestamp",
    "stage",
    "dataset",
    "method",
    "base_method",
    "run_idx",
    "seed",
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
    "base_params_json",
    "trial_params_json",
    "notes",
]


def append_result_row(csv_path, row=None, write_header=False):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        if row is not None:
            writer.writerow(row)


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summarize_error(exc):
    return str(exc).replace("\r", " ").replace("\n", " | ")[:1000]


def robust_from_row(row):
    return float(row["robust_score"])


def select_base_params_from_full_pipeline(grace_dir, dataset, method):
    csv_path = os.path.join(grace_dir, "results", f"{dataset_slug(dataset)}_full_pipeline_results.csv")
    if not os.path.exists(csv_path):
        return None

    if method == "grace":
        rows = [
            row
            for row in read_csv_rows(csv_path)
            if row.get("stage") == "baseline" and row.get("method") == "grace"
        ]
    else:
        rows = [
            row
            for row in read_csv_rows(csv_path)
            if row.get("stage") == "top_verify" and row.get("method") == method
        ]
    if not rows:
        return None

    grouped = {}
    for row in rows:
        rank = str(row.get("candidate_rank", "0" if method == "grace" else "")).strip()
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
        rank_num = int(rank) if rank.isdigit() else 9999
        return (safe_mean(scores) or float("-inf"), -rank_num)

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
    if method == "grace":
        return None

    csv_path = os.path.join(
        grace_dir,
        "results",
        f"grid_search_{METHOD_FILE_SLUG[method]}_{dataset_slug(dataset)}_results.csv",
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

    if method == "grace":
        return {
            "params": {},
            "source": "config_default",
            "source_path": "config.yaml",
            "candidate_rank": "0",
            "source_robust_score": None,
        }

    raise RuntimeError(
        f"No base parameters found for {dataset}/{method}. "
        "Run the method-comparison pipeline or grid search first."
    )


def add_float_if_present(updates, params, key):
    if has_value(params, key):
        updates[key] = to_float(params[key])


def add_int_if_present(updates, params, key):
    if has_value(params, key):
        updates[key] = to_int(params[key])


def params_to_dataset_updates(params, method):
    if method in {"ifl-gr", "ifl-gc"}:
        return ifl_params_to_dataset_updates(params, method)

    updates = {}
    add_float_if_present(updates, params, "drop_edge_rate_1")
    add_float_if_present(updates, params, "drop_edge_rate_2")
    add_float_if_present(updates, params, "drop_feature_rate_1")
    add_float_if_present(updates, params, "drop_feature_rate_2")
    add_float_if_present(updates, params, "tau")

    if method == "gca":
        if has_value(params, "gca_drop_scheme"):
            updates["gca_drop_scheme"] = params["gca_drop_scheme"]
        add_int_if_present(updates, params, "gca_pr_k")
    elif method != "grace":
        raise ValueError(f"Unsupported efficiency method: {method}")

    return updates


def build_trial_updates(base_config, dataset, method, base_params, run_idx):
    updates = params_to_dataset_updates(base_params, method)
    base_seed = int(base_config[dataset].get("seed", 0))
    updates["seed"] = base_seed + int(run_idx)
    return updates


def parse_timing_stats(output_text):
    epochs = []
    epoch_times = []
    total_times = []
    refresh_epoch_times = []
    non_refresh_epoch_times = []
    warmup_epoch_times = []
    corrected_epoch_times = []
    refresh_count = 0

    for line in output_text.splitlines():
        match = EPOCH_TIMING_PATTERN.search(line)
        if not match:
            continue

        epoch = int(match.group("epoch"))
        phase = match.group("phase").strip()
        epoch_time = float(match.group("epoch_time"))
        total_time = float(match.group("total_time"))
        refresh_match = REFRESH_PATTERN.search(line)
        is_refresh = bool(refresh_match and refresh_match.group("refresh") == "1")

        epochs.append(epoch)
        epoch_times.append(epoch_time)
        total_times.append(total_time)
        if is_refresh:
            refresh_count += 1
            refresh_epoch_times.append(epoch_time)
        else:
            non_refresh_epoch_times.append(epoch_time)

        if phase in {"warmup", "warmup-gca"}:
            warmup_epoch_times.append(epoch_time)
        if phase in {"corrected", "corrected-gca"}:
            corrected_epoch_times.append(epoch_time)

    train_total = total_times[-1] if total_times else None
    return {
        "epochs_observed": len(epochs),
        "train_total_sec": train_total,
        "epoch_time_mean_sec": safe_mean(epoch_times),
        "epoch_time_std_sec": safe_pstdev(epoch_times),
        "epoch_time_last_sec": epoch_times[-1] if epoch_times else None,
        "epoch_time_median_sec": median(epoch_times),
        "throughput_epoch_per_sec": (len(epoch_times) / train_total) if train_total and train_total > 0 else None,
        "refresh_count": refresh_count,
        "refresh_epoch_time_mean_sec": safe_mean(refresh_epoch_times),
        "non_refresh_epoch_time_mean_sec": safe_mean(non_refresh_epoch_times),
        "warmup_epoch_time_mean_sec": safe_mean(warmup_epoch_times),
        "corrected_epoch_time_mean_sec": safe_mean(corrected_epoch_times),
    }


def aggregate_records(records):
    def metric_mean(key):
        return safe_mean([record[key] for record in records if record[key] is not None])

    f1mi_mean_values = [record["metrics"]["F1Mi_mean"] for record in records]
    f1mi_std_values = [record["metrics"]["F1Mi_std"] for record in records]
    f1ma_mean_values = [record["metrics"]["F1Ma_mean"] for record in records]
    f1ma_std_values = [record["metrics"]["F1Ma_std"] for record in records]
    robust_values = [record["robust_score"] for record in records]
    wall_times = [record["wall_time_sec"] for record in records]
    train_totals = [record["timing"]["train_total_sec"] for record in records if record["timing"]["train_total_sec"] is not None]

    return {
        "num_runs": len(records),
        "num_epochs_config": records[0]["num_epochs_config"] if records else None,
        "epochs_observed": metric_mean("epochs_observed"),
        "wall_time_sec": safe_mean(wall_times),
        "wall_time_std_sec": safe_pstdev(wall_times),
        "train_total_sec": safe_mean(train_totals),
        "train_total_std_sec": safe_pstdev(train_totals),
        "eval_overhead_sec": metric_mean("eval_overhead_sec"),
        "epoch_time_mean_sec": metric_mean("epoch_time_mean_sec"),
        "epoch_time_std_sec": metric_mean("epoch_time_std_sec"),
        "epoch_time_last_sec": metric_mean("epoch_time_last_sec"),
        "epoch_time_median_sec": metric_mean("epoch_time_median_sec"),
        "throughput_epoch_per_sec": metric_mean("throughput_epoch_per_sec"),
        "refresh_count": metric_mean("refresh_count"),
        "refresh_epoch_time_mean_sec": metric_mean("refresh_epoch_time_mean_sec"),
        "non_refresh_epoch_time_mean_sec": metric_mean("non_refresh_epoch_time_mean_sec"),
        "warmup_epoch_time_mean_sec": metric_mean("warmup_epoch_time_mean_sec"),
        "corrected_epoch_time_mean_sec": metric_mean("corrected_epoch_time_mean_sec"),
        "F1Mi_mean": safe_mean(f1mi_mean_values),
        "F1Mi_std": safe_pstdev(f1mi_mean_values),
        "F1Ma_mean": safe_mean(f1ma_mean_values),
        "F1Ma_std": safe_pstdev(f1ma_mean_values),
        "within_run_F1Mi_std_mean": safe_mean(f1mi_std_values),
        "within_run_F1Ma_std_mean": safe_mean(f1ma_std_values),
        "robust_score": safe_mean(robust_values),
        "robust_score_std": safe_pstdev(robust_values),
    }


def enrich_record_for_aggregate(record):
    timing = record["timing"]
    train_total = timing["train_total_sec"]
    wall_time = record["wall_time_sec"]
    eval_overhead = wall_time - train_total if train_total is not None else None
    enriched = {
        **record,
        "epochs_observed": timing["epochs_observed"],
        "eval_overhead_sec": eval_overhead,
        "epoch_time_mean_sec": timing["epoch_time_mean_sec"],
        "epoch_time_std_sec": timing["epoch_time_std_sec"],
        "epoch_time_last_sec": timing["epoch_time_last_sec"],
        "epoch_time_median_sec": timing["epoch_time_median_sec"],
        "throughput_epoch_per_sec": timing["throughput_epoch_per_sec"],
        "refresh_count": timing["refresh_count"],
        "refresh_epoch_time_mean_sec": timing["refresh_epoch_time_mean_sec"],
        "non_refresh_epoch_time_mean_sec": timing["non_refresh_epoch_time_mean_sec"],
        "warmup_epoch_time_mean_sec": timing["warmup_epoch_time_mean_sec"],
        "corrected_epoch_time_mean_sec": timing["corrected_epoch_time_mean_sec"],
    }
    return enriched


def make_output_path(grace_dir, args, dataset):
    if args.out:
        return args.out if os.path.isabs(args.out) else os.path.join(grace_dir, args.out)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(grace_dir, args.out_dir)
    return os.path.join(out_dir, f"efficiency_{dataset_slug(dataset)}_results.csv")


def build_base_note(selected):
    robust_text = fmt_float(selected.get("source_robust_score"))
    return (
        f"base_source={selected['source']}; "
        f"base_path={selected['source_path']}; "
        f"base_rank={selected['candidate_rank']}; "
        f"base_robust={robust_text}"
    )


def row_common(dataset, method, base_method, base_params, trial_updates, notes):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "method": method,
        "base_method": base_method,
        "base_params_json": fmt_json(base_params),
        "trial_params_json": fmt_json(trial_updates),
        "notes": notes,
    }


def write_run_row(csv_path, dataset, method, run_idx, seed, record, base_params, trial_updates, notes):
    timing = record["timing"]
    metrics = record["metrics"]
    row = row_common(dataset, method, BASE_METHOD[method], base_params, trial_updates, notes)
    row.update(
        {
            "stage": "run",
            "run_idx": run_idx,
            "seed": seed,
            "num_runs": "",
            "num_epochs_config": record["num_epochs_config"],
            "epochs_observed": timing["epochs_observed"],
            "wall_time_sec": fmt_float(record["wall_time_sec"]),
            "wall_time_std_sec": "",
            "train_total_sec": fmt_float(timing["train_total_sec"]),
            "train_total_std_sec": "",
            "eval_overhead_sec": fmt_float(record["eval_overhead_sec"]),
            "epoch_time_mean_sec": fmt_float(timing["epoch_time_mean_sec"]),
            "epoch_time_std_sec": fmt_float(timing["epoch_time_std_sec"]),
            "epoch_time_last_sec": fmt_float(timing["epoch_time_last_sec"]),
            "epoch_time_median_sec": fmt_float(timing["epoch_time_median_sec"]),
            "throughput_epoch_per_sec": fmt_float(timing["throughput_epoch_per_sec"]),
            "refresh_count": fmt_float(timing["refresh_count"], digits=0),
            "refresh_epoch_time_mean_sec": fmt_float(timing["refresh_epoch_time_mean_sec"]),
            "non_refresh_epoch_time_mean_sec": fmt_float(timing["non_refresh_epoch_time_mean_sec"]),
            "warmup_epoch_time_mean_sec": fmt_float(timing["warmup_epoch_time_mean_sec"]),
            "corrected_epoch_time_mean_sec": fmt_float(timing["corrected_epoch_time_mean_sec"]),
            "time_vs_grace_sec": "",
            "time_ratio_vs_grace": "",
            "overhead_vs_base_sec": "",
            "overhead_ratio_vs_base": "",
            "F1Mi_mean": fmt_float(metrics["F1Mi_mean"]),
            "F1Mi_std": fmt_float(metrics["F1Mi_std"]),
            "F1Ma_mean": fmt_float(metrics["F1Ma_mean"]),
            "F1Ma_std": fmt_float(metrics["F1Ma_std"]),
            "robust_score": fmt_float(record["robust_score"]),
            "robust_score_std": "",
        }
    )
    append_result_row(csv_path, row=row)


def write_failed_row(csv_path, dataset, method, run_idx, seed, base_params, trial_updates, notes):
    row = row_common(dataset, method, BASE_METHOD[method], base_params, trial_updates, notes)
    row.update(
        {
            "stage": "run_failed",
            "run_idx": run_idx,
            "seed": seed,
            "num_runs": "",
            "num_epochs_config": "",
            "epochs_observed": "",
            "wall_time_sec": "",
            "wall_time_std_sec": "",
            "train_total_sec": "",
            "train_total_std_sec": "",
            "eval_overhead_sec": "",
            "epoch_time_mean_sec": "",
            "epoch_time_std_sec": "",
            "epoch_time_last_sec": "",
            "epoch_time_median_sec": "",
            "throughput_epoch_per_sec": "",
            "refresh_count": "",
            "refresh_epoch_time_mean_sec": "",
            "non_refresh_epoch_time_mean_sec": "",
            "warmup_epoch_time_mean_sec": "",
            "corrected_epoch_time_mean_sec": "",
            "time_vs_grace_sec": "",
            "time_ratio_vs_grace": "",
            "overhead_vs_base_sec": "",
            "overhead_ratio_vs_base": "",
            "F1Mi_mean": "",
            "F1Mi_std": "",
            "F1Ma_mean": "",
            "F1Ma_std": "",
            "robust_score": "",
            "robust_score_std": "",
        }
    )
    append_result_row(csv_path, row=row)


def compute_summary_ratios(method, summaries):
    method_time = summaries[method]["train_total_sec"] or summaries[method]["wall_time_sec"]
    grace_summary = summaries.get("grace", {})
    grace_time = grace_summary.get("train_total_sec") or grace_summary.get("wall_time_sec")
    base_method = BASE_METHOD[method]
    base_summary = summaries.get(base_method, {}) if base_method else {}
    base_time = (base_summary.get("train_total_sec") or base_summary.get("wall_time_sec")) if base_method else None

    time_vs_grace = None
    time_ratio_vs_grace = None
    if grace_time is not None and method_time is not None:
        time_vs_grace = method_time - grace_time
        if grace_time > 0:
            time_ratio_vs_grace = method_time / grace_time

    overhead_vs_base = None
    overhead_ratio_vs_base = None
    if base_time is not None and method_time is not None:
        overhead_vs_base = method_time - base_time
        if base_time > 0:
            overhead_ratio_vs_base = overhead_vs_base / base_time

    return time_vs_grace, time_ratio_vs_grace, overhead_vs_base, overhead_ratio_vs_base


def write_summary_rows(csv_path, dataset, summaries, base_params_by_method, trial_updates_by_method):
    for method in METHOD_CHOICES:
        if method not in summaries:
            continue

        summary = summaries[method]
        time_vs_grace, time_ratio_vs_grace, overhead_vs_base, overhead_ratio_vs_base = compute_summary_ratios(
            method, summaries
        )
        row = row_common(
            dataset,
            method,
            BASE_METHOD[method],
            base_params_by_method.get(method, {}),
            trial_updates_by_method.get(method, {}),
            "summary across successful efficiency runs",
        )
        row.update(
            {
                "stage": "summary",
                "run_idx": "",
                "seed": "",
                "num_runs": summary["num_runs"],
                "num_epochs_config": summary["num_epochs_config"],
                "epochs_observed": fmt_float(summary["epochs_observed"]),
                "wall_time_sec": fmt_float(summary["wall_time_sec"]),
                "wall_time_std_sec": fmt_float(summary["wall_time_std_sec"]),
                "train_total_sec": fmt_float(summary["train_total_sec"]),
                "train_total_std_sec": fmt_float(summary["train_total_std_sec"]),
                "eval_overhead_sec": fmt_float(summary["eval_overhead_sec"]),
                "epoch_time_mean_sec": fmt_float(summary["epoch_time_mean_sec"]),
                "epoch_time_std_sec": fmt_float(summary["epoch_time_std_sec"]),
                "epoch_time_last_sec": fmt_float(summary["epoch_time_last_sec"]),
                "epoch_time_median_sec": fmt_float(summary["epoch_time_median_sec"]),
                "throughput_epoch_per_sec": fmt_float(summary["throughput_epoch_per_sec"]),
                "refresh_count": fmt_float(summary["refresh_count"]),
                "refresh_epoch_time_mean_sec": fmt_float(summary["refresh_epoch_time_mean_sec"]),
                "non_refresh_epoch_time_mean_sec": fmt_float(summary["non_refresh_epoch_time_mean_sec"]),
                "warmup_epoch_time_mean_sec": fmt_float(summary["warmup_epoch_time_mean_sec"]),
                "corrected_epoch_time_mean_sec": fmt_float(summary["corrected_epoch_time_mean_sec"]),
                "time_vs_grace_sec": fmt_float(time_vs_grace),
                "time_ratio_vs_grace": fmt_float(time_ratio_vs_grace),
                "overhead_vs_base_sec": fmt_float(overhead_vs_base),
                "overhead_ratio_vs_base": fmt_float(overhead_ratio_vs_base),
                "F1Mi_mean": fmt_float(summary["F1Mi_mean"]),
                "F1Mi_std": fmt_float(summary["F1Mi_std"]),
                "F1Ma_mean": fmt_float(summary["F1Ma_mean"]),
                "F1Ma_std": fmt_float(summary["F1Ma_std"]),
                "robust_score": fmt_float(summary["robust_score"]),
                "robust_score_std": fmt_float(summary["robust_score_std"]),
            }
        )
        append_result_row(csv_path, row=row)


def run_one_dataset(grace_dir, base_config, dataset, methods, args, out_csv):
    records_by_method = {method: [] for method in methods}
    base_params_by_method = {}
    trial_updates_by_method = {}
    temp_dir = os.path.join(grace_dir, "results", "_efficiency_tmp")

    for method in methods:
        selected = select_base_params(grace_dir, dataset, method)
        base_params = selected["params"]
        base_params_by_method[method] = base_params
        base_note = build_base_note(selected)
        print("=" * 90)
        print(f"[efficiency] dataset={dataset} | method={method}")
        print(f"[efficiency] {base_note}")

        for run_idx in range(1, args.runs + 1):
            trial_updates = build_trial_updates(
                base_config=base_config,
                dataset=dataset,
                method=method,
                base_params=base_params,
                run_idx=run_idx,
            )
            summary_updates = dict(trial_updates)
            summary_updates.pop("seed", None)
            summary_updates.setdefault("seed_list", [])
            trial_updates_by_method.setdefault(method, summary_updates)
            seed = trial_updates["seed"]
            temp_cfg = None

            try:
                print(f"  [{dataset}/{method}] run {run_idx}/{args.runs} start | seed={seed}")
                temp_cfg = make_temp_config(base_config, dataset, trial_updates, temp_dir)
                wall_start = t()
                metrics, combined = run_train(
                    grace_dir,
                    temp_cfg,
                    dataset=dataset,
                    method=method,
                    gpu_id=args.gpu_id,
                    verbose_output=args.verbose_train_output,
                )
                wall_time = t() - wall_start
                timing = parse_timing_stats(combined)
                score = robust_score(metrics, args.std_weight)
                train_total = timing["train_total_sec"]
                eval_overhead = wall_time - train_total if train_total is not None else None
                record = {
                    "num_epochs_config": int(base_config[dataset]["num_epochs"]),
                    "wall_time_sec": wall_time,
                    "eval_overhead_sec": eval_overhead,
                    "timing": timing,
                    "metrics": metrics,
                    "robust_score": score,
                }
                records_by_method[method].append(enrich_record_for_aggregate(record))
                trial_updates_by_method[method]["seed_list"].append(seed)

                write_run_row(
                    csv_path=out_csv,
                    dataset=dataset,
                    method=method,
                    run_idx=run_idx,
                    seed=seed,
                    record=record,
                    base_params=base_params,
                    trial_updates=trial_updates,
                    notes=base_note,
                )
                print(
                    f"    success: wall={wall_time:.2f}s, "
                    f"train_total={timing['train_total_sec'] or 0.0:.2f}s, "
                    f"epoch_mean={timing['epoch_time_mean_sec'] or 0.0:.4f}s"
                )
            except Exception as exc:
                error = summarize_error(exc)
                write_failed_row(
                    csv_path=out_csv,
                    dataset=dataset,
                    method=method,
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
        method: aggregate_records(records)
        for method, records in records_by_method.items()
        if records
    }
    write_summary_rows(out_csv, dataset, summaries, base_params_by_method, trial_updates_by_method)
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
    print(f"[efficiency] saved rows for {dataset}: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SG-GCL efficiency experiments and measure training time."
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
        help="Stream raw train.py output for each efficiency run.",
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

        run_one_dataset(
            grace_dir=grace_dir,
            base_config=base_config,
            dataset=dataset,
            methods=methods,
            args=args,
            out_csv=out_csv,
        )

    print("[efficiency] all requested efficiency experiments finished")


if __name__ == "__main__":
    main()
