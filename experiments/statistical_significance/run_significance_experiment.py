import argparse
import csv
import os
import sys
from datetime import datetime

import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.component_ablation.run_component_ablation import (
    fmt_float,
    fmt_json,
    make_temp_config,
    safe_mean,
    safe_pstdev,
)
from experiments.efficiency.run_efficiency_experiment import (
    params_to_dataset_updates,
    select_base_params,
)
from experiments.method_comparison.run_full_pipeline import robust_score, run_train


DATASET_CHOICES = ["Cora", "CiteSeer", "PubMed", "DBLP"]
METHOD_CHOICES = ["grace", "gca", "sg-gr", "sg-gc"]

CSV_HEADERS = [
    "timestamp",
    "stage",
    "dataset",
    "method",
    "run_idx",
    "seed",
    "eval_seed",
    "num_runs",
    "F1Mi_mean",
    "F1Mi_std",
    "F1Ma_mean",
    "F1Ma_std",
    "robust_score",
    "robust_score_std",
    "params_json",
    "metric",
    "baseline_method",
    "target_method",
    "test_name",
    "n_pairs",
    "mean_delta",
    "median_delta",
    "ci95_low",
    "ci95_high",
    "p_value",
    "p_value_holm",
    "p_value_paired_ttest",
    "effect_size",
    "effect_size_name",
    "cohen_dz",
    "rank_biserial",
    "significant",
    "notes",
]


def dataset_slug(dataset):
    return dataset.lower()


def seed_for_run(base_config, dataset, run_idx):
    return int(base_config[dataset].get("seed", 0)) + int(run_idx)


def build_trial_updates(base_config, dataset, method, base_params, run_idx, eval_repeats):
    updates = params_to_dataset_updates(base_params, method)
    seed = seed_for_run(base_config, dataset, run_idx)
    updates["seed"] = seed
    updates["eval_seed"] = seed
    updates["eval_repeats"] = int(eval_repeats)
    return updates


def append_result_row(csv_path, row=None, write_header=False):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        if row is not None:
            writer.writerow(row)


def make_output_path(grace_dir, args, dataset):
    if args.out:
        return args.out if os.path.isabs(args.out) else os.path.join(grace_dir, args.out)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(grace_dir, args.out_dir)
    return os.path.join(out_dir, f"significance_{dataset_slug(dataset)}_results.csv")


def row_base(dataset, method, notes=""):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "method": method,
        "notes": notes,
    }


def summarize_records(records):
    f1mi_values = [record["metrics"]["F1Mi_mean"] for record in records]
    f1ma_values = [record["metrics"]["F1Ma_mean"] for record in records]
    robust_values = [record["robust_score"] for record in records]
    return {
        "num_runs": len(records),
        "F1Mi_mean": safe_mean(f1mi_values),
        "F1Mi_std": safe_pstdev(f1mi_values),
        "F1Ma_mean": safe_mean(f1ma_values),
        "F1Ma_std": safe_pstdev(f1ma_values),
        "robust_score": safe_mean(robust_values),
        "robust_score_std": safe_pstdev(robust_values),
    }


def write_run_row(csv_path, dataset, method, run_idx, seed, eval_seed, metrics, score, trial_updates, notes):
    row = row_base(dataset, method, notes)
    row.update(
        {
            "stage": "run",
            "run_idx": run_idx,
            "seed": seed,
            "eval_seed": eval_seed,
            "F1Mi_mean": fmt_float(metrics["F1Mi_mean"]),
            "F1Mi_std": fmt_float(metrics["F1Mi_std"]),
            "F1Ma_mean": fmt_float(metrics["F1Ma_mean"]),
            "F1Ma_std": fmt_float(metrics["F1Ma_std"]),
            "robust_score": fmt_float(score),
            "params_json": fmt_json(trial_updates),
        }
    )
    append_result_row(csv_path, row=row)


def write_failed_row(csv_path, dataset, method, run_idx, seed, eval_seed, trial_updates, notes):
    row = row_base(dataset, method, notes)
    row.update(
        {
            "stage": "run_failed",
            "run_idx": run_idx,
            "seed": seed,
            "eval_seed": eval_seed,
            "params_json": fmt_json(trial_updates),
        }
    )
    append_result_row(csv_path, row=row)


def write_summary_rows(csv_path, dataset, records_by_method, params_by_method, notes_by_method):
    for method in METHOD_CHOICES:
        records = records_by_method.get(method, [])
        if not records:
            continue
        summary = summarize_records(records)
        row = row_base(dataset, method, notes_by_method.get(method, ""))
        row.update(
            {
                "stage": "summary",
                "num_runs": summary["num_runs"],
                "F1Mi_mean": fmt_float(summary["F1Mi_mean"]),
                "F1Mi_std": fmt_float(summary["F1Mi_std"]),
                "F1Ma_mean": fmt_float(summary["F1Ma_mean"]),
                "F1Ma_std": fmt_float(summary["F1Ma_std"]),
                "robust_score": fmt_float(summary["robust_score"]),
                "robust_score_std": fmt_float(summary["robust_score_std"]),
                "params_json": fmt_json(params_by_method.get(method, {})),
            }
        )
        append_result_row(csv_path, row=row)


def build_source_note(selected):
    return (
        f"base_source={selected['source']}; "
        f"base_path={selected['source_path']}; "
        f"base_rank={selected['candidate_rank']}; "
        f"base_robust={fmt_float(selected.get('source_robust_score'))}"
    )


def run_one_dataset(grace_dir, base_config, dataset, methods, args, out_csv):
    records_by_method = {method: [] for method in methods}
    params_by_method = {}
    notes_by_method = {}
    temp_dir = os.path.join(grace_dir, "results", "_significance_tmp")

    for method in methods:
        selected = select_base_params(grace_dir, dataset, method)
        base_params = selected["params"]
        source_note = build_source_note(selected)
        params_by_method[method] = base_params
        notes_by_method[method] = source_note

        print("=" * 90)
        print(f"[significance] dataset={dataset} | method={method}")
        print(f"[significance] {source_note}")

        for run_idx in range(1, args.runs + 1):
            trial_updates = build_trial_updates(
                base_config=base_config,
                dataset=dataset,
                method=method,
                base_params=base_params,
                run_idx=run_idx,
                eval_repeats=args.eval_repeats,
            )
            seed = trial_updates["seed"]
            eval_seed = trial_updates["eval_seed"]
            temp_cfg = None

            try:
                print(
                    f"  [{dataset}/{method}] paired run {run_idx}/{args.runs} "
                    f"start | seed={seed} | eval_seed={eval_seed}"
                )
                temp_cfg = make_temp_config(base_config, dataset, trial_updates, temp_dir)
                metrics, _combined = run_train(
                    grace_dir,
                    temp_cfg,
                    dataset=dataset,
                    method=method,
                    gpu_id=args.gpu_id,
                    verbose_output=args.verbose_train_output,
                )
                score = robust_score(metrics, args.std_weight)
                records_by_method[method].append({"metrics": metrics, "robust_score": score})
                write_run_row(
                    csv_path=out_csv,
                    dataset=dataset,
                    method=method,
                    run_idx=run_idx,
                    seed=seed,
                    eval_seed=eval_seed,
                    metrics=metrics,
                    score=score,
                    trial_updates=trial_updates,
                    notes=source_note,
                )
                print(
                    f"    success: F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}, "
                    f"robust={score:.4f}"
                )
            except Exception as exc:
                error = str(exc).replace("\r", " ").replace("\n", " | ")[:1000]
                write_failed_row(
                    csv_path=out_csv,
                    dataset=dataset,
                    method=method,
                    run_idx=run_idx,
                    seed=seed,
                    eval_seed=eval_seed,
                    trial_updates=trial_updates,
                    notes=f"{source_note}; error={error}",
                )
                print(f"    failed: {error}")
                if not args.continue_on_error:
                    raise
            finally:
                if temp_cfg and os.path.exists(temp_cfg):
                    os.remove(temp_cfg)

    write_summary_rows(out_csv, dataset, records_by_method, params_by_method, notes_by_method)
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
    print(f"[significance] saved rows for {dataset}: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paired-seed statistical significance experiments for method comparison."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=DATASET_CHOICES, choices=DATASET_CHOICES)
    parser.add_argument("--methods", nargs="+", default=METHOD_CHOICES, choices=METHOD_CHOICES)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--eval_repeats", type=int, default=3)
    parser.add_argument("--std_weight", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--out", type=str, default=None, help="Custom output CSV. Only valid for one dataset.")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--verbose_train_output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs <= 1:
        raise RuntimeError("--runs must be >= 2 for paired significance testing.")
    if args.eval_repeats <= 0:
        raise RuntimeError("--eval_repeats must be >= 1.")

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

    print("[significance] all requested paired-seed runs finished")


if __name__ == "__main__":
    main()
