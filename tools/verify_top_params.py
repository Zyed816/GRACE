import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import yaml

# Verification pipeline:
# - read top-k rows from grid-search CSV
# - rebuild temporary config per method
# - rerun multiple times to estimate stability

F1_PATTERN = re.compile(
    r"\(E\) \| label_classification: "
    r"F1Mi=(?P<f1mi_mean>\d+\.\d+)\+-(?P<f1mi_std>\d+\.\d+), "
    r"F1Ma=(?P<f1ma_mean>\d+\.\d+)\+-(?P<f1ma_std>\d+\.\d+)"
)


def parse_metrics(output_text):
    match = F1_PATTERN.search(output_text)
    if not match:
        raise RuntimeError("Failed to parse evaluation metrics from train output.")

    return {
        "F1Mi_mean": float(match.group("f1mi_mean")),
        "F1Mi_std": float(match.group("f1mi_std")),
        "F1Ma_mean": float(match.group("f1ma_mean")),
        "F1Ma_std": float(match.group("f1ma_std")),
    }


def run_train(grace_dir, config_path, dataset, method, gpu_id):
    cmd = [
        sys.executable,
        "train.py",
        "--dataset",
        dataset,
        "--method",
        method,
        "--config",
        config_path,
        "--gpu_id",
        str(gpu_id),
    ]

    proc = subprocess.run(
        cmd,
        cwd=grace_dir,
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Training failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    combined = proc.stdout + "\n" + proc.stderr
    metrics = parse_metrics(combined)
    return metrics, combined


def read_grid_csv(csv_path, topk):
    # Read ranked candidates from CSV top to bottom.
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= topk:
                break
            results.append(row)
    return results


def make_temp_config_from_row(base_config, dataset_key, csv_row):
    import copy
    cfg = copy.deepcopy(base_config)

    dataset_updates = {
        "similarity_percentile": float(csv_row["similarity_percentile"]),
        "max_du_per_node": int(csv_row["max_du_per_node"]),
        "unlabeled_weight": float(csv_row["unlabeled_weight"]),
        "warmup_epochs": int(csv_row["warmup_epochs"]),
        "update_interval": int(csv_row["update_interval"]),
        "beta": float(csv_row["beta"]),
        "use_mutual_topk": csv_row["use_mutual_topk"].lower() == "true",
        "corrected_ramp_epochs": int(csv_row["corrected_ramp_epochs"]),
    }

    if "tau" in csv_row and csv_row["tau"] != "":
        dataset_updates["tau"] = float(csv_row["tau"])

    # Handle null values
    if csv_row.get("similarity_threshold", "").lower() in ["none", "null", ""]:
        dataset_updates["similarity_threshold"] = None
    else:
        dataset_updates["similarity_threshold"] = float(csv_row["similarity_threshold"])

    cfg[dataset_key].update(dataset_updates)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        return f.name


def make_temp_config_from_row_for_method(base_config, dataset_key, csv_row, method):
    # Dispatch CSV->config mapping by method-specific columns.
    if method == "ifl-gr":
        return make_temp_config_from_row(base_config, dataset_key, csv_row)

    if method == "ifl-gc":
        import copy
        cfg = copy.deepcopy(base_config)

        dataset_updates = {
            "gca_drop_scheme": csv_row["gca_drop_scheme"],
            "similarity_percentile": float(csv_row["similarity_percentile"]),
            "max_du_per_node": int(float(csv_row["max_du_per_node"])),
            "unlabeled_weight": float(csv_row["unlabeled_weight"]),
            "iflgc_refl_du_weight": float(csv_row["iflgc_refl_du_weight"]),
            "warmup_epochs": int(float(csv_row["warmup_epochs"])),
            "tau": float(csv_row["tau"]),
            "drop_edge_rate_1": float(csv_row["drop_edge_rate_1"]),
            "drop_edge_rate_2": float(csv_row["drop_edge_rate_2"]),
            "drop_feature_rate_1": float(csv_row["drop_feature_rate_1"]),
            "drop_feature_rate_2": float(csv_row["drop_feature_rate_2"]),
            "update_interval": int(float(csv_row["update_interval"])),
            "beta": float(csv_row["beta"]),
            "use_mutual_topk": csv_row["use_mutual_topk"].lower() == "true",
            "corrected_ramp_epochs": int(float(csv_row["corrected_ramp_epochs"])),
        }

        if "gca_pr_k" in csv_row and csv_row["gca_pr_k"] != "":
            dataset_updates["gca_pr_k"] = int(float(csv_row["gca_pr_k"]))

        if csv_row.get("similarity_threshold", "").lower() in ["none", "null", ""]:
            dataset_updates["similarity_threshold"] = None
        elif "similarity_threshold" in csv_row:
            dataset_updates["similarity_threshold"] = float(csv_row["similarity_threshold"])

        cfg[dataset_key].update(dataset_updates)

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
            return f.name

    if method == "gca":
        import copy
        cfg = copy.deepcopy(base_config)

        dataset_updates = {
            "gca_drop_scheme": csv_row["gca_drop_scheme"],
            "drop_edge_rate_1": float(csv_row["drop_edge_rate_1"]),
            "drop_edge_rate_2": float(csv_row["drop_edge_rate_2"]),
            "drop_feature_rate_1": float(csv_row["drop_feature_rate_1"]),
            "drop_feature_rate_2": float(csv_row["drop_feature_rate_2"]),
            "tau": float(csv_row["tau"]),
        }

        if "gca_pr_k" in csv_row and csv_row["gca_pr_k"] != "":
            dataset_updates["gca_pr_k"] = int(float(csv_row["gca_pr_k"]))

        cfg[dataset_key].update(dataset_updates)

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
            return f.name

    raise ValueError(f"Unsupported method for verification: {method}")


def print_param_summary(csv_row, method):
    if method == "ifl-gr":
        tau_text = f", tau={csv_row['tau']}" if "tau" in csv_row and csv_row["tau"] != "" else ""
        print(
            f"sim_p={csv_row['similarity_percentile']}, "
            f"max_du={csv_row['max_du_per_node']}, "
            f"lambda_u={csv_row['unlabeled_weight']}, "
            f"warmup={csv_row['warmup_epochs']}"
            f"{tau_text}"
        )
        return

    if method == "gca":
        print(
            f"scheme={csv_row['gca_drop_scheme']}, "
            f"de1={csv_row['drop_edge_rate_1']}, "
            f"de2={csv_row['drop_edge_rate_2']}, "
            f"df1={csv_row['drop_feature_rate_1']}, "
            f"df2={csv_row['drop_feature_rate_2']}, "
            f"tau={csv_row['tau']}"
        )
        return

    if method == "ifl-gc":
        print(
            f"scheme={csv_row['gca_drop_scheme']}, "
            f"sim_p={csv_row['similarity_percentile']}, "
            f"max_du={csv_row['max_du_per_node']}, "
            f"lambda_u={csv_row['unlabeled_weight']}, "
            f"alpha_refl={csv_row['iflgc_refl_du_weight']}, "
            f"tau={csv_row['tau']}, "
            f"de=({csv_row['drop_edge_rate_1']},{csv_row['drop_edge_rate_2']}), "
            f"df=({csv_row['drop_feature_rate_1']},{csv_row['drop_feature_rate_2']})"
        )
        return

    raise ValueError(f"Unsupported method for parameter summary: {method}")


def main():
    parser = argparse.ArgumentParser(description="Verify top IFL-GR parameters by multiple runs")
    parser.add_argument("--top_params", type=str, required=True, help="Path to grid_search CSV")
    parser.add_argument("--topk", type=int, default=3, help="Number of top params to verify")
    parser.add_argument("--runs", type=int, default=3, help="Runs per parameter")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed", "DBLP"])
    parser.add_argument("--method", type=str, default="ifl-gr", choices=["ifl-gr", "gca", "ifl-gc"])
    args = parser.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    grace_dir = os.path.abspath(os.path.join(tools_dir, ".."))
    config_path = os.path.join(grace_dir, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    print(f"[1/2] Reading top {args.topk} parameters from {args.top_params}...")
    top_rows = read_grid_csv(args.top_params, args.topk)

    print(f"[2/2] Verifying {args.topk} parameter sets with {args.runs} runs each...")

    verification_results = []

    for param_idx, csv_row in enumerate(top_rows, start=1):
        print(f"\n--- Parameter Set #{param_idx} ---")
        print_param_summary(csv_row, args.method)

        run_metrics = []

        for run_idx in range(1, args.runs + 1):
            temp_cfg = make_temp_config_from_row_for_method(base_config, args.dataset, csv_row, args.method)

            try:
                metrics, _ = run_train(
                    grace_dir,
                    temp_cfg,
                    dataset=args.dataset,
                    method=args.method,
                    gpu_id=args.gpu_id,
                )
                run_metrics.append(metrics)

                print(
                    f"  Run {run_idx:02d}: "
                    f"F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}, "
                    f"F1Ma={metrics['F1Ma_mean']:.4f}+-{metrics['F1Ma_std']:.4f}"
                )
            except Exception as exc:
                print(f"  Run {run_idx:02d} FAILED: {exc}")
            finally:
                if os.path.exists(temp_cfg):
                    os.remove(temp_cfg)

        if not run_metrics:
            print(f"SKIPPED: All {args.runs} runs failed for parameter set #{param_idx}")
            continue

        # Aggregate metrics across runs
        f1mi_means = np.array([m["F1Mi_mean"] for m in run_metrics])
        f1mi_stds = np.array([m["F1Mi_std"] for m in run_metrics])
        f1ma_means = np.array([m["F1Ma_mean"] for m in run_metrics])
        f1ma_stds = np.array([m["F1Ma_std"] for m in run_metrics])

        avg_f1mi_mean = float(np.mean(f1mi_means))
        avg_f1mi_std = float(np.mean(f1mi_stds))
        std_f1mi_mean = float(np.std(f1mi_means))

        avg_f1ma_mean = float(np.mean(f1ma_means))
        avg_f1ma_std = float(np.mean(f1ma_stds))
        std_f1ma_mean = float(np.std(f1ma_means))
        std_f1mi_std = float(np.std(f1mi_stds))
        std_f1ma_std = float(np.std(f1ma_stds))

        result = {
            "param_rank": param_idx,
            "runs": len(run_metrics),
            "avg_F1Mi_mean": avg_f1mi_mean,
            "std_F1Mi_mean": std_f1mi_mean,
            "avg_F1Mi_std": avg_f1mi_std,
            "avg_F1Ma_mean": avg_f1ma_mean,
            "std_F1Ma_mean": std_f1ma_mean,
            "avg_F1Ma_std": avg_f1ma_std,
        }

        if args.method == "ifl-gr":
            result.update({
                "similarity_percentile": csv_row["similarity_percentile"],
                "max_du_per_node": csv_row["max_du_per_node"],
                "unlabeled_weight": csv_row["unlabeled_weight"],
                "warmup_epochs": csv_row["warmup_epochs"],
                "tau": csv_row.get("tau", ""),
            })
        elif args.method == "gca":
            result.update({
                "gca_drop_scheme": csv_row["gca_drop_scheme"],
                "drop_edge_rate_1": csv_row["drop_edge_rate_1"],
                "drop_edge_rate_2": csv_row["drop_edge_rate_2"],
                "drop_feature_rate_1": csv_row["drop_feature_rate_1"],
                "drop_feature_rate_2": csv_row["drop_feature_rate_2"],
                "tau": csv_row["tau"],
            })
        else:
            result.update({
                "gca_drop_scheme": csv_row["gca_drop_scheme"],
                "similarity_percentile": csv_row["similarity_percentile"],
                "max_du_per_node": csv_row["max_du_per_node"],
                "unlabeled_weight": csv_row["unlabeled_weight"],
                "iflgc_refl_du_weight": csv_row["iflgc_refl_du_weight"],
                "tau": csv_row["tau"],
                "drop_edge_rate_1": csv_row["drop_edge_rate_1"],
                "drop_edge_rate_2": csv_row["drop_edge_rate_2"],
                "drop_feature_rate_1": csv_row["drop_feature_rate_1"],
                "drop_feature_rate_2": csv_row["drop_feature_rate_2"],
            })
        verification_results.append(result)

        print(
            f"  Average across {len(run_metrics)} runs: "
            f"F1Mi = {avg_f1mi_mean:.4f}±{std_f1mi_mean:.4f} "
            f"(within-run std: {avg_f1mi_std:.4f}±{std_f1mi_std:.4f})"
        )

    # Print summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    for res in verification_results:
        if args.method == "ifl-gr":
            tau_tail = f", tau={res['tau']}" if res.get("tau", "") else ""
            print(
                f"#{res['param_rank']}: "
                f"F1Mi={res['avg_F1Mi_mean']:.4f}±{res['std_F1Mi_mean']:.4f}, "
                f"runs={res['runs']}, "
                f"params: sim_p={res['similarity_percentile']}, "
                f"max_du={res['max_du_per_node']}, "
                f"lambda_u={res['unlabeled_weight']}"
                f"{tau_tail}"
            )
        elif args.method == "gca":
            print(
                f"#{res['param_rank']}: "
                f"F1Mi={res['avg_F1Mi_mean']:.4f}±{res['std_F1Mi_mean']:.4f}, "
                f"runs={res['runs']}, "
                f"params: scheme={res['gca_drop_scheme']}, "
                f"de1={res['drop_edge_rate_1']}, "
                f"de2={res['drop_edge_rate_2']}, "
                f"tau={res['tau']}"
            )
        else:
            print(
                f"#{res['param_rank']}: "
                f"F1Mi={res['avg_F1Mi_mean']:.4f}±{res['std_F1Mi_mean']:.4f}, "
                f"runs={res['runs']}, "
                f"params: scheme={res['gca_drop_scheme']}, "
                f"sim_p={res['similarity_percentile']}, "
                f"max_du={res['max_du_per_node']}, "
                f"lambda_u={res['unlabeled_weight']}, "
                f"alpha_refl={res['iflgc_refl_du_weight']}, "
                f"tau={res['tau']}"
            )

    recommendation = verification_results[0] if verification_results else None
    if recommendation:
        print(f"\nRECOMMENDATION: Use parameters from rank #{recommendation['param_rank']}")
        print(f"  Expected stable F1Mi: {recommendation['avg_F1Mi_mean']:.4f}±{recommendation['std_F1Mi_mean']:.4f}")


if __name__ == "__main__":
    main()
