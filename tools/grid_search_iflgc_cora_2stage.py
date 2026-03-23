import argparse
import copy
import csv
import itertools
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime

import yaml

# Two-stage grid search for IFL-GC on Cora:
# Stage 1: sweep GCA + method parameters
# Stage 2: for top-K candidates from stage 1, sweep learning_rate
# Results combined into single CSV with stage indicator

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


def robust_score(metrics, std_weight):
    # Penalize unstable settings by subtracting weighted std.
    return metrics["F1Mi_mean"] - std_weight * metrics["F1Mi_std"]


def run_train(grace_dir, config_path, method, gpu_id):
    cmd = [
        sys.executable,
        "train.py",
        "--dataset",
        "Cora",
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


def make_temp_config(base_config, cora_updates):
    cfg = copy.deepcopy(base_config)
    cfg["Cora"].update(cora_updates)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        return f.name


def main():
    parser = argparse.ArgumentParser(description="Two-stage grid search for IFL-GC on Cora")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--std_weight", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=3, help="Top candidates to advance to stage 2")
    parser.add_argument("--out", type=str, default="results/grid_search_iflgc_cora_2stage_results.csv")
    args = parser.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    grace_dir = os.path.abspath(os.path.join(tools_dir, ".."))
    config_path = os.path.join(grace_dir, args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Stage 1 search space: GCA and method parameters
    stage1_search_space = {
        "gca_drop_scheme": ["degree", "pr", "uniform"],
        "similarity_percentile": [99.3, 99.5],
        "max_du_per_node": [10, 14],
        "unlabeled_weight": [0.2, 0.3],
        "warmup_epochs": [80, 100],
        "iflgc_refl_du_weight": [0.2, 0.5],
        "tau": [0.4, 0.6],
    }

    edge_profiles = [
        {"drop_edge_rate_1": 0.2, "drop_edge_rate_2": 0.4},
        {"drop_edge_rate_1": 0.3, "drop_edge_rate_2": 0.5},
    ]

    feature_profiles = [
        {"drop_feature_rate_1": 0.3, "drop_feature_rate_2": 0.4},
        {"drop_feature_rate_1": 0.2, "drop_feature_rate_2": 0.3},
    ]

    stage1_fixed_overrides = {
        "similarity_threshold": None,
        "update_interval": 5,
        "use_mutual_topk": True,
        "beta": 2.0,
        "corrected_ramp_epochs": 40,
        "gca_pr_k": 200,
    }

    # Stage 2 search space: only learning_rate varies
    stage2_learning_rates = [5e-4, 1e-3, 2e-3]

    print("=" * 80)
    print("[STAGE 1/2] Grid search: IFL-GC core parameters")
    print("=" * 80)

    print("[1/3] Running GRACE baseline on Cora...")
    baseline_metrics, _ = run_train(grace_dir, config_path, method="grace", gpu_id=args.gpu_id)
    baseline_score = robust_score(baseline_metrics, args.std_weight)
    print(
        "Baseline GRACE: "
        f"F1Mi={baseline_metrics['F1Mi_mean']:.4f}+-{baseline_metrics['F1Mi_std']:.4f}, "
        f"robust={baseline_score:.4f}"
    )

    keys = list(stage1_search_space.keys())
    values_product = list(itertools.product(*(stage1_search_space[k] for k in keys)))
    total_trials_stage1 = len(values_product) * len(edge_profiles) * len(feature_profiles)

    print(f"[2/3] Stage 1 grid search trials: {total_trials_stage1}")

    stage1_results = []
    trial_idx = 0
    for values in values_product:
        for edge_cfg in edge_profiles:
            for feat_cfg in feature_profiles:
                trial_idx += 1
                trial_params = dict(zip(keys, values))
                trial_params.update(edge_cfg)
                trial_params.update(feat_cfg)
                trial_params.update(stage1_fixed_overrides)

                cora_epochs = base_config["Cora"]["num_epochs"]
                trial_params["warmup_epochs"] = min(int(trial_params["warmup_epochs"]), cora_epochs - 10)

                temp_cfg = make_temp_config(base_config, trial_params)

                try:
                    metrics, _ = run_train(grace_dir, temp_cfg, method="ifl-gc", gpu_id=args.gpu_id)
                    score = robust_score(metrics, args.std_weight)
                    delta = score - baseline_score

                    row = {
                        **trial_params,
                        **metrics,
                        "robust_score": score,
                        "delta_vs_grace": delta,
                        "stage": "1st_stage",
                    }
                    stage1_results.append(row)

                    params_str = (
                        f"scheme={trial_params['gca_drop_scheme']}, "
                        f"sim_p={trial_params['similarity_percentile']}, "
                        f"max_du={trial_params['max_du_per_node']}, "
                        f"lambda_u={trial_params['unlabeled_weight']}, "
                        f"alpha_refl={trial_params['iflgc_refl_du_weight']}, "
                        f"tau={trial_params['tau']}"
                    )
                    print(
                        f"Trial {trial_idx:03d}/{total_trials_stage1}: "
                        f"F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}, "
                        f"robust={score:.4f}, delta={delta:+.4f} | "
                        f"{params_str}"
                    )
                except Exception as exc:
                    print(f"Trial {trial_idx:03d}/{total_trials_stage1} failed: {exc}")
                finally:
                    if os.path.exists(temp_cfg):
                        os.remove(temp_cfg)

    if not stage1_results:
        raise RuntimeError("Stage 1: All trials failed; no valid results generated.")

    stage1_results.sort(key=lambda x: x["robust_score"], reverse=True)

    topk = min(args.topk, len(stage1_results))
    print(f"[3/3] Top {topk} candidates from stage 1:")
    for i in range(topk):
        r = stage1_results[i]
        print(
            f"#{i+1}: robust={r['robust_score']:.4f}, delta={r['delta_vs_grace']:+.4f}, "
            f"F1Mi={r['F1Mi_mean']:.4f}+-{r['F1Mi_std']:.4f}"
        )

    # Stage 2: Sweep learning_rate for top-K candidates
    print("\n" + "=" * 80)
    print("[STAGE 2/2] Fine-tuning: sweep learning_rate for top candidates")
    print("=" * 80)

    stage2_results = []
    trial_idx_stage2 = 0

    for candidate_rank, candidate_row in enumerate(stage1_results[:topk], start=1):
        print(f"\n--- Candidate #{candidate_rank} from Stage 1 ---")
        print(
            f"Best robust={candidate_row['robust_score']:.4f}, "
            f"F1Mi={candidate_row['F1Mi_mean']:.4f}+-{candidate_row['F1Mi_std']:.4f}"
        )

        # Extract stage 1 best parameters (excluding learning_rate which will vary)
        best_params_stage1 = {k: candidate_row[k] for k in stage1_search_space.keys()}
        best_params_stage1.update(edge_profiles[0])  # Use first profile as base
        best_params_stage1.update(feature_profiles[0])
        best_params_stage1.update(stage1_fixed_overrides)

        for lr_idx, lr in enumerate(stage2_learning_rates, start=1):
            trial_idx_stage2 += 1
            trial_params = copy.deepcopy(best_params_stage1)
            trial_params["learning_rate"] = lr

            # Keep warm-up valid under Cora epochs.
            cora_epochs = base_config["Cora"]["num_epochs"]
            trial_params["warmup_epochs"] = min(trial_params["warmup_epochs"], cora_epochs - 10)

            temp_cfg = make_temp_config(base_config, trial_params)

            try:
                metrics, _ = run_train(grace_dir, temp_cfg, method="ifl-gc", gpu_id=args.gpu_id)
                score = robust_score(metrics, args.std_weight)
                delta = score - baseline_score

                row = {
                    **trial_params,
                    **metrics,
                    "robust_score": score,
                    "delta_vs_grace": delta,
                    "stage": "2nd_stage",
                    "base_candidate_rank": candidate_rank,
                }
                stage2_results.append(row)

                print(
                    f"  Lr {lr_idx}/{len(stage2_learning_rates)}: lr={lr:.0e}, "
                    f"F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}, "
                    f"robust={score:.4f}, delta={delta:+.4f}"
                )
            except Exception as exc:
                print(f"  Lr {lr_idx}/{len(stage2_learning_rates)}: lr={lr:.0e} FAILED: {exc}")
            finally:
                if os.path.exists(temp_cfg):
                    os.remove(temp_cfg)

    print("\n" + "=" * 80)
    print("Stage 2 complete. Writing results to CSV...")
    print("=" * 80)

    # Combine all results
    all_results = stage1_results + stage2_results
    all_results.sort(key=lambda x: x["robust_score"], reverse=True)

    out_path = os.path.join(grace_dir, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    headers = [
        "timestamp",
        "stage",
        "base_candidate_rank",
        "gca_drop_scheme",
        "similarity_percentile",
        "max_du_per_node",
        "unlabeled_weight",
        "iflgc_refl_du_weight",
        "warmup_epochs",
        "tau",
        "drop_edge_rate_1",
        "drop_edge_rate_2",
        "drop_feature_rate_1",
        "drop_feature_rate_2",
        "learning_rate",
        "update_interval",
        "beta",
        "use_mutual_topk",
        "corrected_ramp_epochs",
        "gca_pr_k",
        "F1Mi_mean",
        "F1Mi_std",
        "F1Ma_mean",
        "F1Ma_std",
        "robust_score",
        "delta_vs_grace",
    ]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for r in all_results:
            base_rank = r.get("base_candidate_rank", "")
            writer.writerow({
                "timestamp": ts,
                "stage": r["stage"],
                "base_candidate_rank": base_rank,
                "gca_drop_scheme": r["gca_drop_scheme"],
                "similarity_percentile": r["similarity_percentile"],
                "max_du_per_node": r["max_du_per_node"],
                "unlabeled_weight": r["unlabeled_weight"],
                "iflgc_refl_du_weight": r["iflgc_refl_du_weight"],
                "warmup_epochs": r["warmup_epochs"],
                "tau": r["tau"],
                "drop_edge_rate_1": r["drop_edge_rate_1"],
                "drop_edge_rate_2": r["drop_edge_rate_2"],
                "drop_feature_rate_1": r["drop_feature_rate_1"],
                "drop_feature_rate_2": r["drop_feature_rate_2"],
                "learning_rate": r.get("learning_rate", base_config["Cora"].get("learning_rate", "")),
                "update_interval": r["update_interval"],
                "beta": r["beta"],
                "use_mutual_topk": r["use_mutual_topk"],
                "corrected_ramp_epochs": r["corrected_ramp_epochs"],
                "gca_pr_k": r["gca_pr_k"],
                "F1Mi_mean": f"{r['F1Mi_mean']:.6f}",
                "F1Mi_std": f"{r['F1Mi_std']:.6f}",
                "F1Ma_mean": f"{r['F1Ma_mean']:.6f}",
                "F1Ma_std": f"{r['F1Ma_std']:.6f}",
                "robust_score": f"{r['robust_score']:.6f}",
                "delta_vs_grace": f"{r['delta_vs_grace']:.6f}",
            })

    print(f"Saved {len(all_results)} results ({len(stage1_results)} stage1 + {len(stage2_results)} stage2) to: {out_path}")
    print(f"\nTop 5 overall candidates:")
    for i, r in enumerate(all_results[:5], start=1):
        print(
            f"#{i}: stage={r['stage']}, robust={r['robust_score']:.4f}, delta={r['delta_vs_grace']:+.4f}, "
            f"F1Mi={r['F1Mi_mean']:.4f}+-{r['F1Mi_std']:.4f}"
        )


if __name__ == "__main__":
    main()
