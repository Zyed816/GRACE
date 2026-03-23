import argparse
import copy
import itertools
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime

import yaml

# Two-stage grid search for IFL-GR on Cora:
# Stage 1: sweep 4 key hyperparameters (similarity_percentile, max_du_per_node, etc.)
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

    f1mi_mean = float(match.group("f1mi_mean"))
    f1mi_std = float(match.group("f1mi_std"))
    f1ma_mean = float(match.group("f1ma_mean"))
    f1ma_std = float(match.group("f1ma_std"))

    return {
        "F1Mi_mean": f1mi_mean,
        "F1Mi_std": f1mi_std,
        "F1Ma_mean": f1ma_mean,
        "F1Ma_std": f1ma_std,
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


def ensure_seed_consistency(base_config):
    import random
    import numpy as np
    import torch
    cora_seed = base_config["Cora"].get("seed", 39788)
    torch.manual_seed(cora_seed)
    random.seed(12345)
    try:
        np.random.seed(cora_seed)
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="Two-stage grid search for IFL-GR on Cora")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--std_weight", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=3, help="Top candidates to advance to stage 2")
    parser.add_argument("--out", type=str, default="results/grid_search_iflgr_cora_2stage_results.csv")
    args = parser.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    grace_dir = os.path.abspath(os.path.join(tools_dir, ".."))
    config_path = os.path.join(grace_dir, args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Stage 1 search space: 4 key hyperparameters
    stage1_search_space = {
        "similarity_percentile": [99.3, 99.5, 99.7],
        "max_du_per_node": [6, 10, 14],
        "unlabeled_weight": [0.1, 0.2, 0.3],
        "warmup_epochs": [100, 120],
        "tau": [0.4, 0.6],
    }

    stage1_fixed_overrides = {
        "update_interval": 5,
        "similarity_threshold": None,
        "use_mutual_topk": True,
        "beta": 2.0,
        "corrected_ramp_epochs": 40,
    }

    # Stage 2 search space: only learning_rate varies; other params fixed to stage 1 best
    stage2_learning_rates = [5e-4, 1e-3, 2e-3]

    print("=" * 80)
    print("[STAGE 1/2] Grid search: core method parameters")
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
    total_trials_stage1 = len(values_product)

    print(f"[2/3] Stage 1 grid search trials: {total_trials_stage1}")

    stage1_results = []
    for trial_idx, values in enumerate(values_product, start=1):
        trial_params = dict(zip(keys, values))
        trial_params.update(stage1_fixed_overrides)

        # Keep warm-up valid under Cora epochs.
        cora_epochs = base_config["Cora"]["num_epochs"]
        trial_params["warmup_epochs"] = min(trial_params["warmup_epochs"], cora_epochs - 10)

        temp_cfg = make_temp_config(base_config, trial_params)
        ensure_seed_consistency(base_config)

        try:
            metrics, _ = run_train(grace_dir, temp_cfg, method="ifl-gr", gpu_id=args.gpu_id)
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
                f"sim_p={trial_params['similarity_percentile']}, "
                f"max_du={trial_params['max_du_per_node']}, "
                f"lambda_u={trial_params['unlabeled_weight']}, "
                f"warmup={trial_params['warmup_epochs']}, "
                f"tau={trial_params['tau']}"
            )
            print(
                f"Trial {trial_idx:02d}/{total_trials_stage1}: "
                f"F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}, "
                f"robust={score:.4f}, delta={delta:+.4f} | "
                f"{params_str}"
            )
        except Exception as exc:
            print(f"Trial {trial_idx:02d}/{total_trials_stage1} failed: {exc}")
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
    total_trials_stage2 = topk * len(stage2_learning_rates)
    trial_idx_stage2 = 0

    for candidate_rank, candidate_row in enumerate(stage1_results[:topk], start=1):
        print(f"\n--- Candidate #{candidate_rank} from Stage 1 ---")
        print(
            f"Best robust={candidate_row['robust_score']:.4f}, "
            f"F1Mi={candidate_row['F1Mi_mean']:.4f}+-{candidate_row['F1Mi_std']:.4f}"
        )

        # Extract stage 1 best parameters (excluding learning_rate which will vary)
        best_params_stage1 = {k: candidate_row[k] for k in stage1_search_space.keys()}
        best_params_stage1.update(stage1_fixed_overrides)

        for lr_idx, lr in enumerate(stage2_learning_rates, start=1):
            trial_idx_stage2 += 1
            trial_params = copy.deepcopy(best_params_stage1)
            trial_params["learning_rate"] = lr

            # Keep warm-up valid under Cora epochs.
            cora_epochs = base_config["Cora"]["num_epochs"]
            trial_params["warmup_epochs"] = min(trial_params["warmup_epochs"], cora_epochs - 10)

            temp_cfg = make_temp_config(base_config, trial_params)
            ensure_seed_consistency(base_config)

            try:
                metrics, _ = run_train(grace_dir, temp_cfg, method="ifl-gr", gpu_id=args.gpu_id)
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
        "similarity_percentile",
        "max_du_per_node",
        "unlabeled_weight",
        "warmup_epochs",
        "tau",
        "learning_rate",
        "update_interval",
        "beta",
        "use_mutual_topk",
        "corrected_ramp_epochs",
        "F1Mi_mean",
        "F1Mi_std",
        "F1Ma_mean",
        "F1Ma_std",
        "robust_score",
        "delta_vs_grace",
    ]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [",".join(headers)]
    for r in all_results:
        base_rank = r.get("base_candidate_rank", "")
        lines.append(
            ",".join([
                ts,
                r["stage"],
                str(base_rank),
                str(r["similarity_percentile"]),
                str(r["max_du_per_node"]),
                str(r["unlabeled_weight"]),
                str(r["warmup_epochs"]),
                str(r["tau"]),
                str(r.get("learning_rate", base_config["Cora"].get("learning_rate", ""))),
                str(r["update_interval"]),
                str(r["beta"]),
                str(r["use_mutual_topk"]),
                str(r["corrected_ramp_epochs"]),
                f"{r['F1Mi_mean']:.6f}",
                f"{r['F1Mi_std']:.6f}",
                f"{r['F1Ma_mean']:.6f}",
                f"{r['F1Ma_std']:.6f}",
                f"{r['robust_score']:.6f}",
                f"{r['delta_vs_grace']:.6f}",
            ])
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved {len(all_results)} results ({len(stage1_results)} stage1 + {len(stage2_results)} stage2) to: {out_path}")
    print(f"\nTop 5 overall candidates:")
    for i, r in enumerate(all_results[:5], start=1):
        print(
            f"#{i}: stage={r['stage']}, robust={r['robust_score']:.4f}, delta={r['delta_vs_grace']:+.4f}, "
            f"F1Mi={r['F1Mi_mean']:.4f}+-{r['F1Mi_std']:.4f}"
        )


if __name__ == "__main__":
    main()
