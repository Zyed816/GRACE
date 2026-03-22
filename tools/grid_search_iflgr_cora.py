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

# Grid-search pipeline (IFL-GR on Cora):
# 1) run GRACE baseline
# 2) sweep parameter grid with temporary config files
# 3) rank by robust_score and save CSV

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


def make_temp_config(base_config, dataset_key, dataset_updates):
    cfg = copy.deepcopy(base_config)
    cfg[dataset_key].update(dataset_updates)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        return f.name


def ensure_seed_consistency(base_config, dataset_key):
    import random
    import numpy as np
    import torch
    dataset_seed = base_config[dataset_key].get("seed", 39788)
    torch.manual_seed(dataset_seed)
    random.seed(12345)
    try:
        np.random.seed(dataset_seed)
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="Grid search for IFL-GR on selected dataset")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed", "DBLP"])
    parser.add_argument("--std_weight", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    grace_dir = os.path.abspath(os.path.join(tools_dir, ".."))
    config_path = os.path.join(grace_dir, args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    dataset_key = args.dataset
    dataset_slug = dataset_key.lower()
    out_rel_path = args.out if args.out else f"results/grid_search_iflgr_{dataset_slug}_results.csv"

    # Search key hyperparameters with a compact grid.
    search_space = {
        "similarity_percentile": [99.3, 99.5, 99.7],
        "max_du_per_node": [6, 10, 14],
        "unlabeled_weight": [0.1, 0.2, 0.3],
        "warmup_epochs": [100, 120],
        "tau": [0.4, 0.6],
    }

    fixed_overrides = {
        "update_interval": 5,
        "similarity_threshold": None,
        "use_mutual_topk": True,
        "beta": 2.0,
        "corrected_ramp_epochs": 40,
    }

    print(f"[1/3] Running GRACE baseline on {dataset_key}...")
    baseline_metrics, _ = run_train(
        grace_dir,
        config_path,
        dataset=dataset_key,
        method="grace",
        gpu_id=args.gpu_id,
    )
    baseline_score = robust_score(baseline_metrics, args.std_weight)
    print(
        "Baseline GRACE: "
        f"F1Mi={baseline_metrics['F1Mi_mean']:.4f}+-{baseline_metrics['F1Mi_std']:.4f}, "
        f"robust={baseline_score:.4f}"
    )

    keys = list(search_space.keys())
    values_product = list(itertools.product(*(search_space[k] for k in keys)))
    total_trials = len(values_product)

    print(f"[2/3] Grid search trials: {total_trials}")

    results = []
    for trial_idx, values in enumerate(values_product, start=1):
        trial_params = dict(zip(keys, values))
        trial_params.update(fixed_overrides)

        # Keep warm-up valid under current dataset epochs.
        num_epochs = base_config[dataset_key]["num_epochs"]
        trial_params["warmup_epochs"] = min(trial_params["warmup_epochs"], num_epochs - 10)

        temp_cfg = make_temp_config(base_config, dataset_key, trial_params)
        ensure_seed_consistency(base_config, dataset_key)

        try:
            metrics, _ = run_train(
                grace_dir,
                temp_cfg,
                dataset=dataset_key,
                method="ifl-gr",
                gpu_id=args.gpu_id,
            )
            score = robust_score(metrics, args.std_weight)
            delta = score - baseline_score

            row = {
                **trial_params,
                **metrics,
                "robust_score": score,
                "delta_vs_grace": delta,
            }
            results.append(row)

            params_str = (
                f"sim_p={trial_params['similarity_percentile']}, "
                f"max_du={trial_params['max_du_per_node']}, "
                f"lambda_u={trial_params['unlabeled_weight']}, "
                f"warmup={trial_params['warmup_epochs']}, "
                f"tau={trial_params['tau']}"
            )
            print(
                f"Trial {trial_idx:02d}/{total_trials}: "
                f"F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}, "
                f"robust={score:.4f}, delta={delta:+.4f} | "
                f"{params_str}"
            )
        except Exception as exc:
            print(f"Trial {trial_idx:02d}/{total_trials} failed: {exc}")
        finally:
            if os.path.exists(temp_cfg):
                os.remove(temp_cfg)

    if not results:
        raise RuntimeError("All trials failed; no valid results generated.")

    results.sort(key=lambda x: x["robust_score"], reverse=True)

    topk = min(args.topk, len(results))
    print("[3/3] Top candidates:")
    for i in range(topk):
        r = results[i]
        print(
            f"#{i+1}: robust={r['robust_score']:.4f}, delta={r['delta_vs_grace']:+.4f}, "
            f"F1Mi={r['F1Mi_mean']:.4f}+-{r['F1Mi_std']:.4f}, "
            f"params={{similarity_percentile={r['similarity_percentile']}, "
            f"max_du_per_node={r['max_du_per_node']}, "
            f"unlabeled_weight={r['unlabeled_weight']}, "
            f"warmup_epochs={r['warmup_epochs']}, "
            f"tau={r['tau']}}}"
        )

    out_path = os.path.join(grace_dir, out_rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = [
        "timestamp",
        "similarity_percentile",
        "max_du_per_node",
        "unlabeled_weight",
        "warmup_epochs",
        "tau",
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
    for r in results:
        lines.append(
            ",".join([
                ts,
                str(r["similarity_percentile"]),
                str(r["max_du_per_node"]),
                str(r["unlabeled_weight"]),
                str(r["warmup_epochs"]),
                str(r["tau"]),
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

    print(f"Saved full results to: {out_path}")


if __name__ == "__main__":
    main()
