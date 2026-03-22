import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from time import perf_counter as t

import yaml


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
    return metrics["F1Mi_mean"] - std_weight * metrics["F1Mi_std"]


def run_train(grace_dir, config_path, method, gpu_id):
    start = t()
    print(f"[train:{method}] start")
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
    print(
        f"[train:{method}] done in {t() - start:.1f}s | "
        f"F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}"
    )
    return metrics, combined


def run_grid_script(grace_dir, script_name, gpu_id, topk, std_weight):
    start = t()
    print(f"[grid:{script_name}] start")
    cmd = [
        sys.executable,
        os.path.join("tools", script_name),
        "--gpu_id",
        str(gpu_id),
        "--topk",
        str(topk),
        "--std_weight",
        str(std_weight),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=grace_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
    )

    output_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line)
        print(f"[grid:{script_name}] {line.rstrip()}")

    proc.wait()
    combined_output = "".join(output_lines)

    if proc.returncode != 0:
        raise RuntimeError(
            "Grid search failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {proc.returncode}\n"
            f"OUTPUT:\n{combined_output}"
        )

    print(f"[grid:{script_name}] done in {t() - start:.1f}s")
    return combined_output


def read_top_rows(csv_path, topk):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if idx > topk:
                break
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in grid csv: {csv_path}")
    return rows


def make_temp_config_for_method(base_config, csv_row, method):
    cfg = copy.deepcopy(base_config)

    if method == "ifl-gr":
        cora_updates = {
            "similarity_percentile": float(csv_row["similarity_percentile"]),
            "max_du_per_node": int(float(csv_row["max_du_per_node"])),
            "unlabeled_weight": float(csv_row["unlabeled_weight"]),
            "warmup_epochs": int(float(csv_row["warmup_epochs"])),
            "update_interval": int(float(csv_row["update_interval"])),
            "beta": float(csv_row["beta"]),
            "use_mutual_topk": csv_row["use_mutual_topk"].lower() == "true",
            "corrected_ramp_epochs": int(float(csv_row["corrected_ramp_epochs"])),
        }
        if csv_row.get("similarity_threshold", "").lower() in ["none", "null", ""]:
            cora_updates["similarity_threshold"] = None
        elif "similarity_threshold" in csv_row:
            cora_updates["similarity_threshold"] = float(csv_row["similarity_threshold"])

    elif method == "gca":
        cora_updates = {
            "gca_drop_scheme": csv_row["gca_drop_scheme"],
            "drop_edge_rate_1": float(csv_row["drop_edge_rate_1"]),
            "drop_edge_rate_2": float(csv_row["drop_edge_rate_2"]),
            "drop_feature_rate_1": float(csv_row["drop_feature_rate_1"]),
            "drop_feature_rate_2": float(csv_row["drop_feature_rate_2"]),
            "tau": float(csv_row["tau"]),
        }
        if "gca_pr_k" in csv_row and csv_row["gca_pr_k"] != "":
            cora_updates["gca_pr_k"] = int(float(csv_row["gca_pr_k"]))

    elif method == "ifl-gc":
        cora_updates = {
            "gca_drop_scheme": csv_row["gca_drop_scheme"],
            "similarity_percentile": float(csv_row["similarity_percentile"]),
            "max_du_per_node": int(float(csv_row["max_du_per_node"])),
            "unlabeled_weight": float(csv_row["unlabeled_weight"]),
            "iflgc_refl_du_weight": float(csv_row["iflgc_refl_du_weight"]),
            "warmup_epochs": int(float(csv_row["warmup_epochs"])),
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
            cora_updates["gca_pr_k"] = int(float(csv_row["gca_pr_k"]))
        if csv_row.get("similarity_threshold", "").lower() in ["none", "null", ""]:
            cora_updates["similarity_threshold"] = None
        elif "similarity_threshold" in csv_row:
            cora_updates["similarity_threshold"] = float(csv_row["similarity_threshold"])

    else:
        raise ValueError(f"Unsupported method: {method}")

    cfg["Cora"].update(cora_updates)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        return f.name


def append_result_row(csv_path, row, write_header=False):
    headers = [
        "timestamp",
        "stage",
        "method",
        "candidate_rank",
        "run_idx",
        "F1Mi_mean",
        "F1Mi_std",
        "F1Ma_mean",
        "F1Ma_std",
        "robust_score",
        "delta_vs_grace",
        "grid_csv",
        "params_json",
        "notes",
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def method_pipeline(grace_dir, base_config, method, grid_script, grid_csv_name, args, baseline_robust):
    print(f"\n=== [{method}] grid search ===")
    run_grid_script(
        grace_dir=grace_dir,
        script_name=grid_script,
        gpu_id=args.gpu_id,
        topk=max(args.topk_verify, 10),
        std_weight=args.std_weight,
    )

    grid_csv_path = os.path.join(grace_dir, "results", grid_csv_name)
    top_rows = read_top_rows(grid_csv_path, args.topk_verify)

    print(f"=== [{method}] top-{args.topk_verify} verification ({args.runs_per_top} runs each) ===")
    for rank, csv_row in enumerate(top_rows, start=1):
        print(f"[{method}] candidate #{rank}")
        for run_idx in range(1, args.runs_per_top + 1):
            print(f"[{method}] candidate #{rank} run {run_idx}/{args.runs_per_top} start")
            temp_cfg = make_temp_config_for_method(base_config, csv_row, method)
            try:
                metrics, _ = run_train(grace_dir, temp_cfg, method=method, gpu_id=args.gpu_id)
                score = robust_score(metrics, args.std_weight)
                delta = score - baseline_robust

                append_result_row(
                    csv_path=args.out,
                    row={
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stage": "top_verify",
                        "method": method,
                        "candidate_rank": rank,
                        "run_idx": run_idx,
                        "F1Mi_mean": f"{metrics['F1Mi_mean']:.6f}",
                        "F1Mi_std": f"{metrics['F1Mi_std']:.6f}",
                        "F1Ma_mean": f"{metrics['F1Ma_mean']:.6f}",
                        "F1Ma_std": f"{metrics['F1Ma_std']:.6f}",
                        "robust_score": f"{score:.6f}",
                        "delta_vs_grace": f"{delta:.6f}",
                        "grid_csv": os.path.relpath(grid_csv_path, grace_dir).replace("\\", "/"),
                        "params_json": json.dumps(csv_row, ensure_ascii=True),
                        "notes": "",
                    },
                )

                print(
                    f"  run {run_idx}/{args.runs_per_top}: "
                    f"F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}, "
                    f"robust={score:.4f}, delta={delta:+.4f}"
                )
            finally:
                if os.path.exists(temp_cfg):
                    os.remove(temp_cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Automate Cora comparison: GRACE baseline + IFL-GR/GCA/IFL-GC search and top verification"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--std_weight", type=float, default=0.5)
    parser.add_argument("--baseline_runs", type=int, default=3)
    parser.add_argument("--topk_verify", type=int, default=3)
    parser.add_argument("--runs_per_top", type=int, default=3)
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("results", "cora_full_pipeline_results.csv"),
        help="Output CSV (single file for all methods)",
    )
    args = parser.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    grace_dir = os.path.abspath(os.path.join(tools_dir, ".."))
    config_path = os.path.join(grace_dir, args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    out_path = os.path.join(grace_dir, args.out)
    if os.path.exists(out_path):
        os.remove(out_path)

    # Write CSV header once.
    append_result_row(
        csv_path=out_path,
        row={
            "timestamp": "",
            "stage": "",
            "method": "",
            "candidate_rank": "",
            "run_idx": "",
            "F1Mi_mean": "",
            "F1Mi_std": "",
            "F1Ma_mean": "",
            "F1Ma_std": "",
            "robust_score": "",
            "delta_vs_grace": "",
            "grid_csv": "",
            "params_json": "",
            "notes": "",
        },
        write_header=True,
    )

    print("=== [grace] baseline runs ===")
    baseline_scores = []
    for run_idx in range(1, args.baseline_runs + 1):
        metrics, _ = run_train(grace_dir, config_path, method="grace", gpu_id=args.gpu_id)
        score = robust_score(metrics, args.std_weight)
        baseline_scores.append(score)

        append_result_row(
            csv_path=out_path,
            row={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stage": "baseline",
                "method": "grace",
                "candidate_rank": 0,
                "run_idx": run_idx,
                "F1Mi_mean": f"{metrics['F1Mi_mean']:.6f}",
                "F1Mi_std": f"{metrics['F1Mi_std']:.6f}",
                "F1Ma_mean": f"{metrics['F1Ma_mean']:.6f}",
                "F1Ma_std": f"{metrics['F1Ma_std']:.6f}",
                "robust_score": f"{score:.6f}",
                "delta_vs_grace": f"{0.0:.6f}",
                "grid_csv": "",
                "params_json": "{}",
                "notes": "baseline reference",
            },
        )

        print(
            f"  run {run_idx}/{args.baseline_runs}: "
            f"F1Mi={metrics['F1Mi_mean']:.4f}+-{metrics['F1Mi_std']:.4f}, robust={score:.4f}"
        )

    baseline_robust = sum(baseline_scores) / len(baseline_scores)
    print(f"Baseline robust reference (mean over runs): {baseline_robust:.4f}")

    method_pipeline(
        grace_dir=grace_dir,
        base_config=base_config,
        method="ifl-gr",
        grid_script="grid_search_iflgr_cora.py",
        grid_csv_name="grid_search_iflgr_cora_results.csv",
        args=args,
        baseline_robust=baseline_robust,
    )

    method_pipeline(
        grace_dir=grace_dir,
        base_config=base_config,
        method="gca",
        grid_script="grid_search_gca_cora.py",
        grid_csv_name="grid_search_gca_cora_results.csv",
        args=args,
        baseline_robust=baseline_robust,
    )

    method_pipeline(
        grace_dir=grace_dir,
        base_config=base_config,
        method="ifl-gc",
        grid_script="grid_search_iflgc_cora.py",
        grid_csv_name="grid_search_iflgc_cora_results.csv",
        args=args,
        baseline_robust=baseline_robust,
    )

    print(f"\nAll stages completed. Unified results saved to: {out_path}")


if __name__ == "__main__":
    main()
