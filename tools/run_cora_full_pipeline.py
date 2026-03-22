import argparse
import copy
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from time import perf_counter as t
import time

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


def run_train(grace_dir, config_path, dataset, method, gpu_id):
    start = t()
    print(f"[train:{method}] start")
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

    if dataset == "PubMed":
        print(
            f"[train:{method}] PubMed detected; first output may take longer. "
            "Will print heartbeat every 30s."
        )
        with tempfile.NamedTemporaryFile("w+", suffix="_pubmed_train.log", delete=False, encoding="utf-8") as lf:
            log_path = lf.name

        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.Popen(
                    cmd,
                    cwd=grace_dir,
                    text=True,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )

                last_heartbeat = t()
                while proc.poll() is None:
                    now = t()
                    if now - last_heartbeat >= 30.0:
                        elapsed = int(now - start)
                        print(
                            f"[train:{method}] PubMed still running... "
                            f"elapsed={elapsed}s"
                        )
                        last_heartbeat = now
                    time.sleep(2.0)

            with open(log_path, "r", encoding="utf-8") as logf:
                combined = logf.read()

            if proc.returncode != 0:
                raise RuntimeError(
                    "Training failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Return code: {proc.returncode}\n"
                    f"OUTPUT:\n{combined}"
                )
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    else:
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


def run_grid_script(grace_dir, script_name, gpu_id, topk, std_weight, dataset):
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
        "--dataset",
        dataset,
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if dataset == "PubMed":
        print(
            f"[grid:{script_name}] PubMed detected; per-trial output may be sparse. "
            "Will print heartbeat every 30s."
        )
        with tempfile.NamedTemporaryFile("w+", suffix="_pubmed_grid.log", delete=False, encoding="utf-8") as lf:
            log_path = lf.name

        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.Popen(
                    cmd,
                    cwd=grace_dir,
                    text=True,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

                last_heartbeat = t()
                while proc.poll() is None:
                    now = t()
                    if now - last_heartbeat >= 30.0:
                        elapsed = int(now - start)
                        print(
                            f"[grid:{script_name}] PubMed grid search still running... "
                            f"elapsed={elapsed}s"
                        )
                        last_heartbeat = now
                    time.sleep(2.0)

            with open(log_path, "r", encoding="utf-8") as logf:
                combined_output = logf.read()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    else:
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


def try_read_top_rows(csv_path, topk):
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if idx > topk:
                break
            rows.append(row)
    return rows


def make_temp_config_for_method(base_config, dataset_key, csv_row, method):
    cfg = copy.deepcopy(base_config)

    if method == "ifl-gr":
        dataset_updates = {
            "similarity_percentile": float(csv_row["similarity_percentile"]),
            "max_du_per_node": int(float(csv_row["max_du_per_node"])),
            "unlabeled_weight": float(csv_row["unlabeled_weight"]),
            "warmup_epochs": int(float(csv_row["warmup_epochs"])),
            "update_interval": int(float(csv_row["update_interval"])),
            "beta": float(csv_row["beta"]),
            "use_mutual_topk": csv_row["use_mutual_topk"].lower() == "true",
            "corrected_ramp_epochs": int(float(csv_row["corrected_ramp_epochs"])),
        }
        if "tau" in csv_row and csv_row["tau"] != "":
            dataset_updates["tau"] = float(csv_row["tau"])
        if csv_row.get("similarity_threshold", "").lower() in ["none", "null", ""]:
            dataset_updates["similarity_threshold"] = None
        elif "similarity_threshold" in csv_row:
            dataset_updates["similarity_threshold"] = float(csv_row["similarity_threshold"])

    elif method == "gca":
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

    elif method == "ifl-gc":
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

    else:
        raise ValueError(f"Unsupported method: {method}")

    cfg[dataset_key].update(dataset_updates)

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
        if row is not None:
            writer.writerow(row)


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def _safe_std(values):
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0
    return statistics.pstdev(values)


def append_method_summary_rows(csv_path):
    """
    Hierarchical summary output:
    1. For each (method, candidate_rank), output per-candidate mean (one row per candidate).
    2. For each method, then output overall mean across all candidates (one final row per method).
    Baseline (grace) has only 1 candidate, others (ifl-gr, gca, ifl-gc) have multiple.
    """
    if not os.path.exists(csv_path):
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Keep only raw experiment rows; skip any existing summary rows.
    valid_rows = [r for r in rows if r.get("stage") in {"baseline", "top_verify"}]
    if not valid_rows:
        return

    # Group by (method, candidate_rank) and compute per-candidate mean
    by_method_and_rank = {}
    for r in valid_rows:
        method = r.get("method", "").strip()
        candidate_rank = r.get("candidate_rank", "").strip()
        if not method:
            continue

        try:
            f1mi = float(r["F1Mi_mean"])
            f1ma = float(r["F1Ma_mean"])
            robust = float(r["robust_score"])
        except (ValueError, KeyError, TypeError):
            continue

        key = (method, candidate_rank)
        if key not in by_method_and_rank:
            by_method_and_rank[key] = {"f1mi": [], "f1ma": [], "robust": []}
        by_method_and_rank[key]["f1mi"].append(f1mi)
        by_method_and_rank[key]["f1ma"].append(f1ma)
        by_method_and_rank[key]["robust"].append(robust)

    if not by_method_and_rank:
        return

    # Compute per-candidate means and collect for overall method mean
    candidate_means_by_method = {}
    candidate_order_by_method = {}
    for (method, rank), vals in sorted(by_method_and_rank.items()):
        if method not in candidate_means_by_method:
            candidate_means_by_method[method] = {}
            candidate_order_by_method[method] = []

        rank_key = rank  # Keep rank as string for ordering
        candidate_order_by_method[method].append(rank_key)

        candidate_means_by_method[method][rank_key] = {
            "f1mi": _safe_mean(vals["f1mi"]),
            "f1ma": _safe_mean(vals["f1ma"]),
            "robust": _safe_mean(vals["robust"]),
            "count": len(vals["robust"]),
        }

    # Compute baseline robust reference (grace)
    grace_candidates = candidate_means_by_method.get("grace", {})
    grace_robust_ref = _safe_mean([c["robust"] for c in grace_candidates.values()]) if grace_candidates else 0.0

    preferred_order = ["grace", "ifl-gr", "gca", "ifl-gc"]
    methods = [m for m in preferred_order if m in candidate_means_by_method]
    methods.extend(
        sorted([m for m in candidate_means_by_method.keys() if m not in preferred_order])
    )

    # Output per-candidate rows, then overall row for each method
    for method in methods:
        candidates = candidate_means_by_method[method]
        ranks = sorted(set(candidate_order_by_method[method]), key=lambda x: (x != "", int(x) if x else -1))

        # Per-candidate summary rows
        for rank in ranks:
            if rank not in candidates:
                continue
            cand = candidates[rank]
            delta_vs_grace = cand["robust"] - grace_robust_ref

            append_result_row(
                csv_path=csv_path,
                row={
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stage": "summary",
                    "method": method,
                    "candidate_rank": rank,
                    "run_idx": "",
                    "F1Mi_mean": f"{cand['f1mi']:.6f}",
                    "F1Mi_std": "",  # No std at per-candidate level in summary
                    "F1Ma_mean": f"{cand['f1ma']:.6f}",
                    "F1Ma_std": "",
                    "robust_score": f"{cand['robust']:.6f}",
                    "delta_vs_grace": f"{delta_vs_grace:.6f}",
                    "grid_csv": "",
                    "params_json": json.dumps({"n_runs": cand["count"]}, ensure_ascii=True),
                    "notes": f"candidate #{rank} mean",
                },
            )

        # Overall method mean (across all candidates)
        all_robust = [candidates[rank]["robust"] for rank in ranks]
        all_f1mi = [candidates[rank]["f1mi"] for rank in ranks]
        all_f1ma = [candidates[rank]["f1ma"] for rank in ranks]
        total_count = sum(candidates[rank]["count"] for rank in ranks)

        f1mi_overall = _safe_mean(all_f1mi)
        f1mi_std_of_candidates = _safe_std(all_f1mi)
        f1ma_overall = _safe_mean(all_f1ma)
        f1ma_std_of_candidates = _safe_std(all_f1ma)
        robust_overall = _safe_mean(all_robust)
        delta_vs_grace = robust_overall - grace_robust_ref

        append_result_row(
            csv_path=csv_path,
            row={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stage": "summary",
                "method": method,
                "candidate_rank": "overall",
                "run_idx": "",
                "F1Mi_mean": f"{f1mi_overall:.6f}",
                "F1Mi_std": f"{f1mi_std_of_candidates:.6f}",
                "F1Ma_mean": f"{f1ma_overall:.6f}",
                "F1Ma_std": f"{f1ma_std_of_candidates:.6f}",
                "robust_score": f"{robust_overall:.6f}",
                "delta_vs_grace": f"{delta_vs_grace:.6f}",
                "grid_csv": "",
                "params_json": json.dumps(
                    {"n_candidates": len(ranks), "n_runs": total_count},
                    ensure_ascii=True,
                ),
                "notes": "method overall mean across all candidates",
            },
        )


def method_pipeline(grace_dir, base_config, dataset_key, method, grid_script, grid_csv_name, args, baseline_robust, out_csv_path):
    grid_csv_path = os.path.join(grace_dir, "results", grid_csv_name)
    top_rows = []

    if (not args.force_grid) and os.path.exists(grid_csv_path):
        top_rows = try_read_top_rows(grid_csv_path, args.topk_verify)
        if top_rows:
            print(
                f"\n=== [{method}] found existing candidate file: {grid_csv_path} | "
                f"skip grid search, run top-{len(top_rows)} verification directly ==="
            )
        else:
            print(
                f"\n=== [{method}] existing candidate file is empty/invalid: {grid_csv_path} | "
                "will run grid search ==="
            )

    if not top_rows:
        print(f"\n=== [{method}] grid search ===")
        run_grid_script(
            grace_dir=grace_dir,
            script_name=grid_script,
            gpu_id=args.gpu_id,
            topk=max(args.topk_verify, 10),
            std_weight=args.std_weight,
            dataset=dataset_key,
        )
        top_rows = read_top_rows(grid_csv_path, args.topk_verify)

    print(f"=== [{method}] top-{len(top_rows)} verification ({args.runs_per_top} runs each) ===")
    for rank, csv_row in enumerate(top_rows, start=1):
        print(f"[{method}] candidate #{rank}")
        for run_idx in range(1, args.runs_per_top + 1):
            print(f"[{method}] candidate #{rank} run {run_idx}/{args.runs_per_top} start")
            temp_cfg = make_temp_config_for_method(base_config, dataset_key, csv_row, method)
            try:
                metrics, _ = run_train(
                    grace_dir,
                    temp_cfg,
                    dataset=dataset_key,
                    method=method,
                    gpu_id=args.gpu_id,
                )
                score = robust_score(metrics, args.std_weight)
                delta = score - baseline_robust

                append_result_row(
                    csv_path=out_csv_path,
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
        description="Automate dataset comparison: GRACE baseline + IFL-GR/GCA/IFL-GC search and top verification"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed", "DBLP"])
    parser.add_argument("--std_weight", type=float, default=0.5)
    parser.add_argument("--baseline_runs", type=int, default=3)
    parser.add_argument("--topk_verify", type=int, default=3)
    parser.add_argument("--runs_per_top", type=int, default=3)
    parser.add_argument(
        "--force_grid",
        action="store_true",
        help="Force rerun grid search even if an existing candidate CSV is found.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV (single file for all methods)",
    )
    args = parser.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    grace_dir = os.path.abspath(os.path.join(tools_dir, ".."))
    config_path = os.path.join(grace_dir, args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    dataset_slug = args.dataset.lower()
    out_rel_path = args.out if args.out else os.path.join("results", f"{dataset_slug}_full_pipeline_results.csv")
    out_path = os.path.join(grace_dir, out_rel_path)
    if os.path.exists(out_path):
        os.remove(out_path)

    # Write CSV header once.
    append_result_row(
        csv_path=out_path,
        row=None,
        write_header=True,
    )

    print("=== [grace] baseline runs ===")
    baseline_scores = []
    for run_idx in range(1, args.baseline_runs + 1):
        metrics, _ = run_train(
            grace_dir,
            config_path,
            dataset=args.dataset,
            method="grace",
            gpu_id=args.gpu_id,
        )
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
        dataset_key=args.dataset,
        method="ifl-gr",
        grid_script="grid_search_iflgr_cora.py",
        grid_csv_name=f"grid_search_iflgr_{dataset_slug}_results.csv",
        args=args,
        baseline_robust=baseline_robust,
        out_csv_path=out_path,
    )

    method_pipeline(
        grace_dir=grace_dir,
        base_config=base_config,
        dataset_key=args.dataset,
        method="gca",
        grid_script="grid_search_gca_cora.py",
        grid_csv_name=f"grid_search_gca_{dataset_slug}_results.csv",
        args=args,
        baseline_robust=baseline_robust,
        out_csv_path=out_path,
    )

    method_pipeline(
        grace_dir=grace_dir,
        base_config=base_config,
        dataset_key=args.dataset,
        method="ifl-gc",
        grid_script="grid_search_iflgc_cora.py",
        grid_csv_name=f"grid_search_iflgc_{dataset_slug}_results.csv",
        args=args,
        baseline_robust=baseline_robust,
        out_csv_path=out_path,
    )

    print("\n=== [summary] aggregate method-level statistics ===")
    append_method_summary_rows(out_path)
    print("[summary] appended method aggregate rows")

    print(f"\nAll stages completed. Unified results saved to: {out_path}")


if __name__ == "__main__":
    main()
