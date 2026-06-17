import csv
from pathlib import Path

from .constants import METHOD_LABELS, SENSITIVITY_PARAM_LABELS


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def mean(values):
    return sum(values) / len(values) if values else 0.0


def format_method_pair(target, baseline):
    return f"{METHOD_LABELS.get(target, target)} vs {METHOD_LABELS.get(baseline, baseline)}"


def read_csv_rows(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def preview_csv(path, limit=8):
    rows = read_csv_rows(path)
    headers = list(rows[0].keys()) if rows else []
    return {
        "headers": headers,
        "rows": rows[:limit],
        "total_rows": len(rows),
    }


def read_text_file(path):
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def aggregate_method_rows(method, rows):
    robust_scores = [safe_float(row.get("robust_score")) for row in rows]
    f1mi_scores = [safe_float(row.get("F1Mi_mean")) for row in rows]
    f1ma_scores = [safe_float(row.get("F1Ma_mean")) for row in rows]
    robust_scores = [value for value in robust_scores if value is not None]
    f1mi_scores = [value for value in f1mi_scores if value is not None]
    f1ma_scores = [value for value in f1ma_scores if value is not None]

    return {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "robust_score": mean(robust_scores),
        "F1Mi_mean": mean(f1mi_scores),
        "F1Ma_mean": mean(f1ma_scores),
        "n_rows": len(rows),
    }


def build_method_comparison_summary(path):
    rows = read_csv_rows(path)
    if not rows:
        return {"methods": [], "best_method": "", "best_robust": None}

    method_rows = []
    summary_overall = [
        row for row in rows
        if row.get("stage") == "summary" and row.get("candidate_rank") == "overall"
    ]
    if summary_overall:
        for row in summary_overall:
            robust = safe_float(row.get("robust_score"))
            method_rows.append(
                {
                    "method": row.get("method", ""),
                    "label": METHOD_LABELS.get(row.get("method", ""), row.get("method", "")),
                    "robust_score": robust,
                    "F1Mi_mean": safe_float(row.get("F1Mi_mean"), 0.0),
                    "F1Ma_mean": safe_float(row.get("F1Ma_mean"), 0.0),
                    "delta_vs_grace": safe_float(row.get("delta_vs_grace"), 0.0),
                    "stage": row.get("stage", ""),
                }
            )
    else:
        grouped = {}
        for row in rows:
            stage = row.get("stage")
            if stage not in {"baseline", "top_verify"}:
                continue
            grouped.setdefault(row.get("method", ""), []).append(row)
        for method, method_group in grouped.items():
            aggregated = aggregate_method_rows(method, method_group)
            aggregated["delta_vs_grace"] = safe_float(method_group[-1].get("delta_vs_grace"), 0.0)
            aggregated["stage"] = "aggregate"
            method_rows.append(aggregated)

    method_rows = [row for row in method_rows if row.get("method")]
    method_rows.sort(key=lambda item: item.get("robust_score") or float("-inf"), reverse=True)
    best_row = method_rows[0] if method_rows else {}
    return {
        "methods": method_rows,
        "best_method": best_row.get("label", ""),
        "best_robust": best_row.get("robust_score"),
    }


def build_sampling_bias_summary(path):
    rows = read_csv_rows(path)
    points = []
    for row in rows:
        epoch = safe_int(row.get("epoch"))
        violation_rate = safe_float(row.get("violation_rate"))
        mean_margin = safe_float(row.get("mean_margin"))
        if epoch is None or violation_rate is None or mean_margin is None:
            continue
        points.append(
            {
                "epoch": epoch,
                "violation_rate": violation_rate,
                "mean_margin": mean_margin,
            }
        )

    if not points:
        return {
            "epochs": 0,
            "final_violation_rate": None,
            "best_margin": None,
            "points": [],
        }

    return {
        "epochs": len(points),
        "final_violation_rate": points[-1]["violation_rate"],
        "best_margin": max(point["mean_margin"] for point in points),
        "points": points,
    }


def build_sensitivity_series(csv_paths):
    grouped = {}
    param_order = {"t_s": 0, "M": 1, "K": 2}

    for csv_path in csv_paths:
        rows = read_csv_rows(csv_path)
        for row in rows:
            if row.get("stage") != "summary":
                continue

            method = row.get("method", "")
            param = row.get("paper_param", "")
            sweep_value = row.get("sweep_value", "")
            robust_score = safe_float(row.get("robust_score"))
            if not method or not param or robust_score is None or sweep_value == "":
                continue

            grouped.setdefault(param, {}).setdefault(method, []).append(
                {
                    "x": safe_float(sweep_value),
                    "label": str(sweep_value),
                    "value": robust_score,
                }
            )

    series_groups = []
    for param, methods in grouped.items():
        rendered_series = []
        for method, points in methods.items():
            unique_points = {}
            for point in points:
                unique_points[point["label"]] = point
            ordered_points = sorted(
                unique_points.values(),
                key=lambda item: (item["x"] is None, item["x"] if item["x"] is not None else item["label"]),
            )
            rendered_series.append(
                {
                    "method": method,
                    "label": METHOD_LABELS.get(method, method),
                    "points": ordered_points,
                }
            )

        rendered_series.sort(key=lambda item: item["label"])
        series_groups.append(
            {
                "param": param,
                "label": SENSITIVITY_PARAM_LABELS.get(param, param),
                "series": rendered_series,
            }
        )

    series_groups.sort(key=lambda item: (param_order.get(item["param"], 99), item["label"]))
    return series_groups


def build_sensitivity_summary(csv_paths, report_path=None):
    best_rows = []
    total_summary_rows = 0
    methods = set()
    params = set()

    for csv_path in csv_paths:
        rows = read_csv_rows(csv_path)
        summary_rows = [row for row in rows if row.get("stage") == "summary"]
        total_summary_rows += len(summary_rows)

        best_per_method = {}
        for row in summary_rows:
            method = row.get("method", "")
            robust = safe_float(row.get("robust_score"))
            if not method or robust is None:
                continue
            methods.add(method)
            if row.get("paper_param"):
                params.add(row["paper_param"])
            candidate = {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "robust_score": robust,
                "paper_param": row.get("paper_param", ""),
                "sweep_value": row.get("sweep_value", ""),
            }
            previous = best_per_method.get(method)
            if previous is None or robust > previous["robust_score"]:
                best_per_method[method] = candidate

        best_rows.extend(best_per_method.values())

    best_rows.sort(key=lambda item: item.get("robust_score") or float("-inf"), reverse=True)

    return {
        "best_rows": best_rows,
        "summary_rows": total_summary_rows,
        "methods": [METHOD_LABELS.get(method, method) for method in sorted(methods)],
        "params": sorted(params, key=lambda item: {"t_s": 0, "M": 1, "K": 2}.get(item, 99)),
        "sensitivity_series": build_sensitivity_series(csv_paths),
        "report_text": read_text_file(report_path) if report_path else "",
    }


def build_component_ablation_summary(csv_paths, report_path=None):
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]

    summary_rows = []
    methods = set()
    variants = set()
    for csv_path in csv_paths:
        for row in read_csv_rows(csv_path):
            if row.get("stage") != "summary":
                continue
            method = row.get("method", "")
            variant = row.get("variant", "")
            if method:
                methods.add(method)
            if variant:
                variants.add(variant)
            summary_rows.append(row)

    ablation_rows = [row for row in summary_rows if row.get("variant") and row.get("variant") != "full"]
    max_drop_row = None
    for row in ablation_rows:
        drop = safe_float(row.get("drop_vs_full"))
        if drop is None:
            continue
        if max_drop_row is None or drop > max_drop_row["drop_vs_full"]:
            max_drop_row = {
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "label": METHOD_LABELS.get(row.get("method", ""), row.get("method", "")),
                "variant": row.get("variant", ""),
                "drop_vs_full": drop,
                "robust_score": safe_float(row.get("robust_score")),
            }

    drops_by_variant = {}
    for row in ablation_rows:
        variant = row.get("variant", "")
        drop = safe_float(row.get("drop_vs_full"))
        if not variant or drop is None:
            continue
        drops_by_variant.setdefault(variant, []).append(drop)

    mean_drop_by_variant = [
        {
            "variant": variant,
            "label": {
                "no_warmup": "M-off",
                "single_mining": "K-off",
                "uniform_weight": "w-off",
            }.get(variant, variant),
            "mean_drop": mean(values),
            "n_rows": len(values),
        }
        for variant, values in sorted(drops_by_variant.items())
    ]

    return {
        "summary_rows": len(summary_rows),
        "methods": sorted(methods),
        "method_labels": [METHOD_LABELS.get(method, method) for method in sorted(methods)],
        "variants": sorted(variants),
        "variant_count": len(variants),
        "max_drop": max_drop_row,
        "mean_drop_by_variant": mean_drop_by_variant,
        "report_text": read_text_file(report_path) if report_path else "",
    }


def build_efficiency_summary(csv_paths, report_path=None):
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]

    summary_rows = []
    for csv_path in csv_paths:
        summary_rows.extend(row for row in read_csv_rows(csv_path) if row.get("stage") == "summary")

    method_rows = []
    for row in summary_rows:
        method = row.get("method", "")
        train_time = safe_float(row.get("train_total_sec"))
        method_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "train_total_sec": train_time,
                "wall_time_sec": safe_float(row.get("wall_time_sec")),
                "time_ratio_vs_grace": safe_float(row.get("time_ratio_vs_grace")),
                "overhead_ratio_vs_base": safe_float(row.get("overhead_ratio_vs_base")),
            }
        )

    valid_times = [row for row in method_rows if row["train_total_sec"] is not None]
    fastest = min(valid_times, key=lambda row: row["train_total_sec"]) if valid_times else None
    by_method = {row["method"]: row for row in method_rows}

    return {
        "summary_rows": len(summary_rows),
        "methods": [row["method"] for row in method_rows],
        "method_rows": method_rows,
        "fastest_method": fastest,
        "sggr_ratio": (by_method.get("sg-gr") or {}).get("time_ratio_vs_grace"),
        "sggc_ratio": (by_method.get("sg-gc") or {}).get("overhead_ratio_vs_base"),
        "report_text": read_text_file(report_path) if report_path else "",
    }


def build_significance_summary(csv_paths, summary_csv=None, report_path=None):
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]

    source_paths = [summary_csv] if summary_csv else csv_paths
    test_rows = []
    for csv_path in source_paths:
        test_rows.extend(row for row in read_csv_rows(csv_path) if row.get("stage") == "test")

    primary_rows = [
        row
        for row in test_rows
        if row.get("metric") == "robust_score"
        and row.get("notes") == "primary"
        and (row.get("baseline_method"), row.get("target_method")) in {("grace", "sg-gr"), ("gca", "sg-gc")}
    ]
    significant_rows = [row for row in primary_rows if row.get("significant") == "True"]

    best_delta_row = None
    for row in primary_rows:
        delta = safe_float(row.get("mean_delta"))
        if delta is None:
            continue
        if best_delta_row is None or delta > best_delta_row["mean_delta"]:
            best_delta_row = {
                "dataset": row.get("dataset", ""),
                "comparison": format_method_pair(row.get("target_method", ""), row.get("baseline_method", "")),
                "mean_delta": delta,
                "p_value_holm": safe_float(row.get("p_value_holm")),
                "significant": row.get("significant") == "True",
            }

    tests = [
        {
            "dataset": row.get("dataset", ""),
            "comparison": format_method_pair(row.get("target_method", ""), row.get("baseline_method", "")),
            "mean_delta": safe_float(row.get("mean_delta")),
            "p_value_holm": safe_float(row.get("p_value_holm")),
            "significant": row.get("significant") == "True",
        }
        for row in primary_rows
    ]

    return {
        "primary_tests": len(primary_rows),
        "significant_tests": len(significant_rows),
        "tests": tests,
        "best_delta": best_delta_row,
        "report_text": read_text_file(report_path) if report_path else "",
    }
