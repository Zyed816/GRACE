import csv
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml
from django.conf import settings
from django.utils import timezone

from .models import Experiment, ExperimentLog, ExperimentMetric, PipelineResult


F1_PATTERN = re.compile(
    r"\(E\) \| label_classification: "
    r"F1Mi=(?P<f1mi_mean>\d+\.\d+)\+-(?P<f1mi_std>\d+\.\d+), "
    r"F1Ma=(?P<f1ma_mean>\d+\.\d+)\+-(?P<f1ma_std>\d+\.\d+)"
)

EPOCH_PATTERN = re.compile(
    r"\(T\) \| Epoch=(?P<epoch>\d+).*?"
    r"loss=(?P<loss>-?\d+\.\d+).*?"
    r"v_rate=(?P<violation_rate>-?\d+\.\d+).*?"
    r"m_margin=(?P<mean_margin>-?\d+\.\d+).*?"
    r"p10=(?P<p10_margin>-?\d+\.\d+)"
)

GRID_SCRIPT_BY_METHOD = {
    "ifl-gr": "grid_search_iflgr_cora.py",
    "gca": "grid_search_gca_cora.py",
    "ifl-gc": "grid_search_iflgc_cora.py",
}

GRID_CSV_PREFIX_BY_METHOD = {
    "ifl-gr": "iflgr",
    "gca": "gca",
    "ifl-gc": "iflgc",
}

DONE_STATES = [Experiment.STATUS_SUCCEEDED, Experiment.STATUS_FAILED, Experiment.STATUS_CANCELLED]


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    metrics: dict
    exp1_rows: List[Dict[str, str]]
    elapsed_seconds: float
    cancelled: bool
    artifacts: dict


def _dataset_key(dataset: str) -> str:
    return "dblp" if dataset == "DBLP" else dataset


def _load_base_config() -> dict:
    with open(settings.GRACE_CONFIG_FILE, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _build_temp_config(dataset: str, experiment: Experiment) -> str:
    cfg = _load_base_config()
    dataset_key = _dataset_key(dataset)
    section = cfg[dataset_key]
    section["learning_rate"] = float(experiment.learning_rate)
    section["num_hidden"] = int(experiment.hidden_dim)
    section["num_proj_hidden"] = int(experiment.hidden_dim)
    section["tau"] = float(experiment.temperature)
    section["num_epochs"] = int(experiment.epochs)
    section["drop_edge_rate_1"] = float(experiment.drop_edge_rate)
    section["drop_edge_rate_2"] = float(experiment.drop_edge_rate)
    section["drop_feature_rate_1"] = float(experiment.drop_feature_rate)
    section["drop_feature_rate_2"] = float(experiment.drop_feature_rate)

    for key, value in (experiment.extra_params or {}).items():
        section[key] = value

    temp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    try:
        yaml.safe_dump(cfg, temp, sort_keys=False)
        return temp.name
    finally:
        temp.close()


def _parse_final_metrics(output: str) -> dict:
    match = F1_PATTERN.search(output)
    if not match:
        return {}
    return {
        "f1mi_mean": float(match.group("f1mi_mean")),
        "f1mi_std": float(match.group("f1mi_std")),
        "f1ma_mean": float(match.group("f1ma_mean")),
        "f1ma_std": float(match.group("f1ma_std")),
    }


def _parse_epoch_metrics(line: str) -> dict:
    match = EPOCH_PATTERN.search(line)
    if not match:
        return {}
    epoch = int(match.group("epoch"))
    loss = float(match.group("loss"))
    violation_rate = float(match.group("violation_rate"))
    mean_margin = float(match.group("mean_margin"))
    p10_margin = float(match.group("p10_margin"))
    return {
        "epoch": epoch,
        "loss": loss,
        "accuracy": 1.0 - violation_rate,
        "payload": {
            "epoch": epoch,
            "loss": loss,
            "violation_rate": violation_rate,
            "mean_margin": mean_margin,
            "p10_margin": p10_margin,
        },
    }


def _safe_float(value):
    if value in (None, "", "null", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    if value in (None, "", "null", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _infer_dataset_name(csv_path: Path, row: Dict[str, str]) -> str:
    dataset = (row.get("dataset") or "").strip()
    if dataset:
        return dataset
    stem = csv_path.stem.lower()
    if "cora" in stem:
        return "Cora"
    if "citeseer" in stem:
        return "CiteSeer"
    if "pubmed" in stem:
        return "PubMed"
    if "dblp" in stem:
        return "DBLP"
    return ""


def _normalize_pipeline_row(row: Dict[str, str], csv_path: Path) -> dict:
    params_json = row.get("params_json", "")
    if isinstance(params_json, str) and params_json:
        try:
            params_json = json.loads(params_json)
        except json.JSONDecodeError:
            params_json = {"raw": params_json}
    elif params_json in (None, ""):
        params_json = {}

    method_key = row.get("method", row.get("model_name", ""))
    method_name = row.get("method_name", method_key)
    stage = row.get("stage", PipelineResult.STAGE_SUMMARY)

    normalized = {
        "dataset": _infer_dataset_name(csv_path, row),
        "method_key": method_key,
        "method_name": method_name,
        "stage": stage,
        "candidate_rank": _safe_int(row.get("candidate_rank")),
        "run_idx": _safe_int(row.get("run_idx")),
        "F1Mi_mean": _safe_float(row.get("F1Mi_mean") or row.get("f1mi_mean")),
        "F1Mi_std": _safe_float(row.get("F1Mi_std") or row.get("f1mi_std")),
        "F1Ma_mean": _safe_float(row.get("F1Ma_mean") or row.get("f1ma_mean")),
        "F1Ma_std": _safe_float(row.get("F1Ma_std") or row.get("f1ma_std")),
        "robust_score": _safe_float(row.get("robust_score")),
        "delta_vs_grace": _safe_float(row.get("delta_vs_grace")),
        "params_json": params_json,
        "notes": row.get("notes", ""),
        "source_csv": str(csv_path),
    }
    return normalized


def _read_exp1_csv(csv_path: str) -> list:
    rows = []
    if not csv_path or not os.path.exists(csv_path):
        return rows
    with open(csv_path, mode="r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def _upsert_streamed_log(experiment: Experiment, parsed: dict):
    epoch = parsed["epoch"]
    defaults = {
        "loss": parsed["loss"],
        "accuracy": parsed["accuracy"],
        "payload": parsed["payload"],
    }
    ExperimentLog.objects.update_or_create(experiment=experiment, epoch=epoch, defaults=defaults)


def _apply_exp1_rows(experiment: Experiment, exp1_rows: list):
    if not exp1_rows:
        return
    existing = {item.epoch: item for item in ExperimentLog.objects.filter(experiment=experiment)}
    for row in exp1_rows:
        epoch = _safe_int(row.get("epoch"))
        if epoch is None:
            continue
        loss = _safe_float(row.get("loss")) or 0.0
        accuracy = 1.0 - (_safe_float(row.get("violation_rate")) or 0.0)
        payload = dict(row)
        obj = existing.get(epoch)
        if obj is None:
            ExperimentLog.objects.create(
                experiment=experiment,
                epoch=epoch,
                loss=loss,
                accuracy=accuracy,
                payload=payload,
            )
            continue
        obj.loss = loss
        obj.accuracy = accuracy
        obj.payload = payload
        obj.save(update_fields=["loss", "accuracy", "payload"])


def _ensure_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _resolve_output_path(path_value: str) -> str:
    candidate = (path_value or "").strip()
    if not candidate:
        return ""
    path = Path(candidate)
    if path.is_absolute():
        return str(path)
    return str((Path(settings.GRACE_PROJECT_ROOT) / path).resolve())


def _build_command_for_experiment(experiment: Experiment, config_path: str, exp1_path: str):
    params = experiment.task_params or {}
    extra_args = _ensure_list(experiment.extra_cli_args)
    repo_root = Path(settings.GRACE_PROJECT_ROOT)
    task_type = experiment.task_type
    artifacts = {"stdout_log": experiment.stdout_path}

    if task_type == Experiment.TASK_TRAIN:
        cmd = [
            sys.executable,
            str(settings.GRACE_TRAIN_SCRIPT),
            "--dataset",
            experiment.dataset,
            "--method",
            experiment.model_name,
            "--config",
            config_path,
            "--gpu_id",
            "0",
            "--exp1_log_csv",
            exp1_path,
        ]
        cmd.extend(extra_args)
        artifacts["exp1_log_csv"] = exp1_path
        return cmd, artifacts

    if task_type == Experiment.TASK_GRID_SEARCH:
        method = params.get("method", "ifl-gr")
        dataset = params.get("dataset", "Cora")
        topk = int(params.get("topk", 10))
        std_weight = float(params.get("std_weight", 0.5))
        script_name = GRID_SCRIPT_BY_METHOD[method]
        out_value = (params.get("out") or "").strip()
        cmd = [
            sys.executable,
            str(repo_root / "tools" / script_name),
            "--dataset",
            dataset,
            "--gpu_id",
            "0",
            "--topk",
            str(topk),
            "--std_weight",
            str(std_weight),
        ]
        if out_value:
            cmd.extend(["--out", out_value])
            artifacts["result_csv"] = _resolve_output_path(out_value)
        else:
            prefix = GRID_CSV_PREFIX_BY_METHOD[method]
            artifacts["result_csv"] = str(repo_root / "results" / f"grid_search_{prefix}_{dataset.lower()}_results.csv")
        cmd.extend(extra_args)
        return cmd, artifacts

    if task_type == Experiment.TASK_TOP_VERIFY:
        dataset = params.get("dataset", "Cora")
        method = params.get("method", "ifl-gr")
        top_params = params.get("top_params", "")
        topk = int(params.get("topk", 3))
        runs = int(params.get("runs", 3))
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "verify_top_params.py"),
            "--dataset",
            dataset,
            "--method",
            method,
            "--top_params",
            top_params,
            "--topk",
            str(topk),
            "--runs",
            str(runs),
            "--gpu_id",
            "0",
        ]
        cmd.extend(extra_args)
        return cmd, artifacts

    if task_type == Experiment.TASK_FULL_PIPELINE_SINGLE:
        dataset = params.get("dataset", "Cora")
        baseline_runs = int(params.get("baseline_runs", 3))
        topk_verify = int(params.get("topk_verify", 3))
        runs_per_top = int(params.get("runs_per_top", 3))
        force_grid = bool(params.get("force_grid", False))
        out_value = (params.get("out") or "").strip()
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "run_cora_full_pipeline.py"),
            "--dataset",
            dataset,
            "--gpu_id",
            "0",
            "--baseline_runs",
            str(baseline_runs),
            "--topk_verify",
            str(topk_verify),
            "--runs_per_top",
            str(runs_per_top),
        ]
        if force_grid:
            cmd.append("--force_grid")
        if out_value:
            cmd.extend(["--out", out_value])
            artifacts["result_csv"] = _resolve_output_path(out_value)
        else:
            artifacts["result_csv"] = str(repo_root / "results" / f"{dataset.lower()}_full_pipeline_results.csv")
        cmd.extend(extra_args)
        return cmd, artifacts

    if task_type == Experiment.TASK_FULL_PIPELINE_MULTI:
        datasets = params.get("datasets") or ["Cora", "CiteSeer", "PubMed", "DBLP"]
        continue_on_error = bool(params.get("continue_on_error", False))
        baseline_runs = int(params.get("baseline_runs", 3))
        topk_verify = int(params.get("topk_verify", 3))
        runs_per_top = int(params.get("runs_per_top", 3))
        force_grid = bool(params.get("force_grid", False))
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "run_selected_full_pipelines.py"),
            "--datasets",
            *datasets,
            "--gpu_id",
            "0",
        ]
        if continue_on_error:
            cmd.append("--continue_on_error")
        cmd.extend(["--baseline_runs", str(baseline_runs), "--topk_verify", str(topk_verify), "--runs_per_top", str(runs_per_top)])
        if force_grid:
            cmd.append("--force_grid")
        cmd.extend(extra_args)
        artifacts["result_csvs"] = [str(repo_root / "results" / f"{dataset.lower()}_full_pipeline_results.csv") for dataset in datasets]
        return cmd, artifacts

    raise ValueError(f"Unsupported task type: {task_type}")


def _is_cancel_requested(experiment_id: int) -> bool:
    return Experiment.objects.filter(pk=experiment_id, cancel_requested=True).exists()


def _stream_subprocess_output(proc: subprocess.Popen, experiment: Experiment, stdout_path: Path, on_line):
    output_lines: List[str] = []
    cancelled = False
    q = queue.Queue()
    done_sentinel = object()

    def _reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line)
        q.put(done_sentinel)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    reader_done = False
    with open(stdout_path, "w", encoding="utf-8") as log_fp:
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                item = None

            if item is done_sentinel:
                reader_done = True
            elif item is not None:
                raw_line = str(item)
                line = raw_line.rstrip("\n")
                output_lines.append(line)
                log_fp.write(raw_line)
                log_fp.flush()
                on_line(line)

            if not cancelled and _is_cancel_requested(experiment.pk) and proc.poll() is None:
                cancelled = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

            if reader_done and proc.poll() is not None:
                break

    thread.join(timeout=1.0)
    return output_lines, cancelled


def _reset_terminal_records(experiment: Experiment):
    ExperimentLog.objects.filter(experiment=experiment).delete()
    ExperimentMetric.objects.filter(experiment=experiment).delete()


def _sync_pipeline_outputs(artifacts: dict):
    csv_paths = []
    if artifacts.get("result_csv"):
        csv_paths.append(artifacts["result_csv"])
    csv_paths.extend(artifacts.get("result_csvs", []))
    for csv_path in csv_paths:
        path = Path(csv_path)
        if path.exists():
            sync_result_csv(path)


def run_experiment(experiment: Experiment) -> RunResult:
    started = datetime.now().timestamp()
    experiment.refresh_from_db()
    if experiment.status != Experiment.STATUS_PENDING:
        return RunResult(
            returncode=1,
            stdout="",
            stderr="",
            metrics={},
            exp1_rows=[],
            elapsed_seconds=0.0,
            cancelled=False,
            artifacts=experiment.artifacts or {},
        )

    experiment.status = Experiment.STATUS_RUNNING
    experiment.cancel_requested = False
    experiment.started_time = timezone.now()
    experiment.finished_time = None
    experiment.run_seconds = None
    experiment.error_message = ""
    experiment.save(
        update_fields=[
            "status",
            "cancel_requested",
            "started_time",
            "finished_time",
            "run_seconds",
            "error_message",
        ]
    )

    _reset_terminal_records(experiment)

    logs_dir = Path(settings.GRACE_LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"experiment_{experiment.pk}.log"
    exp1_path = logs_dir / f"experiment_{experiment.pk}_exp1.csv"

    config_path = ""
    if experiment.task_type == Experiment.TASK_TRAIN:
        config_path = _build_temp_config(experiment.dataset, experiment)

    cmd, artifacts = _build_command_for_experiment(experiment, config_path=config_path, exp1_path=str(exp1_path))
    experiment.stdout_path = str(stdout_path)
    experiment.exp1_log_path = str(exp1_path) if experiment.task_type == Experiment.TASK_TRAIN else ""
    experiment.artifacts = artifacts
    experiment.save(update_fields=["stdout_path", "exp1_log_path", "artifacts"])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=settings.GRACE_PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
    )

    def _on_line(line: str):
        if experiment.task_type != Experiment.TASK_TRAIN:
            return
        parsed = _parse_epoch_metrics(line)
        if parsed:
            _upsert_streamed_log(experiment, parsed)

    stdout_lines, cancelled = _stream_subprocess_output(proc, experiment, stdout_path, _on_line)
    proc.wait()

    combined_output = "\n".join(stdout_lines)
    metrics = _parse_final_metrics(combined_output)
    exp1_rows = []
    if experiment.task_type == Experiment.TASK_TRAIN:
        exp1_rows = _read_exp1_csv(str(exp1_path))
        _apply_exp1_rows(experiment, exp1_rows)

    elapsed_seconds = datetime.now().timestamp() - started
    if cancelled:
        experiment.status = Experiment.STATUS_CANCELLED
        experiment.error_message = "任务已由用户取消。"
    elif proc.returncode == 0:
        experiment.status = Experiment.STATUS_SUCCEEDED
        experiment.error_message = ""
    else:
        experiment.status = Experiment.STATUS_FAILED
        experiment.error_message = combined_output

    if experiment.status == Experiment.STATUS_SUCCEEDED:
        experiment.final_f1mi = metrics.get("f1mi_mean")
        experiment.final_f1ma = metrics.get("f1ma_mean")
        experiment.final_accuracy = metrics.get("f1mi_mean")
    else:
        experiment.final_f1mi = None
        experiment.final_f1ma = None
        experiment.final_accuracy = None

    experiment.run_seconds = elapsed_seconds
    experiment.finished_time = timezone.now()
    experiment.save(
        update_fields=[
            "status",
            "error_message",
            "final_f1mi",
            "final_f1ma",
            "final_accuracy",
            "run_seconds",
            "finished_time",
        ]
    )

    if experiment.status == Experiment.STATUS_SUCCEEDED and metrics:
        ExperimentMetric.objects.filter(experiment=experiment).delete()
        ExperimentMetric.objects.bulk_create(
            [
                ExperimentMetric(experiment=experiment, name="f1mi_mean", value=metrics["f1mi_mean"]),
                ExperimentMetric(experiment=experiment, name="f1mi_std", value=metrics["f1mi_std"]),
                ExperimentMetric(experiment=experiment, name="f1ma_mean", value=metrics["f1ma_mean"]),
                ExperimentMetric(experiment=experiment, name="f1ma_std", value=metrics["f1ma_std"]),
            ]
        )

    if experiment.status == Experiment.STATUS_SUCCEEDED and experiment.task_type in [
        Experiment.TASK_FULL_PIPELINE_SINGLE,
        Experiment.TASK_FULL_PIPELINE_MULTI,
    ]:
        _sync_pipeline_outputs(artifacts)

    if config_path:
        try:
            Path(config_path).unlink(missing_ok=True)
        except TypeError:
            if Path(config_path).exists():
                Path(config_path).unlink()

    return RunResult(
        returncode=proc.returncode,
        stdout=combined_output,
        stderr="",
        metrics=metrics,
        exp1_rows=exp1_rows,
        elapsed_seconds=elapsed_seconds,
        cancelled=cancelled,
        artifacts=artifacts,
    )


def enqueue_experiment(experiment_id: int):
    experiment = Experiment.objects.get(pk=experiment_id)
    if experiment.status == Experiment.STATUS_RUNNING:
        return experiment

    experiment.status = Experiment.STATUS_PENDING
    experiment.cancel_requested = False
    experiment.started_time = None
    experiment.finished_time = None
    experiment.run_seconds = None
    experiment.error_message = ""
    experiment.save(
        update_fields=[
            "status",
            "cancel_requested",
            "started_time",
            "finished_time",
            "run_seconds",
            "error_message",
        ]
    )
    return experiment


def request_stop(experiment: Experiment):
    if experiment.status == Experiment.STATUS_PENDING:
        experiment.status = Experiment.STATUS_CANCELLED
        experiment.cancel_requested = False
        experiment.finished_time = timezone.now()
        experiment.error_message = "任务在开始执行前已取消。"
        experiment.save(update_fields=["status", "cancel_requested", "finished_time", "error_message"])
        return "cancelled"

    if experiment.status == Experiment.STATUS_RUNNING and not experiment.cancel_requested:
        experiment.cancel_requested = True
        experiment.save(update_fields=["cancel_requested"])
        return "requested"

    return "noop"


def read_terminal_tail(stdout_path: str, max_lines: int = 200) -> dict:
    if not stdout_path:
        return {"terminal_tail": "", "terminal_total_lines": 0, "terminal_updated_at": None}
    path = Path(stdout_path)
    if not path.exists():
        return {"terminal_tail": "", "terminal_total_lines": 0, "terminal_updated_at": None}

    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        lines = fp.readlines()
    tail_lines = lines[-max_lines:]
    updated_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    return {
        "terminal_tail": "".join(tail_lines),
        "terminal_total_lines": len(lines),
        "terminal_updated_at": updated_at,
    }


def import_result_csv(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not csv_path.exists():
        return rows
    with open(csv_path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def sync_result_csv(csv_path: Path) -> List[PipelineResult]:
    rows = import_result_csv(csv_path)
    synced = []
    for row in rows:
        normalized = _normalize_pipeline_row(row, csv_path)
        if not normalized["dataset"] or not normalized["method_key"]:
            continue
        obj, _ = PipelineResult.objects.update_or_create(
            source_csv=normalized["source_csv"],
            dataset=normalized["dataset"],
            method_key=normalized["method_key"],
            stage=normalized["stage"],
            candidate_rank=normalized["candidate_rank"],
            run_idx=normalized["run_idx"],
            defaults=normalized,
        )
        synced.append(obj)
    return synced
