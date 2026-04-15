import csv
import json
import os
import re
import subprocess
import tempfile
import threading
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml
from django.conf import settings
from django.db import close_old_connections
from django.db import transaction
from django.utils import timezone

from .models import Experiment, ExperimentLog, ExperimentMetric


F1_PATTERN = re.compile(
    r"\(E\) \| label_classification: "
    r"F1Mi=(?P<f1mi_mean>\d+\.\d+)\+-(?P<f1mi_std>\d+\.\d+), "
    r"F1Ma=(?P<f1ma_mean>\d+\.\d+)\+-(?P<f1ma_std>\d+\.\d+)"
)


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    metrics: dict
    exp1_rows: List[Dict[str, str]]
    elapsed_seconds: float


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

    for key, value in experiment.extra_params.items():
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


def _read_exp1_csv(csv_path: str) -> list:
    rows = []
    if not csv_path or not os.path.exists(csv_path):
        return rows
    with open(csv_path, mode="r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def run_experiment(experiment: Experiment) -> RunResult:
    started = datetime.now().timestamp()
    experiment.status = Experiment.STATUS_RUNNING
    experiment.started_time = timezone.now()
    experiment.save(update_fields=["status", "started_time"])

    logs_dir = settings.GRACE_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"experiment_{experiment.pk}.log"
    exp1_path = logs_dir / f"experiment_{experiment.pk}_exp1.csv"
    experiment.stdout_path = str(stdout_path)
    experiment.exp1_log_path = str(exp1_path)
    experiment.save(update_fields=["stdout_path", "exp1_log_path"])

    config_path = _build_temp_config(experiment.dataset, experiment)
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
        str(exp1_path),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.run(cmd, cwd=settings.GRACE_PROJECT_ROOT, text=True, capture_output=True, env=env)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    if proc.stderr:
        stdout_path.with_suffix(".err").write_text(proc.stderr, encoding="utf-8")

    metrics = _parse_final_metrics(proc.stdout + "\n" + proc.stderr)
    exp1_rows = _read_exp1_csv(str(exp1_path))

    elapsed_seconds = datetime.now().timestamp() - started
    if proc.returncode == 0:
        experiment.status = Experiment.STATUS_SUCCEEDED
        experiment.final_f1mi = metrics.get("f1mi_mean")
        experiment.final_f1ma = metrics.get("f1ma_mean")
        experiment.final_accuracy = metrics.get("f1mi_mean")
        experiment.run_seconds = elapsed_seconds
        experiment.finished_time = timezone.now()
        experiment.error_message = ""
        experiment.save(
            update_fields=[
                "status",
                "final_f1mi",
                "final_f1ma",
                "final_accuracy",
                "run_seconds",
                "finished_time",
                "error_message",
            ]
        )

        if exp1_rows:
            ExperimentLog.objects.filter(experiment=experiment).delete()
            logs = []
            for row in exp1_rows:
                epoch = int(row.get("epoch", 0))
                loss = float(row.get("loss", 0.0))
                accuracy = 1.0 - float(row.get("violation_rate", 0.0))
                logs.append(
                    ExperimentLog(
                        experiment=experiment,
                        epoch=epoch,
                        loss=loss,
                        accuracy=accuracy,
                        payload=row,
                    )
                )
            ExperimentLog.objects.bulk_create(logs)

        if metrics:
            ExperimentMetric.objects.filter(experiment=experiment).delete()
            ExperimentMetric.objects.bulk_create([
                ExperimentMetric(experiment=experiment, name="f1mi_mean", value=metrics["f1mi_mean"]),
                ExperimentMetric(experiment=experiment, name="f1mi_std", value=metrics["f1mi_std"]),
                ExperimentMetric(experiment=experiment, name="f1ma_mean", value=metrics["f1ma_mean"]),
                ExperimentMetric(experiment=experiment, name="f1ma_std", value=metrics["f1ma_std"]),
            ])
    else:
        experiment.status = Experiment.STATUS_FAILED
        experiment.error_message = proc.stderr or proc.stdout
        experiment.finished_time = timezone.now()
        experiment.run_seconds = elapsed_seconds
        experiment.save(update_fields=["status", "error_message", "finished_time", "run_seconds"])

    try:
        Path(config_path).unlink(missing_ok=True)
    except TypeError:
        if Path(config_path).exists():
            Path(config_path).unlink()

    return RunResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        metrics=metrics,
        exp1_rows=exp1_rows,
        elapsed_seconds=elapsed_seconds,
    )


def enqueue_experiment(experiment_id: int):
    def _runner():
        close_old_connections()
        experiment = Experiment.objects.get(pk=experiment_id)
        run_experiment(experiment)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return thread


def import_result_csv(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not csv_path.exists():
        return rows
    with open(csv_path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows
