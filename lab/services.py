import os
import signal
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from .constants import METHOD_FILE_SLUG
from .models import ExperimentArtifact, ExperimentRun
from .parsers import (
    build_method_comparison_summary,
    build_sampling_bias_summary,
    build_sensitivity_summary,
)


BASE_DIR = Path(settings.BASE_DIR)
ACTIVE_RUN_STATUSES = {ExperimentRun.STATUS_PENDING, ExperimentRun.STATUS_RUNNING}


def _is_within_base(path):
    try:
        path.resolve().relative_to(BASE_DIR.resolve())
        return True
    except ValueError:
        return False


def _safe_unlink(path):
    resolved = path.resolve()
    if not _is_within_base(resolved):
        return
    if resolved.is_file():
        resolved.unlink(missing_ok=True)


def _safe_rmtree(path):
    resolved = path.resolve()
    if not _is_within_base(resolved):
        return
    if resolved.exists() and resolved.is_dir():
        shutil.rmtree(resolved)


def _cleanup_empty_parents(path, stop_at):
    current = path.resolve()
    stop = stop_at.resolve()
    while _is_within_base(current) and current != stop:
        if current.exists() and current.is_dir():
            try:
                current.rmdir()
            except OSError:
                break
        current = current.parent


def _collect_run_paths(run):
    artifact_paths = []
    for artifact in run.artifacts.all():
        absolute_path = (BASE_DIR / artifact.relative_path).resolve()
        if _is_within_base(absolute_path):
            artifact_paths.append(absolute_path)

    run_dirs = [
        (BASE_DIR / "results" / "webapp" / f"run_{run.pk}").resolve(),
        (BASE_DIR / "logs" / "webapp" / f"run_{run.pk}").resolve(),
    ]

    return artifact_paths, run_dirs


def cleanup_experiment_files(run):
    artifact_paths, run_dirs = _collect_run_paths(run)

    for path in artifact_paths:
        if any(run_dir == path or run_dir in path.parents for run_dir in run_dirs):
            continue
        _safe_unlink(path)
        _cleanup_empty_parents(path.parent, BASE_DIR)

    for run_dir in run_dirs:
        _safe_rmtree(run_dir)
        _cleanup_empty_parents(run_dir.parent, BASE_DIR)

    run.artifacts.all().delete()


def delete_experiment_run(run):
    cleanup_experiment_files(run)

    run.delete()


def _terminate_process_tree(pid):
    if not pid:
        raise RuntimeError("This run does not have a recorded worker PID, so it cannot be stopped safely.")

    if sys.platform.startswith("win"):
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        output = (completed.stdout or "").strip()
        if completed.returncode not in {0, 128}:
            raise RuntimeError(output or f"taskkill returned exit code {completed.returncode}.")
        return

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return


def stop_experiment_run(run):
    if run.status not in ACTIVE_RUN_STATUSES:
        raise RuntimeError(f"Run #{run.pk} is no longer active.")

    _terminate_process_tree(run.worker_pid)
    cleanup_experiment_files(run)

    aborted_at = timezone.now()
    log_lines = [run.stdout_log.strip()] if run.stdout_log.strip() else []
    log_lines.append(f"[system] Run aborted by user at {aborted_at:%Y-%m-%d %H:%M:%S}.")
    log_lines.append("[system] Related output files were removed.")

    run.status = ExperimentRun.STATUS_ABORTED
    run.finished_at = aborted_at
    run.worker_pid = None
    run.error_message = ""
    run.result_summary = {}
    run.stdout_log = "\n".join(log_lines)
    run.save(update_fields=["status", "finished_at", "worker_pid", "error_message", "result_summary", "stdout_log"])


def launch_experiment(run):
    manage_py = BASE_DIR / "manage.py"
    command = [sys.executable, str(manage_py), "process_experiment", str(run.pk)]
    kwargs = {
        "cwd": str(BASE_DIR),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    worker = subprocess.Popen(command, **kwargs)
    ExperimentRun.objects.filter(pk=run.pk).update(worker_pid=worker.pid)
    run.worker_pid = worker.pid


def execute_experiment(run_id):
    run = ExperimentRun.objects.get(pk=run_id)
    run.artifacts.all().delete()
    run.status = ExperimentRun.STATUS_RUNNING
    run.started_at = timezone.now()
    run.finished_at = None
    run.error_message = ""
    run.stdout_log = ""
    run.command = ""
    run.result_summary = {}
    run.worker_pid = os.getpid()
    run.save(
        update_fields=[
            "status",
            "started_at",
            "finished_at",
            "error_message",
            "stdout_log",
            "command",
            "result_summary",
            "worker_pid",
        ]
    )

    try:
        if run.experiment_type == ExperimentRun.TYPE_METHOD_COMPARISON:
            summary = _run_method_comparison(run)
        elif run.experiment_type == ExperimentRun.TYPE_SAMPLING_BIAS:
            summary = _run_sampling_bias(run)
        elif run.experiment_type == ExperimentRun.TYPE_SENSITIVITY:
            summary = _run_sensitivity(run)
        else:
            raise RuntimeError(f"Unsupported experiment type: {run.experiment_type}")

        run.result_summary = summary
        run.status = ExperimentRun.STATUS_SUCCEEDED
        run.finished_at = timezone.now()
        run.worker_pid = None
        run.save(update_fields=["result_summary", "status", "finished_at", "worker_pid"])
    except Exception as exc:
        run.refresh_from_db(fields=["status"])
        if run.status == ExperimentRun.STATUS_ABORTED:
            ExperimentRun.objects.filter(pk=run_id).update(worker_pid=None)
            return

        run.status = ExperimentRun.STATUS_FAILED
        run.finished_at = timezone.now()
        run.worker_pid = None
        run.error_message = str(exc)
        run.stdout_log = f"{run.stdout_log}\n{traceback.format_exc()}".strip()
        run.save(update_fields=["status", "finished_at", "worker_pid", "error_message", "stdout_log"])
        raise


def _append_log(run, text):
    run.stdout_log = f"{run.stdout_log}{text}"
    run.save(update_fields=["stdout_log"])


def _append_command(run, command):
    rendered = subprocess.list2cmdline(command)
    run.command = f"{run.command}\n{rendered}".strip()
    run.save(update_fields=["command"])
    return rendered


def _run_command(run, command, label):
    rendered = _append_command(run, command)
    _append_log(run, f"\n[{label}] $ {rendered}\n")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    output_lines = []
    for line in proc.stdout:
        output_lines.append(line)
    return_code = proc.wait()
    output = "".join(output_lines)
    _append_log(run, output)
    if return_code != 0:
        raise RuntimeError(f"{label} failed with exit code {return_code}.")
    return output


def _register_artifact(run, label, artifact_type, relative_path, metadata=None):
    relative = Path(relative_path).as_posix()
    absolute_path = (BASE_DIR / relative).resolve()
    if not absolute_path.exists():
        return None
    return ExperimentArtifact.objects.create(
        run=run,
        label=label,
        artifact_type=artifact_type,
        relative_path=relative,
        metadata=metadata or {},
    )


def _run_method_comparison(run):
    config = run.config
    dataset = config["dataset"]
    dataset_slug = dataset.lower()
    run_dir = Path("results") / "webapp" / f"run_{run.pk}"
    grid_dir = run_dir / "grid"
    out_csv = run_dir / f"{dataset_slug}_full_pipeline_results.csv"

    command = [
        sys.executable,
        "experiments/method_comparison/run_full_pipeline.py",
        "--dataset",
        dataset,
        "--gpu_id",
        str(config["gpu_id"]),
        "--std_weight",
        str(config["std_weight"]),
        "--baseline_runs",
        str(config["baseline_runs"]),
        "--topk_verify",
        str(config["topk_verify"]),
        "--runs_per_top",
        str(config["runs_per_top"]),
        "--grid_dir",
        grid_dir.as_posix(),
        "--out",
        out_csv.as_posix(),
    ]
    if config.get("force_grid"):
        command.append("--force_grid")
    if config.get("verbose_train_output"):
        command.append("--verbose_train_output")

    _run_command(run, command, "method-comparison")

    _register_artifact(run, "Unified Results CSV", ExperimentArtifact.TYPE_CSV, out_csv)
    for method_slug in ("iflgr", "gca", "iflgc"):
        grid_file = grid_dir / f"grid_search_{method_slug}_{dataset_slug}_results.csv"
        _register_artifact(run, f"{method_slug} grid search", ExperimentArtifact.TYPE_CSV, grid_file)

    summary = build_method_comparison_summary(BASE_DIR / out_csv)
    summary.update(
        {
            "dataset": dataset,
            "main_csv": out_csv.as_posix(),
            "grid_dir": grid_dir.as_posix(),
        }
    )
    return summary


def _run_sampling_bias(run):
    config = run.config
    dataset = config["dataset"]
    method = config["method"]
    dataset_slug = dataset.lower()
    method_slug = METHOD_FILE_SLUG[method]
    run_dir = Path("logs") / "webapp" / f"run_{run.pk}"
    csv_path = run_dir / f"exp1_{dataset_slug}_{method_slug}.csv"
    plot_path = run_dir / f"exp1_{dataset_slug}_{method_slug}_curves.png"

    train_command = [
        sys.executable,
        "train.py",
        "--dataset",
        dataset,
        "--method",
        method,
        "--gpu_id",
        str(config["gpu_id"]),
        "--exp1_metrics",
        "--exp1_log_csv",
        csv_path.as_posix(),
    ]
    plot_command = [
        sys.executable,
        "experiments/sampling_bias_validation/plot_sampling_bias_curves.py",
        "--csv",
        csv_path.as_posix(),
        "--out",
        plot_path.as_posix(),
        "--title",
        config.get("title") or f"{dataset} / {method} sampling bias curves",
    ]

    _run_command(run, train_command, "sampling-train")
    _run_command(run, plot_command, "sampling-plot")

    _register_artifact(run, "Sampling Bias CSV", ExperimentArtifact.TYPE_CSV, csv_path)
    _register_artifact(run, "Sampling Bias Plot", ExperimentArtifact.TYPE_IMAGE, plot_path)

    summary = build_sampling_bias_summary(BASE_DIR / csv_path)
    summary.update(
        {
            "dataset": dataset,
            "method": method,
            "csv_path": csv_path.as_posix(),
            "plot_path": plot_path.as_posix(),
        }
    )
    return summary


def _run_sensitivity(run):
    config = run.config
    dataset = config["dataset"]
    dataset_slug = dataset.lower()
    methods = list(config["methods"])
    params = list(config["params"])

    run_dir = Path("results") / "webapp" / f"run_{run.pk}"
    plots_dir = run_dir / "plots"
    csv_paths = []

    for method in methods:
        method_slug = METHOD_FILE_SLUG[method]
        out_csv = run_dir / f"sensitivity_{method_slug}_{dataset_slug}_results.csv"
        command = [
            sys.executable,
            "experiments/hyperparameter_sensitivity/run_ifl_param_sensitivity.py",
            "--datasets",
            dataset,
            "--methods",
            method,
            "--params",
            *params,
            "--gpu_id",
            str(config["gpu_id"]),
            "--base_rank",
            str(config["base_rank"]),
            "--runs",
            str(config["runs"]),
            "--std_weight",
            str(config["std_weight"]),
            "--neighbor_span",
            str(config["neighbor_span"]),
            "--out",
            out_csv.as_posix(),
        ]
        if config.get("continue_on_error"):
            command.append("--continue_on_error")

        _run_command(run, command, f"sensitivity-{method}")
        _register_artifact(run, f"{method_slug} sensitivity csv", ExperimentArtifact.TYPE_CSV, out_csv)
        csv_paths.append(out_csv)

    plot_png = plots_dir / f"{dataset_slug}_ifl_sensitivity_overview.png"
    report_md = plots_dir / f"{dataset_slug}_ifl_sensitivity_analysis.md"
    plot_command = [
        sys.executable,
        "experiments/hyperparameter_sensitivity/plot_ifl_param_sensitivity.py",
        "--dataset",
        dataset,
        "--methods",
        *methods,
        "--inputs",
        *[path.as_posix() for path in csv_paths],
        "--out_dir",
        plots_dir.as_posix(),
    ]
    _run_command(run, plot_command, "sensitivity-plot")

    _register_artifact(run, "Sensitivity Overview", ExperimentArtifact.TYPE_IMAGE, plot_png)
    _register_artifact(run, "Sensitivity Report", ExperimentArtifact.TYPE_REPORT, report_md)

    summary = build_sensitivity_summary(
        [BASE_DIR / path for path in csv_paths],
        BASE_DIR / report_md,
    )
    summary.update(
        {
            "dataset": dataset,
            "methods": methods,
            "params": params,
            "plot_path": plot_png.as_posix(),
            "report_path": report_md.as_posix(),
        }
    )
    return summary
