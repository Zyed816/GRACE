import csv
import json
import mimetypes
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.db.models import Avg, Count
from django.http import FileResponse, Http404, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from apps.datasets.catalog import DATASET_CATALOG
from apps.models.catalog import METHOD_CATALOG

from .forms import ExperimentTaskForm
from .models import Experiment, ExperimentLog, ExperimentMetric, PipelineResult
from .services import enqueue_experiment, read_terminal_tail, request_stop, sync_result_csv


PROJECT_OVERVIEW = [
    {
        "title": "训练核心复用",
        "description": "直接复用 train.py、model.py 与 eval.py，不改动现有算法内部实现。",
    },
    {
        "title": "任务统一编排",
        "description": "把训练、参数搜索、最优验证和全流程脚本统一纳入任务队列调度。",
    },
    {
        "title": "结果统一沉淀",
        "description": "统一保存终端日志、训练指标和导入的 CSV 结果，便于持续追踪。",
    },
    {
        "title": "可视化监控",
        "description": "在同一页面中实时展示终端输出、训练曲线和实验结果产物。",
    },
]

PAPER_FLOW = [
    "加载数据集与配置",
    "构建 GRACE / GCA / IFL 的对比视图",
    "使用对比损失训练编码器",
    "在启用时挖掘无标签正样本",
    "通过线性分类评估嵌入表示",
]

CSV_PREVIEW_LIMIT = 8
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg"}


def _artifact_roots():
    return {
        "logs": Path(settings.GRACE_LOGS_DIR).resolve(),
        "results": Path(settings.GRACE_RESULTS_DIR).resolve(),
    }


def _safe_artifact_path(bucket: str, filename: str) -> Path:
    roots = _artifact_roots()
    if bucket not in roots:
        raise Http404("Unsupported artifact bucket.")
    base = roots[bucket]
    candidate = (base / filename).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise Http404("Invalid artifact path.") from exc
    if not candidate.exists() or not candidate.is_file():
        raise Http404("Artifact not found.")
    return candidate


def _bucket_for_path(file_path: Path):
    resolved = file_path.resolve()
    for bucket, root in _artifact_roots().items():
        try:
            resolved.relative_to(root)
            return bucket
        except ValueError:
            continue
    return None


def _format_size_kb(file_path: Path) -> str:
    return f"{file_path.stat().st_size / 1024:.1f} KB"


def _read_csv_preview(file_path: Path, limit: int = CSV_PREVIEW_LIMIT):
    headers = []
    rows = []
    total_rows = 0
    with open(file_path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        headers = reader.fieldnames or []
        for row in reader:
            total_rows += 1
            if len(rows) < limit:
                rows.append(row)
    return headers, rows, total_rows


def _build_artifact_url(bucket: str, file_path: Path) -> str:
    return reverse("artifact-file", kwargs={"bucket": bucket, "filename": file_path.name})


def _build_asset_entry(bucket: str, file_path: Path, preview_csv: bool = True) -> dict:
    suffix = file_path.suffix.lower()
    entry = {
        "name": file_path.name,
        "bucket": bucket,
        "url": _build_artifact_url(bucket, file_path),
        "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime),
        "size_label": _format_size_kb(file_path),
        "is_csv": suffix == ".csv",
        "is_image": suffix in IMAGE_SUFFIXES,
        "preview_headers": [],
        "preview_rows": [],
        "preview_total_rows": 0,
    }
    if entry["is_csv"] and preview_csv:
        headers, rows, total_rows = _read_csv_preview(file_path)
        entry["preview_headers"] = headers
        entry["preview_rows"] = rows
        entry["preview_total_rows"] = total_rows
    return entry


def _matching_curve_image(csv_path: Path):
    candidate = csv_path.with_name(f"{csv_path.stem}_curves.png")
    return candidate if candidate.exists() else None


def _collect_sampling_bias_assets():
    logs_dir = Path(settings.GRACE_LOGS_DIR)
    assets = []
    for csv_path in sorted(logs_dir.glob("*exp1*.csv"), key=lambda item: item.stat().st_mtime, reverse=True):
        entry = _build_asset_entry("logs", csv_path)
        curve_path = _matching_curve_image(csv_path)
        if curve_path:
            entry["curve_asset"] = _build_asset_entry("logs", curve_path, preview_csv=False)
        assets.append(entry)
    return assets


def _collect_result_csv_assets(pattern: str):
    results_dir = Path(settings.GRACE_RESULTS_DIR)
    assets = []
    for csv_path in sorted(results_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True):
        assets.append(_build_asset_entry("results", csv_path))
    return assets


def _collect_experiment_asset_entries(experiment: Experiment):
    entries = []
    seen = set()
    candidate_paths = []

    if experiment.exp1_log_path:
        candidate_paths.append(Path(experiment.exp1_log_path))

    artifacts = experiment.artifacts or {}
    if artifacts.get("result_csv"):
        candidate_paths.append(Path(artifacts["result_csv"]))
    for item in artifacts.get("result_csvs", []):
        candidate_paths.append(Path(item))
    if artifacts.get("exp1_log_csv"):
        candidate_paths.append(Path(artifacts["exp1_log_csv"]))

    for file_path in candidate_paths:
        if not file_path or not file_path.exists():
            continue
        resolved = file_path.resolve()
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        bucket = _bucket_for_path(resolved)
        if not bucket:
            continue
        entry = _build_asset_entry(bucket, resolved)
        if entry["is_csv"]:
            curve_path = _matching_curve_image(resolved)
            if curve_path and _bucket_for_path(curve_path) == bucket:
                entry["curve_asset"] = _build_asset_entry(bucket, curve_path, preview_csv=False)
        entries.append(entry)
    return entries


def dashboard(request):
    recent_experiments = Experiment.objects.all()[:5]
    results_dir = Path(settings.GRACE_RESULTS_DIR)
    for csv_path in sorted(results_dir.glob("*_full_pipeline_results.csv")):
        sync_result_csv(csv_path)

    recent_results = PipelineResult.objects.filter(stage=PipelineResult.STAGE_SUMMARY).order_by("-created_time")[:6]
    method_summary = (
        Experiment.objects.filter(status=Experiment.STATUS_SUCCEEDED, task_type=Experiment.TASK_TRAIN)
        .values("model_name")
        .annotate(total=Count("id"), avg_f1mi=Avg("final_f1mi"), avg_f1ma=Avg("final_f1ma"))
        .order_by("model_name")
    )
    context = {
        "method_count": len(METHOD_CATALOG),
        "dataset_count": len(DATASET_CATALOG),
        "experiment_count": Experiment.objects.count(),
        "recent_experiments": recent_experiments,
        "recent_results": recent_results,
        "project_overview": PROJECT_OVERVIEW,
        "paper_flow": PAPER_FLOW,
        "method_summary": method_summary,
        "latest_success": Experiment.objects.filter(status=Experiment.STATUS_SUCCEEDED).first(),
    }
    return render(request, "dashboard.html", context)


def experiment_create(request):
    if request.method == "POST":
        form = ExperimentTaskForm(request.POST)
        if form.is_valid():
            payload = form.build_experiment_payload()
            run_now = form.cleaned_data.get("run_now", True)
            if not run_now:
                payload["status"] = Experiment.STATUS_CANCELLED
                payload["error_message"] = "任务已创建但未入队，可稍后点击“开始任务”加入队列。"

            experiment = Experiment.objects.create(**payload)
            if run_now:
                enqueue_experiment(experiment.pk)
            return redirect("experiment-detail", pk=experiment.pk)
    else:
        form = ExperimentTaskForm()
    return render(request, "experiments/form.html", {"form": form, "task_choices": Experiment.TASK_TYPE_CHOICES})


def experiment_detail(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    logs = list(ExperimentLog.objects.filter(experiment=experiment))
    metrics = list(ExperimentMetric.objects.filter(experiment=experiment))
    task_params_json = json.dumps(experiment.task_params or {}, ensure_ascii=False, indent=2)
    extra_cli_args_json = json.dumps(experiment.extra_cli_args or [], ensure_ascii=False, indent=2)
    artifacts_json = json.dumps(experiment.artifacts or {}, ensure_ascii=False, indent=2)
    artifact_entries = _collect_experiment_asset_entries(experiment)
    return render(
        request,
        "experiments/detail.html",
        {
            "experiment": experiment,
            "logs": logs,
            "metrics": metrics,
            "task_params_json": task_params_json,
            "extra_cli_args_json": extra_cli_args_json,
            "artifacts_json": artifacts_json,
            "artifact_entries": artifact_entries,
            "is_train_task": experiment.task_type == Experiment.TASK_TRAIN,
        },
    )


def experiment_history(request):
    experiments = Experiment.objects.all()
    return render(request, "experiments/history.html", {"experiments": experiments})


def experiment_monitor(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    logs = list(ExperimentLog.objects.filter(experiment=experiment).order_by("epoch").values("epoch", "loss", "accuracy", "payload"))
    terminal = read_terminal_tail(experiment.stdout_path, max_lines=120)
    return JsonResponse(
        {
            "experiment_id": experiment.pk,
            "task_type": experiment.task_type,
            "status": experiment.status,
            "cancel_requested": experiment.cancel_requested,
            "logs": logs,
            "final_accuracy": experiment.final_accuracy,
            "final_f1mi": experiment.final_f1mi,
            "final_f1ma": experiment.final_f1ma,
            **terminal,
        }
    )


def results_overview(request):
    method_entries = list(METHOD_CATALOG.values())
    datasets = list(DATASET_CATALOG.keys())
    series_map = {method["display_name"]: [] for method in method_entries}
    pivot = []

    results_dir = Path(settings.GRACE_RESULTS_DIR)
    for csv_path in sorted(results_dir.glob("*_full_pipeline_results.csv")):
        sync_result_csv(csv_path)

    for dataset_name in DATASET_CATALOG:
        row = {"dataset": dataset_name}
        for method_key, method in METHOD_CATALOG.items():
            q = PipelineResult.objects.filter(
                dataset=dataset_name,
                method_key=method_key,
                stage=PipelineResult.STAGE_SUMMARY,
            )
            value = q.aggregate(v=Avg("F1Mi_mean"))["v"]
            row[method["display_name"]] = value
            series_map[method["display_name"]].append(value if value is not None else 0.0)
        pivot.append(row)

    chart_option = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": list(series_map.keys())},
        "xAxis": {"type": "category", "data": datasets},
        "yAxis": {"type": "value"},
        "series": [{"name": name, "type": "bar", "data": values} for name, values in series_map.items()],
    }

    csv_rows = list(
        PipelineResult.objects.filter(
            stage__in=[PipelineResult.STAGE_BASELINE, PipelineResult.STAGE_TOP_VERIFY, PipelineResult.STAGE_SUMMARY]
        ).order_by("dataset", "method_key", "stage", "candidate_rank", "run_idx")[:500]
    )
    sampling_bias_assets = _collect_sampling_bias_assets()
    grid_search_assets = _collect_result_csv_assets("grid_search_*_results.csv")
    pipeline_assets = _collect_result_csv_assets("*_full_pipeline_results.csv")

    return render(
        request,
        "experiments/results.html",
        {
            "pivot": pivot,
            "methods": method_entries,
            "csv_rows": csv_rows,
            "chart_option_json": json.dumps(chart_option, ensure_ascii=False),
            "sampling_bias_assets": sampling_bias_assets,
            "grid_search_assets": grid_search_assets,
            "pipeline_assets": pipeline_assets,
        },
    )


def api_monitor(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    logs = list(ExperimentLog.objects.filter(experiment=experiment).order_by("epoch").values("epoch", "loss", "accuracy", "payload")[:500])
    terminal = read_terminal_tail(experiment.stdout_path, max_lines=200)
    return JsonResponse(
        {
            "id": experiment.pk,
            "task_type": experiment.task_type,
            "status": experiment.status,
            "cancel_requested": experiment.cancel_requested,
            "logs": logs,
            "artifacts": experiment.artifacts or {},
            "task_params": experiment.task_params or {},
            "metrics": {
                "f1mi": experiment.final_f1mi,
                "f1ma": experiment.final_f1ma,
                "accuracy": experiment.final_accuracy,
                "run_seconds": experiment.run_seconds,
            },
            "final_accuracy": experiment.final_accuracy,
            "run_seconds": experiment.run_seconds,
            **terminal,
        }
    )


def experiment_start(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    enqueue_experiment(experiment.pk)
    return redirect("experiment-detail", pk=experiment.pk)


@require_POST
def experiment_stop(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    action = request_stop(experiment)
    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JsonResponse({"id": experiment.pk, "action": action, "status": experiment.status})
    return redirect("experiment-detail", pk=experiment.pk)


@require_POST
def api_experiment_stop(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    action = request_stop(experiment)
    return JsonResponse({"id": experiment.pk, "action": action, "status": experiment.status})


def api_experiment_create(request):
    if request.method != "POST":
        return HttpResponseBadRequest("仅支持 POST 请求。")
    form = ExperimentTaskForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"ok": False, "errors": form.errors}, status=400)
    payload = form.build_experiment_payload()
    experiment = Experiment.objects.create(**payload)
    enqueue_experiment(experiment.pk)
    return JsonResponse({"ok": True, "id": experiment.pk})


def artifact_file(request, bucket, filename):
    file_path = _safe_artifact_path(bucket, filename)
    content_type, _ = mimetypes.guess_type(str(file_path))
    response = FileResponse(open(file_path, "rb"), content_type=content_type or "application/octet-stream")
    response["Content-Disposition"] = f'inline; filename="{file_path.name}"'
    return response
