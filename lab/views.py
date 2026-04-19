import json
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from .forms import MethodComparisonForm, SamplingBiasForm, SensitivityForm
from .models import ExperimentArtifact, ExperimentRun
from .parsers import preview_csv, read_text_file
from .services import delete_experiment_run, launch_experiment, stop_experiment_run
from .visuals import render_bar_chart, render_line_chart


BASE_DIR = Path(settings.BASE_DIR).resolve()


def _compact_value(value, depth=0):
    if isinstance(value, dict):
        if depth >= 2:
            return f"{{{len(value)} keys omitted}}"
        return {key: _compact_value(subvalue, depth + 1) for key, subvalue in value.items()}

    if isinstance(value, list):
        if len(value) <= 6 and depth < 2:
            return [_compact_value(item, depth + 1) for item in value]
        return {
            "items": len(value),
            "preview": [_compact_value(item, depth + 1) for item in value[:3]],
            "note": "truncated for page layout",
        }

    if isinstance(value, str) and len(value) > 320:
        return f"{value[:320]}... [truncated]"

    return value


def _dashboard_context(method_form=None, sampling_form=None, sensitivity_form=None):
    recent_runs = ExperimentRun.objects.prefetch_related("artifacts").all()[:8]
    return {
        "method_form": method_form or MethodComparisonForm(prefix="method"),
        "sampling_form": sampling_form or SamplingBiasForm(prefix="sampling"),
        "sensitivity_form": sensitivity_form or SensitivityForm(prefix="sensitivity"),
        "recent_runs": recent_runs,
    }


@require_GET
def dashboard(request):
    return render(request, "lab/dashboard.html", _dashboard_context())


@require_POST
def create_method_comparison_run(request):
    form = MethodComparisonForm(request.POST, prefix="method")
    if not form.is_valid():
        return render(request, "lab/dashboard.html", _dashboard_context(method_form=form))

    cleaned = form.cleaned_data
    run = ExperimentRun.objects.create(
        name=cleaned["name"],
        experiment_type=ExperimentRun.TYPE_METHOD_COMPARISON,
        dataset=cleaned["dataset"],
        config={
            "dataset": cleaned["dataset"],
            "gpu_id": cleaned["gpu_id"],
            "std_weight": cleaned["std_weight"],
            "baseline_runs": cleaned["baseline_runs"],
            "topk_verify": cleaned["topk_verify"],
            "runs_per_top": cleaned["runs_per_top"],
            "force_grid": cleaned["force_grid"],
        },
    )
    launch_experiment(run)
    messages.success(request, f"Method comparison run #{run.pk} has been queued.")
    return redirect(run)


@require_POST
def create_sampling_bias_run(request):
    form = SamplingBiasForm(request.POST, prefix="sampling")
    if not form.is_valid():
        return render(request, "lab/dashboard.html", _dashboard_context(sampling_form=form))

    cleaned = form.cleaned_data
    run = ExperimentRun.objects.create(
        name=cleaned["name"],
        experiment_type=ExperimentRun.TYPE_SAMPLING_BIAS,
        dataset=cleaned["dataset"],
        config={
            "dataset": cleaned["dataset"],
            "method": cleaned["method"],
            "gpu_id": cleaned["gpu_id"],
            "title": cleaned["title"],
        },
    )
    launch_experiment(run)
    messages.success(request, f"Sampling bias run #{run.pk} has been queued.")
    return redirect(run)


@require_POST
def create_sensitivity_run(request):
    form = SensitivityForm(request.POST, prefix="sensitivity")
    if not form.is_valid():
        return render(request, "lab/dashboard.html", _dashboard_context(sensitivity_form=form))

    cleaned = form.cleaned_data
    run = ExperimentRun.objects.create(
        name=cleaned["name"],
        experiment_type=ExperimentRun.TYPE_SENSITIVITY,
        dataset=cleaned["dataset"],
        config={
            "dataset": cleaned["dataset"],
            "methods": cleaned["methods"],
            "params": cleaned["params"],
            "gpu_id": cleaned["gpu_id"],
            "base_rank": cleaned["base_rank"],
            "runs": cleaned["runs"],
            "std_weight": cleaned["std_weight"],
            "neighbor_span": cleaned["neighbor_span"],
            "continue_on_error": cleaned["continue_on_error"],
        },
    )
    launch_experiment(run)
    messages.success(request, f"Sensitivity run #{run.pk} has been queued.")
    return redirect(run)


@require_POST
def stop_run(request, pk):
    run = get_object_or_404(ExperimentRun, pk=pk)
    next_url = request.POST.get("next") or run.get_absolute_url()

    if not run.is_active:
        messages.error(request, f"Run #{run.pk} is not running anymore.")
        return redirect(next_url)

    if not run.worker_pid:
        messages.error(request, f"Run #{run.pk} has no recorded worker PID, so it cannot be stopped safely.")
        return redirect(next_url)

    try:
        stop_experiment_run(run)
    except RuntimeError as exc:
        messages.error(request, f"Run #{run.pk} could not be stopped: {exc}")
    else:
        messages.success(request, f"Run #{run.pk} was aborted and its related output files were removed.")

    return redirect(next_url)


@require_POST
def delete_run(request, pk):
    run = get_object_or_404(ExperimentRun.objects.prefetch_related("artifacts"), pk=pk)

    if run.is_active:
        messages.error(request, f"Run #{run.pk} is still active. Stop it before deleting the record.")
        next_url = request.POST.get("next") or run.get_absolute_url()
        return redirect(next_url)

    run_label = run.display_name
    run_id = run.pk
    delete_experiment_run(run)
    messages.success(request, f"Run #{run_id} ({run_label}) and its related files were deleted.")

    next_url = request.POST.get("next")
    return redirect(next_url or "lab:dashboard")


def _safe_artifact_path(relative_path):
    candidate = (BASE_DIR / relative_path).resolve()
    if not str(candidate).startswith(str(BASE_DIR)):
        raise Http404("Invalid artifact path")
    if not candidate.exists() or not candidate.is_file():
        raise Http404("Artifact not found")
    return candidate


def _build_detail_payload(run, artifacts):
    summary = run.result_summary or {}
    best_robust = summary.get("best_robust")
    final_violation_rate = summary.get("final_violation_rate")
    best_margin = summary.get("best_margin")
    csv_previews = []
    image_artifacts = []
    report_blocks = []
    chart_svgs = []

    for artifact in artifacts:
        absolute_path = _safe_artifact_path(artifact.relative_path)
        if artifact.artifact_type == ExperimentArtifact.TYPE_CSV:
            csv_previews.append(
                {
                    "artifact": artifact,
                    "preview": preview_csv(absolute_path),
                }
            )
        elif artifact.artifact_type == ExperimentArtifact.TYPE_IMAGE:
            image_artifacts.append(artifact)
        elif artifact.artifact_type == ExperimentArtifact.TYPE_REPORT:
            report_blocks.append(
                {
                    "artifact": artifact,
                    "content": read_text_file(absolute_path),
                }
            )

    summary_cards = []
    if run.experiment_type == ExperimentRun.TYPE_METHOD_COMPARISON and summary.get("methods"):
        summary_cards = [
            {"label": "Best Method", "value": summary.get("best_method", "-")},
            {"label": "Best Robust Score", "value": f"{(best_robust or 0.0):.4f}"},
            {"label": "Compared Methods", "value": str(len(summary.get("methods", [])))},
        ]
        chart_svgs.append(
            render_bar_chart(
                "Method Comparison Robust Score",
                [
                    {"label": item["label"], "value": item["robust_score"]}
                    for item in summary["methods"]
                    if item.get("robust_score") is not None
                ],
            )
        )
    elif run.experiment_type == ExperimentRun.TYPE_SAMPLING_BIAS and summary.get("points"):
        summary_cards = [
            {"label": "Epochs", "value": str(summary.get("epochs", 0))},
            {"label": "Final violation_rate", "value": f"{(final_violation_rate or 0.0):.4f}"},
            {"label": "Best mean_margin", "value": f"{(best_margin or 0.0):.4f}"},
        ]
        chart_svgs.append(
            render_line_chart(
                "Violation Rate Curve",
                summary["points"],
                "epoch",
                "violation_rate",
                "#c2410c",
                width=560,
                height=220,
            )
        )
        chart_svgs.append(
            render_line_chart(
                "Mean Margin Curve",
                summary["points"],
                "epoch",
                "mean_margin",
                "#1d4ed8",
                width=560,
                height=220,
            )
        )
    elif run.experiment_type == ExperimentRun.TYPE_SENSITIVITY and summary.get("best_rows"):
        summary_cards = [
            {"label": "Summary Rows", "value": str(summary.get("summary_rows", 0))},
            {"label": "Methods", "value": " / ".join(summary.get("methods", []))},
            {"label": "Params", "value": " / ".join(summary.get("params", []))},
        ]
        chart_svgs.append(
            render_bar_chart(
                "Sensitivity Best Robust Score",
                [
                    {"label": item["label"], "value": item["robust_score"]}
                    for item in summary["best_rows"]
                ],
            )
        )
        if summary.get("report_text") and not report_blocks:
            report_blocks.append(
                {
                    "artifact": None,
                    "content": summary["report_text"],
                }
            )

    return {
        "summary_cards": summary_cards,
        "chart_svgs": chart_svgs,
        "csv_previews": csv_previews,
        "image_artifacts": image_artifacts,
        "report_blocks": report_blocks,
    }


@require_GET
def run_detail(request, pk):
    run = get_object_or_404(ExperimentRun.objects.prefetch_related("artifacts"), pk=pk)
    artifacts = list(run.artifacts.all())
    payload = _build_detail_payload(run, artifacts)
    compact_summary = _compact_value(run.result_summary)
    summary_json = json.dumps(run.result_summary, ensure_ascii=False, indent=2)
    summary_json_compact = json.dumps(compact_summary, ensure_ascii=False, indent=2)

    context = {
        "run": run,
        "config_json": json.dumps(run.config, ensure_ascii=False, indent=2),
        "summary_json": summary_json,
        "summary_json_compact": summary_json_compact,
        "summary_json_truncated": summary_json != summary_json_compact,
        **payload,
    }
    return render(request, "lab/run_detail.html", context)


@require_GET
def artifact_file(request, run_id, artifact_id):
    artifact = get_object_or_404(ExperimentArtifact, pk=artifact_id, run_id=run_id)
    absolute_path = _safe_artifact_path(artifact.relative_path)
    return FileResponse(absolute_path.open("rb"), as_attachment=False, filename=artifact.filename)
