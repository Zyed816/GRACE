import json
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST

from .constants import METHOD_DISPLAY_ORDER, METHOD_LABELS
from .forms import (
    ComponentAblationForm,
    EfficiencyForm,
    MethodComparisonForm,
    SamplingBiasForm,
    SignificanceForm,
    SensitivityForm,
)
from .models import ExperimentArtifact, ExperimentRun
from .official_results import get_official_result, list_official_results
from .parsers import build_sensitivity_series, preview_csv, read_text_file
from .services import delete_experiment_run, launch_experiment, stop_experiment_run
from .ui_text import (
    experiment_type_label,
    get_ui_language,
    localized_run_name,
    localize_artifact_label,
    sensitivity_param_label,
    status_label,
    text,
)
from .visuals import render_bar_chart, render_line_chart, render_multi_line_chart


BASE_DIR = Path(settings.BASE_DIR).resolve()
OFFICIAL_ROOTS = [(BASE_DIR / "results").resolve(), (BASE_DIR / "logs").resolve()]
OFFICIAL_BLOCKED_DIRS = [(BASE_DIR / "results" / "webapp").resolve(), (BASE_DIR / "logs" / "webapp").resolve()]
METHOD_DISPLAY_INDEX = {method: index for index, method in enumerate(METHOD_DISPLAY_ORDER)}


def _compact_value(value, lang, depth=0):
    if isinstance(value, dict):
        if depth >= 2:
            return "{" + text("summary.keys_omitted", lang, count=len(value)) + "}"
        return {key: _compact_value(subvalue, lang, depth + 1) for key, subvalue in value.items()}

    if isinstance(value, list):
        if len(value) <= 6 and depth < 2:
            return [_compact_value(item, lang, depth + 1) for item in value]
        return {
            "items": len(value),
            "preview": [_compact_value(item, lang, depth + 1) for item in value[:3]],
            "note": text("summary.truncated_note", lang),
        }

    if isinstance(value, str) and len(value) > 320:
        return f"{value[:320]}... {text('summary.truncated_suffix', lang)}"

    return value


def _serialize_recent_run(run, lang):
    return {
        "pk": run.pk,
        "status": run.status,
        "status_label": status_label(run.status, lang),
        "url": run.get_absolute_url(),
        "title": localized_run_name(run, lang),
        "type_label": experiment_type_label(run.experiment_type, lang),
        "dataset": run.dataset,
        "created_at": run.created_at,
        "can_stop": run.can_stop,
        "is_active": run.is_active,
        "can_delete": run.can_delete,
        "stop_confirm_text": text("confirm.stop_run", lang, run_id=run.pk),
        "delete_confirm_text": text("confirm.delete_run", lang, run_id=run.pk),
    }


def _dashboard_context(
    *,
    lang,
    method_form=None,
    sampling_form=None,
    sensitivity_form=None,
    ablation_form=None,
    efficiency_form=None,
    significance_form=None,
):
    recent_runs = [
        _serialize_recent_run(run, lang)
        for run in ExperimentRun.objects.prefetch_related("artifacts").all()[:8]
    ]
    official_results = [
        {
            **entry,
            "url": reverse("lab:official_result_detail", kwargs={"slug": entry["slug"]}),
        }
        for entry in list_official_results(lang)
    ]
    return {
        "method_form": method_form or MethodComparisonForm(prefix="method", lang=lang),
        "sampling_form": sampling_form or SamplingBiasForm(prefix="sampling", lang=lang),
        "sensitivity_form": sensitivity_form or SensitivityForm(prefix="sensitivity", lang=lang),
        "ablation_form": ablation_form or ComponentAblationForm(prefix="ablation", lang=lang),
        "efficiency_form": efficiency_form or EfficiencyForm(prefix="efficiency", lang=lang),
        "significance_form": significance_form or SignificanceForm(prefix="significance", lang=lang),
        "recent_runs": recent_runs,
        "official_results": official_results,
    }


def _safe_run_artifact_path(relative_path):
    candidate = (BASE_DIR / relative_path).resolve()
    if not str(candidate).startswith(str(BASE_DIR)):
        raise Http404("Invalid artifact path")
    if not candidate.exists() or not candidate.is_file():
        raise Http404("Artifact not found")
    return candidate


def _safe_official_artifact_path(relative_path):
    candidate = (BASE_DIR / relative_path).resolve()
    if not any(candidate == root or root in candidate.parents for root in OFFICIAL_ROOTS):
        raise Http404("Invalid artifact path")
    if any(blocked == candidate or blocked in candidate.parents for blocked in OFFICIAL_BLOCKED_DIRS):
        raise Http404("Artifact not found")
    if not candidate.exists() or not candidate.is_file():
        raise Http404("Artifact not found")
    return candidate


def _serialize_run_artifact(run, artifact, lang):
    absolute_path = _safe_run_artifact_path(artifact.relative_path)
    return {
        "label": localize_artifact_label(artifact.label, lang),
        "artifact_type": artifact.artifact_type,
        "relative_path": artifact.relative_path,
        "metadata": artifact.metadata or {},
        "url": reverse("lab:artifact_file", kwargs={"run_id": run.pk, "artifact_id": artifact.pk}),
        "absolute_path": absolute_path,
    }


def _serialize_official_artifact(artifact, lang):
    absolute_path = _safe_official_artifact_path(artifact["relative_path"])
    return {
        **artifact,
        "label": localize_artifact_label(artifact["label"], lang),
        "url": reverse("lab:official_artifact_file", kwargs={"relative_path": artifact["relative_path"]}),
        "absolute_path": absolute_path,
    }


def _order_method_rows(method_rows):
    return sorted(
        method_rows,
        key=lambda item: (
            METHOD_DISPLAY_INDEX.get(item.get("method", ""), len(METHOD_DISPLAY_INDEX)),
            item.get("label", ""),
        ),
    )


def _format_float(value, digits=4, suffix=""):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def _build_detail_payload(experiment_type, summary, artifacts, lang):
    best_robust = summary.get("best_robust")
    final_violation_rate = summary.get("final_violation_rate")
    best_margin = summary.get("best_margin")
    csv_previews = []
    image_artifacts = []
    report_blocks = []
    chart_svgs = []
    sensitivity_csv_paths = []
    chart_grid_class = ""
    csv_stack_class = ""

    for artifact in artifacts:
        absolute_path = artifact["absolute_path"]
        if artifact["artifact_type"] == ExperimentArtifact.TYPE_CSV:
            csv_previews.append({"artifact": artifact, "preview": preview_csv(absolute_path)})
            if experiment_type == ExperimentRun.TYPE_SENSITIVITY:
                sensitivity_csv_paths.append(absolute_path)
        elif artifact["artifact_type"] == ExperimentArtifact.TYPE_IMAGE:
            image_artifacts.append(artifact)
        elif artifact["artifact_type"] == ExperimentArtifact.TYPE_REPORT:
            report_blocks.append({"artifact": artifact, "content": read_text_file(absolute_path)})

    summary_cards = []
    if experiment_type == ExperimentRun.TYPE_METHOD_COMPARISON and summary.get("methods"):
        chart_grid_class = "chart-grid-wide"
        ordered_methods = _order_method_rows(summary["methods"])
        summary_cards = [
            {"label": text("summary.best_method", lang), "value": summary.get("best_method", "-")},
            {"label": text("summary.best_robust_score", lang), "value": f"{(best_robust or 0.0):.4f}"},
            {"label": text("summary.compared_methods", lang), "value": str(len(summary.get("methods", [])))},
        ]
        chart = render_bar_chart(
            text("charts.method_robust_score", lang),
            [
                {"label": item["label"], "value": item["robust_score"]}
                for item in ordered_methods
                if item.get("robust_score") is not None
            ],
            width=920,
            height=360,
        )
        if chart:
            chart_svgs.append(chart)
    elif experiment_type == ExperimentRun.TYPE_SAMPLING_BIAS and summary.get("points"):
        summary_cards = [
            {"label": text("summary.epochs", lang), "value": str(summary.get("epochs", 0))},
            {"label": text("summary.final_violation_rate", lang), "value": f"{(final_violation_rate or 0.0):.4f}"},
            {"label": text("summary.best_mean_margin", lang), "value": f"{(best_margin or 0.0):.4f}"},
        ]
        violation_chart = render_line_chart(
            text("charts.violation_rate", lang),
            summary["points"],
            "epoch",
            "violation_rate",
            "#c2410c",
            width=560,
            height=220,
        )
        margin_chart = render_line_chart(
            text("charts.mean_margin", lang),
            summary["points"],
            "epoch",
            "mean_margin",
            "#1d4ed8",
            width=560,
            height=220,
        )
        if violation_chart:
            chart_svgs.append(violation_chart)
        if margin_chart:
            chart_svgs.append(margin_chart)
    elif experiment_type == ExperimentRun.TYPE_SENSITIVITY and summary.get("best_rows"):
        sensitivity_series = summary.get("sensitivity_series") or build_sensitivity_series(sensitivity_csv_paths)
        methods = summary.get("methods") or [item.get("label", "-") for item in summary.get("best_rows", [])]
        params = summary.get("params") or [item["param"] for item in sensitivity_series]
        rendered_methods = [METHOD_LABELS.get(method, method) for method in methods]
        rendered_params = [sensitivity_param_label(param, lang) for param in params]
        summary_cards = [
            {"label": text("summary.summary_rows", lang), "value": str(summary.get("summary_rows", 0))},
            {"label": text("summary.methods", lang), "value": " / ".join(rendered_methods)},
            {"label": text("summary.params", lang), "value": " / ".join(rendered_params)},
        ]

        best_chart = render_bar_chart(
            text("charts.sensitivity_best", lang),
            [
                {"label": item["label"], "value": item["robust_score"]}
                for item in summary["best_rows"]
                if item.get("robust_score") is not None
            ],
        )
        if best_chart:
            chart_svgs.append(best_chart)

        for series_group in sensitivity_series:
            chart = render_multi_line_chart(
                text("charts.sensitivity_series", lang, label=series_group["label"]),
                series_group["series"],
                width=600,
                height=250,
            )
            if chart:
                chart_svgs.append(chart)

        if summary.get("report_text") and not report_blocks:
            report_blocks.append({"artifact": None, "content": summary["report_text"]})
    elif experiment_type == ExperimentRun.TYPE_COMPONENT_ABLATION:
        max_drop = summary.get("max_drop") or {}
        methods = summary.get("method_labels") or [METHOD_LABELS.get(method, method) for method in summary.get("methods", [])]
        summary_cards = [
            {"label": text("summary.summary_rows", lang), "value": str(summary.get("summary_rows", 0))},
            {"label": text("summary.methods", lang), "value": " / ".join(methods) or "-"},
            {"label": text("summary.variants", lang), "value": str(summary.get("variant_count", 0))},
            {
                "label": text("summary.max_drop", lang),
                "value": _format_float(max_drop.get("drop_vs_full"), 4),
            },
        ]
        if summary.get("report_text") and not report_blocks:
            report_blocks.append({"artifact": None, "content": summary["report_text"]})
    elif experiment_type == ExperimentRun.TYPE_EFFICIENCY:
        fastest = summary.get("fastest_method") or {}
        summary_cards = [
            {"label": text("summary.summary_rows", lang), "value": str(summary.get("summary_rows", 0))},
            {"label": text("summary.fastest_method", lang), "value": fastest.get("label", "-")},
            {
                "label": text("summary.train_total_time", lang),
                "value": _format_float(fastest.get("train_total_sec"), 2, "s"),
            },
            {"label": text("summary.sggr_ratio", lang), "value": _format_float(summary.get("sggr_ratio"), 2, "x")},
            {"label": text("summary.sggc_ratio", lang), "value": _format_float(summary.get("sggc_ratio"), 2, "x")},
        ]
        if summary.get("report_text") and not report_blocks:
            report_blocks.append({"artifact": None, "content": summary["report_text"]})
    elif experiment_type == ExperimentRun.TYPE_SIGNIFICANCE:
        best_delta = summary.get("best_delta") or {}
        summary_cards = [
            {"label": text("summary.primary_tests", lang), "value": str(summary.get("primary_tests", 0))},
            {"label": text("summary.significant_tests", lang), "value": str(summary.get("significant_tests", 0))},
            {"label": text("summary.best_delta", lang), "value": _format_float(best_delta.get("mean_delta"), 4)},
        ]
        if summary.get("report_text") and not report_blocks:
            report_blocks.append({"artifact": None, "content": summary["report_text"]})

    if len(csv_previews) > 1:
        csv_stack_class = "csv-stack-carousel"

    return {
        "summary_cards": summary_cards,
        "chart_svgs": chart_svgs,
        "chart_grid_class": chart_grid_class,
        "csv_previews": csv_previews,
        "csv_stack_class": csv_stack_class,
        "image_artifacts": image_artifacts,
        "report_blocks": report_blocks,
    }


def _build_detail_context(
    *,
    lang,
    detail_eyebrow,
    detail_title,
    experiment_type,
    type_label,
    dataset,
    status_label_text,
    status_class,
    config,
    summary,
    artifacts,
    back_url,
    created_at=None,
    created_label=None,
    started_at=None,
    finished_at=None,
    page_notice="",
    config_title=None,
    show_execution=False,
    command_text="",
    runtime_log="",
    error_message="",
    stop_action_url="",
    stop_next="",
    stop_button_label=None,
    stop_confirm_text="",
    show_disabled_stop=False,
    disabled_stop_label=None,
    delete_action_url="",
    delete_next="",
    delete_button_label=None,
    delete_confirm_text="",
):
    payload = _build_detail_payload(experiment_type, summary, artifacts, lang)
    compact_summary = _compact_value(summary, lang)
    summary_json = json.dumps(summary or {}, ensure_ascii=False, indent=2)
    summary_json_compact = json.dumps(compact_summary, ensure_ascii=False, indent=2)

    return {
        "detail_eyebrow": detail_eyebrow,
        "detail_title": detail_title,
        "type_label": type_label,
        "dataset": dataset,
        "status_label": status_label_text,
        "status_class": status_class,
        "back_url": back_url,
        "created_at": created_at,
        "created_label": created_label or text("labels.created_at", lang),
        "started_at": started_at,
        "started_label": text("detail.started_at", lang),
        "finished_at": finished_at,
        "finished_label": text("detail.finished_at", lang),
        "page_notice": page_notice,
        "config_title": config_title or text("config.run", lang),
        "config_json": json.dumps(config or {}, ensure_ascii=False, indent=2),
        "summary_json": summary_json,
        "summary_json_compact": summary_json_compact,
        "summary_json_truncated": summary_json != summary_json_compact,
        "show_execution": show_execution,
        "command_text": command_text,
        "runtime_log": runtime_log,
        "error_message": error_message,
        "show_stop_action": bool(stop_action_url),
        "stop_action_url": stop_action_url,
        "stop_next": stop_next,
        "stop_button_label": stop_button_label or text("button.stop", lang),
        "stop_confirm_text": stop_confirm_text,
        "show_disabled_stop": show_disabled_stop,
        "disabled_stop_label": disabled_stop_label or text("button.stop_unavailable", lang),
        "show_delete_action": bool(delete_action_url),
        "delete_action_url": delete_action_url,
        "delete_next": delete_next,
        "delete_button_label": delete_button_label or text("button.delete", lang),
        "delete_confirm_text": delete_confirm_text,
        **payload,
    }


@require_GET
def dashboard(request):
    lang = get_ui_language(request)
    return render(request, "lab/dashboard.html", _dashboard_context(lang=lang))


@require_POST
def create_method_comparison_run(request):
    lang = get_ui_language(request)
    form = MethodComparisonForm(request.POST, prefix="method", lang=lang)
    if not form.is_valid():
        return render(request, "lab/dashboard.html", _dashboard_context(lang=lang, method_form=form))

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
    messages.success(request, text("message.method_queued", lang, run_id=run.pk))
    return redirect(run)


@require_POST
def create_sampling_bias_run(request):
    lang = get_ui_language(request)
    form = SamplingBiasForm(request.POST, prefix="sampling", lang=lang)
    if not form.is_valid():
        return render(request, "lab/dashboard.html", _dashboard_context(lang=lang, sampling_form=form))

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
    messages.success(request, text("message.sampling_queued", lang, run_id=run.pk))
    return redirect(run)


@require_POST
def create_sensitivity_run(request):
    lang = get_ui_language(request)
    form = SensitivityForm(request.POST, prefix="sensitivity", lang=lang)
    if not form.is_valid():
        return render(request, "lab/dashboard.html", _dashboard_context(lang=lang, sensitivity_form=form))

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
    messages.success(request, text("message.sensitivity_queued", lang, run_id=run.pk))
    return redirect(run)


@require_POST
def create_component_ablation_run(request):
    lang = get_ui_language(request)
    form = ComponentAblationForm(request.POST, prefix="ablation", lang=lang)
    if not form.is_valid():
        return render(request, "lab/dashboard.html", _dashboard_context(lang=lang, ablation_form=form))

    cleaned = form.cleaned_data
    run = ExperimentRun.objects.create(
        name=cleaned["name"],
        experiment_type=ExperimentRun.TYPE_COMPONENT_ABLATION,
        dataset=cleaned["dataset"],
        config={
            "dataset": cleaned["dataset"],
            "methods": cleaned["methods"],
            "gpu_id": cleaned["gpu_id"],
            "runs": cleaned["runs"],
            "std_weight": cleaned["std_weight"],
            "continue_on_error": cleaned["continue_on_error"],
        },
    )
    launch_experiment(run)
    messages.success(request, text("message.ablation_queued", lang, run_id=run.pk))
    return redirect(run)


@require_POST
def create_efficiency_run(request):
    lang = get_ui_language(request)
    form = EfficiencyForm(request.POST, prefix="efficiency", lang=lang)
    if not form.is_valid():
        return render(request, "lab/dashboard.html", _dashboard_context(lang=lang, efficiency_form=form))

    cleaned = form.cleaned_data
    run = ExperimentRun.objects.create(
        name=cleaned["name"],
        experiment_type=ExperimentRun.TYPE_EFFICIENCY,
        dataset=cleaned["dataset"],
        config={
            "dataset": cleaned["dataset"],
            "methods": cleaned["methods"],
            "gpu_id": cleaned["gpu_id"],
            "runs": cleaned["runs"],
            "std_weight": cleaned["std_weight"],
            "continue_on_error": cleaned["continue_on_error"],
        },
    )
    launch_experiment(run)
    messages.success(request, text("message.efficiency_queued", lang, run_id=run.pk))
    return redirect(run)


@require_POST
def create_significance_run(request):
    lang = get_ui_language(request)
    form = SignificanceForm(request.POST, prefix="significance", lang=lang)
    if not form.is_valid():
        return render(request, "lab/dashboard.html", _dashboard_context(lang=lang, significance_form=form))

    cleaned = form.cleaned_data
    run = ExperimentRun.objects.create(
        name=cleaned["name"],
        experiment_type=ExperimentRun.TYPE_SIGNIFICANCE,
        dataset=cleaned["dataset"],
        config={
            "dataset": cleaned["dataset"],
            "comparison_pairs": cleaned["comparison_pairs"],
            "gpu_id": cleaned["gpu_id"],
            "runs": cleaned["runs"],
            "eval_repeats": cleaned["eval_repeats"],
            "std_weight": cleaned["std_weight"],
            "alpha": cleaned["alpha"],
            "continue_on_error": cleaned["continue_on_error"],
        },
    )
    launch_experiment(run)
    messages.success(request, text("message.significance_queued", lang, run_id=run.pk))
    return redirect(run)


@require_POST
def stop_run(request, pk):
    lang = get_ui_language(request)
    run = get_object_or_404(ExperimentRun, pk=pk)
    next_url = request.POST.get("next") or run.get_absolute_url()

    if not run.is_active:
        messages.error(request, text("message.not_running", lang, run_id=run.pk))
        return redirect(next_url)

    if not run.worker_pid:
        messages.error(request, text("message.missing_pid", lang, run_id=run.pk))
        return redirect(next_url)

    try:
        stop_experiment_run(run)
    except RuntimeError as exc:
        messages.error(request, text("message.stop_failed", lang, run_id=run.pk, error=exc))
    else:
        messages.success(request, text("message.stop_success", lang, run_id=run.pk))

    return redirect(next_url)


@require_POST
def delete_run(request, pk):
    lang = get_ui_language(request)
    run = get_object_or_404(ExperimentRun.objects.prefetch_related("artifacts"), pk=pk)

    if run.is_active:
        messages.error(request, text("message.delete_active", lang, run_id=run.pk))
        next_url = request.POST.get("next") or run.get_absolute_url()
        return redirect(next_url)

    run_label = localized_run_name(run, lang)
    run_id = run.pk
    delete_experiment_run(run)
    messages.success(request, text("message.delete_success", lang, run_id=run_id, run_label=run_label))

    next_url = request.POST.get("next")
    return redirect(next_url or "lab:dashboard")


@require_GET
def run_detail(request, pk):
    lang = get_ui_language(request)
    run = get_object_or_404(ExperimentRun.objects.prefetch_related("artifacts"), pk=pk)
    artifacts = [_serialize_run_artifact(run, artifact, lang) for artifact in run.artifacts.all()]

    page_notice = ""
    if run.status == ExperimentRun.STATUS_RUNNING:
        page_notice = text("notice.running", lang)
    elif run.status == ExperimentRun.STATUS_ABORTED:
        page_notice = text("notice.aborted", lang)

    context = _build_detail_context(
        lang=lang,
        detail_eyebrow=text("detail_eyebrow.run", lang),
        detail_title=localized_run_name(run, lang),
        experiment_type=run.experiment_type,
        type_label=experiment_type_label(run.experiment_type, lang),
        dataset=run.dataset,
        status_label_text=status_label(run.status, lang),
        status_class=run.status,
        config=run.config,
        summary=run.result_summary or {},
        artifacts=artifacts,
        back_url=reverse("lab:dashboard"),
        created_at=run.created_at,
        created_label=text("labels.created_at", lang),
        started_at=run.started_at,
        finished_at=run.finished_at,
        page_notice=page_notice,
        config_title=text("config.run", lang),
        show_execution=True,
        command_text=run.command,
        runtime_log=run.stdout_log,
        error_message=run.error_message,
        stop_action_url=reverse("lab:stop_run", kwargs={"pk": run.pk}) if run.can_stop else "",
        stop_next=run.get_absolute_url(),
        stop_button_label=text("button.stop", lang),
        stop_confirm_text=text("confirm.stop_run", lang, run_id=run.pk),
        show_disabled_stop=run.is_active and not run.can_stop,
        disabled_stop_label=text("button.stop_unavailable", lang),
        delete_action_url=reverse("lab:delete_run", kwargs={"pk": run.pk}) if run.can_delete else "",
        delete_next=reverse("lab:dashboard"),
        delete_button_label=text("button.delete", lang),
        delete_confirm_text=text("confirm.delete_run", lang, run_id=run.pk),
    )
    return render(request, "lab/run_detail.html", context)


@require_GET
def official_result_detail(request, slug):
    lang = get_ui_language(request)
    entry = get_official_result(slug, lang)
    if entry is None:
        raise Http404(text("error.official_not_found", lang))

    artifacts = [_serialize_official_artifact(artifact, lang) for artifact in entry["artifacts"]]
    context = _build_detail_context(
        lang=lang,
        detail_eyebrow=text("detail_eyebrow.official", lang),
        detail_title=entry["title"],
        experiment_type=entry["experiment_type"],
        type_label=entry["type_label"],
        dataset=entry["dataset"],
        status_label_text=status_label("official", lang),
        status_class="official",
        config=entry["config"],
        summary=entry["summary"],
        artifacts=artifacts,
        back_url=reverse("lab:dashboard"),
        created_at=entry["updated_at"],
        created_label=text("labels.updated_at", lang),
        page_notice=text("notice.official", lang),
        config_title=text("config.result_metadata", lang),
        show_execution=False,
    )
    return render(request, "lab/run_detail.html", context)


@require_GET
def artifact_file(request, run_id, artifact_id):
    artifact = get_object_or_404(ExperimentArtifact, pk=artifact_id, run_id=run_id)
    absolute_path = _safe_run_artifact_path(artifact.relative_path)
    return FileResponse(absolute_path.open("rb"), as_attachment=False, filename=Path(artifact.relative_path).name)


@require_GET
def official_artifact_file(request, relative_path):
    absolute_path = _safe_official_artifact_path(relative_path)
    return FileResponse(absolute_path.open("rb"), as_attachment=False, filename=absolute_path.name)
