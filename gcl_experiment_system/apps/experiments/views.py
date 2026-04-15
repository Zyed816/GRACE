import json
from pathlib import Path

from django.conf import settings
from django.db.models import Avg
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from apps.datasets.catalog import DATASET_CATALOG
from apps.models.catalog import METHOD_CATALOG

from .forms import ExperimentForm
from .models import Experiment, ExperimentLog
from .services import enqueue_experiment, import_result_csv


def dashboard(request):
    recent_experiments = Experiment.objects.all()[:5]
    context = {
        "method_count": len(METHOD_CATALOG),
        "dataset_count": len(DATASET_CATALOG),
        "experiment_count": Experiment.objects.count(),
        "recent_experiments": recent_experiments,
        "latest_success": Experiment.objects.filter(status=Experiment.STATUS_SUCCEEDED).first(),
    }
    return render(request, "dashboard.html", context)


def experiment_create(request):
    if request.method == "POST":
        form = ExperimentForm(request.POST)
        if form.is_valid():
            experiment = form.save(commit=False)
            experiment.extra_params = form.cleaned_data.get("extra_params_text", {})
            experiment.status = Experiment.STATUS_PENDING
            experiment.save()
            if form.cleaned_data.get("run_now"):
                enqueue_experiment(experiment.id)
            return redirect("experiment-detail", pk=experiment.pk)
    else:
        form = ExperimentForm()
    return render(request, "experiments/form.html", {"form": form})


def experiment_detail(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    logs = ExperimentLog.objects.filter(experiment=experiment).order_by("epoch")[:200]
    return render(request, "experiments/detail.html", {"experiment": experiment, "logs": logs})


def experiment_history(request):
    experiments = Experiment.objects.all()
    return render(request, "experiments/history.html", {"experiments": experiments})


def experiment_monitor(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    logs = list(
        ExperimentLog.objects.filter(experiment=experiment)
        .order_by("epoch")
        .values("epoch", "loss", "accuracy", "payload")
    )
    return JsonResponse({
        "experiment_id": experiment.pk,
        "status": experiment.status,
        "logs": logs,
        "final_accuracy": experiment.final_accuracy,
        "final_f1mi": experiment.final_f1mi,
        "final_f1ma": experiment.final_f1ma,
    })


def results_overview(request):
    method_entries = list(METHOD_CATALOG.values())
    pivot = []
    datasets = list(DATASET_CATALOG.keys())
    series_map = {method["display_name"]: [] for method in method_entries}
    for dataset_name in DATASET_CATALOG:
        row = {"dataset": dataset_name}
        for method_key, method in METHOD_CATALOG.items():
            q = Experiment.objects.filter(dataset=dataset_name, model_name=method_key, status=Experiment.STATUS_SUCCEEDED)
            value = q.aggregate(v=Avg("final_accuracy"))["v"]
            row[method["display_name"]] = value
            series_map[method["display_name"]].append(value if value is not None else 0.0)
        pivot.append(row)

    chart_option = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": list(series_map.keys())},
        "xAxis": {"type": "category", "data": datasets},
        "yAxis": {"type": "value"},
        "series": [
            {"name": name, "type": "bar", "data": values}
            for name, values in series_map.items()
        ],
    }

    csv_rows = []
    results_dir = Path(settings.GRACE_RESULTS_DIR)
    for csv_path in sorted(results_dir.glob("*_full_pipeline_results.csv")):
        csv_rows.extend(import_result_csv(csv_path))

    return render(request, "experiments/results.html", {
        "pivot": pivot,
        "methods": method_entries,
        "csv_rows": csv_rows,
        "chart_option_json": json.dumps(chart_option, ensure_ascii=False),
    })


def api_monitor(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    logs = list(
        ExperimentLog.objects.filter(experiment=experiment)
        .order_by("epoch")
        .values("epoch", "loss", "accuracy")[:500]
    )
    return JsonResponse({
        "id": experiment.pk,
        "status": experiment.status,
        "logs": logs,
        "final_accuracy": experiment.final_accuracy,
        "run_seconds": experiment.run_seconds,
    })


def experiment_start(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    if experiment.status == Experiment.STATUS_PENDING:
        enqueue_experiment(experiment.pk)
    return redirect("experiment-detail", pk=experiment.pk)
