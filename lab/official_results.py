from datetime import datetime, timezone as dt_timezone
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from .constants import DATASET_CHOICES, EXPERIMENT_TYPE_LABELS, METHOD_LABELS
from .models import ExperimentArtifact, ExperimentRun
from .parsers import (
    build_method_comparison_summary,
    build_sampling_bias_summary,
    build_sensitivity_summary,
)


BASE_DIR = Path(settings.BASE_DIR).resolve()
RESULTS_DIR = (BASE_DIR / "results").resolve()
LOGS_DIR = (BASE_DIR / "logs").resolve()
PLOTS_DIR = (RESULTS_DIR / "plots").resolve()
DATASET_LABELS = {label.lower(): label for _, label in DATASET_CHOICES}
DATASET_ORDER = {label.lower(): index for index, (_, label) in enumerate(DATASET_CHOICES)}
TYPE_ORDER = {
    ExperimentRun.TYPE_SAMPLING_BIAS: 0,
    ExperimentRun.TYPE_METHOD_COMPARISON: 1,
    ExperimentRun.TYPE_SENSITIVITY: 2,
}


def _dataset_label(dataset_slug):
    return DATASET_LABELS.get(dataset_slug.lower(), dataset_slug.replace("_", " ").title())


def _relative_path(path):
    return path.resolve().relative_to(BASE_DIR).as_posix()


def _updated_at(paths):
    existing = [path.resolve() for path in paths if path.exists() and path.is_file()]
    if not existing:
        return None
    timestamp = max(path.stat().st_mtime for path in existing)
    return datetime.fromtimestamp(timestamp, tz=dt_timezone.utc).astimezone(timezone.get_current_timezone())


def _artifact(label, artifact_type, path, metadata=None):
    resolved = path.resolve()
    if not resolved.exists() or not resolved.is_file():
        return None
    return {
        "label": label,
        "artifact_type": artifact_type,
        "relative_path": _relative_path(resolved),
        "metadata": metadata or {},
    }


def _method_key_from_slug(method_slug):
    return {
        "iflgr": "ifl-gr",
        "iflgc": "ifl-gc",
    }.get(method_slug, method_slug)


def _method_comparison_entries():
    entries = []
    for full_csv in sorted(RESULTS_DIR.glob("*_full_pipeline_results.csv")):
        dataset_slug = full_csv.name[: -len("_full_pipeline_results.csv")]
        dataset = _dataset_label(dataset_slug)

        artifacts = [
            _artifact("Unified Results CSV", ExperimentArtifact.TYPE_CSV, full_csv),
            _artifact(
                "IFL-GR Grid Search",
                ExperimentArtifact.TYPE_CSV,
                RESULTS_DIR / f"grid_search_iflgr_{dataset_slug}_results.csv",
            ),
            _artifact(
                "GCA Grid Search",
                ExperimentArtifact.TYPE_CSV,
                RESULTS_DIR / f"grid_search_gca_{dataset_slug}_results.csv",
            ),
            _artifact(
                "IFL-GC Grid Search",
                ExperimentArtifact.TYPE_CSV,
                RESULTS_DIR / f"grid_search_iflgc_{dataset_slug}_results.csv",
            ),
        ]
        artifacts = [artifact for artifact in artifacts if artifact]

        summary = build_method_comparison_summary(full_csv)
        summary.update(
            {
                "dataset": dataset,
                "main_csv": _relative_path(full_csv),
            }
        )

        entries.append(
            {
                "slug": f"method-comparison-{dataset_slug}",
                "title": f"Method Comparison Pipeline / {dataset}",
                "experiment_type": ExperimentRun.TYPE_METHOD_COMPARISON,
                "type_label": EXPERIMENT_TYPE_LABELS[ExperimentRun.TYPE_METHOD_COMPARISON],
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "updated_at": _updated_at([(BASE_DIR / artifact["relative_path"]).resolve() for artifact in artifacts]),
                "summary": summary,
                "config": {
                    "source": "Official archived result",
                    "storage": "results/ (outside webapp)",
                    "dataset": dataset,
                    "entry_type": EXPERIMENT_TYPE_LABELS[ExperimentRun.TYPE_METHOD_COMPARISON],
                    "artifacts": [artifact["relative_path"] for artifact in artifacts],
                },
                "artifacts": artifacts,
            }
        )

    return entries


def _sampling_bias_entries():
    entries = []
    for csv_path in sorted(LOGS_DIR.glob("exp1_*.csv")):
        dataset_slug = csv_path.stem[len("exp1_") :]
        dataset = _dataset_label(dataset_slug)
        plot_path = LOGS_DIR / f"exp1_{dataset_slug}_curves.png"

        artifacts = [
            _artifact("Sampling Bias CSV", ExperimentArtifact.TYPE_CSV, csv_path),
            _artifact("Sampling Bias Plot", ExperimentArtifact.TYPE_IMAGE, plot_path),
        ]
        artifacts = [artifact for artifact in artifacts if artifact]

        summary = build_sampling_bias_summary(csv_path)
        summary.update(
            {
                "dataset": dataset,
                "csv_path": _relative_path(csv_path),
                "plot_path": _relative_path(plot_path) if plot_path.exists() else "",
            }
        )

        entries.append(
            {
                "slug": f"sampling-bias-{dataset_slug}",
                "title": f"Sampling Bias Validation / {dataset}",
                "experiment_type": ExperimentRun.TYPE_SAMPLING_BIAS,
                "type_label": EXPERIMENT_TYPE_LABELS[ExperimentRun.TYPE_SAMPLING_BIAS],
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "updated_at": _updated_at([csv_path, plot_path]),
                "summary": summary,
                "config": {
                    "source": "Official archived result",
                    "storage": "logs/ (outside webapp)",
                    "dataset": dataset,
                    "entry_type": EXPERIMENT_TYPE_LABELS[ExperimentRun.TYPE_SAMPLING_BIAS],
                    "artifacts": [artifact["relative_path"] for artifact in artifacts],
                },
                "artifacts": artifacts,
            }
        )

    return entries


def _sensitivity_entries():
    grouped = {}
    for csv_path in sorted(RESULTS_DIR.glob("sensitivity_*_results.csv")):
        parts = csv_path.stem.split("_")
        if len(parts) < 4:
            continue
        dataset_slug = parts[-2]
        grouped.setdefault(dataset_slug, []).append(csv_path)

    entries = []
    for dataset_slug, csv_paths in grouped.items():
        dataset = _dataset_label(dataset_slug)
        plot_path = PLOTS_DIR / f"{dataset_slug}_ifl_sensitivity_overview.png"
        report_path = PLOTS_DIR / f"{dataset_slug}_ifl_sensitivity_analysis.md"

        artifacts = []
        for csv_path in csv_paths:
            method_slug = csv_path.stem.split("_")[1]
            method_key = _method_key_from_slug(method_slug)
            artifacts.append(
                _artifact(
                    f"{METHOD_LABELS.get(method_key, method_slug.upper())} Sensitivity CSV",
                    ExperimentArtifact.TYPE_CSV,
                    csv_path,
                )
            )
        artifacts.extend(
            [
                _artifact("Sensitivity Overview", ExperimentArtifact.TYPE_IMAGE, plot_path),
                _artifact("Sensitivity Report", ExperimentArtifact.TYPE_REPORT, report_path),
            ]
        )
        artifacts = [artifact for artifact in artifacts if artifact]

        summary = build_sensitivity_summary(csv_paths, report_path)
        summary.update(
            {
                "dataset": dataset,
                "plot_path": _relative_path(plot_path) if plot_path.exists() else "",
                "report_path": _relative_path(report_path) if report_path.exists() else "",
            }
        )

        entries.append(
            {
                "slug": f"sensitivity-{dataset_slug}",
                "title": f"Sensitivity Analysis / {dataset}",
                "experiment_type": ExperimentRun.TYPE_SENSITIVITY,
                "type_label": EXPERIMENT_TYPE_LABELS[ExperimentRun.TYPE_SENSITIVITY],
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "updated_at": _updated_at([*csv_paths, plot_path, report_path]),
                "summary": summary,
                "config": {
                    "source": "Official archived result",
                    "storage": "results/ and results/plots/ (outside webapp)",
                    "dataset": dataset,
                    "entry_type": EXPERIMENT_TYPE_LABELS[ExperimentRun.TYPE_SENSITIVITY],
                    "artifacts": [artifact["relative_path"] for artifact in artifacts],
                },
                "artifacts": artifacts,
            }
        )

    return entries


def list_official_results():
    entries = [
        *_sampling_bias_entries(),
        *_method_comparison_entries(),
        *_sensitivity_entries(),
    ]
    return sorted(
        entries,
        key=lambda entry: (
            TYPE_ORDER.get(entry["experiment_type"], 99),
            DATASET_ORDER.get(entry["dataset_slug"], 99),
            entry["title"],
        ),
    )


def get_official_result(slug):
    for entry in list_official_results():
        if entry["slug"] == slug:
            return entry
    return None
