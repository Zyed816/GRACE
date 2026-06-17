from datetime import datetime, timezone as dt_timezone
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from .constants import DATASET_CHOICES, METHOD_LABELS
from .models import ExperimentArtifact, ExperimentRun
from .parsers import (
    build_component_ablation_summary,
    build_efficiency_summary,
    build_method_comparison_summary,
    build_sampling_bias_summary,
    build_significance_summary,
    build_sensitivity_summary,
)
from .ui_text import experiment_type_label, text


BASE_DIR = Path(settings.BASE_DIR).resolve()
RESULTS_DIR = (BASE_DIR / "results").resolve()
LOGS_DIR = (BASE_DIR / "logs").resolve()
PLOTS_DIR = (RESULTS_DIR / "plots").resolve()
DATASET_LABELS = {label.lower(): label for _, label in DATASET_CHOICES}
DATASET_ORDER = {label.lower(): index for index, (_, label) in enumerate(DATASET_CHOICES)}
TYPE_ORDER = {
    ExperimentRun.TYPE_SAMPLING_BIAS: 0,
    ExperimentRun.TYPE_METHOD_COMPARISON: 1,
    ExperimentRun.TYPE_COMPONENT_ABLATION: 2,
    ExperimentRun.TYPE_EFFICIENCY: 3,
    ExperimentRun.TYPE_SIGNIFICANCE: 4,
    ExperimentRun.TYPE_SENSITIVITY: 5,
}


def _dataset_label(dataset_slug):
    return DATASET_LABELS.get(dataset_slug.lower(), dataset_slug.replace("_", " ").title())


def _all_dataset_label(language):
    return "全部数据集" if language == "zh" else "All Datasets"


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


def _image_pair_artifacts(label, base_path):
    base = Path(base_path)
    return [
        _artifact(label, ExperimentArtifact.TYPE_IMAGE, base.with_suffix(".png")),
        _artifact(f"{label} SVG", ExperimentArtifact.TYPE_IMAGE, base.with_suffix(".svg")),
    ]


def _method_key_from_slug(method_slug):
    return {
        "sggr": "sg-gr",
        "sggc": "sg-gc",
    }.get(method_slug, method_slug)


def _official_config(language, *, location_key, dataset, experiment_type, artifacts):
    return {
        text("official.config_source", language): text("official.source_archive", language),
        text("official.config_location", language): text(location_key, language),
        "dataset": dataset,
        text("official.config_result_type", language): experiment_type_label(experiment_type, language),
        text("official.config_artifacts", language): [artifact["relative_path"] for artifact in artifacts],
    }


def _method_comparison_entries(language):
    entries = []
    for full_csv in sorted(RESULTS_DIR.glob("*_full_pipeline_results.csv")):
        dataset_slug = full_csv.name[: -len("_full_pipeline_results.csv")]
        dataset = _dataset_label(dataset_slug)
        experiment_type = ExperimentRun.TYPE_METHOD_COMPARISON

        artifacts = [
            _artifact("Unified Results CSV", ExperimentArtifact.TYPE_CSV, full_csv),
            _artifact(
                "SG-GR Grid Search",
                ExperimentArtifact.TYPE_CSV,
                RESULTS_DIR / f"grid_search_sggr_{dataset_slug}_results.csv",
            ),
            _artifact(
                "GCA Grid Search",
                ExperimentArtifact.TYPE_CSV,
                RESULTS_DIR / f"grid_search_gca_{dataset_slug}_results.csv",
            ),
            _artifact(
                "SG-GC Grid Search",
                ExperimentArtifact.TYPE_CSV,
                RESULTS_DIR / f"grid_search_sggc_{dataset_slug}_results.csv",
            ),
        ]
        artifacts = [artifact for artifact in artifacts if artifact]

        summary = build_method_comparison_summary(full_csv)
        summary.update({"dataset": dataset, "main_csv": _relative_path(full_csv)})

        type_label = experiment_type_label(experiment_type, language)
        entries.append(
            {
                "slug": f"method-comparison-{dataset_slug}",
                "title": f"{type_label} / {dataset}",
                "experiment_type": experiment_type,
                "type_label": type_label,
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "updated_at": _updated_at([(BASE_DIR / artifact["relative_path"]).resolve() for artifact in artifacts]),
                "summary": summary,
                "config": _official_config(
                    language,
                    location_key="official.location.results",
                    dataset=dataset,
                    experiment_type=experiment_type,
                    artifacts=artifacts,
                ),
                "artifacts": artifacts,
            }
        )

    return entries


def _sampling_bias_entries(language):
    entries = []
    for csv_path in sorted(LOGS_DIR.glob("exp1_*.csv")):
        dataset_slug = csv_path.stem[len("exp1_") :]
        dataset = _dataset_label(dataset_slug)
        plot_path = LOGS_DIR / f"exp1_{dataset_slug}_curves.png"
        experiment_type = ExperimentRun.TYPE_SAMPLING_BIAS

        artifacts = [
            _artifact("Sampling Bias CSV", ExperimentArtifact.TYPE_CSV, csv_path),
            _artifact("Sampling Bias Curve", ExperimentArtifact.TYPE_IMAGE, plot_path),
            _artifact("Sampling Bias Curve SVG", ExperimentArtifact.TYPE_IMAGE, plot_path.with_suffix(".svg")),
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

        type_label = experiment_type_label(experiment_type, language)
        entries.append(
            {
                "slug": f"sampling-bias-{dataset_slug}",
                "title": f"{type_label} / {dataset}",
                "experiment_type": experiment_type,
                "type_label": type_label,
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "updated_at": _updated_at([csv_path, plot_path]),
                "summary": summary,
                "config": _official_config(
                    language,
                    location_key="official.location.logs",
                    dataset=dataset,
                    experiment_type=experiment_type,
                    artifacts=artifacts,
                ),
                "artifacts": artifacts,
            }
        )

    return entries


def _sensitivity_entries(language):
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
        plot_path = PLOTS_DIR / f"{dataset_slug}_sg_sensitivity_overview.png"
        report_path = PLOTS_DIR / f"{dataset_slug}_sg_sensitivity_analysis.md"
        experiment_type = ExperimentRun.TYPE_SENSITIVITY

        artifacts = []
        for csv_path in csv_paths:
            method_slug = csv_path.stem.split("_")[1]
            method_key = _method_key_from_slug(method_slug)
            method_label = METHOD_LABELS.get(method_key, method_slug.upper())
            artifacts.append(_artifact(f"{method_label} Sensitivity CSV", ExperimentArtifact.TYPE_CSV, csv_path))
        artifacts.extend(
            [
            _artifact("Sensitivity Overview Plot", ExperimentArtifact.TYPE_IMAGE, plot_path),
            _artifact("Sensitivity Overview Plot SVG", ExperimentArtifact.TYPE_IMAGE, plot_path.with_suffix(".svg")),
            _artifact("Sensitivity Analysis Report", ExperimentArtifact.TYPE_REPORT, report_path),
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

        type_label = experiment_type_label(experiment_type, language)
        entries.append(
            {
                "slug": f"sensitivity-{dataset_slug}",
                "title": f"{type_label} / {dataset}",
                "experiment_type": experiment_type,
                "type_label": type_label,
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "updated_at": _updated_at([*csv_paths, plot_path, report_path]),
                "summary": summary,
                "config": _official_config(
                    language,
                    location_key="official.location.sensitivity",
                    dataset=dataset,
                    experiment_type=experiment_type,
                    artifacts=artifacts,
                ),
                "artifacts": artifacts,
            }
        )

    entries.extend(_sensitivity_combined_entries(language, sorted(RESULTS_DIR.glob("sensitivity_*_results.csv"))))
    return entries


def _sensitivity_combined_entries(language, csv_paths):
    if not csv_paths:
        return []

    experiment_type = ExperimentRun.TYPE_SENSITIVITY
    plot_specs = [
        ("Sensitivity t_s Effect Plot", PLOTS_DIR / "sg_sensitivity_ts_effect"),
        ("Sensitivity M Effect Plot", PLOTS_DIR / "sg_sensitivity_M_effect"),
        ("Sensitivity K Effect Plot", PLOTS_DIR / "sg_sensitivity_K_effect"),
    ]
    artifacts = []
    for label, base_path in plot_specs:
        artifacts.extend(_image_pair_artifacts(label, base_path))
    artifacts = [artifact for artifact in artifacts if artifact]
    if not artifacts:
        return []

    summary = build_sensitivity_summary(csv_paths)
    summary.update({"dataset": _all_dataset_label(language), "csv_count": len(csv_paths)})

    type_label = experiment_type_label(experiment_type, language)
    return [
        {
            "slug": "sensitivity-all",
            "title": f"{type_label} / {_all_dataset_label(language)}",
            "experiment_type": experiment_type,
            "type_label": type_label,
            "dataset": _all_dataset_label(language),
            "dataset_slug": "all",
            "updated_at": _updated_at([(BASE_DIR / artifact["relative_path"]).resolve() for artifact in artifacts]),
            "summary": summary,
            "config": _official_config(
                language,
                location_key="official.location.sensitivity",
                dataset=_all_dataset_label(language),
                experiment_type=experiment_type,
                artifacts=artifacts,
            ),
            "artifacts": artifacts,
        }
    ]


def _component_ablation_entries(language):
    csv_paths = sorted(RESULTS_DIR.glob("extra_ablation_*_results.csv"))
    if not csv_paths:
        return []

    experiment_type = ExperimentRun.TYPE_COMPONENT_ABLATION
    artifacts = [_artifact("Component Ablation CSV", ExperimentArtifact.TYPE_CSV, path) for path in csv_paths]
    artifacts.extend(
        [
            _artifact("Component Ablation Analysis Report", ExperimentArtifact.TYPE_REPORT, PLOTS_DIR / "extra_ablation_analysis.md"),
        ]
    )
    artifacts.extend(_image_pair_artifacts("Component Ablation M Effect Plot", PLOTS_DIR / "extra_ablation_warmup_M_effect"))
    artifacts.extend(_image_pair_artifacts("Component Ablation K Effect Plot", PLOTS_DIR / "extra_ablation_update_K_effect"))
    artifacts.extend(_image_pair_artifacts("Component Ablation w Effect Plot", PLOTS_DIR / "extra_ablation_weight_w_effect"))
    artifacts = [artifact for artifact in artifacts if artifact]
    report_path = PLOTS_DIR / "extra_ablation_analysis.md"
    summary = build_component_ablation_summary(csv_paths, report_path)
    summary.update({"dataset": _all_dataset_label(language), "csv_count": len(csv_paths)})

    type_label = experiment_type_label(experiment_type, language)
    return [
        {
            "slug": "component-ablation-all",
            "title": f"{type_label} / {_all_dataset_label(language)}",
            "experiment_type": experiment_type,
            "type_label": type_label,
            "dataset": _all_dataset_label(language),
            "dataset_slug": "all",
            "updated_at": _updated_at([(BASE_DIR / artifact["relative_path"]).resolve() for artifact in artifacts]),
            "summary": summary,
            "config": _official_config(
                language,
                location_key="official.location.extra",
                dataset=_all_dataset_label(language),
                experiment_type=experiment_type,
                artifacts=artifacts,
            ),
            "artifacts": artifacts,
        }
    ]


def _efficiency_entries(language):
    csv_paths = sorted(RESULTS_DIR.glob("efficiency_*_results.csv"))
    if not csv_paths:
        return []

    experiment_type = ExperimentRun.TYPE_EFFICIENCY
    artifacts = [_artifact("Efficiency CSV", ExperimentArtifact.TYPE_CSV, path) for path in csv_paths]
    artifacts.extend(
        [
            _artifact("Efficiency Analysis Report", ExperimentArtifact.TYPE_REPORT, PLOTS_DIR / "efficiency_analysis.md"),
        ]
    )
    artifacts.extend(_image_pair_artifacts("Efficiency Train Total Time Plot", PLOTS_DIR / "efficiency_train_total_time"))
    artifacts.extend(_image_pair_artifacts("Efficiency Wall Time Plot", PLOTS_DIR / "efficiency_wall_time"))
    artifacts = [artifact for artifact in artifacts if artifact]
    report_path = PLOTS_DIR / "efficiency_analysis.md"
    summary = build_efficiency_summary(csv_paths, report_path)
    summary.update({"dataset": _all_dataset_label(language), "csv_count": len(csv_paths)})

    type_label = experiment_type_label(experiment_type, language)
    return [
        {
            "slug": "efficiency-all",
            "title": f"{type_label} / {_all_dataset_label(language)}",
            "experiment_type": experiment_type,
            "type_label": type_label,
            "dataset": _all_dataset_label(language),
            "dataset_slug": "all",
            "updated_at": _updated_at([(BASE_DIR / artifact["relative_path"]).resolve() for artifact in artifacts]),
            "summary": summary,
            "config": _official_config(
                language,
                location_key="official.location.extra",
                dataset=_all_dataset_label(language),
                experiment_type=experiment_type,
                artifacts=artifacts,
            ),
            "artifacts": artifacts,
        }
    ]


def _significance_entries(language):
    csv_paths = sorted(RESULTS_DIR.glob("significance_*_results.csv"))
    if not csv_paths:
        return []

    experiment_type = ExperimentRun.TYPE_SIGNIFICANCE
    summary_csv = PLOTS_DIR / "significance_tests_summary.csv"
    report_path = PLOTS_DIR / "significance_analysis.md"
    artifacts = [_artifact("Significance CSV", ExperimentArtifact.TYPE_CSV, path) for path in csv_paths]
    artifacts.extend(
        [
            _artifact("Significance Tests Summary CSV", ExperimentArtifact.TYPE_CSV, summary_csv),
            _artifact("Significance Analysis Report", ExperimentArtifact.TYPE_REPORT, report_path),
        ]
    )
    artifacts.extend(_image_pair_artifacts("Significance Mean/Std Plot", PLOTS_DIR / "significance_mean_std"))
    artifacts.extend(_image_pair_artifacts("Significance Paired Delta Plot", PLOTS_DIR / "significance_paired_delta"))
    artifacts = [artifact for artifact in artifacts if artifact]
    summary = build_significance_summary(csv_paths, summary_csv if summary_csv.exists() else None, report_path)
    summary.update({"dataset": _all_dataset_label(language), "csv_count": len(csv_paths)})

    type_label = experiment_type_label(experiment_type, language)
    return [
        {
            "slug": "significance-all",
            "title": f"{type_label} / {_all_dataset_label(language)}",
            "experiment_type": experiment_type,
            "type_label": type_label,
            "dataset": _all_dataset_label(language),
            "dataset_slug": "all",
            "updated_at": _updated_at([(BASE_DIR / artifact["relative_path"]).resolve() for artifact in artifacts]),
            "summary": summary,
            "config": _official_config(
                language,
                location_key="official.location.extra",
                dataset=_all_dataset_label(language),
                experiment_type=experiment_type,
                artifacts=artifacts,
            ),
            "artifacts": artifacts,
        }
    ]


def list_official_results(language="zh"):
    entries = [
        *_sampling_bias_entries(language),
        *_method_comparison_entries(language),
        *_component_ablation_entries(language),
        *_efficiency_entries(language),
        *_significance_entries(language),
        *_sensitivity_entries(language),
    ]
    return sorted(
        entries,
        key=lambda entry: (
            TYPE_ORDER.get(entry["experiment_type"], 99),
            DATASET_ORDER.get(entry["dataset_slug"], 99),
            entry["title"],
        ),
    )


def get_official_result(slug, language="zh"):
    for entry in list_official_results(language):
        if entry["slug"] == slug:
            return entry
    return None
