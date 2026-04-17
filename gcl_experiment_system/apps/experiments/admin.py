from django.contrib import admin

from .models import Experiment, ExperimentLog, ExperimentMetric, PipelineResult


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ("id", "task_type", "dataset", "model_name", "status", "cancel_requested", "final_accuracy", "created_time")
    list_filter = ("task_type", "dataset", "model_name", "status")
    search_fields = ("dataset", "model_name", "task_type")


@admin.register(ExperimentLog)
class ExperimentLogAdmin(admin.ModelAdmin):
    list_display = ("experiment", "epoch", "loss", "accuracy", "created_time")


@admin.register(ExperimentMetric)
class ExperimentMetricAdmin(admin.ModelAdmin):
    list_display = ("experiment", "name", "value", "step")


@admin.register(PipelineResult)
class PipelineResultAdmin(admin.ModelAdmin):
    list_display = ("dataset", "method_key", "stage", "candidate_rank", "run_idx", "F1Mi_mean", "robust_score")
    list_filter = ("dataset", "method_key", "stage")
    search_fields = ("dataset", "method_key", "method_name", "source_csv")
