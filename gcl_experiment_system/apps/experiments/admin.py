from django.contrib import admin

from .models import Experiment, ExperimentLog, ExperimentMetric


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "model_name", "status", "final_accuracy", "created_time")
    list_filter = ("dataset", "model_name", "status")
    search_fields = ("dataset", "model_name")


@admin.register(ExperimentLog)
class ExperimentLogAdmin(admin.ModelAdmin):
    list_display = ("experiment", "epoch", "loss", "accuracy", "created_time")


@admin.register(ExperimentMetric)
class ExperimentMetricAdmin(admin.ModelAdmin):
    list_display = ("experiment", "name", "value", "step")
