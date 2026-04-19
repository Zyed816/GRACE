from django.contrib import admin

from .models import ExperimentArtifact, ExperimentRun


@admin.register(ExperimentRun)
class ExperimentRunAdmin(admin.ModelAdmin):
    list_display = ("id", "display_name", "experiment_type", "dataset", "status", "created_at")
    list_filter = ("experiment_type", "status", "dataset")
    search_fields = ("name", "dataset", "stdout_log", "error_message")
    readonly_fields = ("created_at", "started_at", "finished_at")


@admin.register(ExperimentArtifact)
class ExperimentArtifactAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "label", "artifact_type", "relative_path", "created_at")
    list_filter = ("artifact_type",)
    search_fields = ("label", "relative_path")
