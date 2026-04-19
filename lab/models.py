from pathlib import Path

from django.db import models
from django.urls import reverse


class ExperimentRun(models.Model):
    TYPE_METHOD_COMPARISON = "method_comparison"
    TYPE_SAMPLING_BIAS = "sampling_bias"
    TYPE_SENSITIVITY = "sensitivity"

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_SUCCEEDED = "succeeded"
    STATUS_FAILED = "failed"
    STATUS_ABORTED = "aborted"

    EXPERIMENT_TYPE_CHOICES = [
        (TYPE_METHOD_COMPARISON, "Method Comparison"),
        (TYPE_SAMPLING_BIAS, "Sampling Bias"),
        (TYPE_SENSITIVITY, "Sensitivity Analysis"),
    ]

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_SUCCEEDED, "Succeeded"),
        (STATUS_FAILED, "Failed"),
        (STATUS_ABORTED, "Aborted"),
    ]

    name = models.CharField("Run Name", max_length=120, blank=True)
    experiment_type = models.CharField("Experiment Type", max_length=32, choices=EXPERIMENT_TYPE_CHOICES)
    dataset = models.CharField("Dataset", max_length=32, blank=True)
    status = models.CharField("Status", max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    config = models.JSONField("Config", default=dict, blank=True)
    command = models.TextField("Command", blank=True)
    stdout_log = models.TextField("Stdout Log", blank=True)
    error_message = models.TextField("Error Message", blank=True)
    result_summary = models.JSONField("Result Summary", default=dict, blank=True)
    worker_pid = models.IntegerField("Worker PID", null=True, blank=True)
    created_at = models.DateTimeField("Created At", auto_now_add=True)
    started_at = models.DateTimeField("Started At", null=True, blank=True)
    finished_at = models.DateTimeField("Finished At", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Experiment Run"
        verbose_name_plural = "Experiment Runs"

    def __str__(self):
        return self.display_name

    @property
    def display_name(self):
        if self.name:
            return self.name
        dataset = f" / {self.dataset}" if self.dataset else ""
        return f"{self.get_experiment_type_display()}{dataset}"

    def get_absolute_url(self):
        return reverse("lab:run_detail", kwargs={"pk": self.pk})

    @property
    def is_active(self):
        return self.status in {self.STATUS_PENDING, self.STATUS_RUNNING}

    @property
    def can_stop(self):
        return self.is_active and bool(self.worker_pid)

    @property
    def can_delete(self):
        return not self.is_active


class ExperimentArtifact(models.Model):
    TYPE_CSV = "csv"
    TYPE_IMAGE = "image"
    TYPE_REPORT = "report"
    TYPE_OTHER = "other"

    ARTIFACT_TYPE_CHOICES = [
        (TYPE_CSV, "CSV"),
        (TYPE_IMAGE, "Image"),
        (TYPE_REPORT, "Report"),
        (TYPE_OTHER, "Other"),
    ]

    run = models.ForeignKey(ExperimentRun, related_name="artifacts", on_delete=models.CASCADE)
    label = models.CharField("Label", max_length=120)
    artifact_type = models.CharField("Artifact Type", max_length=16, choices=ARTIFACT_TYPE_CHOICES, default=TYPE_OTHER)
    relative_path = models.CharField("Relative Path", max_length=255)
    metadata = models.JSONField("Metadata", default=dict, blank=True)
    created_at = models.DateTimeField("Created At", auto_now_add=True)

    class Meta:
        ordering = ["artifact_type", "id"]
        verbose_name = "Experiment Artifact"
        verbose_name_plural = "Experiment Artifacts"

    def __str__(self):
        return f"{self.label} ({self.relative_path})"

    @property
    def filename(self):
        return Path(self.relative_path).name
