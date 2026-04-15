from django.db import models


class Experiment(models.Model):
    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_SUCCEEDED = "succeeded"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_SUCCEEDED, "Succeeded"),
        (STATUS_FAILED, "Failed"),
        (STATUS_CANCELLED, "Cancelled"),
    ]

    dataset = models.CharField(max_length=32)
    model_name = models.CharField(max_length=32)
    learning_rate = models.FloatField(default=0.01)
    hidden_dim = models.PositiveIntegerField(default=256)
    epochs = models.PositiveIntegerField(default=200)
    temperature = models.FloatField(default=0.5)
    drop_edge_rate = models.FloatField(default=0.2)
    drop_feature_rate = models.FloatField(default=0.2)
    extra_params = models.JSONField(default=dict, blank=True)
    final_accuracy = models.FloatField(null=True, blank=True)
    final_f1mi = models.FloatField(null=True, blank=True)
    final_f1ma = models.FloatField(null=True, blank=True)
    run_seconds = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    stdout_path = models.CharField(max_length=512, blank=True)
    exp1_log_path = models.CharField(max_length=512, blank=True)
    created_time = models.DateTimeField(auto_now_add=True)
    started_time = models.DateTimeField(null=True, blank=True)
    finished_time = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_time"]

    def __str__(self):
        return f"{self.dataset} / {self.model_name} / {self.status}"


class ExperimentLog(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name="logs")
    epoch = models.PositiveIntegerField()
    loss = models.FloatField()
    accuracy = models.FloatField(null=True, blank=True)
    payload = models.JSONField(default=dict, blank=True)
    created_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["epoch"]


class ExperimentMetric(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name="metrics")
    name = models.CharField(max_length=64)
    value = models.FloatField()
    step = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["name", "step"]
