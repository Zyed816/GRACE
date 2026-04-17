from django.db import models


class Experiment(models.Model):
    TASK_TRAIN = "train"
    TASK_GRID_SEARCH = "grid_search"
    TASK_TOP_VERIFY = "top_verify"
    TASK_FULL_PIPELINE_SINGLE = "full_pipeline_single"
    TASK_FULL_PIPELINE_MULTI = "full_pipeline_multi"

    TASK_TYPE_CHOICES = [
        (TASK_TRAIN, "训练任务"),
        (TASK_GRID_SEARCH, "参数搜索"),
        (TASK_TOP_VERIFY, "最优参数验证"),
        (TASK_FULL_PIPELINE_SINGLE, "单数据集全流程"),
        (TASK_FULL_PIPELINE_MULTI, "多数据集全流程"),
    ]

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_SUCCEEDED = "succeeded"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    STATUS_CHOICES = [
        (STATUS_PENDING, "等待中"),
        (STATUS_RUNNING, "运行中"),
        (STATUS_SUCCEEDED, "已完成"),
        (STATUS_FAILED, "失败"),
        (STATUS_CANCELLED, "已取消"),
    ]

    task_type = models.CharField(max_length=32, choices=TASK_TYPE_CHOICES, default=TASK_TRAIN)
    task_params = models.JSONField(default=dict, blank=True)
    extra_cli_args = models.JSONField(default=list, blank=True)
    cancel_requested = models.BooleanField(default=False)
    artifacts = models.JSONField(default=dict, blank=True)
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
        return f"{self.task_type} / {self.dataset} / {self.model_name} / {self.status}"


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


class PipelineResult(models.Model):
    STAGE_BASELINE = "baseline"
    STAGE_TOP_VERIFY = "top_verify"
    STAGE_SUMMARY = "summary"

    STAGE_CHOICES = [
        (STAGE_BASELINE, "基线"),
        (STAGE_TOP_VERIFY, "最优参数验证"),
        (STAGE_SUMMARY, "汇总"),
    ]

    dataset = models.CharField(max_length=32)
    method_key = models.CharField(max_length=32)
    method_name = models.CharField(max_length=64, blank=True)
    stage = models.CharField(max_length=32, choices=STAGE_CHOICES, default=STAGE_SUMMARY)
    candidate_rank = models.PositiveIntegerField(null=True, blank=True)
    run_idx = models.PositiveIntegerField(null=True, blank=True)
    F1Mi_mean = models.FloatField(null=True, blank=True)
    F1Mi_std = models.FloatField(null=True, blank=True)
    F1Ma_mean = models.FloatField(null=True, blank=True)
    F1Ma_std = models.FloatField(null=True, blank=True)
    robust_score = models.FloatField(null=True, blank=True)
    delta_vs_grace = models.FloatField(null=True, blank=True)
    params_json = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)
    source_csv = models.CharField(max_length=512, blank=True)
    created_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["dataset", "method_key", "stage", "candidate_rank", "run_idx"]
        indexes = [
            models.Index(fields=["dataset", "method_key", "stage"]),
        ]

    def __str__(self):
        return f"{self.dataset} / {self.method_key} / {self.stage}"
