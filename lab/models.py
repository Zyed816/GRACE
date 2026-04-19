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
        (TYPE_METHOD_COMPARISON, "方法比较流水线"),
        (TYPE_SAMPLING_BIAS, "采样偏差验证"),
        (TYPE_SENSITIVITY, "超参数敏感性分析"),
    ]

    STATUS_CHOICES = [
        (STATUS_PENDING, "待运行"),
        (STATUS_RUNNING, "运行中"),
        (STATUS_SUCCEEDED, "已完成"),
        (STATUS_FAILED, "失败"),
        (STATUS_ABORTED, "已中止"),
    ]

    name = models.CharField("运行名称", max_length=120, blank=True)
    experiment_type = models.CharField("实验类型", max_length=32, choices=EXPERIMENT_TYPE_CHOICES)
    dataset = models.CharField("数据集", max_length=32, blank=True)
    status = models.CharField("状态", max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    config = models.JSONField("配置", default=dict, blank=True)
    command = models.TextField("命令", blank=True)
    stdout_log = models.TextField("标准输出日志", blank=True)
    error_message = models.TextField("错误信息", blank=True)
    result_summary = models.JSONField("结果摘要", default=dict, blank=True)
    worker_pid = models.IntegerField("Worker PID", null=True, blank=True)
    created_at = models.DateTimeField("创建时间", auto_now_add=True)
    started_at = models.DateTimeField("开始时间", null=True, blank=True)
    finished_at = models.DateTimeField("完成时间", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "实验记录"
        verbose_name_plural = "实验记录"

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
        (TYPE_IMAGE, "图像"),
        (TYPE_REPORT, "报告"),
        (TYPE_OTHER, "其他"),
    ]

    run = models.ForeignKey(ExperimentRun, related_name="artifacts", on_delete=models.CASCADE)
    label = models.CharField("名称", max_length=120)
    artifact_type = models.CharField("产物类型", max_length=16, choices=ARTIFACT_TYPE_CHOICES, default=TYPE_OTHER)
    relative_path = models.CharField("相对路径", max_length=255)
    metadata = models.JSONField("元数据", default=dict, blank=True)
    created_at = models.DateTimeField("创建时间", auto_now_add=True)

    class Meta:
        ordering = ["artifact_type", "id"]
        verbose_name = "实验产物"
        verbose_name_plural = "实验产物"

    def __str__(self):
        return f"{self.label} ({self.relative_path})"

    @property
    def filename(self):
        return Path(self.relative_path).name
