import json

from django import forms
from django.core.exceptions import ValidationError

from apps.datasets.catalog import DATASET_CATALOG
from apps.models.catalog import METHOD_CATALOG

from .models import Experiment


DATASET_CHOICES = [(name, name) for name in DATASET_CATALOG.keys()]
METHOD_CHOICES = [(item["key"], item["display_name"]) for item in METHOD_CATALOG.values()]
SEARCH_METHOD_CHOICES = [("ifl-gr", "IFL-GR"), ("gca", "GCA"), ("ifl-gc", "IFL-GC")]


class ExperimentTaskForm(forms.Form):
    task_type = forms.ChoiceField(
        choices=Experiment.TASK_TYPE_CHOICES,
        initial=Experiment.TASK_TRAIN,
        widget=forms.HiddenInput(),
    )
    run_now = forms.BooleanField(required=False, initial=True, label="提交后立即入队")

    dataset = forms.ChoiceField(choices=DATASET_CHOICES, required=False, initial="Cora", label="数据集")
    model_name = forms.ChoiceField(choices=METHOD_CHOICES, required=False, initial="grace", label="方法")
    learning_rate = forms.FloatField(required=False, initial=0.01, label="学习率")
    hidden_dim = forms.IntegerField(required=False, min_value=1, initial=256, label="隐藏维度")
    epochs = forms.IntegerField(required=False, min_value=1, initial=200, label="训练轮数")
    temperature = forms.FloatField(required=False, min_value=0.0, initial=0.5, label="温度系数")
    drop_edge_rate = forms.FloatField(required=False, min_value=0.0, initial=0.2, label="边丢弃率")
    drop_feature_rate = forms.FloatField(required=False, min_value=0.0, initial=0.2, label="特征丢弃率")
    extra_params_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 5, "placeholder": '{"warmup_epochs": 100, "update_interval": 10}'}),
        help_text="填写 JSON 对象，合并进 config.yaml 的当前数据集配置段。",
        label="高级训练参数（JSON 对象）",
    )

    grid_method = forms.ChoiceField(choices=SEARCH_METHOD_CHOICES, required=False, initial="ifl-gr", label="方法族")
    grid_dataset = forms.ChoiceField(choices=DATASET_CHOICES, required=False, initial="Cora", label="数据集")
    grid_topk = forms.IntegerField(required=False, min_value=1, initial=10, label="Top-K")
    grid_std_weight = forms.FloatField(required=False, min_value=0.0, initial=0.5, label="标准差权重")
    grid_out = forms.CharField(required=False, max_length=512, initial="", label="输出 CSV")

    verify_dataset = forms.ChoiceField(choices=DATASET_CHOICES, required=False, initial="Cora", label="数据集")
    verify_method = forms.ChoiceField(choices=SEARCH_METHOD_CHOICES, required=False, initial="ifl-gr", label="方法")
    verify_top_params = forms.CharField(required=False, max_length=512, initial="", label="最优参数 CSV")
    verify_topk = forms.IntegerField(required=False, min_value=1, initial=3, label="Top-K")
    verify_runs = forms.IntegerField(required=False, min_value=1, initial=3, label="每组参数运行次数")

    pipeline_dataset = forms.ChoiceField(choices=DATASET_CHOICES, required=False, initial="Cora", label="数据集")
    pipeline_baseline_runs = forms.IntegerField(required=False, min_value=1, initial=3, label="基线运行次数")
    pipeline_topk_verify = forms.IntegerField(required=False, min_value=1, initial=3, label="Top-K 验证数")
    pipeline_runs_per_top = forms.IntegerField(required=False, min_value=1, initial=3, label="每个候选运行次数")
    pipeline_force_grid = forms.BooleanField(required=False, initial=False, label="强制重新参数搜索")
    pipeline_out = forms.CharField(required=False, max_length=512, initial="", label="输出 CSV")

    multi_datasets = forms.MultipleChoiceField(
        choices=DATASET_CHOICES,
        required=False,
        initial=["Cora", "CiteSeer", "PubMed", "DBLP"],
        widget=forms.SelectMultiple(),
        label="数据集列表",
    )
    multi_continue_on_error = forms.BooleanField(required=False, initial=False, label="遇错继续")
    multi_baseline_runs = forms.IntegerField(required=False, min_value=1, initial=3, label="基线运行次数")
    multi_topk_verify = forms.IntegerField(required=False, min_value=1, initial=3, label="Top-K 验证数")
    multi_runs_per_top = forms.IntegerField(required=False, min_value=1, initial=3, label="每个候选运行次数")
    multi_force_grid = forms.BooleanField(required=False, initial=False, label="强制重新参数搜索")

    extra_cli_args_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 4, "placeholder": '["--std_weight", "0.6"]'}),
        label="额外 CLI 参数（JSON 数组）",
        help_text="可选透传参数，例如 [\"--force_grid\", \"--std_weight\", \"0.6\"]。",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            if name == "task_type":
                continue
            if isinstance(field.widget, forms.CheckboxInput):
                css_class = "form-check-input"
            elif isinstance(field.widget, forms.SelectMultiple):
                css_class = "form-select"
            else:
                css_class = "form-control"
            existing = field.widget.attrs.get("class", "")
            field.widget.attrs["class"] = f"{existing} {css_class}".strip()

    def clean_extra_params_text(self):
        raw = (self.cleaned_data.get("extra_params_text") or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"JSON 对象格式不正确：{exc}") from exc
        if not isinstance(parsed, dict):
            raise ValidationError("高级训练参数必须是 JSON 对象。")
        return parsed

    def clean_extra_cli_args_text(self):
        raw = (self.cleaned_data.get("extra_cli_args_text") or "").strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"JSON 数组格式不正确：{exc}") from exc
        if not isinstance(parsed, list):
            raise ValidationError("额外 CLI 参数必须是 JSON 数组。")
        normalized = []
        for item in parsed:
            if not isinstance(item, (str, int, float, bool)):
                raise ValidationError("额外 CLI 参数中的每一项都必须是字符串、数字或布尔值。")
            token = str(item).strip()
            if not token:
                continue
            normalized.append(token)
        return normalized

    def clean(self):
        cleaned = super().clean()
        task_type = cleaned.get("task_type")

        if task_type == Experiment.TASK_TRAIN:
            required_fields = [
                "dataset",
                "model_name",
                "learning_rate",
                "hidden_dim",
                "epochs",
                "temperature",
                "drop_edge_rate",
                "drop_feature_rate",
            ]
            for field in required_fields:
                if cleaned.get(field) in (None, ""):
                    self.add_error(field, "训练任务必须填写该字段。")

        elif task_type == Experiment.TASK_TOP_VERIFY:
            if not (cleaned.get("verify_top_params") or "").strip():
                self.add_error("verify_top_params", "必须提供最优参数 CSV 路径。")

        elif task_type == Experiment.TASK_FULL_PIPELINE_MULTI:
            if not cleaned.get("multi_datasets"):
                self.add_error("multi_datasets", "至少选择一个数据集。")

        return cleaned

    def build_experiment_payload(self) -> dict:
        cleaned = self.cleaned_data
        task_type = cleaned["task_type"]

        payload = {
            "task_type": task_type,
            "status": Experiment.STATUS_PENDING,
            "cancel_requested": False,
            "task_params": {},
            "extra_cli_args": cleaned.get("extra_cli_args_text", []),
            "artifacts": {},
            "error_message": "",
            "final_accuracy": None,
            "final_f1mi": None,
            "final_f1ma": None,
            "run_seconds": None,
            "stdout_path": "",
            "exp1_log_path": "",
            "extra_params": {},
        }

        if task_type == Experiment.TASK_TRAIN:
            payload.update({
                "dataset": cleaned["dataset"],
                "model_name": cleaned["model_name"],
                "learning_rate": cleaned["learning_rate"],
                "hidden_dim": cleaned["hidden_dim"],
                "epochs": cleaned["epochs"],
                "temperature": cleaned["temperature"],
                "drop_edge_rate": cleaned["drop_edge_rate"],
                "drop_feature_rate": cleaned["drop_feature_rate"],
                "extra_params": cleaned.get("extra_params_text", {}),
                "task_params": {
                    "dataset": cleaned["dataset"],
                    "method": cleaned["model_name"],
                    "gpu_id": 0,
                    "learning_rate": cleaned["learning_rate"],
                    "hidden_dim": cleaned["hidden_dim"],
                    "epochs": cleaned["epochs"],
                    "temperature": cleaned["temperature"],
                    "drop_edge_rate": cleaned["drop_edge_rate"],
                    "drop_feature_rate": cleaned["drop_feature_rate"],
                },
            })
            return payload

        payload.update({
            "learning_rate": 0.01,
            "hidden_dim": 256,
            "epochs": 200,
            "temperature": 0.5,
            "drop_edge_rate": 0.2,
            "drop_feature_rate": 0.2,
        })

        if task_type == Experiment.TASK_GRID_SEARCH:
            payload.update({
                "dataset": cleaned["grid_dataset"],
                "model_name": cleaned["grid_method"],
                "task_params": {
                    "dataset": cleaned["grid_dataset"],
                    "method": cleaned["grid_method"],
                    "topk": cleaned["grid_topk"],
                    "std_weight": cleaned["grid_std_weight"],
                    "out": (cleaned.get("grid_out") or "").strip(),
                    "gpu_id": 0,
                },
            })
            return payload

        if task_type == Experiment.TASK_TOP_VERIFY:
            payload.update({
                "dataset": cleaned["verify_dataset"],
                "model_name": cleaned["verify_method"],
                "task_params": {
                    "dataset": cleaned["verify_dataset"],
                    "method": cleaned["verify_method"],
                    "top_params": (cleaned["verify_top_params"] or "").strip(),
                    "topk": cleaned["verify_topk"],
                    "runs": cleaned["verify_runs"],
                    "gpu_id": 0,
                },
            })
            return payload

        if task_type == Experiment.TASK_FULL_PIPELINE_SINGLE:
            payload.update({
                "dataset": cleaned["pipeline_dataset"],
                "model_name": "pipeline",
                "task_params": {
                    "dataset": cleaned["pipeline_dataset"],
                    "baseline_runs": cleaned["pipeline_baseline_runs"],
                    "topk_verify": cleaned["pipeline_topk_verify"],
                    "runs_per_top": cleaned["pipeline_runs_per_top"],
                    "force_grid": bool(cleaned.get("pipeline_force_grid")),
                    "out": (cleaned.get("pipeline_out") or "").strip(),
                    "gpu_id": 0,
                },
            })
            return payload

        payload.update({
            "dataset": "Multi",
            "model_name": "pipeline-batch",
            "task_params": {
                "datasets": cleaned.get("multi_datasets", []),
                "continue_on_error": bool(cleaned.get("multi_continue_on_error")),
                "baseline_runs": cleaned["multi_baseline_runs"],
                "topk_verify": cleaned["multi_topk_verify"],
                "runs_per_top": cleaned["multi_runs_per_top"],
                "force_grid": bool(cleaned.get("multi_force_grid")),
                "gpu_id": 0,
            },
        })
        return payload
