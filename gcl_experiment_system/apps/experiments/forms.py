from django import forms
from django.core.exceptions import ValidationError

from apps.models.catalog import METHOD_CATALOG

from .models import Experiment


DATASET_CHOICES = [(name, name) for name in ["Cora", "CiteSeer", "PubMed", "DBLP"]]
METHOD_CHOICES = [(item["key"], item["display_name"]) for item in METHOD_CATALOG.values()]


class ExperimentForm(forms.ModelForm):
    run_now = forms.BooleanField(required=False, initial=True, label="创建后立即运行")
    extra_params_text = forms.CharField(
        required=False,
        label="高级参数（JSON）",
        widget=forms.Textarea(attrs={"rows": 8, "placeholder": '{"warmup_epochs": 100, "update_interval": 10}'}),
        help_text="可选。填写 JSON 字典覆盖 config.yaml 中对应数据集参数。",
    )

    class Meta:
        model = Experiment
        fields = [
            "dataset",
            "model_name",
            "learning_rate",
            "hidden_dim",
            "epochs",
            "temperature",
            "drop_edge_rate",
            "drop_feature_rate",
        ]
        widgets = {
            "dataset": forms.Select(choices=DATASET_CHOICES),
            "model_name": forms.Select(choices=METHOD_CHOICES),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            cls = "form-check-input" if isinstance(field.widget, forms.CheckboxInput) else "form-control"
            existing = field.widget.attrs.get("class", "")
            field.widget.attrs["class"] = f"{existing} {cls}".strip()

    def clean_extra_params_text(self):
        raw = self.cleaned_data.get("extra_params_text", "").strip()
        if not raw:
            return {}
        try:
            import json

            value = json.loads(raw)
        except Exception as exc:
            raise ValidationError(f"JSON 解析失败: {exc}")
        if not isinstance(value, dict):
            raise ValidationError("高级参数必须是 JSON 对象（字典）。")
        return value
