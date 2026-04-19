from django import forms

from .constants import (
    DATASET_CHOICES,
    METHOD_CHOICES,
    SENSITIVITY_METHOD_CHOICES,
    SENSITIVITY_PARAM_CHOICES,
)


INPUT_CLASS = "form-input"
CHECKBOX_CLASS = "checkbox-grid"


class StyledFormMixin:
    def _apply_styles(self):
        for field in self.fields.values():
            widget = field.widget
            if isinstance(widget, (forms.CheckboxInput, forms.CheckboxSelectMultiple)):
                continue
            css = widget.attrs.get("class", "")
            widget.attrs["class"] = f"{css} {INPUT_CLASS}".strip()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_styles()


class MethodComparisonForm(StyledFormMixin, forms.Form):
    name = forms.CharField(label="Run Name", max_length=120, required=False, initial="Method Comparison Pipeline")
    dataset = forms.ChoiceField(label="Dataset", choices=DATASET_CHOICES, initial="Cora")
    gpu_id = forms.IntegerField(label="GPU ID", min_value=0, max_value=7, initial=0)
    std_weight = forms.FloatField(label="Std Weight", min_value=0.0, initial=0.5)
    baseline_runs = forms.IntegerField(label="Baseline Runs", min_value=1, initial=3)
    topk_verify = forms.IntegerField(label="Top-K Verify", min_value=1, initial=3)
    runs_per_top = forms.IntegerField(label="Runs Per Candidate", min_value=1, initial=3)
    force_grid = forms.BooleanField(label="Force Grid Search", required=False)


class SamplingBiasForm(StyledFormMixin, forms.Form):
    name = forms.CharField(label="Run Name", max_length=120, required=False, initial="Sampling Bias Validation")
    dataset = forms.ChoiceField(label="Dataset", choices=DATASET_CHOICES, initial="Cora")
    method = forms.ChoiceField(label="Method", choices=METHOD_CHOICES, initial="grace")
    gpu_id = forms.IntegerField(label="GPU ID", min_value=0, max_value=7, initial=0)
    title = forms.CharField(
        label="Plot Title",
        max_length=160,
        required=False,
        initial="Experiment 1: Violation Rate and Margin",
    )


class SensitivityForm(StyledFormMixin, forms.Form):
    name = forms.CharField(label="Run Name", max_length=120, required=False, initial="Hyperparameter Sensitivity")
    dataset = forms.ChoiceField(label="Dataset", choices=DATASET_CHOICES, initial="Cora")
    methods = forms.MultipleChoiceField(
        label="Methods",
        choices=SENSITIVITY_METHOD_CHOICES,
        initial=["ifl-gr", "ifl-gc"],
        widget=forms.CheckboxSelectMultiple(attrs={"class": CHECKBOX_CLASS}),
    )
    params = forms.MultipleChoiceField(
        label="Paper Params",
        choices=SENSITIVITY_PARAM_CHOICES,
        initial=["t_s", "M", "K"],
        widget=forms.CheckboxSelectMultiple(attrs={"class": CHECKBOX_CLASS}),
    )
    gpu_id = forms.IntegerField(label="GPU ID", min_value=0, max_value=7, initial=0)
    base_rank = forms.IntegerField(label="Base Rank", min_value=1, initial=1)
    runs = forms.IntegerField(label="Runs", min_value=1, initial=3)
    std_weight = forms.FloatField(label="Std Weight", min_value=0.0, initial=0.5)
    neighbor_span = forms.IntegerField(label="Neighbor Span", min_value=0, initial=1)
    continue_on_error = forms.BooleanField(label="Continue On Error", required=False)
