from django import forms

from .constants import (
    DATASET_CHOICES,
    METHOD_CHOICES,
    SENSITIVITY_METHOD_CHOICES,
    SG_METHOD_CHOICES,
    SIGNIFICANCE_COMPARISON_CHOICES,
)
from .ui_text import DEFAULT_UI_LANGUAGE, default_run_name, sensitivity_param_choices, text


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

    def _apply_localized_text(self):
        return None

    def __init__(self, *args, **kwargs):
        self.ui_lang = kwargs.pop("lang", DEFAULT_UI_LANGUAGE)
        super().__init__(*args, **kwargs)
        self._apply_localized_text()
        self._apply_styles()


class MethodComparisonForm(StyledFormMixin, forms.Form):
    name = forms.CharField(label="Run Name", max_length=120, required=False, initial="Method Comparison Pipeline")
    dataset = forms.ChoiceField(label="Dataset", choices=DATASET_CHOICES, initial="Cora")
    gpu_id = forms.IntegerField(label="gpu_id", min_value=0, max_value=7, initial=0)
    std_weight = forms.FloatField(label="std_weight", min_value=0.0, initial=0.5)
    baseline_runs = forms.IntegerField(label="baseline_runs", min_value=1, initial=3)
    topk_verify = forms.IntegerField(label="topk_verify", min_value=1, initial=3)
    runs_per_top = forms.IntegerField(label="runs_per_top", min_value=1, initial=3)
    force_grid = forms.BooleanField(label="force_grid", required=False)

    def _apply_localized_text(self):
        self.fields["name"].label = text("forms.run_name", self.ui_lang)
        self.fields["name"].initial = default_run_name("method_comparison", self.ui_lang)
        self.fields["dataset"].label = text("forms.dataset", self.ui_lang)
        self.fields["force_grid"].label = text("forms.force_grid", self.ui_lang)


class SamplingBiasForm(StyledFormMixin, forms.Form):
    name = forms.CharField(label="Run Name", max_length=120, required=False, initial="Sampling Bias Validation")
    dataset = forms.ChoiceField(label="Dataset", choices=DATASET_CHOICES, initial="Cora")
    method = forms.ChoiceField(label="Method", choices=METHOD_CHOICES, initial="grace")
    gpu_id = forms.IntegerField(label="gpu_id", min_value=0, max_value=7, initial=0)
    title = forms.CharField(
        label="Plot Title",
        max_length=160,
        required=False,
        initial="Experiment 1: Violation Rate and Margin",
    )

    def _apply_localized_text(self):
        self.fields["name"].label = text("forms.run_name", self.ui_lang)
        self.fields["name"].initial = default_run_name("sampling_bias", self.ui_lang)
        self.fields["dataset"].label = text("forms.dataset", self.ui_lang)
        self.fields["method"].label = text("forms.method", self.ui_lang)
        self.fields["title"].label = text("forms.chart_title", self.ui_lang)
        self.fields["title"].initial = "Experiment 1: Violation Rate and Margin"


class SensitivityForm(StyledFormMixin, forms.Form):
    name = forms.CharField(label="Run Name", max_length=120, required=False, initial="Sensitivity Analysis")
    dataset = forms.ChoiceField(label="Dataset", choices=DATASET_CHOICES, initial="Cora")
    methods = forms.MultipleChoiceField(
        label="Methods",
        choices=SENSITIVITY_METHOD_CHOICES,
        initial=["ifl-gr", "ifl-gc"],
        widget=forms.CheckboxSelectMultiple(attrs={"class": CHECKBOX_CLASS}),
    )
    params = forms.MultipleChoiceField(
        label="Paper Params",
        choices=sensitivity_param_choices(DEFAULT_UI_LANGUAGE),
        initial=["t_s", "M", "K"],
        widget=forms.CheckboxSelectMultiple(attrs={"class": CHECKBOX_CLASS}),
    )
    gpu_id = forms.IntegerField(label="gpu_id", min_value=0, max_value=7, initial=0)
    base_rank = forms.IntegerField(label="base_rank", min_value=1, initial=1)
    runs = forms.IntegerField(label="runs", min_value=1, initial=3)
    std_weight = forms.FloatField(label="std_weight", min_value=0.0, initial=0.5)
    neighbor_span = forms.IntegerField(label="neighbor_span", min_value=0, initial=1)
    continue_on_error = forms.BooleanField(label="continue_on_error", required=False)

    def _apply_localized_text(self):
        self.fields["name"].label = text("forms.run_name", self.ui_lang)
        self.fields["name"].initial = default_run_name("sensitivity", self.ui_lang)
        self.fields["dataset"].label = text("forms.dataset", self.ui_lang)
        self.fields["methods"].label = text("forms.methods", self.ui_lang)
        self.fields["params"].label = text("forms.paper_params", self.ui_lang)
        self.fields["params"].choices = sensitivity_param_choices(self.ui_lang)
        self.fields["continue_on_error"].label = text("forms.continue_on_error", self.ui_lang)


class ComponentAblationForm(StyledFormMixin, forms.Form):
    name = forms.CharField(label="Run Name", max_length=120, required=False, initial="Component Ablation")
    dataset = forms.ChoiceField(label="Dataset", choices=DATASET_CHOICES, initial="Cora")
    methods = forms.MultipleChoiceField(
        label="Methods",
        choices=SG_METHOD_CHOICES,
        initial=["ifl-gr", "ifl-gc"],
        widget=forms.CheckboxSelectMultiple(attrs={"class": CHECKBOX_CLASS}),
    )
    gpu_id = forms.IntegerField(label="gpu_id", min_value=0, max_value=7, initial=0)
    runs = forms.IntegerField(label="runs", min_value=1, initial=3)
    std_weight = forms.FloatField(label="std_weight", min_value=0.0, initial=0.5)
    continue_on_error = forms.BooleanField(label="continue_on_error", required=False)

    def _apply_localized_text(self):
        self.fields["name"].label = text("forms.run_name", self.ui_lang)
        self.fields["name"].initial = default_run_name("component_ablation", self.ui_lang)
        self.fields["dataset"].label = text("forms.dataset", self.ui_lang)
        self.fields["methods"].label = text("forms.methods", self.ui_lang)
        self.fields["continue_on_error"].label = text("forms.continue_on_error", self.ui_lang)


class EfficiencyForm(StyledFormMixin, forms.Form):
    name = forms.CharField(label="Run Name", max_length=120, required=False, initial="Efficiency Experiment")
    dataset = forms.ChoiceField(label="Dataset", choices=DATASET_CHOICES, initial="Cora")
    methods = forms.MultipleChoiceField(
        label="Methods",
        choices=METHOD_CHOICES,
        initial=["grace", "gca", "ifl-gr", "ifl-gc"],
        widget=forms.CheckboxSelectMultiple(attrs={"class": CHECKBOX_CLASS}),
    )
    gpu_id = forms.IntegerField(label="gpu_id", min_value=0, max_value=7, initial=0)
    runs = forms.IntegerField(label="runs", min_value=1, initial=3)
    std_weight = forms.FloatField(label="std_weight", min_value=0.0, initial=0.5)
    continue_on_error = forms.BooleanField(label="continue_on_error", required=False)

    def _apply_localized_text(self):
        self.fields["name"].label = text("forms.run_name", self.ui_lang)
        self.fields["name"].initial = default_run_name("efficiency", self.ui_lang)
        self.fields["dataset"].label = text("forms.dataset", self.ui_lang)
        self.fields["methods"].label = text("forms.methods", self.ui_lang)
        self.fields["continue_on_error"].label = text("forms.continue_on_error", self.ui_lang)


class SignificanceForm(StyledFormMixin, forms.Form):
    name = forms.CharField(label="Run Name", max_length=120, required=False, initial="Statistical Significance")
    dataset = forms.ChoiceField(label="Dataset", choices=DATASET_CHOICES, initial="Cora")
    comparison_pairs = forms.MultipleChoiceField(
        label="Comparison Pairs",
        choices=SIGNIFICANCE_COMPARISON_CHOICES,
        initial=["sg_gr_vs_grace", "sg_gc_vs_gca"],
        widget=forms.CheckboxSelectMultiple(attrs={"class": CHECKBOX_CLASS}),
    )
    gpu_id = forms.IntegerField(label="gpu_id", min_value=0, max_value=7, initial=0)
    runs = forms.IntegerField(label="runs", min_value=2, initial=10)
    eval_repeats = forms.IntegerField(label="eval_repeats", min_value=1, initial=3)
    std_weight = forms.FloatField(label="std_weight", min_value=0.0, initial=0.5)
    alpha = forms.FloatField(label="alpha", min_value=0.0, max_value=1.0, initial=0.05)
    continue_on_error = forms.BooleanField(label="continue_on_error", required=False)

    def _apply_localized_text(self):
        self.fields["name"].label = text("forms.run_name", self.ui_lang)
        self.fields["name"].initial = default_run_name("significance", self.ui_lang)
        self.fields["dataset"].label = text("forms.dataset", self.ui_lang)
        self.fields["comparison_pairs"].label = text("forms.comparison_pairs", self.ui_lang)
        self.fields["continue_on_error"].label = text("forms.continue_on_error", self.ui_lang)
