import re


DEFAULT_UI_LANGUAGE = "zh"
LANGUAGE_SESSION_KEY = "lab_ui_language"
SUPPORTED_UI_LANGUAGES = ("zh", "en")

EXPERIMENT_TYPE_TEXT = {
    "method_comparison": {"zh": "方法比较流水线", "en": "Method Comparison Pipeline"},
    "sampling_bias": {"zh": "采样偏差验证", "en": "Sampling Bias Validation"},
    "sensitivity": {"zh": "超参数敏感性分析", "en": "Sensitivity Analysis"},
    "component_ablation": {"zh": "组件级消融实验", "en": "Component Ablation"},
    "efficiency": {"zh": "效率实验", "en": "Efficiency Experiment"},
    "significance": {"zh": "统计显著性实验", "en": "Statistical Significance"},
}

STATUS_TEXT = {
    "pending": {"zh": "待运行", "en": "Pending"},
    "running": {"zh": "运行中", "en": "Running"},
    "succeeded": {"zh": "已完成", "en": "Succeeded"},
    "failed": {"zh": "失败", "en": "Failed"},
    "aborted": {"zh": "已中止", "en": "Aborted"},
    "official": {"zh": "官方", "en": "Official"},
}

SENSITIVITY_PARAM_TEXT = {
    "t_s": {"zh": "t_s / 相似度阈值", "en": "t_s / Similarity Threshold"},
    "M": {"zh": "M / 预热轮数", "en": "M / Warmup Epochs"},
    "K": {"zh": "K / 更新间隔", "en": "K / Update Interval"},
}

DEFAULT_RUN_NAMES = {
    "method_comparison": {"zh": "方法比较流水线", "en": "Method Comparison Pipeline"},
    "sampling_bias": {"zh": "采样偏差验证", "en": "Sampling Bias Validation"},
    "sensitivity": {"zh": "超参数敏感性分析", "en": "Sensitivity Analysis"},
    "component_ablation": {"zh": "组件级消融实验", "en": "Component Ablation"},
    "efficiency": {"zh": "效率实验", "en": "Efficiency Experiment"},
    "significance": {"zh": "统计显著性实验", "en": "Statistical Significance"},
}

TEXT = {
    "site.title": {"zh": "GRACE 实验系统", "en": "GRACE Experiment System"},
    "site.brand_copy": {"zh": "图学习实验控制台", "en": "Graph Learning Experiment Console"},
    "site.language_switch": {"zh": "语言切换", "en": "Language Switch"},
    "site.lang_zh": {"zh": "中文", "en": "中文"},
    "site.lang_en": {"zh": "EN", "en": "EN"},
    "forms.run_name": {"zh": "运行名称", "en": "Run Name"},
    "forms.dataset": {"zh": "数据集", "en": "Dataset"},
    "forms.method": {"zh": "Method", "en": "Method"},
    "forms.chart_title": {"zh": "图标题", "en": "Plot Title"},
    "forms.methods": {"zh": "Methods", "en": "Methods"},
    "forms.paper_params": {"zh": "Paper Params", "en": "Paper Params"},
    "forms.comparison_pairs": {"zh": "Comparison Pairs", "en": "Comparison Pairs"},
    "forms.force_grid": {"zh": "force_grid", "en": "force_grid"},
    "forms.continue_on_error": {"zh": "continue_on_error", "en": "continue_on_error"},
    "dashboard.hero_eyebrow": {"zh": "Django 实验系统", "en": "Django Experiment System"},
    "dashboard.hero_title": {
        "zh": "将现有的 GRACE 脚本整理为可配置、可追踪的图形化实验控制台。",
        "en": "Turn the existing GRACE scripts into a configurable, traceable visual experiment console.",
    },
    "dashboard.hero_text": {
        "zh": (
            "下方六个入口分别复用当前项目中的采样偏差验证、方法比较流水线、组件级消融、效率、统计显著性与超参数敏感性分析脚本。"
            "同时，实验室已保存的 results/ 与 logs/ 归档结果也会在这里统一展示，"
            "并与网页端新运行实验共用同一套结果页样式。"
        ),
        "en": (
            "The six entries below reuse the project's sampling-bias, method-comparison, component-ablation, "
            "efficiency, statistical-significance, and hyperparameter sensitivity scripts. Archived results under results/ and logs/ are also "
            "presented here with the same result-page style used by newly launched web experiments."
        ),
    },
    "sampling_bias_label": {"zh": "采样偏差验证", "en": "Sampling Bias Validation"},
    "method_comparison_label": {"zh": "方法比较流水线", "en": "Method Comparison Pipeline"},
    "sensitivity_label": {"zh": "超参数敏感性分析", "en": "Sensitivity Analysis"},
    "component_ablation_label": {"zh": "组件级消融实验", "en": "Component Ablation"},
    "efficiency_label": {"zh": "效率实验", "en": "Efficiency Experiment"},
    "significance_label": {"zh": "统计显著性实验", "en": "Statistical Significance"},
    "dashboard.stat_datasets": {"zh": "数据集", "en": "Datasets"},
    "dashboard.stat_modules": {"zh": "实验模块", "en": "Experiment Modules"},
    "dashboard.stat_recent_runs": {"zh": "最近实验", "en": "Recent Runs"},
    "dashboard.stat_official_results": {"zh": "官方结果", "en": "Official Results"},
    "dashboard.sampling_desc": {
        "zh": "启用偏差指标记录，自动生成 violation_rate 与 mean_margin 曲线，快速观察训练过程中的采样偏差变化。",
        "en": "Record bias metrics and automatically generate violation_rate and mean_margin curves to inspect sampling bias throughout training.",
    },
    "dashboard.sampling_start": {"zh": "启动采样偏差实验", "en": "Launch Sampling Bias Run"},
    "dashboard.method_desc": {
        "zh": "面向单个数据集运行完整方法比较流程，并在结果页中统一展示 robust_score、CSV 产物与相关图表。",
        "en": "Run the full method-comparison workflow on a single dataset and present robust_score, CSV outputs, and charts in one result page.",
    },
    "dashboard.method_start": {"zh": "启动方法比较实验", "en": "Launch Method Comparison Run"},
    "dashboard.sensitivity_desc": {
        "zh": "针对不同方法执行参数扫描，生成敏感性总览图与明细 CSV，并在详情页中统一查看。",
        "en": "Run parameter sweeps across methods, generate overview plots and detailed CSV files, and inspect them from a unified detail page.",
    },
    "dashboard.sensitivity_start": {"zh": "启动敏感性实验", "en": "Launch Sensitivity Run"},
    "dashboard.ablation_desc": {
        "zh": "对 SG-GR 与 SG-GC 运行 M-off、K-off、w-off 组件消融，并生成稳健性评分和相对完整方法变化图。",
        "en": "Run M-off, K-off, and w-off component ablations for SG-GR and SG-GC, with robust-score and change plots.",
    },
    "dashboard.ablation_start": {"zh": "启动消融实验", "en": "Launch Ablation Run"},
    "dashboard.efficiency_desc": {
        "zh": "记录不同方法的训练循环耗时、端到端耗时和相对时间比，评估 SG-GCL 的时间可行性。",
        "en": "Measure train-loop time, wall time, and relative time ratios to inspect SG-GCL efficiency.",
    },
    "dashboard.efficiency_start": {"zh": "启动效率实验", "en": "Launch Efficiency Run"},
    "dashboard.significance_desc": {
        "zh": "使用相同 seed 的配对运行检验 SG-GR 与 SG-GC 的主要提升是否稳定显著。",
        "en": "Use paired-seed runs to test whether the main SG-GR and SG-GC improvements are statistically stable.",
    },
    "dashboard.significance_start": {"zh": "启动显著性实验", "en": "Launch Significance Run"},
    "dashboard.official_title": {"zh": "官方结果", "en": "Official Results"},
    "dashboard.official_desc": {
        "zh": "这里展示实验室保存在 results/ 与 logs/ 中的归档结果，并使用与网页新实验一致的详情页样式。",
        "en": "This section shows archived lab results from results/ and logs/ using the same detail-page layout as newly launched web experiments.",
    },
    "dashboard.official_empty": {
        "zh": "当前尚未在 webapp/ 之外发现可展示的官方结果文件。",
        "en": "No displayable official result files were found outside webapp/ yet.",
    },
    "dashboard.recent_title": {"zh": "最近实验", "en": "Recent Runs"},
    "dashboard.recent_desc": {
        "zh": "打开任意实验记录，即可查看状态、实验产物、运行日志和自动生成的可视化内容。",
        "en": "Open any run record to inspect status, artifacts, runtime logs, and automatically generated visualizations.",
    },
    "dashboard.recent_empty": {
        "zh": "当前还没有实验记录，可以先从上方任意实验卡片开始。",
        "en": "There are no run records yet. You can start from any experiment card above.",
    },
    "dashboard.view_result": {"zh": "查看结果", "en": "View Results"},
    "dashboard.view_run": {"zh": "查看", "en": "View"},
    "dashboard.created_at": {"zh": "创建时间", "en": "Created At"},
    "status_official": {"zh": "官方", "en": "Official"},
    "common.open_file": {"zh": "打开文件", "en": "Open File"},
    "common.open_source_file": {"zh": "打开源文件", "en": "Open Source File"},
    "common.rows_count": {"zh": "{count} 行", "en": "{count} rows"},
    "common.expand": {"zh": "展开", "en": "Expand"},
    "common.error": {"zh": "错误", "en": "Error"},
    "detail.back_home": {"zh": "返回首页", "en": "Back to Home"},
    "detail.image_artifacts": {"zh": "图像产物", "en": "Image Artifacts"},
    "detail.reports": {"zh": "报告", "en": "Reports"},
    "detail.in_memory_report": {"zh": "内存中的报告", "en": "In-Memory Report"},
    "detail.csv_preview": {"zh": "CSV 预览", "en": "CSV Preview"},
    "detail.no_preview_data": {"zh": "当前暂无可预览的数据。", "en": "No previewable data is available right now."},
    "detail.technical_details": {"zh": "技术细节", "en": "Technical Details"},
    "detail.technical_details_hint": {"zh": "为保持页面简洁，次要信息默认折叠展示。", "en": "Secondary details stay collapsed by default to keep the page clean."},
    "detail.result_summary_json": {"zh": "结果摘要 JSON", "en": "Result Summary JSON"},
    "detail.view_full_json": {"zh": "查看完整 JSON", "en": "View Full JSON"},
    "detail.execution_info": {"zh": "执行信息", "en": "Execution Info"},
    "detail.executed_commands": {"zh": "执行命令", "en": "Executed Commands"},
    "detail.runtime_log": {"zh": "运行日志", "en": "Runtime Log"},
    "detail.started_at": {"zh": "开始时间", "en": "Started At"},
    "detail.finished_at": {"zh": "结束时间", "en": "Finished At"},
    "detail_eyebrow.run": {"zh": "实验详情", "en": "Experiment Detail"},
    "detail_eyebrow.official": {"zh": "官方参考结果", "en": "Official Reference Result"},
    "labels.created_at": {"zh": "创建时间", "en": "Created At"},
    "labels.updated_at": {"zh": "更新时间", "en": "Updated At"},
    "config.run": {"zh": "运行配置", "en": "Run Config"},
    "config.result_metadata": {"zh": "结果元数据", "en": "Result Metadata"},
    "button.stop": {"zh": "停止实验", "en": "Stop Run"},
    "button.stop_unavailable": {"zh": "当前不可停止", "en": "Stop Unavailable"},
    "button.delete": {"zh": "删除记录", "en": "Delete Record"},
    "confirm.stop_run": {
        "zh": "确认立即停止实验 #{run_id} 吗？相关输出文件会被删除。",
        "en": "Stop run #{run_id} now? Related output files will be removed.",
    },
    "confirm.delete_run": {
        "zh": "确认删除实验 #{run_id} 及其相关文件吗？",
        "en": "Delete run #{run_id} and its related files?",
    },
    "notice.running": {
        "zh": "该实验仍在后台运行。刷新页面后可查看最新日志与实验产物。",
        "en": "This run is still executing in the background. Refresh the page to see the latest logs and artifacts.",
    },
    "notice.aborted": {
        "zh": "该实验已被用户中止，相关输出文件已自动清理。",
        "en": "This run was aborted by the user, and related output files were cleaned up automatically.",
    },
    "notice.official": {
        "zh": "此页面展示的是实验室归档结果，来源于 results/ 与 logs/ 中 webapp/ 之外的内容。网页端新创建的实验会使用同一套结果页样式，方便直接对照。",
        "en": "This page shows archived lab results from content outside webapp/ under results/ and logs/. Newly launched web experiments use the same result-page layout for direct comparison.",
    },
    "message.method_queued": {"zh": "方法比较实验 #{run_id} 已加入队列。", "en": "Method comparison run #{run_id} has been queued."},
    "message.sampling_queued": {"zh": "采样偏差实验 #{run_id} 已加入队列。", "en": "Sampling bias run #{run_id} has been queued."},
    "message.sensitivity_queued": {"zh": "超参数敏感性实验 #{run_id} 已加入队列。", "en": "Sensitivity analysis run #{run_id} has been queued."},
    "message.ablation_queued": {"zh": "组件级消融实验 #{run_id} 已加入队列。", "en": "Component ablation run #{run_id} has been queued."},
    "message.efficiency_queued": {"zh": "效率实验 #{run_id} 已加入队列。", "en": "Efficiency run #{run_id} has been queued."},
    "message.significance_queued": {"zh": "统计显著性实验 #{run_id} 已加入队列。", "en": "Statistical significance run #{run_id} has been queued."},
    "message.not_running": {"zh": "实验 #{run_id} 当前不在运行。", "en": "Run #{run_id} is not running right now."},
    "message.missing_pid": {
        "zh": "实验 #{run_id} 缺少 worker PID 记录，当前无法安全停止。",
        "en": "Run #{run_id} does not have a recorded worker PID, so it cannot be stopped safely.",
    },
    "message.stop_failed": {"zh": "实验 #{run_id} 停止失败：{error}", "en": "Failed to stop run #{run_id}: {error}"},
    "message.stop_success": {
        "zh": "实验 #{run_id} 已中止，相关输出文件已删除。",
        "en": "Run #{run_id} was aborted and its related output files were removed.",
    },
    "message.delete_active": {
        "zh": "实验 #{run_id} 仍处于活动状态，请先停止再删除记录。",
        "en": "Run #{run_id} is still active. Stop it before deleting the record.",
    },
    "message.delete_success": {
        "zh": "实验 #{run_id}（{run_label}）及其相关文件已删除。",
        "en": "Run #{run_id} ({run_label}) and its related files were deleted.",
    },
    "error.official_not_found": {"zh": "未找到对应的官方结果。", "en": "The requested official result was not found."},
    "summary.best_method": {"zh": "最优方法", "en": "Best Method"},
    "summary.best_robust_score": {"zh": "最佳 robust_score", "en": "Best robust_score"},
    "summary.compared_methods": {"zh": "比较方法数", "en": "Compared Methods"},
    "summary.epochs": {"zh": "训练轮数", "en": "Epochs"},
    "summary.final_violation_rate": {"zh": "最终 violation_rate", "en": "Final violation_rate"},
    "summary.best_mean_margin": {"zh": "最佳 mean_margin", "en": "Best mean_margin"},
    "summary.summary_rows": {"zh": "汇总行数", "en": "Summary Rows"},
    "summary.methods": {"zh": "方法列表", "en": "Methods"},
    "summary.params": {"zh": "参数列表", "en": "Params"},
    "summary.variants": {"zh": "变体数", "en": "Variants"},
    "summary.max_drop": {"zh": "最大下降", "en": "Max Drop"},
    "summary.fastest_method": {"zh": "最快方法", "en": "Fastest Method"},
    "summary.train_total_time": {"zh": "训练总耗时", "en": "Train Total Time"},
    "summary.sggr_ratio": {"zh": "SG-GR/GRACE", "en": "SG-GR/GRACE"},
    "summary.sggc_ratio": {"zh": "SG-GC/GCA", "en": "SG-GC/GCA"},
    "summary.primary_tests": {"zh": "主比较数", "en": "Primary Tests"},
    "summary.significant_tests": {"zh": "显著比较数", "en": "Significant Tests"},
    "summary.best_delta": {"zh": "最大正向差值", "en": "Best Delta"},
    "summary.keys_omitted": {"zh": "{count} 个键已省略", "en": "{count} keys omitted"},
    "summary.truncated_note": {"zh": "为页面布局截断显示", "en": "truncated for page layout"},
    "summary.truncated_suffix": {"zh": "[已截断]", "en": "[truncated]"},
    "charts.method_robust_score": {"zh": "方法比较 robust_score", "en": "Method Comparison Robust Score"},
    "charts.violation_rate": {"zh": "violation_rate 曲线", "en": "Violation Rate Curve"},
    "charts.mean_margin": {"zh": "mean_margin 曲线", "en": "Mean Margin Curve"},
    "charts.sensitivity_best": {"zh": "敏感性最佳 robust_score", "en": "Sensitivity Best Robust Score"},
    "charts.sensitivity_series": {"zh": "{label} 的 robust_score 变化", "en": "{label} Robust Score Trend"},
    "charts.ablation_drop": {"zh": "消融平均下降", "en": "Ablation Mean Drop"},
    "charts.efficiency_train_time": {"zh": "训练总耗时", "en": "Train Total Time"},
    "charts.significance_delta": {"zh": "主比较 robust_score 差值", "en": "Primary Robust Score Delta"},
    "official.config_source": {"zh": "来源", "en": "Source"},
    "official.config_location": {"zh": "存放位置", "en": "Location"},
    "official.config_result_type": {"zh": "结果类型", "en": "Result Type"},
    "official.config_artifacts": {"zh": "artifacts", "en": "artifacts"},
    "official.source_archive": {"zh": "实验室归档结果", "en": "Archived Lab Result"},
    "official.location.results": {"zh": "results/（不含 webapp）", "en": "results/ (excluding webapp)"},
    "official.location.logs": {"zh": "logs/（不含 webapp）", "en": "logs/ (excluding webapp)"},
    "official.location.sensitivity": {
        "zh": "results/ 与 results/plots/（不含 webapp）",
        "en": "results/ and results/plots/ (excluding webapp)",
    },
    "official.location.extra": {
        "zh": "results/ 与 results/plots/（不含 webapp）",
        "en": "results/ and results/plots/ (excluding webapp)",
    },
}

EXACT_ARTIFACT_KEYS = {
    "Unified Results CSV": "artifact.unified_results_csv",
    "汇总结果 CSV": "artifact.unified_results_csv",
    "Sampling Bias CSV": "artifact.sampling_bias_csv",
    "采样偏差 CSV": "artifact.sampling_bias_csv",
    "Sampling Bias Curve": "artifact.sampling_bias_curve",
    "采样偏差曲线图": "artifact.sampling_bias_curve",
    "Sensitivity Overview Plot": "artifact.sensitivity_overview_plot",
    "敏感性总览图": "artifact.sensitivity_overview_plot",
    "Sensitivity Analysis Report": "artifact.sensitivity_analysis_report",
    "敏感性分析报告": "artifact.sensitivity_analysis_report",
    "Component Ablation CSV": "artifact.component_ablation_csv",
    "Component Ablation Overview Plot": "artifact.component_ablation_overview_plot",
    "Component Ablation Change Plot": "artifact.component_ablation_change_plot",
    "Component Ablation Analysis Report": "artifact.component_ablation_analysis_report",
    "Efficiency CSV": "artifact.efficiency_csv",
    "Efficiency Train Total Time Plot": "artifact.efficiency_train_total_time_plot",
    "Efficiency Wall Time Plot": "artifact.efficiency_wall_time_plot",
    "Efficiency Time Ratio Plot": "artifact.efficiency_time_ratio_plot",
    "Efficiency Analysis Report": "artifact.efficiency_analysis_report",
    "Significance CSV": "artifact.significance_csv",
    "Significance Tests Summary CSV": "artifact.significance_tests_summary_csv",
    "Significance Mean/Std Plot": "artifact.significance_mean_std_plot",
    "Significance Paired Delta Plot": "artifact.significance_paired_delta_plot",
    "Significance Analysis Report": "artifact.significance_analysis_report",
}

ARTIFACT_TEXT = {
    "artifact.unified_results_csv": {"zh": "汇总结果 CSV", "en": "Unified Results CSV"},
    "artifact.sampling_bias_csv": {"zh": "采样偏差 CSV", "en": "Sampling Bias CSV"},
    "artifact.sampling_bias_curve": {"zh": "采样偏差曲线图", "en": "Sampling Bias Curve"},
    "artifact.sensitivity_overview_plot": {"zh": "敏感性总览图", "en": "Sensitivity Overview Plot"},
    "artifact.sensitivity_analysis_report": {"zh": "敏感性分析报告", "en": "Sensitivity Analysis Report"},
    "artifact.component_ablation_csv": {"zh": "组件消融 CSV", "en": "Component Ablation CSV"},
    "artifact.component_ablation_overview_plot": {"zh": "组件消融总览图", "en": "Component Ablation Overview Plot"},
    "artifact.component_ablation_change_plot": {"zh": "组件消融变化图", "en": "Component Ablation Change Plot"},
    "artifact.component_ablation_analysis_report": {"zh": "组件消融分析报告", "en": "Component Ablation Analysis Report"},
    "artifact.efficiency_csv": {"zh": "效率实验 CSV", "en": "Efficiency CSV"},
    "artifact.efficiency_train_total_time_plot": {"zh": "训练总耗时图", "en": "Train Total Time Plot"},
    "artifact.efficiency_wall_time_plot": {"zh": "端到端耗时图", "en": "Wall Time Plot"},
    "artifact.efficiency_time_ratio_plot": {"zh": "时间比值图", "en": "Time Ratio Plot"},
    "artifact.efficiency_analysis_report": {"zh": "效率分析报告", "en": "Efficiency Analysis Report"},
    "artifact.significance_csv": {"zh": "显著性实验 CSV", "en": "Significance CSV"},
    "artifact.significance_tests_summary_csv": {"zh": "显著性检验汇总 CSV", "en": "Significance Tests Summary CSV"},
    "artifact.significance_mean_std_plot": {"zh": "均值标准差图", "en": "Mean/Std Plot"},
    "artifact.significance_paired_delta_plot": {"zh": "配对差值图", "en": "Paired Delta Plot"},
    "artifact.significance_analysis_report": {"zh": "显著性分析报告", "en": "Significance Analysis Report"},
}

GRID_SEARCH_PATTERN = re.compile(r"^(?P<method>[A-Z-]+) (Grid Search|网格搜索)$")
SENSITIVITY_CSV_PATTERN = re.compile(r"^(?P<method>[A-Z-]+) (Sensitivity CSV|敏感性 CSV)$")


def normalize_ui_language(language):
    if language in SUPPORTED_UI_LANGUAGES:
        return language
    return DEFAULT_UI_LANGUAGE


def get_ui_language(request):
    if request is None:
        return DEFAULT_UI_LANGUAGE

    requested = request.GET.get("lang")
    if requested in SUPPORTED_UI_LANGUAGES:
        request.session[LANGUAGE_SESSION_KEY] = requested
        return requested

    stored = request.session.get(LANGUAGE_SESSION_KEY, DEFAULT_UI_LANGUAGE)
    return normalize_ui_language(stored)


def ui_html_lang(language):
    return "en" if normalize_ui_language(language) == "en" else "zh-CN"


def text(key, language, **kwargs):
    normalized = normalize_ui_language(language)
    variants = TEXT.get(key)
    if not variants:
        template = key
    else:
        template = variants.get(normalized) or variants.get(DEFAULT_UI_LANGUAGE) or key
    return template.format(**kwargs) if kwargs else template


def build_language_switch_url(request, language):
    normalized = normalize_ui_language(language)
    if request is None:
        return f"/?lang={normalized}"

    params = request.GET.copy()
    params["lang"] = normalized
    query = params.urlencode()
    return f"{request.path}?{query}" if query else request.path


def experiment_type_label(experiment_type, language):
    normalized = normalize_ui_language(language)
    variants = EXPERIMENT_TYPE_TEXT.get(experiment_type, {})
    return variants.get(normalized) or variants.get(DEFAULT_UI_LANGUAGE) or experiment_type


def status_label(status, language):
    normalized = normalize_ui_language(language)
    variants = STATUS_TEXT.get(status, {})
    return variants.get(normalized) or variants.get(DEFAULT_UI_LANGUAGE) or status


def sensitivity_param_label(param, language):
    normalized = normalize_ui_language(language)
    variants = SENSITIVITY_PARAM_TEXT.get(param, {})
    return variants.get(normalized) or variants.get(DEFAULT_UI_LANGUAGE) or param


def sensitivity_param_choices(language):
    return [(param, sensitivity_param_label(param, language)) for param in SENSITIVITY_PARAM_TEXT]


def default_run_name(experiment_type, language):
    normalized = normalize_ui_language(language)
    variants = DEFAULT_RUN_NAMES.get(experiment_type, {})
    return variants.get(normalized) or variants.get(DEFAULT_UI_LANGUAGE) or experiment_type


def is_default_run_name(name, experiment_type):
    stripped = (name or "").strip()
    if not stripped:
        return True
    variants = DEFAULT_RUN_NAMES.get(experiment_type, {})
    return stripped in variants.values()


def localized_run_name(run, language):
    if is_default_run_name(getattr(run, "name", ""), getattr(run, "experiment_type", "")):
        return default_run_name(run.experiment_type, language)
    return run.name


def localize_artifact_label(label, language):
    normalized = normalize_ui_language(language)
    key = EXACT_ARTIFACT_KEYS.get(label)
    if key:
        return ARTIFACT_TEXT[key][normalized]

    grid_match = GRID_SEARCH_PATTERN.match(label or "")
    if grid_match:
        suffix = "网格搜索" if normalized == "zh" else "Grid Search"
        return f"{grid_match.group('method')} {suffix}"

    sensitivity_match = SENSITIVITY_CSV_PATTERN.match(label or "")
    if sensitivity_match:
        suffix = "敏感性 CSV" if normalized == "zh" else "Sensitivity CSV"
        return f"{sensitivity_match.group('method')} {suffix}"

    return label
