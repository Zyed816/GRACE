# Cora Sensitivity Analysis

- Plot: `results\plots\cora_ifl_sensitivity_overview.png`

## Data Status
- `results\sensitivity_iflgr_cora_results.csv`: status=ok, rows=60, summary_rows=15
- `results\sensitivity_iflgc_cora_results.csv`: status=ok, rows=60, summary_rows=15

## Findings
- SG-GR / t_s: best robust at 0.9834 (robust=0.8427, F1Mi=0.8444); robust range=0.0109. Delta vs anchor=+0.0057. Trend is non-monotonic, suggesting a local optimum.
- SG-GR / M: best robust at 100 (robust=0.8425, F1Mi=0.8450); robust range=0.0079. Delta vs anchor=+0.0000. Trend is non-monotonic, suggesting a local optimum.
- SG-GR / K: best robust at 5 (robust=0.8385, F1Mi=0.8424); robust range=0.0108. Delta vs anchor=+0.0060. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / t_s: best robust at 0.9759 (robust=0.8385, F1Mi=0.8416); robust range=0.0081. Delta vs anchor=+0.0020. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / M: best robust at 100 (robust=0.8388, F1Mi=0.8411); robust range=0.0039. Delta vs anchor=+0.0033. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / K: best robust at 1 (robust=0.8369, F1Mi=0.8403); robust range=0.0083. Delta vs anchor=+0.0034. Trend is non-monotonic, suggesting a local optimum.
