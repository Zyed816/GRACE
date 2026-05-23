# CiteSeer Sensitivity Analysis

- Plot: `results\plots\citeseer_ifl_sensitivity_overview.png`

## Data Status
- `results\sensitivity_iflgr_citeseer_results.csv`: status=ok, rows=56, summary_rows=14
- `results\sensitivity_iflgc_citeseer_results.csv`: status=ok, rows=60, summary_rows=15

## Findings
- SG-GR / t_s: best robust at 1.0000 (robust=0.7048, F1Mi=0.7093); robust range=0.0060. Delta vs anchor=+0.0030. Trend is non-monotonic, suggesting a local optimum.
- SG-GR / M: best robust at 60 (robust=0.7096, F1Mi=0.7119); robust range=0.0042. Delta vs anchor=+0.0000. Trend is non-monotonic, suggesting a local optimum.
- SG-GR / K: best robust at 5 (robust=0.7097, F1Mi=0.7120); robust range=0.0071. Delta vs anchor=+0.0047. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / t_s: best robust at 0.9891 (robust=0.7183, F1Mi=0.7210); robust range=0.0136. Delta vs anchor=+0.0136. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / M: best robust at 120 (robust=0.7127, F1Mi=0.7171); robust range=0.0101. Delta vs anchor=+0.0082. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / K: best robust at 1 (robust=0.7172, F1Mi=0.7191); robust range=0.0054. Delta vs anchor=+0.0049. Trend is non-monotonic, suggesting a local optimum.
