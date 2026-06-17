# DBLP Sensitivity Analysis

- Plot: `results\plots\dblp_sg_sensitivity_overview.png`

## Data Status
- `results\sensitivity_sggr_dblp_results.csv`: status=ok, rows=60, summary_rows=15
- `results\sensitivity_sggc_dblp_results.csv`: status=ok, rows=60, summary_rows=15

## Findings
- SG-GR / t_s: best robust at 0.8662 (robust=0.7217, F1Mi=0.7240); robust range=0.0100. Delta vs anchor=+0.0100. Trend is non-monotonic, suggesting a local optimum.
- SG-GR / M: best robust at 140 (robust=0.7238, F1Mi=0.7247); robust range=0.0106. Delta vs anchor=+0.0092. Trend is non-monotonic, suggesting a local optimum.
- SG-GR / K: best robust at 4 (robust=0.7227, F1Mi=0.7245); robust range=0.0100. Delta vs anchor=+0.0088. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / t_s: best robust at 0.8347 (robust=0.7134, F1Mi=0.7158); robust range=0.0134. Delta vs anchor=+0.0000. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / M: best robust at 120 (robust=0.7180, F1Mi=0.7201); robust range=0.0119. Delta vs anchor=+0.0077. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / K: best robust at 3 (robust=0.7173, F1Mi=0.7202); robust range=0.0082. Delta vs anchor=+0.0000. Trend is non-monotonic, suggesting a local optimum.
