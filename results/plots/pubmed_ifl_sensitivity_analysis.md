# PubMed Sensitivity Analysis

- Plot: `results\plots\pubmed_ifl_sensitivity_overview.png`

## Data Status
- `results\sensitivity_iflgr_pubmed_results.csv`: status=ok, rows=60, summary_rows=15
- `results\sensitivity_iflgc_pubmed_results.csv`: status=ok, rows=60, summary_rows=15

## Findings
- SG-GR / t_s: best robust at 0.9273 (robust=0.7930, F1Mi=0.7955); robust range=0.0074. Delta vs anchor=+0.0000. Trend is non-monotonic, suggesting a local optimum.
- SG-GR / M: best robust at 60 (robust=0.7924, F1Mi=0.7937); robust range=0.0091. Delta vs anchor=+0.0091. Trend is non-monotonic, suggesting a local optimum.
- SG-GR / K: best robust at 3 (robust=0.7944, F1Mi=0.7960); robust range=0.0082. Delta vs anchor=+0.0000. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / t_s: best robust at 0.8960 (robust=0.7964, F1Mi=0.7979); robust range=0.0105. Delta vs anchor=+0.0000. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / M: best robust at 120 (robust=0.7915, F1Mi=0.7936); robust range=0.0042. Delta vs anchor=+0.0037. Trend is non-monotonic, suggesting a local optimum.
- SG-GC / K: best robust at 1 (robust=0.7966, F1Mi=0.7994); robust range=0.0109. Delta vs anchor=+0.0053. Trend is monotonic decreasing in the tested range.
