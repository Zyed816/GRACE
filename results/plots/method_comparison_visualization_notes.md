# Method Comparison Visualization Notes

## Metric Choice
- `robust_score`: robust_score combines mean performance and stability, and is therefore well suited to summarize the overall effectiveness of each method.
- `F1Mi_mean`: Micro-F1 reflects the overall classification accuracy across all nodes and is the most direct indicator of downstream classification quality.
- `F1Ma_mean`: Macro-F1 gives equal weight to each class and is helpful for observing whether the method also improves balanced performance across categories.
- `delta_vs_grace`: Delta vs GRACE is a supplementary metric that directly reveals the gain brought by each method relative to the vanilla contrastive baseline.

## Statistical Rule
- For each dataset and each method, the figure uses the best verified candidate selected by the highest mean robust score.
- The error bars are computed from repeated verification runs of the selected candidate.
- GRACE uses its baseline repeated runs directly.

## Winner Snapshot
- Cora: best robust_score = SG-GR (0.8394); best Micro-F1 = SG-GR (0.8424); best Macro-F1 = SG-GC (0.8292).
- CiteSeer: best robust_score = SG-GC (0.7149); best Micro-F1 = GCA (0.7188); best Macro-F1 = SG-GC (0.6516).
- PubMed: best robust_score = SG-GC (0.7961); best Micro-F1 = SG-GC (0.7973); best Macro-F1 = SG-GC (0.7968).
- DBLP: best robust_score = SG-GR (0.7221); best Micro-F1 = SG-GR (0.7247); best Macro-F1 = SG-GC (0.6448).

## Generated Files
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_best_verified_summary.csv`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_overview.png`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_overview.pdf`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_overview.svg`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_robust_score.png`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_robust_score.pdf`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_robust_score.svg`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_micro_f1.png`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_micro_f1.pdf`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_micro_f1.svg`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_macro_f1.png`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_macro_f1.pdf`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_macro_f1.svg`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_delta_vs_grace.png`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_delta_vs_grace.pdf`
- `D:/dissertation/openSourceCode/GRACE/results/plots/method_comparison_delta_vs_grace.svg`
