# SG-GCL Component Ablation Analysis

## Data Status
- `results\extra_ablation_cora_results.csv`: status=ok, rows=8, summary_rows=4

## Component Impact
- w/o Warmup: mean drop=0.0047, max drop=0.0047, positive drops=1/1.
- w/o Dynamic Update: mean drop=-0.0053, max drop=-0.0053, positive drops=0/1.
- w/o Semantic Weight: mean drop=-0.0004, max drop=-0.0004, positive drops=0/1.

## Dataset And Method Details
- Cora / SG-GR: full robust=0.8195; w/o Warmup robust=0.8147, drop=+0.0047; w/o Dynamic Update robust=0.8248, drop=-0.0053; w/o Semantic Weight robust=0.8198, drop=-0.0004.

## Generated Files
- `results\plots\extra_ablation_overview.png`
- `results\plots\extra_ablation_overview.pdf`
- `results\plots\extra_ablation_drop_vs_full.png`
- `results\plots\extra_ablation_drop_vs_full.pdf`
- `results\plots\extra_ablation_analysis.md`
