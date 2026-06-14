# SG-GCL Component Ablation Analysis

## Data Status
- `results\extra_ablation_citeseer_results.csv`: status=ok, rows=32, summary_rows=8
- `results\extra_ablation_cora_results.csv`: status=ok, rows=32, summary_rows=8
- `results\extra_ablation_dblp_results.csv`: status=ok, rows=32, summary_rows=8
- `results\extra_ablation_pubmed_results.csv`: status=ok, rows=32, summary_rows=8

## Component Impact
- M-off: mean drop=0.0014, max drop=0.0152, positive drops=5/8.
- K-off: mean drop=-0.0017, max drop=0.0052, positive drops=3/8.
- w-off: mean drop=0.0010, max drop=0.0091, positive drops=4/8.

## Dataset And Method Details
- Cora / SG-GR: full robust=0.8206; M-off robust=0.8196, drop=+0.0010; K-off robust=0.8285, drop=-0.0079; w-off robust=0.8176, drop=+0.0030.
- Cora / SG-GC: full robust=0.8230; M-off robust=0.8280, drop=-0.0050; K-off robust=0.8236, drop=-0.0006; w-off robust=0.8248, drop=-0.0018.
- CiteSeer / SG-GR: full robust=0.7099; M-off robust=0.7086, drop=+0.0013; K-off robust=0.7122, drop=-0.0023; w-off robust=0.7109, drop=-0.0011.
- CiteSeer / SG-GC: full robust=0.7132; M-off robust=0.6981, drop=+0.0152; K-off robust=0.7080, drop=+0.0052; w-off robust=0.7106, drop=+0.0026.
- PubMed / SG-GR: full robust=0.7968; M-off robust=0.7979, drop=-0.0010; K-off robust=0.7965, drop=+0.0003; w-off robust=0.7982, drop=-0.0013.
- PubMed / SG-GC: full robust=0.7935; M-off robust=0.8033, drop=-0.0097; K-off robust=0.8003, drop=-0.0067; w-off robust=0.7964, drop=-0.0029.
- DBLP / SG-GR: full robust=0.7153; M-off robust=0.7070, drop=+0.0083; K-off robust=0.7109, drop=+0.0044; w-off robust=0.7062, drop=+0.0091.
- DBLP / SG-GC: full robust=0.7029; M-off robust=0.7018, drop=+0.0011; K-off robust=0.7089, drop=-0.0060; w-off robust=0.7027, drop=+0.0002.

## Generated Files
- `results\plots\extra_ablation_warmup_M_effect.png`
- `results\plots\extra_ablation_warmup_M_effect.pdf`
- `results\plots\extra_ablation_warmup_M_effect.svg`
- `results\plots\extra_ablation_update_K_effect.png`
- `results\plots\extra_ablation_update_K_effect.pdf`
- `results\plots\extra_ablation_update_K_effect.svg`
- `results\plots\extra_ablation_weight_w_effect.png`
- `results\plots\extra_ablation_weight_w_effect.pdf`
- `results\plots\extra_ablation_weight_w_effect.svg`
- `results\plots\extra_ablation_analysis.md`
