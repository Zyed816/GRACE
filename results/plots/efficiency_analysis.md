# SG-GCL Efficiency Analysis

## Data Status
- `results\efficiency_citeseer_results.csv`: status=ok, rows=16, summary_rows=4
- `results\efficiency_cora_results.csv`: status=ok, rows=16, summary_rows=4
- `results\efficiency_dblp_results.csv`: status=ok, rows=16, summary_rows=4
- `results\efficiency_pubmed_results.csv`: status=ok, rows=16, summary_rows=4

## Time Summary
- Cora: fastest=GRACE (train=3.80s); GRACE train=3.80s, wall=15.59s, 1.00x vs GRACE; GCA train=3.83s, wall=15.81s, 1.01x vs GRACE; SG-GR train=4.36s, wall=16.23s, 1.15x vs GRACE; SG-GC train=4.87s, wall=16.85s, 1.28x vs GRACE.
- CiteSeer: fastest=GCA (train=4.29s); GRACE train=4.32s, wall=26.62s, 1.00x vs GRACE; GCA train=4.29s, wall=26.28s, 0.99x vs GRACE; SG-GR train=5.48s, wall=28.32s, 1.27x vs GRACE; SG-GC train=6.37s, wall=28.76s, 1.48x vs GRACE.
- PubMed: fastest=GCA (train=5.03s); GRACE train=5.29s, wall=16.76s, 1.00x vs GRACE; GCA train=5.03s, wall=15.94s, 0.95x vs GRACE; SG-GR train=194.24s, wall=205.21s, 36.70x vs GRACE; SG-GC train=369.64s, wall=380.68s, 69.84x vs GRACE.
- DBLP: fastest=GCA (train=4.86s); GRACE train=5.19s, wall=17.42s, 1.00x vs GRACE; GCA train=4.86s, wall=17.48s, 0.94x vs GRACE; SG-GR train=201.09s, wall=213.31s, 38.76x vs GRACE; SG-GC train=409.38s, wall=421.92s, 78.91x vs GRACE.

## Generated Files
- `results\plots\efficiency_train_total_time.png`
- `results\plots\efficiency_train_total_time.pdf`
- `results\plots\efficiency_wall_time.png`
- `results\plots\efficiency_wall_time.pdf`
- `results\plots\efficiency_time_ratio.png`
- `results\plots\efficiency_time_ratio.pdf`
- `results\plots\efficiency_analysis.md`
