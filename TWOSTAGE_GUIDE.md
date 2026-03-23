# Two-Stage Grid Search Guide

This document explains how to use the two-stage parameter search strategy for experiments on Cora.

## Overview

**Two-stage search** separates hyperparameter exploration into two phases:

1. **Stage 1**: Grid search over method-specific parameters (e.g., `similarity_percentile`, `max_du_per_node`, `tau`, `gca_drop_scheme`, etc.)
2. **Stage 2**: For each top-K candidate from Stage 1, scan `learning_rate` ∈ {5e-4, 1e-3, 2e-3} to find optimal LR

**Benefit**: Dramatic reduction in total trials:
- Stage 1 (IFL-GC example): ~360 candidates
- Stage 2 (per top-3): 3 candidates × 3 LR values = 9 trials
- **Total: ~369 trials** (vs. full grid ≈3500+ if LR was included in Stage 1)

**Results tracked**: Each result row includes `stage` column (e.g., "top_verify") and `base_candidate_rank` column to trace back to Stage 1 candidate.

---

## Running Two-Stage Search

### Option 1: Automated Two-Stage Pipeline (Recommended)

Run the all-in-one coordinator script for IFL-GR and IFL-GC:

```bash
cd GRACE
python tools/run_cora_full_pipeline_2stage.py \
  --gpu_id 0 \
  --baseline_runs 3 \
  --topk_verify 3 \
  --runs_per_top 3 \
  --out "results/cora_full_pipeline_2stage_results.csv"
```

**What it does**:
1. Runs GRACE baseline 3 times → CSV row `stage=baseline`
2. Calls `grid_search_iflgr_cora_2stage.py` → Stage 1 grid search → picks top-3
3. For each top-3 IFL-GR candidate: runs 3 verification runs with best LR → rows `stage=top_verify`
4. Calls `grid_search_iflgc_cora_2stage.py` → Stage 1 grid search → picks top-3
5. For each top-3 IFL-GC candidate: runs 3 verification runs with best LR → rows `stage=top_verify`
6. Appends two-level summary rows (per-candidate mean + method overall mean) → rows `stage=summary`

**Output**: Single unified CSV at `results/cora_full_pipeline_2stage_results.csv` containing:
- Baseline runs
- Top candidates from each method (Stage 2 verification)
- Summary statistics per candidate and per method

**Arguments**:
- `--gpu_id 0`: GPU device ID
- `--baseline_runs 3`: Number of GRACE baseline runs
- `--topk_verify 3`: Number of top Stage-1 candidates to verify in Stage 2
- `--runs_per_top 3`: Number of runs per top candidate (robustness)
- `--out <path>`: Output CSV path

---

### Option 2: Run Stage 1 & Stage 2 Separately

For manual control, run Stage 1 and Stage 2 scripts independently:

#### IFL-GR Two-Stage

```bash
# Stage 1: Grid search over IFL-GR parameters + tau
python GRACE/tools/grid_search_iflgr_cora_2stage.py --gpu_id 0 --topk 3 --std_weight 0.5

# Output: GRACE/results/grid_search_iflgr_cora_2stage_results.csv
# Columns include: stage=topK_results, base_candidate_rank, learning_rate, F1Mi_mean, F1Mi_std, robust_score, etc.
```

#### IFL-GC Two-Stage

```bash
# Stage 1: Grid search over IFL-GC parameters + tau
python GRACE/tools/grid_search_iflgc_cora_2stage.py --gpu_id 0 --topk 3 --std_weight 0.5

# Output: GRACE/results/grid_search_iflgc_cora_2stage_results.csv
```

After these complete, top-K candidates are already ranked by `robust_score`. You can then:
1. Extract top rows from each CSV
2. Use `verify_top_params.py` to run representative verification runs
3. Manually merge results into a summary table

---

### Option 3: One-Stage Search (Original)

If you prefer the original single-stage grid search (learning_rate not optimized separately):

```bash
# Original one-stage pipelines still available
python tools/run_cora_full_pipeline.py \
  --gpu_id 0 \
  --baseline_runs 3 \
  --topk_verify 3 \
  --runs_per_top 3 \
  --out "results/cora_full_pipeline_results_1stage.csv"
```

**Methods covered**: GRACE, IFL-GR, IFL-GC, GCA (all use one-stage search)

---

## Understanding Two-Stage Results

### Stage 1 Grid Search Output

Each two-stage script generates a results CSV with columns:

- `stage`: "stage1_grid" (intermediate) or "topK_results" (top candidates from stage 1)
- `method`: "ifl-gr" or "ifl-gc"
- `candidate_rank`: Rank within top-K, ordered by `robust_score`
- `base_candidate_rank`: (Always same as `candidate_rank` for stage 1 results; used in stage 2)
- `learning_rate`: Best learning_rate found in Stage 2 for this candidate
- `F1Mi_mean`, `F1Mi_std`: Macro F1 metrics (stage 1 = single run)
- `robust_score`: F1Mi_mean - std_weight × F1Mi_std
- `*_profile`: Edge/feature drop distribution fingerprint (IFL-GC only)

### Stage 2 Results (from pipeline)

When run via `run_cora_full_pipeline_2stage.py`, additional rows are appended with:

- `stage`: "top_verify" (representative runs)
- `base_candidate_rank`: Links to Stage 1 candidate
- `run_idx`: 1, 2, 3, ... for multiple runs per candidate
- `F1Mi_mean`, `F1Mi_std`: Averaged metrics from that run

### Summary Rows

Final rows with `stage=summary`:

- `candidate_rank`: "overall" (method average) or "1", "2", "3", ... (per-candidate mean)
- `F1Mi_mean`, `F1Mi_std`: Mean and std-dev across candidates (if "overall")

---

## Parameter Ranges: Two-Stage

### IFL-GR Stage 1
- `similarity_percentile`: [0.1, 0.3, 0.5, 0.7, 0.9]
- `max_du_per_node`: [1, 3, 5, 10]
- `unlabeled_weight`: [0.1, 0.3, 0.5, 1.0]
- `warmup_epochs`: [0, 5, 10]
- `tau`: [0.4, 0.5, 0.6]

**Stage 2 (for each top-K candidate)**:
- `learning_rate`: [5e-4, 1e-3, 2e-3]

### IFL-GC Stage 1
- All IFL-GR parameters + GCA parameters:
  - `gca_drop_scheme`: ["degree", "pr", "uniform"]
  - `drop_edge_rate_1`, `drop_edge_rate_2`: [0.1, 0.3, 0.5]
  - `drop_feature_rate_1`, `drop_feature_rate_2`: [0.1, 0.3, 0.5]
  - `tau`: [0.4, 0.5, 0.6]

**Stage 2**: Same as IFL-GR

---

## Workflow Example

1. **Run automated pipeline** (recommended first time):
   ```bash
   python tools/run_cora_full_pipeline_2stage.py --gpu_id 0 --topk_verify 3 --out results/my_2stage_results.csv
   ```
   → Produces `my_2stage_results.csv` with all stages

2. **Inspect top results**:
   ```python
   import pandas as pd
   df = pd.read_csv("GRACE/results/my_2stage_results.csv")
   
   # Top 3 IFL-GR candidates (stage=top_verify)
   df[(df['method'] == 'ifl-gr') & (df['stage'] == 'top_verify')].groupby('candidate_rank')[['F1Mi_mean', 'robust_score']].mean()
   ```

3. **Compare methods**:
   ```python
   # Summary rows only
   summary_df = df[df['stage'] == 'summary']
   print(summary_df[['method', 'candidate_rank', 'F1Mi_mean', 'robust_score']])
   ```

---

## Troubleshooting

### Script exits during Stage 1 grid search
- Check error message for GPU memory / dataset loading issues
- Verify `--gpu_id` is correct
- Check `GRACE/datasets/Cora` exists and is readable

### Learning_rate not appearing in results
- Verify your 2-stage grid scripts include learning_rate in output headers
- Check that `make_temp_config_for_method` reads `learning_rate` from CSV row

### Results CSV is empty
- Ensure baseline runs complete successfully first
- Check `results/` directory has write permissions
- Verify `--out` path is correct

---

## Files Reference

| File | Purpose | Stage |
|------|---------|-------|
| `grid_search_iflgr_cora_2stage.py` | IFL-GR param search + LR scan | Stage 1+2 |
| `grid_search_iflgc_cora_2stage.py` | IFL-GC param search + LR scan | Stage 1+2 |
| `run_cora_full_pipeline_2stage.py` | Orchestrator: baseline + both 2stage + verification + summary | Full pipeline |
| `run_cora_full_pipeline.py` | Original one-stage orchestrator | 1-stage |
| `grid_search_iflgr_cora.py` | IFL-GR one-stage | 1-stage |
| `grid_search_iflgc_cora.py` | IFL-GC one-stage | 1-stage |

---

## Next Steps

1. Run the two-stage pipeline:
   ```bash
   python GRACE/tools/run_cora_full_pipeline_2stage.py --gpu_id 0
   ```

2. Check results CSV for convergence and method comparisons

3. If two-stage results are promising (shorter time, better robustness):
   - Consider combining with GCA one-stage results for final comparison
   - Document optimal parameters per method

4. Optional: Profile Stage 2 learning_rate distribution to validate choice of {5e-4, 1e-3, 2e-3}

---

## Citation / Method References

- **GRACE**: Graph Contrastive Learning with Augmentation
- **IFL-GR**: Iterative False-Label Removal + GRACE
- **IFL-GC**: IFL + GCA (structure-aware contrastive learning)
- **GCA**: Graph Contrastive Augmentation

All implemented in `GRACE/train.py` and tested on Cora using Cora as node classification benchmark.
