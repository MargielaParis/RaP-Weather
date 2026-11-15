# Results

Statistics behind the paper tables, written by `analysis/statistical_audit.py`
from the saved per-query retrieval results. All comparisons use the 4,943
confirmatory queries (the 200 model-selection queries are excluded),
per-query bootstrap 95% CIs (2,000 resamples), paired Wilcoxon signed-rank
tests against RaP-Weather and Holm correction per family.

| File | Content |
|---|---|
| `primary_main_stats.csv` | Weather-observation MSE of RaP-Weather, Group 1, Group 2 and Pixel-NN for 4 regions and k in {1, 3, 5, 10, 50, 100}. Holm family of 150 comparisons. Tables 3 and 4, Fig. 6. |
| `visual_metric_stats.csv` | LPIPS, PSNR and SSIM comparisons. Holm family of 216 comparisons. Table 2. |
| `per_region_reference_stats.csv` | Per-region fine-tuned encoders. Holm family of 72 comparisons. Table 9 (Appendix B). |
| `primary_cell_claims.csv`, `per_region_cell_claims.csv` | Per region and depth: lowest mean, significantly better or worse comparisons. |
| `audit_summary.json` | Counts quoted in the text (for example 23 of 24 lowest means, 16 of 24 significant against all baselines). |
| `model_selection_queries.csv` | The 200 model-selection query indices excluded from the confirmatory set. |
| `tau_resweep_d02_retrieval.csv`, `best_tau_d02.json` | Appendix A tau re-check at temporal dropout 0.2 (`scripts/tau_resweep_d02.py`). |

Re-running the analysis scripts overwrites these files. Set `RAP_STATS_DIR` to
write elsewhere.
