# Numbers quoted in the text

| Quantity | Value | Source |
|---|---|---|
| Line A: share of hold hours in longest 10% of holds (H1) | 0.741 [0.710, 0.767] | `hypotheses_summary.csv` |
| Line A: Weibull shape (H2a) | 0.518 [0.514, 0.522] | `hypotheses_summary.csv` |
| Line A: median remaining after 8 h / median at onset (H2b) | 8.83 [8.11, 9.46] | `hypotheses_summary.csv` |
| Line B: share of hold hours in longest 10% of holds (H1) | 0.822 [0.811, 0.832] | `hypotheses_summary.csv` |
| Line B: Weibull shape (H2a) | 0.437 [0.436, 0.439] | `hypotheses_summary.csv` |
| Line B: median remaining after 8 h / median at onset (H2b) | 8.49 [8.13, 8.95] | `hypotheses_summary.csv` |
| Line A: AUC(remaining > 8 h): Ensemble > LightGBM onset (H3a) | ΔAUC +0.033, p_Holm < 0.001 | `hypotheses_summary.csv` |
| Line A: AUC(remaining > 8 h): Ensemble > Landmark GBM (H3b) | ΔAUC +0.022, p_Holm < 0.001 | `hypotheses_summary.csv` |
| Line A: AUC(remaining > 8 h): Ensemble > Kaplan–Meier (H3c) | ΔAUC +0.148, p_Holm < 0.001 | `hypotheses_summary.csv` |
| Line A: AUC(remaining > 8 h): Landmark GBM > Static p90 (H3d) | ΔAUC +0.241, p_Holm < 0.001 | `hypotheses_summary.csv` |
| Line B: AUC(remaining > 8 h): Ensemble > LightGBM onset (H3a) | ΔAUC +0.048, p_Holm < 0.001 | `hypotheses_summary.csv` |
| Line B: AUC(remaining > 8 h): Ensemble > Landmark GBM (H3b) | ΔAUC +0.012, p_Holm < 0.001 | `hypotheses_summary.csv` |
| Line B: AUC(remaining > 8 h): Ensemble > Kaplan–Meier (H3c) | ΔAUC +0.085, p_Holm < 0.001 | `hypotheses_summary.csv` |
| Line B: AUC(remaining > 8 h): Landmark GBM > Static p90 (H3d) | ΔAUC +0.343, p_Holm < 0.001 | `hypotheses_summary.csv` |
| Line A: ensemble vs oldest-first, rolling origins, B = 6 | +11.1% [-1.0, +29.8] | `tables/forest_gain_vs_age.csv` |
| Line A: ensemble vs oldest-first, rolling origins, B = 12 | +12.2% [+6.0, +20.2] | `tables/forest_gain_vs_age.csv` |
| Line A: ensemble vs oldest-first, rolling origins, B = 24 | +9.7% [+6.3, +13.8] | `tables/forest_gain_vs_age.csv` |
| Line A: ensemble vs oldest-first, rolling origins, B = 48 | +5.6% [+3.7, +7.7] | `tables/forest_gain_vs_age.csv` |
| Line B: ensemble vs oldest-first, rolling origins, B = 6 | +23.4% [+8.6, +41.3] | `tables/forest_gain_vs_age.csv` |
| Line B: ensemble vs oldest-first, rolling origins, B = 12 | +13.9% [-0.6, +29.5] | `tables/forest_gain_vs_age.csv` |
| Line B: ensemble vs oldest-first, rolling origins, B = 24 | +11.6% [+6.3, +18.6] | `tables/forest_gain_vs_age.csv` |
| Line B: ensemble vs oldest-first, rolling origins, B = 48 | +7.0% [+2.6, +13.2] | `tables/forest_gain_vs_age.csv` |
| Pooled: ensemble vs oldest-first, rolling origins, B = 6 | +17.4% [+6.7, +29.2] | `tables/forest_gain_vs_age.csv` |
| Pooled: ensemble vs oldest-first, rolling origins, B = 12 | +12.5% [+6.2, +19.2] | `tables/forest_gain_vs_age.csv` |
| Pooled: ensemble vs oldest-first, rolling origins, B = 24 | +10.3% [+7.1, +13.5] | `tables/forest_gain_vs_age.csv` |
| Pooled: ensemble vs oldest-first, rolling origins, B = 48 | +5.8% [+3.9, +7.6] | `tables/forest_gain_vs_age.csv` |
| H9 heterogeneity, B = 6 | I² = 0.13, Q p = 0.28 | `hypotheses_summary.csv` |
| H9 heterogeneity, B = 12 | I² = 0.00, Q p = 0.83 | `hypotheses_summary.csv` |
| H9 heterogeneity, B = 24 | I² = 0.00, Q p = 0.60 | `hypotheses_summary.csv` |
| H9 heterogeneity, B = 48 | I² = 0.00, Q p = 0.62 | `hypotheses_summary.csv` |
| Line A: static p90 vs oldest-first (H5), B = 6..48 | -10.4, -17.1, -27.8, -37.7 | `hypotheses_summary.csv` |
| Line B: static p90 vs oldest-first (H5), B = 6..48 | -15.1, -45.2, -53.0, -54.3 | `hypotheses_summary.csv` |
| Line A: benchmark AUC, MLP survival (paper features) | 0.746 | `benchmark_A/tables/bench_prediction_pooled.csv` |
| Line A: benchmark AUC, MLP survival (ext.) | 0.783 | `benchmark_A/tables/bench_prediction_pooled.csv` |
| Line A: benchmark AUC, LightGBM landmark (paper features) | 0.749 | `benchmark_A/tables/bench_prediction_pooled.csv` |
| Line A: benchmark AUC, LightGBM landmark (ext. features) | 0.769 | `benchmark_A/tables/bench_prediction_pooled.csv` |
| Line A: benchmark AUC, LotSeq decision-focused ranker | 0.790 | `benchmark_A/tables/bench_prediction_pooled.csv` |
| Line A: seconds per fit | LightGBM landmark (paper features): 0.8; LightGBM landmark (ext. features): 1.7; LightGBM gain regression (ext.): 1.4; LightGBM LambdaRank (ext.): 1.2; MLP survival (paper features): 2.2; MLP survival (ext.): 2.3; FT-Transformer survival (ext.): 70.3; LotSeq survival (ext. + history): 10.9; LotSeq decision-focused ranker: 8.3 | `benchmark_A/tables/bench_runtime.csv` |
| Line B: benchmark AUC, MLP survival (paper features) | 0.751 | `benchmark_B/tables/bench_prediction_pooled.csv` |
| Line B: benchmark AUC, MLP survival (ext.) | 0.759 | `benchmark_B/tables/bench_prediction_pooled.csv` |
| Line B: benchmark AUC, LightGBM landmark (paper features) | 0.727 | `benchmark_B/tables/bench_prediction_pooled.csv` |
| Line B: benchmark AUC, LightGBM landmark (ext. features) | 0.753 | `benchmark_B/tables/bench_prediction_pooled.csv` |
| Line B: benchmark AUC, LotSeq decision-focused ranker | 0.725 | `benchmark_B/tables/bench_prediction_pooled.csv` |
| Line B: seconds per fit | LightGBM landmark (paper features): 2.2; LightGBM landmark (ext. features): 4.5; LightGBM gain regression (ext.): 1.6; LightGBM LambdaRank (ext.): 2.5; MLP survival (paper features): 10.8; MLP survival (ext.): 9.4; FT-Transformer survival (ext.): 70.6; LotSeq survival (ext. + history): 51.7; LotSeq decision-focused ranker: 29.8 | `benchmark_B/tables/bench_runtime.csv` |
