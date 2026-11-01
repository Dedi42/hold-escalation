**Table A1. Effect of decision-time context in the benchmark: without → with context features (test period, mean over five seeds).**

| Line, model | AUC(> 8 h) | B = 6 | B = 12 | B = 24 | B = 48 |
|---|---|---|---|---|---|
| A, LightGBM landmark | 0.749 → 0.769 | 66.7 → 65.8 | 90.2 → 91.6 | 122.5 → 125.0 | 145.6 → 146.2 |
| A, MLP survival | 0.746 → 0.783 | 66.7 → 71.3 | 88.3 → 96.1 | 122.9 → 126.5 | 146.3 → 146.3 |
| B, LightGBM landmark | 0.727 → 0.753 | 477 → 479 | 815 → 821 | 1180 → 1224 | 1503 → 1593 |
| B, MLP survival | 0.751 → 0.759 | 490 → 478 | 820 → 796 | 1191 → 1195 | 1551 → 1567 |

Note: columns B = 6 to 48 give hold hours recovered per day; the benchmark divides by elapsed review days.
