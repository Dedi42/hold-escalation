**Table 5. Hold hours recovered per day on the test period (α = 0.5, δ = 2 h).**

| Policy | A, B = 6 | A, B = 12 | A, B = 24 | A, B = 48 | B, B = 6 | B, B = 12 | B, B = 24 | B, B = 48 |
|---|---|---|---|---|---|---|---|---|
| Oldest-first | 61.0 | 80.2 | 111.9 | 136.3 | 259.4 | 692.0 | 1089.6 | 1453.7 |
| Static p90 rule | 54.7 | 66.5 | 80.8 | 85.0 | 220.2 | 379.1 | 512.2 | 664.4 |
| Kaplan–Meier (department × reason) | 63.2 | 79.4 | 112.4 | 138.6 | 453.8 | 770.6 | 1160.1 | 1485.5 |
| LightGBM onset | 63.5 | 86.5 | 122.0 | 140.8 | 441.2 | 756.5 | 1000.2 | 1419.8 |
| Landmark GBM | 65.3 | 88.5 | 123.1 | 142.8 | 456.6 | 755.0 | 1112.8 | 1413.9 |
| Ensemble (Equation 4) | 66.5 | 90.4 | 122.7 | 142.2 | 445.2 | 777.5 | 1131.4 | 1456.1 |
| Ensemble, gain-based | 67.9 | 89.2 | 122.5 | 141.8 | 445.1 | 779.5 | 1122.2 | 1458.0 |
| Oracle | 88.4 | 113.4 | 136.7 | 147.2 | 474.3 | 926.2 | 1393.2 | 1788.7 |
| Ensemble vs oldest-first, test period (%) | +8.9 | +12.7* | +9.7* | +4.4* | +71.6† | +12.3† | +3.8† | +0.2† |
| Ensemble vs oldest-first, rolling origins (%) | +11.1* | +12.2* | +9.7* | +5.6* | +23.4* | +13.9* | +11.6* | +7.0* |
| Pooled over both lines, rolling origins (%) | +17.4* | +12.5* | +10.3* | +5.8* |  |  |  |  |

Note: * p_Holm < 0.05 (one-sided, weekly block bootstrap). † fewer than 6 test weeks; inference relies on rolling origins. Pooled row: inverse-variance meta-analysis, listed under the first line.
