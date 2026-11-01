**Table 4. Discrimination among open holds on the test period (landmarks 0.5–48 h).**

| Model | C-index A | AUC(> 8 h) A | C-index B | AUC(> 8 h) B |
|---|---|---|---|---|
| Static p90 rule | 0.541 [0.535, 0.548] | 0.595 [0.580, 0.610] | 0.489 [0.484, 0.494] | 0.486 [0.476, 0.495] |
| Kaplan–Meier (department × reason) | 0.608 [0.600, 0.615] | 0.710 [0.696, 0.727] | 0.652 [0.646, 0.657] | 0.756 [0.747, 0.766] |
| LightGBM onset | 0.673 [0.664, 0.681] | 0.825 [0.806, 0.842] | 0.686 [0.681, 0.692] | 0.793 [0.784, 0.802] |
| Landmark GBM | 0.674 [0.665, 0.681] | 0.836 [0.819, 0.850] | 0.693 [0.687, 0.699] | 0.828 [0.821, 0.838] |
| Ensemble (Equation 4) | 0.693 [0.685, 0.700] | 0.858 [0.842, 0.871] | 0.712 [0.707, 0.718] | 0.841 [0.833, 0.849] |

Note: 95% lot cluster bootstrap intervals in brackets.
