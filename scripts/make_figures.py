"""Legacy (v1.x): redraw the landmark and frontier figures from the result tables of run_pipeline.py.
The figures of the paper are rebuilt by scripts/make_paper.py (see README, "Reproducing the paper").

    python scripts/make_figures.py --results results --out figures
(Figure 2 needs the hold-level data and is drawn by run_pipeline.py.)
"""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from holdesc import figures as F

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="figures")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
M = pd.read_csv(os.path.join(a.results, "table4_landmark_metrics.csv"))
F.fig_landmark(M, os.path.join(a.out, "figure3_landmark.png"),
               ["Static p90 rule", "Kaplan–Meier (D×R)", "Cox PH", "LightGBM (log T)", "Landmark GBM", "Ensemble (proposed)"])
F.fig_frontier(pd.read_csv(os.path.join(a.results, "table5_policies.csv")), os.path.join(a.out, "figure4_frontier.png"))
print("figures written to", a.out)
