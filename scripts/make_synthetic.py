"""Generate a synthetic hold data set with the same schema as the study data.

The generator reproduces the qualitative features reported in the paper (heavy-tailed
durations, a decreasing release hazard, a dominant department, reason bursts, and
workload effects) but none of the plant's actual values. Use it to test the pipeline.

    python scripts/make_synthetic.py --out data/synthetic --holds 30000
"""
import argparse, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--holds", type=int, default=30000); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); os.makedirs(a.out, exist_ok=True)
    n, weeks = a.holds, 20
    deps = np.array(["DEP_A", "DEP_B", "DEP_C", "DEP_D", "DEP_E"]); pdep = [0.85, 0.06, 0.05, 0.035, 0.005]
    reasons = np.array([f"RSN_{i:02d}" for i in range(25)]); preason = rng.dirichlet(np.ones(25) * 0.5)
    steps = np.array([f"STEP_{i:03d}" for i in range(120)])
    n_lots = n // 9
    lots = np.array([f"LOT{i:06d}" for i in range(n_lots)])
    start0 = pd.Timestamp("2030-01-01")
    lot_release = start0 + pd.to_timedelta(rng.uniform(0, weeks * 7 - 14, n_lots), unit="D")
    lot_type = rng.choice(["PROD", "ENG", "QUAL"], n_lots, p=[0.65, 0.3, 0.05])
    lot_fam = rng.choice([f"FAM_{i}" for i in range(12)], n_lots)
    li = rng.integers(0, n_lots, n)
    start = lot_release[li] + pd.to_timedelta(rng.uniform(0, 12, n), unit="D")
    dep = rng.choice(deps, n, p=pdep); rsn = rng.choice(reasons, n, p=preason); stp = rng.choice(steps, n)
    # duration: log-normal mixture -> decreasing hazard; reason, department and lot type shift the scale
    r_eff = dict(zip(reasons, rng.normal(0, 0.6, 25))); s_eff = dict(zip(steps, rng.normal(0, 0.5, 120)))
    d_eff = {"DEP_A": 0, "DEP_B": 0.1, "DEP_C": -0.9, "DEP_D": -0.3, "DEP_E": 1.0}
    mu = -0.3 + np.vectorize(r_eff.get)(rsn) + np.vectorize(s_eff.get)(stp) + np.vectorize(d_eff.get)(dep) + 0.4 * (lot_type[li] != "PROD")
    sig = np.where(rng.random(n) < 0.2, 2.2, 1.2)
    dur = np.exp(mu + sig * rng.standard_normal(n))
    release = start + pd.to_timedelta(dur, unit="h")
    end = start0 + pd.Timedelta(weeks=weeks)
    release = release.where(release <= end, pd.NaT)
    holds = pd.DataFrame({"lot": lots[li], "step_code": stp, "reason_code": rsn, "dept_code": dep,
                          "hold_start": start, "hold_release": release, "lot_priority": rng.integers(1, 10, n),
                          "lot_kind": lot_type[li], "family": lot_fam[li], "wafers": rng.integers(1, 26, n),
                          "rework_flag": rng.choice(["Y", "N"], n, p=[0.05, 0.95])}).sort_values("hold_start")
    holds.to_csv(f"{a.out}/holds.csv", index=False)
    hold_h = holds.assign(h=(holds.hold_release.fillna(end) - holds.hold_start).dt.total_seconds() / 3600).groupby("lot").h.sum()
    lt = pd.DataFrame({"lot": lots, "release": lot_release})
    lt["completion"] = lt.release + pd.to_timedelta(rng.gamma(4, 1.5, n_lots), unit="D") + pd.to_timedelta(lt.lot.map(hold_h).fillna(0), unit="h")
    lt["due"] = (lt.release + pd.to_timedelta(rng.normal(8, 1.5, n_lots), unit="D")).dt.normalize() + pd.Timedelta(hours=23, minutes=59)
    lt["risk_score"] = np.clip(rng.beta(1, 6, n_lots) + 0.3 * (lt.completion > lt.due), 0, 1)
    lt.to_csv(f"{a.out}/lots.csv", index=False)
    print(f"wrote {len(holds)} holds and {len(lt)} lots to {a.out}")


if __name__ == "__main__":
    main()
