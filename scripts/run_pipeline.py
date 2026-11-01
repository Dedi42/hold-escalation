"""End-to-end pipeline: load -> anonymise -> features -> models -> escalation replay -> tables and figures.

    python scripts/run_pipeline.py --config config/local.yaml --out results
    python scripts/run_pipeline.py --config config/synthetic.yaml --out results_synthetic

Only anonymised, aggregated results are written to --out. Nothing in --out contains
raw codes, lot identifiers, or calendar dates.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from holdesc.io import load_config, load_holds, load_lots
from holdesc.anonymize import anonymize
from holdesc.features import build, temporal_split, purge_split, landmark_frame
from holdesc.survival import StaticModels, residual_life_table, department_table, weibull_shapes
from holdesc.landmark import LandmarkModel
from holdesc.evaluate import landmark_metrics, pooled_bootstrap
from holdesc.policy import review_candidates, simulate, summarise, block_bootstrap
from holdesc import figures as F
from lifelines.utils import concordance_index

CAP = 168.0
KAPPA, URGENCY_DAYS = 1.0, 3.0
EVAL_LANDMARKS = [0.5, 1, 2, 4, 8, 16, 24, 48]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--budgets", default="6,12,24,48"); ap.add_argument("--bootstrap", type=int, default=2000); ap.add_argument("--lm-bootstrap", dest="lm_bootstrap", type=int, default=1000)
    ap.add_argument("--purge", action="store_true", help="purged temporal split (robustness check against label leakage)")
    ap.add_argument("--hetero-draws", dest="hetero_draws", type=int, default=20, help="draws of group-specific alpha and delta")
    ap.add_argument("--no-revision", dest="revision", action="store_false", help="skip the revision analyses (gain index, heterogeneous effect)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    log("load and anonymise")
    h_raw, lots_raw = load_holds(cfg), load_lots(cfg)
    h, lots, _keys = anonymize(h_raw, lots_raw)
    del h_raw, lots_raw, _keys
    h, oc = build(h)
    h["split"] = temporal_split(h)
    # data diagnostics: arrival rate, time-average number of open holds (Little's law check), straddling holds
    v0, t0 = h.start_time[h.split == "val"].min(), h.start_time[h.split == "test"].min()
    span_d = (h.start_time.max() - h.start_time.min()).total_seconds() / 86400
    grid_t = pd.date_range(h.start_time.min() + pd.Timedelta(days=1), h.start_time.max(), freq="1h").values.astype("int64")
    diag = dict(arrivals_per_day=len(h) / span_d,
                arrivals_per_day_by_split={s: float((h.split == s).sum() / ((g.start_time.max() - g.start_time.min()).total_seconds() / 86400))
                                           for s, g in h.groupby("split")},
                open_holds_time_average=float(oc(grid_t).mean()), open_holds_p10_p90=[float(x) for x in np.percentile(oc(grid_t), [10, 90])],
                littles_law_open=float(len(h) / span_d * h.dur_h.mean() / 24),
                littles_law_open_cap30d=float(len(h) / span_d * h.dur_h.clip(upper=720).mean() / 24),
                train_holds_open_at_val_start=int(((h.split == "train") & (h.rel > v0)).sum()),
                val_holds_open_at_test_start=int(((h.split == "val") & (h.rel > t0)).sum()),
                holds_open_at_data_start=int((h.start_time <= h.start_time.min() + pd.Timedelta(hours=1)).sum()))
    if a.purge:
        h["split"] = purge_split(h, h["split"])
        diag["purged_holds"] = int((h.split == "purged").sum())
    tr, va, te = (h[h.split == s] for s in ["train", "val", "test"])
    trva = pd.concat([tr, va])
    res = dict(holds=len(h), lots=int(h.lot_id.nunique()), weeks=round((h.start_time.max() - h.start_time.min()).days / 7, 1),
               split={s: int((h.split == s).sum()) for s in ["train", "val", "test"]}, diagnostics=diag, purged=a.purge,
               released_share=float(h.event.mean()),
               quantiles_h={str(q): float(h.dur_h.quantile(q)) for q in [.5, .75, .9, .95, .99]},
               top_share_of_hours={str(q): float(np.sort(h.dur_h.values)[::-1][:int(len(h) * q)].sum() / h.dur_h.sum()) for q in [.01, .05, .1]})
    residual_life_table(h).to_csv(f"{a.out}/table3_residual_life.csv", index=False)
    dt = department_table(h)
    wb = weibull_shapes(h, list(dt.index[:4]))
    dt = dt.join(wb.drop(index="All", errors="ignore"))
    dt.to_csv(f"{a.out}/table2_departments.csv")
    res["weibull_all"] = wb.loc["All"].to_dict()
    F.fig_duration(h, f"{a.out}/figure2_duration_residual_life.png")

    log("static models")
    sm = StaticModels(CAP).fit(trva)
    S = sm.survival(te)
    yt, et = te.dur_h.values, te.event.values
    km0 = sm.km_cond_mean(te.assign(e=0.0))
    res["c_index_onset"] = {"Kaplan–Meier (D×R)": concordance_index(yt, km0, et),
                            **{k: concordance_index(yt, v, et) for k, v in sm.onset_scores.items()}}
    log("landmark model")
    lm = LandmarkModel(CAP).fit(tr, va, oc)
    res["landmark_trees"] = lm.n_trees
    res["hyperparameters"] = dict(landmark=dict(lm.model.get_params(), n_estimators=lm.n_trees),
                                  onset_lightgbm={k: v for k, v in sm.gbm.get_params().items() if v is not None},
                                  onset_lightgbm_sigma=sm.gbm_sigma)
    res["landmark_importance"] = lm.importance().head(10).round(4).to_dict()

    log("landmark evaluation")
    pos = pd.Series(np.arange(len(te)), index=te.index)
    L = landmark_frame(te, oc, EVAL_LANDMARKS, CAP)
    p = pos.loc[L.hold_index].values
    for name, Sm in S.items():
        L["s_" + name] = sm.grid.cond_mean(Sm[p], L.e.values)
    L["s_Kaplan–Meier (D×R)"] = sm.km_cond_mean(L)
    L["s_Static p90 rule"] = sm.p90_ratio(L)
    L["_lm"] = lm.predict(L); L["_gbm"] = L["s_LightGBM (log T)"]
    L["s_Landmark GBM"] = L["_lm"]
    L["s_Ensemble (proposed)"] = L.groupby("e")["_lm"].rank(pct=True) + L.groupby("e")["_gbm"].rank(pct=True)
    models = ["Static p90 rule", "Kaplan–Meier (D×R)", "Cox PH", "Log-normal AFT", "XGBoost AFT", "LightGBM (log T)", "Landmark GBM", "Ensemble (proposed)"]
    M = landmark_metrics(L, {m: "s_" + m for m in models})
    M.to_csv(f"{a.out}/table4_landmark_metrics.csv", index=False)
    log("bootstrap of pooled landmark metrics (clusters = lots)")
    L["s_Ensemble pooled"] = L["_lm"].rank(pct=True) + L["_gbm"].rank(pct=True)
    cols = {m: ("s_Ensemble pooled" if m == "Ensemble (proposed)" else "s_" + m) for m in models}
    bci, bdiff = pooled_bootstrap(L, cols, B=a.lm_bootstrap,
                                  pairs=[("Ensemble (proposed)", "LightGBM (log T)"), ("Ensemble (proposed)", "Landmark GBM"),
                                         ("Landmark GBM", "Static p90 rule")])
    bci.to_csv(f"{a.out}/table4_pooled_ci.csv", index=False); bdiff.to_csv(f"{a.out}/table4_pooled_differences.csv", index=False)
    F.fig_landmark(M, f"{a.out}/figure3_landmark.png", ["Static p90 rule", "Kaplan–Meier (D×R)", "Cox PH", "LightGBM (log T)", "Landmark GBM", "Ensemble (proposed)"])

    log("escalation replay")
    C = review_candidates(te, oc)
    pc = pos.loc[C.hold_index].values
    C["s_Age (oldest first)"] = C.e
    C["s_Static p90 rule"] = sm.p90_rule(C)
    C["s_Kaplan–Meier (D×R)"] = sm.km_cond_mean(C)
    C["s_Cox PH"] = sm.grid.cond_mean(S["Cox PH"][pc], C.e.values)
    C["s_LightGBM (log T)"] = sm.grid.cond_mean(S["LightGBM (log T)"][pc], C.e.values)
    C["s_Landmark GBM"] = lm.predict(C)
    g = C.groupby("review")
    C["s_Ensemble (proposed)"] = g["s_Landmark GBM"].rank(pct=True) + g["s_LightGBM (log T)"].rank(pct=True)
    # due-date urgency u_j = exp(-slack_j / 3 days), slack measured at the review time (no look-ahead)
    if lots is not None and "due_time" in lots:
        slack_d = (C.lot_id.map(lots["due_time"]) - C.review).dt.total_seconds() / 86400
        C["urgency"] = np.exp(-np.clip(slack_d, 0, None) / URGENCY_DAYS).fillna(0)
    else:
        C["urgency"] = 0.0
    C["s_Ensemble + due-date urgency"] = C["s_Ensemble (proposed)"] * (1 + KAPPA * C.urgency.values)
    C["s_Oracle"] = C.R
    days = (C.review.max() - C.review.min()).total_seconds() / 86400
    total = te.dur_h.clip(upper=CAP).sum()
    policies = ["Age (oldest first)", "Static p90 rule", "Kaplan–Meier (D×R)", "Cox PH", "LightGBM (log T)", "Landmark GBM",
                "Ensemble (proposed)", "Ensemble + due-date urgency", "Oracle"]
    rows, recs = [], {}
    for B in [int(x) for x in a.budgets.split(",")]:
        b = max(1, B // 6)
        for pol in policies:
            rec = simulate(C, "s_" + pol, b, eligible_only=(pol == "Static p90 rule"))
            recs[(B, pol)] = rec
            rows.append(dict(policy=pol, budget=B, **summarise(rec, days, total, lots)))
        for pol in ["Static p90 rule", "LightGBM (log T)", "Landmark GBM", "Ensemble (proposed)", "Ensemble + due-date urgency", "Oracle"]:
            rel, lo, hi = block_bootstrap(recs[(B, pol)], recs[(B, "Age (oldest first)")], a.bootstrap)
            for r in rows:
                if r["policy"] == pol and r["budget"] == B:
                    r.update(rel_vs_age=rel, rel_lo=lo, rel_hi=hi)
    P = pd.DataFrame(rows); P.to_csv(f"{a.out}/table5_policies.csv", index=False)
    F.fig_frontier(P, f"{a.out}/figure4_frontier.png")
    sens = []
    for alpha in [0.3, 0.5, 0.8]:
        for delta in [1, 2, 4]:
            for pol in ["Age (oldest first)", "Static p90 rule", "Kaplan–Meier (D×R)", "Ensemble (proposed)", "Oracle"]:
                rec = simulate(C, "s_" + pol, 4, alpha, delta, eligible_only=(pol == "Static p90 rule"))
                sens.append(dict(alpha=alpha, delta=delta, policy=pol, saved_h_per_day=rec.saved_h.sum() / days))
    pd.DataFrame(sens).pivot_table(index=["alpha", "delta"], columns="policy", values="saved_h_per_day").to_csv(f"{a.out}/table6_sensitivity.csv")
    res.update(test_days=days, review_epochs=int(C.review.nunique()))
    if a.revision:
        revision_analyses(a, C, S, sm, tr, va, oc, pc, days, total, lots, recs, res)
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    log("done -> " + a.out)


def revision_analyses(a, C, S, sm, tr, va, oc, pc, days, total, lots, recs, res, delta0=2.0):
    """R1: index built from the expected gain E[(R - delta)^+] of Eq. (2) instead of the mean remaining time.
    R2: heterogeneous escalation effect, alpha and delta drawn per department x reason group."""
    log("revision R1: gain-based index")
    lmg = LandmarkModel(CAP, shift=delta0).fit(tr, va, oc)
    C["s_Landmark gain"] = lmg.predict(C)
    C["s_LightGBM gain"] = sm.grid.cond_gain(S["LightGBM (log T)"][pc], C.e.values, delta0)
    g = C.groupby("review")
    C["s_Ensemble (gain)"] = g["s_Landmark gain"].rank(pct=True) + g["s_LightGBM gain"].rank(pct=True)
    budgets = [int(x) for x in a.budgets.split(",")]
    rows = []
    for B in budgets:
        b = max(1, B // 6)
        for pol in ["Landmark gain", "LightGBM gain", "Ensemble (gain)"]:
            recs[(B, pol)] = simulate(C, "s_" + pol, b)
        for pol in ["Age (oldest first)", "Ensemble (proposed)", "Landmark gain", "LightGBM gain", "Ensemble (gain)", "Oracle"]:
            r = dict(policy=pol, budget=B, **summarise(recs[(B, pol)], days, total, lots))
            if pol != "Age (oldest first)":
                r.update(zip(["rel_vs_age", "rel_lo", "rel_hi"], block_bootstrap(recs[(B, pol)], recs[(B, "Age (oldest first)")], a.bootstrap)))
            if pol == "Ensemble (gain)":
                r.update(zip(["rel_vs_ensemble", "rel_vs_ens_lo", "rel_vs_ens_hi"],
                             block_bootstrap(recs[(B, pol)], recs[(B, "Ensemble (proposed)")], a.bootstrap)))
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"{a.out}/tableR1_gain_index.csv", index=False)

    log("revision R2: heterogeneous escalation effect")
    grp = C.department.astype(str) + "|" + C.reason.astype(str)
    groups = grp.unique()
    pols = ["Age (oldest first)", "Static p90 rule", "Kaplan–Meier (D×R)", "Ensemble (proposed)", "Ensemble (gain)", "Oracle"]
    rows = []
    for k in range(a.hetero_draws):
        rng = np.random.default_rng(100 + k)
        al = dict(zip(groups, rng.uniform(0.2, 0.8, len(groups))))
        de = dict(zip(groups, rng.uniform(1.0, 8.0, len(groups))))
        C["alpha_h"], C["delta_h"] = grp.map(al).values, grp.map(de).values
        # effect-aware benchmarks: they know alpha_g and delta_g (upper reference, not implementable today)
        C["s_LightGBM gain, effect-aware"] = C.alpha_h * sm.grid.cond_gain(S["LightGBM (log T)"][pc], C.e.values, C.delta_h.values)
        C["s_Oracle, effect-aware"] = C.alpha_h * np.maximum(np.minimum(C.R, CAP) - C.delta_h, 0)
        for B in budgets:
            b = max(1, B // 6)
            for pol in pols + ["LightGBM gain, effect-aware", "Oracle, effect-aware"]:
                rec = simulate(C, "s_" + pol, b, eligible_only=(pol == "Static p90 rule"), alpha_col="alpha_h", delta_col="delta_h")
                rows.append(dict(draw=k, budget=B, policy=pol, saved_h_per_day=rec.saved_h.sum() / days))
    R2 = pd.DataFrame(rows)
    R2.to_csv(f"{a.out}/tableR2_heterogeneous_draws.csv", index=False)
    base = R2[R2.policy == "Age (oldest first)"].set_index(["draw", "budget"]).saved_h_per_day
    R2["rel_vs_age"] = R2.saved_h_per_day.values / base.loc[list(zip(R2.draw, R2.budget))].values - 1
    summ = R2.groupby(["budget", "policy"]).agg(saved_mean=("saved_h_per_day", "mean"), rel_mean=("rel_vs_age", "mean"),
                                                rel_p05=("rel_vs_age", lambda x: np.percentile(x, 5)),
                                                rel_p95=("rel_vs_age", lambda x: np.percentile(x, 95)),
                                                share_draws_better_than_age=("rel_vs_age", lambda x: float((x > 0).mean())))
    summ.to_csv(f"{a.out}/tableR2_heterogeneous_summary.csv")
    C.drop(columns=["alpha_h", "delta_h"], inplace=True)
    # realised escalations per day and recovered time per escalation (Section 6.1)
    res["escalations_per_day"] = {f"{p}|{B}": len(recs[(B, p)]) / days for B in budgets for p in ["Age (oldest first)", "Ensemble (proposed)", "Static p90 rule"]}


if __name__ == "__main__":
    main()
