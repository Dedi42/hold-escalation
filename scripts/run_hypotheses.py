"""Pre-specified hypotheses of the hold-escalation study, tested per production line and across lines.

    python scripts/run_hypotheses.py --line A=config/local_lineA.yaml --line B=config/local_lineB.yaml --out results_hyp
    python scripts/run_hypotheses.py --line A=config/local_lineA.yaml --out results_hyp_A          # one line
    python scripts/run_hypotheses.py --line A=config/synthetic.yaml --out results_hyp_quick --quick   # smoke test

Hypotheses (H0 in brackets; one-sided alternatives; alpha = 0.05; Holm correction within each family)
  H1  Heavy tail: the longest 10% of holds carry more than half of the hold hours        [share <= 0.5]
  H2a Decreasing hazard of release: Weibull shape < 1                                   [shape >= 1]
  H2b Remaining time grows with elapsed time: median remaining after 8 h > at onset       [ratio <= 1]
  H3  Prediction (pooled AUC of remaining > 8 h on test landmarks, lot bootstrap):
      H3a ensemble > LightGBM onset   H3b ensemble > landmark GBM   H3c ensemble > Kaplan-Meier
      H3d landmark GBM > static p90 rule
  H4  Policy: ensemble index recovers more hold hours than oldest-first, per budget B   [ratio <= 1]
      H4-RO the same pooled over rolling origins (expanding window, K non-overlapping test windows)
  H5  Policy: static p90 rule recovers fewer hold hours than oldest-first, per budget    [ratio >= 1]
  H6  Robustness: H4 holds when the escalation effect differs by department x reason
      (share of random draws with gain > 0 and 5th percentile of the gain > 0; descriptive)
  H7  Equivalence: the index built from the expected gain E[(R - delta)^+] and the mean-based index
      recover the same hold hours within +/- margin (TOST)
  H8  Transfer: the ensemble trained on another line beats oldest-first on the target line (H8a),
      and is non-inferior to the in-line ensemble within the margin (H8b)
  H9  Consistency across lines: pooled (inverse-variance) gain of the ensemble over oldest-first > 0;
      Cochran's Q and I^2 report heterogeneity
  H10 Delivery (lines with due dates): ensemble + due-date urgency removes more lateness than oldest-first

Outputs in --out: hypotheses_summary.csv (one row per test with estimate, CI, p, Holm p, decision),
hypotheses.md (readable report), per-line tables, and fig_forest.png (gains over oldest-first).
Only anonymised aggregates are written.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from holdesc.io import load_config, load_holds, load_lots
from holdesc.anonymize import anonymize
from holdesc.features import build, temporal_split, purge_split, landmark_frame
from holdesc.survival import StaticModels, residual_life_table, weibull_shapes
from holdesc.landmark import LandmarkModel
from holdesc.evaluate import pooled_bootstrap
from holdesc.policy import review_candidates, simulate
from holdesc import stats as S

CAP, ALPHA0, DELTA0, KAPPA, URG_D = 168.0, 0.5, 2.0, 1.0, 3.0
EVAL_LM = [0.5, 1, 2, 4, 8, 16, 24, 48]
ROWS = []
LOGF = None


def log(m):
    line = time.strftime("%H:%M:%S ") + m
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n"); LOGF.flush()


def add(hyp, line, test, estimate, lo=np.nan, hi=np.nan, p=np.nan, family=None, unit="", note="", decision=None, **kw):
    ROWS.append(dict(hypothesis=hyp, line=line, test=test, estimate=estimate, ci_lo=lo, ci_hi=hi, p=p,
                     family=family or f"{hyp}|{line}", unit=unit, note=note, decision_rule=decision, **kw))


# ------------------------------------------------------------------------------------ data
def load_lines(specs):
    """Load every line, anonymise them jointly (shared code labels, so models transfer), split again."""
    hs, ls = [], []
    for name, path in specs:
        cfg = load_config(path)
        h = load_holds(cfg); h["line"] = name
        h["lot_id"] = name + ":" + h["lot_id"].astype(str)          # lot ids are unique within a line only
        hs.append(h)
        l = load_lots(cfg)
        if l is not None:
            l = l[~l.index.duplicated()]
            l.index = name + ":" + l.index.astype(str)
            ls.append(l)
    h_all = pd.concat(hs, ignore_index=True)
    lots_all = pd.concat(ls) if ls else None
    h_all, lots_all, _ = anonymize(h_all, lots_all)
    return {name: h_all[h_all.line == name].drop(columns="line").reset_index(drop=True) for name, _ in specs}, lots_all


def prepare(h_line, purge=False):
    h, oc = build(h_line)
    h["split"] = temporal_split(h)
    if purge:
        h["split"] = purge_split(h, h["split"])
    tr, va, te = (h[h.split == s] for s in ["train", "val", "test"])
    return dict(h=h, oc=oc, tr=tr, va=va, te=te)


def fit(tr, va, oc, shift=0.0):
    sm = StaticModels(CAP).fit(pd.concat([tr, va]))
    lm = LandmarkModel(CAP).fit(tr, va, oc)
    lmg = LandmarkModel(CAP, shift=shift).fit(tr, va, oc) if shift else None
    return sm, lm, lmg


def score_candidates(C, rows_df, sm, lm, lmg=None, lots=None):
    """Policy scores on review candidates C whose holds are rows of rows_df."""
    Sv = sm.survival(rows_df)
    pos = pd.Series(np.arange(len(rows_df)), index=rows_df.index).loc[C.hold_index].values
    sc = {"Age (oldest first)": C.e.values, "Static p90 rule": sm.p90_rule(C).astype(float),
          "Kaplan–Meier (D×R)": sm.km_cond_mean(C)}
    onset = sm.grid.cond_mean(Sv["LightGBM (log T)"][pos], C.e.values)
    land = lm.predict(C)
    g = C.assign(a=land, b=onset).groupby("review")
    sc["LightGBM onset"] = onset
    sc["Landmark GBM"] = land
    sc["Ensemble"] = g["a"].rank(pct=True).values + g["b"].rank(pct=True).values
    if lmg is not None:
        gain_on = sm.grid.cond_gain(Sv["LightGBM (log T)"][pos], C.e.values, DELTA0)
        gg = C.assign(a=lmg.predict(C), b=gain_on).groupby("review")
        sc["Ensemble (gain)"] = gg["a"].rank(pct=True).values + gg["b"].rank(pct=True).values
    if lots is not None and "due_time" in lots:
        slack = (C.lot_id.map(lots["due_time"]) - C.review).dt.total_seconds() / 86400
        u = np.exp(-np.clip(slack, 0, None) / URG_D).fillna(0).values
        sc["Ensemble + urgency"] = sc["Ensemble"] * (1 + KAPPA * u)
    sc["Oracle"] = C.R.values
    return sc


def replay(C, scores, budgets, **kw):
    Ce = C.assign(**{"s::" + k: v for k, v in scores.items()})
    return {(B, k): simulate(Ce, "s::" + k, max(1, B // 6), eligible_only=(k == "Static p90 rule"), **kw)
            for B in budgets for k in scores}


def day_index(C):
    return pd.date_range(C.review.min().normalize(), C.review.max().normalize(), freq="D")


def line_summary(name, h, tr, va, te, lots, a):
    """Table 2 of the paper: descriptive statistics of one line (no dates, no codes)."""
    days = (h.start_time.max() - h.start_time.min()).total_seconds() / 86400
    tdays = (te.start_time.max() - te.start_time.min()).total_seconds() / 86400
    dep = h.department.astype(str).value_counts(normalize=True)
    lot_ids = set(h.lot_id)
    sod = int(lots.loc[lots.index.isin(lot_ids), "due_time"].notna().sum()) if lots is not None and "due_time" in lots else 0
    row = dict(line=name, package_families=int(h.product_family.nunique()) if "product_family" in h else np.nan,
               months=round(days / 30.44), holds=len(h), lots=int(h.lot_id.nunique()), train=len(tr), val=len(va),
               test=len(te), test_days=round(tdays, 1), test_weeks=int(round(tdays / 7)),
               median_h=float(np.nanmedian(h.dur_h)), share_gt8h=float((h.dur_h > 8).mean()),
               departments=int(dep.size), largest_department_share=float(dep.iloc[0]), lots_with_sod=sod)
    pd.DataFrame([row]).to_csv(f"{a.out}/tables/{name}_line_summary.csv", index=False)


# ------------------------------------------------------------------------------------ per-line hypotheses
def analyze_line(name, D, lots, a, budgets):
    h, oc, tr, va, te = D["h"], D["oc"], D["tr"], D["va"], D["te"]
    log(f"[{name}] holds {len(h):,} (train {len(tr):,} / val {len(va):,} / test {len(te):,})")
    info = dict(line=name, holds=len(h), lots=int(h.lot_id.nunique()), train=len(tr), val=len(va), test=len(te),
                start=str(h.start_time.min().date()), end=str(h.start_time.max().date()))
    line_summary(name, h, tr, va, te, lots, a)

    # H1 heavy tail (lot bootstrap)
    def top10(d):
        x = np.sort(d.dur_h.values)[::-1]
        return x[:max(1, int(0.1 * len(x)))].sum() / x.sum()
    est = top10(h); bs = S.lot_bootstrap(h, top10, a.lm_bootstrap)
    add("H1", name, "share of hold hours in the longest 10% of holds > 0.5", est, *np.quantile(bs, [.025, .975]),
        S.one_sided_p(bs, 0.5), unit="share")

    # H2a Weibull shape < 1, H2b remaining time grows with elapsed time
    try:
        wb = weibull_shapes(h, []).loc["All"]
        add("H2a", name, "Weibull shape of hold duration < 1", wb.weibull_shape, wb.weibull_lo, wb.weibull_hi,
            S.weibull_p_less_than_one(wb.weibull_shape, wb.weibull_lo, wb.weibull_hi), unit="shape")
    except Exception:
        log(f"[{name}] Weibull fit failed:\n{traceback.format_exc()}")

    def ratio8(d):
        T = d.dur_h.values
        return np.median(T[T > 8] - 8) / np.median(T)
    est = ratio8(h); bs = S.lot_bootstrap(h, ratio8, a.lm_bootstrap)
    add("H2b", name, "median remaining after 8 h / median at onset > 1", est, *np.quantile(bs, [.025, .975]),
        S.one_sided_p(bs, 1.0), unit="ratio")
    residual_life_table(h).to_csv(f"{a.out}/tables/{name}_residual_life.csv", index=False)

    # models
    log(f"[{name}] fitting models")
    sm, lm, lmg = fit(tr, va, oc, shift=DELTA0)

    # H3 prediction on test landmarks
    L = landmark_frame(te, oc, EVAL_LM, CAP)
    pos = pd.Series(np.arange(len(te)), index=te.index).loc[L.hold_index].values
    Sv = sm.survival(te)
    L["s_onset"] = sm.grid.cond_mean(Sv["LightGBM (log T)"][pos], L.e.values)
    L["s_land"] = lm.predict(L)
    L["s_km"] = sm.km_cond_mean(L)
    L["s_p90"] = sm.p90_ratio(L)
    L["s_ens"] = L["s_land"].rank(pct=True) + L["s_onset"].rank(pct=True)
    cols = {"Ensemble": "s_ens", "LightGBM onset": "s_onset", "Landmark GBM": "s_land", "Kaplan–Meier": "s_km",
            "Static p90": "s_p90"}
    pairs = [("Ensemble", "LightGBM onset"), ("Ensemble", "Landmark GBM"), ("Ensemble", "Kaplan–Meier"),
             ("Landmark GBM", "Static p90")]
    ci, diffs = pooled_bootstrap(L, cols, B=a.lm_bootstrap, pairs=pairs)
    ci.to_csv(f"{a.out}/tables/{name}_prediction_ci.csv", index=False)
    for (x, y), lab in zip(pairs, ["H3a", "H3b", "H3c", "H3d"]):
        r = diffs[(diffs.model_a == x) & (diffs.model_b == y) & (diffs.metric == "AUC")].iloc[0]
        p1 = r.p_two_sided / 2 if r["diff"] > 0 else 1 - r.p_two_sided / 2
        add(lab, name, f"AUC(remaining > 8 h): {x} > {y}", r["diff"], r.lo, r.hi, p1, family=f"H3|{name}", unit="ΔAUC")

    # policy replay on the test period
    log(f"[{name}] policy replay")
    C = review_candidates(te, oc)
    days = day_index(C)
    sc = score_candidates(C, te, sm, lm, lmg, lots)
    recs = replay(C, sc, budgets)
    tab = []
    for (B, k), rec in recs.items():
        tab.append(dict(line=name, budget=B, policy=k, h_per_day=S.weekly_series(rec, days).mean(), escalations=len(rec)))
    pd.DataFrame(tab).to_csv(f"{a.out}/tables/{name}_policies.csv", index=False)
    forest = []
    for B in budgets:
        r = S.block_test(recs[(B, "Ensemble")], recs[(B, "Age (oldest first)")], days, a.bootstrap)
        add("H4", name, f"Ensemble > oldest first, B = {B}", r["rel"], r["rel_lo"], r["rel_hi"], r["p_greater"],
            family=f"H4|{name}", unit="relative gain", budget=B, weeks=r["weeks"], diff_h_per_day=r["diff_h_per_day"])
        forest.append(dict(line=name, budget=B, kind="test period", **{k: v for k, v in r.items() if k != "rel_bs"}))
        r5 = S.block_test(recs[(B, "Age (oldest first)")], recs[(B, "Static p90 rule")], days, a.bootstrap)
        loss = 1 / (1 + r5["rel_bs"]) - 1                       # p90 / age - 1
        add("H5", name, f"oldest first > static p90 rule, B = {B}", 1 / (1 + r5["rel"]) - 1,
            np.quantile(loss, .025), np.quantile(loss, .975), r5["p_greater"], family=f"H5|{name}",
            unit="p90 relative to oldest first", budget=B, weeks=r5["weeks"])
        # H7 equivalence of gain-based and mean-based index
        if "Ensemble (gain)" in sc:
            r7 = S.block_test(recs[(B, "Ensemble (gain)")], recs[(B, "Ensemble")], days, a.bootstrap)
            t = S.tost_bootstrap(r7["rel_bs"], a.margin)
            add("H7", name, f"gain-based ≈ mean-based index within ±{a.margin:.0%}, B = {B}", r7["rel"], t["ci90_lo"],
                t["ci90_hi"], t["p_tost"], family=f"H7|{name}", unit="relative difference (90% CI)", budget=B)
        # H10 delivery
        if "Ensemble + urgency" in sc and lots is not None and "due_time" in lots:
            add_h10(name, recs[(B, "Ensemble + urgency")], recs[(B, "Age (oldest first)")], lots, B, a)

    # H6 heterogeneous escalation effect
    log(f"[{name}] heterogeneous effect ({a.hetero_draws} draws)")
    grp = C.department.astype(str) + "|" + C.reason.astype(str); groups = grp.unique()
    Ce = C.assign(**{"s::" + k: v for k, v in sc.items()})
    for B in budgets:
        rels = []
        for k in range(a.hetero_draws):
            rng = np.random.default_rng(100 + k)
            al = dict(zip(groups, rng.uniform(0.2, 0.8, len(groups)))); de = dict(zip(groups, rng.uniform(1, 8, len(groups))))
            Ch = Ce.assign(alpha_h=grp.map(al).values, delta_h=grp.map(de).values)
            ra = simulate(Ch, "s::Ensemble", max(1, B // 6), alpha_col="alpha_h", delta_col="delta_h").saved_h.sum()
            rb = simulate(Ch, "s::Age (oldest first)", max(1, B // 6), alpha_col="alpha_h", delta_col="delta_h").saved_h.sum()
            rels.append(ra / max(rb, 1e-12) - 1)
        rels = np.array(rels)
        add("H6", name, f"Ensemble > oldest first under heterogeneous α, δ, B = {B}", rels.mean(),
            np.quantile(rels, .05), np.quantile(rels, .95), np.nan, family=f"H6|{name}", unit="relative gain (5–95% of draws)",
            budget=B, share_draws_positive=float((rels > 0).mean()),
            decision="supported if 5th percentile > 0")

    # H4-RO rolling origins
    if a.origins > 0:
        forest += rolling_origins(name, h, oc, lots, a, budgets)
    return info, forest, (sm, lm)


def add_h10(name, rec_a, rec_b, lots, B, a):
    late = (lots.completion_time - lots.due_time).dt.total_seconds() / 3600
    late = late[(late > 0) & late.index.astype(str).str.startswith(name + ":")]   # late lots of this line only
    def removed(rec):
        s = rec.groupby("lot_id").saved_h.sum().reindex(late.index, fill_value=0.0)
        return np.minimum(s.values, late.values)
    xa, xb = removed(rec_a), removed(rec_b)
    if len(late) == 0 or (xa.sum() + xb.sum()) == 0:
        return
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(late), (a.bootstrap, len(late)))
    d = xa[idx].sum(1) - xb[idx].sum(1)
    rel = xa.sum() / max(xb.sum(), 1e-12) - 1
    rel_bs = xa[idx].sum(1) / np.maximum(xb[idx].sum(1), 1e-12) - 1
    add("H10", name, f"Ensemble + urgency removes more lateness than oldest first, B = {B}", rel,
        np.quantile(rel_bs, .025), np.quantile(rel_bs, .975), (np.sum(d <= 0) + 1) / (a.bootstrap + 1),
        family=f"H10|{name}", unit="relative lateness removed", budget=B, late_lots=len(late))


def rolling_origins(name, h, oc, lots, a, budgets):
    """Expanding-window evaluation: K consecutive, non-overlapping test windows of W days at the end of the data;
    for each origin the models are refitted on the holds started before it (last 20% of them for validation)."""
    log(f"[{name}] rolling origins: {a.origins} windows of {a.window_days} days")
    t_end = h.start_time.max().normalize()
    recs_all = {(B, k): [] for B in budgets for k in ("Ensemble", "Age (oldest first)")}
    days_all = []
    for k in range(a.origins, 0, -1):
        t0 = t_end - pd.Timedelta(days=k * a.window_days)
        t1 = t0 + pd.Timedelta(days=a.window_days)
        past = h[h.start_time < t0]
        test = h[(h.start_time >= t0) & (h.start_time < t1)]
        if len(past) < 2000 or len(test) < 200:
            log(f"   origin {t0.date()}: too few holds, skipped"); continue
        cut = past.start_time.quantile(0.8)
        tr, va = past[past.start_time <= cut], past[past.start_time > cut]
        sm, lm, _ = fit(tr, va, oc)
        C = review_candidates(test, oc)
        C = C[C.review < t1]
        sc = score_candidates(C, test, sm, lm)
        sc = {k2: sc[k2] for k2 in ("Ensemble", "Age (oldest first)")}
        recs = replay(C, sc, budgets)
        for key, rec in recs.items():
            recs_all[key].append(rec)
        days_all.append(day_index(C))
        log(f"   origin {t0.date()}: train {len(tr):,}, test holds {len(test):,}")
    if not days_all:
        return []
    days = days_all[0]
    for d in days_all[1:]:
        days = days.union(d)
    out = []
    for B in budgets:
        ra = pd.concat(recs_all[(B, "Ensemble")]); rb = pd.concat(recs_all[(B, "Age (oldest first)")])
        r = S.block_test(ra, rb, days, a.bootstrap)
        add("H4-RO", name, f"Ensemble > oldest first over {len(days_all)} rolling origins, B = {B}", r["rel"], r["rel_lo"],
            r["rel_hi"], r["p_greater"], family=f"H4-RO|{name}", unit="relative gain", budget=B, weeks=r["weeks"])
        out.append(dict(line=name, budget=B, kind="rolling origins", **{k: v for k, v in r.items() if k != "rel_bs"}))
    return out


# ------------------------------------------------------------------------------------ transfer and pooling
def transfer(src, tgt, Ds, models, lots, a, budgets):
    log(f"transfer {src} -> {tgt}")
    sm, lm = models[src]
    T = Ds[tgt]
    C = review_candidates(T["te"], T["oc"])
    days = day_index(C)
    sc_t = score_candidates(C, T["te"], sm, lm)
    sm2, lm2 = models[tgt]
    sc_in = score_candidates(C, T["te"], sm2, lm2)
    scores = {"Transfer ensemble": sc_t["Ensemble"], "In-line ensemble": sc_in["Ensemble"], "Age (oldest first)": sc_in["Age (oldest first)"]}
    recs = replay(C, scores, budgets)
    for B in budgets:
        r = S.block_test(recs[(B, "Transfer ensemble")], recs[(B, "Age (oldest first)")], days, a.bootstrap)
        add("H8a", f"{src}→{tgt}", f"transferred ensemble > oldest first, B = {B}", r["rel"], r["rel_lo"], r["rel_hi"],
            r["p_greater"], family=f"H8a|{src}→{tgt}", unit="relative gain", budget=B, weeks=r["weeks"])
        r2 = S.block_test(recs[(B, "Transfer ensemble")], recs[(B, "In-line ensemble")], days, a.bootstrap)
        add("H8b", f"{src}→{tgt}", f"transferred ensemble non-inferior to in-line ensemble (margin {a.margin:.0%}), B = {B}",
            r2["rel"], r2["rel_lo"], r2["rel_hi"], S.noninferior(r2["rel_bs"], a.margin), family=f"H8b|{src}→{tgt}",
            unit="relative difference", budget=B, weeks=r2["weeks"])


def pool_lines(forest, budgets, min_weeks=6):
    """H9: inverse-variance pooling across lines. Rolling-origin estimates are pooled when available (more test
    weeks); single test periods are pooled only when every line has at least min_weeks weekly blocks, because the
    block-bootstrap standard error is unreliable with very few weeks."""
    F = pd.DataFrame(forest)
    for kind in ("rolling origins", "test period"):
        for B in budgets:
            g = F[(F.budget == B) & (F.kind == kind)]
            if len(g) < 2:
                continue
            if (g.weeks < min_weeks).any():
                log(f"H9 {kind} B={B}: skipped, a line has fewer than {min_weeks} test weeks")
                continue
            m = S.fixed_effect(g.log_ratio.values, g.log_ratio_se.values)
            add("H9", "pooled", f"pooled gain over oldest first ({kind}, {len(g)} lines), B = {B}", m["rel"], m["rel_lo"],
                m["rel_hi"], m["p_greater"], family=f"H9|{kind}", unit="relative gain", budget=B, Q=m["Q"], Q_p=m["Q_p"], I2=m["I2"])
            F = pd.concat([F, pd.DataFrame([dict(line="pooled", budget=B, kind=kind, rel=m["rel"], rel_lo=m["rel_lo"],
                                                 rel_hi=m["rel_hi"])])], ignore_index=True)
    return F


# ------------------------------------------------------------------------------------ report
def finalize(a, infos, F):
    R = pd.DataFrame(ROWS)
    R["p_holm"] = np.nan
    for fam, g in R.groupby("family"):
        ok = g.p.notna()
        if ok.any():
            R.loc[g.index[ok], "p_holm"] = S.holm(g.p[ok].values)
    def decide(r):
        if r.hypothesis == "H6":
            return "supported" if r.ci_lo > 0 else "not supported"
        if r.hypothesis == "H7":
            return "equivalent" if r.p_holm < a.alpha else "not shown"
        if r.hypothesis in ("H4", "H5", "H8a", "H8b") and r.get("weeks", 99) < a.min_weeks:
            return ("supported" if r.p_holm < a.alpha else "not supported") + f" (only {int(r.weeks)} test weeks)"
        return "supported" if (not np.isnan(r.p_holm) and r.p_holm < a.alpha) else "not supported"
    R["decision"] = R.apply(decide, axis=1)
    R.to_csv(f"{a.out}/hypotheses_summary.csv", index=False)
    F.to_csv(f"{a.out}/tables/forest_gain_vs_age.csv", index=False)
    try:
        forest_plot(F, f"{a.out}/fig_forest.png")
    except Exception:
        log("forest plot failed:\n" + traceback.format_exc())
    L = ["# Hypothesis tests\n", f"alpha = {a.alpha}; Holm correction within each family (hypothesis × line); "
         f"bootstrap replicates: {a.bootstrap} (policies, weekly blocks), {a.lm_bootstrap} (lots); equivalence margin ±{a.margin:.0%}.\n",
         "## Data\n", pd.DataFrame(infos).to_string(index=False), "\n## Verdicts\n"]
    show = R.copy()
    for c in ["estimate", "ci_lo", "ci_hi"]:
        show[c] = show[c].astype(float).round(4)
    show["p"] = show.p.map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    show["p_holm"] = show.p_holm.map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    L.append(show[["hypothesis", "line", "test", "estimate", "ci_lo", "ci_hi", "p", "p_holm", "decision"]].to_string(index=False))
    L.append("\n## Overall verdict per hypothesis and line\n")
    ov = R.groupby(["hypothesis", "line"]).decision.agg(lambda s: f"{(s.isin(['supported', 'equivalent'])).sum()}/{len(s)} tests supported")
    L.append(ov.to_string())
    with open(f"{a.out}/hypotheses.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def forest_plot(F, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = F.dropna(subset=["rel"]).reset_index(drop=True)
    F["label"] = F.line + " · " + F.kind.fillna("") + " · B=" + F.budget.astype(str)
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(F) + 1))
    y = np.arange(len(F))[::-1]
    col = np.where(F.line == "pooled", "#1f4e9a", np.where(F.kind == "rolling origins", "#b8621b", "#444444"))
    ax.errorbar(100 * F.rel, y, xerr=[100 * (F.rel - F.rel_lo), 100 * (F.rel_hi - F.rel)], fmt="none", ecolor=col, lw=1)
    ax.scatter(100 * F.rel, y, c=col, s=18, zorder=3, marker="o")
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(F.label, fontsize=7)
    ax.set_xlabel("Gain of the ensemble index over oldest-first (%), 95% CI")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", action="append", required=True, help="NAME=config.yaml (repeat for several lines)")
    ap.add_argument("--out", default="results_hyp"); ap.add_argument("--budgets", default="6,12,24,48")
    ap.add_argument("--bootstrap", type=int, default=2000); ap.add_argument("--lm-bootstrap", dest="lm_bootstrap", type=int, default=500)
    ap.add_argument("--hetero-draws", dest="hetero_draws", type=int, default=20)
    ap.add_argument("--origins", type=int, default=4); ap.add_argument("--window-days", dest="window_days", type=int, default=14)
    ap.add_argument("--margin", type=float, default=0.03); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-weeks", dest="min_weeks", type=int, default=6, help="minimum weekly blocks for a reliable bootstrap")
    ap.add_argument("--purge", action="store_true"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.bootstrap, a.lm_bootstrap, a.hetero_draws, a.origins, a.budgets = 200, 50, 3, 2, "6,24"
    os.makedirs(f"{a.out}/tables", exist_ok=True)
    LOGF = open(f"{a.out}/hypotheses.log", "a", encoding="utf-8")
    budgets = [int(x) for x in a.budgets.split(",")]
    specs = [tuple(s.split("=", 1)) for s in a.line]
    t0 = time.time()
    lines, lots = load_lines(specs)
    Ds, infos, forest, models = {}, [], [], {}
    for name, _ in specs:
        try:
            Ds[name] = prepare(lines[name], a.purge)
            info, fo, mdl = analyze_line(name, Ds[name], lots, a, budgets)
            infos.append(info); forest += fo; models[name] = mdl
        except Exception:
            log(f"!! line {name} failed:\n{traceback.format_exc()}")
    names = list(models)
    for s in names:
        for t in names:
            if s != t:
                try:
                    transfer(s, t, Ds, models, lots, a, budgets)
                except Exception:
                    log(f"!! transfer {s}->{t} failed:\n{traceback.format_exc()}")
    F = pool_lines(forest, budgets, a.min_weeks) if forest else pd.DataFrame()
    finalize(a, infos, F)
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}/hypotheses.md")


if __name__ == "__main__":
    main()
