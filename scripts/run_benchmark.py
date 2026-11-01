"""Fair benchmark of baselines, tuned gradient boosting, and deep learning for hold escalation.

    python scripts/run_benchmark.py --config config/local.yaml --out results_bench
    python scripts/run_benchmark.py --config config/synthetic.yaml --out results_bench_quick --quick

Protocol (identical for every learned model)
  * same temporal split as the paper (60/20/20 by onset; --purge for the purged split);
  * learners are fitted on the training period only; the validation period is used for early
    stopping and for model selection; the test period is touched once;
  * the same number of random-search trials per learner family (--trials), each scored by the
    same decision-aligned criterion: hold hours recovered per day in a replay of the validation
    period, averaged over budgets 6 and 24 (alpha = 0.5, delta = 2 h);
  * the selected configuration is refitted with --seeds seeds; tables report the mean and SD
    over seeds and the seed ensemble (mean prediction), compared with age-based escalation and
    with the paper's ensemble by a weekly block bootstrap;
  * the paper baselines (age, p90 rule, Kaplan-Meier, LightGBM at onset, landmark GBM, ensemble)
    are computed exactly as in run_pipeline.py and refitted on training + validation data, which
    favours them.

Families (see --models): lgb_lm_paper, lgb_lm_ext, lgb_gain_ext, lgb_rank_ext, mlp_paper, mlp_ext, ft_ext,
lotseq_ext, lotseq_rank. Only anonymised aggregates are written to --out/tables; --out/cache holds
per-hold predictions and must not be shared.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, sys, time, traceback
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from holdesc.io import load_config, load_holds, load_lots
from holdesc.anonymize import anonymize
from holdesc.features import build, temporal_split, purge_split, landmark_frame, CAT
from holdesc.survival import StaticModels
from holdesc.landmark import LandmarkModel
from holdesc.policy import review_candidates, simulate, simulate_flex, summarise, block_bootstrap
from holdesc.tabprep import (Prep, add_context, PAPER_NUM, REGIME, HIST_AGG, CAP, gain_target, surv_at,
                             mean_remaining, expected_gain, group_index)
from holdesc import bench_gbm as G
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

DELTA0, ALPHA0 = 2.0, 0.5
BENCH_LM = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 24, 48, 72, 96, 168]
EVAL_LM = [0.5, 1, 2, 4, 8, 16, 24, 48]
EXT_NUM = PAPER_NUM + REGIME + HIST_AGG
FAMILIES = {  # name: (kind, feature set, architecture)
    "lgb_lm_paper": ("gbm_mean", "paper", None), "lgb_lm_ext": ("gbm_mean", "ext", None),
    "lgb_gain_ext": ("gbm_gain", "ext", None), "lgb_rank_ext": ("gbm_rank", "ext", None),
    "mlp_paper": ("surv", "paper", "mlp"), "mlp_ext": ("surv", "ext", "mlp"), "ft_ext": ("surv", "ext", "ft"),
    "lotseq_ext": ("surv", "ext", "lotseq"), "lotseq_rank": ("rank", "ext", "lotseq"),
}
LABEL = {"lgb_lm_paper": "LightGBM landmark (paper features)", "lgb_lm_ext": "LightGBM landmark (ext. features)",
         "lgb_gain_ext": "LightGBM gain regression (ext.)", "lgb_rank_ext": "LightGBM LambdaRank (ext.)",
         "mlp_paper": "MLP survival (paper features)", "mlp_ext": "MLP survival (ext.)",
         "ft_ext": "FT-Transformer survival (ext.)", "lotseq_ext": "LotSeq survival (ext. + history)",
         "lotseq_rank": "LotSeq decision-focused ranker"}

LOGF = None


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n"); LOGF.flush()


def h8(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


class Cache:
    def __init__(self, root):
        self.root = root; os.makedirs(root, exist_ok=True)

    def path(self, key):
        return os.path.join(self.root, key + ".pkl")

    def get(self, key):
        p = self.path(key)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def put(self, key, obj):
        with open(self.path(key) + ".tmp", "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(self.path(key) + ".tmp", self.path(key))


# ------------------------------------------------------------------------------------ evaluation helpers
def replay(C, score_col, budgets, days, alpha=ALPHA0, delta=DELTA0):
    return {B: simulate(C, score_col, max(1, B // 6), alpha, delta) for B in budgets}


def per_day(rec, days):
    return rec.saved_h.sum() / days


def val_criterion(Cva, scores, days_va, budgets=(6, 24)):
    C = Cva.assign(_s=scores)
    return float(np.mean([per_day(simulate(C, "_s", max(1, B // 6)), days_va) for B in budgets]))


def within_review_rank(C, col):
    return C.groupby("review")[col].rank(pct=True).values


def pred_metrics(L, score, S=None, thr=8.0):
    """Pooled and per-landmark discrimination; integrated Brier score when a survival curve is given."""
    rows = []
    y = L.R_true.values
    yc = np.minimum(y, CAP)
    for e in sorted(L.e.unique()) + ["pooled"]:
        m = np.ones(len(L), bool) if e == "pooled" else (L.e.values == e)
        if e == "pooled":
            p = L.assign(_p=score).groupby("e")["_p"].rank(pct=True).values  # pooled over landmarks: ranks per e
        else:
            p = score[m]
        yy, ev = yc[m], L.event.values[m]
        ok = len(np.unique(p)) > 1
        lab = (y[m] > thr).astype(int)
        k = max(1, int(0.1 * m.sum()))
        rows.append(dict(elapsed_h=e, n=int(m.sum()), C=concordance_index(yy, p, ev) if ok else 0.5,
                         AUC_8h=roc_auc_score(lab, p) if ok and 0 < lab.mean() < 1 else np.nan,
                         top10_capture=yy[np.argsort(-p)[:k]].sum() / yy.sum()))
    out = pd.DataFrame(rows)
    if S is not None:
        hs = np.array([1, 2, 4, 8, 16, 24, 48, 96])
        bs = [np.mean((surv_at(S, hh) - (y > hh)) ** 2) for hh in hs]
        out["IBS"] = np.trapezoid(bs, hs) / (hs[-1] - hs[0]) if hasattr(np, "trapezoid") else np.trapz(bs, hs) / (hs[-1] - hs[0])
    return out


# ------------------------------------------------------------------------------------ hyperparameter spaces
def sample_dl(arch, rng, kind):
    hp = dict(lr=float(rng.choice([3e-4, 1e-3, 2e-3])), wd=float(rng.choice([1e-6, 1e-5, 1e-4])),
              dropout=float(rng.choice([0.0, 0.1, 0.2])))
    if arch in ("mlp", "lotseq"):
        hp.update(hidden=int(rng.choice([128, 256, 512])), layers=int(rng.choice([2, 3])))
    if arch == "ft":
        hp.update(d=int(rng.choice([32, 64])), layers=int(rng.choice([2, 3])), heads=4)
    if arch == "lotseq":
        hp.update(d_seq=int(rng.choice([32, 64])), seq_layers=int(rng.choice([1, 2])), heads=4)
    if kind == "rank":
        hp.update(lr=float(rng.choice([1e-4, 3e-4, 1e-3])), power=float(rng.choice([0.5, 1.0])),
                  warm=bool(rng.choice([True, False])))
    return hp


# ------------------------------------------------------------------------------------ main
def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", default="results_bench")
    ap.add_argument("--models", default=",".join(FAMILIES)); ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--budgets", default="6,12,24,48")
    ap.add_argument("--bootstrap", type=int, default=2000); ap.add_argument("--hetero-draws", dest="hetero_draws", type=int, default=20)
    ap.add_argument("--device", default="auto"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5); ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--K", type=int, default=16, help="max previous holds per lot for LotSeq")
    ap.add_argument("--rank-review-h", dest="rank_review_h", type=float, default=2.0)
    ap.add_argument("--purge", action="store_true"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.quick:
        a.trials, a.seeds, a.epochs, a.patience, a.bootstrap, a.hetero_draws = 2, 2, 3, 2, 100, 2
        a.budgets = "6,24"
    os.makedirs(f"{a.out}/tables", exist_ok=True)
    LOGF = open(f"{a.out}/benchmark.log", "a", encoding="utf-8")
    cache = Cache(f"{a.out}/cache")
    budgets = [int(x) for x in a.budgets.split(",")]
    fams = [m.strip() for m in a.models.split(",") if m.strip()]
    log(f"benchmark start: families={fams} trials={a.trials} seeds={a.seeds} quick={a.quick}")
    t_start = time.time()

    # ---------------- data (identical to run_pipeline.py)
    cfg = load_config(a.config)
    h_raw, lots_raw = load_holds(cfg), load_lots(cfg)
    h, lots, _ = anonymize(h_raw, lots_raw)
    del h_raw, lots_raw
    h, oc = build(h)
    h["split"] = temporal_split(h)
    if a.purge:
        h["split"] = purge_split(h, h["split"])
    tr, va, te = (h[h.split == s] for s in ["train", "val", "test"])
    trva = pd.concat([tr, va])
    log(f"holds train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    # ---------------- frames with decision-time context (cached)
    fkey = f"frames_{h8([len(h), a.purge, a.K, a.rank_review_h, BENCH_LM])}"
    fr = cache.get(fkey)
    if fr is None:
        log("building frames (landmarks, review epochs, context features, lot history)")
        fr = {}
        for name, d, lms in [("Ltr", tr, BENCH_LM), ("Lva", va, BENCH_LM), ("Lte", te, EVAL_LM)]:
            fr[name] = add_context(h, landmark_frame(d, oc, lms, CAP), a.K)
        for name, d, rh in [("Ctr", tr, a.rank_review_h), ("Cva", va, 4.0), ("Cte", te, 4.0)]:
            fr[name] = add_context(h, review_candidates(d, oc, review_h=rh), a.K)
        cache.put(fkey, fr)
    (Ltr, Str), (Lva, Sva), (Lte, Ste) = fr["Ltr"], fr["Lva"], fr["Lte"]
    (Ctr, SCtr), (Cva, SCva), (Cte, SCte) = fr["Ctr"], fr["Cva"], fr["Cte"]
    days_va = (Cva.review.max() - Cva.review.min()).total_seconds() / 86400
    days_te = (Cte.review.max() - Cte.review.min()).total_seconds() / 86400
    total_te = te.dur_h.clip(upper=CAP).sum()
    log(f"frames: Ltr {len(Ltr)}, Lva {len(Lva)}, Lte {len(Lte)}, Ctr {len(Ctr)}, Cva {len(Cva)}, Cte {len(Cte)}; "
        f"test days {days_te:.1f}")

    # ---------------- paper baselines (as in run_pipeline.py)
    log("paper baselines")
    sm = StaticModels(CAP).fit(trva)
    S_te = sm.survival(te)
    pos = pd.Series(np.arange(len(te)), index=te.index)
    lm = LandmarkModel(CAP).fit(tr, va, oc)
    base_L, base_C = {}, {}
    pL, pC = pos.loc[Lte.hold_index].values, pos.loc[Cte.hold_index].values
    base_L["Static p90 rule"] = sm.p90_ratio(Lte).astype(float)
    base_L["Kaplan–Meier (D×R)"] = sm.km_cond_mean(Lte)
    base_L["LightGBM (log T)"] = sm.grid.cond_mean(S_te["LightGBM (log T)"][pL], Lte.e.values)
    base_L["Landmark GBM"] = lm.predict(Lte)
    base_L["Ensemble (paper)"] = (Lte.assign(a=base_L["Landmark GBM"]).groupby("e")["a"].rank(pct=True).values +
                                  Lte.assign(b=base_L["LightGBM (log T)"]).groupby("e")["b"].rank(pct=True).values)
    base_C["Age (oldest first)"] = Cte.e.values
    base_C["Static p90 rule"] = sm.p90_rule(Cte).astype(float)
    base_C["Kaplan–Meier (D×R)"] = sm.km_cond_mean(Cte)
    base_C["LightGBM (log T)"] = sm.grid.cond_mean(S_te["LightGBM (log T)"][pC], Cte.e.values)
    base_C["Landmark GBM"] = lm.predict(Cte)
    Ctmp = Cte.assign(a=base_C["Landmark GBM"], b=base_C["LightGBM (log T)"])
    base_C["Ensemble (paper)"] = within_review_rank(Ctmp, "a") + within_review_rank(Ctmp, "b")
    base_C["Oracle"] = Cte.R_true.values
    base_C_va = {"Age (oldest first)": Cva.e.values}

    # ---------------- inputs
    preps = {"paper": Prep(PAPER_NUM).fit(Ltr, h), "ext": Prep(EXT_NUM).fit(Ltr, h)}
    D = {fs: {k: p.transform(df, seq) for k, (df, seq) in
              dict(Ltr=(Ltr, Str), Lva=(Lva, Sva), Lte=(Lte, Ste), Ctr=(Ctr, SCtr), Cva=(Cva, SCva), Cte=(Cte, SCte)).items()}
         for fs, p in preps.items()}
    gcats = {c: pd.Categorical(Ltr[c].astype(str)).categories for c in CAT}
    for f_ in (Ltr, Lva, Lte, Ctr, Cva, Cte):
        for c in CAT:
            f_[c] = f_[c].astype(str)
    X = {fs: {k: G.matrix(df, PAPER_NUM if fs == "paper" else EXT_NUM, gcats) for k, df in
              dict(Ltr=Ltr, Lva=Lva, Lte=Lte, Ctr=Ctr, Cva=Cva, Cte=Cte).items()} for fs in preps}
    targets = dict(Ltr=np.minimum(Ltr.R_true.values, CAP), Lva=np.minimum(Lva.R_true.values, CAP))
    gtr_rank, gva_rank = gain_target(Ctr.R_true.values, DELTA0), gain_target(Cva.R_true.values, DELTA0)
    grp_tr, grp_va = group_index(Ctr.review.values), group_index(Cva.review.values)
    meta = {fs: dict(n_num=D[fs]["Ltr"]["num"].shape[1], cards=preps[fs].cards, seq_cards=preps[fs].seq_cards,
                     n_seq_num=D[fs]["Ltr"]["seq_num"].shape[2]) for fs in preps}

    dl_needed = any(FAMILIES[f][0] in ("surv", "rank") for f in fams)
    if dl_needed:
        try:
            import torch
        except ImportError:
            log("!! PyTorch is not installed (pip install -r requirements-dl.txt); deep families are skipped")
            fams = [f for f in fams if FAMILIES[f][0] not in ("surv", "rank")]
            dl_needed = False
    if dl_needed:
        from holdesc import deep as DL
        if a.threads:
            torch.set_num_threads(a.threads)
        dev = DL.device_auto(a.device)
        log(f"torch {torch.__version__} on {dev}")
        ylab = {k: DL.surv_labels(df.R_true.values, df.event.values) for k, df in dict(Ltr=Ltr, Lva=Lva).items()}

    # ---------------- one fit = one (family, hp, seed); returns predictions on all evaluation frames
    surv_trunks = {}

    def fit_one(fam, hp, seed):
        kind, fs, arch = FAMILIES[fam]
        Dx, Xx = D[fs], X[fs]
        out = dict(hp=hp, seed=seed)
        t0 = time.time()
        if kind in ("gbm_mean", "gbm_gain"):
            ytr, yva = targets["Ltr"], targets["Lva"]
            if kind == "gbm_gain":
                ytr, yva = gain_target(Ltr.R_true.values, DELTA0), gain_target(Lva.R_true.values, DELTA0)
            m = G.fit_reg(Xx["Ltr"], ytr, Xx["Lva"], yva, hp, seed)
            for k in ("Lte", "Cva", "Cte"):
                out[k] = m.predict(Xx[k])
            out["n_trees"] = m.best_iteration_
        elif kind == "gbm_rank":
            m = G.fit_rank(Xx["Ctr"], gtr_rank, grp_tr[1], Xx["Cva"], gva_rank, grp_va[1], hp, seed)
            for k in ("Lte", "Cva", "Cte"):
                out[k] = m.predict(Xx[k])
            out["n_trees"] = m.best_iteration_
        elif kind == "surv":
            model, info = DL.train_surv(arch, meta[fs], hp, Dx["Ltr"], ylab["Ltr"], Dx["Lva"], ylab["Lva"], seed, dev,
                                        a.epochs, a.patience, a.batch, log)
            for k in ("Lte", "Cva", "Cte"):
                out["S_" + k] = DL.predict_surv(model, arch, Dx[k], dev).astype(np.float32)
            out["Lte"] = mean_remaining(out["S_Lte"])
            out["Cva"] = expected_gain(out["S_Cva"], DELTA0)
            out["Cte"] = expected_gain(out["S_Cte"], DELTA0)
            out["Cte_mean"] = mean_remaining(out["S_Cte"])
            out.update(epochs=info["epochs"], best_val=info["best_val"], params=DL.n_params(model))
            if arch == "lotseq":
                out["trunk"] = {k: v.detach().cpu() for k, v in model.trunk.state_dict().items()}
        elif kind == "rank":
            init = None
            if hp.get("warm"):
                init = surv_trunks.get((fs, seed), surv_trunks.get((fs, 1)))
                if init is not None and surv_trunks.get(("hp", fs)) is not None:
                    hp = {**surv_trunks[("hp", fs)], "lr": hp["lr"], "wd": hp["wd"], "power": hp["power"], "warm": True}
                    out["hp"] = hp
                else:
                    init = None
            wtr, wva = gtr_rank ** hp["power"], gva_rank ** hp["power"]
            model, info = DL.train_ranker(arch, meta[fs], hp, Dx["Ctr"], grp_tr, wtr.astype(np.float32), Dx["Cva"], grp_va,
                                          wva.astype(np.float32), seed, dev, a.epochs, a.patience, init_trunk=init, log=log)
            for k in ("Lte", "Cva", "Cte"):
                out[k] = DL.predict_score(model, arch, Dx[k], dev)
            out.update(epochs=info["epochs"], best_val=info["best_val"], params=DL.n_params(model))
        out["val_criterion"] = val_criterion(Cva, out["Cva"], days_va)
        out["sec"] = time.time() - t0
        return out

    # ---------------- tuning (same budget per family) and seeds
    tuning_rows, final = [], {}
    order = sorted(fams, key=lambda f: 1 if FAMILIES[f][0] == "rank" else 0)  # survival trunks before the ranker
    for fam in order:
        kind, fs, arch = FAMILIES[fam]
        rng = np.random.default_rng(sum(map(ord, fam)))
        best = None
        for trial in range(a.trials):
            hp = G.sample_hp(rng) if kind.startswith("gbm") else sample_dl(arch, rng, kind)
            key = f"tune_{fam}_{h8([hp, fkey, a.epochs, a.purge])}"
            res = cache.get(key)
            if res is None:
                log(f"  tune {fam} trial {trial + 1}/{a.trials}: {hp}")
                try:
                    res = fit_one(fam, hp, seed=0)
                    res = {k: v for k, v in res.items() if not k.startswith("S_") and k not in ("Lte", "Cte", "Cte_mean", "trunk")}
                    cache.put(key, res)
                except Exception:
                    log(f"  !! {fam} trial failed:\n{traceback.format_exc()}")
                    continue
            tuning_rows.append(dict(family=fam, trial=trial, val_criterion=res["val_criterion"], sec=res.get("sec"),
                                    epochs=res.get("epochs"), n_trees=res.get("n_trees"), hp=json.dumps(res["hp"])))
            log(f"    {fam} trial {trial + 1}: val recovered {res['val_criterion']:.2f} h/day")
            if best is None or res["val_criterion"] > best["val_criterion"]:
                best = res
        if best is None:
            log(f"  !! {fam}: no successful trial, skipped")
            continue
        log(f"  {fam}: best val {best['val_criterion']:.2f} h/day with {best['hp']}")
        runs = []
        for seed in range(1, a.seeds + 1):
            key = f"final_{fam}_{h8([best['hp'], fkey, a.epochs, a.purge])}_s{seed}"
            res = cache.get(key)
            if res is None:
                log(f"  final {fam} seed {seed}/{a.seeds}")
                try:
                    res = fit_one(fam, best["hp"], seed)
                    cache.put(key, res)
                except Exception:
                    log(f"  !! {fam} seed {seed} failed:\n{traceback.format_exc()}")
                    continue
            if kind == "surv" and arch == "lotseq" and "trunk" in res:
                surv_trunks[(fs, seed)] = res["trunk"]; surv_trunks[("hp", fs)] = best["hp"]
            runs.append(res)
        if runs:
            final[fam] = runs
    pd.DataFrame(tuning_rows).to_csv(f"{a.out}/tables/bench_tuning.csv", index=False)

    # ---------------- prediction metrics on test landmarks
    log("evaluation: prediction")
    prow = []
    for name, sc in base_L.items():
        M = pred_metrics(Lte, np.asarray(sc, float))
        M["model"], M["seed"] = name, "-"
        prow.append(M)
    for fam, runs in final.items():
        for r in runs:
            M = pred_metrics(Lte, r["Lte"], r.get("S_Lte"))
            M["model"], M["seed"] = LABEL[fam], r["seed"]
            prow.append(M)
        ens = np.mean([pd.Series(r["Lte"]).rank(pct=True).values for r in runs], axis=0)
        M = pred_metrics(Lte, ens); M["model"], M["seed"] = LABEL[fam], "ensemble"
        prow.append(M)
    P = pd.concat(prow, ignore_index=True)
    P.to_csv(f"{a.out}/tables/bench_prediction_all.csv", index=False)
    pooled = P[P.elapsed_h == "pooled"]
    agg = pooled[pooled.seed.astype(str) != "ensemble"].groupby("model").agg(
        C_mean=("C", "mean"), C_sd=("C", "std"), AUC8_mean=("AUC_8h", "mean"), AUC8_sd=("AUC_8h", "std"),
        top10_mean=("top10_capture", "mean"), **({"IBS_mean": ("IBS", "mean")} if "IBS" in pooled else {}))
    agg.to_csv(f"{a.out}/tables/bench_prediction_pooled.csv")

    # ---------------- policy replay on the test period
    log("evaluation: policy replay")
    score_te = dict(base_C)
    score_va = dict(base_C_va)
    for fam, runs in final.items():
        score_te[LABEL[fam]] = np.mean([r["Cte"] for r in runs], axis=0)
        score_va[LABEL[fam]] = np.mean([r["Cva"] for r in runs], axis=0)
        if FAMILIES[fam][0] == "surv":
            score_te[LABEL[fam] + " [mean score]"] = np.mean([r["Cte_mean"] for r in runs], axis=0)
    dl_fams = [f for f in final if FAMILIES[f][0] in ("surv", "rank")]
    best_dl = max(dl_fams, key=lambda f: np.mean([r["val_criterion"] for r in final[f]])) if dl_fams else None
    if best_dl and "lgb_lm_ext" in final:
        Ct = Cte.assign(a=score_te[LABEL[best_dl]], b=score_te[LABEL["lgb_lm_ext"]])
        score_te[f"Ensemble new ({LABEL[best_dl]} + LightGBM ext.)"] = within_review_rank(Ct, "a") + within_review_rank(Ct, "b")
    Ce = Cte.copy()
    for k, v in score_te.items():
        Ce["s::" + k] = v
    recs, rows = {}, []
    for B in budgets:
        b = max(1, B // 6)
        for k in score_te:
            recs[(B, k)] = simulate(Ce, "s::" + k, b, eligible_only=(k == "Static p90 rule"))
        for k in score_te:
            r = dict(policy=k, budget=B, **summarise(recs[(B, k)], days_te, total_te, lots))
            if k != "Age (oldest first)":
                r.update(zip(["rel_vs_age", "lo_age", "hi_age"], block_bootstrap(recs[(B, k)], recs[(B, "Age (oldest first)")], a.bootstrap)))
            if k not in ("Ensemble (paper)", "Age (oldest first)"):
                r.update(zip(["rel_vs_paper_ens", "lo_ens", "hi_ens"], block_bootstrap(recs[(B, k)], recs[(B, "Ensemble (paper)")], a.bootstrap)))
            rows.append(r)
    # seed spread of the recovered hours
    for fam, runs in final.items():
        for B in budgets:
            vals = [per_day(simulate(Ce.assign(_s=r["Cte"]), "_s", max(1, B // 6)), days_te) for r in runs]
            for r in rows:
                if r["policy"] == LABEL[fam] and r["budget"] == B:
                    r.update(seed_mean=float(np.mean(vals)), seed_sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan)
    PT = pd.DataFrame(rows)
    PT.to_csv(f"{a.out}/tables/bench_policy.csv", index=False)

    # ---------------- flexible daily budget (absolute scores only; threshold chosen on validation)
    log("evaluation: flexible daily budget")
    flex_rows = []
    absolute = ["Age (oldest first)"] + [LABEL[f] for f in final if FAMILIES[f][0] in ("surv", "gbm_mean", "gbm_gain")]
    Cv = Cva.copy()
    for k in absolute:
        if k not in score_va:
            continue
        Cv["s::" + k] = score_va[k]
        qs = np.unique(np.quantile(score_va[k], [0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]))
        for B in budgets:
            cand = [(-np.inf, per_day(simulate_flex(Cv, "s::" + k, B, -np.inf), days_va))]
            cand += [(q, per_day(simulate_flex(Cv, "s::" + k, B, q), days_va)) for q in qs]
            theta, vbest = max(cand, key=lambda x: x[1])
            rec = simulate_flex(Ce, "s::" + k, B, theta)
            rel, lo, hi = block_bootstrap(rec, recs[(B, k)], a.bootstrap)
            flex_rows.append(dict(policy=k, budget=B, theta=theta, val_h_per_day=vbest, fixed_h_per_day=per_day(recs[(B, k)], days_te),
                                  flex_h_per_day=per_day(rec, days_te), rel_flex_vs_fixed=rel, lo=lo, hi=hi,
                                  escalations_per_day=len(rec) / days_te))
    pd.DataFrame(flex_rows).to_csv(f"{a.out}/tables/bench_flex_budget.csv", index=False)

    # ---------------- heterogeneous escalation effect (distributional models can use alpha_g and delta_g)
    log("evaluation: heterogeneous escalation effect")
    grp = Cte.department.astype(str) + "|" + Cte.reason.astype(str)
    groups = grp.unique()
    hrows = []
    surv_best = max([f for f in final if FAMILIES[f][0] == "surv"], key=lambda f: np.mean([r["val_criterion"] for r in final[f]]),
                    default=None)
    for k in range(a.hetero_draws):
        rng = np.random.default_rng(100 + k)
        al = dict(zip(groups, rng.uniform(0.2, 0.8, len(groups)))); de = dict(zip(groups, rng.uniform(1.0, 8.0, len(groups))))
        Ch = Ce.assign(alpha_h=grp.map(al).values, delta_h=grp.map(de).values)
        pols = ["Age (oldest first)", "Ensemble (paper)"] + ([LABEL[best_dl]] if best_dl else []) + \
               ([LABEL["lgb_gain_ext"]] if "lgb_gain_ext" in final else [])
        if surv_best:
            S_mean = np.mean([r["S_Cte"] for r in final[surv_best]], axis=0)
            Ch["s::effect-aware " + LABEL[surv_best]] = Ch.alpha_h.values * expected_gain(S_mean, Ch.delta_h.values)
            pols.append("effect-aware " + LABEL[surv_best])
        Ch["s::Oracle, effect-aware"] = Ch.alpha_h * np.maximum(np.minimum(Ch.R_true, CAP) - Ch.delta_h, 0)
        pols += ["Oracle", "Oracle, effect-aware"]
        for B in budgets:
            for p_ in pols:
                rec = simulate(Ch, "s::" + p_, max(1, B // 6), alpha_col="alpha_h", delta_col="delta_h")
                hrows.append(dict(draw=k, budget=B, policy=p_, saved_h_per_day=per_day(rec, days_te)))
    H = pd.DataFrame(hrows)
    if len(H):
        base = H[H.policy == "Age (oldest first)"].set_index(["draw", "budget"]).saved_h_per_day
        H["rel_vs_age"] = H.saved_h_per_day.values / base.loc[list(zip(H.draw, H.budget))].values - 1
        H.groupby(["budget", "policy"]).agg(saved_mean=("saved_h_per_day", "mean"), rel_mean=("rel_vs_age", "mean"),
                                           rel_p05=("rel_vs_age", lambda x: np.percentile(x, 5)),
                                           rel_p95=("rel_vs_age", lambda x: np.percentile(x, 95)),
                                           share_better=("rel_vs_age", lambda x: float((x > 0).mean()))).to_csv(
            f"{a.out}/tables/bench_hetero.csv")

    # ---------------- runtime and model size
    rt = [dict(model=LABEL[f], seeds=len(r), sec_per_fit=np.mean([x["sec"] for x in r]),
               epochs=np.mean([x.get("epochs") or np.nan for x in r]), params=r[0].get("params"),
               n_trees=np.mean([x.get("n_trees") or np.nan for x in r]), hp=json.dumps(r[0]["hp"])) for f, r in final.items()]
    pd.DataFrame(rt).to_csv(f"{a.out}/tables/bench_runtime.csv", index=False)

    # ---------------- figure and summary
    try:
        write_figure(PT, budgets, f"{a.out}/fig_bench_frontier.png")
    except Exception:
        log("figure failed:\n" + traceback.format_exc())
    write_summary(a, agg, PT, pd.DataFrame(flex_rows), budgets, best_dl, time.time() - t_start,
                  dict(train=len(tr), val=len(va), test=len(te), Ltr=len(Ltr), Ctr=len(Ctr), test_days=days_te))
    log(f"done in {(time.time() - t_start) / 60:.1f} min -> {a.out}")


def write_figure(PT, budgets, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    show = PT[~PT.policy.str.contains(r"\[mean score\]")]
    for k, g in show.groupby("policy"):
        g = g.sort_values("budget")
        ls = "--" if k in ("Oracle",) else ("-" if "Ensemble" in k or "LotSeq" in k else ":")
        ax.plot(g.budget, g.saved_h_per_day, ls, marker="o", ms=3, lw=1, label=k)
    ax.set_xscale("log", base=2); ax.set_xticks(budgets); ax.set_xticklabels(budgets)
    ax.set_xlabel("Escalation budget B (holds per day)"); ax.set_ylabel("Hold hours recovered per day")
    ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def write_summary(a, agg, PT, FX, budgets, best_dl, sec, sizes):
    L = [f"# Benchmark results\n", f"- sizes: {sizes}", f"- trials per family: {a.trials}; seeds: {a.seeds}; "
         f"epochs <= {a.epochs}; purge: {a.purge}; quick: {a.quick}", f"- best DL family on validation: {best_dl}",
         f"- runtime: {sec / 60:.1f} min\n", "## Prediction on test landmarks (pooled, mean over seeds)\n",
         agg.round(4).to_string(), "",
         "## Hold hours recovered per day (seed ensemble) and relative difference to age and to the paper ensemble\n"]
    piv = PT.pivot_table(index="policy", columns="budget", values="saved_h_per_day").round(1)
    L.append(piv.to_string()); L.append("")
    for B in budgets:
        s = PT[PT.budget == B][["policy", "rel_vs_age", "lo_age", "hi_age", "rel_vs_paper_ens", "lo_ens", "hi_ens"]].copy()
        for c in s.columns[1:]:
            s[c] = (100 * s[c]).round(1)
        L.append(f"### B = {B} (percent)\n"); L.append(s.to_string(index=False)); L.append("")
    if len(FX):
        L.append("## Flexible daily budget vs fixed per review (percent change)\n")
        f = FX[["policy", "budget", "fixed_h_per_day", "flex_h_per_day", "rel_flex_vs_fixed", "lo", "hi"]].copy()
        for c in ["rel_flex_vs_fixed", "lo", "hi"]:
            f[c] = (100 * f[c]).round(1)
        L.append(f.round(2).to_string(index=False))
    with open(f"{a.out}/results_benchmark.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
