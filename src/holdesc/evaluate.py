"""Discrimination metrics for onset and landmark predictions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score


def landmark_metrics(L: pd.DataFrame, score_cols: dict, threshold_h: float = 8.0) -> pd.DataFrame:
    rows = []
    for e in sorted(L.e.unique()) + ["pooled"]:
        s = L if e == "pooled" else L[L.e == e]
        y = s.remaining.values
        for name, col in score_cols.items():
            p = s[col].values
            if e == "pooled" and name == "Ensemble (proposed)":
                p = s["_lm"].rank(pct=True).values + s["_gbm"].rank(pct=True).values
            ok = len(np.unique(p)) > 1
            auc = roc_auc_score((y > threshold_h).astype(int), p) if ok and 0 < (y > threshold_h).mean() < 1 else np.nan
            k = max(1, int(0.1 * len(s)))
            rows.append(dict(elapsed_h=e, model=name, n=len(s), C=concordance_index(y, p) if ok else 0.5,
                             AUC_8h=auc, top10_capture=y[np.argsort(-p)[:k]].sum() / y.sum()))
    return pd.DataFrame(rows)


def pooled_bootstrap(L: pd.DataFrame, score_cols: dict, B: int = 200, seed: int = 1, threshold_h: float = 8.0, pairs=()):
    """Cluster (lot) bootstrap of pooled C-index and AUC, and of paired differences between models."""
    rng = np.random.default_rng(seed)
    lots = L.lot_id.values
    uniq, inv = np.unique(lots, return_inverse=True)
    rows_of = pd.Series(np.arange(len(L))).groupby(inv).apply(np.array).values
    y = L.remaining.values; yb = (y > threshold_h).astype(int)
    S = {m: L[c].values for m, c in score_cols.items()}
    stats = {m: {"C": [], "AUC": []} for m in S}
    for _ in range(B):
        idx = np.concatenate(rows_of[rng.integers(0, len(uniq), len(uniq))])
        for m, p in S.items():
            stats[m]["C"].append(concordance_index(y[idx], p[idx]))
            stats[m]["AUC"].append(roc_auc_score(yb[idx], p[idx]))
    ci = []
    for m in S:
        ci.append(dict(model=m, C=concordance_index(y, S[m]), C_lo=np.percentile(stats[m]["C"], 2.5), C_hi=np.percentile(stats[m]["C"], 97.5),
                       AUC=roc_auc_score(yb, S[m]), AUC_lo=np.percentile(stats[m]["AUC"], 2.5), AUC_hi=np.percentile(stats[m]["AUC"], 97.5)))
    diffs = []
    for a, b in pairs:
        for k in ["C", "AUC"]:
            d = np.array(stats[a][k]) - np.array(stats[b][k])
            diffs.append(dict(model_a=a, model_b=b, metric=k, diff=np.mean(d), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                              p_two_sided=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    return pd.DataFrame(ci), pd.DataFrame(diffs)
