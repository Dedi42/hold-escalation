"""Budgeted escalation: index policies replayed on held-out holds.

At each review epoch (every `review_h` hours) the open, not yet escalated holds are
ranked by a policy score, and the top b = B * review_h / 24 are escalated. An
escalation ends the hold after `delta` hours with probability `alpha` (Eq. 1 of the
paper); the expected recovered hold time is alpha * max(min(R, cap) - delta, 0).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def review_candidates(test: pd.DataFrame, oc, review_h: float = 4.0, warmup_h: float = 24.0) -> pd.DataFrame:
    t0 = (test.start_time.min() + pd.Timedelta(hours=warmup_h)).ceil(f"{int(review_h)}h")
    reviews = pd.date_range(t0, test.start_time.max(), freq=f"{int(review_h)}h")
    st, rl = test.start_time.values, test.rel.values
    out = []
    for r in reviews:
        m = (st < r.to_datetime64()) & (rl > r.to_datetime64())
        c = test[m].copy()
        if c.empty:
            continue
        c["review"] = r
        c["e"] = (r - c.start_time).dt.total_seconds() / 3600
        c["R"] = (c.rel - r).dt.total_seconds() / 3600
        c["hold_index"] = c.index
        c["open_all_e"] = oc(np.full(len(c), r.value))
        c["open_dep_e"] = [oc(np.array([r.value]), d)[0] for d in c.department]
        c["hour_e"] = r.hour
        out.append(c)
    return pd.concat(out, ignore_index=True)


def simulate(C: pd.DataFrame, score: str, b: int, alpha=0.5, delta=2.0, cap=168.0, eligible_only=False,
             alpha_col: str | None = None, delta_col: str | None = None):
    """alpha_col / delta_col: optional columns of C with a hold-specific escalation effect
    (heterogeneous alpha and delta, e.g. per department x reason group). The policy score
    does not see them unless it is built from them."""
    escalated, records = set(), []
    for r, c in C.groupby("review", sort=True):
        c = c[~c.hold_index.isin(escalated)]
        if eligible_only:
            c = c[c[score] >= 0]
        if c.empty:
            continue
        pick = c.loc[c[score].nlargest(b).index]
        al = pick[alpha_col].values if alpha_col else np.full(len(pick), alpha)
        de = pick[delta_col].values if delta_col else np.full(len(pick), delta)
        for h, R, lot, a_, d_ in zip(pick.hold_index, pick.R, pick.lot_id, al, de):
            escalated.add(h)
            records.append((r.normalize(), lot, a_ * max(min(R, cap) - d_, 0.0)))
    return pd.DataFrame(records, columns=["day", "lot_id", "saved_h"])


def summarise(rec: pd.DataFrame, days: float, total_hold_h: float, lots: pd.DataFrame | None):
    out = dict(escalations=len(rec), saved_h_per_day=rec.saved_h.sum() / days,
               saved_per_escalation=rec.saved_h.mean() if len(rec) else 0,
               share_of_hold_hours=rec.saved_h.sum() / total_hold_h)
    if lots is not None and "due_time" in lots:
        s = rec.groupby("lot_id").saved_h.sum()
        L = lots.reindex(s.index)
        late = (L.completion_time - L.due_time).dt.total_seconds() / 3600
        late = late[late > 0]
        out["lateness_removed_h"] = float(np.minimum(s.reindex(late.index), late).sum())
        out["late_lots_recovered"] = int((s.reindex(late.index) >= late).sum())
    return out


def block_bootstrap(rec_a: pd.DataFrame, rec_b: pd.DataFrame, B: int = 2000, seed: int = 1):
    """Relative difference (a - b) / b of recovered hours with a weekly block bootstrap."""
    da, db = rec_a.groupby("day").saved_h.sum(), rec_b.groupby("day").saved_h.sum()
    idx = da.index.union(db.index)
    d = da.reindex(idx, fill_value=0) - db.reindex(idx, fill_value=0)
    wk = d.groupby(pd.Index(idx).to_period("W")).sum()
    base = db.reindex(idx, fill_value=0).groupby(pd.Index(idx).to_period("W")).sum().mean()
    rng = np.random.default_rng(seed)
    bs = [wk.values[rng.integers(0, len(wk), len(wk))].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.sum() / db.sum(), lo / base, hi / base


def simulate_flex(C: pd.DataFrame, score: str, B_daily: int, theta: float, alpha=0.5, delta=2.0, cap=168.0,
                  alpha_col: str | None = None, delta_col: str | None = None):
    """Flexible daily budget: the B_daily escalations of a calendar day can be spent at any review
    of that day. At each review, open and not yet escalated holds whose score is at least theta are
    escalated in order of their score while the day's budget lasts. Unused budget is lost at the
    end of the day. theta = -inf spends the budget at the first reviews of the day (greedy)."""
    escalated, records = set(), []
    day, left = None, 0
    for r, c in C.groupby("review", sort=True):
        d = r.normalize()
        if d != day:
            day, left = d, B_daily
        if left <= 0:
            continue
        c = c[~c.hold_index.isin(escalated)]
        c = c[c[score] >= theta]
        if c.empty:
            continue
        pick = c.loc[c[score].nlargest(left).index]
        left -= len(pick)
        al = pick[alpha_col].values if alpha_col else np.full(len(pick), alpha)
        de = pick[delta_col].values if delta_col else np.full(len(pick), delta)
        for h, R, lot, a_, d_ in zip(pick.hold_index, pick.R, pick.lot_id, al, de):
            escalated.add(h)
            records.append((d, lot, a_ * max(min(R, cap) - d_, 0.0)))
    return pd.DataFrame(records, columns=["day", "lot_id", "saved_h"])
