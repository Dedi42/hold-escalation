"""Features known at hold onset and at a landmark time (no look-ahead)."""
from __future__ import annotations
import numpy as np
import pandas as pd

NUM = ["priority", "quantity", "rework", "non_production", "hour", "weekday", "lot_prev_n",
       "lot_prev_h", "same_step_repeat", "open_all", "open_dep", "burst_reason"]
CAT = ["department", "reason", "step", "product_family", "lot_type"]
LM_EXTRA = ["e", "open_all_e", "open_dep_e", "hour_e"]


class OpenCounter:
    """Number of holds open at a given time, overall or per department."""

    def __init__(self, h: pd.DataFrame):
        self.s = np.sort(h.start_time.values.astype("int64"))
        self.r = np.sort(h.rel.values.astype("int64"))
        self.dep = {d: (np.sort(g.start_time.values.astype("int64")), np.sort(g.rel.values.astype("int64")))
                    for d, g in h.groupby("department")}

    def __call__(self, t: np.ndarray, dep: str | None = None) -> np.ndarray:
        s, r = (self.s, self.r) if dep is None else self.dep[dep]
        return np.searchsorted(s, t, "right") - np.searchsorted(r, t, "right")


def build(h: pd.DataFrame, end_time: pd.Timestamp | None = None) -> tuple[pd.DataFrame, OpenCounter]:
    h = h.sort_values("start_time").reset_index(drop=True).copy()
    end_time = end_time or max(h.start_time.max(), h.release_time.max())
    h["rel"] = h.release_time.fillna(end_time)
    h["event"] = h.release_time.notna().astype(int)
    h["dur_h"] = (h.rel - h.start_time).dt.total_seconds() / 3600
    h = h[h.dur_h > 0].reset_index(drop=True)
    h["hour"] = h.start_time.dt.hour
    h["weekday"] = h.start_time.dt.weekday
    h["non_production"] = (h.lot_type != h.lot_type.value_counts().index[0]).astype(int)
    # earlier holds of the same lot that were already released at onset
    prev_h = np.zeros(len(h)); prev_n = np.zeros(len(h))
    for _, g in h.groupby("lot_id"):
        st, rl, d, idx = g.start_time.values, g.rel.values, g.dur_h.values, g.index.values
        for k in range(1, len(g)):
            m = rl[:k] <= st[k]
            prev_h[idx[k]] = d[:k][m].sum(); prev_n[idx[k]] = m.sum()
    h["lot_prev_h"], h["lot_prev_n"] = prev_h, prev_n
    h["same_step_repeat"] = h.groupby(["lot_id", "step"]).cumcount()
    oc = OpenCounter(h)
    t = h.start_time.values.astype("int64")
    h["open_all"] = oc(t)
    h["open_dep"] = 0
    for d, g in h.groupby("department"):
        h.loc[g.index, "open_dep"] = oc(t[g.index], d)
    h["burst_reason"] = 0
    day = int(24 * 3600e9)
    for _, g in h.groupby("reason"):
        s = g.start_time.values.astype("int64")
        h.loc[g.index, "burst_reason"] = np.searchsorted(s, s, "left") - np.searchsorted(s, s - day, "left")
    return h, oc


def temporal_split(h: pd.DataFrame, train=0.6, val=0.2) -> pd.Series:
    q = h.start_time.quantile([train, train + val]).values
    return pd.Series(np.where(h.start_time <= q[0], "train", np.where(h.start_time <= q[1], "val", "test")), index=h.index)


def purge_split(h: pd.DataFrame, split: pd.Series) -> pd.Series:
    """Purged temporal split: drop training holds still open at the start of the validation
    period and validation holds still open at the start of the test period, so that no
    training or validation label uses information from a later period."""
    out = split.copy()
    v0 = h.start_time[split == "val"].min(); t0 = h.start_time[split == "test"].min()
    out[(split == "train") & (h.rel > v0)] = "purged"
    out[(split == "val") & (h.rel > t0)] = "purged"
    return out


def landmark_frame(h: pd.DataFrame, oc: OpenCounter, landmarks, cap: float) -> pd.DataFrame:
    """One row per hold still open at each landmark e; target = min(T, cap) - e."""
    rows = []
    for e in landmarks:
        g = h[h.dur_h > e].copy()
        if g.empty:
            continue
        t = (g.start_time + pd.to_timedelta(e, unit="h")).values.astype("int64")
        g["e"] = e
        g["open_all_e"] = oc(t)
        g["open_dep_e"] = 0
        for d, gg in g.groupby("department"):
            g.loc[gg.index, "open_dep_e"] = oc(t[g.index.get_indexer(gg.index)], d)
        g["hour_e"] = ((g.hour + e) % 24).astype(int)
        g["remaining"] = np.minimum(g.dur_h, cap) - e
        g["hold_index"] = g.index
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def encode(df: pd.DataFrame, cats: dict, cols) -> pd.DataFrame:
    X = df[cols].copy()
    for c in CAT:
        X[c] = pd.Categorical(df[c], categories=cats[c])
    return X
