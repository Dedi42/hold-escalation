"""Decision-time context of an open hold (no look-ahead).

Two groups of features are computed for (hold, decision time t) pairs:

* regime features: for the hold's reason, department, and step, the number of holds of the
  same group open at t, started in the 24 h before t, and released in the window before t,
  together with the mean log duration and the share of long (> 8 h) holds among those
  released. They describe the current backlog and recent disposition speed of the group,
  which the static and landmark features of the paper do not capture;
* lot history: the previous holds of the same lot that started before t (at most K, most
  recent last), as indices into the hold table for sequence models, and as aggregates for
  tabular models.

Only information known at t is used: a previous hold contributes its duration only if it
was released by t, otherwise its elapsed time at t.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

GROUPS = ["reason", "department", "step"]
NS_H = 3600e9


def _ns(x) -> np.ndarray:
    return np.asarray(pd.to_datetime(x).values if not isinstance(x, np.ndarray) else x).astype("datetime64[ns]").astype("int64")


def regime_features(h: pd.DataFrame, hold_idx: np.ndarray, t: np.ndarray, window_h: float = 168.0,
                    long_h: float = 8.0) -> pd.DataFrame:
    """Backlog and recent disposition speed of the hold's reason, department and step at time t."""
    t = np.asarray(t, dtype="int64")
    st_all = _ns(h.start_time.values); rl_all = _ns(h.rel.values)
    logd = np.log1p(h.dur_h.values); long_ = (h.dur_h.values > long_h).astype(float)
    out = {}
    w, d24 = int(window_h * NS_H), int(24 * NS_H)
    for g in GROUPS:
        key_h = h[g].astype(str).to_numpy(dtype=object)
        key_q = key_h[hold_idx]
        cols = {f"{g}_open": np.zeros(len(t)), f"{g}_start24h": np.zeros(len(t)), f"{g}_rel_n": np.zeros(len(t)),
                f"{g}_rel_mlog": np.full(len(t), np.nan), f"{g}_rel_plong": np.full(len(t), np.nan)}
        order_q = pd.Series(np.arange(len(t))).groupby(key_q).indices
        grp_h = pd.Series(np.arange(len(h))).groupby(key_h).indices
        for v, qi in order_q.items():
            hi = grp_h.get(v, np.array([], dtype=int))
            s = np.sort(st_all[hi])
            ro = np.argsort(rl_all[hi]); r = rl_all[hi][ro]
            cl = np.r_[0.0, np.cumsum(logd[hi][ro])]; cg = np.r_[0.0, np.cumsum(long_[hi][ro])]
            tq = t[qi]
            n_start = np.searchsorted(s, tq, "right")
            n_rel = np.searchsorted(r, tq, "right")
            cols[f"{g}_open"][qi] = n_start - n_rel
            cols[f"{g}_start24h"][qi] = n_start - np.searchsorted(s, tq - d24, "right")
            lo = np.searchsorted(r, tq - w, "right")
            n = n_rel - lo
            cols[f"{g}_rel_n"][qi] = n
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"{g}_rel_mlog"][qi] = np.where(n > 0, (cl[n_rel] - cl[lo]) / np.maximum(n, 1), np.nan)
                cols[f"{g}_rel_plong"][qi] = np.where(n > 0, (cg[n_rel] - cg[lo]) / np.maximum(n, 1), np.nan)
        out.update(cols)
    return pd.DataFrame(out)


def lot_history(h: pd.DataFrame, hold_idx: np.ndarray, t: np.ndarray, K: int = 16):
    """Previous holds of the same lot that started before t.

    Returns (hist, agg): hist is an (n, K) int array of row positions in h (-1 = padding,
    most recent last); agg is a DataFrame of aggregates over all previous holds."""
    t = np.asarray(t, dtype="int64")
    st = _ns(h.start_time.values); rl = _ns(h.rel.values); dur = h.dur_h.values
    reason = h.reason.astype(str).to_numpy(dtype=object); step = h.step.astype(str).to_numpy(dtype=object)
    n = len(t)
    hist = np.full((n, K), -1, dtype=np.int64)
    agg = {k: np.zeros(n) for k in ["hist_n", "hist_open", "hist_known_h", "hist_max_h", "hist_same_reason",
                                    "hist_same_step", "hist_last_gap_h"]}
    agg["hist_last_gap_h"][:] = np.nan
    lot_arr = h.lot_id.astype(str).to_numpy(dtype=object)
    lots_h = pd.Series(np.arange(len(h))).groupby(lot_arr).indices
    lot_q = lot_arr[hold_idx]
    for lot, qi in pd.Series(np.arange(n)).groupby(lot_q).indices.items():
        hi = lots_h[lot]
        hi = hi[np.argsort(st[hi], kind="stable")]
        s = st[hi]
        for q in qi:
            self_ = hold_idx[q]; tq = t[q]
            k = np.searchsorted(s, tq, "right")
            prev = hi[:k]
            prev = prev[prev != self_]
            if len(prev) == 0:
                continue
            known = np.where(rl[prev] <= tq, dur[prev], (tq - st[prev]) / NS_H)
            agg["hist_n"][q] = len(prev)
            agg["hist_open"][q] = (rl[prev] > tq).sum()
            agg["hist_known_h"][q] = known.sum()
            agg["hist_max_h"][q] = known.max()
            agg["hist_same_reason"][q] = (reason[prev] == reason[self_]).sum()
            agg["hist_same_step"][q] = (step[prev] == step[self_]).sum()
            agg["hist_last_gap_h"][q] = (tq - st[prev].max()) / NS_H
            take = prev[-K:]
            hist[q, K - len(take):] = take
    return hist, pd.DataFrame(agg)


def history_event_features(h: pd.DataFrame, hist: np.ndarray, t: np.ndarray, hold_idx: np.ndarray) -> np.ndarray:
    """Numeric features of each history event at decision time t: log known duration, released
    flag, log age at t, same step, same reason. Shape (n, K, 5); zeros at padding."""
    t = np.asarray(t, dtype="int64")
    st = _ns(h.start_time.values); rl = _ns(h.rel.values); dur = h.dur_h.values
    reason = h.reason.astype(str).to_numpy(dtype=object); step = h.step.astype(str).to_numpy(dtype=object)
    m = hist >= 0
    j = np.where(m, hist, 0)
    tq = t[:, None]
    released = (rl[j] <= tq)
    known = np.where(released, dur[j], (tq - st[j]) / NS_H)
    age = (tq - st[j]) / NS_H
    f = np.stack([np.log1p(np.maximum(known, 0)), released.astype(float), np.log1p(np.maximum(age, 0)),
                  (step[j] == step[hold_idx][:, None]).astype(float),
                  (reason[j] == reason[hold_idx][:, None]).astype(float)], axis=-1)
    return (f * m[..., None]).astype(np.float32)
