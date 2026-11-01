"""Model inputs shared by every learner in the benchmark (identical features for GBM and DL).

Feature sets
  paper : static attributes at onset + elapsed time + workload at the decision time (Sections 4.2-4.3)
  ext   : paper + regime features + lot-history aggregates (holdesc.context)
The sequence model additionally receives the raw lot history (holdesc.context.lot_history).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import NUM, CAT, LM_EXTRA
from .context import regime_features, lot_history, history_event_features, GROUPS

CAP = 168.0
EDGES = np.array([0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 120, 144, 168.0])
PAPER_NUM = NUM + LM_EXTRA
REGIME = [f"{g}_{k}" for g in GROUPS for k in ["open", "start24h", "rel_n", "rel_mlog", "rel_plong"]]
HIST_AGG = ["hist_n", "hist_open", "hist_known_h", "hist_max_h", "hist_same_reason", "hist_same_step", "hist_last_gap_h"]
LOG_COLS = {"lot_prev_h", "lot_prev_n", "open_all", "open_dep", "burst_reason", "same_step_repeat", "e", "open_all_e",
            "open_dep_e", "hist_n", "hist_open", "hist_known_h", "hist_max_h", "hist_same_reason", "hist_same_step",
            "hist_last_gap_h"} | {c for c in REGIME if c.endswith(("_open", "_start24h", "_rel_n"))}
CYCLIC = {"hour": 24, "weekday": 7, "hour_e": 24}


def decision_time(rows: pd.DataFrame) -> np.ndarray:
    """Decision time of each row: review time for replay rows, onset + e for landmark rows."""
    if "review" in rows:
        t = pd.to_datetime(rows["review"])
    else:
        t = rows.start_time + pd.to_timedelta(rows.e, unit="h")
    return t.values.astype("datetime64[ns]").astype("int64")


def add_context(h: pd.DataFrame, rows: pd.DataFrame, K: int = 16) -> tuple[pd.DataFrame, dict]:
    """Attach regime features and history aggregates; return the raw history for sequence models."""
    t = decision_time(rows)
    idx = rows.hold_index.values.astype(int)
    reg = regime_features(h, idx, t)
    hist, agg = lot_history(h, idx, t, K)
    out = pd.concat([rows.reset_index(drop=True), reg, agg], axis=1)
    out["R_true"] = (out["dur_h"].values - out["e"].values) if "R" not in out else out["R"].values
    out["event"] = h.event.values[idx]
    seq = dict(hist=hist, ev=history_event_features(h, hist, t, idx))
    return out, seq


def gain_target(R: np.ndarray, delta: float | np.ndarray) -> np.ndarray:
    return np.maximum(np.minimum(R, CAP) - delta, 0.0)


class Prep:
    """Fit on training rows only: log transform, standardisation, NaN flags, categorical vocabularies."""

    def __init__(self, num_cols, cat_cols=CAT, min_count: int = 5):
        self.num_cols, self.cat_cols, self.min_count = list(num_cols), list(cat_cols), min_count

    def _num(self, df):
        X = {}
        for c in self.num_cols:
            v = pd.to_numeric(df[c], errors="coerce").astype(float).values
            if c in CYCLIC:
                X[c + "_sin"] = np.sin(2 * np.pi * v / CYCLIC[c]); X[c + "_cos"] = np.cos(2 * np.pi * v / CYCLIC[c])
                continue
            if c in LOG_COLS:
                v = np.log1p(np.maximum(v, 0))
            X[c] = v
        return pd.DataFrame(X)

    def fit(self, df: pd.DataFrame, h: pd.DataFrame):
        X = self._num(df)
        self.nan_cols = [c for c in X if X[c].isna().any()]
        self.mu, self.sd = X.mean(), X.std().replace(0, 1).fillna(1)
        self.vocab = {}
        for c in self.cat_cols:
            vc = df[c].astype(str).value_counts()
            keep = vc[vc >= self.min_count].index
            self.vocab[c] = {v: i + 2 for i, v in enumerate(keep)}  # 0 = padding, 1 = unknown
        self.h_codes = {g: self._codes(h[g], g) for g in GROUPS}
        self.num_names = list(X.columns) + [c + "_nan" for c in self.nan_cols]
        return self

    def _codes(self, s, c):
        return s.astype(str).map(self.vocab[c]).fillna(1).astype(np.int64).values

    def transform(self, df: pd.DataFrame, seq: dict | None = None) -> dict:
        X = self._num(df)
        flags = np.column_stack([X[c].isna().values for c in self.nan_cols]) if self.nan_cols else np.zeros((len(X), 0))
        X = ((X - self.mu) / self.sd).fillna(0.0)
        D = dict(num=np.hstack([X.values, flags]).astype(np.float32),
                 cat=np.column_stack([self._codes(df[c], c) for c in self.cat_cols]).astype(np.int64))
        if seq is not None:
            hist = seq["hist"]; m = hist >= 0; j = np.where(m, hist, 0)
            D["seq_cat"] = np.stack([np.where(m, self.h_codes[g][j], 0) for g in GROUPS], axis=-1).astype(np.int64)
            D["seq_num"] = seq["ev"]
            D["mask"] = ~m  # True = padding
        return D

    @property
    def cards(self):
        return [len(self.vocab[c]) + 2 for c in self.cat_cols]

    @property
    def seq_cards(self):
        return [len(self.vocab[g]) + 2 for g in GROUPS]


def _rates(S, edges):
    """Constant hazard within each bin implied by the survival curve at the bin edges."""
    w = np.diff(edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -np.log(np.clip(S[:, 1:], 1e-300, None) / np.clip(S[:, :-1], 1e-300, None)) / w
    return np.nan_to_num(np.maximum(lam, 0.0), nan=0.0, posinf=1e6), w


def _part(S0, lam, x):
    """Integral of S0 * exp(-lam * u) for u in [0, x]."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v = S0 * -np.expm1(-lam * x) / lam
    return np.where(lam > 1e-12, v, S0 * x)


def surv_integral(S: np.ndarray, upper, edges: np.ndarray = EDGES) -> np.ndarray:
    """Integral from 0 to upper of the survival curve with constant hazard within bins (S: n x len(edges))."""
    n = S.shape[0]
    upper = np.clip(np.broadcast_to(np.asarray(upper, float), (n,)), 0, edges[-1])
    lam, w = _rates(S, edges)
    full = _part(S[:, :-1], lam, w[None, :])
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(full, axis=1)], axis=1)
    k = np.clip(np.searchsorted(edges, upper, "right") - 1, 0, len(w) - 1)
    r = np.arange(n)
    return cum[r, k] + _part(S[r, k], lam[r, k], upper - edges[k])


def surv_at(S: np.ndarray, x, edges: np.ndarray = EDGES) -> np.ndarray:
    """P(R > x) with constant hazard within bins."""
    n = S.shape[0]
    x = np.clip(np.broadcast_to(np.asarray(x, float), (n,)), 0, edges[-1])
    lam, w = _rates(S, edges)
    k = np.clip(np.searchsorted(edges, x, "right") - 1, 0, len(w) - 1)
    r = np.arange(n)
    return S[r, k] * np.exp(-lam[r, k] * (x - edges[k]))


def mean_remaining(S):
    return surv_integral(S, CAP)


def expected_gain(S, delta):
    return surv_integral(S, CAP) - surv_integral(S, delta)


def group_index(review_values: np.ndarray):
    """Rows must be sorted by review; returns start offsets and sizes of each review group."""
    codes = np.unique(review_values, return_inverse=True)[1].ravel()
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    sizes = np.diff(np.r_[starts, len(codes)])
    return starts, sizes
