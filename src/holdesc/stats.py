"""Statistical tests for the pre-specified hypotheses (run_hypotheses.py).

All resampling respects the dependence in the data:
  * policy comparisons: non-overlapping weekly block bootstrap of daily recovered hours (weeks are resampled
    whole, so autocorrelation within a week is kept);
  * prediction and hazard statistics: cluster bootstrap over lots (holds of one lot stay together);
  * lateness: bootstrap over late lots.
One-sided p-values use the bootstrap distribution with the (count + 1) / (B + 1) correction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as st


# ------------------------------------------------------------------------------------ resampling
def weekly_series(rec: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Recovered hours per calendar day (0 on days without escalations) for the given day index."""
    s = rec.groupby("day").saved_h.sum() if len(rec) else pd.Series(dtype=float)
    return s.reindex(days, fill_value=0.0)


def block_test(rec_a: pd.DataFrame, rec_b: pd.DataFrame, days: pd.DatetimeIndex, B: int = 2000, seed: int = 1,
               level: float = 0.95) -> dict:
    """Policy a vs policy b on the same days. Effect = relative difference of total recovered hours,
    (A - B) / B. Weekly block bootstrap; p_greater tests H1: A > B."""
    a, b = weekly_series(rec_a, days), weekly_series(rec_b, days)
    wk = pd.Index(days).to_period("W")
    A, Bw = a.groupby(wk).sum().values, b.groupby(wk).sum().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(A), (B, len(A)))
    ra, rb = A[idx].sum(1), Bw[idx].sum(1)
    rel_bs = ra / np.maximum(rb, 1e-12) - 1
    q = (1 - level) / 2
    n_days = len(days)
    return dict(a_h_per_day=A.sum() / n_days, b_h_per_day=Bw.sum() / n_days, diff_h_per_day=(A.sum() - Bw.sum()) / n_days,
                rel=A.sum() / max(Bw.sum(), 1e-12) - 1, rel_lo=np.quantile(rel_bs, q), rel_hi=np.quantile(rel_bs, 1 - q),
                log_ratio=np.log(max(A.sum(), 1e-12) / max(Bw.sum(), 1e-12)),
                log_ratio_se=np.std(np.log(np.maximum(ra, 1e-12) / np.maximum(rb, 1e-12)), ddof=1),
                p_greater=(np.sum(ra - rb <= 0) + 1) / (B + 1), p_less=(np.sum(ra - rb >= 0) + 1) / (B + 1),
                weeks=len(A), rel_bs=rel_bs)


def lot_bootstrap(df: pd.DataFrame, stat, B: int = 1000, seed: int = 1, lot_col: str = "lot_id") -> np.ndarray:
    """Cluster bootstrap over lots: stat(df_resampled) for B resamples."""
    rng = np.random.default_rng(seed)
    groups = df.groupby(lot_col).indices
    keys = np.array(list(groups.keys()), dtype=object)
    rows = [groups[k] for k in keys]
    out = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([rows[j] for j in pick])
        out[i] = stat(df.iloc[idx])
    return out


def one_sided_p(boot: np.ndarray, null: float, greater: bool = True) -> float:
    """Bootstrap p-value for H1: theta > null (greater) or theta < null."""
    n = np.sum(boot <= null) if greater else np.sum(boot >= null)
    return (n + 1) / (len(boot) + 1)


# ------------------------------------------------------------------------------------ multiplicity, equivalence
def holm(p: list | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def tost_bootstrap(boot: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests for equivalence |theta| < margin using the bootstrap distribution of theta."""
    p_low = (np.sum(boot <= -margin) + 1) / (len(boot) + 1)
    p_high = (np.sum(boot >= margin) + 1) / (len(boot) + 1)
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return dict(p_tost=max(p_low, p_high), ci90_lo=lo, ci90_hi=hi, equivalent=bool(max(p_low, p_high) < alpha))


def noninferior(boot: np.ndarray, margin: float) -> float:
    """p-value for H1: theta > -margin (non-inferiority)."""
    return (np.sum(boot <= -margin) + 1) / (len(boot) + 1)


# ------------------------------------------------------------------------------------ meta-analysis across lines
def fixed_effect(log_ratios, ses) -> dict:
    """Inverse-variance fixed-effect pooling of log ratios across lines, with Cochran's Q and I^2."""
    y, s = np.asarray(log_ratios, float), np.asarray(ses, float)
    w = 1 / np.maximum(s, 1e-12) ** 2
    mu = np.sum(w * y) / np.sum(w)
    se = np.sqrt(1 / np.sum(w))
    z = mu / se
    Q = float(np.sum(w * (y - mu) ** 2)); df = len(y) - 1
    I2 = max(0.0, (Q - df) / Q) if Q > 0 and df > 0 else 0.0
    return dict(rel=np.exp(mu) - 1, rel_lo=np.exp(mu - 1.96 * se) - 1, rel_hi=np.exp(mu + 1.96 * se) - 1,
                p_greater=float(1 - st.norm.cdf(z)), Q=Q, Q_p=float(1 - st.chi2.cdf(Q, df)) if df > 0 else np.nan, I2=I2)


def weibull_p_less_than_one(shape: float, lo: float, hi: float) -> float:
    """One-sided p for H1: Weibull shape < 1 from a 95% Wald interval."""
    se = (hi - lo) / (2 * 1.96)
    return float(st.norm.cdf((shape - 1) / max(se, 1e-12)))
