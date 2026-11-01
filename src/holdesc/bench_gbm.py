"""Gradient-boosting learners of the benchmark, tuned with the same budget as the deep models."""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from .features import CAT, encode

N_LEVELS = 31


def matrix(df: pd.DataFrame, num_cols, cats: dict) -> pd.DataFrame:
    return encode(df, cats, list(num_cols) + CAT)


def fit_reg(Xtr, ytr, Xva, yva, hp: dict, seed: int):
    m = lgb.LGBMRegressor(n_estimators=4000, learning_rate=hp["lr"], num_leaves=hp["leaves"],
                          min_child_samples=hp["min_child"], subsample=hp["bagging"], subsample_freq=1,
                          colsample_bytree=hp["colsample"], reg_lambda=hp["l2"], random_state=seed, verbose=-1,
                          n_jobs=hp.get("n_jobs", -1))
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def gain_levels(g: np.ndarray) -> np.ndarray:
    """Graded relevance for LambdaRank whose label gain is proportional to the hold-hours gained."""
    return np.clip(np.floor(3 * np.log2(1 + np.maximum(g, 0))), 0, N_LEVELS - 1).astype(int)


LABEL_GAIN = [2 ** (l / 3) - 1 for l in range(N_LEVELS)]


def fit_rank(Xtr, gtr, sizes_tr, Xva, gva, sizes_va, hp: dict, seed: int, eval_at: int = 10):
    m = lgb.LGBMRanker(n_estimators=4000, learning_rate=hp["lr"], num_leaves=hp["leaves"],
                       min_child_samples=hp["min_child"], subsample=hp["bagging"], subsample_freq=1,
                       colsample_bytree=hp["colsample"], reg_lambda=hp["l2"], random_state=seed, verbose=-1,
                       label_gain=LABEL_GAIN, n_jobs=hp.get("n_jobs", -1))
    m.fit(Xtr, gain_levels(gtr), group=sizes_tr, eval_set=[(Xva, gain_levels(gva))], eval_group=[sizes_va],
          eval_at=[eval_at], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def sample_hp(rng) -> dict:
    return dict(lr=float(rng.choice([0.02, 0.05, 0.1])), leaves=int(rng.choice([15, 31, 63, 127, 255])),
                min_child=int(rng.choice([20, 50, 100, 200])), bagging=float(rng.choice([0.6, 0.8, 1.0])),
                colsample=float(rng.choice([0.5, 0.7, 0.9, 1.0])), l2=float(rng.choice([0.0, 1.0, 10.0])))
