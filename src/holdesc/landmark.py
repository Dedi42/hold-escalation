"""Landmark gradient-boosting model of the remaining hold time."""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from .features import NUM, CAT, LM_EXTRA, landmark_frame, encode

LANDMARKS = [0.5, 1, 2, 4, 8, 16, 24, 48, 72, 96, 168]
PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1, n_jobs=1)


class LandmarkModel:
    def __init__(self, cap: float = 168.0, seed: int = 1, landmarks=LANDMARKS, shift: float = 0.0):
        """shift = 0: target is the truncated remaining time min(T, cap) - e (Eq. 3).
        shift = delta > 0: target is (min(T, cap) - e - delta)^+, i.e. the gain g of Eq. (2) per unit alpha."""
        self.cap, self.seed, self.landmarks, self.shift = cap, seed, landmarks, shift
        self.cols = NUM + LM_EXTRA + CAT

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, oc) -> "LandmarkModel":
        both = pd.concat([train, val])
        self.cats = {c: pd.Categorical(both[c]).categories for c in CAT}
        lt, lv, lb = (landmark_frame(d, oc, self.landmarks, self.cap) for d in (train, val, both))
        for d in (lt, lv, lb):
            d["remaining"] = np.maximum(d.remaining - self.shift, 0.0)
        m = lgb.LGBMRegressor(n_estimators=2000, random_state=self.seed, **PARAMS)
        m.fit(encode(lt, self.cats, self.cols), lt.remaining, eval_set=[(encode(lv, self.cats, self.cols), lv.remaining)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        self.n_trees = max(m.best_iteration_, 20)
        self.model = lgb.LGBMRegressor(n_estimators=self.n_trees, random_state=self.seed, **PARAMS)
        self.model.fit(encode(lb, self.cats, self.cols), lb.remaining)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict(encode(rows, self.cats, self.cols))

    def importance(self) -> pd.Series:
        g = pd.Series(self.model.booster_.feature_importance("gain"), index=self.cols)
        return (g / g.sum()).sort_values(ascending=False)
