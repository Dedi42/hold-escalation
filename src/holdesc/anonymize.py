"""Remove company-identifying information before anything is analysed or saved.

* department, reason, step, product and lot-type codes are replaced by rank-ordered
  labels (D1, D2, ... by number of holds), so that no internal code appears in outputs;
* lot identifiers are replaced by salted SHA-256 hashes; the salt is read from the
  environment variable HOLDESC_SALT and is never written to disk;
* calendar time is shifted by a whole number of weeks so that the week of the first
  hold starts on Monday 2000-01-03; hour of day and weekday are preserved, the
  calendar date is not.
The code-to-label keys are returned in memory and never saved.
"""
from __future__ import annotations
import hashlib
import os
import pandas as pd

ORIGIN = pd.Timestamp("2000-01-03")  # a Monday


def _rank_codes(s: pd.Series, prefix: str, width: int) -> tuple[pd.Series, dict]:
    s = s.fillna("NA").astype(str)
    mapping = {c: f"{prefix}{i + 1:0{width}d}" for i, c in enumerate(s.value_counts().index)}
    return s.map(mapping), mapping


def anonymize(holds: pd.DataFrame, lots: pd.DataFrame | None = None, salt: str | None = None):
    salt = os.environ.get("HOLDESC_SALT", "") if salt is None else salt
    h = holds.copy()
    keys = {}
    for col, prefix, width in [("department", "D", 1), ("reason", "R", 2), ("step", "S", 3),
                               ("product_family", "P", 2), ("lot_type", "T", 1)]:
        h[col], keys[col] = _rank_codes(h[col], prefix, width)

    def hash_(x):
        return hashlib.sha256((salt + str(x)).encode()).hexdigest()[:12]

    h["lot_id"] = h["lot_id"].map(hash_)
    first = h["start_time"].min().normalize()
    monday = first - pd.Timedelta(days=first.weekday())
    shift = pd.Timedelta(weeks=round((monday - ORIGIN) / pd.Timedelta(weeks=1)))
    for c in ["start_time", "release_time"]:
        h[c] = h[c] - shift
    out = None
    if lots is not None:
        out = lots.copy()
        out.index = out.index.map(hash_)
        for c in ["completion_time", "due_time"]:
            if c in out:
                out[c] = out[c] - shift
    return h, out, keys
