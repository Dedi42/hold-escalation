"""Load hold events and lot data and map them to the standard schema."""
from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
from .schema import HOLD_COLUMNS, LOT_COLUMNS, REQUIRED_HOLD


def _read(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    if path.endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _times(s: pd.Series, fmt, nchar) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s.astype(str).str.slice(0, nchar), format=fmt, errors="coerce")


def load_holds(cfg: dict) -> pd.DataFrame:
    """Read hold events and rename columns according to cfg['holds']['columns']."""
    h = cfg["holds"]
    raw = _read(h["path"])
    for col, val in (h.get("filters") or {}).items():
        raw = raw[raw[col].astype(str).isin([str(v) for v in np.atleast_1d(val)])]
    for col, val in (h.get("exclude") or {}).items():
        raw = raw[~raw[col].astype(str).isin([str(v) for v in np.atleast_1d(val)])]
    mapping = {v: k for k, v in h["columns"].items() if v}
    df = raw.rename(columns=mapping)
    df = df[[c for c in HOLD_COLUMNS if c in df.columns]].copy()
    missing = [c for c in REQUIRED_HOLD if c not in df]
    if missing:
        raise ValueError(f"missing required hold columns: {missing}")
    for c in ["start_time", "release_time"]:
        df[c] = _times(df[c], h.get("time_format"), h.get("time_chars", 30))
    yes = (h.get("yes_values") or {}).get("rework")
    if yes is not None and "rework" in df:
        df["rework"] = df["rework"].astype(str).isin([str(v) for v in yes]).astype(int)
    for c in ["priority", "quantity", "rework"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df else 0
    for c in ["lot_type", "product_family"]:
        df[c] = df[c].astype(str) if c in df else "NA"
    return df.dropna(subset=["start_time"]).reset_index(drop=True)


def load_lots(cfg: dict) -> pd.DataFrame | None:
    if not cfg.get("lots"):
        return None
    l = cfg["lots"]
    raw = _read(l["path"])
    mapping = {v: k for k, v in l["columns"].items() if v}
    df = raw.rename(columns=mapping)
    df = df[[c for c in LOT_COLUMNS if c in df.columns]].copy()
    for c in ["completion_time", "due_time"]:
        if c in df:
            df[c] = _times(df[c], l.get("time_format"), l.get("time_chars", 30))
    return df.drop_duplicates("lot_id").set_index("lot_id")
