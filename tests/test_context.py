"""Decision-time context features agree with a brute-force computation and use no future information."""
import os, subprocess, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from holdesc.features import build, landmark_frame  # noqa: E402
from holdesc.context import regime_features, lot_history  # noqa: E402
from holdesc.tabprep import EDGES, mean_remaining, expected_gain, surv_at, CAP  # noqa: E402


def _holds(tmp_path):
    subprocess.run([sys.executable, "scripts/make_synthetic.py", "--out", str(tmp_path), "--holds", "4000"], cwd=ROOT, check=True)
    raw = pd.read_csv(tmp_path / "holds.csv", parse_dates=["hold_start", "hold_release"])
    h = raw.rename(columns=dict(lot="lot_id", step_code="step", reason_code="reason", dept_code="department",
                                hold_start="start_time", hold_release="release_time", lot_priority="priority",
                                lot_kind="lot_type", family="product_family", wafers="quantity", rework_flag="rework"))
    h["rework"] = (h.rework == "Y").astype(int)
    return build(h)


def test_context_bruteforce(tmp_path):
    h, oc = _holds(tmp_path)
    L = landmark_frame(h, oc, [0.5, 4, 24], 168)
    t = (L.start_time + pd.to_timedelta(L.e, unit="h")).values.astype("datetime64[ns]").astype("int64")
    R = regime_features(h, L.hold_index.values, t)
    hist, agg = lot_history(h, L.hold_index.values, t, K=8)
    st = h.start_time.values.astype("datetime64[ns]").astype("int64"); rl = h.rel.values.astype("datetime64[ns]").astype("int64")
    lots = h.lot_id.astype(str).to_numpy(dtype=object); reason = h.reason.astype(str).to_numpy(dtype=object)
    for q in np.random.default_rng(0).choice(len(L), 200, replace=False):
        i, tq = L.hold_index.values[q], t[q]
        same = reason == reason[i]
        assert ((st <= tq) & (rl > tq) & same).sum() == R["reason_open"].values[q]
        m = same & (rl <= tq) & (rl > tq - 168 * 3600e9)            # only holds released by t
        if m.any():
            assert np.isclose(np.log1p(h.dur_h.values[m]).mean(), R["reason_rel_mlog"].values[q])
        prev = np.where((lots == lots[i]) & (st <= tq) & (np.arange(len(h)) != i))[0]
        assert len(prev) == agg.hist_n.values[q]
        assert (st[hist[q][hist[q] >= 0]] <= tq).all()                # history never contains later holds


def test_survival_integrals():
    lam = np.array([0.05, 0.5, 3.0])
    S = np.exp(-lam[:, None] * EDGES[None, :])
    assert np.allclose(mean_remaining(S), (1 - np.exp(-lam * CAP)) / lam)
    assert np.allclose(expected_gain(S, 2.0), (np.exp(-2 * lam) - np.exp(-lam * CAP)) / lam)
    assert np.allclose(surv_at(S, 8.0), np.exp(-8 * lam))
