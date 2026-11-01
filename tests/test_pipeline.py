"""Smoke test: the full pipeline runs on a small synthetic data set and writes no raw codes."""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_pipeline(tmp_path):
    data = tmp_path / "data"
    subprocess.run([sys.executable, "scripts/make_synthetic.py", "--out", str(data), "--holds", "6000"], cwd=ROOT, check=True)
    cfg = tmp_path / "cfg.yaml"
    text = open(os.path.join(ROOT, "config/synthetic.yaml")).read().replace("data/synthetic", str(data))
    cfg.write_text(text)
    out = tmp_path / "out"
    subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(cfg), "--out", str(out),
                    "--budgets", "6,12", "--bootstrap", "50"], cwd=ROOT, check=True)
    s = json.load(open(out / "summary.json"))
    assert s["holds"] > 5000
    for f in os.listdir(out):
        if f.endswith(".csv"):
            body = open(out / f).read()
            assert "DEP_" not in body and "RSN_" not in body and "LOT0" not in body, f"raw code in {f}"
