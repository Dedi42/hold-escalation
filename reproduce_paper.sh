#!/usr/bin/env bash
# Rebuild all tables and figures of the paper.
#   bash reproduce_paper.sh            -> from the archived aggregated results in paper_results/ (seconds)
#   bash reproduce_paper.sh synthetic  -> full pipeline on two synthetic lines (about 10-20 min on a laptop)
#   bash reproduce_paper.sh own config/local_lineA.yaml config/local_lineB.yaml  -> full pipeline on your own data
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mode="${1:-archived}"
if [ "$mode" = "archived" ]; then
  python scripts/make_paper.py --hyp paper_results/hypotheses \
    --bench A=paper_results/benchmark_A --bench B=paper_results/benchmark_B --out paper_output
  exit 0
fi
if [ "$mode" = "synthetic" ]; then
  python scripts/make_synthetic.py --out data/synthetic --holds 20000 --seed 7
  python scripts/make_synthetic.py --out data/synthetic_B --holds 30000 --seed 11
  CA=config/synthetic.yaml; CB=config/synthetic_B.yaml; OUT=results_synthetic; EXTRA="--quick"
else
  CA="$2"; CB="$3"; OUT=results_paper; EXTRA=""
fi
python scripts/run_hypotheses.py --line A="$CA" --line B="$CB" --out "$OUT/hypotheses" $EXTRA
MODELS=$(python -c "import importlib.util as u; print('' if u.find_spec('torch') else 'lgb_lm_paper,lgb_lm_ext,lgb_gain_ext,lgb_rank_ext')")
python scripts/run_benchmark.py --config "$CA" --out "$OUT/benchmark_A" $EXTRA ${MODELS:+--models $MODELS}
python scripts/run_benchmark.py --config "$CB" --out "$OUT/benchmark_B" $EXTRA ${MODELS:+--models $MODELS}
python scripts/make_paper.py --hyp "$OUT/hypotheses" --bench A="$OUT/benchmark_A" --bench B="$OUT/benchmark_B" --out "${OUT}_paper_output"
