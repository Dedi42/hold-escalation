# Windows PowerShell version of reproduce_paper.sh
#   .\reproduce_paper.ps1                 -> from the archived aggregated results in paper_results\
#   .\reproduce_paper.ps1 synthetic       -> full pipeline on two synthetic lines
#   .\reproduce_paper.ps1 own config\local_lineA.yaml config\local_lineB.yaml
param([string]$mode = "archived", [string]$CA = "", [string]$CB = "")
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
if ($mode -eq "archived") {
  python scripts/make_paper.py --hyp paper_results/hypotheses --bench A=paper_results/benchmark_A --bench B=paper_results/benchmark_B --out paper_output
  exit
}
$extra = @()
if ($mode -eq "synthetic") {
  python scripts/make_synthetic.py --out data/synthetic --holds 20000 --seed 7
  python scripts/make_synthetic.py --out data/synthetic_B --holds 30000 --seed 11
  $CA = "config/synthetic.yaml"; $CB = "config/synthetic_B.yaml"; $out = "results_synthetic"; $extra = @("--quick")
} else { $out = "results_paper" }
python scripts/run_hypotheses.py --line "A=$CA" --line "B=$CB" --out "$out/hypotheses" @extra
python scripts/run_benchmark.py --config $CA --out "$out/benchmark_A" @extra
python scripts/run_benchmark.py --config $CB --out "$out/benchmark_B" @extra
python scripts/make_paper.py --hyp "$out/hypotheses" --bench "A=$out/benchmark_A" --bench "B=$out/benchmark_B" --out "${out}_paper_output"
