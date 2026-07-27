$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Split-List([string]$Value, [string]$Default) {
  $Raw = if ($Value) { $Value } else { $Default }
  return @($Raw -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Run-BaseSetting([string]$Tag, [string]$D0) {
  $env:M = "6"
  $env:T = "6"
  $env:Q_MAX = "3"
  $env:INITIAL_LOGGED_DAYS = $D0
  $env:NEW_LOGGED_DAYS = "100"
  $env:BUDGET = "10"
  $env:SEEDS = "2026"
  $env:LAMBDA_PESS = "0.005"
  $env:INIT_SHARE = "0.2"
  $env:AUTO_ALPHA = "0"
  $env:FIXED_TEMPERATURE = "0.05"
  $env:INIT_TEMPERATURE = "0.05"
  $env:MIN_TEMPERATURE = "0.05"
  $env:MAX_TEMPERATURE = "0.05"
  $env:CENTER_RESIDUAL = "1"
  $env:REAL_ONLINE_UPDATE = "1"
  $env:METHODS = if ($env:METHODS) { $env:METHODS } else { "sacsa_reserve,dfl_pto" }
  $env:RUN_TAG = $Tag
  $env:SUMMARY_DIR = "summary_${Tag}"
  Write-Host "[setting] D0=$D0 N=100 B=10 seed=2026 tag=$Tag"
  & (Join-Path $PSScriptRoot "run_gpu_q3_5090_windows.ps1")
}

$D0List = Split-List $env:D0_LIST "800,1000"
foreach ($D0 in $D0List) {
  Run-BaseSetting "synT6_abD0fix_D${D0}_N100_B10_lp0p005_a0t0p05_i0p2_s2026" $D0
}

$SweepSummary = "summary_synT6_abD0fix"
$FigureDir = "paper_figures_synT6_abD0fix"
if ($env:DRY_RUN -eq "1" -or $env:DRY_RUN -eq "true" -or $env:DRY_RUN -eq "True") {
  Write-Host "[dry-run] $Python summarize_iterative_results.py --root . --pattern results_synT6_abD0fix_* --out_dir $SweepSummary"
  Write-Host "[dry-run] $Python make_synthetic_t6_ablation_paper_figures.py --summary_dir $SweepSummary --facet d0 --out_dir $FigureDir --name_prefix fig_synT6_abD0"
} else {
  & $Python summarize_iterative_results.py --root . --pattern "results_synT6_abD0fix_*" --out_dir $SweepSummary
  if ($LASTEXITCODE -ne 0) { throw "sweep summary failed with exit code $LASTEXITCODE" }
  & $Python make_synthetic_t6_ablation_paper_figures.py --summary_dir $SweepSummary --facet d0 --out_dir $FigureDir --name_prefix fig_synT6_abD0
  if ($LASTEXITCODE -ne 0) { throw "ablation figures failed with exit code $LASTEXITCODE" }
}
Write-Host "[done] $SweepSummary"
