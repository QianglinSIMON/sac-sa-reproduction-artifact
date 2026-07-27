$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Split-List([string]$Value, [string]$Default) {
  $Raw = if ($Value) { $Value } else { $Default }
  return @($Raw -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

$SeedList = Split-List $env:SEED_LIST "2026,2028"
$SeedTag = ($SeedList -join "x")
$Tag = "synT6_abSeedfix_D800_N100_B10_lp0p005_a0t0p05_i0p2_s${SeedTag}"

$env:M = "6"
$env:T = "6"
$env:Q_MAX = "3"
$env:INITIAL_LOGGED_DAYS = "800"
$env:NEW_LOGGED_DAYS = "100"
$env:BUDGET = "10"
$env:SEEDS = ($SeedList -join ",")
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

Write-Host "[setting] D0=800 N=100 B=10 seeds=$($env:SEEDS) tag=$Tag"
& (Join-Path $PSScriptRoot "run_gpu_q3_5090_windows.ps1")

$SweepSummary = "summary_synT6_abSeedfix"
$FigureDir = "paper_figures_synT6_abSeedfix"
if ($env:DRY_RUN -eq "1" -or $env:DRY_RUN -eq "true" -or $env:DRY_RUN -eq "True") {
  Write-Host "[dry-run] $Python summarize_iterative_results.py --root . --pattern results_synT6_abSeedfix_* --out_dir $SweepSummary"
  Write-Host "[dry-run] $Python make_synthetic_t6_ablation_paper_figures.py --summary_dir $SweepSummary --facet seed --out_dir $FigureDir --name_prefix fig_synT6_abSeed"
} else {
  & $Python summarize_iterative_results.py --root . --pattern "results_synT6_abSeedfix_*" --out_dir $SweepSummary
  if ($LASTEXITCODE -ne 0) { throw "sweep summary failed with exit code $LASTEXITCODE" }
  & $Python make_synthetic_t6_ablation_paper_figures.py --summary_dir $SweepSummary --facet seed --out_dir $FigureDir --name_prefix fig_synT6_abSeed
  if ($LASTEXITCODE -ne 0) { throw "ablation figures failed with exit code $LASTEXITCODE" }
}
Write-Host "[done] $SweepSummary"
