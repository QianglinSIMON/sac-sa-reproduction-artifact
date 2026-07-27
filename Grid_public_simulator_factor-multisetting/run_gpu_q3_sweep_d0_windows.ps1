$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Methods = if ($env:METHODS_OVERRIDE) { $env:METHODS_OVERRIDE } else { "sacsa_reserve,dfl_pto" }

function Split-List([string]$Value, [string]$Default) {
    $Raw = if ($Value) { $Value } else { $Default }
    return @($Raw -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Set-MainSetting([string]$Stem, [string]$D0Value) {
    Remove-Item Env:\RUN_TAG -ErrorAction SilentlyContinue
    $env:OUTPUT_STEM = "pc_$Stem"
    $env:METHODS = $Methods
    $env:M = "3"; $env:T = "20"; $env:BUDGET = "10"; $env:Q_MAX = "3"; $env:SEED = "2026"
    $env:INITIAL_LOGGED_DAYS = $D0Value; $env:NEW_LOGGED_DAYS = "300"
    $env:DGP_CALIBRATION_DAYS = "10000"; $env:DGP_BASELINE_EPOCHS = "100"
    $env:RESIDUAL_CALIBRATION_DAYS = "1200"; $env:VALIDATION_LOGGED_DAYS = "800"; $env:HOLDOUT_LOGGED_DAYS = "800"
    $env:SIM_EPOCHS = "600"; $env:B_BOOT = "12"; $env:BOOTSTRAP_EPOCHS = "120"
    $env:SYNTHETIC_TRAIN_EPISODES = "7000"; $env:SYNTHETIC_TRAIN_EPISODES_LATER_ROUNDS = "3000"
    $env:LAMBDA_PESS = "0.015"; $env:BEHAVIOR_SUPPORT_LAMBDA = "0.03"; $env:SUPPORT_PROTOTYPE_COUNT = "512"
    $env:AUTO_ALPHA = "1"; $env:FIXED_TEMPERATURE = "0.08"; $env:INIT_SHARE = "0.10"
    $env:REAL_EXPLORATION_EPSILON = "0.25"; $env:REAL_GRADIENT_STEPS_PER_STEP = "0.04"; $env:REAL_CRITIC_WARMUP_UPDATES = "200"
    $env:SIMULATOR_SELECTION_EPISODES = "1200"; $env:EVAL_EPISODES = "2000"
    $env:DFL_EPOCHS = "150"; $env:DFL_BATCH_SIZE = "1024"; $env:DFL_CANDIDATE_RANDOM_COUNT = "128"
    $env:DFL_LEARNING_RATE = "0.001"; $env:DFL_DECISION_WEIGHT = "1.0"; $env:DFL_SOFTMAX_TEMPERATURE = "1.0"
    $env:DFL_BUDGET_FRACTIONS = "0.25,0.50,0.75,1.00"
    $env:STRICT_SIMULATOR_QUALITY = "1"; $env:SIM_MIN_R2_Z = "0.80"; $env:SIM_MIN_R2_Z_BY_T = "0.45"
}

$D0List = Split-List $env:D0_LIST "4500,3500"
foreach ($D0Value in $D0List) {
    $Stem = if ($D0Value -eq "4500") { "sensitivity_base" } else { "sensitivity_d0_D${D0Value}" }
    Set-MainSetting $Stem $D0Value
    Write-Host "[setting] M=3 B=10 D0=$D0Value N=300 seed=2026 tag=$Stem"
    if ($env:DRY_RUN -eq "1") { continue }
    & (Join-Path $PSScriptRoot "run_public_city_panel_gpu_q3_5090_windows.ps1")
}

$FigureDir = "paper_figures_pc_sensitivity_d0"
$ResultPattern = "results_pc_sensitivity_base_*,results_pc_sensitivity_d0_*"
if ($env:DRY_RUN -eq "1") {
    Write-Host "[dry-run] $Python make_public_city_ablation_paper_figures.py --root . --pattern $ResultPattern --facet d0 --out_dir $FigureDir --name_prefix fig_pc_sensitivity_d0"
} else {
    Remove-Item "$FigureDir\fig_pc_sensitivity_d0_*.pdf" -ErrorAction SilentlyContinue
    Remove-Item "$FigureDir\source_data\fig_pc_sensitivity_d0_*.csv" -ErrorAction SilentlyContinue
    $LASTEXITCODE = $null
    & $Python make_public_city_ablation_paper_figures.py --root . --pattern $ResultPattern --facet d0 --out_dir $FigureDir --name_prefix fig_pc_sensitivity_d0
    $FigureSucceeded = $?
    $FigureExitCode = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } elseif ($FigureSucceeded) { 0 } else { -1 }
    $FigureComplete = (
        (Test-Path "$FigureDir\fig_pc_sensitivity_d0_final_boxplot.pdf") -and
        (Test-Path "$FigureDir\fig_pc_sensitivity_d0_round_curve.pdf") -and
        (Test-Path "$FigureDir\fig_pc_sensitivity_d0_reward_spend.pdf") -and
        (Test-Path "$FigureDir\source_data\fig_pc_sensitivity_d0_round_eval_summary.csv")
    )
    if ($FigureExitCode -ne 0 -and $FigureComplete) {
        Write-Warning "D0 figure process reported exit code $FigureExitCode after publishing complete outputs; continuing."
    } elseif ($FigureExitCode -ne 0) {
        throw "D0 sweep figure aggregation failed with exit code $FigureExitCode"
    }
}
Write-Host "[done] public-city D0 sweep"
