$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Convert-ToTag([string]$Value) {
  return (($Value -replace "\.", "p") -replace "-", "m")
}

$PessLambda = if ($env:PESS_LAMBDA) { $env:PESS_LAMBDA } else { "0.005" }
$LambdaTag = Convert-ToTag $PessLambda
$RequestedVariants = if ($env:COMPONENT_VARIANTS) {
  @($env:COMPONENT_VARIANTS -split "," | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
} else {
  @("full", "nopess", "noreplay")
}
$AllowedVariants = @("full", "nopess", "noreplay")
foreach ($VariantName in $RequestedVariants) {
  if ($AllowedVariants -notcontains $VariantName) {
    throw "Unknown component variant '$VariantName'. Allowed values: $($AllowedVariants -join ', ')."
  }
}
$JobTag = if ($env:COMPONENT_JOB_TAG) {
  $env:COMPONENT_JOB_TAG
} else {
  "lp${LambdaTag}_$($RequestedVariants -join 'x')"
}
$LockFile = Join-Path $PSScriptRoot ".component_ablation_${JobTag}.lock"
if (Test-Path $LockFile) {
  throw "Component ablation is already running, or a stale lock exists: $LockFile"
}
Set-Content -Path $LockFile -Value "pid=$PID started=$(Get-Date -Format o)"

try {
  $Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
  $GpuTag = if ($env:GPU_ID) { $env:GPU_ID } else { "0" }
  $env:MPLCONFIGDIR = if ($env:MPLCONFIGDIR) { $env:MPLCONFIGDIR } else { Join-Path $PSScriptRoot ".matplotlib_component_gpu$GpuTag" }
  # Locked paper-main protocol. Each variant below changes one component only.
  $env:M = "6"
  $env:T = "6"
  $env:BUDGET = "10"
  $env:Q_MAX = "3"
  $env:SEEDS = "2026"
  $env:INITIAL_LOGGED_DAYS = "800"
  $env:NEW_LOGGED_DAYS = "100"
  $env:VALIDATION_LOGGED_DAYS = "1200"
  $env:HOLDOUT_LOGGED_DAYS = "1200"
  $env:AUTO_ALPHA = "0"
  $env:FIXED_TEMPERATURE = "0.05"
  $env:INIT_SHARE = "0.2"
  $env:CENTER_RESIDUAL = "1"
  $env:SIM_EPOCHS = "700"
  $env:SIM_FINETUNE_EPOCHS = "100"
  $env:SIM_BATCH_SIZE = "4096"
  $env:SIM_LR = "0.0008"
  $env:SIM_DROPOUT = "0.02"
  $env:SIM_WEIGHT_DECAY = "0.000005"
  $env:B_BOOT = "8"
  $env:BOOTSTRAP_EPOCHS = "50"
  $env:BOOTSTRAP_FINETUNE_EPOCHS = "25"
  $env:SYNTHETIC_TRAIN_EPISODES = "5000"
  $env:ROLLOUT_BATCH_SIZE = "256"
  $env:WARMUP_EPISODES = "200"
  $env:UPDATE_AFTER_EPISODES = "40"
  $env:GRADIENT_STEPS_PER_UPDATE = "4"
  $env:BATCH_ROLLOUT_UPDATE_CAP = "512"
  $env:BATCH_SIZE = "2048"
  $env:REPLAY_SIZE = "1500000"
  $env:REAL_BATCH_SIZE = "256"
  $env:REAL_UPDATE_AFTER_STEPS = "2"
  $env:REAL_GRADIENT_STEPS_PER_STEP = "2"
  $env:VALIDATION_EPISODES = "500"
  $env:EVAL_EPISODES = "2000"
  $env:METHODS = "sacsa_reserve"
  $env:RESUME_COMPLETED = "1"

  $AllVariants = @(
    [pscustomobject]@{ Name = "full";     Pess = "1"; Lambda = $PessLambda; Residual = "1"; Replay = "1" },
    [pscustomobject]@{ Name = "nopess";   Pess = "0"; Lambda = "0";         Residual = "1"; Replay = "1" },
    [pscustomobject]@{ Name = "noreplay"; Pess = "1"; Lambda = $PessLambda; Residual = "1"; Replay = "0" }
  )
  $Variants = @($AllVariants | Where-Object { $RequestedVariants -contains $_.Name })

  foreach ($Variant in $Variants) {
    $env:USE_PESSIMISM = $Variant.Pess
    $env:LAMBDA_PESS = $Variant.Lambda
    $env:RESIDUAL_BOOTSTRAP = $Variant.Residual
    $env:REAL_ONLINE_UPDATE = $Variant.Replay
    if ($Variant.Name -eq "nopess") {
      $RunTag = "synT6_comp_nopess"
    } else {
      $RunTag = "synT6_comp_lp${LambdaTag}_$($Variant.Name)"
    }
    $env:RUN_TAG = $RunTag
    $env:SUMMARY_DIR = "summary_${RunTag}"
    Write-Host "[component] seed=2026 lambda_pess=$($Variant.Lambda) variant=$($Variant.Name)"
    & "$PSScriptRoot\run_gpu_q3_5090_windows.ps1"
  }

  Write-Host "[done] component job=$JobTag variants=$($RequestedVariants -join ',')"
} finally {
  Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
}
