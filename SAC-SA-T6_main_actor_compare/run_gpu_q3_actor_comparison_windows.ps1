$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$LockFile = Join-Path $PSScriptRoot ".actor_comparison.lock"
if (Test-Path $LockFile) {
  throw "Actor comparison is already running, or a stale lock exists: $LockFile"
}
Set-Content -Path $LockFile -Value "pid=$PID started=$(Get-Date -Format o)"

try {
  $Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

  $GpuTag = if ($env:GPU_ID) { $env:GPU_ID } else { "0" }
  $env:MPLCONFIGDIR = if ($env:MPLCONFIGDIR) { $env:MPLCONFIGDIR } else { Join-Path $PSScriptRoot ".matplotlib_actor_gpu$GpuTag" }

  # Locked synthetic T=6 paper protocol. Only the actor parameterization changes.
  $env:M = "6"
  $env:T = "6"
  $env:BUDGET = "10"
  $env:Q_MAX = "3"
  $env:SEEDS = "2026"
  $env:INITIAL_LOGGED_DAYS = "800"
  $env:NEW_LOGGED_DAYS = "100"
  $env:VALIDATION_LOGGED_DAYS = "1200"
  $env:HOLDOUT_LOGGED_DAYS = "1200"
  $env:LAMBDA_PESS = "0.005"
  $env:AUTO_ALPHA = "0"
  $env:FIXED_TEMPERATURE = "0.05"
  $env:INIT_SHARE = "0.2"
  $env:GAUSSIAN_PROJECTION_INIT_LOG_STD = "-4.0"
  $env:CENTER_RESIDUAL = "1"
  $env:REAL_ONLINE_UPDATE = "1"

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
  $env:RESUME_COMPLETED = "1"

  $DryRun = $env:DRY_RUN -eq "1" -or $env:DRY_RUN -eq "true" -or $env:DRY_RUN -eq "True"
  if (-not $DryRun) {
    & $Python check_actor_parameterizations.py
    if ($LASTEXITCODE -ne 0) {
      throw "Actor feasibility preflight failed."
    }
  }

  $Actors = @(
    [pscustomobject]@{ Tag = "bd";    Method = "sacsa_reserve" },
    [pscustomobject]@{ Tag = "bln";   Method = "sac_logistic_normal_reserve" },
    [pscustomobject]@{ Tag = "ddir";  Method = "sacsa" },
    [pscustomobject]@{ Tag = "dln";   Method = "sac_logistic_normal" },
    [pscustomobject]@{ Tag = "gproj"; Method = "sac_gaussian_projection" }
  )

  foreach ($Actor in $Actors) {
    $env:METHODS = $Actor.Method
    $env:RUN_TAG = "synT6_actor_$($Actor.Tag)"
    $env:SUMMARY_DIR = "summary_synT6_actor_$($Actor.Tag)"
    Write-Host "[actor] tag=$($Actor.Tag) method=$($Actor.Method)"
    & "$PSScriptRoot\run_gpu_q3_5090_windows.ps1"
  }

  if (-not $DryRun) {
    & $Python make_actor_comparison_figures.py
    if ($LASTEXITCODE -ne 0) {
      throw "Actor-comparison figure generation failed."
    }
  }
} finally {
  Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
}
