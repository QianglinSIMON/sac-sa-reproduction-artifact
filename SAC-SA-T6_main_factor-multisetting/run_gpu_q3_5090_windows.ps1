$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$GpuId = if ($env:GPU_ID) { $env:GPU_ID } elseif ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:CUDA_VISIBLE_DEVICES = $GpuId
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "4" }
$env:MKL_NUM_THREADS = if ($env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS } else { "4" }
$env:NUMEXPR_NUM_THREADS = if ($env:NUMEXPR_NUM_THREADS) { $env:NUMEXPR_NUM_THREADS } else { "4" }
$env:MPLCONFIGDIR = if ($env:MPLCONFIGDIR) { $env:MPLCONFIGDIR } else { Join-Path $PSScriptRoot ".matplotlib" }

function Convert-ToTag([string]$Value) {
  return (($Value -replace "\.", "p") -replace "-", "m")
}

if ($env:SEEDS) {
  $Seeds = @($env:SEEDS -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
} else {
  $Seeds = @("2026")
}

$QMax = if ($env:Q_MAX) { $env:Q_MAX } else { "3" }
$Device = if ($env:DEVICE) { $env:DEVICE } else { "cuda" }
$M = if ($env:M) { $env:M } else { "6" }
$T = if ($env:T) { $env:T } else { "6" }
$Budget = if ($env:BUDGET) { $env:BUDGET } else { "10" }
$LambdaPess = if ($env:LAMBDA_PESS) { $env:LAMBDA_PESS } else { "0.005" }
$InitShare = if ($env:INIT_SHARE) { $env:INIT_SHARE } else { "0.2" }

$AutoAlpha = if ($env:AUTO_ALPHA) { $env:AUTO_ALPHA } else { "0" }
$FixedTemperature = if ($env:FIXED_TEMPERATURE) { $env:FIXED_TEMPERATURE } else { "0.05" }
$InitTemperature = if ($env:INIT_TEMPERATURE) { $env:INIT_TEMPERATURE } else { "0.05" }
$MinTemperature = if ($env:MIN_TEMPERATURE) { $env:MIN_TEMPERATURE } else { "1e-5" }
$MaxTemperature = if ($env:MAX_TEMPERATURE) { $env:MAX_TEMPERATURE } else { "0.2" }
$TargetEntropyScale = if ($env:TARGET_ENTROPY_SCALE) { $env:TARGET_ENTROPY_SCALE } else { "2.5" }
$SacEntropySpace = if ($env:SAC_ENTROPY_SPACE) { $env:SAC_ENTROPY_SPACE } else { "share_clipped" }

$InitialLoggedDays = if ($env:INITIAL_LOGGED_DAYS) { $env:INITIAL_LOGGED_DAYS } else { "800" }
$NewLoggedDays = if ($env:NEW_LOGGED_DAYS) { $env:NEW_LOGGED_DAYS } else { "100" }
$ValidationLoggedDays = if ($env:VALIDATION_LOGGED_DAYS) { $env:VALIDATION_LOGGED_DAYS } else { "1200" }
$HoldoutLoggedDays = if ($env:HOLDOUT_LOGGED_DAYS) { $env:HOLDOUT_LOGGED_DAYS } else { "1200" }
$RealOnlineUpdate = if ($env:REAL_ONLINE_UPDATE) { $env:REAL_ONLINE_UPDATE } else { "1" }
$CenterResidual = if ($env:CENTER_RESIDUAL) { $env:CENTER_RESIDUAL } else { "1" }

$SimEpochs = if ($env:SIM_EPOCHS) { $env:SIM_EPOCHS } else { "700" }
$SimFinetuneEpochs = if ($env:SIM_FINETUNE_EPOCHS) { $env:SIM_FINETUNE_EPOCHS } else { "100" }
$SimBatchSize = if ($env:SIM_BATCH_SIZE) { $env:SIM_BATCH_SIZE } else { "4096" }
$SimLr = if ($env:SIM_LR) { $env:SIM_LR } else { "8e-4" }
$SimDropout = if ($env:SIM_DROPOUT) { $env:SIM_DROPOUT } else { "0.02" }
$SimWeightDecay = if ($env:SIM_WEIGHT_DECAY) { $env:SIM_WEIGHT_DECAY } else { "5e-6" }
$Bboot = if ($env:B_BOOT) { $env:B_BOOT } else { "8" }
$BootstrapEpochs = if ($env:BOOTSTRAP_EPOCHS) { $env:BOOTSTRAP_EPOCHS } else { "50" }
$BootstrapFinetuneEpochs = if ($env:BOOTSTRAP_FINETUNE_EPOCHS) { $env:BOOTSTRAP_FINETUNE_EPOCHS } else { "25" }

$SyntheticTrainEpisodes = if ($env:SYNTHETIC_TRAIN_EPISODES) { $env:SYNTHETIC_TRAIN_EPISODES } else { "5000" }
$RolloutBatchSize = if ($env:ROLLOUT_BATCH_SIZE) { $env:ROLLOUT_BATCH_SIZE } else { "256" }
$WarmupEpisodes = if ($env:WARMUP_EPISODES) { $env:WARMUP_EPISODES } else { "200" }
$UpdateAfterEpisodes = if ($env:UPDATE_AFTER_EPISODES) { $env:UPDATE_AFTER_EPISODES } else { "40" }
$GradientStepsPerUpdate = if ($env:GRADIENT_STEPS_PER_UPDATE) { $env:GRADIENT_STEPS_PER_UPDATE } else { "4" }
$BatchRolloutUpdateCap = if ($env:BATCH_ROLLOUT_UPDATE_CAP) { $env:BATCH_ROLLOUT_UPDATE_CAP } else { "512" }
$BatchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "2048" }
$ReplaySize = if ($env:REPLAY_SIZE) { $env:REPLAY_SIZE } else { "1500000" }
$RealBatchSize = if ($env:REAL_BATCH_SIZE) { $env:REAL_BATCH_SIZE } else { "256" }
$RealUpdateAfterSteps = if ($env:REAL_UPDATE_AFTER_STEPS) { $env:REAL_UPDATE_AFTER_STEPS } else { "2" }
$RealGradientStepsPerStep = if ($env:REAL_GRADIENT_STEPS_PER_STEP) { $env:REAL_GRADIENT_STEPS_PER_STEP } else { "2" }
$ValidationEpisodes = if ($env:VALIDATION_EPISODES) { $env:VALIDATION_EPISODES } else { "500" }
$EvalEpisodes = if ($env:EVAL_EPISODES) { $env:EVAL_EPISODES } else { "2000" }
$DflEpochs = if ($env:DFL_EPOCHS) { $env:DFL_EPOCHS } else { "150" }
$DflBatchSize = if ($env:DFL_BATCH_SIZE) { $env:DFL_BATCH_SIZE } else { "2048" }
$CandidateRandomCount = if ($env:CANDIDATE_RANDOM_COUNT) { $env:CANDIDATE_RANDOM_COUNT } else { "128" }

$DefaultMethods = "sacsa_reserve,dfl_pto"
$Methods = if ($env:METHODS) {
  @($env:METHODS -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
} else {
  @($DefaultMethods -split ",")
}

$ReserveSacMethods = @("sacsa_reserve", "sac_logistic_normal_reserve")
$DryRun = $env:DRY_RUN -eq "1" -or $env:DRY_RUN -eq "true" -or $env:DRY_RUN -eq "True"

$SeedTag = ($Seeds -join "x")
$UseAutoAlpha = $AutoAlpha -eq "1" -or $AutoAlpha -eq "true" -or $AutoAlpha -eq "True"
$PessTag = Convert-ToTag $LambdaPess
$InitShareTag = Convert-ToTag $InitShare
$MaxTempTag = Convert-ToTag $MaxTemperature
$FixedTempTag = Convert-ToTag $FixedTemperature
$TempModeTag = if ($UseAutoAlpha) { "a1t${MaxTempTag}" } else { "a0t${FixedTempTag}" }
$RunTagBase = if ($env:RUN_TAG) {
  $env:RUN_TAG
} else {
  "synT${T}_M${M}_B${Budget}_D${InitialLoggedDays}_N${NewLoggedDays}_q${QMax}_lp${PessTag}_${TempModeTag}_i${InitShareTag}_s${SeedTag}"
}
$LogDir = "logs_$RunTagBase"
$SummaryDir = if ($env:SUMMARY_DIR) {
  $env:SUMMARY_DIR
} else {
  "summary_${RunTagBase}"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

$RealUpdateArg = "--real_online_update"
if ($RealOnlineUpdate -eq "0" -or $RealOnlineUpdate -eq "false" -or $RealOnlineUpdate -eq "False") {
  $RealUpdateArg = "--no_real_online_update"
}

$ResidualArg = "--center_residual"
if ($CenterResidual -eq "0" -or $CenterResidual -eq "false" -or $CenterResidual -eq "False") {
  $ResidualArg = "--no_center_residual"
}

Write-Host "[python] $Python"
Write-Host "[gpu-id] CUDA_VISIBLE_DEVICES=$($env:CUDA_VISIBLE_DEVICES)"
Write-Host "[config] lambda_pess=$LambdaPess auto_alpha=$AutoAlpha fixed_temperature=$FixedTemperature init_share=$InitShare"
& $Python -c "import torch; print('[torch]', torch.__version__, 'cuda=', torch.cuda.is_available()); print('[gpu]', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

foreach ($ThisSeed in $Seeds) {
  foreach ($Method in $Methods) {
    Write-Host "[run] q=$QMax seed=$ThisSeed method=$Method device=$Device"
    $SeedPart = if ($Seeds.Count -gt 1) { "seed${ThisSeed}_" } else { "" }
    $OutDir = "results_${RunTagBase}_${SeedPart}${Method}"
    $LogFile = Join-Path $LogDir "${SeedPart}${Method}.log"
    if ($ReserveSacMethods -contains $Method) {
      if ($UseAutoAlpha) {
        $TempArgs = @(
          "--auto_alpha",
          "--init_temperature", $InitTemperature,
          "--min_temperature", $MinTemperature,
          "--max_temperature", $MaxTemperature,
          "--target_entropy_scale", $TargetEntropyScale,
          "--sac_entropy_space", $SacEntropySpace
        )
      } else {
        $TempArgs = @(
          "--no_auto_alpha",
          "--fixed_temperature", $FixedTemperature,
          "--sac_entropy_space", $SacEntropySpace
        )
      }
    } else {
      $TempArgs = @("--auto_alpha", "--max_temperature", $MaxTemperature)
    }
    $CmdArgs = @(
      "iterative_synthetic_sacsa.py",
      "--method", $Method,
      "--device", $Device,
      "--seed", $ThisSeed,
      "--M", $M,
      "--T", $T,
      "--daily_budget", $Budget,
      "--q_max", $QMax,
      "--initial_logged_days", $InitialLoggedDays,
      "--new_logged_days", $NewLoggedDays,
      "--validation_logged_days", $ValidationLoggedDays,
      "--holdout_logged_days", $HoldoutLoggedDays,
      "--sim_update_mode", "finetune",
      "--sim_epochs", $SimEpochs,
      "--sim_finetune_epochs", $SimFinetuneEpochs,
      "--sim_batch_size", $SimBatchSize,
      "--sim_lr", $SimLr,
      "--sim_hidden_layers", "512", "256", "128",
      "--sim_dropout", $SimDropout,
      "--sim_weight_decay", $SimWeightDecay,
      "--logging_policy", "rich",
      $ResidualArg,
      "--use_pessimism",
      "--lambda_pess", $LambdaPess,
      "--b_boot", $Bboot,
      "--bootstrap_epochs", $BootstrapEpochs,
      "--bootstrap_finetune_epochs", $BootstrapFinetuneEpochs,
      "--synthetic_train_episodes", $SyntheticTrainEpisodes,
      "--rollout_batch_size", $RolloutBatchSize,
      "--warmup_episodes", $WarmupEpisodes,
      "--update_after_episodes", $UpdateAfterEpisodes,
      "--gradient_steps_per_update", $GradientStepsPerUpdate,
      "--batch_rollout_update_cap", $BatchRolloutUpdateCap,
      "--batch_size", $BatchSize,
      "--replay_size", $ReplaySize,
      $RealUpdateArg,
      "--real_batch_size", $RealBatchSize,
      "--real_update_after_steps", $RealUpdateAfterSteps,
      "--real_gradient_steps_per_step", $RealGradientStepsPerStep,
      "--candidate_random_count", $CandidateRandomCount,
      "--dfl_epochs", $DflEpochs,
      "--dfl_batch_size", $DflBatchSize,
      "--validation_episodes", $ValidationEpisodes,
      "--eval_episodes", $EvalEpisodes,
      "--dirichlet_init_total_spend_share", $InitShare,
      "--ln_init_total_spend_share", $InitShare,
      "--td3_init_total_spend_share", $InitShare
    ) + $TempArgs + @(
      "--save_models",
      "--progress", "compact",
      "--out_dir", $OutDir
    )
    if ($DryRun) {
      Write-Host "[dry-run] $Python $($CmdArgs -join ' ') > $LogFile 2>&1"
    } else {
      $PreviousErrorActionPreference = $ErrorActionPreference
      $ErrorActionPreference = "Continue"
      try {
        & $Python @CmdArgs > $LogFile 2>&1
        $PythonExitCode = $LASTEXITCODE
      } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
      }
      if ($PythonExitCode -ne 0) {
        Write-Host "[failed] seed=$ThisSeed method=$Method exit_code=$PythonExitCode"
        Write-Host "[log-tail] $LogFile"
        Get-Content $LogFile -Tail 100
        throw "Python command failed with exit code $PythonExitCode"
      }
    }
  }
}

Write-Host "[summary] $SummaryDir"
if ($DryRun) {
  Write-Host "[dry-run] $Python summarize_iterative_results.py --root . --pattern results_${RunTagBase}_* --out_dir $SummaryDir"
} else {
  & $Python summarize_iterative_results.py `
    --root . `
    --pattern "results_${RunTagBase}_*" `
    --out_dir $SummaryDir
  if ($LASTEXITCODE -ne 0) {
    throw "summarize_iterative_results.py failed with exit code $LASTEXITCODE"
  }
}

Write-Host "[done] $SummaryDir"
Write-Host "Key outputs:"
Write-Host "  $SummaryDir\result_dirs_manifest.csv"
Write-Host "  $SummaryDir\round_eval_summary_all_methods.csv"
