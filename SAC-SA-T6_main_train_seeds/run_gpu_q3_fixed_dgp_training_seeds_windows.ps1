$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$GpuId = if ($env:GPU_ID) { $env:GPU_ID } elseif ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:CUDA_VISIBLE_DEVICES = $GpuId
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "4" }
$env:MKL_NUM_THREADS = if ($env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS } else { "4" }
$env:NUMEXPR_NUM_THREADS = if ($env:NUMEXPR_NUM_THREADS) { $env:NUMEXPR_NUM_THREADS } else { "4" }
$env:MPLCONFIGDIR = if ($env:MPLCONFIGDIR) { $env:MPLCONFIGDIR } else { Join-Path $PSScriptRoot ".matplotlib" }

$ProtocolPath = if ($env:FIXED_DGP_PROTOCOL) {
  if ([System.IO.Path]::IsPathRooted($env:FIXED_DGP_PROTOCOL)) {
    $env:FIXED_DGP_PROTOCOL
  } else {
    Join-Path $PSScriptRoot $env:FIXED_DGP_PROTOCOL
  }
} else {
  Join-Path $PSScriptRoot "synthetic_t6_fixed_dgp_protocol.json"
}
if (-not (Test-Path $ProtocolPath)) {
  throw "Missing locked protocol: $ProtocolPath"
}
$Protocol = Get-Content $ProtocolPath -Raw | ConvertFrom-Json
$A = $Protocol.locked_args
$Fixed = $Protocol.fixed_seeds
$DryRun = $env:DRY_RUN -in @("1", "true", "True")
$ForceRerun = $env:FORCE_RERUN -in @("1", "true", "True")
$RunPrefix = if ($env:RUN_PREFIX) { $env:RUN_PREFIX } else { "synT6_fdg2" }
$LogDir = "logs_$RunPrefix"
$SummaryDir = "summary_$RunPrefix"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

Write-Host "[protocol] $($Protocol.protocol_id)"
Write-Host "[fixed] DGP=$($Fixed.dgp_seed) data=$($Fixed.data_seed) evaluation=$($Fixed.evaluation_seed)"
Write-Host "[training-seeds] $($Protocol.training_seeds -join ',')"
Write-Host "[methods] $($Protocol.methods -join ',')"
Write-Host "[gpu-id] CUDA_VISIBLE_DEVICES=$($env:CUDA_VISIBLE_DEVICES)"
& $Python -c "import torch; print('[torch]', torch.__version__, 'cuda=', torch.cuda.is_available()); print('[gpu]', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

foreach ($TrainingSeed in $Protocol.training_seeds) {
  foreach ($Method in $Protocol.methods) {
    $OutDir = "results_${RunPrefix}_ts${TrainingSeed}_${Method}"
    $LogFile = Join-Path $LogDir "ts${TrainingSeed}_${Method}.log"
    if ((-not $ForceRerun) -and (Test-Path (Join-Path $OutDir "final_eval_raw.csv"))) {
      Write-Host "[skip-complete] training_seed=$TrainingSeed method=$Method out=$OutDir"
      Copy-Item $ProtocolPath (Join-Path $OutDir "protocol_lock.json") -Force
      continue
    }
    $CmdArgs = @(
      "iterative_synthetic_sacsa.py",
      "--protocol_id", $Protocol.protocol_id,
      "--method", $Method,
      "--device", "cuda",
      "--seed", $TrainingSeed,
      "--training_seed", $TrainingSeed,
      "--dgp_seed", $Fixed.dgp_seed,
      "--data_seed", $Fixed.data_seed,
      "--evaluation_seed", $Fixed.evaluation_seed,
      "--M", $A.M,
      "--T", $A.T,
      "--daily_budget", $A.daily_budget,
      "--q_max", $A.q_max,
      "--initial_logged_days", $A.initial_logged_days,
      "--new_logged_days", $A.new_logged_days,
      "--validation_logged_days", $A.validation_logged_days,
      "--holdout_logged_days", $A.holdout_logged_days,
      "--logging_policy", $A.logging_policy,
      "--logging_spend_scale", $A.logging_spend_scale,
      "--sim_update_mode", $A.sim_update_mode,
      "--sim_epochs", $A.sim_epochs,
      "--sim_finetune_epochs", $A.sim_finetune_epochs,
      "--sim_batch_size", $A.sim_batch_size,
      "--sim_lr", $A.sim_lr,
      "--sim_hidden_layers", $A.sim_hidden_layers[0], $A.sim_hidden_layers[1], $A.sim_hidden_layers[2],
      "--sim_dropout", $A.sim_dropout,
      "--sim_weight_decay", $A.sim_weight_decay,
      "--center_residual",
      "--use_pessimism",
      "--lambda_pess", $A.lambda_pess,
      "--b_boot", $A.b_boot,
      "--bootstrap_epochs", $A.bootstrap_epochs,
      "--bootstrap_finetune_epochs", $A.bootstrap_finetune_epochs,
      "--synthetic_train_episodes", $A.synthetic_train_episodes,
      "--rollout_batch_size", $A.rollout_batch_size,
      "--warmup_episodes", $A.warmup_episodes,
      "--update_after_episodes", $A.update_after_episodes,
      "--gradient_steps_per_update", $A.gradient_steps_per_update,
      "--batch_rollout_update_cap", $A.batch_rollout_update_cap,
      "--batch_size", $A.batch_size,
      "--replay_size", $A.replay_size,
      "--real_online_update",
      "--real_batch_size", $A.real_batch_size,
      "--real_update_after_steps", $A.real_update_after_steps,
      "--real_gradient_steps_per_step", $A.real_gradient_steps_per_step,
      "--real_exploration_epsilon", $A.real_exploration_epsilon,
      "--hidden_dim", $A.hidden_dim,
      "--lr_actor", $A.lr_actor,
      "--lr_critic", $A.lr_critic,
      "--lr_alpha", $A.lr_alpha,
      "--gamma", $A.gamma,
      "--tau", $A.tau,
      "--critic_loss", $A.critic_loss,
      "--fixed_temperature", $A.fixed_temperature,
      "--sac_entropy_space", $A.sac_entropy_space,
      "--dirichlet_init_total_spend_share", $A.dirichlet_init_total_spend_share,
      "--dirichlet_init_concentration", $A.dirichlet_init_concentration,
      "--td3_init_total_spend_share", $(if ($null -ne $A.td3_init_total_spend_share) { $A.td3_init_total_spend_share } else { $A.dirichlet_init_total_spend_share }),
      "--candidate_random_count", $A.candidate_random_count,
      "--dfl_epochs", $A.dfl_epochs,
      "--dfl_batch_size", $A.dfl_batch_size,
      "--dfl_lr", $A.dfl_lr,
      "--dfl_decision_weight", $A.dfl_decision_weight,
      "--dfl_softmax_temperature", $A.dfl_softmax_temperature,
      "--validation_episodes", $A.validation_episodes,
      "--eval_episodes", $A.eval_episodes,
      "--final_eval_seed_offset", $A.final_eval_seed_offset,
      "--no_auto_accelerator_defaults",
      "--save_models",
      "--progress", "compact",
      "--out_dir", $OutDir
    )

    Write-Host "[run] training_seed=$TrainingSeed method=$Method out=$OutDir"
    if ($DryRun) {
      Write-Host "[dry-run] $Python $($CmdArgs -join ' ')"
      continue
    }

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
      & $Python @CmdArgs > $LogFile 2>&1
      $PythonExitCode = $LASTEXITCODE
    } finally {
      $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($PythonExitCode -ne 0) {
      Write-Host "[failed] training_seed=$TrainingSeed method=$Method exit_code=$PythonExitCode"
      Get-Content $LogFile -Tail 120
      throw "Python command failed with exit code $PythonExitCode"
    }
    Copy-Item $ProtocolPath (Join-Path $OutDir "protocol_lock.json") -Force
  }
}

if ($DryRun) {
  Write-Host "[dry-run] summary skipped"
  exit 0
}

& $Python summarize_iterative_results.py --root . --pattern "results_${RunPrefix}_*" --out_dir $SummaryDir
if ($LASTEXITCODE -ne 0) {
  throw "summarize_iterative_results.py failed with exit code $LASTEXITCODE"
}
& $Python summarize_fixed_dgp_training_seeds.py `
  --root . `
  --pattern "results_${RunPrefix}_*" `
  --protocol $ProtocolPath `
  --out_dir $SummaryDir
if ($LASTEXITCODE -ne 0) {
  throw "summarize_fixed_dgp_training_seeds.py failed with exit code $LASTEXITCODE"
}
Copy-Item $ProtocolPath (Join-Path $SummaryDir "protocol_lock.json") -Force

Write-Host "[done] $SummaryDir"
Write-Host "  $SummaryDir\paired_gap_by_training_seed.csv"
Write-Host "  $SummaryDir\paired_gap_training_seed_summary.csv"
Write-Host "  $SummaryDir\return_training_seed_summary.csv"
Write-Host "  $SummaryDir\protocol_verification.csv"
