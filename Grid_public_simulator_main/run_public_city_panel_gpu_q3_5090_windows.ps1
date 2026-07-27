$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$GpuId = if ($env:GPU_ID) { $env:GPU_ID } else { "0" }
$env:CUDA_VISIBLE_DEVICES = $GpuId

$M = if ($env:M) { $env:M } else { "3" }
$T = if ($env:T) { $env:T } else { "20" }
$Budget = if ($env:BUDGET) { $env:BUDGET } else { "10" }
$D0 = if ($env:INITIAL_LOGGED_DAYS) { $env:INITIAL_LOGGED_DAYS } else { "3000" }
$NewDays = if ($env:NEW_LOGGED_DAYS) { $env:NEW_LOGGED_DAYS } else { "200" }
$QMax = if ($env:Q_MAX) { $env:Q_MAX } else { "3" }
$Seed = if ($env:SEED) { $env:SEED } else { "2026" }
$Methods = if ($env:METHODS) { $env:METHODS -split "[, ]+" } else { @("sacsa_reserve", "dfl_pto") }

$LambdaPess = if ($env:LAMBDA_PESS) { $env:LAMBDA_PESS } else { "0.02" }
$AutoAlpha = if ($env:AUTO_ALPHA) { $env:AUTO_ALPHA } else { "0" }
$Temperature = if ($env:FIXED_TEMPERATURE) { $env:FIXED_TEMPERATURE } else { "0.05" }
$InitShare = if ($env:INIT_SHARE) { $env:INIT_SHARE } else { "0.10" }
$BehaviorSupport = if ($env:BEHAVIOR_SUPPORT_LAMBDA) { $env:BEHAVIOR_SUPPORT_LAMBDA } else { "0.05" }
$SupportPrototypes = if ($env:SUPPORT_PROTOTYPE_COUNT) { $env:SUPPORT_PROTOTYPE_COUNT } else { "512" }
$RealExploration = if ($env:REAL_EXPLORATION_EPSILON) { $env:REAL_EXPLORATION_EPSILON } else { "0.40" }
$SimulatorSelectionEpisodes = if ($env:SIMULATOR_SELECTION_EPISODES) { $env:SIMULATOR_SELECTION_EPISODES } else { "1200" }
$SimEpochs = if ($env:SIM_EPOCHS) { $env:SIM_EPOCHS } else { "300" }
$BootstrapModels = if ($env:B_BOOT) { $env:B_BOOT } else { "8" }
$PolicyEpisodes = if ($env:SYNTHETIC_TRAIN_EPISODES) { $env:SYNTHETIC_TRAIN_EPISODES } else { "6000" }
$LaterPolicyEpisodes = if ($env:SYNTHETIC_TRAIN_EPISODES_LATER_ROUNDS) { $env:SYNTHETIC_TRAIN_EPISODES_LATER_ROUNDS } else { "0" }
$RealUpdateRatio = if ($env:REAL_GRADIENT_STEPS_PER_STEP) { $env:REAL_GRADIENT_STEPS_PER_STEP } else { "1.0" }
$RealCriticWarmup = if ($env:REAL_CRITIC_WARMUP_UPDATES) { $env:REAL_CRITIC_WARMUP_UPDATES } else { "0" }
$EvalEpisodes = if ($env:EVAL_EPISODES) { $env:EVAL_EPISODES } else { "1000" }
$ResidualDays = if ($env:RESIDUAL_CALIBRATION_DAYS) { $env:RESIDUAL_CALIBRATION_DAYS } else { "800" }
$ValidationDays = if ($env:VALIDATION_LOGGED_DAYS) { $env:VALIDATION_LOGGED_DAYS } else { "600" }
$HoldoutDays = if ($env:HOLDOUT_LOGGED_DAYS) { $env:HOLDOUT_LOGGED_DAYS } else { "600" }
$DgpDays = if ($env:DGP_CALIBRATION_DAYS) { $env:DGP_CALIBRATION_DAYS } else { "10000" }
$DgpEpochs = if ($env:DGP_BASELINE_EPOCHS) { $env:DGP_BASELINE_EPOCHS } else { "100" }
$BootstrapEpochs = if ($env:BOOTSTRAP_EPOCHS) { $env:BOOTSTRAP_EPOCHS } else { "100" }
$DflEpochs = if ($env:DFL_EPOCHS) { $env:DFL_EPOCHS } else { "150" }
$DflBatchSize = if ($env:DFL_BATCH_SIZE) { $env:DFL_BATCH_SIZE } else { "1024" }
$DflCandidates = if ($env:DFL_CANDIDATE_RANDOM_COUNT) { $env:DFL_CANDIDATE_RANDOM_COUNT } else { "128" }
$DflLearningRate = if ($env:DFL_LEARNING_RATE) { $env:DFL_LEARNING_RATE } else { "0.001" }
$DflDecisionWeight = if ($env:DFL_DECISION_WEIGHT) { $env:DFL_DECISION_WEIGHT } else { "1.0" }
$DflSoftmaxTemperature = if ($env:DFL_SOFTMAX_TEMPERATURE) { $env:DFL_SOFTMAX_TEMPERATURE } else { "1.0" }
$DflBudgetFractions = if ($env:DFL_BUDGET_FRACTIONS) { $env:DFL_BUDGET_FRACTIONS } else { "0.25,0.50,0.75,1.00" }
$StrictSimulator = if ($env:STRICT_SIMULATOR_QUALITY) { $env:STRICT_SIMULATOR_QUALITY } else { "1" }
$MinR2Z = if ($env:SIM_MIN_R2_Z) { $env:SIM_MIN_R2_Z } else { "0.75" }
$MinR2ZByT = if ($env:SIM_MIN_R2_Z_BY_T) { $env:SIM_MIN_R2_Z_BY_T } else { "0.30" }

function Compact-Decimal([string]$Value) {
    return $Value.Replace(".", "p")
}

$RunTag = if ($env:RUN_TAG) { "_$($env:RUN_TAG)" } else { "" }
$Tag = "pc${RunTag}_M${M}T${T}_B${Budget}_D${D0}N${NewDays}_q${QMax}_lp$(Compact-Decimal $LambdaPess)_bs$(Compact-Decimal $BehaviorSupport)_a${AutoAlpha}t$(Compact-Decimal $Temperature)_i$(Compact-Decimal $InitShare)_d3e${DflEpochs}c${DflCandidates}b${DflBatchSize}_s${Seed}"
$ResultDirs = @()

Write-Host "[config] GPU=$GpuId $Tag methods=$($Methods -join ',')"
& $Python -c "import torch; print('[torch]', torch.__version__, 'cuda=', torch.cuda.is_available()); print('[gpu]', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
if ($LASTEXITCODE -ne 0) { throw "Python/CUDA preflight failed." }

foreach ($Method in $Methods) {
    if ([string]::IsNullOrWhiteSpace($Method)) { continue }
    $OutDir = "results_${Tag}_${Method}"
    $LogDir = "logs_${Tag}"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $LogFile = Join-Path $LogDir "${Method}.log"
    $CmdArgs = @(
        "train_public_city_panel.py",
        "--method", $Method,
        "--device", "cuda",
        "--seed", $Seed,
        "--M", $M,
        "--T", $T,
        "--daily_budget", $Budget,
        "--q_max", $QMax,
        "--initial_logged_days", $D0,
        "--new_logged_days", $NewDays,
        "--residual_calibration_days", $ResidualDays,
        "--validation_logged_days", $ValidationDays,
        "--holdout_logged_days", $HoldoutDays,
        "--eval_episodes", $EvalEpisodes,
        "--dgp_calibration_days", $DgpDays,
        "--dgp_baseline_epochs", $DgpEpochs,
        "--sim_hidden_dims", "768,512,256",
        "--sim_head_dim", "256",
        "--sim_epochs", $SimEpochs,
        "--sim_finetune_epochs", "120",
        "--sim_batch_size", "4096",
        "--bootstrap_models", $BootstrapModels,
        "--bootstrap_epochs", $BootstrapEpochs,
        "--min_holdout_r2_z", $MinR2Z,
        "--min_holdout_r2_z_by_t", $MinR2ZByT,
        "--policy_hidden_dims", "512,512,256",
        "--synthetic_train_episodes", $PolicyEpisodes,
        "--synthetic_train_episodes_later_rounds", $LaterPolicyEpisodes,
        "--rollout_batch_size", "256",
        "--updates_per_rollout_batch", "128",
        "--policy_batch_size", "1024",
        "--init_spend_share", $InitShare,
        "--lambda_pess", $LambdaPess,
        "--fixed_temperature", $Temperature,
        "--real_gradient_steps_per_step", $RealUpdateRatio,
        "--real_critic_warmup_updates", $RealCriticWarmup,
        "--real_exploration_epsilon", $RealExploration,
        "--behavior_support_lambda", $BehaviorSupport,
        "--support_prototype_count", $SupportPrototypes,
        "--simulator_selection_episodes", $SimulatorSelectionEpisodes,
        "--dfl_epochs", $DflEpochs,
        "--dfl_batch_size", $DflBatchSize,
        "--dfl_candidate_random_count", $DflCandidates,
        "--dfl_hidden_dims", "256,256",
        "--dfl_learning_rate", $DflLearningRate,
        "--dfl_decision_weight", $DflDecisionWeight,
        "--dfl_softmax_temperature", $DflSoftmaxTemperature,
        "--dfl_budget_fractions", $DflBudgetFractions,
        "--out_dir", $OutDir
    )
    if ($AutoAlpha -eq "1") { $CmdArgs += "--auto_alpha" }
    if ($StrictSimulator -eq "1") { $CmdArgs += "--strict_simulator_quality" }
    Write-Host "[run] method=$Method out=$OutDir"
    $OldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @CmdArgs *> $LogFile
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $OldPreference
    if ($ExitCode -ne 0) {
        Get-Content $LogFile -Tail 50
        throw "Python command failed with exit code $ExitCode"
    }
    $ResultDirs += $OutDir
}

if ($ResultDirs.Count -gt 0) {
    & $Python "summarize_public_city_panel.py" --result_dirs $ResultDirs --out_dir "summary_${Tag}"
    if ($LASTEXITCODE -ne 0) { throw "Result summarization failed." }
}
Write-Host "[done] summary_${Tag}"
