$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:SEEDS = "2026"
$env:PESS_LAMBDA = "0"
$env:COMPONENT_VARIANTS = "nopess"
$env:COMPONENT_JOB_TAG = "nopess"
& "$PSScriptRoot\run_gpu_q3_component_ablation_windows.ps1"
