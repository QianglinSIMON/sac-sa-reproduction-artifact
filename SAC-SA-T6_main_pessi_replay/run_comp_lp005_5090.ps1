$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:SEEDS = "2026"
$env:PESS_LAMBDA = "0.005"
$env:COMPONENT_VARIANTS = "full,noreplay"
$env:COMPONENT_JOB_TAG = "lp0p005"
& "$PSScriptRoot\run_gpu_q3_component_ablation_windows.ps1"
