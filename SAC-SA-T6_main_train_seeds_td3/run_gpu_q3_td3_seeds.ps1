$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PreviousProtocol = $env:FIXED_DGP_PROTOCOL
$PreviousPrefix = $env:RUN_PREFIX
try {
    $env:FIXED_DGP_PROTOCOL = "synthetic_t6_td3_protocol.json"
    $env:RUN_PREFIX = "synT6_fdg_td3v2"
    & "$PSScriptRoot\run_gpu_q3_fixed_dgp_training_seeds_windows.ps1"
} finally {
    if ($null -eq $PreviousProtocol) {
        Remove-Item Env:\FIXED_DGP_PROTOCOL -ErrorAction SilentlyContinue
    } else {
        $env:FIXED_DGP_PROTOCOL = $PreviousProtocol
    }
    if ($null -eq $PreviousPrefix) {
        Remove-Item Env:\RUN_PREFIX -ErrorAction SilentlyContinue
    } else {
        $env:RUN_PREFIX = $PreviousPrefix
    }
}
