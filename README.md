# SAC-SA Anonymous Reproduction Artifact

The core experiment directories contain the code used for the
fully synthetic and public-grid-calibrated experiments in the paper. The real-data-calibrated
semi-synthetic experiment is not included because its source data are
confidential.

## Scope

| Paper experiment | Directory | Entry point |
|---|---|---|
| Fully synthetic, \(T=6\) | `SAC-SA-T6_main` | `run_gpu_q3_5090_windows.ps1` |
| Fully synthetic, \(T=12\) | `SAC-SA-T12_main` | `run_gpu_q3_5090_windows.ps1` |
| Public-grid main experiment | `Grid_public_simulator_main` | `run_public_grid_main_5090.ps1` |
| Synthetic one-factor sensitivity | `SAC-SA-T6_main_factor-multisetting` | four `run_gpu_q3_sweep_*_windows.ps1` scripts |
| Public-grid one-factor sensitivity | `Grid_public_simulator_factor-multisetting` | five `run_gpu_q3_sweep_*_windows.ps1` scripts |
| Pessimism and observed-replay ablations | `SAC-SA-T6_main_pessi_replay` | five `run_comp_*_5090.ps1` scripts |
| SAC-SA and DFL-PTO training seeds | `SAC-SA-T6_main_train_seeds` | `run_gpu_q3_fixed_dgp_training_seeds_windows.ps1` |
| TD3 training seeds | `SAC-SA-T6_main_train_seeds_td3` | `run_gpu_q3_td3_seeds.ps1` |
| Actor parameterizations | `SAC-SA-T6_main_actor_compare` | `run_gpu_q3_actor_comparison_windows.ps1` |

Raw episode files, summaries, figures, and checkpoints are intentionally
excluded. Each experiment regenerates its outputs locally.

## RL Baseline Comparison

The [SAC-SA-RL-Baselines](SAC-SA-RL-Baselines/README.md) supplement adds
CQL-FT, MOPO-FT, SAC-Lag+Filter and TD3-PiNet, together with SAC-SA,
TD3-Feasible, TD3-Projection and DFL-PTO controls. It includes frozen
configurations, execution and summary scripts, and seed-level result records.


Synthetic T6/T12 and Public-grid T20 include their required public inputs.
A/A T6/T12 code is included, but its confidential data and calibration
artifacts are excluded. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) and
the supplement README before running those scenarios. The supplement retains
its own dependency lists and source/configuration integrity checks.

## Controlled Code Variants

The experiment folders are standalone snapshots so each entry point can run
without importing another experiment directory. The artifact audit checks the
following relationships:

- all Synthetic T=6 folders use the same `DGP_sim_env.py`;
- the Synthetic T=6 main and one-factor folders use identical simulator,
  policy, and training-loop source;
- the component folder adds only the pessimism and observed-replay switches;
- the fixed-DGP folders separate training randomness from the fixed DGP, data,
  and evaluation seeds;
- the actor folder adds the four alternative feasible actors and Gaussian
  projection;
- the public-grid sensitivity folder adds a sixth city specification for the
  \(M=6\) sensitivity run. Its \(M=3\) reference setting uses the first three
  city specifications, as in the main experiment.

## Environment

The experiments were run with Python 3.11 and CUDA-enabled PyTorch. Install a
PyTorch build compatible with the local NVIDIA driver, then install the
remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

The scripts use `python` from the active environment. To select another
interpreter:

```powershell
$env:PYTHON="C:\path\to\python.exe"
```

Select a GPU with `GPU_ID`:

```powershell
$env:GPU_ID="0"
```

## Main Experiments

```powershell
Set-Location .\SAC-SA-T6_main
.\run_gpu_q3_5090_windows.ps1

Set-Location ..\SAC-SA-T12_main
.\run_gpu_q3_5090_windows.ps1

Set-Location ..\Grid_public_simulator_main
.\run_public_grid_main_5090.ps1
```

The two synthetic runners support a command-only check:

```powershell
$env:DRY_RUN="1"
.\run_gpu_q3_5090_windows.ps1
Remove-Item Env:\DRY_RUN
```

The locked main configurations are:

| setting | \(M\) | \(T\) | \(B\) | \(D_0\) | \(n^{new}\) | rounds | seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Synthetic T=6 | 6 | 6 | 10 | 800 | 100 | 3 | 2026 |
| Synthetic T=12 | 6 | 12 | 10 | 1500 | 150 | 3 | 2026 |
| Public grid | 3 | 20 | 10 | 4500 | 300 | 3 | 2026 |

## Sensitivity Analyses

The synthetic sensitivity study changes one factor at a time around
\(D_0=800\), \(n^{\mathrm{new}}=100\), \(M=6\), and seed 2026:

```powershell
Set-Location .\SAC-SA-T6_main_factor-multisetting
.\run_gpu_q3_sweep_d0_windows.ps1
.\run_gpu_q3_sweep_newdays_windows.ps1
.\run_gpu_q3_sweep_m_windows.ps1
.\run_gpu_q3_sweep_seed_windows.ps1
```

The public-grid sensitivity study changes \(B\), \(D_0\),
\(n^{\mathrm{new}}\), \(M\), and the seed around the main configuration:

```powershell
Set-Location .\Grid_public_simulator_factor-multisetting
.\run_gpu_q3_sweep_budget_windows.ps1
.\run_gpu_q3_sweep_d0_windows.ps1
.\run_gpu_q3_sweep_newdays_windows.ps1
.\run_gpu_q3_sweep_m_windows.ps1
.\run_gpu_q3_sweep_seed_windows.ps1
```

Run these wrappers sequentially in one checkout. They share the main-setting
output directory and use lock files to prevent concurrent writes. Independent
settings may instead be run from separate copies of the artifact.

## Ablations and Training Seeds

The component study changes only pessimism or observed replay. The five
wrappers use `lambda_pess` values 0, 0.005, 0.02, 0.05, and 0.10; every
nonzero setting runs both full SAC-SA and its no-replay variant.

```powershell
Set-Location .\SAC-SA-T6_main_pessi_replay
.\run_comp_nopess_5090.ps1
.\run_comp_lp005_5090.ps1
.\run_comp_lp02_5090.ps1
.\run_comp_lp05_5090.ps1
.\run_comp_lp10_5090.ps1

Set-Location ..\SAC-SA-T6_main_actor_compare
.\run_gpu_q3_actor_comparison_windows.ps1
```

The actor comparison retrains Beta-Dirichlet, Beta-Logistic-Normal, Direct
Dirichlet, Direct Logistic-Normal, and Gaussian Projection actors while keeping
the Synthetic T=6 setting fixed.

The fixed-DGP studies vary only training randomness over seeds 2026--2030.
Their complete locked configurations are stored in
`synthetic_t6_fixed_dgp_protocol.json` and `synthetic_t6_td3_protocol.json`.

```powershell
Set-Location ..\SAC-SA-T6_main_train_seeds
.\run_gpu_q3_fixed_dgp_training_seeds_windows.ps1

Set-Location ..\SAC-SA-T6_main_train_seeds_td3
.\run_gpu_q3_td3_seeds.ps1
```

Each long-running job should use its own GPU. Do not launch two wrappers that
write to the same result or log directory.

## Reproducibility Checks

Run the artifact audit from the repository root:

```powershell
python .\tools\check_artifact.py
```

The check verifies expected entry points, JSON files, Python syntax, absence of
known personal paths, absence of generated checkpoints or large files, core
source consistency, and the locked five-seed protocols.

This audit retains the core benchmark packaging restrictions. With the
RL baseline package present, it also flags the required Grid calibration
files and source-provenance names; it scans locally ignored A/A inputs as
well. Its implementation is unchanged. The supplement's
`verify_package.py` checks its own source/configuration hashes and recorded
summaries, and requires the private A/A inputs for a full-package check.
Git publication excludes those private inputs as documented above.

## Data and Licensing

See `DATA_AVAILABILITY.md` for data scope and
`THIRD_PARTY_NOTICE.md` for the public-grid simulator notice. The relevant code
is included with permission from the rights holder; identifying attribution is
withheld during double-blind review and will be restored in the non-anonymous
release.
