# SAC-SA: Expanded RL Baseline Comparison

Code for the additional RL baseline comparison in the KDD 2027 rebuttal (E1 / response Table 2). The package follows the **response's result provenance**: seven methods use the original E1 implementation, and TD3-PiNet uses the subsequent PiNet implementation. Algorithm files, frozen configurations, and environment inputs are copied without modification.

中文整理说明见 [整理说明.md](整理说明.md)。Code and result sources are documented in [docs/PROVENANCE.md](docs/PROVENANCE.md).

**Public repository scope:** A/A T6/T12 code is included, but the 34 confidential files under those scenarios' `data_sim/` and `aa_data_based_simulator_outputs/` directories are withheld. Their original hashes and configuration references remain intact. Restore the authorized private inputs before running A/A scenarios or `verify_package.py`, which checks the complete five-scenario package. Synthetic T6/T12 and Public-grid T20 include their required inputs.

## Methods and scenarios

| CLI method | Reported name | Source used by the launcher |
|---|---|---|
| `sacsa_reserve` | SAC-SA | Original E1 |
| `td3_direct_softmax` | TD3-Feasible | Original E1 |
| `td3_projection` | TD3-Projection | Original E1 |
| `dfl_pto` | DFL-PTO | Original E1 |
| `cql_ft` | CQL-FT | Original E1 |
| `mopo_ft` | MOPO-FT | Original E1 |
| `sac_lag` | SAC-Lag+Filter | Original E1 |
| `td3_pinet` | TD3-Πnet | E1-next, PiNet only |

Operational references are evaluated by the original scenario pipelines. Best op. is the best mean among those references; it has no training-seed SD.

| CLI scenario | Setting | Cities | Horizon | Initial log episodes | New episodes per round |
|---|---|---:|---:|---:|---:|
| `synthetic_t6` | Synthetic T6 | 6 | 6 | 800 | 100 |
| `synthetic_t12` | Synthetic T12 | 6 | 12 | 1,500 | 150 |
| `aa_t6` | A/A-calibrated T6 | 2 | 6 | 1,000 | 100 |
| `aa_t12` | A/A-calibrated T12 | 2 | 12 | 1,500 | 200 |
| `grid_t20` | Public-grid T20 | 3 | 20 | 4,500 | 300 |

Formal evaluation uses budget B=10, final round q=3, training seeds 2026–2030, and 2,000 evaluation episodes per trained policy. The scenario-specific training and data-use schedules are preserved. SAC-SA/CQL/MOPO use deterministic distribution-mean actions; TD3 uses noise-free actions; SAC-Lag evaluates its latent mean after sigmoid and the execution filter. Evaluation dynamics remain stochastic.

## Contents

```text
run_baselines.py          Unified entry point; routes each method to its recorded source
summarize_baselines.py    Aggregate new runs at their configured final round
verify_package.py        Hash, syntax, configuration, and recorded-summary checks
experiments/
  original_e1/            Five standalone scenario packages for the seven E1 methods
  pinet/                 Five standalone E1-next packages, used only for TD3-PiNet
results/
  recorded/              200 seed-level records and their E1 + PiNet aggregate
provenance/              Copied-file hashes, recorded run configurations/environments,
                         and this packaging task's validation report
docs/                    Code/result provenance explanation
```

The two source families are intentionally kept separate. Combining their shared modules would change the source identities associated with the results. Some legacy methods and configuration files remain inside these frozen packages because the original protocol verifiers enumerate them; the root launcher exposes only the eight methods above. Use the root launcher for the response comparison.

## Environment

Install a PyTorch build appropriate for your machine, then install the non-PyTorch dependencies:

```bash
# Python 3.11+ dependency list inherited from the source package
python -m pip install -r requirements.txt
# For the inherited Python 3.10 compatibility list, use requirements-py310.txt instead.
```

The eight selected methods do not require JAX or CVXPY/cvxpylayers. The retained, unused HardNet class imports its solver dependencies only when instantiated.

These requirement lists are inherited launch dependencies, **not a single environment lock for all recorded results**. Saved runs used Python 3.12.3 / PyTorch 2.11.0 on PPU-ZW810; the original E1 and PiNet batches used different NumPy/pandas/SciPy versions, and PiNet itself spans two recorded NumPy/pandas combinations. Every selected run's recorded environment is retained in `provenance/recorded_runs.json`. See [docs/PROVENANCE.md](docs/PROVENANCE.md). This packaging task did not reinstall these historical environments.

## Check and run

Run these commands from this directory. The wrapper sets each subprocess's working directory, so relative data paths and the Grid calibration cache remain valid.

```bash
# Complete-package check; requires the private A/A inputs to have been restored
python verify_package.py

# Validate 120 formal configurations for the three public scenarios; no training
python run_baselines.py --profile formal --scenarios synthetic_t6 synthetic_t12 grid_t20 --dry-run --device cpu --output-root runs/public_formal_plan

# Optional small execution check: one scenario, one method, one seed
python run_baselines.py --profile smoke --scenarios synthetic_t6 --methods cql_ft --seeds 2026 --device cuda --output-root runs/smoke_cql

# Four additional baselines across three public scenarios and five seeds: 60 jobs
python run_baselines.py --profile formal --scenarios synthetic_t6 synthetic_t12 grid_t20 --methods cql_ft mopo_ft sac_lag td3_pinet --device cuda --output-root runs/additional_baselines

# Complete eight-method comparison: 200 jobs; requires the private A/A inputs
python run_baselines.py --profile formal --device cuda --output-root runs/full_comparison

# Combine the completed campaign's final-round results
python summarize_baselines.py runs/full_comparison
```

Execution is serial. Select scenarios/methods/seeds explicitly to divide work across machines, using different output roots. Existing output roots are refused rather than overwritten. Failed jobs remain recorded and stop the queue; automatic resume is not added. `--dry-run` executes the original protocol/configuration checks and requires PyTorch, but performs no training or evaluation.

`smoke` uses the original small diagnostic overrides. It is not a formal experiment, and Grid's smoke profile relaxes the simulator quality gate. Formal runs retain the original quality gate. Smoke results must not be used for the response table.

## Results and interpretation

- `results/recorded/seed_results.csv` contains all 200 selected E1/PiNet runs. Mean and sample SD are calculated across the five **policy-level mean returns**, not across pooled evaluation episodes. Original raw evaluation CSV paths are recorded relative to the original result collection; those raw files and trained checkpoints are not bundled.
- `results/recorded/TABLE_ORIGINAL_E1_PLUS_PINET.csv` represents those 200 runs.
- CQL-FT and MOPO-FT are budget-feasible adaptations with scheduled limited-interaction fine-tuning. Original E1 MOPO uses elite models for both sampling and uncertainty; the broader E1-next MOPO variant is not substituted here. SAC-Lag's executed feasibility relies on its filter. TD3-PiNet is the project's PyTorch budget-constrained adaptation.

The aggregate uses final-round evaluations from the 200 recorded E1/PiNet runs. Results can vary across hardware and library versions.

## Validation and source preservation

At extraction, the 200 saved runs' 4,240 recorded code/configuration hashes matched their source directories. All 200 final evaluation CSV hashes were checked; their q=3 rows contained 2,000 unique episodes per run (400,000 total), and recomputed policy means agreed with the saved seed records to 1e-9. These are checks of saved evidence, not new training or checkpoint replay.

The packaged entry point also passed a 200-job formal configuration dry-run. Every generated configuration matched its recorded launch configuration after excluding output/settings paths and the validation device. Runtime imports passed for all ten source packages. The new summarizer reproduced all 40 saved mean/SD pairs from the 200 original evaluation files and rejected dry-run/incomplete inputs. These checks used the existing local Python 3.10.20 environment for runtime imports and configuration generation; no new training or rollout evaluation was performed.

`provenance/packaging_validation.json` records these checks. `verify_package.py` is portable and does not require the original workspace. `.gitattributes` preserves exact file bytes so Git newline conversion does not invalidate the source/configuration hashes.

Keep the hidden `experiments/**/.cache/public_city_panel/*.pt` files: they are frozen target-environment calibration inputs, not disposable training checkpoints. A/A `data_sim/` and `aa_data_based_simulator_outputs/` files must remain private and are ignored by Git. The repository also excludes virtual environments, training checkpoints, smoke outputs, unrelated E2–E11 experiments, and remote-machine schedulers.
