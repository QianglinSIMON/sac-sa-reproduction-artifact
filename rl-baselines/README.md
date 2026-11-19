# SAC-SA RL benchmarks

Compare SAC-SA with seven RL and optimization baselines using fixed scenario configurations. The package includes runnable code for three public scenarios, code for two scenarios requiring confidential inputs, and 200 recorded seed-level results.

## Quick start

Run the following commands from `rl-baselines/`. The integrity check uses only the Python standard library and does not run experiments:

```bash
python verify_package.py
```

For configuration checks or execution, install PyTorch for your hardware and the launch dependencies in a separate Python environment. Use Python 3.12 with `requirements.txt`, or Python 3.10 with `requirements-py310.txt`:

```bash
python -m pip install -r requirements.txt

# Check all 120 public formal configurations; no training or evaluation.
python run_baselines.py --profile formal --dry-run --device cpu --output-root runs/public_plan
```

The default scenarios are `synthetic_t6`, `synthetic_t12`, and `grid_t20`. All eight methods and five seeds are selected for the formal profile. The frozen public-grid calibration inputs are included, so this check does not require confidential data.

To execute a small diagnostic run, then a formal comparison:

```bash
# One diagnostic job; this is not a formal result.
python run_baselines.py --profile smoke --scenarios synthetic_t6 --methods cql_ft --seeds 2026 --device cuda --output-root runs/smoke_cql

# Four additional baselines, three public scenarios, five seeds: 60 training jobs.
python run_baselines.py --profile formal --methods cql_ft mopo_ft sac_lag td3_pinet --device cuda --output-root runs/additional_baselines

# Aggregate a completed formal campaign.
python summarize_baselines.py runs/additional_baselines
```

`--dry-run` imports the scenario runtime and checks its inputs and configurations; it requires PyTorch but performs no training or evaluation. The `smoke` profile uses diagnostic overrides, including a relaxed Grid simulator quality gate. Its outputs must not be used as formal results. Formal runs retain the frozen training schedules and quality gate.

## Methods

| CLI method | Reported name | Implementation directory |
|---|---|---|
| `sacsa_reserve` | SAC-SA | `experiments/original_e1/` |
| `td3_direct_softmax` | TD3-Feasible | `experiments/original_e1/` |
| `td3_projection` | TD3-Projection | `experiments/original_e1/` |
| `dfl_pto` | DFL-PTO | `experiments/original_e1/` |
| `cql_ft` | CQL-FT | `experiments/original_e1/` |
| `mopo_ft` | MOPO-FT | `experiments/original_e1/` |
| `sac_lag` | SAC-Lag+Filter | `experiments/original_e1/` |
| `td3_pinet` | TD3-PiNet | `experiments/pinet/` |

CQL-FT and MOPO-FT are budget-feasible adaptations with scheduled limited-interaction fine-tuning. SAC-Lag's executed feasibility relies on its filter. TD3-PiNet is a PyTorch specialization for the budget constraint. These implementations should not be described as unmodified upstream algorithms; see [implementation provenance](docs/PROVENANCE.md) and [third-party references](../THIRD_PARTY_NOTICE.md).

The launcher selects the implementation associated with each recorded result. The two implementation directories remain separate because they contain different algorithm variants. Internal snapshot identifiers are retained for source traceability. Additional configurations retained by the original protocol verifiers are not exposed by the root launcher.

## Scenarios and protocol

| CLI scenario | Setting | Cities | Horizon | Initial log episodes | New episodes per round | Inputs |
|---|---|---:|---:|---:|---:|---|
| `synthetic_t6` | Fully synthetic | 6 | 6 | 800 | 100 | Included |
| `synthetic_t12` | Fully synthetic | 6 | 12 | 1,500 | 150 | Included |
| `grid_t20` | Public-grid calibrated | 3 | 20 | 4,500 | 300 | Included |
| `aa_t6` | A/A calibrated | 2 | 6 | 1,000 | 100 | Private |
| `aa_t12` | A/A calibrated | 2 | 12 | 1,500 | 200 | Private |

Formal evaluation uses budget B=10, final round q=3, training seeds 2026-2030, and 2,000 evaluation episodes per trained policy. Scenario-specific training and data-use schedules are preserved. SAC-SA/CQL/MOPO use deterministic distribution-mean actions; TD3 uses noise-free actions; SAC-Lag evaluates its latent mean after sigmoid and its execution filter. Evaluation dynamics remain stochastic.

Operational reference policies are evaluated by the scenario pipelines. Best op. is the best mean among those references; it has no training-seed SD.

### Confidential scenarios

The 34 files in the A/A `data_sim/` and `aa_data_based_simulator_outputs/` directories are not distributed. Their expected package-relative paths and hashes are recorded with `availability: private` in `provenance/source_files.json`. Authorized users can restore these inputs and then run:

```bash
python verify_package.py --include-private

# All five scenarios, eight methods, five seeds: 200 training jobs.
python run_baselines.py --profile formal --scenarios synthetic_t6 synthetic_t12 aa_t6 aa_t12 grid_t20 --device cuda --output-root runs/all_scenarios
```

The launcher refuses A/A jobs when the required inputs are absent. The package does not substitute a different DGP for confidential inputs. See [data availability](../DATA_AVAILABILITY.md).

## Outputs and execution

Jobs run serially. Each campaign writes `campaign_manifest.json`, scenario-level job records, and result directories beneath `--output-root`. The summarizer produces `seed_results.csv`, `method_summary.csv`, and `operational_references.csv` from completed final-round evaluations.

Select scenarios, methods, and seeds explicitly to divide work across machines; use a different output root for each campaign. Existing output roots are refused. A failed job stops the queue, and automatic resume is not supported. The wrapper sets each scenario's working directory so relative inputs remain valid.

## Recorded results

- [Seed-level results](results/recorded/seed_results.csv): 200 policy-level mean returns, covering five scenarios, eight methods, and five seeds.
- [Method comparison](results/recorded/method_comparison.md): means and sample SDs across the five policy-level means; also available as [CSV](results/recorded/method_comparison.csv).
- [Unrounded summaries](results/recorded/combined_method_summary.csv) and [operational references](results/recorded/operational_baselines.csv).
- [Recorded run metadata](provenance/recorded_runs.json): scientific configurations, runtime environments, and original code/configuration hashes.

Raw episode-level evaluation files and trained policies are not bundled. The published summaries can be checked from the seed-level records; replaying the saved policies is not possible from this package alone. A/A results are provided as aggregate evidence, while reproducing those scenarios requires authorized inputs.

The recorded runs used Python 3.12.3 and PyTorch 2.11.0 on PPU-ZW810. NumPy, pandas, and SciPy versions differ across batches. The dependency lists are launch requirements, not an identical environment lock for every recorded run. [Provenance details](docs/PROVENANCE.md) document these differences. Results can vary across hardware and library versions.

## Package layout and verification

```text
run_baselines.py          Select scenarios, methods, seeds, and profile
summarize_baselines.py    Aggregate completed campaigns
verify_package.py        Check hashes, syntax, configurations, and summaries
experiments/             Frozen algorithm sources, configurations, and inputs
results/recorded/        Published seed records and summaries
provenance/              Source hashes and recorded run metadata
docs/PROVENANCE.md       Implementation mapping and evidence scope
```

The verifier checks all 328 public manifest entries, Python syntax, 40 routed method/scenario configurations, and the 200 recorded seeds. It recomputes the 40 mean/SD pairs. `--include-private` additionally checks the 34 confidential inputs when supplied locally. Neither check trains or replays a policy.

Keep `experiments/**/.cache/public_city_panel/*.pt`: these two files define the frozen target environment. They are required calibration inputs, not disposable training outputs. `.gitattributes` preserves file bytes for hash verification; `.gitignore` excludes private A/A inputs and transient run outputs.
