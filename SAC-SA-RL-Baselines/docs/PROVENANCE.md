# Implementation and result provenance

This package combines frozen implementations and recorded final-round results across five scenarios. It contains seven methods from the original E1 archive and TD3-PiNet from the subsequent implementation. The source records associate each result with its implementation and configuration hashes.

## Source mapping

| Packaged component | Source archive within the original result collection |
|---|---|
| `experiments/original_e1/*` | `SAC-E1-simple/*_E1_baselines_5seeds/` |
| `experiments/pinet/*` | `E1-next-simple/*_E1_baselines_5seeds_next/` |
| 175 recorded runs for the seven original methods | `SAC-E1-simple/formal_all_ppu12_20260929/` |
| 20 PiNet runs: Synthetic T6/T12, A/A T6, Grid T20 | `E1-next-simple/formal_shard*_8gpu_w3_nohup_20261001/jobs/` |
| Five completed A/A T12 PiNet runs | `E1-rest-simple-T12-AA-td3-pinet/` |
| Recorded aggregate and seed provenance | `E1_pinet_replacement_20261001/` |

The A/A T12 PiNet completion package supplies the completed results, while the E1-next scenario directory supplies its implementation. The saved run hashes match that implementation. No incomplete snapshot is counted alongside its completed replacement.

For the public GitHub supplement, the 34 A/A data/residual/calibration files under `data_sim/` and `aa_data_based_simulator_outputs/` are withheld and ignored by Git. Their original source-manifest entries remain unchanged so authorized users can verify the private inputs when supplied locally. The full-package verifier requires these inputs; the three public scenarios can run without them.

## Why two implementation directories are needed

The later E1-next sources change the Beta gate parameterization and MOPO uncertainty computation in addition to introducing PiNet. The seven non-PiNet methods use the original E1 definitions associated with their recorded results. Replacing them with all of E1-next would make the released implementation inconsistent with those records.

The common root launcher routes only `td3_pinet` to E1-next. It does not enable the development learning-rate plans, `mopo_short_ft`, `sac_fixed_penalty` or `td3_hardnet_cvx`. Supporting source/config files for these branches are retained where imported by the frozen runtime or enumerated by its protocol verifier. The algorithm files are byte-identical copies; no losses, DGPs, budgets, training schedules, checkpoint-selection rules, or evaluation rules were changed during extraction.

## Recorded environment combinations

The following values come from the selected runs' `runtime_environment.json` files, not from the README defaults:

| Runs | Python | PyTorch | NumPy | pandas | SciPy |
|---|---|---|---|---|---|
| Original E1, 175 | 3.12.3 | 2.11.0 | 2.3.5 | 3.0.5 | 1.17.1 |
| PiNet, 12 | 3.12.3 | 2.11.0 | 2.2.6 | 2.3.3 | 1.11.3 |
| PiNet, 13 | 3.12.3 | 2.11.0 | 1.26.4 | 2.2.3 | 1.11.3 |

The records identify PPU-ZW810 and CUDA runtime 13.0. Full per-run associations are in `provenance/recorded_runs.json`. Shared method configurations do not imply an identical software environment across these batches. The inherited requirement files are retained as launch dependency lists; no claim of an environment-identical reproduction is made.

## Provenance files and scope

- `source_files.json`: original path relative to its recorded source base, packaged path, SHA-256 and size. Translated documentation retains its original source hash alongside the updated packaged-file hash.
- `recorded_runs.json`: selected run identity, original launch configuration, baseline settings, saved runtime environment and source hashes. Historical absolute paths inside these immutable records are provenance, not dependencies of the root launcher.
- `packaging_validation.json`: checks actually performed during extraction, separated from historical validation.
- `results/recorded/seed_results.csv`: portable source paths and verified policy-level means for all 200 selected runs.

The raw episode-level evaluation files and policy checkpoints remain in the original collection. Machine-specific schedulers, development checks requiring the original checkout, and unrelated experiments are not included.
