# Implementation and result provenance

The published comparison associates each of 200 recorded runs with its implementation, scientific configuration, and software environment. Seven methods use the frozen baseline implementation; TD3-PiNet uses a separate PiNet implementation. Algorithm sources and experiment configurations retain their original bytes.

## Source mapping

| Packaged component | Relative archive identifier |
|---|---|
| `experiments/original_e1/*` | `SAC-E1-simple/*_E1_baselines_5seeds/` |
| `experiments/pinet/*` | `E1-next-simple/*_E1_baselines_5seeds_next/` |
| 175 runs for the seven baseline methods | `SAC-E1-simple/formal_all_ppu12_20260929/` |
| 20 PiNet runs: Synthetic T6/T12, A/A T6, Grid T20 | `E1-next-simple/formal_shard*_8gpu_w3_nohup_20261001/jobs/` |
| Five completed A/A T12 PiNet runs | `E1-rest-simple-T12-AA-td3-pinet/` |
| Recorded aggregate and seed provenance | `E1_pinet_replacement_20261001/` |

These identifiers refer to the archived experiment collection, not directories required outside this repository. The A/A T12 completion archive supplies the completed results, while the PiNet scenario directory supplies their implementation. Their saved source hashes match. Incomplete snapshots are not counted alongside completed runs.

## Implementation variants

The PiNet source family also contains changes to the Beta gate parameterization and MOPO uncertainty calculation. The seven non-PiNet methods use the baseline sources associated with their recorded results. In particular, the recorded MOPO implementation uses elite models for both sampling and uncertainty. Substituting the later MOPO implementation would change the experiment.

The root launcher routes only `td3_pinet` to `experiments/pinet/`. Other retained development variants are not selected. Supporting files remain where they are imported by the frozen runtime or enumerated by its protocol verifier. No losses, DGPs, budgets, training schedules, checkpoint-selection rules, or evaluation rules were changed for distribution.

## Recorded environments

The following values come from the selected runs' saved runtime records:

| Runs | Python | PyTorch | NumPy | pandas | SciPy |
|---|---|---|---|---|---|
| Baseline implementations, 175 | 3.12.3 | 2.11.0 | 2.3.5 | 3.0.5 | 1.17.1 |
| PiNet, 12 | 3.12.3 | 2.11.0 | 2.2.6 | 2.3.3 | 1.11.3 |
| PiNet, 13 | 3.12.3 | 2.11.0 | 1.26.4 | 2.2.3 | 1.11.3 |

The records identify PPU-ZW810 and CUDA runtime 13.0. Per-run associations are retained in `provenance/recorded_runs.json`. Shared method configurations do not imply an identical software environment. The launch dependency lists do not recreate every historical environment.

## Manifest conventions

- `provenance/source_files.json` records each package-relative path, relative archive source, packaged SHA-256, byte count, and `public` or `private` availability. For edited documentation or path-only metadata, `source_sha256` and `source_bytes` retain the original identity, and `transformation` describes the edit.
- Scenario `source_manifest.json` files retain original source and configuration hashes. Machine-specific `path` fields are omitted. Runtime checks resolve inputs by manifest keys within the scenario directory.
- PiNet `parent_e1_manifest.json` files retain parent file hashes. The original absolute `source_directory` is omitted. These hashes describe the historical parent snapshot, including files not distributed in this package.
- `provenance/recorded_runs.json` retains scientific launch settings, baseline settings, runtime environments, and historical code/configuration hashes. Only `out_dir` and `e1_options` were omitted from each saved launch configuration because they identify machine-specific output and settings locations. The launcher generates valid local paths when running a campaign.
- `results/recorded/seed_results.csv` records policy-level mean returns and evaluation-file hashes. Source paths are relative archive identifiers; raw evaluation files and trained policies are not bundled.

The 34 confidential A/A input records are retained with `availability: private`, but the files are excluded from Git. The default verifier checks public files only. Authorized users can supply the private inputs and run `verify_package.py --include-private`.

## Evidence and validation scope

During extraction, all 4,240 saved code/configuration hashes for the 200 selected runs matched their archived source files. All 200 final evaluation CSV hashes were verified. Each final-round evaluation contained 2,000 unique episodes, for 400,000 episodes in total; recomputed policy means matched the seed-level records to 1e-9. The campaign summarizer reproduced all 40 saved mean/SD pairs from those raw files.

These are checks of existing evidence, not new training or policy replay. The full five-scenario configuration dry-run and runtime import checks used the existing Python 3.10.20 validation environment. The historical checks are recorded in `provenance/packaging_validation.json`; they do not imply that confidential inputs or raw evaluation files are distributed.

For the public distribution, `verify_package.py` checks the available input/source hashes, syntax, fixed configurations, result coverage, and the means/SDs that can be recomputed from the bundled seed records. The public configuration dry-run covers 120 jobs across Synthetic T6/T12 and Grid T20 without private inputs. Run it using the commands in [the package README](../README.md).
