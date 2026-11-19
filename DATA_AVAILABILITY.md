# Data availability

## Public benchmarks

The fully synthetic benchmarks generate their inputs and trajectories from the included code. The public-grid-calibrated benchmark generates a city-time panel from the included simulator and constructs the controlled subsidy DGP.

The [RL benchmark package](rl-baselines/README.md) includes all required inputs for Synthetic T6, Synthetic T12, and Public-grid T20. Its two `experiments/**/.cache/public_city_panel/*.pt` files are frozen target-environment calibrations and must be retained. They are not trained policy checkpoints.

## Confidential A/A benchmarks

The A/A-calibrated benchmarks use non-public marketplace records that cannot be redistributed. The core experiment directories exclude these benchmarks. The RL benchmark package provides their code and configurations, but excludes all 34 data, residual, and calibration files under the A/A `data_sim/` and `aa_data_based_simulator_outputs/` directories. The ignore rules keep those files out of Git.

Expected package-relative paths and original hashes are retained in `rl-baselines/provenance/source_files.json` with `availability: private`. Authorized users must supply these inputs locally before running A/A scenarios or the optional `verify_package.py --include-private` check. The default verifier and launcher work with public inputs. No replacement data or alternative DGP is silently substituted.

## Recorded results and generated outputs

The RL benchmark package includes 200 seed-level policy means, aggregate summaries, configuration records, and source hashes for five scenarios and eight methods. A/A summaries contain aggregate experiment results, not source marketplace records or derived row-level inputs. Raw evaluation trajectories and trained policies are not included.

Running an experiment generates its own episode outputs, summaries, figures, and checkpoints locally. Virtual environments, credentials, and transient outputs are excluded from the distribution. Published metadata omits machine-specific paths while retaining scientific settings and source/result hashes.
