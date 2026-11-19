# Final-round baseline results (q=3)

| Method | Synthetic T6 | Synthetic T12 | A/A T6 | A/A T12 | Grid T20 |
|---|---:|---:|---:|---:|---:|
| SAC-SA | 70.27 ± 0.24 | 160.76 ± 0.96 | 156.62 ± 4.12 | 164.52 ± 0.72 | 893.10 ± 0.90 |
| TD3-Feasible | 67.20 ± 0.29 | 155.34 ± 7.10 | 157.00 ± 0.81 | 161.64 ± 2.33 | 897.85 ± 3.34 |
| TD3-Projection | 66.71 ± 0.46 | 144.33 ± 9.87 | 153.24 ± 9.54 | 132.95 ± 14.86 | 883.62 ± 11.08 |
| DFL-PTO | 69.30 ± 0.44 | 156.24 ± 0.32 | 151.36 ± 0.34 | 154.95 ± 0.28 | 886.88 ± 0.98 |
| Best op. | 67.57 | 157.74 | 155.05 | 157.11 | 891.06 |
| CQL-FT | 67.33 ± 0.27 | 144.13 ± 0.44 | 136.94 ± 0.75 | 149.00 ± 4.78 | 874.26 ± 0.64 |
| MOPO-FT | 69.07 ± 1.35 | 159.74 ± 0.68 | 157.16 ± 1.11 | 163.47 ± 1.51 | 890.92 ± 10.26 |
| SAC-Lag+Filter | 68.05 ± 0.46 | 160.47 ± 0.27 | 158.11 ± 0.46 | 163.84 ± 0.43 | 884.11 ± 1.56 |
| TD3-Πnet | 66.62 ± 0.36 | 144.23 ± 9.69 | 143.70 ± 10.38 | 136.65 ± 16.30 | 885.67 ± 12.04 |

Each learning-method entry reports the mean and sample SD across five policy-level mean returns (training seeds 2026-2030). Each trained policy was evaluated over 2,000 episodes using its prescribed deterministic action rule. Results use the final round without selecting the best round or seed. Best op. is the best mean among operational reference policies and has no training-seed SD.

The table covers five scenarios, eight learning methods, and 200 completed runs. SAC-SA, CQL, and MOPO retain the baseline gate parameterization; MOPO uses elite models for both sampling and uncertainty. TD3-PiNet uses the separate PyTorch implementation specialized for the shared-budget constraint. See [implementation provenance](../../docs/PROVENANCE.md) for the source mapping and recorded environment differences.

PiNet shows substantial variation across seeds in A/A T12. Seeds 2027 and 2030 have near-zero mean spending and returns close to the no-subsidy policy. All five seeds are included. Budget feasibility alone does not establish training stability or optimal returns.

See [seed-level results](seed_results.csv) and [unrounded method summaries](combined_method_summary.csv) for the underlying policy means. Reproducing the A/A scenarios requires confidential inputs; their aggregate results are included here for reference.
