# Final-round baseline results (q=3)

Seven methods retain the original E1 implementation, while TD3-PiNet uses the subsequent implementation. The results do not represent an all-method comparison using the revised E1-next implementation.

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

Each learning-method entry reports the mean and sample SD across training seeds 2026–2030. Each trained policy was evaluated over 2,000 episodes using its prescribed deterministic action rule. Results use the final round, without selecting the best round, seed or implementation version. Best op. is an operational reference and has no training-seed SD.

The collection contains 200 completed runs across five scenarios, eight learning methods and five training seeds. TD3-PiNet occupies the projection-layer baseline slot; the other seven methods consistently use original E1 results.

SAC-SA, CQL and MOPO retain the original gate parameterization. MOPO computes uncertainty using the original elite subset. These rows do not use the revised squareplus gate or full-ensemble uncertainty calculation.

PiNet is the project's PyTorch adaptation to the shared-budget constraint. Four scenarios come from E1-next-simple, and the five A/A T12 results come from E1-rest-simple-T12-AA-td3-pinet. Those five records were verified as completed snapshots of the original tasks. The original HardNet runs remain in the source collection and were neither overwritten nor relabeled.

PiNet shows substantial variation across seeds in A/A T12. Seeds 2027 and 2030 have near-zero mean spending and returns close to the no-subsidy policy. All five seeds are included, and budget feasibility alone does not establish training stability or optimal returns.
