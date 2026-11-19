# Data Availability

The initial-submission artifact contains two reproducible benchmark families.

1. The fully synthetic benchmark generates all states, actions, responses, and
   evaluation trajectories from the included code.
2. The public-grid-calibrated benchmark generates a city-time panel from the
   included grid-simulator code and then constructs the controlled subsidy DGP.

The real-data-calibrated semi-synthetic benchmark is excluded. Its underlying
marketplace records are non-public and cannot be redistributed. No private raw
data, derived row-level records, credentials, user identifiers, or local
absolute paths are intended to be present in this artifact. Running an
experiment regenerates its episode-level outputs, summaries, figures, and
model checkpoints locally.

## Rebuttal supplement

`SAC-SA-RL-Baselines/` includes code for the five E1 comparison scenarios.
The A/A T6/T12 `data_sim/` and `aa_data_based_simulator_outputs/` directories
are private inputs: all 34 data, residual and calibration files in those
directories are excluded from Git. Their experiment code and configuration
remain unchanged. Running A/A scenarios or the full-package integrity check
requires the original authorized inputs at the paths recorded in each
scenario's source manifest. The public artifact does not regenerate or
replace them with a different DGP.

The two Public-grid `.cache/public_city_panel/*.pt` files in the supplement
are required frozen target-environment calibrations and are included. They
are not disposable training outputs. Aggregate and seed-level E1 results,
source hashes and recorded configurations are also included. Provenance
records retain original filesystem paths as metadata; execution uses the
packaged scenario paths. Virtual environments, credentials and transient
training outputs are not included.
