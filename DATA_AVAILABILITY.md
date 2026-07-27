# Data Availability

The artifact contains two reproducible benchmark families.

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
