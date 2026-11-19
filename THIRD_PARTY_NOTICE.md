# Third-party code and references

## Public-grid simulator

The public-grid experiment adapts the ride-hailing simulation associated with [MDPOD: Simulation of Ridesharing Market and the MDP Order Dispatch Policy](https://github.com/callmespring/MDPOD). The upstream [demonstration notebook](https://github.com/callmespring/MDPOD/blob/main/Demonstration.ipynb) implements a synthetic grid setting inspired by *Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms: A Learning and Planning Approach*.

The adapted simulator is included with permission from the rights holder, as recorded in the original artifact notice. The local implementation adds batched simulation, city-time calibration, and a controlled subsidy DGP. It should not be described as an unmodified upstream implementation.

## RL method references

The baseline implementations adapt the following method references to this project's budget constraints, data schedules, and evaluation protocol:

- [Conservative Q-Learning (CQL)](https://github.com/aviralkumar2907/CQL).
- [Model-based Offline Policy Optimization (MOPO)](https://github.com/tianheyu927/mopo).
- [PiNet](https://github.com/antonioterpin/pinet), with its [associated paper](https://arxiv.org/abs/2508.10480v2). The local implementation uses a PyTorch budget specialization with a Douglas-Rachford forward solve and a dense implicit adjoint.

The scenario `upstream_references.json` files retain the recorded upstream file hashes or commit identifiers and retrieval dates. The package uses local adaptations rather than the original upstream training scripts. Third-party dependencies retain their respective licenses.

## Repository license status

No repository-wide software license is specified in this distribution. The recorded permission to include the adapted simulator does not specify a general license for the entire repository.
