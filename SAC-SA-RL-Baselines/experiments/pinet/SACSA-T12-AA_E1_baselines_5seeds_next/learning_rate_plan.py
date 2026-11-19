"""Opt-in learning rates selected on local development evidence."""
from __future__ import annotations
import json
from pathlib import Path

TUNED_METHODS = {"sacsa_reserve", "mopo_ft", "sac_lag",
                 "td3_direct_softmax", "td3_projection", "td3_pinet"}
ALLOWED_FACTORS = {0.5, 1.0, 2.0}


def read_learning_rate_plan(path, package_name):
    if path is None:
        return None
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    scenario = package_name.removesuffix("_E1_baselines_5seeds_next")
    if plan.get("scenario") != scenario:
        raise ValueError("Learning-rate plan belongs to another scenario.")
    if plan.get("purpose") != "development_candidates_require_formal_validation":
        raise ValueError("Learning-rate plan must declare its development evidence boundary.")
    factors = plan.get("factors")
    if not isinstance(factors, dict) or set(factors) != TUNED_METHODS:
        raise ValueError("Learning-rate plan must cover all six compared RL methods.")
    for method, factor in factors.items():
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor not in ALLOWED_FACTORS:
            raise ValueError(f"Invalid learning-rate factor for {method}.")
    return plan


def apply_learning_rate_factor(config, factor):
    if factor not in ALLOWED_FACTORS:
        raise ValueError("Learning-rate factor must be 0.5, 1 or 2.")
    if "policy" in config:
        rates = config["policy"]
        for key in ["actor_lr", "critic_lr", "alpha_lr"]:
            rates[key] = rates.get(key, 3e-4) * factor
    else:
        for key in ["lr_actor", "lr_critic", "lr_alpha"]:
            config[key] *= factor
    return config

