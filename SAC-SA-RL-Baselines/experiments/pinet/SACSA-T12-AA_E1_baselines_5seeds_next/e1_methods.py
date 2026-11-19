"""Method names and variant metadata shared by this standalone experiment."""

METHOD_LABELS = {
    "sacsa_reserve": "SAC-SA",
    "td3_direct_softmax": "TD3-Feasible",
    "td3_projection": "TD3-Projection",
    "dfl_pto": "DFL-PTO",
    "cql_ft": "CQL-FT",
    "mopo_ft": "MOPO-FT",
    "sac_lag": "SAC-Lag+Filter",
    "td3_pinet": "TD3-Πnet",
    "mopo_short_ft": "MOPO-FT (h=1)",
    "sac_fixed_penalty": "SAC-FixedPenalty+Filter",
}
METHODS = list(METHOD_LABELS)
NEW_METHODS = {
    method: METHOD_LABELS[method]
    for method in ("cql_ft", "mopo_ft", "sac_lag", "td3_pinet", "mopo_short_ft", "sac_fixed_penalty")
}
METHOD_VARIANTS = {
    "mopo_short_ft": {"parent": "mopo_ft", "rollout_length": 1},
    "sac_fixed_penalty": {"parent": "sac_lag", "multiplier": 1.0, "dual_adaptation": False},
}
VARIANT_PARENTS = {method: settings["parent"] for method, settings in METHOD_VARIANTS.items()}
