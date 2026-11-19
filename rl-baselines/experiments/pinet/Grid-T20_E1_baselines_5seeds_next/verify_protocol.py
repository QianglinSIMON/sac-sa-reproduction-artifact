"""Check frozen configurations and packaged inputs without the original checkout."""
import hashlib
import json
from pathlib import Path

from e1_methods import METHODS, VARIANT_PARENTS

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    # These are launch checks and must also run under python -O.
    if not condition:
        raise ValueError(message)


def verify_packaged_inputs(manifest):
    modified_sources = {"iterative_synthetic_sacsa.py", "sacsa_rl_components.py",
                        "city_panel_models.py", "city_panel_training.py"}
    checked = []
    for name, source in manifest["source_files"].items():
        if name in modified_sources:
            continue
        # The source manifest was generated on Windows; the package also runs on Linux.
        path = ROOT / name.replace("\\", "/")
        require(path.is_file(), f"Missing frozen input: {path}. Copy the complete experiment package.")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"],
                f"Frozen input differs from the original experiment: {path}")
        checked.append(name.replace("\\", "/"))
    return checked


def verify():
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    checked_inputs = verify_packaged_inputs(manifest)
    parent = json.loads((ROOT / "parent_e1_manifest.json").read_text(encoding="utf-8"))
    for base, info in manifest["source_configs"].items():
        path = ROOT / "configs" / "source" / f"{base}.json"
        require(hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"],
                f"Historical source config changed: {path}")
    changes = {}
    for method in METHODS:
        base = {"td3_pinet": "td3_hardnet_cvx", **VARIANT_PARENTS}.get(method, method)
        source = ROOT / "configs" / "parent_e1" / f"{base}.json"
        key = f"configs/{base}.json"
        hashes = {k.replace("\\", "/"): v for k, v in parent["files"].items()}
        require(hashlib.sha256(source.read_bytes()).hexdigest() == hashes[key],
                f"Parent E1 configuration hash differs: {source}")
        old = json.loads(source.read_text(encoding="utf-8"))
        new = json.loads((ROOT / "configs" / f"{method}.json").read_text(encoding="utf-8"))
        import copy
        expected = copy.deepcopy(old)
        expected["method"] = method
        grid = "experiment" in new
        if not grid and method in {"sacsa_reserve", "cql_ft", "mopo_ft", "mopo_short_ft"}:
            expected["dirichlet_logit_clip"] = 30.0
        if method in {"sacsa_reserve", "cql_ft", "mopo_ft", "mopo_short_ft"}:
            (expected["policy"] if grid else expected)["bd_gate_parameterization"] = "squareplus"
        require(new == expected, f"Undeclared E1-next changes for {method}")
        diff = {key: {"parent_e1": old.get(key), "next": new.get(key)}
                for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
        if not grid:
            require(all(new[key] == 2026 for key in ["seed", "training_seed", "dgp_seed", "data_seed", "evaluation_seed"]),
                    f"Incorrect base seed roles: {method}")
            require(new["q_max"] == 3 and new["eval_episodes"] == 2000,
                    f"Incorrect final evaluation protocol: {method}")
        else:
            require(new["experiment"]["q_max"] == 3 and new["experiment"]["eval_episodes"] == 2000,
                    f"Incorrect Grid final evaluation protocol: {method}")
            require(new["checkpoint_selection"]["target_dgp_episodes"] == 0 and new["simulator"]["strict_quality_gate"] is True,
                    f"Incorrect Grid selection or quality gate: {method}")
        changes[method] = {"parent_config": f"configs/parent_e1/{base}.json", "allowed_differences": diff}
    settings_path = ROOT / "baseline_settings_parent.json"
    require(hashlib.sha256(settings_path.read_bytes()).hexdigest() == parent["files"]["baseline_settings.json"],
            "Parent baseline settings hash differs")
    options = json.loads(settings_path.read_text(encoding="utf-8"))
    options.pop("hardnet")
    from pinet_budget import DEFAULT_SETTINGS
    options["pinet"] = DEFAULT_SETTINGS
    require(options == json.loads((ROOT / "baseline_settings.json").read_text(encoding="utf-8")),
            "Undeclared new-method hyperparameter changes")
    report = {"status": "PASS", "scenario": manifest["scenario"],
              "formal_training_seeds": [2026, 2027, 2028, 2029, 2030],
              "formal_jobs": 50, "configuration_comparisons": changes,
              "verified_packaged_inputs": checked_inputs,
              "training_budget": "Full parent E1 schedules preserved; no 2/3 reduction.",
              "declared_code_changes": ["PiNet replaces HardNet-Cvx", "MOPO h=1 and fixed-penalty ablations added",
                                        "BD gate uses squareplus; city-share entropy and training losses unchanged", "BD concentration clip corrected for SAC-SA/CQL/MOPO",
                                        "MOPO uncertainty uses all ensemble members; sampling uses elites"],
              "note": "Only formal base configs checked here; smoke overrides are separately labeled."}
    return report


if __name__ == "__main__":
    report = verify()
    (ROOT / "protocol_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {report['scenario']} preserves full parent training budgets and checks declared changes.")
