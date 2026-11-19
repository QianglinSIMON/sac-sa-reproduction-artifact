"""Check frozen configurations and packaged inputs without the original checkout."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ["sacsa_reserve", "td3_direct_softmax", "td3_projection", "dfl_pto",
           "cql_ft", "mopo_ft", "sac_lag", "td3_hardnet_cvx"]


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
    changes = {}
    for method in METHODS:
        base = ("td3_projection" if method == "td3_hardnet_cvx" else method)
        if base not in manifest["source_configs"]:
            base = "sacsa_reserve"
        source = ROOT / "configs" / "source" / f"{base}.json"
        require(hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_configs"][base]["sha256"],
                f"Original configuration hash differs: {source}")
        old = json.loads(source.read_text(encoding="utf-8"))
        new = json.loads((ROOT / "configs" / f"{method}.json").read_text(encoding="utf-8"))
        if "args" in old:
            old = old["args"]
            allowed = {"seed", "training_seed", "dgp_seed", "data_seed", "evaluation_seed",
                       "protocol_id", "method", "out_dir"}
            diff = {key: {"source": old.get(key), "formal": new.get(key)}
                    for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
            require(not (diff.keys() - allowed), f"Undeclared configuration changes for {method}: {diff}")
            require(all(new[key] == 2026 for key in ["seed", "training_seed", "dgp_seed", "data_seed", "evaluation_seed"]),
                    f"Incorrect base seed roles: {method}")
            require(new["q_max"] == 3 and new["eval_episodes"] == 2000,
                    f"Incorrect final evaluation protocol: {method}")
        else:
            require(old["simulator"] == new["simulator"] and old["policy"] == new["policy"],
                    f"Grid simulator or policy settings changed: {method}")
            expected_experiment = dict(old["experiment"], training_seed=2026, data_seed=2026, evaluation_seed=2026)
            expected_selection = dict(old["checkpoint_selection"], seed_role="evaluation_seed")
            require(new["experiment"] == expected_experiment, f"Grid experiment settings changed: {method}")
            require(new["checkpoint_selection"] == expected_selection, f"Grid selection settings changed: {method}")
            require(new["experiment"]["q_max"] == 3 and new["experiment"]["eval_episodes"] == 2000,
                    f"Incorrect Grid final evaluation protocol: {method}")
            require(new["checkpoint_selection"]["target_dgp_episodes"] == 0 and new["simulator"]["strict_quality_gate"] is True,
                    f"Incorrect Grid selection or quality gate: {method}")
            diff = {"method": {"source": old["method"], "formal": method},
                    "seed_roles": "Training seed separated from fixed data/evaluation and frozen DGP."}
        require(new["method"] == method, f"Method label differs from its configuration: {method}")
        changes[method] = {"source_config": f"configs/source/{base}.json", "allowed_differences": diff}
    report = {"status": "PASS", "scenario": manifest["scenario"],
              "formal_training_seeds": [2026, 2027, 2028, 2029, 2030],
              "formal_jobs": 40, "configuration_comparisons": changes,
              "verified_packaged_inputs": checked_inputs,
              "note": "Only formal base configs checked here; smoke overrides are separately labeled."}
    return report


if __name__ == "__main__":
    report = verify()
    (ROOT / "protocol_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {report['scenario']} preserves original settings in all eight formal configurations.")
