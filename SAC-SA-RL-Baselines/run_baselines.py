"""Run the eight-method E1 comparison using the sources cited by the rebuttal."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCENARIOS = {
    "synthetic_t6": ("SAC-SA-T6", "Synthetic T6"),
    "synthetic_t12": ("SAC-SA-T12", "Synthetic T12"),
    "aa_t6": ("SACSA-T6-AA", "A/A T6"),
    "aa_t12": ("SACSA-T12-AA", "A/A T12"),
    "grid_t20": ("Grid-T20", "Grid T20"),
}
METHODS = {
    "sacsa_reserve": "SAC-SA",
    "td3_direct_softmax": "TD3-Feasible",
    "td3_projection": "TD3-Projection",
    "dfl_pto": "DFL-PTO",
    "cql_ft": "CQL-FT",
    "mopo_ft": "MOPO-FT",
    "sac_lag": "SAC-Lag+Filter",
    "td3_pinet": "TD3-PiNet",
}
SEEDS = [2026, 2027, 2028, 2029, 2030]


def package_for(scenario: str, method: str) -> Path:
    prefix = SCENARIOS[scenario][0]
    variant = "pinet" if method == "td3_pinet" else "original_e1"
    suffix = "_next" if variant == "pinet" else ""
    return ROOT / "experiments" / variant / f"{prefix}_E1_baselines_5seeds{suffix}"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Verify inputs and generate configs; do not train.")
    args = parser.parse_args()
    seeds = args.seeds or ([2026] if args.profile == "smoke" else SEEDS)
    for name, values in [("scenarios", args.scenarios), ("methods", args.methods), ("seeds", seeds)]:
        if len(values) != len(set(values)):
            parser.error(f"Duplicate {name} are not allowed.")
    output = (args.output_root or ROOT / "runs" / f"{args.profile}_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite an existing output directory: {output}")

    groups = []
    for scenario in args.scenarios:
        for variant in ["original_e1", "pinet"]:
            methods = [m for m in args.methods if (m == "td3_pinet") == (variant == "pinet")]
            if not methods:
                continue
            package = package_for(scenario, methods[0])
            relative_output = f"{scenario}/{variant}"
            command = [sys.executable, str(package / "run_e1.py"), "--profile", args.profile,
                       "--methods", *methods, "--seeds", *map(str, seeds), "--device", args.device,
                       "--output-root", str(output / relative_output)]
            if args.dry_run:
                command.append("--dry-run")
            groups.append({"scenario": scenario, "variant": variant, "methods": methods,
                           "package": package.relative_to(ROOT).as_posix(), "output": relative_output,
                           "command": command, "status": "planned"})

    output.mkdir(parents=True)
    manifest = {"profile": args.profile, "dry_run": args.dry_run, "seeds": seeds,
                "scenarios": args.scenarios, "methods": args.methods,
                "source_policy": "Original E1 for seven methods; E1-next for TD3-PiNet only.",
                "training_jobs": len(args.scenarios) * len(args.methods) * len(seeds),
                "groups": groups}
    manifest_path = output / "campaign_manifest.json"
    write_json(manifest_path, manifest)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    print(f"{'PLAN' if args.dry_run else 'RUN'}: {manifest['training_jobs']} jobs; {output}", flush=True)
    for group in groups:
        print(f"{group['scenario']} / {group['variant']}: {', '.join(group['methods'])}", flush=True)
        group["status"] = "checking" if args.dry_run else "running"
        write_json(manifest_path, manifest)
        result = subprocess.run(group["command"], cwd=ROOT / group["package"], env=env)
        group["returncode"] = result.returncode
        group["status"] = ("planned" if args.dry_run else "completed") if result.returncode == 0 else "failed"
        write_json(manifest_path, manifest)
        if result.returncode:
            return result.returncode
    print("Configuration checks passed; no training executed." if args.dry_run else "Campaign completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
