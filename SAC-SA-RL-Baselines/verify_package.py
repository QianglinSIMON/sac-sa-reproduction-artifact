"""Check copied-file integrity, Python syntax, configurations and recorded summaries."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

from run_baselines import METHODS, SCENARIOS, SEEDS, package_for

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = json.loads((ROOT / "provenance/source_files.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["path"]
        require(path.is_file(), f"Missing packaged file: {path}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], f"Modified copied file: {path}")
    py_files = [*ROOT.glob("*.py"), *ROOT.glob("experiments/**/*.py")]
    for path in py_files:
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    for scenario in SCENARIOS:
        for method in METHODS:
            path = package_for(scenario, method)
            cfg = json.loads((path / "configs" / f"{method}.json").read_text(encoding="utf-8"))
            require(cfg["method"] == method, f"Wrong method in {path}")
            values = cfg.get("experiment", cfg)
            require(values["q_max"] == 3 and values["eval_episodes"] == 2000, f"Wrong endpoint: {path}")
            require(values["daily_budget"] == 10, f"Wrong budget: {path}")

    with (ROOT / "results/recorded/seed_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(label, method, seed) for _, label in SCENARIOS.values() for method in METHODS for seed in SEEDS}
    observed = {(r["scenario"], r["method"], int(r["seed"])) for r in rows}
    require(len(rows) == 200 and observed == expected, "Recorded result coverage is not 5 x 8 x 5.")
    with (ROOT / "results/recorded/combined_method_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summary = list(csv.DictReader(stream))
    require(len(summary) == 40, "Expected 40 recorded method/scenario groups.")
    for item in summary:
        values = [float(r["net_reward"]) for r in rows if (r["scenario"], r["method"]) == (item["scenario"], item["method"])]
        require(len(values) == 5, f"Missing recorded seeds: {item}")
        require(abs(statistics.mean(values) - float(item["mean"])) < 1e-9, f"Mean differs: {item}")
        require(abs(statistics.stdev(values) - float(item["sd"])) < 1e-9, f"Sample SD differs: {item}")
    print(f"PASS: {len(manifest['files'])} copied hashes, {len(py_files)} Python files, 40 routed configs, 200 recorded seeds.")
    print("This checks package integrity and saved summaries; it does not train or replay policies.")


if __name__ == "__main__":
    main()
