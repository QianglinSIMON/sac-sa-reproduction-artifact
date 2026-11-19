"""Combine final-round E1/PiNet results from run_baselines.py."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from run_baselines import METHODS, SCENARIOS, SEEDS


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    manifest = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
    if manifest["dry_run"]:
        raise ValueError("A dry-run contains no experimental results.")
    seeds, refs = [], {}
    for group in manifest["groups"]:
        if group["status"] != "completed" or group.get("returncode") != 0:
            raise ValueError(f"Incomplete group: {group['scenario']} / {group['variant']}")
        group_root = root / group["output"]
        run = json.loads((group_root / "run_manifest.json").read_text(encoding="utf-8"))
        expected = {(m, s) for m in group["methods"] for s in manifest["seeds"]}
        jobs = run["jobs"]
        if len(jobs) != len(expected) or {(j["method"], j["training_seed"]) for j in jobs} != expected:
            raise ValueError(f"Unexpected or duplicate jobs: {group_root}")
        if run["profile"] != manifest["profile"] or run["dry_run"]:
            raise ValueError(f"Profile mismatch: {group_root}")
        for job in jobs:
            if job["status"] != "completed" or job.get("returncode") != 0:
                raise ValueError(f"Incomplete job: {group_root} / {job['method']}")
            method, seed = job["method"], job["training_seed"]
            tag = f"{method}_seed{seed}"
            cfg = json.loads((group_root / f"config_{tag}.json").read_text(encoding="utf-8"))
            cfg = cfg.get("experiment", cfg)
            raw_path = group_root / tag / "final_eval_raw.csv"
            with raw_path.open(encoding="utf-8-sig", newline="") as stream:
                raw = list(csv.DictReader(stream))
            key = "method_key" if "method_key" in raw[0] else "method"
            episode = "eval_episode" if "eval_episode" in raw[0] else "episode"
            final = [r for r in raw if r[key] == method and int(r["round"]) == cfg["q_max"]]
            n = cfg["eval_episodes"]
            if len(final) != n or {int(r[episode]) for r in final} != set(range(n)):
                raise ValueError(f"Missing or duplicate final episodes: {raw_path}")
            if not all(math.isfinite(float(r[k])) for r in final for k in ["net_reward", "spend", "final_budget"]):
                raise ValueError(f"Nonfinite evaluation: {raw_path}")
            seeds.append({"scenario": SCENARIOS[group["scenario"]][1], "method": method,
                          "seed": seed, "profile": manifest["profile"], "round": cfg["q_max"],
                          "eval_episodes": n, "net_reward": statistics.mean(float(r["net_reward"]) for r in final),
                          "spend": statistics.mean(float(r["spend"]) for r in final),
                          "budget_violation_episodes_1e5": sum(float(r["spend"]) > cfg["daily_budget"] + 1e-5
                                                               or float(r["final_budget"]) < -1e-5 for r in final),
                          "source_variant": group["variant"], "source": raw_path.relative_to(root).as_posix()})
            for op in ["no_subsidy", "random_pacing", "uniform_pacing", "gap_heuristic", "gap_pacing"]:
                op_rows = [r for r in raw if r[key] == op]
                if not op_rows:
                    continue
                last = max(int(r["round"]) for r in op_rows)
                values = [float(r["net_reward"]) for r in op_rows if int(r["round"]) == last]
                if len(values) != n or not all(map(math.isfinite, values)):
                    raise ValueError(f"Invalid operational reference: {raw_path} / {op}")
                ref_key = (group["scenario"], op)
                value = statistics.mean(values)
                if ref_key in refs and not math.isclose(refs[ref_key], value, rel_tol=1e-6, abs_tol=1e-5):
                    raise ValueError(f"Operational reference differs across runs: {ref_key}")
                refs[ref_key] = value

    summaries = []
    for scenario in manifest["scenarios"]:
        for method in manifest["methods"]:
            runs = [r for r in seeds if r["scenario"] == SCENARIOS[scenario][1] and r["method"] == method]
            values = [r["net_reward"] for r in runs]
            complete = sorted(r["seed"] for r in runs) == SEEDS
            summaries.append({"scenario": SCENARIOS[scenario][1], "method": METHODS[method],
                              "method_key": method, "n": len(values), "mean": statistics.mean(values),
                              "sample_sd": statistics.stdev(values) if len(values) > 1 else "",
                              "complete_five_seed_formal": complete and manifest["profile"] == "formal"})
    operations = [{"scenario": SCENARIOS[s][1], "method": m, "net_reward": v} for (s, m), v in refs.items()]
    write_csv(root / "seed_results.csv", seeds)
    write_csv(root / "method_summary.csv", summaries)
    if operations:
        write_csv(root / "operational_references.csv", operations)
    print(f"Summarized {len(seeds)} runs at the configured final round ({manifest['profile']}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    summarize(parser.parse_args().run_root.resolve())
