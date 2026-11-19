"""Training-seed summaries, with explicit completeness and smoke labels."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from e1_methods import METHOD_LABELS as LABELS

OPERATIONAL = {"no_subsidy", "random_pacing", "uniform_pacing", "gap_pacing"}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    seeds, summaries, refs = [], [], {}
    for job in manifest["jobs"]:
        if job["status"] != "completed":
            continue
        out = root / f"{job['method']}_seed{job['training_seed']}"
        with (out / "final_eval_raw.csv").open(encoding="utf-8-sig") as f:
            raw = list(csv.DictReader(f))
        cfg = json.loads((root / f"config_{job['method']}_seed{job['training_seed']}.json").read_text(encoding="utf-8"))["experiment"]
        rows = [row for row in raw if row["method"] == job["method"] and int(row["round"]) == cfg["q_max"]]
        if not rows:
            raise ValueError(f"No final evaluation for {job['method']}: {out}")
        rewards = [float(row["net_reward"]) for row in rows]
        spending = [float(row["spend"]) for row in rows]
        if not all(math.isfinite(value) for value in rewards):
            raise ValueError(f"Nonfinite evaluation: {out}")
        if len(rows) != cfg["eval_episodes"]:
            raise ValueError(f"Incomplete evaluation: {out}")
        if not all(math.isfinite(value) for value in spending):
            raise ValueError(f"Nonfinite spending: {out}")
        if any(value < -1e-5 or value > cfg["daily_budget"] + 1e-3 for value in spending):
            raise ValueError(f"Budget violation: {out}")
        with (out / "simulator_policy_selection_diagnostics.csv").open(encoding="utf-8-sig") as f:
            selection = list(csv.DictReader(f))
        if any(int(r["target_dgp_episodes"]) != 0 for r in selection):
            raise ValueError("Target-DGP episodes used for Grid checkpoint selection.")
        with (out / "simulator_quality_gate.csv").open(encoding="utf-8-sig") as f:
            quality = list(csv.DictReader(f))
        audit = {"warning_count": sum(int(r["pass"]) == 0 for r in quality)}
        seeds.append({"method": job["method"], "training_seed": job["training_seed"],
                      "profile": manifest["profile"], "round": cfg["q_max"],
                      "eval_episodes": len(rows), "net_reward": statistics.mean(rewards),
                      "elapsed_seconds": job["elapsed_seconds"],
                      "audit_warning_count": audit.get("warning_count", 0), "source": str(out)})
        seed_row = seeds[-1]
        seed_row["spend_mean"] = statistics.mean(spending)
        seed_row["executed_episode_budget_violation_rate"] = statistics.mean(
            float(value > cfg["daily_budget"] + 1e-5) for value in spending)
        for metric in ["proposal_violation", "proposal_overspend", "filter_distance",
                       "solver_constraint_residual", "solver_numerical_repair",
                       "pinet_fixed_point_residual", "pinet_iterations"]:
            values = [float(row[metric]) for row in rows if row.get(metric, "") != ""]
            if values:
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Nonfinite constraint diagnostic: {metric}")
                seed_row[metric + "_mean"] = statistics.mean(values)
        for key in OPERATIONAL:
            baseline = [float(r["net_reward"]) for r in raw if r["method"] == key]
            if baseline:
                if not all(math.isfinite(value) for value in baseline):
                    raise ValueError(f"Nonfinite operational reference: {key}: {out}")
                value = statistics.mean(baseline)
                if key in refs and not math.isclose(refs[key], value, rel_tol=1e-6, abs_tol=1e-5):
                    raise ValueError(f"Operational reference differs across fixed evaluation streams: {key}")
                refs[key] = value
    for method, label in LABELS.items():
        runs = [r for r in seeds if r["method"] == method]
        values = [r["net_reward"] for r in runs]
        observed = sorted(r["training_seed"] for r in runs)
        complete = observed == [2026, 2027, 2028, 2029, 2030]
        mean = statistics.mean(values) if values else None
        sd = statistics.stdev(values) if len(values) >= 2 else None
        summaries.append({"method": label, "method_key": method, "profile": manifest["profile"],
                          "n_training_seeds": len(values), "seeds_complete": complete,
                          "ready_for_table2": bool(complete and manifest["profile"] == "formal"),
                          "mean": mean, "sample_sd": sd,
                          "display": f"{mean:.2f} +/- {sd:.2f}" if sd is not None else "pending five seeds"})
    if refs:
        best = max(refs, key=refs.get)
        summaries.append({"method": "Best op.", "method_key": best, "profile": manifest["profile"],
                          "n_training_seeds": 0, "mean": refs[best], "display": f"{refs[best]:.2f}"})
    write_csv(root / "seed_results.csv", seeds)
    write_csv(root / "table2_summary.csv", summaries)
    print(f"Wrote seed_results.csv and table2_summary.csv ({manifest['profile']}; {len(seeds)} runs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    summarize(parser.parse_args().run_root.resolve())
