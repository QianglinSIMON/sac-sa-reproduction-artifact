"""Confirmatory statistics for the fixed-DGP Synthetic T=6 protocol.

The script keeps two uncertainty sources separate:

1. Conditional Monte Carlo uncertainty is computed from episode-paired return
   differences for one trained policy.
2. Training uncertainty is the sample standard deviation of those seed-level
   means across independent training seeds.

It intentionally does not pool all evaluation episodes across training seeds
into one standard error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


METHOD_LABELS = {
    "sacsa_reserve": "Beta-Dirichlet SAC-SA",
    "dfl_pto": "DFL-PTO",
    "td3_direct_softmax": "TD3--Feasible",
    "td3_projection": "TD3--Projection",
    "no_subsidy": "No Subsidy",
    "random_pacing": "Random Pacing",
    "uniform_pacing": "Uniform Pacing",
    "gap_heuristic": "Demand-Supply Gap",
}
OPERATIONAL_METHODS = {"no_subsidy", "random_pacing", "uniform_pacing", "gap_heuristic"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    return float(value)


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            values_match(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def mean_sd(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def paired_mc_summary(differences: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot summarize an empty paired-difference sample.")
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    se = sd / math.sqrt(values.size) if values.size > 1 else 0.0
    return {
        "paired_gap_mean": mean,
        "paired_gap_episode_sd": sd,
        "paired_gap_mc_se": se,
        "paired_gap_mc_ci95_lower": mean - 1.96 * se,
        "paired_gap_mc_ci95_upper": mean + 1.96 * se,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pattern", type=str, default="results_synT6_fdg2_*")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_methods = {str(value) for value in protocol["methods"]}
    expected_training_seeds = {int(value) for value in protocol["training_seeds"]}
    fixed_seeds = {key: int(value) for key, value in protocol["fixed_seeds"].items()}
    locked_args = dict(protocol["locked_args"])
    result_dirs = sorted(path for path in args.root.glob(args.pattern) if (path / "config.json").exists())
    if not result_dirs:
        raise SystemExit(f"No result directories matched {args.root / args.pattern}.")

    verification_rows: List[Dict[str, Any]] = []
    method_rows: Dict[Tuple[int, int, str, int, int], Dict[str, Any]] = {}
    operational_candidates: Dict[Tuple[int, int, str, int, int], List[float]] = defaultdict(list)
    observed_jobs: set[Tuple[int, str]] = set()

    for result_dir in result_dirs:
        config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
        run_args = dict(config.get("args", {}))
        method = str(run_args.get("method", ""))
        training_seed = int(run_args.get("training_seed", run_args.get("seed", -1)))
        observed_jobs.add((training_seed, method))
        issues: List[str] = []
        if str(run_args.get("protocol_id", "")) != str(protocol["protocol_id"]):
            issues.append("protocol_id")
        if method not in expected_methods:
            issues.append("method")
        if training_seed not in expected_training_seeds:
            issues.append("training_seed")
        for name, expected in fixed_seeds.items():
            if int(run_args.get(name, -1)) != expected:
                issues.append(name)
        for name, expected in locked_args.items():
            if not values_match(run_args.get(name), expected):
                issues.append(name)
        verification_rows.append(
            {
                "result_dir": result_dir.name,
                "training_seed": training_seed,
                "method_key": method,
                "protocol_ok": int(not issues),
                "mismatched_fields": ";".join(sorted(set(issues))),
            }
        )

        raw_path = result_dir / "round_eval_raw.csv"
        if not raw_path.exists():
            raise SystemExit(f"Missing {raw_path}.")
        for row in read_csv(raw_path):
            method_key = str(row.get("method_key", ""))
            q = int(float(row["round"]))
            eval_seed = int(float(row["eval_seed"]))
            eval_episode = int(float(row["eval_episode"]))
            net_reward = as_float(row["net_reward"])
            key = (training_seed, q, method_key, eval_seed, eval_episode)
            if method_key == method:
                method_rows[key] = {
                    "training_seed": training_seed,
                    "round": q,
                    "method_key": method_key,
                    "method": METHOD_LABELS.get(method_key, method_key),
                    "eval_seed": eval_seed,
                    "eval_episode": eval_episode,
                    "net_reward": net_reward,
                }
            elif method_key in OPERATIONAL_METHODS:
                operational_candidates[key].append(net_reward)

    if any(not int(row["protocol_ok"]) for row in verification_rows):
        write_csv(args.out_dir / "protocol_verification.csv", verification_rows)
        bad = [row for row in verification_rows if not int(row["protocol_ok"])]
        raise SystemExit(f"Protocol verification failed for {len(bad)} result directories.")
    expected_jobs = {(seed, method) for seed in expected_training_seeds for method in expected_methods}
    missing_jobs = sorted(expected_jobs - observed_jobs)
    if missing_jobs:
        raise SystemExit(f"Missing fixed-DGP jobs: {missing_jobs}")

    operational_rows: Dict[Tuple[int, int, str, int, int], float] = {}
    for key, values in operational_candidates.items():
        if max(values) - min(values) > 1e-6:
            raise SystemExit(f"Repeated operational baseline is not identical for paired key {key}.")
        operational_rows[key] = float(values[0])

    return_seed_rows: List[Dict[str, Any]] = []
    grouped_returns: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)
    for row in method_rows.values():
        grouped_returns[(row["training_seed"], row["round"], row["method_key"])].append(row["net_reward"])
    for (training_seed, q, method_key), values in sorted(grouped_returns.items()):
        mean, episode_sd = mean_sd(values)
        return_seed_rows.append(
            {
                "training_seed": training_seed,
                "round": q,
                "method_key": method_key,
                "method": METHOD_LABELS.get(method_key, method_key),
                "eval_episodes": len(values),
                "net_reward_mean": mean,
                "net_reward_episode_sd": episode_sd,
                "net_reward_mc_se": episode_sd / math.sqrt(len(values)),
            }
        )

    # Operational comparator is selected by its mean within each paired seed/round,
    # then the same baseline method is used for every episode in that comparison.
    best_operational: Dict[Tuple[int, int], str] = {}
    for training_seed in sorted(expected_training_seeds):
        for q in range(1, int(locked_args["q_max"]) + 1):
            means: Dict[str, float] = {}
            for method_key in OPERATIONAL_METHODS:
                vals = [
                    value
                    for (seed, round_idx, key_method, _, _), value in operational_rows.items()
                    if seed == training_seed and round_idx == q and key_method == method_key
                ]
                if vals:
                    means[method_key] = float(np.mean(vals))
            if not means:
                raise SystemExit(f"No operational baselines for training_seed={training_seed}, q={q}.")
            best_operational[(training_seed, q)] = max(means, key=means.get)

    comparisons: List[Tuple[str, str]] = [
        (method, "best_operational") for method in sorted(expected_methods)
    ]
    if {"sacsa_reserve", "dfl_pto"}.issubset(expected_methods):
        comparisons.append(("sacsa_reserve", "dfl_pto"))
    paired_seed_rows: List[Dict[str, Any]] = []
    for training_seed in sorted(expected_training_seeds):
        for q in range(1, int(locked_args["q_max"]) + 1):
            for method_key, comparator_key in comparisons:
                method_episode_rows = {
                    (eval_seed, eval_episode): row
                    for (seed, round_idx, key_method, eval_seed, eval_episode), row in method_rows.items()
                    if seed == training_seed and round_idx == q and key_method == method_key
                }
                if comparator_key == "best_operational":
                    actual_comparator = best_operational[(training_seed, q)]
                    comparator_episode_rows = {
                        (eval_seed, eval_episode): value
                        for (seed, round_idx, key_method, eval_seed, eval_episode), value in operational_rows.items()
                        if seed == training_seed and round_idx == q and key_method == actual_comparator
                    }
                else:
                    actual_comparator = comparator_key
                    comparator_episode_rows = {
                        (eval_seed, eval_episode): row["net_reward"]
                        for (seed, round_idx, key_method, eval_seed, eval_episode), row in method_rows.items()
                        if seed == training_seed and round_idx == q and key_method == actual_comparator
                    }
                common = sorted(set(method_episode_rows) & set(comparator_episode_rows))
                if len(common) != int(locked_args["eval_episodes"]):
                    raise SystemExit(
                        f"Expected {locked_args['eval_episodes']} paired episodes for seed={training_seed}, "
                        f"q={q}, {method_key} vs {actual_comparator}; found {len(common)}."
                    )
                differences = [
                    float(method_episode_rows[key]["net_reward"]) - float(comparator_episode_rows[key])
                    for key in common
                ]
                paired_seed_rows.append(
                    {
                        "training_seed": training_seed,
                        "round": q,
                        "method_key": method_key,
                        "method": METHOD_LABELS.get(method_key, method_key),
                        "comparator_key": comparator_key,
                        "actual_comparator_key": actual_comparator,
                        "actual_comparator": METHOD_LABELS.get(actual_comparator, actual_comparator),
                        "paired_eval_episodes": len(common),
                        **paired_mc_summary(differences),
                    }
                )

    paired_training_rows: List[Dict[str, Any]] = []
    grouped_gaps: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in paired_seed_rows:
        grouped_gaps[(int(row["round"]), str(row["method_key"]), str(row["comparator_key"]))].append(row)
    for (q, method_key, comparator_key), rows in sorted(grouped_gaps.items()):
        gaps = [float(row["paired_gap_mean"]) for row in rows]
        mean, training_sd = mean_sd(gaps)
        paired_training_rows.append(
            {
                "round": q,
                "method_key": method_key,
                "method": METHOD_LABELS.get(method_key, method_key),
                "comparator_key": comparator_key,
                "training_seeds": len(rows),
                "mean_paired_gap_across_training_seeds": mean,
                "training_seed_sd_of_paired_gap": training_sd,
                "min_seed_paired_gap": min(gaps),
                "max_seed_paired_gap": max(gaps),
            }
        )

    return_training_rows: List[Dict[str, Any]] = []
    grouped_seed_returns: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for row in return_seed_rows:
        grouped_seed_returns[(int(row["round"]), str(row["method_key"]))].append(float(row["net_reward_mean"]))
    for (q, method_key), values in sorted(grouped_seed_returns.items()):
        mean, training_sd = mean_sd(values)
        return_training_rows.append(
            {
                "round": q,
                "method_key": method_key,
                "method": METHOD_LABELS.get(method_key, method_key),
                "training_seeds": len(values),
                "net_reward_mean_across_training_seeds": mean,
                "training_seed_sd_net_reward": training_sd,
                "min_seed_net_reward": min(values),
                "max_seed_net_reward": max(values),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "protocol_verification.csv", verification_rows)
    write_csv(args.out_dir / "return_by_training_seed.csv", return_seed_rows)
    write_csv(args.out_dir / "return_training_seed_summary.csv", return_training_rows)
    write_csv(args.out_dir / "paired_gap_by_training_seed.csv", paired_seed_rows)
    write_csv(args.out_dir / "paired_gap_training_seed_summary.csv", paired_training_rows)
    print(f"Wrote fixed-DGP confirmatory summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
