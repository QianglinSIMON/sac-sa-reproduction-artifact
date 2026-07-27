"""
Aggregate iterative synthetic experiment outputs across methods.

Example:
    python summarize_iterative_results.py \
        --root . \
        --pattern 'results_cpu_pilot_*' \
        --out_dir results_cpu_pilot_summary
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

_default_mplconfig = Path.cwd() / ".matplotlib"
_default_mplconfig.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_default_mplconfig))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "No Subsidy",
    "Random Pacing",
    "Uniform Pacing",
    "Demand-Supply Gap",
    "DFL-PTO",
    "Finite-Candidate MPC-3",
    "Original Logistic-Normal SAC",
    "Reserve-Gated Logistic-Normal SAC",
    "TD3--Feasible",
    "TD3--Projection",
    "Original Augmented-Dirichlet SAC-SA",
    "Proposed Beta-Dirichlet SAC-SA",
]

METHOD_KEY_LABELS = {
    "no_subsidy": "No Subsidy",
    "random_pacing": "Random Pacing",
    "uniform_pacing": "Uniform Pacing",
    "gap_heuristic": "Demand-Supply Gap",
    "dfl_pto": "DFL-PTO",
    "mpc3": "Finite-Candidate MPC-3",
    "sac_logistic_normal_reserve": "Reserve-Gated Logistic-Normal SAC",
    "sac_logistic_normal": "Original Logistic-Normal SAC",
    "td3_direct_softmax": "TD3--Feasible",
    "td3_projection": "TD3--Projection",
    "sacsa_reserve": "Proposed Beta-Dirichlet SAC-SA",
    "sacsa": "Original Augmented-Dirichlet SAC-SA",
}
METRICS = ["net_reward", "response_score", "spend", "final_budget"]
OPERATIONAL_REFERENCE_KEYS = {"no_subsidy", "random_pacing", "uniform_pacing", "gap_heuristic"}
CONFIG_DISPLAY_FIELDS = [
    "M",
    "T",
    "daily_budget",
    "lambda_pess",
    "use_pessimism",
    "initial_logged_days",
    "new_logged_days",
    "q_max",
    "center_residual",
    "real_online_update",
    "real_exploration_epsilon",
]
CONFIG_KEY_FIELDS = [
    "protocol_id",
    "dgp_seed",
    "data_seed",
    "evaluation_seed",
    "M",
    "T",
    "daily_budget",
    "logging_spend_scale",
    "sigma_x",
    "sigma_y",
    "sigma_c",
    "sigma2_z",
    "rho_corr",
    "initial_logged_days",
    "new_logged_days",
    "validation_logged_days",
    "holdout_logged_days",
    "q_max",
    "use_pessimism",
    "lambda_pess",
    "b_boot",
    "bootstrap_epochs",
    "bootstrap_finetune_epochs",
    "save_bootstrap_ensemble",
    "sim_update_mode",
    "sim_epochs",
    "sim_finetune_epochs",
    "sim_batch_size",
    "sim_hidden_dim",
    "sim_hidden_layers",
    "sim_head_hidden_dim",
    "sim_dropout",
    "sim_activation",
    "sim_lr",
    "sim_weight_decay",
    "logging_policy",
    "center_residual",
    "synthetic_train_episodes",
    "warmup_episodes",
    "update_after_episodes",
    "update_every_steps",
    "gradient_steps_per_update",
    "rollout_batch_size",
    "batch_rollout_update_cap",
    "hidden_dim",
    "batch_size",
    "replay_size",
    "real_online_update",
    "real_batch_size",
    "real_update_after_steps",
    "real_gradient_steps_per_step",
    "real_exploration_epsilon",
    "lr_actor",
    "lr_critic",
    "lr_alpha",
    "init_temperature",
    "fixed_temperature",
    "min_temperature",
    "max_temperature",
    "auto_alpha",
    "target_entropy_scale",
    "sac_entropy_space",
    "alpha_logp_clip",
    "dirichlet_init_total_spend_share",
    "dirichlet_init_concentration",
    "dirichlet_min_alpha",
    "dirichlet_max_alpha",
    "dirichlet_logit_clip",
    "ln_init_total_spend_share",
    "ln_init_log_std",
    "td3_init_total_spend_share",
    "td3_policy_noise",
    "td3_noise_clip",
    "td3_policy_delay",
    "candidate_random_count",
    "candidate_budget_fracs",
    "dfl_epochs",
    "dfl_batch_size",
    "dfl_lr",
    "dfl_decision_weight",
    "dfl_softmax_temperature",
    "mpc_horizon",
    "eval_episodes",
    "validation_episodes",
    "inner_early_stop",
    "inner_eval_every_episodes",
    "inner_patience",
    "inner_min_delta",
    "final_eval_seed_offset",
]
COMPARISON_CONFIG_FIELDS = [
    "protocol_id",
    "dgp_seed",
    "data_seed",
    "evaluation_seed",
    "M",
    "T",
    "daily_budget",
    "logging_spend_scale",
    "sigma_x",
    "sigma_y",
    "sigma_c",
    "sigma2_z",
    "rho_corr",
    "initial_logged_days",
    "new_logged_days",
    "validation_logged_days",
    "holdout_logged_days",
    "q_max",
    "use_pessimism",
    "lambda_pess",
    "b_boot",
    "bootstrap_epochs",
    "bootstrap_finetune_epochs",
    "save_bootstrap_ensemble",
    "sim_update_mode",
    "sim_epochs",
    "sim_finetune_epochs",
    "sim_batch_size",
    "sim_hidden_dim",
    "sim_hidden_layers",
    "sim_head_hidden_dim",
    "sim_dropout",
    "sim_activation",
    "sim_lr",
    "sim_weight_decay",
    "logging_policy",
    "center_residual",
    "real_online_update",
    "real_batch_size",
    "real_update_after_steps",
    "real_gradient_steps_per_step",
    "real_exploration_epsilon",
    "synthetic_train_episodes",
    "warmup_episodes",
    "update_after_episodes",
    "update_every_steps",
    "gradient_steps_per_update",
    "rollout_batch_size",
    "batch_rollout_update_cap",
    "hidden_dim",
    "batch_size",
    "replay_size",
    "lr_actor",
    "lr_critic",
    "lr_alpha",
    "dirichlet_init_total_spend_share",
    "ln_init_total_spend_share",
    "td3_init_total_spend_share",
    "candidate_random_count",
    "candidate_budget_fracs",
    "dfl_epochs",
    "dfl_batch_size",
    "dfl_lr",
    "dfl_decision_weight",
    "dfl_softmax_temperature",
    "mpc_horizon",
    "eval_episodes",
    "validation_episodes",
    "final_eval_seed_offset",
]
MAIN_SENSITIVITY_DEFAULTS = {
    "M": 6.0,
    "T": 6.0,
    "daily_budget": 10.0,
    "lambda_pess": 0.001,
    "q_max": 4.0,
    "source_seed": "2026",
}
SENSITIVITY_PLOT_SPECS = [
    ("seed", "source_seed", "Seed sensitivity", "sensitivity_seed"),
    ("budget", "config_daily_budget", "Budget sensitivity", "sensitivity_budget"),
    ("lambda_pess", "config_lambda_pess", "Pessimism sensitivity", "sensitivity_lambda"),
    ("M", "config_M", "City-count sensitivity", "sensitivity_M"),
]


def parse_config_key(config_key: Any) -> Dict[str, Any]:
    if not config_key:
        return {}
    try:
        parsed = json.loads(str(config_key))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def short_config_label(config: Dict[str, Any]) -> str:
    if not config:
        return "config_unknown"
    m = config.get("M", "")
    t = config.get("T", "")
    budget = config.get("daily_budget", "")
    lam = config.get("lambda_pess", "")
    pessimism = config.get("use_pessimism", "")
    q_max = config.get("q_max", "")
    d0 = config.get("initial_logged_days", "")
    center = config.get("center_residual", "")
    real = config.get("real_online_update", "")
    suffix = ""
    if d0 not in ("", None):
        suffix += f"_D0{int(float(d0))}"
    if center not in ("", None):
        suffix += f"_center{int(bool(center)) if isinstance(center, bool) else center}"
    if real not in ("", None):
        suffix += f"_real{int(bool(real)) if isinstance(real, bool) else real}"
    return f"M{m}_T{t}_B{budget}_lambda{lam}_pess{pessimism}_Q{q_max}{suffix}"


def add_config_metadata(item: Dict[str, Any], config_key: Any) -> None:
    config = parse_config_key(config_key)
    item["config_label"] = short_config_label(config)
    for field in CONFIG_DISPLAY_FIELDS:
        item[f"config_{field}"] = config.get(field, "")


def copy_config_metadata_from_row(item: Dict[str, Any], row: Dict[str, Any]) -> None:
    add_config_metadata(item, row.get("config_key", ""))


def method_sort_key(name: str) -> int:
    try:
        return METHOD_ORDER.index(name)
    except ValueError:
        return len(METHOD_ORDER)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x == "" or x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def metric_value(row: Dict[str, Any], metric: str) -> Any:
    if metric == "response_score" and "response_score" not in row and "gmv" in row:
        return row.get("gmv")
    return row.get(metric)


def grouping_config_key(row: Dict[str, Any]) -> str:
    """Use method-specific configs for learned methods and shared configs for repeated references."""
    if str(row.get("method_key", "")) in OPERATIONAL_REFERENCE_KEYS:
        return str(row.get("comparison_config_key", row.get("config_key", "")))
    return str(row.get("config_key", ""))


def summarize_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric in METRICS:
        vals = np.array([to_float(metric_value(r, metric)) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[metric] = float(vals.mean())
        out[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out[f"{metric}_se"] = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        out[f"{metric}_median"] = float(np.median(vals))
        out[f"{metric}_p25"] = float(np.percentile(vals, 25))
        out[f"{metric}_p75"] = float(np.percentile(vals, 75))
    return out


def draw_boxplot(ax: Any, data: Sequence[Sequence[float]], labels: Sequence[str], showfliers: bool = False) -> None:
    """Matplotlib renamed labels to tick_labels in newer versions."""
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=showfliers)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=showfliers)


def discover_result_dirs(root: Path, pattern: str) -> List[Path]:
    return sorted(
        [
            p
            for p in root.glob(pattern)
            if p.is_dir() and ((p / "final_eval_raw.csv").exists() or (p / "round_eval_raw.csv").exists())
        ]
    )


def add_source(rows: Iterable[Dict[str, str]], source_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    source_method_key = ""
    source_seed = ""
    config_key = ""
    comparison_config_key = ""
    cfg_path = source_dir / "config.json"
    if cfg_path.exists():
        try:
            with cfg_path.open("r") as f:
                cfg = json.load(f)
            args = cfg.get("args", {})
            source_method_key = str(args.get("method", ""))
            source_seed = str(args.get("training_seed", args.get("seed", "")))
            config_key = json.dumps(
                {k: args.get(k) for k in CONFIG_KEY_FIELDS},
                sort_keys=True,
                separators=(",", ":"),
            )
            comparison_config_key = json.dumps(
                {k: args.get(k) for k in COMPARISON_CONFIG_FIELDS},
                sort_keys=True,
                separators=(",", ":"),
            )
        except (OSError, json.JSONDecodeError, TypeError):
            source_method_key = ""
            source_seed = ""
            config_key = ""
            comparison_config_key = ""
    if source_method_key:
        source_label = METHOD_KEY_LABELS.get(source_method_key, source_method_key)
        source_method = f"{source_label} seed {source_seed}" if source_seed else source_label
    else:
        source_method = (
            source_dir.name
            .replace("results_cpu_pilot_", "")
            .replace("results_cpu_medium_", "")
            .replace("results_iterative_synthetic_", "")
            .replace("subsidy_smoke_", "")
        )
    for row in rows:
        item: Dict[str, Any] = dict(row)
        method_key = str(item.get("method_key", ""))
        if method_key:
            item["method"] = METHOD_KEY_LABELS.get(method_key, item.get("method", method_key))
        item["source_dir"] = source_dir.name
        item["source_method"] = source_method
        item["source_method_key"] = source_method_key
        item["source_seed"] = source_seed
        item["training_seed"] = source_seed
        item["config_key"] = config_key
        item["comparison_config_key"] = comparison_config_key or config_key
        add_config_metadata(item, config_key)
        if "response_score" not in item and "gmv" in item:
            item["response_score"] = item["gmv"]
        out.append(item)
    return out


def dedupe_final_raw(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        method_key = row.get("method_key", "")
        config_for_key = grouping_config_key(row) if method_key in OPERATIONAL_REFERENCE_KEYS else str(row.get("config_key", ""))
        key = (
            config_for_key,
            row.get("source_seed", ""),
            row.get("method", ""),
            method_key,
            row.get("stage", ""),
            row.get("round", ""),
            row.get("eval_seed", ""),
            row.get("eval_episode", ""),
        )
        # Operational baselines are repeated in every method-specific result
        # directory under the same evaluation seeds; keep one copy.
        if method_key in OPERATIONAL_REFERENCE_KEYS:
            if key in seen:
                continue
            seen.add(key)
        out.append(row)
    return out


def make_summary(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(grouping_config_key(row), str(row.get("method", "")))].append(row)
    summary = []
    for (config_key, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "config_key": config_key,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "eval_episodes": len(rows),
            "num_source_seeds": len({str(r.get("source_seed", "")) for r in rows if str(r.get("source_seed", ""))}),
            "num_source_methods": len(
                {str(r.get("source_method_key", "")) for r in rows if str(r.get("source_method_key", ""))}
            ),
        }
        add_config_metadata(item, config_key)
        item.update(summarize_group(rows))
        summary.append(item)
    return sorted(summary, key=lambda r: (str(r.get("config_label", "")), method_sort_key(str(r["method"]))))


def make_round_summary(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[
            (
                grouping_config_key(row),
                str(row.get("round", "")),
                str(row.get("method", "")),
            )
        ].append(row)
    summary = []
    for (config_key, round_idx, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "config_key": config_key,
            "round": round_idx,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "eval_episodes": len(rows),
            "num_source_seeds": len({str(r.get("source_seed", "")) for r in rows if str(r.get("source_seed", ""))}),
            "num_source_methods": len(
                {str(r.get("source_method_key", "")) for r in rows if str(r.get("source_method_key", ""))}
            ),
        }
        add_config_metadata(item, config_key)
        item.update(summarize_group(rows))
        summary.append(item)
    return sorted(
        summary,
        key=lambda r: (str(r.get("config_label", "")), to_float(r.get("round"), 0.0), method_sort_key(str(r["method"]))),
    )


def make_method_ranking(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranking: List[Dict[str, Any]] = []
    for row in summary_rows:
        if not math.isfinite(to_float(row.get("net_reward"))):
            continue
        item = dict(row)
        item["rank_by_net_reward"] = 0
        ranking.append(item)
    ranking.sort(key=lambda r: to_float(r.get("net_reward")), reverse=True)
    for idx, row in enumerate(ranking, start=1):
        row["rank_by_net_reward"] = idx
        row["rank"] = idx
    return ranking


def make_paired_deltas(
    raw_rows: Sequence[Dict[str, Any]],
    baseline_key: str = "no_subsidy",
) -> List[Dict[str, Any]]:
    def pair_key(row: Dict[str, Any]) -> tuple[str, str, str, str, str, str]:
        return (
            str(row.get("comparison_config_key", row.get("config_key", ""))),
            str(row.get("source_seed", "")),
            str(row.get("stage", "")),
            str(row.get("round", "")),
            str(row.get("eval_seed", "")),
            str(row.get("eval_episode", "")),
        )

    baseline_by_seed = {
        pair_key(r): r
        for r in raw_rows
        if str(r.get("method_key", "")) == baseline_key
    }
    out: List[Dict[str, Any]] = []
    for row in raw_rows:
        method_key = str(row.get("method_key", ""))
        eval_seed = str(row.get("eval_seed", ""))
        base = baseline_by_seed.get(pair_key(row))
        if method_key == baseline_key or base is None:
            continue
        item: Dict[str, Any] = {
            "stage": row.get("stage", ""),
            "round": row.get("round", ""),
            "method": row.get("method", ""),
            "method_key": method_key,
            "baseline_method": base.get("method", "No Subsidy"),
            "baseline_method_key": baseline_key,
            "eval_seed": eval_seed,
            "eval_episode": row.get("eval_episode", ""),
            "source_seed": row.get("source_seed", ""),
            "source_dir": row.get("source_dir", ""),
            "source_method": row.get("source_method", ""),
        }
        item["config_key"] = grouping_config_key(row)
        item["comparison_config_key"] = row.get("comparison_config_key", row.get("config_key", ""))
        for metric in METRICS:
            item[metric] = to_float(metric_value(row, metric))
            item[f"baseline_{metric}"] = to_float(metric_value(base, metric))
            item[f"delta_{metric}"] = item[metric] - item[f"baseline_{metric}"]
        out.append(item)
    return out


def make_paired_deltas_to_best_operational(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def pair_key(row: Dict[str, Any]) -> tuple[str, str, str, str, str, str]:
        return (
            str(row.get("comparison_config_key", row.get("config_key", ""))),
            str(row.get("source_seed", "")),
            str(row.get("stage", "")),
            str(row.get("round", "")),
            str(row.get("eval_seed", "")),
            str(row.get("eval_episode", "")),
        )

    refs_by_seed: Dict[tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if str(row.get("method_key", "")) in OPERATIONAL_REFERENCE_KEYS:
            refs_by_seed[pair_key(row)].append(row)

    out: List[Dict[str, Any]] = []
    for row in raw_rows:
        refs = refs_by_seed.get(pair_key(row), [])
        if not refs:
            continue
        best_ref = max(refs, key=lambda r: to_float(metric_value(r, "net_reward"), -float("inf")))
        method_key = str(row.get("method_key", ""))
        item: Dict[str, Any] = {
            "stage": row.get("stage", ""),
            "round": row.get("round", ""),
            "method": row.get("method", ""),
            "method_key": method_key,
            "baseline_method": "Best Operational",
            "baseline_method_key": "best_operational",
            "best_operational_method": best_ref.get("method", ""),
            "best_operational_method_key": best_ref.get("method_key", ""),
            "eval_seed": row.get("eval_seed", ""),
            "eval_episode": row.get("eval_episode", ""),
            "source_seed": row.get("source_seed", ""),
            "source_dir": row.get("source_dir", ""),
            "source_method": row.get("source_method", ""),
            "config_key": grouping_config_key(row),
            "comparison_config_key": row.get("comparison_config_key", row.get("config_key", "")),
        }
        for metric in METRICS:
            item[metric] = to_float(metric_value(row, metric))
            item[f"baseline_{metric}"] = to_float(metric_value(best_ref, metric))
            item[f"delta_{metric}"] = item[metric] - item[f"baseline_{metric}"]
        out.append(item)
    return out


def make_paired_delta_summary(delta_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        grouped[(grouping_config_key(row), str(row.get("method", "")))].append(row)
    summary: List[Dict[str, Any]] = []
    for (config_key, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "config_key": config_key,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "baseline_method": rows[0].get("baseline_method", "No Subsidy"),
            "eval_episodes": len(rows),
            "num_source_seeds": len({str(r.get("source_seed", "")) for r in rows if str(r.get("source_seed", ""))}),
        }
        add_config_metadata(item, config_key)
        for metric in METRICS:
            key = f"delta_{metric}"
            vals = np.array([to_float(r.get(key)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            item[key] = float(vals.mean())
            item[f"{key}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            item[f"{key}_se"] = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
            item[f"{key}_median"] = float(np.median(vals))
            item[f"{key}_p25"] = float(np.percentile(vals, 25))
            item[f"{key}_p75"] = float(np.percentile(vals, 75))
        summary.append(item)
    return sorted(summary, key=lambda r: (str(r.get("config_label", "")), method_sort_key(str(r["method"]))))


def make_seed_mean_paired_delta_rows(delta_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        grouped[
            (
                grouping_config_key(row),
                str(row.get("source_seed", "")),
                str(row.get("stage", "")),
                str(row.get("method", "")),
            )
        ].append(row)

    seed_rows: List[Dict[str, Any]] = []
    for (config_key, source_seed, stage, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "config_key": config_key,
            "source_seed": source_seed,
            "stage": stage,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "baseline_method": rows[0].get("baseline_method", "No Subsidy"),
            "paired_eval_episodes": len(rows),
        }
        add_config_metadata(item, config_key)
        for metric in METRICS:
            key = f"delta_{metric}"
            vals = np.array([to_float(r.get(key)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                item[key] = float(vals.mean())
        seed_rows.append(item)
    return sorted(
        seed_rows,
        key=lambda r: (
            str(r.get("config_key", "")),
            method_sort_key(str(r.get("method", ""))),
            str(r.get("source_seed", "")),
            str(r.get("stage", "")),
        ),
    )


def make_seed_level_paired_delta_summary(seed_delta_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in seed_delta_rows:
        grouped[
            (
                grouping_config_key(row),
                str(row.get("stage", "")),
                str(row.get("method", "")),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []
    for (config_key, stage, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "config_key": config_key,
            "stage": stage,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "baseline_method": rows[0].get("baseline_method", "No Subsidy"),
            "num_seeds": len({str(r.get("source_seed", "")) for r in rows}),
        }
        add_config_metadata(item, config_key)
        for metric in METRICS:
            key = f"delta_{metric}"
            vals = np.array([to_float(r.get(key)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            item[key] = float(vals.mean())
            item[f"{key}_std_across_seeds"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            item[f"{key}_se_across_seeds"] = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
            item[f"{key}_median_across_seeds"] = float(np.median(vals))
            item[f"{key}_p25_across_seeds"] = float(np.percentile(vals, 25))
            item[f"{key}_p75_across_seeds"] = float(np.percentile(vals, 75))
        out.append(item)
    return sorted(
        out,
        key=lambda r: (str(r.get("config_key", "")), method_sort_key(str(r.get("method", ""))), str(r.get("stage", ""))),
    )


def make_seed_mean_rows(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[
            (
                grouping_config_key(row),
                str(row.get("source_seed", "")),
                str(row.get("method", "")),
            )
        ].append(row)

    seed_rows: List[Dict[str, Any]] = []
    for (config_key, source_seed, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "config_key": config_key,
            "source_seed": source_seed,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "eval_episodes": len(rows),
        }
        add_config_metadata(item, config_key)
        for metric in METRICS:
            vals = np.array([to_float(metric_value(r, metric)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                item[metric] = float(vals.mean())
        seed_rows.append(item)
    return sorted(seed_rows, key=lambda r: (str(r.get("config_key", "")), method_sort_key(str(r["method"])), str(r.get("source_seed", ""))))


def make_seed_level_summary(seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(grouping_config_key(row), str(row.get("method", "")))].append(row)

    out: List[Dict[str, Any]] = []
    for (config_key, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "config_key": config_key,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "num_seeds": len({str(r.get("source_seed", "")) for r in rows}),
        }
        add_config_metadata(item, config_key)
        for metric in METRICS:
            vals = np.array([to_float(r.get(metric)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            item[metric] = float(vals.mean())
            item[f"{metric}_std_across_seeds"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            item[f"{metric}_se_across_seeds"] = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
            item[f"{metric}_median_across_seeds"] = float(np.median(vals))
        out.append(item)
    return sorted(out, key=lambda r: (str(r.get("config_key", "")), method_sort_key(str(r["method"]))))


def make_round_seed_mean_rows(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[
            (
                grouping_config_key(row),
                str(row.get("source_seed", "")),
                str(row.get("round", "")),
                str(row.get("method", "")),
            )
        ].append(row)

    seed_rows: List[Dict[str, Any]] = []
    for (config_key, source_seed, round_idx, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "config_key": config_key,
            "source_seed": source_seed,
            "round": round_idx,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "eval_episodes": len(rows),
        }
        add_config_metadata(item, config_key)
        for metric in METRICS:
            vals = np.array([to_float(metric_value(r, metric)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                item[metric] = float(vals.mean())
        seed_rows.append(item)
    return sorted(
        seed_rows,
        key=lambda r: (
            str(r.get("config_label", "")),
            str(r.get("source_seed", "")),
            to_float(r.get("round"), 0.0),
            method_sort_key(str(r["method"])),
        ),
    )


def make_round_seed_level_summary(round_seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in round_seed_rows:
        grouped[
            (
                grouping_config_key(row),
                str(row.get("round", "")),
                str(row.get("method", "")),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []
    for (config_key, round_idx, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "config_key": config_key,
            "round": round_idx,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "num_seeds": len({str(r.get("source_seed", "")) for r in rows if str(r.get("source_seed", ""))}),
        }
        add_config_metadata(item, config_key)
        for metric in METRICS:
            vals = np.array([to_float(r.get(metric)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            item[metric] = float(vals.mean())
            item[f"{metric}_std_across_seeds"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            item[f"{metric}_se_across_seeds"] = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
            item[f"{metric}_median_across_seeds"] = float(np.median(vals))
        out.append(item)
    return sorted(
        out,
        key=lambda r: (str(r.get("config_label", "")), to_float(r.get("round"), 0.0), method_sort_key(str(r["method"]))),
    )


def add_calibration_aliases(rows: Sequence[Dict[str, Any]]) -> None:
    for row in rows:
        if row.get("source_method_key") and not row.get("method_key"):
            row["method_key"] = row.get("source_method_key")
        if row.get("true_method_key") and not row.get("evaluated_policy_key"):
            row["evaluated_policy_key"] = row.get("true_method_key")


def safe_filename(text: str) -> str:
    keep = []
    for ch in str(text).lower():
        keep.append(ch if ch.isalnum() else "_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    if not out:
        return "value"
    max_len = 40
    if len(out) <= max_len:
        return out
    digest = hashlib.sha1(out.encode("utf-8")).hexdigest()[:8]
    return f"{out[:max_len - 9]}_{digest}"


def _num_equal(value: Any, target: float, tol: float = 1e-9) -> bool:
    v = to_float(value)
    return math.isfinite(v) and abs(v - float(target)) <= tol


def _is_main_config_except(row: Dict[str, Any], varying: str) -> bool:
    checks = {
        "config_M": MAIN_SENSITIVITY_DEFAULTS["M"],
        "config_T": MAIN_SENSITIVITY_DEFAULTS["T"],
        "config_daily_budget": MAIN_SENSITIVITY_DEFAULTS["daily_budget"],
        "config_lambda_pess": MAIN_SENSITIVITY_DEFAULTS["lambda_pess"],
        "config_q_max": MAIN_SENSITIVITY_DEFAULTS["q_max"],
    }
    for field, expected in checks.items():
        if field == varying:
            continue
        if not _num_equal(row.get(field), expected):
            return False
    if varying != "source_seed":
        seed = str(row.get("source_seed", ""))
        if seed and seed != str(MAIN_SENSITIVITY_DEFAULTS["source_seed"]):
            return False
    return True


def make_sensitivity_rows(seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build one-factor-at-a-time sensitivity rows from seed-level final means.

    Seed-level rows avoid accidentally mixing the seed sweep into budget,
    lambda, or M sensitivity panels. The main setting is reused across views.
    """
    out: List[Dict[str, Any]] = []
    for view, varying, title, _prefix in SENSITIVITY_PLOT_SPECS:
        for row in seed_rows:
            if not _is_main_config_except(row, varying):
                continue
            item = dict(row)
            item["sensitivity_view"] = view
            item["sensitivity_title"] = title
            item["sensitivity_variable"] = varying
            item["sensitivity_level"] = str(row.get(varying, ""))
            out.append(item)
    return sorted(
        out,
        key=lambda r: (
            str(r.get("sensitivity_view", "")),
            to_float(r.get("sensitivity_level"), float("inf"))
            if str(r.get("sensitivity_variable", "")) != "source_seed"
            else to_float(r.get("source_seed"), float("inf")),
            method_sort_key(str(r.get("method", ""))),
            str(r.get("source_seed", "")),
        ),
    )


def make_sensitivity_summary(sensitivity_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in sensitivity_rows:
        grouped[
            (
                str(row.get("sensitivity_view", "")),
                str(row.get("sensitivity_level", "")),
                str(row.get("method", "")),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []
    for (view, level, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "sensitivity_view": view,
            "sensitivity_title": rows[0].get("sensitivity_title", ""),
            "sensitivity_variable": rows[0].get("sensitivity_variable", ""),
            "sensitivity_level": level,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "num_rows": len(rows),
            "num_source_seeds": len({str(r.get("source_seed", "")) for r in rows if str(r.get("source_seed", ""))}),
        }
        for metric in METRICS:
            vals = np.array([to_float(r.get(metric)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            item[metric] = float(vals.mean())
            item[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            item[f"{metric}_se"] = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(item)
    return sorted(
        out,
        key=lambda r: (
            str(r.get("sensitivity_view", "")),
            to_float(r.get("sensitivity_level"), float("inf")),
            method_sort_key(str(r.get("method", ""))),
        ),
    )


def make_round_sensitivity_summary(round_seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sens_rows = make_sensitivity_rows(round_seed_rows)
    grouped: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in sens_rows:
        grouped[
            (
                str(row.get("sensitivity_view", "")),
                str(row.get("sensitivity_level", "")),
                str(row.get("round", "")),
                str(row.get("method", "")),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []
    for (view, level, round_idx, method), rows in grouped.items():
        item: Dict[str, Any] = {
            "sensitivity_view": view,
            "sensitivity_title": rows[0].get("sensitivity_title", ""),
            "sensitivity_variable": rows[0].get("sensitivity_variable", ""),
            "sensitivity_level": level,
            "round": round_idx,
            "method": method,
            "method_key": rows[0].get("method_key", ""),
            "num_rows": len(rows),
            "num_source_seeds": len({str(r.get("source_seed", "")) for r in rows if str(r.get("source_seed", ""))}),
        }
        for metric in METRICS:
            vals = np.array([to_float(r.get(metric)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            item[metric] = float(vals.mean())
            item[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            item[f"{metric}_se"] = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(item)
    return sorted(
        out,
        key=lambda r: (
            str(r.get("sensitivity_view", "")),
            to_float(r.get("sensitivity_level"), float("inf")),
            to_float(r.get("round"), float("inf")),
            method_sort_key(str(r.get("method", ""))),
        ),
    )


def plot_boxplots(raw_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    methods = sorted({str(r.get("method", "")) for r in raw_rows}, key=method_sort_key)
    for metric in METRICS:
        data = []
        labels = []
        for method in methods:
            vals = [to_float(metric_value(r, metric)) for r in raw_rows if str(r.get("method", "")) == method]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                data.append(vals)
                labels.append(method)
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(labels)), 5))
        draw_boxplot(ax, data, labels, showfliers=False)
        ax.set_title(f"Final true-DGP evaluation: {metric}")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"final_boxplot_{metric}.png", dpi=220)
        plt.close(fig)


def plot_mean_bars(summary_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    for metric in METRICS:
        rows = [r for r in summary_rows if metric in r]
        if not rows:
            continue
        labels = [str(r["method"]) for r in rows]
        vals = [to_float(r.get(metric)) for r in rows]
        ses = [to_float(r.get(f"{metric}_se"), 0.0) for r in rows]
        fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(labels)), 5))
        ax.bar(labels, vals, yerr=ses, capsize=3)
        ax.set_title(f"Final true-DGP evaluation mean +/- SE: {metric}")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"final_mean_{metric}.png", dpi=220)
        plt.close(fig)


def plot_final_boxplots_by_config(raw_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not raw_rows:
        return
    config_labels = sorted({str(r.get("config_label", "config_unknown")) for r in raw_rows})
    for config_label in config_labels:
        config_rows = [r for r in raw_rows if str(r.get("config_label", "config_unknown")) == config_label]
        methods = sorted({str(r.get("method", "")) for r in config_rows}, key=method_sort_key)
        for metric in METRICS:
            data = []
            labels = []
            for method in methods:
                vals = [
                    to_float(metric_value(r, metric))
                    for r in config_rows
                    if str(r.get("method", "")) == method
                ]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    data.append(vals)
                    labels.append(method)
            if not data:
                continue
            fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(labels)), 5))
            draw_boxplot(ax, data, labels, showfliers=False)
            ax.set_title(f"Final true-DGP evaluation: {metric}\n{config_label}")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(out_dir / f"final_boxplot_{metric}_{safe_filename(config_label)}.png", dpi=220)
            plt.close(fig)


def plot_final_boxplots_faceted_by_config(raw_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    """Facet final true-DGP boxplots by experiment setting.

    The legacy final_boxplot_*.png files pool all result directories that match
    the requested pattern. That is fine for one setting, but misleading for
    server sweeps over M/T/budget/lambda. This plot keeps each config in its
    own panel so settings are visually separated.
    """
    if not raw_rows:
        return
    config_labels = sorted({str(r.get("config_label", "config_unknown")) for r in raw_rows})
    if not config_labels:
        return
    ncols = min(2, len(config_labels))
    nrows = int(math.ceil(len(config_labels) / ncols))
    for metric in METRICS:
        fig, axes = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 4.8 * nrows), squeeze=False)
        plotted_any = False
        for idx, config_label in enumerate(config_labels):
            ax = axes[idx // ncols][idx % ncols]
            config_rows = [r for r in raw_rows if str(r.get("config_label", "config_unknown")) == config_label]
            methods = sorted({str(r.get("method", "")) for r in config_rows}, key=method_sort_key)
            data = []
            labels = []
            for method in methods:
                vals = [
                    to_float(metric_value(r, metric))
                    for r in config_rows
                    if str(r.get("method", "")) == method
                ]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    data.append(vals)
                    labels.append(method)
            if data:
                draw_boxplot(ax, data, labels, showfliers=False)
                ax.set_title(config_label, fontsize=9)
                ax.set_ylabel(metric)
                ax.tick_params(axis="x", rotation=30, labelsize=8)
                plotted_any = True
            else:
                ax.axis("off")
        for idx in range(len(config_labels), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        if plotted_any:
            fig.suptitle(f"Final true-DGP evaluation by setting: {metric}", y=1.02)
            fig.tight_layout()
            fig.savefig(out_dir / f"final_boxplot_{metric}_by_config.png", dpi=220)
        plt.close(fig)


def plot_seed_mean_points(
    seed_rows: Sequence[Dict[str, Any]],
    seed_level_summary: Sequence[Dict[str, Any]],
    out_dir: Path,
) -> None:
    if not seed_rows:
        return
    for config_label in sorted({str(r.get("config_label", "config_unknown")) for r in seed_rows}):
        config_seed_rows = [r for r in seed_rows if str(r.get("config_label", "config_unknown")) == config_label]
        config_summary_rows = [
            r for r in seed_level_summary if str(r.get("config_label", "config_unknown")) == config_label
        ]
        methods = sorted({str(r.get("method", "")) for r in config_seed_rows}, key=method_sort_key)
        xs = np.arange(len(methods), dtype=np.float64)
        for metric in METRICS:
            means_by_method = {
                str(r.get("method", "")): to_float(r.get(metric))
                for r in config_summary_rows
                if math.isfinite(to_float(r.get(metric)))
            }
            if not means_by_method:
                continue
            fig, ax = plt.subplots(figsize=(max(9, 1.25 * len(methods)), 5))
            means = [means_by_method.get(method, float("nan")) for method in methods]
            ses = []
            for method in methods:
                match = [r for r in config_summary_rows if str(r.get("method", "")) == method]
                ses.append(to_float(match[0].get(f"{metric}_se_across_seeds"), 0.0) if match else 0.0)
            ax.bar(xs, means, yerr=ses, capsize=3, alpha=0.65)
            for i, method in enumerate(methods):
                vals = [
                    (str(r.get("source_seed", "")), to_float(r.get(metric)))
                    for r in config_seed_rows
                    if str(r.get("method", "")) == method and math.isfinite(to_float(r.get(metric)))
                ]
                for j, (seed, value) in enumerate(vals):
                    offset = 0.0 if len(vals) <= 1 else (j - (len(vals) - 1) / 2.0) * min(0.08, 0.35 / len(vals))
                    ax.scatter(i + offset, value, s=22, color="black", alpha=0.75)
                    if len(methods) <= 8 and len(vals) <= 5:
                        ax.text(i + offset, value, str(seed), fontsize=6, rotation=45, ha="left", va="bottom")
            ax.set_title(f"Final seed-level mean +/- SE across seeds: {metric}\n{config_label}")
            ax.set_ylabel(metric)
            ax.set_xticks(xs, labels=methods, rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / f"final_seed_mean_{metric}_{safe_filename(config_label)}.png", dpi=220)
            plt.close(fig)


def plot_sensitivity_final_net_reward(
    sensitivity_summary: Sequence[Dict[str, Any]],
    out_dir: Path,
) -> None:
    if not sensitivity_summary:
        return
    for view, _varying, title, prefix in SENSITIVITY_PLOT_SPECS:
        rows = [r for r in sensitivity_summary if str(r.get("sensitivity_view", "")) == view]
        if not rows:
            continue
        levels = sorted(
            {str(r.get("sensitivity_level", "")) for r in rows},
            key=lambda x: to_float(x, float("inf")),
        )
        methods = sorted({str(r.get("method", "")) for r in rows}, key=method_sort_key)
        x = np.arange(len(levels), dtype=np.float64)
        width = min(0.8 / max(len(methods), 1), 0.18)
        fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(levels) + 4), 5))
        plotted = False
        for idx, method in enumerate(methods):
            vals = []
            ses = []
            for level in levels:
                match = [
                    r
                    for r in rows
                    if str(r.get("method", "")) == method
                    and str(r.get("sensitivity_level", "")) == level
                ]
                vals.append(to_float(match[0].get("net_reward")) if match else float("nan"))
                ses.append(to_float(match[0].get("net_reward_se"), 0.0) if match else 0.0)
            if not any(math.isfinite(v) for v in vals):
                continue
            offset = (idx - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, yerr=ses, capsize=2, label=method)
            plotted = True
        if plotted:
            ax.set_title(f"{title}: final true-DGP net_reward")
            ax.set_xlabel(view)
            ax.set_ylabel("net_reward")
            ax.set_xticks(x, labels=levels)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{prefix}_final_net_reward.png", dpi=220)
        plt.close(fig)


def plot_sensitivity_final_delta(
    sensitivity_rows: Sequence[Dict[str, Any]],
    out_dir: Path,
) -> None:
    if not sensitivity_rows:
        return
    # Compute best operational reference within each sensitivity level and
    # source seed, then average paired deltas by method.
    for view, _varying, title, prefix in SENSITIVITY_PLOT_SPECS:
        rows = [r for r in sensitivity_rows if str(r.get("sensitivity_view", "")) == view]
        if not rows:
            continue
        ref_by_pair: Dict[tuple[str, str], Dict[str, Any]] = {}
        for row in rows:
            if str(row.get("method_key", "")) not in OPERATIONAL_REFERENCE_KEYS:
                continue
            key = (str(row.get("sensitivity_level", "")), str(row.get("source_seed", "")))
            current = ref_by_pair.get(key)
            if current is None or to_float(row.get("net_reward"), -float("inf")) > to_float(
                current.get("net_reward"), -float("inf")
            ):
                ref_by_pair[key] = row

        grouped: Dict[tuple[str, str], List[float]] = defaultdict(list)
        for row in rows:
            method = str(row.get("method", ""))
            method_key = str(row.get("method_key", ""))
            if method_key in OPERATIONAL_REFERENCE_KEYS:
                continue
            key = (str(row.get("sensitivity_level", "")), str(row.get("source_seed", "")))
            ref = ref_by_pair.get(key)
            if not ref:
                continue
            delta = to_float(row.get("net_reward")) - to_float(ref.get("net_reward"))
            if math.isfinite(delta):
                grouped[(str(row.get("sensitivity_level", "")), method)].append(delta)

        levels = sorted({level for level, _method in grouped}, key=lambda x: to_float(x, float("inf")))
        methods = sorted({method for _level, method in grouped}, key=method_sort_key)
        if not levels or not methods:
            continue
        x = np.arange(len(levels), dtype=np.float64)
        width = min(0.8 / max(len(methods), 1), 0.18)
        fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(levels) + 4), 5))
        plotted = False
        for idx, method in enumerate(methods):
            vals = []
            ses = []
            for level in levels:
                ds = grouped.get((level, method), [])
                vals.append(float(np.mean(ds)) if ds else float("nan"))
                ses.append(float(np.std(ds, ddof=1) / math.sqrt(len(ds))) if len(ds) > 1 else 0.0)
            if not any(math.isfinite(v) for v in vals):
                continue
            offset = (idx - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, yerr=ses, capsize=2, label=method)
            plotted = True
        if plotted:
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
            ax.set_title(f"{title}: final delta vs best operational")
            ax.set_xlabel(view)
            ax.set_ylabel("delta_net_reward")
            ax.set_xticks(x, labels=levels)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{prefix}_final_delta_vs_best_operational.png", dpi=220)
        plt.close(fig)


def plot_paired_delta_net_reward(
    delta_rows: Sequence[Dict[str, Any]],
    delta_summary: Sequence[Dict[str, Any]],
    out_dir: Path,
    suffix: str = "vs_no_subsidy",
    reference_label: str = "No Subsidy",
) -> None:
    methods = sorted({str(r.get("method", "")) for r in delta_rows}, key=method_sort_key)
    data = []
    labels = []
    for method in methods:
        vals = [to_float(r.get("delta_net_reward")) for r in delta_rows if str(r.get("method", "")) == method]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            data.append(vals)
            labels.append(method)
    if data:
        fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(labels)), 5))
        ax.axhline(0.0, color="black", linewidth=1)
        draw_boxplot(ax, data, labels, showfliers=False)
        ax.set_title(f"Paired final true-DGP delta vs {reference_label}: net_reward")
        ax.set_ylabel("delta_net_reward")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"final_boxplot_delta_net_reward_{suffix}.png", dpi=220)
        plt.close(fig)

    rows = [r for r in delta_summary if "delta_net_reward" in r]
    if rows:
        labels = [str(r["method"]) for r in rows]
        vals = [to_float(r.get("delta_net_reward")) for r in rows]
        ses = [to_float(r.get("delta_net_reward_se"), 0.0) for r in rows]
        fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(labels)), 5))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.bar(labels, vals, yerr=ses, capsize=3)
        ax.set_title(f"Paired final true-DGP mean delta vs {reference_label}: net_reward")
        ax.set_ylabel("delta_net_reward")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"final_mean_delta_net_reward_{suffix}.png", dpi=220)
        plt.close(fig)


def plot_round_eval_curves(round_summary_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not round_summary_rows:
        return
    for metric, filename, title, zero_line in [
        ("net_reward", "round_eval_curve_net_reward.png", "Per-round true-DGP evaluation: net_reward", False),
        ("response_score", "round_eval_curve_response_score.png", "Per-round true-DGP evaluation: response_score", False),
        ("spend", "round_eval_curve_spend.png", "Per-round true-DGP evaluation: spend", False),
        ("final_budget", "round_eval_curve_final_budget.png", "Per-round true-DGP evaluation: final_budget", False),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for method in sorted({str(r.get("method", "")) for r in round_summary_rows}, key=method_sort_key):
            rows = sorted(
                [r for r in round_summary_rows if str(r.get("method", "")) == method],
                key=lambda r: to_float(r.get("round"), 0.0),
            )
            pairs = [
                (to_float(r.get("round")), to_float(r.get(metric)))
                for r in rows
                if math.isfinite(to_float(r.get("round"))) and math.isfinite(to_float(r.get(metric)))
            ]
            if not pairs:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=method)
            plotted = True
        if plotted:
            if zero_line:
                ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel("round")
            ax.set_ylabel(metric)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)


def plot_round_eval_curves_by_config(round_summary_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not round_summary_rows:
        return
    config_labels = sorted({str(r.get("config_label", "config_unknown")) for r in round_summary_rows})
    for config_label in config_labels:
        config_rows = [r for r in round_summary_rows if str(r.get("config_label", "config_unknown")) == config_label]
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(8, 5))
            plotted = False
            for method in sorted({str(r.get("method", "")) for r in config_rows}, key=method_sort_key):
                rows = sorted(
                    [r for r in config_rows if str(r.get("method", "")) == method],
                    key=lambda r: to_float(r.get("round"), 0.0),
                )
                pairs = [
                    (to_float(r.get("round")), to_float(r.get(metric)))
                    for r in rows
                    if math.isfinite(to_float(r.get("round"))) and math.isfinite(to_float(r.get(metric)))
                ]
                if not pairs:
                    continue
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=method)
                plotted = True
            if plotted:
                ax.set_title(f"Per-round true-DGP {metric}\n{config_label}")
                ax.set_xlabel("round")
                ax.set_ylabel(metric)
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(
                    out_dir / f"round_eval_curve_{metric}_{safe_filename(config_label)}.png",
                    dpi=220,
                )
            plt.close(fig)


def plot_round_seed_curves(round_seed_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not round_seed_rows:
        return
    for config_label in sorted({str(r.get("config_label", "config_unknown")) for r in round_seed_rows}):
        config_rows = [r for r in round_seed_rows if str(r.get("config_label", "config_unknown")) == config_label]
        seeds = sorted({str(r.get("source_seed", "")) for r in config_rows if str(r.get("source_seed", ""))})
        if not seeds:
            continue
        ncols = min(3, len(seeds))
        nrows = int(math.ceil(len(seeds) / ncols))
        for metric in ["net_reward", "response_score", "spend", "final_budget"]:
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)
            plotted_any = False
            for idx, seed in enumerate(seeds):
                ax = axes[idx // ncols][idx % ncols]
                seed_rows = [r for r in config_rows if str(r.get("source_seed", "")) == seed]
                plotted = False
                for method in sorted({str(r.get("method", "")) for r in seed_rows}, key=method_sort_key):
                    rows = sorted(
                        [r for r in seed_rows if str(r.get("method", "")) == method],
                        key=lambda r: to_float(r.get("round"), 0.0),
                    )
                    pairs = [
                        (to_float(r.get("round")), to_float(r.get(metric)))
                        for r in rows
                        if math.isfinite(to_float(r.get("round"))) and math.isfinite(to_float(r.get(metric)))
                    ]
                    if not pairs:
                        continue
                    ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=method)
                    plotted = True
                    plotted_any = True
                ax.set_title(f"seed {seed}")
                ax.set_xlabel("round")
                ax.set_ylabel(metric)
                if plotted:
                    ax.legend(fontsize=7)
            for idx in range(len(seeds), nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")
            if plotted_any:
                fig.suptitle(f"Per-round true-DGP {metric} by seed\n{config_label}", y=1.02)
                fig.tight_layout()
                fig.savefig(out_dir / f"round_eval_curve_{metric}_by_seed_{safe_filename(config_label)}.png", dpi=220)
            plt.close(fig)


def plot_round_delta_curves(delta_rows: Sequence[Dict[str, Any]], out_dir: Path, suffix: str, title: str) -> None:
    if not delta_rows:
        return
    for metric in ["delta_net_reward", "delta_response_score", "delta_spend"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for method in sorted({str(r.get("method", "")) for r in delta_rows}, key=method_sort_key):
            rows = sorted(
                [r for r in delta_rows if str(r.get("method", "")) == method],
                key=lambda r: to_float(r.get("round"), 0.0),
            )
            by_round: Dict[float, List[float]] = defaultdict(list)
            for row in rows:
                x = to_float(row.get("round"))
                y = to_float(row.get(metric))
                if math.isfinite(x) and math.isfinite(y):
                    by_round[x].append(y)
            pairs = [(x, float(np.mean(vals))) for x, vals in sorted(by_round.items()) if vals]
            if not pairs:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=method)
            plotted = True
        if plotted:
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
            ax.set_title(f"{title}: {metric}")
            ax.set_xlabel("round")
            ax.set_ylabel(metric)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"round_eval_curve_{metric}_{suffix}.png", dpi=220)
        plt.close(fig)


def plot_sensitivity_round_curves(
    round_sensitivity_summary: Sequence[Dict[str, Any]],
    out_dir: Path,
) -> None:
    if not round_sensitivity_summary:
        return
    for view, _varying, title, prefix in SENSITIVITY_PLOT_SPECS:
        rows = [r for r in round_sensitivity_summary if str(r.get("sensitivity_view", "")) == view]
        if not rows:
            continue
        levels = sorted(
            {str(r.get("sensitivity_level", "")) for r in rows},
            key=lambda x: to_float(x, float("inf")),
        )
        ncols = min(3, len(levels))
        nrows = int(math.ceil(len(levels) / ncols))
        for metric in ["net_reward", "spend"]:
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), squeeze=False)
            plotted_any = False
            for idx, level in enumerate(levels):
                ax = axes[idx // ncols][idx % ncols]
                level_rows = [r for r in rows if str(r.get("sensitivity_level", "")) == level]
                for method in sorted({str(r.get("method", "")) for r in level_rows}, key=method_sort_key):
                    method_rows = sorted(
                        [r for r in level_rows if str(r.get("method", "")) == method],
                        key=lambda r: to_float(r.get("round"), 0.0),
                    )
                    pairs = [
                        (to_float(r.get("round")), to_float(r.get(metric)))
                        for r in method_rows
                        if math.isfinite(to_float(r.get("round"))) and math.isfinite(to_float(r.get(metric)))
                    ]
                    if not pairs:
                        continue
                    ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=method)
                    plotted_any = True
                ax.set_title(f"{view}={level}", fontsize=9)
                ax.set_xlabel("round")
                ax.set_ylabel(metric)
                if plotted_any:
                    ax.legend(fontsize=7)
            for idx in range(len(levels), nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")
            if plotted_any:
                fig.suptitle(f"{title}: per-round {metric}", y=1.02)
                fig.tight_layout()
                fig.savefig(out_dir / f"{prefix}_round_eval_curve_{metric}.png", dpi=220)
            plt.close(fig)


def plot_sensitivity_round_delta_vs_best_operational(
    round_sensitivity_summary: Sequence[Dict[str, Any]],
    out_dir: Path,
) -> None:
    if not round_sensitivity_summary:
        return
    for view, _varying, title, prefix in SENSITIVITY_PLOT_SPECS:
        rows = [r for r in round_sensitivity_summary if str(r.get("sensitivity_view", "")) == view]
        if not rows:
            continue
        ref_by_level_round: Dict[tuple[str, str], float] = {}
        for row in rows:
            if str(row.get("method_key", "")) not in OPERATIONAL_REFERENCE_KEYS:
                continue
            key = (str(row.get("sensitivity_level", "")), str(row.get("round", "")))
            current = ref_by_level_round.get(key, -float("inf"))
            ref_by_level_round[key] = max(current, to_float(row.get("net_reward"), -float("inf")))

        levels = sorted(
            {str(r.get("sensitivity_level", "")) for r in rows},
            key=lambda x: to_float(x, float("inf")),
        )
        ncols = min(3, len(levels))
        nrows = int(math.ceil(len(levels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), squeeze=False)
        plotted_any = False
        for idx, level in enumerate(levels):
            ax = axes[idx // ncols][idx % ncols]
            level_rows = [r for r in rows if str(r.get("sensitivity_level", "")) == level]
            for method in sorted({str(r.get("method", "")) for r in level_rows}, key=method_sort_key):
                method_key = next((str(r.get("method_key", "")) for r in level_rows if str(r.get("method", "")) == method), "")
                if method_key in OPERATIONAL_REFERENCE_KEYS:
                    continue
                method_rows = sorted(
                    [r for r in level_rows if str(r.get("method", "")) == method],
                    key=lambda r: to_float(r.get("round"), 0.0),
                )
                pairs = []
                for row in method_rows:
                    q = str(row.get("round", ""))
                    ref = ref_by_level_round.get((level, q), float("nan"))
                    delta = to_float(row.get("net_reward")) - ref
                    if math.isfinite(to_float(q)) and math.isfinite(delta):
                        pairs.append((to_float(q), delta))
                if not pairs:
                    continue
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=method)
                plotted_any = True
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
            ax.set_title(f"{view}={level}", fontsize=9)
            ax.set_xlabel("round")
            ax.set_ylabel("delta_net_reward")
            if plotted_any:
                ax.legend(fontsize=7)
        for idx in range(len(levels), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        if plotted_any:
            fig.suptitle(f"{title}: per-round delta vs best operational", y=1.02)
            fig.tight_layout()
            fig.savefig(out_dir / f"{prefix}_round_eval_curve_delta_vs_best_operational.png", dpi=220)
        plt.close(fig)


def plot_validation_curves(iter_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    for metric in [
        "val_uncertainty_penalty",
        "mean_policy_uncertainty",
        "residual_mean_l1_raw",
        "residual_mean_linf_raw",
        "holdout_mse_w_raw",
        "holdout_mse_z",
        "holdout_mse_x_next",
        "holdout_mse_y_next",
        "holdout_mse_c_next",
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for source in sorted({str(r.get("source_method", "")) for r in iter_rows}):
            rows = sorted(
                [r for r in iter_rows if str(r.get("source_method", "")) == source],
                key=lambda r: to_float(r.get("round"), 0.0),
            )
            xs = [to_float(r.get("round")) for r in rows]
            ys = [to_float(r.get(metric)) for r in rows]
            pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
            if not pairs:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=source)
            plotted = True
        if plotted:
            ax.set_title(f"Learned-simulator diagnostic trajectory: {metric}")
            ax.set_xlabel("round")
            ax.set_ylabel(metric)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"validation_curve_{metric}.png", dpi=220)
        plt.close(fig)

    for metric, filename, title in [
        ("val_uncertainty_penalty", "validation_curve_uncertainty.png", "Diagnostic rollout uncertainty penalty"),
        ("holdout_mse_w_raw", "holdout_mse_by_round.png", "Fixed holdout response MSE by round"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for source in sorted({str(r.get("source_method", "")) for r in iter_rows}):
            rows = sorted(
                [r for r in iter_rows if str(r.get("source_method", "")) == source],
                key=lambda r: to_float(r.get("round"), 0.0),
            )
            pairs = [
                (to_float(r.get("round")), to_float(r.get(metric)))
                for r in rows
                if math.isfinite(to_float(r.get("round"))) and math.isfinite(to_float(r.get(metric)))
            ]
            if not pairs:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=source)
            plotted = True
        if plotted:
            ax.set_title(title)
            ax.set_xlabel("round")
            ax.set_ylabel(metric)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)

    rows_with_spend_unc = [
        r
        for r in iter_rows
        if math.isfinite(to_float(r.get("val_spend"))) and math.isfinite(to_float(r.get("mean_policy_uncertainty")))
    ]
    if rows_with_spend_unc:
        fig, ax = plt.subplots(figsize=(8, 5))
        for source in sorted({str(r.get("source_method", "")) for r in rows_with_spend_unc}):
            rows = [r for r in rows_with_spend_unc if str(r.get("source_method", "")) == source]
            ax.scatter(
                [to_float(r.get("mean_policy_uncertainty")) for r in rows],
                [to_float(r.get("val_spend")) for r in rows],
                label=source,
                alpha=0.8,
            )
        ax.set_title("Validation spend vs bootstrap uncertainty")
        ax.set_xlabel("mean_policy_uncertainty")
        ax.set_ylabel("val_spend")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "policy_spend_vs_uncertainty.png", dpi=220)
        plt.close(fig)


def plot_training_curves(train_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    metrics = [
        "synthetic_return",
        "actor_loss",
        "q1_loss",
        "q2_loss",
        "dfl_loss",
        "dfl_pred_loss",
        "dfl_decision_loss",
        "dfl_teacher_candidate_spend",
        "dfl_selected_candidate_spend",
        "dfl_candidate_score_range",
        "dfl_candidate_penalty_mean",
        "synthetic_raw_return",
        "synthetic_uncertainty_penalty",
    ]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for source in sorted({str(r.get("source_method", "")) for r in train_rows}):
            rows = [r for r in train_rows if str(r.get("source_method", "")) == source]
            rows = sorted(rows, key=lambda r: (to_float(r.get("round"), 0.0), to_float(r.get("step"), 0.0)))
            xs = list(range(1, len(rows) + 1))
            ys = [to_float(r.get(metric)) for r in rows]
            pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
            if not pairs:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=source)
            plotted = True
        if plotted:
            ax.set_title(f"Training log trajectory: {metric}")
            ax.set_xlabel("logged training checkpoint")
            ax.set_ylabel(metric)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"training_curve_{metric}.png", dpi=220)
        plt.close(fig)


def plot_simulator_diagnostic_curves(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    metrics = [
        ("sim_minus_true_net_reward", "Simulator rollout minus true DGP: net_reward"),
        ("sim_minus_true_response_score", "Simulator rollout minus true DGP: response_score"),
        ("sim_minus_true_spend", "Simulator rollout minus true DGP: spend"),
    ]
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for source in sorted({str(r.get("source_method", "")) for r in rows}, key=method_sort_key):
            source_rows = [r for r in rows if str(r.get("source_method", "")) == source]
            policies = sorted({str(r.get("diagnostic_policy", "")) for r in source_rows}, key=method_sort_key)
            for policy in policies:
                policy_rows = sorted(
                    [r for r in source_rows if str(r.get("diagnostic_policy", "")) == policy],
                    key=lambda r: to_float(r.get("round"), 0.0),
                )
                pairs = [
                    (to_float(r.get("round")), to_float(r.get(metric)))
                    for r in policy_rows
                    if math.isfinite(to_float(r.get("round"))) and math.isfinite(to_float(r.get(metric)))
                ]
                if not pairs:
                    continue
                label = f"{source}: {policy}"
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=label)
                plotted = True
        if plotted:
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel("round")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"simulator_diagnostic_curve_{metric}.png", dpi=220)
        plt.close(fig)


def plot_step_diagnostics(step_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not step_rows:
        return
    metrics = [
        ("step_net_reward", "True-DGP step net reward"),
        ("step_response_score", "True-DGP step response score"),
        ("step_spend", "True-DGP step subsidy spend"),
        ("remaining_budget_after", "True-DGP remaining budget after step"),
        ("action_mean", "Mean city-level action"),
        ("action_max", "Max city-level action"),
        ("action_nonzero_share", "Share of cities with positive subsidy"),
    ]
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        methods = sorted({str(r.get("method", "")) for r in step_rows}, key=method_sort_key)
        for method in methods:
            rows = [r for r in step_rows if str(r.get("method", "")) == method]
            by_t: Dict[float, List[float]] = defaultdict(list)
            for row in rows:
                t = to_float(row.get("t"))
                value = to_float(row.get(metric))
                if math.isfinite(t) and math.isfinite(value):
                    by_t[t].append(value)
            pairs = [(t, float(np.mean(vals))) for t, vals in sorted(by_t.items()) if vals]
            if not pairs:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=method)
            plotted = True
        if plotted:
            ax.set_title(title)
            ax.set_xlabel("time interval")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"step_diagnostic_curve_{metric}.png", dpi=220)
        plt.close(fig)


def plot_interval_simulator_r2_heatmaps(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    sources = sorted({str(r.get("source_method", "")) for r in rows if str(r.get("source_method", ""))})
    for source in sources:
        source_rows = [r for r in rows if str(r.get("source_method", "")) == source]
        rounds = sorted({int(to_float(r.get("round"))) for r in source_rows if math.isfinite(to_float(r.get("round")))})
        times = sorted({int(to_float(r.get("t"))) for r in source_rows if math.isfinite(to_float(r.get("t")))})
        if not rounds or not times:
            continue
        mat = np.full((len(times), len(rounds)), np.nan, dtype=np.float64)
        for i, t in enumerate(times):
            for j, q in enumerate(rounds):
                vals = [
                    to_float(r.get("holdout_r2_z", r.get("r2_z")))
                    for r in source_rows
                    if int(to_float(r.get("t"), -1)) == t and int(to_float(r.get("round"), -1)) == q
                ]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    mat[i, j] = float(np.mean(vals))
        if np.isfinite(mat).any():
            fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(rounds) + 3), max(4, 0.35 * len(times) + 2)))
            im = ax.imshow(mat, aspect="auto", vmin=np.nanmin(mat), vmax=np.nanmax(mat), cmap="viridis")
            ax.set_title(f"Interval holdout R2_z: {source}")
            ax.set_xlabel("round")
            ax.set_ylabel("time interval")
            ax.set_xticks(range(len(rounds)), labels=[str(x) for x in rounds])
            ax.set_yticks(range(len(times)), labels=[str(x) for x in times])
            fig.colorbar(im, ax=ax, label="holdout_r2_z")
            fig.tight_layout()
            fig.savefig(out_dir / f"interval_holdout_r2_z_heatmap_{safe_filename(source)}.png", dpi=220)
            plt.close(fig)


def plot_policy_support_time_curves(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    metrics = [
        ("policy_step_spend_over_B_mean", "Policy step spend / B"),
        ("policy_action_hhi_mean", "Policy action concentration HHI"),
        ("policy_action_entropy_mean", "Policy action entropy"),
        ("extrapolation_risk_share", "Action support extrapolation risk"),
    ]
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for source in sorted({str(r.get("source_method", "")) for r in rows}, key=method_sort_key):
            source_rows = [r for r in rows if str(r.get("source_method", "")) == source]
            by_t: Dict[float, List[float]] = defaultdict(list)
            for row in source_rows:
                t = to_float(row.get("t"))
                value = to_float(row.get(metric))
                if math.isfinite(t) and math.isfinite(value):
                    by_t[t].append(value)
            pairs = [(t, float(np.mean(vals))) for t, vals in sorted(by_t.items()) if vals]
            if not pairs:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=source)
            plotted = True
        if plotted:
            ax.set_title(title)
            ax.set_xlabel("time interval")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"policy_support_time_curve_{metric}.png", dpi=220)
        plt.close(fig)


def plot_simulator_bias_by_policy(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    metric = "sim_minus_true_net_reward"
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for policy in sorted({str(r.get("diagnostic_policy", "")) for r in rows}, key=method_sort_key):
        policy_rows = [r for r in rows if str(r.get("diagnostic_policy", "")) == policy]
        by_round: Dict[float, List[float]] = defaultdict(list)
        for row in policy_rows:
            q = to_float(row.get("round"))
            value = to_float(row.get(metric))
            if math.isfinite(q) and math.isfinite(value):
                by_round[q].append(value)
        pairs = [(q, float(np.mean(vals))) for q, vals in sorted(by_round.items()) if vals]
        if not pairs:
            continue
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=policy)
        plotted = True
    if plotted:
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.set_title("Simulator bias by evaluated policy")
        ax.set_xlabel("round")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "simulator_bias_by_policy_round.png", dpi=220)
    plt.close(fig)


def _markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str], max_rows: int = 10) -> str:
    if not rows:
        return "_No rows._"
    shown = list(rows[:max_rows])
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    text_columns = {"method", "method_key", "source_method", "source_method_key", "diagnostic_policy", "evaluated_policy_key"}
    int_columns = {"rank", "rank_by_net_reward", "round", "episode", "eval_episodes", "num_seeds"}
    for row in shown:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if col in text_columns:
                vals.append(value)
            elif col in int_columns:
                try:
                    vals.append(str(int(float(value))))
                except (TypeError, ValueError):
                    vals.append(str(value))
            elif isinstance(value, (int, np.integer)):
                vals.append(str(int(value)))
            else:
                vals.append(_metric_for_report(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def _metric_for_report(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return ""
    if abs(v) >= 1000 or (0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.4f}"


def write_run_report(
    out_dir: Path,
    final_ranking: Sequence[Dict[str, Any]],
    paired_best_op_summary: Sequence[Dict[str, Any]],
    simulator_rows: Sequence[Dict[str, Any]],
    support_rows: Sequence[Dict[str, Any]],
    training_rows: Sequence[Dict[str, Any]],
) -> None:
    best = final_ranking[0] if final_ranking else {}
    policy_sim_rows = [r for r in simulator_rows if str(r.get("diagnostic_policy", "")) == "policy"]
    support_by_method: List[Dict[str, Any]] = []
    for source in sorted({str(r.get("source_method", "")) for r in support_rows}, key=method_sort_key):
        rows = [r for r in support_rows if str(r.get("source_method", "")) == source]
        risks = [to_float(r.get("extrapolation_risk_share")) for r in rows]
        spends = [to_float(r.get("policy_step_spend_over_B_mean")) for r in rows]
        hhis = [to_float(r.get("policy_action_hhi_mean")) for r in rows]
        support_by_method.append(
            {
                "method": source,
                "mean_extrapolation_risk": float(np.nanmean([v for v in risks if math.isfinite(v)])) if any(math.isfinite(v) for v in risks) else "",
                "mean_step_spend_over_B": float(np.nanmean([v for v in spends if math.isfinite(v)])) if any(math.isfinite(v) for v in spends) else "",
                "mean_action_hhi": float(np.nanmean([v for v in hhis if math.isfinite(v)])) if any(math.isfinite(v) for v in hhis) else "",
            }
        )
    training_tail: List[Dict[str, Any]] = []
    for source in sorted({str(r.get("source_method", "")) for r in training_rows}, key=method_sort_key):
        rows = sorted(
            [r for r in training_rows if str(r.get("source_method", "")) == source],
            key=lambda r: (to_float(r.get("round"), 0.0), to_float(r.get("episode"), 0.0), to_float(r.get("step"), 0.0)),
        )
        if rows:
            last = rows[-1]
            training_tail.append(
                {
                    "method": source,
                    "round": last.get("round", ""),
                    "episode": last.get("episode", ""),
                    "synthetic_return": last.get("synthetic_return", ""),
                    "actor_loss": last.get("actor_loss", ""),
                    "q1_loss": last.get("q1_loss", ""),
                    "q2_loss": last.get("q2_loss", ""),
                }
            )

    lines = [
        "# SAC-SA Experiment Run Report",
        "",
        "## Executive Snapshot",
        "",
        f"- Best final true-DGP method: **{best.get('method', 'N/A')}**",
        f"- Best net reward: `{_metric_for_report(best.get('net_reward', ''))}`",
        f"- Best spend: `{_metric_for_report(best.get('spend', ''))}`",
        "",
        "## Final Ranking",
        "",
        _markdown_table(final_ranking, ["rank", "method", "net_reward", "net_reward_se", "response_score", "spend", "final_budget"], max_rows=12),
        "",
        "## Paired Delta vs Best Operational Baseline",
        "",
        _markdown_table(paired_best_op_summary, ["method", "delta_net_reward", "delta_net_reward_se", "delta_response_score", "delta_spend"], max_rows=12),
        "",
        "## Simulator Calibration",
        "",
        _markdown_table(policy_sim_rows, ["source_method", "round", "sim_minus_true_net_reward", "sim_minus_true_response_score", "holdout_r2_z", "holdout_mse_z"], max_rows=16),
        "",
        "## Policy Support Risk",
        "",
        _markdown_table(support_by_method, ["method", "mean_extrapolation_risk", "mean_step_spend_over_B", "mean_action_hhi"], max_rows=12),
        "",
        "## Training Tail",
        "",
        _markdown_table(training_tail, ["method", "round", "episode", "synthetic_return", "actor_loss", "q1_loss", "q2_loss"], max_rows=12),
        "",
        "## Key Plots",
        "",
        "- [Final net reward boxplot](final_boxplot_net_reward.png)",
        "- [Final mean net reward](final_mean_net_reward.png)",
        "- [Paired delta vs best operational](final_mean_delta_net_reward_vs_best_operational.png)",
        "- [Per-round true-DGP net reward](round_eval_curve_net_reward.png)",
        "- [Seed sensitivity final net reward](sensitivity_seed_final_net_reward.png)",
        "- [Budget sensitivity final net reward](sensitivity_budget_final_net_reward.png)",
        "- [Pessimism sensitivity final net reward](sensitivity_lambda_final_net_reward.png)",
        "- [M sensitivity final net reward](sensitivity_M_final_net_reward.png)",
        "- [Simulator bias by policy](simulator_bias_by_policy_round.png)",
        "- [Policy spend over time](policy_support_time_curve_policy_step_spend_over_B_mean.png)",
        "- [Policy action concentration over time](policy_support_time_curve_policy_action_hhi_mean.png)",
        "",
        "## Interpretation Checklist",
        "",
        "- Final ranking should be read from true-DGP evaluation, not from learned-simulator validation diagnostics.",
        "- Simulator calibration is healthy when policy `sim_minus_true_net_reward` is near zero and holdout `R2_z` is stable.",
        "- High extrapolation risk means the learned policy is acting outside logged action support and should be interpreted cautiously.",
        "- `response_score` is a log-scale synthetic market-response index, not original-currency GMV.",
    ]
    (out_dir / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize iterative synthetic results across methods.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--pattern", type=str, default="results_cpu_pilot_*")
    parser.add_argument("--out_dir", type=str, default="results_cpu_pilot_summary")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result_dirs = discover_result_dirs(root, args.pattern)
    if not result_dirs:
        raise SystemExit(f"No result directories found under {root} matching {args.pattern!r}.")

    final_raw: List[Dict[str, Any]] = []
    round_raw: List[Dict[str, Any]] = []
    iterative_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    simulator_diagnostic_rows: List[Dict[str, Any]] = []
    interval_simulator_rows: List[Dict[str, Any]] = []
    simulator_calibration_rows: List[Dict[str, Any]] = []
    policy_support_rows: List[Dict[str, Any]] = []
    round_step_summary_rows: List[Dict[str, Any]] = []
    final_step_summary_rows: List[Dict[str, Any]] = []
    for d in result_dirs:
        final_raw.extend(add_source(read_csv(d / "final_eval_raw.csv"), d))
        round_raw.extend(add_source(read_csv(d / "round_eval_raw.csv"), d))
        iterative_rows.extend(add_source(read_csv(d / "iterative_rounds.csv"), d))
        training_rows.extend(add_source(read_csv(d / "synthetic_training_rows.csv"), d))
        simulator_diagnostic_rows.extend(add_source(read_csv(d / "simulator_diagnostics.csv"), d))
        interval_rows = read_csv(d / "interval_simulator_diagnostics.csv")
        if not interval_rows:
            interval_rows = read_csv(d / "per_time_simulator_diagnostics.csv")
        interval_simulator_rows.extend(add_source(interval_rows, d))
        simulator_calibration_rows.extend(add_source(read_csv(d / "simulator_vs_true_policy_calibration.csv"), d))
        policy_support_rows.extend(add_source(read_csv(d / "policy_support_diagnostics.csv"), d))
        round_step_summary_rows.extend(add_source(read_csv(d / "round_eval_step_summary.csv"), d))
        final_step_summary_rows.extend(add_source(read_csv(d / "final_eval_step_summary.csv"), d))

    add_calibration_aliases(simulator_diagnostic_rows)
    add_calibration_aliases(simulator_calibration_rows)
    round_raw = dedupe_final_raw(round_raw)
    final_raw = dedupe_final_raw(final_raw)
    if not final_raw and round_raw:
        last_round = max(to_float(r.get("round"), 0.0) for r in round_raw)
        final_raw = [r for r in round_raw if to_float(r.get("round"), -1.0) == last_round]
    final_summary = make_summary(final_raw)
    round_summary = make_round_summary(round_raw)
    method_ranking = make_method_ranking(final_summary)
    seed_mean_rows = make_seed_mean_rows(final_raw)
    seed_level_summary = make_seed_level_summary(seed_mean_rows)
    round_seed_mean_rows = make_round_seed_mean_rows(round_raw)
    round_seed_level_summary = make_round_seed_level_summary(round_seed_mean_rows)
    sensitivity_rows = make_sensitivity_rows(seed_mean_rows)
    sensitivity_summary = make_sensitivity_summary(sensitivity_rows)
    round_sensitivity_summary = make_round_sensitivity_summary(round_seed_mean_rows)
    paired_delta_rows = make_paired_deltas(final_raw)
    paired_delta_summary = make_paired_delta_summary(paired_delta_rows)
    paired_delta_best_op_rows = make_paired_deltas_to_best_operational(final_raw)
    paired_delta_best_op_summary = make_paired_delta_summary(paired_delta_best_op_rows)
    round_paired_delta_rows = make_paired_deltas(round_raw)
    round_paired_delta_best_op_rows = make_paired_deltas_to_best_operational(round_raw)
    seed_mean_paired_delta_rows = make_seed_mean_paired_delta_rows(paired_delta_rows)
    seed_level_paired_delta_summary = make_seed_level_paired_delta_summary(seed_mean_paired_delta_rows)

    manifest_rows = []
    for d in result_dirs:
        cfg = {}
        cfg_path = d / "config.json"
        if cfg_path.exists():
            try:
                with cfg_path.open("r") as f:
                    cfg = json.load(f).get("args", {})
            except (OSError, json.JSONDecodeError, TypeError):
                cfg = {}
        item: Dict[str, Any] = {
            "source_dir": d.name,
            "method_key": cfg.get("method", ""),
            "source_seed": cfg.get("seed", ""),
        }
        config_key = json.dumps({k: cfg.get(k) for k in CONFIG_KEY_FIELDS}, sort_keys=True, separators=(",", ":")) if cfg else ""
        comparison_config_key = (
            json.dumps({k: cfg.get(k) for k in COMPARISON_CONFIG_FIELDS}, sort_keys=True, separators=(",", ":"))
            if cfg
            else ""
        )
        item["comparison_config_key"] = comparison_config_key or config_key
        add_config_metadata(item, config_key)
        manifest_rows.append(item)

    write_csv(out_dir / "result_dirs_manifest.csv", manifest_rows)
    write_csv(out_dir / "final_eval_raw_all_methods.csv", final_raw)
    write_csv(out_dir / "final_eval_summary_all_methods.csv", final_summary)
    write_csv(out_dir / "final_eval_setting_summary.csv", final_summary)
    write_csv(out_dir / "final_eval_method_ranking.csv", method_ranking)
    write_csv(out_dir / "final_eval_seed_means.csv", seed_mean_rows)
    write_csv(out_dir / "final_eval_seed_level_summary.csv", seed_level_summary)
    write_csv(out_dir / "sensitivity_final_seed_means.csv", sensitivity_rows)
    write_csv(out_dir / "sensitivity_final_summary.csv", sensitivity_summary)
    write_csv(out_dir / "sensitivity_round_summary.csv", round_sensitivity_summary)
    write_csv(out_dir / "final_eval_paired_delta_vs_no_subsidy.csv", paired_delta_rows)
    write_csv(out_dir / "final_eval_paired_delta_summary.csv", paired_delta_summary)
    write_csv(out_dir / "final_eval_paired_delta_vs_best_operational.csv", paired_delta_best_op_rows)
    write_csv(out_dir / "final_eval_paired_delta_vs_best_operational_summary.csv", paired_delta_best_op_summary)
    write_csv(out_dir / "final_eval_seed_mean_paired_delta_vs_no_subsidy.csv", seed_mean_paired_delta_rows)
    write_csv(out_dir / "final_eval_seed_level_paired_delta_summary.csv", seed_level_paired_delta_summary)
    write_csv(out_dir / "iterative_rounds_all_methods.csv", iterative_rows)
    write_csv(out_dir / "synthetic_training_rows_all_methods.csv", training_rows)
    write_csv(out_dir / "round_eval_raw_all_methods.csv", round_raw)
    write_csv(out_dir / "round_eval_summary_all_methods.csv", round_summary)
    write_csv(out_dir / "round_eval_setting_summary.csv", round_summary)
    write_csv(out_dir / "round_eval_seed_means.csv", round_seed_mean_rows)
    write_csv(out_dir / "round_eval_seed_level_summary.csv", round_seed_level_summary)
    write_csv(out_dir / "round_eval_paired_delta_vs_no_subsidy.csv", round_paired_delta_rows)
    write_csv(out_dir / "round_eval_paired_delta_vs_best_operational.csv", round_paired_delta_best_op_rows)
    write_csv(out_dir / "simulator_diagnostics_all_methods.csv", simulator_diagnostic_rows)
    write_csv(out_dir / "interval_simulator_diagnostics_all_methods.csv", interval_simulator_rows)
    write_csv(out_dir / "simulator_vs_true_policy_calibration_all_methods.csv", simulator_calibration_rows)
    write_csv(out_dir / "policy_support_diagnostics_all_methods.csv", policy_support_rows)
    write_csv(out_dir / "round_eval_step_summary_all_methods.csv", round_step_summary_rows)
    write_csv(out_dir / "final_eval_step_summary_all_methods.csv", final_step_summary_rows)

    plot_boxplots(final_raw, out_dir)
    plot_final_boxplots_by_config(final_raw, out_dir)
    plot_final_boxplots_faceted_by_config(final_raw, out_dir)
    plot_mean_bars(final_summary, out_dir)
    plot_seed_mean_points(seed_mean_rows, seed_level_summary, out_dir)
    plot_sensitivity_final_net_reward(sensitivity_summary, out_dir)
    plot_sensitivity_final_delta(sensitivity_rows, out_dir)
    plot_paired_delta_net_reward(
        paired_delta_rows,
        paired_delta_summary,
        out_dir,
        suffix="vs_no_subsidy",
        reference_label="No Subsidy",
    )
    plot_paired_delta_net_reward(
        paired_delta_best_op_rows,
        paired_delta_best_op_summary,
        out_dir,
        suffix="vs_best_operational",
        reference_label="Best Operational",
    )
    plot_validation_curves(iterative_rows, out_dir)
    plot_round_eval_curves(round_summary, out_dir)
    plot_round_eval_curves_by_config(round_summary, out_dir)
    plot_round_seed_curves(round_seed_mean_rows, out_dir)
    plot_sensitivity_round_curves(round_sensitivity_summary, out_dir)
    plot_sensitivity_round_delta_vs_best_operational(round_sensitivity_summary, out_dir)
    plot_round_delta_curves(
        round_paired_delta_rows,
        out_dir,
        suffix="vs_no_subsidy",
        title="Per-round paired true-DGP delta vs No Subsidy",
    )
    plot_round_delta_curves(
        round_paired_delta_best_op_rows,
        out_dir,
        suffix="vs_best_operational",
        title="Per-round paired true-DGP delta vs best operational",
    )
    plot_simulator_diagnostic_curves(simulator_diagnostic_rows, out_dir)
    plot_training_curves(training_rows, out_dir)
    plot_step_diagnostics(final_step_summary_rows or round_step_summary_rows, out_dir)
    plot_interval_simulator_r2_heatmaps(interval_simulator_rows, out_dir)
    plot_policy_support_time_curves(policy_support_rows, out_dir)
    plot_simulator_bias_by_policy(simulator_diagnostic_rows, out_dir)
    write_run_report(
        out_dir,
        method_ranking,
        paired_delta_best_op_summary,
        simulator_diagnostic_rows,
        policy_support_rows,
        training_rows,
    )

    print(f"Found {len(result_dirs)} result directories.")
    print(f"Saved combined tables and plots to: {out_dir}")


if __name__ == "__main__":
    main()
