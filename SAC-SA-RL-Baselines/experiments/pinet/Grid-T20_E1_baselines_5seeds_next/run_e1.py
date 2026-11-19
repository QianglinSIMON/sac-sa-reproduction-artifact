"""Serial E1 runner. The default is smoke; formal training requires --profile formal."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import platform
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from learning_rate_plan import read_learning_rate_plan, apply_learning_rate_factor

from e1_methods import METHODS, METHOD_VARIANTS, VARIANT_PARENTS
SEEDS = [2026, 2027, 2028, 2029, 2030]
ROOT = Path(__file__).resolve().parent


def effective_config(method, seed, profile, device):
    cfg = json.loads((ROOT / "configs" / f"{method}.json").read_text(encoding="utf-8"))
    cfg["experiment"].update(seed=seed, training_seed=seed, device=device)
    if profile == "smoke":
        cfg["experiment"].update(initial_logged_days=8, new_logged_days=2, q_max=3,
            residual_calibration_days=8, validation_logged_days=8, holdout_logged_days=8,
            eval_episodes=2, logging_batch_size=4)
        # Use the unchanged full frozen target DGP, already supplied as a cache.
        # A one-epoch diagnostic model cannot satisfy the formal quality gate.
        cfg["simulator"].update(hidden_dims=[32, 16], head_dim=16, epochs=1,
            finetune_epochs=1, batch_size=64, patience=1, bootstrap_models=2,
            bootstrap_epochs=1, strict_quality_gate=False)
        cfg["policy"].update(hidden_dims=[32, 32], train_episodes=4,
            later_round_train_episodes=4, rollout_batch_size=2, updates_per_rollout_batch=2,
            warmup_episodes=2, batch_size=8, replay_size=512, real_gradient_steps_per_step=.1,
            real_critic_warmup_updates=1, support_prototype_count=4, simulator_selection_episodes=2,
            dfl_epochs=1, dfl_batch_size=8, dfl_candidate_random_count=4, dfl_hidden_dims=[32, 16])
    cfg["profile"] = profile
    return cfg


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def runtime_environment(device, methods, dry_run):
    """Record the environment and fail before training if a required backend fails."""
    import torch

    packages = {}
    for name in ["torch", "numpy", "scipy", "pandas"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    target = torch.device(device)
    uses_cuda = target.type == "cuda"
    available = torch.cuda.is_available()
    report = {"python": sys.version, "platform": platform.platform(), "packages": packages,
              "cuda_runtime": torch.version.cuda, "cuda_available": available,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu": torch.cuda.get_device_name(target) if uses_cuda and available else None,
              "preflight": "not_run_dry_run" if dry_run else "running"}
    if dry_run:
        return report
    if uses_cuda and not available:
        raise RuntimeError("CUDA was requested but is unavailable; install a compatible PyTorch build.")
    x = torch.ones((16, 16), device=target, requires_grad=True)
    (x @ x).square().mean().backward()
    if not torch.isfinite(x.grad).all():
        raise FloatingPointError("Device forward/backward preflight failed.")
    if uses_cuda:
        torch.cuda.synchronize(target)
        report["gpu_capability"] = list(torch.cuda.get_device_capability(target))
        report["torch_arch_list"] = torch.cuda.get_arch_list()
    if "td3_pinet" in methods:
        from e1_baselines import PiNetActor
        # Test the actual PiNet forward and implicit backward on the requested device.
        actor = PiNetActor(56, 3, (8, 8)).to(target)
        states = torch.zeros(2, 56, device=target)
        states[:, -5] = 1.0
        proposal = torch.tensor([[.8, .6, -.2], [.3, .2, .1]], device=target, requires_grad=True)
        action = actor.action_from_fraction(states, proposal)
        action.square().sum().backward()
        if not torch.isfinite(proposal.grad).all() or not proposal.grad.abs().sum() > 0:
            raise FloatingPointError("PiNet solver/autograd preflight failed.")
        report["pinet_solver_autograd"] = "PASS"
    report["preflight"] = "PASS"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--learning-rate-plan", type=Path, default=None,
                        help="Opt-in local development candidates; original formal defaults remain unchanged.")
    args = parser.parse_args()
    learning_rate_plan = read_learning_rate_plan(args.learning_rate_plan, ROOT.name)
    if len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be unique; duplicate jobs would overwrite their outputs.")
    from verify_protocol import verify
    protocol_report = verify()
    seeds = args.seeds or ([2026] if args.profile == "smoke" else SEEDS)
    if len(set(seeds)) != len(seeds) or any(seed not in SEEDS for seed in seeds):
        parser.error("Seeds must be unique members of 2026..2030.")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output_root or ROOT / f"runs_{args.profile}_{stamp}").resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite a run directory: {out}")
    out.mkdir(parents=True)
    write_json(out / "protocol_verification.json", protocol_report)
    options = json.loads((ROOT / "baseline_settings.json").read_text(encoding="utf-8"))
    if args.profile == "smoke":
        options["mopo"].update(ensemble_size=2, elite_size=1, model_epochs=1,
                               model_batch_size=16, rollout_batch_size=4,
                               rollout_length=2, rollout_every_updates=2,
                               model_buffer_size=128, real_ratio=0.25)
    write_json(out / "baseline_settings.json", options)
    runtime = runtime_environment(args.device, args.methods, args.dry_run)
    write_json(out / "runtime_environment.json", runtime)
    code_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted([*ROOT.glob("*.py"), *ROOT.glob("configs/*.json"), ROOT / "baseline_settings.json"])}
    write_json(out / "code_and_config_hashes.json", code_hashes)

    manifest = {"profile": args.profile, "scenario": ROOT.name, "seeds": seeds,
                "expected_formal_seeds": SEEDS, "methods": args.methods,
                "python": sys.executable, "device": args.device,
                "dry_run": args.dry_run, "old_results_reused": False, "variant": "E1-next-final-20261001",
                "training_budget": "full parent E1 schedules; no reductions",
                "bd_gate_parameterization": "squareplus for SAC-SA/CQL/both MOPO variants",
                "method_variants": METHOD_VARIANTS,
                "source_manifest_sha256": hashlib.sha256((ROOT / "source_manifest.json").read_bytes()).hexdigest(),
                "jobs": []}
    if learning_rate_plan:
        write_json(out / "learning_rate_plan.json", learning_rate_plan)
        manifest.update(variant="E1-next-final-20261001+development-lr",
            learning_rate_plan_sha256=hashlib.sha256(args.learning_rate_plan.read_bytes()).hexdigest(),
            learning_rate_factors=learning_rate_plan["factors"],
            learning_rate_evidence="Local development only; full-scale superiority unverified.")
    for method in args.methods:
        for seed in seeds:
            run_dir = out / f"{method}_seed{seed}"
            cfg = effective_config(method, seed, args.profile, args.device)
            rate_method = VARIANT_PARENTS.get(method, method)
            if learning_rate_plan and rate_method in learning_rate_plan["factors"]:
                cfg = apply_learning_rate_factor(cfg, learning_rate_plan["factors"][rate_method])
            cfg["out_dir"] = str(run_dir)
            cfg["e1_options"] = str(out / "baseline_settings.json")
            cfg_file = out / f"config_{method}_seed{seed}.json"
            write_json(cfg_file, cfg)
            command = [sys.executable, str(ROOT / "e1_entry.py"), "--e1_config", str(cfg_file)]
            job = {"method": method, "training_seed": seed, "output": str(run_dir),
                   "config": str(cfg_file), "command": command, "status": "planned"}
            manifest["jobs"].append(job)
    write_json(out / "run_manifest.json", manifest)
    if args.dry_run:
        print(f"Planned {len(manifest['jobs'])} jobs: {out}")
        return
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    for job in manifest["jobs"]:
        start = time.perf_counter()
        job["status"] = "running"
        write_json(out / "run_manifest.json", manifest)
        print(f"START {job['method']} seed={job['training_seed']} ({args.profile})", flush=True)
        log = out / f"{job['method']}_seed{job['training_seed']}.log"
        job["log"] = str(log)
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.run(job["command"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        job.update(status="completed" if proc.returncode == 0 else "failed",
                   returncode=proc.returncode, elapsed_seconds=time.perf_counter() - start)
        write_json(out / "run_manifest.json", manifest)
        print(f"{job['status'].upper()} {job['method']} {job['elapsed_seconds']:.1f}s; log={log}", flush=True)
        if proc.returncode:
            raise SystemExit(proc.returncode)
    subprocess.run([sys.executable, str(ROOT / "summarize_e1.py"), str(out)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
