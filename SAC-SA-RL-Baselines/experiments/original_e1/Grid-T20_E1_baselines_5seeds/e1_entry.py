"""Run one locked E1 Grid configuration. Prefer the serial run_e1.py launcher."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import hashlib
import numpy as np
import torch
from city_panel_training import ExperimentConfig, SimulatorConfig, PolicyConfig, run_experiment, make_panel_config
from public_city_panel_dgp import _cache_key, _cache_path
from verify_protocol import verify_packaged_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1_config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.e1_config.read_text(encoding="utf-8"))
    manifest = json.loads(Path(__file__).with_name("source_manifest.json").read_text(encoding="utf-8"))
    verify_packaged_inputs(manifest)
    options = json.loads(Path(cfg["e1_options"]).read_text(encoding="utf-8"))
    ec = dict(cfg["experiment"])
    ec["e1_options"] = options
    if str(ec["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is used.")
    random.seed(ec["training_seed"])
    np.random.seed(ec["training_seed"])
    torch.manual_seed(ec["training_seed"])
    if cfg["profile"] == "smoke":
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    os.environ["PUBLIC_CITY_CALIBRATION_CACHE_DIR"] = str(Path(__file__).parent / ".cache/public_city_panel")
    # Cache keys contain the device spelling. Reuse the identical frozen DGP
    # for cuda:0 / cuda:1 / cpu without inadvertently recalibrating the target.
    canonical = Path(__file__).parent / ".cache/public_city_panel/calibration_v8_59e87f571650ad804f9d.pt"
    requested = _cache_path(_cache_key(make_panel_config(ExperimentConfig(**ec))))
    if requested != canonical:
        if requested.exists():
            if hashlib.sha256(requested.read_bytes()).digest() != hashlib.sha256(canonical.read_bytes()).digest():
                raise RuntimeError("The requested device cache differs from the pinned target DGP.")
        else:
            shutil.copy2(canonical, requested)
    run_experiment(cfg["method"], ExperimentConfig(**ec), SimulatorConfig(**cfg["simulator"]),
                   PolicyConfig(**cfg["policy"]), Path(cfg["out_dir"]))


if __name__ == "__main__":
    main()
