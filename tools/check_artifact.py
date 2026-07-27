from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ROOT_FILES = (
    "README.md",
    "DATA_AVAILABILITY.md",
    "THIRD_PARTY_NOTICE.md",
    "requirements.txt",
)

EXPECTED = {
    "SAC-SA-T6_main": ("run_gpu_q3_5090_windows.ps1",),
    "SAC-SA-T12_main": ("run_gpu_q3_5090_windows.ps1",),
    "Grid_public_simulator_main": (
        "run_public_grid_main_5090.ps1",
        "run_public_city_panel_gpu_q3_5090_windows.ps1",
    ),
    "SAC-SA-T6_main_factor-multisetting": (
        "run_gpu_q3_sweep_d0_windows.ps1",
        "run_gpu_q3_sweep_newdays_windows.ps1",
        "run_gpu_q3_sweep_m_windows.ps1",
        "run_gpu_q3_sweep_seed_windows.ps1",
        "make_synthetic_t6_sensitivity_overview.py",
    ),
    "Grid_public_simulator_factor-multisetting": (
        "run_gpu_q3_sweep_budget_windows.ps1",
        "run_gpu_q3_sweep_d0_windows.ps1",
        "run_gpu_q3_sweep_newdays_windows.ps1",
        "run_gpu_q3_sweep_m_windows.ps1",
        "run_gpu_q3_sweep_seed_windows.ps1",
        "make_public_city_sensitivity_overview.py",
    ),
    "SAC-SA-T6_main_pessi_replay": (
        "run_comp_nopess_5090.ps1",
        "run_comp_lp005_5090.ps1",
        "run_comp_lp02_5090.ps1",
        "run_comp_lp05_5090.ps1",
        "run_comp_lp10_5090.ps1",
        "make_synthetic_t6_pessimism_sweep.py",
    ),
    "SAC-SA-T6_main_train_seeds": (
        "run_gpu_q3_fixed_dgp_training_seeds_windows.ps1",
        "synthetic_t6_fixed_dgp_protocol.json",
    ),
    "SAC-SA-T6_main_train_seeds_td3": (
        "run_gpu_q3_td3_seeds.ps1",
        "synthetic_t6_td3_protocol.json",
    ),
    "SAC-SA-T6_main_actor_compare": (
        "run_gpu_q3_actor_comparison_windows.ps1",
        "check_actor_parameterizations.py",
        "make_actor_comparison_figures.py",
    ),
}

CORE_EQUALITY_GROUPS = {
    "synthetic T=6 DGP": tuple(
        ROOT / directory / "DGP_sim_env.py"
        for directory in (
            "SAC-SA-T6_main",
            "SAC-SA-T6_main_factor-multisetting",
            "SAC-SA-T6_main_pessi_replay",
            "SAC-SA-T6_main_train_seeds",
            "SAC-SA-T6_main_train_seeds_td3",
            "SAC-SA-T6_main_actor_compare",
        )
    ),
    "synthetic main and one-factor RL components": tuple(
        ROOT / directory / "sacsa_rl_components.py"
        for directory in (
            "SAC-SA-T6_main",
            "SAC-SA-T6_main_factor-multisetting",
        )
    ),
    "synthetic main and one-factor training loop": tuple(
        ROOT / directory / "iterative_synthetic_sacsa.py"
        for directory in (
            "SAC-SA-T6_main",
            "SAC-SA-T6_main_factor-multisetting",
        )
    ),
    "public-grid microscopic environment": tuple(
        ROOT / directory / "Env-DGP.py"
        for directory in (
            "Grid_public_simulator_main",
            "Grid_public_simulator_factor-multisetting",
        )
    ),
    "public-grid model definitions": tuple(
        ROOT / directory / "city_panel_models.py"
        for directory in (
            "Grid_public_simulator_main",
            "Grid_public_simulator_factor-multisetting",
        )
    ),
    "public-grid command-line entry": tuple(
        ROOT / directory / "train_public_city_panel.py"
        for directory in (
            "Grid_public_simulator_main",
            "Grid_public_simulator_factor-multisetting",
        )
    ),
}

PROTOCOL_EXPECTATIONS = {
    "SAC-SA-T6_main_train_seeds/synthetic_t6_fixed_dgp_protocol.json": {
        "methods": ["sacsa_reserve", "dfl_pto"],
        "training_seeds": [2026, 2027, 2028, 2029, 2030],
        "fixed_seeds": {
            "dgp_seed": 2026,
            "data_seed": 2026,
            "evaluation_seed": 2026,
        },
    },
    "SAC-SA-T6_main_train_seeds_td3/synthetic_t6_td3_protocol.json": {
        "methods": ["td3_direct_softmax", "td3_projection"],
        "training_seeds": [2026, 2027, 2028, 2029, 2030],
        "fixed_seeds": {
            "dgp_seed": 2026,
            "data_seed": 2026,
            "evaluation_seed": 2026,
        },
    },
}

REQUIRED_TEXT = {
    "SAC-SA-T6_main_actor_compare/run_gpu_q3_actor_comparison_windows.ps1": (
        'Method = "sacsa_reserve"',
        'Method = "sac_logistic_normal_reserve"',
        'Method = "sacsa"',
        'Method = "sac_logistic_normal"',
        'Method = "sac_gaussian_projection"',
    ),
    "SAC-SA-T6_main_pessi_replay/run_gpu_q3_component_ablation_windows.ps1": (
        '@("full", "nopess", "noreplay")',
        '$env:INITIAL_LOGGED_DAYS = "800"',
        '$env:NEW_LOGGED_DAYS = "100"',
        '$env:EVAL_EPISODES = "2000"',
    ),
}

FORBIDDEN_DIR_PREFIXES = (
    "logs_",
    "results_",
    "summary_",
    "summary_input_",
    "paper_figures_",
)
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".npz",
}
FORBIDDEN_NAMES = {
    ".DS_Store",
    ".env",
}
FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".matplotlib",
    ".pytest_cache",
}
FORBIDDEN_PRIVATE_FRAGMENTS = {
    "realAAdata",
    "AA_T6",
    "AA_T12",
}
PERSONAL_PATTERNS = {
    "Windows user path": re.compile(r"C:\\Users\\", re.IGNORECASE),
    "local Miniconda path": re.compile(r"B:\\programs\\", re.IGNORECASE),
    "macOS user path": re.compile(r"/Users/[^/\s]+/", re.IGNORECASE),
}
INTERNAL_TRACE_PATTERNS = {
    "legacy directory typo": re.compile(r"_mian\b", re.IGNORECASE),
    "exploratory public-grid tag": re.compile(
        r"\b(?:D[1-5](?:ab\w*|_simsel)?|cfg[A-E]|config[_ -]?[A-E]|try[_ -]?\d+)\b",
        re.IGNORECASE,
    ),
    "result-selection wording": re.compile(
        r"\bselected[_ -](?:ablation|configuration|setting)\b", re.IGNORECASE
    ),
}
PACKAGE_EXTERNAL_PATTERNS = {
    "external paper-figure source path": re.compile(r"\bfigs-sac-sa\b"),
    "identifying upstream project name": re.compile(r"\bMDPOD\b", re.IGNORECASE),
    "identifying upstream account": re.compile(r"\bcallmespring\b", re.IGNORECASE),
    "identifying upstream repository URL": re.compile(
        r"github\.com/callmespring", re.IGNORECASE
    ),
}
REQUIRED_DEPENDENCIES = {
    "numpy",
    "pandas",
    "bottleneck",
    "matplotlib",
    "torch",
    "tqdm",
}


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []

    requirements_path = ROOT / "requirements.txt"
    if not requirements_path.is_file():
        errors.append("missing requirements.txt")
    else:
        requirement_names = {
            re.split(r"[<>=!~\[]", line.strip(), maxsplit=1)[0].lower()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        missing_dependencies = REQUIRED_DEPENDENCIES - requirement_names
        if missing_dependencies:
            errors.append(
                "requirements.txt is missing: "
                + ", ".join(sorted(missing_dependencies))
            )

    for directory, entry_points in EXPECTED.items():
        path = ROOT / directory
        if not path.is_dir():
            errors.append(f"missing directory: {directory}")
        for entry_point in entry_points:
            if not (path / entry_point).is_file():
                errors.append(f"missing entry point: {directory}/{entry_point}")

    for relative in REQUIRED_ROOT_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing root file: {relative}")

    for label, paths in CORE_EQUALITY_GROUPS.items():
        missing = [path for path in paths if not path.is_file()]
        if missing:
            errors.append(
                f"{label}: missing files: "
                + ", ".join(str(path.relative_to(ROOT)) for path in missing)
            )
            continue
        digests = {file_digest(path) for path in paths}
        if len(digests) != 1:
            errors.append(f"{label}: files expected to match have drifted")

    for relative, expected in PROTOCOL_EXPECTATIONS.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing protocol: {relative}")
            continue
        protocol = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if protocol.get(key) != value:
                errors.append(
                    f"{relative}: {key}={protocol.get(key)!r}, expected {value!r}"
                )

    for relative, fragments in REQUIRED_TEXT.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing protocol script: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in text:
                errors.append(f"{relative}: missing locked fragment {fragment!r}")

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        relative_text = relative.as_posix()
        if path.is_symlink():
            errors.append(f"symbolic link committed: {relative}")
            continue
        if any(fragment.lower() in relative_text.lower() for fragment in FORBIDDEN_PRIVATE_FRAGMENTS):
            errors.append(f"private experiment path committed: {relative}")
        if path.is_dir() and path.name in FORBIDDEN_DIR_NAMES:
            errors.append(f"local/generated directory committed: {relative}")
            continue
        if path.is_dir() and path.name.startswith(FORBIDDEN_DIR_PREFIXES):
            errors.append(f"generated directory committed: {relative}")
            continue
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() == ".lock":
            errors.append(f"local/generated file committed: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"model/generated artifact committed: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file exceeds 10 MiB: {relative}")
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"invalid JSON {relative}: {exc}")
        if path.suffix == ".py":
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"Python compile failure {relative}: {exc}")
        if (
            path.resolve() != Path(__file__).resolve()
            and path.suffix.lower() in {".py", ".ps1", ".md", ".json", ".txt", ".csv"}
        ):
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in PERSONAL_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label} in {relative}")
            for label, pattern in INTERNAL_TRACE_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label} in {relative}")
            for label, pattern in PACKAGE_EXTERNAL_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label} in {relative}")

    if errors:
        print("Artifact check failed:")
        for error in sorted(set(errors)):
            print(f"  - {error}")
        return 1

    print(
        "Artifact check passed: "
        f"{len(EXPECTED)} experiment modules and all declared entry points verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
