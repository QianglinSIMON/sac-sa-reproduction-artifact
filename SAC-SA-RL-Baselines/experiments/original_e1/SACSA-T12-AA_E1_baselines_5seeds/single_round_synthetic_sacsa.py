#!/usr/bin/env python
"""Single-round synthetic subsidy training/evaluation entrypoint.

This script is intentionally a thin, explicit q=1 entrypoint. It reuses the
maintained simulator fitting, policy training, diagnostics, and true-DGP
evaluation implementation in iterative_synthetic_sacsa.py, but fixes the outer
loop to exactly one round. Use this before the iterative q>=2 pipeline to audit
whether the initial logged data support a useful learned simulator and whether
the one-round trained policies behave sensibly in true-DGP evaluation.
"""

from __future__ import annotations

import sys
from typing import Sequence

from iterative_synthetic_sacsa import main as iterative_main


def _has_cli_option(argv: Sequence[str], option: str) -> bool:
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in argv)


def main() -> None:
    argv = list(sys.argv[1:])
    if _has_cli_option(argv, "--q_max"):
        raise SystemExit("single_round_synthetic_sacsa.py fixes --q_max=1; remove --q_max from this command.")
    if not _has_cli_option(argv, "--out_dir"):
        argv.extend(["--out_dir", "results_single_round_synthetic_sacsa"])

    sys.argv = [sys.argv[0], *argv, "--q_max", "1"]
    iterative_main()


if __name__ == "__main__":
    main()
