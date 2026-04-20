#!/usr/bin/env python
"""
Baseline reference: same harness as diagnose_truncation.py but with no human.

If truncations vanish here but persist with the human injected, the human is
confirmed as the source of simulation instability.

Usage:
    python scripts/diagnose_no_human.py --episodes 30 --max-steps 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_truncation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/diagnose/truncation_no_human.csv"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.argv = [
        "diagnose_truncation",
        "--episodes", str(args.episodes),
        "--max-steps", str(args.max_steps),
        "--out", str(args.out),
        "--seed", str(args.seed),
        "--no-human",
    ]
    diagnose_truncation.main()


if __name__ == "__main__":
    main()
