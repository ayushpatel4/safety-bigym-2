#!/usr/bin/env python3
"""Safe snapshot cleanup — free disk by deleting intermediate snapshot_<step>.pt
files, with hard guarantees it can NEVER delete a canonical / referenced one.

Why this exists: on 2026-06-08 an over-broad
    find exp_local -name 'snapshot_[0-9]*.pt' ! -name snapshot_best.pt -delete
(run manually during a disk-full crisis, and as a 10-min `nohup` janitor loop)
wiped registry-referenced saucepan baselines (snapshot_2588 / snapshot_28203).
NEVER use a bare `find ... snapshot ... -delete` again. Use this instead.

Three independent guards — a snapshot is deleted only if ALL hold:
  1. SCOPE  — it lives under the campaign ALLOWLIST (exp_local/{isolation,
     curriculum,safety}). Pre-existing experiment dirs are never touched.
  2. UNREFERENCED — its path is not referenced by filters/snapshots.py or any
     *.sh/*.py/*.yaml/*.json under the repo (canonical baselines/warm-starts).
  3. NOT-BEST / NOT-RECENT — it is not snapshot_best.pt and not among the
     --keep-recent newest snapshots in its run dir (resume/basin inputs).

DRY-RUN by default. Pass --apply to actually delete.

  venv/bin/python scripts/safe_snapshot_cleanup.py                 # preview
  venv/bin/python scripts/safe_snapshot_cleanup.py --keep-recent 4 --apply
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Hard allowlist — the ONLY directories this tool will ever delete from.
ALLOWLIST = [(REPO / "exp_local" / d).resolve() for d in ("isolation", "curriculum", "safety")]
STEP_RE = re.compile(r"^snapshot_(\d+)\.pt$")


def referenced_snapshots() -> set[Path]:
    """Absolute paths of any snapshot_<step>.pt referenced anywhere in code/config."""
    out: set[Path] = set()
    try:
        r = subprocess.run(
            ["grep", "-rhoE", r"exp_local/[^ '\"]*snapshot_[0-9]+\.pt",
             "--include=*.sh", "--include=*.py", "--include=*.yaml", "--include=*.json",
             str(REPO)],
            capture_output=True, text=True)
        for line in r.stdout.splitlines():
            out.add((REPO / line.strip()).resolve())
    except Exception:
        pass
    return out


def in_allowlist(p: Path) -> bool:
    rp = p.resolve()
    return any(str(rp).startswith(str(a) + "/") for a in ALLOWLIST)


def main() -> int:
    ap = argparse.ArgumentParser(description="Guarded intermediate-snapshot cleanup.")
    ap.add_argument("--keep-recent", type=int, default=2,
                    help="keep the N most-recent snapshot_<step>.pt per run dir (default 2)")
    ap.add_argument("--apply", action="store_true", help="delete (default: dry-run preview)")
    args = ap.parse_args()

    protected = referenced_snapshots()
    print(f"[safe-cleanup] mode={'APPLY' if args.apply else 'DRY-RUN'}  keep-recent={args.keep_recent}")
    print(f"[safe-cleanup] allowlist: {[str(a) for a in ALLOWLIST]}")
    print(f"[safe-cleanup] {len(protected)} code-referenced snapshots protected\n")

    total = freed = 0
    for root in ALLOWLIST:
        if not root.exists():
            continue
        runs: dict[Path, list[Path]] = {}
        for p in root.rglob("snapshot_*.pt"):
            if STEP_RE.match(p.name):          # excludes snapshot_best.pt
                runs.setdefault(p.parent, []).append(p)
        for rundir, snaps in runs.items():
            snaps.sort(key=lambda p: int(STEP_RE.match(p.name).group(1)))
            keep = set(snaps[-args.keep_recent:]) if args.keep_recent > 0 else set()
            for p in snaps:
                if p in keep or p.resolve() in protected or not in_allowlist(p):
                    continue
                sz = p.stat().st_size
                total += 1
                freed += sz
                if args.apply:
                    p.unlink()
                elif total <= 40:
                    print(f"  {'delete' if args.apply else 'would delete'}: {p}  ({sz/1e6:.0f} MB)")
    print(f"\n[safe-cleanup] {'deleted' if args.apply else 'would delete'} "
          f"{total} files, {freed/1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
