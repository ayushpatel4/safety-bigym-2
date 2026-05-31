#!/usr/bin/env python
"""Summarise a P3 (E3.1 cost-signal) or P4 (E3.2 budget) run: per-cell success /
proximity, grouped by the swept variable (cost form, or budget d), seed-averaged.
For a budget sweep it also suggests the d_knee. Pure stdlib.

Reads each cell's `final_metrics.json` (prefers `last_eval` — converged success +
proximity from the same cycle; `eval/`-prefixed keys), falling back to
`best_eval` (unprefixed) or the last `ty==eval` row of `metrics.jsonl` for runs
still in flight.

Usage:
  python scripts/analyze_e3.py --in-dir exp_local/e3_1_cost_signal/<run>
  python scripts/analyze_e3.py --in-dir exp_local/e3_2_cost_budget/<run> --success-floor 0.4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUCC = "success_rate"
PROX = "ep_proximity_violation_rate"


def _cell_metrics(cell: Path) -> Optional[Tuple[Optional[float], Optional[float], str]]:
    """Return (success_rate, proximity, source) for a cell dir, or None if no data."""
    fm = cell / "final_metrics.json"
    if fm.is_file():
        try:
            m = json.loads(fm.read_text())
        except Exception:
            m = {}
        le = m.get("last_eval") or {}
        s = le.get(f"eval/{SUCC}")
        p = le.get(f"eval/{PROX}")
        if s is not None or p is not None:
            return _flt(s), _flt(p), "last_eval"
        be = m.get("best_eval") or {}
        if be.get(SUCC) is not None or be.get(PROX) is not None:
            return _flt(be.get(SUCC)), _flt(be.get(PROX)), "best_eval"
    # In-flight fallback: last eval row of metrics.jsonl.
    mj = cell / "metrics.jsonl"
    if mj.is_file():
        last = None
        for line in mj.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("ty") == "eval":
                last = row
        if last is not None:
            return _flt(last.get(f"eval/{SUCC}")), _flt(last.get(f"eval/{PROX}")), "jsonl"
    return None


def _flt(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _split_cell(name: str) -> Tuple[str, str]:
    """`continuous_seed0` -> (continuous, 0); `d0p01_seed2` -> (d0p01, 2)."""
    if "_seed" in name:
        var, seed = name.rsplit("_seed", 1)
        return var, seed
    return name, "?"


def _d_value(var: str) -> Optional[float]:
    """`d0p01` -> 0.01, else None (not a budget cell)."""
    if not var.startswith("d"):
        return None
    try:
        return float(var[1:].replace("p", "."))
    except ValueError:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True, help="A P3/P4 run dir of cell subdirs.")
    ap.add_argument("--success-floor", type=float, default=0.4,
                    help="Min mean success_rate for a d to be a knee candidate (budget sweeps).")
    args = ap.parse_args(argv)

    # cell -> (var, seed, success, prox, source)
    groups: Dict[str, List[Tuple[str, Optional[float], Optional[float], str]]] = {}
    for cell in sorted(p for p in args.in_dir.iterdir() if p.is_dir()):
        got = _cell_metrics(cell)
        if got is None:
            continue
        s, p, src = got
        var, seed = _split_cell(cell.name)
        groups.setdefault(var, []).append((seed, s, p, src))
    if not groups:
        raise SystemExit(f"No cells with metrics under {args.in_dir} "
                         "(final_metrics.json / metrics.jsonl). Runs not started?")

    is_budget = all(_d_value(v) is not None for v in groups)
    order = sorted(groups, key=_d_value) if is_budget else sorted(groups)

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.fmean(xs) if xs else None

    rows = []
    print(f"{'cell':<14} {'n':>2} {'success(mean)':>13} {'proximity(mean)':>16}  seeds(success/prox)")
    print("-" * 72)
    for var in order:
        cells = groups[var]
        succs = [c[1] for c in cells]
        proxs = [c[2] for c in cells]
        ms, mp = _mean(succs), _mean(proxs)
        rows.append((var, ms, mp))
        per = " ".join(
            f"s{c[0]}:{_fmt(c[1])}/{_fmt(c[2])}" for c in sorted(cells)
        )
        partial = "" if all(c[3] in ("last_eval", "best_eval") for c in cells) else "  [in-flight]"
        print(f"{var:<14} {len(cells):>2} {_fmt(ms):>13} {_fmt(mp):>16}  {per}{partial}")

    if is_budget:
        print()
        cand = [(v, ms, mp) for (v, ms, mp) in rows
                if ms is not None and mp is not None and ms >= args.success_floor]
        if cand:
            knee = min(cand, key=lambda r: r[2])  # lowest proximity among success>=floor
            print(f"Suggested d_knee = {knee[0]} (d={_d_value(knee[0])}): "
                  f"lowest proximity ({_fmt(knee[2])}) with success>={args.success_floor} "
                  f"(={_fmt(knee[1])}).")
        else:
            best = max((r for r in rows if r[1] is not None), key=lambda r: r[1], default=None)
            print(f"No d meets success>={args.success_floor}. Highest-success d = "
                  f"{best[0] if best else 'n/a'}; inspect the table / loosen --success-floor.")
        print("(Confirm against the benchmark_policy eval before locking ROW3.)")
    return 0


def _fmt(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
