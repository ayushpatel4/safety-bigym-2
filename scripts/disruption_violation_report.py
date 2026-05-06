#!/usr/bin/env python
"""Sampler-side report: aggressive-disruption distribution check.

Samples N scenarios from the default ParameterSpace and reports the
empirical disruption-type mix and key trajectory parameter histograms.
Used to confirm that scenario_sampler tuning lands the expected
distribution before spinning up MuJoCo for an end-to-end violation test.

Usage:
    python scripts/disruption_violation_report.py
    python scripts/disruption_violation_report.py --n 5000

The full env-stepping violation report (which actually rolls out
EpisodeSafetyMetrics episodes per disruption type and reports
ssm_violation_rate / pfl_violation_rate) is the human's hand-off
task — see plan file for the criteria.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from safety_bigym.scenarios import (
    DisruptionType,
    ParameterSpace,
    ScenarioSampler,
)


def _hist_line(name: str, values: list[float], buckets: list[float]) -> str:
    arr = np.asarray(values)
    counts = []
    edges = [-np.inf, *buckets, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        counts.append(int(np.sum((arr >= lo) & (arr < hi))))
    parts = [f"{name:>22}:"]
    for hi, c in zip(edges[1:], counts):
        if hi == np.inf:
            parts.append(f"  >={edges[-2]:.2f}: {c:4d}")
        else:
            parts.append(f"  <{hi:.2f}: {c:4d}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000, help="Number of scenarios to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    args = parser.parse_args()

    sampler = ScenarioSampler(
        parameter_space=ParameterSpace(clip_paths=["dummy.npz"]),
    )

    type_counts: Counter[DisruptionType] = Counter()
    closest_by_type: dict[DisruptionType, list[float]] = {d: [] for d in DisruptionType}
    loiter_by_type: dict[DisruptionType, list[float]] = {d: [] for d in DisruptionType}
    embed_values: list[float] = []
    closest_all: list[float] = []
    loiter_all: list[float] = []

    for i in range(args.n):
        s = sampler.sample_scenario(args.seed + i)
        type_counts[s.disruption_type] += 1
        closest_by_type[s.disruption_type].append(s.closest_approach)
        loiter_by_type[s.disruption_type].append(s.loiter_duration)
        closest_all.append(s.closest_approach)
        loiter_all.append(s.loiter_duration)
        if s.disruption_type == DisruptionType.CONTACT:
            embed_values.append(s.disruption_config.embed_distance)

    print("=" * 70)
    print(f"Sampled {args.n} scenarios from ParameterSpace defaults")
    print("=" * 70)

    print("\nDisruption type mix (target ~60% violation-producing CONTACT+OBSTRUCTION+DIRECT):")
    for dtype in DisruptionType:
        n = type_counts[dtype]
        pct = 100.0 * n / args.n
        ca = closest_by_type[dtype]
        ld = loiter_by_type[dtype]
        ca_mean = float(np.mean(ca)) if ca else 0.0
        ld_mean = float(np.mean(ld)) if ld else 0.0
        print(
            f"  {dtype.name:<18} {n:>5} ({pct:5.1f}%)  "
            f"mean closest={ca_mean:.2f}m  mean loiter={ld_mean:.2f}s"
        )

    print()
    print(_hist_line("closest_approach (m)", closest_all, [0.05, 0.2, 0.5, 0.8]))
    print(_hist_line("loiter_duration (s)", loiter_all, [3.0, 6.0, 10.0, 30.0]))
    if embed_values:
        print(_hist_line("CONTACT embed (m)", embed_values, [0.01, 0.025, 0.05]))

    pct_close = 100.0 * sum(1 for c in closest_all if c < 0.3) / args.n
    pct_long_loiter = 100.0 * sum(1 for d in loiter_all if d >= 6.0) / args.n
    print()
    print(f"Episodes with closest_approach < 0.3m: {pct_close:.1f}%  (target ≥ 30%)")
    print(f"Episodes with loiter_duration ≥ 6s:    {pct_long_loiter:.1f}%  (target ≥ 40%)")


if __name__ == "__main__":
    main()
