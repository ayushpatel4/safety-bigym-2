# Headline-numbers checklist (working note, not part of report)

Canonical values — must read identically in abstract / intro / results / conclusion.
Source-of-truth locations in v16 noted per number.

| Number | Meaning | Source of truth |
|---|---|---|
| 0.296 → 0.228 (−22.8%) at 0.76 success | fixed-λ policy proximity reduction, saucepan, 3 seeds | E4.1 headline table |
| 0.85 | baseline success, saucepan | E4.1 headline table |
| 0.146 → 0.048 (−67%) at 0.53 success | unconditional speed-scaler SSM-actual, saucepan | E4.1 headline table |
| 0.85 → 0.44 | composition success cost (both-axis config) | joint-coverage section |
| ≈42% / ≈58% | exogenous vs robot-controllable proximity (frozen-robot sweep) | R0 decomposition |
| −50% (dish, 0.67 succ, −0.10) / −22% (drawers, 0.73, −0.09; near-free −10% at −0.02) | critic-gated speed-scaling SSM reductions | R2 methods table |
| 2.46 → 6.03 m/s (3.77 at R=2.25) | binary veto max-velocity break, chunked policy | E2.4 |
| 0.44 → 0.79 mean vel (2.46 → 4.14 max) | velocity rises when λ binds, dishwasher | E3.3 |
| 61.5% [57.2,65.4] vs 26.5% [16.3,36.7] / 19.2% [12.7,25.3]; operating points 51–63% vs ≈27%; unconditional trigger 44.2% | gate-active fractions at matched R=2.75 | gate-activity figure (F7) |
| success ≥0.60 ∧ SSM-actual ≤0.08 | pre-registered cross-task rule — no row passes | E4.4 table |
| 0.433 / 0.072 vs 0.44 / 0.065 | unconditional control reproduces composition (snapshot caveat) | snapshot caveat box |
| 0.236 oracle vs 0.198 noisy | perception robustness, constrained policy | E3.6 |
| −31% (0.084 → 0.058) | OOD generalisation, gentler coworker_eval | E5.2 |
| CVaR_0.95 min-sep ≈ 0.005 m invariant | exogenous tail unmoved | E5.1 |
| λ = 0.0 / 0.267 / 3.855 across 3 seeds | PID seed instability at d=0.3 | E3.2 |
| WCSAC: dish 0.47 success, drawers 0% every budget | external corroboration (single seed, oracle) | E3.7 table |
| 37,883 transitions, 16.8% unsafe (saucepan SVF); 52,794 (shared dish/drawers SVF) | SVF datasets | method ch + appendix |
| ≈450–500 GPU-h, ≈40 kg CO2e | compute | setup + declarations |
| 87 references, all cited | bibliography | references.bib |

Audit greps (run at every gate):
- errors/undefined: `grep -c '^!' main.log; grep -ci 'undefined' main.log`
- banned codes outside appendix: `grep -nE '\bE[0-9]+\.[0-9]+\b|\bR[0-3]\b|\bC[1-6]\b|Option[~ ][AB]|\bv3op?\b|Phase[~ ][0-9]' main.tex` (hand-review: `$R$`, `R = 2.75`, C51 legitimate)
- stale terms: `grep -nE 'to our knowledge|load-bearing|byte-for-byte|else-branch|working draft|VERIFY' main.tex`
- em-dashes: `grep -c -- '---' main.tex main.bbl` → 0 0
