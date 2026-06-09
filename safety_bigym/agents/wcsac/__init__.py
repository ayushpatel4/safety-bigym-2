"""WCSAC (Worst-Case Soft Actor-Critic) external safe-RL baseline.

Faithful reimplementation of WCSAC (Yang, Simao, Tindemans & Spaan,
"WCSAC: Worst-Case Soft Actor Critic for Safety-Constrained Reinforcement
Learning", AAAI 2021) for the 76-DOF humanoid SafetyBiGym tasks. This is the
E3.7 / P9 *external* distributional safe-RL reference, distinct from the
project's value-based B-value-CVaR (E3.5) variant.

It is a standalone actor-critic agent (stochastic squashed-Gaussian actor,
twin reward critics, a Gaussian safety critic with mean + variance heads, a
CVaR_alpha constraint enforced by a learnable Lagrange multiplier, and a
learnable SAC entropy temperature). It plugs into the existing
``train_cqn_as.py`` stack via ``agent=wcsac`` and consumes the same per-step
``cost`` signal that the CQN-AS replay already carries -- so it reuses all of
the cost/env/replay/eval plumbing without touching RoboBase.
"""
