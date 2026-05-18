"""Vendored CQN-AS (Coarse-to-fine Q-Network with Action Sequence) agent.

Upstream: https://github.com/younggyoseo/CQN-AS at commit 8cf806e.
Paper: Seo et al. 2024, arXiv:2411.12155.

The reference clone lives at /Users/ayushpatel/Documents/FYP3/CQN-AS/ and is
kept pristine. Vendored modules:

- cqn_utils: action encode/decode + zoom_in (verbatim).
- utils: general training utilities (verbatim).
- replay_buffer: action-sequence replay buffer (verbatim).
- agent: agent + encoder + C2F critic network (import paths converted to relative).

The env_adapter and any cost-critic extensions for Phase 3 are NOT vendored from
upstream and live in sibling modules in this package.
"""
