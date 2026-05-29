"""Snapshot -> (PolicyMeta, payload) dispatch.

Detects the checkpoint kind from its top-level payload keys:

  * ``{agent_state, config}``  -> CQN-AS   (train_cqn_as.save_snapshot)
  * ``{agent, cfg}``           -> ACT/RoboBase (svf_collect.load_snapshot_policy)
  * ``snapshot_path is None``  -> random policy
  * anything else              -> loud ValueError listing the observed keys

Returns the (already-loaded) payload alongside the meta so the runner builder
(:func:`safety_bigym.benchmark.runners.build_cell_runner`) doesn't re-read it. Env
construction lives in the runner builder, not here, because the CQN-AS path builds its
own adapter from the snapshot's embedded config (not the gym builder).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

__all__ = ["PolicyMeta", "load_policy", "detect_kind"]


def _import_collector():
    """Lazy import of scripts/svf_collect_dataset.py (the ACT loader's home)."""
    import sys

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import svf_collect_dataset as collector  # type: ignore

    return collector


def _act_policy_from_snapshot(snapshot_path, env):
    """Build the ACT _SnapshotPolicy callable (reuses the tested collector loader)."""
    return _import_collector().load_snapshot_policy(snapshot_path, env)


@dataclass
class PolicyMeta:
    kind: str  # "random" | "act" | "cqn_as"
    bodyslam_mode: str = "off"  # peeked from snapshot cfg (informational)
    cameras: Tuple[str, ...] = ()
    camera_resolution: Tuple[int, int] = (84, 84)
    pixels: bool = False
    raw_payload_keys: Tuple[str, ...] = field(default_factory=tuple)


def detect_kind(payload_keys) -> str:
    """Pure kind detection from a payload's top-level key set (unit-testable)."""
    keys = set(payload_keys)
    if {"agent_state", "config"} <= keys:
        return "cqn_as"
    if {"agent", "cfg"} <= keys:
        return "act"
    raise ValueError(
        f"Unrecognized snapshot: top-level keys {sorted(keys)} match neither CQN-AS "
        f"({{agent_state, config}}) nor ACT ({{agent, cfg}})."
    )


def load_policy(snapshot_path: Optional[Path]) -> Tuple[PolicyMeta, Any]:
    """Return ``(PolicyMeta, payload)`` for a snapshot (or random when ``None``).

    ``payload`` is the loaded checkpoint dict (``None`` for random). The runner builder
    consumes ``meta`` + ``payload`` to construct the env + policy.
    """
    if snapshot_path is None:
        return PolicyMeta(kind="random"), None

    import torch

    payload = torch.load(str(snapshot_path), map_location="cpu", weights_only=False)
    keys = tuple(payload.keys()) if isinstance(payload, dict) else ()
    kind = detect_kind(keys)

    if kind == "act":
        # Reuse the ACT snapshot peekers from the collector (lazy script import).
        import sys

        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import svf_collect_dataset as collector  # type: ignore

        bodyslam_mode = collector.peek_snapshot_bodyslam_mode(snapshot_path)
        cameras, resolution = collector.peek_snapshot_cameras(snapshot_path)
        return (
            PolicyMeta(
                kind="act",
                bodyslam_mode=bodyslam_mode,
                cameras=tuple(cameras),
                camera_resolution=tuple(resolution),
                pixels=bool(cameras),
                raw_payload_keys=keys,
            ),
            payload,
        )

    # kind == "cqn_as"
    config = payload.get("config", {})
    env_cfg = (config.get("env", {}) if isinstance(config, dict) else {}) or {}
    bodyslam = (env_cfg.get("bodyslam", {}) if isinstance(env_cfg, dict) else {}) or {}
    return (
        PolicyMeta(
            kind="cqn_as",
            bodyslam_mode=str(bodyslam.get("mode", "off")),
            cameras=tuple(env_cfg.get("cameras", []) or []),
            pixels=bool(config.get("pixels", False)) if isinstance(config, dict) else False,
            raw_payload_keys=keys,
        ),
        payload,
    )
