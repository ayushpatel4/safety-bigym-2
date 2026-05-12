"""Per-task snapshot path lookup — single source of truth for SVF scripts.

Update :data:`SNAPSHOTS` after each Phase-0 retrain by inspecting W&B
``pretrain_eval/episode_success`` curves and picking the peak step. See
the session-handling rules in CLAUDE.md.

Tasks left as ``None`` are *deliberately* skipped by the snapshot-source
collection / snapshot-policy eval paths. A task whose value points to a
nonexistent file raises :class:`FileNotFoundError` from
:func:`resolve_snapshot` — that's a typo / stale path, not a deliberate skip.

Paths are resolved relative to the safety_bigym repo root (i.e. the parent
of this package), so the dict stays portable across local and GPU layouts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

# Repo root is two levels up from this file: safety_bigym/filters/snapshots.py
_REPO_ROOT = Path(__file__).resolve().parents[2]


# Keyed by safety_bigym task_key. Values are paths relative to the repo root,
# absolute paths, or ``None`` for "no snapshot yet — skip this task".
#
# Set after each retrain; pick the W&B `pretrain_eval/episode_success` peak.
SNAPSHOTS: Dict[str, Optional[str]] = {
    "reach_target_single": None,
    "dishwasher_close": None,
    "dishwasher_load_plates": None,
    "saucepan_to_hob": None,
}


def _resolve_path(value: str) -> Path:
    p = Path(os.path.expanduser(value))
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def resolve_snapshot(
    task_key: str,
    *,
    overrides: Optional[Mapping[str, str]] = None,
) -> Optional[Path]:
    """Return the on-disk path for ``task_key``, or ``None`` if deliberately unset.

    Parameters
    ----------
    task_key
        The task name (matches ``TASK_REGISTRY`` keys in svf_collect_dataset.py).
    overrides
        Optional per-task override mapping — typically populated from CLI flags.
        Values in this mapping take precedence over :data:`SNAPSHOTS`.

    Raises
    ------
    KeyError
        If ``task_key`` is not in :data:`SNAPSHOTS` and no override is provided.
    FileNotFoundError
        If the resolved path is non-empty but doesn't exist on disk (a stale
        / typo'd entry — distinct from deliberate ``None``).
    """
    raw = None
    if overrides and task_key in overrides:
        raw = overrides[task_key]
    elif task_key in SNAPSHOTS:
        raw = SNAPSHOTS[task_key]
    else:
        raise KeyError(
            f"Unknown task {task_key!r} — add it to SNAPSHOTS in "
            "safety_bigym/filters/snapshots.py"
        )

    if raw is None or raw == "":
        return None

    path = _resolve_path(str(raw))
    if not path.is_file():
        raise FileNotFoundError(
            f"Snapshot for task {task_key!r} listed but missing on disk: {path}. "
            "Update SNAPSHOTS in safety_bigym/filters/snapshots.py."
        )
    return path


__all__ = ["SNAPSHOTS", "resolve_snapshot"]
