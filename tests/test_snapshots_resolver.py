"""Tests for filters/snapshots.py — per-task snapshot path resolver."""

from pathlib import Path

import pytest

from safety_bigym.filters import snapshots as snapmod
from safety_bigym.filters.snapshots import resolve_snapshot


def test_unset_task_returns_none(monkeypatch):
    monkeypatch.setattr(snapmod, "SNAPSHOTS", {"reach_target_single": None})
    assert resolve_snapshot("reach_target_single") is None


def test_empty_string_treated_as_unset(monkeypatch):
    monkeypatch.setattr(snapmod, "SNAPSHOTS", {"t": ""})
    assert resolve_snapshot("t") is None


def test_unknown_task_raises_keyerror(monkeypatch):
    monkeypatch.setattr(snapmod, "SNAPSHOTS", {})
    with pytest.raises(KeyError, match="Unknown task"):
        resolve_snapshot("never_seen_task")


def test_existing_absolute_path_resolves(tmp_path, monkeypatch):
    snap = tmp_path / "snap.pt"
    snap.write_bytes(b"")
    monkeypatch.setattr(snapmod, "SNAPSHOTS", {"t": str(snap)})
    assert resolve_snapshot("t") == snap


def test_relative_path_resolved_against_repo_root(tmp_path, monkeypatch):
    """Relative paths are resolved against the repo root, so the dict stays
    portable across local and GPU layouts."""
    fake_root = tmp_path / "repo"
    (fake_root / "exp_local").mkdir(parents=True)
    snap = fake_root / "exp_local" / "snap.pt"
    snap.write_bytes(b"")
    monkeypatch.setattr(snapmod, "_REPO_ROOT", fake_root)
    monkeypatch.setattr(snapmod, "SNAPSHOTS", {"t": "exp_local/snap.pt"})
    assert resolve_snapshot("t") == snap


def test_missing_file_raises_file_not_found(monkeypatch):
    monkeypatch.setattr(snapmod, "SNAPSHOTS", {"t": "/nope/never_existed.pt"})
    with pytest.raises(FileNotFoundError, match="missing on disk"):
        resolve_snapshot("t")


def test_override_wins_over_dict(tmp_path, monkeypatch):
    dict_snap = tmp_path / "dict.pt"
    dict_snap.write_bytes(b"")
    override_snap = tmp_path / "override.pt"
    override_snap.write_bytes(b"")
    monkeypatch.setattr(snapmod, "SNAPSHOTS", {"t": str(dict_snap)})
    out = resolve_snapshot("t", overrides={"t": str(override_snap)})
    assert out == override_snap


def test_override_can_introduce_unknown_task(tmp_path, monkeypatch):
    """Overrides extend the dict, so a CLI flag for a brand-new task works."""
    snap = tmp_path / "snap.pt"
    snap.write_bytes(b"")
    monkeypatch.setattr(snapmod, "SNAPSHOTS", {})
    out = resolve_snapshot("brand_new", overrides={"brand_new": str(snap)})
    assert out == snap


def test_override_to_none_skips(monkeypatch):
    """Override can also explicitly set None to skip a task that has a
    SNAPSHOTS entry — useful for one-off ablations."""
    monkeypatch.setattr(snapmod, "SNAPSHOTS", {"t": "/some/path.pt"})
    assert resolve_snapshot("t", overrides={"t": None}) is None
