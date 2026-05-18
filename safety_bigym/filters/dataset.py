"""Sharded SVF transition dataset.

One ``.npz`` shard per (source, task, batch_index); a ``manifest.json`` at the
root records the spec, shard list, and per-source violation rates. Pixel data
is **never** stored — the critic is decoupled from the actor's encoder.

The dataset is read-only at training time; collection scripts use
``TransitionShardWriter`` to lay down shards incrementally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from safety_bigym.filters.feature_extractor import CriticFeatureSpec

PathLike = Union[str, Path]

_SHARD_REQUIRED_ARRAYS = {
    "action",
    "r_safe",
    "done",
    "ssm_margin",
    "min_separation",
    "pfl_force_ratio",
    "source",
    "task_id",
}


def _obs_array_name(prefix: str, key: str) -> str:
    return f"{prefix}__{key}"


class TransitionShardWriter:
    """Writes one or more ``.npz`` shards into a dataset directory.

    Maintains a ``manifest.json`` at ``root`` describing the spec, shard list,
    and per-source violation rates. The manifest is rewritten after every
    shard so partial collections remain consistent.
    """

    def __init__(self, spec: CriticFeatureSpec, root: PathLike):
        self.spec = spec
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._manifest = self._load_or_init_manifest()

    def _load_or_init_manifest(self) -> dict:
        if self._manifest_path.exists():
            data = json.loads(self._manifest_path.read_text())
            if data.get("spec") and data["spec"]["obs_keys"] != list(self.spec.obs_keys):
                raise ValueError(
                    "Existing manifest uses a different CriticFeatureSpec; "
                    "refusing to mix shards. "
                    f"existing={data['spec']}, new={self.spec.to_dict()}"
                )
            return data
        return {
            "spec": self.spec.to_dict(),
            "shards": [],
            "total_transitions": 0,
            "violations_total": 0,
            "violation_rate_total": 0.0,
            "by_source": {},
        }

    def write_shard(
        self,
        name: str,
        *,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        r_safe: np.ndarray,
        done: np.ndarray,
        ssm_margin: np.ndarray,
        min_separation: np.ndarray,
        pfl_force_ratio: np.ndarray,
        source: np.ndarray,
        task_id: np.ndarray,
    ) -> Path:
        """Persist one shard. ``name`` is the filename stem (no extension).

        ``min_separation`` and ``pfl_force_ratio`` are stored per-step so the
        binary ``r_safe`` label can be recomputed later (e.g., proximity
        threshold sweep, PFL retrofit once the contact-detection bug lands)
        without re-collecting transitions. ``pfl_force_ratio`` is identically
        zero under the current PFL bug — stored anyway for forward
        compatibility; relabelling needs a future re-collection through a
        PFL-fixed env, but the schema is already in place.
        """
        n = len(action)
        if n == 0:
            raise ValueError("Refusing to write empty shard")

        for key, dim in zip(self.spec.obs_keys, self.spec.obs_dims):
            for label, source_dict in (("obs", obs), ("next_obs", next_obs)):
                if key not in source_dict:
                    raise KeyError(
                        f"Missing {label!r}[{key!r}] required by spec"
                    )
                arr = np.asarray(source_dict[key], dtype=np.float32)
                if arr.shape != (n, dim):
                    raise ValueError(
                        f"{label}[{key}] shape {arr.shape} != expected ({n}, {dim})"
                    )

        if action.shape != (n, self.spec.action_dim):
            raise ValueError(
                f"action shape {action.shape} != ({n}, {self.spec.action_dim})"
            )

        payload: Dict[str, np.ndarray] = {
            "action": action.astype(np.float32, copy=False),
            "r_safe": r_safe.astype(np.float32, copy=False),
            "done": done.astype(np.bool_, copy=False),
            "ssm_margin": ssm_margin.astype(np.float32, copy=False),
            "min_separation": min_separation.astype(np.float32, copy=False),
            "pfl_force_ratio": pfl_force_ratio.astype(np.float32, copy=False),
            "source": source.astype(np.uint8, copy=False),
            "task_id": task_id.astype(np.uint8, copy=False),
        }
        for key in self.spec.obs_keys:
            payload[_obs_array_name("obs", key)] = obs[key].astype(np.float32, copy=False)
            payload[_obs_array_name("next_obs", key)] = next_obs[key].astype(
                np.float32, copy=False
            )

        path = self.root / f"{name}.npz"
        np.savez(path, **payload)

        # Update manifest
        n_violations = int((r_safe == 0.0).sum())
        self._manifest["shards"].append(path.name)
        self._manifest["total_transitions"] += n
        self._manifest["violations_total"] += n_violations
        total = self._manifest["total_transitions"]
        self._manifest["violation_rate_total"] = (
            self._manifest["violations_total"] / total if total else 0.0
        )
        for src in np.unique(source):
            key = str(int(src))
            mask = source == src
            entry = self._manifest["by_source"].setdefault(
                key, {"transitions": 0, "violations": 0}
            )
            entry["transitions"] += int(mask.sum())
            entry["violations"] += int(((r_safe == 0.0) & mask).sum())
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))
        return path


@dataclass
class _ShardIndex:
    """Pointer into a memory-mapped shard."""

    path: Path
    n: int
    offset: int  # global index where this shard's transitions begin


class SafetyTransitionDataset(Dataset):
    """Read-only view over a directory of ``.npz`` shards.

    Each shard is loaded lazily on first access (mmap mode) so the dataset
    stays cheap for large collections. Indices are flat across shards.
    """

    def __init__(self, root: PathLike):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            self.spec = CriticFeatureSpec.from_dict(manifest["spec"])
            shard_names = manifest["shards"]
        else:
            shard_names = sorted(p.name for p in self.root.glob("*.npz"))
            if not shard_names:
                raise ValueError(f"No shards or manifest found at {self.root}")
            sample = np.load(self.root / shard_names[0])
            self.spec = self._infer_spec_from_arrays(sample)

        if not shard_names:
            raise ValueError(f"Dataset at {self.root} has no shards")

        self._shards: List[_ShardIndex] = []
        self._caches: Dict[int, dict] = {}
        offset = 0
        for name in shard_names:
            path = self.root / name
            with np.load(path, mmap_mode="r") as data:
                n = int(data["action"].shape[0])
            self._shards.append(_ShardIndex(path=path, n=n, offset=offset))
            offset += n
        self._total = offset

        violation_indices: List[int] = []
        safe_indices: List[int] = []
        for shard_idx, shard in enumerate(self._shards):
            r_safe = self._load_shard(shard_idx)["r_safe"]
            base = shard.offset
            v = np.where(r_safe == 0.0)[0]
            s = np.where(r_safe == 1.0)[0]
            violation_indices.extend((base + v).tolist())
            safe_indices.extend((base + s).tolist())
        self.violation_indices: np.ndarray = np.asarray(violation_indices, dtype=np.int64)
        self.safe_indices: np.ndarray = np.asarray(safe_indices, dtype=np.int64)

    @staticmethod
    def _infer_spec_from_arrays(npz) -> CriticFeatureSpec:
        obs_keys: List[str] = []
        obs_dims: List[int] = []
        for k in npz.files:
            if k.startswith("obs__"):
                key = k.removeprefix("obs__")
                obs_keys.append(key)
                obs_dims.append(int(npz[k].shape[-1]))
        order = np.argsort(obs_keys)
        obs_keys = [obs_keys[i] for i in order]
        obs_dims = [obs_dims[i] for i in order]
        action_dim = int(npz["action"].shape[-1])
        return CriticFeatureSpec(
            obs_keys=tuple(obs_keys), obs_dims=tuple(obs_dims), action_dim=action_dim
        )

    def _load_shard(self, shard_idx: int) -> dict:
        if shard_idx in self._caches:
            return self._caches[shard_idx]
        shard = self._shards[shard_idx]
        with np.load(shard.path) as data:
            cached = {k: np.asarray(data[k]) for k in data.files}
        self._caches[shard_idx] = cached
        return cached

    def _locate(self, index: int) -> Tuple[int, int]:
        if not 0 <= index < self._total:
            raise IndexError(f"index {index} out of range for size {self._total}")
        # Linear scan is fine — shard count is small (tens, not millions)
        for i, shard in enumerate(self._shards):
            if index < shard.offset + shard.n:
                return i, index - shard.offset
        raise IndexError(index)

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int) -> dict:
        shard_idx, local_idx = self._locate(index)
        data = self._load_shard(shard_idx)
        obs = {
            k: data[_obs_array_name("obs", k)][local_idx] for k in self.spec.obs_keys
        }
        next_obs = {
            k: data[_obs_array_name("next_obs", k)][local_idx]
            for k in self.spec.obs_keys
        }
        return {
            "obs": obs,
            "action": data["action"][local_idx],
            "next_obs": next_obs,
            "r_safe": float(data["r_safe"][local_idx]),
            "done": bool(data["done"][local_idx]),
            "ssm_margin": float(data["ssm_margin"][local_idx]),
            "min_separation": float(data["min_separation"][local_idx]),
            "pfl_force_ratio": float(data["pfl_force_ratio"][local_idx]),
            "source": int(data["source"][local_idx]),
            "task_id": int(data["task_id"][local_idx]),
        }


def make_oversampler(
    dataset: SafetyTransitionDataset,
    target_violation_rate: float = 0.3,
    seed: Optional[int] = None,
) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that oversamples violating transitions.

    The per-batch expected fraction of violation samples approaches
    ``target_violation_rate``. Falls back to uniform when no violations exist
    in the dataset.
    """
    n = len(dataset)
    n_pos = len(dataset.violation_indices)
    n_neg = len(dataset.safe_indices)
    if n_pos == 0 or n_neg == 0:
        weights = torch.ones(n, dtype=torch.float64)
    else:
        weights = torch.zeros(n, dtype=torch.float64)
        weights[dataset.violation_indices] = float(target_violation_rate) / n_pos
        weights[dataset.safe_indices] = float(1.0 - target_violation_rate) / n_neg
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights, num_samples=n, replacement=True, generator=generator
    )


__all__ = [
    "SafetyTransitionDataset",
    "TransitionShardWriter",
    "make_oversampler",
]
