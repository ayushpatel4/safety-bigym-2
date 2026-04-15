"""Path resolution helpers for external data."""
from __future__ import annotations

import os
from pathlib import Path

_AMASS_ENV = "AMASS_DATA_DIR"


def get_amass_data_dir(override: str | Path | None = None) -> Path:
    """Resolve the AMASS CMU motion-clip root directory.

    Precedence: ``override`` argument > ``$AMASS_DATA_DIR`` environment variable.
    Raises :class:`FileNotFoundError` if neither is set, or if the resolved path
    does not exist on disk.
    """
    raw = override if override is not None else os.environ.get(_AMASS_ENV)
    if raw is None:
        raise FileNotFoundError(
            f"AMASS motion-clip directory not configured. "
            f"Set ${_AMASS_ENV} or pass an explicit path. "
            f"See CLAUDE.md > 'Critical gotchas' for details."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        source = "override" if override is not None else f"${_AMASS_ENV}"
        raise FileNotFoundError(
            f"AMASS directory does not exist: {path} (resolved from {source})"
        )
    return path
