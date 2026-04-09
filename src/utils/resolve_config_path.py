"""Utility for resolving configuration file paths."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_config_path(
    config_path: str | os.PathLike[str] | None,
    project_root: str | os.PathLike[str],
    current_dir: str | os.PathLike[str],
    default_relative_path: str | os.PathLike[str] = os.path.join(
        "configs", "config.cfg"
    ),
) -> str:
    """Resolve a config file path against the project and source directories."""
    project_root_path = os.fspath(project_root)
    current_dir_path = os.fspath(current_dir)
    default_relative = os.fspath(default_relative_path)
    default_config_path = (
        default_relative
        if os.path.isabs(default_relative)
        else os.path.join(project_root_path, default_relative)
    )

    if config_path is None or not str(config_path).strip():
        if not os.path.exists(default_config_path):
            raise FileNotFoundError(
                f"Configuration file not found. Checked: {default_config_path}"
            )
        return os.path.abspath(default_config_path)

    raw_path = os.fspath(config_path)
    candidates = (
        [raw_path]
        if os.path.isabs(raw_path)
        else [os.path.join(project_root_path, raw_path)]
    )
    if not os.path.isabs(raw_path) and len(Path(raw_path).parts) == 1:
        candidates.extend(
            [
                os.path.join(project_root_path, "configs", os.path.basename(raw_path)),
                os.path.join(current_dir_path, os.path.basename(raw_path)),
            ]
        )

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    checked_paths = ", ".join(candidates)
    raise FileNotFoundError(f"Configuration file not found. Checked: {checked_paths}")
