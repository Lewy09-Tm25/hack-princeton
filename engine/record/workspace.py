"""Workspace directory layout helpers for experiment storage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(identifier: str) -> str:
    """Convert an arbitrary identifier (such as a model or series ID) into a safe filename stem."""
    cleaned = _UNSAFE_PATH_CHARS.sub("__", identifier.strip()).strip("._")
    return cleaned or "unnamed"


@dataclass(frozen=True)
class WorkspaceLayout:
    """Paths inside a local experiment workspace."""

    root: Path
    db_path: Path
    models_dir: Path
    runs_dir: Path
    reports_dir: Path
    exports_dir: Path


def init_workspace(workspace: Path | str) -> WorkspaceLayout:
    """Create the workspace directory tree if missing and return its layout."""
    root = Path(workspace).resolve()
    models_dir = root / "models"
    runs_dir = root / "runs"
    reports_dir = root / "reports"
    exports_dir = root / "exports"

    for directory in (root, models_dir, runs_dir, reports_dir, exports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return WorkspaceLayout(
        root=root,
        db_path=root / "record.sqlite",
        models_dir=models_dir,
        runs_dir=runs_dir,
        reports_dir=reports_dir,
        exports_dir=exports_dir,
    )


def ensure_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Ensure the per-run folder and standard subfolders exist inside `runs/`."""
    safe_id = safe_filename(run_id)
    if safe_id != run_id and ("/" in run_id or "\\" in run_id or ".." in run_id):
        raise ValueError(f"Invalid run_id for workspace path: {run_id!r}")

    target = runs_dir / safe_id
    (target / "code").mkdir(parents=True, exist_ok=True)
    (target / "activations").mkdir(parents=True, exist_ok=True)
    logs_path = target / "logs.txt"
    if not logs_path.exists():
        logs_path.write_text("", encoding="utf-8")
    return target
