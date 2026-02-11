"""Local file storage for swap results."""

from __future__ import annotations

import os
import time
from pathlib import Path

from api.config import settings


def result_path(job_id: str, ext: str = ".jpg") -> str:
    """Return the full path for a job result file."""
    return os.path.join(settings.storage_path, f"{job_id}{ext}")


def save_result(job_id: str, data: bytes, ext: str = ".jpg") -> str:
    """Write result bytes to disk, return the path."""
    path = result_path(job_id, ext)
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def cleanup_old_results() -> int:
    """Delete result files older than configured TTL. Returns count of deleted files."""
    storage = Path(settings.storage_path)
    if not storage.exists():
        return 0
    cutoff = time.time() - (settings.result_ttl_hours * 3600)
    deleted = 0
    for f in storage.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            deleted += 1
    return deleted
