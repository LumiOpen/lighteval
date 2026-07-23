"""Helpers for immutable, checksum-verified runtime assets."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import requests


def lighteval_cache_dir() -> Path:
    """Return the writable cache root used for downloaded LightEval assets."""
    if xdg_cache := os.environ.get("XDG_CACHE_HOME"):
        cache_home = Path(xdg_cache).expanduser()
    else:
        cache_home = Path.home() / ".cache"
    return cache_home / "huggingface" / "lighteval" / "assets"


def ensure_cached_asset(*, relative_path: str, url: str, sha256: str, timeout: float = 60) -> Path:
    """Download an asset into the writable cache and verify its expected digest.

    Downloads are written to a temporary file and atomically published. Multiple
    processes may download the same missing asset concurrently, but they can only
    publish content matching the pinned digest.
    """
    target = lighteval_cache_dir() / relative_path
    if target.is_file() and _sha256(target.read_bytes()) == sha256:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    content = response.content
    actual_sha256 = _sha256(content)
    if actual_sha256 != sha256:
        raise RuntimeError(f"Checksum mismatch for {url}: expected {sha256}, got {actual_sha256}")

    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    return target


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
