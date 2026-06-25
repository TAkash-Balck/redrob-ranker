"""
loader.py — Fast streaming JSONL loader with progress bar.

Streams candidates one-by-one to keep RAM usage minimal even for
very large files (>1 GB).  Uses orjson when available for ~3× faster
parsing, falls back gracefully to stdlib json.
"""

from __future__ import annotations

import json
import os
from typing import Generator, Optional

from tqdm import tqdm

# Try to import orjson for faster parsing; fallback to stdlib
try:
    import orjson  # type: ignore

    def _parse(line: bytes) -> dict:  # type: ignore[return]
        return orjson.loads(line)

except ImportError:
    def _parse(line: str) -> dict:  # type: ignore[return]
        return json.loads(line)


def stream_candidates(
    path: str,
    max_candidates: Optional[int] = None,
    show_progress: bool = True,
) -> Generator[dict, None, None]:
    """Yield parsed candidate dicts one at a time from a JSONL file.

    Args:
        path: Absolute or relative path to the JSONL file.
        max_candidates: If set, stop after this many candidates (for testing).
        show_progress: Whether to show a tqdm progress bar.

    Yields:
        Parsed candidate dict (may contain nested dicts/lists).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If a line cannot be parsed as JSON (skipped with warning).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Candidate file not found: {path}")

    file_size = os.path.getsize(path)

    with open(path, "rb") as fh:
        pbar = tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc="Loading candidates",
            disable=not show_progress,
        )
        count = 0
        for raw_line in fh:
            pbar.update(len(raw_line))
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                candidate = _parse(stripped)
                yield candidate
                count += 1
                if max_candidates and count >= max_candidates:
                    break
            except Exception as exc:
                # Skip malformed lines but warn
                tqdm.write(f"[WARN] Skipping malformed line: {exc}")
                continue
        pbar.close()


def load_all_candidates(
    path: str,
    max_candidates: Optional[int] = None,
    show_progress: bool = True,
) -> list[dict]:
    """Load all candidates into memory (use only when RAM permits).

    For the full 100K dataset this will use ~2 GB RAM.
    Prefer :func:`stream_candidates` in production runs.

    Args:
        path: Path to the JSONL file.
        max_candidates: Optional cap.
        show_progress: Show progress bar.

    Returns:
        List of parsed candidate dicts.
    """
    return list(stream_candidates(path, max_candidates=max_candidates, show_progress=show_progress))


def safe_get(d: dict, *keys: str, default=None):
    """Safely traverse nested dict with a chain of keys.

    Args:
        d: The root dict.
        *keys: Sequence of keys to traverse.
        default: Value returned when any key is missing or value is None.

    Returns:
        The nested value, or *default*.
    """
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current
