from __future__ import annotations

from pathlib import Path
from typing import Any


def file_bytes(paths: list[str | Path]) -> int:
    total = 0
    for item in paths:
        path = Path(item)
        if path.is_dir():
            total += sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        else:
            total += path.stat().st_size
    return total


def audit_space(manifest: dict[str, Any]) -> dict[str, Any]:
    raw_paths = manifest.get("raw_data_paths") or []
    index_paths = manifest.get("index_output_paths") or []
    if not raw_paths:
        raise ValueError("build manifest missing raw_data_paths")
    if not index_paths:
        raise ValueError("build manifest missing index_output_paths")
    raw_bytes = file_bytes(raw_paths)
    index_bytes = file_bytes(index_paths)
    ratio = float("inf") if raw_bytes <= 0 else index_bytes / raw_bytes
    return {
        "raw_data_paths": [str(p) for p in raw_paths],
        "index_output_paths": [str(p) for p in index_paths],
        "raw_bytes": raw_bytes,
        "index_bytes": index_bytes,
        "expansion_ratio": ratio,
    }

