from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Small exact mock ANN adapter for acceptance smoke tests.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build")
    build.add_argument("--vectors", required=True)
    build.add_argument("--ids", required=True)
    build.add_argument("--labels", required=True)
    build.add_argument("--output-manifest", required=True)
    build.add_argument("--state-dir", required=True)
    build.add_argument("--index-dir", required=True)
    build.add_argument("--threads", type=int, required=True)
    build.add_argument("--latency-ms", type=float, default=1.0)

    insert = sub.add_parser("insert")
    insert.add_argument("--vectors", required=True)
    insert.add_argument("--ids", required=True)
    insert.add_argument("--labels", required=True)
    insert.add_argument("--output", required=True)
    insert.add_argument("--state-dir", required=True)
    insert.add_argument("--threads", type=int, required=True)
    insert.add_argument("--latency-ms", type=float, default=1.0)

    delete = sub.add_parser("delete")
    delete.add_argument("--ids", required=True)
    delete.add_argument("--output", required=True)
    delete.add_argument("--state-dir", required=True)
    delete.add_argument("--threads", type=int, required=True)
    delete.add_argument("--latency-ms", type=float, default=1.0)

    search = sub.add_parser("search")
    search.add_argument("--queries", required=True)
    search.add_argument("--selector", required=True)
    search.add_argument("--k", type=int, required=True)
    search.add_argument("--limit", type=int, required=True)
    search.add_argument("--output", required=True)
    search.add_argument("--state-dir", required=True)
    search.add_argument("--threads", type=int, required=True)
    search.add_argument("--latency-ms", type=float, default=1.0)
    search.add_argument("--allocate-mb", type=int, default=0)

    selectivity = sub.add_parser("selectivity")
    selectivity.add_argument("--selector", required=True)
    selectivity.add_argument("--output", required=True)
    selectivity.add_argument("--state-dir", required=True)
    selectivity.add_argument("--threads", type=int, required=True)
    selectivity.add_argument("--offset", type=int, default=0)

    args = parser.parse_args()
    time.sleep(max(0.0, getattr(args, "latency_ms", 0.0)) / 1000.0)
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "insert":
        cmd_insert(args)
    elif args.cmd == "delete":
        cmd_delete(args)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "selectivity":
        cmd_selectivity(args)


def cmd_build(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    vectors, ids = _load_vectors_with_ids(Path(args.vectors), Path(args.ids))
    labels = _read_labels(Path(args.labels))
    state = {"vectors": {}, "labels": {}}
    for vector_id, vector in zip(ids, vectors, strict=False):
        state["vectors"][str(vector_id)] = vector.astype(float).tolist()
        state["labels"][str(vector_id)] = labels[str(vector_id)]
    _save_state(state_dir, state)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_file = index_dir / "mock.index"
    index_file.write_bytes(b"mock-index\n")
    _write_json(
        Path(args.output_manifest),
        {"raw_data_paths": [str(Path(args.vectors).resolve())], "index_output_paths": [str(index_file.resolve())]},
    )


def cmd_insert(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    state = _load_state(state_dir)
    vectors, ids = _load_vectors_with_ids(Path(args.vectors), Path(args.ids))
    labels = _read_labels(Path(args.labels))
    for vector_id, vector in zip(ids, vectors, strict=False):
        state["vectors"][str(vector_id)] = vector.astype(float).tolist()
        state["labels"][str(vector_id)] = labels[str(vector_id)]
    _save_state(state_dir, state)
    _write_json(Path(args.output), {"live_count": len(state["labels"])})


def cmd_delete(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    state = _load_state(state_dir)
    for vector_id in _read_ids(Path(args.ids)):
        state["vectors"].pop(str(vector_id), None)
        state["labels"].pop(str(vector_id), None)
    _save_state(state_dir, state)
    _write_json(Path(args.output), {"live_count": len(state["labels"])})


def cmd_search(args: argparse.Namespace) -> None:
    state = _load_state(Path(args.state_dir))
    allocation = bytearray(max(0, args.allocate_mb) * 1024 * 1024)
    if allocation:
        for offset in range(0, len(allocation), 4096):
            allocation[offset] = 1
    queries = _load_npy_float32_2d(Path(args.queries))[: args.limit]
    selector = _read_json(Path(args.selector))
    candidate_ids = [int(vector_id) for vector_id, row in state["labels"].items() if _matches(row, selector)]
    candidate_ids.sort()
    results = []
    for query_id, query in enumerate(queries):
        if len(candidate_ids) == 0:
            ids: list[int] = []
        else:
            scored = []
            for vector_id in candidate_ids:
                vector = state["vectors"][str(vector_id)]
                distance = sum((a - b) * (a - b) for a, b in zip(vector, query, strict=False))
                scored.append((distance, vector_id))
            scored.sort(key=lambda item: (item[0], item[1]))
            ids = [vector_id for _, vector_id in scored[: args.k]]
        results.append({"query_id": query_id, "ids": ids, "latency_ms": args.latency_ms})
    if allocation:
        time.sleep(0.25)
    _write_json(
        Path(args.output),
        {
            "results": results,
            "summary": {
                "candidate_count": len(candidate_ids),
                "avg_latency_ms": args.latency_ms,
                "p50_latency_ms": args.latency_ms,
                "p95_latency_ms": args.latency_ms,
                "p99_latency_ms": args.latency_ms,
            },
        },
    )


def cmd_selectivity(args: argparse.Namespace) -> None:
    state = _load_state(Path(args.state_dir))
    selector = _read_json(Path(args.selector))
    matched = sum(1 for row in state["labels"].values() if _matches(row, selector)) + args.offset
    total = len(state["labels"])
    _write_json(
        Path(args.output),
        {
            "matched_count": matched,
            "total_live_count": total,
            "selectivity": 0.0 if total == 0 else matched / total,
        },
    )


def _load_vectors_with_ids(vectors_path: Path, ids_path: Path) -> tuple[np.ndarray, list[int]]:
    import numpy as np

    vectors = np.load(vectors_path).astype(np.float32)
    ids = _read_ids(ids_path)
    return vectors[: len(ids)], ids


def _load_npy_float32_2d(path: Path) -> list[list[float]]:
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"not an npy file: {path}")
        major, _minor = handle.read(2)
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        else:
            header_len = struct.unpack("<I", handle.read(4))[0]
        header = handle.read(header_len).decode("latin1")
        if "'<f4'" not in header and '"<f4"' not in header and "'|f4'" not in header and '"|f4"' not in header:
            raise ValueError(f"only float32 npy is supported by mock search: {path}")
        shape_start = header.index("(") + 1
        shape_end = header.index(")", shape_start)
        shape = [int(part.strip()) for part in header[shape_start:shape_end].split(",") if part.strip()]
        if len(shape) != 2:
            raise ValueError(f"only 2D npy arrays are supported by mock search: {path}")
        rows, dim = shape
        values = struct.unpack("<" + "f" * (rows * dim), handle.read(rows * dim * 4))
    return [list(values[row * dim : (row + 1) * dim]) for row in range(rows)]


def _read_ids(path: Path) -> list[int]:
    ids: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.lower() == "id":
                continue
            ids.append(int(line.split(",", 1)[0]))
    return ids


def _read_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: dict(row) for row in csv.DictReader(handle)}


def _matches(row: dict[str, str], selector: dict[str, Any]) -> bool:
    selector_type = selector.get("selector_type")
    if selector_type == "match_all":
        return True
    if selector_type == "equality":
        return str(row.get(selector["field"])) == str(selector["value"])
    if selector_type == "range":
        value = float(row.get(selector["field"], "nan"))
        return float(selector.get("lower", float("-inf"))) <= value <= float(selector.get("upper", float("inf")))
    if selector_type == "intersect":
        return all(_matches(row, condition) for condition in selector.get("conditions", []))
    return False


def _state_path(state_dir: Path) -> Path:
    return state_dir / "mock_state.json"


def _load_state(state_dir: Path) -> dict[str, Any]:
    path = _state_path(state_dir)
    if not path.exists():
        return {"vectors": {}, "labels": {}}
    return _read_json(path)


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    temp = _state_path(state_dir).with_suffix(".tmp")
    _write_json(temp, state)
    shutil.move(str(temp), _state_path(state_dir))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
