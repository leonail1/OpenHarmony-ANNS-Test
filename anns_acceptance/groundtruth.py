from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .config import VectorFormat
from .labels import LiveLabelStore, read_ids


def load_vectors(path: Path, vector_format: VectorFormat, dimensions: int | None = None) -> np.ndarray:
    if vector_format == "npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path} must be a 2D npy array")
        return arr.astype(np.float32, copy=False)
    if vector_format == "float32":
        if dimensions is None:
            raise ValueError("dimensions is required for raw float32 vectors")
        data = np.fromfile(path, dtype=np.float32)
        if data.size % dimensions != 0:
            raise ValueError(f"{path} size is not divisible by dimensions={dimensions}")
        return data.reshape((-1, dimensions))
    if vector_format == "fbin":
        with path.open("rb") as handle:
            header = np.fromfile(handle, dtype=np.int32, count=2)
            if header.size != 2:
                raise ValueError(f"{path} missing fbin header")
            n, d = int(header[0]), int(header[1])
            data = np.fromfile(handle, dtype=np.float32, count=n * d)
        if data.size != n * d:
            raise ValueError(f"{path} truncated fbin payload")
        return data.reshape((n, d))
    raise ValueError(f"unsupported vector format: {vector_format}")


def write_vectors(path: Path, vectors: np.ndarray, vector_format: VectorFormat) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = vectors.astype(np.float32, copy=False)
    if vector_format == "npy":
        np.save(path, vectors)
        return
    if vector_format == "float32":
        vectors.tofile(path)
        return
    if vector_format == "fbin":
        with path.open("wb") as handle:
            np.array([vectors.shape[0], vectors.shape[1]], dtype=np.int32).tofile(handle)
            vectors.tofile(handle)
        return
    raise ValueError(f"unsupported vector format: {vector_format}")


def sliced_vector_path(original: Path, output_dir: Path, prefix: str, vector_format: VectorFormat) -> Path:
    if vector_format == "npy":
        return output_dir / f"{prefix}.npy"
    if vector_format == "fbin":
        return output_dir / f"{prefix}.fbin"
    return output_dir / f"{prefix}.f32"


def vector_count(path: Path, vector_format: VectorFormat, dimensions: int | None = None) -> int:
    return int(load_vectors(path, vector_format, dimensions).shape[0])


def load_vector_sources(
    sources: list[tuple[Path, Path | None]],
    vector_format: VectorFormat,
    dimensions: int | None,
) -> dict[int, np.ndarray]:
    vectors_by_id: dict[int, np.ndarray] = {}
    for vectors_path, ids_path in sources:
        vectors = load_vectors(vectors_path, vector_format, dimensions)
        ids = read_ids(ids_path, len(vectors))
        for vector_id, vector in zip(ids, vectors, strict=False):
            vectors_by_id[int(vector_id)] = vector
    return vectors_by_id


def compute_groundtruth(
    vectors_by_id: dict[int, np.ndarray],
    query_file: Path,
    vector_format: VectorFormat,
    dimensions: int | None,
    labels: LiveLabelStore,
    selector: dict,
    k: int,
    query_limit: int,
    scratch_dir: Path | None = None,
) -> dict[int, list[int]]:
    candidate_ids = [vector_id for vector_id in labels.matching_ids(selector) if vector_id in vectors_by_id]
    if len(candidate_ids) < k:
        return {}
    if scratch_dir is None:
        raise ValueError("scratch_dir is required for PipeANN C++ groundtruth")
    try:
        return _compute_groundtruth_with_pipeann_cpp(
            vectors_by_id,
            query_file,
            vector_format,
            dimensions,
            candidate_ids,
            k,
            query_limit,
            scratch_dir,
        )
    except Exception:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        raise


def _pipeann_gt_binary() -> Path:
    return Path(
        os.environ.get("ANNS_GT_BINARY", "/mnt/nvme1n1/PipeANN-github/build/tests/utils/compute_groundtruth")
    )


def _compute_groundtruth_with_pipeann_cpp(
    vectors_by_id: dict[int, np.ndarray],
    query_file: Path,
    vector_format: VectorFormat,
    dimensions: int | None,
    candidate_ids: list[int],
    k: int,
    query_limit: int,
    scratch_dir: Path,
) -> dict[int, list[int]]:
    binary = _pipeann_gt_binary()
    if not binary.exists():
        raise FileNotFoundError(f"PipeANN compute_groundtruth binary not found: {binary}")
    if any(vector_id < 0 or vector_id > np.iinfo(np.uint32).max for vector_id in candidate_ids):
        raise ValueError("PipeANN C++ groundtruth requires uint32-compatible vector ids")
    run_dir = scratch_dir / f"run_{os.getpid()}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    base_path = run_dir / "candidate_base.fbin"
    query_path = run_dir / "queries.fbin"
    tags_path = run_dir / "candidate_tags.u32bin"
    truth_path = run_dir / "truthset.bin"

    candidate_matrix = np.stack([vectors_by_id[vector_id] for vector_id in candidate_ids]).astype(np.float32, copy=False)
    queries = load_vectors(query_file, vector_format, dimensions)[:query_limit]
    _write_fbin(base_path, candidate_matrix)
    _write_fbin(query_path, queries.astype(np.float32, copy=False))
    _write_u32_bin(tags_path, np.asarray(candidate_ids, dtype=np.uint32).reshape((-1, 1)))

    threads = int(os.environ.get("ANNS_GT_THREADS", str(os.cpu_count() or 1)))
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_PROC_BIND", "spread")
    env.setdefault("OMP_PLACES", "cores")
    command = [
        str(binary),
        "float",
        "l2",
        str(base_path),
        str(query_path),
        str(k),
        str(truth_path),
        str(tags_path),
        "null",
    ]
    subprocess.run(command, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    groundtruth = _read_pipeann_truthset_tags(truth_path, candidate_ids)

    if os.environ.get("ANNS_GT_KEEP_SCRATCH", "0") != "1":
        shutil.rmtree(run_dir, ignore_errors=True)
    return groundtruth


def _write_fbin(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    with path.open("wb") as handle:
        np.asarray([vectors.shape[0], vectors.shape[1]], dtype=np.int32).tofile(handle)
        vectors.tofile(handle)


def _write_u32_bin(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype=np.uint32)
    with path.open("wb") as handle:
        np.asarray([values.shape[0], values.shape[1]], dtype=np.int32).tofile(handle)
        values.tofile(handle)


def _read_pipeann_truthset_tags(path: Path, candidate_ids: list[int]) -> dict[int, list[int]]:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"{path} missing truthset header")
        nqueries, k = int(header[0]), int(header[1])
        ids = np.fromfile(handle, dtype=np.int32, count=nqueries * k)
        dists = np.fromfile(handle, dtype=np.float32, count=nqueries * k)
        tags = np.fromfile(handle, dtype=np.uint32, count=nqueries * k)
    if ids.size != nqueries * k or dists.size != nqueries * k:
        raise ValueError(f"{path} truncated PipeANN truthset")
    if tags.size == nqueries * k:
        values = tags.reshape((nqueries, k)).astype(np.int64, copy=False)
    elif tags.size == 0:
        local_ids = ids.reshape((nqueries, k))
        mapped = np.empty_like(local_ids, dtype=np.int64)
        for query_idx in range(nqueries):
            for rank in range(k):
                local_id = int(local_ids[query_idx, rank])
                mapped[query_idx, rank] = -1 if local_id < 0 else int(candidate_ids[local_id])
        values = mapped
    else:
        raise ValueError(f"{path} has partial PipeANN tag block")
    return {
        query_idx: [int(value) for value in values[query_idx] if int(value) >= 0]
        for query_idx in range(nqueries)
    }
