from __future__ import annotations

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
) -> dict[int, list[int]]:
    candidate_ids = [vector_id for vector_id in labels.matching_ids(selector) if vector_id in vectors_by_id]
    if len(candidate_ids) < k:
        return {}
    matrix = np.stack([vectors_by_id[vector_id] for vector_id in candidate_ids]).astype(np.float32, copy=False)
    queries = load_vectors(query_file, vector_format, dimensions)[:query_limit]
    groundtruth: dict[int, list[int]] = {}
    for query_idx, query in enumerate(queries):
        diff = matrix - query
        distances = np.einsum("ij,ij->i", diff, diff)
        order = np.argsort(distances, kind="stable")[:k]
        groundtruth[query_idx] = [int(candidate_ids[i]) for i in order]
    return groundtruth
