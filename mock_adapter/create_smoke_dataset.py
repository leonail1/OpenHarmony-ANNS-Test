from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny mock acceptance dataset and configs.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260605)
    n = 1000
    d = 8
    q = 16
    vectors0 = rng.normal(size=(n, d)).astype(np.float32)
    queries = vectors0[:q].copy()
    np.save(out / "base.npy", vectors0)
    np.save(out / "queries.npy", queries)
    _write_ids(out / "base.ids", list(range(n)))
    for cycle in range(1, 3):
        vectors = rng.normal(size=(n, d)).astype(np.float32) + cycle
        np.save(out / f"insert{cycle}.npy", vectors)
        _write_ids(out / f"insert{cycle}.ids", list(range(cycle * 1000, cycle * 1000 + n)))
    _write_manifest(out / "adapter_manifest.positive.yaml", latency_ms=1.0, selectivity_offset=0)
    _write_manifest(out / "adapter_manifest.negative.yaml", latency_ms=25.0, selectivity_offset=0)
    _write_manifest(out / "adapter_manifest.rss_negative.yaml", latency_ms=1.0, selectivity_offset=0, allocate_mb=64)
    _write_config(out / "acceptance_config.positive.yaml", out, "adapter_manifest.positive.yaml")
    _write_config(out / "acceptance_config.negative.yaml", out, "adapter_manifest.negative.yaml")
    _write_config(out / "acceptance_config.rss_negative.yaml", out, "adapter_manifest.rss_negative.yaml")


def _write_manifest(path: Path, latency_ms: float, selectivity_offset: int, allocate_mb: int = 0) -> None:
    mock = "{repo_root}/mock_adapter/mock_ann.py"
    commands = {
        "ann_build_index": [
            "{python}",
            mock,
            "build",
            "--vectors",
            "{vectors_path}",
            "--ids",
            "{ids_path}",
            "--labels",
            "{labels_path}",
            "--output-manifest",
            "{output_manifest_path}",
            "--state-dir",
            "{state_dir}",
            "--index-dir",
            "{index_dir}",
            "--threads",
            "{threads}",
        ],
        "ann_filter_search": [
            "{python}",
            mock,
            "search",
            "--queries",
            "{query_file}",
            "--selector",
            "{selector_file}",
            "--k",
            "{k}",
            "--limit",
            "{query_limit}",
            "--output",
            "{output_path}",
            "--state-dir",
            "{state_dir}",
            "--threads",
            "{threads}",
            "--latency-ms",
            str(latency_ms),
            "--allocate-mb",
            str(allocate_mb),
        ],
        "ann_apply_insert": [
            "{python}",
            mock,
            "insert",
            "--vectors",
            "{insert_vectors_path}",
            "--ids",
            "{insert_ids_path}",
            "--labels",
            "{labels_path}",
            "--output",
            "{output_path}",
            "--state-dir",
            "{state_dir}",
            "--threads",
            "{threads}",
        ],
        "ann_apply_delete": [
            "{python}",
            mock,
            "delete",
            "--ids",
            "{delete_ids_path}",
            "--output",
            "{output_path}",
            "--state-dir",
            "{state_dir}",
            "--threads",
            "{threads}",
        ],
        "ann_label_selectivity": [
            "{python}",
            mock,
            "selectivity",
            "--selector",
            "{selector_file}",
            "--output",
            "{output_path}",
            "--state-dir",
            "{state_dir}",
            "--threads",
            "{threads}",
            "--offset",
            str(selectivity_offset),
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"commands": {k: {"command": v} for k, v in commands.items()}}, handle, sort_keys=False)


def _write_config(path: Path, out: Path, manifest_name: str) -> None:
    if "rss_negative" in manifest_name:
        suffix = "rss_negative"
    elif "negative" in manifest_name:
        suffix = "negative"
    else:
        suffix = "positive"
    payload = {
        "adapter_manifest": manifest_name,
        "results_dir": str(out / f"results_{suffix}"),
        "work_dir": str(out / f"work_{suffix}"),
        "target_npoints": 1000,
        "dynamic_cycles": 1,
        "delete_fraction": 0.60,
        "pilot_query_limit": 8,
        "full_query_limit": 12,
        "foreground_search_threads": [1, 8],
        "k": 5,
        "datasets": [
            {
                "name": "mock",
                "base_vectors": str(out / "base.npy"),
                "base_ids": str(out / "base.ids"),
                "query_file": str(out / "queries.npy"),
                "vector_format": "npy",
                "dimensions": 8,
                "build_threads": 1,
                "search_threads": [1],
                "insert_batches": [
                    {"name": "initial", "vectors": str(out / "base.npy"), "ids": str(out / "base.ids"), "count": 1000},
                    {
                        "name": "cycle1",
                        "vectors": str(out / "insert1.npy"),
                        "ids": str(out / "insert1.ids"),
                        "count": 1000,
                    },
                ],
            }
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_ids(path: Path, ids: list[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for vector_id in ids:
            handle.write(f"{vector_id}\n")


if __name__ == "__main__":
    main()
