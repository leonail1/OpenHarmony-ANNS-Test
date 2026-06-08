from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from anns_acceptance.adapter import AnnAdapter
from anns_acceptance.artifacts import append_jsonl, ensure_dir
from anns_acceptance.config import AcceptanceConfig, DatasetConfig, VectorBatchConfig
from anns_acceptance.foreground import wait_with_foreground
from anns_acceptance.groundtruth import (
    load_vector_sources,
    load_vectors,
    sliced_vector_path,
    vector_count,
    write_vectors,
)
from anns_acceptance.labels import (
    LiveLabelStore,
    generate_labels_csv,
    generate_rebalanced_insert_labels_csv,
    read_ids,
    selector_to_json,
    write_ids,
    write_label_schema,
)
from anns_acceptance.metrics import parse_search_output
from anns_acceptance.selectors import all_search_selectors, selectivity_check_selectors
from anns_acceptance.testsuite.conftest import (
    StaticContext,
    _append_failure,
    run_search_case,
    run_selectivity_case,
)


@pytest.mark.dynamic
def test_dynamic_update_chain(acceptance_config: AcceptanceConfig, adapter: AnnAdapter):
    failures: list[dict[str, Any]] = []
    for dataset in acceptance_config.datasets:
        failures.extend(_run_dataset_chain(acceptance_config, adapter, dataset))
    assert not failures, f"dynamic update failures: {failures[:5]}"


def _run_dataset_chain(
    config: AcceptanceConfig,
    adapter: AnnAdapter,
    dataset: DatasetConfig,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    root = (config.work_dir or config.results_dir / "work") / "dynamic" / dataset.name
    if root.exists():
        shutil.rmtree(root)
    root = ensure_dir(root)
    state_dir = ensure_dir(root / "state")
    generated = ensure_dir(root / "generated")
    write_label_schema(generated / "label_schema.json")
    batches = _dynamic_batches(config, dataset)
    live_labels = LiveLabelStore({})
    vector_sources: list[tuple[Path, Path | None]] = []
    vectors_by_id: dict[int, Any] = {}
    foreground_failures: list[dict[str, Any]] = []
    foreground_calibration = _load_static_foreground_selector(config, dataset)

    initial = batches[0]
    insert_rows, batch_ids_path, batch_vectors_path = _insert_batch(
        config,
        adapter,
        dataset,
        root,
        state_dir,
        live_labels,
        initial,
        cycle=0,
        phase="initial_insert",
        foreground_vectors_by_id=vectors_by_id,
        foreground_failures=foreground_failures,
        foreground_calibration=foreground_calibration,
    )
    live_labels.merge(insert_rows)
    vector_sources.append((batch_vectors_path, batch_ids_path))
    vectors_by_id = load_vector_sources(vector_sources, dataset.vector_format, dataset.dimensions)
    cycle0_failures, _ = _checkpoint(
        config, adapter, dataset, root, state_dir, live_labels, vectors_by_id, "cycle0"
    )
    failures.extend(cycle0_failures)

    for cycle in range(1, config.dynamic_cycles + 1):
        delete_ids = _choose_delete_ids(live_labels.ids, config.delete_fraction)
        if len(delete_ids) / max(1, live_labels.total()) < config.delete_fraction:
            row = {"dataset": dataset.name, "cycle": cycle, "failure_reason": "delete fraction too small"}
            _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
            failures.append(row)
        delete_ids_path = generated / f"cycle{cycle}_delete.ids"
        write_ids(delete_ids_path, delete_ids)
        delete_output_path = root / "mutations" / f"cycle{cycle}_delete.json"
        started = time.monotonic()
        running = adapter.start_delete(
            dataset_name=dataset.name,
            delete_ids_path=delete_ids_path,
            threads=dataset.build_threads,
            output_path=delete_output_path,
            state_dir=state_dir,
        )
        result = wait_with_foreground(
            running,
            lambda: _foreground_search(
                config,
                adapter,
                dataset,
                root,
                state_dir,
                live_labels,
                vectors_by_id,
                f"cycle{cycle}_delete",
                foreground_failures,
                foreground_calibration,
            ),
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if not result.ok:
            row = {"dataset": dataset.name, "cycle": cycle, "failure_reason": "delete command failed"}
            _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
            failures.append(row)
            delete_payload = {"live_count": -1}
        else:
            delete_payload = adapter.parse_mutation_output(delete_output_path)
        live_labels.delete(delete_ids)
        expected_live_count = live_labels.total()
        observed_live_count = int(delete_payload.get("live_count", -1))
        delete_ms_per_vector = elapsed_ms / max(1, len(delete_ids))
        delete_row = {
            "dataset": dataset.name,
            "cycle": cycle,
            "deleted_count": len(delete_ids),
            "wall_time_ms": elapsed_ms,
            "delete_ms_per_vector": delete_ms_per_vector,
            "expected_live_count": expected_live_count,
            "observed_live_count": observed_live_count,
            "pass": (
                delete_ms_per_vector <= config.thresholds.delete_ms_per_vector_max
                and observed_live_count == expected_live_count
            ),
            **result.resource_dict(),
        }
        append_jsonl(config.results_dir / "delete_api_timing.jsonl", delete_row)
        if observed_live_count != expected_live_count:
            row = {**delete_row, "failure_reason": "delete live_count mismatch"}
            _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
            failures.append(row)
        elif not delete_row["pass"]:
            row = {**delete_row, "failure_reason": "delete ms/vector above threshold"}
            _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
            failures.append(row)

        batch = batches[cycle]
        insert_rows, batch_ids_path, batch_vectors_path = _insert_batch(
            config,
            adapter,
            dataset,
            root,
            state_dir,
            live_labels,
            batch,
            cycle=cycle,
            phase="insert",
            foreground_vectors_by_id=vectors_by_id,
            foreground_failures=foreground_failures,
            foreground_calibration=foreground_calibration,
        )
        overlap = set(insert_rows.ids) & set(delete_ids)
        if overlap:
            row = {"dataset": dataset.name, "cycle": cycle, "failure_reason": "insert ids overlap deleted ids"}
            _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
            failures.append(row)
        live_labels.merge(insert_rows)
        vector_sources.append((batch_vectors_path, batch_ids_path))
        vectors_by_id = load_vector_sources(vector_sources, dataset.vector_format, dataset.dimensions)
        checkpoint_failures, _ = _checkpoint(
            config, adapter, dataset, root, state_dir, live_labels, vectors_by_id, f"cycle{cycle}"
        )
        failures.extend(checkpoint_failures)
    failures.extend(foreground_failures)
    return failures


def _dynamic_batches(config: AcceptanceConfig, dataset: DatasetConfig) -> list[VectorBatchConfig]:
    batches = list(dataset.insert_batches)
    if not batches:
        batches = [
            VectorBatchConfig(
                name="base_as_initial",
                vectors=dataset.base_vectors,
                ids=dataset.base_ids,
                count=config.target_npoints,
            )
        ]
    needed = config.dynamic_cycles + 1
    if len(batches) < needed:
        raise AssertionError(
            f"dataset {dataset.name} needs {needed} insert batches for dynamic test, got {len(batches)}"
        )
    return batches[:needed]


def _insert_batch(
    config: AcceptanceConfig,
    adapter: AnnAdapter,
    dataset: DatasetConfig,
    root: Path,
    state_dir: Path,
    live_labels: LiveLabelStore,
    batch: VectorBatchConfig,
    cycle: int,
    phase: str,
    foreground_vectors_by_id: dict[int, Any],
    foreground_failures: list[dict[str, Any]],
    foreground_calibration: dict[str, Any] | None,
) -> tuple[LiveLabelStore, Path, Path]:
    generated = ensure_dir(root / "generated")
    available = vector_count(batch.vectors, dataset.vector_format, dataset.dimensions)
    desired = config.target_npoints if live_labels.total() == 0 else config.target_npoints - live_labels.total()
    npoints = min(available, batch.count or desired, desired)
    if desired > 0 and npoints < desired:
        row = {
            "dataset": dataset.name,
            "cycle": cycle,
            "phase": phase,
            "available_vectors": available,
            "requested_insert_count": desired,
            "inserted_count": npoints,
            "failure_reason": "insert batch cannot restore target_npoints",
        }
        _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
        raise AssertionError(
            f"insert batch {batch.name} for {dataset.name} has {npoints} vectors, "
            f"but {desired} are required to restore target_npoints"
        )
    ids = read_ids(batch.ids, npoints)
    if len(ids) < npoints:
        row = {
            "dataset": dataset.name,
            "cycle": cycle,
            "phase": phase,
            "available_vectors": available,
            "requested_insert_count": desired,
            "inserted_count": len(ids),
            "failure_reason": "insert batch has fewer ids than required",
        }
        _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
        raise AssertionError(
            f"insert batch {batch.name} for {dataset.name} has {len(ids)} ids, "
            f"but {npoints} are required"
        )
    ids_path = generated / f"cycle{cycle}_{phase}.ids"
    write_ids(ids_path, ids)
    vectors_path = sliced_vector_path(
        batch.vectors,
        generated,
        f"cycle{cycle}_{phase}",
        dataset.vector_format,
    )
    write_vectors(
        vectors_path,
        load_vectors(batch.vectors, dataset.vector_format, dataset.dimensions)[:npoints],
        dataset.vector_format,
    )
    labels_path = generated / f"cycle{cycle}_{phase}.labels.csv"
    if live_labels.total() == 0:
        insert_labels = generate_labels_csv(ids, labels_path, config.label_seed + cycle)
    else:
        insert_labels = generate_rebalanced_insert_labels_csv(
            ids,
            live_labels,
            config.target_npoints,
            labels_path,
            config.label_seed + cycle,
        )
    output_path = root / "mutations" / f"cycle{cycle}_{phase}.json"
    started = time.monotonic()
    running = adapter.start_insert(
        dataset_name=dataset.name,
        insert_vectors_path=vectors_path,
        insert_ids_path=ids_path,
        labels_path=labels_path,
        threads=dataset.build_threads,
        output_path=output_path,
        state_dir=state_dir,
    )
    result = wait_with_foreground(
        running,
        lambda: _foreground_search(
            config,
            adapter,
            dataset,
            root,
            state_dir,
            live_labels,
            foreground_vectors_by_id,
            f"cycle{cycle}_{phase}",
            foreground_failures,
            foreground_calibration,
        ),
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if not result.ok:
        insert_payload = {"live_count": -1}
    else:
        insert_payload = adapter.parse_mutation_output(output_path)
    expected_live_count = live_labels.total() + len(ids)
    observed_live_count = int(insert_payload.get("live_count", -1))
    row = {
        "dataset": dataset.name,
        "cycle": cycle,
        "phase": phase,
        "inserted_count": len(ids),
        "wall_time_ms": elapsed_ms,
        "expected_live_count": expected_live_count,
        "observed_live_count": observed_live_count,
        "pass": result.ok and observed_live_count == expected_live_count,
        **result.resource_dict(),
    }
    append_jsonl(config.results_dir / "insert_api_timing.jsonl", row)
    if not result.ok:
        _append_failure(
            config.results_dir / "dynamic_update_failures.csv",
            {**row, "failure_reason": "insert command failed"},
        )
        raise AssertionError(f"insert command failed for {dataset.name} cycle {cycle}: {result.stderr}")
    if observed_live_count != expected_live_count:
        _append_failure(
            config.results_dir / "dynamic_update_failures.csv",
            {**row, "failure_reason": "insert live_count mismatch"},
        )
        raise AssertionError(
            f"insert live_count mismatch for {dataset.name} cycle {cycle}: "
            f"expected {expected_live_count}, observed {observed_live_count}"
        )
    return insert_labels, ids_path, vectors_path


def _checkpoint(
    config: AcceptanceConfig,
    adapter: AnnAdapter,
    dataset: DatasetConfig,
    root: Path,
    state_dir: Path,
    labels: LiveLabelStore,
    vectors_by_id: dict[int, Any],
    checkpoint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ctx = StaticContext(
        config=config,
        dataset=dataset,
        adapter=adapter,
        state_dir=state_dir,
        work_dir=root / "checkpoint" / checkpoint,
        labels_path=root / "unused.labels.csv",
        label_schema_path=root / "generated" / "label_schema.json",
        ids_path=root / "unused.ids",
        labels=labels,
        manifest={},
        vectors_by_id=vectors_by_id,
    )
    failures: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    for selector in selectivity_check_selectors():
        row = run_selectivity_case(ctx, selector, checkpoint)
        if not row["pass"]:
            failures.append({**row, "checkpoint": checkpoint})
    for selector in all_search_selectors():
        row = run_search_case(
            ctx,
            selector,
            config.full_query_limit,
            dataset.search_threads[0],
            checkpoint,
            compute_recall=True,
        )
        if row.get("invalid_selector"):
            continue
        recall_ok = row["recall_at_10"] >= config.thresholds.recall_at_10_min
        latency_ok = row["avg_latency_ms"] < config.thresholds.avg_latency_ms_lt
        row["checkpoint"] = checkpoint
        row["pass"] = recall_ok and latency_ok
        if not recall_ok:
            row["failure_reason"] = "recall below threshold"
        elif not latency_ok:
            row["failure_reason"] = "avg latency above threshold"
        append_jsonl(config.results_dir / "dynamic_update_chain_results.jsonl", row)
        search_rows.append(row)
        if not row["pass"]:
            _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
            failures.append(row)
    return failures, search_rows


def _load_static_foreground_selector(
    config: AcceptanceConfig,
    dataset: DatasetConfig,
) -> dict[str, Any]:
    artifact_path = config.results_dir / "static_foreground_worst_selectors.jsonl"
    if not artifact_path.exists():
        row = {
            "dataset": dataset.name,
            "expected_artifact": str(artifact_path),
            "failure_reason": (
                "static foreground worst selector artifact is missing in this results_dir; "
                "run static filtered search and dynamic in the same test run"
            ),
        }
        _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
        raise AssertionError(row["failure_reason"])
    artifact: dict[str, Any] | None = None
    with artifact_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = json.loads(line)
            if candidate.get("dataset") == dataset.name:
                artifact = candidate
    if artifact is None:
        row = {
            "dataset": dataset.name,
            "artifact_path": str(artifact_path),
            "failure_reason": "static foreground worst selector artifact has no row for dataset",
        }
        _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
        raise AssertionError(row["failure_reason"])
    selector_by_id = {selector["selector_id"]: selector for selector in all_search_selectors()}
    selector_id = str(artifact["selector_id"])
    if selector_id not in selector_by_id:
        row = {
            "dataset": dataset.name,
            "selector_id": selector_id,
            "artifact_path": str(artifact_path),
            "failure_reason": "static foreground worst selector is not in selector workload",
        }
        _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
        raise AssertionError(row["failure_reason"])
    selector = selector_by_id[selector_id]
    dynamic_artifact = {
        **artifact,
        "checkpoint": "static",
        "selector_id": selector_id,
        "selector_type": selector.get("selector_type"),
        "target_selectivity": selector.get("target_selectivity"),
        "artifact_path": str(artifact_path),
    }
    append_jsonl(config.results_dir / "dynamic_foreground_worst_selectors.jsonl", dynamic_artifact)
    return {"selector": selector, "artifact": dynamic_artifact}


def _foreground_search(
    config: AcceptanceConfig,
    adapter: AnnAdapter,
    dataset: DatasetConfig,
    root: Path,
    state_dir: Path,
    labels: LiveLabelStore,
    vectors_by_id: dict[int, Any],
    prefix: str,
    foreground_failures: list[dict[str, Any]] | None = None,
    foreground_calibration: dict[str, Any] | None = None,
) -> None:
    if labels.total() < config.k or not vectors_by_id:
        return
    if foreground_calibration is None:
        return
    counter_path = root / "foreground" / f"{prefix}.counter"
    ensure_dir(counter_path.parent)
    counter = int(counter_path.read_text(encoding="utf-8")) if counter_path.exists() else 0
    counter_path.write_text(str(counter + 1), encoding="utf-8")
    selector = foreground_calibration["selector"]
    calibration = foreground_calibration["artifact"]
    selector_path = root / "foreground" / f"{prefix}_{counter}.selector.json"
    output_path = root / "foreground" / f"{prefix}_{counter}.search.json"
    selector_to_json(selector_path, selector)
    threads = config.foreground_search_threads[counter % len(config.foreground_search_threads)]
    candidate_count = labels.count(selector)
    if candidate_count < config.k:
        row = {
            "dataset": dataset.name,
            "phase": prefix,
            "threads": threads,
            "selector_id": selector["selector_id"],
            "selector_type": selector.get("selector_type"),
            "target_selectivity": selector.get("target_selectivity"),
            "candidate_count": candidate_count,
            "foreground_selector_source": "calibrated_static_worst",
            "calibration_checkpoint": calibration["checkpoint"],
            "calibration_avg_latency_ms": calibration["calibration_avg_latency_ms"],
            "calibration_p95_latency_ms": calibration["calibration_p95_latency_ms"],
            "pass": False,
            "failure_reason": "foreground calibrated selector candidate count is smaller than k",
        }
        append_jsonl(config.results_dir / "dynamic_update_foreground_latency.jsonl", row)
        _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
        if foreground_failures is not None:
            foreground_failures.append(row)
        return
    try:
        payload, result = adapter.filter_search(
            dataset_name=dataset.name,
            query_file=dataset.query_file,
            selector_file=selector_path,
            k=config.k,
            query_limit=1,
            threads=threads,
            output_path=output_path,
            state_dir=state_dir,
        )
        _, summary = parse_search_output(payload, expected_query_count=1)
        row = {
            "dataset": dataset.name,
            "phase": prefix,
            "threads": threads,
            "selector_id": selector["selector_id"],
            "selector_type": selector.get("selector_type"),
            "target_selectivity": selector.get("target_selectivity"),
            "candidate_count": candidate_count,
            "avg_latency_ms": summary.get("avg_latency_ms"),
            "pass": float(summary.get("avg_latency_ms", 1e9)) < config.thresholds.avg_latency_ms_lt,
            "foreground_selector_source": "calibrated_static_worst",
            "calibration_checkpoint": calibration["checkpoint"],
            "calibration_avg_latency_ms": calibration["calibration_avg_latency_ms"],
            "calibration_p95_latency_ms": calibration["calibration_p95_latency_ms"],
            **result.resource_dict(),
        }
    except Exception as exc:
        row = {
            "dataset": dataset.name,
            "phase": prefix,
            "threads": threads,
            "selector_id": selector["selector_id"],
            "selector_type": selector.get("selector_type"),
            "target_selectivity": selector.get("target_selectivity"),
            "candidate_count": candidate_count,
            "foreground_selector_source": "calibrated_static_worst",
            "calibration_checkpoint": calibration["checkpoint"],
            "calibration_avg_latency_ms": calibration["calibration_avg_latency_ms"],
            "calibration_p95_latency_ms": calibration["calibration_p95_latency_ms"],
            "pass": False,
            "failure_reason": str(exc),
        }
    append_jsonl(config.results_dir / "dynamic_update_foreground_latency.jsonl", row)
    if not row["pass"]:
        _append_failure(config.results_dir / "dynamic_update_failures.csv", row)
        if foreground_failures is not None:
            foreground_failures.append(row)


def _choose_delete_ids(ids: list[int], fraction: float) -> list[int]:
    count = int(len(ids) * fraction)
    if count < len(ids) * fraction:
        count += 1
    return sorted(ids, key=lambda value: hashlib.blake2b(str(value).encode(), digest_size=8).digest())[:count]
