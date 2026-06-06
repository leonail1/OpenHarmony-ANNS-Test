from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from anns_acceptance.adapter import AnnAdapter
from anns_acceptance.artifacts import append_jsonl, ensure_dir, write_csv, write_json
from anns_acceptance.config import AcceptanceConfig, DatasetConfig, load_adapter_manifest, load_config
from anns_acceptance.groundtruth import (
    compute_groundtruth,
    load_vector_sources,
    load_vectors,
    sliced_vector_path,
    vector_count,
    write_vectors,
)
from anns_acceptance.labels import (
    LiveLabelStore,
    generate_labels_csv,
    read_ids,
    selector_to_json,
    write_ids,
    write_label_schema,
)
from anns_acceptance.metrics import parse_search_output, recall_at_k
from anns_acceptance.selectivity import compare_selectivity, expected_selectivity


@dataclass
class StaticContext:
    config: AcceptanceConfig
    dataset: DatasetConfig
    adapter: AnnAdapter
    state_dir: Path
    work_dir: Path
    labels_path: Path
    label_schema_path: Path
    ids_path: Path
    labels: LiveLabelStore
    manifest: dict[str, Any]
    vectors_by_id: dict[int, Any]


@pytest.fixture(scope="session")
def acceptance_config(pytestconfig: pytest.Config) -> AcceptanceConfig:
    path = pytestconfig.getoption("--config")
    if not path:
        pytest.skip("--config is required for acceptance tests")
    cfg = load_config(path)
    if cfg.results_dir.exists():
        shutil.rmtree(cfg.results_dir)
    ensure_dir(cfg.results_dir)
    ensure_dir(cfg.work_dir or cfg.results_dir / "work")
    return cfg


@pytest.fixture(scope="session")
def adapter(acceptance_config: AcceptanceConfig) -> AnnAdapter:
    manifest = load_adapter_manifest(acceptance_config.adapter_manifest)
    return AnnAdapter(
        manifest,
        {
            "python": sys.executable,
            "repo_root": Path(__file__).resolve().parents[2],
            "results_dir": acceptance_config.results_dir,
            "work_dir": acceptance_config.work_dir,
        },
    )


@pytest.fixture(scope="session")
def static_contexts(acceptance_config: AcceptanceConfig, adapter: AnnAdapter) -> list[StaticContext]:
    contexts: list[StaticContext] = []
    for dataset in acceptance_config.datasets:
        contexts.append(_prepare_static_context(acceptance_config, adapter, dataset))
    return contexts


def _prepare_static_context(
    config: AcceptanceConfig,
    adapter: AnnAdapter,
    dataset: DatasetConfig,
) -> StaticContext:
    root = (config.work_dir or config.results_dir / "work") / "static" / dataset.name
    if root.exists():
        shutil.rmtree(root)
    root = ensure_dir(root)
    state_dir = ensure_dir(root / "state")
    index_dir = ensure_dir(root / "index")
    generated = ensure_dir(root / "generated")
    npoints = vector_count(dataset.base_vectors, dataset.vector_format, dataset.dimensions)
    npoints = min(npoints, config.target_npoints)
    ids = read_ids(dataset.base_ids, npoints)
    vectors_path = sliced_vector_path(dataset.base_vectors, generated, "base", dataset.vector_format)
    write_vectors(
        vectors_path,
        load_vectors(dataset.base_vectors, dataset.vector_format, dataset.dimensions)[:npoints],
        dataset.vector_format,
    )
    ids_path = generated / "base.ids"
    write_ids(ids_path, ids)
    labels_path = generated / "labels.csv"
    labels = generate_labels_csv(ids, labels_path, config.label_seed)
    label_schema_path = generated / "label_schema.json"
    write_label_schema(label_schema_path)
    output_manifest_path = root / "build_manifest.json"
    manifest, command_result = adapter.build_index(
        dataset_name=dataset.name,
        vectors_path=vectors_path,
        ids_path=ids_path,
        labels_path=labels_path,
        label_schema_path=label_schema_path,
        threads=dataset.build_threads,
        output_manifest_path=output_manifest_path,
        state_dir=state_dir,
        index_dir=index_dir,
    )
    append_jsonl(
        config.results_dir / "build_resource.jsonl",
        {
            "dataset": dataset.name,
            "command": command_result.command,
            **command_result.resource_dict(),
        },
    )
    vectors_by_id = load_vector_sources(
        [(vectors_path, ids_path)], dataset.vector_format, dataset.dimensions
    )
    return StaticContext(
        config=config,
        dataset=dataset,
        adapter=adapter,
        state_dir=state_dir,
        work_dir=root,
        labels_path=labels_path,
        label_schema_path=label_schema_path,
        ids_path=ids_path,
        labels=labels,
        manifest=manifest,
        vectors_by_id=vectors_by_id,
    )


def run_selectivity_case(ctx: StaticContext, selector: dict[str, Any], artifact_prefix: str) -> dict[str, Any]:
    selector_path = ctx.work_dir / "selectors" / f"{artifact_prefix}_{selector['selector_id']}.json"
    output_path = ctx.work_dir / "selectivity" / f"{artifact_prefix}_{selector['selector_id']}.json"
    selector_to_json(selector_path, selector)
    observed, command_result = ctx.adapter.label_selectivity(
        dataset_name=ctx.dataset.name,
        selector_file=selector_path,
        threads=ctx.dataset.search_threads[0],
        output_path=output_path,
        state_dir=ctx.state_dir,
    )
    expected = expected_selectivity(ctx.labels, selector)
    ok, reason = compare_selectivity(
        expected, observed, ctx.config.thresholds.selectivity_abs_tolerance
    )
    row = {
        "dataset": ctx.dataset.name,
        "selector_id": selector["selector_id"],
        "selector_type": selector["selector_type"],
        "target_selectivity": selector.get("target_selectivity"),
        "expected_matched_count": expected["matched_count"],
        "observed_matched_count": observed.get("matched_count"),
        "expected_selectivity": expected["selectivity"],
        "observed_selectivity": observed.get("selectivity"),
        "pass": ok,
        "failure_reason": reason,
        **command_result.resource_dict(),
    }
    append_jsonl(ctx.config.results_dir / "label_selectivity_results.jsonl", row)
    if not ok:
        _append_failure(ctx.config.results_dir / "label_selectivity_failures.csv", row)
    return row


def run_search_case(
    ctx: StaticContext,
    selector: dict[str, Any],
    query_limit: int,
    threads: int,
    artifact_prefix: str,
    compute_recall: bool,
) -> dict[str, Any]:
    selector_path = ctx.work_dir / "selectors" / f"{artifact_prefix}_{selector['selector_id']}.json"
    output_path = ctx.work_dir / "search" / f"{artifact_prefix}_{selector['selector_id']}_t{threads}.json"
    selector_to_json(selector_path, selector)
    actual_query_limit = min(
        query_limit,
        vector_count(ctx.dataset.query_file, ctx.dataset.vector_format, ctx.dataset.dimensions),
    )
    if compute_recall and ctx.labels.count(selector) < ctx.config.k:
        return {
            "dataset": ctx.dataset.name,
            "selector_id": selector["selector_id"],
            "selector_type": selector["selector_type"],
            "target_selectivity": selector.get("target_selectivity"),
            "invalid_selector": True,
            "failure_reason": "candidate count is smaller than k",
        }
    payload, command_result = ctx.adapter.filter_search(
        dataset_name=ctx.dataset.name,
        query_file=ctx.dataset.query_file,
        selector_file=selector_path,
        k=ctx.config.k,
        query_limit=actual_query_limit,
        threads=threads,
        output_path=output_path,
        state_dir=ctx.state_dir,
    )
    results, summary = parse_search_output(payload, expected_query_count=actual_query_limit)
    recall = None
    if compute_recall:
        groundtruth = compute_groundtruth(
            ctx.vectors_by_id,
            ctx.dataset.query_file,
            ctx.dataset.vector_format,
            ctx.dataset.dimensions,
            ctx.labels,
            selector,
            ctx.config.k,
            actual_query_limit,
            ctx.work_dir / "groundtruth" / f"{artifact_prefix}_{selector['selector_id']}_t{threads}",
        )
        recall = recall_at_k(results, groundtruth, ctx.config.k)
    row = {
        "dataset": ctx.dataset.name,
        "selector_id": selector["selector_id"],
        "selector_type": selector["selector_type"],
        "target_selectivity": selector.get("target_selectivity"),
        "threads": threads,
        "query_limit": actual_query_limit,
        "recall_at_10": recall,
        **summary,
        **command_result.resource_dict(),
    }
    return row


def _append_failure(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "dataset",
        "selector_id",
        "selector_type",
        "target_selectivity",
        "cycle",
        "phase",
        "failure_reason",
        "available_vectors",
        "requested_insert_count",
        "inserted_count",
        "deleted_count",
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "recall_at_10",
        "candidate_count",
        "threads",
        "max_rss_bytes",
        "single_query_max_rss_bytes_lt",
        "foreground_selector_source",
        "calibration_checkpoint",
        "calibration_avg_latency_ms",
        "calibration_p95_latency_ms",
    ]
    current = {field: row.get(field) for field in fields}
    if path.exists():
        import csv

        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writerow(current)
    else:
        write_csv(path, [current], fields)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config_path = session.config.getoption("--config")
    if not config_path:
        return
    try:
        cfg = load_config(config_path)
    except Exception:
        return
    summary = {
        "exitstatus": exitstatus,
        "pass": exitstatus == 0,
        "artifacts": sorted(str(path.name) for path in cfg.results_dir.glob("*")),
    }
    failure_files = [
        "label_selectivity_failures.csv",
        "static_filtered_search_failures.csv",
        "dynamic_update_failures.csv",
        "single_query_resource_failures.csv",
    ]
    summary["failure_files"] = [name for name in failure_files if (cfg.results_dir / name).exists()]
    write_json(cfg.results_dir / "acceptance_summary.json", summary)
