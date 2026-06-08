from __future__ import annotations

import pytest

from anns_acceptance.artifacts import append_jsonl
from anns_acceptance.selectors import all_search_selectors
from anns_acceptance.testsuite.conftest import _append_failure, run_search_case


@pytest.mark.static
def test_static_filtered_search(static_contexts):
    failures = []
    for ctx in static_contexts:
        worst_candidates = []
        for selector in all_search_selectors():
            threads = ctx.dataset.search_threads[0]
            pilot = run_search_case(
                ctx,
                selector,
                ctx.config.pilot_query_limit,
                threads,
                "pilot",
                compute_recall=True,
            )
            if pilot.get("invalid_selector"):
                append_jsonl(ctx.config.results_dir / "static_filtered_search_results.jsonl", pilot)
                continue
            if pilot["avg_latency_ms"] > ctx.config.thresholds.pilot_skip_latency_ms:
                pilot["failure_reason"] = "pilot latency exceeds skip threshold"
                pilot["pass"] = False
                pilot["search_stage"] = "pilot"
                append_jsonl(ctx.config.results_dir / "static_filtered_search_results.jsonl", pilot)
                worst_candidates.append(pilot)
                _append_failure(ctx.config.results_dir / "static_filtered_search_failures.csv", pilot)
                failures.append(pilot)
                continue
            full = run_search_case(
                ctx,
                selector,
                ctx.config.full_query_limit,
                threads,
                "full",
                compute_recall=True,
            )
            recall_ok = full["recall_at_10"] >= ctx.config.thresholds.recall_at_10_min
            latency_ok = full["avg_latency_ms"] < ctx.config.thresholds.avg_latency_ms_lt
            full["pass"] = recall_ok and latency_ok
            if not recall_ok:
                full["failure_reason"] = "recall below threshold"
            elif not latency_ok:
                full["failure_reason"] = "avg latency above threshold"
            full["search_stage"] = "full"
            append_jsonl(ctx.config.results_dir / "static_filtered_search_results.jsonl", full)
            append_jsonl(ctx.config.results_dir / "static_filtered_search_resource.jsonl", full)
            worst_candidates.append(full)
            if not full["pass"]:
                _append_failure(ctx.config.results_dir / "static_filtered_search_failures.csv", full)
                failures.append(full)
        _write_worst_selector(ctx, worst_candidates)
    assert not failures, f"static filtered search failures: {failures[:5]}"


def _write_worst_selector(ctx, rows):
    candidates = [row for row in rows if row.get("avg_latency_ms") is not None]
    if not candidates:
        return
    worst = max(
        candidates,
        key=lambda row: (
            float(row.get("avg_latency_ms") or float("-inf")),
            float(row.get("p95_latency_ms") or float("-inf")),
            float(row.get("p99_latency_ms") or float("-inf")),
            str(row.get("selector_id", "")),
        ),
    )
    append_jsonl(
        ctx.config.results_dir / "static_foreground_worst_selectors.jsonl",
        {
            "dataset": ctx.dataset.name,
            "selector_id": worst.get("selector_id"),
            "selector_type": worst.get("selector_type"),
            "target_selectivity": worst.get("target_selectivity"),
            "candidate_count": worst.get("candidate_count"),
            "calibration_avg_latency_ms": worst.get("avg_latency_ms"),
            "calibration_p95_latency_ms": worst.get("p95_latency_ms"),
            "calibration_p99_latency_ms": worst.get("p99_latency_ms"),
            "recall_at_10": worst.get("recall_at_10"),
            "source_test": "static_filtered_search",
            "source_stage": worst.get("search_stage"),
        },
    )
