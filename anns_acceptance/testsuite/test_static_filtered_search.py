from __future__ import annotations

import pytest

from anns_acceptance.artifacts import append_jsonl
from anns_acceptance.selectors import all_search_selectors
from anns_acceptance.testsuite.conftest import _append_failure, run_search_case


@pytest.mark.static
def test_static_filtered_search(static_contexts):
    failures = []
    for ctx in static_contexts:
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
                append_jsonl(ctx.config.results_dir / "static_filtered_search_results.jsonl", pilot)
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
            append_jsonl(ctx.config.results_dir / "static_filtered_search_results.jsonl", full)
            append_jsonl(ctx.config.results_dir / "static_filtered_search_resource.jsonl", full)
            if not full["pass"]:
                _append_failure(ctx.config.results_dir / "static_filtered_search_failures.csv", full)
                failures.append(full)
    assert not failures, f"static filtered search failures: {failures[:5]}"
