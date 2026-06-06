from __future__ import annotations

import pytest

from anns_acceptance.artifacts import append_jsonl
from anns_acceptance.selectors import typical_single_query_selectors
from anns_acceptance.testsuite.conftest import _append_failure, run_search_case


@pytest.mark.single_query
def test_single_query_resource(static_contexts):
    failures = []
    for ctx in static_contexts:
        for selector in typical_single_query_selectors():
            if ctx.labels.count(selector) < ctx.config.k and selector["selector_type"] != "match_all":
                continue
            for threads in ctx.config.foreground_search_threads:
                row = run_search_case(
                    ctx,
                    selector,
                    1,
                    threads,
                    "single_query",
                    compute_recall=False,
                )
                latency_ok = row["avg_latency_ms"] < ctx.config.thresholds.avg_latency_ms_lt
                rss = row.get("max_rss_bytes")
                rss_recorded = rss is not None
                rss_ok = rss_recorded and int(rss) < ctx.config.thresholds.single_query_max_rss_bytes_lt
                row["single_query_max_rss_bytes_lt"] = ctx.config.thresholds.single_query_max_rss_bytes_lt
                row["pass"] = latency_ok and rss_ok
                if not rss_recorded:
                    row["failure_reason"] = "max RSS was not recorded"
                elif not rss_ok:
                    row["failure_reason"] = "single query max RSS above threshold"
                elif not latency_ok:
                    row["failure_reason"] = "single query latency above threshold"
                append_jsonl(ctx.config.results_dir / "single_query_latency.jsonl", row)
                append_jsonl(ctx.config.results_dir / "single_query_resource.jsonl", row)
                if not row["pass"]:
                    _append_failure(ctx.config.results_dir / "single_query_resource_failures.csv", row)
                    failures.append(row)
    assert not failures, f"single query resource failures: {failures[:5]}"
