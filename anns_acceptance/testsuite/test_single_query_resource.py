from __future__ import annotations

import pytest

from anns_acceptance.artifacts import append_jsonl
from anns_acceptance.selectors import typical_single_query_selectors
from anns_acceptance.testsuite.conftest import run_search_case


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
                row["pass"] = row["avg_latency_ms"] < ctx.config.thresholds.avg_latency_ms_lt
                append_jsonl(ctx.config.results_dir / "single_query_latency.jsonl", row)
                append_jsonl(ctx.config.results_dir / "single_query_resource.jsonl", row)
                if row.get("max_rss_bytes") is None:
                    row["pass"] = False
                    row["failure_reason"] = "max RSS was not recorded"
                elif not row["pass"]:
                    row["failure_reason"] = "single query latency above threshold"
                if not row["pass"]:
                    failures.append(row)
    assert not failures, f"single query resource failures: {failures[:5]}"
