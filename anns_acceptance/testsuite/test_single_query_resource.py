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
                psutil_rss = row.get("psutil_max_rss_bytes")
                time_v_rss = row.get("time_v_max_rss_bytes")
                time_v_recorded = time_v_rss is not None
                psutil_recorded = psutil_rss is not None
                rss_delta = row.get("rss_measurement_delta_bytes")
                rss_ratio = row.get("rss_measurement_ratio")
                rss_ok = time_v_recorded and int(time_v_rss) < ctx.config.thresholds.single_query_max_rss_bytes_lt
                if not ctx.config.thresholds.require_psutil_rss_sanity:
                    psutil_sanity_ok = True
                elif time_v_recorded and psutil_recorded:
                    psutil_sanity_ok = (
                        int(rss_delta or 0) <= ctx.config.thresholds.rss_measurement_max_abs_diff_bytes
                        or float(rss_ratio or float("inf")) <= ctx.config.thresholds.rss_measurement_max_ratio
                    )
                else:
                    psutil_sanity_ok = False
                row["single_query_max_rss_bytes_lt"] = ctx.config.thresholds.single_query_max_rss_bytes_lt
                row["require_time_v_rss_measurement"] = ctx.config.thresholds.require_time_v_rss_measurement
                row["require_psutil_rss_sanity"] = ctx.config.thresholds.require_psutil_rss_sanity
                row["rss_measurement_max_abs_diff_bytes"] = (
                    ctx.config.thresholds.rss_measurement_max_abs_diff_bytes
                )
                row["rss_measurement_max_ratio"] = ctx.config.thresholds.rss_measurement_max_ratio
                row["pass"] = latency_ok and rss_ok and psutil_sanity_ok
                if not time_v_recorded and ctx.config.thresholds.require_time_v_rss_measurement:
                    row["failure_reason"] = "GNU time-v RSS was not recorded"
                elif not rss_ok:
                    row["failure_reason"] = "single query time-v max RSS above threshold"
                elif not psutil_recorded and ctx.config.thresholds.require_psutil_rss_sanity:
                    row["failure_reason"] = "psutil RSS sanity measurement was not recorded"
                elif not psutil_sanity_ok:
                    row["failure_reason"] = "psutil sanity RSS differs too much from time-v RSS"
                elif not latency_ok:
                    row["failure_reason"] = "single query latency above threshold"
                append_jsonl(ctx.config.results_dir / "single_query_latency.jsonl", row)
                append_jsonl(ctx.config.results_dir / "single_query_resource.jsonl", row)
                if not row["pass"]:
                    _append_failure(ctx.config.results_dir / "single_query_resource_failures.csv", row)
                    failures.append(row)
    assert not failures, f"single query resource failures: {failures[:5]}"
