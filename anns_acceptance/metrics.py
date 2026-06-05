from __future__ import annotations

import statistics
from typing import Any


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "avg_latency_ms": float("inf"),
            "p50_latency_ms": float("inf"),
            "p95_latency_ms": float("inf"),
            "p99_latency_ms": float("inf"),
        }
    values = sorted(float(v) for v in latencies_ms)
    return {
        "avg_latency_ms": statistics.fmean(values),
        "p50_latency_ms": percentile(values, 50),
        "p95_latency_ms": percentile(values, 95),
        "p99_latency_ms": percentile(values, 99),
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("inf")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = rank - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def recall_at_k(results: list[dict[str, Any]], groundtruth: dict[int, list[int]], k: int) -> float:
    if not groundtruth:
        return float("nan")
    recalls: list[float] = []
    for row in results:
        query_id = int(row.get("query_id", len(recalls)))
        gt = groundtruth.get(query_id, [])
        if not gt:
            continue
        returned = [int(v) for v in row.get("ids", [])[:k]]
        recalls.append(len(set(returned) & set(gt[:k])) / min(k, len(gt)))
    if not recalls:
        return float("nan")
    return statistics.fmean(recalls)


def parse_search_output(
    payload: dict[str, Any],
    expected_query_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("search output must contain a list field named results")
    if expected_query_count is not None:
        _validate_search_rows(results, expected_query_count)
    latencies = [float(row["latency_ms"]) for row in results if "latency_ms" in row]
    summary = dict(payload.get("summary") or {})
    # Acceptance latency is computed by the harness from per-query rows; adapter
    # summaries are retained only for non-latency fields.
    summary.update(latency_summary(latencies))
    return results, summary


def _validate_search_rows(results: list[dict[str, Any]], expected_query_count: int) -> None:
    if len(results) != expected_query_count:
        raise ValueError(f"search output returned {len(results)} rows, expected {expected_query_count}")
    seen: set[int] = set()
    for row in results:
        if "query_id" not in row:
            raise ValueError("search result row missing query_id")
        if "latency_ms" not in row:
            raise ValueError("search result row missing latency_ms")
        query_id = int(row["query_id"])
        if query_id in seen:
            raise ValueError(f"duplicate query_id in search output: {query_id}")
        seen.add(query_id)
        if "ids" not in row or not isinstance(row["ids"], list):
            raise ValueError(f"search result row {query_id} must contain list field ids")
    expected = set(range(expected_query_count))
    if seen != expected:
        missing = sorted(expected - seen)[:10]
        extra = sorted(seen - expected)[:10]
        raise ValueError(f"search query_id set mismatch; missing={missing}, extra={extra}")
