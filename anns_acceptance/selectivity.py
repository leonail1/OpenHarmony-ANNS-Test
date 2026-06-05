from __future__ import annotations

from typing import Any

from .labels import LiveLabelStore


def expected_selectivity(labels: LiveLabelStore, selector: dict[str, Any]) -> dict[str, Any]:
    matched = labels.count(selector)
    total = labels.total()
    return {
        "matched_count": matched,
        "total_live_count": total,
        "selectivity": 0.0 if total == 0 else matched / total,
    }


def compare_selectivity(
    expected: dict[str, Any],
    observed: dict[str, Any],
    tolerance: float,
) -> tuple[bool, str]:
    expected_count = int(expected["matched_count"])
    observed_count = int(observed.get("matched_count", -1))
    if expected_count != observed_count:
        return False, f"matched_count expected {expected_count}, observed {observed_count}"
    expected_total = int(expected["total_live_count"])
    observed_total = int(observed.get("total_live_count", -1))
    if expected_total != observed_total:
        return False, f"total_live_count expected {expected_total}, observed {observed_total}"
    expected_sel = float(expected["selectivity"])
    observed_sel = float(observed.get("selectivity", float("nan")))
    if abs(expected_sel - observed_sel) > tolerance:
        return False, f"selectivity expected {expected_sel}, observed {observed_sel}"
    return True, ""

