from __future__ import annotations

from typing import Any

from .labels import SELECTIVITY_SPECS, eq_field, int_a_field, int_b_field


def all_search_selectors() -> list[dict[str, Any]]:
    selectors: list[dict[str, Any]] = [match_all_selector()]
    for name, fraction in SELECTIVITY_SPECS:
        selectors.append(equality_selector(name, fraction))
        selectors.append(range_selector(name, fraction))
        selectors.append(intersect_selector(name, fraction))
    return selectors


def selectivity_check_selectors() -> list[dict[str, Any]]:
    selectors: list[dict[str, Any]] = []
    for name, fraction in SELECTIVITY_SPECS:
        selectors.append(equality_selector(name, fraction))
        selectors.append(range_selector(name, fraction))
    return selectors


def typical_single_query_selectors() -> list[dict[str, Any]]:
    return [match_all_selector(), equality_selector("s10", 0.10), range_selector("s10", 0.10)]


def match_all_selector() -> dict[str, Any]:
    return {
        "selector_id": "match_all_s100",
        "selector_type": "match_all",
        "target_selectivity": 1.0,
    }


def equality_selector(name: str, fraction: float) -> dict[str, Any]:
    return {
        "selector_id": f"equality_{name}",
        "selector_type": "equality",
        "target_selectivity": fraction,
        "field": eq_field(name),
        "value": 1,
    }


def range_selector(name: str, fraction: float) -> dict[str, Any]:
    return {
        "selector_id": f"range_{name}",
        "selector_type": "range",
        "target_selectivity": fraction,
        "field": "range_uniform",
        "lower": 0.0,
        "upper": fraction,
    }


def intersect_selector(name: str, fraction: float) -> dict[str, Any]:
    return {
        "selector_id": f"intersect_{name}",
        "selector_type": "intersect",
        "target_selectivity": fraction,
        "conditions": [
            {
                "selector_type": "equality",
                "field": int_a_field(name),
                "value": 1,
            },
            {
                "selector_type": "equality",
                "field": int_b_field(name),
                "value": 1,
            },
        ],
    }

