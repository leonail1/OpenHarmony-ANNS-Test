from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--config", action="store", help="Acceptance config JSON/YAML")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "space: build output space expansion tests")
    config.addinivalue_line("markers", "selectivity: label selectivity consistency tests")
    config.addinivalue_line("markers", "static: static filtered search quality tests")
    config.addinivalue_line("markers", "dynamic: dynamic insert/delete chain tests")
    config.addinivalue_line("markers", "single_query: single query resource tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    marker_order = {
        "space": 0,
        "selectivity": 1,
        "static": 2,
        "dynamic": 3,
        "single_query": 4,
    }

    def order_key(item: pytest.Item) -> tuple[int, str]:
        marks = {mark.name for mark in item.iter_markers()}
        order = min((marker_order[name] for name in marks if name in marker_order), default=99)
        return order, item.nodeid

    items.sort(key=order_key)
