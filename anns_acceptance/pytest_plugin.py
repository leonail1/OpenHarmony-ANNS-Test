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
