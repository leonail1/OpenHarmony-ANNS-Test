from __future__ import annotations

import pytest

from anns_acceptance.selectors import selectivity_check_selectors
from anns_acceptance.testsuite.conftest import run_selectivity_case


@pytest.mark.selectivity
def test_label_selectivity(static_contexts):
    failures = []
    for ctx in static_contexts:
        for selector in selectivity_check_selectors():
            row = run_selectivity_case(ctx, selector, "static")
            if not row["pass"]:
                failures.append(row)
    assert not failures, f"label selectivity failures: {failures[:5]}"
