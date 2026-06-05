from __future__ import annotations

import pytest

from anns_acceptance.artifacts import write_csv, write_json
from anns_acceptance.space import audit_space


@pytest.mark.space
def test_build_space(static_contexts):
    rows: list[dict] = []
    failures: list[dict] = []
    for ctx in static_contexts:
        audit = audit_space(ctx.manifest)
        audit["dataset"] = ctx.dataset.name
        audit["pass"] = audit["expansion_ratio"] < ctx.config.thresholds.expansion_ratio_lt
        rows.append(audit)
        if not audit["pass"]:
            failures.append(audit)
    if rows:
        write_json(ctx.config.results_dir / "space_audit.json", rows)
        write_csv(ctx.config.results_dir / "space_audit.csv", rows)
    assert not failures, f"space expansion failures: {failures}"
