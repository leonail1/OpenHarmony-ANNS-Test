from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ensure_dir, write_json


SELECTIVITY_SPECS: list[tuple[str, float]] = [
    ("s0001", 0.0001),
    ("s001", 0.001),
    ("s01", 0.01),
    ("s05", 0.05),
    ("s10", 0.10),
    ("s25", 0.25),
    ("s50", 0.50),
    ("s100", 1.00),
]


def eq_field(spec: str) -> str:
    return f"eq_{spec}"


def int_a_field(spec: str) -> str:
    return f"int_{spec}_a"


def int_b_field(spec: str) -> str:
    return f"int_{spec}_b"


def label_fieldnames() -> list[str]:
    fields = ["id"]
    fields.extend(eq_field(name) for name, _ in SELECTIVITY_SPECS)
    fields.append("range_uniform")
    for name, _ in SELECTIVITY_SPECS:
        fields.extend([int_a_field(name), int_b_field(name)])
    return fields


def write_label_schema(path: Path) -> None:
    payload = {
        "version": 1,
        "distribution": "uniform",
        "selectivities": [{"name": name, "fraction": fraction} for name, fraction in SELECTIVITY_SPECS],
        "fields": label_fieldnames(),
    }
    write_json(path, payload)


def read_ids(path: Path | None, n: int | None = None) -> list[int]:
    if path is None:
        if n is None:
            raise ValueError("n is required when id path is not provided")
        return list(range(n))
    ids: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                line = line.split(",", 1)[0]
            if line.lower() == "id":
                continue
            ids.append(int(line))
            if n is not None and len(ids) >= n:
                break
    return ids


def write_ids(path: Path, ids: list[int]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for vector_id in ids:
            handle.write(f"{vector_id}\n")


def generate_labels_csv(ids: list[int], path: Path, seed: int) -> "LiveLabelStore":
    rows = _base_rows(ids)
    total = len(ids)
    for name, fraction in SELECTIVITY_SPECS:
        count = _target_count(total, fraction)
        selected = set(_stable_order(ids, f"{seed}:eq:{name}")[:count])
        selected_intersect = set(_stable_order(ids, f"{seed}:intersect:{name}")[:count])
        for vector_id in ids:
            row = rows[vector_id]
            row[eq_field(name)] = "1" if vector_id in selected else "0"
            row[int_a_field(name)] = "1" if vector_id in selected_intersect else "0"
            row[int_b_field(name)] = "1" if vector_id in selected_intersect else "0"
    order = _stable_order(ids, f"{seed}:range")
    if total:
        for rank, vector_id in enumerate(order):
            rows[vector_id]["range_uniform"] = f"{(rank + 0.5) / total:.12f}"
    _write_rows(path, rows)
    return LiveLabelStore(rows)


def generate_rebalanced_insert_labels_csv(
    insert_ids: list[int],
    survivor_store: "LiveLabelStore",
    target_total: int,
    path: Path,
    seed: int,
) -> "LiveLabelStore":
    rows = _base_rows(insert_ids)
    remaining_by_spec: dict[str, list[int]] = {
        name: _stable_order(insert_ids, f"{seed}:insert:eq:{name}") for name, _ in SELECTIVITY_SPECS
    }
    for name, fraction in SELECTIVITY_SPECS:
        target = _target_count(target_total, fraction)
        survivor_count = survivor_store.count(
            {"selector_type": "equality", "field": eq_field(name), "value": 1}
        )
        deficit = max(0, min(len(insert_ids), target - survivor_count))
        for vector_id in remaining_by_spec[name][:deficit]:
            rows[vector_id][eq_field(name)] = "1"

        survivor_intersect = survivor_store.count(
            {
                "selector_type": "intersect",
                "conditions": [
                    {"selector_type": "equality", "field": int_a_field(name), "value": 1},
                    {"selector_type": "equality", "field": int_b_field(name), "value": 1},
                ],
            }
        )
        intersect_deficit = max(0, min(len(insert_ids), target - survivor_intersect))
        for vector_id in _stable_order(insert_ids, f"{seed}:insert:intersect:{name}")[:intersect_deficit]:
            rows[vector_id][int_a_field(name)] = "1"
            rows[vector_id][int_b_field(name)] = "1"

    _rebalance_range(rows, insert_ids, survivor_store, target_total, seed)
    _write_rows(path, rows)
    return LiveLabelStore(rows)


def _rebalance_range(
    rows: dict[int, dict[str, str]],
    insert_ids: list[int],
    survivor_store: "LiveLabelStore",
    target_total: int,
    seed: int,
) -> None:
    ordered_insert = _stable_order(insert_ids, f"{seed}:insert:range")
    cursor = 0
    previous = 0.0
    previous_target = 0
    for _, fraction in SELECTIVITY_SPECS:
        target_cumulative = _target_count(target_total, fraction)
        target_bucket = target_cumulative - previous_target
        survivor_bucket = survivor_store.count_range_bucket(previous, fraction)
        deficit = max(0, target_bucket - survivor_bucket)
        midpoint = (previous + fraction) / 2.0 if fraction > previous else fraction
        for vector_id in ordered_insert[cursor : cursor + deficit]:
            rows[vector_id]["range_uniform"] = f"{midpoint:.12f}"
        cursor += deficit
        previous = fraction
        previous_target = target_cumulative
    for vector_id in ordered_insert[cursor:]:
        rows[vector_id]["range_uniform"] = "0.999999999999"


def _base_rows(ids: list[int]) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for vector_id in ids:
        row = {field: "0" for field in label_fieldnames()}
        row["id"] = str(vector_id)
        row["range_uniform"] = "0.999999999999"
        rows[vector_id] = row
    return rows


def _write_rows(path: Path, rows: dict[int, dict[str, str]]) -> None:
    ensure_dir(path.parent)
    fields = label_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for vector_id in sorted(rows):
            writer.writerow(rows[vector_id])


def _stable_order(ids: list[int], salt: str) -> list[int]:
    return sorted(ids, key=lambda vector_id: _stable_hash_int(f"{salt}:{vector_id}"))


def _stable_hash_int(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def _target_count(total: int, fraction: float) -> int:
    return min(total, max(0, int(round(total * fraction))))


@dataclass
class LiveLabelStore:
    rows: dict[int, dict[str, str]]

    @classmethod
    def from_csv(cls, path: Path) -> "LiveLabelStore":
        rows: dict[int, dict[str, str]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                vector_id = int(row["id"])
                rows[vector_id] = dict(row)
        return cls(rows)

    @property
    def ids(self) -> list[int]:
        return sorted(self.rows)

    def total(self) -> int:
        return len(self.rows)

    def merge(self, other: "LiveLabelStore") -> None:
        self.rows.update(other.rows)

    def delete(self, ids: list[int]) -> None:
        for vector_id in ids:
            self.rows.pop(vector_id, None)

    def write_csv(self, path: Path) -> None:
        _write_rows(path, self.rows)

    def count(self, selector: dict[str, Any]) -> int:
        selector_type = selector.get("selector_type")
        if selector_type == "match_all":
            return self.total()
        if selector_type == "equality":
            field = selector["field"]
            value = str(selector["value"])
            return sum(1 for row in self.rows.values() if str(row.get(field)) == value)
        if selector_type == "range":
            field = selector["field"]
            lower = float(selector.get("lower", float("-inf")))
            upper = float(selector.get("upper", float("inf")))
            return sum(1 for row in self.rows.values() if lower <= float(row.get(field, "nan")) <= upper)
        if selector_type == "intersect":
            return sum(1 for row in self.rows.values() if _matches(row, selector))
        raise ValueError(f"unsupported selector type: {selector_type}")

    def count_range_bucket(self, lower_exclusive: float, upper_inclusive: float) -> int:
        return sum(
            1
            for row in self.rows.values()
            if lower_exclusive < float(row["range_uniform"]) <= upper_inclusive
        )

    def matching_ids(self, selector: dict[str, Any]) -> list[int]:
        if selector.get("selector_type") == "match_all":
            return self.ids
        return sorted(vector_id for vector_id, row in self.rows.items() if _matches(row, selector))


def _matches(row: dict[str, str], selector: dict[str, Any]) -> bool:
    selector_type = selector.get("selector_type")
    if selector_type == "match_all":
        return True
    if selector_type == "equality":
        return str(row.get(selector["field"])) == str(selector["value"])
    if selector_type == "range":
        value = float(row.get(selector["field"], "nan"))
        return float(selector.get("lower", float("-inf"))) <= value <= float(
            selector.get("upper", float("inf"))
        )
    if selector_type == "intersect":
        return all(_matches(row, condition) for condition in selector.get("conditions", []))
    return False


def selector_to_json(path: Path, selector: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(selector, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

