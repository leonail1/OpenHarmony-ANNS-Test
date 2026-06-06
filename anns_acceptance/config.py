from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


VectorFormat = Literal["npy", "fbin", "float32"]


class CommandSpec(BaseModel):
    command: str | list[str]
    timeout_seconds: float | None = None
    cwd: Path | None = None

    def resolve(self, base_dir: Path) -> "CommandSpec":
        cwd = self.cwd
        if cwd is not None and not cwd.is_absolute():
            cwd = (base_dir / cwd).resolve()
        return self.model_copy(update={"cwd": cwd})

    def contains_variable(self, name: str) -> bool:
        needle = "{" + name + "}"
        if isinstance(self.command, str):
            return needle in self.command
        return any(needle in part for part in self.command)


class AdapterManifest(BaseModel):
    commands: dict[str, CommandSpec]

    @model_validator(mode="after")
    def require_commands(self) -> "AdapterManifest":
        required = {
            "ann_build_index",
            "ann_filter_search",
            "ann_apply_insert",
            "ann_apply_delete",
            "ann_label_selectivity",
        }
        missing = sorted(required - set(self.commands))
        if missing:
            raise ValueError(f"adapter manifest missing commands: {', '.join(missing)}")
        missing_threads = sorted(
            name for name in required if not self.commands[name].contains_variable("threads")
        )
        if missing_threads:
            raise ValueError(
                "adapter commands must include a {threads} template variable: "
                + ", ".join(missing_threads)
            )
        return self

    def resolve(self, base_dir: Path) -> "AdapterManifest":
        return self.model_copy(
            update={"commands": {k: v.resolve(base_dir) for k, v in self.commands.items()}}
        )


class VectorBatchConfig(BaseModel):
    name: str
    vectors: Path
    ids: Path | None = None
    count: int | None = None

    def resolve(self, base_dir: Path) -> "VectorBatchConfig":
        return self.model_copy(
            update={
                "vectors": _resolve_path(self.vectors, base_dir),
                "ids": _resolve_optional_path(self.ids, base_dir),
            }
        )


class DatasetConfig(BaseModel):
    name: str
    base_vectors: Path
    base_ids: Path | None = None
    query_file: Path
    vector_format: VectorFormat = "npy"
    dimensions: int | None = None
    build_threads: int = 1
    search_threads: list[int] = Field(default_factory=lambda: [1])
    insert_batches: list[VectorBatchConfig] = Field(default_factory=list)

    @field_validator("search_threads")
    @classmethod
    def nonempty_threads(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("search_threads must not be empty")
        if any(v <= 0 for v in value):
            raise ValueError("thread counts must be positive")
        return value

    def resolve(self, base_dir: Path) -> "DatasetConfig":
        return self.model_copy(
            update={
                "base_vectors": _resolve_path(self.base_vectors, base_dir),
                "base_ids": _resolve_optional_path(self.base_ids, base_dir),
                "query_file": _resolve_path(self.query_file, base_dir),
                "insert_batches": [b.resolve(base_dir) for b in self.insert_batches],
            }
        )


class Thresholds(BaseModel):
    expansion_ratio_lt: float = 2.0
    recall_at_10_min: float = 0.98
    avg_latency_ms_lt: float = 10.0
    pilot_skip_latency_ms: float = 20.0
    delete_ms_per_vector_max: float = 0.5
    single_query_max_rss_bytes_lt: int = 30_000_000
    selectivity_abs_tolerance: float = 1e-9


class AcceptanceConfig(BaseModel):
    adapter_manifest: Path
    results_dir: Path = Path("results")
    work_dir: Path | None = None
    target_npoints: int = 1_000_000
    dynamic_cycles: int = 5
    delete_fraction: float = 0.60
    pilot_query_limit: int = 50
    full_query_limit: int = 1000
    foreground_search_threads: list[int] = Field(default_factory=lambda: [1, 8])
    k: int = 10
    label_seed: int = 20260605
    thresholds: Thresholds = Field(default_factory=Thresholds)
    datasets: list[DatasetConfig]

    @field_validator("foreground_search_threads")
    @classmethod
    def foreground_threads_valid(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("foreground_search_threads must not be empty")
        if any(v <= 0 for v in value):
            raise ValueError("thread counts must be positive")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> "AcceptanceConfig":
        if self.target_npoints <= 0:
            raise ValueError("target_npoints must be positive")
        if not 0 < self.delete_fraction <= 1:
            raise ValueError("delete_fraction must be in (0, 1]")
        if self.dynamic_cycles < 0:
            raise ValueError("dynamic_cycles must be non-negative")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not self.datasets:
            raise ValueError("at least one dataset is required")
        return self

    def resolve(self, base_dir: Path) -> "AcceptanceConfig":
        results_dir = _resolve_path(self.results_dir, base_dir)
        work_dir = self.work_dir or (results_dir / "work")
        work_dir = _resolve_path(work_dir, base_dir)
        return self.model_copy(
            update={
                "adapter_manifest": _resolve_path(self.adapter_manifest, base_dir),
                "results_dir": results_dir,
                "work_dir": work_dir,
                "datasets": [d.resolve(base_dir) for d in self.datasets],
            }
        )


def _resolve_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_optional_path(path: Path | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    return _resolve_path(path, base_dir)


def load_config(path: str | Path) -> AcceptanceConfig:
    config_path = Path(path).resolve()
    data = _load_mapping(config_path)
    config = AcceptanceConfig.model_validate(data).resolve(config_path.parent)
    return config


def load_adapter_manifest(path: str | Path) -> AdapterManifest:
    manifest_path = Path(path).resolve()
    data = _load_mapping(manifest_path)
    return AdapterManifest.model_validate(data).resolve(manifest_path.parent)


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        else:
            raise ValueError(f"unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data
