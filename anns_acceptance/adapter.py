from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import read_json
from .command import CommandResult, RunningCommand, run_command, start_command
from .config import AdapterManifest, CommandSpec


class AnnAdapter:
    def __init__(self, manifest: AdapterManifest, base_variables: dict[str, Any] | None = None):
        self.manifest = manifest
        self.base_variables = dict(base_variables or {})

    def build_index(self, **variables: Any) -> tuple[dict[str, Any], CommandResult]:
        output_manifest_path = Path(variables["output_manifest_path"])
        result = self._run("ann_build_index", variables)
        self._require_ok("ann_build_index", result)
        return read_json(output_manifest_path), result

    def filter_search(self, **variables: Any) -> tuple[dict[str, Any], CommandResult]:
        output_path = Path(variables["output_path"])
        result = self._run("ann_filter_search", variables)
        self._require_ok("ann_filter_search", result)
        return read_json(output_path), result

    def label_selectivity(self, **variables: Any) -> tuple[dict[str, Any], CommandResult]:
        output_path = Path(variables["output_path"])
        result = self._run("ann_label_selectivity", variables)
        self._require_ok("ann_label_selectivity", result)
        return read_json(output_path), result

    def apply_insert(self, **variables: Any) -> tuple[dict[str, Any], CommandResult]:
        output_path = Path(variables["output_path"])
        result = self._run("ann_apply_insert", variables)
        self._require_ok("ann_apply_insert", result)
        return read_json(output_path), result

    def apply_delete(self, **variables: Any) -> tuple[dict[str, Any], CommandResult]:
        output_path = Path(variables["output_path"])
        result = self._run("ann_apply_delete", variables)
        self._require_ok("ann_apply_delete", result)
        return read_json(output_path), result

    def start_insert(self, **variables: Any) -> RunningCommand:
        return self._start("ann_apply_insert", variables)

    def start_delete(self, **variables: Any) -> RunningCommand:
        return self._start("ann_apply_delete", variables)

    def parse_mutation_output(self, output_path: Path) -> dict[str, Any]:
        return read_json(output_path)

    def _run(self, name: str, variables: dict[str, Any]) -> CommandResult:
        return run_command(self._command(name), self._vars(variables))

    def _start(self, name: str, variables: dict[str, Any]) -> RunningCommand:
        return start_command(self._command(name), self._vars(variables))

    def _command(self, name: str) -> CommandSpec:
        return self.manifest.commands[name]

    def _vars(self, variables: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self.base_variables)
        merged.update(variables)
        return merged

    @staticmethod
    def _require_ok(name: str, result: CommandResult) -> None:
        if result.ok:
            return
        command = result.command if isinstance(result.command, str) else " ".join(result.command)
        raise RuntimeError(
            f"{name} failed with exit code {result.returncode}\n"
            f"command: {command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def load_json_output(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload

