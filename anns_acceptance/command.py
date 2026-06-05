from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import CommandSpec
from .resources import ProcessSampler, ResourceSample, parse_time_v


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    resource: ResourceSample
    command: str | list[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def resource_dict(self) -> dict[str, Any]:
        return asdict(self.resource)


class RunningCommand:
    def __init__(
        self,
        popen: subprocess.Popen[str],
        command: str | list[str],
        timeout_seconds: float | None = None,
    ):
        self.popen = popen
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.sampler = ProcessSampler(popen.pid)
        self._stdout: str | None = None
        self._stderr: str | None = None

    def poll(self) -> int | None:
        self.sampler.sample()
        return self.popen.poll()

    def wait(self, timeout: float | None = None) -> CommandResult:
        try:
            stdout, stderr = self.popen.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.popen.kill()
            stdout, stderr = self.popen.communicate()
        self._stdout = stdout
        self._stderr = stderr
        resource = self.sampler.finish()
        parsed = parse_time_v(stderr)
        if "max_rss_bytes" in parsed:
            resource.max_rss_bytes = int(parsed["max_rss_bytes"])
        if "user_cpu_s" in parsed:
            resource.user_cpu_s = float(parsed["user_cpu_s"])
        if "system_cpu_s" in parsed:
            resource.system_cpu_s = float(parsed["system_cpu_s"])
        return CommandResult(
            returncode=self.popen.returncode,
            stdout=stdout,
            stderr=stderr,
            resource=resource,
            command=self.command,
        )


def render_command(spec: CommandSpec, variables: dict[str, Any]) -> str | list[str]:
    safe_vars = {k: "" if v is None else str(v) for k, v in variables.items()}
    if isinstance(spec.command, str):
        return spec.command.format_map(_MissingAsEmpty(safe_vars))
    return [part.format_map(_MissingAsEmpty(safe_vars)) for part in spec.command]


def start_command(spec: CommandSpec, variables: dict[str, Any]) -> RunningCommand:
    command = render_command(spec, variables)
    cwd = str(spec.cwd) if spec.cwd else None
    if isinstance(command, str):
        popen = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        popen = subprocess.Popen(
            command,
            shell=False,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return RunningCommand(popen, command, spec.timeout_seconds)


def run_command(spec: CommandSpec, variables: dict[str, Any]) -> CommandResult:
    running = start_command(spec, variables)
    deadline = None if spec.timeout_seconds is None else time.monotonic() + spec.timeout_seconds
    while True:
        if running.poll() is not None:
            return running.wait(timeout=0)
        if deadline is not None and time.monotonic() > deadline:
            running.popen.kill()
            return running.wait(timeout=5)
        time.sleep(0.05)


class _MissingAsEmpty(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""
