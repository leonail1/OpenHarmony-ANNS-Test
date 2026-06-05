from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class ResourceSample:
    wall_time_s: float = 0.0
    max_rss_bytes: int | None = None
    user_cpu_s: float | None = None
    system_cpu_s: float | None = None
    read_bytes: int | None = None
    write_bytes: int | None = None


@dataclass
class ProcessSampler:
    pid: int
    started_at: float = field(default_factory=time.monotonic)
    max_rss_bytes: int = 0
    _proc: psutil.Process | None = field(init=False, default=None)
    _read_bytes_start: int | None = field(init=False, default=None)
    _write_bytes_start: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        try:
            self._proc = psutil.Process(self.pid)
            io_counters = getattr(self._proc, "io_counters", None)
            if io_counters is not None:
                io = io_counters()
                self._read_bytes_start = getattr(io, "read_bytes", None)
                self._write_bytes_start = getattr(io, "write_bytes", None)
        except (psutil.Error, OSError):
            self._proc = None

    def sample(self) -> None:
        if self._proc is None:
            return
        try:
            rss = self._proc.memory_info().rss
            self.max_rss_bytes = max(self.max_rss_bytes, rss)
            for child in self._proc.children(recursive=True):
                try:
                    self.max_rss_bytes = max(self.max_rss_bytes, child.memory_info().rss)
                except (psutil.Error, OSError):
                    continue
        except (psutil.Error, OSError):
            return

    def finish(self) -> ResourceSample:
        self.sample()
        wall = time.monotonic() - self.started_at
        sample = ResourceSample(wall_time_s=wall, max_rss_bytes=self.max_rss_bytes or None)
        if self._proc is None:
            return sample
        try:
            cpu = self._proc.cpu_times()
            sample.user_cpu_s = getattr(cpu, "user", None)
            sample.system_cpu_s = getattr(cpu, "system", None)
        except (psutil.Error, OSError):
            pass
        try:
            io_counters = getattr(self._proc, "io_counters", None)
            if io_counters is not None:
                io = io_counters()
                read_now = getattr(io, "read_bytes", None)
                write_now = getattr(io, "write_bytes", None)
                if read_now is not None and self._read_bytes_start is not None:
                    sample.read_bytes = max(0, read_now - self._read_bytes_start)
                if write_now is not None and self._write_bytes_start is not None:
                    sample.write_bytes = max(0, write_now - self._write_bytes_start)
        except (psutil.Error, OSError):
            pass
        return sample


def parse_time_v(stderr: str) -> dict[str, int | float]:
    """Parse GNU /usr/bin/time -v output when an adapter chooses to use it."""
    parsed: dict[str, int | float] = {}
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    if rss:
        parsed["max_rss_bytes"] = int(rss.group(1)) * 1024
    user = re.search(r"User time \(seconds\):\s*([0-9.]+)", stderr)
    if user:
        parsed["user_cpu_s"] = float(user.group(1))
    system = re.search(r"System time \(seconds\):\s*([0-9.]+)", stderr)
    if system:
        parsed["system_cpu_s"] = float(system.group(1))
    return parsed
