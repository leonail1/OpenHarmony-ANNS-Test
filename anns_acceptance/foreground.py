from __future__ import annotations

import time
from collections.abc import Callable

from .command import CommandResult, RunningCommand


def wait_with_foreground(
    running: RunningCommand,
    foreground: Callable[[], None],
    poll_interval_s: float = 0.05,
) -> CommandResult:
    ran_foreground = False
    deadline = None
    if running.timeout_seconds is not None:
        deadline = time.monotonic() + running.timeout_seconds
    while running.poll() is None:
        if deadline is not None and time.monotonic() > deadline:
            running.popen.kill()
            return running.wait(timeout=5)
        foreground()
        ran_foreground = True
        time.sleep(poll_interval_s)
    if not ran_foreground:
        foreground()
    return running.wait(timeout=0)
