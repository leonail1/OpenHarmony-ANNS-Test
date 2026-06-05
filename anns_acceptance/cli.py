from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import load_config

app = typer.Typer(help="Run OpenHarmony ANNS acceptance tests.")
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Acceptance config JSON/YAML."),
    marker: Optional[str] = typer.Option(None, "--marker", "-m", help="Optional pytest marker."),
) -> None:
    cfg = load_config(config)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    test_path = Path(__file__).resolve().parent / "testsuite"
    args = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "anns_acceptance.pytest_plugin",
        str(test_path),
        "--config",
        str(config.resolve()),
    ]
    if marker:
        args.extend(["-m", marker])
    console.print(f"[bold]Running acceptance tests[/bold] with config: {config}")
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(args, cwd=Path(config).resolve().parent, env=env, check=False)
    raise typer.Exit(completed.returncode)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.argv.pop(1)
    app()


if __name__ == "__main__":
    main()
