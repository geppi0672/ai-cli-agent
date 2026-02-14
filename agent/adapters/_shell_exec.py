from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..models import StepResult


def run_external_command(command: list[str], workdir: Path) -> StepResult:
    binary = command[0]
    if shutil.which(binary) is None:
        return StepResult(False, f"Command not found: {binary}")

    completed = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        capture_output=True,
    )
    combined = (completed.stdout + completed.stderr).strip()
    output = combined[:6000] if combined else "(no output)"
    return StepResult(completed.returncode == 0, output)

