from __future__ import annotations

import os

from ._shell_exec import run_external_command
from .base import ExternalAgentAdapter, ExternalAgentTask
from ..models import StepResult


class CodexAdapter(ExternalAgentAdapter):
    name = "codex_agent"

    def __init__(self, binary: str | None = None) -> None:
        self.binary = binary or os.getenv("CODEX_CLI_BIN", "codex")

    def run(self, task: ExternalAgentTask) -> StepResult:
        # Assumes codex CLI can receive the task as a single prompt argument.
        command = [self.binary, task.instruction]
        return run_external_command(command, task.workdir)

