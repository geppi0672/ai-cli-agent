from __future__ import annotations

import os

from ._shell_exec import run_external_command
from .base import ExternalAgentAdapter, ExternalAgentTask
from ..models import StepResult


class AntigravityAdapter(ExternalAgentAdapter):
    name = "antigravity_agent"

    def __init__(self, binary: str | None = None) -> None:
        self.binary = binary or os.getenv("ANTIGRAVITY_CLI_BIN", "antigravity")

    def run(self, task: ExternalAgentTask) -> StepResult:
        # Assumes antigravity CLI can receive the task as a single prompt argument.
        command = [self.binary, task.instruction]
        return run_external_command(command, task.workdir)

