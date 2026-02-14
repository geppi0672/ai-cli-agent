from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..models import StepResult


@dataclass
class ExternalAgentTask:
    instruction: str
    workdir: Path


class ExternalAgentAdapter(Protocol):
    name: str

    def run(self, task: ExternalAgentTask) -> StepResult:
        """Execute an external agent task."""

