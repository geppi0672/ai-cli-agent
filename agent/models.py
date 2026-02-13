from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecision:
    thought: str
    action: Action


@dataclass
class StepResult:
    ok: bool
    output: str

