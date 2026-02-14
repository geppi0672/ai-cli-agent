from __future__ import annotations

from typing import Protocol

from ..models import AgentDecision


class PlannerProvider(Protocol):
    def decide(self, objective: str, history_text: str, available_tools: list[str]) -> AgentDecision:
        """Return the next action decision."""

