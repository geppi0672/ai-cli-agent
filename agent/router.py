from __future__ import annotations

import re
from dataclasses import dataclass

from .budget import BudgetManager
from .tools import ToolRunner


COMPLEXITY_HINT_RE = re.compile(
    r"(実装|設計|修正|改修|リファクタ|最適化|テスト失敗|不具合|bug|fix|refactor|migrate|architecture)",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    allowed_tools: list[str]
    mode: str
    note: str


class Router:
    def __init__(self, escalate_after_failures: int = 1) -> None:
        self.escalate_after_failures = max(1, escalate_after_failures)

    def decide(
        self,
        objective: str,
        history: list[dict],
        tools: ToolRunner,
        budget: BudgetManager,
    ) -> RouteDecision:
        all_tools = tools.available_tools()
        external = tools.external_tool_names()
        base_tools = [name for name in all_tools if name not in external]

        if not external:
            return RouteDecision(
                allowed_tools=all_tools,
                mode="base_only",
                note="No external adapters enabled.",
            )

        if not budget.can_call_external():
            return RouteDecision(
                allowed_tools=base_tools,
                mode="base_only",
                note="External call budget exhausted; continue with base tools only.",
            )

        failures = _failure_streak(history)
        complex_objective = bool(COMPLEXITY_HINT_RE.search(objective))
        stalled = _looks_stalled(history)
        should_escalate = failures >= self.escalate_after_failures or (complex_objective and stalled)

        if should_escalate:
            return RouteDecision(
                allowed_tools=all_tools,
                mode="escalated",
                note=(
                    "External adapters allowed. Escalation triggered by "
                    f"failures={failures}, stalled={stalled}, complex_objective={complex_objective}."
                ),
            )

        return RouteDecision(
            allowed_tools=base_tools,
            mode="base_only",
            note=(
                "Stay on cheap/base tools first. "
                f"failures={failures}, stalled={stalled}, complex_objective={complex_objective}."
            ),
        )


def _failure_streak(history: list[dict]) -> int:
    streak = 0
    for item in reversed(history):
        if item.get("ok"):
            break
        streak += 1
    return streak


def _looks_stalled(history: list[dict]) -> bool:
    if len(history) < 3:
        return False
    recent = history[-3:]
    outputs = [str(item.get("output", "")).strip().lower()[:140] for item in recent]
    same_output = len(set(outputs)) == 1
    all_fail = all(not item.get("ok") for item in recent)
    return same_output or all_fail

