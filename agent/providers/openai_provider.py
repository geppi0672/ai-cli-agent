from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from ..models import Action, AgentDecision
from .base import PlannerProvider


BASE_SYSTEM_PROMPT = """You are an autonomous coding CLI agent.
Return ONLY valid JSON with this schema:
{
  "thought": "short rationale",
  "action": {
    "tool": "<one of allowed tools>",
    "args": { ... }
  }
}

Rules:
- Prefer one action at a time.
- Use relative paths whenever possible.
- If the goal is complete, use tool=finish with args={"final_message":"..."}.
- Never include markdown fences.
"""


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return json.loads(raw)

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"JSON not found in model output: {raw}")
    return json.loads(match.group(0))


class OpenAIProvider(PlannerProvider):
    def __init__(self, model: str) -> None:
        self.model = model
        self.client = OpenAI()

    def decide(self, objective: str, history_text: str, available_tools: list[str]) -> AgentDecision:
        tools_text = ", ".join(available_tools)
        system_prompt = f"{BASE_SYSTEM_PROMPT}\nAllowed tools: {tools_text}"
        user_prompt = (
            f"Objective:\n{objective}\n\n"
            f"History:\n{history_text}\n\n"
            "Decide the next single action."
        )
        parsed: dict[str, Any] | None = None
        last_error = ""
        for attempt in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            try:
                parsed = _extract_json(content)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                user_prompt += (
                    "\n\nYour previous output was invalid JSON. "
                    "Return only one valid JSON object with no extra text."
                )
        if parsed is None:
            return AgentDecision(
                thought=f"Planner JSON parse failed. {last_error}",
                action=Action(tool="finish", args={"final_message": f"Stopped: invalid planner JSON ({last_error})"}),
            )
        action = parsed.get("action", {})
        return AgentDecision(
            thought=str(parsed.get("thought", "")),
            action=Action(
                tool=str(action.get("tool", "")).strip(),
                args=action.get("args", {}) or {},
            ),
        )
