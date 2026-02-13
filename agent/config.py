from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
    model: str = "gpt-4o-mini"
    max_steps: int = 20
    max_external_calls: int = 4
    max_seconds: int = 900
    compress_recent_steps: int = 6
    escalate_after_failures: int = 1
    workdir: Path = Path.cwd()
    auto_approve_safe: bool = True

    @classmethod
    def from_args(
        cls,
        model: str | None,
        max_steps: int | None,
        max_external_calls: int | None,
        max_seconds: int | None,
        compress_recent_steps: int | None,
        escalate_after_failures: int | None,
        workdir: str | None,
        auto_approve_safe: bool,
    ) -> "AgentConfig":
        resolved_model = model or os.getenv("OPENAI_MODEL", cls.model)
        resolved_steps = max_steps if max_steps is not None else cls.max_steps
        resolved_external_calls = (
            max_external_calls if max_external_calls is not None else cls.max_external_calls
        )
        resolved_seconds = max_seconds if max_seconds is not None else cls.max_seconds
        resolved_recent_steps = (
            compress_recent_steps
            if compress_recent_steps is not None
            else cls.compress_recent_steps
        )
        resolved_escalate_after_failures = (
            escalate_after_failures
            if escalate_after_failures is not None
            else cls.escalate_after_failures
        )
        resolved_workdir = Path(workdir).expanduser().resolve() if workdir else Path.cwd()
        return cls(
            model=resolved_model,
            max_steps=resolved_steps,
            max_external_calls=resolved_external_calls,
            max_seconds=resolved_seconds,
            compress_recent_steps=resolved_recent_steps,
            escalate_after_failures=resolved_escalate_after_failures,
            workdir=resolved_workdir,
            auto_approve_safe=auto_approve_safe,
        )
