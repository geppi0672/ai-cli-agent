from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class BudgetSnapshot:
    max_steps: int
    max_external_calls: int
    max_seconds: int
    used_external_calls: int
    elapsed_seconds: int


class BudgetManager:
    def __init__(self, max_steps: int, max_external_calls: int, max_seconds: int) -> None:
        self.max_steps = max_steps
        self.max_external_calls = max_external_calls
        self.max_seconds = max_seconds
        self._start = time.monotonic()
        self._used_external_calls = 0

    def check_before_step(self, step: int) -> str | None:
        if step > self.max_steps:
            return f"Stopped: step budget exceeded ({self.max_steps})."
        if self.elapsed_seconds() > self.max_seconds:
            return f"Stopped: time budget exceeded ({self.max_seconds}s)."
        return None

    def can_call_external(self) -> bool:
        return self._used_external_calls < self.max_external_calls

    def record_tool_use(self, is_external: bool) -> None:
        if is_external:
            self._used_external_calls += 1

    def used_external_calls(self) -> int:
        return self._used_external_calls

    def elapsed_seconds(self) -> int:
        return int(time.monotonic() - self._start)

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            max_steps=self.max_steps,
            max_external_calls=self.max_external_calls,
            max_seconds=self.max_seconds,
            used_external_calls=self._used_external_calls,
            elapsed_seconds=self.elapsed_seconds(),
        )

