from __future__ import annotations

from collections.abc import Callable

from .budget import BudgetManager
from .history import HistoryCompressor
from .memory import JsonlMemory
from .models import AgentDecision, StepResult
from .providers import PlannerProvider
from .router import RouteDecision, Router
from .tools import ToolRunner


class AgentRunner:
    def __init__(
        self,
        objective: str,
        planner: PlannerProvider,
        tools: ToolRunner,
        memory: JsonlMemory,
        budget: BudgetManager,
        router: Router,
        compressor: HistoryCompressor,
        on_event: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        forced_allowed_tools: list[str] | None = None,
        noop_streak_limit: int = 0,
    ) -> None:
        self.objective = objective
        self.planner = planner
        self.tools = tools
        self.memory = memory
        self.budget = budget
        self.router = router
        self.compressor = compressor
        self.on_event = on_event
        self.should_stop = should_stop
        self.forced_allowed_tools = forced_allowed_tools
        self.noop_streak_limit = max(0, noop_streak_limit)
        self.history: list[dict] = []

    def run(self) -> str:
        step = 1
        invalid_tool_streak = 0
        noop_streak = 0
        while True:
            stop_reason = self.budget.check_before_step(step)
            if stop_reason:
                self.memory.append(
                    {
                        "event": "budget_stop",
                        "step": step,
                        "reason": stop_reason,
                        "budget": self.budget.snapshot().__dict__,
                    }
                )
                self._emit(
                    {
                        "event": "budget_stop",
                        "step": step,
                        "reason": stop_reason,
                        "budget": self.budget.snapshot().__dict__,
                    }
                )
                return stop_reason
            if self.should_stop is not None and self.should_stop():
                cancelled = "Stopped: cancellation requested."
                self.memory.append({"event": "cancel_stop", "step": step, "reason": cancelled})
                self._emit({"event": "cancel_stop", "step": step, "reason": cancelled})
                return cancelled

            route = self.router.decide(self.objective, self.history, self.tools, self.budget)
            if self.forced_allowed_tools is not None:
                allowed = [name for name in route.allowed_tools if name in self.forced_allowed_tools]
                route.allowed_tools = allowed
                route.mode = f"{route.mode}+forced"
                route.note = f"{route.note} Forced allowed tools={allowed}."
            decision = self._decide(route)
            tool = decision.action.tool
            args = decision.action.args

            self.memory.append(
                {
                    "event": "decision",
                    "step": step,
                    "route_mode": route.mode,
                    "route_note": route.note,
                    "thought": decision.thought,
                    "tool": tool,
                    "args": args,
                    "budget": self.budget.snapshot().__dict__,
                }
            )
            self._emit(
                {
                    "event": "decision",
                    "step": step,
                    "route_mode": route.mode,
                    "route_note": route.note,
                    "tool": tool,
                    "args": args,
                    "thought": decision.thought,
                    "budget": self.budget.snapshot().__dict__,
                }
            )

            if tool == "finish":
                final = str(args.get("final_message", "Done."))
                self.memory.append({"event": "finish", "step": step, "message": final})
                self._emit({"event": "finish", "step": step, "message": final})
                return final

            available_tools = self.tools.available_tools()
            if not tool.strip() or tool not in available_tools:
                invalid_tool_streak += 1
                result = StepResult(
                    True,
                    f"No-op: planner returned invalid tool '{tool or '(empty)'}'; replanning.",
                )
            elif tool not in route.allowed_tools:
                invalid_tool_streak = 0
                _ = self.tools.run(
                    "append_file",
                    {
                        "path": "runs/router-blocked.log",
                        "content": f"blocked tool={tool} step={step} reason={route.note}\n",
                    },
                )
                result = StepResult(False, f"Tool blocked by router: {tool}. {route.note}")
            else:
                invalid_tool_streak = 0
                result = self.tools.run(tool, args)

            if str(result.output).lower().startswith("no-op:"):
                noop_streak += 1
            else:
                noop_streak = 0

            self.budget.record_tool_use(is_external=self.tools.is_external_tool(tool) and result.ok)
            record = {
                "thought": decision.thought,
                "tool": tool,
                "command": str(args.get("command", "")) if tool == "shell" else "",
                "ok": result.ok,
                "output": result.output,
            }
            self.history.append(record)
            self.memory.append(
                {
                    "event": "observation",
                    "step": step,
                    "ok": result.ok,
                    "output": result.output,
                    "budget": self.budget.snapshot().__dict__,
                }
            )
            self._emit(
                {
                    "event": "observation",
                    "step": step,
                    "tool": tool,
                    "ok": result.ok,
                    "output": result.output,
                    "budget": self.budget.snapshot().__dict__,
                }
            )
            print(f"\n[step {step}] {tool} ok={result.ok}\n{result.output[:500]}\n")

            if _is_pytest_no_tests(tool, args, result.output):
                final = "Finished: pytest found no tests to run (collected 0 items)."
                self.memory.append({"event": "finish", "step": step, "message": final})
                self._emit({"event": "finish", "step": step, "message": final})
                return final
            if invalid_tool_streak >= 3:
                final = "Stopped: planner kept returning invalid tools."
                self.memory.append({"event": "finish", "step": step, "message": final})
                self._emit({"event": "finish", "step": step, "message": final})
                return final
            if self.noop_streak_limit > 0 and noop_streak >= self.noop_streak_limit:
                final = f"Finished: no-op repeated {noop_streak} times."
                self.memory.append({"event": "finish", "step": step, "message": final})
                self._emit({"event": "finish", "step": step, "message": final})
                return final
            if _is_repeated_pytest_success(self.history, repeat_threshold=2):
                final = "Finished: pytest success repeated; stopping redundant test loop."
                self.memory.append({"event": "finish", "step": step, "message": final})
                self._emit({"event": "finish", "step": step, "message": final})
                return final
            step += 1

    def _decide(self, route: RouteDecision) -> AgentDecision:
        history_text = self.compressor.compress(self.history)
        enriched_history = f"Routing:\n{route.note}\n\n{history_text}"
        return self.planner.decide(self.objective, enriched_history, route.allowed_tools)

    def _emit(self, event: dict) -> None:
        if self.on_event is None:
            return
        self.on_event(event)


def _is_pytest_no_tests(tool: str, args: dict, output: str) -> bool:
    if tool != "shell":
        return False
    command = str(args.get("command", "")).lower()
    if "pytest" not in command:
        return False
    text = output.lower()
    if "error:" in text or "file or directory not found" in text:
        return False
    return "collected 0 items" in text or "no tests ran" in text


def _is_repeated_pytest_success(history: list[dict], repeat_threshold: int = 2) -> bool:
    if len(history) < repeat_threshold:
        return False
    streak = 0
    last_sig = ""
    for item in reversed(history):
        if item.get("tool") != "shell" or not item.get("ok"):
            break
        command = str(item.get("command", "")).lower()
        output = str(item.get("output", "")).lower()
        if "pytest" not in command:
            break
        if "passed" not in output and "no tests ran" not in output:
            break
        sig = output[:220]
        if not last_sig:
            last_sig = sig
        elif sig != last_sig:
            break
        streak += 1
        if streak >= repeat_threshold:
            return True
    return False
