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
        self.history: list[dict] = []

    def run(self) -> str:
        step = 1
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

            if tool not in route.allowed_tools:
                _ = self.tools.run(
                    "append_file",
                    {
                        "path": "runs/router-blocked.log",
                        "content": f"blocked tool={tool} step={step} reason={route.note}\n",
                    },
                )
                result = StepResult(False, f"Tool blocked by router: {tool}. {route.note}")
            else:
                result = self.tools.run(tool, args)

            self.budget.record_tool_use(is_external=self.tools.is_external_tool(tool) and result.ok)
            record = {
                "thought": decision.thought,
                "tool": tool,
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
            step += 1

    def _decide(self, route: RouteDecision) -> AgentDecision:
        history_text = self.compressor.compress(self.history)
        enriched_history = f"Routing:\n{route.note}\n\n{history_text}"
        return self.planner.decide(self.objective, enriched_history, route.allowed_tools)

    def _emit(self, event: dict) -> None:
        if self.on_event is None:
            return
        self.on_event(event)
