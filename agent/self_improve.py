from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class RouterProfile:
    escalate_after_failures: int = 1
    compress_recent_steps: int = 6
    max_external_calls: int = 4
    updated_at: str = ""
    notes: str = ""


@dataclass
class RunSignals:
    observations: int
    failures: int
    max_failure_streak: int
    router_blocked: int
    repeated_error_pairs: int
    max_external_used: int
    configured_max_external: int | None
    invalid_tool_noops: int
    noop_events: int
    empty_shell_noops: int
    pytest_no_tests_error_events: int


@dataclass
class FailurePatternProfile:
    total_runs: int = 0
    bad_runs: int = 0
    invalid_tool_noops: int = 0
    noop_events: int = 0
    empty_shell_noops: int = 0
    router_blocked_events: int = 0
    pytest_no_tests_error_events: int = 0
    updated_at: str = ""
    notes: str = ""


def load_router_profile(path: Path) -> RouterProfile | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return RouterProfile(
        escalate_after_failures=int(data.get("escalate_after_failures", 1)),
        compress_recent_steps=int(data.get("compress_recent_steps", 6)),
        max_external_calls=int(data.get("max_external_calls", 4)),
        updated_at=str(data.get("updated_at", "")),
        notes=str(data.get("notes", "")),
    )


def save_router_profile(path: Path, profile: RouterProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "escalate_after_failures": profile.escalate_after_failures,
        "compress_recent_steps": profile.compress_recent_steps,
        "max_external_calls": profile.max_external_calls,
        "updated_at": profile.updated_at,
        "notes": profile.notes,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze_run_log(path: Path) -> RunSignals:
    if not path.exists():
        return RunSignals(0, 0, 0, 0, 0, 0, None, 0, 0, 0, 0)

    observations: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "observation":
            observations.append(event)

    failures = 0
    max_failure_streak = 0
    streak = 0
    router_blocked = 0
    repeated_error_pairs = 0
    prev_fail_output = ""
    max_external_used = 0
    configured_max_external: int | None = None
    invalid_tool_noops = 0
    noop_events = 0
    empty_shell_noops = 0
    pytest_no_tests_error_events = 0

    for obs in observations:
        ok = bool(obs.get("ok"))
        output = str(obs.get("output", "")).strip().lower()
        budget = obs.get("budget", {}) if isinstance(obs.get("budget"), dict) else {}
        used_external = int(budget.get("used_external_calls", 0) or 0)
        configured = budget.get("max_external_calls")
        if configured is not None:
            configured_max_external = int(configured)
        max_external_used = max(max_external_used, used_external)
        if output.startswith("no-op:"):
            noop_events += 1
        if "planner returned invalid tool" in output:
            invalid_tool_noops += 1
        if "no-op: empty shell command" in output:
            empty_shell_noops += 1
        if "collected 0 items" in output and (
            "error:" in output or "file or directory not found" in output
        ):
            pytest_no_tests_error_events += 1
        if ok:
            streak = 0
            prev_fail_output = ""
            continue
        failures += 1
        streak += 1
        max_failure_streak = max(max_failure_streak, streak)
        if "tool blocked by router" in output:
            router_blocked += 1
        if prev_fail_output and prev_fail_output == output:
            repeated_error_pairs += 1
        prev_fail_output = output

    return RunSignals(
        observations=len(observations),
        failures=failures,
        max_failure_streak=max_failure_streak,
        router_blocked=router_blocked,
        repeated_error_pairs=repeated_error_pairs,
        max_external_used=max_external_used,
        configured_max_external=configured_max_external,
        invalid_tool_noops=invalid_tool_noops,
        noop_events=noop_events,
        empty_shell_noops=empty_shell_noops,
        pytest_no_tests_error_events=pytest_no_tests_error_events,
    )


def tune_router_profile(
    current: RouterProfile,
    signals: RunSignals,
) -> RouterProfile:
    escalate = current.escalate_after_failures
    compress_recent = current.compress_recent_steps
    max_external_calls = current.max_external_calls
    notes: list[str] = []

    if signals.max_failure_streak >= 3:
        escalate = max(1, escalate - 1)
        compress_recent = min(12, compress_recent + 2)
        if signals.configured_max_external is not None and signals.max_external_used >= signals.configured_max_external:
            max_external_calls = min(20, max_external_calls + 2)
        notes.append("High failure streak: escalate earlier, keep more recent context.")

    if signals.router_blocked >= 2:
        escalate = 1
        notes.append("Router blocked tools repeatedly: always allow earlier escalation.")

    if signals.failures == 0 and signals.observations >= 8:
        escalate = min(3, escalate + 1)
        compress_recent = max(4, compress_recent - 1)
        if signals.max_external_used <= max(1, max_external_calls // 3):
            max_external_calls = max(2, max_external_calls - 1)
        notes.append("Stable run: reduce escalation sensitivity and context size.")

    if signals.repeated_error_pairs >= 2:
        compress_recent = min(14, compress_recent + 2)
        notes.append("Repeated same error: increase detailed recent context.")

    if not notes:
        notes.append("No significant pattern detected; keep previous settings.")

    return RouterProfile(
        escalate_after_failures=escalate,
        compress_recent_steps=compress_recent,
        max_external_calls=max_external_calls,
        updated_at=datetime.now(UTC).isoformat(),
        notes=" ".join(notes),
    )


def load_failure_profile(path: Path) -> FailurePatternProfile | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return FailurePatternProfile(
        total_runs=int(data.get("total_runs", 0)),
        bad_runs=int(data.get("bad_runs", 0)),
        invalid_tool_noops=int(data.get("invalid_tool_noops", 0)),
        noop_events=int(data.get("noop_events", 0)),
        empty_shell_noops=int(data.get("empty_shell_noops", 0)),
        router_blocked_events=int(data.get("router_blocked_events", 0)),
        pytest_no_tests_error_events=int(data.get("pytest_no_tests_error_events", 0)),
        updated_at=str(data.get("updated_at", "")),
        notes=str(data.get("notes", "")),
    )


def save_failure_profile(path: Path, profile: FailurePatternProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_runs": profile.total_runs,
        "bad_runs": profile.bad_runs,
        "invalid_tool_noops": profile.invalid_tool_noops,
        "noop_events": profile.noop_events,
        "empty_shell_noops": profile.empty_shell_noops,
        "router_blocked_events": profile.router_blocked_events,
        "pytest_no_tests_error_events": profile.pytest_no_tests_error_events,
        "updated_at": profile.updated_at,
        "notes": profile.notes,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def tune_failure_profile(current: FailurePatternProfile, signals: RunSignals) -> FailurePatternProfile:
    bad_run = (
        signals.invalid_tool_noops >= 2
        or signals.noop_events >= 3
        or signals.router_blocked >= 2
        or signals.pytest_no_tests_error_events >= 1
        or signals.max_failure_streak >= 3
    )

    total_runs = current.total_runs + 1
    bad_runs = current.bad_runs + (1 if bad_run else 0)
    notes: list[str] = []
    if bad_run:
        notes.append("Detected unstable run pattern.")
    if signals.invalid_tool_noops:
        notes.append("Invalid-tool no-op observed.")
    if signals.empty_shell_noops:
        notes.append("Empty shell command observed.")
    if signals.pytest_no_tests_error_events:
        notes.append("Pytest no-tests with error observed.")
    if not notes:
        notes.append("No failure pattern escalation for this run.")

    return FailurePatternProfile(
        total_runs=total_runs,
        bad_runs=bad_runs,
        invalid_tool_noops=current.invalid_tool_noops + signals.invalid_tool_noops,
        noop_events=current.noop_events + signals.noop_events,
        empty_shell_noops=current.empty_shell_noops + signals.empty_shell_noops,
        router_blocked_events=current.router_blocked_events + signals.router_blocked,
        pytest_no_tests_error_events=(
            current.pytest_no_tests_error_events + signals.pytest_no_tests_error_events
        ),
        updated_at=datetime.now(UTC).isoformat(),
        notes=" ".join(notes),
    )


def recommend_guardrails(
    profile: FailurePatternProfile | None,
    max_steps: int,
    noop_streak_limit: int,
) -> tuple[int, int, str]:
    if profile is None or profile.total_runs < 3:
        return max_steps, noop_streak_limit, "guardrails: baseline"

    effective_noop = noop_streak_limit if noop_streak_limit > 0 else 3
    bad_rate = profile.bad_runs / max(1, profile.total_runs)
    noisy_rate = (profile.noop_events + profile.invalid_tool_noops) / max(1, profile.total_runs)
    tightened = False

    if bad_rate >= 0.5:
        max_steps = min(max_steps, 10)
        effective_noop = min(effective_noop, 2)
        tightened = True
    if noisy_rate >= 2.0:
        effective_noop = min(effective_noop, 2)
        tightened = True

    note = "guardrails: tightened from failure patterns" if tightened else "guardrails: baseline"
    return max_steps, effective_noop, note
