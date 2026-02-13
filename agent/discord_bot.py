from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import discord
from discord.ext import commands

from .adapters import AntigravityAdapter, CodexAdapter, ExternalAgentAdapter
from .budget import BudgetManager
from .config import AgentConfig
from .history import HistoryCompressor
from .memory import JsonlMemory
from .providers import OpenAIProvider
from .router import Router
from .runner import AgentRunner
from .self_improve import (
    RouterProfile,
    analyze_run_log,
    load_router_profile,
    save_router_profile,
    tune_router_profile,
)
from .supervisor import Supervisor
from .tools import ToolRunner


@dataclass
class ChannelRunState:
    running: bool = False
    objective: str = ""
    step: int = 0
    last_event: str = ""
    run_log: str = ""
    final_message: str = ""
    cancelled: bool = False
    output_tail: str = ""
    profile_note: str = ""
    mode: str = "agent"
    lock: threading.Lock = field(default_factory=threading.Lock)


class RunRegistry:
    def __init__(self) -> None:
        self._states: dict[int, ChannelRunState] = {}
        self._lock = threading.Lock()

    def get(self, channel_id: int) -> ChannelRunState:
        with self._lock:
            if channel_id not in self._states:
                self._states[channel_id] = ChannelRunState()
            return self._states[channel_id]


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip().strip("'").strip('"')
    if not key:
        return None
    return key, value


def _load_dotenv_files(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_dotenv_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


def _profile_path(project_root: Path) -> Path:
    return project_root / ".agent_state" / "router_profile.json"


def _build_external_adapters(enable: bool) -> dict[str, ExternalAgentAdapter]:
    if not enable:
        return {}
    codex = CodexAdapter()
    antigravity = AntigravityAdapter()
    return {codex.name: codex, antigravity.name: antigravity}


def _load_allowed_channel_ids() -> set[int]:
    raw = os.getenv("DISCORD_ALLOWED_CHANNEL_IDS", "").strip()
    if not raw:
        return set()
    ids: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            ids.add(int(token))
    return ids


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _load_profile_overrides(project_root: Path) -> RouterProfile | None:
    return load_router_profile(_profile_path(project_root))


def _run_cmd(command: str, workdir: Path) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=workdir,
        shell=True,
        text=True,
        capture_output=True,
    )
    out = (completed.stdout + completed.stderr).strip()
    return completed.returncode, out[:8000] if out else "(no output)"


def _collect_changed_files(workdir: Path) -> list[str]:
    code, out = _run_cmd("git status --porcelain", workdir)
    if code != 0:
        return []
    files: list[str] = []
    for raw in out.splitlines():
        line = raw.rstrip()
        if len(line) < 4:
            continue
        path_part = line[3:]
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        files.append(path_part.strip())
    return files


def _runtime_noise_paths() -> tuple[str, ...]:
    return (
        ".agent_state/router_profile.json",
        ".pytest_cache/",
        ".venv/",
        "agent/__pycache__/",
        "agent/adapters/__pycache__/",
        "agent/providers/__pycache__/",
        "runs/run-",
        "runs/router-blocked.log",
    )


def _is_runtime_noise(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _runtime_noise_paths())


def _effective_changed_files(workdir: Path) -> list[str]:
    return [path for path in _collect_changed_files(workdir) if not _is_runtime_noise(path)]


def _forbidden_path_prefixes() -> list[str]:
    raw = os.getenv("DISCORD_FORBIDDEN_PATH_PREFIXES", "").strip()
    if raw:
        return [token.strip() for token in raw.split(",") if token.strip()]
    return [".env", ".venv/", "agent/__pycache__/", "__pycache__/"]


def _evaluate_dod(workdir: Path, validation_ok: bool, alerts: list[str]) -> tuple[bool, str]:
    max_changed = max(1, _env_int("DISCORD_MAX_CHANGED_FILES", 25))
    files = _effective_changed_files(workdir)
    forbidden_prefixes = _forbidden_path_prefixes()
    forbidden_hits = [path for path in files if any(path.startswith(prefix) for prefix in forbidden_prefixes)]
    over_limit = len(files) > max_changed
    has_alerts = len(alerts) > 0

    ok = validation_ok and not has_alerts and not forbidden_hits and not over_limit
    lines = [
        "# DoD Report",
        "",
        f"- validation_ok: {validation_ok}",
        f"- validation_alerts: {len(alerts)}",
        f"- changed_files_count: {len(files)} (limit={max_changed})",
        f"- forbidden_hits_count: {len(forbidden_hits)}",
        f"- status: {'PASS' if ok else 'FAIL'}",
        "",
        "## Changed Files",
    ]
    if files:
        lines.extend(f"- {path}" for path in files[:120])
    else:
        lines.append("- (none)")
    lines.extend(["", "## Forbidden Hits"])
    if forbidden_hits:
        lines.extend(f"- {path}" for path in forbidden_hits[:120])
    else:
        lines.append("- (none)")
    if alerts:
        lines.extend(["", "## Validation Alerts"])
        lines.extend(f"- {line}" for line in alerts)
    return ok, "\n".join(lines)


def _review_findings(workdir: Path, project_root: Path) -> str:
    files = _effective_changed_files(workdir)
    forbidden_prefixes = _forbidden_path_prefixes()
    max_changed = max(1, _env_int("DISCORD_MAX_CHANGED_FILES", 25))
    _, diff_text = _run_cmd("git diff --", workdir)
    dod_path = project_root / "runs" / "dod_report.md"
    val_path = project_root / "runs" / "validation_report.md"
    validation = val_path.read_text(encoding="utf-8") if val_path.exists() else ""

    critical: list[str] = []
    high: list[str] = []
    medium: list[str] = []

    forbidden_hits = [path for path in files if any(path.startswith(prefix) for prefix in forbidden_prefixes)]
    if forbidden_hits:
        critical.append(f"Forbidden path changes: {', '.join(forbidden_hits[:8])}")
    secret_hits = _detect_secret_like_additions(diff_text)
    if secret_hits:
        preview = "; ".join(secret_hits[:3])
        critical.append(f"Possible secret exposure in added lines: {preview}")

    if len(files) > max_changed:
        high.append(f"Changed files exceed limit: {len(files)} > {max_changed}.")
    if "exit_code: 1" in validation or "exit_code: 2" in validation:
        high.append("Validation report contains failing command(s).")
    if "error:" in validation.lower() or "failed" in validation.lower():
        high.append("Validation report contains errors or failures.")
    if not files:
        high.append("No effective changed files found.")

    added_lines = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed_lines = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    if added_lines + removed_lines > 800:
        medium.append(f"Large diff size: +{added_lines}/-{removed_lines}.")
    if "todo" in diff_text.lower() or "fixme" in diff_text.lower():
        medium.append("Diff includes TODO/FIXME markers.")
    if dod_path.exists():
        dod = dod_path.read_text(encoding="utf-8")
        if "- status: FAIL" in dod:
            medium.append("Latest DoD report indicates FAIL.")

    lines = ["# Review Report", ""]
    lines.append("## Critical")
    lines.extend([f"- {item}" for item in critical] or ["- none"])
    lines.append("")
    lines.append("## High")
    lines.extend([f"- {item}" for item in high] or ["- none"])
    lines.append("")
    lines.append("## Medium")
    lines.extend([f"- {item}" for item in medium] or ["- none"])
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- changed_files={len(files)}")
    lines.append(f"- diff_lines=+{added_lines}/-{removed_lines}")
    score = "BLOCKED" if critical else ("RISKY" if high else "OK")
    lines.append(f"- status={score}")
    return "\n".join(lines)


def _detect_secret_like_additions(diff_text: str) -> list[str]:
    """
    Scan only added diff lines and detect concrete secret formats.
    README/docs/examples/sample contexts are excluded to reduce false positives.
    """
    patterns: list[tuple[str, re.Pattern[str]]] = [
        ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
        ("Google API key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
        ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
        ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
        ("Discord bot token", re.compile(r"\b[MN][A-Za-z\d]{23}\.[\w-]{6}\.[\w-]{20,}\b")),
        ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        (
            "Private key header",
            re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
        ),
    ]

    ignored_path_tokens = ("readme", "docs/", "example", "examples/", "sample", "samples/")
    findings: list[str] = []
    current_path = ""

    for raw in diff_text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("+++ b/"):
            current_path = line[6:]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue

        path_l = current_path.lower()
        if any(token in path_l for token in ignored_path_tokens):
            continue

        content = line[1:]
        content_l = content.lower()
        if any(token in content_l for token in ("example", "sample", "placeholder", "dummy")):
            continue

        for label, pattern in patterns:
            if pattern.search(content):
                findings.append(f"{label} in {current_path or '(unknown file)'}")
                break

    # de-duplicate while preserving order
    unique: list[str] = []
    for item in findings:
        if item in unique:
            continue
        unique.append(item)
    return unique


def _make_release_note(
    workdir: Path,
    objective: str,
    deliver_status: str,
    implementer_final: str,
    validation_report_path: Path,
    dod_report_path: Path,
) -> str:
    files = _effective_changed_files(workdir)
    validation_excerpt = ""
    if validation_report_path.exists():
        validation_excerpt = validation_report_path.read_text(encoding="utf-8")[:1200]
    dod_status = "UNKNOWN"
    if dod_report_path.exists():
        dod_text = dod_report_path.read_text(encoding="utf-8")
        if "- status: PASS" in dod_text:
            dod_status = "PASS"
        elif "- status: FAIL" in dod_text:
            dod_status = "FAIL"
    lines = [
        "# Release Note",
        "",
        "## Objective",
        objective,
        "",
        "## Delivery Status",
        f"- deliver: {deliver_status}",
        f"- implementer: {implementer_final}",
        f"- dod: {dod_status}",
        "",
        "## Changed Files",
    ]
    lines.extend([f"- {path}" for path in files[:120]] or ["- (none)"])
    lines.extend(
        [
            "",
            "## Validation Snapshot",
            "```text",
            validation_excerpt or "(no validation report)",
            "```",
            "",
            "## PR Body",
            "### Summary",
            "- Implemented requested changes and ran validation suite.",
            "### Validation",
            f"- See `{validation_report_path}`.",
            f"- DoD result recorded in `{dod_report_path}`.",
            "### Risks / Follow-ups",
            "- Review DoD and Review Report before merge.",
        ]
    )
    return "\n".join(lines)


def _dod_status(project_root: Path) -> str:
    path = project_root / "runs" / "dod_report.md"
    if not path.exists():
        return "MISSING"
    text = path.read_text(encoding="utf-8")
    if "- status: PASS" in text:
        return "PASS"
    if "- status: FAIL" in text:
        return "FAIL"
    return "UNKNOWN"


def _review_status(project_root: Path) -> str:
    path = project_root / "runs" / "review_report.md"
    if not path.exists():
        return "MISSING"
    text = path.read_text(encoding="utf-8")
    if "- status=OK" in text:
        return "OK"
    if "- status=RISKY" in text:
        return "RISKY"
    if "- status=BLOCKED" in text:
        return "BLOCKED"
    return "UNKNOWN"


def _is_placeholder_commit_message(message: str) -> bool:
    text = message.strip()
    lower = text.lower()
    if not text or len(text) < 6:
        return True
    if re.fullmatch(r"<[^>]+>", text):
        return True
    placeholders = {
        "<message>",
        "message",
        "commit message",
        "fix",
        "update",
        "wip",
        "test",
        "todo",
    }
    return lower in placeholders


def _write_pr_ready_bundle(project_root: Path, workdir: Path) -> Path:
    release_note_path = project_root / "runs" / "release_note.md"
    review_report_path = project_root / "runs" / "review_report.md"
    dod_report_path = project_root / "runs" / "dod_report.md"
    validation_report_path = project_root / "runs" / "validation_report.md"
    bundle_path = project_root / "runs" / "pr_ready.md"

    dod = _dod_status(project_root)
    review = _review_status(project_root)
    changed_files = _effective_changed_files(workdir)
    branch_code, branch_out = _run_cmd("git rev-parse --abbrev-ref HEAD", workdir)
    branch = branch_out.splitlines()[-1] if branch_code == 0 and branch_out else "unknown"

    lines = [
        "# PR Ready Bundle",
        "",
        "## Status",
        f"- branch: {branch}",
        f"- dod: {dod}",
        f"- review: {review}",
        f"- changed_files: {len(changed_files)}",
        "",
        "## Artifacts",
        f"- release_note: {release_note_path}",
        f"- review_report: {review_report_path}",
        f"- dod_report: {dod_report_path}",
        f"- validation_report: {validation_report_path}",
        "",
        "## Changed Files",
    ]
    lines.extend([f"- {path}" for path in changed_files[:150]] or ["- (none)"])
    lines.extend(
        [
            "",
            "## Merge Checklist",
            f"- [ ] DoD PASS (`{dod}`)",
            f"- [ ] Review OK (`{review}`)",
            "- [ ] Validation reviewed",
            "- [ ] Release note reviewed",
        ]
    )
    bundle_path.write_text("\n".join(lines), encoding="utf-8")
    return bundle_path


def _is_git_repo(workdir: Path) -> bool:
    code, _ = _run_cmd("git rev-parse --is-inside-work-tree", workdir)
    return code == 0


def _latest_logs(runs_dir: Path, limit: int = 5) -> list[Path]:
    files = sorted(runs_dir.glob("run-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[: max(1, limit)]


def _detect_project_type(workdir: Path) -> str:
    has_py = (workdir / "requirements.txt").exists() or bool(list(workdir.glob("agent/**/*.py")))
    # Treat Node as present only when package.json exists in project root.
    has_js = (workdir / "package.json").exists()
    if has_py and has_js:
        return "mixed"
    if has_py:
        return "python"
    if has_js:
        return "node"
    return "unknown"


def _tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return f"file not found: {path}"
    content = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(content[-max(1, lines) :])[:3500]


def _validation_commands(workdir: Path) -> list[str]:
    project_type = _detect_project_type(workdir)
    if project_type == "python":
        cmds: list[str] = []
        if (workdir / ".venv" / "bin" / "python").exists():
            cmds.append(".venv/bin/python -m pytest -q")
            cmds.append(".venv/bin/python -m compileall agent")
        else:
            cmds.append("python3 -m pytest -q")
            cmds.append("python3 -m compileall agent")
        return cmds
    if project_type == "node":
        return ["npm test --silent"]
    if project_type == "mixed":
        return ["python3 -m pytest -q", "npm test --silent"]
    return []


def _run_validation_suite(workdir: Path) -> tuple[bool, str]:
    commands = _validation_commands(workdir)
    if not commands:
        return True, "No validation commands for project type."

    lines = ["# Validation Report", ""]
    ok_all = True
    for command in commands:
        if command.startswith("npm ") and not (workdir / "package.json").exists():
            lines.append(f"## Command: `{command}`")
            lines.append("- skipped: package.json not found")
            lines.append("")
            continue
        code, out = _run_cmd(command, workdir)
        lines.append(f"## Command: `{command}`")
        lines.append(f"- exit_code: {code}")
        lines.append("```text")
        lines.append(out[:2500])
        lines.append("```")
        lines.append("")
        if code != 0:
            ok_all = False
    return ok_all, "\n".join(lines)


def _extract_validation_alert_lines(report_text: str) -> list[str]:
    alerts: list[str] = []
    keywords = ("error", "failed", "warning", "traceback", "exception", "not found")
    for raw in report_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(token in lowered for token in keywords):
            alerts.append(line)
    # Keep the output concise for prompt/context usage.
    unique: list[str] = []
    for line in alerts:
        if line in unique:
            continue
        unique.append(line)
        if len(unique) >= 12:
            break
    return unique


def _ensure_branch(workdir: Path) -> str:
    code, out = _run_cmd("git rev-parse --abbrev-ref HEAD", workdir)
    if code != 0:
        return "unknown"
    branch = out.strip().splitlines()[-1]
    if branch in {"main", "master"}:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        new_branch = f"agent/{ts}"
        _run_cmd(f"git checkout -b {new_branch}", workdir)
        return new_branch
    return branch


def _build_runtime_config(project_root: Path, workdir: Path) -> tuple[AgentConfig, dict[str, ExternalAgentAdapter], RouterProfile | None]:
    profile = _load_profile_overrides(project_root)
    max_steps = _env_int("DISCORD_MAX_STEPS", 30)
    max_external_calls = _env_int("DISCORD_MAX_EXTERNAL_CALLS", 8)
    max_seconds = _env_int("DISCORD_MAX_SECONDS", 3600)
    compress_recent_steps = _env_int("DISCORD_COMPRESS_RECENT_STEPS", 8)
    escalate_after_failures = _env_int("DISCORD_ESCALATE_AFTER_FAILURES", 1)
    if profile:
        max_external_calls = profile.max_external_calls
        compress_recent_steps = profile.compress_recent_steps
        escalate_after_failures = profile.escalate_after_failures
    config = AgentConfig(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        max_steps=max_steps,
        max_external_calls=max_external_calls,
        max_seconds=max_seconds,
        compress_recent_steps=compress_recent_steps,
        escalate_after_failures=escalate_after_failures,
        workdir=workdir,
        auto_approve_safe=False,
    )
    external_enabled = _env_bool("DISCORD_ENABLE_EXTERNAL_AGENTS", True)
    external_adapters = _build_external_adapters(external_enabled)
    return config, external_adapters, profile


def _run_agent_job(
    *,
    project_root: Path,
    workdir: Path,
    objective: str,
    state: ChannelRunState,
    progress_hook,
    worker_name: str = "agent",
    max_steps_override: int | None = None,
    forced_allowed_tools: list[str] | None = None,
    noop_streak_limit: int = 0,
) -> tuple[str, str, str]:
    config, external_adapters, profile = _build_runtime_config(project_root, workdir)
    if max_steps_override is not None:
        config.max_steps = max(1, max_steps_override)
    planner = OpenAIProvider(model=config.model)
    approval_policy = os.getenv("DISCORD_APPROVAL_POLICY", "allow").strip().lower() or "allow"
    strict_shell_allowlist = False
    extra_safe_shell_prefixes: tuple[str, ...] | None = None
    shell_allow_prefixes: tuple[str, ...] | None = None
    if worker_name == "tester":
        project_type = _detect_project_type(workdir)
        if project_type == "python":
            strict_shell_allowlist = True
            extra_safe_shell_prefixes = (
                "python -m pytest",
                "python3 -m pytest",
                "pytest",
                "python -m compileall",
                "python3 -m compileall",
            )
            objective = (
                objective
                + "\n制約: このプロジェクトはPython扱い。testerは pytest / compileall 系以外の検証コマンドを使わないこと。"
            )
    if worker_name == "deliver_implementer":
        strict_shell_allowlist = True
        shell_allow_prefixes = (
            "pytest",
            "python -m pytest",
            "python3 -m pytest",
            ".venv/bin/python -m pytest",
            "python -m compileall",
            "python3 -m compileall",
            ".venv/bin/python -m compileall",
        )
        objective = (
            objective
            + "\n制約: 使用ツールは read_file/write_file/shell のみ。"
            + " shellは pytest/compileall 系コマンドのみ許可。"
            + " 同じ成功結果を繰り返さず、完了したら即 finish。"
        )

    tools = ToolRunner(
        workdir=config.workdir,
        auto_approve_safe=False,
        external_adapters=external_adapters,
        approval_policy=approval_policy,
        allow_dangerous_commands=_env_bool("DISCORD_ALLOW_DANGEROUS_COMMANDS", False),
        strict_shell_allowlist=strict_shell_allowlist,
        extra_safe_shell_prefixes=extra_safe_shell_prefixes,
        shell_allow_prefixes=shell_allow_prefixes,
    )
    runs_dir = project_root / "runs"
    memory = JsonlMemory(output_dir=runs_dir)
    budget = BudgetManager(
        max_steps=config.max_steps,
        max_external_calls=config.max_external_calls,
        max_seconds=config.max_seconds,
    )
    router = Router(escalate_after_failures=config.escalate_after_failures)
    compressor = HistoryCompressor(recent_steps=config.compress_recent_steps)

    def on_event(event: dict) -> None:
        evt = str(event.get("event", ""))
        step = int(event.get("step", 0) or 0)
        with state.lock:
            state.step = step
            state.last_event = evt
            if evt == "observation":
                state.output_tail = str(event.get("output", ""))[:400]
        if evt == "observation":
            ok = bool(event.get("ok"))
            tool = str(event.get("tool", ""))
            text = str(event.get("output", "")).replace("\n", " ")
            progress_hook(f"[step {step}] {tool} ok={ok}\n{text[:500]}")

    def should_stop() -> bool:
        with state.lock:
            return state.cancelled

    runner = AgentRunner(
        objective=objective,
        planner=planner,
        tools=tools,
        memory=memory,
        budget=budget,
        router=router,
        compressor=compressor,
        on_event=on_event,
        should_stop=should_stop,
        forced_allowed_tools=forced_allowed_tools,
        noop_streak_limit=noop_streak_limit,
    )
    final = runner.run()
    signals = analyze_run_log(memory.path)
    base_profile = profile or RouterProfile(
        max_external_calls=config.max_external_calls,
        escalate_after_failures=config.escalate_after_failures,
        compress_recent_steps=config.compress_recent_steps,
    )
    tuned_profile = tune_router_profile(base_profile, signals)
    profile_file = _profile_path(project_root)
    save_router_profile(profile_file, tuned_profile)
    note = (
        f"next: max_external_calls={tuned_profile.max_external_calls}, "
        f"escalate_after_failures={tuned_profile.escalate_after_failures}, "
        f"compress_recent_steps={tuned_profile.compress_recent_steps}"
    )
    return final, str(memory.path), note


def main() -> int:
    parser = argparse.ArgumentParser(description="Discord control bot for ai-cli-agent.")
    parser.add_argument("--workdir", default=None, help="Default workdir for runs.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else Path.cwd()
    _load_dotenv_files([project_root / ".env", Path.cwd() / ".env", workdir / ".env"])

    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        print("DISCORD_BOT_TOKEN is not set.", flush=True)
        return 1

    allowed_channels = _load_allowed_channel_ids()
    registry = RunRegistry()
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready() -> None:
        print(f"Discord bot logged in as {bot.user}", flush=True)
        print(
            "Commands: !agent !supervise !deliver !autopr !review !status !cancel !runs !tail !diff !approve !rollback",
            flush=True,
        )

    @bot.event
    async def on_command_error(ctx: commands.Context, error: Exception) -> None:
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.reply(
                "引数が不足しています。`!agent <指示>` / `!supervise <指示>` / `!autopr <指示>` の形式で送ってください。"
            )
            return
        if isinstance(error, commands.CommandNotFound):
            return
        await ctx.reply(f"コマンドエラー: {type(error).__name__}: {error}")

    def is_allowed_channel(channel_id: int) -> bool:
        return not allowed_channels or channel_id in allowed_channels

    async def with_channel_lock(ctx: commands.Context, mode: str, objective: str) -> ChannelRunState | None:
        if not is_allowed_channel(ctx.channel.id):
            return None
        state = registry.get(ctx.channel.id)
        with state.lock:
            if state.running:
                await ctx.reply("このチャンネルでは既に実行中です。`!status` で確認してください。")
                return None
            state.running = True
            state.mode = mode
            state.cancelled = False
            state.objective = objective
            state.step = 0
            state.last_event = "queued"
            state.final_message = ""
            state.output_tail = ""
            state.profile_note = ""
        return state

    def release_state(state: ChannelRunState, final_message: str, run_log: str = "", profile_note: str = "") -> None:
        with state.lock:
            state.running = False
            state.final_message = final_message
            if run_log:
                state.run_log = run_log
            if profile_note:
                state.profile_note = profile_note

    @bot.command(name="status")
    async def status_cmd(ctx: commands.Context) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        state = registry.get(ctx.channel.id)
        with state.lock:
            if state.running:
                await ctx.reply(
                    f"実行中 mode={state.mode}\nstep={state.step}\nobjective={state.objective}\nlast={state.last_event}\nlog={state.run_log}"
                )
                return
            if state.objective:
                await ctx.reply(
                    f"直近実行 mode={state.mode}\nobjective={state.objective}\nfinal={state.final_message or '(none)'}\nlog={state.run_log}\nnotes={state.profile_note or '(none)'}"
                )
                return
        await ctx.reply("実行履歴はまだありません。`!agent <指示>` で開始できます。")

    @bot.command(name="cancel")
    async def cancel_cmd(ctx: commands.Context) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        state = registry.get(ctx.channel.id)
        with state.lock:
            if not state.running:
                await ctx.reply("現在実行中のタスクはありません。")
                return
            state.cancelled = True
        await ctx.reply("キャンセル要求を受け付けました。次のステップ境界で停止します。")

    @bot.command(name="runs")
    async def runs_cmd(ctx: commands.Context, count: int = 5) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        runs = _latest_logs(project_root / "runs", limit=max(1, min(count, 20)))
        if not runs:
            await ctx.reply("runログはまだありません。")
            return
        lines = [f"{idx+1}. {p.name}" for idx, p in enumerate(runs)]
        await ctx.reply("最新runログ:\n" + "\n".join(lines))

    @bot.command(name="tail")
    async def tail_cmd(ctx: commands.Context, lines: int = 40) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        state = registry.get(ctx.channel.id)
        with state.lock:
            run_log = state.run_log
        if not run_log:
            await ctx.reply("参照できるrunログがありません。")
            return
        text = _tail_file(Path(run_log), max(5, min(lines, 200)))
        await ctx.reply(f"```text\n{text[:1800]}\n```")

    @bot.command(name="diff")
    async def diff_cmd(ctx: commands.Context) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        if not _is_git_repo(workdir):
            await ctx.reply("このworkdirはGitリポジトリではありません。`git init` か既存リポジトリを指定してください。")
            return
        code, out = _run_cmd("git diff --", workdir)
        if code != 0:
            await ctx.reply(f"git diffに失敗しました。\n```text\n{out[:1600]}\n```")
            return
        preview = out[:1800] if out else "(no diff)"
        await ctx.reply(f"```diff\n{preview}\n```")

    @bot.command(name="review")
    async def review_cmd(ctx: commands.Context) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        if not _is_git_repo(workdir):
            await ctx.reply("このworkdirはGitリポジトリではありません。`!review` は利用できません。")
            return
        report = _review_findings(workdir, project_root)
        report_path = project_root / "runs" / "review_report.md"
        report_path.write_text(report, encoding="utf-8")
        await ctx.reply(f"```markdown\n{report[:1700]}\n```\nreport={report_path}")

    @bot.command(name="approve")
    async def approve_cmd(ctx: commands.Context, *, message: str) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        if not _is_git_repo(workdir):
            await ctx.reply("このworkdirはGitリポジトリではありません。`git init` 後に再実行してください。")
            return
        if _is_placeholder_commit_message(message):
            await ctx.reply(
                "コミットメッセージがダミー形式です。具体的な変更内容を書いてください。\n"
                "例: `feat: add deliver DoD gates and PR bundle generation`"
            )
            return
        enforce_gates = _env_bool("DISCORD_ENFORCE_APPROVE_GATES", True)
        if enforce_gates:
            dod = _dod_status(project_root)
            review = _review_status(project_root)
            if dod != "PASS" or review != "OK":
                await ctx.reply(
                    "approveをブロックしました。ゲート未達です。\n"
                    f"- DoD: {dod} (required: PASS)\n"
                    f"- Review: {review} (required: OK)\n"
                    "先に `!deliver ...` と `!review` を実行してください。"
                )
                return
        add_code, add_out = _run_cmd("git add -A", workdir)
        if add_code != 0:
            await ctx.reply(f"git add失敗\n```text\n{add_out[:1500]}\n```")
            return
        commit_code, commit_out = _run_cmd(f"git commit -m {message!r}", workdir)
        if commit_code != 0:
            await ctx.reply(f"git commit失敗\n```text\n{commit_out[:1500]}\n```")
            return
        await ctx.reply(f"コミット完了\n```text\n{commit_out[:1500]}\n```")

    @bot.command(name="rollback")
    async def rollback_cmd(ctx: commands.Context, ref: str = "HEAD") -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        if not _is_git_repo(workdir):
            await ctx.reply("このworkdirはGitリポジトリではありません。rollbackは利用できません。")
            return
        code, out = _run_cmd(f"git revert --no-edit {ref}", workdir)
        if code != 0:
            await ctx.reply(f"rollback失敗\n```text\n{out[:1500]}\n```")
            return
        await ctx.reply(f"rollback完了 (git revert {ref})\n```text\n{out[:1500]}\n```")

    @bot.command(name="agent")
    async def agent_cmd(ctx: commands.Context, *, objective: str) -> None:
        state = await with_channel_lock(ctx, "agent", objective)
        if state is None:
            return
        await ctx.reply(f"開始 mode=agent\nobjective={objective}\nworkdir={workdir}")
        runtime_loop = asyncio.get_running_loop()

        async def send_progress(message: str) -> None:
            try:
                await ctx.send(message[:1800])
            except Exception:
                pass

        def progress_hook(msg: str) -> None:
            runtime_loop.call_soon_threadsafe(asyncio.create_task, send_progress(msg))

        try:
            final, run_log, note = await asyncio.to_thread(
                _run_agent_job,
                project_root=project_root,
                workdir=workdir,
                objective=objective,
                state=state,
                progress_hook=progress_hook,
                worker_name="agent",
            )
            release_state(state, final, run_log, note)
            await ctx.send(f"完了\nfinal={final}\nlog={run_log}\n{note}")
        except Exception as exc:
            release_state(state, f"error: {type(exc).__name__}: {exc}")
            await ctx.send(f"エラーで停止しました: {type(exc).__name__}: {exc}")

    @bot.command(name="supervise")
    async def supervise_cmd(ctx: commands.Context, *, objective: str) -> None:
        state = await with_channel_lock(ctx, "supervise", objective)
        if state is None:
            return
        if not _is_git_repo(workdir):
            await ctx.reply(
                "注意: このworkdirはGitリポジトリではありません。`git_status/git_diff` は失敗するため、ファイル編集中心で実行します。"
            )
        await ctx.reply(f"開始 mode=supervise\nobjective={objective}")
        runtime_loop = asyncio.get_running_loop()
        supervisor = Supervisor()

        async def send_stage(message: str) -> None:
            try:
                await ctx.send(message[:1800])
            except Exception:
                pass

        def progress_hook(msg: str) -> None:
            runtime_loop.call_soon_threadsafe(asyncio.create_task, send_stage(msg))

        def run_supervised() -> tuple[str, str, str]:
            final_log = ""
            final_note = ""
            results = supervisor.run(
                objective,
                run_worker=lambda name, worker_objective: _run_agent_job(
                    project_root=project_root,
                    workdir=workdir,
                    objective=f"[worker={name}] {worker_objective}",
                    state=state,
                    progress_hook=lambda m: progress_hook(f"[{name}] {m}"),
                    worker_name=name,
                ),
            )
            summary_lines = []
            for item in results:
                summary_lines.append(f"{item.name}: {item.final}")
                final_log = item.run_log
                final_note = item.profile_note
            return "\n".join(summary_lines), final_log, final_note

        try:
            final, run_log, note = await asyncio.to_thread(run_supervised)
            release_state(state, final, run_log, note)
            await ctx.send(f"完了 mode=supervise\n{final}\nlog={run_log}\n{note}")
        except Exception as exc:
            release_state(state, f"error: {type(exc).__name__}: {exc}")
            await ctx.send(f"エラーで停止しました: {type(exc).__name__}: {exc}")

    @bot.command(name="autopr")
    async def autopr_cmd(ctx: commands.Context, *, objective: str) -> None:
        state = await with_channel_lock(ctx, "autopr", objective)
        if state is None:
            return
        if not _is_git_repo(workdir):
            release_state(state, "error: workdir is not a git repository")
            await ctx.reply("autoprはGitリポジトリが必要です。`git init` または既存repoで再実行してください。")
            return
        runtime_loop = asyncio.get_running_loop()
        await ctx.reply(f"開始 mode=autopr\nobjective={objective}")

        async def send_msg(message: str) -> None:
            try:
                await ctx.send(message[:1800])
            except Exception:
                pass

        def progress_hook(msg: str) -> None:
            runtime_loop.call_soon_threadsafe(asyncio.create_task, send_msg(msg))

        def run_autopr() -> tuple[str, str, str]:
            branch = _ensure_branch(workdir)
            progress_hook(f"作業ブランチ: {branch}")
            worker_objective = (
                f"{objective}\n"
                "必ず変更後に検証を実施する。優先順: pytest, npm test, make test。"
                "最後に変更要約を `runs/pr_body.md` と `runs/commit_msg.txt` に出力して終了する。"
            )
            final, run_log, note = _run_agent_job(
                project_root=project_root,
                workdir=workdir,
                objective=worker_objective,
                state=state,
                progress_hook=progress_hook,
                worker_name="autopr",
            )
            status_code, status = _run_cmd("git status --short", workdir)
            diff_code, diff = _run_cmd("git diff --", workdir)
            summary = (
                f"branch={branch}\n"
                f"final={final}\n"
                f"status_code={status_code}, diff_code={diff_code}\n"
                f"status:\n{status[:1000]}\n\n"
                f"diff_preview:\n{diff[:2000]}"
            )
            summary_path = project_root / "runs" / "autopr-summary.txt"
            summary_path.write_text(summary, encoding="utf-8")
            return final, run_log, f"{note}\nautopr_summary={summary_path}"

        try:
            final, run_log, note = await asyncio.to_thread(run_autopr)
            release_state(state, final, run_log, note)
            await ctx.send(f"完了 mode=autopr\nfinal={final}\nlog={run_log}\n{note}")
        except Exception as exc:
            release_state(state, f"error: {type(exc).__name__}: {exc}")
            await ctx.send(f"エラーで停止しました: {type(exc).__name__}: {exc}")

    @bot.command(name="deliver")
    async def deliver_cmd(ctx: commands.Context, *, objective: str) -> None:
        state = await with_channel_lock(ctx, "deliver", objective)
        if state is None:
            return
        runtime_loop = asyncio.get_running_loop()
        await ctx.reply(f"開始 mode=deliver\nobjective={objective}")

        async def send_msg(message: str) -> None:
            try:
                await ctx.send(message[:1800])
            except Exception:
                pass

        def progress_hook(msg: str) -> None:
            runtime_loop.call_soon_threadsafe(asyncio.create_task, send_msg(msg))

        def run_deliver() -> tuple[str, str, str]:
            max_repair_loops = max(1, _env_int("DISCORD_MAX_REPAIR_LOOPS", 3))
            latest_log = ""
            latest_note = ""
            impl_final = ""
            progress_hook("phase=implement")
            impl_final, impl_log, impl_note = _run_agent_job(
                project_root=project_root,
                workdir=workdir,
                objective=(
                    f"{objective}\n"
                    "要件: 実装後に検証で確認できる状態にする。"
                ),
                state=state,
                progress_hook=lambda m: progress_hook(f"[implement] {m}"),
                worker_name="deliver_implementer",
                max_steps_override=12,
                forced_allowed_tools=["read_file", "write_file", "shell", "finish"],
                noop_streak_limit=3,
            )
            latest_log = impl_log
            latest_note = impl_note

            validation_path = project_root / "runs" / "validation_report.md"
            dod_path = project_root / "runs" / "dod_report.md"
            release_note_path = project_root / "runs" / "release_note.md"
            ok, report = _run_validation_suite(workdir)
            validation_path.write_text(report, encoding="utf-8")
            alerts = _extract_validation_alert_lines(report)
            dod_ok, dod_report = _evaluate_dod(workdir, ok, alerts)
            dod_path.write_text(dod_report, encoding="utf-8")
            progress_hook(f"validation: ok={ok} report={validation_path}")

            attempt = 0
            while not ok and attempt < max_repair_loops:
                attempt += 1
                progress_hook(f"phase=repair attempt={attempt}")
                fix_objective = (
                    "以下の検証レポートに基づいて失敗を修正してください。"
                    f"\nreport_path=runs/validation_report.md\n\n{objective}"
                )
                fix_final, fix_log, fix_note = _run_agent_job(
                    project_root=project_root,
                    workdir=workdir,
                    objective=fix_objective,
                    state=state,
                    progress_hook=lambda m: progress_hook(f"[repair-{attempt}] {m}"),
                    worker_name="deliver_implementer",
                    max_steps_override=12,
                    forced_allowed_tools=["read_file", "write_file", "shell", "finish"],
                    noop_streak_limit=3,
                )
                latest_log = fix_log
                latest_note = fix_note
                ok, report = _run_validation_suite(workdir)
                validation_path.write_text(report, encoding="utf-8")
                alerts = _extract_validation_alert_lines(report)
                dod_ok, dod_report = _evaluate_dod(workdir, ok, alerts)
                dod_path.write_text(dod_report, encoding="utf-8")
                progress_hook(f"validation_retry: ok={ok} attempt={attempt}")

            alert_text = "\n".join(f"- {line}" for line in alerts) if alerts else "- (none)"
            doc_objective = (
                "以下の必須見出しで `runs/summary.md` を更新すること: "
                "Objective, Changes, Validation, Unresolved, Next Steps. "
                "Validationは `runs/validation_report.md` を参照して記述する。"
                "\nさらに Validation Alerts セクションを作り、以下の抽出行を必ず転記すること:\n"
                f"{alert_text}"
            )
            doc_final, doc_log, doc_note = _run_agent_job(
                project_root=project_root,
                workdir=workdir,
                objective=doc_objective,
                state=state,
                progress_hook=lambda m: progress_hook(f"[documenter] {m}"),
                worker_name="documenter",
            )
            latest_log = doc_log
            latest_note = doc_note

            implementer_bad = (
                "stopped:" in impl_final.lower()
                and "pytest found no tests" not in impl_final.lower()
                and "pytest success repeated" not in impl_final.lower()
                and "no-op repeated" not in impl_final.lower()
            )
            has_alerts = len(alerts) > 0
            status = "SUCCESS" if (ok and dod_ok and not implementer_bad and not has_alerts) else "NEEDS_REVIEW"
            release_note = _make_release_note(
                workdir=workdir,
                objective=objective,
                deliver_status=status,
                implementer_final=impl_final,
                validation_report_path=validation_path,
                dod_report_path=dod_path,
            )
            release_note_path.write_text(release_note, encoding="utf-8")
            review_report = _review_findings(workdir, project_root)
            review_report_path = project_root / "runs" / "review_report.md"
            review_report_path.write_text(review_report, encoding="utf-8")
            pr_ready_path = _write_pr_ready_bundle(project_root, workdir)
            final = (
                f"deliver={status}\n"
                f"implementer={impl_final}\n"
                f"documenter={doc_final}\n"
                f"validation_report={validation_path}\n"
                f"dod_report={dod_path}\n"
                f"release_note={release_note_path}\n"
                f"review_report={review_report_path}\n"
                f"pr_ready={pr_ready_path}\n"
                f"repair_attempts={attempt}/{max_repair_loops}\n"
                f"validation_alerts={len(alerts)}"
            )
            return final, latest_log, latest_note

        try:
            final, run_log, note = await asyncio.to_thread(run_deliver)
            release_state(state, final, run_log, note)
            await ctx.send(f"完了 mode=deliver\n{final}\nlog={run_log}\n{note}")
        except Exception as exc:
            release_state(state, f"error: {type(exc).__name__}: {exc}")
            await ctx.send(f"エラーで停止しました: {type(exc).__name__}: {exc}")

    bot.run(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
