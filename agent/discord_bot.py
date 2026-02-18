from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import discord
from discord.ext import commands
from openai import OpenAI

from .adapters import AntigravityAdapter, CodexAdapter, ExternalAgentAdapter
from .budget import BudgetManager
from .config import AgentConfig
from .history import HistoryCompressor
from .memory import JsonlMemory
from .providers import OpenAIProvider
from .router import Router
from .runner import AgentRunner
from .self_improve import (
    FailurePatternProfile,
    RouterProfile,
    analyze_run_log,
    load_router_profile,
    load_failure_profile,
    recommend_guardrails,
    save_failure_profile,
    save_router_profile,
    tune_failure_profile,
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


@dataclass
class AutoPilotState:
    enabled: bool = False
    interval_seconds: int = 900
    objectives: list[str] = field(default_factory=list)
    next_index: int = 0
    last_run_at: str = ""
    last_final: str = ""
    last_log: str = ""
    last_daily_summary_date: str = ""
    last_daily_summary_path: str = ""
    task: asyncio.Task[None] | None = None


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


def _failure_profile_path(project_root: Path) -> Path:
    return project_root / ".agent_state" / "failure_patterns.json"


def _autopilot_objectives_path(project_root: Path) -> Path:
    return project_root / ".agent_state" / "autopilot_objectives.txt"


def _autopilot_history_path(project_root: Path) -> Path:
    return project_root / ".agent_state" / "autopilot_history.jsonl"


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


def _default_autopilot_objectives() -> list[str]:
    return [
        "あなた専用タスク: 直近作業から次の小さな一手を3件に絞り、`runs/auto_todo.md` を更新して finish する。",
        "あなた専用タスク: 直近runログを見て失敗/停滞の傾向を1分で読める形で `runs/auto_health.md` に更新して finish する。",
    ]


def _load_autopilot_objectives(project_root: Path) -> list[str]:
    profile_file = _autopilot_objectives_path(project_root)
    if profile_file.exists():
        rows = [line.strip() for line in profile_file.read_text(encoding="utf-8").splitlines()]
        saved = [line for line in rows if line and not line.startswith("#")]
        if saved:
            return saved
    raw = os.getenv("DISCORD_AUTOPILOT_OBJECTIVES", "").strip()
    if not raw:
        return _default_autopilot_objectives()
    parts = [item.strip() for item in raw.split("||")]
    values = [item for item in parts if item]
    return values or _default_autopilot_objectives()


def _save_autopilot_objectives(project_root: Path, objectives: list[str]) -> Path:
    path = _autopilot_objectives_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(objectives) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path


def _append_autopilot_history(project_root: Path, record: dict) -> Path:
    path = _autopilot_history_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _tail_autopilot_history(project_root: Path, limit: int = 20) -> list[dict]:
    path = _autopilot_history_path(project_root)
    if not path.exists():
        return []
    rows = path.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    for raw in rows[-max(1, limit) :]:
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default


def _transcribe_audio_file(path: Path) -> str:
    model = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"
    client = OpenAI()
    with path.open("rb") as f:
        text = client.audio.transcriptions.create(model=model, file=f, response_format="text")
    return str(text).strip()
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


def _current_branch(workdir: Path) -> str:
    code, out = _run_cmd("git rev-parse --abbrev-ref HEAD", workdir)
    if code != 0 or not out:
        return "unknown"
    return out.splitlines()[-1].strip()


def _has_git_identity(workdir: Path) -> bool:
    name_code, name = _run_cmd("git config --get user.name", workdir)
    email_code, email = _run_cmd("git config --get user.email", workdir)
    return (
        name_code == 0
        and email_code == 0
        and bool(name.strip())
        and bool(email.strip())
    )


def _collect_changed_files(workdir: Path) -> list[str]:
    # Use name-only lists instead of parsing porcelain columns to avoid
    # edge cases caused by local git status formatting.
    commands = [
        "git diff --name-only",
        "git diff --name-only --cached",
        "git ls-files --others --exclude-standard",
    ]
    files: list[str] = []
    for command in commands:
        code, out = _run_cmd(command, workdir)
        if code != 0 or not out or out == "(no output)":
            continue
        for raw in out.splitlines():
            path = raw.strip()
            if not path:
                continue
            files.append(path)
    # de-duplicate while preserving order
    unique: list[str] = []
    for path in files:
        if path in unique:
            continue
        unique.append(path)
    return unique


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
    return [".env", ".venv/", ".agent_state/", "runs/", "path/to/", "agent/__pycache__/", "__pycache__/"]


def _forbidden_extensions() -> list[str]:
    raw = os.getenv("DISCORD_FORBIDDEN_EXTENSIONS", "").strip()
    if raw:
        return [token.strip().lower() for token in raw.split(",") if token.strip()]
    return [".pyc", ".pyo", ".pyd"]


def _forbidden_extension_hits(paths: list[str]) -> list[str]:
    exts = _forbidden_extensions()
    hits: list[str] = []
    for path in paths:
        lowered = path.lower()
        if any(lowered.endswith(ext) for ext in exts):
            hits.append(path)
    return hits


def _autopilot_allowed_prefixes() -> list[str]:
    raw = os.getenv("DISCORD_AUTOPILOT_ALLOWED_PREFIXES", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return ["runs/"]


def _evaluate_autopilot_guard(workdir: Path, project_root: Path) -> tuple[bool, Path]:
    all_changed = _collect_changed_files(workdir)
    max_changed = max(1, _env_int("DISCORD_AUTOPILOT_MAX_CHANGED_FILES", 20))
    allowed_prefixes = _autopilot_allowed_prefixes()
    filtered = [path for path in all_changed if not _is_runtime_noise(path)]
    disallowed = [
        path for path in filtered if not any(path.startswith(prefix) for prefix in allowed_prefixes)
    ]
    ok = len(disallowed) == 0 and len(filtered) <= max_changed
    report_path = project_root / "runs" / "autopilot_guard_report.md"
    lines = [
        "# Autopilot Guard Report",
        "",
        f"- status: {'PASS' if ok else 'FAIL'}",
        f"- changed_files_count: {len(filtered)} (limit={max_changed})",
        f"- disallowed_count: {len(disallowed)}",
        f"- allowed_prefixes: {', '.join(allowed_prefixes)}",
        "",
        "## Changed Files",
    ]
    lines.extend([f"- {path}" for path in filtered[:120]] or ["- (none)"])
    lines.extend(["", "## Disallowed"])
    lines.extend([f"- {path}" for path in disallowed[:120]] or ["- (none)"])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return ok, report_path


def _diff_line_counts(workdir: Path) -> tuple[int, int]:
    code, out = _run_cmd("git diff --numstat --", workdir)
    if code != 0 or not out:
        return 0, 0
    added = 0
    removed = 0
    for raw in out.splitlines():
        parts = raw.split("\t")
        if len(parts) < 3:
            continue
        a, d = parts[0].strip(), parts[1].strip()
        if a.isdigit():
            added += int(a)
        if d.isdigit():
            removed += int(d)
    return added, removed


def _worktree_fingerprint(workdir: Path) -> tuple[int, int, int]:
    files = _effective_changed_files(workdir)
    added, removed = _diff_line_counts(workdir)
    return len(files), added, removed


def _repair_attempt_limit(failure_type: str, default_limit: int) -> int:
    key_map = {
        "test_failure": "DISCORD_REPAIR_TEST_MAX_ATTEMPTS",
        "syntax_failure": "DISCORD_REPAIR_SYNTAX_MAX_ATTEMPTS",
        "permission_failure": "DISCORD_REPAIR_PERMISSION_MAX_ATTEMPTS",
        "unknown_failure": "DISCORD_REPAIR_UNKNOWN_MAX_ATTEMPTS",
    }
    key = key_map.get(failure_type)
    if key is None:
        return max(1, default_limit)
    return max(1, _env_int(key, default_limit))


def _repair_stagnant_limit(failure_type: str) -> int:
    key_map = {
        "test_failure": "DISCORD_REPAIR_TEST_STAGNANT_LIMIT",
        "syntax_failure": "DISCORD_REPAIR_SYNTAX_STAGNANT_LIMIT",
        "permission_failure": "DISCORD_REPAIR_PERMISSION_STAGNANT_LIMIT",
        "unknown_failure": "DISCORD_REPAIR_UNKNOWN_STAGNANT_LIMIT",
    }
    default_map = {
        "test_failure": 2,
        "syntax_failure": 1,
        "permission_failure": 1,
        "unknown_failure": 2,
    }
    key = key_map.get(failure_type, "DISCORD_REPAIR_UNKNOWN_STAGNANT_LIMIT")
    default_value = default_map.get(failure_type, 2)
    return max(1, _env_int(key, default_value))


def _evaluate_dod(workdir: Path, validation_ok: bool, alerts: list[str]) -> tuple[bool, str]:
    max_changed = max(1, _env_int("DISCORD_MAX_CHANGED_FILES", 25))
    min_diff_lines = max(1, _env_int("DISCORD_MIN_DIFF_LINES", 1))
    files = _effective_changed_files(workdir)
    added_lines, removed_lines = _diff_line_counts(workdir)
    diff_lines = added_lines + removed_lines
    forbidden_prefixes = _forbidden_path_prefixes()
    forbidden_hits = [path for path in files if any(path.startswith(prefix) for prefix in forbidden_prefixes)]
    forbidden_ext_hits = _forbidden_extension_hits(files)
    over_limit = len(files) > max_changed
    has_alerts = len(alerts) > 0
    insufficient_diff = diff_lines < min_diff_lines
    no_effective_changes = len(files) == 0

    ok = (
        validation_ok
        and not has_alerts
        and not forbidden_hits
        and not forbidden_ext_hits
        and not over_limit
        and not insufficient_diff
        and not no_effective_changes
    )
    lines = [
        "# DoD Report",
        "",
        f"- validation_ok: {validation_ok}",
        f"- validation_alerts: {len(alerts)}",
        f"- changed_files_count: {len(files)} (limit={max_changed})",
        f"- diff_lines_total: {diff_lines} (min={min_diff_lines})",
        f"- forbidden_hits_count: {len(forbidden_hits)}",
        f"- forbidden_extension_hits_count: {len(forbidden_ext_hits)}",
        f"- insufficient_diff: {insufficient_diff}",
        f"- no_effective_changes: {no_effective_changes}",
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
    lines.extend(["", "## Forbidden Extension Hits"])
    if forbidden_ext_hits:
        lines.extend(f"- {path}" for path in forbidden_ext_hits[:120])
    else:
        lines.append("- (none)")
    if alerts:
        lines.extend(["", "## Validation Alerts"])
        lines.extend(f"- {line}" for line in alerts)
    return ok, "\n".join(lines)


def _review_findings(workdir: Path, project_root: Path) -> str:
    files = _effective_changed_files(workdir)
    forbidden_prefixes = _forbidden_path_prefixes()
    forbidden_exts = _forbidden_extensions()
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
    forbidden_ext_hits = _forbidden_extension_hits(files)
    if forbidden_ext_hits:
        critical.append(
            f"Forbidden extension changes ({','.join(forbidden_exts)}): {', '.join(forbidden_ext_hits[:8])}"
        )
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
        high.append("Diff includes TODO/FIXME markers.")
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


def _low_quality_commit_message_reason(message: str) -> str | None:
    text = message.strip()
    lower = text.lower()
    if len(text) < 10:
        return "短すぎます。変更内容が分かる文にしてください。"
    if len(text) > 100:
        return "長すぎます。100文字以内に要約してください。"
    if ":" not in text:
        return "`type: summary` 形式を推奨します（例: `feat: tighten ...`）。"
    request_like_tokens = (
        "してください",
        "してほしい",
        "お願いします",
        "まとめて",
        "リサーチ",
        "検索して",
        "調査して",
        "実行して",
    )
    if any(token in text for token in request_like_tokens):
        return "依頼文ではなく、実装結果を要約したコミット文にしてください。"
    valid_prefixes = ("feat:", "fix:", "chore:", "docs:", "refactor:", "test:", "perf:", "ci:", "build:")
    if not lower.startswith(valid_prefixes):
        return "先頭に `feat:` などの種別を付けてください。"
    summary = text.split(":", 1)[1].strip() if ":" in text else ""
    if len(summary) < 6:
        return "コロン以降の要約が短すぎます。"
    return None


def _write_pr_ready_bundle(project_root: Path, workdir: Path) -> Path:
    release_note_path = project_root / "runs" / "release_note.md"
    review_report_path = project_root / "runs" / "review_report.md"
    dod_report_path = project_root / "runs" / "dod_report.md"
    validation_report_path = project_root / "runs" / "validation_report.md"
    bundle_path = project_root / "runs" / "pr_ready.md"

    dod = _dod_status(project_root)
    review = _review_status(project_root)
    changed_files = _effective_changed_files(workdir)
    branch = _current_branch(workdir)

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


def _infer_failure_kind(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("permission denied", "authentication", "forbidden", "blocked")):
        return "permission_failure"
    if any(
        token in lowered
        for token in ("syntaxerror", "jsondecodeerror", "indentationerror", "typeerror", "traceback")
    ):
        return "syntax_failure"
    if any(token in lowered for token in ("pytest", "failed", "assert", "no tests ran", "collected 0 items")):
        return "test_failure"
    return "unknown_failure"


def _write_manual_checklist(
    project_root: Path,
    objective: str,
    final_message: str,
    run_log: str,
) -> Path:
    failure_kind = _infer_failure_kind(final_message)
    path = project_root / "runs" / "manual_checklist.md"
    lines = [
        "# Manual Checklist",
        "",
        f"- generated_at: {datetime.now(UTC).isoformat()}",
        f"- failure_kind: {failure_kind}",
        f"- objective: {objective}",
        f"- final: {final_message}",
        f"- run_log: {run_log or '(none)'}",
        "",
        "## Actions",
        "- 1) run `!status` and confirm current state",
        "- 2) read `runs/validation_report.md` and identify the first failing command",
        "- 3) run `!review` and check Critical/High findings",
        "- 4) apply minimal fix and rerun `!deliver ...`",
        "- 5) if auth/permission related, refresh credentials and retry",
        "",
        "## Notes",
        "- このチェックリストは自動生成です。必要に応じて追記してください。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_daily_summary(project_root: Path, auto: AutoPilotState) -> Path:
    runs_dir = project_root / "runs"
    summary_path = runs_dir / "daily_summary.md"
    latest = _latest_logs(runs_dir, limit=5)
    auto_todo = runs_dir / "auto_todo.md"
    auto_health = runs_dir / "auto_health.md"
    todo_text = auto_todo.read_text(encoding="utf-8")[:1200] if auto_todo.exists() else "(none)"
    health_text = auto_health.read_text(encoding="utf-8")[:1200] if auto_health.exists() else "(none)"
    lines = [
        "# Daily Summary",
        "",
        f"- date: {datetime.now(UTC).date().isoformat()}",
        f"- autopilot_last_run: {auto.last_run_at or '(none)'}",
        f"- autopilot_last_final: {auto.last_final or '(none)'}",
        "",
        "## Latest Runs",
    ]
    lines.extend([f"- {item.name}" for item in latest] or ["- (none)"])
    lines.extend(
        [
            "",
            "## Auto TODO",
            "```text",
            todo_text,
            "```",
            "",
            "## Auto Health",
            "```text",
            health_text,
            "```",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


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


def _has_no_tests_signal(command: str, output: str) -> bool:
    command_l = command.lower()
    if "pytest" not in command_l:
        return False
    output_l = output.lower()
    return "collected 0 items" in output_l or "no tests ran" in output_l


def _run_validation_suite(workdir: Path) -> tuple[bool, str]:
    commands = _validation_commands(workdir)
    if not commands:
        return True, "No validation commands for project type."

    fail_on_no_tests = _env_bool("DISCORD_FAIL_ON_NO_TESTS", True)
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
        if fail_on_no_tests and _has_no_tests_signal(command, out):
            ok_all = False
            lines.append("- policy_violation: pytest reported no tests; treated as failure")
            lines.append("")
    return ok_all, "\n".join(lines)


def _has_no_tests_signal_in_text(text: str) -> bool:
    lowered = text.lower()
    return "collected 0 items" in lowered or "no tests ran" in lowered


def _detect_no_tests_in_runlog(run_log_path: str) -> bool:
    path = Path(run_log_path)
    if not path.exists():
        return False
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except Exception:
            continue
        if row.get("event") != "observation":
            continue
        output = str(row.get("output", ""))
        if _has_no_tests_signal_in_text(output):
            return True
    return False


def _classify_validation_failure(report_text: str) -> str:
    sections: list[tuple[str, int | None, str]] = []
    current_command = ""
    current_exit_code: int | None = None
    body_lines: list[str] = []

    for raw in report_text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("## Command:"):
            if current_command:
                sections.append((current_command, current_exit_code, "\n".join(body_lines)))
            current_command = line.replace("## Command:", "", 1).strip().strip("`")
            current_exit_code = None
            body_lines = []
            continue
        if line.startswith("- exit_code:"):
            value = line.split(":", 1)[1].strip()
            try:
                current_exit_code = int(value)
            except ValueError:
                current_exit_code = None
            continue
        body_lines.append(line)

    if current_command:
        sections.append((current_command, current_exit_code, "\n".join(body_lines)))

    failing_sections = [
        (cmd, code, body)
        for cmd, code, body in sections
        if (code is not None and code != 0) or "policy_violation" in body.lower()
    ]

    permission_tokens = (
        "permission denied",
        "operation not permitted",
        "access denied",
        "authentication failed",
        "could not read from remote repository",
        "403",
        "forbidden",
        "blocked by strict shell allowlist",
    )
    syntax_tokens = (
        "syntaxerror",
        "jsondecodeerror",
        "indentationerror",
        "nameerror",
        "typeerror",
    )

    for command, _, body in failing_sections:
        text = f"{command}\n{body}".lower()
        if any(token in text for token in permission_tokens):
            return "permission_failure"

    for command, _, body in failing_sections:
        text = f"{command}\n{body}".lower()
        if "pytest" in command.lower() or "no tests ran" in text or "collected 0 items" in text:
            return "test_failure"

    for command, _, body in failing_sections:
        text = f"{command}\n{body}".lower()
        if "compileall" in command.lower() or any(token in text for token in syntax_tokens):
            return "syntax_failure"

    if failing_sections:
        return "unknown_failure"

    text = report_text.lower()
    if any(token in text for token in permission_tokens):
        return "permission_failure"
    if "pytest" in text:
        return "test_failure"
    if "compileall" in text or any(token in text for token in syntax_tokens):
        return "syntax_failure"
    return "unknown_failure"


def _build_repair_objective(base_objective: str, failure_type: str) -> str:
    if failure_type == "test_failure":
        strategy = (
            "修復戦略: テスト失敗を最優先。"
            " failing test/exit_code を特定し、最小変更で修正し、pytestを再実行して結果を確認する。"
            " テストが0件の場合は有効なテスト対象を指定して再実行する。"
        )
    elif failure_type == "syntax_failure":
        strategy = (
            "修復戦略: 構文/実行時エラーを最優先。"
            " tracebackやcompileall出力の先頭エラーから順に修正し、compileall -> pytest の順で再検証する。"
        )
    elif failure_type == "permission_failure":
        strategy = (
            "修復戦略: 権限/認証エラー。"
            " コード変更で解決できない場合が多いため、原因を明示して安全に停止し、"
            " 必要な手動操作（認証/権限設定）を summary に残す。"
        )
    else:
        strategy = (
            "修復戦略: 汎用。"
            " validation_report の先頭失敗コマンドを起点に原因を切り分け、"
            " 最小変更で修正後に再検証する。"
        )
    return (
        "以下の検証レポートに基づいて失敗を修正してください。"
        f"\nreport_path=runs/validation_report.md\nfailure_type={failure_type}\n"
        f"{strategy}\n\n{base_objective}"
    )


def _build_repair_objective_with_stagnation_note(
    base_objective: str,
    failure_type: str,
    stagnant_repair_count: int,
) -> str:
    objective = _build_repair_objective(base_objective, failure_type)
    if stagnant_repair_count <= 0:
        return objective
    return (
        objective
        + "\n追加制約: 前回までの修復で有効な差分がほぼ増えていません。"
        " 今回は必ず `write_file` を使って最小限の修正を行い、"
        " その後に1回だけpytest/compileallで再検証してfinishしてください。"
    )


def _extract_validation_alert_lines(report_text: str) -> list[str]:
    alerts: list[str] = []
    keywords = (
        "error",
        "failed",
        "warning",
        "traceback",
        "exception",
        "not found",
        "no tests ran",
        "collected 0 items",
        "policy_violation",
    )
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


def _extract_validation_brief(report_text: str) -> list[str]:
    brief: list[str] = []
    for raw in report_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if line.startswith("## Command:"):
            brief.append(line)
        elif line.startswith("- exit_code:"):
            brief.append(line)
        elif " passed in " in lower or " failed in " in lower:
            brief.append(line)
        if len(brief) >= 12:
            break
    return brief


def _write_summary_template(
    *,
    workdir: Path,
    project_root: Path,
    objective: str,
    impl_final: str,
    validation_ok: bool,
    validation_report: str,
    alerts: list[str],
) -> str:
    summary_path = project_root / "runs" / "summary.md"
    changed_files = _effective_changed_files(workdir)
    validation_brief = _extract_validation_brief(validation_report)
    lines = [
        "## Objective",
        objective,
        "",
        "## Changes",
    ]
    lines.extend([f"- {path}" for path in changed_files[:80]] or ["- (none)"])
    lines.extend(
        [
            "",
            "## Validation",
            f"- status: {'ok' if validation_ok else 'failed'}",
            f"- implementer: {impl_final}",
        ]
    )
    lines.extend([f"- {line}" for line in validation_brief] or ["- (no validation details)"])
    lines.extend(
        [
            "",
            "## Validation Alerts",
        ]
    )
    lines.extend([f"- {line}" for line in alerts] or ["- (none)"])
    lines.extend(
        [
            "",
            "## Unresolved",
            "- (none)",
            "",
            "## Next Steps",
            "- Run `!review` and confirm `status=OK` before `!approve`.",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return str(summary_path)


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
    failure_profile_file = _failure_profile_path(project_root)
    failure_profile = load_failure_profile(failure_profile_file)
    config.max_steps, effective_noop_streak_limit, guardrail_note = recommend_guardrails(
        failure_profile,
        config.max_steps,
        noop_streak_limit,
    )
    planner = OpenAIProvider(model=config.model)
    approval_policy = os.getenv("DISCORD_APPROVAL_POLICY", "allow").strip().lower() or "allow"
    strict_shell_allowlist = False
    extra_safe_shell_prefixes: tuple[str, ...] | None = None
    shell_allow_prefixes: tuple[str, ...] | None = None
    write_allow_prefixes: tuple[str, ...] | None = None
    write_allow_extensions: tuple[str, ...] | None = None
    write_allow_paths: tuple[str, ...] | None = None
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
    if worker_name == "autopilot":
        write_allow_prefixes = ("runs/",)
        write_allow_extensions = (".md",)
        write_allow_paths = ("runs/auto_todo.md", "runs/auto_health.md")
        objective = (
            objective
            + "\n制約: autopilot の書き込み先は `runs/auto_todo.md` と `runs/auto_health.md` のみ。"
            + " それ以外のパスへは write_file/append_file を使わないこと。"
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
        write_allow_prefixes=write_allow_prefixes,
        write_allow_extensions=write_allow_extensions,
        write_allow_paths=write_allow_paths,
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
        noop_streak_limit=effective_noop_streak_limit,
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
    tuned_failure_profile = tune_failure_profile(
        failure_profile or FailurePatternProfile(),
        signals,
    )
    save_failure_profile(failure_profile_file, tuned_failure_profile)
    note = (
        f"next: max_external_calls={tuned_profile.max_external_calls}, "
        f"escalate_after_failures={tuned_profile.escalate_after_failures}, "
        f"compress_recent_steps={tuned_profile.compress_recent_steps}, "
        f"guardrails={guardrail_note}"
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
    autopilot_states: dict[int, AutoPilotState] = {}
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    def get_autopilot_state(channel_id: int) -> AutoPilotState:
        if channel_id not in autopilot_states:
            autopilot_states[channel_id] = AutoPilotState(
                enabled=False,
                interval_seconds=max(60, _env_int("DISCORD_AUTOPILOT_INTERVAL_SECONDS", 900)),
                objectives=_load_autopilot_objectives(project_root),
            )
        return autopilot_states[channel_id]

    async def _run_autopilot_once(channel_id: int, trigger: str = "timer") -> tuple[bool, str]:
        if not is_allowed_channel(channel_id):
            return False, "blocked channel"
        state = registry.get(channel_id)
        auto = get_autopilot_state(channel_id)
        if not auto.objectives:
            return False, "no objectives configured"
        with state.lock:
            if state.running:
                return False, "already running"
            objective = auto.objectives[auto.next_index % len(auto.objectives)]
            auto.next_index = (auto.next_index + 1) % len(auto.objectives)
            state.running = True
            state.mode = "autopilot"
            state.cancelled = False
            state.objective = objective
            state.step = 0
            state.last_event = "queued"
            state.final_message = ""
            state.output_tail = ""
            state.profile_note = ""

        channel = bot.get_channel(channel_id)
        started_at = datetime.now(UTC)
        if isinstance(channel, discord.abc.Messageable):
            try:
                await channel.send(
                    f"autopilot開始 trigger={trigger}\nobjective={objective}"
                )
            except Exception:
                pass

        def progress_hook(_: str) -> None:
            return

        try:
            final, run_log, note = await asyncio.to_thread(
                _run_agent_job,
                project_root=project_root,
                workdir=workdir,
                objective=objective,
                state=state,
                progress_hook=progress_hook,
                worker_name="autopilot",
                max_steps_override=max(3, _env_int("DISCORD_AUTOPILOT_MAX_STEPS", 8)),
                forced_allowed_tools=["read_file", "write_file", "finish"],
                noop_streak_limit=2,
            )
            release_state(state, final, run_log, note)
            auto.last_run_at = datetime.now(UTC).isoformat()
            auto.last_final = final
            auto.last_log = run_log
            guard_ok, guard_report_path = _evaluate_autopilot_guard(workdir, project_root)
            if not guard_ok:
                final = "Stopped: autopilot guard violation detected."
                auto.last_final = final
                if isinstance(channel, discord.abc.Messageable):
                    try:
                        await channel.send(f"autopilotガード違反: {guard_report_path}")
                    except Exception:
                        pass
            final_lower = final.lower()
            if any(token in final_lower for token in ("stopped:", "needs_review", "error", "failed")):
                checklist_path = _write_manual_checklist(project_root, objective, final, run_log)
                if isinstance(channel, discord.abc.Messageable):
                    try:
                        await channel.send(f"autopilot失敗チェックリストを生成しました: {checklist_path}")
                    except Exception:
                        pass
            if isinstance(channel, discord.abc.Messageable):
                try:
                    await channel.send(
                        f"autopilot完了\nfinal={final}\nlog={run_log}\n{note}"
                    )
                except Exception:
                    pass
            history_path = _append_autopilot_history(
                project_root,
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "channel_id": channel_id,
                    "trigger": trigger,
                    "objective": objective,
                    "final": final,
                    "run_log": run_log,
                    "duration_seconds": int((datetime.now(UTC) - started_at).total_seconds()),
                    "success": final.lower().startswith("finished:") and guard_ok,
                    "guard_ok": guard_ok,
                    "guard_report": str(guard_report_path),
                },
            )
            if isinstance(channel, discord.abc.Messageable):
                try:
                    await channel.send(f"autopilot履歴を更新: {history_path}")
                except Exception:
                    pass
            return True, final
        except Exception as exc:
            err = f"error: {type(exc).__name__}: {exc}"
            release_state(state, err)
            auto.last_run_at = datetime.now(UTC).isoformat()
            auto.last_final = err
            history_path = _append_autopilot_history(
                project_root,
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "channel_id": channel_id,
                    "trigger": trigger,
                    "objective": objective,
                    "final": err,
                    "run_log": "",
                    "duration_seconds": int((datetime.now(UTC) - started_at).total_seconds()),
                    "success": False,
                    "guard_ok": False,
                    "guard_report": "",
                },
            )
            if isinstance(channel, discord.abc.Messageable):
                try:
                    await channel.send(f"autopilotエラー: {err}\nhistory={history_path}")
                except Exception:
                    pass
            return False, err

    async def _maybe_post_daily_summary(channel_id: int, force: bool = False) -> None:
        auto = get_autopilot_state(channel_id)
        today = datetime.now(UTC).date().isoformat()
        if not force and auto.last_daily_summary_date == today:
            return
        summary_path = _write_daily_summary(project_root, auto)
        auto.last_daily_summary_date = today
        auto.last_daily_summary_path = str(summary_path)
        channel = bot.get_channel(channel_id)
        if isinstance(channel, discord.abc.Messageable):
            try:
                await channel.send(f"daily_summary更新: {summary_path}")
            except Exception:
                pass

    async def _autopilot_loop(channel_id: int) -> None:
        auto = get_autopilot_state(channel_id)
        while auto.enabled:
            await _run_autopilot_once(channel_id, trigger="timer")
            await _maybe_post_daily_summary(channel_id)
            await asyncio.sleep(max(30, auto.interval_seconds))

    @bot.event
    async def on_ready() -> None:
        print(f"Discord bot logged in as {bot.user}", flush=True)
        print(
            "Commands: !agent !supervise !deliver !autopr !review !status !cancel !runs !tail !diff !approve !rollback !auto_on !auto_off !auto_status !auto_now !auto_set !auto_daily !voice",
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

    @bot.command(name="auto_on")
    async def auto_on_cmd(ctx: commands.Context, interval_seconds: int | None = None) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        auto = get_autopilot_state(ctx.channel.id)
        if interval_seconds is not None:
            auto.interval_seconds = max(60, interval_seconds)
        auto.enabled = True
        if auto.task is None or auto.task.done():
            auto.task = asyncio.create_task(_autopilot_loop(ctx.channel.id))
        await ctx.reply(
            "autopilotを有効化しました。\n"
            f"interval_seconds={auto.interval_seconds}\n"
            f"objectives={len(auto.objectives)}"
        )

    @bot.command(name="auto_off")
    async def auto_off_cmd(ctx: commands.Context) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        auto = get_autopilot_state(ctx.channel.id)
        auto.enabled = False
        if auto.task is not None:
            auto.task.cancel()
            auto.task = None
        await ctx.reply("autopilotを停止しました。")

    @bot.command(name="auto_status")
    async def auto_status_cmd(ctx: commands.Context) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        auto = get_autopilot_state(ctx.channel.id)
        history = _tail_autopilot_history(project_root, limit=20)
        success_count = sum(1 for row in history if bool(row.get("success")))
        fail_count = len(history) - success_count
        await ctx.reply(
            "autopilot状態\n"
            f"enabled={auto.enabled}\n"
            f"interval_seconds={auto.interval_seconds}\n"
            f"objectives={len(auto.objectives)}\n"
            f"last_run_at={auto.last_run_at or '(none)'}\n"
            f"last_final={auto.last_final or '(none)'}\n"
            f"last_log={auto.last_log or '(none)'}\n"
            f"last_daily_summary_date={auto.last_daily_summary_date or '(none)'}\n"
            f"last_daily_summary_path={auto.last_daily_summary_path or '(none)'}\n"
            f"history_recent={len(history)} success={success_count} fail={fail_count}"
        )

    @bot.command(name="auto_now")
    async def auto_now_cmd(ctx: commands.Context) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        ok, message = await _run_autopilot_once(ctx.channel.id, trigger="manual")
        if not ok:
            await ctx.reply(f"autopilot単発実行をスキップ: {message}")
            return
        await _maybe_post_daily_summary(ctx.channel.id)

    @bot.command(name="auto_set")
    async def auto_set_cmd(ctx: commands.Context, *, objectives: str) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        values = [item.strip() for item in objectives.split("||") if item.strip()]
        if not values:
            await ctx.reply("設定対象が空です。`||` 区切りで1件以上指定してください。")
            return
        auto = get_autopilot_state(ctx.channel.id)
        auto.objectives = values
        auto.next_index = 0
        path = _save_autopilot_objectives(project_root, values)
        await ctx.reply(
            "autopilot目標を更新しました。\n"
            f"count={len(values)}\n"
            f"profile={path}"
        )

    @bot.command(name="auto_daily")
    async def auto_daily_cmd(ctx: commands.Context) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        await _maybe_post_daily_summary(ctx.channel.id, force=True)

    @bot.command(name="voice")
    async def voice_cmd(ctx: commands.Context, mode: str = "agent") -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        selected_mode = mode.strip().lower()
        if selected_mode not in {"agent", "deliver", "supervise"}:
            await ctx.reply("modeは `agent` / `deliver` / `supervise` のみ指定できます。例: `!voice deliver`")
            return
        if not ctx.message.attachments:
            await ctx.reply("音声ファイルを添付してください。例: `!voice agent` + m4a/mp3/wav")
            return
        attachment = ctx.message.attachments[0]
        max_mb = max(1, _env_int("DISCORD_VOICE_MAX_MB", 30))
        if attachment.size > max_mb * 1024 * 1024:
            await ctx.reply(f"音声ファイルが大きすぎます（上限: {max_mb}MB）。")
            return
        suffix = Path(attachment.filename or "voice.m4a").suffix or ".m4a"
        with tempfile.NamedTemporaryFile(prefix="voice-", suffix=suffix, delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            await attachment.save(temp_path)
            transcript = await asyncio.to_thread(_transcribe_audio_file, temp_path)
        except Exception as exc:
            await ctx.reply(f"音声文字起こしに失敗しました: {type(exc).__name__}: {exc}")
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        if not transcript:
            await ctx.reply("文字起こし結果が空でした。もう一度試してください。")
            return
        await ctx.reply(
            "音声を文字起こししました。\n"
            f"mode={selected_mode}\n"
            f"objective={transcript[:500]}"
        )
        if selected_mode == "agent":
            await agent_cmd(ctx, objective=transcript)
        elif selected_mode == "deliver":
            await deliver_cmd(ctx, objective=transcript)
        else:
            await supervise_cmd(ctx, objective=transcript)

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
        enforce_branch = _env_bool("DISCORD_ENFORCE_APPROVE_BRANCH", True)
        branch = _current_branch(workdir)
        if enforce_branch and branch in {"main", "master"}:
            await ctx.reply(
                "approveをブロックしました。`main/master` への直接コミットは禁止です。\n"
                "先にブランチを切ってください。例: `git checkout -b feat/your-task`"
            )
            return
        if not _has_git_identity(workdir):
            await ctx.reply(
                "Gitの user.name / user.email が未設定です。先に設定してください。\n"
                "`git config --global user.name \"Your Name\"`\n"
                "`git config --global user.email \"you@example.com\"`"
            )
            return
        if _is_placeholder_commit_message(message):
            await ctx.reply(
                "コミットメッセージがダミー形式です。具体的な変更内容を書いてください。\n"
                "例: `feat: add deliver DoD gates and PR bundle generation`"
            )
            return
        low_quality_reason = _low_quality_commit_message_reason(message)
        if low_quality_reason is not None:
            await ctx.reply(
                "コミットメッセージ品質チェックでブロックしました。\n"
                f"- reason: {low_quality_reason}\n"
                "例: `feat: tighten deliver convergence and validation guardrails`"
            )
            return
        files = _effective_changed_files(workdir)
        forbidden_prefixes = _forbidden_path_prefixes()
        forbidden_hits = [path for path in files if any(path.startswith(prefix) for prefix in forbidden_prefixes)]
        forbidden_ext_hits = _forbidden_extension_hits(files)
        if forbidden_hits or forbidden_ext_hits:
            await ctx.reply(
                "approveをブロックしました。禁止パス/拡張子の変更が残っています。\n"
                f"- forbidden_paths: {', '.join(forbidden_hits[:8]) or 'none'}\n"
                f"- forbidden_ext: {', '.join(forbidden_ext_hits[:8]) or 'none'}"
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
            repair_strategies: list[str] = []
            stagnant_repair_count = 0
            repair_type_counts: dict[str, int] = {}
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
            if _detect_no_tests_in_runlog(impl_log):
                alerts.append("implement_phase_no_tests_detected")
                ok = False
            dod_ok, dod_report = _evaluate_dod(workdir, ok, alerts)
            dod_path.write_text(dod_report, encoding="utf-8")
            progress_hook(f"validation: ok={ok} report={validation_path}")

            attempt = 0
            while not ok and attempt < max_repair_loops:
                attempt += 1
                failure_type = _classify_validation_failure(report)
                repair_strategies.append(failure_type)
                repair_type_counts[failure_type] = repair_type_counts.get(failure_type, 0) + 1
                type_limit = _repair_attempt_limit(failure_type, max_repair_loops)
                if repair_type_counts[failure_type] > type_limit:
                    progress_hook(
                        "repair_stop=type_attempt_limit "
                        f"failure_type={failure_type} limit={type_limit}"
                    )
                    break
                progress_hook(f"phase=repair attempt={attempt}")
                progress_hook(f"repair_strategy={failure_type}")
                fix_objective = _build_repair_objective_with_stagnation_note(
                    objective,
                    failure_type,
                    stagnant_repair_count,
                )
                pre_fp = _worktree_fingerprint(workdir)
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
                post_fp = _worktree_fingerprint(workdir)
                if post_fp == pre_fp:
                    stagnant_repair_count += 1
                    progress_hook(
                        "repair_progress=stagnant "
                        f"(count={stagnant_repair_count}, fp={post_fp})"
                    )
                else:
                    stagnant_repair_count = 0
                latest_log = fix_log
                latest_note = fix_note
                ok, report = _run_validation_suite(workdir)
                validation_path.write_text(report, encoding="utf-8")
                alerts = _extract_validation_alert_lines(report)
                if _detect_no_tests_in_runlog(fix_log):
                    alerts.append("repair_phase_no_tests_detected")
                    ok = False
                dod_ok, dod_report = _evaluate_dod(workdir, ok, alerts)
                dod_path.write_text(dod_report, encoding="utf-8")
                progress_hook(f"validation_retry: ok={ok} attempt={attempt}")
                if not ok and failure_type == "permission_failure":
                    progress_hook("repair_stop=permission_failure requires manual intervention")
                    break
                stagnant_limit = _repair_stagnant_limit(failure_type)
                if not ok and stagnant_repair_count >= stagnant_limit:
                    progress_hook(
                        "repair_stop=stagnant_repair "
                        f"failure_type={failure_type} limit={stagnant_limit}"
                    )
                    break

            summary_path = _write_summary_template(
                workdir=workdir,
                project_root=project_root,
                objective=objective,
                impl_final=impl_final,
                validation_ok=ok,
                validation_report=report,
                alerts=alerts,
            )
            doc_final = (
                "Summary template written with sections: "
                "Objective, Changes, Validation, Validation Alerts, Unresolved, Next Steps."
            )
            progress_hook(f"[documenter] summary generated: {summary_path}")

            implementer_bad = (
                "stopped:" in impl_final.lower()
                and "pytest found no tests" not in impl_final.lower()
                and "pytest success repeated" not in impl_final.lower()
                and "no-op repeated" not in impl_final.lower()
            )
            has_alerts = len(alerts) > 0
            review_report = _review_findings(workdir, project_root)
            review_report_path = project_root / "runs" / "review_report.md"
            review_report_path.write_text(review_report, encoding="utf-8")
            review_ok = _review_status(project_root) == "OK"
            status = (
                "SUCCESS"
                if (ok and dod_ok and review_ok and not implementer_bad and not has_alerts)
                else "NEEDS_REVIEW"
            )
            release_note = _make_release_note(
                workdir=workdir,
                objective=objective,
                deliver_status=status,
                implementer_final=impl_final,
                validation_report_path=validation_path,
                dod_report_path=dod_path,
            )
            release_note_path.write_text(release_note, encoding="utf-8")
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
                f"repair_strategies={','.join(repair_strategies) if repair_strategies else '(none)'}\n"
                f"repair_type_counts={repair_type_counts}\n"
                f"repair_stagnant_count={stagnant_repair_count}\n"
                f"validation_alerts={len(alerts)}\n"
                f"review_status={'OK' if review_ok else 'NOT_OK'}"
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
