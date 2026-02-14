from __future__ import annotations

import subprocess
from pathlib import Path

from .adapters import ExternalAgentAdapter, ExternalAgentTask
from .models import StepResult


DANGEROUS_TOKENS = {
    "rm -rf",
    "shutdown",
    "reboot",
    "mkfs",
    "dd ",
    ":(){",
    "curl ",
    "wget ",
    "scp ",
    "nc ",
}

SAFE_SHELL_PREFIXES = (
    "ls",
    "pwd",
    "cat",
    "echo",
    "head",
    "tail",
    "wc",
    "rg",
    "find",
    "git status",
    "git diff",
    "git log",
    "pytest",
    "python -m pytest",
)


def _is_dangerous(command: str) -> bool:
    normalized = command.strip().lower()
    return any(token in normalized for token in DANGEROUS_TOKENS)


def _is_safe_prefix(command: str) -> bool:
    normalized = command.strip().lower()
    return any(normalized.startswith(prefix) for prefix in SAFE_SHELL_PREFIXES)


def _resolve_path(workdir: Path, rel_path: str) -> Path:
    candidate = (workdir / rel_path).resolve()
    if not str(candidate).startswith(str(workdir.resolve())):
        raise ValueError("Path escapes workdir and is blocked.")
    return candidate


class ToolRunner:
    def __init__(
        self,
        workdir: Path,
        auto_approve_safe: bool = True,
        external_adapters: dict[str, ExternalAgentAdapter] | None = None,
        approval_policy: str | None = None,
        allow_dangerous_commands: bool = False,
        strict_shell_allowlist: bool = False,
        extra_safe_shell_prefixes: tuple[str, ...] | None = None,
        shell_allow_prefixes: tuple[str, ...] | None = None,
    ) -> None:
        self.workdir = workdir
        self.auto_approve_safe = auto_approve_safe
        self.external_adapters = external_adapters or {}
        self.allow_dangerous_commands = allow_dangerous_commands
        self.strict_shell_allowlist = strict_shell_allowlist
        self.extra_safe_shell_prefixes = extra_safe_shell_prefixes or ()
        self.shell_allow_prefixes = shell_allow_prefixes
        if approval_policy is not None:
            self.approval_policy = approval_policy
        else:
            self.approval_policy = "deny" if auto_approve_safe else "prompt"

    def available_tools(self) -> list[str]:
        tools = [
            "shell",
            "read_file",
            "write_file",
            "append_file",
            "git_status",
            "git_diff",
            "finish",
        ]
        tools.extend(sorted(self.external_adapters.keys()))
        return tools

    def external_tool_names(self) -> list[str]:
        return sorted(self.external_adapters.keys())

    def is_external_tool(self, tool: str) -> bool:
        return tool in self.external_adapters

    def run(self, tool: str, args: dict) -> StepResult:
        try:
            if tool == "shell":
                return self._shell(str(args.get("command", "")))
            if tool == "read_file":
                return self._read_file(str(args.get("path", "")))
            if tool == "write_file":
                return self._write_file(str(args.get("path", "")), str(args.get("content", "")))
            if tool == "append_file":
                return self._append_file(str(args.get("path", "")), str(args.get("content", "")))
            if tool == "git_status":
                return self._shell("git status --short")
            if tool == "git_diff":
                return self._shell("git diff --")
            if tool in self.external_adapters:
                return self._run_external(tool, args)
            return StepResult(False, f"Unknown tool: {tool}")
        except Exception as exc:
            return StepResult(False, f"{type(exc).__name__}: {exc}")

    def _shell(self, command: str) -> StepResult:
        if not command.strip():
            return StepResult(True, "No-op: empty shell command; replanning.")
        if _is_dangerous(command) and not self.allow_dangerous_commands:
            return StepResult(False, f"Blocked dangerous command: {command}")

        safe_prefixes = (
            self.shell_allow_prefixes
            if self.shell_allow_prefixes is not None
            else SAFE_SHELL_PREFIXES + tuple(self.extra_safe_shell_prefixes)
        )
        normalized = command.strip().lower()
        is_safe = any(normalized.startswith(prefix) for prefix in safe_prefixes)
        if self.strict_shell_allowlist and not is_safe:
            return StepResult(False, f"Blocked by strict shell allowlist: {command}")

        needs_approval = not is_safe
        if needs_approval and not self._approved(command):
            return StepResult(False, "User rejected shell command.")

        completed = subprocess.run(
            command,
            shell=True,
            cwd=self.workdir,
            text=True,
            capture_output=True,
        )
        combined = (completed.stdout + completed.stderr).strip()
        output = combined[:4000] if combined else "(no output)"
        return StepResult(completed.returncode == 0, output)

    def _confirm(self, command: str) -> bool:
        answer = input(f"[approval required] Run shell command? {command} [y/N]: ").strip().lower()
        return answer == "y"

    def _approved(self, command: str) -> bool:
        if self.approval_policy == "allow":
            return True
        if self.approval_policy == "prompt":
            return self._confirm(command)
        return False

    def _read_file(self, rel_path: str) -> StepResult:
        path = _resolve_path(self.workdir, rel_path)
        if not path.exists():
            return StepResult(False, f"File not found: {rel_path}")
        if path.is_dir():
            entries = sorted(p.name for p in path.iterdir())
            preview = "\n".join(entries[:200])
            return StepResult(True, f"Path is a directory. Listing:\n{preview}")
        text = path.read_text(encoding="utf-8")
        return StepResult(True, text[:6000])

    def _write_file(self, rel_path: str, content: str) -> StepResult:
        path = _resolve_path(self.workdir, rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return StepResult(True, f"Wrote {rel_path} ({len(content)} chars).")

    def _append_file(self, rel_path: str, content: str) -> StepResult:
        path = _resolve_path(self.workdir, rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(content)
        return StepResult(True, f"Appended {len(content)} chars to {rel_path}.")

    def _run_external(self, tool: str, args: dict) -> StepResult:
        instruction = str(args.get("instruction", "")).strip()
        if not instruction:
            return StepResult(False, f"{tool} requires args.instruction.")

        if not self._approved(f"{tool}: {instruction[:200]}"):
            return StepResult(False, f"User rejected external agent call: {tool}")

        adapter = self.external_adapters[tool]
        task = ExternalAgentTask(instruction=instruction, workdir=self.workdir)
        return adapter.run(task)
