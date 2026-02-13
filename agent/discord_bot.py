from __future__ import annotations

import argparse
import asyncio
import os
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


def _is_git_repo(workdir: Path) -> bool:
    code, _ = _run_cmd("git rev-parse --is-inside-work-tree", workdir)
    return code == 0


def _latest_logs(runs_dir: Path, limit: int = 5) -> list[Path]:
    files = sorted(runs_dir.glob("run-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[: max(1, limit)]


def _tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return f"file not found: {path}"
    content = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(content[-max(1, lines) :])[:3500]


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
) -> tuple[str, str, str]:
    config, external_adapters, profile = _build_runtime_config(project_root, workdir)
    planner = OpenAIProvider(model=config.model)
    approval_policy = os.getenv("DISCORD_APPROVAL_POLICY", "allow").strip().lower() or "allow"
    tools = ToolRunner(
        workdir=config.workdir,
        auto_approve_safe=False,
        external_adapters=external_adapters,
        approval_policy=approval_policy,
        allow_dangerous_commands=_env_bool("DISCORD_ALLOW_DANGEROUS_COMMANDS", False),
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
            "Commands: !agent !supervise !autopr !status !cancel !runs !tail !diff !approve !rollback",
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

    @bot.command(name="approve")
    async def approve_cmd(ctx: commands.Context, *, message: str) -> None:
        if not is_allowed_channel(ctx.channel.id):
            return
        if not _is_git_repo(workdir):
            await ctx.reply("このworkdirはGitリポジトリではありません。`git init` 後に再実行してください。")
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

    bot.run(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
