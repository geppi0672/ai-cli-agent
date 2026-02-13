from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
from .tools import ToolRunner


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


def _load_dotenv_files(paths: list[Path]) -> list[Path]:
    loaded: list[Path] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_dotenv_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)
        loaded.append(path)
    return loaded


def _resolve_workdir(raw_workdir: str | None) -> Path:
    return Path(raw_workdir).expanduser().resolve() if raw_workdir else Path.cwd()


def _show_file_preview(workdir: Path, file_path: str) -> None:
    path = (workdir / file_path).resolve()
    if not str(path).startswith(str(workdir.resolve())):
        print(f"[show-file] blocked path outside workdir: {file_path}", file=sys.stderr)
        return
    if not path.exists() or not path.is_file():
        print(f"[show-file] file not found: {file_path}", file=sys.stderr)
        return
    content = path.read_text(encoding="utf-8")
    print("-" * 60)
    print(f"File preview: {file_path}")
    print(content[:4000])
    if len(content) > 4000:
        print("\n...(truncated)...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous CLI agent (MVP).")
    parser.add_argument("objective", help="Goal for the agent.")
    parser.add_argument("--workdir", default=None, help="Working directory for tools.")
    parser.add_argument("--model", default=None, help="OpenAI model name.")
    parser.add_argument(
        "--planner-provider",
        default="openai",
        choices=["openai"],
        help="Planner provider implementation.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum loop steps.")
    parser.add_argument(
        "--max-external-calls",
        type=int,
        default=None,
        help="Maximum number of successful external agent calls.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Maximum run time in seconds.",
    )
    parser.add_argument(
        "--compress-recent-steps",
        type=int,
        default=None,
        help="How many recent steps to keep verbatim in context.",
    )
    parser.add_argument(
        "--escalate-after-failures",
        type=int,
        default=None,
        help="Router escalation threshold for consecutive failures.",
    )
    parser.add_argument(
        "--show-file",
        default=None,
        help="Show file content after run (path relative to workdir).",
    )
    parser.add_argument(
        "--interactive-approval",
        action="store_true",
        help="Ask before non-safe shell commands. By default those commands are blocked.",
    )
    parser.add_argument(
        "--enable-external-agents",
        action="store_true",
        help="Enable codex/antigravity adapter tools.",
    )
    return parser.parse_args()


def _build_external_adapters(enable: bool) -> dict[str, ExternalAgentAdapter]:
    if not enable:
        return {}
    codex = CodexAdapter()
    antigravity = AntigravityAdapter()
    return {codex.name: codex, antigravity.name: antigravity}


def _profile_path(project_root: Path) -> Path:
    return project_root / ".agent_state" / "router_profile.json"


def _resolve_router_profile(
    project_root: Path,
    arg_max_external_calls: int | None,
    arg_escalate_after_failures: int | None,
    arg_compress_recent_steps: int | None,
) -> tuple[int | None, int | None, int | None, RouterProfile | None]:
    profile = load_router_profile(_profile_path(project_root))
    max_external_calls = arg_max_external_calls
    escalate = arg_escalate_after_failures
    compress_recent = arg_compress_recent_steps
    if profile:
        if max_external_calls is None:
            max_external_calls = profile.max_external_calls
        if escalate is None:
            escalate = profile.escalate_after_failures
        if compress_recent is None:
            compress_recent = profile.compress_recent_steps
    return max_external_calls, escalate, compress_recent, profile


def main() -> int:
    args = parse_args()
    workdir = _resolve_workdir(args.workdir)

    project_root = Path(__file__).resolve().parents[1]
    dotenv_candidates = [project_root / ".env", Path.cwd() / ".env", workdir / ".env"]
    loaded_envs = _load_dotenv_files(dotenv_candidates)
    tuned_max_external_calls, tuned_escalate, tuned_compress_recent, loaded_profile = _resolve_router_profile(
        project_root=project_root,
        arg_max_external_calls=args.max_external_calls,
        arg_escalate_after_failures=args.escalate_after_failures,
        arg_compress_recent_steps=args.compress_recent_steps,
    )

    config = AgentConfig.from_args(
        model=args.model,
        max_steps=args.max_steps,
        max_external_calls=tuned_max_external_calls,
        max_seconds=args.max_seconds,
        compress_recent_steps=tuned_compress_recent,
        escalate_after_failures=tuned_escalate,
        workdir=str(workdir),
        auto_approve_safe=not args.interactive_approval,
    )

    if not config.workdir.exists():
        print(f"workdir does not exist: {config.workdir}", file=sys.stderr)
        return 1

    runs_dir = Path(__file__).resolve().parents[1] / "runs"
    if args.planner_provider != "openai":
        print(f"Unsupported planner provider: {args.planner_provider}", file=sys.stderr)
        return 1

    planner = OpenAIProvider(model=config.model)
    external_adapters = _build_external_adapters(args.enable_external_agents)
    tools = ToolRunner(
        workdir=config.workdir,
        auto_approve_safe=config.auto_approve_safe,
        external_adapters=external_adapters,
    )
    memory = JsonlMemory(output_dir=runs_dir)
    budget = BudgetManager(
        max_steps=config.max_steps,
        max_external_calls=config.max_external_calls,
        max_seconds=config.max_seconds,
    )
    router = Router(escalate_after_failures=config.escalate_after_failures)
    compressor = HistoryCompressor(recent_steps=config.compress_recent_steps)
    runner = AgentRunner(
        objective=args.objective,
        planner=planner,
        tools=tools,
        memory=memory,
        budget=budget,
        router=router,
        compressor=compressor,
    )

    print(f"Objective: {args.objective}")
    print(f"Workdir:   {config.workdir}")
    print(f"Model:     {config.model}")
    print(f"Planner:   {args.planner_provider}")
    print(
        "Budget:    "
        f"steps={config.max_steps}, external_calls={config.max_external_calls}, seconds={config.max_seconds}"
    )
    print(
        "Routing:   "
        f"escalate_after_failures={config.escalate_after_failures}, "
        f"compress_recent_steps={config.compress_recent_steps}"
    )
    print(f"Run log:   {memory.path}")
    if external_adapters:
        print(f"Adapters:  {', '.join(sorted(external_adapters.keys()))}")
    if loaded_envs:
        pretty = ", ".join(str(p) for p in loaded_envs)
        print(f"Loaded .env: {pretty}")
    if loaded_profile:
        print(
            "Loaded router profile: "
            f"max_external_calls={loaded_profile.max_external_calls}, "
            f"escalate_after_failures={loaded_profile.escalate_after_failures}, "
            f"compress_recent_steps={loaded_profile.compress_recent_steps}"
        )
    print("-" * 60)
    final = runner.run()
    print("-" * 60)
    print(f"Final: {final}")

    signals = analyze_run_log(memory.path)
    base_profile = loaded_profile or RouterProfile(
        max_external_calls=config.max_external_calls,
        escalate_after_failures=config.escalate_after_failures,
        compress_recent_steps=config.compress_recent_steps,
    )
    tuned_profile = tune_router_profile(base_profile, signals)
    profile_file = _profile_path(project_root)
    save_router_profile(profile_file, tuned_profile)
    print(
        "Next router profile: "
        f"max_external_calls={tuned_profile.max_external_calls}, "
        f"escalate_after_failures={tuned_profile.escalate_after_failures}, "
        f"compress_recent_steps={tuned_profile.compress_recent_steps}"
    )
    print(f"Profile saved: {profile_file}")
    print(f"Tuning note: {tuned_profile.notes}")

    if args.show_file:
        _show_file_preview(config.workdir, args.show_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
