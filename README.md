# ai-cli-agent (MVP)

AI that autonomously works through a CLI loop:

1. Plan next action with LLM
2. Execute tool
3. Observe output
4. Repeat until `finish`

## Features

- ReAct style loop (`plan -> act -> observe`)
- Planner Provider abstraction (`openai` now, extensible)
- BudgetManager:
  - step budget
  - external-call budget
  - time budget
- Router:
  - cheap/base tools first
  - escalation to external adapters on failure/stall
- Context compression:
  - older history summarized
  - only recent N steps kept in detail
- Self-improvement:
  - analyze each run log in `runs/*.jsonl`
  - auto-tune next router settings
  - persist profile to `.agent_state/router_profile.json`
  - auto-tune `max_external_calls` too
- Tools:
  - `shell`
  - `read_file`
  - `write_file`
  - `append_file`
  - `git_status`
  - `git_diff`
  - external agent adapters (optional):
    - `codex_agent`
    - `antigravity_agent`
- Basic guardrails:
  - dangerous command block list
  - non-safe shell command approval mode
- JSONL run memory under `runs/`

## Setup

```bash
cd /Users/tanaka/ai-cli-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key
```

## Run

```bash
python -m agent.cli "READMEを作成して、内容を要約して"
```

With explicit workdir:

```bash
python -m agent.cli "テストを実行して失敗原因を調べる" --workdir /path/to/repo
```

Enable interactive approval for non-safe shell commands:

```bash
python -m agent.cli "依存関係を調査して" --interactive-approval
```

Enable external adapters (`codex` / `antigravity`):

```bash
python -m agent.cli "外部エージェントに委譲して実装方針を作る" \
  --enable-external-agents \
  --interactive-approval
```

Optional binary override:

```bash
export CODEX_CLI_BIN=codex
export ANTIGRAVITY_CLI_BIN=antigravity
```

Tune budget / routing / compression:

```bash
python -m agent.cli "大きめの実装タスクを進める" \
  --enable-external-agents \
  --interactive-approval \
  --max-steps 30 \
  --max-external-calls 8 \
  --max-seconds 1800 \
  --escalate-after-failures 1 \
  --compress-recent-steps 8
```

Show generated summary right after the run:

```bash
python -m agent.cli "このディレクトリのファイル一覧を確認して要約して" \
  --workdir /Users/tanaka/ai-cli-agent \
  --show-file summary.txt
```

## Discord Control

Use Discord to run and monitor tasks remotely.

1. Prepare env variables in `.env`:

```dotenv
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_ALLOWED_CHANNEL_IDS=123456789012345678,234567890123456789
DISCORD_ENABLE_EXTERNAL_AGENTS=true
DISCORD_MAX_STEPS=30
DISCORD_MAX_EXTERNAL_CALLS=8
DISCORD_MAX_SECONDS=3600
DISCORD_ESCALATE_AFTER_FAILURES=1
DISCORD_COMPRESS_RECENT_STEPS=8
DISCORD_ALLOW_DANGEROUS_COMMANDS=false
```

2. Start bot:

```bash
python -m agent.discord_bot --workdir /Users/tanaka/ai-cli-agent
```

3. Discord commands:

- `!agent <objective>` : run autonomous task
- `!supervise <objective>` : supervisor mode (planner/implementer/tester/documenter)
- `!deliver <objective>` : implement -> validate -> auto-repair loop -> document
  - implement/repair フェーズは既定で `max_steps=12`
  - `no-op` が3連続で出たらそのフェーズを早期終了
  - 許可ツールは `read_file/write_file/shell` に制限
  - `shell` は `pytest/compileall` 系のみ許可
  - `runs/validation_report.md` から失敗/警告行を抽出し、`runs/summary.md` に転記
  - 成功条件は `validation all green` かつ `validation alerts=0` かつ `implementer重大停止なし` かつ `DoD=PASS`
  - `runs/dod_report.md` を生成（変更ファイル数上限・禁止パス変更を検査）
  - `runs/release_note.md` を固定フォーマットで生成（PR本文利用向け）
  - `runs/review_report.md` を自動更新し、`runs/pr_ready.md` を生成（提出用バンドル）
- `!autopr <objective>` : branch + implementation/test + PR artifact generation
- `!review` : current diff を `Critical/High/Medium` で自動査読（`runs/review_report.md` 出力）
- `!status` : check running/latest status
- `!cancel` : request stop on next step boundary
- `!runs [count]` : list latest run logs
- `!tail [lines]` : show tail of latest run log for this channel
- `!diff` : show current `git diff` preview
- `!approve <commit message>` : `git add -A` + `git commit`
  - 既定で `DoD=PASS` かつ `review=OK` のときのみ実行（未達はブロック）
- `!rollback [ref]` : safe rollback via `git revert --no-edit <ref>`
- `supervise` の tester はプロジェクト種別を自動判定し、Pythonプロジェクトでは `pytest/compileall` 系のみ許可

4. Optional Discord runtime env:

```dotenv
DISCORD_APPROVAL_POLICY=allow
DISCORD_MAX_REPAIR_LOOPS=3
DISCORD_MAX_CHANGED_FILES=25
DISCORD_FORBIDDEN_PATH_PREFIXES=.env,.venv/,agent/__pycache__/,__pycache__/
DISCORD_ENFORCE_APPROVE_GATES=true
```

- `allow`: non-safe shell/external calls are allowed automatically
- `prompt`: requires terminal confirmation
- `deny`: blocks non-safe operations
- `DISCORD_ALLOW_DANGEROUS_COMMANDS=true` にすると危険コマンドブロックを解除（推奨は `false`）
- `DISCORD_MAX_REPAIR_LOOPS` は `!deliver` の自動修正リトライ回数
- `DISCORD_MAX_CHANGED_FILES` は DoD の変更ファイル数上限
- `DISCORD_FORBIDDEN_PATH_PREFIXES` は DoD で禁止する変更パス接頭辞（`,` 区切り）
- `DISCORD_ENFORCE_APPROVE_GATES` は `!approve` の DoD/Review ゲート強制（既定 `true`）

Additional runtime safeguards:

- planner JSON invalid output: retry up to 3 times, then safe finish
- `pytest` で `collected 0 items` / `no tests ran` は早期終了として扱う
- ただし `ERROR:` や `file or directory not found` を含む場合は no-tests 早期終了しない
- `pytest` 成功出力の同一反復を検知したら早期終了（無限再実行防止）
- plannerが空/不正なtool名を連続返却したら安全停止
- 空の `shell` コマンドは no-op として再計画扱い

## Notes

- Default model is `gpt-4o-mini` (override by `--model` or `OPENAI_MODEL`).
- Default planner provider is `openai` (`--planner-provider openai`).
- Default budgets:
  - `max_steps=20`
  - `max_external_calls=4`
  - `max_seconds=900`
- Default routing/compression:
  - `escalate_after_failures=1`
  - `compress_recent_steps=6`
- Auto-tuning behavior:
  - if failure streak grows, escalate earlier and keep more recent context
  - if external-call budget is repeatedly saturated under failures, increase `max_external_calls`
  - if router blocks repeatedly, force early escalation
  - if run is stable, relax escalation/reduce context and lower `max_external_calls`
- `.env` is auto-loaded from:
  - `ai-cli-agent/.env`
  - current directory `.env`
  - `--workdir` directory `.env`
  (existing shell env vars take priority)
- This is an MVP and does not yet include:
  - dynamic task graph planner
  - long-term vector memory
  - sandboxed subprocess isolation
  - browser automation
