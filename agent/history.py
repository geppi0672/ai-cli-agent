from __future__ import annotations

from collections import Counter


class HistoryCompressor:
    def __init__(self, recent_steps: int = 6, max_output_chars: int = 220) -> None:
        self.recent_steps = max(1, recent_steps)
        self.max_output_chars = max(80, max_output_chars)

    def compress(self, history: list[dict]) -> str:
        if not history:
            return "(empty)"
        if len(history) <= self.recent_steps:
            return _render_entries(history, self.max_output_chars)

        older = history[: -self.recent_steps]
        recent = history[-self.recent_steps :]
        return (
            _summarize_older(older)
            + "\n\nRecent Steps:\n"
            + _render_entries(recent, self.max_output_chars)
        )


def _summarize_older(older: list[dict]) -> str:
    total = len(older)
    ok = sum(1 for item in older if item.get("ok"))
    fail = total - ok
    by_tool = Counter(str(item.get("tool", "")) for item in older)
    most_common = ", ".join(f"{tool}:{count}" for tool, count in by_tool.most_common(5))
    fail_examples: list[str] = []
    for item in older:
        if item.get("ok"):
            continue
        output = str(item.get("output", "")).strip().replace("\n", " ")
        fail_examples.append(f"{item.get('tool')}: {output[:120]}")
        if len(fail_examples) >= 3:
            break

    fail_text = "; ".join(fail_examples) if fail_examples else "none"
    return (
        "Older Summary:\n"
        f"- total_steps={total}\n"
        f"- successes={ok}, failures={fail}\n"
        f"- tool_usage={most_common or 'none'}\n"
        f"- failure_examples={fail_text}"
    )


def _render_entries(entries: list[dict], max_output_chars: int) -> str:
    lines: list[str] = []
    for idx, item in enumerate(entries, start=1):
        output = str(item.get("output", "")).strip().replace("\n", " ")
        lines.append(
            f"[{idx}] tool={item.get('tool')} ok={item.get('ok')} "
            f"thought={str(item.get('thought', ''))[:120]} "
            f"output={output[:max_output_chars]}"
        )
    return "\n".join(lines)

