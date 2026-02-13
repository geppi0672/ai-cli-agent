from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class WorkerTask:
    name: str
    objective: str


@dataclass
class WorkerResult:
    name: str
    final: str
    run_log: str
    profile_note: str


class Supervisor:
    def build_default_plan(self, objective: str) -> list[WorkerTask]:
        return [
            WorkerTask(
                name="planner",
                objective=(
                    "以下の目標に対して実装計画を作成し、必要なら作業メモを `runs/plan.md` に出力してから終了する: "
                    f"{objective}"
                ),
            ),
            WorkerTask(
                name="implementer",
                objective=(
                    "以下を実装して必要なファイル変更を行う。途中経過を短く残し、完了時にfinishする: "
                    f"{objective}"
                ),
            ),
            WorkerTask(
                name="tester",
                objective=(
                    "変更後の検証を行う。可能ならテストや静的チェックを実行し、失敗時は原因を記録して終了する。"
                ),
            ),
            WorkerTask(
                name="documenter",
                objective=(
                    "最終的な変更点・検証結果・未解決事項を `runs/summary.md` に要約して終了する。"
                ),
            ),
        ]

    def run(
        self,
        objective: str,
        run_worker: Callable[[str, str], tuple[str, str, str]],
    ) -> list[WorkerResult]:
        results: list[WorkerResult] = []
        for task in self.build_default_plan(objective):
            final, run_log, profile_note = run_worker(task.name, task.objective)
            results.append(
                WorkerResult(
                    name=task.name,
                    final=final,
                    run_log=run_log,
                    profile_note=profile_note,
                )
            )
        return results

