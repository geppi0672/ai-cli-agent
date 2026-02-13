from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


class JsonlMemory:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = output_dir / f"run-{timestamp}.jsonl"

    def append(self, event: dict[str, Any]) -> None:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            **event,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

