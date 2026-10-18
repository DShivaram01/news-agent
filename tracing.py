"""Every run writes a JSONL trace: one line per model reply, tool call and stop.

When the agent does something weird, this file is how you find out why.
(Named tracing.py because `trace` is already a module in Python's stdlib.)
"""
import json
from datetime import datetime
from pathlib import Path


class Tracer:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.path = directory / f"run-{stamp}.jsonl"

    def log(self, kind: str, data: dict) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "kind": kind, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
