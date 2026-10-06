from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
            rows.append(item)
    return rows


def normalize_math_id(value: Any) -> str:
    return str(value)


def math_item_to_tutor_row(
    item: dict[str, Any],
    *,
    fallback_id: str,
) -> dict[str, Any]:
    metadata = dict(item.get("meta") or item.get("metadata") or {})
    if item.get("solution") is not None:
        metadata["solution"] = str(item["solution"])
    metadata.setdefault("source", "math")
    return {
        "id": normalize_math_id(item.get("id", fallback_id)),
        "task": str(item["problem"]),
        "ground_truth": str(item["answer"]),
        "metadata": metadata,
    }


def load_math_rows(path: Path) -> list[dict[str, Any]]:
    return [
        math_item_to_tutor_row(item, fallback_id=f"{path.stem}-{idx}")
        for idx, item in enumerate(load_jsonl(path))
    ]


def build_math_splits(
    *,
    train_jsonl_path: Path,
    test_jsonl_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return load_math_rows(train_jsonl_path), load_math_rows(test_jsonl_path)
