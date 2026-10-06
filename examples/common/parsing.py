from __future__ import annotations

import json
from typing import Any


def extract_json_object(raw_output: str) -> str | None:
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw_output[start : end + 1]


def parse_json_dict(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    raw_output = (raw_output or "").strip()
    if not raw_output:
        return None, "Empty output"
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        candidate = extract_json_object(raw_output)
        if candidate is None:
            return None, "Invalid JSON: no JSON object found"
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "Output is not a JSON object"
    return parsed, None


def join_errors(*errors: str | None) -> str | None:
    values = [error for error in errors if error]
    if not values:
        return None
    return " | ".join(values)
