#!/usr/bin/env python3
"""Consolidate task metrics and Ped-RM scores into one report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

CONFIGS = (
    "problem_solving",
    "socratic_questioning",
    "student_solution_correctness",
    "mistake_location",
    "mistake_correction",
    "scaffolding_generation",
    "pedagogy_following",
    "scaffolding_generation_hard",
    "pedagogy_following_hard",
)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    tasks: dict[str, Any] = {}
    for config in CONFIGS:
        path = run_dir / "tasks" / config / "metrics.json"
        if not path.is_file():
            raise SystemExit(f"missing task metrics: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        tasks[payload["task_name"]] = payload

    pedrm_file = run_dir / "pedrm" / "pedrm_metrics.json"
    pedrm = (
        json.loads(pedrm_file.read_text(encoding="utf-8"))
        if pedrm_file.is_file()
        else {}
    )
    leaderboard = {
        "problem_solving": tasks["problem_solving"]["metrics"].get(
            "accuracy_flexible-extract"
        ),
        "socratic_questioning": tasks["socratic_questioning"]["metrics"].get("bleu"),
        "solution_correctness": tasks["solution_correctness"]["metrics"].get("f1"),
        "mistake_location": tasks["mistake_location"]["metrics"].get("f1_micro"),
        "mistake_correction": tasks["mistake_correction"]["metrics"].get("accuracy"),
    }
    for task in (
        "scaffolding_generation",
        "pedagogy_following",
        "scaffolding_generation_hard",
        "pedagogy_following_hard",
    ):
        leaderboard[task] = pedrm.get(task, {}).get("win_rate")

    report = {
        "math_tutor_bench_revision": args.upstream_revision,
        "leaderboard": leaderboard,
        "official_task_metrics": tasks,
        "official_pedrm_metrics": pedrm,
    }
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    yaml_text = yaml.safe_dump(report, sort_keys=False, allow_unicode=True)
    atomic_text(run_dir / "summary.json", json_text)
    atomic_text(run_dir / "summary.yaml", yaml_text)
    print(yaml.safe_dump({"leaderboard": leaderboard}, sort_keys=False))


if __name__ == "__main__":
    main()
