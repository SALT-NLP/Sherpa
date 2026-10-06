"""Merge disjoint shards and report preference metrics from latest raw records."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def gate_counts(rows: list[dict], preference: str) -> tuple[int, int]:
    """Exclude turn 1 for preferences whose first turn is an automatic pass."""
    passed = sampled = 0
    for row in rows:
        gate = row.get("adaptive_gate", {})
        n = gate.get("sampled_turn_count", 0)
        k = gate.get("passed_turn_count", 0)
        if preference in {"attempt-diagnosis", "independent-verification"} and n:
            if "turn1_sampled" not in gate or "turn1_passed" not in gate:
                raise ValueError(
                    "Missing turn-1 gate metadata; cannot exclude the first turn"
                )
            first_sampled = bool(gate["turn1_sampled"])
            n -= int(first_sampled)
            k -= int(first_sampled and bool(gate["turn1_passed"]))
        if not 0 <= k <= n:
            raise ValueError("Inconsistent gate counts")
        passed += k
        sampled += n
    return passed, sampled


def guidance_gate_fail_counts(rows: list[dict]) -> tuple[int, int]:
    """Guidance gate failures over all recorded teacher turns, including end actions."""
    failed = turns = 0
    for row in rows:
        k, n = row["guidance_gate_fail_count"], row["num_turns"]
        if not 0 <= k <= n:
            raise ValueError("Inconsistent guidance gate counts")
        failed += k
        turns += n
    return failed, turns


def summarize(directories: list[Path]) -> dict:
    manifests = [json.loads((p / "experiment.json").read_text()) for p in directories]
    first = manifests[0]
    count = first["shard_count"]
    if len(directories) != count or {m["shard_index"] for m in manifests} != set(
        range(count)
    ):
        raise ValueError("Supply each shard exactly once")
    for manifest in manifests[1:]:
        for key in (
            "experiment",
            "protocol_sha256",
            "source_sha256",
            "limit",
            "shard_count",
            "adapter_files_sha256",
        ):
            if manifest.get(key) != first.get(key):
                raise ValueError(f"Shards disagree on {key}")
    records = {}
    for directory in directories:
        completion = json.loads((directory / "completion.json").read_text())
        if not completion.get("complete"):
            raise ValueError(f"Incomplete shard: {directory}")
        latest = {}
        with (directory / "evaluation/results.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                latest[row["key"]] = row
        if records.keys() & latest.keys():
            raise ValueError("Shards overlap; refusing to double-count episodes")
        records.update(latest)
    evaluation = first["experiment"]["evaluation"]
    questions = (
        min(first["limit"], evaluation["expected_questions"])
        if first["limit"]
        else evaluation["expected_questions"]
    )
    expected = questions * evaluation["attempts"]
    cells = {}
    for preference in evaluation["preferences"]:
        rows = [
            r
            for r in records.values()
            if r.get("adaptive_gate", {}).get("preference", "none") == preference
        ]
        if len(rows) != expected:
            raise ValueError(f"{preference}: expected {expected}, found {len(rows)}")
        for row in rows:
            failed = any(
                row.get(k)
                for k in (
                    "error",
                    "student_call_failed",
                    "answer_judge_failed_count",
                    "guidance_gate_error_count",
                    "teacher_pre_error_count",
                )
            )
            # Gate errors are sampled FAILs, not missing episode results.
            # gate_counts keeps them in the denominator, never in passed turns.
            replay = row.get("generalization", {}).get("original", {})
            if (
                failed
                or replay.get("replay_count") != 8
                or replay.get("score") is None
                or row.get("no_teaching_baseline") is None
            ):
                raise ValueError(f"Unresolved diagnostic/retest result: {row['key']}")
        passed, sampled = gate_counts(rows, preference)
        failed, turns = guidance_gate_fail_counts(rows)
        cells[preference] = {
            "episodes": len(rows),
            "split": "ID" if preference in evaluation["id_preferences"] else "OOD",
            "improvement_pp": 100
            * statistics.mean(
                r["generalization"]["original"]["score"] - r["no_teaching_baseline"]
                for r in rows
            ),
            "retest_percent": 100
            * statistics.mean(r["generalization"]["original"]["score"] for r in rows),
            "gate_pass_percent": 100 * passed / sampled if sampled else None,
            "gate_passed_turn_count": passed,
            "gate_sampled_turn_count": sampled,
            "gate_first_turn_included": preference
            not in {"attempt-diagnosis", "independent-verification"},
            "guidance_gate_fail_percent": 100 * failed / turns if turns else None,
            "guidance_gate_fail_turn_count": failed,
            "teacher_turn_count": turns,
        }
    aggregates = {}
    for split in ("ID", "OOD", "Overall"):
        selected = [
            cell
            for cell in cells.values()
            if split == "Overall" or cell["split"] == split
        ]
        aggregates[split] = {
            key: statistics.mean(c[key] for c in selected) if selected else None
            for key in ("improvement_pp", "retest_percent")
        }
    return {
        "complete": True,
        "checkpoint": first["experiment"]["teacher"]["checkpoint"],
        "grouping_note": "ID/OOD are configured comparison groups, not inferred training membership",
        "cells": cells,
        "aggregates": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directories)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
