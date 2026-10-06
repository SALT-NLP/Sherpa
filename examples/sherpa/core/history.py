from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .types import GuidanceGateResult, TurnTrace


def trace_to_history_record(
    trace: TurnTrace,
    guidance_gate_result: GuidanceGateResult | None,
    student_error: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "round_idx": trace.turn_idx,
        "teacher_raw_output": trace.tutor_raw_output,
        "teacher_action": trace.tutor_visible_output,
        "student_answer": trace.student_output,
        "student_error": student_error,
        "reward": trace.reward,
        "reward_components": dict(trace.reward_components),
        "guidance_gate_failed": trace.guidance_gate_failed,
        "guidance_gate_masked": trace.guidance_gate_masked,
        "teacher_format_error": trace.tutor_format_error,
        "teacher_ended": trace.teacher_ended,
        "teacher_exact_repeat": trace.teacher_exact_repeat,
        "public_history_before": trace.public_history_before,
        "public_history_after": trace.public_history_after,
    }
    if guidance_gate_result is not None:
        record["guidance_gate_feedback"] = guidance_gate_result.feedback
    return record


def trace_to_json(trace: TurnTrace) -> dict[str, Any]:
    return asdict(trace)
