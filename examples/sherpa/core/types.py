from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GuidanceGateMode = Literal["disabled", "masked_continue"]


@dataclass(slots=True)
class JudgeResult:
    raw_output: str
    correct: bool
    feedback: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class GuidanceGateResult:
    raw_output: str
    failed: bool
    feedback: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class AdaptiveGateResult:
    """One adaptive gate check on one teacher message.

    `passed` is what routes the turn: True calls the student, False replaces its
    reply with a complaint. `reason` keeps the judge's analysis. `error` is set only
    when every retry came back unclean, in which case `passed` is False -- the
    conservative default, so a broken check never lets through a message that may
    violate the preference.
    """

    raw_output: str
    passed: bool
    reason: str
    error: str | None = None
    attempts: int = 1


@dataclass(slots=True)
class PublicHistoryState:
    summary: str = ""
    turn_count: int = 0
    # The dialogue as real messages: [{"role": "teacher"|"student", "content": ...}].
    # `summary` is kept only for debug traces and logging; prompts are built from
    # `turns` so the model sees an ordinary multi-turn chat.
    turns: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class TeacherPreSolveAttempt:
    attempt: int
    raw_output: str
    error: str | None
    accepted: bool
    judge_result: JudgeResult | None = None


@dataclass(slots=True)
class TeacherPreSolveResult:
    accepted: bool
    attempts: list[TeacherPreSolveAttempt] = field(default_factory=list)
    raw_output: str = ""
    error: str | None = None
    verification_enabled: bool = True


@dataclass(slots=True)
class TutorTurnState:
    task: str
    ground_truth: str
    public_history: PublicHistoryState
    previous_tutor_visible_output: str
    turn_idx: int
    max_turns: int
    teacher_pre_solve_result: TeacherPreSolveResult | None = None
    student_reply_before_teacher: str = ""
    # Exact prior teacher replies, parallel to the teacher turns in
    # ``public_history``. Keeping this outside PublicHistoryState prevents the
    # student's conversation and re-test views from ever receiving private
    # reasoning. A tuple snapshots the history stored on each training row.
    previous_tutor_raw_outputs: tuple[str, ...] = ()


@dataclass(slots=True)
class StudentTurnState:
    task: str
    public_history: PublicHistoryState
    previous_student_output: str
    latest_tutor_visible_output: str
    # What this student demands of the teacher's manner, or "" for the open gate.
    # Stored per episode because one workflow serves many concurrent episodes.
    student_preference: str = ""


@dataclass(slots=True)
class TurnArtifact:
    turn_idx: int
    tutor_state: TutorTurnState
    # The exact messages sent to the teacher this turn. Training tokens are
    # re-rendered from these, so they must be what generation actually saw.
    tutor_messages: list[dict[str, str]]
    tutor_response: Any
    tutor_raw_output: str
    tutor_visible_output: str
    guidance_gate_result: GuidanceGateResult
    public_history_before: list[dict[str, str]]
    public_history_after: list[dict[str, str]]
    tutor_format_error: str | None = None
    # A valid policy action that ends the dialogue before another student call.
    # It has no student-visible text, but remains a trainable teacher turn so the
    # terminal outcome can teach the policy when to stop.
    teacher_ended: bool = False
    student_state: StudentTurnState | None = None
    student_output: str = ""
    student_error: str | None = None
    # In masked_continue mode a teacher turn that fails the guidance gate is kept
    # only in the teacher's history. The real student is not called;
    # student_output is the injected user reply, and student/re-test history stays
    # on its prior branch.
    guidance_gate_masked: bool = False
    # None when this student has no preference, or when the turn ended on a format
    # error or a guidance gate failure before the adaptive gate could run.
    adaptive_gate_result: AdaptiveGateResult | None = None
    # True when the gate closed and the student never answered: `student_output` holds
    # the injected complaint instead of a student reply. Such a turn is not a student
    # attempt but still consumes one turn of the budget.
    adaptive_gate_failed: bool = False
    # True when this student-visible reply exactly matches any earlier
    # student-visible teacher reply after whitespace normalization. A terminating
    # repeat is trained but is not appended to either the student conversation or
    # the re-test context.
    teacher_exact_repeat: bool = False


@dataclass(slots=True)
class EpisodeArtifact:
    task: str
    ground_truth: str
    turns: list[TurnArtifact]
    termination_reason: str
    guidance_gate_fail_count: int
    latest_student_answer: str
    teacher_pre_solve_result: TeacherPreSolveResult | None = None
    student_name: str = ""
    student_model: str = ""


@dataclass(slots=True)
class RewardAssignment:
    reward: float
    reward_components: dict[str, float]
    # The part of reward that belongs to the turn that produced it and must
    # not be accumulated backward onto earlier turns by ReBN. The actor adds it
    # back after advantage normalization.
    local_reward: float = 0.0


@dataclass(slots=True)
class TurnTrace:
    turn_idx: int
    tutor_state: TutorTurnState
    tutor_raw_output: str
    tutor_visible_output: str
    guidance_gate_failed: bool
    student_output: str
    reward: float
    reward_components: dict[str, float]
    public_history_before: list[dict[str, str]]
    public_history_after: list[dict[str, str]]
    guidance_gate_masked: bool = False
    tutor_format_error: str | None = None
    teacher_ended: bool = False
    # None when this student has no preference, or when the turn ended on a format
    # error or a guidance gate failure before the adaptive gate could run.
    adaptive_gate_result: AdaptiveGateResult | None = None
    # True when the gate closed and the student never answered: `student_output` holds
    # the injected complaint instead of a student reply. Such a turn is not a student
    # attempt but still consumes one turn of the budget.
    adaptive_gate_failed: bool = False
    teacher_exact_repeat: bool = False
