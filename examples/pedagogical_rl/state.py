from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from examples.pedagogical_rl.prompts import (
    INITIAL_ATTEMPT_WRAPPER,
    SIMPLE_STUDENT_PROMPT,
    STUDENT_FINAL_PROMPT,
    STUDENT_INITIAL_ATTEMPT_PROMPT,
    render,
    render_teacher_prompt,
)
from examples.sherpa.core.parsers import parse_tagged_teacher_action


class ConversationType(str, Enum):
    GUIDED = "GUIDED"
    ATTEMPTED = "ATTEMPTED"


STUDENT_NAMES: tuple[str | None, ...] = (
    "Alex",
    "Jamie",
    "Taylor",
    "Jordan",
    "Sam",
    "Casey",
    "Morgan",
    "Riley",
    None,
)


def student_visible_text(content: str) -> str:
    """Return the public part of a tagged teacher action."""

    visible, ended, error = parse_tagged_teacher_action(
        content,
        allow_end=True,
        require_nonempty_output=True,
    )
    if error or ended:
        return ""
    return visible or ""


@dataclass(slots=True)
class NativeJudgeDecision:
    rule: str
    reasoning: str
    decision: str

    @property
    def rejected(self) -> bool:
        return self.decision == "REJECT"


@dataclass(slots=True)
class ClassroomEpisode:
    problem: str
    answer: str
    conversation: list[dict[str, Any]] = field(default_factory=list)
    native_judges: list[NativeJudgeDecision] = field(default_factory=list)
    final_solutions: list[str] = field(default_factory=list)
    initial_attempt: str | None = None
    termination_reason: str | None = None
    format_errors: list[str] = field(default_factory=list)
    teacher_ended: bool = False
    conversation_type: ConversationType = field(init=False)
    student_name: str | None = field(init=False)
    teacher_system_prompt: str = field(init=False)
    student_system_prompt: str = field(init=False)
    student_initial_prompt: str = field(init=False)
    student_final_prompt: str = field(init=False)

    def __post_init__(self) -> None:
        problem_hash = hash(self.problem)
        self.conversation_type = (
            ConversationType.ATTEMPTED if problem_hash % 2 else ConversationType.GUIDED
        )
        self.student_name = STUDENT_NAMES[problem_hash % len(STUDENT_NAMES)]
        self.teacher_system_prompt = render_teacher_prompt(
            student_name=self.student_name,
            problem=self.problem,
        )
        self.student_system_prompt = render(
            SIMPLE_STUDENT_PROMPT,
            student_name=self.student_name,
            problem=self.problem,
        )
        self.student_initial_prompt = render(
            STUDENT_INITIAL_ATTEMPT_PROMPT, problem=self.problem
        )
        self.student_final_prompt = render(STUDENT_FINAL_PROMPT)

    @property
    def teacher_turns(self) -> int:
        return sum(message["role"] == "teacher" for message in self.conversation)

    @property
    def failed_native_judges(self) -> bool:
        return any(decision.rejected for decision in self.native_judges)

    def add_initial_attempt(self, attempt: str) -> None:
        self.initial_attempt = attempt
        self.conversation.append(
            {
                "role": "student",
                "content": render(INITIAL_ATTEMPT_WRAPPER, attempt=attempt),
            }
        )

    @property
    def format_failed(self) -> bool:
        return bool(self.format_errors)

    def add_teacher(self, content: str) -> None:
        self.conversation.append(
            {
                "role": "teacher",
                "content": content,
                "student_visible": True,
            }
        )
        _visible, ended, error = parse_tagged_teacher_action(
            content,
            allow_end=True,
            require_nonempty_output=True,
        )
        if error:
            self.format_errors.append(error)
            self.termination_reason = "format_error"
        elif ended:
            # The shared action contract makes <end></end> control-only; it
            # is not an empty teacher message in the student's transcript.
            self.conversation[-1]["student_visible"] = False
            self.teacher_ended = True
            self.termination_reason = "end_of_conversation"

    def add_student(self, content: str) -> None:
        self.conversation.append(
            {
                "role": "student",
                "content": content,
                "student_visible": True,
            }
        )

    def teacher_messages(self) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.teacher_system_prompt}]
        messages.extend(
            {
                "role": "assistant" if message["role"] == "teacher" else "user",
                "content": message["content"],
            }
            for message in self.conversation
        )
        return messages

    def student_messages(self, *, final: bool = False) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.student_system_prompt}]
        messages.extend(
            {
                "role": "assistant" if message["role"] == "student" else "user",
                "content": (
                    student_visible_text(message["content"])
                    if message["role"] == "teacher"
                    else message["content"]
                ),
            }
            for message in self.conversation
            if message.get("student_visible", True)
        )
        if final:
            messages.append({"role": "user", "content": self.student_final_prompt})
        return messages

    def initial_student_messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.student_initial_prompt}]

    def hidden_conversation(self) -> list[dict[str, str]]:
        return [
            {
                "role": message["role"],
                "content": (
                    student_visible_text(message["content"])
                    if message["role"] == "teacher"
                    else message["content"]
                ),
            }
            for message in self.conversation
            if message.get("student_visible", True)
        ]

    def content_token_count(self, tokenizer: Any) -> int:
        return sum(
            len(tokenizer.encode(message["content"])) for message in self.conversation
        )

    def should_stop_dialogue(
        self,
        *,
        tokenizer: Any,
        max_teacher_turns: int,
        max_tokens_in_conversation: int,
    ) -> bool:
        if self.format_failed:
            return True
        if self.teacher_turns >= max_teacher_turns:
            self.termination_reason = self.termination_reason or "max_turns"
            return True
        if self.teacher_ended:
            return True
        if self.content_token_count(tokenizer) > max_tokens_in_conversation:
            self.termination_reason = self.termination_reason or "max_tokens"
            return True
        return False

    def to_trace(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "answer": self.answer,
            "conversation_type": self.conversation_type.value,
            "student_name": self.student_name,
            "teacher_system_prompt": self.teacher_system_prompt,
            "student_system_prompt": self.student_system_prompt,
            "student_initial_prompt": self.student_initial_prompt,
            "student_final_prompt": self.student_final_prompt,
            "conversation": self.conversation,
            "initial_attempt": self.initial_attempt,
            "termination_reason": self.termination_reason,
            "format_errors": self.format_errors,
            "teacher_ended": self.teacher_ended,
            "native_judges": [
                {
                    "rule": decision.rule,
                    "reasoning": decision.reasoning,
                    "decision": decision.decision,
                }
                for decision in self.native_judges
            ],
            "final_solutions": self.final_solutions,
        }
