from __future__ import annotations

import asyncio
import json
import os
import random
import re
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import torch

from examples.common.chat_budget import ChatContextBudget
from examples.common.openai_utils import AsyncLLMCaller, AuxModelConfig
from examples.common.parsing import parse_json_dict
from examples.sherpa.configs import (
    NO_PREFERENCE,
    TUTOR_EVAL_STUDENT_FIELD,
    TUTOR_TRAIN_STUDENT_FIELD,
    TutorStudentModelConfig,
)
from examples.sherpa.core.callers import (
    ApiAuxiliaryCaller,
    AReaLEngineActorCaller,
    AReaLEngineAuxiliaryCaller,
    AReaLEngineChatCaller,
    ExternalActorCaller,
    TextCallResult,
)
from examples.sherpa.core.generation_budget import (
    CONTEXT_BUDGET_TERMINATION_REASON,
    ContextBudgetLimitExceeded,
)
from examples.sherpa.core.history import (
    trace_to_history_record,
    trace_to_json,
)
from examples.sherpa.core.math import score_math_answer
from examples.sherpa.core.parsers import (
    parse_guidance_gate_result,
    parse_tagged_teacher_action,
    parse_thinking_teacher_action,
)
from examples.sherpa.core.repetition import normalize_exact_teacher_output
from examples.sherpa.core.rewards import EpisodeRewardComputer, artifact_to_trace
from examples.sherpa.core.tensors import response_to_tensordict
from examples.sherpa.core.text import (
    strip_reasoning_for_context as _strip_reasoning_for_context,
)
from examples.sherpa.core.types import (
    AdaptiveGateResult,
    EpisodeArtifact,
    GuidanceGateMode,
    GuidanceGateResult,
    JudgeResult,
    PublicHistoryState,
    StudentTurnState,
    TeacherPreSolveAttempt,
    TeacherPreSolveResult,
    TurnArtifact,
    TurnTrace,
    TutorTurnState,
)
from examples.sherpa.prompts import (
    ADAPTIVE_GATE_NO_LAST_STUDENT_MESSAGE,
    ADAPTIVE_GATE_SYSTEM_PROMPT,
    ADAPTIVE_GATE_USER_TEMPLATE,
    ANSWER_JUDGE_SYSTEM_PROMPT,
    ANSWER_JUDGE_USER_TEMPLATE,
    FREE_CHAT_STUDENT_RETEST_TEMPLATE,
    FREE_CHAT_STUDENT_SYSTEM_PROMPT,
    FREE_CHAT_TEACHER_OPEN_PROMPT,
    FREE_CHAT_TEACHER_SOLVE_PROMPT,
    FREE_CHAT_TEACHER_SYSTEM_PROMPT,
    GUIDANCE_GATE_FAILED_FEEDBACK_TEMPLATE,
    GUIDANCE_GATE_PENDING_FEEDBACK,
    GUIDANCE_GATE_SYSTEM_PROMPT,
    GUIDANCE_GATE_USER_TEMPLATE,
    NON_THINKING_TEACHER_OUTPUT_FORMAT_WITH_END_PROMPT,
    PUBLIC_HISTORY_ENTRY_TEMPLATE,
    TEACHER_GUIDANCE_INSTRUCTION,
    TEACHER_HISTORY_MASKED_TEMPLATE,
    THINKING_TEACHER_OUTPUT_FORMAT_WITH_END_PROMPT,
    render_prompt,
)

from areal import workflow_context
from areal.api import ModelResponse, RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("TutorWorkflow")

FORMAT_TERMINATION_REASON = "format_error"
TEACHER_EXACT_REPEAT_TERMINATION_REASON = "teacher_exact_repeat"
TEACHER_END_TERMINATION_REASON = "teacher_end"
TEACHER_PRE_SKIPPED_TERMINATION_REASON = "pre_solve_skipped"
GUIDANCE_GATE_MODES = {
    "disabled",
    "masked_continue",
}
GUIDANCE_GATE_STUDENT_REPLY = (
    "Please do not leak the answer to me; let me think by myself."
)


def load_adaptive_gate_prompts(path: str) -> dict[str, dict[str, str]]:
    """{name: {"source": citation, "preference": prompt}} from a JSON file.

    `source` is a required citation for the learner type and is never shown to the
    model.
    """
    normalized_path = str(path or "").strip()
    if not normalized_path:
        return {}

    file_path = Path(normalized_path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"adaptive gate prompt file not found: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"adaptive gate prompts must be valid JSON: {file_path}: {exc.msg}"
        ) from exc

    entries = payload.get("preferences") if isinstance(payload, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError(
            "preference prompts must be an object with a non-empty "
            f"'preferences' mapping: {file_path}"
        )

    prompts: dict[str, dict[str, str]] = {}
    for name, entry in entries.items():
        clean_name = str(name or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", clean_name):
            raise ValueError(
                f"preference name {name!r} is not a valid name segment, and it "
                f"becomes part of a student name: {file_path}"
            )
        if clean_name == NO_PREFERENCE:
            raise ValueError(
                f"{NO_PREFERENCE!r} is the open gate and must not have a prompt: "
                f"{file_path}"
            )
        if not isinstance(entry, dict):
            raise ValueError(
                f"preference {clean_name!r} must be an object with 'source' and "
                f"'preference': {file_path}"
            )
        preference = str(entry.get("preference") or "").strip()
        source = str(entry.get("source") or "").strip()
        if not preference:
            raise ValueError(
                f"preference {clean_name!r} has an empty 'preference' prompt: {file_path}"
            )
        if not source:
            raise ValueError(
                f"preference {clean_name!r} has no 'source'; each preference "
                f"must cite the learner type it is based on: {file_path}"
            )
        prompts[clean_name] = {"source": source, "preference": preference}
    return prompts


def _parse_adaptive_gate_reply(text: str) -> tuple[bool, str, str | None]:
    """Return ``(passed, reason, parse_error)`` from a binary gate reply.

    The gate asks for two XML-style tags so mathematical backslashes in the
    reasoning cannot corrupt the envelope. A JSON object with the same two fields
    is also accepted. Anything without an exact PASS/FAIL verdict is retried and
    ultimately fails closed.
    """
    body = _strip_reasoning_for_context(str(text or "")).strip()
    if not body:
        return False, "", "adaptive gate returned an empty response."
    # Tolerate a surrounding code fence; its content is still the requested reply.
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
        body = re.sub(r"\n?```$", "", body).strip()

    xml_match = re.fullmatch(
        r"<reasoning>(.*?)</reasoning>\s*<verdict>\s*([^<]*?)\s*</verdict>",
        body,
        flags=re.DOTALL,
    )
    if xml_match is not None:
        reason = xml_match.group(1).strip()
        verdict = xml_match.group(2).strip().upper()
        if verdict == "PASS":
            return True, reason, None
        if verdict == "FAIL":
            return False, reason, None
        return (
            False,
            reason,
            f"adaptive gate verdict was {verdict!r}, not PASS/FAIL.",
        )
    if any(
        tag in body
        for tag in ("<reasoning>", "</reasoning>", "<verdict>", "</verdict>")
    ):
        return False, "", "adaptive gate reply was not valid tagged XML."

    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return False, "", "adaptive gate reply contained no JSON object."
    try:
        payload = json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        return False, "", f"adaptive gate reply was not valid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return False, "", "adaptive gate reply was not a JSON object."
    verdict = str(payload.get("verdict") or "").strip().upper()
    reason = str(payload.get("reasoning") or "").strip()
    if verdict == "PASS":
        return True, reason, None
    if verdict == "FAIL":
        return False, reason, None
    return False, reason, f"adaptive gate verdict was {verdict!r}, not PASS/FAIL."


def load_adaptive_gate_complaints(path: str) -> dict[str, tuple[str, ...]]:
    """{name: complaints} from a JSON file.

    Every complaint names the remedy the student wants, so the lists are
    per-preference.
    """
    normalized_path = str(path or "").strip()
    if not normalized_path:
        return {}

    file_path = Path(normalized_path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"adaptive gate complaint file not found: {file_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"adaptive gate complaints must be valid JSON: {file_path}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"adaptive gate complaints must be an object: {file_path}")

    def clean_list(raw: Any, label: str) -> tuple[str, ...]:
        if not isinstance(raw, list) or not raw:
            raise ValueError(
                f"adaptive gate complaints {label} must be a non-empty array: {file_path}"
            )
        lines = tuple(str(line).strip() for line in raw if str(line).strip())
        if not lines:
            raise ValueError(
                f"adaptive gate complaints {label} has no non-empty lines: {file_path}"
            )
        return lines

    raw_explain = payload.get("explain")
    if not isinstance(raw_explain, dict) or not raw_explain:
        raise ValueError(
            f"adaptive gate complaints need a non-empty 'explain' mapping: {file_path}"
        )
    return {
        str(name).strip(): clean_list(lines, f"explain.{name}")
        for name, lines in raw_explain.items()
    }


def _safe_scalar(**metrics: Any) -> None:
    try:
        stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
    except Exception:
        logger.debug("Skipping stats logging outside workflow context.")


def _safe_generalize_scalar(**metrics: Any) -> None:
    try:
        ctx = workflow_context.get()
        split = "test" if bool(getattr(ctx, "is_eval", False)) else "train"
        scoped_metrics = {f"{split}/{key}": value for key, value in metrics.items()}
        stats_tracker.get("generalize").scalar(**scoped_metrics)
    except Exception:
        logger.debug("Skipping generalize stats logging outside workflow context.")


# The re-test: the original task, solved from scratch on an independent branch of
# the same chat after the conversation ends.
ORIGINAL_RETEST_LEVEL = "original"


@dataclass(slots=True)
class StudentGeneralizationResult:
    level: str
    task: str = ""
    ground_truth: str = ""
    attempted: bool = False
    skipped: bool = False
    skip_reason: str = ""
    student_output: str = ""
    student_error: str | None = None
    judge_result: JudgeResult | None = None
    replay_count: int = 0
    replay_correct: int = 0
    reward: float = 0.0
    public_history: str = ""
    reward_turn_idx: int | None = None


@dataclass(slots=True)
class StudentGeneralizationAnchor:
    public_history: PublicHistoryState
    reward_turn_idx: int | None


@dataclass(slots=True)
class StudentModelRuntime:
    name: str
    model: str
    weight: float
    caller: ApiAuxiliaryCaller
    # What this student demands of the teacher's manner, or "" for the open gate.
    # See TutorStudentModelConfig.preference.
    preference: str = ""


@dataclass(slots=True)
class SelectedStudent:
    name: str
    model: str
    caller: ApiAuxiliaryCaller
    preference: str = ""


class TutorAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: Any,
        tokenizer: str | Any,
        max_turns: int = 10,
        enable_thinking: bool = False,
        teacher_response_format: str = "non_thinking",
        guidance_gate_mode: GuidanceGateMode = "masked_continue",
        mask_rejected_turns: bool = True,
        retest_replays: int = 8,
        aux_mode: str = "self",
        aux_enable_thinking: bool = False,
        aux_base_url: str = "",
        aux_model: str = "",
        aux_api_key: str = "EMPTY",
        aux_timeout: int = 120,
        aux_max_tokens: int = 1024,
        aux_temperature: float = 0.0,
        aux_top_p: float | None = 1.0,
        max_concurrent_aux_calls: int = 32,
        aux_request_params: dict[str, Any] | None = None,
        answer_judge_enabled: bool = True,
        answer_judge_max_tokens: int = 256,
        student_models: list[dict[str, Any]] | None = None,
        adaptive_gate: dict[str, Any] | None = None,
        guidance_gate_fail_penalty: float = 0.0,
        format_error_penalty: float = -0.5,
        teacher_exact_repeat_penalty: float = -0.5,
        adaptive_gate_fail_penalty: float = 0.0,
        soft_overlong_penalty: dict[str, Any] | None = None,
        length_retry_enabled: bool = True,
        length_retry_attempts: int = 3,
        teacher_pre_enabled: bool = True,
        teacher_pre_verify: bool = True,
        teacher_pre_attempts: int = 3,
        teacher_pre_max_tokens: int = 4096,
        seed: int = 0,
        debug_trace_dir: str | None = None,
        debug_trace_every_n_rollouts: int = 1,
        max_train_sample_tokens: int | None = None,
        tokenizer_path: str | None = None,
        model_context_length: int | None = None,
        context_window_margin: int = 256,
        eval_repeat_count: int = 1,
    ):
        self.eval_repeat_count = int(eval_repeat_count)
        if self.eval_repeat_count < 1:
            raise ValueError("eval_repeat_count must be >= 1.")
        self._eval_repeat_outcomes: dict[int, list[float]] = {}
        self.max_turns = int(max_turns)
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        # task_id -> fraction the student solves unaided. A workflow instance
        # serves one rollout group, so the trajectories of a group share one
        # estimate and every group of the problem draws its own.
        self._no_teaching_baselines: dict[str, float] = {}
        self._no_teaching_baseline_lock = asyncio.Lock()
        # (problem, weight version) -> in-flight pre-solve task shared by the group.
        # Keyed on the version because the teacher is being trained; versions older
        # than the previous one are dropped.
        self._teacher_pre_solve_shared: dict[tuple[str, int], Any] = {}
        self._teacher_pre_solve_shared_lock = asyncio.Lock()
        self.enable_thinking = enable_thinking
        if teacher_response_format not in {"non_thinking", "thinking"}:
            raise ValueError("teacher_response_format must be non_thinking or thinking")
        self.teacher_response_format = teacher_response_format
        if guidance_gate_mode not in GUIDANCE_GATE_MODES:
            raise ValueError(
                "guidance_gate_mode must be 'disabled' or 'masked_continue'."
            )
        self.guidance_gate_mode: GuidanceGateMode = guidance_gate_mode
        self.gconfig = gconfig
        self.temperature = gconfig.temperature
        self.top_p = gconfig.top_p
        self.max_completion_tokens = gconfig.max_new_tokens
        if aux_mode not in {"api", "self"}:
            raise ValueError(f"aux_mode must be 'api' or 'self', got {aux_mode!r}")
        self.aux_mode = aux_mode
        self.aux_enable_thinking = bool(aux_enable_thinking)
        self.aux_base_url = aux_base_url
        self.aux_model = aux_model
        self.aux_api_key = aux_api_key
        self.aux_timeout = int(aux_timeout)
        self.aux_max_tokens = int(aux_max_tokens)
        self.aux_temperature = float(aux_temperature)
        self.aux_top_p = aux_top_p
        self.aux_request_params = dict(aux_request_params or {})
        self.max_concurrent_aux_calls = int(max_concurrent_aux_calls)
        self._student_model_configs = self._normalize_student_model_configs(
            student_models
        )
        positive_student_configs = [
            config
            for config in self._student_model_configs
            if float(config["weight"]) > 0.0
        ]
        self._stratified_student_scores = {
            str(config["name"]): 0.0 for config in positive_student_configs
        }
        self._stratified_student_batch_index = 0
        self._self_aux_semaphore = asyncio.Semaphore(
            max(1, self.max_concurrent_aux_calls)
        )
        self.context_window_margin = int(context_window_margin)
        if format_error_penalty > 0.0:
            raise ValueError("format_error_penalty must be <= 0.")
        if teacher_exact_repeat_penalty >= 0.0:
            raise ValueError(
                "teacher_exact_repeat_penalty must be < 0: an exact repeat ends the "
                "episode and this is what that turn is charged."
            )
        if adaptive_gate_fail_penalty > 0.0:
            raise ValueError("adaptive_gate_fail_penalty must be <= 0.")
        self.guidance_gate_fail_penalty = float(guidance_gate_fail_penalty)
        self.format_error_penalty = float(format_error_penalty)
        self.teacher_exact_repeat_penalty = float(teacher_exact_repeat_penalty)
        self.adaptive_gate_fail_penalty = float(adaptive_gate_fail_penalty)
        soft_overlong = dict(soft_overlong_penalty or {})
        self.soft_overlong_enabled = bool(soft_overlong.get("enabled", False))
        self.soft_overlong_buffer_tokens = int(soft_overlong.get("buffer_tokens", 512))
        self.soft_overlong_max_penalty = float(soft_overlong.get("max_penalty", -0.05))
        if self.soft_overlong_buffer_tokens <= 0:
            raise ValueError("soft_overlong buffer_tokens must be positive.")
        if self.soft_overlong_max_penalty > 0.0:
            raise ValueError("soft_overlong max_penalty must be <= 0.")
        if self.soft_overlong_enabled:
            if self.soft_overlong_max_penalty == 0.0:
                raise ValueError("soft_overlong enabled=true requires max_penalty < 0.")
            if self.soft_overlong_buffer_tokens >= self.max_completion_tokens:
                raise ValueError(
                    "soft_overlong buffer_tokens must be smaller than the teacher "
                    "generation limit."
                )
        self.teacher_pre_enabled = bool(teacher_pre_enabled)
        self.teacher_pre_verify = bool(teacher_pre_verify)
        self.teacher_pre_attempts = int(teacher_pre_attempts)
        if self.teacher_pre_attempts < 1:
            raise ValueError("teacher_pre_attempts must be >= 1.")
        self.teacher_pre_max_tokens = int(teacher_pre_max_tokens)
        self.seed = int(seed)
        self.answer_judge_enabled = bool(answer_judge_enabled)
        self.answer_judge_max_tokens = max(1, int(answer_judge_max_tokens))
        self._answer_judge_cache: dict[tuple[str, str, str], JudgeResult] = {}
        self.debug_trace_dir = debug_trace_dir.strip() if debug_trace_dir else ""
        self.debug_trace_every_n_rollouts = max(1, int(debug_trace_every_n_rollouts))
        self.max_train_sample_tokens = max_train_sample_tokens
        self.retest_replays = max(1, int(retest_replays))
        # With mask_rejected_turns the re-test reward is credited only to turns
        # that passed both the adaptive gate and the guidance gate.
        self.mask_rejected_turns = bool(mask_rejected_turns)
        self.length_retry_enabled = bool(length_retry_enabled)
        self.length_retry_attempts = max(1, int(length_retry_attempts))
        self.last_history: list[dict[str, Any]] = []
        self.last_traces: list[TurnTrace] = []
        self.last_student_generalization_results: list[StudentGeneralizationResult] = []
        self.last_teacher_pre_solve_result: TeacherPreSolveResult | None = None
        self.tokenizer = (
            load_hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.tokenizer_path = tokenizer_path
        self.model_context_length = model_context_length
        self.teacher_context_budget = ChatContextBudget(
            tokenizer_path=tokenizer_path,
            context_length=model_context_length,
            safety_margin=context_window_margin,
        )
        self.aux_caller = None
        if self.aux_mode == "api":
            aux_config = AuxModelConfig(
                base_url=aux_base_url,
                model=aux_model,
                api_key=aux_api_key,
                timeout=aux_timeout,
                max_tokens=aux_max_tokens,
                temperature=aux_temperature,
                top_p=aux_top_p,
                max_concurrency=max_concurrent_aux_calls,
                request_params=self.aux_request_params,
                tokenizer_path=tokenizer_path,
                context_length=model_context_length,
                context_window_margin=context_window_margin,
            )
            self.aux_caller = ApiAuxiliaryCaller(AsyncLLMCaller(aux_config))
        self.student_model_runtimes = self._build_student_model_runtimes(
            tokenizer_path=tokenizer_path,
            context_length=model_context_length,
            context_window_margin=context_window_margin,
        )
        self._configure_adaptive_gate(adaptive_gate)

    def _configure_adaptive_gate(self, adaptive_gate: dict[str, Any] | None) -> None:
        """Load the preference prompts and complaints, and check the pool against them.

        Validation runs at construction so a missing prompt fails before rollout.
        """
        settings = dict(adaptive_gate or {})
        self.adaptive_gate_retries = max(1, int(settings.get("retries", 3)))
        self.adaptive_gate_prompts = load_adaptive_gate_prompts(
            str(settings.get("prompts_path", "") or "")
        )
        self.adaptive_gate_complaints = load_adaptive_gate_complaints(
            str(settings.get("complaints_path", "") or "")
        )
        self._adaptive_gate_fallback_rng = random.Random(f"{self.seed}:personality")

        demanded = sorted(
            {
                runtime.preference
                for runtime in self.student_model_runtimes.values()
                if runtime.preference and runtime.preference != NO_PREFERENCE
            }
        )
        self.adaptive_gate_active = bool(demanded)
        if not demanded:
            return
        if not self.adaptive_gate_prompts:
            raise ValueError(
                "adaptive_gate_prompts_path is required: students demand "
                f"{demanded} and no preference prompts were loaded."
            )
        missing_prompts = [p for p in demanded if p not in self.adaptive_gate_prompts]
        if missing_prompts:
            raise ValueError(
                f"no preference prompt for preferences {missing_prompts}; "
                f"the file defines {sorted(self.adaptive_gate_prompts)}."
            )
        if not self.adaptive_gate_complaints:
            raise ValueError(
                "adaptive_gate_complaints_path is required: a closed gate has to put "
                "something in the student's slot."
            )
        missing_complaints = [
            p for p in demanded if p not in self.adaptive_gate_complaints
        ]
        if missing_complaints:
            raise ValueError(
                f"no complaints for preferences {missing_complaints}; "
                f"the file defines {sorted(self.adaptive_gate_complaints)}."
            )
        logger.info(
            "adaptive gate active: %s",
            ", ".join(
                f"{name}[{self.adaptive_gate_prompts[name]['source'].split(':')[0]}]"
                for name in demanded
            ),
        )

    def _turn_hidden_from_student(self, artifact: TurnArtifact) -> bool:
        """Whether this teacher/fake-student pair belongs only to teacher history.

        Turns that failed the guidance gate or the adaptive gate are shown to
        the teacher but never to the student.
        """

        return bool(artifact.guidance_gate_masked or artifact.adaptive_gate_failed)

    def _adaptive_gate_rng(self, *, kind: str, turn_idx: int) -> random.Random:
        """Task-scoped, so every rollout of one task draws from the same stream.

        Falls back to a shared stream when there is no task id.
        """
        try:
            task_id = getattr(workflow_context.get(), "task_id", None)
        except Exception:
            task_id = None
        if task_id is None:
            return self._adaptive_gate_fallback_rng
        return random.Random(
            f"{self.seed}:personality-{kind}:{int(task_id)}:{int(turn_idx)}"
        )

    def _draw_adaptive_gate_complaint(self, preference: str, *, turn_idx: int) -> str:
        """The student's reply to a message that failed the gate.

        Drawn from a fixed file rather than generated, since this reply is the only
        channel that conveys the preference to the teacher.
        """
        rng = self._adaptive_gate_rng(kind="complaint", turn_idx=turn_idx)
        return rng.choice(self.adaptive_gate_complaints[preference])

    async def _run_adaptive_gate(
        self,
        preference: str,
        teacher_message: str,
        *,
        task: str,
        turn_idx: int,
        previous_student_message: str | None = None,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> AdaptiveGateResult | None:
        """One gate outcome for a teacher message, or None when no gate applies."""
        if not preference or preference == NO_PREFERENCE:
            return None
        entry = self.adaptive_gate_prompts.get(preference)
        if entry is None:
            raise ValueError(f"no adaptive gate prompt for preference {preference!r}.")

        user_prompt = ADAPTIVE_GATE_USER_TEMPLATE.format(
            preference=entry["preference"],
            task=str(task or "").strip(),
            last_student_message=(
                str(previous_student_message or "").strip()
                or ADAPTIVE_GATE_NO_LAST_STUDENT_MESSAGE
            ),
            teacher_message=teacher_message.strip(),
        )
        system_prompt = ADAPTIVE_GATE_SYSTEM_PROMPT
        rid_prefix = f"personality-gate-v3-{preference}-{turn_idx}"
        last_error = "adaptive gate produced no verdict."
        raw_output = ""
        for attempt in range(1, self.adaptive_gate_retries + 1):
            result = await self._call_auxiliary_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                aux_caller=aux_caller,
                rid_prefix=rid_prefix,
            )
            raw_output = result.raw_text or result.text
            if result.error:
                last_error = str(result.error)
                continue
            verdict, reason, parse_error = _parse_adaptive_gate_reply(result.text)
            if parse_error:
                last_error = parse_error
                continue
            return AdaptiveGateResult(
                raw_output=raw_output,
                passed=verdict,
                reason=reason,
                error=None,
                attempts=attempt,
            )
        # Fail closed when no retry produced a clean verdict; the rate is reported
        # as the gate_error metric.
        return AdaptiveGateResult(
            raw_output=raw_output,
            passed=False,
            reason="",
            error=last_error,
            attempts=self.adaptive_gate_retries,
        )

    @staticmethod
    def _normalize_student_model_configs(
        student_models: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for raw_config in student_models or []:
            if isinstance(raw_config, TutorStudentModelConfig):
                config = raw_config
            elif isinstance(raw_config, dict):
                config = TutorStudentModelConfig(**raw_config)
            else:
                raise TypeError(
                    "student_models entries must be dictionaries or "
                    "TutorStudentModelConfig instances."
                )
            if config.name in seen_names:
                raise ValueError("student_models names must be unique.")
            seen_names.add(config.name)
            normalized.append(asdict(config))

        if not any(config["weight"] > 0.0 for config in normalized):
            raise ValueError(
                "student_models must contain at least one student with positive weight."
            )
        return normalized

    def _build_student_model_runtimes(
        self,
        *,
        tokenizer_path: str | None,
        context_length: int | None,
        context_window_margin: int,
    ) -> dict[str, StudentModelRuntime]:
        runtimes: dict[str, StudentModelRuntime] = {}
        for student in self._student_model_configs:
            config = AuxModelConfig(
                base_url=student["base_url"],
                model=student["model"],
                api_key=student["api_key"],
                timeout=student["timeout"],
                max_tokens=student["max_tokens"],
                temperature=student["temperature"],
                top_p=student["top_p"],
                max_concurrency=student["max_concurrent_calls"],
                request_params=student["request_params"],
                tokenizer_path=tokenizer_path,
                context_length=context_length,
                context_window_margin=context_window_margin,
            )
            api_caller = AsyncLLMCaller(config)
            runtimes[student["name"]] = StudentModelRuntime(
                name=student["name"],
                model=student["model"],
                weight=student["weight"],
                caller=ApiAuxiliaryCaller(api_caller),
                preference=str(student.get("preference", "") or ""),
            )
        return runtimes

    def _group_rng(
        self, role: str, group_key: str, rollout_version: int | None
    ) -> random.Random:
        """A draw that is constant across one problem's rollouts in one step.

        `gconfig.n_samples` rollouts of a problem form the GRPO group, and
        `actor.group_baseline='episode'` subtracts that group's mean return from
        every member, so a per-rollout draw would add variance the teacher cannot
        control to the advantage.

        Seeding on the problem holds the draw fixed across the group; including
        the weight version redraws it each step. A None version (external client,
        no local weights) fixes the draw per problem for the whole run.
        """
        version = int(rollout_version) if rollout_version is not None else -1
        return random.Random(f"{self.seed}:{role}:{group_key}:{version}")

    def _stratified_student_assignments(self, batch_size: int) -> list[str]:
        """Return smooth weighted quotas for one input batch.

        The scheduler carries its residual scores between batches. Equal weights
        therefore differ by at most one when the batch is not divisible by the
        student count, and are exactly equal when it is divisible.
        """
        active = [
            (str(config["name"]), float(config["weight"]))
            for config in self._student_model_configs
            if float(config["weight"]) > 0.0
        ]
        total_weight = sum(weight for _name, weight in active)
        assignments: list[str] = []
        for _ in range(batch_size):
            for name, weight in active:
                self._stratified_student_scores[name] += weight
            selected_name, _selected_weight = max(
                active,
                key=lambda item: self._stratified_student_scores[item[0]],
            )
            self._stratified_student_scores[selected_name] -= total_weight
            assignments.append(selected_name)

        rng = random.Random(
            f"{self.seed}:student-stratified:{self._stratified_student_batch_index}"
        )
        rng.shuffle(assignments)
        self._stratified_student_batch_index += 1
        return assignments

    def prepare_rollout_batch(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Assign students to one input batch in fixed proportions (stratified)."""
        if len(self._stratified_student_scores) < 2 or not data:
            return data
        if any(TUTOR_TRAIN_STUDENT_FIELD in item for item in data):
            raise ValueError(
                f"Training dataset contains reserved field "
                f"{TUTOR_TRAIN_STUDENT_FIELD!r}."
            )
        assignments = self._stratified_student_assignments(len(data))
        return [
            {**item, TUTOR_TRAIN_STUDENT_FIELD: student_name}
            for item, student_name in zip(data, assignments, strict=True)
        ]

    def _select_student(
        self,
        data: dict[str, Any],
        *,
        group_key: str = "",
        rollout_version: int | None = None,
    ) -> SelectedStudent:
        try:
            is_eval = bool(getattr(workflow_context.get(), "is_eval", False))
        except Exception:
            is_eval = False
        eval_forced_name = (
            str(data.get(TUTOR_EVAL_STUDENT_FIELD) or "").strip() if is_eval else ""
        )
        train_forced_name = (
            str(data.get(TUTOR_TRAIN_STUDENT_FIELD) or "").strip()
            if not is_eval and len(self._stratified_student_scores) >= 2
            else ""
        )
        forced_name = eval_forced_name or train_forced_name

        if forced_name:
            runtime = self.student_model_runtimes.get(forced_name)
            if runtime is None:
                raise ValueError(
                    f"Unknown forced student {forced_name!r}; expected "
                    f"one of {sorted(self.student_model_runtimes)}."
                )
        else:
            runtimes = list(self.student_model_runtimes.values())
            # Group-scoped, not per-rollout: see _group_rng.
            rng = self._group_rng(
                "student", group_key or str(data.get("id") or ""), rollout_version
            )
            runtime = rng.choices(
                runtimes,
                weights=[item.weight for item in runtimes],
                k=1,
            )[0]
        return SelectedStudent(
            name=runtime.name,
            model=runtime.model,
            caller=runtime.caller,
            preference=runtime.preference,
        )

    @staticmethod
    def _student_metric_name(name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-")
        return normalized or "unknown"

    def _teacher_output_format_prompt(self) -> str:
        if self.teacher_response_format == "thinking":
            return THINKING_TEACHER_OUTPUT_FORMAT_WITH_END_PROMPT
        return NON_THINKING_TEACHER_OUTPUT_FORMAT_WITH_END_PROMPT

    def _parse_tutor_action(self, raw_output: str) -> tuple[str, bool, str | None]:
        if self.teacher_response_format == "thinking":
            output, ended, error = parse_thinking_teacher_action(
                raw_output, allow_end=True
            )
            return output or "", ended, error
        output, ended, parse_error = parse_tagged_teacher_action(
            raw_output,
            allow_end=True,
            require_nonempty_output=True,
        )
        if output is None:
            return "", False, parse_error or "failed to parse tagged teacher output"
        return _strip_reasoning_for_context(output), ended, None

    async def arun_episode(self, engine, data: dict[str, Any]):
        return await self._run_episode(data, engine=engine)

    async def _run_episode(
        self,
        data: dict[str, Any],
        engine: Any | None = None,
        external_client: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if (engine is None) == (external_client is None):
            raise ValueError("Exactly one tutor generation source must be provided.")

        task = str(data["task"])
        ground_truth = str(data["ground_truth"])
        # Identifies the GRPO group; the pre-solve cache and the student draw are
        # scoped to it.
        group_key = str(data.get("id") or task)
        trajectory_id = uuid.uuid4().int & ((1 << 63) - 1)
        self.last_student_generalization_results = []
        turn_artifacts: list[TurnArtifact] = []
        guidance_gate_fail_count = 0
        termination_reason = "max_turns"
        episode_lora_version = None
        if engine is not None:
            try:
                context_lora_version = getattr(
                    workflow_context.get(), "lora_version", None
                )
            except Exception:
                context_lora_version = None
            if context_lora_version is not None:
                episode_lora_version = int(context_lora_version)
            elif hasattr(engine, "get_version"):
                episode_lora_version = int(engine.get_version())
        actor_chat_caller = (
            self._make_engine_chat_caller(
                engine,
                enable_thinking=self.enable_thinking,
            )
            if engine is not None
            else None
        )
        aux_chat_caller = (
            self._make_engine_chat_caller(
                engine,
                enable_thinking=self.aux_enable_thinking,
            )
            if engine is not None and self.aux_mode == "self"
            else None
        )
        actor_caller = self._make_actor_caller(
            chat_caller=actor_chat_caller,
            external_client=external_client,
        )
        aux_caller = self._make_auxiliary_caller(chat_caller=aux_chat_caller)
        selected_student = self._select_student(
            data,
            group_key=group_key,
            rollout_version=episode_lora_version,
        )
        student_caller = selected_student.caller
        answer_judge_caller = self._make_answer_judge_caller(
            chat_caller=aux_chat_caller
        )
        teacher_pre_solve_result: TeacherPreSolveResult | None = None
        teacher_pre_cache_hit = False
        self.last_teacher_pre_solve_result = None
        if self.teacher_pre_enabled:
            (
                teacher_pre_solve_result,
                teacher_pre_cache_hit,
            ) = await self._teacher_pre_solve_for_group(
                task,
                ground_truth,
                actor_caller=actor_caller,
                answer_judge_caller=answer_judge_caller,
                lora_version=episode_lora_version,
                group_key=group_key,
            )
            self.last_teacher_pre_solve_result = teacher_pre_solve_result
            # The pre-solve result is shared by the group, so a rejected draft drops
            # the whole group for this step rather than thinning it.
            if not teacher_pre_solve_result.accepted:
                self.last_history = []
                self.last_traces = []
                self.last_student_generalization_results = []
                completed_repeat_outcome = self._log_rollout_stats(
                    total_reward=0.0,
                    traces=[],
                    termination_reason=TEACHER_PRE_SKIPPED_TERMINATION_REASON,
                    guidance_gate_fail_count=0,
                    teacher_pre_solve_result=teacher_pre_solve_result,
                    teacher_pre_cache_hit=teacher_pre_cache_hit,
                    student_name=selected_student.name,
                )
                if completed_repeat_outcome is not None and self.debug_trace_dir:
                    await self._dump_eval_repeat_outcomes(*completed_repeat_outcome)
                await self._maybe_dump_debug_trace(
                    trajectory_id=trajectory_id,
                    task=task,
                    ground_truth=ground_truth,
                    latest_student_answer="",
                    total_reward=0.0,
                    traces=[],
                    termination_reason=TEACHER_PRE_SKIPPED_TERMINATION_REASON,
                    guidance_gate_fail_count=0,
                    teacher_pre_solve_result=teacher_pre_solve_result,
                    student_name=selected_student.name,
                    student_model=selected_student.model,
                )
                return None

        # The teacher opens into an empty history, so its first turn is generated
        # from its system prompt alone. Nothing is judged until the re-test.
        public_history = PublicHistoryState(
            summary="",
            turn_count=0,
            turns=[],
        )
        # Aliases the complete history until the first hidden turn, which advances
        # only `public_history`; later student calls continue from this branch.
        student_visible_history = public_history
        student_history_filtered = False
        # All successfully parsed teacher replies in this episode, including turns
        # that failed the guidance gate or the adaptive gate. The visibility masks
        # control what the real student and re-test can see, not whether the teacher
        # may repeat itself. Keeping the whole episode also makes A-B-A a repeat.
        teacher_outputs: set[str] = set()
        previous_tutor_visible_output = ""
        previous_tutor_raw_outputs: tuple[str, ...] = ()
        previous_student_output = ""
        student_visible_previous_output = ""
        # Unlike previous_student_output, this never becomes a scripted gate
        # complaint. The gate judges the teacher against the student's latest
        # actual work even after one or more teacher-only gate failures.
        last_real_student_output = ""

        for turn_idx in range(1, self.max_turns + 1):
            tutor_state = TutorTurnState(
                task=task,
                ground_truth=ground_truth,
                public_history=public_history,
                previous_tutor_visible_output=previous_tutor_visible_output,
                turn_idx=turn_idx,
                max_turns=self.max_turns,
                teacher_pre_solve_result=teacher_pre_solve_result,
                student_reply_before_teacher=previous_student_output,
                previous_tutor_raw_outputs=previous_tutor_raw_outputs,
            )
            try:
                response, tutor_raw_output = await self._generate_tutor_response(
                    tutor_state,
                    actor_caller=actor_caller,
                    lora_version=episode_lora_version,
                )
            except ContextBudgetLimitExceeded as exc:
                logger.info(
                    "Terminating tutor episode at turn %s due to context budget: %s",
                    turn_idx,
                    exc,
                )
                termination_reason = CONTEXT_BUDGET_TERMINATION_REASON
                break
            (
                tutor_visible_output,
                teacher_ended,
                tutor_format_error,
            ) = self._parse_tutor_action(tutor_raw_output)
            public_before = list(public_history.turns)
            if teacher_ended:
                # END is a trainable policy action, but not a message.  Keep the
                # complete teacher state in the artifact and the filtered student
                # state as the re-test anchor; neither receives an additional turn.
                termination_reason = TEACHER_END_TERMINATION_REASON
                end_student_state = StudentTurnState(
                    task=task,
                    public_history=student_visible_history,
                    previous_student_output=student_visible_previous_output,
                    latest_tutor_visible_output="",
                    student_preference=selected_student.preference,
                )
                turn_artifacts.append(
                    TurnArtifact(
                        turn_idx=turn_idx,
                        tutor_state=tutor_state,
                        tutor_messages=list(self._build_tutor_messages(tutor_state)),
                        tutor_response=response,
                        tutor_raw_output=tutor_raw_output,
                        tutor_visible_output="",
                        guidance_gate_result=self._teacher_end_guidance_gate_result(),
                        public_history_before=public_before,
                        public_history_after=public_before,
                        teacher_ended=True,
                        student_state=end_student_state,
                    )
                )
                break
            if tutor_format_error:
                # Stop rather than write a blank turn into the student's and the
                # tutor's history; the turn keeps its format_error penalty.
                termination_reason = FORMAT_TERMINATION_REASON
                turn_artifacts.append(
                    TurnArtifact(
                        turn_idx=turn_idx,
                        tutor_state=tutor_state,
                        tutor_messages=list(self._build_tutor_messages(tutor_state)),
                        tutor_response=response,
                        tutor_raw_output=tutor_raw_output,
                        tutor_visible_output=tutor_visible_output,
                        guidance_gate_result=self._pending_guidance_gate_result(),
                        public_history_before=public_before,
                        public_history_after=public_before,
                        tutor_format_error=tutor_format_error,
                    )
                )
                break
            guidance_gate_result = self._pending_guidance_gate_result()
            if self.guidance_gate_mode == "masked_continue":
                guidance_gate_result = await self._run_guidance_gate(
                    task,
                    ground_truth,
                    tutor_visible_output,
                    aux_caller=aux_caller,
                )
            guidance_gate_masked = bool(
                self.guidance_gate_mode == "masked_continue"
                and guidance_gate_result.failed
            )

            student_state = StudentTurnState(
                task=task,
                public_history=student_visible_history,
                previous_student_output=student_visible_previous_output,
                latest_tutor_visible_output=tutor_visible_output,
                student_preference=selected_student.preference,
            )
            # The adaptive gate runs after the format parse and the guidance gate, so
            # an ended turn costs no auxiliary call, and before the student call,
            # since a failed gate means the student does not answer. A turn masked by
            # the guidance gate is already hidden from the student, so it is skipped.
            adaptive_gate_result = (
                None
                if guidance_gate_masked
                else await self._run_adaptive_gate(
                    selected_student.preference,
                    tutor_visible_output,
                    task=task,
                    turn_idx=turn_idx,
                    previous_student_message=last_real_student_output,
                    aux_caller=aux_caller,
                )
            )
            adaptive_gate_failed = (
                adaptive_gate_result is not None and not adaptive_gate_result.passed
            )

            normalized_teacher_output = normalize_exact_teacher_output(
                tutor_visible_output
            )
            teacher_exact_repeat = bool(
                normalized_teacher_output
                and normalized_teacher_output in teacher_outputs
            )
            if teacher_exact_repeat:
                # Keep the offending sample for its local penalty, but stop before
                # the student call and leave both student-visible histories at the
                # last completed round. The teacher's prompt already contains all
                # private gate feedback that preceded this turn.
                termination_reason = TEACHER_EXACT_REPEAT_TERMINATION_REASON
                turn_artifacts.append(
                    TurnArtifact(
                        turn_idx=turn_idx,
                        tutor_state=tutor_state,
                        tutor_messages=list(self._build_tutor_messages(tutor_state)),
                        tutor_response=response,
                        tutor_raw_output=tutor_raw_output,
                        tutor_visible_output=tutor_visible_output,
                        guidance_gate_result=guidance_gate_result,
                        public_history_before=public_before,
                        public_history_after=public_before,
                        tutor_format_error=tutor_format_error,
                        adaptive_gate_result=adaptive_gate_result,
                        adaptive_gate_failed=adaptive_gate_failed,
                        teacher_exact_repeat=True,
                    )
                )
                break

            if guidance_gate_masked:
                # This synthetic reply is a user turn in the teacher's
                # conversation. The real student is never called, and the
                # filtered student/re-test branch advances by neither message.
                student_answer_raw = GUIDANCE_GATE_STUDENT_REPLY
                student_error = None
            elif adaptive_gate_failed:
                # No student call: the complaint IS the teacher-visible student
                # turn. It gives the teacher feedback but never enters a later
                # real-student call or re-test.
                student_answer_raw = self._draw_adaptive_gate_complaint(
                    selected_student.preference, turn_idx=turn_idx
                )
                student_error = None
            else:
                student_answer_raw, student_error = await self._run_student(
                    student_state,
                    aux_caller=student_caller,
                )
            student_answer = _strip_reasoning_for_context(student_answer_raw)

            termination_reason = (
                "max_turns" if turn_idx == self.max_turns else "continue"
            )

            next_public_history = self._append_public_history_turn(
                old_public_history=public_history,
                tutor_visible_output=tutor_visible_output,
                current_student_answer=student_answer,
            )
            (
                next_student_visible_history,
                next_student_visible_output,
                student_history_filtered,
            ) = self._advance_student_visible_history(
                complete_history_after=next_public_history,
                student_visible_history=student_visible_history,
                previous_student_output=student_visible_previous_output,
                tutor_visible_output=tutor_visible_output,
                current_student_output=student_answer,
                adaptive_gate_failed=adaptive_gate_failed,
                guidance_gate_masked=guidance_gate_masked,
                history_already_filtered=student_history_filtered,
            )
            turn_artifacts.append(
                TurnArtifact(
                    turn_idx=turn_idx,
                    tutor_state=tutor_state,
                    tutor_messages=list(self._build_tutor_messages(tutor_state)),
                    tutor_response=response,
                    tutor_raw_output=tutor_raw_output,
                    tutor_visible_output=tutor_visible_output,
                    guidance_gate_result=guidance_gate_result,
                    guidance_gate_masked=guidance_gate_masked,
                    public_history_before=public_before,
                    public_history_after=list(next_public_history.turns),
                    tutor_format_error=tutor_format_error,
                    student_state=student_state,
                    student_output=student_answer,
                    student_error=student_error,
                    adaptive_gate_result=adaptive_gate_result,
                    adaptive_gate_failed=adaptive_gate_failed,
                    teacher_exact_repeat=teacher_exact_repeat,
                )
            )

            if normalized_teacher_output:
                teacher_outputs.add(normalized_teacher_output)

            public_history = next_public_history
            student_visible_history = next_student_visible_history
            previous_tutor_raw_outputs = (
                *previous_tutor_raw_outputs,
                tutor_raw_output if not tutor_format_error else "",
            )
            previous_tutor_visible_output = tutor_visible_output
            previous_student_output = student_answer
            student_visible_previous_output = next_student_visible_output
            if not adaptive_gate_failed and not guidance_gate_masked:
                last_real_student_output = student_answer

        guidance_gate_fail_count = sum(
            1 for artifact in turn_artifacts if artifact.guidance_gate_result.failed
        )
        episode_artifact = EpisodeArtifact(
            task=task,
            ground_truth=ground_truth,
            turns=turn_artifacts,
            termination_reason=termination_reason,
            guidance_gate_fail_count=guidance_gate_fail_count,
            latest_student_answer=previous_student_output,
            teacher_pre_solve_result=teacher_pre_solve_result,
            student_name=selected_student.name,
            student_model=selected_student.model,
        )
        # Passed explicitly rather than stored on the instance, since one workflow
        # serves every concurrent episode; the value is cached per problem.
        episode_no_teaching_baseline = await self._no_teaching_baseline(
            data,
            aux_caller=student_caller,
            answer_judge_caller=answer_judge_caller,
        )
        student_generalization_results = await self._run_student_generalization(
            episode_artifact,
            aux_caller=student_caller,
            answer_judge_caller=answer_judge_caller,
            no_teaching_baseline=episode_no_teaching_baseline,
        )
        reward_computer = EpisodeRewardComputer(
            guidance_gate_fail_penalty=self.guidance_gate_fail_penalty,
            format_error_penalty=self.format_error_penalty,
            teacher_exact_repeat_penalty=self.teacher_exact_repeat_penalty,
            adaptive_gate_fail_penalty=self.adaptive_gate_fail_penalty,
            soft_overlong_enabled=self.soft_overlong_enabled,
            soft_overlong_max_tokens=self.max_completion_tokens,
            soft_overlong_buffer_tokens=self.soft_overlong_buffer_tokens,
            soft_overlong_max_penalty=self.soft_overlong_max_penalty,
        )
        assignments = await reward_computer.compute(episode_artifact)
        self._apply_student_generalization_rewards(
            turn_artifacts, assignments, student_generalization_results
        )
        traces = [
            artifact_to_trace(artifact, assignment)
            for artifact, assignment in zip(turn_artifacts, assignments, strict=True)
        ]
        history = [
            trace_to_history_record(
                trace,
                artifact.guidance_gate_result
                if artifact.guidance_gate_result.failed
                else None,
                artifact.student_error,
            )
            for artifact, trace in zip(turn_artifacts, traces, strict=True)
        ]
        # With mask_rejected_turns the re-test reward is credited only to turns
        # that passed both gates; the actor splits the
        # return into that masked part and the rest.
        results = [
            response_to_tensordict(
                artifact.tutor_response,
                reward=assignment.reward,
                local_reward=assignment.local_reward,
                gate_masked_reward=(
                    assignment.reward_components.get("student_generalize_original", 0.0)
                    if self.mask_rejected_turns
                    else None
                ),
                gate_credit_mask=(
                    not (artifact.adaptive_gate_failed or artifact.guidance_gate_masked)
                    if self.mask_rejected_turns
                    else None
                ),
                trajectory_id=trajectory_id,
                turn_idx=artifact.turn_idx,
            )
            for artifact, assignment in zip(turn_artifacts, assignments, strict=True)
        ]
        total_reward = float(sum(assignment.reward for assignment in assignments))
        self.last_history = history
        self.last_traces = traces
        self.last_student_generalization_results = student_generalization_results
        completed_repeat_outcome = self._log_rollout_stats(
            total_reward=total_reward,
            traces=traces,
            termination_reason=episode_artifact.termination_reason,
            guidance_gate_fail_count=episode_artifact.guidance_gate_fail_count,
            student_generalization_results=student_generalization_results,
            teacher_pre_solve_result=episode_artifact.teacher_pre_solve_result,
            teacher_pre_cache_hit=teacher_pre_cache_hit,
            student_name=selected_student.name,
            student_call_failed=any(
                artifact.student_error for artifact in turn_artifacts
            ),
            no_teaching_baseline=episode_no_teaching_baseline,
        )
        if completed_repeat_outcome is not None and self.debug_trace_dir:
            await self._dump_eval_repeat_outcomes(*completed_repeat_outcome)
        await self._maybe_dump_debug_trace(
            trajectory_id=trajectory_id,
            task=episode_artifact.task,
            ground_truth=episode_artifact.ground_truth,
            latest_student_answer=episode_artifact.latest_student_answer,
            total_reward=total_reward,
            traces=traces,
            termination_reason=episode_artifact.termination_reason,
            guidance_gate_fail_count=episode_artifact.guidance_gate_fail_count,
            student_generalization_results=student_generalization_results,
            teacher_pre_solve_result=episode_artifact.teacher_pre_solve_result,
            student_name=selected_student.name,
            student_model=selected_student.model,
        )
        # Format-terminated episodes are trained: the offending turn carries its
        # penalty and the re-test scores the prefix through the last completed round.
        return concat_padded_tensors(results) if results else None

    def _make_engine_chat_caller(
        self,
        engine: Any,
        *,
        enable_thinking: bool,
    ) -> AReaLEngineChatCaller:
        return AReaLEngineChatCaller(
            engine=engine,
            tokenizer=self.tokenizer,
            enable_thinking=enable_thinking,
        )

    def _make_actor_caller(
        self,
        engine: Any | None = None,
        external_client: Any | None = None,
        chat_caller: AReaLEngineChatCaller | None = None,
    ) -> AReaLEngineActorCaller | ExternalActorCaller:
        if chat_caller is None and engine is not None:
            chat_caller = self._make_engine_chat_caller(
                engine,
                enable_thinking=self.enable_thinking,
            )
        if chat_caller is not None:
            return AReaLEngineActorCaller(
                chat_caller=chat_caller,
                gconfig=self._generation_config(),
                max_completion_tokens=self.max_completion_tokens,
                max_train_sample_tokens=self.max_train_sample_tokens,
            )
        if external_client is None:
            raise ValueError("Exactly one tutor generation source must be provided.")
        return ExternalActorCaller(
            client=external_client,
            tokenizer=self.tokenizer,
            context_budget=self.teacher_context_budget,
            temperature=self.temperature,
            top_p=self.top_p,
            enable_thinking=self.enable_thinking,
            max_completion_tokens=self.max_completion_tokens,
            max_train_sample_tokens=self.max_train_sample_tokens,
        )

    def _make_auxiliary_caller(
        self,
        engine: Any | None = None,
        chat_caller: AReaLEngineChatCaller | None = None,
    ) -> ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller:
        if self.aux_mode == "api":
            if self.aux_caller is None:
                raise RuntimeError("API auxiliary caller is not initialized.")
            return self.aux_caller
        if chat_caller is None and engine is not None:
            chat_caller = self._make_engine_chat_caller(
                engine,
                enable_thinking=self.aux_enable_thinking,
            )
        if chat_caller is None:
            raise RuntimeError(
                "auxiliary_model.mode='self' requires an AReaL inference engine. "
                "Use arun_episode(engine, data) or switch auxiliary_model.mode to 'api'."
            )
        return AReaLEngineAuxiliaryCaller(
            chat_caller=chat_caller,
            base_gconfig=self.gconfig,
            max_completion_tokens=self.aux_max_tokens,
            temperature=self.aux_temperature,
            top_p=self.aux_top_p,
            max_concurrency=self.max_concurrent_aux_calls,
            context_length=self.teacher_context_budget.context_length,
            context_window_margin=self.context_window_margin,
            semaphore=self._self_aux_semaphore,
        )

    def _make_answer_judge_caller(
        self,
        *,
        chat_caller: AReaLEngineChatCaller | None = None,
    ) -> ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None:
        if not self.answer_judge_enabled:
            return None
        if self.aux_mode == "api":
            config = AuxModelConfig(
                base_url=self.aux_base_url,
                model=self.aux_model,
                api_key=self.aux_api_key,
                timeout=self.aux_timeout,
                max_tokens=self.answer_judge_max_tokens,
                temperature=0.0,
                top_p=self.aux_top_p,
                max_concurrency=self.max_concurrent_aux_calls,
                request_params=self.aux_request_params,
                tokenizer_path=self.tokenizer_path,
                context_length=self.model_context_length,
                context_window_margin=self.context_window_margin,
            )
            return ApiAuxiliaryCaller(AsyncLLMCaller(config))
        if chat_caller is None:
            raise RuntimeError(
                "answer_judge with auxiliary_model.mode='self' requires an "
                "AReaL inference engine."
            )
        return AReaLEngineAuxiliaryCaller(
            chat_caller=chat_caller,
            base_gconfig=self.gconfig,
            max_completion_tokens=self.answer_judge_max_tokens,
            temperature=0.0,
            top_p=self.aux_top_p,
            max_concurrency=self.max_concurrent_aux_calls,
            context_length=self.teacher_context_budget.context_length,
            context_window_margin=self.context_window_margin,
            semaphore=self._self_aux_semaphore,
        )

    def _teacher_pre_solve_tokens(self) -> int:
        if self.teacher_pre_max_tokens > 0:
            return self.teacher_pre_max_tokens
        return self.max_completion_tokens

    def _build_teacher_pre_solve_messages(self, *, task: str) -> list[dict[str, str]]:
        """Context the pre-solve is generated in.

        These are the first two messages of the teacher context the draft is later
        placed in.
        """
        return [
            {
                "role": "system",
                "content": self._free_chat_teacher_system(task),
            },
            {"role": "user", "content": FREE_CHAT_TEACHER_SOLVE_PROMPT},
        ]

    async def _teacher_pre_solve_for_group(
        self,
        task: str,
        ground_truth: str,
        *,
        actor_caller: AReaLEngineActorCaller | ExternalActorCaller,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
        lora_version: int | None,
        group_key: str,
    ) -> tuple[TeacherPreSolveResult, bool]:
        """The draft for this problem at this weight version, shared by its group.

        The draft is part of the teacher's prompt, so sharing it gives every rollout
        of the GRPO group a common prompt and costs one generation per problem. A
        None `lora_version` (no local engine) generates a draft per episode.

        The key carries the weight version and versions older than the previous one
        are dropped, so a draft is never reused across a weight update.

        Returns the result and whether this rollout reused another's draft, which is
        what `teacher_pre/cache_hit` reports.
        """

        def run() -> Any:
            return self._run_teacher_pre_solve(
                task,
                ground_truth,
                actor_caller=actor_caller,
                answer_judge_caller=answer_judge_caller,
                lora_version=lora_version,
            )

        if lora_version is None:
            return (await run(), False)

        version = int(lora_version)
        key = (str(group_key), version)
        # The lock guards the dict only; the task is awaited outside it so one
        # problem's generation does not serialise the rest.
        async with self._teacher_pre_solve_shared_lock:
            pending = self._teacher_pre_solve_shared.get(key)
            cache_hit = pending is not None
            if pending is None:
                pending = asyncio.ensure_future(run())
                self._teacher_pre_solve_shared[key] = pending
                for stale in [
                    cached
                    for cached in self._teacher_pre_solve_shared
                    if cached[1] < version - 1
                ]:
                    self._teacher_pre_solve_shared.pop(stale, None)
        try:
            # shield keeps one awaiting rollout's cancellation from cancelling the
            # draft the rest of the group is waiting on.
            result = await asyncio.shield(pending)
        except Exception:
            # Do not let one transient failure poison the whole group for this step
            # -- drop the entry so the next sibling retries.
            async with self._teacher_pre_solve_shared_lock:
                if self._teacher_pre_solve_shared.get(key) is pending:
                    del self._teacher_pre_solve_shared[key]
            raise
        return (result, cache_hit)

    async def _run_teacher_pre_solve(
        self,
        task: str,
        ground_truth: str,
        *,
        actor_caller: AReaLEngineActorCaller | ExternalActorCaller,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
        lora_version: int | None,
    ) -> TeacherPreSolveResult:
        attempts: list[TeacherPreSolveAttempt] = []
        messages = self._build_teacher_pre_solve_messages(task=task)
        max_completion_tokens = self._teacher_pre_solve_tokens()
        verification_enabled = self.teacher_pre_verify
        attempt_count = self.teacher_pre_attempts if verification_enabled else 1
        for attempt_idx in range(1, attempt_count + 1):
            try:
                result = await actor_caller.generate(
                    messages,
                    lora_version=lora_version,
                    rid_prefix=f"teacher-pre-{attempt_idx}",
                    max_completion_tokens=max_completion_tokens,
                )
            except Exception as exc:
                attempts.append(
                    TeacherPreSolveAttempt(
                        attempt=attempt_idx,
                        raw_output="",
                        error=str(exc),
                        accepted=False,
                        judge_result=None,
                    )
                )
                continue

            raw_output = str(result.raw_text or "").strip()
            if not verification_enabled:
                attempts.append(
                    TeacherPreSolveAttempt(
                        attempt=attempt_idx,
                        raw_output=raw_output,
                        error=None,
                        accepted=True,
                        judge_result=None,
                    )
                )
                return TeacherPreSolveResult(
                    accepted=True,
                    attempts=attempts,
                    raw_output=raw_output,
                    error=None,
                    verification_enabled=False,
                )
            judge_result = await self._score_answer_async(
                task,
                ground_truth,
                raw_output,
                answer_judge_caller=answer_judge_caller,
            )
            accepted = bool(judge_result.correct)
            attempts.append(
                TeacherPreSolveAttempt(
                    attempt=attempt_idx,
                    raw_output=raw_output,
                    error=None,
                    accepted=accepted,
                    judge_result=judge_result,
                )
            )
            if accepted:
                return TeacherPreSolveResult(
                    accepted=True,
                    attempts=attempts,
                    raw_output=raw_output,
                    error=None,
                    verification_enabled=True,
                )

        return TeacherPreSolveResult(
            accepted=False,
            attempts=attempts,
            raw_output="",
            error=(
                f"no correct teacher pre-solve after {attempt_count} attempts"
                if verification_enabled
                else "teacher pre-solve draft generation failed"
            ),
            verification_enabled=verification_enabled,
        )

    async def _generate_tutor_response(
        self,
        tutor_state: TutorTurnState,
        *,
        actor_caller: AReaLEngineActorCaller | ExternalActorCaller,
        lora_version: int | None = None,
        rid_prefix: str = "tutor",
    ) -> tuple[ModelResponse, str]:
        messages = self._build_tutor_messages(tutor_state)
        # A reply that stops on 'length' closed no tags and is a format error, so
        # it is resampled; the last attempt is kept if every attempt hits the limit.
        attempts = 1
        if self.length_retry_enabled:
            attempts = self.length_retry_attempts
        for attempt in range(1, attempts + 1):
            result = await actor_caller.generate(
                messages,
                lora_version=lora_version,
                # Attempt 1 keeps the base rid; retries add a -len<k> suffix.
                rid_prefix=(
                    f"{rid_prefix}-{tutor_state.turn_idx}"
                    if attempt == 1
                    else f"{rid_prefix}-{tutor_state.turn_idx}-len{attempt - 1}"
                ),
            )
            if getattr(result.response, "stop_reason", None) != "length":
                break
        if attempts > 1:
            # Only reported when the feature is on, so the series never carries
            # zeros that mean "not measured".
            _safe_scalar(
                **{
                    "length_retry/attempts": float(attempt),
                    "length_retry/retried": float(attempt > 1),
                    "length_retry/exhausted": float(
                        getattr(result.response, "stop_reason", None) == "length"
                    ),
                }
            )
        return result.response, result.raw_text

    async def _run_student(
        self,
        state: StudentTurnState,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> tuple[str, str | None]:
        """One student turn. Returns (visible_turn_text, error)."""
        messages = self._build_student_messages(state)
        rid = f"student-{state.public_history.turn_count}"
        result = await self._call_auxiliary_messages(
            messages,
            aux_caller=aux_caller,
            rid_prefix=rid,
        )
        if result.error:
            return "", result.error
        return result.text, None

    @staticmethod
    def _pending_guidance_gate_result() -> GuidanceGateResult:
        return GuidanceGateResult(
            raw_output="",
            failed=False,
            feedback=GUIDANCE_GATE_PENDING_FEEDBACK,
            parse_error=None,
            raw_result={"pending": True},
        )

    @staticmethod
    def _teacher_end_guidance_gate_result() -> GuidanceGateResult:
        return GuidanceGateResult(
            raw_output="",
            failed=False,
            feedback="Guidance gate skipped for teacher end action.",
            parse_error=None,
            raw_result={"teacher_end": True},
        )

    async def _run_guidance_gate(
        self,
        task: str,
        ground_truth: str,
        teacher_action: str,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> GuidanceGateResult:
        return await self._run_guidance_gate_judge(
            task,
            ground_truth,
            teacher_action,
            aux_caller=aux_caller,
        )

    async def _run_guidance_gate_judge(
        self,
        task: str,
        ground_truth: str,
        teacher_action: str,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> GuidanceGateResult:
        del task
        # The rollout passes an already-extracted student-visible action here.
        # Parsing it again would discard valid plain text in non-thinking mode.
        teacher_message = _strip_reasoning_for_context(teacher_action)
        prompt = render_prompt(
            GUIDANCE_GATE_USER_TEMPLATE,
            ground_truth=ground_truth,
            teacher_action=teacher_message,
        )
        result = await self._call_auxiliary_prompt(
            system_prompt=GUIDANCE_GATE_SYSTEM_PROMPT,
            user_prompt=prompt,
            aux_caller=aux_caller,
            rid_prefix="rawbase-leak-check",
        )
        if result.error:
            return GuidanceGateResult(
                raw_output="",
                failed=True,
                feedback=GUIDANCE_GATE_FAILED_FEEDBACK_TEMPLATE.format(
                    error=result.error
                ),
                parse_error=result.error,
                raw_result={"method": "rawbase_llm"},
            )
        guidance_gate_result = parse_guidance_gate_result(result.text)
        guidance_gate_result.raw_result["method"] = "rawbase_llm"
        return guidance_gate_result

    async def _call_auxiliary_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
        rid_prefix: str = "auxiliary",
    ) -> TextCallResult:
        return await self._call_auxiliary_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            aux_caller=aux_caller,
            rid_prefix=rid_prefix,
        )

    async def _call_auxiliary_messages(
        self,
        messages: list[dict[str, str]],
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
        rid_prefix: str = "auxiliary",
    ) -> TextCallResult:
        caller = aux_caller or self._make_auxiliary_caller(engine=None)
        return await caller.call_text(messages, rid_prefix=rid_prefix)

    def _append_public_history_turn(
        self,
        *,
        old_public_history: PublicHistoryState,
        tutor_visible_output: str,
        current_student_answer: str,
    ) -> PublicHistoryState:
        entries = []
        existing_history = old_public_history.summary.strip()
        if existing_history:
            entries.append(existing_history)

        turn_idx = old_public_history.turn_count + 1
        entries.append(
            self._format_public_history_entry(
                "Tutor",
                turn_idx,
                tutor_visible_output,
            )
        )
        entries.append(
            self._format_public_history_entry(
                "Student",
                turn_idx,
                current_student_answer,
            )
        )
        turns = list(old_public_history.turns)
        turns.append({"role": "teacher", "content": tutor_visible_output})
        turns.append({"role": "student", "content": current_student_answer})
        return PublicHistoryState(
            summary="\n\n".join(entry for entry in entries if entry),
            turn_count=old_public_history.turn_count + 1,
            turns=turns,
        )

    def _advance_student_visible_history(
        self,
        *,
        complete_history_after: PublicHistoryState,
        student_visible_history: PublicHistoryState,
        previous_student_output: str,
        tutor_visible_output: str,
        current_student_output: str,
        adaptive_gate_failed: bool,
        history_already_filtered: bool,
        guidance_gate_masked: bool = False,
    ) -> tuple[PublicHistoryState, str, bool]:
        """Advance the real student's branch without changing teacher history."""

        if guidance_gate_masked or adaptive_gate_failed:
            return student_visible_history, previous_student_output, True
        if not history_already_filtered:
            return complete_history_after, current_student_output, False
        return (
            self._append_public_history_turn(
                old_public_history=student_visible_history,
                tutor_visible_output=tutor_visible_output,
                current_student_answer=current_student_output,
            ),
            current_student_output,
            True,
        )

    def _build_student_probe_messages(
        self,
        *,
        anchor: StudentGeneralizationAnchor,
        task: str,
    ) -> list[dict[str, str]]:
        """The re-test continues the tutoring chat: the same dialogue the student
        saw, then one new user turn asking it to solve the task from scratch.

        Each probe is built fresh from the anchor and never written back into
        `turns`, so replays are independent branches.
        """
        # Same system prompt as the conversation and same re-test template as
        # `_no_teaching_baseline`, whose score is subtracted from this probe's.
        system, free_chat_retest_template = self._free_chat_student_prompts()
        final_turn = render_prompt(free_chat_retest_template, task=task)
        return [
            {"role": "system", "content": system},
            *self._render_conversation(
                list(anchor.public_history.turns), speaker="student"
            ),
            {"role": "user", "content": final_turn},
        ]

    def _format_public_history_entry(
        self, speaker: str, round_idx: int, text: str
    ) -> str:
        visible_text = _strip_reasoning_for_context(text)
        return PUBLIC_HISTORY_ENTRY_TEMPLATE.format(
            speaker=speaker,
            round_idx=round_idx,
            visible_text=visible_text,
        )

    def _score_answer(
        self, task: str, ground_truth: str, student_answer: str
    ) -> JudgeResult:
        return score_math_answer(task, ground_truth, student_answer)

    async def _score_answer_async(
        self,
        task: str,
        ground_truth: str,
        student_answer: str,
        *,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
    ) -> JudgeResult:
        exact_result = self._score_answer(task, ground_truth, student_answer)
        if exact_result.correct or not self.answer_judge_enabled:
            return exact_result

        extracted_answer = str(exact_result.raw_result.get("extracted_answer") or "")
        cache_key = (str(task), str(ground_truth), extracted_answer)
        cached = self._answer_judge_cache.get(cache_key)
        if cached is not None:
            return cached

        if answer_judge_caller is None:
            result = self._answer_judge_failed_result(
                exact_result,
                error="answer judge caller is unavailable",
            )
            self._answer_judge_cache[cache_key] = result
            return result

        prompt = self._build_answer_judge_prompt(task, ground_truth, extracted_answer)
        judge_call = await self._call_auxiliary_prompt(
            system_prompt=ANSWER_JUDGE_SYSTEM_PROMPT,
            user_prompt=prompt,
            aux_caller=answer_judge_caller,
            rid_prefix="answer-judge",
        )
        if judge_call.error:
            result = self._answer_judge_failed_result(
                exact_result,
                error=judge_call.error,
            )
            self._answer_judge_cache[cache_key] = result
            return result

        result = self._parse_answer_judge_result(exact_result, judge_call)
        self._answer_judge_cache[cache_key] = result
        return result

    def _answer_judge_failed_result(
        self, exact_result: JudgeResult, *, error: str
    ) -> JudgeResult:
        raw_result = dict(exact_result.raw_result)
        raw_result["exact_match_correct"] = bool(exact_result.correct)
        raw_result["answer_judge"] = {
            "enabled": True,
            "used": False,
            "error": error,
        }
        return JudgeResult(
            raw_output=exact_result.raw_output,
            correct=exact_result.correct,
            feedback=exact_result.feedback,
            parse_error=exact_result.parse_error,
            raw_result=raw_result,
        )

    def _parse_answer_judge_result(
        self, exact_result: JudgeResult, judge_call: TextCallResult
    ) -> JudgeResult:
        parsed, parse_error = parse_json_dict(judge_call.text)
        if not isinstance(parsed, dict):
            return self._answer_judge_failed_result(
                exact_result,
                error=parse_error or "Expected JSON object from answer judge.",
            )

        correct = parsed.get("correct")
        if not isinstance(correct, bool):
            return self._answer_judge_failed_result(
                exact_result,
                error='Answer judge field "correct" must be a boolean.',
            )
        if parse_error:
            return self._answer_judge_failed_result(
                exact_result,
                error=parse_error,
            )

        raw_result = dict(exact_result.raw_result)
        raw_result["exact_match_correct"] = bool(exact_result.correct)
        raw_result["answer_judge"] = {
            "enabled": True,
            "used": True,
            "correct": correct,
            "raw_output": judge_call.raw_text or judge_call.text,
            "raw_result": parsed,
        }
        return JudgeResult(
            raw_output=judge_call.raw_text or judge_call.text,
            correct=correct,
            feedback=exact_result.feedback,
            parse_error=None,
            raw_result=raw_result,
        )

    def _build_answer_judge_prompt(
        self, task: str, ground_truth: str, extracted_answer: str
    ) -> str:
        return render_prompt(
            ANSWER_JUDGE_USER_TEMPLATE,
            task=task,
            ground_truth=ground_truth,
            extracted_answer=extracted_answer,
        )

    def _student_visible_history_after(
        self, artifact: TurnArtifact
    ) -> PublicHistoryState:
        """Return the transcript a real student retains after this turn.

        The stored public history is always complete because it is the teacher's
        state; this projection removes hidden exchanges. It is built from
        ``student_state``, which already holds the filtered prefix the real student
        call used.
        """

        before = artifact.student_state.public_history
        if artifact.teacher_ended or self._turn_hidden_from_student(artifact):
            # END contributes no teacher or student message, and a hidden turn
            # never reached the student: either way the re-test runs on the
            # filtered prefix the student last saw.
            return PublicHistoryState(
                summary=before.summary,
                turn_count=before.turn_count,
                turns=list(before.turns),
            )
        return self._append_public_history_turn(
            old_public_history=before,
            tutor_visible_output=artifact.tutor_visible_output,
            current_student_answer=artifact.student_output,
        )

    def _turn_generalization_anchor(
        self, artifact: TurnArtifact
    ) -> StudentGeneralizationAnchor:
        return StudentGeneralizationAnchor(
            public_history=self._student_visible_history_after(artifact),
            reward_turn_idx=int(artifact.turn_idx),
        )

    def _empty_generalization_anchor(
        self, episode_artifact: EpisodeArtifact
    ) -> StudentGeneralizationAnchor | None:
        """Re-test with no conversation at all: the student solves it alone.

        None only when the episode produced no turn, leaving nothing to attach a
        reward to.
        """
        if not episode_artifact.turns:
            return None
        return StudentGeneralizationAnchor(
            public_history=PublicHistoryState(summary="", turn_count=0, turns=[]),
            reward_turn_idx=int(episode_artifact.turns[-1].turn_idx),
        )

    def _student_generalization_anchor(
        self, episode_artifact: EpisodeArtifact
    ) -> StudentGeneralizationAnchor | None:
        for artifact in reversed(episode_artifact.turns):
            if artifact.student_state is not None:
                return self._turn_generalization_anchor(artifact)

        # No completed round (the first teacher turn was malformed): re-test on the
        # empty transcript rather than scoring 0, crediting the only turn.
        return self._empty_generalization_anchor(episode_artifact)

    async def _run_student_generalization(
        self,
        episode_artifact: EpisodeArtifact,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
        no_teaching_baseline: float | None = None,
    ) -> list[StudentGeneralizationResult]:
        """The re-test: after the chat, the student solves the task on its own.

        Replayed `retest_replays` times. The reward is the fraction correct minus
        the no-teaching baseline, credited to the last turn the student saw.
        """
        level = ORIGINAL_RETEST_LEVEL
        anchor = self._student_generalization_anchor(episode_artifact)
        if anchor is None:
            return [
                StudentGeneralizationResult(
                    level=level,
                    skipped=True,
                    skip_reason="base_not_solved",
                )
            ]

        probe_messages = self._build_student_probe_messages(
            anchor=anchor,
            task=episode_artifact.task,
        )
        replay_results = await asyncio.gather(
            *[
                self._call_auxiliary_messages(
                    probe_messages,
                    aux_caller=aux_caller,
                    rid_prefix=(
                        f"student-transfer-{level}-"
                        f"{anchor.public_history.turn_count}-r{replay_idx}"
                    ),
                )
                for replay_idx in range(self.retest_replays)
            ]
        )
        # The first successful replay is canonical for logging; replay 1 is used
        # only when every replay failed.
        student_result = next(
            (result for result in replay_results if not result.error),
            replay_results[0],
        )
        if student_result.error:
            student_output_raw = ""
            student_error = student_result.error
        else:
            student_output_raw = student_result.text
            student_error = None
        student_output = _strip_reasoning_for_context(student_output_raw)
        judge_result = None
        reward = 0.0
        replay_correct = 0
        replay_scored = 0
        if student_error is None:
            judge_results = await asyncio.gather(
                *[
                    self._score_answer_async(
                        episode_artifact.task,
                        episode_artifact.ground_truth,
                        result.text,
                        answer_judge_caller=answer_judge_caller,
                    )
                    for result in replay_results
                    if not result.error
                ]
            )
            judge_result = judge_results[0]
            replay_scored = len(judge_results)
            replay_correct = sum(1 for judged in judge_results if judged.correct)
            reward = replay_correct / max(replay_scored, 1)
            if no_teaching_baseline is not None:
                # Gain over the no-teaching baseline; negative when the conversation
                # leaves the student worse off than the bare problem statement.
                reward = reward - no_teaching_baseline
        return [
            StudentGeneralizationResult(
                level=level,
                task=episode_artifact.task,
                ground_truth=episode_artifact.ground_truth,
                attempted=True,
                student_output=student_output,
                student_error=student_error,
                judge_result=judge_result,
                replay_count=replay_scored,
                replay_correct=replay_correct,
                reward=reward,
                public_history=anchor.public_history.summary,
                reward_turn_idx=anchor.reward_turn_idx,
            )
        ]

    def _apply_student_generalization_rewards(
        self,
        turn_artifacts: list[TurnArtifact],
        assignments: list[Any],
        student_generalization_results: list[StudentGeneralizationResult],
    ) -> None:
        assignment_by_turn_idx = {
            int(artifact.turn_idx): assignment
            for artifact, assignment in zip(turn_artifacts, assignments, strict=True)
        }
        for result in student_generalization_results:
            if not result.reward or result.reward_turn_idx is None:
                continue
            assignment = assignment_by_turn_idx.get(int(result.reward_turn_idx))
            if assignment is None:
                continue
            key = f"student_generalize_{result.level}"
            assignment.reward_components[key] = assignment.reward_components.get(
                key, 0.0
            ) + float(result.reward)
            assignment.reward += float(result.reward)

    def _log_generalize_stats(
        self,
        *,
        solved: float,
        student_generalization_results: list[StudentGeneralizationResult] | None,
    ) -> None:
        """``solved`` is the episode outcome score: the re-test fraction."""

        try:
            is_eval = bool(getattr(workflow_context.get(), "is_eval", False))
        except Exception:
            is_eval = False

        metrics: dict[str, float] = {}
        if is_eval:
            metrics["teacher_success"] = float(solved)

        results = student_generalization_results or []
        level = ORIGINAL_RETEST_LEVEL
        level_result = next(
            (result for result in results if result.level == level), None
        )
        attempted = bool(level_result is not None and level_result.attempted)
        correct = bool(
            attempted
            and level_result is not None
            and level_result.judge_result is not None
            and level_result.judge_result.correct
        )
        # The headline series is the replay fraction correct; the binary series
        # tracks attempt 1 only.
        replay_count = int(getattr(level_result, "replay_count", 0) or 0)
        replay_correct = int(getattr(level_result, "replay_correct", 0) or 0)
        score = replay_correct / replay_count if replay_count > 0 else float(correct)
        metrics[f"student_{level}_attempted"] = float(attempted)
        metrics[f"student_{level}_success"] = float(score)
        metrics[f"student_{level}_success_binary"] = float(correct)
        if attempted:
            metrics[f"student_{level}_correct_given_attempted"] = float(score)
            metrics[f"student_{level}_correct_given_attempted_binary"] = float(correct)
            if replay_count > 0:
                metrics[f"student_{level}_replay_count"] = float(replay_count)
        elif level_result is not None and level_result.skipped:
            metrics[f"student_{level}_skipped"] = 1.0

        _safe_generalize_scalar(**metrics)

    @staticmethod
    def _student_generalization_result_to_json(
        result: StudentGeneralizationResult,
    ) -> dict[str, Any]:
        judge = result.judge_result
        return {
            "level": result.level,
            "task": result.task,
            "ground_truth": result.ground_truth,
            "attempted": bool(result.attempted),
            "skipped": bool(result.skipped),
            "skip_reason": result.skip_reason,
            "student_output": result.student_output,
            "student_error": result.student_error,
            "judge_correct": bool(judge.correct) if judge is not None else False,
            "judge_feedback": judge.feedback if judge is not None else "",
            # judge_correct is attempt 1 only; these are the whole replay set the
            # reward is actually computed from.
            "replay_count": int(result.replay_count),
            "replay_correct": int(result.replay_correct),
            "reward": float(result.reward),
            "reward_turn_idx": result.reward_turn_idx,
            "public_history": result.public_history,
        }

    @staticmethod
    def _teacher_pre_solve_result_to_json(
        result: TeacherPreSolveResult,
    ) -> dict[str, Any]:
        return {
            "verification_enabled": bool(result.verification_enabled),
            "accepted": bool(result.accepted),
            "raw_output": result.raw_output,
            "error": result.error,
            "attempt_count": len(result.attempts),
            "attempts": [
                {
                    "attempt": attempt.attempt,
                    "raw_output": attempt.raw_output,
                    "error": attempt.error,
                    "accepted": bool(attempt.accepted),
                    "judge_correct": (
                        bool(attempt.judge_result.correct)
                        if attempt.judge_result is not None
                        else False
                    ),
                    "judge_feedback": (
                        attempt.judge_result.feedback
                        if attempt.judge_result is not None
                        else ""
                    ),
                    "judge_raw_result": (
                        dict(attempt.judge_result.raw_result)
                        if attempt.judge_result is not None
                        else {}
                    ),
                }
                for attempt in result.attempts
            ],
        }

    @staticmethod
    def _render_conversation(
        turns: list[dict[str, str]],
        *,
        speaker: str,
        own_turn_template: str | None = None,
    ) -> list[dict[str, str]]:
        """Shared dialogue seen from one side: own turns assistant, other user.

        ``own_turn_template`` re-wraps the speaker's OWN turns, and is how the
        teacher gets the shape of its replies back after `public_history`
        stripped the tags off them. It takes a single ``{visible}`` field. Only
        the teacher passes it. The student's view stays plain text because its
        state contains only the public transcript.
        """
        rendered = []
        for turn in turns:
            is_own = turn["role"] == speaker
            content = turn["content"]
            if is_own and own_turn_template is not None:
                content = own_turn_template.format(visible=content)
            rendered.append(
                {"role": "assistant" if is_own else "user", "content": content}
            )
        return rendered

    def _free_chat_teacher_system(self, task: str) -> str:
        """Teacher system prompt for the free-chat rollout: the setting only.

        Budget, task, and what the student will be tested on; the student is not
        given the task until the re-test. Reply directives (format contract and
        guidance instruction) live in FREE_CHAT_TEACHER_OPEN_PROMPT, so this prompt is
        also the context the pre-solve is conditioned on.

        Takes the task rather than a TutorTurnState because the pre-solve runs
        before any turn state exists.
        """
        return render_prompt(
            FREE_CHAT_TEACHER_SYSTEM_PROMPT,
            budget=int(self.max_turns),
            task=task,
        )

    def _free_chat_open_prompt(self) -> str:
        """The user turn that starts the conversation and carries its directives."""
        parts = [FREE_CHAT_TEACHER_OPEN_PROMPT]
        parts.append(self._teacher_output_format_prompt())
        parts.append(TEACHER_GUIDANCE_INSTRUCTION)
        return "\n\n".join(parts)

    def _free_chat_preamble(self, tutor_state: TutorTurnState) -> list[dict[str, str]]:
        """Everything before the first teacher reply, after the system turn.

        Two messages when the pre-solve ran and was accepted -- the request and
        the draft that answered it -- then the turn that opens the conversation.
        With `teacher_pre.enabled` off the first two are absent and nothing else
        changes.
        """
        messages: list[dict[str, str]] = []
        pre_solve = tutor_state.teacher_pre_solve_result
        if self.teacher_pre_enabled and pre_solve is not None and pre_solve.accepted:
            raw_output = _strip_reasoning_for_context(
                str(pre_solve.raw_output or "")
            ).strip()
            if raw_output:
                messages.append(
                    {"role": "user", "content": FREE_CHAT_TEACHER_SOLVE_PROMPT}
                )
                messages.append({"role": "assistant", "content": raw_output})
        messages.append({"role": "user", "content": self._free_chat_open_prompt()})
        return messages

    def _build_tutor_messages(
        self, tutor_state: TutorTurnState
    ) -> list[dict[str, str]]:
        """Messages for one teacher turn.

        A preamble sits between the system turn and the conversation: the
        pre-solve exchange when there is one, then the turn that opens the
        conversation. It is a pure function of the state, so it is identical in the
        rollout prompt and in the training prompt, and it lands entirely on the
        prompt side of the split -- the draft is never trained on.
        """
        return [
            {
                "role": "system",
                "content": self._free_chat_teacher_system(tutor_state.task),
            },
            *self._free_chat_preamble(tutor_state),
            *self._render_conversation(
                tutor_state.public_history.turns,
                speaker="teacher",
                own_turn_template=(
                    "{visible}"
                    if self.teacher_response_format == "thinking"
                    else TEACHER_HISTORY_MASKED_TEMPLATE
                ),
            ),
        ]

    @staticmethod
    def _free_chat_student_prompts() -> tuple[str, str]:
        """The student's system prompt and re-test template.

        Shared by the dialogue turn, the re-test probe, and the no-teaching
        baseline; the baseline is subtracted from the probe's score, so both must
        use the same prompts.
        """
        return FREE_CHAT_STUDENT_SYSTEM_PROMPT, FREE_CHAT_STUDENT_RETEST_TEMPLATE

    def _build_student_messages(self, state: StudentTurnState) -> list[dict[str, str]]:
        # The student gets no task or instruction and learns about the problem only
        # from what the teacher says.
        system, _ = self._free_chat_student_prompts()
        turns = list(state.public_history.turns)
        latest_teacher_output = state.latest_tutor_visible_output.strip()
        if latest_teacher_output:
            turns.append({"role": "teacher", "content": latest_teacher_output})
        return [
            {"role": "system", "content": system},
            *self._render_conversation(turns, speaker="student"),
        ]

    def _generation_config(self):
        if self.gconfig is not None and hasattr(self.gconfig, "new"):
            return self.gconfig.new(
                n_samples=1,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_completion_tokens,
            )
        return self.gconfig

    async def _no_teaching_baseline(
        self,
        data: dict[str, Any],
        *,
        aux_caller: Any,
        answer_judge_caller: Any,
    ) -> float | None:
        """What this problem is worth with no teaching, shared within the group.

        The same messages the re-test uses, with an empty transcript: the student
        gets the task and nothing else. None when every student call failed, and
        the caller then leaves the reward alone rather than subtracting a
        fabricated zero.
        """
        task = str(data["task"])
        ground_truth = str(data["ground_truth"])
        key = str(data.get("id", task))
        cached = self._no_teaching_baselines.get(key)
        if cached is not None:
            return cached
        async with self._no_teaching_baseline_lock:
            # Re-check: another episode of the same group may have filled it in
            # while this one waited.
            cached = self._no_teaching_baselines.get(key)
            if cached is not None:
                return cached
            # The same two messages _build_student_probe_messages produces for an
            # empty transcript, since the baseline is subtracted from its score.
            system, retest_template = self._free_chat_student_prompts()
            final_turn = render_prompt(retest_template, task=task)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": final_turn},
            ]
            attempts = await asyncio.gather(
                *[
                    self._call_auxiliary_messages(
                        messages,
                        aux_caller=aux_caller,
                        rid_prefix=f"no-teaching-baseline-r{index}",
                    )
                    for index in range(self.retest_replays)
                ]
            )
            usable = [attempt for attempt in attempts if not attempt.error]
            if not usable:
                return None
            judged = await asyncio.gather(
                *[
                    self._score_answer_async(
                        task,
                        ground_truth,
                        attempt.text,
                        answer_judge_caller=answer_judge_caller,
                    )
                    for attempt in usable
                ]
            )
            baseline = sum(1 for item in judged if item.correct) / len(judged)
            self._no_teaching_baselines[key] = baseline
            return baseline

    @staticmethod
    def _free_chat_outcome_score(
        results: list[StudentGeneralizationResult] | None,
    ) -> float:
        """The episode's outcome: the solo re-test score.

        The fraction of re-test replays the student got right, the same quantity
        evaluation reports. A skipped re-test (the episode produced no turn)
        scores 0.
        """
        for result in results or []:
            if result.level != ORIGINAL_RETEST_LEVEL:
                continue
            if result.skipped or not result.attempted:
                return 0.0
            if result.replay_count > 0:
                return result.replay_correct / result.replay_count
            judge = result.judge_result
            return float(bool(judge is not None and judge.correct))
        return 0.0

    def _log_rollout_stats(
        self,
        *,
        total_reward: float,
        traces: list[TurnTrace],
        termination_reason: str,
        guidance_gate_fail_count: int,
        student_generalization_results: list[StudentGeneralizationResult] | None = None,
        teacher_pre_solve_result: TeacherPreSolveResult | None = None,
        teacher_pre_cache_hit: bool = False,
        student_name: str = "",
        student_call_failed: bool = False,
        no_teaching_baseline: float | None = None,
    ) -> tuple[int, list[float]] | None:
        # Nothing in the conversation is judged, so the re-test is the outcome.
        outcome_score = self._free_chat_outcome_score(student_generalization_results)
        is_eval = bool(workflow_context.get().is_eval)
        completed_repeat_outcome = None
        if is_eval:
            completed_repeat_outcome = self._record_eval_repeat_outcomes(
                final_correct=outcome_score,
            )
        metrics = {
            "reward": float(total_reward),
            "turns": len(traces),
            "guidance_gate_fails": int(guidance_gate_fail_count),
            "format_errors": sum(
                int(bool(trace.tutor_format_error)) for trace in traces
            ),
            "teacher_exact_repeats": sum(
                int(bool(trace.teacher_exact_repeat)) for trace in traces
            ),
            "teacher_ends": sum(int(bool(trace.teacher_ended)) for trace in traces),
            "solved": outcome_score,
            "final_correct": outcome_score,
            # final_correct ignores guidance gate failures; this variant scores them 0.
            "final_correct_guidance_gated": (
                0.0 if guidance_gate_fail_count else outcome_score
            ),
            "stop/max_turns": float(termination_reason == "max_turns"),
            "stop/context_limit": float(
                termination_reason == CONTEXT_BUDGET_TERMINATION_REASON
            ),
            "stop/format_error": float(termination_reason == FORMAT_TERMINATION_REASON),
            "stop/teacher_exact_repeat": float(
                termination_reason == TEACHER_EXACT_REPEAT_TERMINATION_REASON
            ),
            "stop/teacher_end": float(
                termination_reason == TEACHER_END_TERMINATION_REASON
            ),
            "stop/teacher_end_turn": float(
                next((trace.turn_idx for trace in traces if trace.teacher_ended), 0)
            ),
            "stop/teacher_pre_skipped": float(
                termination_reason == TEACHER_PRE_SKIPPED_TERMINATION_REASON
            ),
        }
        metrics.update(self._adaptive_gate_metrics(traces, student_name=student_name))
        if teacher_pre_solve_result is not None:
            metrics["teacher_pre/verification_enabled"] = float(
                teacher_pre_solve_result.verification_enabled
            )
            metrics["teacher_pre/accepted"] = float(teacher_pre_solve_result.accepted)
            metrics["teacher_pre/attempts"] = float(
                len(teacher_pre_solve_result.attempts)
            )
            # Whether this rollout reused its group's draft. On a hit, `attempts`
            # above is the shared draft's attempt count, not this rollout's calls.
            metrics["teacher_pre/cache_hit"] = float(bool(teacher_pre_cache_hit))

        baseline = no_teaching_baseline
        if baseline is not None:
            metrics["retest/no_teaching_baseline"] = float(baseline)
            # The headline stays the raw fraction; this is the gain over a
            # student that was handed the problem and no conversation.
            metrics["retest/improvement"] = float(outcome_score - baseline)
            metrics["retest/improved"] = float(outcome_score > baseline)
            metrics["retest/made_it_worse"] = float(outcome_score < baseline)

        for configured_name in self.student_model_runtimes:
            metric_name = self._student_metric_name(configured_name)
            metrics[f"student/{metric_name}/selected"] = float(
                configured_name == student_name
            )
        if student_name:
            metric_name = self._student_metric_name(student_name)
            prefix = f"student/{metric_name}"
            metrics[f"{prefix}/solved"] = outcome_score
            metrics[f"{prefix}/final_correct"] = outcome_score
            metrics[f"{prefix}/reward"] = float(total_reward)
            metrics[f"{prefix}/turns"] = float(len(traces))
            metrics[f"{prefix}/call_failed"] = float(student_call_failed)
            if baseline is not None:
                metrics[f"{prefix}/retest/no_teaching_baseline"] = float(baseline)
                metrics[f"{prefix}/retest/improvement"] = float(
                    outcome_score - baseline
                )
                metrics[f"{prefix}/retest/improved"] = float(outcome_score > baseline)
                metrics[f"{prefix}/retest/made_it_worse"] = float(
                    outcome_score < baseline
                )

        metrics.update(self._reward_component_metrics(traces))
        self._log_generalize_stats(
            solved=outcome_score,
            student_generalization_results=student_generalization_results,
        )
        _safe_scalar(**metrics)
        return completed_repeat_outcome

    def _record_eval_repeat_outcomes(
        self, *, final_correct: float
    ) -> tuple[int, list[float]] | None:
        task_id = getattr(workflow_context.get(), "task_id", None)
        if task_id is None:
            return None

        task_outcomes = self._eval_repeat_outcomes.setdefault(int(task_id), [])
        task_outcomes.append(float(final_correct))

        if len(task_outcomes) < self.eval_repeat_count:
            return None
        completed = self._eval_repeat_outcomes.pop(int(task_id))
        self._log_eval_repeat_metrics(completed)
        return int(task_id), completed

    @staticmethod
    def _log_eval_repeat_metrics(values: list[float]) -> None:
        mean = sum(values) / len(values)
        sample_variance = (
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            if len(values) > 1
            else 0.0
        )
        _safe_scalar(
            **{"repeat/final_correct/mean_task_sample_variance": sample_variance}
        )

        repeat_pairs = list(combinations(values, 2))
        if not repeat_pairs:
            repeat_pairs = [(values[0], values[0])]
        for left, right in repeat_pairs:
            if left or right:
                _safe_scalar(
                    **{
                        "repeat/final_correct/pairwise_success_jaccard": float(
                            left and right
                        )
                    }
                )

    async def _dump_eval_repeat_outcomes(
        self, task_id: int, outcomes: list[float]
    ) -> None:
        ctx = workflow_context.get()

        try:
            out_dir = Path(self.debug_trace_dir) / "eval" / "repeat_outcomes"
            await aiofiles.os.makedirs(out_dir, exist_ok=True)
            shard_path = out_dir / (
                f"{socket.gethostname()}_{os.getpid()}_final_correct.jsonl"
            )
            payload = {
                "task_id": int(task_id),
                "lora_version": getattr(ctx, "lora_version", None),
                "final_correct": [int(value) for value in outcomes],
            }
            async with aiofiles.open(shard_path, "a", encoding="utf-8") as outcome_file:
                await outcome_file.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
        except Exception:
            logger.exception("Failed to dump tutor eval repeat outcomes.")

    def _enabled_reward_component_keys(self) -> list[str]:
        keys = []
        if self.guidance_gate_fail_penalty:
            keys.append("guidance_gate_fail")
        if self.format_error_penalty:
            keys.append("format_error")
        if self.teacher_exact_repeat_penalty:
            keys.append("teacher_exact_repeat")
        if self.soft_overlong_enabled:
            keys.append("soft_overlong")
        return keys

    def _adaptive_gate_metrics(
        self, traces: list[TurnTrace], *, student_name: str = ""
    ) -> dict[str, float]:
        """Adaptive gate metrics for the drawn student, keyed by preference."""
        if not self.adaptive_gate_active:
            return {}
        results = [
            trace.adaptive_gate_result
            for trace in traces
            if trace.adaptive_gate_result is not None
        ]
        if not results:
            return {}
        preference = ""
        for runtime in self.student_model_runtimes.values():
            if runtime.name == student_name:
                preference = runtime.preference
                break
        if not preference:
            return {}
        prefix = f"adaptive_gate/{self._student_metric_name(preference)}"
        gated = [result for result in results if not result.passed]

        # Attempt diagnosis is undefined until the teacher has actually seen a
        # real student response. Turns before that point are auto-passed by the
        # binary gate, but counting those passes inflates compliance. Scripted
        # guidance gate replies and adaptive gate complaints are not student attempts.
        has_real_student_turn = False
        eligible: list[AdaptiveGateResult] = []
        for trace in traces:
            result = trace.adaptive_gate_result
            if result is not None and (
                preference != "attempt-diagnosis" or has_real_student_turn
            ):
                eligible.append(result)
            if (
                bool(str(trace.student_output or "").strip())
                and not trace.guidance_gate_masked
                and not trace.adaptive_gate_failed
            ):
                has_real_student_turn = True
        eligible_passed = [result for result in eligible if result.passed]
        metrics: dict[str, float] = {
            f"{prefix}/gated_turns": float(len(gated)),
            f"{prefix}/gate_calls": float(len(results)),
            f"{prefix}/gate_error": float(sum(1 for result in results if result.error)),
            # Batch-aggregated counts; RLTrainer converts them to a ratio after
            # rollout-worker reduction instead of averaging per-episode ratios.
            f"{prefix}/eligible_gate_calls": float(len(eligible)),
            f"{prefix}/eligible_gate_passes": float(len(eligible_passed)),
            f"{prefix}/compliance": float(sum(1 for result in results if result.passed))
            / float(len(results)),
        }
        # On turn 1 nothing in the teacher's prompt identifies the student.
        first = next(
            (
                trace.adaptive_gate_result
                for trace in traces
                if trace.turn_idx == 1 and trace.adaptive_gate_result is not None
            ),
            None,
        )
        if first is not None:
            metrics[f"{prefix}/compliance_turn1"] = float(first.passed)
        # Marks episodes with no gated turn, so the re-test can be read on episodes
        # where nothing was withheld from the student.
        metrics[f"{prefix}/clean_episode"] = float(not gated)
        return metrics

    def _reward_component_metrics(self, traces: list[TurnTrace]) -> dict[str, float]:
        component_totals = dict.fromkeys(self._enabled_reward_component_keys(), 0.0)
        for trace in traces:
            for component_key, value in trace.reward_components.items():
                component_totals[component_key] = component_totals.get(
                    component_key, 0.0
                ) + float(value)

        if not component_totals:
            return {}

        total_abs = sum(abs(value) for value in component_totals.values())
        metrics: dict[str, float] = {}
        for component_key, value in component_totals.items():
            metrics[f"reward_component/{component_key}"] = value
            metrics[f"reward_share/{component_key}"] = (
                abs(value) / total_abs if total_abs else 0.0
            )
        return metrics

    async def _maybe_dump_debug_trace(
        self,
        *,
        trajectory_id: int,
        task: str,
        ground_truth: str,
        latest_student_answer: str,
        total_reward: float,
        traces: list[TurnTrace],
        termination_reason: str,
        guidance_gate_fail_count: int,
        student_generalization_results: list[StudentGeneralizationResult] | None = None,
        teacher_pre_solve_result: TeacherPreSolveResult | None = None,
        student_name: str = "",
        student_model: str = "",
    ) -> None:
        if not self.debug_trace_dir:
            return
        try:
            ctx = workflow_context.get()
            task_id = ctx.task_id
            if task_id is None:
                return
            if task_id % self.debug_trace_every_n_rollouts != 0:
                return
            out_dir = Path(self.debug_trace_dir) / ("eval" if ctx.is_eval else "train")
            await aiofiles.os.makedirs(out_dir, exist_ok=True)
            file_path = out_dir / f"task_{task_id:08d}_{int(time.time() * 1000)}.json"
            payload = {
                "task_id": task_id,
                "trajectory_id": int(trajectory_id),
                "is_eval": bool(ctx.is_eval),
                "termination_reason": termination_reason,
                "total_reward": float(total_reward),
                "num_turns": len(traces),
                "guidance_gate_fail_count": int(guidance_gate_fail_count),
                "student": {
                    "name": student_name,
                    "model": student_model,
                },
                "task": task,
                "ground_truth": ground_truth,
                "latest_student_answer": latest_student_answer,
                "turns": [trace_to_json(trace) for trace in traces],
                "teacher_pre_solve": (
                    self._teacher_pre_solve_result_to_json(teacher_pre_solve_result)
                    if teacher_pre_solve_result is not None
                    else None
                ),
                "student_generalization": [
                    self._student_generalization_result_to_json(result)
                    for result in (student_generalization_results or [])
                ],
            }
            async with aiofiles.open(file_path, "w", encoding="utf-8") as trace_file:
                await trace_file.write(
                    json.dumps(payload, ensure_ascii=False, indent=2)
                )
            logger.info("Tutor debug trace dumped to %s", os.fspath(file_path))
        except Exception:
            logger.exception("Failed to dump tutor debug trace.")


SherpaWorkflow = TutorAgentWorkflow
