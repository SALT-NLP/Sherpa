from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

from tqdm import tqdm

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - handled with a runtime error
    AsyncOpenAI = None

from examples.sherpa import train as tutor_train
from examples.sherpa.configs import TUTOR_EVAL_STUDENT_FIELD, TutorConfig
from examples.sherpa.core.history import trace_to_json
from examples.sherpa.eval.guidance_gate import EvalGuidanceGateMixin
from examples.sherpa.workflow import TutorAgentWorkflow

from areal import workflow_context
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.infra.workflow_context import WorkflowContext
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("TutorApiTeacherEval")

_T = TypeVar("_T")
_baseline_evaluation: ContextVar[bool] = ContextVar(
    "api_teacher_baseline_evaluation", default=False
)

# The only probe the workflow runs: the original task, re-tested after tutoring.
RETEST_LEVELS = ("original",)

_RESERVED_REQUEST_FIELDS = {
    "messages",
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
}
_PROXY_ENV_VARS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)


@dataclass(frozen=True, slots=True)
class PresolveMode:
    name: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    mode: PresolveMode
    dataset_index: int
    attempt: int
    row: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.mode.name}:{self.dataset_index}:{self.attempt}"


@dataclass(slots=True)
class EpisodeResult:
    key: str
    mode: str
    presolve_enabled: bool
    dataset_index: int
    attempt: int
    item_id: str
    student_name: str
    student_model: str
    termination_reason: str
    error: str | None
    num_turns: int
    guidance_gate_fail_count: int
    guidance_gate_error_count: int
    format_error_count: int
    student_call_failed: bool
    answer_judge_used_count: int
    answer_judge_failed_count: int
    answer_judge_override_correct_count: int
    total_reward: float
    teacher_pre_accepted: bool | None
    teacher_pre_attempts: int
    teacher_pre_error_count: int
    generalization: dict[str, dict[str, Any]]
    latest_student_answer_preview: str
    trace_path: str | None
    duration_seconds: float
    outcome_score: float | None = None
    no_teaching_baseline: float | None = None
    # Per-episode adaptive gate evidence. Keeping the counts here (rather than only
    # in debug traces) lets full evaluations save traces only for errors without
    # losing the turn-1 and post-complaint adaptation measurements.
    adaptive_gate: dict[str, Any] = field(default_factory=dict)


class RecordingTutorWorkflow(EvalGuidanceGateMixin, TutorAgentWorkflow):
    """Tutor workflow that captures the existing eval outputs without trainers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.captured_stats: dict[str, Any] | None = None
        self.captured_trace: dict[str, Any] | None = None
        self.guidance_gate_error_count = 0
        self.guidance_gate_diagnostics: list[dict[str, Any]] = []
        self.answer_judge_used_count = 0
        self.answer_judge_failed_count = 0
        self.answer_judge_override_correct_count = 0
        self.extra_api_callers: list[Any] = []
        super().__init__(*args, **kwargs)

    def _log_rollout_stats(self, **kwargs: Any) -> None:
        self.captured_stats = dict(kwargs)

    async def _maybe_dump_debug_trace(self, **kwargs: Any) -> None:
        self.captured_trace = dict(kwargs)

    def _make_answer_judge_caller(self, **kwargs: Any) -> Any:
        caller = super()._make_answer_judge_caller(**kwargs)
        if caller is not None and hasattr(caller, "caller"):
            self.extra_api_callers.append(caller)
            if getattr(self, "capture_api_requests", False):
                from examples.sherpa.eval.api_request_trace import attach_wrapper

                attach_wrapper(caller, "answer_judge")
        return caller

    async def _run_guidance_gate(self, *args: Any, **kwargs: Any) -> Any:
        result = await super()._run_guidance_gate(*args, **kwargs)
        if result.parse_error:
            self.guidance_gate_error_count += 1
        return result

    async def _no_teaching_baseline(self, *args: Any, **kwargs: Any) -> Any:
        # Scope strictness to this eval measurement, including its parallel
        # replay/judge tasks. Training and unrelated concurrent episodes retain
        # their existing behavior.
        token = _baseline_evaluation.set(True)
        try:
            return await super()._no_teaching_baseline(*args, **kwargs)
        finally:
            _baseline_evaluation.reset(token)

    async def _retry_baseline_call(
        self,
        operation: Callable[[], Awaitable[_T]],
        error_of: Callable[[_T], Any],
        label: str,
    ) -> _T:
        attempts = 3 if _baseline_evaluation.get() else 1
        for attempt in range(attempts):
            result = await operation()
            if attempts == 1:
                return result
            error = error_of(result)
            if not error:
                return result
            if attempt + 1 < attempts:
                logger.warning(
                    "Retrying failed no-teaching baseline %s (%d/%d)",
                    label,
                    attempt + 2,
                    attempts,
                )
                await asyncio.sleep(0.5 * 2**attempt)
        # Raising before the shared workflow averages/caches its usable subset
        # makes the episode an explicit error, eligible for normal resume/retry.
        raise RuntimeError(
            f"Incomplete no-teaching baseline: {label} failed after {attempts} attempts: {error}"
        )

    async def _call_auxiliary_messages(self, *args: Any, **kwargs: Any) -> Any:
        operation = super()._call_auxiliary_messages
        return await self._retry_baseline_call(
            lambda: operation(*args, **kwargs),
            lambda result: result.error,
            str(kwargs.get("rid_prefix", "auxiliary")),
        )

    async def _score_answer_async(self, *args: Any, **kwargs: Any) -> Any:
        operation = super()._score_answer_async
        result = await self._retry_baseline_call(
            lambda: operation(*args, **kwargs),
            lambda result: (result.raw_result.get("answer_judge") or {}).get("error"),
            "answer judge",
        )
        answer_judge = result.raw_result.get("answer_judge")
        if isinstance(answer_judge, dict) and answer_judge.get("enabled") is True:
            if answer_judge.get("used") is True:
                self.answer_judge_used_count += 1
                if (
                    result.correct
                    and result.raw_result.get("exact_match_correct") is False
                ):
                    self.answer_judge_override_correct_count += 1
            elif answer_judge.get("error"):
                self.answer_judge_failed_count += 1
        return result


class ApiTeacherClient:
    """Force a configured model while retaining the workflow's token budget."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
        request_params: dict[str, Any],
        client: Any | None = None,
    ) -> None:
        if client is None:
            if AsyncOpenAI is None:
                raise RuntimeError("The openai package is required for API evaluation.")
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "EMPTY",
                timeout=timeout,
                max_retries=max(0, int(max_retries)),
            )
        self.model = model
        self.request_params = deepcopy(request_params)
        self._client = client
        self.chat = SimpleNamespace(completions=_ApiTeacherCompletions(self))

    async def list_models(self) -> list[str]:
        response = await self._client.models.list()
        return [str(item.id) for item in response.data]

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result


class _ApiTeacherCompletions:
    def __init__(self, parent: ApiTeacherClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> Any:
        request = merge_dicts(self._parent.request_params, kwargs)
        request["model"] = self._parent.model
        return await self._parent._client.chat.completions.create(**request)


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


@contextmanager
def without_proxy_environment(enabled: bool = True) -> Any:
    """Temporarily clear proxy environment variables, then restore them exactly."""

    if not enabled:
        yield
        return
    saved = {name: os.environ[name] for name in _PROXY_ENV_VARS if name in os.environ}
    for name in _PROXY_ENV_VARS:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name in _PROXY_ENV_VARS:
            os.environ.pop(name, None)
        os.environ.update(saved)


async def run_without_proxy_environment(
    operation: Callable[[], Awaitable[_T]],
    *,
    enabled: bool = True,
) -> _T:
    """Keep proxy variables cleared for an entire asynchronous API phase."""

    with without_proxy_environment(enabled=enabled):
        return await operation()


@contextmanager
def without_config_snapshot_writes() -> Any:
    """Load standalone-eval configs without writing trainer log snapshots."""

    previous_rank = os.environ.get("RANK")
    os.environ["RANK"] = "1"
    try:
        yield
    finally:
        if previous_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = previous_rank


def teacher_request_defaults(seed: int) -> dict[str, Any]:
    """Request fields for a locally served teacher: a fixed seed, native thinking off.

    `thinking` is the DeepSeek chat-template switch; Qwen's `enable_thinking`
    comes from teacher_api_request_params.
    """

    return {
        "seed": int(seed),
        "extra_body": {"chat_template_kwargs": {"thinking": False}},
    }


def prepare_episode_workflow_kwargs(
    workflow_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Copy mutable role settings without changing configured request seeds."""

    resolved = dict(workflow_kwargs)
    resolved["student_models"] = deepcopy(workflow_kwargs["student_models"])
    resolved["aux_request_params"] = deepcopy(
        workflow_kwargs.get("aux_request_params") or {}
    )
    return resolved


def load_json_object(value: str, *, label: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def load_request_params(
    value: str,
    path: Path | None,
    *,
    label: str,
) -> dict[str, Any]:
    params = load_json_object(value, label=label)
    if path is not None:
        file_params = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(file_params, dict):
            raise ValueError(f"{label}-file must contain a JSON object.")
        params = merge_dicts(params, file_params)
    reserved = sorted(_RESERVED_REQUEST_FIELDS.intersection(params))
    if reserved:
        raise ValueError(
            f"{label} cannot set dedicated fields {reserved}; use the matching CLI "
            "options instead."
        )
    return params


def normalize_base_url(value: str) -> str:
    base_url = str(value or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("--teacher-base-url is required (or set TEACHER_BASE_URL).")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def resolve_presolve_modes(choice: str, config_enabled: bool) -> list[PresolveMode]:
    if choice == "both":
        return [
            PresolveMode(name="presolve_off", enabled=False),
            PresolveMode(name="presolve_on", enabled=True),
        ]
    enabled = config_enabled if choice == "config" else choice == "on"
    return [
        PresolveMode(
            name="presolve_on" if enabled else "presolve_off",
            enabled=enabled,
        )
    ]


def snapshot_student_models(config_path: str) -> list[dict[str, Any]]:
    """Resolve students before any role-specific command-line overrides."""

    # StatsLogger imports HTTP clients while loading config. Do not let a global
    # SOCKS proxy make this purely local operation require optional socksio.
    with without_proxy_environment(), without_config_snapshot_writes():
        baseline, _ = load_expr_config(["--config", config_path], TutorConfig)
    students = [asdict(student) for student in baseline.student_models]
    if not students:
        raise ValueError(
            "The source config must define at least one student_models entry."
        )
    return students


def load_experiment_config(
    config_path: str,
    overrides: list[str],
) -> tuple[TutorConfig, list[dict[str, Any]]]:
    students = snapshot_student_models(config_path)
    with without_proxy_environment(), without_config_snapshot_writes():
        config, _ = load_expr_config(
            ["--config", config_path, *overrides],
            TutorConfig,
        )
    return config, students


def select_student_models(
    student_models: list[dict[str, Any]], requested_names: list[str] | None
) -> list[dict[str, Any]]:
    """Keep an explicit eval subset without mutating the source config."""

    if not requested_names:
        return deepcopy(student_models)
    names = list(dict.fromkeys(str(name) for name in requested_names))
    available = {str(student["name"]): student for student in student_models}
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(
            "Unknown --student-name value(s): "
            f"{missing}. Available students: {sorted(available)}"
        )
    return [deepcopy(available[name]) for name in names]


def effective_eval_presolve_enabled(config: TutorConfig) -> bool:
    override = config.evaluator.teacher_pre_enabled
    return bool(config.teacher_pre.enabled if override is None else override)


def validate_effective_eval_semantics(
    *,
    workflow_kwargs: dict[str, Any],
    student_models: list[dict[str, Any]],
) -> None:
    """Fail before API calls if the standalone evaluator cannot run this config."""

    if workflow_kwargs.get("aux_mode") != "api":
        raise ValueError(
            "The standalone API evaluator has no AReaL inference engine, so "
            "auxiliary_model.mode='self' cannot run here. Use "
            "--self-aux-via-teacher when the external teacher endpoint serves the "
            "same actor checkpoint, or evaluate with an explicit API auxiliary "
            "model."
        )
    if not student_models:
        raise ValueError("Evaluation needs at least one selected student.")


def build_eval_workflow_kwargs(
    *,
    config: TutorConfig,
    student_models: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    presolve_enabled: bool,
    external_self_aux: dict[str, Any] | None = None,
) -> dict[str, Any]:
    auxiliary_model = config.auxiliary_model
    teacher_pre = config.teacher_pre
    base_eval_gconfig = config.eval_gconfig or config.gconfig
    eval_gconfig = base_eval_gconfig.new(
        n_samples=1,
        temperature=float(args.teacher_temperature),
        top_p=args.teacher_top_p,
        max_new_tokens=int(args.teacher_max_tokens),
    )
    presolve_attempts = (
        int(args.presolve_attempts)
        if int(args.presolve_attempts) > 0
        else int(teacher_pre.attempts)
    )
    presolve_max_tokens = (
        int(args.presolve_max_tokens)
        if args.presolve_max_tokens is not None
        else int(teacher_pre.max_tokens)
    )

    # The same arguments training builds, so the standalone evaluator cannot
    # drift from the regular validation pass.
    workflow_kwargs = tutor_train.build_workflow_kwargs(
        config, tokenizer=tokenizer, student_models=student_models
    )
    if external_self_aux is not None:
        if auxiliary_model.mode != "self":
            raise ValueError(
                "--self-aux-via-teacher requires auxiliary_model.mode='self'; "
                f"the loaded config uses {auxiliary_model.mode!r}."
            )
        required = {"base_url", "model", "api_key", "request_params"}
        missing = sorted(required - set(external_self_aux))
        if missing:
            raise ValueError(
                "external self-auxiliary bridge is missing: " + ", ".join(missing)
            )
        # Regular validation's 'self' auxiliary uses the current actor adapter.
        # A standalone evaluator has no AReaL engine, so reproduce that path with
        # the same OpenAI endpoint and the same per-request lora_path as the
        # external teacher. Keeping the source auxiliary sampling settings while
        # merging the teacher request body preserves its judge and gate call semantics.
        workflow_kwargs["aux_mode"] = "api"
        workflow_kwargs["aux_base_url"] = str(external_self_aux["base_url"])
        workflow_kwargs["aux_model"] = str(external_self_aux["model"])
        workflow_kwargs["aux_api_key"] = str(external_self_aux["api_key"])
        workflow_kwargs["aux_request_params"] = merge_dicts(
            deepcopy(auxiliary_model.request_params),
            dict(external_self_aux["request_params"]),
        )
    # This is the canonical conversion used by the trainer for its regular
    # validation pass.
    eval_workflow_kwargs = tutor_train._build_eval_workflow_kwargs(
        workflow_kwargs, config
    )
    eval_workflow_kwargs["gconfig"] = eval_gconfig
    eval_workflow_kwargs["teacher_pre_enabled"] = bool(presolve_enabled)
    eval_workflow_kwargs["teacher_pre_attempts"] = presolve_attempts
    eval_workflow_kwargs["teacher_pre_max_tokens"] = presolve_max_tokens
    # Traces are owned by this standalone evaluator's --save-traces path.
    eval_workflow_kwargs["debug_trace_dir"] = None
    validate_effective_eval_semantics(
        workflow_kwargs=eval_workflow_kwargs,
        student_models=student_models,
    )
    return eval_workflow_kwargs


def prepare_test_dataset(
    config: TutorConfig,
    student_models: list[dict[str, Any]],
    *,
    tokenizer: Any,
    limit: int,
    stratified_max_samples: int = 0,
) -> Any:
    valid_config = tutor_train._without_remote_dataset_loading(config.valid_dataset)
    dataset = get_custom_dataset(
        split="test",
        dataset_config=valid_config,
        tokenizer=tokenizer,
    )
    eval_max_samples = config.evaluator.max_samples
    if eval_max_samples is not None:
        eval_max_samples = int(eval_max_samples)
        if stratified_max_samples > 0 and eval_max_samples > 0:
            raise ValueError(
                "Use either evaluator.max_samples or --stratified-max-samples, "
                "not both."
            )
        if 0 < eval_max_samples < len(dataset):
            rng = random.Random(config.seed)
            indices = sorted(rng.sample(range(len(dataset)), k=eval_max_samples))
            dataset = dataset.select(indices)
    if 0 < stratified_max_samples < len(dataset):
        dataset = dataset.select(
            stratified_math_subset_indices(
                dataset,
                sample_count=stratified_max_samples,
                seed=int(config.seed),
            )
        )
    if limit > 0 and limit < len(dataset):
        dataset = dataset.select(range(limit))
    dataset = tutor_train._expand_eval_dataset_for_students(
        dataset,
        [str(student["name"]) for student in student_models],
    )
    return dataset


def stratified_math_subset_indices(
    dataset: Any,
    *,
    sample_count: int,
    seed: int,
) -> list[int]:
    """Select a deterministic proportional subset over MATH type x level."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    dataset_size = len(dataset)
    if sample_count >= dataset_size:
        return list(range(dataset_size))

    strata: dict[tuple[str, str], list[int]] = {}
    for index in range(dataset_size):
        row = dataset[index]
        metadata = row.get("metadata") if hasattr(row, "get") else None
        if not isinstance(metadata, dict):
            raise ValueError(
                "Stratified evaluation requires every row to have metadata."
            )
        math_type = str(metadata.get("type") or "").strip()
        level = str(metadata.get("level") or "").strip()
        if not math_type or not level:
            raise ValueError(
                "Stratified evaluation requires metadata.type and metadata.level."
            )
        strata.setdefault((math_type, level), []).append(index)

    allocations: dict[tuple[str, str], int] = {}
    remainders: dict[tuple[str, str], int] = {}
    for key, indices in strata.items():
        numerator = sample_count * len(indices)
        allocations[key], remainders[key] = divmod(numerator, dataset_size)
    unallocated = sample_count - sum(allocations.values())
    remainder_order = sorted(strata, key=lambda key: (-remainders[key], key))
    for key in remainder_order[:unallocated]:
        allocations[key] += 1

    rng = random.Random(seed)
    selected: list[int] = []
    for key in sorted(strata):
        selected.extend(rng.sample(strata[key], allocations[key]))
    selected.sort()
    if len(selected) != sample_count or len(set(selected)) != sample_count:
        raise RuntimeError("Stratified subset selection produced invalid indices.")
    return selected


def dataset_sha256(dataset: Any) -> str:
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        payload = json.dumps(
            dict(dataset[index]),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        digest.update(payload.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def preview_text(value: Any, max_chars: int = 500) -> str:
    text = str(value or "")
    return text[: max(0, int(max_chars))]


def serialize_generalization(
    workflow: RecordingTutorWorkflow,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for result in workflow.last_student_generalization_results:
        judge_result = result.judge_result
        replay_count = int(result.replay_count or 0)
        replay_correct = int(result.replay_correct or 0)
        score = None
        if result.attempted and not result.skipped:
            score = (
                replay_correct / replay_count
                if replay_count > 0
                else float(bool(judge_result is not None and judge_result.correct))
            )
        payload[str(result.level)] = {
            "attempted": bool(result.attempted),
            "skipped": bool(result.skipped),
            "skip_reason": str(result.skip_reason or ""),
            "correct": bool(judge_result.correct) if judge_result is not None else None,
            "student_error": result.student_error,
            "replay_count": replay_count,
            "replay_correct": replay_correct,
            "score": score,
        }
    return payload


def summarize_adaptive_gate(
    workflow: RecordingTutorWorkflow, *, student_name: str
) -> dict[str, Any]:
    """Keep the gate sequence needed to distinguish blind and adaptive teaching.

    Every teacher turn that passes the format check and the guidance gate is
    checked; the `sampled_*` fields count those checks. "Post complaint" starts
    after the first gated turn, and the first-post value is the next check.
    """

    preference = "none"
    for runtime in workflow.student_model_runtimes.values():
        if runtime.name == student_name:
            preference = str(runtime.preference or "none")
            break

    traces = list(workflow.last_traces)
    gate_traces = [trace for trace in traces if trace.adaptive_gate_result is not None]
    active = preference not in {"", "none"}
    if not active:
        return {
            "preference": "none",
            "active": False,
            "eligible_turn_count": 0,
            "sampled_turn_count": 0,
            "passed_turn_count": 0,
            "gated_turn_count": 0,
            "gate_error_count": 0,
            "compliance": None,
            "turn1_sampled": False,
            "turn1_passed": None,
            "first_complaint_turn": None,
            "first_post_complaint_sampled_turn": None,
            "first_post_complaint_passed": None,
            "post_complaint_sampled_turn_count": 0,
            "post_complaint_passed_turn_count": 0,
            "post_complaint_compliance": None,
            "post_complaint_all_passed": None,
        }

    sampled = gate_traces
    passed = [trace for trace in sampled if bool(trace.adaptive_gate_result.passed)]
    gated = [trace for trace in gate_traces if bool(trace.adaptive_gate_failed)]
    errors = [trace for trace in sampled if bool(trace.adaptive_gate_result.error)]

    turn1 = next((trace for trace in gate_traces if trace.turn_idx == 1), None)
    first_complaint = gated[0] if gated else None
    post_sampled = (
        [trace for trace in sampled if trace.turn_idx > first_complaint.turn_idx]
        if first_complaint is not None
        else []
    )
    post_passed = [
        trace for trace in post_sampled if bool(trace.adaptive_gate_result.passed)
    ]
    first_post = post_sampled[0] if post_sampled else None

    return {
        "preference": preference,
        "active": True,
        "eligible_turn_count": len(gate_traces),
        "sampled_turn_count": len(sampled),
        "passed_turn_count": len(passed),
        "gated_turn_count": len(gated),
        "gate_error_count": len(errors),
        "compliance": _rate(len(passed), len(sampled)),
        "turn1_sampled": turn1 is not None,
        "turn1_passed": (
            bool(turn1.adaptive_gate_result.passed) if turn1 is not None else None
        ),
        "first_complaint_turn": (
            int(first_complaint.turn_idx) if first_complaint is not None else None
        ),
        "first_post_complaint_sampled_turn": (
            int(first_post.turn_idx) if first_post is not None else None
        ),
        "first_post_complaint_passed": (
            bool(first_post.adaptive_gate_result.passed)
            if first_post is not None
            else None
        ),
        "post_complaint_sampled_turn_count": len(post_sampled),
        "post_complaint_passed_turn_count": len(post_passed),
        "post_complaint_compliance": _rate(len(post_passed), len(post_sampled)),
        "post_complaint_all_passed": (
            len(post_passed) == len(post_sampled) if post_sampled else None
        ),
    }


def build_trace_payload(
    *,
    workflow: RecordingTutorWorkflow,
    spec: EpisodeSpec,
    result: EpisodeResult,
) -> dict[str, Any]:
    captured = workflow.captured_trace or {}
    teacher_pre = workflow.last_teacher_pre_solve_result
    return {
        "result": asdict(result),
        "dataset_row": spec.row,
        "task": captured.get("task", spec.row.get("task", "")),
        "ground_truth": captured.get("ground_truth", spec.row.get("ground_truth", "")),
        "latest_student_answer": captured.get("latest_student_answer", ""),
        "turns": [trace_to_json(trace) for trace in workflow.last_traces],
        "history": workflow.last_history,
        "guidance_gate_diagnostics": workflow.guidance_gate_diagnostics,
        "teacher_pre_solve": asdict(teacher_pre) if teacher_pre is not None else None,
        "student_generalization": [
            asdict(item) for item in workflow.last_student_generalization_results
        ],
    }


def result_from_workflow(
    *,
    workflow: RecordingTutorWorkflow,
    spec: EpisodeSpec,
    duration_seconds: float,
) -> EpisodeResult:
    stats = workflow.captured_stats
    if stats is None:
        raise RuntimeError("Tutor workflow did not emit episode statistics.")
    termination_reason = str(stats.get("termination_reason") or "unknown")
    traces = workflow.last_traces
    captured = workflow.captured_trace or {}
    student_call_failed = bool(stats.get("student_call_failed"))
    teacher_pre = workflow.last_teacher_pre_solve_result
    teacher_pre_errors = (
        [attempt.error for attempt in teacher_pre.attempts if attempt.error]
        if teacher_pre is not None
        else []
    )
    all_teacher_pre_attempts_failed = bool(
        teacher_pre is not None
        and teacher_pre.attempts
        and len(teacher_pre_errors) == len(teacher_pre.attempts)
    )
    outcome_score = float(
        workflow._free_chat_outcome_score(workflow.last_student_generalization_results)
    )
    no_teaching_baseline = stats.get("no_teaching_baseline")
    return EpisodeResult(
        key=spec.key,
        mode=spec.mode.name,
        presolve_enabled=spec.mode.enabled,
        dataset_index=spec.dataset_index,
        attempt=spec.attempt,
        item_id=str(spec.row.get("id", spec.dataset_index)),
        student_name=str(stats.get("student_name") or ""),
        student_model=str(captured.get("student_model") or ""),
        termination_reason=termination_reason,
        error=(
            "TeacherPreSolveError: all presolve attempts failed: "
            + " | ".join(teacher_pre_errors)
            if all_teacher_pre_attempts_failed
            else None
        ),
        num_turns=len(traces),
        guidance_gate_fail_count=int(stats.get("guidance_gate_fail_count") or 0),
        guidance_gate_error_count=workflow.guidance_gate_error_count,
        format_error_count=sum(bool(trace.tutor_format_error) for trace in traces),
        student_call_failed=student_call_failed,
        answer_judge_used_count=workflow.answer_judge_used_count,
        answer_judge_failed_count=workflow.answer_judge_failed_count,
        answer_judge_override_correct_count=(
            workflow.answer_judge_override_correct_count
        ),
        total_reward=float(stats.get("total_reward") or 0.0),
        teacher_pre_accepted=(
            bool(teacher_pre.accepted) if teacher_pre is not None else None
        ),
        teacher_pre_attempts=(
            len(teacher_pre.attempts) if teacher_pre is not None else 0
        ),
        teacher_pre_error_count=len(teacher_pre_errors),
        generalization=serialize_generalization(workflow),
        latest_student_answer_preview=preview_text(
            captured.get("latest_student_answer", "")
        ),
        trace_path=None,
        duration_seconds=float(duration_seconds),
        outcome_score=outcome_score,
        no_teaching_baseline=(
            float(no_teaching_baseline) if no_teaching_baseline is not None else None
        ),
        adaptive_gate=summarize_adaptive_gate(
            workflow,
            student_name=str(stats.get("student_name") or ""),
        ),
    )


def error_result(
    spec: EpisodeSpec,
    error: Exception,
    *,
    duration_seconds: float,
) -> EpisodeResult:
    return EpisodeResult(
        key=spec.key,
        mode=spec.mode.name,
        presolve_enabled=spec.mode.enabled,
        dataset_index=spec.dataset_index,
        attempt=spec.attempt,
        item_id=str(spec.row.get("id", spec.dataset_index)),
        student_name=str(spec.row.get(TUTOR_EVAL_STUDENT_FIELD) or ""),
        student_model="",
        termination_reason="error",
        error=f"{type(error).__name__}: {error}",
        num_turns=0,
        guidance_gate_fail_count=0,
        guidance_gate_error_count=0,
        format_error_count=0,
        student_call_failed=False,
        answer_judge_used_count=0,
        answer_judge_failed_count=0,
        answer_judge_override_correct_count=0,
        total_reward=0.0,
        teacher_pre_accepted=None,
        teacher_pre_attempts=0,
        teacher_pre_error_count=0,
        generalization={},
        latest_student_answer_preview="",
        trace_path=None,
        duration_seconds=float(duration_seconds),
    )


async def close_workflow_api_clients(workflow: RecordingTutorWorkflow) -> None:
    """Close per-episode auxiliary/student clients created by the workflow."""

    wrappers = [
        workflow.aux_caller,
        *workflow.extra_api_callers,
    ]
    for runtime in workflow.student_model_runtimes.values():
        wrappers.append(runtime.caller)

    clients: dict[int, Any] = {}
    for wrapper in wrappers:
        caller = getattr(wrapper, "caller", None)
        client = getattr(caller, "_client", None)
        if client is not None:
            clients[id(client)] = client
    for client in clients.values():
        close = getattr(client, "close", None)
        if close is None:
            continue
        try:
            result = close()
            if asyncio.iscoroutine(result):
                async with asyncio.timeout(10.0):
                    await result
        except TimeoutError:
            logger.warning("Timed out closing an episode API client after 10s.")
        except Exception as exc:  # pragma: no cover - transport-specific cleanup
            logger.warning("Failed to close an episode API client: %s", exc)


def safe_path_token(value: Any) -> str:
    text = str(value)
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    return safe.strip("._") or "item"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()


def rewrite_results_jsonl(path: Path, results: list[EpisodeResult]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(asdict(result), ensure_ascii=False, sort_keys=True) + "\n"
            for result in results
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_existing_results(path: Path) -> list[EpisodeResult]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty_line_numbers = [
        line_number for line_number, line in enumerate(lines, 1) if line.strip()
    ]
    last_nonempty_line = nonempty_line_numbers[-1] if nonempty_line_numbers else 0
    results_by_key: dict[str, EpisodeResult] = {}
    needs_rewrite = False
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            result = EpisodeResult(**payload)
        except json.JSONDecodeError as exc:
            if line_number == last_nonempty_line:
                logger.warning(
                    "Ignoring a truncated final resume record at %s:%s: %s",
                    path,
                    line_number,
                    exc,
                )
                needs_rewrite = True
                break
            raise ValueError(
                f"Invalid resume record at {path}:{line_number}: {exc}"
            ) from exc
        except TypeError as exc:
            raise ValueError(
                f"Invalid resume record at {path}:{line_number}: {exc}"
            ) from exc
        if result.key in results_by_key:
            logger.warning(
                "Duplicate resume key %s; keeping the latest record.", result.key
            )
            needs_rewrite = True
        results_by_key[result.key] = result
    results = list(results_by_key.values())
    if needs_rewrite:
        rewrite_results_jsonl(path, results)
    return results


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def latest_results(results: list[EpisodeResult]) -> list[EpisodeResult]:
    """Keep the last record for each episode key, matching JSONL resume semantics."""

    by_key: dict[str, EpisodeResult] = {}
    for result in results:
        by_key[result.key] = result
    return list(by_key.values())


def result_needs_retry(
    result: EpisodeResult,
    *,
    retry_errors: bool = True,
    retry_diagnostic_failures: bool = False,
    generalization_levels: tuple[str, ...] = (),
    expected_generalization_replays: int = 0,
) -> bool:
    """Return whether resume should replace an unreliable episode record."""

    return bool(
        result_retry_reasons(
            result,
            retry_errors=retry_errors,
            retry_diagnostic_failures=retry_diagnostic_failures,
            generalization_levels=generalization_levels,
            expected_generalization_replays=expected_generalization_replays,
        )
    )


def result_retry_reasons(
    result: EpisodeResult,
    *,
    retry_errors: bool = True,
    retry_diagnostic_failures: bool = False,
    generalization_levels: tuple[str, ...] = (),
    expected_generalization_replays: int = 0,
) -> list[str]:
    """Describe infrastructure failures that make an episode unsafe to score."""

    reasons: list[str] = []
    if retry_errors and result.error is not None:
        reasons.append(f"error={result.error}")
    if not retry_diagnostic_failures:
        return reasons
    if result.student_call_failed:
        reasons.append("student_call_failed")
    if result.guidance_gate_error_count:
        reasons.append(f"guidance_gate_errors={result.guidance_gate_error_count}")
    if result.answer_judge_failed_count:
        reasons.append(f"answer_judge_failed={result.answer_judge_failed_count}")
    if result.teacher_pre_error_count:
        reasons.append(f"teacher_pre_errors={result.teacher_pre_error_count}")
    gate = result.adaptive_gate or {}
    if int(gate.get("gate_error_count", 0) or 0):
        reasons.append(f"adaptive_gate_errors={gate['gate_error_count']}")
    expected_replays = max(0, int(expected_generalization_replays))
    if expected_replays:
        generalization = result.generalization or {}
        for level in generalization_levels:
            replay = generalization.get(level)
            if not isinstance(replay, dict):
                reasons.append(f"{level}_retest=missing")
                continue
            actual_replays = int(replay.get("replay_count", -1) or 0)
            if actual_replays != expected_replays:
                reasons.append(f"{level}_replays={actual_replays}/{expected_replays}")
            if replay.get("score") is None:
                reasons.append(f"{level}_score=missing")
            if replay.get("student_error"):
                reasons.append(f"{level}_student_error={replay['student_error']}")
    return reasons


def aggregate_mode(
    results: list[EpisodeResult],
    *,
    expected: int,
) -> dict[str, Any]:
    results = latest_results(results)
    completed = [result for result in results if result.error is None]
    outcome_scores = [float(result.outcome_score) for result in completed]
    pre_solve_skipped = sum(
        result.termination_reason == "pre_solve_skipped" for result in completed
    )
    covered = len(completed) - pre_solve_skipped
    presolve_results = [result for result in completed if result.presolve_enabled]
    presolve_accepted = sum(
        result.teacher_pre_accepted is True for result in presolve_results
    )
    presolve_enabled = any(result.presolve_enabled for result in results)
    generalization: dict[str, dict[str, Any]] = {}
    levels = sorted(
        {level for result in completed for level in result.generalization}
        | set(RETEST_LEVELS)
    )
    for level in levels:
        level_results = [
            (result, result.generalization[level])
            for result in completed
            if level in result.generalization
        ]
        attempted = [pair for pair in level_results if pair[1].get("attempted")]
        evaluable = [pair for pair in attempted if not pair[1].get("student_error")]
        correct = sum(item.get("correct") is True for _, item in attempted)
        evaluable_correct = sum(item.get("correct") is True for _, item in evaluable)
        student_errors = sum(
            bool(item.get("student_error")) for _, item in level_results
        )
        replay_count = sum(int(item.get("replay_count") or 0) for _, item in attempted)
        replay_correct = sum(
            int(item.get("replay_correct") or 0) for _, item in attempted
        )

        def _score(item: dict[str, Any]) -> float:
            if item.get("score") is not None:
                return float(item["score"])
            item_replays = int(item.get("replay_count") or 0)
            if item_replays > 0:
                return int(item.get("replay_correct") or 0) / item_replays
            return float(item.get("correct") is True)

        attempted_score = sum(_score(item) for _, item in attempted)
        evaluable_score = sum(_score(item) for _, item in evaluable)
        generalization[level] = {
            "present_episode_count": len(level_results),
            "attempted": len(attempted),
            "evaluable": len(evaluable),
            "correct": correct,
            "student_error_count": student_errors,
            "replay_attempt_count": replay_count,
            "replay_correct_count": replay_correct,
            "accuracy_on_replays": _rate(replay_correct, replay_count),
            "mean_episode_score_on_attempted": _rate(attempted_score, len(attempted)),
            "mean_episode_score_on_evaluable": _rate(evaluable_score, len(evaluable)),
            "mean_episode_score_full_set": _rate(attempted_score, expected),
            "conditional_accuracy_on_attempted": _rate(correct, len(attempted)),
            "conditional_accuracy_on_evaluable": _rate(
                evaluable_correct, len(evaluable)
            ),
            "attempt_coverage_full_set": _rate(len(attempted), expected),
            "end_to_end_correct_rate_full_set": _rate(correct, expected),
        }

    baseline_pairs = [
        (float(result.no_teaching_baseline), outcome_score)
        for result, outcome_score in zip(completed, outcome_scores, strict=True)
        if result.no_teaching_baseline is not None
    ]

    gate_payloads = [
        result.adaptive_gate or {}
        for result in completed
        if isinstance(result.adaptive_gate, dict)
    ]
    active_gates = [gate for gate in gate_payloads if gate.get("active") is True]
    sampled_turns = sum(
        int(gate.get("sampled_turn_count", 0) or 0) for gate in active_gates
    )
    passed_turns = sum(
        int(gate.get("passed_turn_count", 0) or 0) for gate in active_gates
    )
    gated_turns = sum(
        int(gate.get("gated_turn_count", 0) or 0) for gate in active_gates
    )
    turn1_sampled = [gate for gate in active_gates if gate.get("turn1_sampled")]
    complaint_episodes = [
        gate for gate in active_gates if gate.get("first_complaint_turn") is not None
    ]
    first_post_checked = [
        gate
        for gate in complaint_episodes
        if gate.get("first_post_complaint_passed") is not None
    ]
    post_sampled_turns = sum(
        int(gate.get("post_complaint_sampled_turn_count", 0) or 0)
        for gate in complaint_episodes
    )
    post_passed_turns = sum(
        int(gate.get("post_complaint_passed_turn_count", 0) or 0)
        for gate in complaint_episodes
    )
    post_sustained = [
        gate
        for gate in complaint_episodes
        if gate.get("post_complaint_all_passed") is not None
    ]
    episode_compliances = [
        float(gate["compliance"])
        for gate in active_gates
        if gate.get("compliance") is not None
    ]
    adaptive_gate = {
        "active_episode_count": len(active_gates),
        "ungated_episode_count": len(gate_payloads) - len(active_gates),
        "preferences": dict(
            sorted(
                Counter(
                    str(gate.get("preference") or "none") for gate in gate_payloads
                ).items()
            )
        ),
        "eligible_turn_count": sum(
            int(gate.get("eligible_turn_count", 0) or 0) for gate in active_gates
        ),
        "sampled_turn_count": sampled_turns,
        "passed_turn_count": passed_turns,
        "gated_turn_count": gated_turns,
        "gate_error_count": sum(
            int(gate.get("gate_error_count", 0) or 0) for gate in active_gates
        ),
        "gate_error_episode_count": sum(
            int(gate.get("gate_error_count", 0) or 0) > 0 for gate in active_gates
        ),
        "micro_compliance": _rate(passed_turns, sampled_turns),
        "mean_episode_compliance": (
            sum(episode_compliances) / len(episode_compliances)
            if episode_compliances
            else None
        ),
        "turn1_sampled_episode_count": len(turn1_sampled),
        "turn1_passed_episode_count": sum(
            gate.get("turn1_passed") is True for gate in turn1_sampled
        ),
        "turn1_compliance": _rate(
            sum(gate.get("turn1_passed") is True for gate in turn1_sampled),
            len(turn1_sampled),
        ),
        "complaint_episode_count": len(complaint_episodes),
        "complaint_episode_rate": _rate(len(complaint_episodes), len(active_gates)),
        "first_post_complaint_checked_episode_count": len(first_post_checked),
        "first_post_complaint_passed_episode_count": sum(
            gate.get("first_post_complaint_passed") is True
            for gate in first_post_checked
        ),
        "first_post_complaint_compliance": _rate(
            sum(
                gate.get("first_post_complaint_passed") is True
                for gate in first_post_checked
            ),
            len(first_post_checked),
        ),
        "post_complaint_sampled_turn_count": post_sampled_turns,
        "post_complaint_passed_turn_count": post_passed_turns,
        "post_complaint_micro_compliance": _rate(post_passed_turns, post_sampled_turns),
        "post_complaint_sustained_episode_count": sum(
            gate.get("post_complaint_all_passed") is True for gate in post_sustained
        ),
        "post_complaint_sustained_rate": _rate(
            sum(
                gate.get("post_complaint_all_passed") is True for gate in post_sustained
            ),
            len(post_sustained),
        ),
    }

    return {
        "expected_attempts": int(expected),
        "recorded_attempts": len(results),
        "completed_attempts": len(completed),
        "error_count": len(results) - len(completed),
        "recorded_coverage_rate": _rate(len(results), expected),
        "execution_coverage_rate": _rate(len(completed), expected),
        "termination_histogram": dict(
            sorted(Counter(result.termination_reason for result in results).items())
        ),
        "outcome_score_sum": sum(outcome_scores),
        "outcome_score_mean_completed": _rate(sum(outcome_scores), len(completed)),
        "outcome_score_mean_full_set": _rate(sum(outcome_scores), expected),
        "no_teaching_baseline_mean": (
            _rate(sum(pair[0] for pair in baseline_pairs), len(baseline_pairs))
            if baseline_pairs
            else None
        ),
        "improvement_over_no_teaching_baseline_mean": (
            _rate(
                sum(outcome - baseline for baseline, outcome in baseline_pairs),
                len(baseline_pairs),
            )
            if baseline_pairs
            else None
        ),
        "workflow_covered_count": covered,
        "workflow_coverage_rate_full_set": _rate(covered, expected),
        "presolve_covered_count": covered if presolve_enabled else None,
        "presolve_coverage_rate": (
            _rate(covered, len(completed)) if presolve_enabled else None
        ),
        "presolve_coverage_rate_full_set": (
            _rate(covered, expected) if presolve_enabled else None
        ),
        "presolve_skipped_count": pre_solve_skipped,
        "presolve_accepted_count": presolve_accepted,
        "presolve_acceptance_rate": _rate(presolve_accepted, len(presolve_results)),
        "presolve_acceptance_rate_full_set": (
            _rate(presolve_accepted, expected) if presolve_enabled else None
        ),
        "teacher_pre_error_count": sum(
            result.teacher_pre_error_count for result in results
        ),
        "guidance_gate_failed_episode_count": sum(
            result.guidance_gate_fail_count > 0 for result in completed
        ),
        "guidance_gate_error_episode_count": sum(
            result.guidance_gate_error_count > 0 for result in completed
        ),
        "answer_judge_used_count": sum(
            result.answer_judge_used_count for result in completed
        ),
        "answer_judge_failed_episode_count": sum(
            result.answer_judge_failed_count > 0 for result in completed
        ),
        "answer_judge_override_correct_count": sum(
            result.answer_judge_override_correct_count for result in completed
        ),
        "format_error_episode_count": sum(
            result.format_error_count > 0 for result in completed
        ),
        "student_call_failed_count": sum(
            result.student_call_failed for result in completed
        ),
        "avg_turns_completed": (
            sum(result.num_turns for result in completed) / len(completed)
            if completed
            else None
        ),
        "adaptive_gate": adaptive_gate,
        "generalization": generalization,
    }


def aggregate_report(
    results: list[EpisodeResult],
    *,
    modes: list[PresolveMode],
    dataset_size: int,
    attempts: int,
) -> dict[str, Any]:
    results = latest_results(results)
    expected_per_mode = dataset_size * attempts
    mode_summaries = {
        mode.name: aggregate_mode(
            [result for result in results if result.mode == mode.name],
            expected=expected_per_mode,
        )
        for mode in modes
    }
    return {
        "dataset_rows": dataset_size,
        "attempts_per_row": attempts,
        "expected_total_attempts": dataset_size * attempts * len(modes),
        "recorded_total_attempts": len(results),
        "modes": mode_summaries,
    }


def redact_sensitive(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if any(
        token in lowered
        for token in (
            "api_key",
            "api-key",
            "authorization",
            "secret",
        )
    ):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            item_key: redact_sensitive(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def build_run_signature(
    *,
    args: argparse.Namespace,
    config: TutorConfig,
    student_models: list[dict[str, Any]],
    modes: list[PresolveMode],
    dataset_size: int,
    dataset_hash: str,
    attempts: int,
    teacher_base_url: str,
    teacher_request_params: dict[str, Any],
    workflow_kwargs_by_mode: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    effective_kwargs = next(iter(workflow_kwargs_by_mode.values()))
    return redact_sensitive(
        {
            "config": str(config_path),
            "config_sha256": config_hash,
            "evaluator_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
                + Path(__file__).with_name("guidance_gate.py").read_bytes()
            ).hexdigest(),
            "overrides": list(args.overrides),
            "dataset_size": dataset_size,
            "dataset_sha256": dataset_hash,
            "dataset_selection": {
                "strategy": (
                    "math_type_level_stratified"
                    if int(args.stratified_max_samples) > 0
                    else "config"
                ),
                "stratified_max_samples": int(args.stratified_max_samples),
            },
            "shard": {"count": int(args.shard_count), "index": int(args.shard_index)},
            "attempts": attempts,
            "modes": [asdict(mode) for mode in modes],
            "presolve": {
                "verify": bool(effective_kwargs["teacher_pre_verify"]),
                "attempts": int(effective_kwargs["teacher_pre_attempts"]),
                "max_tokens": int(effective_kwargs["teacher_pre_max_tokens"]),
            },
            "teacher": {
                "base_url": teacher_base_url,
                "model": args.teacher_model,
                "temperature": args.teacher_temperature,
                "top_p": args.teacher_top_p,
                "max_tokens": args.teacher_max_tokens,
                "timeout": args.teacher_timeout,
                "request_params": teacher_request_params,
                "test_flow_budget": {
                    "tokenizer_path": config.tokenizer_path,
                    "model_context_length": config.sglang.context_length,
                    "max_train_sample_tokens": effective_kwargs[
                        "max_train_sample_tokens"
                    ],
                    "note": (
                        "Token counts use the experiment tokenizer rather than "
                        "the API model tokenizer."
                    ),
                },
            },
            "reliability": {
                "episode_error_retries": int(args.episode_error_retries),
                "episode_error_retry_backoff_seconds": float(
                    args.episode_error_retry_backoff_seconds
                ),
                "retry_diagnostic_failures": bool(args.retry_diagnostic_failures),
            },
            "auxiliary": {
                "source_mode": config.auxiliary_model.mode,
                "effective_mode": effective_kwargs["aux_mode"],
                "base_url": effective_kwargs["aux_base_url"],
                "model": effective_kwargs["aux_model"],
                "temperature": config.auxiliary_model.temperature,
                "top_p": config.auxiliary_model.top_p,
                "max_tokens": config.auxiliary_model.max_tokens,
                "timeout": config.auxiliary_model.timeout,
                "max_concurrent_calls": effective_kwargs["max_concurrent_aux_calls"],
                "request_params": effective_kwargs["aux_request_params"],
            },
            "students": student_models,
            "test_semantics": {
                key: effective_kwargs[key]
                for key in (
                    "max_turns",
                    "guidance_gate_mode",
                    "teacher_response_format",
                    "adaptive_gate",
                    "retest_replays",
                    "length_retry_enabled",
                    "length_retry_attempts",
                    "mask_rejected_turns",
                )
            },
        }
    )


def resolve_output_dir(args: argparse.Namespace, config: TutorConfig) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return (
        Path(config.cluster.fileroot)
        / "api_teacher_eval"
        / safe_path_token(config.experiment_name)
        / safe_path_token(config.trial_name)
        / timestamp
    ).resolve()


def prepare_output_dir(
    output_dir: Path,
    *,
    signature: dict[str, Any],
    resume: bool,
    allow_evaluator_code_change: bool = False,
    allow_diagnostic_backfill: bool = False,
) -> None:
    signature_path = output_dir / "run_config.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --resume or choose "
                "a new directory."
            )
        if not signature_path.exists():
            raise ValueError(f"Resume directory is missing {signature_path.name}.")
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
        previous_signature = previous.get("signature")
        if previous_signature != signature:
            if allow_diagnostic_backfill and isinstance(previous_signature, dict):
                before = deepcopy(previous_signature)
                after = deepcopy(signature)
                old_policy = before.get("reliability", {}).get(
                    "retry_diagnostic_failures"
                )
                new_policy = after.get("reliability", {}).get(
                    "retry_diagnostic_failures"
                )
                if old_policy is False and new_policy is True:
                    after["reliability"]["retry_diagnostic_failures"] = False
                    if allow_evaluator_code_change:
                        before.pop("evaluator_sha256", None)
                        after.pop("evaluator_sha256", None)
                    if before == after:
                        append_jsonl(
                            output_dir / "resume_policy_events.jsonl",
                            {
                                "recorded_at": datetime.now(UTC).isoformat(),
                                "reason": "Explicit diagnostic backfill; all non-retry settings match",
                                "previous_signature": previous_signature,
                                "resumed_signature": signature,
                            },
                        )
                        logger.warning(
                            "Explicit diagnostic backfill enabled; all non-retry settings match."
                        )
                        return
            compatible_code_change = False
            if allow_evaluator_code_change and isinstance(previous_signature, dict):
                previous_without_code = dict(previous_signature)
                current_without_code = dict(signature)
                previous_hash = previous_without_code.pop("evaluator_sha256", None)
                current_hash = current_without_code.pop("evaluator_sha256", None)
                compatible_code_change = (
                    previous_hash is not None
                    and current_hash is not None
                    and previous_hash != current_hash
                    and previous_without_code == current_without_code
                )
            if not compatible_code_change:
                raise ValueError(
                    "Resume settings differ from the existing run_config.json."
                )
            logger.warning(
                "Resuming across an explicitly allowed evaluator code change; "
                "all other run signature fields match exactly (old_sha256=%s, "
                "new_sha256=%s).",
                previous_hash,
                current_hash,
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        signature_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "signature": signature,
        },
    )


async def run_episode(
    *,
    spec: EpisodeSpec,
    workflow_kwargs: dict[str, Any],
    teacher_client: ApiTeacherClient,
    output_dir: Path,
    save_traces: str,
    keep_env_proxy: bool,
    execution_try: int = 1,
    generalization_levels: tuple[str, ...] = (),
    expected_generalization_replays: int = 0,
    episode_timeout_seconds: float = 300.0,
) -> EpisodeResult:
    started = time.monotonic()
    workflow: RecordingTutorWorkflow | None = None
    from examples.sherpa.eval.api_request_trace import (
        ACTIVE_TRACE,
        attach,
        attach_wrapper,
    )

    capture = getattr(teacher_client, "capture_api_requests", False)
    capture_token = ACTIVE_TRACE.set(
        {
            "path": str(output_dir / "api_requests.jsonl"),
            "episode_key": spec.key,
            "execution_try": execution_try,
        }
        if capture
        else None
    )
    try:
        episode_workflow_kwargs = prepare_episode_workflow_kwargs(workflow_kwargs)
        with without_proxy_environment(enabled=not keep_env_proxy):
            workflow = RecordingTutorWorkflow(**episode_workflow_kwargs)
        if capture:
            workflow.capture_api_requests = True
            attach(teacher_client._client, "teacher")
            attach_wrapper(workflow.aux_caller, "judge")
            for caller in workflow.extra_api_callers:
                attach_wrapper(caller, "answer_judge")
            for name, runtime in workflow.student_model_runtimes.items():
                attach_wrapper(runtime.caller, f"student:{name}")
        workflow_context.set(
            WorkflowContext(
                is_eval=True,
                task_id=spec.dataset_index,
                lora_version=None,
            )
        )
        # Client-level timeouts do not bound the whole episode, which fans out into
        # teacher, student, judge and replay calls; a hard outer deadline keeps one
        # stalled call from holding the episode indefinitely.
        async with asyncio.timeout(float(episode_timeout_seconds)):
            await workflow._run_episode(
                dict(spec.row),
                external_client=teacher_client,
            )
        result = result_from_workflow(
            workflow=workflow,
            spec=spec,
            duration_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        result = error_result(
            spec,
            exc,
            duration_seconds=time.monotonic() - started,
        )
    finally:
        ACTIVE_TRACE.reset(capture_token)
        if workflow is not None:
            await close_workflow_api_clients(workflow)

    should_trace = save_traces == "all" or (
        save_traces == "errors"
        and result_needs_retry(
            result,
            retry_errors=True,
            retry_diagnostic_failures=True,
            generalization_levels=generalization_levels,
            expected_generalization_replays=expected_generalization_replays,
        )
    )
    if should_trace:
        retry_suffix = "" if execution_try <= 1 else f"_retry_{execution_try - 1:02d}"
        trace_path = (
            output_dir
            / "traces"
            / spec.mode.name
            / (
                f"row_{spec.dataset_index:05d}_id_{safe_path_token(result.item_id)}_"
                f"attempt_{spec.attempt:02d}{retry_suffix}.json"
            )
        )
        if result.error is None and workflow is not None:
            trace_payload = build_trace_payload(
                workflow=workflow,
                spec=spec,
                result=result,
            )
        else:
            trace_payload = {
                "result": asdict(result),
                "dataset_row": spec.row,
                "guidance_gate_diagnostics": (
                    workflow.guidance_gate_diagnostics if workflow is not None else []
                ),
            }
        result.trace_path = str(trace_path)
        trace_payload["result"]["trace_path"] = str(trace_path)
        await asyncio.to_thread(write_json, trace_path, trace_payload)
    return result


async def preflight_teacher(client: ApiTeacherClient, model: str) -> None:
    models = await client.list_models()
    normalized = {item.lower() for item in models}
    if model.lower() not in normalized:
        raise ValueError(
            f"Teacher model {model!r} was not returned by /v1/models: {models}."
        )
    logger.info("Teacher endpoint ready; available models=%s", models)


async def run_all(
    *,
    specs: list[EpisodeSpec],
    completed_keys: set[str],
    workflow_kwargs_by_mode: dict[str, dict[str, Any]],
    teacher_client: ApiTeacherClient,
    output_dir: Path,
    save_traces: str,
    concurrency: int,
    log_every: int,
    keep_env_proxy: bool,
    error_retries: int = 0,
    retry_backoff_seconds: float = 1.0,
    retry_diagnostic_failures: bool = False,
    episode_timeout_seconds: float = 300.0,
) -> list[EpisodeResult]:
    pending = [spec for spec in specs if spec.key not in completed_keys]
    if not pending:
        logger.info(
            "All %s attempts are already present; nothing to resume.", len(specs)
        )
        return []

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    write_lock = asyncio.Lock()
    results_path = output_dir / "results.jsonl"
    retry_events_path = output_dir / "retry_events.jsonl"
    previous_results = (
        [
            result
            for result in latest_results(load_existing_results(results_path))
            if result.key in completed_keys
        ]
        if results_path.exists()
        else []
    )
    processed = 0
    error_count = sum(result.error is not None for result in previous_results)
    diagnostic_failure_count = sum(
        result.student_call_failed
        or result.guidance_gate_error_count > 0
        or result.answer_judge_failed_count > 0
        or result.teacher_pre_error_count > 0
        for result in previous_results
    )
    mode_counts: Counter[str] = Counter(result.mode for result in previous_results)
    progress = tqdm(
        total=len(specs),
        initial=len(specs) - len(pending),
        desc="Tutor API eval",
        unit="episode",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    async def _run(spec: EpisodeSpec) -> EpisodeResult:
        nonlocal diagnostic_failure_count, error_count, processed
        result: EpisodeResult | None = None
        total_execution_tries = max(0, int(error_retries)) + 1
        workflow_kwargs = workflow_kwargs_by_mode[spec.mode.name]
        expected_generalization_replays = int(workflow_kwargs["retest_replays"])
        async with semaphore:
            for execution_try in range(1, total_execution_tries + 1):
                result = await run_episode(
                    spec=spec,
                    workflow_kwargs=workflow_kwargs,
                    teacher_client=teacher_client,
                    output_dir=output_dir,
                    save_traces=save_traces,
                    keep_env_proxy=keep_env_proxy,
                    execution_try=execution_try,
                    generalization_levels=RETEST_LEVELS,
                    expected_generalization_replays=expected_generalization_replays,
                    episode_timeout_seconds=episode_timeout_seconds,
                )
                reasons = result_retry_reasons(
                    result,
                    retry_errors=True,
                    retry_diagnostic_failures=retry_diagnostic_failures,
                    generalization_levels=RETEST_LEVELS,
                    expected_generalization_replays=expected_generalization_replays,
                )
                if not reasons:
                    break

                will_retry = execution_try < total_execution_tries
                retry_event = {
                    "attempt": spec.attempt,
                    "dataset_index": spec.dataset_index,
                    "episode_key": spec.key,
                    "execution_try": execution_try,
                    "item_id": result.item_id,
                    "max_execution_tries": total_execution_tries,
                    "reasons": reasons,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "will_retry": will_retry,
                }
                async with write_lock:
                    await asyncio.to_thread(
                        append_jsonl, retry_events_path, retry_event
                    )
                if not will_retry:
                    logger.error(
                        "Episode %s exhausted %s execution tries; recording it for "
                        "later backfill and continuing: %s",
                        spec.key,
                        total_execution_tries,
                        "; ".join(reasons),
                    )
                    break

                delay = max(0.0, float(retry_backoff_seconds)) * (
                    2 ** (execution_try - 1)
                )
                logger.warning(
                    "Episode %s failed execution try %s/%s; retrying in %.1fs: %s",
                    spec.key,
                    execution_try,
                    total_execution_tries,
                    delay,
                    "; ".join(reasons),
                )
                if delay:
                    await asyncio.sleep(delay)
        assert result is not None
        async with write_lock:
            await asyncio.to_thread(append_jsonl, results_path, asdict(result))
            processed += 1
            error_count += int(result.error is not None)
            diagnostic_failure_count += int(
                result.student_call_failed
                or result.guidance_gate_error_count > 0
                or result.answer_judge_failed_count > 0
                or result.teacher_pre_error_count > 0
            )
            mode_counts[result.mode] += 1
            progress.update(1)
            progress.set_postfix(
                errors=error_count,
                diagnostic=diagnostic_failure_count,
                off=mode_counts["presolve_off"],
                on=mode_counts["presolve_on"],
                refresh=False,
            )
            if processed == len(pending) or processed % max(1, log_every) == 0:
                logger.info(
                    "Completed %s/%s pending attempts (termination=%s, id=%s)",
                    processed,
                    len(pending),
                    result.termination_reason,
                    result.item_id,
                )
        return result

    try:
        return await asyncio.gather(*[_run(spec) for spec in pending])
    finally:
        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the TutorAgentWorkflow test split without training, against an "
            "OpenAI-compatible teacher endpoint, with the source config's auxiliary "
            "judge and API students."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--teacher-base-url",
        default=os.getenv("TEACHER_BASE_URL", ""),
        help="OpenAI-compatible teacher endpoint, with or without the /v1 suffix.",
    )
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument(
        "--api-key",
        default="",
        help="Prefer the TEACHER_API_KEY environment variable over this option.",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=None,
        help="Defaults to eval_gconfig.temperature from --config.",
    )
    parser.add_argument(
        "--teacher-top-p",
        type=float,
        default=None,
        help="Defaults to eval_gconfig.top_p from --config.",
    )
    parser.add_argument(
        "--teacher-max-tokens",
        type=int,
        default=None,
        help="Defaults to eval_gconfig.max_new_tokens from --config.",
    )
    parser.add_argument("--teacher-timeout", type=float, default=300.0)
    parser.add_argument("--teacher-request-params", default="")
    parser.add_argument("--teacher-request-params-file", type=Path, default=None)
    parser.add_argument(
        "--self-aux-via-teacher",
        action="store_true",
        help=(
            "For a source config with auxiliary_model.mode=self, reproduce regular "
            "validation by routing auxiliary calls to this external teacher "
            "endpoint with the same request params (including lora_path). Required "
            "when evaluating an actor checkpoint whose self judges must use that "
            "same checkpoint."
        ),
    )
    parser.add_argument(
        "--teacher-presolve",
        choices=["config", "off", "on", "both"],
        default="config",
    )
    parser.add_argument("--presolve-attempts", type=int, default=0)
    parser.add_argument("--presolve-max-tokens", type=int, default=None)
    parser.add_argument(
        "--student-name",
        action="append",
        default=[],
        help=(
            "Evaluate only this configured student; repeat for multiple students. "
            "By default all configured students are evaluated."
        ),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=0,
        help="Attempts per test row; 0 uses evaluator.average_rollouts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Base validation-item limit before student expansion.",
    )
    parser.add_argument(
        "--stratified-max-samples",
        type=int,
        default=0,
        help=(
            "Select this many base rows proportionally by metadata.type x "
            "metadata.level using the config seed; 0 disables stratification."
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split the expanded rows into this many disjoint shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Evaluate the rows whose global index modulo --shard-count is this.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument(
        "--episode-error-retries",
        type=int,
        default=0,
        help=(
            "Retry a whole episode this many additional times after an API or "
            "diagnostic infrastructure failure."
        ),
    )
    parser.add_argument(
        "--episode-error-retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Initial whole-episode retry delay; subsequent delays double.",
    )
    parser.add_argument(
        "--episode-timeout-seconds",
        type=float,
        default=300.0,
        help=(
            "Hard wall-clock limit for one whole episode, including teaching, "
            "judging, and replays. A timeout is handled by the normal whole-episode "
            "retry/backfill policy."
        ),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-diagnostic-backfill-on-resume",
        action="store_true",
        help="Allow only retry_diagnostic_failures false->true on resume; retain other settings.",
    )
    parser.add_argument(
        "--allow-evaluator-code-change-on-resume",
        action="store_true",
        help=(
            "Allow --resume only when evaluator_sha256 is the sole run-signature "
            "difference. All dataset, model, prompt, and generation settings must "
            "still match exactly."
        ),
    )
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On resume, rerun error records by default; use --no-retry-errors to "
            "keep them."
        ),
    )
    parser.add_argument(
        "--retry-diagnostic-failures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Retry whole episodes with student, guidance gate, answer-judge, or "
            "teacher-presolve call failures, and incomplete re-test replays, both "
            "within a run and on resume. Disabled by default to "
            "avoid conditional resampling."
        ),
    )
    parser.add_argument(
        "--save-traces",
        choices=["all", "errors", "none"],
        default="all",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--save-api-requests",
        action="store_true",
        help="Record actual serialized API messages and responses, excluding headers/credentials.",
    )
    parser.add_argument(
        "--keep-env-proxy",
        action="store_true",
        help=(
            "Let OpenAI clients inherit HTTP(S)/ALL_PROXY. By default clients are "
            "created with those variables temporarily cleared, then the environment "
            "is restored."
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label in (
        "teacher_max_tokens",
        "concurrency",
    ):
        if int(getattr(args, label)) <= 0:
            raise ValueError(f"--{label.replace('_', '-')} must be positive.")
    if args.attempts < 0:
        raise ValueError("--attempts must be non-negative.")
    if args.stratified_max_samples < 0:
        raise ValueError("--stratified-max-samples must be non-negative.")
    if args.stratified_max_samples > 0 and args.limit > 0:
        raise ValueError("Use either --stratified-max-samples or --limit, not both.")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count).")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative.")
    if args.episode_error_retries < 0:
        raise ValueError("--episode-error-retries must be non-negative.")
    if args.episode_error_retry_backoff_seconds < 0:
        raise ValueError("--episode-error-retry-backoff-seconds must be non-negative.")
    if args.episode_timeout_seconds <= 0:
        raise ValueError("--episode-timeout-seconds must be positive.")
    if args.presolve_attempts < 0:
        raise ValueError("--presolve-attempts must be non-negative.")
    if args.presolve_max_tokens is not None and args.presolve_max_tokens < 0:
        raise ValueError("--presolve-max-tokens must be non-negative.")
    if float(args.teacher_temperature) < 0.0:
        raise ValueError("--teacher-temperature must be non-negative.")
    if args.teacher_top_p is not None and not 0.0 < float(args.teacher_top_p) <= 1.0:
        raise ValueError("--teacher-top-p must be in (0, 1].")


def resolve_teacher_generation_args(
    args: argparse.Namespace, config: TutorConfig
) -> None:
    eval_gconfig = config.eval_gconfig or config.gconfig
    if args.teacher_temperature is None:
        args.teacher_temperature = float(eval_gconfig.temperature)
    if args.teacher_top_p is None:
        args.teacher_top_p = eval_gconfig.top_p
    if args.teacher_max_tokens is None:
        args.teacher_max_tokens = int(eval_gconfig.max_new_tokens)


async def main_async(args: argparse.Namespace) -> None:
    teacher_base_url = normalize_base_url(args.teacher_base_url)
    env_api_key = os.getenv("TEACHER_API_KEY", "")
    teacher_api_key = env_api_key or args.api_key or "EMPTY"

    config, student_models = load_experiment_config(args.config, args.overrides)
    tutor_train._apply_eval_average_rollouts(config)
    resolve_teacher_generation_args(args, config)
    validate_args(args)
    student_models = select_student_models(student_models, args.student_name)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    dataset = prepare_test_dataset(
        config,
        student_models,
        tokenizer=tokenizer,
        limit=max(0, int(args.limit)),
        stratified_max_samples=int(args.stratified_max_samples),
    )
    modes = resolve_presolve_modes(
        args.teacher_presolve,
        effective_eval_presolve_enabled(config),
    )
    attempts = (
        int(args.attempts)
        if int(args.attempts) > 0
        else max(1, int(config.evaluator.average_rollouts))
    )

    teacher_request_params = merge_dicts(
        merge_dicts(
            teacher_request_defaults(config.seed)
            if config.teacher_response_format == "non_thinking"
            else {},
            config.teacher_api_request_params,
        ),
        load_request_params(
            args.teacher_request_params,
            args.teacher_request_params_file,
            label="--teacher-request-params",
        ),
    )

    external_self_aux = (
        {
            "base_url": teacher_base_url,
            "model": args.teacher_model,
            "api_key": teacher_api_key,
            "request_params": teacher_request_params,
        }
        if args.self_aux_via_teacher
        else None
    )
    workflow_kwargs_by_mode = {
        mode.name: build_eval_workflow_kwargs(
            config=config,
            student_models=student_models,
            tokenizer=tokenizer,
            args=args,
            presolve_enabled=mode.enabled,
            external_self_aux=external_self_aux,
        )
        for mode in modes
    }
    retry_generalization_replays = int(
        next(iter(workflow_kwargs_by_mode.values()))["retest_replays"]
    )
    signature = build_run_signature(
        args=args,
        config=config,
        student_models=student_models,
        modes=modes,
        dataset_size=len(dataset),
        dataset_hash=dataset_sha256(dataset),
        attempts=attempts,
        teacher_base_url=teacher_base_url,
        teacher_request_params=teacher_request_params,
        workflow_kwargs_by_mode=workflow_kwargs_by_mode,
    )
    output_dir = resolve_output_dir(args, config)
    prepare_output_dir(
        output_dir,
        signature=signature,
        resume=args.resume,
        allow_evaluator_code_change=args.allow_evaluator_code_change_on_resume,
        allow_diagnostic_backfill=args.allow_diagnostic_backfill_on_resume,
    )
    results_path = output_dir / "results.jsonl"
    existing_results = load_existing_results(results_path) if args.resume else []
    completed_keys = {
        result.key
        for result in existing_results
        if not result_needs_retry(
            result,
            retry_errors=args.retry_errors,
            retry_diagnostic_failures=args.retry_diagnostic_failures,
            generalization_levels=RETEST_LEVELS,
            expected_generalization_replays=retry_generalization_replays,
        )
    }

    # A shard owns the expanded rows whose global index is congruent to its
    # index, so every shard keeps all students and the shards partition the set.
    shard_rows = range(args.shard_index, len(dataset), args.shard_count)
    specs = [
        EpisodeSpec(
            mode=mode,
            dataset_index=index,
            attempt=attempt,
            row=dict(dataset[index]),
        )
        for mode in modes
        for index in shard_rows
        for attempt in range(1, attempts + 1)
    ]

    async def _run_api_phase() -> list[EpisodeResult]:
        teacher_client = ApiTeacherClient(
            base_url=teacher_base_url,
            api_key=teacher_api_key,
            model=args.teacher_model,
            timeout=args.teacher_timeout,
            max_retries=args.max_retries,
            request_params=teacher_request_params,
        )
        teacher_client.capture_api_requests = args.save_api_requests
        try:
            if not args.skip_preflight:
                await preflight_teacher(teacher_client, args.teacher_model)
            return await run_all(
                specs=specs,
                completed_keys=completed_keys,
                workflow_kwargs_by_mode=workflow_kwargs_by_mode,
                teacher_client=teacher_client,
                output_dir=output_dir,
                save_traces=args.save_traces,
                concurrency=args.concurrency,
                log_every=args.log_every,
                keep_env_proxy=args.keep_env_proxy,
                error_retries=args.episode_error_retries,
                retry_backoff_seconds=args.episode_error_retry_backoff_seconds,
                retry_diagnostic_failures=args.retry_diagnostic_failures,
                episode_timeout_seconds=args.episode_timeout_seconds,
            )
        finally:
            await teacher_client.close()

    new_results = await run_without_proxy_environment(
        _run_api_phase,
        enabled=not args.keep_env_proxy,
    )

    all_results = latest_results([*existing_results, *new_results])
    pending_backfill = [
        result
        for result in all_results
        if result_needs_retry(
            result,
            retry_errors=True,
            retry_diagnostic_failures=True,
            generalization_levels=RETEST_LEVELS,
            expected_generalization_replays=retry_generalization_replays,
        )
    ]
    pending_backfill_path = output_dir / "pending_backfill.jsonl"
    rewrite_results_jsonl(pending_backfill_path, pending_backfill)
    report = aggregate_report(
        all_results,
        modes=modes,
        dataset_size=len(shard_rows),
        attempts=attempts,
    )
    report["output_dir"] = str(output_dir)
    report["pending_backfill"] = {
        "count": len(pending_backfill),
        "path": str(pending_backfill_path),
    }
    report["finished_at"] = datetime.now(UTC).isoformat()
    write_json(output_dir / "summary.json", report)
    logger.info("API teacher evaluation complete: %s", output_dir)
    logger.info("Summary: %s", json.dumps(report["modes"], ensure_ascii=False))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
