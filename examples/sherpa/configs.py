import re
from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import (
    MISSING,
    EvaluatorConfig,
    GRPOConfig,
)

_STUDENT_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

TUTOR_EVAL_STUDENT_FIELD = "__tutor_student_name"
TUTOR_TRAIN_STUDENT_FIELD = "__tutor_train_student_name"


@dataclass
class TutorAuxiliaryModelConfig:
    mode: str = field(
        default="self",
        metadata={
            "help": (
                "Auxiliary caller backend: 'api' uses base_url, "
                "'self' uses the actor base model without LoRA."
            ),
            "choices": ["api", "self"],
        },
    )
    enable_thinking: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable thinking mode for self auxiliary calls. "
                "Only affects mode='self'."
            )
        },
    )
    base_url: str = field(
        default="",
        metadata={"help": "OpenAI-compatible API base URL for mode='api'."},
    )
    model: str = field(
        default="",
        metadata={"help": "Model name sent to chat.completions.create in mode='api'."},
    )
    api_key: str = field(default="EMPTY")
    timeout: int = field(default=120)
    max_tokens: int = field(default=1024)
    temperature: float = field(default=0.0)
    top_p: float | None = field(default=1.0)
    max_concurrent_calls: int = field(default=32)
    request_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Additional OpenAI chat.completions.create keyword arguments for "
                "mode='api'. Use extra_body for backend-specific parameters."
            )
        },
    )
    answer_judge_enabled: bool = field(
        default=True,
        metadata={
            "help": (
                "Use this auxiliary model as an LLM fallback judge when exact "
                "answer matching fails."
            )
        },
    )
    answer_judge_max_tokens: int = field(
        default=256,
        metadata={"help": "Maximum completion tokens for answer judge JSON output."},
    )


NO_PREFERENCE = "none"


@dataclass
class TutorStudentModelConfig:
    name: str = field(
        default=MISSING,
        metadata={
            "help": (
                "Unique student identifier used for rollout selection, traces, and "
                "per-student metrics. Use letters, digits, dots, underscores, or dashes."
            )
        },
    )
    base_url: str = field(
        default=MISSING,
        metadata={"help": "OpenAI-compatible API base URL for this student."},
    )
    model: str = field(
        default=MISSING,
        metadata={"help": "Model name sent to chat.completions.create."},
    )
    weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Non-negative training rollout sampling weight. Weights are "
                "normalized across all configured students."
            )
        },
    )
    api_key: str = field(default="EMPTY")
    timeout: int = field(default=120)
    max_tokens: int = field(default=2048)
    temperature: float = field(default=0.7)
    top_p: float | None = field(default=0.8)
    max_concurrent_calls: int = field(default=24)
    # Two entries may share base_url and model and differ only here: that is the
    # intended way to define several students over one served endpoint, and it is
    # why `name` rather than `model` keys the per-student metrics.
    preference: str = field(
        default="",
        metadata={
            "help": (
                "What this student requires of the TEACHER'S MANNER before it will "
                "engage: the name keys a preference prompt in "
                "adaptive_gate_prompts_path, and on every teacher turn an auxiliary "
                "model is asked that prompt about the teacher's message. FAIL means "
                "the student does not answer at all -- an injected complaint takes "
                "its slot and the turn is spent. Empty or 'none' leaves the gate "
                "open and costs no call.\n\n"
                "Each value is a learner type from prior work rather than a style "
                "chosen here; see adaptive_gate_prompts_path for the per-value "
                "citation. This is NOT a prompted persona: nothing asks the student "
                "to act a certain way. The requirement is enforced structurally, by "
                "withholding the student's engagement."
            )
        },
    )
    request_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Additional OpenAI chat.completions.create keyword arguments. "
                "Use extra_body for backend-specific parameters."
            )
        },
    )

    def __post_init__(self) -> None:
        missing = [
            field_name
            for field_name in ("name", "base_url", "model")
            if getattr(self, field_name) is None
            or getattr(self, field_name) is MISSING
            or str(getattr(self, field_name)).strip() in {"", "???"}
        ]
        if missing:
            raise ValueError(
                "student_models entries require non-empty values for: "
                f"{', '.join(missing)}."
            )

        self.name = str(self.name).strip()
        self.base_url = str(self.base_url).strip()
        self.model = str(self.model).strip()
        if not _STUDENT_MODEL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                "student_models.name must start with a letter or digit and contain "
                "only letters, digits, dots, underscores, or dashes."
            )
        if not self.base_url or not self.model:
            raise ValueError("student_models base_url and model must be non-empty.")
        self.weight = float(self.weight)
        if self.weight < 0.0:
            raise ValueError("student_models.weight must be non-negative.")
        self.timeout = int(self.timeout)
        if self.timeout <= 0:
            raise ValueError("student_models.timeout must be positive.")
        self.max_tokens = int(self.max_tokens)
        if self.max_tokens <= 0:
            raise ValueError("student_models.max_tokens must be positive.")
        self.max_concurrent_calls = int(self.max_concurrent_calls)
        if self.max_concurrent_calls <= 0:
            raise ValueError("student_models.max_concurrent_calls must be positive.")
        self.preference = str(self.preference or "").strip()


@dataclass
class TutorLengthRetryConfig:
    """Resample a teacher turn that ran to the generation cap.

    A turn that stops on 'length' never closed its tags, so it is a format error by
    construction, and under a token-mean loss it carries its whole length into the
    gradient.

    Retrying removes such a sample instead of shrinking it: the discarded draft
    never reaches the batch, and the episode is not terminated, so it keeps the
    re-test that a format termination would forfeit. When every attempt still hits
    the cap the last one is kept and the normal format handling applies.
    """

    enabled: bool = field(
        default=True,
        metadata={
            "help": (
                "Resample a teacher turn whose generation stopped on the token "
                "limit, up to `attempts` times. Discarded drafts never enter "
                "training. Off keeps the first draft, whatever it is."
            )
        },
    )
    attempts: int = field(
        default=3,
        metadata={
            "help": (
                "Total teacher generations allowed for one turn, the first "
                "included. 1 disables retrying even with enabled true. Each "
                "attempt is a full generation at gconfig.max_new_tokens."
            )
        },
    )


@dataclass
class TutorTeacherPreConfig:
    enabled: bool = field(
        default=True,
        metadata={
            "help": (
                "Privately ask the teacher to solve the task before tutoring. "
                "An accepted solution is hidden from the student and kept in the "
                "teacher's context as a private reference. One draft is shared by "
                "all rollouts of a problem at one weight version."
            )
        },
    )
    verify: bool = field(
        default=True,
        metadata={
            "help": (
                "Judge teacher pre-solve drafts and skip the rollout unless one "
                "is correct. When disabled, generate exactly one unverified draft "
                "and continue without calling the answer judge."
            )
        },
    )
    attempts: int = field(
        default=3,
        metadata={
            "help": (
                "Maximum teacher pre-solve attempts when verification is enabled. "
                "If no attempt is correct, the problem's whole rollout group is "
                "skipped for this step."
            )
        },
    )
    max_tokens: int = field(
        default=4096,
        metadata={
            "help": (
                "Maximum completion tokens for teacher pre-solve. Non-positive "
                "values reuse the tutor rollout max_new_tokens."
            )
        },
    )

    def __post_init__(self) -> None:
        if int(self.attempts) < 1:
            raise ValueError("teacher_pre.attempts must be >= 1.")


@dataclass
class TutorEvaluatorConfig(EvaluatorConfig):
    max_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of validation samples to evaluate. "
                "None or non-positive values evaluate the full validation set."
            )
        },
    )
    teacher_pre_verify: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Override teacher_pre.verify during evaluation only. None keeps "
                "eval identical to training. False evaluates under the deployment "
                "condition: the teacher pre-solve produces one unverified draft, "
                "the answer judge is not called, and no sample is skipped for "
                "having failed it. Training can still verify, while evaluation "
                "does not depend on a checker that is unavailable at deployment."
            )
        },
    )
    teacher_pre_enabled: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Override teacher_pre.enabled during evaluation only. None keeps eval "
                "identical to training, which with teacher_pre_verify False is the "
                "usual setting: the teacher drafts a solution but nothing checks it. "
                "False evaluates a teacher that never drafts at all. True forces the "
                "draft on where training had it off. Together with "
                "teacher_pre_verify this spans the three eval regimes: "
                "draft-unverified, draft-verified, and no draft."
            )
        },
    )
    average_rollouts: int = field(
        default=3,
        metadata={
            "help": (
                "Number of independent tutor rollout episodes to run for each "
                "validation sample. Metrics are averaged over rollout attempts, and "
                "per-task correctness stability is reported under eval-rollout/repeat."
            )
        },
    )
    student_model_names: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "Optional subset of student_models to evaluate. None evaluates every "
                "configured student model."
            )
        },
    )

    def __post_init__(self) -> None:
        self.average_rollouts = int(self.average_rollouts)
        if self.average_rollouts < 1:
            raise ValueError("evaluator.average_rollouts must be >= 1.")
        if self.student_model_names is None:
            return

        self.student_model_names = [
            str(name).strip() for name in self.student_model_names
        ]
        if not self.student_model_names or any(
            not name for name in self.student_model_names
        ):
            raise ValueError(
                "evaluator.student_model_names must contain non-empty names."
            )
        if len(self.student_model_names) != len(set(self.student_model_names)):
            raise ValueError("evaluator.student_model_names must be unique.")


@dataclass
class TutorSoftOverlongPenaltyConfig:
    """DAPO-style soft penalty near the teacher generation-token limit."""

    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable a linear per-turn penalty over the final buffer_tokens "
                "of gconfig.max_new_tokens. The generated-token count includes "
                "the teacher's reasoning and visible output."
            )
        },
    )
    buffer_tokens: int = field(default=512)
    max_penalty: float = field(
        default=-0.05,
        metadata={
            "help": (
                "Penalty at gconfig.max_new_tokens. Intermediate penalties are "
                "linearly interpolated from zero at max_new_tokens-buffer_tokens."
            )
        },
    )

    def __post_init__(self) -> None:
        self.buffer_tokens = int(self.buffer_tokens)
        self.max_penalty = float(self.max_penalty)
        if self.buffer_tokens <= 0:
            raise ValueError("reward.soft_overlong.buffer_tokens must be positive.")
        if self.max_penalty > 0.0:
            raise ValueError("reward.soft_overlong.max_penalty must be <= 0.")
        if self.enabled and self.max_penalty == 0.0:
            raise ValueError(
                "reward.soft_overlong.enabled=true requires max_penalty < 0."
            )


@dataclass
class TutorRewardConfig:
    """Per-turn penalties. The re-test reward itself has no knobs.

    Every penalty here belongs to the turn that raised it: it is kept out of the
    return that ReBN accumulates backward, and added to that turn's advantage after
    normalization, in normalized-advantage units.
    """

    guidance_gate_fail_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Charged once per episode, on the first teacher turn that fails "
                "the guidance gate."
            )
        },
    )
    format_error_penalty: float = field(
        default=-0.5,
        metadata={
            "help": (
                "Charged on a teacher turn whose output cannot be parsed; the "
                "episode ends at that turn. Must be <= 0."
            )
        },
    )
    teacher_exact_repeat_penalty: float = field(
        default=-0.5,
        metadata={
            "help": (
                "Charged on a teacher reply that exactly matches any earlier "
                "student-visible teacher reply after whitespace normalization; the "
                "episode ends before the student is called. Must be < 0."
            )
        },
    )
    adaptive_gate_fail_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Charged on a teacher turn rejected by the adaptive gate. "
                "Must be <= 0; 0 disables it."
            )
        },
    )
    soft_overlong: TutorSoftOverlongPenaltyConfig = field(
        default_factory=TutorSoftOverlongPenaltyConfig
    )

    def __post_init__(self) -> None:
        if self.format_error_penalty > 0.0:
            raise ValueError("reward.format_error_penalty must be <= 0.")
        self.teacher_exact_repeat_penalty = float(self.teacher_exact_repeat_penalty)
        if self.teacher_exact_repeat_penalty >= 0.0:
            raise ValueError(
                "reward.teacher_exact_repeat_penalty must be < 0: an exact repeat "
                "ends the episode and this is what that turn is charged."
            )
        self.adaptive_gate_fail_penalty = float(self.adaptive_gate_fail_penalty)
        if self.adaptive_gate_fail_penalty > 0.0:
            raise ValueError("reward.adaptive_gate_fail_penalty must be <= 0.")


@dataclass
class TutorConfig(GRPOConfig):
    workflow: str = field(
        default="examples.sherpa.workflow.SherpaWorkflow",
        metadata={"help": "Training workflow import path."},
    )
    eval_workflow: str = field(
        default="examples.sherpa.workflow.SherpaWorkflow",
        metadata={"help": "Evaluation workflow import path."},
    )
    max_turns: int = field(
        default=10,
        metadata={
            "help": (
                "Number of (teacher, student) rounds. The teacher is told this "
                "budget in its system prompt."
            )
        },
    )
    teacher_response_format: str = field(
        default="non_thinking",
        metadata={
            "help": "Teacher reply contract, independent of native model thinking.",
            "choices": ["non_thinking", "thinking"],
        },
    )
    teacher_api_request_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "External teacher API options, e.g. reasoning_effort; never used for student/judge calls."
        },
    )
    enable_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable thinking mode for the tutor rollout model."
        },
    )
    guidance_gate: bool = field(
        default=True,
        metadata={
            "help": (
                "Run the guidance gate before each student call: a teacher turn that "
                "fails it is hidden from the real student and re-test, a fixed user "
                "reply goes into teacher-only history, and the dialogue continues. "
                "False skips the gate."
            ),
        },
    )
    adaptive_gate: bool = field(
        default=True,
        metadata={
            "help": (
                "Run the adaptive gate on each teacher turn that passed the guidance "
                "gate: a student with a preference answers only if the turn meets "
                "it; otherwise a complaint takes its slot and the gated exchange "
                "stays in the teacher's history only. It runs at evaluation too, "
                "and never during the re-test. False requires every student's "
                "preference to be 'none'."
            ),
        },
    )
    adaptive_gate_prompts_path: str = field(
        default="",
        metadata={
            "help": (
                "JSON of the preference prompts: {preferences: {name: {source, "
                "preference}}}. `source` cites the work the category comes from and "
                "is never shown to the model. Required as soon as any student names "
                "a preference."
            )
        },
    )
    adaptive_gate_complaints_path: str = field(
        default="",
        metadata={
            "help": (
                "JSON of the student's replies to a failed turn: {explain: {name: "
                "[...]}}. Each reply names the remedy the student wants, so the "
                "lists are per-preference."
            )
        },
    )
    # The gate reply is bounded by auxiliary_model.max_tokens, which must leave room
    # for the reasoning that precedes the verdict.
    adaptive_gate_retries: int = field(
        default=3,
        metadata={
            "help": (
                "Retries before an unclean adaptive-gate verdict becomes FAIL. "
                "Invalid replies, API errors, and timeouts take this path. FAIL is "
                "the conservative default."
            )
        },
    )
    mask_rejected_turns: bool = field(
        default=True,
        metadata={
            "help": (
                "Credit the re-test reward only to teacher turns the student "
                "actually saw -- turns that failed the adaptive gate or the guidance "
                "gate get none of it -- and compare turns against a per-turn "
                "group baseline (actor.group_baseline='turn'). False credits every "
                "turn: all turns of an episode share its return and one group "
                "baseline (actor.group_baseline='episode')."
            )
        },
    )
    retest_replays: int = field(
        default=8,
        metadata={
            "help": (
                "How many times the student re-solves the task after the "
                "conversation, and how many samples estimate the no-teaching "
                "baseline. The reward is the fraction correct minus that baseline."
            )
        },
    )
    teacher_pre: TutorTeacherPreConfig = field(default_factory=TutorTeacherPreConfig)
    length_retry: TutorLengthRetryConfig = field(default_factory=TutorLengthRetryConfig)
    auxiliary_model: TutorAuxiliaryModelConfig = field(
        default_factory=TutorAuxiliaryModelConfig
    )
    student_models: list[TutorStudentModelConfig] = field(
        default_factory=list,
        metadata={
            "help": (
                "Pool of API students. Each training episode draws one, in fixed "
                "proportions per batch."
            )
        },
    )
    evaluator: TutorEvaluatorConfig = field(default_factory=TutorEvaluatorConfig)
    reward: TutorRewardConfig = field(default_factory=TutorRewardConfig)
    debug_trace_dir: str = field(
        default="",
        metadata={
            "help": "Optional directory to dump readable per-rollout tutor traces."
        },
    )
    debug_trace_every_n_rollouts: int = field(
        default=10,
        metadata={"help": "Dump one readable tutor trace every N rollout episodes."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.teacher_response_format not in {"non_thinking", "thinking"}:
            raise ValueError("teacher_response_format must be non_thinking or thinking")
        self.adaptive_gate_prompts_path = str(self.adaptive_gate_prompts_path or "").strip()
        self.adaptive_gate_complaints_path = str(
            self.adaptive_gate_complaints_path or ""
        ).strip()
        self.adaptive_gate_retries = int(self.adaptive_gate_retries)
        if self.adaptive_gate_retries < 1:
            raise ValueError(
                "adaptive_gate_retries must be at least 1, got "
                f"{self.adaptive_gate_retries}."
            )
        if not self.adaptive_gate:
            gated = [
                student.name
                for student in self.student_models
                if student.preference not in ("", NO_PREFERENCE)
            ]
            if gated:
                raise ValueError(
                    "adaptive_gate is false, but these students have a preference: "
                    f"{gated}. Set their preference to 'none' or remove them."
                )
        if self.actor.mask_no_eos_with_zero:
            raise ValueError(
                "Tutor does not support actor.mask_no_eos_with_zero because its "
                "turn-level tensors are dynamically padded."
            )
        if (
            self.reward.soft_overlong.enabled
            and self.reward.soft_overlong.buffer_tokens
            >= int(self.gconfig.max_new_tokens)
        ):
            raise ValueError(
                "reward.soft_overlong.buffer_tokens must be smaller than "
                "gconfig.max_new_tokens."
            )
        student_names = [student.name for student in self.student_models]
        if len(student_names) != len(set(student_names)):
            raise ValueError("student_models names must be unique.")
        eval_student_names = self.evaluator.student_model_names
        if eval_student_names is not None:
            unknown_eval_students = sorted(set(eval_student_names) - set(student_names))
            if unknown_eval_students:
                raise ValueError(
                    "evaluator.student_model_names must reference configured "
                    f"student_models; unknown names: {unknown_eval_students}."
                )
        if not any(student.weight > 0.0 for student in self.student_models):
            raise ValueError(
                "student_models must contain at least one student with positive weight."
            )
        # The turn-local penalties and the group baselines below exist only in the
        # ReBN estimator.
        if self.actor.advantage_estimator != "rebn":
            raise ValueError("Tutor requires actor.advantage_estimator='rebn'.")
        # One switch decides both whether rejected turns are masked out of the
        # re-test credit and which group baseline the actor uses, so the two cannot
        # be set inconsistently.
        group_baseline = "turn" if self.mask_rejected_turns else "episode"
        if self.actor.group_baseline is None:
            self.actor.group_baseline = group_baseline
        elif self.actor.group_baseline != group_baseline:
            raise ValueError(
                f"mask_rejected_turns={self.mask_rejected_turns} implies "
                f"actor.group_baseline={group_baseline!r}; leave "
                "actor.group_baseline unset."
            )
        if self.mask_rejected_turns:
            if not self.actor.group_baseline_leave1out:
                raise ValueError(
                    "mask_rejected_turns requires actor.group_baseline_leave1out=true."
                )
            if (
                self.actor.adv_norm is not None
                and self.actor.adv_norm.mean_level is not None
            ):
                raise ValueError(
                    "mask_rejected_turns requires actor.adv_norm.mean_level=null so "
                    "the explicit group-relative credit is not mean-centered a "
                    "second time."
                )
