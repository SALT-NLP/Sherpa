from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import GRPOConfig, PPOActorConfig


@dataclass
class PedagogicalAPIModelConfig:
    """Frozen OpenAI-compatible model used by the classroom workflow."""

    # ``api`` calls an OpenAI-compatible server. ``self`` uses the rollout
    # engine's base model with the trainable LoRA disabled.  The student remains
    # an API client even in offline runs because the launcher serves it locally.
    mode: str = "api"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout: float = 120.0
    max_retries: int = 2
    max_concurrent_calls: int = 4
    seed: int | None = None
    top_k: int = 20
    min_p: float = 0.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": False}
    )

    def __post_init__(self) -> None:
        if self.mode not in {"api", "self"}:
            raise ValueError("model mode must be 'api' or 'self'")
        if self.max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class PedagogicalGenerationConfig:
    max_teacher_turns: int = 10
    max_tokens_in_conversation: int = 24576
    max_tokens_per_teacher_turn: int = 4096
    max_tokens_per_student_turn: int = 2048
    max_tokens_per_student_attempt: int = 2048
    max_tokens_per_judge_attempt: int = 1024
    number_student_attempts: int = 8
    number_judge_attempts: int = 2
    student_temperature: float = 0.7
    student_top_p: float = 0.8
    judge_temperature: float = 0.0
    judge_top_p: float = 1.0
    extra_penalty_for_rejected_judges: float = 1.0
    format_error_penalty: float = -0.5

    def __post_init__(self) -> None:
        positive_fields = {
            "max_teacher_turns": self.max_teacher_turns,
            "max_tokens_in_conversation": self.max_tokens_in_conversation,
            "max_tokens_per_teacher_turn": self.max_tokens_per_teacher_turn,
            "max_tokens_per_student_turn": self.max_tokens_per_student_turn,
            "max_tokens_per_student_attempt": self.max_tokens_per_student_attempt,
            "max_tokens_per_judge_attempt": self.max_tokens_per_judge_attempt,
            "number_student_attempts": self.number_student_attempts,
            "number_judge_attempts": self.number_judge_attempts,
        }
        invalid = [name for name, value in positive_fields.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"generation values must be positive: {invalid}")
        if self.format_error_penalty > 0.0:
            raise ValueError("generation.format_error_penalty must be <= 0")


@dataclass
class PedagogicalActorConfig(PPOActorConfig):
    """PPO actor with PedagogicalRL's rollout reuse count."""

    behave_imp_weight_cap: float | None = None
    behave_imp_weight_mode: str = "disabled"
    num_iterations: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_iterations < 1:
            raise ValueError("actor.num_iterations must be positive")


@dataclass
class PedagogicalRLConfig(GRPOConfig):
    workflow: str = "examples.pedagogical_rl.workflow.PedagogicalRLWorkflow"
    actor: PedagogicalActorConfig = field(default_factory=PedagogicalActorConfig)
    student_model: PedagogicalAPIModelConfig = field(
        default_factory=PedagogicalAPIModelConfig
    )
    judge_model: PedagogicalAPIModelConfig = field(
        default_factory=PedagogicalAPIModelConfig
    )
    generation: PedagogicalGenerationConfig = field(
        default_factory=PedagogicalGenerationConfig
    )
    debug_trace_dir: str = ""
    debug_trace_every_n_rollouts: int = 10
    max_train_examples: int = -1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.debug_trace_every_n_rollouts < 1:
            raise ValueError("debug_trace_every_n_rollouts must be positive")
        if self.max_train_examples == 0 or self.max_train_examples < -1:
            raise ValueError("max_train_examples must be -1 or positive")
        if self.valid_dataset is not None:
            raise ValueError(
                "valid_dataset must be null: trained checkpoints are evaluated "
                "with the Sherpa evaluation protocol (examples/sherpa/eval)"
            )
        if self.actor.kl_ctl < 0.0:
            raise ValueError("actor.kl_ctl (PedagogicalRL beta) must be >= 0")
        if self.critic is not None:
            raise ValueError("the PedagogicalRL baseline does not use a critic")
        if self.actor.kl_ctl > 0.0 and self.ref is None:
            raise ValueError(
                "actor.kl_ctl > 0 requires a frozen ref model; for a LoRA actor "
                "configure ref as the same base checkpoint with use_lora=false"
            )
        if self.actor.kl_ctl == 0.0 and self.ref is not None:
            raise ValueError("ref is unnecessary when actor.kl_ctl=0")
