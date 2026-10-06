import pathlib
import random
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from typing import Any

from examples.sherpa.configs import TUTOR_EVAL_STUDENT_FIELD, TutorConfig

from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("TutorTrain")


def _without_remote_dataset_loading(dataset_config: Any) -> Any:
    local_config = deepcopy(dataset_config)
    local_config.scheduling_spec = None
    return local_config


def _apply_eval_average_rollouts(config: TutorConfig) -> None:
    if config.eval_gconfig is None:
        config.eval_gconfig = config.gconfig.new()
    config.eval_gconfig = config.eval_gconfig.new(
        n_samples=config.evaluator.average_rollouts
    )


def _resolve_eval_student_names(config: TutorConfig) -> list[str]:
    configured_names = [student.name for student in config.student_models]
    requested_names = config.evaluator.student_model_names
    return configured_names if requested_names is None else list(requested_names)


def _expand_eval_dataset_for_students(dataset: Any, student_names: list[str]) -> Any:
    if not student_names:
        return dataset
    if TUTOR_EVAL_STUDENT_FIELD in dataset.column_names:
        raise ValueError(
            f"Validation dataset already contains reserved column "
            f"{TUTOR_EVAL_STUDENT_FIELD!r}."
        )

    from datasets import concatenate_datasets

    expanded = [
        dataset.add_column(TUTOR_EVAL_STUDENT_FIELD, [name] * len(dataset))
        for name in student_names
    ]
    return concatenate_datasets(expanded)


def build_workflow_kwargs(
    config: TutorConfig,
    *,
    tokenizer: Any | None = None,
    student_models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """TutorAgentWorkflow arguments for a config.

    Shared by training and the standalone evaluator, so the two build the
    workflow the same way. `tokenizer` defaults to config.tokenizer_path and
    `student_models` to every configured student.
    """
    auxiliary_model = config.auxiliary_model
    teacher_pre = config.teacher_pre
    reward = config.reward
    return dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path if tokenizer is None else tokenizer,
        max_turns=config.max_turns,
        enable_thinking=config.enable_thinking,
        teacher_response_format=config.teacher_response_format,
        guidance_gate_mode="masked_continue" if config.guidance_gate else "disabled",
        mask_rejected_turns=config.mask_rejected_turns,
        retest_replays=config.retest_replays,
        aux_mode=auxiliary_model.mode,
        aux_enable_thinking=auxiliary_model.enable_thinking,
        aux_base_url=auxiliary_model.base_url,
        aux_model=auxiliary_model.model,
        aux_api_key=auxiliary_model.api_key,
        aux_timeout=auxiliary_model.timeout,
        aux_max_tokens=auxiliary_model.max_tokens,
        aux_temperature=auxiliary_model.temperature,
        aux_top_p=auxiliary_model.top_p,
        max_concurrent_aux_calls=auxiliary_model.max_concurrent_calls,
        aux_request_params=deepcopy(auxiliary_model.request_params),
        answer_judge_enabled=auxiliary_model.answer_judge_enabled,
        answer_judge_max_tokens=auxiliary_model.answer_judge_max_tokens,
        student_models=(
            [asdict(student) for student in config.student_models]
            if student_models is None
            else deepcopy(student_models)
        ),
        # Also used at eval (eval_workflow_kwargs copies this dict): the preference
        # gate defines the student, so it stays on during evaluation.
        adaptive_gate={
            "prompts_path": config.adaptive_gate_prompts_path,
            "complaints_path": config.adaptive_gate_complaints_path,
            "retries": config.adaptive_gate_retries,
        },
        guidance_gate_fail_penalty=reward.guidance_gate_fail_penalty,
        format_error_penalty=reward.format_error_penalty,
        teacher_exact_repeat_penalty=reward.teacher_exact_repeat_penalty,
        adaptive_gate_fail_penalty=reward.adaptive_gate_fail_penalty,
        soft_overlong_penalty=asdict(reward.soft_overlong),
        length_retry_enabled=config.length_retry.enabled,
        length_retry_attempts=config.length_retry.attempts,
        teacher_pre_enabled=teacher_pre.enabled,
        teacher_pre_verify=teacher_pre.verify,
        teacher_pre_attempts=teacher_pre.attempts,
        teacher_pre_max_tokens=teacher_pre.max_tokens,
        seed=config.seed,
        debug_trace_dir=config.debug_trace_dir or None,
        debug_trace_every_n_rollouts=config.debug_trace_every_n_rollouts,
        max_train_sample_tokens=config.gconfig.max_tokens,
        tokenizer_path=config.tokenizer_path,
        model_context_length=config.sglang.context_length,
    )


def _build_eval_workflow_kwargs(
    workflow_kwargs: dict[str, Any], config: TutorConfig
) -> dict[str, Any]:
    if config.eval_gconfig is None:
        raise ValueError("eval_gconfig must be set before building eval workflow.")
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig.new(n_samples=1)
    eval_workflow_kwargs["eval_repeat_count"] = config.evaluator.average_rollouts
    # Optional eval-only override: no checker verifies the teacher's pre-solve at
    # deployment, so evaluation can skip verification while training keeps its own
    # setting.
    if config.evaluator.teacher_pre_verify is not None:
        eval_workflow_kwargs["teacher_pre_verify"] = bool(
            config.evaluator.teacher_pre_verify
        )
    # Optional eval-only override of whether the teacher drafts a pre-solve at all.
    if config.evaluator.teacher_pre_enabled is not None:
        eval_workflow_kwargs["teacher_pre_enabled"] = bool(
            config.evaluator.teacher_pre_enabled
        )
    return eval_workflow_kwargs


def main(args):
    from areal import PPOTrainer

    config_path = pathlib.Path(args[args.index("--config") + 1])
    has_trial_name_override = any(arg.startswith("trial_name=") for arg in args)
    if not has_trial_name_override:
        trial_name = next(
            line.split(":", 1)[1].strip().strip("'").strip('"')
            for line in config_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("trial_name:")
        )
        args = [*args, f"trial_name={datetime.now():%Y%m%d_%H%M%S}_{trial_name}"]
    config, _ = load_expr_config(args, TutorConfig)
    _apply_eval_average_rollouts(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    # The validation set is expanded per student below, which needs the rows
    # locally rather than behind a remote dataset service.
    valid_dataset_config = config.valid_dataset
    if valid_dataset_config is not None:
        valid_dataset_config = _without_remote_dataset_loading(valid_dataset_config)
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=valid_dataset_config,
        tokenizer=tokenizer,
    )
    eval_max_samples = config.evaluator.max_samples
    if eval_max_samples is not None:
        eval_max_samples = int(eval_max_samples)
        if eval_max_samples <= 0:
            eval_max_samples = None
    if eval_max_samples is not None and eval_max_samples < len(valid_dataset):
        rng = random.Random(config.seed)
        eval_indices = sorted(rng.sample(range(len(valid_dataset)), k=eval_max_samples))
        valid_dataset = valid_dataset.select(eval_indices)
    valid_dataset = _expand_eval_dataset_for_students(
        valid_dataset,
        _resolve_eval_student_names(config),
    )

    workflow_kwargs = build_workflow_kwargs(config)
    eval_workflow_kwargs = _build_eval_workflow_kwargs(workflow_kwargs, config)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
