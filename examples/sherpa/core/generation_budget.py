from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

CONTEXT_BUDGET_TERMINATION_REASON = "context budget limit"


@dataclass(frozen=True, slots=True)
class TrainSampleBudgetResult:
    input_len: int
    max_train_sample_tokens: int | None
    remaining_tokens: int | None
    max_new_tokens: int
    gconfig: Any | None
    over_budget: bool


class ContextBudgetLimitExceeded(RuntimeError):
    pass


def prepare_train_sample_generation_config(
    *,
    input_ids: list[int],
    gconfig: Any | None,
    max_completion_tokens: int,
    max_train_sample_tokens: int | None,
) -> TrainSampleBudgetResult:
    input_len = len(input_ids)
    requested_max_new_tokens = _requested_max_new_tokens(
        gconfig,
        max_completion_tokens=max_completion_tokens,
    )
    if max_train_sample_tokens is None:
        return TrainSampleBudgetResult(
            input_len=input_len,
            max_train_sample_tokens=None,
            remaining_tokens=None,
            max_new_tokens=requested_max_new_tokens,
            gconfig=with_max_new_tokens(gconfig, requested_max_new_tokens),
            over_budget=False,
        )

    remaining_tokens = int(max_train_sample_tokens) - input_len
    if remaining_tokens <= 0:
        return TrainSampleBudgetResult(
            input_len=input_len,
            max_train_sample_tokens=int(max_train_sample_tokens),
            remaining_tokens=remaining_tokens,
            max_new_tokens=0,
            gconfig=None,
            over_budget=True,
        )

    max_new_tokens = min(requested_max_new_tokens, remaining_tokens)
    return TrainSampleBudgetResult(
        input_len=input_len,
        max_train_sample_tokens=int(max_train_sample_tokens),
        remaining_tokens=remaining_tokens,
        max_new_tokens=max_new_tokens,
        gconfig=with_max_new_tokens(gconfig, max_new_tokens),
        over_budget=False,
    )


def ensure_response_within_train_sample_budget(
    *,
    input_len: int,
    output_len: int,
    max_train_sample_tokens: int | None,
) -> None:
    if max_train_sample_tokens is None:
        return
    total_len = int(input_len) + int(output_len)
    if total_len > int(max_train_sample_tokens):
        raise ContextBudgetLimitExceeded(
            f"tutor train sample length exceeded context budget: "
            f"{total_len} > {int(max_train_sample_tokens)}"
        )


def raise_if_over_budget(result: TrainSampleBudgetResult) -> None:
    if not result.over_budget:
        return
    raise ContextBudgetLimitExceeded(
        f"tutor prompt exceeded context budget: "
        f"{result.input_len} >= {result.max_train_sample_tokens}"
    )


def _requested_max_new_tokens(
    gconfig: Any | None,
    *,
    max_completion_tokens: int,
) -> int:
    raw_max_new_tokens = (
        getattr(gconfig, "max_new_tokens", None) if gconfig is not None else None
    )
    if raw_max_new_tokens is None:
        raw_max_new_tokens = max_completion_tokens
    return max(1, int(raw_max_new_tokens))


def with_max_new_tokens(gconfig: Any | None, max_new_tokens: int) -> Any | None:
    """A copy of gconfig with a different generation cap, or None if there is none."""
    if gconfig is None:
        return None
    if hasattr(gconfig, "new"):
        return gconfig.new(max_new_tokens=int(max_new_tokens))
    copied = copy.copy(gconfig)
    setattr(copied, "max_new_tokens", int(max_new_tokens))
    return copied
