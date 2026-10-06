from __future__ import annotations

from typing import Any

import torch


def response_to_tensordict(
    response: Any,
    *,
    reward: float,
    local_reward: float | None = None,
    gate_masked_reward: float | None = None,
    gate_credit_mask: bool | None = None,
    trajectory_id: int | None = None,
    turn_idx: int | None = None,
) -> dict[str, torch.Tensor]:
    input_tokens = list(response.input_tokens)
    input_len = len(input_tokens)
    output_tokens = list(response.output_tokens)
    output_len = len(output_tokens)
    full_ids = input_tokens + output_tokens
    output_logprobs = list(response.output_logprobs)
    if len(output_logprobs) != output_len:
        raise ValueError(
            "ModelResponse output_logprobs length mismatch: "
            f"output_tokens={output_len}, "
            f"output_logprobs={len(output_logprobs)}, "
            f"stop_reason={getattr(response, 'stop_reason', None)!r}."
        )
    output_versions = list(response.output_versions)
    if len(output_versions) != output_len:
        raise ValueError(
            "ModelResponse output_versions length mismatch: "
            f"output_tokens={output_len}, "
            f"output_versions={len(output_versions)}, "
            f"stop_reason={getattr(response, 'stop_reason', None)!r}."
        )
    trajectory_value = 0 if trajectory_id is None else int(trajectory_id)
    turn_value = 0 if turn_idx is None else int(turn_idx)
    if (gate_masked_reward is None) != (gate_credit_mask is None):
        raise ValueError(
            "gate_masked_reward and gate_credit_mask must be provided together."
        )
    effective_reward = float(reward)
    result = {
        "input_ids": torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
        "logprobs": torch.tensor(
            [0.0] * input_len + output_logprobs,
            dtype=torch.float32,
        ).unsqueeze(0),
        "loss_mask": torch.tensor(
            [0] * input_len + [1] * output_len,
            dtype=torch.long,
        ).unsqueeze(0),
        "versions": torch.tensor(
            [-1] * input_len + output_versions,
            dtype=torch.long,
        ).unsqueeze(0),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.bool).unsqueeze(0),
        "rewards": torch.tensor([effective_reward], dtype=torch.float32),
        "trajectory_id": torch.tensor([trajectory_value], dtype=torch.long),
        "turn_idx": torch.tensor([turn_value], dtype=torch.long),
    }
    if local_reward is not None:
        effective_local_reward = float(local_reward)
        result["local_rewards"] = torch.tensor(
            [effective_local_reward], dtype=torch.float32
        )
    if gate_masked_reward is not None:
        effective_gate_reward = float(gate_masked_reward)
        result["gate_masked_rewards"] = torch.tensor(
            [effective_gate_reward], dtype=torch.float32
        )
        result["gate_credit_mask"] = torch.tensor(
            [bool(gate_credit_mask)], dtype=torch.bool
        )
    return result
