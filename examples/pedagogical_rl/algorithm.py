from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from areal import PPOTrainer
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import PPOActorConfig
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.trainer.ppo.actor import PPOActor
from areal.utils import stats_tracker
from areal.utils.environ import is_single_controller


def pedagogical_grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict[str, Any],
    *,
    ppo_loss_fn: Callable[..., torch.Tensor],
    beta: float,
    **loss_kwargs: Any,
) -> torch.Tensor:
    """Add upstream PedagogicalRL's sampled forward-KL term to AReaL GRPO.

    Upstream applies ``beta * (exp(ref-logp) - (ref-logp) - 1)`` directly in
    the token loss.  It is intentionally not folded into rewards or advantages:
    doing so would change the group baseline and would no longer be their
    algorithm.
    """

    loss = ppo_loss_fn(logprobs, entropy, input_data, **loss_kwargs)
    if beta == 0.0:
        return loss
    ref_logp = input_data.get("ref_logp")
    if ref_logp is None:
        raise ValueError("PedagogicalRL beta > 0 requires ref_logp")
    loss_mask = input_data["loss_mask"].bool()
    delta = ref_logp.detach().to(logprobs.dtype) - logprobs
    per_token_kl = torch.exp(delta) - delta - 1.0
    token_count = loss_mask.count_nonzero().clamp_min(1)
    kl_loss = torch.where(loss_mask, per_token_kl, 0.0).sum() / token_count
    stats_tracker.stat(
        pedagogical_reference_kl=per_token_kl.detach(),
        denominator="n_valid_tokens",
    )
    stats_tracker.scalar(pedagogical_beta=float(beta))
    return loss + float(beta) * kl_loss


class PedagogicalDistributedSampler(DistributedSampler):
    """Reproduce PedagogicalRL's stateful seeded permutations across epochs."""

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            for _ in range(self.epoch + 1):
                indices = torch.randperm(
                    len(self.dataset), generator=generator
                ).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * ((padding_size // len(indices)) + 1))[
                    :padding_size
                ]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(
                f"sampler produced {len(indices)} samples; expected {self.num_samples}"
            )
        return iter(indices)


class PedagogicalPPOActor(PPOActor):
    """GRPO advantage used by PedagogicalRL.

    One group-normalized episode advantage is broadcast to every trainable
    teacher token. Student, judge, and final-solution tokens never enter the
    teacher trajectory and therefore never receive loss.
    """

    def _compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        reward_score = (data["rewards"] + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, min=-self.reward_clip, max=self.reward_clip
        )
        if self.reward_norm is not None:
            reward_score = self.reward_norm(reward_score)

        loss_mask = torch.roll(data["loss_mask"].float(), shifts=-1, dims=-1)
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            old_logp = data.get("prox_logp")
            if old_logp is None:
                raise ValueError("prox_logp is required when recompute_logprob=True")
            data["logprobs"] = old_logp
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                data["prox_logp"] = old_logp
        old_logp = old_logp * loss_mask

        ref_logp = data.get("ref_logp")
        if float(getattr(self.config, "kl_ctl", 0.0)) > 0.0:
            if ref_logp is None:
                raise ValueError("PedagogicalRL beta > 0 requires ref_logp")
            # compute_logp returns the same next-token alignment as prox_logp.
            data["ref_logp"] = ref_logp * loss_mask

        advantages = reward_score.float().unsqueeze(-1).expand_as(loss_mask)
        advantages = advantages * loss_mask
        token_rewards = torch.zeros_like(advantages)
        attention_lengths = data["attention_mask"].sum(-1).long()
        batch_indices = torch.arange(
            attention_lengths.shape[0], device=attention_lengths.device
        )
        terminal_indices = torch.clamp(attention_lengths - 2, min=0)
        token_rewards[batch_indices, terminal_indices] = reward_score.float()

        data["advantages"] = advantages
        data["returns"] = advantages
        data["kl_rewards"] = torch.zeros_like(advantages)
        data["tot_rewards"] = token_rewards
        data["loss_mask"] = loss_mask
        data["logprobs"] = old_logp
        return data


class PedagogicalFSDPPPOActor(FSDPPPOActor):
    """FSDP PPO actor that performs PedagogicalRL's μ full-batch updates."""

    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PedagogicalPPOActor(config, self)

    def train_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Wrap AReaL's policy loss without changing its training engine."""

        return super().train_batch(
            input_,
            loss_fn=functools.partial(
                pedagogical_grpo_loss_fn,
                ppo_loss_fn=loss_fn,
                beta=float(self.config.kl_ctl),
            ),
            loss_weight_fn=loss_weight_fn,
        )

    def ppo_update(self, data: list[dict[str, Any]]) -> None:
        iterations = int(getattr(self.config, "num_iterations", 1))
        for iteration in range(iterations):
            self.actor.ppo_update(data)
            # The outer AReaL trainer steps once after this method. Step between
            # internal updates so μ updates also correspond to μ scheduler steps.
            if iteration + 1 < iterations:
                self.lr_scheduler_step()


class PedagogicalPPOTrainer(PPOTrainer):
    """PPO trainer selecting the example-local FSDP algorithm adapter."""

    def _create_dataloader(
        self,
        dataset,
        dataset_config,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        if dataset_config is not self.config.train_dataset:
            return super()._create_dataloader(
                dataset,
                dataset_config=dataset_config,
                rank=rank,
                world_size=world_size,
            )
        if dataset_config.batch_size % world_size != 0:
            raise ValueError(
                f"batch size({dataset_config.batch_size}) must be divisible by "
                f"world_size({world_size})"
            )
        sampler = PedagogicalDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=dataset_config.shuffle,
            seed=self.config.seed,
            drop_last=True,
        )
        return StatefulDataLoader(
            dataset,
            batch_size=dataset_config.batch_size // world_size,
            sampler=sampler,
            drop_last=dataset_config.drop_last,
            num_workers=dataset_config.num_workers,
            collate_fn=lambda rows: rows,
        )

    def _create_train_engine(
        self, actor_config: PPOActorConfig, alloc: ModelAllocation
    ) -> Any:
        if alloc.backend != "fsdp":
            raise ValueError(
                "the PedagogicalRL μ-update actor supports only actor.backend=fsdp"
            )
        if is_single_controller():
            actor = PedagogicalFSDPPPOActor.as_controller(actor_config, self.scheduler)
        else:
            actor = PedagogicalFSDPPPOActor(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor
