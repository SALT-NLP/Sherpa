from __future__ import annotations

import json
import os
import socket
import uuid
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from examples.pedagogical_rl.api import (
    PedagogicalAPIClient,
    PedagogicalEngineClient,
)
from examples.pedagogical_rl.config import (
    PedagogicalAPIModelConfig,
    PedagogicalGenerationConfig,
)
from examples.pedagogical_rl.judges import run_whole_dialogue_judge
from examples.pedagogical_rl.prompts import WHOLE_DIALOGUE_JUDGE_PROMPTS
from examples.pedagogical_rl.scoring import native_answer_correct
from examples.pedagogical_rl.state import (
    ClassroomEpisode,
    ConversationType,
    NativeJudgeDecision,
)

from areal import workflow_context
from areal.api import RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.experimental.openai import ArealOpenAI
from areal.utils import logging, stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("PedagogicalRLWorkflow")


def _coerce_config(config_cls: type, value: Any) -> Any:
    if isinstance(value, config_cls):
        return value
    if is_dataclass(value):
        value = asdict(value)
    elif not isinstance(value, dict):
        try:
            from omegaconf import OmegaConf

            value = OmegaConf.to_container(value, resolve=True)
        except Exception:
            pass
    if not isinstance(value, dict):
        raise TypeError(
            f"cannot convert {type(value).__name__} to {config_cls.__name__}"
        )
    return config_cls(**value)


class PedagogicalRLWorkflow(RolloutWorkflow):
    """PedagogicalRL's classroom method running on an AReaL teacher actor.

    Training uses PedagogicalRL's two whole-dialogue judges as hard gates.
    Trained checkpoints are evaluated with the tutor evaluation protocol.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: Any,
        student_model: PedagogicalAPIModelConfig | dict[str, Any],
        judge_model: PedagogicalAPIModelConfig | dict[str, Any],
        generation: PedagogicalGenerationConfig | dict[str, Any],
        debug_trace_dir: str = "",
        debug_trace_every_n_rollouts: int = 10,
        student_client: PedagogicalAPIClient | None = None,
        judge_client: PedagogicalAPIClient | None = None,
        actor_client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.gconfig = gconfig
        self.tokenizer = (
            load_hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.student_model = _coerce_config(PedagogicalAPIModelConfig, student_model)
        self.judge_model = _coerce_config(PedagogicalAPIModelConfig, judge_model)
        self.generation = _coerce_config(PedagogicalGenerationConfig, generation)
        if student_client is None and self.student_model.mode != "api":
            raise ValueError(
                "the student must use mode='api'; offline launchers provide a "
                "local OpenAI-compatible server"
            )
        self.student_client = student_client or PedagogicalAPIClient(self.student_model)
        self.judge_client = judge_client
        if self.judge_client is None and self.judge_model.mode == "api":
            self.judge_client = PedagogicalAPIClient(self.judge_model)
        self._engine_judge_clients: dict[int, PedagogicalEngineClient] = {}
        self.debug_trace_dir = debug_trace_dir
        self.debug_trace_every_n_rollouts = int(debug_trace_every_n_rollouts)
        self.actor_client_factory = actor_client_factory or ArealOpenAI

    def _new_actor_client(self, engine: Any) -> Any:
        return self.actor_client_factory(
            engine=engine,
            tokenizer=self.tokenizer,
            chat_template_type="concat",
            engine_max_tokens=self.gconfig.max_tokens,
        )

    def _new_judge_client(self, engine: Any) -> Any:
        if self.judge_client is not None:
            return self.judge_client
        if self.judge_model.mode != "self":
            raise RuntimeError(
                f"unsupported judge model mode: {self.judge_model.mode!r}"
            )
        key = id(engine)
        if key not in self._engine_judge_clients:
            self._engine_judge_clients[key] = PedagogicalEngineClient(
                self.judge_model,
                engine=engine,
                tokenizer=self.tokenizer,
                base_gconfig=self.gconfig,
            )
        return self._engine_judge_clients[key]

    async def _teacher_turn(self, client: Any, episode: ClassroomEpisode) -> str:
        response = await client.chat.completions.create(
            messages=episode.teacher_messages(),
            n=1,
            max_completion_tokens=self.generation.max_tokens_per_teacher_turn,
            max_total_tokens=self.gconfig.max_tokens,
            temperature=self.gconfig.temperature,
            top_p=self.gconfig.top_p,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content or ""

    async def _student_turn(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        max_tokens: int,
    ) -> list[str]:
        return await self.student_client.generate(
            messages,
            n=n,
            max_tokens=max_tokens,
            temperature=self.generation.student_temperature,
            top_p=self.generation.student_top_p,
        )

    async def _run_native_judges(
        self,
        episode: ClassroomEpisode,
        *,
        judge_client: Any,
        stop_on_reject: bool,
    ) -> list[NativeJudgeDecision]:
        hidden = episode.hidden_conversation()
        all_decisions: list[NativeJudgeDecision] = []

        async def call(prompt: str) -> list[str]:
            return await judge_client.generate(
                [{"role": "user", "content": prompt}],
                n=1,
                max_tokens=self.generation.max_tokens_per_judge_attempt,
                temperature=self.generation.judge_temperature,
                top_p=self.generation.judge_top_p,
            )

        for rule in WHOLE_DIALOGUE_JUDGE_PROMPTS:
            decisions = await run_whole_dialogue_judge(
                rule=rule,
                conversation=hidden,
                call=call,
                attempts=self.generation.number_judge_attempts,
            )
            all_decisions.extend(decisions)
            if stop_on_reject and any(d.rejected for d in decisions):
                break
        episode.native_judges.extend(all_decisions)
        return all_decisions

    @staticmethod
    def _native_rejected(episode: ClassroomEpisode, rule: str) -> bool:
        return any(
            decision.rule == rule and decision.rejected
            for decision in episode.native_judges
        )

    def _native_reward(self, episode: ClassroomEpisode) -> dict[str, float]:
        end_reward = (
            sum(
                native_answer_correct(solution, episode.answer)
                for solution in episode.final_solutions
            )
            / len(episode.final_solutions)
            if episode.final_solutions
            else -self.generation.extra_penalty_for_rejected_judges
        )
        if episode.failed_native_judges and episode.final_solutions:
            end_reward -= self.generation.extra_penalty_for_rejected_judges

        teacher_outputs: list[str] = []
        for message in episode.conversation:
            if message["role"] != "teacher":
                continue
            teacher_outputs.append(message["content"])

        # The shared XML action contract takes the slot of PedagogicalRL's
        # <think>-tag reward, which is zero in its non-thinking setting.
        thinking_reward = 0.0
        format_reward = 0.0
        if episode.format_failed:
            format_reward = self.generation.format_error_penalty
        if episode.final_solutions:
            eoc_reward = 0.1 if episode.teacher_ended else 0.0
        else:
            # PedagogicalRL's Conversation reward helpers return zero when the
            # native judge gate rejects the dialogue and no final solutions are
            # sampled. The rejection penalty and length penalty still apply.
            eoc_reward = 0.0
        length_reward = (
            -0.5
            if any(
                len(self.tokenizer.encode(output))
                >= self.generation.max_tokens_per_teacher_turn - 1
                for output in teacher_outputs
            )
            else 0.0
        )
        return {
            "end_rm_reward": float(end_reward),
            "thinking_reward": float(thinking_reward),
            "format_reward": float(format_reward),
            "end_of_conversation_reward": float(eoc_reward),
            "length_reward": float(length_reward),
            "total_reward": float(
                end_reward
                + thinking_reward
                + format_reward
                + eoc_reward
                + length_reward
            ),
        }

    async def _train_episode(
        self,
        actor_client: Any,
        episode: ClassroomEpisode,
        *,
        judge_client: Any,
    ) -> tuple[float, dict[str, float]]:
        if episode.conversation_type is ConversationType.ATTEMPTED:
            initial = await self._student_turn(
                episode.initial_student_messages(),
                n=1,
                max_tokens=self.generation.max_tokens_per_student_attempt,
            )
            episode.add_initial_attempt(initial[0])

        while not episode.should_stop_dialogue(
            tokenizer=self.tokenizer,
            max_teacher_turns=self.generation.max_teacher_turns,
            max_tokens_in_conversation=self.generation.max_tokens_in_conversation,
        ):
            teacher_output = await self._teacher_turn(actor_client, episode)
            episode.add_teacher(teacher_output)
            if episode.format_failed:
                break
            if episode.should_stop_dialogue(
                tokenizer=self.tokenizer,
                max_teacher_turns=self.generation.max_teacher_turns,
                max_tokens_in_conversation=self.generation.max_tokens_in_conversation,
            ):
                break
            student = await self._student_turn(
                episode.student_messages(),
                n=1,
                max_tokens=self.generation.max_tokens_per_student_turn,
            )
            episode.add_student(student[0])

        if not episode.format_failed:
            await self._run_native_judges(
                episode,
                judge_client=judge_client,
                stop_on_reject=True,
            )
            if not episode.failed_native_judges:
                episode.final_solutions = await self._student_turn(
                    episode.student_messages(final=True),
                    n=self.generation.number_student_attempts,
                    max_tokens=self.generation.max_tokens_per_student_attempt,
                )
        components = self._native_reward(episode)
        return components["total_reward"], components

    @staticmethod
    def _safe_stats(metrics: dict[str, float]) -> None:
        try:
            stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
        except Exception:
            logger.debug("Skipping stats logging outside workflow context")

    async def _dump_trace(
        self,
        *,
        episode: ClassroomEpisode,
        metrics: dict[str, float],
    ) -> None:
        if not self.debug_trace_dir:
            return
        ctx = workflow_context.get()
        task_id = getattr(ctx, "task_id", None)
        if (
            task_id is not None
            and int(task_id) % self.debug_trace_every_n_rollouts != 0
        ):
            return
        output_dir = Path(self.debug_trace_dir) / "train"
        payload = episode.to_trace()
        payload.update(
            {
                "metrics": metrics,
                "task_id": task_id,
                "lora_version": getattr(ctx, "lora_version", None),
            }
        )
        try:
            await aiofiles.os.makedirs(output_dir, exist_ok=True)
            path = output_dir / (
                f"{socket.gethostname()}_{os.getpid()}_{task_id}_{uuid.uuid4().hex}.json"
            )
            async with aiofiles.open(path, "w", encoding="utf-8") as trace_file:
                await trace_file.write(
                    json.dumps(payload, ensure_ascii=False, indent=2)
                )
        except Exception:
            logger.exception("Failed to write PedagogicalRL trace")

    async def arun_episode(self, engine: Any, data: dict[str, Any]) -> dict | None:
        episode = ClassroomEpisode(
            problem=str(data["task"]),
            answer=str(data["ground_truth"]),
        )
        actor_client = self._new_actor_client(engine)
        judge_client = self._new_judge_client(engine)
        reward, components = await self._train_episode(
            actor_client,
            episode,
            judge_client=judge_client,
        )
        solved = float(
            any(
                native_answer_correct(solution, episode.answer)
                for solution in episode.final_solutions
            )
        )
        turns = float(episode.teacher_turns)
        native_leak = float(self._native_rejected(episode, "does_not_leak_answer"))
        student_metric_prefix = f"student/{self.student_model.model}"
        metrics = {
            "reward": reward,
            "final_correct": solved,
            "solved": solved,
            "leaks": native_leak,
            "turns": turns,
            f"{student_metric_prefix}/selected": 1.0,
            f"{student_metric_prefix}/solved": solved,
            f"{student_metric_prefix}/turns": turns,
            f"{student_metric_prefix}/reward": reward,
            "native_judge_rejected": float(episode.failed_native_judges),
            "format_errors": float(episode.format_failed),
            "stop/format_error": float(episode.format_failed),
            "native_leak/rejected": native_leak,
            "native_pedagogy/rejected": float(
                self._native_rejected(episode, "follows_pedagogical_values")
            ),
            **components,
        }
        self._safe_stats(metrics)
        await self._dump_trace(episode=episode, metrics=metrics)

        actor_client.set_last_reward(float(reward))
        return actor_client.export_interactions(style="concat")
