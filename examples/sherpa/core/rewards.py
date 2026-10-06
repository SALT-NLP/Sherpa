from __future__ import annotations

from .types import (
    EpisodeArtifact,
    GuidanceGateResult,
    RewardAssignment,
    TurnArtifact,
    TurnTrace,
)

# Penalties that belong to the turn that raised them. ReBN would otherwise sum
# them backward into the returns of every earlier turn; the actor holds them out
# of that accumulation and adds them after advantage normalization.
TURN_LOCAL_COMPONENTS = frozenset(
    {
        "guidance_gate_fail",
        "format_error",
        "teacher_exact_repeat",
        "adaptive_gate_fail",
        "soft_overlong",
    }
)


class EpisodeRewardComputer:
    def __init__(
        self,
        *,
        guidance_gate_fail_penalty: float,
        format_error_penalty: float = 0.0,
        teacher_exact_repeat_penalty: float = 0.0,
        adaptive_gate_fail_penalty: float = 0.0,
        soft_overlong_enabled: bool = False,
        soft_overlong_max_tokens: int = 0,
        soft_overlong_buffer_tokens: int = 0,
        soft_overlong_max_penalty: float = 0.0,
    ) -> None:
        if format_error_penalty > 0.0:
            raise ValueError("format_error_penalty must be <= 0.")
        if teacher_exact_repeat_penalty > 0.0:
            raise ValueError("teacher_exact_repeat_penalty must be <= 0.")
        if adaptive_gate_fail_penalty > 0.0:
            raise ValueError("adaptive_gate_fail_penalty must be <= 0.")
        if soft_overlong_max_penalty > 0.0:
            raise ValueError("soft_overlong_max_penalty must be <= 0.")
        if soft_overlong_enabled:
            if soft_overlong_max_tokens <= 0:
                raise ValueError("soft_overlong_max_tokens must be positive.")
            if not 0 < soft_overlong_buffer_tokens < soft_overlong_max_tokens:
                raise ValueError(
                    "soft_overlong_buffer_tokens must be positive and smaller "
                    "than soft_overlong_max_tokens."
                )
            if soft_overlong_max_penalty == 0.0:
                raise ValueError(
                    "soft_overlong_enabled=true requires a negative max penalty."
                )

        self.guidance_gate_fail_penalty = float(guidance_gate_fail_penalty)
        self.format_error_penalty = float(format_error_penalty)
        self.teacher_exact_repeat_penalty = float(teacher_exact_repeat_penalty)
        self.adaptive_gate_fail_penalty = float(adaptive_gate_fail_penalty)
        self.soft_overlong_enabled = bool(soft_overlong_enabled)
        self.soft_overlong_max_tokens = int(soft_overlong_max_tokens)
        self.soft_overlong_buffer_tokens = int(soft_overlong_buffer_tokens)
        self.soft_overlong_max_penalty = float(soft_overlong_max_penalty)
        self.turn_local_components = TURN_LOCAL_COMPONENTS

    async def compute(self, episode: EpisodeArtifact) -> list[RewardAssignment]:
        episode_guidance_gate_component = self._episode_guidance_gate_component(
            episode.turns
        )
        assignments: list[RewardAssignment] = []
        for artifact in episode.turns:
            components: dict[str, float] = {}
            if (
                episode_guidance_gate_component is not None
                and artifact.turn_idx == episode_guidance_gate_component[0]
            ):
                name, value = episode_guidance_gate_component[1]
                components[name] = value
            if artifact.tutor_format_error and self.format_error_penalty:
                components["format_error"] = self.format_error_penalty
            if artifact.teacher_exact_repeat and self.teacher_exact_repeat_penalty:
                components["teacher_exact_repeat"] = self.teacher_exact_repeat_penalty
            if artifact.adaptive_gate_failed and self.adaptive_gate_fail_penalty:
                components["adaptive_gate_fail"] = self.adaptive_gate_fail_penalty
            soft_overlong_penalty = self._soft_overlong_penalty(
                len(getattr(artifact.tutor_response, "output_tokens", ()) or ())
            )
            if soft_overlong_penalty:
                components["soft_overlong"] = soft_overlong_penalty
            reward = float(sum(components.values()))
            local_reward = float(
                sum(
                    value
                    for name, value in components.items()
                    if name in self.turn_local_components
                )
            )
            assignments.append(
                RewardAssignment(
                    reward=reward,
                    reward_components=components,
                    local_reward=local_reward,
                )
            )
        return assignments

    def _guidance_gate_component(
        self, result: GuidanceGateResult
    ) -> tuple[str, float] | None:
        if result.failed:
            return "guidance_gate_fail", self.guidance_gate_fail_penalty
        return None

    def _episode_guidance_gate_component(
        self, turns: list[TurnArtifact]
    ) -> tuple[int, tuple[str, float]] | None:
        """Charged once per episode, on the first turn that fails the guidance gate."""
        for artifact in turns:
            component = self._guidance_gate_component(artifact.guidance_gate_result)
            if component is not None:
                return int(artifact.turn_idx), component
        return None

    def _soft_overlong_penalty(self, generated_tokens: int) -> float:
        """Linear penalty over the last buffer tokens before the generation cap."""
        if not self.soft_overlong_enabled:
            return 0.0
        penalty_start = self.soft_overlong_max_tokens - self.soft_overlong_buffer_tokens
        excess_tokens = max(0, int(generated_tokens) - penalty_start)
        if excess_tokens == 0:
            return 0.0
        fraction = min(1.0, excess_tokens / self.soft_overlong_buffer_tokens)
        return float(self.soft_overlong_max_penalty * fraction)


def artifact_to_trace(
    artifact: TurnArtifact, assignment: RewardAssignment
) -> TurnTrace:
    return TurnTrace(
        turn_idx=artifact.turn_idx,
        tutor_state=artifact.tutor_state,
        tutor_raw_output=artifact.tutor_raw_output,
        tutor_visible_output=artifact.tutor_visible_output,
        guidance_gate_failed=artifact.guidance_gate_result.failed,
        student_output=artifact.student_output,
        reward=assignment.reward,
        reward_components=assignment.reward_components,
        public_history_before=artifact.public_history_before,
        public_history_after=artifact.public_history_after,
        guidance_gate_masked=artifact.guidance_gate_masked,
        tutor_format_error=artifact.tutor_format_error,
        teacher_ended=artifact.teacher_ended,
        adaptive_gate_result=artifact.adaptive_gate_result,
        adaptive_gate_failed=artifact.adaptive_gate_failed,
        teacher_exact_repeat=artifact.teacher_exact_repeat,
    )
