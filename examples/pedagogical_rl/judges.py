"""PedagogicalRL's whole-dialogue judges.

In training they are hard gates: a REJECT from either rule skips the final
student attempts and applies the rejection penalty.

The retry-and-fail-open behaviour is theirs: up to ``MAX_JUDGE_ROUNDS`` rounds
to collect ``attempts`` parseable verdicts, then any shortfall counts as OK.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

from examples.pedagogical_rl.prompts import WHOLE_DIALOGUE_JUDGE_PROMPTS, render
from examples.pedagogical_rl.state import NativeJudgeDecision

logger = logging.getLogger(__name__)

MAX_JUDGE_ROUNDS = 5


def parse_native_judge_output(raw_output: str, rule: str) -> NativeJudgeDecision | None:
    """Match PedagogicalRL's permissive JSON slicing and backslash removal."""

    start = (raw_output or "").find("{")
    end = (raw_output or "").rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = raw_output[start : end + 1].replace("\\", "")
    try:
        parsed = json.loads(candidate, strict=False)
    except Exception:
        return None
    reasoning = parsed.get("reasoning")
    decision = parsed.get("decision")
    if not isinstance(reasoning, str) or decision not in {"OK", "REJECT"}:
        return None
    return NativeJudgeDecision(rule=rule, reasoning=reasoning, decision=decision)


async def run_whole_dialogue_judge(
    *,
    rule: str,
    conversation: list[dict[str, str]],
    call: Callable[[str], Awaitable[list[str]]],
    attempts: int,
) -> list[NativeJudgeDecision]:
    """Collect exactly ``attempts`` verdicts for one rule."""

    if attempts < 1:
        raise ValueError("attempts must be positive")
    # conversation entries use PedagogicalRL's internal teacher/student roles;
    # their template capitalises them.
    prompt = render(WHOLE_DIALOGUE_JUDGE_PROMPTS[rule], conversation=conversation)
    valid: list[NativeJudgeDecision] = []
    for _ in range(MAX_JUDGE_ROUNDS):
        missing = attempts - len(valid)
        if missing <= 0:
            break
        results = await asyncio.gather(
            *(call(prompt) for _ in range(missing)), return_exceptions=True
        )
        outputs: list[str] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error("whole-dialogue judge failed for %s: %s", rule, result)
            else:
                outputs.extend(result)
        valid.extend(
            decision
            for output in outputs
            if (decision := parse_native_judge_output(output, rule)) is not None
        )
    while len(valid) < attempts:
        # PedagogicalRL fails open once retries are exhausted.
        valid.append(
            NativeJudgeDecision(
                rule=rule, reasoning="max turns exceeded", decision="OK"
            )
        )
    return valid[:attempts]
