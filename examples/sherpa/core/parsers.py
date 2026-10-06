from __future__ import annotations

import re

from examples.common.parsing import join_errors, parse_json_dict

from .types import GuidanceGateResult

_TEACHER_RESPONSE_TAGS = (
    "<reasoning>",
    "</reasoning>",
    "<output>",
    "</output>",
)
TEACHER_END_TAG = "<end></end>"


def parse_thinking_teacher_action(
    raw_output: str, *, allow_end: bool = False
) -> tuple[str | None, bool, str | None]:
    """Final-content contract for native reasoning models; never expose thoughts."""
    text = (raw_output or "").strip()
    if allow_end and text == "<end>":
        return "", True, None
    if text and not re.search(
        r"</?(?:output|end|reasoning|think|thinking)\b", text, re.IGNORECASE
    ):
        return text, False, None
    return (
        None,
        False,
        (
            "teacher response must be non-empty student-facing text without protocol tags"
            + (" or <end>" if allow_end else "")
        ),
    )


def parse_tagged_teacher_output(raw_output: str) -> tuple[str | None, str | None]:
    """Extract the student-visible section from a tagged teacher response."""
    text = raw_output or ""
    for tag in _TEACHER_RESPONSE_TAGS:
        count = text.count(tag)
        if count != 1:
            return None, f"teacher response must contain exactly one {tag} tag"

    reasoning_open = text.index("<reasoning>")
    reasoning_close = text.index("</reasoning>")
    output_open = text.index("<output>")
    output_close = text.index("</output>")
    if not reasoning_open < reasoning_close < output_open < output_close:
        return None, "teacher response tags are not in the required order"

    output_start = output_open + len("<output>")
    return text[output_start:output_close].strip(), None


def parse_tagged_teacher_action(
    raw_output: str,
    *,
    allow_end: bool = False,
    require_nonempty_output: bool = False,
) -> tuple[str | None, bool, str | None]:
    """Parse a normal visible reply or the explicit teacher end action.

    Returns ``(visible_output, ended, parse_error)``. ``parse_tagged_teacher_output``
    accepts an empty output; ``require_nonempty_output`` rejects it.
    """

    text = raw_output or ""
    if allow_end:
        reasoning_open_count = text.count("<reasoning>")
        reasoning_close_count = text.count("</reasoning>")
        if reasoning_open_count == 1 and reasoning_close_count == 1:
            reasoning_open = text.index("<reasoning>")
            reasoning_close = text.index("</reasoning>")
            if reasoning_open < reasoning_close:
                suffix = text[reasoning_close + len("</reasoning>") :].strip()
                # Reserve every end-like XML tag in the action position.  Only
                # the exact paired tag is valid; a bare or mixed tag is a format
                # error rather than a normal student-visible message.
                if re.search(r"</?end\b[^>]*>", suffix):
                    if text[:reasoning_open].strip() or suffix != TEACHER_END_TAG:
                        return (
                            None,
                            False,
                            (
                                "teacher end action must be exactly "
                                "<reasoning>...</reasoning><end></end>"
                            ),
                        )
                    return "", True, None

    output, parse_error = parse_tagged_teacher_output(text)
    if output is None:
        return None, False, parse_error
    if require_nonempty_output and not output:
        return None, False, "teacher <output> must contain a non-empty message"
    return output, False, None


def parse_guidance_gate_result(raw_output: str) -> GuidanceGateResult:
    parsed, parse_error = parse_json_dict(raw_output)
    if parsed is None:
        return GuidanceGateResult(
            raw_output=raw_output,
            failed=True,
            feedback="Failed to parse guidance gate output.",
            parse_error=parse_error,
            raw_result={},
        )
    failed = parsed.get("leaked")
    feedback = parsed.get("feedback", "")
    if not isinstance(failed, bool):
        parse_error = join_errors(parse_error, '"leaked" must be a boolean')
        failed = True
    if not isinstance(feedback, str):
        parse_error = join_errors(parse_error, '"feedback" must be a string')
        feedback = str(feedback)
    if parse_error:
        failed = True
    return GuidanceGateResult(
        raw_output=raw_output,
        failed=failed,
        feedback=feedback
        or (
            "The teacher revealed the answer directly. The student did not see this turn."
            if failed
            else "No answer leakage detected."
        ),
        parse_error=parse_error,
        raw_result=parsed,
    )
