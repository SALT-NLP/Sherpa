from __future__ import annotations

from functools import cache
from typing import Any

NON_THINKING_TEACHER_OUTPUT_FORMAT_WITH_END_PROMPT = """\
For each reply, first use this tagged section:
<reasoning>
your private thinking about what the student needs next
</reasoning>
Then use exactly one of:
<output>
your non-empty message shown to the student
</output>
<end></end>
Use <end></end> when you think the conversation can end early.
The student will see only the text inside <output>...</output>. Put all private
thinking inside <reasoning>...</reasoning>. Do not use JSON or Markdown code fences."""

THINKING_TEACHER_OUTPUT_FORMAT_WITH_END_PROMPT = """\
Reply directly to the student with a non-empty message. \
When you think the conversation can end early, reply with only <end> instead. \
Do not combine <end> with a message to the student."""

# How the teacher sees its own earlier turns in the non-thinking format.
# `public_history` stores them with the tags stripped; this restores the tag
# skeleton so earlier replies still exemplify the format, and replaces the
# `<reasoning>` content with a placeholder so private reasoning is not replayed.
# The placeholder is a description rather than text the model might imitate as
# its own reasoning.
TEACHER_HISTORY_MASKED_TEMPLATE = """\
<reasoning>
(your earlier private reasoning, omitted from this transcript)
</reasoning>
<output>
{visible}
</output>"""

TEACHER_GUIDANCE_INSTRUCTION = (
    "Do not reveal the problem's answer or an equivalent expression to the student."
)

FILTER_SOLVER_SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem independently. "
    "Show your reasoning if useful. Put the final answer in the last "
    "\\boxed{...}. Do not use any answer key or hidden solution."
)

FILTER_SOLVER_USER_TEMPLATE = """\
Task:
{task}

Solve the problem. Put your final answer in \\boxed{{}}.
"""

GUIDANCE_GATE_SYSTEM_PROMPT = (
    "Mechanically judge answer containment. Find every value in the teacher "
    "message that is either a number or an expression made only from numbers "
    "and arithmetic operators. Convert LaTeX arithmetic notation, compute "
    "those values exactly, and normalize them with the ground truth. First "
    "write feedback with a short reason based on that comparison, then set "
    "leaked=true when any value is equal to the ground truth. Return valid "
    "JSON only with keys feedback (string) and leaked (boolean), in that order."
)

ANSWER_JUDGE_SYSTEM_PROMPT = (
    "You are a strict math answer equivalence judge. Compare only the extracted "
    "student answer with the ground-truth answer for the given task. Mark correct "
    "only if they are mathematically equivalent final answers. Return valid JSON "
    "only with key correct (boolean)."
)

ADAPTIVE_GATE_SYSTEM_PROMPT = """You judge whether a tutor message follows \
one stated student preference.
Judge the teaching approach, not mathematical correctness. Ignore tone,
politeness, and verbosity.
If the preference requires a previous student response but no previous real
student message is available, return PASS so the student can respond first.

Reply with exactly these two XML elements, with the reasoning before the
verdict:
<reasoning>brief reason</reasoning>
<verdict>PASS</verdict>

Set verdict to PASS exactly when the tutor message satisfies the preference;
otherwise set it to FAIL. Ensure the verdict agrees with the reasoning.
Use only PASS or FAIL inside the verdict element. Put nothing outside the two
elements."""

ADAPTIVE_GATE_NO_LAST_STUDENT_MESSAGE = "No previous real student message is available."

ADAPTIVE_GATE_USER_TEMPLATE = """<student_preference>
{preference}
</student_preference>

<problem>
{task}
</problem>

<latest_student_message>
{last_student_message}
</latest_student_message>

<tutor_message>
{teacher_message}
</tutor_message>"""


# ---------------------------------------------------------------------------
# Free-chat rollout.
#
# The teacher opens, the pair talks for a fixed budget of rounds, and the student
# is then tested alone on a fresh branch. That re-test provides the episode
# reward, so the student is given no task, no subject and no instruction about
# what to do: what the conversation is about is the teacher's decision.
#
# The teacher's context is a conversation, not one string:
#
#   system     FREE_CHAT_TEACHER_SYSTEM_PROMPT
#   user       FREE_CHAT_TEACHER_SOLVE_PROMPT   ) both only when teacher_pre is
#   assistant  the accepted pre-solve draft     ) on and a draft was accepted
#   user       FREE_CHAT_TEACHER_OPEN_PROMPT    (+ output format, + guidance instruction)
#   ...        the conversation itself
#
# `_build_tutor_messages` assembles it. With teacher_pre disabled the two
# pre-solve messages are omitted and nothing else changes.
# ---------------------------------------------------------------------------

# Deliberately one line, so the prompt adds no prior on how the student behaves.
# The first name keeps the student from identifying itself as an AI assistant
# when asked who it is.
#
# Shared by the conversation, the scored re-test and the no-teaching baseline
# (see _build_student_probe_messages and _no_teaching_baseline), so a change
# affects all three alike instead of being measured as teaching.
FREE_CHAT_STUDENT_SYSTEM_PROMPT = "You are Sam, a student talking with a teacher."

# The task lives here because the teacher's copy is the only one in the episode:
# the student is not given it until the re-test. It goes last so the problem
# statement stays close to the pre-solve and opening turns appended after this
# block. "a problem" rather than "the problem" because the task has not been
# named yet at that point in the text.
FREE_CHAT_TEACHER_SYSTEM_PROMPT = """\
You are a teacher. You have {{ budget }} turn budgets to talk with a student. \
After the conversation, we will ask the student to solve a problem from scratch \
to see whether the student understands. Different students may behave \
differently and have different learning needs. Infer the student's \
characteristics from the interaction and adapt your teaching accordingly. Your \
goal is to teach the student so that they can solve it on their own.

The math problem is:
{{ task }}"""

# The pre-solve is a turn of the conversation, not a block of the system prompt:
# this is the request, and the accepted draft is the assistant turn that answers
# it. It precedes the output-format contract, so a plain-prose draft violates
# nothing, and the teacher reads the draft as its own answer to an explicit
# request rather than as reference material. "Before interacting with the
# student" and "ourselves" already mark the draft as private to the teacher.
FREE_CHAT_TEACHER_SOLVE_PROMPT = """\
Before interacting with the student, let us solve the problem ourselves first. \
Put your final answer in \\boxed{}."""

# Opens the conversation and carries the per-reply directives -- the
# output-format contract and the guidance instruction, appended in that order by
# `_free_chat_open_prompt` -- because neither applies until the conversation
# starts; keeping them out of the system prompt leaves the pre-solve reply free
# of the format contract.
FREE_CHAT_TEACHER_OPEN_PROMPT = """\
Now you can start the conversation with the student."""

# Appended to a replay of the conversation, on an independent branch. This is
# the first and only time the student is shown the task.
FREE_CHAT_STUDENT_RETEST_TEMPLATE = """\
Now try to solve the problem from scratch:
{{ task }}

Put your final answer in \\boxed{}."""

GUIDANCE_GATE_PENDING_FEEDBACK = "Guidance gate pending."
GUIDANCE_GATE_FAILED_FEEDBACK_TEMPLATE = "Guidance gate call failed: {error}"
PUBLIC_HISTORY_ENTRY_TEMPLATE = "{speaker} round {round_idx}:\n{visible_text}"

GUIDANCE_GATE_USER_TEMPLATE = """\
Ground Truth:
{{ ground_truth }}

Teacher Message:
{{ teacher_action or '(empty)' }}

After extracting and computing numeric content from the teacher message, does
any extracted value equal the ground truth?

Return JSON only with this schema:
{
  "feedback": "<short explanation>",
  "leaked": <boolean>
}
"""

ANSWER_JUDGE_USER_TEMPLATE = """\
Task:
{{ task }}

Ground Truth:
{{ ground_truth }}

Extracted Student Answer:
{{ extracted_answer or '(empty)' }}

Decide whether the extracted student answer is mathematically equivalent to the
ground truth as a final answer to the task. Ignore superficial notation
differences, such as including the function name on the left side of an equation,
when the right-hand side is equivalent. Do not use any hidden student reasoning.

Return JSON only with this schema:
{
  "correct": true or false
}
"""


@cache
def _get_template_env():
    from jinja2 import Environment, StrictUndefined

    return Environment(
        autoescape=False,
        lstrip_blocks=True,
        trim_blocks=True,
        undefined=StrictUndefined,
    )


def render_prompt(template: str, **context: Any) -> str:
    return _get_template_env().from_string(template).render(**context).strip()
