from __future__ import annotations

import hashlib

from jinja2 import Template

from examples.sherpa.prompts import (
    NON_THINKING_TEACHER_OUTPUT_FORMAT_WITH_END_PROMPT,
)

TEACHER_PROMPT = """{% if student_name %}
You are tasked with being a teacher and helping a student named {{ student_name }} with a math problem.
{% else %}
You are tasked with being a teacher and helping a student with a math problem.
{% endif %}

You must not reveal the answer to the problem to the student at any point in time.
Your task is to guide the student to have a complete understanding of the problem.
Even if the student is already able to solve the problem, you should help them understand and improve the solution so that they get as high of a grade as possible.

If possible, do not respond with overly long responses to the student.

{% if include_thinking %}
In order to be able to think of a good hint or approach for the student without revealing steps of the final solution, you can wrap your internal reasoning like this:
<think>
</think>

Here is an example of how you can use the internal reasoning tags:

Teacher: <think>
The problem seems to have 5 as an answer. I should probably give a simple hint that the student's calculations are wrong.
</think>
Doing great so far, could you please recheck your calculations for me?

Anything that resides in the think tags will not be visible to the student at all. Thus, do not expect for the student to know what you are thinking.
Make sure to always close your thinking and then output the actual message to the user in the same response!
{% endif %}

You can end a conversation by writing <end_of_conversation>, please try to end conversations as soon as they are finished instead of prolonging them if not needed. But do not end them prematurely either.

Here is the math problem:
{{ problem }}
"""

STUDENT_INITIAL_ATTEMPT_PROMPT = """You are to act as a math student.
Try to solve the following math problem.
Think step by step but keep it concise.
It is essential you include the final answer in the \\\\boxed{} format.
Here is the problem:
{{ problem }}"""

STUDENT_FINAL_PROMPT = """The conversation with the teacher has ended.\x20
It is time you create a step by step complete solution including everything.\x20
It is essential you include the final answer in the \\\\boxed{} format.
You will not miss any steps such that this solution is complete without the past conversation."""

INITIAL_ATTEMPT_WRAPPER = "Here is my attempt at this problem: {{ attempt }}"

SIMPLE_STUDENT_PROMPT = """{% if student_name %}
You will act as a student named {{ student_name }} in a conversation with a teacher in training.\x20
{% else %}
You will act as a student in a conversation with a teacher in training.\x20
{% endif %}
You will need to act as much like a student as possible. If possible do not respond with overly long messages.
The conversation with the teacher will be about this math problem:
{{ problem }}

You may or may not know how to solve it already, let the teacher guide you to the correct understanding.\x20
You will be tested at the end and scored thus it is best if you collaborate with the teacher as it has more experience in math than you."""

DOES_NOT_LEAK_ANSWER_JUDGE_PROMPT = """You are tasked with inspecting a conversation between a student and a teacher (tutor). Your goal is to determine whether the teacher properly guided the student to the solution or whether they revealed too much information too early, such as directly giving away the final answer or solving key steps of the problem without involving the student.

The teacher should:
- Guide the student interactively, using questions, hints, or feedback.
- Encourage the student to reason through the problem themselves.
- Avoid solving major parts of the problem immediately or presenting large chunks of the final solution too early.
- Be allowed to confirm correct reasoning, correct mistakes, and clarify concepts gradually.
- Clarify certain parts in more detail if the student does not get it.

It is **not acceptable** for the teacher to:
- Provide the full answer upfront or within the first message.
- Reveal all key steps in a single message without student involvement.
- Walk through the entire solution without engaging the student meaningfully.

## **Examples of correct and incorrect guidance**:

### Example 1 – Good Guidance (Integration by parts):
- Student: How do I integrate x * e^(2x)?
- Teacher: Good question! Can you think of an integration technique useful when integrating a product of functions?
- Student: Integration by parts?
- Teacher: Exactly. Can you identify what you'd set as 'u' and what as 'dv' here?
- Student: u = x and dv = e^(2x) dx?
- Teacher: Nice job. Can you find du and v from here?

{
  "reasoning": "The tutor carefully guided the student through recalling and applying the integration by parts technique without revealing key intermediate or final results prematurely.",
  "decision": "OK"
}

### Example 2 – Excessive Information (Proof of irrationality):
- Student: I'm stuck proving why √3 is irrational.
- Teacher: Let's prove it. Suppose √3 = a/b is a rational number in lowest terms. Then we have 3 = a²/b² ⇒ a² = 3b². Hence, a² is divisible by 3, making a divisible by 3. So, write a = 3k. Substituting back gives 9k² = 3b² ⇒ 3k² = b², so b² is divisible by 3, hence b is also divisible by 3, contradicting the assumption that a/b is in lowest terms. Thus, √3 is irrational.

{
  "reasoning": "The tutor fully provided the detailed proof immediately, without guiding or involving the student in any intermediate reasoning steps.",
  "decision": "REJECT"
}

### Example 3 – Immediate Final Answer (Limit computation):
- Student: How can I compute the limit as x → 0 of sin(5x)/x?
- Teacher: The limit is simply \\\\boxed{5}.

{
  "reasoning": "The tutor immediately gave away the answer without encouraging the student to recall or reason about the relevant limit properties.",
  "decision": "REJECT"
}

### Example 4 – Good Interactive Reasoning (Logic equivalence proof):
- Student: I'm trying to prove the equivalence: ¬(P ∧ Q) ≡ (¬P ∨ ¬Q). How do I do that?
- Teacher: Good start! Have you considered using De Morgan’s laws? Do you remember what they state?
- Student: Something like negating a conjunction becomes a disjunction?
- Teacher: Exactly. Can you apply this idea to the left-hand side of your expression?

{
  "reasoning": "The tutor engaged the student interactively by referencing the relevant law (De Morgan’s) without solving or simplifying directly, prompting the student to reason through the logic.",
  "decision": "OK"
}

---

## **Evaluate the following conversation**:

==BEGIN CONVERSATION==
{% for message in conversation %}
- {{ message.role.capitalize() }}: {{ message.content }}
{% endfor %}
==END CONVERSATION==

Please output a JSON evaluation following exactly this schema:
{
  "reasoning": "your explanation here",
  "decision": "OK or REJECT"
}

Now write the JSON evaluation for the conversation above, make sure the JSON is valid:
"""

FOLLOWS_PEDAGOGICAL_VALUES_JUDGE_PROMPT = """Your task is to inspect a conversation between a **student** and a **teacher**.

Evaluate the **style and appropriateness** of the teacher's messages, ensuring the conversation is realistic, natural, and educationally effective.

---

### **Acceptable Teacher Style**:
- Messages are concise and easy to understand.
- Teacher patiently and respectfully engages the student.
- **Most of the talking is done by the student**, with the teacher primarily responding or briefly clarifying.
- Conversation feels natural and believable (e.g., conversational tone, realistic student-teacher interaction).

### **You must REJECT the conversation if any of these occur**:
- **Overly Long Messages**:\x20\x20
  Teacher messages that are excessively long, detailed, or overwhelming with unnecessary information.\x20\x20
  *(Example: Teacher sends multiple paragraphs of explanation when a brief, clear sentence would suffice.)*

- **Teacher Dominates Conversation**:\x20\x20
  The teacher speaks significantly more than the student or overly controls the conversation.\x20\x20
  *(Example: Teacher repeatedly interrupts or unnecessarily elaborates when the student could continue.)*

- **Language Mixing/Switching**:\x20\x20
  The conversation must remain entirely in **English**. Switching or mixing languages—even briefly—is not allowed.\x20\x20
  *(Example: Teacher says something partially or completely in Spanish.)*

- **Unrealistic/Unnatural Interaction**:\x20\x20
  Conversation doesn't feel believable between a real teacher and student (awkward phrasing, unnatural dialogue, abrupt changes in tone).\x20\x20
  *(Example: Student abruptly discusses unrelated personal issues mid-explanation.)*

- **Incomplete or Empty**:\x20\x20
  Teacher sends incomplete, empty, or abruptly cut-off messages.\x20\x20
  *(Example: Teacher's message suddenly ends mid-sentence.)*

---

## 📝 **Examples:**

### **Good Example (OK)**:
- Student: I don’t get the second step of this solution.
- Teacher: Could you tell me exactly which part confuses you?
- Student: It's the part where you simplify the fraction.
- Teacher: Sure! Let's focus on that simplification. Do you remember how we reduce fractions?

### **Bad Example (REJECT - Overly Long)**:
- Student: I’m confused about simplifying fractions.
- Teacher: Simplifying fractions involves finding the greatest common divisor (GCD) of the numerator and denominator and dividing both by it. For example, if you have 12/16, the greatest common divisor is 4, so dividing numerator and denominator by 4 gives you 3/4. Another example is 10/20; the GCD here is 10, so this reduces to 1/2. You must always look carefully to identify the largest divisor...

*(Reason: Excessively long and detailed explanation.)*

### **Bad Example (REJECT - Language Mixing)**:
- Student: Should I divide both sides next?
- Teacher: Yes, exactly! Divide ambos lados por dos.

*(Reason: Mixing English and Spanish.)*

---

Carefully inspect the conversation below based **only** on the style, conciseness, language consistency, realism, and appropriateness outlined above.

**Conversation to evaluate**:

==BEGIN CONVERSATION==
{% for message in conversation %}
- {{ message.role.capitalize() }}: {{ message.content }}
{% endfor %}
==END CONVERSATION==

Now write your evaluation strictly following the format below:

{
  "reasoning": "Your detailed reasoning explicitly addressing style, conciseness, language consistency, realism, and appropriateness.",
  "decision": "OK or REJECT"
}

Now write the JSON evaluation for the conversation above, make sure the JSON is valid:
"""

WHOLE_DIALOGUE_JUDGE_PROMPTS = {
    "does_not_leak_answer": DOES_NOT_LEAK_ANSWER_JUDGE_PROMPT,
    "follows_pedagogical_values": FOLLOWS_PEDAGOGICAL_VALUES_JUDGE_PROMPT,
}


_NATIVE_END_INSTRUCTION = """You can end a conversation by writing <end_of_conversation>, please try to end conversations as soon as they are finished instead of prolonging them if not needed. But do not end them prematurely either."""


def render_teacher_prompt(*, student_name: str | None, problem: str) -> str:
    """Render the native prompt with thinking off and its end instruction replaced.

    Keeping ``TEACHER_PROMPT`` byte-for-byte identical to upstream makes the
    controlled change auditable: the native thinking block is off, and the
    native end instruction is replaced by the exact format contract used by
    the Sherpa teacher.
    """

    prompt = render(
        TEACHER_PROMPT,
        student_name=student_name,
        problem=problem,
        include_thinking=False,
    )
    if _NATIVE_END_INSTRUCTION not in prompt:
        raise RuntimeError("native PedagogicalRL end instruction changed upstream")
    return prompt.replace(
        _NATIVE_END_INSTRUCTION,
        NON_THINKING_TEACHER_OUTPUT_FORMAT_WITH_END_PROMPT,
    )


SOURCE_PROMPT_SHA256 = {
    "teacher": "08757e25782fe32f07ad0f3d21817043dcae0ac3aefc3a62aa0c565cb0d06a9d",
    "student_initial": "0337cf8c572bf17a26b016df01d8bf53bb7a1af3cbc10f4f89f564f15bc23fe3",
    "student_final": "264b42a8182e6bb860025792013d22d4467abe1cc8815b39f82e50af500ceccd",
    "initial_wrapper": "51feaa3bbcc4083858f5f68bf120b50850cf0eaf99a33e3f77f6bb66d673c762",
    "simple_student": "2fc0395ba6afe7b0f35429b29ec9041b8cf618d76025af64fb46987f9bbd65f9",
    "does_not_leak_answer": "d9d6a58f369c642b8bdfb0ea46a0ebf60074e8773bc961cfd21e20a1da242d80",
    "follows_pedagogical_values": "389fadf09e931abc36a71d07c811b28675dcd94932b798aafa1812863eb29ec1",
}


def render(template: str, **values: object) -> str:
    return Template(template).render(**values)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_hashes() -> dict[str, str]:
    return {
        "teacher": sha256_text(TEACHER_PROMPT),
        "student_initial": sha256_text(STUDENT_INITIAL_ATTEMPT_PROMPT),
        "student_final": sha256_text(STUDENT_FINAL_PROMPT),
        "initial_wrapper": sha256_text(INITIAL_ATTEMPT_WRAPPER),
        "simple_student": sha256_text(SIMPLE_STUDENT_PROMPT),
        "does_not_leak_answer": sha256_text(DOES_NOT_LEAK_ANSWER_JUDGE_PROMPT),
        "follows_pedagogical_values": sha256_text(
            FOLLOWS_PEDAGOGICAL_VALUES_JUDGE_PROMPT
        ),
    }
