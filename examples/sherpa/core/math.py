from __future__ import annotations

import json

from .text import strip_reasoning_for_context
from .types import JudgeResult


def score_math_answer(task: str, ground_truth: str, student_answer: str) -> JudgeResult:
    visible_answer = strip_reasoning_for_context(student_answer)
    extracted_answer = extract_math_answer(visible_answer)
    target_answer = extract_ground_truth_answer(ground_truth)
    normalized_prediction = strip_string(extracted_answer)
    normalized_target = strip_string(target_answer)
    correct = is_equiv(extracted_answer, target_answer)
    raw_result = {
        "method": "lm_eval_hendrycks_math_exact_match",
        "task": task,
        "student_answer": visible_answer,
        "extracted_answer": extracted_answer,
        "target_answer": target_answer,
        "normalized_prediction": normalized_prediction,
        "normalized_target": normalized_target,
    }
    return JudgeResult(
        raw_output=json.dumps(
            {
                "correct": correct,
                "feedback": "Correct." if correct else "Incorrect.",
                "scoring": raw_result,
            },
            ensure_ascii=True,
            indent=2,
        ),
        correct=correct,
        feedback="Correct." if correct else "Incorrect.",
        parse_error=None,
        raw_result=raw_result,
    )


def extract_math_answer(response: str) -> str:
    boxed = last_boxed_only_string(response)
    if boxed is not None:
        return remove_boxed(boxed)
    return ""


def extract_ground_truth_answer(ground_truth: str) -> str:
    boxed = last_boxed_only_string(ground_truth)
    if boxed is not None:
        return remove_boxed(boxed)
    return str(ground_truth)


def is_equiv(str1: str | None, str2: str | None) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return strip_string(str1) == strip_string(str2)
    except Exception:
        return str1 == str2


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if not s.startswith(left):
            return s
        return s[len(left) :]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    left = "\\fbox{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


def last_boxed_only_string(string: str) -> str | None:
    string = string or ""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string
