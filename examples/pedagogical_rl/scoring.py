from __future__ import annotations


def extract_boxed_answer(solution: str) -> str | None:
    """Replicate PedagogicalRL's last-balanced-``\\boxed{}`` extraction."""

    solution = solution or ""
    last_boxed_start = solution.rfind("\\boxed{")
    if last_boxed_start == -1:
        return None
    start_index = last_boxed_start + len("\\boxed{")
    depth = 1
    for index in range(start_index, len(solution)):
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            depth -= 1
            if depth == 0:
                return solution[start_index:index]
    return None


def native_answer_correct(solution: str, ground_truth: str) -> bool:
    """Replicate PedagogicalRL's ``Answer`` reward model exactly."""

    extracted = extract_boxed_answer(solution)
    return str(ground_truth).strip().lower() == str(extracted).strip().lower()
