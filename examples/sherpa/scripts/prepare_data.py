"""Prepare the training/evaluation dataset.

build (default)
    Write the paper's filtered MATH split, shipped as JSONL next to the output
    directory, as the on-disk dataset the configs read:

        python examples/sherpa/scripts/prepare_data.py

    With --math-train-jsonl/--math-test-jsonl it converts MATH-style rows
    ({id, problem, answer, solution, meta}) instead, e.g. as input to `filter`.

filter
    Build a split for another student/teacher pair. A problem is kept when the
    student fails every one of --student-attempts attempts without teaching and
    the teacher then solves it within --teacher-attempts. Answers are scored as
    in training: the exact math scorer, then the LLM answer judge. The student
    gets the prompt of the no-teaching baseline, the teacher the pre-solve prompt.

        python examples/sherpa/scripts/prepare_data.py filter \\
            --input examples/sherpa/data/math --output examples/sherpa/data/my_split

    Endpoints come from STUDENT_BASE_URL / STUDENT_API_KEY and TEACHER_BASE_URL /
    TEACHER_API_KEY (environment or .env); the teacher also judges answers
    unless --judge none.

The paper split is this filter on MATH with Qwen3-1.7B (two attempts) and
Qwen3-8B (one attempt). A re-run gives a similar but not identical split; use the
shipped split to reproduce the paper.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv

from examples.common.openai_utils import AsyncLLMCaller, AuxModelConfig
from examples.common.parsing import parse_json_dict
from examples.sherpa.core.math import score_math_answer
from examples.sherpa.data_formats.math import build_math_splits
from examples.sherpa.prompts import (
    ANSWER_JUDGE_SYSTEM_PROMPT,
    ANSWER_JUDGE_USER_TEMPLATE,
    FILTER_SOLVER_SYSTEM_PROMPT,
    FILTER_SOLVER_USER_TEMPLATE,
    FREE_CHAT_STUDENT_RETEST_TEMPLATE,
    FREE_CHAT_STUDENT_SYSTEM_PROMPT,
    render_prompt,
)

PAPER_SPLIT = "examples/sherpa/data/math_1.7b_8b/math_filtered"
# Qwen3 non-thinking sampling, as used to build the paper split.
REQUEST_PARAMS = {
    "seed": 42,
    "extra_body": {
        "top_k": 20,
        "min_p": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(args: argparse.Namespace) -> None:
    if bool(args.math_train_jsonl) != bool(args.math_test_jsonl):
        raise SystemExit("--math-train-jsonl and --math-test-jsonl go together.")
    if args.math_train_jsonl:
        train, test = build_math_splits(
            train_jsonl_path=Path(args.math_train_jsonl),
            test_jsonl_path=Path(args.math_test_jsonl),
        )
        output = Path(args.output or "examples/sherpa/data/math")
    else:
        source = Path(args.source)
        train = load_jsonl(source.with_name(f"{source.name}.train.jsonl"))
        test = load_jsonl(source.with_name(f"{source.name}.test.jsonl"))
        output = Path(args.output or args.source)
    DatasetDict(
        {"train": Dataset.from_list(train), "test": Dataset.from_list(test)}
    ).save_to_disk(str(output))
    print(f"Wrote {output}: {len(train)} train / {len(test)} test problems.")


class AnswerScorer:
    """The exact math scorer, then the answer judge when it says incorrect."""

    def __init__(self, judge: AsyncLLMCaller | None):
        self.judge = judge
        self._judged: dict[tuple[str, str, str], bool] = {}

    async def correct(self, task: str, ground_truth: str, answer: str) -> bool:
        exact = score_math_answer(task, ground_truth, answer)
        if exact.correct or self.judge is None:
            return bool(exact.correct)
        extracted = str(exact.raw_result.get("extracted_answer") or "")
        key = (task, ground_truth, extracted)
        if key not in self._judged:
            prompt = render_prompt(
                ANSWER_JUDGE_USER_TEMPLATE,
                task=task,
                ground_truth=ground_truth,
                extracted_answer=extracted,
            )
            try:
                reply = await self.judge.call(
                    [
                        {"role": "system", "content": ANSWER_JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )
                parsed, error = parse_json_dict(reply.text)
                verdict = (
                    not error and isinstance(parsed, dict) and parsed.get("correct")
                )
            except Exception:
                verdict = False
            self._judged[key] = verdict is True
        return self._judged[key]


async def first_correct(
    caller: AsyncLLMCaller,
    messages: list[dict[str, str]],
    *,
    attempts: int,
    row: dict[str, Any],
    scorer: AnswerScorer,
) -> tuple[bool | None, list[bool | None]]:
    """(solved, per-attempt outcomes); solved is None when every call failed."""
    outcomes: list[bool | None] = []
    for _ in range(attempts):
        try:
            reply = await caller.call(messages)
        except Exception:
            outcomes.append(None)
            continue
        correct = await scorer.correct(
            str(row["task"]), str(row["ground_truth"]), reply.text
        )
        outcomes.append(correct)
        if correct:
            return True, outcomes
    if all(outcome is None for outcome in outcomes):
        return None, outcomes
    return False, outcomes


async def classify(
    row: dict[str, Any],
    *,
    student: AsyncLLMCaller,
    teacher: AsyncLLMCaller,
    scorer: AnswerScorer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task = str(row["task"])
    student_messages = [
        {"role": "system", "content": FREE_CHAT_STUDENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_prompt(FREE_CHAT_STUDENT_RETEST_TEMPLATE, task=task),
        },
    ]
    student_solved, student_outcomes = await first_correct(
        student,
        student_messages,
        attempts=args.student_attempts,
        row=row,
        scorer=scorer,
    )
    record = {"id": row.get("id"), "student": student_outcomes, "teacher": []}
    # A failed student call cannot show the problem is hard for this student.
    if student_solved is not False or None in student_outcomes:
        record["kept"] = False
        return record
    teacher_messages = [
        {"role": "system", "content": FILTER_SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": FILTER_SOLVER_USER_TEMPLATE.format(task=task)},
    ]
    teacher_solved, record["teacher"] = await first_correct(
        teacher,
        teacher_messages,
        attempts=args.teacher_attempts,
        row=row,
        scorer=scorer,
    )
    record["kept"] = teacher_solved is True
    return record


def api_caller(
    role: str, model: str, args: argparse.Namespace, **overrides: Any
) -> AsyncLLMCaller:
    base_url = os.environ.get(f"{role}_BASE_URL", "")
    if not base_url:
        raise SystemExit(f"Set {role}_BASE_URL (and {role}_API_KEY if needed).")
    settings = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        **overrides,
    }
    return AsyncLLMCaller(
        AuxModelConfig(
            base_url=base_url,
            model=model,
            api_key=os.environ.get(f"{role}_API_KEY", "EMPTY"),
            timeout=args.timeout,
            max_concurrency=args.concurrency,
            request_params=json.loads(json.dumps(REQUEST_PARAMS)),
            **settings,
        )
    )


async def run_filter(args: argparse.Namespace) -> None:
    load_dotenv(".env", override=False)
    dataset = load_from_disk(args.input)
    splits = args.splits or list(dataset)
    student = api_caller("STUDENT", args.student_model, args)
    teacher = api_caller("TEACHER", args.teacher_model, args)
    judge = None
    if args.judge == "teacher":
        judge = api_caller(
            "TEACHER", args.teacher_model, args, max_tokens=256, temperature=0.0
        )
    scorer = AnswerScorer(judge)
    filtered: dict[str, Dataset] = {}
    report: dict[str, Any] = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "student_attempts": args.student_attempts,
        "teacher_attempts": args.teacher_attempts,
        "judge": args.judge,
        "splits": {},
    }
    for split in splits:
        rows = dataset[split].to_list()
        records = await asyncio.gather(
            *(
                classify(
                    row, student=student, teacher=teacher, scorer=scorer, args=args
                )
                for row in rows
            )
        )
        kept = [row for row, record in zip(rows, records) if record["kept"]]
        filtered[split] = Dataset.from_list(kept, features=dataset[split].features)
        report["splits"][split] = {
            "input": len(rows),
            "kept": len(kept),
            "rows": records,
        }
        print(f"{split}: kept {len(kept)} of {len(rows)} problems.")
    output = Path(args.output)
    DatasetDict(filtered).save_to_disk(str(output))
    report_path = output.with_name(f"{output.name}.report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output} and {report_path}.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command")

    build_parser = commands.add_parser("build", help="Write an on-disk dataset.")
    build_parser.add_argument(
        "--source",
        default=PAPER_SPLIT,
        help="Prefix of SOURCE.train.jsonl / SOURCE.test.jsonl (default: paper split).",
    )
    build_parser.add_argument("--math-train-jsonl")
    build_parser.add_argument("--math-test-jsonl")
    build_parser.add_argument(
        "--output",
        help="Output directory (default: SOURCE, or examples/sherpa/data/math).",
    )

    filter_parser = commands.add_parser("filter", help="Build a new split.")
    filter_parser.add_argument("--input", required=True, help="On-disk DatasetDict.")
    filter_parser.add_argument("--output", required=True)
    filter_parser.add_argument("--splits", nargs="*", help="Default: every split.")
    filter_parser.add_argument("--student-model", default="qwen3-1.7b")
    filter_parser.add_argument("--teacher-model", default="qwen3-8b")
    filter_parser.add_argument("--student-attempts", type=int, default=2)
    filter_parser.add_argument("--teacher-attempts", type=int, default=1)
    filter_parser.add_argument(
        "--judge", choices=["teacher", "none"], default="teacher"
    )
    filter_parser.add_argument("--max-tokens", type=int, default=2048)
    filter_parser.add_argument("--temperature", type=float, default=0.7)
    filter_parser.add_argument("--top-p", type=float, default=0.8)
    filter_parser.add_argument("--timeout", type=int, default=120)
    filter_parser.add_argument("--concurrency", type=int, default=8)

    if not argv or argv[0] not in {"build", "filter", "-h", "--help"}:
        argv = ["build", *argv]
    args = parser.parse_args(argv)
    if (
        args.command == "filter"
        and min(args.student_attempts, args.teacher_attempts) < 1
    ):
        parser.error("attempt counts must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "filter":
        asyncio.run(run_filter(args))
    else:
        build(args)


if __name__ == "__main__":
    main()
