#!/usr/bin/env python3
"""Score all four open-ended tasks with the official MathTutorBench Ped-RM."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

PEDAGOGY_TASKS = (
    "scaffolding_generation",
    "pedagogy_following",
    "scaffolding_generation_hard",
    "pedagogy_following_hard",
)
SYSTEM_PROMPT = (
    "Judge the pedagogical quality of the responses provided by two teachers. "
    "Focus on the quality of the scaffolding guidance, correctness, and "
    "actionability of the feedback through nudges, questions and hints. Do not "
    "give high scores for revealing the full answer."
)


def resolve_local_model(model: str) -> Path:
    local = Path(model).expanduser()
    if local.exists():
        snapshot = local.resolve()
    else:
        from huggingface_hub import snapshot_download

        try:
            snapshot = Path(snapshot_download(model, local_files_only=True)).resolve()
        except Exception as error:
            raise SystemExit(
                f"Ped-RM {model!r} is not fully available in the local cache. "
                "This runner will not download this multi-GB model automatically; "
                "set PED_RM_MODEL to a complete local snapshot."
            ) from error
    if not (snapshot / "config.json").is_file():
        raise SystemExit(f"invalid Ped-RM snapshot (missing config.json): {snapshot}")
    if not (
        list(snapshot.glob("*.safetensors"))
        or list(snapshot.glob("pytorch_model*.bin"))
    ):
        raise SystemExit(f"invalid Ped-RM snapshot (missing model weights): {snapshot}")
    return snapshot


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def response_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value or "")


def conversation(item: dict[str, Any], response: str) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Problem: "
                + str(item.get("problem", ""))
                + "\nReference Solution: "
                + str(item.get("reference_solution", ""))
            ),
        },
    ]
    for entry in item["dialog_history"]:
        role = "assistant" if entry["user"] in ("Teacher", "Tutor") else "user"
        messages.append({"role": role, "content": str(entry["text"])})
    messages.append({"role": "assistant", "content": response})
    return messages


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel"
    )
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--tasks-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_path = resolve_local_model(args.model)
    if args.resolve_only:
        print(model_path)
        return
    if args.tasks_root is None or args.output is None:
        parser.error(
            "--tasks-root and --output are required unless --resolve-only is used"
        )

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    set_seed(42)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks_root = args.tasks_root.resolve()

    source_data: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    aggregate: dict[str, Any] = {}
    pending: list[str] = []
    for task in PEDAGOGY_TASKS:
        source = tasks_root / task / "generations.json"
        if not source.is_file():
            raise SystemExit(f"missing generations for Ped-RM: {source}")
        data = json.loads(source.read_text(encoding="utf-8"))
        source_data[task] = (source, data)
        metrics_file = output / task / "metrics.json"
        if metrics_file.is_file():
            cached = json.loads(metrics_file.read_text(encoding="utf-8"))
            if int(cached.get("total_samples", -1)) == len(data):
                aggregate[task] = cached
                print(f"[pedrm] {task}: using complete cached score")
                continue
        pending.append(task)

    if pending:
        print(f"[pedrm] loading official reward model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            num_labels=1,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        @torch.inference_mode()
        def score(messages: list[dict[str, str]]) -> float:
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt"
            ).to(device)
            return float(model(inputs).logits[0][0].item())

        for task in pending:
            _, data = source_data[task]
            candidate_scores: list[float] = []
            reference_scores: list[float] = []
            enriched = deepcopy(data)
            for index, item in enumerate(tqdm(data, desc=f"Ped-RM {task}")):
                candidate = score(
                    conversation(item, str(item["generated_teacher_utterance"]))
                )
                reference = score(
                    conversation(item, response_text(item["ground_truth_response"]))
                )
                candidate_scores.append(candidate)
                reference_scores.append(reference)
                enriched[index]["chosen_score"] = candidate
                enriched[index]["rejected_score"] = reference

            margins = [
                candidate - reference
                for candidate, reference in zip(candidate_scores, reference_scores)
            ]
            metrics = {
                "win_rate": (
                    sum(margin > 0 for margin in margins) / len(margins)
                    if margins
                    else 0.0
                ),
                "score": mean(candidate_scores),
                "baseline_score": mean(reference_scores),
                "mean_margin": mean(margins),
                "total_samples": len(margins),
            }
            task_output = output / task
            task_output.mkdir(parents=True, exist_ok=True)
            write_json(task_output / "metrics.json", metrics)
            write_json(task_output / "enriched_generations.json", enriched)
            aggregate[task] = metrics
            print(f"[pedrm] {task}: win_rate={metrics['win_rate']:.6f}")

    write_json(output / "pedrm_metrics.json", aggregate)


if __name__ == "__main__":
    main()
