# MathTutorBench

Runs the full [MathTutorBench](https://github.com/eth-lre/mathtutorbench) leaderboard
(nine tasks, pinned to commit
[`6faed173`](https://github.com/eth-lre/mathtutorbench/commit/6faed173ec2bef55cb899b2a3e0f93982f9cb176))
for one teacher. Generation calls the teacher's API. The four pedagogy tasks are then
scored by the official Ped-RM
([`eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel`](https://huggingface.co/eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel)),
which requires a GPU.

## Setup (once, with network access)

```bash
.venv/bin/python examples/math_tutor_bench/prepare.py \
  --upstream examples/math_tutor_bench/.runtime/upstream \
  --dependency-root examples/math_tutor_bench/.runtime/python \
  --python .venv/bin/python
HF_HUB_CACHE=examples/math_tutor_bench/.runtime/hf_hub \
  .venv/bin/python examples/math_tutor_bench/prefetch_datasets.py \
  --cache-dir examples/math_tutor_bench/.runtime/hf_datasets
hf download eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel
```

This stages the pinned benchmark code, its GSM8K and StepVerify data and the Ped-RM. The
evaluation then runs offline.

## Run

Serve the teacher as for the [Sherpa evaluation](../sherpa/README.md#evaluate) (no
student or judge is needed) and set `TEACHER_BASE_URL`, `TEACHER_API_KEY` and, for an
adapter, `TEACHER_ADAPTER` in `.env` (see [`.env.example`](../../.env.example)). Then
run one of the configs in [`configs/`](configs)
([`sherpa-qwen3-8b`](configs/sherpa-qwen3-8b.yaml), [`qwen3-8b`](configs/qwen3-8b.yaml),
[`pedrl-qwen3-8b`](configs/pedrl-qwen3-8b.yaml),
[`gemini-3.8-flash`](configs/gemini-3.8-flash.yaml)):

```bash
GPU_ID=0 bash examples/math_tutor_bench/run.sh sherpa-qwen3-8b
```

Ped-RM runs on the GPU `GPU_ID` (default 0). Re-running a command resumes it. Other
options: `MAX_SAMPLES` (examples per task; 0 = all), `REQUEST_CONCURRENCY`,
`SKIP_PED_RM=1`, `RUN_DIR` (default `results/<run_name>`), and `LORA_PROBE=warn`, which
continues when an adapter leaves the probe reply unchanged. Results: `summary.yaml`, and per task `tasks/<task>/` with metrics and raw
replies.

## Decoding and response processing

As in the benchmark: problem solving, Socratic questioning, solution correctness and
mistake location use the completion API, and the five dialogue tasks use chat, with
temperature 0, seed 42, 2048 output tokens and native thinking off. For APIs without a completion endpoint, every task is sent as one chat
message, and commercial APIs use the provider's own sampling and output length.

Replies are processed as in the paper (`teacher-student-boundary-v1`): an initial
`Teacher:` label is removed, a reply stops only at a later line-start `Teacher:` or
`Student:` label, and Solution Correctness takes the last explicit Yes/No judgment. The
summary reports Solution Correctness F1 and Mistake Location micro-F1, as the official
leaderboard does.
