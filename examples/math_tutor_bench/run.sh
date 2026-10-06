#!/usr/bin/env bash
# Usage: [GPU_ID=0] bash examples/math_tutor_bench/run.sh CONFIG
#
# Generates all nine tasks through the teacher's OpenAI-compatible API, scores the
# four pedagogy tasks with the official Ped-RM on GPU_ID, and writes the
# leaderboard summary. CONFIG (a name in configs/ or a YAML path) names the
# teacher; with `provider` it is a commercial API (see api_generate.py).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${PYTHON:-$REPO_ROOT/.venv/bin/python}
UPSTREAM_DIR="$SCRIPT_DIR/.runtime/upstream"
DEPENDENCY_ROOT="$SCRIPT_DIR/.runtime/python"
DATASETS_CACHE="$SCRIPT_DIR/.runtime/hf_datasets"
DATASET_HUB_CACHE="$SCRIPT_DIR/.runtime/hf_hub"
UPSTREAM_REVISION=6faed173ec2bef55cb899b2a3e0f93982f9cb176

usage() {
  printf 'Usage: [GPU_ID=0] bash %s CONFIG\n' "$0" >&2
  printf 'CONFIG is a name in %s/configs or a YAML path.\n' "$SCRIPT_DIR" >&2
}

if (( $# != 1 )); then
  usage
  exit 2
fi
CONFIG=$1
if [[ ! -f "$CONFIG" ]]; then
  CONFIG="$SCRIPT_DIR/configs/${1%.yaml}.yaml"
fi
if [[ ! -f "$CONFIG" ]]; then
  printf 'No such config: %s\n' "$1" >&2
  usage
  exit 2
fi
CONFIG=$(realpath -e -- "$CONFIG")
if [[ ! -x "$PYTHON" ]]; then
  printf 'Python interpreter is not executable: %s\n' "$PYTHON" >&2
  exit 1
fi
cd "$REPO_ROOT"

GPU_ID=${GPU_ID:-0}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_SAMPLES=${MAX_SAMPLES:-0}
PED_RM_MODEL=${PED_RM_MODEL:-eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel}
SKIP_PED_RM=${SKIP_PED_RM:-0}
# The LoRA liveness probe fails when the adapter leaves the probe prompt's greedy
# reply unchanged, which a weakly trained adapter can do although the server
# loaded and applied it. LORA_PROBE=warn reports that and continues.
LORA_PROBE=${LORA_PROBE:-strict}

for numeric in GPU_ID MAX_TOKENS MAX_SAMPLES; do
  value=${!numeric}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a nonnegative integer; got %s\n' "$numeric" "$value" >&2
    exit 2
  fi
done
if (( MAX_TOKENS < 1 )); then
  printf 'MAX_TOKENS must be positive.\n' >&2
  exit 2
fi
if [[ "$SKIP_PED_RM" != 0 && "$SKIP_PED_RM" != 1 ]]; then
  printf 'SKIP_PED_RM must be 0 or 1.\n' >&2
  exit 2
fi
if [[ "$LORA_PROBE" != strict && "$LORA_PROBE" != warn ]]; then
  printf 'LORA_PROBE must be strict or warn.\n' >&2
  exit 2
fi

# The teacher's address, key and adapter come from the repository .env; the
# process environment wins.
RESOLVED=$(
  "$PYTHON" - "$CONFIG" <<'PY'
import os
import re
import shlex
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(Path(".env"), override=False)
config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
fields = {"version", "run_name", "model", "endpoint_env", "key_env", "adapter_env"}
if (
    not isinstance(config, dict)
    or not fields <= set(config) <= fields | {"provider"}
    or config["version"] != 1
    or not re.fullmatch(r"[A-Za-z0-9_.-]+", config["run_name"])
):
    raise SystemExit(f"Unknown or missing fields in {sys.argv[1]}")
adapter_env = config["adapter_env"] or ""
values = {
    "RUN_NAME": config["run_name"],
    "MODEL": config["model"],
    "PROVIDER": "1" if config.get("provider") else "0",
    "ENDPOINT_ENV": config["endpoint_env"],
    "ENDPOINT": os.environ.get(config["endpoint_env"], "").rstrip("/"),
    "KEY_ENV": config["key_env"],
    "API_KEY": os.environ.get(config["key_env"], ""),
    "ADAPTER_ENV": adapter_env,
    "ADAPTER": os.environ.get(adapter_env, "") if adapter_env else "",
}
for name, value in values.items():
    print(f"{name}={shlex.quote(value)}")
PY
)
eval "$RESOLVED"
RUN_DIR=${RUN_DIR:-$SCRIPT_DIR/results/$RUN_NAME}

# Resolve the Ped-RM downloaded into the default Hugging Face cache before any
# benchmark-specific cache is set. An API run that skips scoring does not need it;
# a self-served run records it in run.json either way.
PED_RM_PATH=
if [[ "$PROVIDER" == 0 || "$SKIP_PED_RM" == 0 ]]; then
  PED_RM_PATH=$(
    "$PYTHON" "$SCRIPT_DIR/score_pedrm.py" --model "$PED_RM_MODEL" --resolve-only
  )
fi

if [[ "$PROVIDER" == 1 ]]; then
  # A commercial API: every task as one chat message.
  "$PYTHON" -m examples.math_tutor_bench.api_generate generate \
    --config "$CONFIG" --env-file "$REPO_ROOT/.env" --output-dir "$RUN_DIR" \
    --limit "$MAX_SAMPLES" --concurrency "${REQUEST_CONCURRENCY:-8}"
  if [[ "$SKIP_PED_RM" == 0 ]]; then
    "$PYTHON" -m examples.math_tutor_bench.api_generate score \
      --config "$CONFIG" --output-dir "$RUN_DIR" --gpu-id "$GPU_ID" \
      --pedrm-model "$PED_RM_PATH"
  fi
  exit 0
fi

if [[ -z "$ENDPOINT" ]]; then
  printf 'Set %s to the teacher server base URL (e.g. http://127.0.0.1:30000/v1).\n' "$ENDPOINT_ENV" >&2
  exit 1
fi
if [[ -z "$API_KEY" ]]; then
  printf 'Set %s (EMPTY for an unauthenticated server).\n' "$KEY_ENV" >&2
  exit 1
fi
export MTB_API_KEY="$API_KEY"
if [[ -n "$ADAPTER_ENV" && -z "$ADAPTER" ]]; then
  printf 'Set %s to the adapter path the teacher server registered.\n' "$ADAPTER_ENV" >&2
  exit 1
fi
EVALUATION_MODE=base
if [[ -n "$ADAPTER" ]]; then
  EVALUATION_MODE=lora
fi
REQUEST_CONCURRENCY=${REQUEST_CONCURRENCY:-16}
if [[ ! "$REQUEST_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  printf 'REQUEST_CONCURRENCY must be a positive integer; got %s\n' "$REQUEST_CONCURRENCY" >&2
  exit 2
fi

# Benchmark assets are staged once with prepare.py; generation and scoring then
# run offline.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/tasks" "$SCRIPT_DIR/.runtime"
RUN_DIR=$(realpath -e -- "$RUN_DIR")

"$PYTHON" "$SCRIPT_DIR/prepare.py" \
  --upstream "$UPSTREAM_DIR" \
  --dependency-root "$DEPENDENCY_ROOT" \
  --python "$PYTHON" \
  --offline

export PYTHONPATH="$DEPENDENCY_ROOT:$UPSTREAM_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE="$DATASETS_CACHE"
export HF_HUB_CACHE="$DATASET_HUB_CACHE"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# Validate the already-staged official datasets before generating. Offline mode
# makes a missing artifact fail immediately without any network retry.
"$PYTHON" "$SCRIPT_DIR/prefetch_datasets.py" --cache-dir "$HF_DATASETS_CACHE"

MANIFEST="$RUN_DIR/run.json"
"$PYTHON" - "$MANIFEST" "$CONFIG" "$MODEL" "$ENDPOINT" "$ADAPTER" "$PED_RM_PATH" \
  "$UPSTREAM_REVISION" "$GPU_ID" "$MAX_TOKENS" "$MAX_SAMPLES" "$EVALUATION_MODE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from examples.math_tutor_bench.run_task import RESPONSE_PROCESSING

path = Path(sys.argv[1])
new = {
    "config": Path(sys.argv[2]).name,
    "model": sys.argv[3],
    "endpoint_sha256": hashlib.sha256(sys.argv[4].encode()).hexdigest(),
    "adapter": sys.argv[5] or None,
    "pedrm_model": sys.argv[6],
    "math_tutor_bench_revision": sys.argv[7],
    "gpu_id": sys.argv[8],
    "temperature": 0.0,
    "seed": 42,
    "max_tokens": int(sys.argv[9]),
    "max_samples": int(sys.argv[10]),
    "evaluation_mode": sys.argv[11],
    "response_processing": RESPONSE_PROCESSING,
    "status": "running",
}
if path.exists():
    old = json.loads(path.read_text(encoding="utf-8"))
    immutable = ("model", "adapter", "pedrm_model", "math_tutor_bench_revision", "max_tokens", "evaluation_mode", "response_processing")
    mismatches = [key for key in immutable if old.get(key) != new.get(key)]
    if mismatches:
        raise SystemExit(f"RUN_DIR belongs to incompatible settings: {mismatches}; use a new RUN_DIR to preserve old results")
    if old.get("max_samples", 0) not in (0, new["max_samples"]) and new["max_samples"]:
        raise SystemExit("cannot mix two nonzero MAX_SAMPLES values in one RUN_DIR")
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

printf '[run] mode:       %s\n' "$EVALUATION_MODE"
printf '[run] model:      %s\n' "$MODEL"
printf '[run] output:     %s\n' "$RUN_DIR"

if [[ "$EVALUATION_MODE" == lora ]]; then
  probe_args=(
    --base-url "$ENDPOINT"
    --model "$MODEL"
    --lora-path "$ADAPTER"
  )
  if [[ "$LORA_PROBE" == warn ]]; then
    probe_args+=(--warn-if-inactive)
  fi
  "$PYTHON" "$SCRIPT_DIR/probe_adapter.py" "${probe_args[@]}"
fi

TASKS=(
  student_solution_correctness
  mistake_location
  problem_solving
  socratic_questioning
  scaffolding_generation
  pedagogy_following
  mistake_correction
  scaffolding_generation_hard
  pedagogy_following_hard
)
for task in "${TASKS[@]}"; do
  log="$RUN_DIR/logs/task-$task.log"
  printf '[task] %s\n' "$task"
  task_args=(
    --upstream "$UPSTREAM_DIR"
    --task "$task"
    --base-url "$ENDPOINT"
    --model "$MODEL"
    --output "$RUN_DIR/tasks/$task"
    --concurrency "$REQUEST_CONCURRENCY"
    --max-tokens "$MAX_TOKENS"
    --max-samples "$MAX_SAMPLES"
  )
  if [[ "$EVALUATION_MODE" == lora ]]; then
    task_args+=(--lora-path "$ADAPTER")
  fi
  if ! "$PYTHON" "$SCRIPT_DIR/run_task.py" "${task_args[@]}" >>"$log" 2>&1; then
    printf 'Task %s failed; see %s\n' "$task" "$log" >&2
    exit 1
  fi
done

if [[ "$SKIP_PED_RM" == 0 ]]; then
  printf '[pedrm] scoring four open-ended tasks on GPU %s\n' "$GPU_ID"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/score_pedrm.py" \
    --model "$PED_RM_PATH" \
    --tasks-root "$RUN_DIR/tasks" \
    --output "$RUN_DIR/pedrm" 2>&1 | tee -a "$RUN_DIR/logs/pedrm.log"
fi

"$PYTHON" "$SCRIPT_DIR/summarize.py" \
  --run-dir "$RUN_DIR" \
  --upstream-revision "$UPSTREAM_REVISION" | tee "$RUN_DIR/leaderboard.txt"

"$PYTHON" - "$MANIFEST" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["status"] = "complete"
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

printf '[done] full report: %s/summary.yaml\n' "$RUN_DIR"
