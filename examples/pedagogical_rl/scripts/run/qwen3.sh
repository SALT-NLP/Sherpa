#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  printf '%s\n' \
    'PedagogicalRL training on one node: local Qwen3-1.7B student + in-engine Qwen3-8B judges.' \
    '' \
    'Usage:' \
    '  bash examples/pedagogical_rl/scripts/run/qwen3.sh [CONFIG.yaml] [CONFIG_OVERRIDE ...]' \
    '' \
    'The GPU count and the actor/rollout split are read from the config' \
    '(cluster.n_gpus_per_node, fsdp:d<actor>p1t1, sglang:d<rollout>p1t1).' \
    'The student server shares the rollout GPUs. Trailing arguments are' \
    'config overrides and are forwarded to training unchanged.' \
    '' \
    'Environment overrides:' \
    '  CUDA_VISIBLE_DEVICES           GPU ids to use; the first n_gpus_per_node are taken.' \
    '  STUDENT_MODEL_PATH             Student weights: HF id or local path (default Qwen/Qwen3-1.7B).' \
    '  STUDENT_PORT                   Local OpenAI port (default 30001).' \
    '  STUDENT_MEM_FRACTION_STATIC    Student GPU allocation (default 0.45).' \
    '  DRY_RUN=1                      Validate and print commands only.'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG="examples/pedagogical_rl/configs/pedrl_qwen3_8b.yaml"
if (( $# > 0 )) && [[ "$1" == *.yaml || "$1" == *.yml ]]; then
  CONFIG="$1"
  shift
fi

USER_OVERRIDES=("$@")

if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing executable venv Python: %s\nRun: uv sync --extra cuda\n' "$PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  printf 'Config does not exist: %s\n' "$CONFIG" >&2
  exit 1
fi
for command in curl setsid; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf '%s is required.\n' "$command" >&2
    exit 1
  fi
done

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

# AReaL's local scheduler launches workers with the literal `python3` command,
# so the venv must come first on PATH.
unset PYTHONHOME
export VIRTUAL_ENV="$ROOT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r
if [[ "$(command -v python3)" != "$VIRTUAL_ENV/bin/python3" ]]; then
  printf 'Worker python3 did not resolve to the project venv: %s\n' \
    "$(command -v python3)" >&2
  exit 1
fi

# All inference is local: models load from the local HF cache, and HTTP clients
# must not route localhost through a proxy.
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export DO_NOT_TRACK=1
export PYTHONPATH="$ROOT_DIR"
export PYTHONUNBUFFERED=1
# GUIDED/ATTEMPTED assignment and student names hash the problem text.
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export TOKENIZERS_PARALLELISM=false

DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  printf 'DRY_RUN must be 0 or 1; got %q.\n' "$DRY_RUN" >&2
  exit 2
fi

STUDENT_HOST=127.0.0.1
STUDENT_PORT="${STUDENT_PORT:-30001}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-Qwen/Qwen3-1.7B}"
STUDENT_MODEL="${STUDENT_MODEL:-qwen3-1.7b}"
STUDENT_CONTEXT_LENGTH="${STUDENT_CONTEXT_LENGTH:-40960}"
STUDENT_MEM_FRACTION_STATIC="${STUDENT_MEM_FRACTION_STATIC:-0.45}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
STUDENT_PROBE_TIMEOUT="${STUDENT_PROBE_TIMEOUT:-120}"

if [[ ! "$STUDENT_PORT" =~ ^[1-9][0-9]*$ ]] || (( STUDENT_PORT > 65535 )); then
  printf 'STUDENT_PORT must be in 1..65535; got %q.\n' "$STUDENT_PORT" >&2
  exit 2
fi
if [[ "$STUDENT_MODEL_PATH" == /* || "$STUDENT_MODEL_PATH" == ./* ]]; then
  if [[ ! -f "$STUDENT_MODEL_PATH/config.json" ]]; then
    printf 'Local STUDENT_MODEL_PATH has no config.json: %s\n' "$STUDENT_MODEL_PATH" >&2
    exit 1
  fi
fi

STUDENT_BASE_URL="http://$STUDENT_HOST:$STUDENT_PORT/v1"
export STUDENT_BASE_URL STUDENT_MODEL

CONFIG_SUMMARY="$($PYTHON -B - "$CONFIG" "$STUDENT_BASE_URL" "$STUDENT_MODEL" \
  "${USER_OVERRIDES[@]}" <<'PY'
import json
import re
import sys
from urllib.parse import urlparse

from omegaconf import OmegaConf

from areal.api.cli_args import parse_cli_args, to_structured_cfg
from examples.pedagogical_rl.config import PedagogicalRLConfig

config_path, student_url, student_model, *overrides = sys.argv[1:]
raw, _ = parse_cli_args(["--config", config_path, *overrides])
config = OmegaConf.to_object(to_structured_cfg(raw, PedagogicalRLConfig))
if not isinstance(config, PedagogicalRLConfig):
    raise SystemExit(f"Expected PedagogicalRLConfig, got {type(config).__name__}")


def devices(role, backend, prefix):
    match = re.fullmatch(rf"{prefix}:d(\d+)p(\d+)t(\d+)", backend)
    if match is None:
        raise SystemExit(f"{role} backend has unsupported form: {backend!r}")
    count, pipeline, tensor = map(int, match.groups())
    if pipeline != 1 or tensor != 1:
        raise SystemExit(f"{role} backend must use p1t1: {backend!r}")
    return count


gpu_count = int(config.cluster.n_gpus_per_node)
rollout_gpus = devices("rollout", config.rollout.backend, "sglang")
actor_gpus = devices("actor", config.actor.backend, "fsdp")
if actor_gpus + rollout_gpus != gpu_count:
    raise SystemExit(
        f"Config requests {actor_gpus} actor + {rollout_gpus} rollout GPUs, "
        f"but cluster.n_gpus_per_node={gpu_count}."
    )
if config.cluster.n_nodes != 1:
    raise SystemExit("This launcher requires cluster.n_nodes=1.")
if config.scheduler.type != "local":
    raise SystemExit("This launcher requires scheduler.type=local.")
if config.student_model.mode != "api":
    raise SystemExit("The frozen student must use the launcher's local API server.")
if config.student_model.base_url.rstrip("/") != student_url.rstrip("/"):
    raise SystemExit(
        f"Student URL must resolve to local {student_url}, got "
        f"{config.student_model.base_url}."
    )
if urlparse(config.student_model.base_url).hostname not in {"127.0.0.1", "localhost"}:
    raise SystemExit("Refusing to call a non-local student endpoint.")
if config.student_model.model != student_model:
    raise SystemExit(
        f"Configured student {config.student_model.model!r} does not match "
        f"served model {student_model!r}."
    )
if config.judge_model.mode != "self":
    raise SystemExit("This launcher requires judge_model.mode=self.")
if config.judge_model.base_url or config.judge_model.api_key:
    raise SystemExit("Self-hosted judge must not contain an API endpoint or key.")

print(json.dumps({
    "config": config_path,
    "experiment_name": config.experiment_name,
    "trial_name": config.trial_name,
    "gpus": gpu_count,
    "actor_gpus": actor_gpus,
    "rollout_gpus": rollout_gpus,
    "student": f"local {config.student_model.model}",
    "student_url": config.student_model.base_url,
    "judges": "rollout base model (LoRA disabled)",
}, indent=2, sort_keys=True))
PY
)"
printf '%s\n' "$CONFIG_SUMMARY"
read_count() { sed -n "s/.*\"$1\": \([0-9][0-9]*\).*/\1/p" <<<"$CONFIG_SUMMARY"; }
GPU_COUNT="$(read_count gpus)"
ACTOR_GPU_COUNT="$(read_count actor_gpus)"
ROLLOUT_GPU_COUNT="$(read_count rollout_gpus)"
for count in "$GPU_COUNT" "$ACTOR_GPU_COUNT" "$ROLLOUT_GPU_COUNT"; do
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Could not read the GPU split from the resolved config.\n' >&2
    exit 1
  fi
done

GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a RAW_GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
else
  VISIBLE_COUNT="$($PYTHON -B -c 'import torch; print(torch.cuda.device_count())')"
  RAW_GPU_IDS=()
  for ((index = 0; index < VISIBLE_COUNT; index++)); do
    RAW_GPU_IDS+=("$index")
  done
  if [[ "$DRY_RUN" == 1 && ${#RAW_GPU_IDS[@]} -lt $GPU_COUNT ]]; then
    RAW_GPU_IDS=()
    for ((index = 0; index < GPU_COUNT; index++)); do
      RAW_GPU_IDS+=("$index")
    done
  fi
fi
declare -A SEEN_GPU_IDS=()
for raw_gpu_id in "${RAW_GPU_IDS[@]}"; do
  gpu_id="${raw_gpu_id//[[:space:]]/}"
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_GPU_IDS[$gpu_id]:-}" ]]; then
    printf 'CUDA_VISIBLE_DEVICES must contain unique integer ids; got %q.\n' \
      "${CUDA_VISIBLE_DEVICES:-}" >&2
    exit 2
  fi
  SEEN_GPU_IDS[$gpu_id]=1
  GPU_IDS+=("$gpu_id")
done
if (( ${#GPU_IDS[@]} < GPU_COUNT )); then
  printf '%d GPUs are required; only %d are visible: %s\n' \
    "$GPU_COUNT" "${#GPU_IDS[@]}" "${CUDA_VISIBLE_DEVICES:-<none>}" >&2
  exit 1
fi
SELECTED_GPU_IDS=("${GPU_IDS[@]:0:GPU_COUNT}")
join_by_comma() { local IFS=,; printf '%s' "$*"; }
SELECTED_GPU_SPEC="$(join_by_comma "${SELECTED_GPU_IDS[@]}")"
export CUDA_VISIBLE_DEVICES="$SELECTED_GPU_SPEC"

if [[ "$DRY_RUN" != 1 ]]; then
  VISIBLE_COUNT="$($PYTHON -B -c 'import torch; print(torch.cuda.device_count())')"
  if [[ "$VISIBLE_COUNT" != "$GPU_COUNT" ]]; then
    printf 'PyTorch sees %s GPUs after CUDA_VISIBLE_DEVICES=%s; expected %s.\n' \
      "$VISIBLE_COUNT" "$CUDA_VISIBLE_DEVICES" "$GPU_COUNT" >&2
    exit 1
  fi
fi

# AReaL's scheduler gives the actor the leading ids and the rollout engine the
# trailing ones; the student is pinned to the rollout GPUs.
ACTOR_GPU_IDS=("${SELECTED_GPU_IDS[@]:0:ACTOR_GPU_COUNT}")
ROLLOUT_GPU_IDS=("${SELECTED_GPU_IDS[@]:ACTOR_GPU_COUNT:ROLLOUT_GPU_COUNT}")
ACTOR_GPU_SPEC="$(join_by_comma "${ACTOR_GPU_IDS[@]}")"
ROLLOUT_GPU_SPEC="$(join_by_comma "${ROLLOUT_GPU_IDS[@]}")"

LOCAL_MODEL_LOG_ROOT="${LOCAL_MODEL_LOG_ROOT:-$ROOT_DIR/output/local_models}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
STUDENT_SERVER_LOG="$LOCAL_MODEL_LOG_ROOT/student-pedagogical-rl-$stamp.log"

STUDENT_CMD=(
  "$PYTHON" -m sglang.launch_server
  --model-path "$STUDENT_MODEL_PATH"
  --served-model-name "$STUDENT_MODEL"
  --host "$STUDENT_HOST"
  --port "$STUDENT_PORT"
  --tp-size 1
  --dp-size "$ROLLOUT_GPU_COUNT"
  --load-balance-method round_robin
  --context-length "$STUDENT_CONTEXT_LENGTH"
  --mem-fraction-static "$STUDENT_MEM_FRACTION_STATIC"
)
TRAIN_CMD=("$PYTHON" -m examples.pedagogical_rl.train --config "$CONFIG" "${USER_OVERRIDES[@]}")

printf 'GPU plan: actor=[%s], rollout+student=[%s]\n' \
  "$ACTOR_GPU_SPEC" "$ROLLOUT_GPU_SPEC"
printf 'Student endpoint: %s\nStudent log: %s\n' \
  "$STUDENT_BASE_URL" "$STUDENT_SERVER_LOG"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY RUN: student command:\n  CUDA_VISIBLE_DEVICES=%q setsid ' "$ROLLOUT_GPU_SPEC"
  printf '%q ' "${STUDENT_CMD[@]}"
  printf '\nDRY RUN: training command:\n  CUDA_VISIBLE_DEVICES=%q setsid ' "$SELECTED_GPU_SPEC"
  printf '%q ' "${TRAIN_CMD[@]}"
  printf '\n'
  exit 0
fi

if "$PYTHON" -B - "$STUDENT_HOST" "$STUDENT_PORT" <<'PY'
import socket
import sys
try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
then
  printf 'Port %s:%s is occupied; refusing to reuse an unverified server.\n' \
    "$STUDENT_HOST" "$STUDENT_PORT" >&2
  exit 1
fi

STUDENT_SERVER_PID=""
TRAIN_PID=""
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "$TRAIN_PID" "$STUDENT_SERVER_PID"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      sleep 1
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$LOCAL_MODEL_LOG_ROOT"
printf 'Starting local student with DP=%s.\n' "$ROLLOUT_GPU_COUNT"
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU_SPEC" setsid "${STUDENT_CMD[@]}" \
  >"$STUDENT_SERVER_LOG" 2>&1 &
STUDENT_SERVER_PID=$!

deadline=$((SECONDS + SERVER_READY_TIMEOUT))
while (( SECONDS < deadline )); do
  if ! kill -0 "$STUDENT_SERVER_PID" 2>/dev/null; then
    printf 'Student server exited during startup.\n' >&2
    tail -n 120 "$STUDENT_SERVER_LOG" >&2 || true
    exit 1
  fi
  if curl --silent --show-error --fail --max-time 3 \
      "$STUDENT_BASE_URL/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! curl --silent --show-error --fail --max-time 3 \
    "$STUDENT_BASE_URL/models" >/dev/null 2>&1; then
  printf 'Student server was not ready after %s seconds.\n' "$SERVER_READY_TIMEOUT" >&2
  tail -n 120 "$STUDENT_SERVER_LOG" >&2 || true
  exit 1
fi

if ! "$PYTHON" -B - "$STUDENT_BASE_URL" "$STUDENT_MODEL" \
    "$STUDENT_PROBE_TIMEOUT" <<'PY'
import json
import sys
import urllib.request

base_url, model, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "temperature": 0,
    "top_p": 1,
    "top_k": 20,
    "min_p": 0,
    "max_tokens": 2,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
request = urllib.request.Request(
    base_url + "/chat/completions",
    data=payload,
    headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=timeout) as response:
    result = json.load(response)
if not result.get("choices"):
    raise SystemExit("Local student decode returned no choices.")
print(f"Local student decode probe passed for {model}.")
PY
then
  printf 'Student decode probe failed.\n' >&2
  tail -n 120 "$STUDENT_SERVER_LOG" >&2 || true
  exit 1
fi

printf 'Local student is healthy; starting PedagogicalRL training.\n'
CUDA_VISIBLE_DEVICES="$SELECTED_GPU_SPEC" setsid "${TRAIN_CMD[@]}" &
TRAIN_PID=$!

while kill -0 "$TRAIN_PID" 2>/dev/null && kill -0 "$STUDENT_SERVER_PID" 2>/dev/null; do
  sleep 2
done
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  if wait "$TRAIN_PID"; then
    exit 0
  else
    exit $?
  fi
fi
printf 'Local student server exited while training was running.\n' >&2
tail -n 120 "$STUDENT_SERVER_LOG" >&2 || true
exit 1
