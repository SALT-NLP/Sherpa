#!/usr/bin/env bash
# Sherpa training with the Ministral-3-8B teacher: checks the Ministral
# environment and checkpoint, then runs the common launcher (qwen3.sh).
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf '%s\n' \
    'Usage: bash examples/sherpa/scripts/run/ministral3.sh [--check-training] [CONFIG.yaml] [OVERRIDES...]' \
    'Defaults to examples/sherpa/configs/ministral3_8b.yaml and needs the Ministral' \
    'environment (envs/ministral; see examples/sherpa/README.md).' \
    'The pinned BF16 snapshot is read from the local Hugging Face cache;' \
    'MINISTRAL_MODEL_PATH overrides it.' \
    '--check-training checks config/tokenizer only; it does not certify GPU training.'
  exit 0
fi
CHECK_ARGS=()
if [[ "${1:-}" == "--check-training" ]]; then
  CHECK_ARGS=(--training-only)
  shift
fi
CONFIG="examples/sherpa/configs/ministral3_8b.yaml"
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then
  CONFIG="$1"
  shift
fi
PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing .venv Python. Create the Ministral environment first.\n' >&2
  exit 1
fi
unset PYTHONHOME
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
export PYTHONPATH="$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
# Keep this environment's compiled kernels and caches apart from the Qwen
# environment's, which uses different torch/Triton/FlashInfer builds.
CACHE_ROOT="${MINISTRAL_CACHE_ROOT:-$ROOT_DIR/.cache/ministral3}"
export AREAL_CACHE_DIR="$CACHE_ROOT/areal"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
if [[ -z "${MINISTRAL_MODEL_PATH:-}" ]]; then
  MINISTRAL_MODEL_PATH="$("$PYTHON" -m examples.sherpa.ministral3.local_model)"
fi
export MINISTRAL_MODEL_PATH
"$PYTHON" -m examples.sherpa.ministral3.preflight \
  "${CHECK_ARGS[@]}" --config "$CONFIG" "$@"
if (( ${#CHECK_ARGS[@]} > 0 )); then exit 0; fi
exec bash "$ROOT_DIR/examples/sherpa/scripts/run/qwen3.sh" "$CONFIG" "$@"
