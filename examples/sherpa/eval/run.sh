#!/usr/bin/env bash
# Usage: bash examples/sherpa/eval/run.sh CONFIG [options]   (CONFIG: a name in configs/)
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"
exec "${SHERPA_PYTHON:-$REPO_DIR/.venv/bin/python}" -m examples.sherpa.eval.runner "$@"
