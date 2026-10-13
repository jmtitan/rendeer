#!/usr/bin/env bash
set -euo pipefail
RENDEER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RENDEER_TEST_OUTPUT="${RENDEER_TEST_OUTPUT:-/scratch/mingyan_root/mingyan/jmtang/outputs/LLM4Policy/synthetic/rendeer}"
export RENDEER_PYTHON="${RENDEER_PYTHON:-/scratch/mingyan_root/mingyan/jmtang/env/rendeer-dev/bin/python}"
mkdir -p "$RENDEER_TEST_OUTPUT"
export MPLCONFIGDIR="$RENDEER_TEST_OUTPUT/cache/matplotlib"
export XDG_CACHE_HOME="$RENDEER_TEST_OUTPUT/cache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RENDEER_ROOT/src:$RENDEER_ROOT/examples${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$RENDEER_TEST_OUTPUT/tmp"
export JUPYTER_RUNTIME_DIR="$RENDEER_TEST_OUTPUT/tmp/jupyter"
export IPYTHONDIR="$RENDEER_TEST_OUTPUT/cache/ipython"
export UV_CACHE_DIR=/scratch/mingyan_root/mingyan/jmtang/env/cache
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR"
cd "$RENDEER_ROOT"
"$RENDEER_PYTHON" "$@"
