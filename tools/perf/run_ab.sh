#!/usr/bin/env bash
# Round-1 A/B driver. One core_ab.py process per (model, config), strictly
# sequential: two benchmarks at once would contend for the same CPU cores and
# memory bandwidth and neither number would mean anything.
#
#   ./run_ab.sh <src-dir> <out-dir> <label> [models...]
#
# <src-dir> is a directory containing src/slowq: the repo root for the optimised
# side, or the baseline snapshot for the "before" side. Get that snapshot from
# `python tools/perf/baseline_tree.py --print-dest` rather than typing its path —
# it is materialised from revision 8968e39 and hash-verified, so a stale or
# hand-edited copy fails there instead of being reported as the baseline.
# With no model list it runs the default generation set.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=tools/perf/out_guard.sh
. tools/perf/out_guard.sh

SRC="${1:?src dir}"
OUT="${2:?out dir}"
LABEL="${3:?label}"
shift 3

PY=".venv/bin/python"
HARNESS="tools/perf/core_ab.py"

# Reserved before measuring rather than at each write: see tools/perf/out_guard.py. An
# existing empty directory is accepted, so an orchestrator that created the run root
# still gets through.
out_guard_reserve "$PY" "$OUT"

# Text-generation checkpoints that the README quotes numbers for.
DEFAULT_MODELS=(
  SmolLM2-135M-Instruct-slowq
  SmolLM2-360M-Instruct-slowq
  Qwen2.5-0.5B-Instruct-slowq
  Qwen3-0.6B-slowq
  Qwen3-1.7B-slowq
)
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then MODELS=("${DEFAULT_MODELS[@]}"); fi

for model in "${MODELS[@]}"; do
  echo "=== ${LABEL} ${model} ===" >&2
  "$PY" "$HARNESS" \
    --src "$SRC" \
    --model "models/${model}" \
    --suite all \
    --label "$LABEL" \
    --out "${OUT}/${model}-${LABEL}.json"
done
echo "=== ${LABEL} done ===" >&2
