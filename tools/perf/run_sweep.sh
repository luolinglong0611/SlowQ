#!/usr/bin/env bash
# Thread-count and prefill-chunk-size sweep for one source tree.
#
#   ./run_sweep.sh <src-dir> <out-dir> <label> [models...]
#
# Each (model, threads, chunk) combination is its own process, run one after
# another — never in parallel, since two benchmarks would contend for the same
# cores and memory bandwidth. `requested` thread counts are recorded by the
# harness along with the BLAS backend NumPy detected; the effective worker count
# is not observable from Python and is stored as null rather than guessed.
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

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(Qwen3-0.6B-slowq SmolLM2-360M-Instruct-slowq)
fi

# Reserved here as well, even though run_round1.sh already reserved this directory
# before calling in: an empty directory is accepted, so the orchestrator's path is
# unaffected, and a standalone `run_sweep.sh . <dir> opt` gets the same refusal instead
# of writing seven JSONs over somebody's seven JSONs.
out_guard_reserve "$PY" "$OUT"

# THREAD_CONFIGS / CHUNK_CONFIGS are "tag:arg" pairs; `none` means "leave the
# environment alone", which is the shipped default and must be measured, not
# assumed to equal any explicit number.
THREAD_CONFIGS=("t-default:none" "t1:1" "t4:4" "t8:8")
CHUNK_CONFIGS=("c64:64" "c256:256" "c512:512")

run() {
  local model="$1" tag="$2"
  shift 2
  echo "=== ${LABEL} ${model} ${tag} ===" >&2
  # --no-chat: the sweep varies threads and chunk size, and the chat suite's
  # cold/warm comparison answers a different question. Running it in all seven
  # configurations would multiply the sweep's cost without adding evidence.
  "$PY" "$HARNESS" \
    --src "$SRC" \
    --model "models/${model}" \
    --suite all \
    --no-chat \
    --label "${LABEL}-${tag}" \
    "$@" \
    --out "${OUT}/${model}-${LABEL}-${tag}.json"
}

for model in "${MODELS[@]}"; do
  for entry in "${THREAD_CONFIGS[@]}"; do
    tag="${entry%%:*}"
    value="${entry##*:}"
    if [ "$value" = "none" ]; then
      run "$model" "$tag"
    else
      run "$model" "$tag" --threads "$value"
    fi
  done
  for entry in "${CHUNK_CONFIGS[@]}"; do
    tag="${entry%%:*}"
    run "$model" "$tag" --prefill-chunk-size "${entry##*:}"
  done
done
echo "=== ${LABEL} sweep done ===" >&2
