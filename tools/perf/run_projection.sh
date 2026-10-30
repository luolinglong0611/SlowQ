#!/usr/bin/env bash
# Price optimisation 1 directly, one process per model, strictly serially.
#
#   ./run_projection.sh [out-root]
#
# This is the probe the A/B tables cannot replace. `core_ab.py` observes the
# projection *count* dropping 4 → 1 (an instrumented spy on `model.forward` /
# `model.encode`), but the base/opt delta mixes that with optimisation 2's sigmoid
# rewrite and single-token mask skip, so the count alone does not say what it was
# worth. `projection_probe.py` times `forward` and `encode` as an interleaved pair
# on one tree and subtracts inside the pair.
#
# It runs *after* run_round1.sh, because the "share of 512-token prefill" figure is
# a ratio against `<out-root>/ab/<model>-base.json`. The probe derives that path
# from --out, so a re-collection prices optimisation 1 against its own prefill
# rather than against the published round's.
#
# Separate script rather than a step inside run_round1.sh: run_round1.sh's hash is
# recorded in the tooling map of every JSON this round produced, and keeping it
# byte-identical is what makes those records reproducible from the same driver.
#
# <out-root> defaults to a fresh directory under artifacts/, created for this run and
# printed to stderr; see tools/perf/out_guard.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=tools/perf/out_guard.sh
. tools/perf/out_guard.sh

PY=".venv/bin/python"
OUT_ROOT="$(out_guard_root "$PY" "${1:-}")"

MODELS=(
  SmolLM2-135M-Instruct-slowq
  SmolLM2-360M-Instruct-slowq
  Qwen2.5-0.5B-Instruct-slowq
  Qwen3-0.6B-slowq
  Qwen3-1.7B-slowq
)

PROJ="${OUT_ROOT}/projection"

# A second attempt must not replace the first one's JSON, and must not be waved through
# by relocating it either. This probe is the least repeatable thing in the round — its
# answer is a *difference* between two ~540 ms medians, so a 1% wobble in either half
# moves the result severalfold — and the first attempt is exactly the evidence that
# shows it. This driver used to archive the occupied directory under a
# projection-attempt-N name and carry on: the bytes survived, but the run that should
# have been refused reported success and the directory the operator pointed at no longer
# held what they put there. Overwriting and relocating are the same failure, and the
# second one is quieter. Re-measure into a new out-root instead; two directories are
# what make the two attempts comparable.
out_guard_reserve "$PY" "$PROJ"

if [ ! -d "${OUT_ROOT}/ab" ]; then
  echo "run_projection: ${OUT_ROOT}/ab does not exist — run run_round1.sh first." >&2
  echo "                Without it the probe records engine_prefill_base=null and the" >&2
  echo "                share of prefill cannot be computed." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  echo "--- projection probe: ${model} ---" >&2
  "$PY" tools/perf/projection_probe.py \
    --model "models/${model}" \
    --out "${PROJ}/${model}.json"
done

echo "=== projection probe done -> ${PROJ} ===" >&2
