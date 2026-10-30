#!/usr/bin/env bash
# How far apart two *independent processes* land on the same measurement, and
# therefore how small a difference this machine cannot resolve.
#
#   ./run_noise_floor.sh [out-root] [rounds] [models...]
#
# Why this exists: every table in the round-1 report prints a within-process
# min–max range, and those ranges are tight (often ±0.5%), which makes a 2–3%
# difference between two configurations look decisive. It is not. A tight range
# says only that the seven repeats *inside one process* agreed; two processes
# started minutes apart can land further apart than that, and the within-process
# range cannot see it.
#
# The round-1 report answered this with a "3% cross-process floor" taken from two
# incidental runs of the same configuration (`ab/` step 1 and `sweep-opt/*
# -t-default`). Two problems: two data points is not a distribution, and taking
# the max of them as a floor for *every* workload assumes the one probe that
# happened to be duplicated is representative of all of them. This script measures
# the floor per workload instead, from its own data.
#
# Design:
#   * Paired. Each round runs base and opt back to back, so the pair — the unit
#     of comparison — is adjacent in time, the same reasoning run_round1.sh uses.
#   * Order-balanced (AB/BA). Half the rounds run base first, half run opt first.
#     Without this, "which tree ran second" is perfectly confounded with "which
#     tree is which": the second process benefits from a warm page cache and a
#     machine that has already been through the load, so an unbalanced design
#     would report that warm-up as a property of the optimisation.
#   * Independent processes. One process per (round, side). Never threads or
#     in-process re-measurement: the quantity of interest is exactly the spread
#     *between* processes, which an in-process repeat cannot contain.
#   * Strictly sequential. Two benchmarks at once contend for the same cores and
#     memory bandwidth, and then the measurement is of the contention.
#
# `rounds` defaults to 4 and must be even for AB/BA to balance; the script refuses
# an odd number rather than silently producing a design with one extra AB pair.
#
# Both within-process and between-process dispersion come out of this: each
# process already stores its own repeat samples (within), and the rounds of one
# side give the across-process spread (between). make_report.py reports them
# separately — collapsing them into one number is what produced the "3% floor".
#
# <out-root> defaults to a fresh directory under artifacts/, created for this run and
# printed to stderr; see tools/perf/out_guard.py. artifacts/perf-round1{,-rerun} are
# refused for the reason given there: they hold the raw evidence behind the published
# report, which a re-measurement adds to rather than replaces.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=tools/perf/out_guard.sh
. tools/perf/out_guard.sh

PY=".venv/bin/python"
CORE="tools/perf/core_ab.py"
# Captured before the shift below, which drops it along with the round count.
REQUESTED_ROOT="${1:-}"
ROUNDS="${2:-4}"

# Argument validation before the run root is resolved: resolving it without an argument
# *creates* a directory, and a driver that refuses the round count must not leave one
# behind for the next attempt to conflict with.
if [ "$ROUNDS" -lt 2 ]; then
  echo "run_noise_floor: need >= 2 rounds; one round cannot separate a" >&2
  echo "             between-process spread from a single observation." >&2
  exit 1
fi

if [ $((ROUNDS % 2)) -ne 0 ]; then
  echo "run_noise_floor: rounds must be even so AB and BA counts balance;" >&2
  echo "             got ${ROUNDS}. An odd count leaves one extra AB pair," >&2
  echo "             which reintroduces the order confound this design removes." >&2
  exit 1
fi

shift 2 2>/dev/null || shift $#

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(
    SmolLM2-135M-Instruct-slowq
    SmolLM2-360M-Instruct-slowq
    Qwen2.5-0.5B-Instruct-slowq
    Qwen3-0.6B-slowq
    Qwen3-1.7B-slowq
  )
fi

OUT_ROOT="$(out_guard_root "$PY" "$REQUESTED_ROOT")"

# Verified against its Git revision rather than trusted as a directory that
# happens to exist; see tools/perf/baseline_tree.py.
BASE_SRC="$("$PY" tools/perf/baseline_tree.py --print-dest)"
OUT="${OUT_ROOT}/noise-floor"
out_guard_reserve "$PY" "$OUT"

echo "=== noise floor: ${#MODELS[@]} models x ${ROUNDS} rounds, AB/BA balanced ===" >&2

for model in "${MODELS[@]}"; do
  for ((round = 1; round <= ROUNDS; round++)); do
    # Odd rounds run base first, even rounds run opt first, so across an even
    # number of rounds each tree is first exactly half the time.
    if [ $((round % 2)) -eq 1 ]; then
      order=(base opt)
    else
      order=(opt base)
    fi
    for position in 1 2; do
      side="${order[$((position - 1))]}"
      if [ "$side" = base ]; then src="$BASE_SRC"; else src=.; fi
      # The order position is recorded in the label, not only in the run order:
      # make_report.py checks that first/second counts actually balance instead of
      # trusting that this loop did what it was written to do.
      echo "--- ${model} round ${round} position ${position} (${side}) ---" >&2
      "$PY" "$CORE" --src "$src" --model "models/${model}" --suite all --no-chat \
        --label "noise-r${round}-${side}-pos${position}" \
        --out "${OUT}/${model}-r${round}-${side}.json"
    done
  done
done

echo "=== noise floor done -> ${OUT} (${ROUNDS} rounds, $((ROUNDS / 2)) AB + $((ROUNDS / 2)) BA) ===" >&2
