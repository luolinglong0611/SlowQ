#!/usr/bin/env bash
# Step 1 again, with a context window big enough for the workload that step 1
# skipped.
#
#   ./run_round1_longctx.sh [out-root]
#
# Why this exists: run_round1.sh step 1 runs with the harness default
# --max-context 1024. The `long-prompt-short-answer` workload's prompt is one fixed
# piece of text, but it tokenises per model, and only SmolLM2's tokenizer makes it long
# enough to skip: 1662 tokens for SmolLM2-135M/-360M (1662 + 32 > 1024), versus 584 for
# Qwen3-0.6B/-1.7B and 601 for Qwen2.5-0.5B, which fit and were measured in step 1. So
# the SmolLM2 base and opt trees both recorded
#   {"skipped": "prompt (1662 tokens) + max_new_tokens (32) = 1694 exceeds
#     max_context=1024 ..."}
# instead of a measurement. Both trees skipped for the same reason, so it is not
# a regression — but it is the one workload that is prefill-dominated (1662 in,
# 32 out on SmolLM2; 584–601 in on Qwen), i.e. exactly the shape optimisation 1
# targets, and shipping a round-1 report without the SmolLM2 rows would leave the
# strongest case for that change unmeasured.
#
# --max-context 2048 covers SmolLM2's 1694 with room to spare. The other four
# workloads — and the Qwen models' long-prompt rows that step 1 already measured — are
# re-measured in the same run rather than reused from ab/, so every row of the table
# comes from one configuration instead of two.
#
# --suite all, not --suite workloads: the workload table's self-check column is
# `workload decode rate / this process's raw decode probe`, and a workloads-only
# run has no raw decode probe in it. Pairing those rows against ab/'s probe from
# hours earlier would produce a *cross-process* ratio, which mixes in the machine
# drift between the two runs — and machine drift is precisely what the self-check
# exists to detect. So this run collects its own micro block in the same window,
# and make_report.py marks any row without one as 待重测 rather than borrowing.
# --no-chat because the chat suite is step 1's job and does not vary with
# --max-context; skipping it keeps this run's cost to what the column needs.
#
# Sequential and interleaved per model, for the same reason as step 1: the pair
# is the unit of comparison, so the pair runs adjacent.
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
OUT_ROOT="$(out_guard_root "$PY" "${1:-}")"

# Verified against its Git revision rather than trusted as a directory that
# happens to exist; see tools/perf/baseline_tree.py.
BASE_SRC="$("$PY" tools/perf/baseline_tree.py --print-dest)"
OUT="${OUT_ROOT}/ab-longctx"

MODELS=(
  SmolLM2-135M-Instruct-slowq
  SmolLM2-360M-Instruct-slowq
  Qwen2.5-0.5B-Instruct-slowq
  Qwen3-0.6B-slowq
  Qwen3-1.7B-slowq
)

# Only this driver's own product. The run root may well exist already — an orchestrator
# created it, or run_round1.sh did — and refusing that would break the only documented
# way to run a whole re-collection.
out_guard_reserve "$PY" "$OUT"

echo "=== micro + workloads at --max-context 2048, 5 models, interleaved ===" >&2
for model in "${MODELS[@]}"; do
  for side in base opt; do
    if [ "$side" = base ]; then src="$BASE_SRC"; else src=.; fi
    echo "--- ${model} ${side} ---" >&2
    "$PY" "$CORE" --src "$src" --model "models/${model}" --suite all --no-chat \
      --max-context 2048 --label "$side" --out "${OUT}/${model}-${side}.json"
  done
done

echo "=== long-context workloads done -> ${OUT} ===" >&2
