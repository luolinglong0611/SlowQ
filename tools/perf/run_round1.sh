#!/usr/bin/env bash
# The measurement runs behind docs/推理加速-第1轮-2026-10-23.md, in one place so
# the report's "commands" section is the thing that actually produced the numbers.
#
#   ./run_round1.sh [out-root]
#
# Everything is strictly sequential. Two benchmarks at once would contend for the
# same cores and the same memory bandwidth, and neither number would mean
# anything — the same reason run_ab.sh and run_sweep.sh are sequential.
#
# Step 1 deliberately interleaves the two trees *within* each model rather than
# running all five baselines and then all five optimised runs. This machine
# throttles and its page cache holds about four Qwen3-1.7B weight sets, so two
# runs an hour apart do not see the same machine; two runs a minute apart
# mostly do. The pair is the unit of comparison, and the pair is adjacent.
#
# <out-root> defaults to a *fresh* directory under artifacts/, created for this run and
# printed to stderr. It used to default to artifacts/perf-round1-rerun — a directory
# that already held the JSON the published documents cite — and the only protection was
#
#   if [ "$OUT_ROOT" = "artifacts/perf-round1" ]; then exit 1; fi
#
# a string comparison that `artifacts/perf-round1/` walks straight past, that an absolute
# path or a symlink walks straight past, and that never covered the default at all. Both
# clobbers were reproduced. See tools/perf/out_guard.py for the rule; re-measuring after a
# source fix produces new evidence, it does not retroactively change what the old run
# measured, and deleting the old numbers to present the new ones as the same series would
# make the correction unauditable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=tools/perf/out_guard.sh
. tools/perf/out_guard.sh

PY=".venv/bin/python"
CORE="tools/perf/core_ab.py"
OUT_ROOT="$(out_guard_root "$PY" "${1:-}")"

# The baseline is a Git revision, not the snapshot directory. baseline_tree.py
# materialises it if absent and hash-verifies it if present, so a stale or
# hand-edited snapshot fails here rather than being measured and reported as
# "before".
BASE_SRC="$("$PY" tools/perf/baseline_tree.py --print-dest)"
AB="${OUT_ROOT}/ab"
SWEEP="${OUT_ROOT}/sweep-opt"

MODELS=(
  SmolLM2-135M-Instruct-slowq
  SmolLM2-360M-Instruct-slowq
  Qwen2.5-0.5B-Instruct-slowq
  Qwen3-0.6B-slowq
  Qwen3-1.7B-slowq
)

# All four product directories in one call, before step 1 rather than at it: a run that
# creates ab/ and vision/ and then discovers documents/ was taken leaves two directories
# of evidence from a run that never finished, and the next attempt conflicts with them.
out_guard_reserve "$PY" "$AB" "$SWEEP" "${OUT_ROOT}/vision" "${OUT_ROOT}/documents"

# 1. Per model: baseline then optimised, back to back, all three suites.
#    `chat` runs on both trees — its cold path is the same standalone
#    `generate`-per-turn that the CLI did before Sessions existed, so
#    base-cold against opt-cold is the regression check on "adding Session did
#    not slow down the plain request path", and opt-cold against opt-warm is
#    the reuse measurement itself.
echo "=== step 1: interleaved A/B, 5 models, micro + workloads + chat ===" >&2
for model in "${MODELS[@]}"; do
  for side in base opt; do
    if [ "$side" = base ]; then src="$BASE_SRC"; else src=.; fi
    echo "--- ${model} ${side} ---" >&2
    "$PY" "$CORE" --src "$src" --model "models/${model}" --suite all \
      --label "$side" --out "${AB}/${model}-${side}.json"
  done
done

# 2. Thread and chunk sweep on the optimised tree. Same two models and same
#    seven configurations as the baseline sweep in ../baseline/, so the columns
#    line up. The sweep answers "which knobs help", not "did this change help",
#    so it does not need the interleaving above.
echo "=== step 2: optimised tree, thread + chunk sweep ===" >&2
bash tools/perf/run_sweep.sh . "$SWEEP" opt

# 3. Image feature cache, both trees, adjacent. The baseline has no
#    `image_cache_bytes` parameter, so its three cache-on paths report
#    themselves unsupported and only the cache-off path runs — which is exactly
#    the "before" number.
echo "=== step 3: image feature cache A/B ===" >&2
VISION_ARGS=(
  --model models/SmolVLM-256M-Instruct-slowq
  --image artifacts/validation/SmolVLM-256M-Instruct-slowq/vision-grid-1x4.png
  --image-other artifacts/validation/SmolVLM-256M-Instruct-slowq/vision-grid-2x4.png
  --warmup 1 --repeat 5 --max-new-tokens 32
)
"$PY" tools/perf/vision_cache_ab.py --src "$BASE_SRC" "${VISION_ARGS[@]}" \
  --label vision-base --out "${OUT_ROOT}/vision/base.json"
"$PY" tools/perf/vision_cache_ab.py --src . "${VISION_ARGS[@]}" \
  --label vision-opt --out "${OUT_ROOT}/vision/opt.json"

# 4. Pre-encoded document index, both trees, adjacent. Same reasoning: the
#    baseline has no `index_documents`, so it reports the search path only.
echo "=== step 4: document index A/B ===" >&2
DOC_ARGS=(
  --model models/Qwen3-Embedding-0.6B-slowq
  --corpus-size 16 --queries 8 --warmup 1 --repeat 3
)
"$PY" tools/perf/document_index_ab.py --src "$BASE_SRC" "${DOC_ARGS[@]}" \
  --label doc-base --out "${OUT_ROOT}/documents/base.json"
"$PY" tools/perf/document_index_ab.py --src . "${DOC_ARGS[@]}" \
  --label doc-opt --out "${OUT_ROOT}/documents/opt.json"

echo "=== round-1 measurements done -> ${OUT_ROOT} ===" >&2
