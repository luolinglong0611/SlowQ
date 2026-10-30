#!/bin/bash
# Sequential integration validation: one model dir per pytest process, no
# parallelism (these contend for the same cores the timing runs need).
#
# Every model runs even after one fails — learning in a single pass which of the
# eight fail is the point — but the accumulated failure is what the script exits
# with. Printing OVERALL_RC=1 and then exiting 0 left the verdict in a status line
# that nothing downstream reads: a `&&` chain, a CI job and a human checking `$?`
# all saw success.
#
# <log-dir> defaults to a fresh directory under artifacts/, printed to stderr, and an
# existing one is refused: these logs are the evidence that a given round of real-model
# validation happened, and a second run writing over the first leaves the first
# unfindable. See tools/perf/out_guard.py.
#
# `set -C` covers what the reserve cannot. Reserving <log-dir> proves it was empty a
# moment ago; the eight per-model logs are written with `>`, which truncates whatever is
# at the path by the time the loop reaches it. Under noclobber that redirect is refused
# instead, the model is not run, and the failure lands in FAIL like any other.
set -uC
cd "$(dirname "$0")/../.."

# shellcheck source=tools/perf/out_guard.sh
. tools/perf/out_guard.sh

PY=".venv/bin/python"
OUT="$(out_guard_root "$PY" "${1:-}")" || exit 1
out_guard_reserve "$PY" "$OUT" || exit 1

MODELS=(
  Qwen3-0.6B-slowq
  Qwen2.5-0.5B-Instruct-slowq
  SmolLM2-135M-Instruct-slowq
  SmolVLM-256M-Instruct-slowq
  Qwen3-Embedding-0.6B-slowq
  Qwen2.5-Coder-0.5B-Instruct-slowq
  Qwen3-1.7B-slowq
  SmolLM2-360M-Instruct-slowq
)

# All eight before the first one, so a conflict in the last model is not discovered an
# hour into the run. These are file paths, so an empty directory at one of them is a
# conflict too — there is no log to write there.
LOGS=()
for m in "${MODELS[@]}"; do LOGS+=("${OUT}/${m}.log"); done
out_guard_check_file "$PY" "${LOGS[@]}" || exit 1

FAIL=0
for m in "${MODELS[@]}"; do
  echo "=== $m ==="
  log="${OUT}/${m}.log"
  # Explicit rather than left to the redirect: a refused redirect would otherwise be
  # reported as pytest's own failure, and `tail -3` below would print the *previous*
  # occupant's log as though it were this model's result.
  if [ -e "$log" ]; then
    echo "run_integration: ${log} 已存在，不覆盖；本模型的集成验证没有运行" >&2
    FAIL=1
    continue
  fi
  SLOWQ_MODEL_DIR="models/$m" "$PY" -m pytest -m integration -q \
    > "$log" 2>&1
  rc=$?
  if [ ! -s "$log" ]; then
    # A model whose log could not be written has no evidence behind whatever rc came
    # back, so its result cannot be reported either way.
    echo "run_integration: ${log} 没有写成，本模型的结论不可追溯" >&2
    FAIL=1
    continue
  fi
  tail -3 "$log"
  # A missing interpreter exits 127 before pytest ever starts, which is a failure
  # of this run and not an empty pass.
  [ $rc -ne 0 ] && FAIL=1
done
echo "OVERALL_RC=$FAIL"
exit $FAIL
