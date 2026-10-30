#!/bin/bash
# Re-capture the SlowQ side of the reference validation with the *current* source,
# then compare it against the stored official reference, for all eight models.
#
# Serial on purpose: these are the same cores the timing runs need, and a capture
# that contends for them produces numbers worth less than the wait it saved.
#
# Only the `numpy` half runs here. The `ref` half needs torch/transformers, which
# this project deliberately does not depend on; the official side was captured once
# and is reused, so what this driver proves is "the current source still reproduces
# the reference", not "the reference is still right".
#
# The capture protocol is read back out of each model's reference.json instead of
# being repeated here: `numpy` takes its protocol from the command line while
# `compare` takes it from reference.json, and the two must agree or the comparison
# is between different experiments. `max_context` is the one value reference.json
# does not record, so it is passed explicitly where the default cannot hold it.
# `set -C` covers what the reserve cannot: it proves $OUT was empty a moment ago, while
# every file below is written with `>`, which truncates whatever is at the path when the
# loop reaches it. Under noclobber such a redirect is refused instead, the command does
# not run, and its nonzero status lands in FAIL like any other failure.
set -uC
cd "$(dirname "$0")/../.."

# shellcheck source=tools/perf/out_guard.sh
. tools/perf/out_guard.sh

TAG="${2:-pre-review}"
PY=.venv/bin/python
# <log-dir> defaults to a fresh directory under artifacts/, printed to stderr. It used
# to be a required argument, and the argument everybody reused was the previous run's
# directory: this driver writes summary.txt and one numpy/compare log pair per model
# into it, so a second run quietly replaced the first run's evidence — including the
# summary that says which generation the IDENTICAL columns were measured against.
# artifacts/validation/<model>/ is a different matter and is deliberately not guarded
# here: rewriting slowq.npz and compare.log there *is* this driver's documented
# protocol, and it keeps a -<tag> backup of the previous generation first.
OUT="$(out_guard_root "$PY" "${1:-}")" || exit 1
# The one line in a compare log that describes the machine rather than the numerics.
# It is dropped from both sides of the log comparison, and the column that results is
# named log_no_timing so nobody reads a filtered comparison as a byte-for-byte one.
TIMING_RE='^\[compare\] took '
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

# The 9-frame sample expands to a 612-token prompt, so the 512 default overflows
# before it captures anything. See docs/validation.md.
MAX_CONTEXT_SmolVLM_256M_Instruct_slowq=1024

# Every file this driver writes into $OUT, checked as *files* before the first capture
# starts: source-identity.txt, summary.txt and one numpy/compare log pair per model.
# Eight models at roughly five minutes each is the same shape of run as the timing
# drivers, and "model 8 had nowhere to write" is not a recoverable place to find that out.
EVIDENCE_FILES=("$OUT/source-identity.txt" "$OUT/summary.txt")
for m in "${MODELS[@]}"; do
  EVIDENCE_FILES+=("$OUT/$m-numpy.log" "$OUT/$m-compare.log")
done
out_guard_check_file "$PY" "${EVIDENCE_FILES[@]}" || exit 1

protocol() {
  $PY - "$1" <<'PYEOF'
import json
import sys

capture = json.load(open(f"artifacts/validation/{sys.argv[1]}/reference.json"))["capture"]
print(capture["prefill_chunk_size"], capture["decode_steps"])
PYEOF
}

FAIL=0
# The record of *which tree* was validated. Its exit status used to be looked at by
# nobody: the loop below collected only the two validate_reference.py codes, so a
# source-identity run that failed — or a redirect that did — left a summary reporting
# eight models validated against a tree the run never named.
$PY tools/perf/source_identity.py --src . > "$OUT/source-identity.txt" 2>&1 || {
  FAIL=1
  echo "run_reference: 源码身份记录写入失败（$OUT/source-identity.txt），本次验证无法关联到某棵树" >&2
}
# The header is part of the evidence: without it a reader cannot tell which
# generation `IDENTICAL` was measured against, and cannot re-decide the column.
{
  echo "# run_reference.sh summary"
  echo "# backup tag: $TAG"
  echo "#   tensors=, worst_numerics_rel_l2=, worst_internal_rel_l2= are parsed from"
  echo "#   this run's own log, <log-dir>/<model>-compare.log."
  echo "#   npz=            raw bytes, artifacts/validation/<model>/slowq.npz vs slowq-$TAG.npz"
  echo "#   log_no_timing=  this run's compare log vs compare-$TAG.log, with the"
  echo "#                   '[compare] took <n> s' line removed from both sides; that"
  echo "#                   line measures the machine, not the numerics."
} > "$OUT/summary.txt"
if [ ! -s "$OUT/summary.txt" ]; then
  FAIL=1
  echo "run_reference: 摘要 ${OUT}/summary.txt 没有写成，本轮的 IDENTICAL 列没有归属" >&2
fi

for m in "${MODELS[@]}"; do
  dir="artifacts/validation/$m"
  echo "=== $m ==="
  numpy_log="$OUT/$m-numpy.log"
  compare_log="$OUT/$m-compare.log"
  # Explicit rather than left to the redirect. Every column below is parsed out of
  # $compare_log, so a refused redirect would not show up as a failure — it would show
  # up as the *previous* occupant's numbers printed next to this run's `npz=` column,
  # which is the exact defect the `cp` under this loop was added to remove.
  if [ -e "$numpy_log" ] || [ -e "$compare_log" ]; then
    echo "run_reference: ${m} 的日志已存在，不覆盖；本模型没有重新验证" >&2
    FAIL=1
    continue
  fi
  read -r CHUNK STEPS <<<"$(protocol "$m")"

  # bash 3.2 treats an empty "${arr[@]}" as unbound under `set -u`, hence the
  # ${arr+...} guard rather than a plain expansion.
  extra=()
  var="MAX_CONTEXT_${m//-/_}"
  if [ -n "${!var:-}" ]; then
    extra=(--max-context "${!var}")
  fi

  # Back up before overwriting, so "did it change" stays answerable from bytes
  # rather than from memory. The existence check keeps the first backup: re-running
  # this driver must not replace the pre-review baseline with a post-review one.
  # Both files are rewritten below, and the log backup is what the `log_no_timing`
  # column compares against, so it has to be taken here rather than after.
  for f in slowq.npz compare.log; do
    if [ -f "$dir/$f" ] && [ ! -f "$dir/${f%%.*}-$TAG.${f##*.}" ]; then
      cp -p "$dir/$f" "$dir/${f%%.*}-$TAG.${f##*.}"
    fi
  done

  $PY tools/validate_reference.py numpy --model "models/$m" \
    --prefill-chunk-size "$CHUNK" --decode-steps "$STEPS" ${extra[@]+"${extra[@]}"} \
    > "$numpy_log" 2>&1
  nrc=$?
  $PY tools/validate_reference.py compare --model "models/$m" \
    > "$compare_log" 2>&1
  crc=$?
  if [ ! -s "$numpy_log" ] || [ ! -s "$compare_log" ]; then
    echo "run_reference: ${m} 的日志没有写成，本模型的结论不可追溯" >&2
    FAIL=1
    continue
  fi

  # `compare` is read-only over the artifact directory, so $dir/compare.log exists
  # only because some command redirected stdout into it. This driver used to redirect
  # into $OUT and then read $dir/compare.log for every column below, which left it
  # holding the *previous* generation's output: the numbers described an older run
  # while `npz=` next to them described this one, and `log=IDENTICAL` compared that
  # stale file against a backup this same script had just copied from it — a column
  # that could not fail, and did not (SmolLM2-135M's own log differed by one timing
  # line while the summary still printed IDENTICAL). Writing it here keeps
  # docs/validation.md's "最近一次 compare 的完整输出" true, and makes the backup
  # taken above a real previous generation to compare against.
  cp "$compare_log" "$dir/compare.log"

  tensors=$(sed -n 's/^PASS: \([0-9][0-9]*\) tensors.*/\1/p' "$compare_log" | tail -1)
  # Two different worsts, from two different sections, in two different formats.
  # Conflating them under one `worst_rel_l2` printed the *internal* number (chunk
  # vs full prefill, often exactly 0) under a name a reader took for the
  # cross-implementation error, which is orders of magnitude larger — Qwen2.5-0.5B
  # read 0.000e+00 while its true cross-implementation worst was 2.719e-05.
  #   [numerics] : whitespace-aligned table, `rel L2` is the 4th field, no `=`.
  #   [internal] : `... rel_l2=<value> ...` per sample line.
  worst_numerics=$(awk '/^\[numerics\]/{f=1;next} /^\[/{f=0} f && $4 ~ /e[+-]/ {print $4}' \
    "$compare_log" | sort -g | tail -1)
  worst_internal=$(grep -o 'rel_l2=[0-9.e+-]*' "$compare_log" |
    sed 's/rel_l2=//' | sort -g | tail -1)
  same_npz=DIFFERS
  same_log=DIFFERS
  if cmp -s "$dir/slowq.npz" "$dir/slowq-$TAG.npz"; then same_npz=IDENTICAL; fi
  if [ -f "$dir/compare-$TAG.log" ] && cmp -s <(grep -v "$TIMING_RE" "$compare_log") <(grep -v "$TIMING_RE" "$dir/compare-$TAG.log"); then same_log=IDENTICAL; fi

  printf '%-34s numpy_rc=%d compare_rc=%d tensors=%s worst_numerics_rel_l2=%s worst_internal_rel_l2=%s npz=%s log_no_timing=%s\n' \
    "$m" "$nrc" "$crc" "${tensors:-NONE}" "${worst_numerics:-n/a}" "${worst_internal:-n/a}" "$same_npz" "$same_log" |
    tee -a "$OUT/summary.txt"
  if [ $nrc -ne 0 ] || [ $crc -ne 0 ]; then FAIL=1; fi
done

echo "OVERALL_RC=$FAIL" | tee -a "$OUT/summary.txt"
exit $FAIL
