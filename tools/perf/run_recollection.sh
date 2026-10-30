#!/usr/bin/env bash
# Re-collect round 1 with the post-review source, in one command.
#
#   ./run_recollection.sh [out-root]
#
# The review's point was that the published numbers were produced by source that has
# since been fixed, so "新的结论必须关联新的执行记录". This chains the four
# measurement drivers over the *current* working tree and writes everything under one
# out-root — by default a fresh directory it creates and prints, so a re-collection can
# never land on the evidence behind the published report. See tools/perf/out_guard.py.
#
# Strictly serial, and not merely by convention: run_round1.sh, run_round1_longctx.sh
# and run_noise_floor.sh each contend for the same cores and the same memory
# bandwidth, and two of them at once would produce numbers that describe the
# contention rather than the code.
#
# Deliberately does NOT run make_report.py. Regenerating
# docs/推理加速-第1轮-2026-10-23.md from a half-finished out-root would replace a
# complete report with one full of MISSING rows, and the doc is the deliverable. Run
# it by hand once every step below reports rc=0:
#
#   .venv/bin/python tools/perf/make_report.py --data <out-root>
#
# `;` between the steps rather than `&&`: each driver already refuses to write into
# the published directory and each logs its own rc, so a failure in step 1 should not
# silently cancel steps 2 and 3 — partial new evidence with an honest status file is
# more useful than no evidence and a guess about where it stopped.
#
# Continuing is not the same as succeeding. run_step accumulates every nonzero rc
# into OVERALL, the status file ends with OVERALL_RC=, and the script exits with it.
# Without that, "keep going after a failure" becomes "always exit 0", which is what
# this driver did: four steps returning 7 and a caller seeing success.
#
# Two accumulators, not one, because a measurement failing and the run failing to
# record itself are different facts. EVIDENCE covers the status file, the environment
# record and the step logs: all four steps can return 0 and the run can still be
# incomplete, and "re-collection done" printed over an unwritable status file is how a
# run with no verdict gets filed as a finished one.
#
# `set -C` is the write-time half of the same rule. The pre-check below runs once,
# before step 1; a log path taken afterwards — a second invocation pointed at the same
# root, an operator's own note — is refused by the shell at the moment of the redirect
# instead of being truncated by it. The JSON products already had this inside
# out_guard.write_json, which opens "x"; the logs had nothing.
#
# Ownership, because noclobber alone still left a hole. `set -C` only governs `>`: the
# two record files are written again at the end with `>>`, which appends to whatever is
# there. So a run that lost the exclusive create — a second invocation created a
# non-empty recollection-environment.txt between the pre-check and the write — went on
# to append its own "## machine at the end" into *that* file, and because the check was
# `[ ! -s ]`, a foreign file that was already non-empty looked like a successful write.
# rc=0, "re-collection done", and somebody else's evidence carrying this run's tail.
# Each record file is therefore created once, by an explicit `: >` whose return code is
# read, and appended to afterwards only if that create succeeded.
set -uC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=tools/perf/out_guard.sh
. tools/perf/out_guard.sh

PY=".venv/bin/python"
OUT="$(out_guard_root "$PY" "${1:-}")" || exit 1
STATUS="${OUT}/recollection-status.txt"
ENVIRONMENT="${OUT}/recollection-environment.txt"
STEP_LOGS=(
  "${OUT}/run_round1.log"
  "${OUT}/run_round1_longctx.log"
  "${OUT}/run_noise_floor.log"
  "${OUT}/run_projection.log"
)

# Checked before step 1 and creating nothing: this driver's own status, environment and
# four step logs, plus every product directory of every driver it chains. The point of
# including the steps' products is the shape of this run — four hours of measurement
# ending in "step 4 had nowhere to write" is not a recoverable failure, and the operator
# is left with a directory that is neither the old evidence nor a new one.
#
# Two calls because these are two kinds of path. The six *files* go through check-file,
# which refuses any existing object at all, an empty directory included: a directory
# sitting on recollection-status.txt is not a free target, it is a write that can never
# succeed, and accepting it let every step run and every status write fail while the
# driver still exited 0. The seven product *directories* go through check, which accepts
# an empty directory because that is what this driver handing $OUT to a sub-driver looks
# like.
#
# The run root itself is deliberately *not* checked. It exists by the time the steps run
# — this driver wrote the environment file into it — and so does every parent an operator
# might reuse; what must be free is the products inside.
out_guard_check_file "$PY" "$STATUS" "$ENVIRONMENT" "${STEP_LOGS[@]}" || exit 1
out_guard_check "$PY" \
  "${OUT}/ab" \
  "${OUT}/sweep-opt" \
  "${OUT}/vision" \
  "${OUT}/documents" \
  "${OUT}/ab-longctx" \
  "${OUT}/noise-floor" \
  "${OUT}/projection" || exit 1

mkdir -p "$OUT"

OVERALL=0
EVIDENCE=0
# 1 only when *this* run won the exclusive create below. Everything that appends to a
# record file reads its own flag first, because an append into a file this run never
# created cannot be detected afterwards: `>>` reports success.
OWN_ENVIRONMENT=0
OWN_STATUS=0

# `: >` under `set -C` is an exclusive create, and its return code is the whole test —
# 0 only when the path was free, and when it is not 0 nothing was written, so a foreign
# file keeps both its bytes and its mtime. `2>/dev/null` comes first so the shell's own
# "cannot overwrite existing file" is swallowed and the message below is the one an
# operator reads. Checked here rather than trusted from the pre-check: that one ran
# before `mkdir -p`, and this is the moment the path is actually taken.
if : 2>/dev/null >"$ENVIRONMENT"; then
  OWN_ENVIRONMENT=1
else
  EVIDENCE=1
  echo "run_recollection: 环境记录 ${ENVIRONMENT} 被占用或无法创建，本次运行不会写它" >&2
fi

# What machine, what source, and how busy the box was when this started. The three
# drivers each record their own source_identity, but none of them records the load —
# and a re-collection run on a machine that was doing something else is a different
# measurement from one that was not.
#
# `>>` rather than `>`: the create above already made this file, so `>` would now be
# refused by the same noclobber that just established ownership.
if [ "$OWN_ENVIRONMENT" -eq 1 ]; then
  {
    echo "# round-1 re-collection"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "out-root: $OUT"
    echo "source-identity digest: $("$PY" tools/perf/source_identity.py --src . --digest-only)"
    echo
    echo "## git"
    git rev-parse HEAD
    git log -1 --pretty='%h %s'
    git -c core.quotepath=false status --porcelain
    echo
    echo "## machine"
    uname -a
    sw_vers 2>/dev/null
    uptime
    sysctl -n vm.swapusage 2>/dev/null
    echo
    echo "## interpreter"
    "$PY" -VV
    "$PY" -c 'import numpy; print("numpy", numpy.__version__); print(numpy.show_config("dicts")["Build Dependencies"])'
    echo
    echo "## busiest processes at launch (a non-idle machine is a caveat, not a secret)"
    ps -Ao pid,pcpu,pmem,comm -r | head -12
  } >>"$ENVIRONMENT" 2>&1

  # `-s`, not `-f`: the block above redirects its own stderr into the file, so a failure
  # inside it still produces a file, and an empty one is the case that means the record
  # could not be written at all.
  if [ ! -s "$ENVIRONMENT" ]; then
    EVIDENCE=1
    echo "run_recollection: 环境记录 ${ENVIRONMENT} 没有写成，本次运行没有机器与源码身份记录" >&2
  fi
fi

if : 2>/dev/null >"$STATUS"; then
  OWN_STATUS=1
else
  EVIDENCE=1
  echo "run_recollection: 状态文件 ${STATUS} 被占用或无法创建，本次运行不会写它" >&2
fi

note() {
  # The one place that writes the status file, so the one place that can notice it
  # failing. Appends rather than truncates: each step adds its own line, and `>>` is
  # deliberately outside `set -C`, which only governs `>`. That is also why the
  # ownership flag is checked instead of the append's own return code — into a file
  # this run did not create, `>>` succeeds, and succeeding is the damage.
  if [ "$OWN_STATUS" -ne 1 ]; then
    # One message for the disowning above is enough; a driver that could not create its
    # status file still has to run its steps, and four repeats of the same fact would
    # bury the step output this is interleaved with.
    return 0
  fi
  printf '%s\n' "$1" >>"$STATUS" 2>/dev/null || {
    EVIDENCE=1
    echo "run_recollection: 状态记录写入失败，本次运行的结论不可追溯：$1" >&2
  }
}

run_step() {
  name="$1"; shift
  echo "=== ${name}: $(date '+%H:%M:%S') ===" >&2
  "$@" > "${OUT}/${name}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ ! -s "${OUT}/${name}.log" ]; then
    # A step that reports success and leaves nothing to check that success against.
    EVIDENCE=1
    echo "run_recollection: ${name} 返回 0，但 ${OUT}/${name}.log 是空的" >&2
  fi
  note "${name} rc=${rc} finished=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "=== ${name} rc=${rc} ===" >&2
  # Accumulated rather than acted on: step 2 still has to run after step 1 fails,
  # because "which steps failed" is what the status file is for. But an
  # accumulation nobody exits with is a decoration, and that is what this used to
  # be — four steps returning 7 and a driver returning 0.
  [ $rc -ne 0 ] && OVERALL=1
  return 0
}

run_step run_round1         bash tools/perf/run_round1.sh "$OUT"
run_step run_round1_longctx bash tools/perf/run_round1_longctx.sh "$OUT"
run_step run_noise_floor    bash tools/perf/run_noise_floor.sh "$OUT" 4
# Optimisation 1's direct price. It was missing from the first three re-collection
# runs, which left make_report.py reading projection/ from a directory that has no
# such subdirectory — so §2.4's table printed MISSING and §9.2's share printed
# 数据缺失 while the published round's numbers sat one directory over. Runs last
# because it divides by <out-root>/ab, which step 1 produces.
run_step run_projection     bash tools/perf/run_projection.sh "$OUT"

# Guarded by the ownership flag rather than by `||`, because this append *succeeds* into
# a file this run did not create — that success is the defect, and it is undetectable
# from the return code. The `||` stays for the case where this run does own the file and
# the write fails anyway: a full disk, a directory that became read-only mid-run.
if [ "$OWN_ENVIRONMENT" -eq 1 ]; then
  {
    echo
    echo "## machine at the end"
    uptime
    sysctl -n vm.swapusage 2>/dev/null
    ps -Ao pid,pcpu,pmem,comm -r | head -12
    echo "finished: $(date '+%Y-%m-%d %H:%M:%S %z')"
  } >>"$ENVIRONMENT" 2>&1 || {
    EVIDENCE=1
    echo "run_recollection: 环境记录 ${ENVIRONMENT} 的收尾部分写入失败" >&2
  }
fi

# In the status file as well as in the exit code: the file is what gets read weeks
# later, and a run that failed must not be findable as a completed one. Written last,
# after every failure this driver knows about has been accumulated, so the line cannot
# claim a verdict the exit code then contradicts.
FINAL=$OVERALL
[ $EVIDENCE -ne 0 ] && FINAL=1
if [ "$OWN_STATUS" -eq 1 ]; then
  printf 'OVERALL_RC=%d\n' "$FINAL" >>"$STATUS" 2>/dev/null
  wrote=$?
  # Read back rather than trusted. An append that reports success and lands nothing — a
  # status file sealed by something else, a full disk — leaves the exit code and the file
  # disagreeing, and the file is the one that survives.
  if [ $wrote -ne 0 ] || ! grep -q '^OVERALL_RC=' "$STATUS" 2>/dev/null; then
    EVIDENCE=1
    FINAL=1
    echo "run_recollection: 无法把 OVERALL_RC 写进 ${STATUS}，本次运行的状态记录不完整" >&2
  fi
else
  # No file to seal and nothing to read back, so the verdict survives only in the exit
  # code — which is exactly why it has to be nonzero even when all four measurements
  # passed. A run with no status record is not a finished run.
  FINAL=1
fi

if [ $FINAL -ne 0 ]; then
  echo "re-collection FAILED -> ${OUT}; see ${STATUS}" >&2
  if [ $OVERALL -ne 0 ]; then
    echo "at least one step exited nonzero, so this out-root is incomplete evidence:" >&2
  else
    echo "every step succeeded but this run could not record itself, so it is incomplete evidence:" >&2
  fi
  echo "do not regenerate the report from it, and do not file it as a finished run." >&2
  exit $FINAL
fi

echo "re-collection done -> ${OUT}; see ${STATUS}" >&2
echo "next: .venv/bin/python tools/perf/make_report.py --data ${OUT}" >&2
