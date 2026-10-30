# Shell half of out_guard.py. Sourced by the drivers; never executed.
#
# The Python half owns the rule — a path that already exists is a conflict, and a
# conflict is a nonzero exit before anything expensive runs. This file owns the one
# piece of shell logic that every driver would otherwise copy: turning "no argument"
# into a fresh run directory instead of into a directory that already holds somebody's
# evidence. Seven drivers had that default wrong in the same way, and the way it stayed
# wrong for seven drivers is that it was written seven times.
#
# bash 3.2 is the target, because that is the /bin/bash macOS ships: no associative
# arrays, no mapfile, and every optional expansion spelled ${x:-} so `set -u` does not
# turn an absent argument into an unbound variable.

# out_guard_root <python> [requested-root]
#
# Echoes the run root to use, and exits nonzero if it cannot be had.
#
# With an argument, that argument verbatim. The driver must not reinterpret a path the
# operator typed — no stripping of the trailing slash, no joining onto $ROOT — because
# the guard is what normalises it, via realpath, and it normalises the same way whether
# it was typed relative, absolute, with a trailing slash or as a symlink. Normalising
# twice in two places is how the two answers come apart.
#
# Without one, a directory created atomically under artifacts/ and announced on stderr:
# a run directory nobody was told about is a run nobody can cite, and "where did this
# JSON come from" is the first question asked of it weeks later.
out_guard_root() {
  local py="$1" requested="${2:-}" fresh
  if [ -n "$requested" ]; then
    printf '%s\n' "$requested"
    return 0
  fi
  fresh="$("$py" tools/perf/out_guard.py fresh-root)" || return 1
  printf 'out_guard: 本次运行输出目录 %s（新建；已有结果、状态与日志一律不覆盖）\n' \
    "$fresh" >&2
  printf '%s\n' "$fresh"
}

# out_guard_reserve <python> <path>...
#
# Create this driver's own products, refusing the whole call if any of them is taken.
# A driver reserves only what it writes, never the run root it was handed: an
# orchestrator creates the root and its sub-drivers create their own subdirectories
# inside it, so "the root exists" is normal and only "<root>/ab exists" is a conflict.
out_guard_reserve() {
  local py="$1"
  shift
  "$py" tools/perf/out_guard.py reserve "$@"
}

# out_guard_check <python> <path>...
#
# Same refusal, creating nothing. For an orchestrator that has to know about *every*
# step's products before step 1 starts — including the ones it never creates itself —
# so that a conflict in the last step is not discovered four hours into the run.
#
# These are the *directory* products. The rule for them accepts an existing empty
# directory, because that is what an orchestrator handing its own fresh root to a
# sub-driver looks like.
out_guard_check() {
  local py="$1"
  shift
  "$py" tools/perf/out_guard.py check "$@"
}

# out_guard_check_file <python> <path>...
#
# Same refusal for the paths that have to *hold a file*: a status, an environment
# record, a step log. Here an empty directory is a conflict, not a free target — there
# is nothing to create inside it and the write can never succeed. Checking these with
# the directory rule is what let run_recollection.sh pass its pre-check with a directory
# sitting on recollection-status.txt, run all four measurement steps, fail every status
# write with "Is a directory", and still exit 0 printing "done".
out_guard_check_file() {
  local py="$1"
  shift
  "$py" tools/perf/out_guard.py check-file "$@"
}
