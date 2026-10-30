"""Refuse to overwrite evidence: one rule, applied at every write site.

The round-1 drivers each defaulted to ``artifacts/perf-round1-rerun`` — a directory
that already held the measurements the published documents cite — and each protected
only ``artifacts/perf-round1``, by comparing a string:

    if [ "$OUT_ROOT" = "$LEGACY" ]; then ... exit 1; fi

``artifacts/perf-round1/`` is not that string, so a trailing slash walked straight
past the check, and nothing else stopped the run: every measurement program ends in
``out.parent.mkdir(parents=True, exist_ok=True); out.write_text(...)``, which is a
silent overwrite by construction. Two reviewers reproduced both clobbers in scratch
copies with stand-in programs, which is the only safe way to demonstrate it.

Overwriting raw evidence is not recoverable and it is not audible. The JSON that
survives is the new one, so the old measurement is not "changed" — it is gone, and
the document derived from it now describes bytes that no longer exist. ``run_projection.sh``
had the same defect wearing a disguise: it ``mv``'d the previous attempt into an
archive directory, which preserves the bytes but still lets the command that was
supposed to conflict succeed.

So the rule here is one rule with no escape hatch: **a path that already exists is a
conflict, and a conflict is a nonzero exit before anything expensive runs.** There is
deliberately no ``--force``. Re-measuring is always possible — into a new directory,
which is what ``fresh_root`` is for — and the new directory is what makes the two
runs comparable instead of making one of them unverifiable.

Three layers, because one is not enough:

* the drivers call :func:`reserve` up front, so a conflict is found before an hour
  of measurement rather than at the end of it, and all of a driver's own products
  are checked together — a half-created tree is its own kind of mess;
* a measurement program run directly calls :func:`preflight` as the first thing
  ``main()`` does, because it has no driver to reserve for it and its own
  ``write_json`` is the last line it executes — the refusal was always correct and
  always came after the model was loaded and the suite had run;
* the measurement programs call :func:`write_json` at the moment of writing, opened
  ``"x"`` so the refusal holds even if something created the file in between.

The early layers never replace the last one. A path that is free when it is checked
can be taken by the time the measurement finishes, and a rule enforced only by a
check performed earlier is a hope with a timestamp.

An orchestrator creates the run root and its sub-drivers create their own products
inside it, so "the root exists" is normal and must not be refused; "``<root>/ab``
exists" is the conflict. Every driver therefore reserves *its own* subdirectories
and never the root it was handed. An existing **empty** directory is accepted for a
*directory* product, since it holds no result — which is what makes ``mkdir -p``-style
invocation keep working without weakening the rule.

That acceptance stops at directory products. A status file, an environment record, a
step log and a measurement JSON all have to *be* files, and a directory sitting on one
of those paths is not a free target but a write that can never succeed: the empty-directory
rule applied to them is what let ``run_recollection.sh`` pass its pre-check with a
directory on ``recollection-status.txt``, run all four measurement steps, fail every
status write with ``Is a directory``, and still exit 0 printing "done". So the guard
takes the kind of path it is being asked about — ``is_file`` — and answers differently.

Paths are compared after ``realpath``, so trailing slashes, ``..``, relative paths,
absolute paths and symlinks all land on the same answer. A symlink whose target sits
inside the published evidence is caught by the target; a dangling symlink is caught
by ``lexists`` on the name that was typed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]

#: Directories holding measurements that published documents cite. Writing into one
#: is refused outright rather than left to the exists-check, so the message names the
#: reason instead of making the operator work out which old run they were about to
#: replace. ``artifacts/validation`` is *not* here: ``run_reference.sh`` rewrites
#: ``slowq.npz`` and ``compare.log`` there by design and keeps a ``-<tag>`` backup of
#: the previous generation first, which is that driver's documented protocol.
PUBLISHED_ROOTS = (
    "artifacts/perf-round1",
    "artifacts/perf-round1-rerun",
)

#: Where ``fresh_root`` puts a new run. Timestamped rather than sequenced: two
#: people, or one person in two terminals, must not be able to collide on a name.
DEFAULT_PREFIX = "artifacts/perf-run"


def canonical(raw: Any) -> Path:
    """The one path a name refers to, with symlinks and ``..`` resolved."""
    return Path(os.path.realpath(str(raw)))


def typed_name(raw: Any) -> Path:
    """The same name made absolute but *not* resolved through symlinks.

    ``canonical`` cannot see a dangling symlink — it reports the target, which does
    not exist — so "is there already something at the name I was given" needs the
    unresolved form.
    """
    return Path(os.path.abspath(str(raw)))


def display(path: Path) -> str:
    """Repo-relative when the path is inside the repo, absolute otherwise."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def published_conflict(path: Any, root: Any = None) -> str | None:
    """Which published evidence root ``path`` lands in, or None.

    ``root`` exists so a caller that has its own idea of the checkout — ``make_report``,
    which resolves every one of its paths against its own ``ROOT`` — asks this question
    in the same terms it asks every other path question. Leaving it to this module's
    ``ROOT`` would give one caller two different notions of "inside the repository",
    and the point of the rule is that there is only one.
    """
    resolved = canonical(path)
    base = ROOT if root is None else root
    for name in PUBLISHED_ROOTS:
        published = canonical(Path(base) / name)
        if resolved == published or published in resolved.parents:
            return name
    return None


def _conflict(raw: Any, *, is_file: bool = False) -> str | None:
    """Why this path may not be written, or None if it may.

    ``is_file`` says the path has to *hold a file* — a status, an environment record, a
    step log, a measurement JSON. An empty directory is then a conflict rather than a
    free target, because there is nothing to create in it and the write can never
    succeed. Without the flag the answer is the directory rule, which accepts an empty
    directory so an orchestrator can hand its own fresh root to a sub-driver.
    """
    published = published_conflict(raw)
    if published is not None:
        return f"落在已发布证据目录 `{published}/` 内"
    name = typed_name(raw)
    target = canonical(raw)
    for candidate in (name, target):
        # ``os.path.lexists``, not ``Path.exists``: a dangling symlink must count as
        # occupied, and ``exists`` follows the link and reports the missing target.
        if not os.path.lexists(candidate):
            continue
        if candidate.is_symlink():
            return f"已存在（符号链接 {display(candidate)}）"
        if candidate.is_dir():
            if is_file:
                return f"已存在（目录 {display(candidate)}），而这个路径要写的是文件"
            # An empty directory holds no result, so accepting it is not a hole in the
            # rule; it is what lets an orchestrator hand its own fresh root down.
            if not any(candidate.iterdir()):
                continue
            return f"已存在（目录 {display(candidate)}）"
        return f"已存在（文件 {display(candidate)}）"
    return None


def _blocked_parent(raw: Any) -> str | None:
    """The nearest existing ancestor of ``raw`` that is not a directory, or None.

    ``--out result.json/model.json`` is not createable, and nothing else catches it: the
    name itself does not exist, so :func:`_conflict` has nothing to say about it, and
    ``realpath`` reports a path straight through the file without complaining. Left to
    the write, it surfaces as ``NotADirectoryError`` after the measurement rather than
    before it.
    """
    for start in (typed_name(raw), canonical(raw)):
        for parent in start.parents:
            if not os.path.lexists(parent):
                continue
            if not parent.is_dir():
                return f"父路径已存在且不是目录（{display(parent)}）"
            break
    return None


def file_conflict(raw: Any) -> str | None:
    """One answer for a path that must hold a file: itself, then its parents."""
    return _conflict(raw, is_file=True) or _blocked_parent(raw)


def preflight(out: Any) -> Path | None:
    """Check one output file path before any expensive work, creating nothing.

    Every measurement program ends in :func:`write_text`, so an existing ``--out`` was
    never overwritten — but the refusal was the *last* line of ``main()``, which means
    the operator learned about it after the model had loaded and the whole suite had
    run. This is the same rule moved to the front. It does not replace the write-time
    one, and creating the file it checks would quietly turn that one into a formality:
    ``open(path, "x")`` can no longer fail on a path the preflight already made.

    ``None`` in, ``None`` out: ``--out`` is optional in all six programs and running
    one without it stays legal.
    """
    if out is None:
        return None
    reason = file_conflict(out)
    if reason is not None:
        _die([f"{out}：{reason}"], files=True)
    return canonical(out)


def reserve(paths: Sequence[Any], *, files: bool = False) -> list[Path]:
    """Check every path, then create them; a conflict creates nothing.

    Checking all of them before creating any is the point: a driver that reserves
    four directories and fails on the third would otherwise leave two behind, and
    the next attempt would then conflict with the evidence of the attempt that never
    ran. It is also why the check has to be the *file* check when ``files`` is set —
    an ``IsADirectoryError`` from the creation loop below raises after the earlier
    paths are already on disk, which is the same half-created tree with a traceback
    instead of a message.
    """
    problems: list[str] = []
    seen: dict[str, Any] = {}
    for raw in paths:
        key = canonical(raw).as_posix()
        if key in seen:
            problems.append(f"{raw}：与 {seen[key]} 规范化后是同一个路径")
            continue
        seen[key] = raw
        reason = file_conflict(raw) if files else _conflict(raw)
        if reason is not None:
            problems.append(f"{raw}：{reason}")
    if problems:
        _die(problems, files)
    created: list[Path] = []
    for raw in paths:
        target = canonical(raw)
        if files:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "x", encoding="utf-8"):
                pass
        else:
            os.makedirs(target, exist_ok=True)
        created.append(target)
    return created


def fresh_root(prefix: str = DEFAULT_PREFIX) -> Path:
    """A run directory that did not exist a moment ago, created atomically.

    ``os.mkdir`` rather than ``makedirs`` for the last component: it fails if the
    name is taken, so two drivers starting in the same second get two directories
    instead of one shared between them.
    """
    parent = canonical(ROOT / Path(prefix).parent)
    os.makedirs(parent, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for suffix in ("", *(f"-{n}" for n in range(2, 1000))):
        candidate = parent / f"{Path(prefix).name}-{stamp}{suffix}"
        published = published_conflict(candidate)
        if published is not None:  # pragma: no cover - the prefix is not published
            raise SystemExit(f" refusing to create inside published evidence {published}/")
        try:
            os.mkdir(candidate)
        except FileExistsError:
            continue
        return candidate
    raise SystemExit(f"{prefix}: no free run directory name under {parent}")


def write_text(out: Any, text: str) -> Path:
    """Write one measurement artefact, refusing to replace anything.

    Opened ``"x"`` so the refusal is enforced by the filesystem at the moment of
    writing rather than by a check performed earlier, which is the difference
    between a rule and a hope.

    Takes the rendered text rather than a payload because ``sigmoid_check.py``
    already builds its own document — the guard's job is to refuse, not to decide
    how an artefact is formatted.
    """
    problems = []
    published = published_conflict(out)
    if published is not None:
        problems.append(f"{out}：落在已发布证据目录 `{published}/` 内")
    elif (reason := file_conflict(out)) is not None:
        problems.append(f"{out}：{reason}，不覆盖")
    if problems:
        _die(problems, files=True)
    target = canonical(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as handle:
        handle.write(text)
    return target


def write_json(out: Any, payload: dict[str, Any]) -> Path:
    """``write_text`` for the common case of one JSON measurement payload."""
    return write_text(out, json.dumps(payload, ensure_ascii=False, indent=2))


def _die(problems: Iterable[str], files: bool) -> None:
    what = "输出文件" if files else "输出目录"
    print(f"out_guard: 拒绝写入，{what}会与已有证据冲突：", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    print(
        "  已有实验结果、历史状态与日志一律不覆盖，也没有 --force："
        "重新测量请换一个新目录（不带参数调用驱动脚本会自动创建一个并打印位置）。",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fresh = sub.add_parser("fresh-root", help="create and print a unique new run root")
    fresh.add_argument("--prefix", default=DEFAULT_PREFIX)

    for name, help_text, is_file in (
        ("reserve", "check then create output directories", False),
        ("reserve-file", "check then create empty output files", True),
        ("check", "check only, create nothing", False),
        (
            "check-file",
            "check only, create nothing; refuse any existing object, an empty "
            "directory included, because this path has to hold a file",
            True,
        ),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("paths", nargs="+")
        command.set_defaults(files=is_file)

    args = parser.parse_args(argv)
    if args.command == "fresh-root":
        print(display(fresh_root(args.prefix)))
        return 0
    if args.command in ("check", "check-file"):
        judge = file_conflict if args.files else _conflict
        problems = [
            f"{raw}：{reason}" for raw in args.paths if (reason := judge(raw))
        ]
        if problems:
            _die(problems, files=args.files)
        return 0
    reserve(args.paths, files=args.files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
