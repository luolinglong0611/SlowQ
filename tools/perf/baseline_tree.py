#!/usr/bin/env python3
"""Materialise the "before" side of an A/B from a Git revision, and verify it.

    .venv/bin/python tools/perf/baseline_tree.py [--dest DIR] [--revision REV]

Round 1 measured its baseline against ``artifacts/perf-round1/baseline-tree/``, a
directory that existed only on this machine. That works until the machine does
not: a reviewer with the repository and no snapshot cannot rebuild the "before"
side, so every gain in the report is unverifiable in principle however carefully
it was measured. The revision is the durable name for the same tree, so this
script makes the revision the source of truth and treats the directory as a cache
of it.

Two modes, chosen by whether ``--dest`` already holds a tree:

*absent*    ``git archive REV | tar -x`` into ``DEST``, then write
            ``DEST/BASELINE-REVISION``. ``git archive`` rather than
            ``git worktree add``: the snapshot is a detached copy with nothing
            registered in ``.git/worktrees``, and it excludes untracked files — a
            stale ``__pycache__`` or ``*.egg-info`` left in a working-tree copy
            would be imported in preference to the revision's own source.
*present*   verify it and change no source file. Every ``src/slowq/*.py`` is
            hashed and compared against ``git show REV:src/slowq/<name>``, and
            the file list is compared in both directions, so a snapshot with one
            extra module or one edited line is reported rather than measured. A
            mismatch exits non-zero: the point of the check is that a wrong
            baseline is loud.

Existing snapshots are never overwritten or "refreshed" in place. The round-1
directory is the record of what round 1 measured, and silently replacing it would
leave the report citing data nobody can point at. Re-materialising means choosing
a different ``--dest``. The one thing this script will add to an existing
snapshot is the ``BASELINE-REVISION`` label it was missing, and only once the
source hashes have already proved the tree is that revision.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The commit round 1's "before" numbers were measured against. ``git log -1`` on
#: it is the authority; the snapshot directory is only a cache.
BASELINE_REVISION = "8968e39"

DEFAULT_DEST = Path("artifacts") / "perf-round1" / "baseline-tree"
SOURCE_SUBDIR = Path("src") / "slowq"

#: Must match ``source_identity.REVISION_FILE``, which reads this label back.
#: Duplicated rather than imported: both tools are loaded by path (that is how
#: ``tests/test_benchmark.py`` loads ``tools/benchmark.py``), and a sibling import
#: would make this one unloadable the same way.
REVISION_FILE = "BASELINE-REVISION"


def git(root: Path, *args: str, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        check=False,
        text=not binary,
    )


def resolve_revision(root: Path, revision: str) -> str:
    done = git(root, "rev-parse", f"{revision}^{{commit}}")
    if done.returncode != 0:
        raise SystemExit(
            f"baseline_tree: {revision!r} does not name a commit in {root}\n"
            f"  git said: {done.stderr.strip()}"
        )
    return done.stdout.strip()


def revision_source_hashes(root: Path, revision: str) -> dict[str, str]:
    """``{filename: sha256}`` for ``src/slowq/*.py`` *as of* ``revision``.

    Read through ``git ls-tree`` and ``git cat-file`` rather than by extracting to
    a scratch directory, so verifying a snapshot reads the filesystem instead of
    writing to it.
    """
    prefix = SOURCE_SUBDIR.as_posix()
    listing = git(root, "ls-tree", "--name-only", f"{revision}:{prefix}")
    if listing.returncode != 0:
        raise SystemExit(
            f"baseline_tree: {revision} has no {prefix}: {listing.stderr.strip()}"
        )
    out: dict[str, str] = {}
    for name in sorted(listing.stdout.split()):
        if not name.endswith(".py"):
            continue
        blob = git(root, "cat-file", "blob", f"{revision}:{prefix}/{name}", binary=True)
        if blob.returncode != 0:
            raise SystemExit(
                f"baseline_tree: cannot read {name} at {revision}: {blob.stderr.decode()!r}"
            )
        out[name] = hashlib.sha256(blob.stdout).hexdigest()
    return out


def snapshot_source_hashes(dest: Path) -> dict[str, str]:
    directory = dest / SOURCE_SUBDIR
    if not directory.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.py"))
    }


def source_differences(dest: Path, want: dict[str, str], have: dict[str, str]) -> list[str]:
    prefix = SOURCE_SUBDIR.as_posix()
    problems = [f"missing {prefix}/{name}" for name in sorted(set(want) - set(have))]
    problems += [f"unexpected {prefix}/{name}" for name in sorted(set(have) - set(want))]
    problems += [
        f"{prefix}/{name} differs (snapshot {have[name][:12]}, revision {want[name][:12]})"
        for name in sorted(set(want) & set(have))
        if want[name] != have[name]
    ]
    return problems


def materialise(dest: Path, root: Path, full_sha: str) -> None:
    """Extract ``full_sha`` into ``dest`` through a tar pipe, then label it."""
    archive = git(root, "archive", "--format=tar", full_sha, binary=True)
    if archive.returncode != 0:
        raise SystemExit(f"baseline_tree: git archive failed: {archive.stderr.decode()!r}")
    dest.mkdir(parents=True, exist_ok=True)
    unpack = subprocess.run(
        ["tar", "-x", "-C", str(dest)], input=archive.stdout, capture_output=True, check=False
    )
    if unpack.returncode != 0:
        raise SystemExit(f"baseline_tree: tar failed: {unpack.stderr.decode()!r}")
    (dest / REVISION_FILE).write_text(full_sha + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--revision", default=BASELINE_REVISION)
    parser.add_argument("--root", default=".", help="repository to read Git objects from")
    parser.add_argument(
        "--print-dest",
        action="store_true",
        help="verify (materialising if needed), then print the tree's path on "
        "stdout and nothing else; for the run scripts, so they ask for the "
        "baseline instead of hardcoding where a snapshot happens to live",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = root / dest
    full_sha = resolve_revision(root, args.revision)
    short = full_sha[:12]

    def report(message: str) -> None:
        # Every human-facing line goes to stderr, which is what keeps stdout a
        # usable channel for --print-dest.
        print(message, file=sys.stderr)

    have = snapshot_source_hashes(dest)
    if have:
        problems = source_differences(dest, revision_source_hashes(root, args.revision), have)
        if problems:
            report(
                f"baseline_tree: {dest} is NOT {args.revision} ({short}):\n  "
                + "\n  ".join(problems)
            )
            return 1
        label = dest / REVISION_FILE
        recorded = label.read_text(encoding="utf-8").strip() if label.is_file() else ""
        if recorded and recorded != full_sha:
            report(f"baseline_tree: {label} says {recorded!r} but the source is {short}")
            return 1
        if not recorded:
            label.write_text(full_sha + "\n", encoding="utf-8")
            report(f"baseline_tree: labelled {label} as {short}")
        report(
            f"baseline_tree: {dest} is {args.revision} ({short}), "
            f"{len(have)} source file(s) verified by hash"
        )
        if args.print_dest:
            print(dest)
        return 0

    if dest.exists() and any(dest.iterdir()):
        report(
            f"baseline_tree: {dest} exists and is not empty but holds no "
            f"{SOURCE_SUBDIR.as_posix()}/. Refusing to write into it; choose another --dest."
        )
        return 1

    materialise(dest, root, full_sha)
    have = snapshot_source_hashes(dest)
    problems = source_differences(dest, revision_source_hashes(root, args.revision), have)
    if problems:
        report(
            "baseline_tree: the materialised tree failed its own verification:\n  "
            + "\n  ".join(problems)
        )
        return 1
    report(
        f"baseline_tree: {args.revision} ({short}) -> {dest}, "
        f"{len(have)} source file(s)"
    )
    if args.print_dest:
        print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
