#!/usr/bin/env python3
"""Verifiable identity for the source tree a measurement ran against.

A timing without the identity of the code that produced it is not reproducible.
"the working tree on 2026-10-23" names a moment rather than a thing, and this
round's optimised side is *uncommitted*, so there is no revision to name at all.
Every harness therefore records this alongside its numbers, and the report prints
it alongside the tables.

    .venv/bin/python tools/perf/source_identity.py [--src .]

Three parts:

``git``      HEAD, whether the tree is dirty, and which tracked files differ. For
             a committed baseline that is the whole identity; for the working tree
             it is the reason the hashes below exist.
``source``   sha256 of every ``src/slowq/*.py`` — the files the measurement
             actually imported. One hash per file rather than one over the tree,
             so a difference names the file instead of merely announcing itself.
``tooling``  sha256 of every ``tools/perf`` script, so the harness version
             travels with the numbers it produced.

Nothing here imports ``slowq``. The harnesses call it while a *different* tree's
``slowq`` is on ``sys.path``, and a tool that identified the wrong tree would be
worse than having no tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

#: Relative to the repository root, which is two levels up from this file.
ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path("src") / "slowq"
TOOL_DIR = Path("tools") / "perf"

#: The label :mod:`tools.perf.baseline_tree` writes into a materialised snapshot.
REVISION_FILE = "BASELINE-REVISION"

#: One ``git status --porcelain`` line: two status columns, one space, the path.
#: Matched rather than sliced at a fixed offset, so a malformed line is skipped
#: instead of contributing a truncated path to the identity record.
STATUS_LINE = re.compile(r"^(..) (.+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path, subdir: Path, suffixes: tuple[str, ...]) -> dict[str, str]:
    """``{relative posix path: sha256}`` for the matching files under ``subdir``.

    Sorted and keyed by the path *inside* the tree, never by an absolute one:
    two checkouts of the same revision at different locations must hash
    identically, or the comparison a reviewer wants to make is impossible.
    ``__pycache__`` is skipped because a stale ``.pyc`` is not source.
    """
    directory = root / subdir
    if not directory.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if "__pycache__" in path.parts:
            continue
        out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def git_state(root: Path) -> dict[str, Any]:
    """HEAD plus the working tree's distance from it.

    ``git`` is shelled out to rather than parsed out of ``.git``, and a failure is
    recorded instead of raised: someone handed a tarball of this repository can
    still measure it, and their JSON should say "no git" rather than not exist.

    Only a checkout *root* is described. A tree materialised by
    :mod:`tools.perf.baseline_tree` lives under ``artifacts/`` inside this very
    repository, so asking git about it would answer with the surrounding
    checkout's HEAD and dirty list — the identity of the wrong tree, which is
    worse than none. Such a tree is labelled by its ``BASELINE-REVISION`` file
    instead, and its real identity is the per-file hashes in :func:`describe`.
    """

    def run(*args: str) -> str | None:
        """Raw stdout, or ``None`` if git could not answer.

        Not stripped: ``git status --porcelain`` puts the index/worktree columns
        in the first two characters, so a leading space *is* data. Stripping the
        block would silently turn ``' M README.md'`` into ``'M README.md'`` and
        every path parsed from it would lose a character.
        """
        try:
            done = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=False
            )
        except OSError as exc:  # git not installed at all
            return f"<git unavailable: {exc}>"
        return done.stdout if done.returncode == 0 else None

    toplevel = run("rev-parse", "--show-toplevel")
    if toplevel is None:
        return {"repository": False, "status_error": "git could not describe this tree"}
    if Path(toplevel.strip()).resolve() != root:
        return {
            "repository": False,
            "enclosing_repository": toplevel.strip(),
            "status_error": None,
        }

    # ``core.quotepath=false`` so a path like ``docs/推理加速-第1轮.md`` is
    # printed verbatim instead of octal-escaped inside quotes: this block exists
    # to name files, and an escaped name does not name one.
    status = run("-c", "core.quotepath=false", "status", "--porcelain")
    modified: list[str] = []
    untracked: list[str] = []
    for entry in (status or "").splitlines():
        match = STATUS_LINE.match(entry)
        if match is None:
            continue
        columns, path = match.groups()
        (untracked if columns == "??" else modified).append(path)
    head = run("rev-parse", "HEAD")
    subject = run("log", "-1", "--pretty=%s")
    return {
        "repository": True,
        "head": head.strip() if head is not None else None,
        "head_subject": subject.strip() if subject is not None else None,
        "dirty": bool(modified or untracked) if status is not None else None,
        "modified_tracked": sorted(modified),
        "untracked": sorted(untracked),
        "status_error": None if status is not None else "git status failed",
    }


def describe(root: Path = ROOT) -> dict[str, Any]:
    """The identity block every measurement JSON carries."""
    root = Path(root).resolve()
    label = root / REVISION_FILE
    return {
        "root": root.as_posix(),
        "git": git_state(root),
        # A snapshot materialised from a known commit carries no ``.git`` of its
        # own, so this label is the only place its revision is written down.
        "revision_label": label.read_text(encoding="utf-8").strip()
        if label.is_file()
        else None,
        "source": tree_hashes(root, SOURCE_DIR, (".py",)),
        "tooling": tree_hashes(root, TOOL_DIR, (".py", ".sh")),
    }


def record(root: Path, harness: Path) -> dict[str, Any]:
    """Identify both sides of a measurement: the tree timed and the tool timing it.

    ``root`` is the tree whose ``slowq`` was imported, which for the baseline side
    is *not* the checkout the harness was launched from. ``harness`` is the
    running script, hashed from the launching checkout, so a JSON says which
    ``core_ab.py`` produced it even when that file has since changed.
    """
    harness = Path(harness).resolve()
    identity = describe(root)
    identity["harness"] = {
        "path": harness.as_posix(),
        "sha256": sha256_file(harness),
    }
    identity["digest"] = digest(identity)
    return identity


def digest(identity: dict[str, Any]) -> str:
    """One hash over the source and tooling maps, for a short comparable label.

    The ``git`` block is excluded on purpose: it contains the checkout path and
    the untracked-file list, which differ between two machines holding identical
    source. This digest answers "did the measured code differ", nothing more.
    """
    payload = json.dumps(
        {"source": identity.get("source"), "tooling": identity.get("tooling")},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=".", help="tree to identify (default: cwd)")
    parser.add_argument(
        "--digest-only", action="store_true", help="print just the short digest"
    )
    args = parser.parse_args(argv)

    identity = describe(Path(args.src))
    if args.digest_only:
        print(digest(identity))
        return 0
    print(json.dumps(identity, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
