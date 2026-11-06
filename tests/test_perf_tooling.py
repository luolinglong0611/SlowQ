"""Tests for the reproducibility tooling in ``tools/perf/``.

Every number in ``docs/推理加速-第1轮-2026-10-23.md`` rests on the claim "this is
the code that produced it". Two tools in ``tools/perf/`` make that claim
checkable — ``source_identity.py`` hashes the measured tree and embeds it in every
measurement JSON, ``baseline_tree.py`` rebuilds the "before" tree from a git
revision — and each has a failure mode that looks like success:

* ``source_identity.py`` records per-file hashes. Its git block once reported the
  first modified path as ``EADME.md``: the whole ``git status`` blob was
  ``.strip()``ed, which ate the leading space that is *part of* the porcelain
  format (``' M README.md'`` is a worktree-only modification). A wrong filename in
  an identity record is worse than no record, because it is believed.
* The same block escaped non-ASCII paths (``docs/\\346\\216\\250...``), so the two
  Chinese-named round-1 reports were unnameable in the very record meant to name
  them.
* ``baseline_tree.py`` rebuilds the "before" tree from a revision. A snapshot that
  drifted from the revision it claims would turn every A/B row in the report into
  a comparison against unknown code — and it would still run, and still produce
  tables.

``make_report.py`` turns those JSONs into the report's verdicts, so its judgement
functions are tested here too. The defect they replaced was *directional*: a row
was called untrustworthy only when it also claimed a slowdown
(``gain <= 1.0 and opt_check > base_check * 1.15``), so a row claiming a 3×
speedup on self-checks that disagreed by 2× was printed as a clean result. A
one-directional filter does not remove noise from a table, it removes the noise
that was inconvenient — and the surviving rows then look more consistent than the
measurement ever was. The tests below pin the symmetry, because it is exactly the
property that a well-meaning later edit would silently break.

Nothing here loads a *real* model; the tools only read files and shell out to git.
The sigmoid-attribution section at the bottom is the exception, and it uses the tiny
synthetic checkpoint from ``conftest`` — a real forward through real code, but no
downloaded weights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = REPO_ROOT / "tools" / "perf"


def load_tool(name: str) -> Any:
    """Import ``tools/perf/<name>.py`` by path.

    The same mechanism ``tests/test_benchmark.py`` uses for ``tools/benchmark.py``,
    and the reason neither tool may import its sibling: a by-path load has no
    package context, so ``from source_identity import ...`` inside
    ``baseline_tree.py`` would make it unloadable here.
    """
    spec = importlib.util.spec_from_file_location(name, TOOL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def identity() -> Any:
    return load_tool("source_identity")


@pytest.fixture(scope="module")
def baseline() -> Any:
    return load_tool("baseline_tree")


@pytest.fixture(scope="module")
def report() -> Any:
    return load_tool("make_report")


# --------------------------------------------------------------------------- #
# the generator's two outputs, and what no test may touch
# --------------------------------------------------------------------------- #

#: Official and historical products. A plain ``pytest`` run must leave all four
#: byte- *and* mtime-identical, so the fingerprint carries both: a regeneration that
#: happens to produce the same bytes still moved the mtime, and "same content" would
#: hide that a test wrote there at all.
#:
#: Two of the four live under a gitignored ``artifacts/``, so a checkout that did not
#: also carry the historical evidence over has only the two ``docs/`` entries. That is
#: fine for the fixture below — a file that is not there cannot be rewritten — and it
#: is why nothing here asserts the four exist. See
#: ``test_a_product_that_never_existed_cannot_be_told_from_a_deleted_one``.
PROTECTED_PRODUCTS = (
    "artifacts/perf-round1/report-tables.md",
    "artifacts/perf-round1-rerun/report-tables.md",
    "docs/推理加速-第1轮-2026-10-23.md",
    "docs/推理加速-第1轮-测量表.md",
)


def product_fingerprint(path: Path) -> tuple[int, int, str] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint_all(names: Sequence[str], root: Path) -> dict[str, tuple[int, int, str] | None]:
    return {name: product_fingerprint(root / name) for name in names}


def changed_since(before: dict[str, tuple[int, int, str] | None], root: Path) -> list[str]:
    """Which of ``before``'s paths no longer match the fingerprint taken earlier.

    Split out of the fixture so the mechanism can be driven against a fabricated tree:
    asserting it only against ``REPO_ROOT`` makes the guard's own test depend on a
    gitignored ``artifacts/``, which is a fact about one machine rather than about the
    comparison.
    """
    return [name for name in before if product_fingerprint(root / name) != before[name]]


#: Two of the four protected products are derived tables under a gitignored
#: ``artifacts/``, so a checkout that did not also carry the historical evidence over
#: has nothing to check. Skipping rather than failing follows the convention the rest of
#: this file already uses for missing historical data — ``ab_artifacts_missing``,
#: ``paired_experiment_missing``, ``identity_artifacts_missing`` and the rest. An
#: unconditional ``is_file()`` made the quick suite red on any machine without
#: ``artifacts/``, which is a statement about that machine and not about the code under
#: test; the check is gated, not dropped, and it still runs in the full checkout the
#: release flow produces.
protected_products_missing = pytest.mark.skipif(
    not all((REPO_ROOT / name).is_file() for name in PROTECTED_PRODUCTS),
    reason="the protected products are gitignored artifacts, absent from this checkout",
)


@pytest.fixture(autouse=True)
def official_products_are_not_test_outputs() -> Iterator[None]:
    """Attribute any rewrite of an official product to the test that caused it.

    ``make_report.main()`` derived ``OUT`` from ``--data`` and wrote it
    unconditionally, so the eight tests parametrized over the two published roots each
    rewrote ``<root>/report-tables.md`` — and rewrote it *without* ``--paired-data``,
    which is how the historical table came to cite the old paired experiment (65 收益
    rows) while the document beside it cited the new one (64). The mtime a reviewer
    finds afterwards does not name the culprit; this does.
    """
    before = fingerprint_all(PROTECTED_PRODUCTS, REPO_ROOT)
    yield
    changed = changed_since(before, REPO_ROOT)
    assert not changed, (
        f"this test rewrote official or historical products: {changed}. Render into "
        "tmp_path via render_report(); nothing under artifacts/ or docs/ is a test "
        "output, and updating the delivered document is a release step, not a test."
    )


class Rendered(NamedTuple):
    """Where one ``main()`` invocation put its two products, plus anything a test had
    to read out of the module globals before they were restored."""

    tables: Path
    doc: Path
    collected: Any = None


#: The globals ``main()`` reassigns and never puts back. The ``report`` fixture is
#: module-scoped, so a test that leaks one of them makes every later test in the
#: module describe a directory that no longer exists.
_REPORT_GLOBALS = (
    "PERF",
    "OUT",
    "DOC",
    "PAIRED_ROOT",
    "SIGMOID_CORRECTION",
    "PAIRED_SUMMARY",
    "PAIRED_BETWEEN",
)


def render_report(
    report: Any,
    tmp_path: Path,
    data: Path,
    paired: Path | None = None,
    tag: str = "render",
    collect: Callable[[Any], Any] | None = None,
) -> Rendered:
    """Run the generator with *both* of its outputs inside ``tmp_path``.

    Input and output used to be the same directory: ``OUT`` was ``<data>/report-
    tables.md``, so pointing a test at a real data root meant writing into it. The two
    are separate arguments now, and this helper is the only place in the test suite
    that saves and restores the module globals — nine call sites each hand-rolled a
    try/finally, and the ones that forgot ``PERF`` or ``PAIRED_ROOT`` were silent
    about it until a later test read the wrong tree.

    ``collect`` runs *inside* the try, while ``PERF`` still points at ``data``: the
    two tests that need ``PAIRED_BETWEEN`` or ``headline_numbers()`` have to read them
    before the restore, and reading them after is reading a different data root.
    """
    tables = tmp_path / f"{tag}-tables.md"
    doc = tmp_path / f"{tag}-doc.md"
    saved = tuple(getattr(report, name) for name in _REPORT_GLOBALS)
    argv = ["--data", str(data), "--out", str(tables), "--doc", str(doc)]
    if paired is not None:
        argv += ["--paired-data", str(paired)]
    try:
        report.main(argv)
        collected = collect(report) if collect is not None else None
    finally:
        for name, value in zip(_REPORT_GLOBALS, saved):
            setattr(report, name, value)
    return Rendered(tables, doc, collected)


def git_available(root: Path) -> bool:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return done.returncode == 0


# --------------------------------------------------------------------------
# source_identity: what gets hashed
# --------------------------------------------------------------------------
def test_every_runtime_module_is_hashed_and_nothing_else(identity: Any) -> None:
    """The map must cover exactly the files a measurement imports.

    A gap here is silent: the harness would embed an identity block that simply
    does not mention the module that was edited, and the report would go on
    describing the run as verifiable.
    """
    described = identity.describe(REPO_ROOT)
    on_disk = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "src" / "slowq").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert set(described["source"]) == on_disk
    assert described["source"], "an empty hash map would verify anything"
    assert all(len(value) == 64 for value in described["source"].values())


def test_hash_keys_are_tree_relative_so_two_checkouts_agree(
    identity: Any, tmp_path: Path
) -> None:
    """The same source at a different path must produce the same map.

    Keying by absolute path would make the one comparison a reviewer actually
    wants — "is this the tree that was measured?" — impossible across machines,
    and impossible between the checkout and the materialised baseline snapshot.
    """
    copy = tmp_path / "elsewhere"
    shutil.copytree(REPO_ROOT / "src", copy / "src")
    here = identity.describe(REPO_ROOT)["source"]
    there = identity.describe(copy)["source"]
    assert not any(key.startswith("/") for key in there)
    assert there == here


def test_the_digest_follows_the_source_and_ignores_the_checkout(
    identity: Any,
) -> None:
    base: dict[str, Any] = {
        "git": {"head": "a" * 40, "dirty": False},
        "source": {"src/slowq/ops.py": "1" * 64},
        "tooling": {"tools/perf/core_ab.py": "2" * 64},
    }
    same_code_elsewhere = dict(base, git={"head": "b" * 40, "dirty": True}, root="/other")
    assert identity.digest(base) == identity.digest(same_code_elsewhere)

    one_edit = dict(base, source={"src/slowq/ops.py": "3" * 64})
    assert identity.digest(base) != identity.digest(one_edit)

    other_tool = dict(base, tooling={"tools/perf/core_ab.py": "4" * 64})
    assert identity.digest(base) != identity.digest(other_tool), (
        "the harness version is part of what produced the numbers"
    )


# --------------------------------------------------------------------------
# source_identity: the git block
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("entry", "columns", "path"),
    [
        (" M README.md", " M", "README.md"),  # worktree-only: the leading space is data
        ("M  README.md", "M ", "README.md"),  # index-only
        ("MM src/slowq/ops.py", "MM", "src/slowq/ops.py"),
        ("?? docs/推理加速-第1轮-2026-10-23.md", "??", "docs/推理加速-第1轮-2026-10-23.md"),
        ("R  old.py -> new.py", "R ", "old.py -> new.py"),
        ("A  docs/a b.md", "A ", "docs/a b.md"),  # a path containing a space
    ],
)
def test_porcelain_lines_are_parsed_by_column_not_by_offset(
    identity: Any, entry: str, columns: str, path: str
) -> None:
    match = identity.STATUS_LINE.match(entry)
    assert match is not None, entry
    assert match.groups() == (columns, path)


@pytest.mark.parametrize("entry", ["", "garbage", "M", " M"])
def test_a_malformed_status_line_is_skipped_not_truncated(
    identity: Any, entry: str
) -> None:
    assert identity.STATUS_LINE.match(entry) is None


def test_every_path_the_git_block_names_exists(identity: Any) -> None:
    """The ``EADME.md`` regression, caught by an invariant rather than a fixture.

    A path git reports is a path on disk, so anything the parser mangles fails
    here — including the truncation, which produced a name that exists nowhere.

    The count check catches the *other* half of the same bug. Once the parse
    became a regex rather than a fixed offset, a stripped status blob no longer
    yielded ``EADME.md``; it yielded no match at all, so the first modified file
    silently vanished from the record instead of being wrong. Both are failures of
    "name the files", and only the line count sees the second one.
    """
    if not git_available(REPO_ROOT):
        pytest.skip("no git in this environment")
    block = identity.git_state(REPO_ROOT)
    assert block["repository"] is True
    assert block["status_error"] is None
    named = block["modified_tracked"] + block["untracked"]
    for path in named:
        assert "\\" not in path, f"{path!r} looks octal-escaped"
        # git collapses an untracked directory to its name, trailing slash or not.
        assert (REPO_ROOT / path.rstrip("/")).exists(), path

    raw = subprocess.run(
        ["git", "-c", "core.quotepath=false", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [line for line in raw.splitlines() if line.strip()]
    assert len(named) == len(lines), (
        f"git reported {len(lines)} entr(ies) but the identity block names "
        f"{len(named)}: {sorted(set(l[3:] for l in lines) - set(named))}"
    )


def test_a_tree_inside_the_checkout_is_not_described_as_a_checkout(identity: Any) -> None:
    """The baseline snapshot lives *inside* this repository, so git will answer for it.

    And it will answer with the surrounding checkout's HEAD and dirty list — the
    identity of the wrong tree, which is worse than no identity because it looks
    authoritative. ``baseline_tree.py`` writes under ``artifacts/`` of the measured
    checkout, so this is the normal case rather than an edge case.

    The scratch directory goes under the gitignored ``artifacts/`` and is removed
    again: a test must not leave the working tree it is identifying dirty.
    """
    if not git_available(REPO_ROOT):
        pytest.skip("no git in this environment")
    inner = REPO_ROOT / "artifacts" / ".test-nested-tree"
    inner.mkdir(parents=True, exist_ok=True)
    try:
        block = identity.git_state(inner)
    finally:
        inner.rmdir()
    assert block["repository"] is False
    assert Path(block["enclosing_repository"]).resolve() == REPO_ROOT
    assert "head" not in block, "a non-root tree must not borrow the outer HEAD"
    assert "modified_tracked" not in block, "nor the outer checkout's dirty list"


def test_a_tree_outside_any_repository_reports_no_git(identity: Any, tmp_path: Path) -> None:
    """Someone handed a tarball of this source can still measure it.

    Their JSON should say "no git" rather than not exist, and rather than raising
    out of the harness that is trying to record a timing.
    """
    block = identity.git_state(tmp_path)
    assert block["repository"] is False
    assert block["status_error"]
    described = identity.describe(tmp_path)
    assert described["source"] == {}
    assert described["revision_label"] is None


def test_the_published_baseline_snapshot_is_not_mistaken_for_a_checkout(
    identity: Any,
) -> None:
    snapshot = REPO_ROOT / "artifacts" / "perf-round1" / "baseline-tree"
    if not snapshot.is_dir():
        pytest.skip("the round-1 baseline snapshot has not been materialised here")
    assert identity.describe(snapshot)["git"]["repository"] is False


def test_a_snapshot_carries_its_revision_as_a_label(
    identity: Any, baseline: Any, tmp_path: Path
) -> None:
    dest = tmp_path / "tree"
    (dest / "src" / "slowq").mkdir(parents=True)
    assert identity.describe(dest)["revision_label"] is None
    (dest / baseline.REVISION_FILE).write_text("f" * 40 + "\n", encoding="utf-8")
    assert identity.describe(dest)["revision_label"] == "f" * 40


# --------------------------------------------------------------------------
# baseline_tree: the "before" side is rebuildable, not a single local snapshot
# --------------------------------------------------------------------------
def test_the_baseline_revision_is_named_and_resolves(baseline: Any) -> None:
    if not git_available(REPO_ROOT):
        pytest.skip("no git in this environment")
    full = baseline.resolve_revision(REPO_ROOT, baseline.BASELINE_REVISION)
    assert len(full) == 40
    assert full.startswith(baseline.BASELINE_REVISION)


def test_a_fresh_snapshot_materialises_to_exactly_the_revision(
    baseline: Any, tmp_path: Path
) -> None:
    """Rebuild the baseline from scratch and hash-verify it against the commit.

    This is the test that makes "the baseline is reproducible" a checked property
    rather than a sentence in the report: it never looks at the existing snapshot
    under ``artifacts/``, it makes a new one from git objects alone.
    """
    if not git_available(REPO_ROOT):
        pytest.skip("no git in this environment")
    dest = tmp_path / "rebuilt"
    full = baseline.resolve_revision(REPO_ROOT, baseline.BASELINE_REVISION)
    baseline.materialise(dest, REPO_ROOT, full)

    have = baseline.snapshot_source_hashes(dest)
    want = baseline.revision_source_hashes(REPO_ROOT, baseline.BASELINE_REVISION)
    assert have and want
    assert baseline.source_differences(dest, want, have) == []
    assert (dest / baseline.REVISION_FILE).read_text(encoding="utf-8").strip() == full


def test_a_drifted_snapshot_is_named_file_by_file(
    baseline: Any, tmp_path: Path
) -> None:
    """A tampered snapshot must fail loudly and say which file, with both hashes.

    "verification failed" alone would send someone re-running the measurement
    against a tree that is not the baseline; naming the file and showing the two
    digests is what makes the difference actionable.
    """
    if not git_available(REPO_ROOT):
        pytest.skip("no git in this environment")
    dest = tmp_path / "drifted"
    full = baseline.resolve_revision(REPO_ROOT, baseline.BASELINE_REVISION)
    baseline.materialise(dest, REPO_ROOT, full)

    want = baseline.revision_source_hashes(REPO_ROOT, baseline.BASELINE_REVISION)
    assert baseline.source_differences(dest, want, baseline.snapshot_source_hashes(dest)) == []

    victim = dest / baseline.SOURCE_SUBDIR / "ops.py"
    victim.write_text(victim.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    problems = baseline.source_differences(dest, want, baseline.snapshot_source_hashes(dest))
    assert len(problems) == 1
    assert "src/slowq/ops.py differs" in problems[0]
    assert want["ops.py"][:12] in problems[0]


def test_a_missing_and_an_extra_file_are_both_reported(baseline: Any, tmp_path: Path) -> None:
    dest = tmp_path / "tree" / "src" / "slowq"
    dest.mkdir(parents=True)
    (dest / "kept.py").write_text("x = 1\n", encoding="utf-8")
    want = {"kept.py": "0" * 64, "gone.py": "1" * 64}
    problems = baseline.source_differences(tmp_path / "tree", want, {"kept.py": "0" * 64, "new.py": "2" * 64})
    assert any(p.startswith("missing src/slowq/gone.py") for p in problems)
    assert any(p.startswith("unexpected src/slowq/new.py") for p in problems)
    assert not any("kept.py" in p for p in problems), "an agreeing file is not a problem"


def test_a_tree_with_no_source_hashes_as_empty(baseline: Any, tmp_path: Path) -> None:
    """An empty map means "there is nothing here to verify", not "everything matched"."""
    assert baseline.snapshot_source_hashes(tmp_path) == {}


# --------------------------------------------------------------------------
# make_report.py: the verdicts must not depend on which way the gain points
# --------------------------------------------------------------------------
def old_untrustworthy(gain: Any, b_check: Any, o_check: Any) -> bool:
    """The predicate ``make_report.py`` used before the symmetry fix.

    Kept here verbatim so a test can show what it did, rather than describing the
    defect in prose and asking the reader to take it on faith.
    """
    return (
        gain is not None
        and gain <= 1.0
        and b_check is not None
        and o_check is not None
        and o_check > b_check * 1.15
    )


@pytest.mark.parametrize(
    "b_check, o_check",
    [
        (1.00, 2.00),   # opt twice as slow per token as its own probe
        (1.00, 1.20),   # just over the 1.15 tolerance
        (2.00, 1.00),   # the mirror image: base is the outlier
        (1.20, 1.00),
    ],
)
def test_a_mismatched_self_check_is_flagged_whichever_side_is_the_outlier(
    report: Any, b_check: float, o_check: float
) -> None:
    """The rule reads a disagreement between the two sides, not a direction.

    Both orderings must flag: a predicate that only notices ``opt > base`` treats
    "the optimised tree's window was slow" as suspicious while treating "the
    baseline's window was slow" as a clean speedup.
    """
    assert report.suspect_self_check(b_check, o_check) is True


@pytest.mark.parametrize(
    "b_check, o_check",
    [(1.00, 1.00), (1.19, 1.21), (0.80, 0.85), (1.14, 1.00)],
)
def test_agreeing_self_checks_pass_whichever_way_the_gain_points(
    report: Any, b_check: float, o_check: float
) -> None:
    """A row that is merely slow, but internally consistent, stays a real result."""
    assert report.suspect_self_check(b_check, o_check) is False


@pytest.mark.parametrize("b_check, o_check", [(1.00, 2.00), (2.00, 1.00), (1.00, 1.30)])
def test_the_verdict_cannot_see_the_gain(b_check: float, o_check: float, report: Any) -> None:
    """The same pair of self-checks gives the same verdict at any gain.

    This is the property the old rule lacked, expressed as an executable claim:
    ``suspect_self_check`` takes no gain argument at all, so a 3× speedup and a
    0.33× slowdown resting on identical self-checks are judged identically.
    """
    verdicts = {report.suspect_self_check(b_check, o_check) for _ in range(3)}
    assert len(verdicts) == 1


def test_the_old_rule_missed_exactly_the_favourable_rows(report: Any) -> None:
    """Pin the asymmetry that was removed, so it cannot be reintroduced quietly.

    One pair of self-checks disagreeing by 2× — unambiguous contamination — is fed
    through both predicates at a speedup and at a slowdown. The old rule flags one
    and waves the other through; the new rule flags both.
    """
    b_check, o_check = 1.00, 2.00
    assert old_untrustworthy(0.33, b_check, o_check) is True, "slowdown: old rule fired"
    assert old_untrustworthy(3.00, b_check, o_check) is False, "speedup: old rule was blind"
    assert report.suspect_self_check(b_check, o_check) is True, "new rule fires on both"


def test_a_row_with_no_self_check_is_suspect_not_trustworthy(report: Any) -> None:
    """"Could not check" is not "passed the check".

    A run that executed only the workloads suite has no in-window decode probe, so
    every one of its rows lands here. Reporting those rows as trustworthy would
    make an under-instrumented run look better-covered than a complete one.
    """
    assert report.suspect_self_check(None, 1.00) is True
    assert report.suspect_self_check(1.00, None) is True
    assert report.suspect_self_check(None, None) is True


@pytest.mark.parametrize(
    "left, right", [(1.0, 2.0), (2.0, 1.0), (0.5, 1.5), (1.5, 0.5), (3.0, 3.0)]
)
def test_ratio_disagreement_is_symmetric_in_its_arguments(
    report: Any, left: float, right: float
) -> None:
    """``max/min`` — swapping the arguments cannot change the answer."""
    forward = report.ratio_disagreement(left, right)
    backward = report.ratio_disagreement(right, left)
    assert forward == pytest.approx(backward)


def test_ratio_disagreement_is_none_when_either_side_is_missing(report: Any) -> None:
    assert report.ratio_disagreement(None, 1.0) is None
    assert report.ratio_disagreement(1.0, None) is None
    assert report.ratio_disagreement(0.0, 1.0) is None


# --------------------------------------------------------------------------
# make_report.py: the raw_prefill_full self-check
# --------------------------------------------------------------------------
def prefill_micro(engine: float, raw: float) -> dict[str, Any]:
    """One side's ``micro`` block with the two 512-token prefill probes set apart."""
    return {
        "engine_prefill": {"512": {"median": engine, "samples": [engine] * 7}},
        "raw_prefill_full": {"512": {"median": raw, "samples": [raw] * 7}},
    }


#: Verbatim medians from the collected artifacts, not invented round numbers.
#: ``CLEAN_*`` is the published batch's Qwen3-1.7B pair, where both probes agree
#: within 4%; ``THRASHED_BASE`` is the re-collection's base side, whose
#: ``raw_prefill_full`` is 10.33× its own ``engine_prefill`` at the same length.
#: Both probes prefill the same 512 tokens and both project the vocabulary once, so
#: within one tree those medians have to be within a few percent — 10.33 is a weight
#: set being paged back in, not work.
CLEAN_BASE = prefill_micro(2.1465962919974118, 1.9770270830049412)
CLEAN_OPT = prefill_micro(1.561900249995233, 1.3785975840000901)
THRASHED_BASE = prefill_micro(2.225816958998621, 23.001566750001075)


def test_the_raw_prefill_self_check_flags_the_row_that_thrashed(report: Any) -> None:
    """"16.52× faster" is a machine-state artefact, and the check says so.

    Without it the re-collection's 整块 prefill table printed base 23001.6 ms
    against opt 1391.9 ms as a speedup, and ``headline_numbers`` turned that into
    the upper bound of a band quoted in the summary.
    """
    _, _, disagreement = report.raw_prefill_self_check(
        THRASHED_BASE, CLEAN_OPT, "512"
    )
    assert disagreement is not None
    assert disagreement > report.SELF_CHECK_TOLERANCE


@pytest.mark.parametrize(
    "left, right",
    [(THRASHED_BASE, CLEAN_OPT), (CLEAN_OPT, THRASHED_BASE)],
)
def test_the_raw_prefill_check_flags_whichever_side_is_the_outlier(
    report: Any, left: dict[str, Any], right: dict[str, Any]
) -> None:
    """Symmetric, because the contaminated side reports a *speedup*.

    A rule keyed on regressions — the defect ``suspect_self_check`` was written to
    replace — waves this row through, since base being 16× slower than opt looks
    like the optimisation working.
    """
    _, _, disagreement = report.raw_prefill_self_check(left, right, "512")
    assert disagreement is not None and disagreement > report.SELF_CHECK_TOLERANCE


def test_a_clean_row_passes_whatever_the_gain(report: Any) -> None:
    """A row that is merely fast, but internally consistent, stays a real result.

    1.43× here is the published batch's measured value; the check has to leave it
    alone or it is filtering the table rather than reading it.
    """
    base_ratio, opt_ratio, disagreement = report.raw_prefill_self_check(
        CLEAN_BASE, CLEAN_OPT, "512"
    )
    assert disagreement is not None
    assert disagreement <= report.SELF_CHECK_TOLERANCE
    assert base_ratio == pytest.approx(1.9770270830049412 / 2.1465962919974118)
    assert opt_ratio == pytest.approx(1.3785975840000901 / 1.561900249995233)


@pytest.mark.parametrize(
    "base, opt",
    [
        ({}, CLEAN_OPT),                                   # no probes at all
        (prefill_micro(1.0, 1.0), {"engine_prefill": {}}),  # opt has no raw probe
        ({"engine_prefill": {"512": {"median": 0.0}}}, CLEAN_OPT),
    ],
)
def test_the_raw_prefill_check_reports_nothing_when_a_probe_is_missing(
    report: Any, base: Any, opt: Any
) -> None:
    """"Could not check" is ``None``, never a passing ratio."""
    _, _, disagreement = report.raw_prefill_self_check(base, opt, "512")
    assert disagreement is None


def test_a_thrashed_raw_prefill_cannot_set_the_headline_band(
    report_at: Any, tmp_path: Path
) -> None:
    """The regression test: the band must exclude the row, and say it did.

    ``headline_numbers`` reduces five models to one ``(min, max)`` pair, so a
    single contaminated cell does not shift the band — it *becomes* its upper
    bound. Verified by removing the guard: this returned
    ``(16.68, 16.68)`` and the failure printed that number as the summary's
    achieved speedup. (The artifacts' own base/opt pair is 16.52×; this fixture
    crosses the re-collected base side with the published opt side, so it differs
    in the second decimal.)
    """
    model = "Qwen3-1.7B-slowq"
    ab = tmp_path / "ab"
    ab.mkdir(parents=True)
    for side, micro in (("base", THRASHED_BASE), ("opt", CLEAN_OPT)):
        (ab / f"{model}-{side}.json").write_text(
            json.dumps({"micro": micro, "environment": {}, "threads": {}}), encoding="utf-8"
        )
    numbers = report_at.headline_numbers()
    assert numbers["raw_prefill_512"] is None, "the only row was contaminated"
    assert numbers["raw_prefill_512_excluded"] == [report_at.SHORT[model]]
    assert numbers["raw_prefill_512_models"] == 0


def test_a_clean_row_still_reaches_the_headline_band(
    report_at: Any, tmp_path: Path
) -> None:
    """The guard must not empty the band: one clean row is one quoted ratio."""
    model = "Qwen3-1.7B-slowq"
    ab = tmp_path / "ab"
    ab.mkdir(parents=True)
    for side, micro in (("base", CLEAN_BASE), ("opt", CLEAN_OPT)):
        (ab / f"{model}-{side}.json").write_text(
            json.dumps({"micro": micro, "environment": {}, "threads": {}}), encoding="utf-8"
        )
    numbers = report_at.headline_numbers()
    assert numbers["raw_prefill_512_excluded"] == []
    expected = 1.9770270830049412 / 1.3785975840000901
    assert numbers["raw_prefill_512"] == pytest.approx((expected, expected))


AB_DIRS = [
    REPO_ROOT / "artifacts" / "perf-round1",
    REPO_ROOT / "artifacts" / "perf-round1-rerun",
]
ab_artifacts_missing = pytest.mark.skipif(
    not all((d / "ab").is_dir() for d in AB_DIRS),
    reason="the A/B artifacts are gitignored; re-run tools/perf/run_round1.sh",
)
#: Batches that actually contain the AB/BA paired experiment. The published round
#: never ran ``run_noise_floor.sh`` — that is a finding the report states, not a gap
#: to paper over — so tests that judge a claim *against* that experiment are
#: parametrized over this list rather than over ``AB_DIRS``. Filtering here instead of
#: skipping inside keeps the quick suite's headline a plain pass count, and a future
#: batch that runs the experiment is picked up without editing the test.
NOISE_FLOOR_DIRS = [d for d in AB_DIRS if (d / "noise-floor").is_dir()]
paired_experiment_missing = pytest.mark.skipif(
    not NOISE_FLOOR_DIRS,
    reason="no batch on disk ran tools/perf/run_noise_floor.sh",
)


@ab_artifacts_missing
def test_the_check_separates_the_collected_batches_with_room_to_spare(
    report: Any,
) -> None:
    """The 1.15 threshold is verified against the data, not chosen to suit it.

    Twenty rows per length across both collected batches: every clean one lands in
    a band barely wider than 1.0, and the contaminated ones are an order of
    magnitude outside it. A threshold that only just separated them would be a
    threshold fitted to the answer.
    """
    clean: list[float] = []
    flagged: list[tuple[str, str, str, float]] = []
    for root in AB_DIRS:
        for model in report.MODELS:
            base = report.load(root / "ab" / f"{model}-base.json")
            opt = report.load(root / "ab" / f"{model}-opt.json")
            if not base or not opt:
                continue
            for length in ("128", "512"):
                _, _, disagreement = report.raw_prefill_self_check(
                    base.get("micro"), opt.get("micro"), length
                )
                if disagreement is None:
                    continue
                if disagreement > report.SELF_CHECK_TOLERANCE:
                    flagged.append((root.name, report.SHORT[model], length, disagreement))
                else:
                    clean.append(disagreement)
    assert len(clean) >= 18, f"expected both batches present, got {len(clean)} clean rows"
    assert max(clean) < 1.12, "a clean row near the threshold means the threshold is arbitrary"
    assert flagged, "the re-collection's Qwen3-1.7B 512-token base side is known contaminated"
    assert min(item[3] for item in flagged) > 5.0, "flagged rows should be far outside, not near"


@ab_artifacts_missing
def test_the_flagged_row_is_the_one_the_engine_probe_contradicts(report: Any) -> None:
    """Name the row, so "the check fired" is checkable rather than asserted.

    The base side's ``raw_prefill_full`` is 23.0 s while its own ``engine_prefill``
    at the same length is 2.2 s. Three independent readings of the same quantity —
    the engine ratio in the same file, the two surviving noise-floor rounds, and the
    published batch — all put it near 1.42×.
    """
    root = REPO_ROOT / "artifacts" / "perf-round1-rerun"
    model = "Qwen3-1.7B-slowq"
    base = report.load(root / "ab" / f"{model}-base.json")
    opt = report.load(root / "ab" / f"{model}-opt.json")
    base_ratio, _, disagreement = report.raw_prefill_self_check(
        base.get("micro"), opt.get("micro"), "512"
    )
    assert disagreement > report.SELF_CHECK_TOLERANCE
    assert base_ratio > 5.0, "it is the base side that spent 23 s, not the opt side"
    engine = report.stat(
        (base["micro"]["engine_prefill"]["512"]), "median"
    ) / report.stat(opt["micro"]["engine_prefill"]["512"], "median")
    assert engine == pytest.approx(1.42, abs=0.05), "the clean probe in the same file says 1.42×"


@ab_artifacts_missing
@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_the_vision_prose_ratios_are_the_tables_ratios(report: Any, root: Path) -> None:
    """§5.4's bullets and §5.3's tables describe one measurement, so they must agree.

    Both used to be written by hand from the same JSON and both had drifted: the
    other-image row read 0.57× in the table while the bullet said 0.56×, and §5.3's
    sentence quoted "视觉字段的倍数（约 142×）" — the *different-question* headline's
    ratio — while the table it sits under is the same-question one at 140.9×. The
    check is on the rendered text rather than on the helper, because the defect was
    two copies of a number, and only comparing the copies catches that.
    """
    if not (root / "vision" / "opt.json").exists():
        pytest.skip(f"{root.name} has no vision/opt.json")
    report.PERF = root.resolve()
    numbers = report.headline_numbers()
    line: list[str] = []
    report.section_vision(line)
    text = "\n".join(line)

    other = numbers["vision_other_image_ratio"]
    assert other, "no other-image ratio was derived"
    row = next(r for r in text.splitlines() if "on-other-image #1" in r)
    assert f"{other:.2f}×" in row, f"§5.3's row says something other than {other:.2f}×"

    metrics = numbers["vision_metrics"]
    for key in ("wall_seconds", "first_token_seconds", "vision_encode_seconds"):
        assert metrics[key], f"{key} ratio missing"
    sentence = next(r for r in text.splitlines() if "远小于视觉字段的倍数" in r)
    assert f"约 {metrics['vision_encode_seconds']:.0f}×" in sentence.replace("\n", ""), (
        "§5.3's sentence must quote the same-question ratio it sits under"
    )


@ab_artifacts_missing
@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_the_summary_share_band_is_the_band_the_table_prints(report: Any, root: Path) -> None:
    """Two functions price optimisation 1's share; they must return the same band.

    ``section_attribution`` prints the band in the sentence under its own table and
    ``headline_numbers`` returns it for §2.5 and the summary. Same JSON, but they
    applied different voiding rules: the table drops a model's whole row when the
    512-token paired median is non-positive, while the summary kept that model's
    share computed from the two surviving lengths. On the published batch the
    summary therefore quoted a 0.51% lower bound and the table's own minimum was
    0.55% — a number a reader sent to verify could not find anywhere in the report.

    The assertion is deliberately about agreement rather than about either rule, so
    it stays true if the rule changes and both sides change with it.
    """
    if not (root / "projection").is_dir():
        pytest.skip(f"{root.name} has no projection/ directory")
    report.PERF = root.resolve()
    line: list[str] = []
    report.section_attribution(line)
    match = re.search(r"占比区间是 ([\d.]+)% ~ ([\d.]+)%", "\n".join(line))
    assert match, f"§2.4 printed no share band for {root.name}"
    band = report.headline_numbers()["opt1_share_percent"]
    assert band, f"the summary produced no share band for {root.name}"
    assert (round(band[0], 2), round(band[1], 2)) == (float(match[1]), float(match[2]))


@ab_artifacts_missing
@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_a_voided_projection_row_contributes_no_share(report: Any, root: Path) -> None:
    """Name the models the summary excludes, so the exclusion is checkable.

    ``projection_voided`` is what makes the summary bullet say which rows were
    thrown out rather than just reporting a narrower band. A band that silently
    lost a model reads as a band over all five.
    """
    if not (root / "projection").is_dir():
        pytest.skip(f"{root.name} has no projection/ directory")
    report.PERF = root.resolve()
    numbers = report.headline_numbers()
    voided = numbers["projection_voided"]
    for model in report.MODELS:
        probe = report.load(root / "projection" / f"{model}.json")
        if not probe:
            continue
        per_length = probe.get("per_length") or {}
        per = per_length.get("512") or {}
        direct = report.stat(per.get("projection_seconds"), "median")
        listed = [v for v in voided if v.startswith(report.SHORT[model])]
        if direct is not None and direct <= 0:
            # Route A subtracts the probe's price from the measured engine saving,
            # so a non-positive price would *inflate* optimisation 2. The row must
            # be both named and excluded.
            assert listed, f"{report.SHORT[model]} has a non-positive 512 tok median and is unlisted"
            base = report.stat(per, "engine_prefill_base")
            skipped = per.get("projections_skipped")
            if base and skipped:
                # Both cost variants, because the leak was the second one: the row's
                # own 512 median is negative and was already excluded, but the
                # median of its two *surviving* lengths is positive and produced a
                # share (0.51%) no table displayed.
                survivors = [
                    m
                    for m in (
                        report.stat(
                            (per_length.get(length) or {}).get("projection_seconds"), "median"
                        )
                        for length in ("128", "256", "512")
                    )
                    if m is not None and m > 0
                ]
                band = numbers["opt1_share_percent"]
                for cost in (direct, statistics.median(survivors) if survivors else None):
                    if not cost:
                        continue
                    share = cost * skipped / base * 100
                    assert not band[0] - 1e-9 <= share <= band[1] + 1e-9, (
                        f"{report.SHORT[model]} is voided but its {share:.3f}% share is inside the band"
                    )
        # A model can also be listed because a *shorter* length came out negative
        # while 512 stayed positive. That is still worth naming — the first table
        # prints 作废 in that cell — but it keeps contributing its 512 share, so
        # there is no exclusion to assert in that direction.


@ab_artifacts_missing
@paired_experiment_missing
@pytest.mark.parametrize("root", NOISE_FLOOR_DIRS or AB_DIRS, ids=lambda p: p.name)
def test_optimisation_1s_share_is_judged_per_model_not_against_a_pooled_median(
    report: Any, root: Path, tmp_path: Path
) -> None:
    """§9.2 must not compare one model's share against two other models' spread.

    The cell read "占 512-token 分块 prefill 的 0.22%–2.04%，不大于本轮观测到的
    约 1.5% 跨进程离散度". Both numbers came from real data and the sentence was
    still false, because they are keyed differently: the band's upper end is
    Qwen3-0.6B's share, while 1.5% is the *median of the two sweep models'* opt-side
    between-process dispersion for ``引擎分块 prefill 512 tok``. Qwen3-0.6B is one of
    those two models and its own value is 0.7% — **below** its 2.04% share by 2.8×.
    Pooling hid exactly the case that mattered, since it is the model that sets the
    maximum, and the hidden direction was the one that would have made a reader doubt
    the "分辨不出来" verdict.

    So the test recomputes both sides per model and requires the rendered cell to
    place every model on the side the arithmetic gives it. The old phrasing is
    asserted *absent* rather than merely outnumbered: a later edit that re-added
    "不大于" next to a correct table would still mislead whoever stops at the first
    clause.

    Renders the document because the defect lives in prose assembled from two
    independently-correct numbers — neither ``headline_numbers()`` nor the paired
    section contains it on its own.
    """
    if not (root / "projection").is_dir() or not (root / report.NOISE_DIR).is_dir():
        pytest.skip(f"{root.name} lacks projection/ or {report.NOISE_DIR}/")
    # main() renders every section before write_doc, which is what fills
    # PAIRED_BETWEEN, so the value has to be read out of the module during the render
    # rather than after it. Zeroed first: it is an output of the render, and the
    # module is shared with every other test in this file.
    report.PAIRED_BETWEEN = {}
    rendered = render_report(
        report,
        tmp_path,
        root,
        tag=root.name,
        collect=lambda module: (
            dict(module.PAIRED_BETWEEN),
            module.headline_numbers()["opt1_share_by_model"],
        ),
    )
    between, shares = rendered.collected
    doc = rendered.doc

    compared = {short: share for short, share in shares.items() if short in between}
    assert compared, f"{root.name}: no model has both a share and a paired-dispersion row"
    over = {s: v for s, v in compared.items() if v > between[s][1]}

    text = doc.read_text(encoding="utf-8")
    cell = text.split("### 9.2 已完成但收益测不出来", 1)[1].split("| 优化 5", 1)[0]
    # write_doc reflows prose to 88 columns, so a contiguous-string assertion has to
    # be made against whitespace-stripped text or it fails on line breaks alone.
    flat = re.sub(r"\s+", "", cell)
    assert not re.search(r"不大于本轮观测到的约[\d.]+%跨进程离散度", flat), (
        "the pooled-median comparison is back: it is false for whichever model sets "
        "the band's upper end"
    )

    def entry(short: str) -> str:
        base_pct, opt_pct = between[short]
        return (
            f"{short}占比{compared[short]:.2f}%vs自己那一行base/opt进程间"
            f"{base_pct:.1f}%/{opt_pct:.1f}%"
        )

    # Each model's two numbers must both be printed, so a reader can redo the
    # comparison from the cell alone instead of trusting the trailing label.
    for short in compared:
        assert entry(short) in flat, f"{short}'s share/dispersion pair is not in §9.2"

    if over:
        assert "高于自身opt侧离散度" in flat, (
            f"{', '.join(sorted(over))} exceed their own dispersion but §9.2 does not say so"
        )
        tail = flat.split("高于自身opt侧离散度", 1)[0]
        head = flat.split("低于自身离散度", 1)[0]
        for short in over:
            # The label trails its whole group, so membership is what pins the side:
            # an over-dispersion model must sit in the segment the "高于" label closes
            # and must not have been folded into the "低于" one.
            assert entry(short) in tail.rsplit("；**", 1)[-1], f"{short} is not in the 高于 group"
        for short in set(compared) - set(over):
            assert entry(short) in head, f"{short} is below its dispersion but not in the 低于 group"
    else:
        assert "高于自身opt侧离散度" not in flat, "no model exceeds its own dispersion"


def _decode_median_from_disk(payload: Any, key: str) -> float | None:
    """The ms/token a batch reports for ``key``, recomputed from the raw JSON.

    Deliberately does *not* call the generator's own ``decode_median``: a test that
    asks the function under test what it thinks and then checks the document against
    that answer passes no matter which way the function is wrong.
    """
    node = (payload.get("micro") or {}).get(key)
    if not isinstance(node, dict):
        return None
    windows = node.get("ms_per_token_windows")
    samples = windows.get("samples") if isinstance(windows, dict) else None
    if samples:
        return statistics.median(samples)
    value = node.get("ms_per_token")
    return value if isinstance(value, (int, float)) else None


@ab_artifacts_missing
@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_the_contradiction_prose_counts_the_workloads_the_data_actually_has(
    report: Any, root: Path, tmp_path: Path
) -> None:
    """§3's "同样四个工作负载" was typed in, and the re-collection has five.

    The published batch's contradiction table happened to span four distinct
    workload names, so the sentence read correctly there. Regenerated from the
    re-collection it spans five names over seven rows — and it was never one number
    anyway, because ``long-prompt-short-answer`` is skipped on the two SmolLM2
    models under ``ab/``, so those two only have four comparable loads while the
    other three have five. A count in prose that a count in the data can contradict
    is the defect; the fix is to print the data's.

    Renders the document rather than calling the helper, because the wrong number
    lives in a sentence and only the assembled text contains it.
    """
    doc = render_report(report, tmp_path, root, tag=root.name).doc
    text = doc.read_text(encoding="utf-8")
    assert "### 两次运行不一致的地方" in text
    section = re.sub(
        r"\s+", "", text.split("### 两次运行不一致的地方", 1)[1].split("\n## ", 1)[0]
    )

    assert "同样四个工作负载" not in section, (
        "the workload count is still typed in; it is four for two models and five "
        "for three, so no single number is right"
    )
    loads = report.compared_workloads(root)
    assert loads, f"{root.name} has no workload both batches measured"
    for short, names in loads.items():
        assert f"{short}{len(names)}个" in section, (
            f"§3 does not state that {short} has {len(names)} comparable workloads"
        )
    rows = report.contradiction_rows(root)
    assert f"本批{len(rows)}行" in section
    assert f"涉及{len({r[1] for r in rows})}个负载" in section
    assert f"{len({r[0] for r in rows})}个模型" in section
    # The closing sentence picks the batch that measured more, and says how many —
    # both counts out of the data, since "五个负载都测全了" was the same defect
    # wearing a different number.
    primary = report.batch_workloads(root, "ab-longctx")
    other = report.batch_workloads(root, "ab")
    assert (
        "逐模型" + "、".join(f"{s}{len(n)}个" for s, n in sorted(primary.items()))
    ) in section
    assert f"有模型只测到{min(len(n) for n in other.values())}个" in section


@ab_artifacts_missing
@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_the_bare_decode_probes_are_cross_checked_between_the_two_batches(
    report: Any, root: Path, tmp_path: Path
) -> None:
    """The two-run disagreement section only ever compared ``total_seconds``.

    End-to-end wall time mixes prefill, decode and detokenising, so a disagreement
    there can always be argued away as one of the other two. The same two batches
    also each ran the bare decode probes on the same two trees, and those disagree
    more sharply: on the re-collection Qwen3-1.7B's short-context gain is 1.064× in
    ``ab/`` and 0.928× in ``ab-longctx/`` — **opposite signs for the same code** —
    at a magnitude of 1.146×, i.e. *under* the 1.15 threshold the rest of the report
    uses. A magnitude-only rule therefore prints that row as 两批一致, burying the
    single strongest statement the batch makes, and burying it in the direction that
    flatters the change. So a sign flip is judged on its own.

    The per-side medians are recomputed from the raw JSON here rather than read back
    out of the generator, and the base-side drift the prose calls "最干净的证据" has
    to be the largest one actually in the table — the same tree cannot change between
    two batches, so that number is the one that cannot be blamed on the diff.
    """
    doc = render_report(report, tmp_path, root, tag=root.name).doc
    text = doc.read_text(encoding="utf-8")
    assert "#### 裸 decode 探针：两遍互相比一次" in text, (
        "the cross-batch decode check is missing; §3 only compared total_seconds"
    )
    flat = re.sub(
        r"\s+",
        "",
        text.split("#### 裸 decode 探针：两遍互相比一次", 1)[1].split("\n## ", 1)[0],
    )

    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for directory in ("ab", "ab-longctx"):
        for model in report.MODELS:
            for side in ("base", "opt"):
                path = root / directory / f"{model}-{side}.json"
                if path.is_file():
                    payloads.setdefault(model, {}).setdefault(directory, {})[side] = json.loads(
                        path.read_text(encoding="utf-8")
                    )

    expected: list[tuple[str, str, dict[str, dict[str, float]]]] = []
    for model, by_dir in payloads.items():
        if set(by_dir) != {"ab", "ab-longctx"}:
            continue
        for key, label in report.MICRO_DECODE_PROBES:
            medians = {}
            for directory, sides in by_dir.items():
                values = {
                    side: _decode_median_from_disk(payload, key)
                    for side, payload in sides.items()
                }
                if all(v is not None for v in values.values()):
                    medians[directory] = values
            if len(medians) == 2:
                expected.append((report.SHORT[model], label, medians))

    if not expected:
        # The published batch's ab-longctx/ ran --suite workloads only, so there is
        # genuinely nothing to compare. Saying so is the finding; staying silent or
        # printing an empty table would both read as "checked and fine".
        assert "这一节读不出来" in flat
        assert "--suiteworkloads" in flat, "the reason has to name what was not run"
        return

    flips = []
    flagged = []
    drifts = []
    for short, label, medians in expected:
        gains = {d: v["base"] / v["opt"] for d, v in medians.items()}
        lo, hi = sorted(gains.values())
        disagreement = hi / lo
        flip = (gains["ab"] - 1) * (gains["ab-longctx"] - 1) < 0
        base_drift = max(medians["ab"]["base"], medians["ab-longctx"]["base"]) / min(
            medians["ab"]["base"], medians["ab-longctx"]["base"]
        )
        compact_label = re.sub(r"\s+", "", label)
        row = flat.split(f"|{short}|{compact_label}|", 1)
        assert len(row) == 2, f"{short} {label} has no row in the decode cross-check table"
        # Whitespace is already gone, so the row is bounded by the next row's leading
        # pipe: ``…|7/7|两批一致||SmolLM2-135M|…``. maxsplit=7 puts the verdict in
        # cells[6] and everything after the empty cell into cells[7].
        cells = row[1].split("|", 7)
        assert cells[0] == (
            f"{medians['ab']['base']:.2f}→{medians['ab-longctx']['base']:.2f}"
        ), f"{short} {label} prints a base median the JSON does not contain"
        assert cells[1] == (
            f"{medians['ab']['opt']:.2f}→{medians['ab-longctx']['opt']:.2f}"
        )
        assert cells[2] == f"{gains['ab']:.3f}×"
        assert cells[3] == f"{gains['ab-longctx']:.3f}×"
        verdict = cells[6]
        if flip:
            flips.append((short, label))
            assert "两批方向相反" in verdict, (
                f"{short} {label} flips sign between the batches and must say so"
            )
        if flip or disagreement > report.SELF_CHECK_TOLERANCE:
            flagged.append((short, label))
            assert "尚不能确认收益或回退" in verdict
        else:
            assert "两批一致" in verdict
            assert "尚不能确认" not in verdict
        drifts.append((base_drift, short, label, medians))

    assert flagged, "the re-collection's Qwen3-1.7B rows must land here"
    assert f"本批{len(expected)}行里{len(flagged)}行" in flat
    if flips:
        assert f"其中{len(flips)}行" in flat

    worst_drift, worst_short, worst_label, worst_medians = max(drifts, key=lambda r: r[0])
    if worst_drift > report.SELF_CHECK_TOLERANCE:
        label_flat = re.sub(r"\s+", "", worst_label)
        assert "最干净的证据在base侧自己身上" in flat
        assert f"{worst_short}的「{label_flat}」两遍差{worst_drift:.3f}×" in flat, (
            "the prose must quote the widest base-side drift in the table, not a "
            "convenient one: that column is the only evidence the code cannot explain"
        )
        assert (
            f"{worst_medians['ab']['base']:.2f}→{worst_medians['ab-longctx']['base']:.2f}ms/tok"
            in flat
        )
    else:
        assert "最干净的证据在base侧自己身上" not in flat, (
            "no base side drifted past the threshold, so nothing is cleanest evidence"
        )


@ab_artifacts_missing
@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_the_pending_row_quotes_the_chat_table_it_points_at(
    report: Any, root: Path, tmp_path: Path
) -> None:
    """§9.4's 待重测 row and §4's note describe one table, so they must agree.

    The row said "四行读起来像回退（0.85× / 1.04× / 0.78× / 0.63×）", typed in from
    the published batch. Two things were wrong with it, and only comparing the two
    copies catches both: the re-collected batch has one such row, not four, and even
    on the published batch there are three — 1.04× is not a regression, it was
    counted as one. A reader sent to §4 to check the claim found a table that did not
    contain it.

    Rendered through ``main`` rather than through ``section_chat``, because the
    defect was in the row's own literal and a section-level check would not see it.
    """
    if not (root / "ab").is_dir():
        pytest.skip(f"{root.name} has no ab/ directory")
    doc = render_report(report, tmp_path, root, tag=root.name).doc
    text = doc.read_text(encoding="utf-8")

    report.PERF = root.resolve()
    try:
        slower = report.headline_numbers()["chat_cold_slower"]
    finally:
        report.PERF = report.DEFAULT_PERF

    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "行的耗时读起来像回退" in line)
    note = lines[start]
    assert note.startswith(f"**{len(slower)} 行"), (
        f"§4 counts {len(slower)} slow rows but its note says otherwise: {note}"
    )
    # The paragraph, not the line: the doc is reflowed to 88 columns, so the verdict
    # constant lands several lines below the count it belongs to.
    paragraph = "\n".join(lines[start:next((i for i in range(start, len(lines))
                                            if not lines[i].strip()), len(lines))])
    row = next(line for line in lines if "cold-vs-cold 计时" in line)
    assert f"**{len(slower)} 行读起来像回退**" in row, (
        f"§9.4's row must name the same {len(slower)} rows §4's table contains"
    )
    for name, turn, value in slower:
        assert f"{name} 第 {turn} 轮 {value:.2f}×" in row
    # The withdrawn inference may be *quoted* — §0 and §3 both quote it in order to
    # retract it — but it must never stand as the report's own claim. Pin the
    # retraction rather than the phrase: a bare "not in text" held for one edit only
    # because the 88-column reflow happened to split the quotation across two lines,
    # and broke as soon as §0 grew a clause. Whitespace-stripped, per paragraph, so
    # the check does not depend on where the line budget lands.
    blocks = [re.sub(r"\s+", "", block) for block in text.split("\n\n")]
    quoting = [block for block in blocks if "1.05×~1.29×" in block]
    assert quoting, "the retracted sentence should still be quoted where it is withdrawn"
    for block in quoting:
        assert "撤回" in block, (
            f"the withdrawn inference is stated without its retraction:\n{block[:300]}"
        )
    assert report.INDETERMINATE in paragraph


@ab_artifacts_missing
@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_the_pending_row_quotes_the_decode_probe_it_points_at(report: Any, root: Path) -> None:
    """§9.4's decode row said "报出 0.52×" — §3's own note is the source of that number.

    Same defect class as the chat row: a figure copied out of one batch's rendered
    text and left behind when the batch changed. Here the check is on the model
    names, because the row lists which rows were flagged and §3 prints the full
    sentence for each.
    """
    if not (root / "ab").is_dir():
        pytest.skip(f"{root.name} has no ab/ directory")
    report.PERF = root.resolve()
    try:
        line: list[str] = []
        report.section_micro(line)
        text = "\n".join(line)
        suspects = report.headline_numbers()["decode_long_suspects"]
    finally:
        report.PERF = report.DEFAULT_PERF

    flagged = re.search(r"\*\*(\d+) 行 decode 数据待重测", text)
    if not suspects:
        assert not flagged, "§3 flagged rows the self-check does not return"
        return
    assert flagged and int(flagged.group(1)) == len(suspects)
    for note in suspects:
        assert note in text, "§9.4's row and §3's note must be the same string"


@ab_artifacts_missing
@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_the_report_names_the_version_of_the_script_that_wrote_it(
    report: Any, root: Path, tmp_path: Path
) -> None:
    """§1.5 identified the measured trees but not the thing writing the conclusions.

    Every number and verdict in the document comes out of ``write_doc``, which was
    edited repeatedly during the review — the same JSON rendered by two versions of
    the script gives different prose, and a reader checking a sentence against the
    artifacts had no way to tell which version produced it. The file sits in the
    working tree, so there is no revision to cite instead; the hash is the handle.

    The line count is checked against ``splitlines`` because the paragraph invites
    exactly that check, and ``count(b"\\n") + 1`` on a file ending in a newline
    reports one more line than ``wc -l`` does.
    """
    doc = render_report(report, tmp_path, root, tag=root.name).doc
    text = doc.read_text(encoding="utf-8")
    section = text.split("### 1.5 源码身份", 1)[1].split("\n## ", 1)[0]

    digest, script, lines = report.generator_identity()
    path = report.ROOT / script
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, "the recorded hash is not this file's"
    assert len(raw.decode().splitlines()) == lines
    assert f"`{digest}`" in section
    assert f"`{script}`" in section
    assert f"{lines} 行" in section
    # The reader has to be able to run the check the paragraph claims.
    assert f"shasum -a 256 {script}" in section
    assert "生成本文的脚本自身" in section


def _recorded_identity_blocks(report: Any, root: Path) -> list[tuple[str, dict]]:
    """The ``(name, identity)`` pairs ``section_identity`` would gather for ``root``.

    Walked from ``IDENTITY_SOURCES`` rather than ``rglob`` so the sample is exactly the
    set the report reasons about, and in the report's own order. Callers de-duplicate by
    digest when they need one row per tree, since every file in a tree shares a digest.
    """
    blocks: list[tuple[str, dict]] = []
    for name in report.IDENTITY_SOURCES:
        payload = report.load(root / name)
        identity = (payload or {}).get("source_identity")
        if isinstance(identity, dict) and identity.get("digest"):
            blocks.append((name, identity))
    return blocks


def test_recorded_trees_relocate_without_confusing_baseline_and_runtime(
    report: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A moved checkout must retain each measured tree's relative location."""
    old = tmp_path / "old checkout"
    current = tmp_path / "new checkout"
    monkeypatch.setattr(report, "ROOT", current)
    data = current / "measurements"
    data.mkdir(parents=True)
    monkeypatch.setattr(report, "PERF", data)
    monkeypatch.setattr(report, "IDENTITY_SOURCES", ["base.json", "opt.json"])
    monkeypatch.setattr(report, "paired_directory", lambda: None)
    relative_file = "src/slowq/model.py"
    originals: dict[Path, bytes] = {}
    for name, tree, content in (
        ("base", Path("artifacts/baseline-tree"), b"baseline code"),
        ("opt", Path("."), b"runtime code"),
    ):
        path = current / tree / relative_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        identity = {
            "root": str(old / tree),
            "harness": {"path": str(old / "tools/perf/core_ab.py")},
            "source": {relative_file: hashlib.sha256(content).hexdigest()},
            "tooling": {},
        }
        identity["digest"] = report.partial_digest(identity, "source", "tooling")
        payload = data / f"{name}.json"
        payload.write_text(json.dumps({"source_identity": identity}), encoding="utf-8")
        originals[payload] = payload.read_bytes()
        assert report.recorded_tree_root(identity) == current / tree
        assert report.recorded_map_still_matches(identity, "source") == (1, [])

    lines: list[str] = []
    report.section_identity(lines)
    rendered = "\n".join(lines)
    assert "源码路径重定位" in rendered
    assert f"| `{old}` | `{current}` |" in rendered
    assert f"| `{old / 'artifacts/baseline-tree'}` | `{current / 'artifacts/baseline-tree'}` |" in rendered
    assert "2/2 棵树的 digest 现在仍能原样复现" in rendered
    assert all(path.read_bytes() == before for path, before in originals.items())


def test_existing_recorded_tree_wins_over_a_matching_relocated_copy(
    report: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not hide an edited original by selecting another matching copy."""
    old, current = tmp_path / "old", tmp_path / "current"
    old.mkdir()
    current.mkdir()
    (old / "file.py").write_bytes(b"changed")
    (current / "file.py").write_bytes(b"recorded")
    monkeypatch.setattr(report, "ROOT", current)
    identity = {
        "root": str(old),
        "harness": {"path": str(old / "tools/perf/core_ab.py")},
        "source": {"file.py": hashlib.sha256(b"recorded").hexdigest()},
    }
    assert report.recorded_tree_root(identity) == old
    assert report.recorded_map_still_matches(identity, "source") == (0, ["file.py"])


@pytest.mark.parametrize("content", [None, b"changed"])
def test_relocated_tree_still_rejects_missing_or_changed_files(
    report: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes | None
) -> None:
    old, current = tmp_path / "old", tmp_path / "current"
    current.mkdir()
    if content is not None:
        (current / "file.py").write_bytes(content)
    monkeypatch.setattr(report, "ROOT", current)
    identity = {
        "root": str(old),
        "harness": {"path": str(old / "tools/perf/core_ab.py")},
        "source": {"file.py": hashlib.sha256(b"recorded").hexdigest()},
    }
    assert report.recorded_tree_root(identity) == current
    assert report.recorded_map_still_matches(identity, "source") == (0, ["file.py"])


@pytest.mark.parametrize("case", ["external", "wrong-harness", "parent-root", "parent-harness", "symlink"])
def test_recorded_tree_relocation_does_not_guess_or_escape_the_checkout(
    report: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    old, current = tmp_path / "old", tmp_path / "current"
    current.mkdir()
    root = old
    harness = old / "tools/perf/core_ab.py"
    if case == "external":
        root = tmp_path / "external-tree"
    elif case == "wrong-harness":
        harness = old / "unrelated/core_ab.py"
    elif case == "parent-root":
        root = old / ".." / "external-tree"
    elif case == "parent-harness":
        harness = old / ".." / "tools/perf/core_ab.py"
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        (current / "snapshot").symlink_to(outside, target_is_directory=True)
        root = old / "snapshot"
    monkeypatch.setattr(report, "ROOT", current)
    identity = {"root": str(root), "harness": {"path": str(harness)}}
    assert report.recorded_tree_root(identity) == root


@pytest.mark.parametrize("case", ["parent", "absolute", "symlink"])
def test_relocated_tree_hash_map_cannot_read_a_file_outside_its_root(
    report: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    old, current = tmp_path / "old", tmp_path / "current"
    current.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"recorded")
    relative = "../outside.py"
    if case == "absolute":
        relative = str(outside)
    elif case == "symlink":
        relative = "link.py"
        (current / relative).symlink_to(outside)
    monkeypatch.setattr(report, "ROOT", current)
    identity = {
        "root": str(old),
        "harness": {"path": str(old / "tools/perf/core_ab.py")},
        "source": {relative: hashlib.sha256(b"recorded").hexdigest()},
    }
    assert report.recorded_map_still_matches(identity, "source") == (0, [relative])


@ab_artifacts_missing
def test_the_partial_digest_mirrors_the_tool_it_copies(report: Any, identity: Any) -> None:
    """``make_report.partial_digest`` is a copy of ``source_identity.digest``; pin it.

    ``tools/perf`` scripts load by path in these tests and may not import one another, so
    ``make_report`` cannot call ``source_identity.digest`` and carries its own copy
    instead. A copy nothing ties to the original drifts silently the moment the payload
    changes: §1.5 would keep printing a "digest" that no longer equals what
    ``source_identity.py --digest-only`` prints, quietly defeating the whole point of the
    paragraph, which is to let a reader re-derive the hash. Every recorded artifact is the
    sample. For each, the restricted copy over ``("source", "tooling")`` must equal both
    the digest stored in the file and the real tool's answer on the same block, so the
    copy cannot diverge from either the recording or the tool it duplicates.
    """
    blocks = [block for root in AB_DIRS for _, block in _recorded_identity_blocks(report, root)]
    assert blocks, "no source_identity artifacts on disk to hold the copy against"
    for block in blocks:
        copied = report.partial_digest(block, "source", "tooling")
        assert copied == block["digest"], "the copy disagrees with the recorded digest"
        assert copied == identity.digest(block), (
            "the copy disagrees with source_identity.digest on the same block"
        )


@ab_artifacts_missing
def test_the_digest_paragraph_classifies_each_tree_by_what_actually_moved(
    report: Any, tmp_path: Path
) -> None:
    """§1.5's reproducibility split must be derived from the files, not asserted.

    Two earlier drafts were both wrong, in opposite directions. One claimed *no* recorded
    digest reproduces any more — but the baseline tree is a ``git archive`` snapshot under
    ``artifacts/`` whose ``tooling`` half is empty, so this round's edits never touched it
    and ``source_identity.py --digest-only`` still returns its digest verbatim. The other
    claimed the only moved file "renders prose and takes no timings" — but a probe script
    that *does* time moved for one tree. Neither draft was checked against the files.

    The paragraph now re-hashes each tree's recorded ``source`` and ``tooling`` maps and
    sorts the trees into reproducible / tooling-only / harness-moved / source-moved. This
    test recomputes that classification independently and holds the rendered prose to it:
    the reproducible count and digests, the tooling-only count, and — the invariant that
    makes the "harness didn't move a character" sentence safe — that no tree reported as
    tooling-only actually has its own harness among the changed files.
    """
    root = report.REPRO_ROOT
    blocks: dict[str, dict] = {}
    harnesses: dict[str, set[str]] = {}
    for _, identity in _recorded_identity_blocks(report, root):
        digest = identity["digest"]
        blocks.setdefault(digest, identity)
        # Every harness that collected a JSON in this tree, matching what the paragraph
        # checks: a tree groups JSONs from several harnesses, and "采集它们的 harness 也
        # 一个字没变" is a claim about all of them, so the independent recomputation has
        # to range over all of them too or it would not be checking the same statement.
        harnesses.setdefault(digest, set()).add(
            Path((identity.get("harness") or {}).get("path") or "?").name
        )
    if not blocks:
        pytest.skip(f"{root} carries no source_identity blocks")

    def changed_tooling(identity: dict) -> list[str]:
        _, tool_changed = report.recorded_map_still_matches(identity, "tooling")
        return tool_changed

    reproducible, tooling_only, harness_moved, source_moved = [], [], [], []
    for digest, identity in blocks.items():
        _, src_changed = report.recorded_map_still_matches(identity, "source")
        tool_changed = changed_tooling(identity)
        if src_changed:
            source_moved.append(digest)
        elif harnesses[digest] & {Path(c).name for c in tool_changed}:
            harness_moved.append(digest)
        elif tool_changed:
            tooling_only.append(digest)
        else:
            reproducible.append(digest)

    doc = render_report(report, tmp_path, root, tag="digest").doc
    section = (
        doc.read_text(encoding="utf-8").split("### 1.5 源码身份", 1)[1].split("\n## ", 1)[0]
    )
    flat = re.sub(r"\s+", "", section)

    assert reproducible, "the empty-tooling baseline snapshot must still reproduce"
    assert f"{len(reproducible)}/{len(blocks)}棵树的digest现在仍能原样复现" in flat
    for digest in reproducible:
        assert f"`{digest[:12]}`" in section, f"{digest[:12]} should be named reproducible"
    if tooling_only:
        assert f"另{len(tooling_only)}棵的digest复现不出来了" in flat

    # The baseline is the tree with an empty tooling half; it must land in reproducible,
    # which is exactly the fact the first wrong draft contradicted.
    empty_tooling = [d for d, i in blocks.items() if not (i.get("tooling") or {})]
    assert empty_tooling and set(empty_tooling) <= set(reproducible)

    # The "harness 一个字没变" sentence is emitted for the tooling-only group, so no tree
    # in it may have *any* of its harnesses among the changed files. Assert the invariant
    # over the full harness set, not the prose: this is the check that would have caught
    # the second wrong draft, which named one changed file harmless without asking
    # whether it was a harness.
    for digest in tooling_only:
        moved = harnesses[digest] & {Path(c).name for c in changed_tooling(blocks[digest])}
        assert not moved, f"{digest[:12]} is reported tooling-only but its harness {moved} moved"


def test_the_integration_spread_covers_every_log_on_disk(report: Any) -> None:
    """The 33× spread was quoted from two logs; there are now four.

    §9.4's row is a claim about the machine, not the code, so its sample is every
    duration recorded for that model — and a spread computed from a subset
    understates exactly the instability the row is about. The re-collection added
    57.17 s, which moves the answer from 33× to 67×; the second review's
    re-validation added 3935.49 s from a fresh ``perf-run-*`` root, which moves it
    again. That last one is the observation the row exists for — it landed in the
    slow band *after* the fix, so a sample that only walks the published roots would
    quietly drop it and report the fix as having removed the instability.

    Also pins the de-duplication: generating from the published root makes
    ``DEFAULT_PERF`` and ``PERF`` the same directory, and walking it twice reported
    each duration twice.
    """
    found = report.integration_durations("Qwen3-1.7B-slowq")
    paths = [path for path, _ in found]
    assert len(paths) == len(set(paths)), f"duplicate log counted: {paths}"

    expected: dict[str, float] = {}
    for root in report.repro_roots():
        for relative in report.INTEGRATION_LOG_DIRS:
            path = root.resolve() / relative / "Qwen3-1.7B-slowq.log"
            if not path.is_file():
                continue
            match = re.search(r"\bin (\d+(?:\.\d+)?)s\b",
                              path.read_text(encoding="utf-8", errors="replace")[-400:])
            assert match, f"{path} has no pytest duration line"
            expected[report.repo_relative(path)] = float(match.group(1))
    assert dict(found) == expected
    if not expected:
        pytest.skip(
            "the integration logs this spread is computed from live under a gitignored "
            "artifacts/; this checkout has none"
        )
    assert len(expected) >= 2, "the row needs at least two windows to be a spread at all"
    # The fastest window lives under the re-collection's root and the slowest under a
    # fresh one, so a sample that followed ``--data`` would report 33× from one root
    # and 67× from another for the same round. Four windows is the whole claim.
    assert len(expected) == 4, f"a window dropped out of the sample: {sorted(expected)}"
    assert any("perf-run-" in path for path in expected), (
        "the fresh root's log is missing, so §9.4 would describe only the published runs"
    )
    seconds = sorted(expected.values())
    assert seconds[-1] / seconds[0] > 10, "a spread this wide is the point of the row"


@ab_artifacts_missing
@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_the_cross_check_column_reproduces_its_own_verdict(report: Any, root: Path) -> None:
    """Each cell's verdict has to follow from the numbers printed in that cell.

    The column first printed only the two opt-side ratios and then a verdict. On
    ``Qwen2.5-0.5B emoji-mixed-script`` that produced "`ab/` 0.95、`ab-longctx/` 0.99
    → **两遍对不上**" — two numbers that agree, next to a claim that they do not,
    because the verdict had actually fired on ``ab/``'s *internal* base↔opt
    disagreement. Naming the comparison that fired is half the fix; the other half
    is printing all three disagreement values, since dividing the rounded per-batch
    ratios gives 1.29 where the judgement used 1.31 and a reader who checks the cell
    concludes the report cannot do its own arithmetic.

    The prose counts are checked against the rendered cells rather than against the
    module's own classification, so the paragraph cannot drift from the table it
    summarises. Which branch is taken is a property of the data: the published
    batch's ``ab-longctx/`` ran only the workloads suite and has no ``micro``, so it
    must say 读不出来, while the re-collection must not.
    """
    original = report.PERF
    report.PERF = root
    try:
        line: list[str] = []
        report.rerun_contradiction(line, "ab-longctx")
    finally:
        report.PERF = original
    # Stop at the bare-decode subsection: it is a second table under the same
    # heading with a different column layout, and this test is about the workload
    # table's cross-check column. Scoping the slice keeps every assertion below as
    # strict as it was; letting the new rows in would only teach the loop to parse
    # cells that were never its subject.
    stop = next(
        (i for i, entry in enumerate(line) if entry.lstrip().startswith("#### ")),
        len(line),
    )
    line, text = line[:stop], "\n".join(line[:stop])
    cells = [
        entry.rsplit("|", 2)[-2].strip()
        for entry in line
        if entry.startswith("|") and "---" not in entry and "总加速" not in entry
    ]
    assert cells, f"{root} produced no contradiction rows"

    tally = {"agreed": 0, "internal": 0, "across": 0, "unreadable": 0}
    for cell in cells:
        verdict = cell.split("→", 1)[1]
        if "读不出来" in verdict:
            tally["unreadable"] += 1
            assert "无 `micro`" in cell, "unreadable must say which batch lacks it"
            continue
        printed = [float(v) for v in re.findall(r"(?:`[^`]+` 内部|跨遍 opt) ([\d.]+)×", cell)]
        assert len(printed) == 3, f"expected the three comparisons, got {printed}"
        failed = [
            label.format(first_dir="ab", second_dir="ab-longctx")
            for (label, _, _), value in zip(report.CROSS_CHECKS, printed)
            if value > report.SELF_CHECK_TOLERANCE
        ]
        for label in failed:
            assert label in verdict, f"{label} exceeds the tolerance but the cell says {verdict}"
        for (label, _, _), value in zip(report.CROSS_CHECKS, printed):
            label = label.format(first_dir="ab", second_dir="ab-longctx")
            if value <= report.SELF_CHECK_TOLERANCE:
                assert label not in verdict, f"{label} is inside the tolerance but was named"
        # The raw ratios are printed so a reader can divide them; if the display
        # precision ever stops reproducing the 2-decimal verdict, that is a defect
        # in the column, not in the reader.
        sides = [(float(a), float(b)) for a, b in re.findall(r"base ([\d.]+) / opt ([\d.]+)", cell)]
        assert len(sides) == 2, f"a readable cell must show both batches' ratios: {cell}"
        for (label, _, _), value, pair in zip(report.CROSS_CHECKS, printed, (
            sides[0], sides[1], (sides[0][1], sides[1][1])
        )):
            assert f"{report.ratio_disagreement(*pair):.2f}" == f"{value:.2f}", (
                f"{label}: dividing the printed ratios gives "
                f"{report.ratio_disagreement(*pair):.2f}, the cell claims {value:.2f}"
            )
        if any(item.startswith(report.CROSS_BATCH_LABEL) for item in failed):
            tally["across"] += 1
        elif failed:
            tally["internal"] += 1
        else:
            assert "三项全部一致" in verdict
            tally["agreed"] += 1

    readable = len(cells) - tally["unreadable"]
    if not readable:
        assert f"**但这 {len(cells)} 行拿不到跨遍自检**" in text
        assert "`micro` 为空" in text, "the reason must be stated, not just the absence"
        return
    counts = re.search(
        r"其中 (\d+) 行三项全部一致；(\d+) 行只是.*?；(\d+) 行连\*\*跨遍 opt\*\*", text, re.S
    )
    assert counts, f"the readable branch must break the rows into three kinds:\n{text}"
    assert [int(group) for group in counts.groups()] == [
        tally["agreed"], tally["internal"], tally["across"]
    ], "the paragraph's counts are not the table's"
    assert f"有 {readable} 行两遍都带同窗口 `micro`" in text
    assert "拿不到跨遍自检" not in text, (
        "this batch has same-window micro in both directories; claiming otherwise "
        "discards the only measurement that could say machine or change"
    )


@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_the_opening_discipline_quotes_what_the_evidence_sections_contain(
    report: Any, root: Path, tmp_path: Path
) -> None:
    """§0's two examples have to be traceable, not remembered.

    Discipline 2 opened the report with "Qwen3-1.7B 的端到端负载先后跑出过 0.59× 和
    1.29×" and "集成测试也先后跑出过 115 s 和 3828 s". Both were typed in from the
    published batch. Generated from the re-collected root, §3's table no longer
    contains that pair and §9.4's span had grown to three windows and 67×, so the
    report began by asserting numbers its own evidence sections contradicted — and
    §0 reads as framing, so nobody checks it against a table six sections later.

    Whitespace-stripped comparison because ``write_doc`` reflows to 88 columns and
    splits these clauses at whatever point the line budget lands on.
    """
    doc = render_report(report, tmp_path, root, tag=root.name).doc
    text = doc.read_text(encoding="utf-8")
    opening = re.sub(r"\s+", "", text.split("## 0. 先说怎么读这份报告", 1)[1].split("## 1. ", 1)[0])

    spread = report.headline_numbers()["end_to_end_spread"]
    if not spread:
        pytest.skip(
            "the two-run disagreement §0 quotes is read off the A/B JSON under a "
            "gitignored artifacts/; this checkout has none"
        )
    short, name, low, high, _, spread_root = spread
    assert f"**{low:.2f}×和{high:.2f}×**" in opening
    assert f"`{spread_root}/`" in opening, "§0 must say which root the pair came from"
    rows = report.contradiction_rows(report.ROOT / spread_root)
    assert any(
        row[0] == short and row[1] == name and sorted(row[2:4]) == [low, high]
        for row in rows
    ), f"§0 quotes {short} {name} {low:.2f}×/{high:.2f}× but {spread_root} has no such row"

    windows = report.integration_durations("Qwen3-1.7B-slowq")
    seconds = sorted(value for _, value in windows)
    assert len(seconds) >= 2, "a span needs at least two windows"
    assert "、".join(f"{value:.2f}s" for value in seconds) in opening, (
        "§0 must quote every recorded window: dropping the fastest one is what made "
        "the old 33× claim understate the spread it was illustrating"
    )
    assert f"相差{seconds[-1] / seconds[0]:.0f}倍" in opening


@pytest.mark.parametrize("root", AB_DIRS, ids=lambda p: p.name)
def test_the_reproduction_block_expects_what_its_own_commands_produce(
    report: Any, root: Path, tmp_path: Path
) -> None:
    """§10's "期望" lines have to say which run they came from.

    Two defects, both invisible from either half on its own. The expectations were
    typed in from the published round ("826 passed") while the suite had grown to
    976, so a reader who reproduced *correctly* saw a number the report called
    wrong. And the expectation cited whichever log directory the data root happened
    to hold — generated from the published root that was ``integration-opt4/``, two
    lines below a command that writes ``integration-fixed/logs/``, sending the reader
    to verify against a run they were never told to do.

    The second is why this renders the document instead of calling the helpers: the
    disagreement is between a command line and a prose line, and only the assembled
    text contains both.

    What "name the same run" means changed when the drivers stopped defaulting to a
    directory that already held evidence: §10's commands now write into a fresh
    ``$RUN``, so an expectation *cannot* come from the run those commands produce.
    Naming the source is what keeps it honest instead — see the assertions below.

    Converges in two steps, not one: §10's command pipes the suite into a log under
    ``$RUN``, so running it verbatim does not disturb the log this reads. Run the
    suite to a temporary file, copy it into place, regenerate, then run again.
    """
    doc = render_report(report, tmp_path, root, tag=root.name).doc
    text = doc.read_text(encoding="utf-8")
    block = text.split("## 10. 复现", 1)[1].split("```", 2)[1]

    totals = report.integration_totals()
    if totals is None:
        pytest.skip(
            "the integration logs this expectation is derived from live under a "
            "gitignored artifacts/; this checkout has none"
        )
    passed, failed, directory = totals
    assert f"{passed} passed、{failed} failed" in block
    assert directory in block, (
        f"§10 quotes an expectation derived from `{directory}` without saying where it came from"
    )
    # The expectation is cited from the last complete run, while §10's commands write
    # into a fresh `$RUN`. That split is deliberate and it reverses what this test used
    # to demand: `directory` sits under a published evidence root, so a reproduction
    # command that wrote there would replace the very logs the expectation was read
    # from — the defect tools/perf/out_guard.py exists to prevent. The invariant is
    # therefore no longer "some command writes the cited directory" but "the cited
    # directory is labelled as a previous run, and no command writes into it".
    assert not any(
        directory.rstrip("/") in line
        for line in block.splitlines()
        if line.startswith("bash tools/perf/run_")
    ), f"§10 still tells the reader to write into the cited evidence directory {directory}"
    assert "$RUN" in block, "§10's writes have no fresh run root to land in"
    assert 'fresh-root' in block, "§10 never shows how to obtain that run root"

    quick = report.quick_suite_expectation()
    assert quick, "no quick-suite log on disk to derive an expectation from"
    summary, source = quick
    assert summary in block
    # Asserted apart from the summary because the summary alone cannot see the
    # defect: a count printed beside a directory it was not read from. That stopped
    # being a hypothetical when out_guard closed the published roots to new logs — the
    # newest count now lives in a fresh `perf-run-*` root, so the line has to follow it.
    assert source in block, f"§10 quotes `{summary}` without naming `{source}`"
    assert (REPO_ROOT / source).is_file(), f"§10 names `{source}`, which is not on disk"
    assert summary == report.pytest_summary(REPO_ROOT / source)
    assert "826 passed" not in block, "the published round's count is not this tree's"
    # The duration is deliberately absent: §9.4 exists to say wall time on this host
    # spans two orders of magnitude, so an expectation that included it could not be
    # reproduced by anyone.
    assert not re.search(r"期望：\d+ passed[^\n]*\bin [\d.]+s", block)


# --------------------------------------------------------------------------
# make_report.py: paired rounds, and the two dispersions kept apart
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gains, expected",
    [
        ([1.50, 1.60, 1.45], "收益"),
        ([0.63, 0.67, 0.60], "回退"),
        ([0.90, 1.20], report_indeterminate := "尚不能确认收益或回退"),
        ([1.05, 0.98], report_indeterminate),
        ([0.99, 1.01], report_indeterminate),
    ],
)
def test_gain_verdict_is_symmetric_about_one(report: Any, gains: list[float], expected: str) -> None:
    """Straddling 1.0 is the test, so neither side of it is privileged.

    ``[0.90, 1.20]`` and ``[1.05, 0.98]`` are the same situation seen from
    opposite sides of a favourable point estimate, and must get the same verdict.
    """
    assert report.gain_verdict(gains) == expected
    assert report.gain_verdict(list(reversed(gains))) == expected


def test_the_verdict_constant_is_the_wording_the_review_asked_for(report: Any) -> None:
    """Pinned so a later edit cannot soften it into "roughly flat"."""
    assert report.INDETERMINATE == "尚不能确认收益或回退"


@pytest.mark.parametrize("gains", [[], [1.5], [0.6], [None], [1.5, None]])
def test_one_round_supports_no_verdict_at_all(report: Any, gains: list[Any]) -> None:
    """A single observation has no dispersion, so it cannot settle a direction.

    Returning a verdict here — a favourable one especially — would be the same
    mistake as the single-window decode rows: a number with nothing attached that
    says how much of it was the machine.
    """
    assert report.gain_verdict(gains) is None


def test_within_and_between_process_dispersion_are_separate_numbers(report: Any) -> None:
    """Two processes that each repeat tightly can still land far apart.

    Each process's own samples agree to well under 1%; their medians differ by
    ~10%. Collapsing these into one "noise floor" is what let a tight
    within-process range be read as the resolution of a cross-process comparison.
    """
    processes = [
        [1.00, 1.002, 0.999, 1.001],
        [1.10, 1.102, 1.099, 1.101],
    ]
    within, between = report.within_between(processes)
    assert within is not None and within < 1.01
    assert between is not None and between > 1.05


@pytest.mark.parametrize(
    "values, expected",
    [([], None), ([1.0], None), ([1.0, 2.0], 2.0), ([2.0, 1.0], 2.0), ([0.0, 1.0], None)],
)
def test_spread_ratio_needs_two_positive_numbers(report: Any, values: Any, expected: Any) -> None:
    """One sample is "measured once", not "did not vary"."""
    assert report.spread_ratio(values) == expected


def write_noise_runs(directory: Path, model: str, spec: list[tuple[int, str, int]]) -> None:
    """Fabricate ``noise-floor`` payloads: ``(round, side, position)`` triples."""
    directory.mkdir(parents=True, exist_ok=True)
    for number, side, position in spec:
        payload = {
            "label": f"noise-r{number}-{side}-pos{position}",
            "micro": {"engine_prefill": {"512": {"median": 1.0, "samples": [1.0, 1.0, 1.0]}}},
        }
        (directory / f"{model}-r{number}-{side}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


@pytest.fixture
def report_at(report: Any, tmp_path: Path) -> Any:
    """Point the generator at a scratch data root, and put it back afterwards.

    ``PERF`` and ``PAIRED_ROOT`` are module globals every section reads, so a test
    that repoints one without restoring it would make the rest of the module describe
    a directory that no longer exists.

    ``PAIRED_ROOT`` moves with ``PERF`` because ``paired_directory()`` no longer
    derives one from the other. Setting both to the scratch root is exactly what
    ``main()`` does for a plain ``--data <root>`` invocation, so these tests exercise
    the default path rather than a private arrangement.
    """
    original = (report.PERF, report.PAIRED_ROOT)
    report.PERF = tmp_path
    report.PAIRED_ROOT = tmp_path
    try:
        yield report
    finally:
        report.PERF, report.PAIRED_ROOT = original


def test_balanced_rounds_are_recognised_as_balanced(report_at: Any, tmp_path: Path) -> None:
    model = "Qwen3-0.6B-slowq"
    write_noise_runs(
        tmp_path / report_at.NOISE_DIR,
        model,
        [(1, "base", 1), (1, "opt", 2), (2, "opt", 1), (2, "base", 2)],
    )
    data = report_at.paired_rounds(model, lambda p: p["micro"]["engine_prefill"]["512"]["median"])
    assert data is not None
    assert data["round_count"] == 2
    assert data["base_first_rounds"] == 1
    assert data["order_balanced"] is True


def test_unbalanced_rounds_are_not_reported_as_balanced(report_at: Any, tmp_path: Path) -> None:
    """The whole point of AB/BA is that neither tree systematically runs second.

    A driver bug that always ran base first would otherwise be invisible in the
    output while quietly crediting the optimisation with the second process's warm
    page cache.
    """
    model = "Qwen3-0.6B-slowq"
    write_noise_runs(
        tmp_path / report_at.NOISE_DIR,
        model,
        [(1, "base", 1), (1, "opt", 2), (2, "base", 1), (2, "opt", 2)],
    )
    data = report_at.paired_rounds(model, lambda p: p["micro"]["engine_prefill"]["512"]["median"])
    assert data is not None
    assert data["order_balanced"] is False
    assert data["base_first_rounds"] == 2


def test_a_missing_position_is_unknown_not_balanced(report_at: Any, tmp_path: Path) -> None:
    model = "Qwen3-0.6B-slowq"
    directory = tmp_path / report_at.NOISE_DIR
    directory.mkdir(parents=True)
    for number, side in ((1, "base"), (1, "opt"), (2, "base"), (2, "opt")):
        (directory / f"{model}-r{number}-{side}.json").write_text(
            '{"label": "unlabelled", "micro": {}}', encoding="utf-8"
        )
    data = report_at.paired_rounds(model, lambda p: None)
    assert data is not None
    assert data["order_balanced"] is None


def test_an_absent_experiment_is_none_so_the_gap_gets_reported(
    report_at: Any, tmp_path: Path
) -> None:
    """No directory must read as "not run", never as "balanced" or "zero spread"."""
    assert report_at.paired_rounds("Qwen3-0.6B-slowq", lambda p: None) is None


def test_a_half_finished_round_is_listed_not_silently_dropped(
    report_at: Any, tmp_path: Path
) -> None:
    model = "Qwen3-0.6B-slowq"
    write_noise_runs(
        tmp_path / report_at.NOISE_DIR,
        model,
        [(1, "base", 1), (1, "opt", 2), (2, "base", 1)],
    )
    data = report_at.paired_rounds(model, lambda p: p["micro"]["engine_prefill"]["512"]["median"])
    assert data is not None
    assert data["round_count"] == 1
    assert data["incomplete_rounds"] == [2]


def make_noise_payload(
    side: str,
    round_number: int,
    position: int,
    prefill_median: float,
    decode_median: float,
    workload_median: float,
) -> dict[str, Any]:
    """One ``noise-floor`` payload with a micro probe, a decode probe and a workload."""

    def node(seconds: float, count: int = 7) -> dict[str, Any]:
        return {"median": seconds, "samples": [seconds] * count}

    return {
        "label": f"noise-r{round_number}-{side}-pos{position}",
        "micro": {
            "engine_prefill": {"512": node(prefill_median / 1e3)},
            "raw_decode": {
                "ms_per_token": decode_median,
                "ms_per_token_windows": node(decode_median, 5),
            },
        },
        # Shaped like the harness writes it, not like a probe node: ``total_seconds``
        # here is the *summary* (median/min/max/stdev, no sample list) and the
        # repeats live in ``runs``, one dict each with a bare float. Fabricating
        # samples onto the summary instead — which this fixture did until the
        # end-to-end row vanished from the paired analysis — models a layout that
        # exists nowhere in noise-floor/, and it is the same class of defect as the
        # one it hides: a workload row quietly dropping out.
        "workloads": {
            "zh-short": {
                "total_seconds": {
                    "median": workload_median / 1e3,
                    "min": workload_median / 1e3,
                    "max": workload_median / 1e3,
                    "stdev": 0.0,
                },
                "runs": [{"total_seconds": workload_median / 1e3} for _ in range(7)],
            },
            "skipped-one": {"skipped": "prompt too long for max_context"},
        },
    }


def test_paired_quantities_covers_probes_decode_and_workloads(report: Any) -> None:
    """Each quantity is judged on its own samples, so each has to be found.

    The "3% floor" this replaces came from one probe and was then applied to
    decode and to end-to-end workloads alike.
    """
    payload = make_noise_payload("base", 1, 1, 1000.0, 6.8, 480.0)
    found = report.paired_quantities(payload)
    assert "引擎分块 prefill 512 tok" in found
    assert "decode 短上下文" in found
    assert "zh-short（端到端）" in found
    # A workload the harness refused has no measurement, so it must not become a
    # row of em-dashes that reads as "measured, no difference".
    assert not any("skipped" in name for name in found)
    assert found["引擎分块 prefill 512 tok"][0] == pytest.approx(1000.0)
    # Seven repeats, in ms: the samples have to come from ``runs``, not from the
    # summary node, which carries none.
    assert found["zh-short（端到端）"][1] == pytest.approx([480.0] * 7)


NOISE_FLOOR_DIR = REPO_ROOT / "artifacts" / "perf-round1-rerun" / "noise-floor"
noise_floor_missing = pytest.mark.skipif(
    not NOISE_FLOOR_DIR.is_dir() or not list(NOISE_FLOOR_DIR.glob("*.json")),
    reason="the noise-floor artifacts are gitignored; re-run tools/perf/run_noise_floor.sh",
)


@noise_floor_missing
def test_the_fixture_shape_is_the_shape_the_harness_writes(report: Any) -> None:
    """Pin the fabrication to the real payload, because a fixture that drifts lies.

    ``make_noise_payload`` invented ``workloads.<name>.total_seconds.samples``.
    Nothing on disk has that key: the summary node is
    median/min/max/stdev and the repeats live under ``runs`` as bare floats. The
    fixture still made the tests pass, because ``paired_quantities`` was reading
    samples off the same invented key — so both sides agreed on a layout that
    exists nowhere, and every end-to-end row dropped out of the real analysis
    without a single test noticing.
    """
    fabricated = make_noise_payload("base", 1, 1, 1000.0, 6.8, 480.0)["workloads"]["zh-short"]
    real: list[dict[str, Any]] = []
    for path in sorted(NOISE_FLOOR_DIR.glob("*.json")):
        payload = report.load(path)
        if not payload:
            continue
        for node in (payload.get("workloads") or {}).values():
            if isinstance(node, dict) and "skipped" not in node:
                real.append(node)
    assert real, "expected the collected runs to contain at least one measured workload"
    for node in real:
        summary = node.get("total_seconds")
        assert isinstance(summary, dict)
        assert "samples" not in summary, "the summary node carries no sample list"
        assert {"median", "min", "max", "stdev"} <= set(summary)
        runs = node.get("runs")
        assert isinstance(runs, list) and runs
        assert all(isinstance(run.get("total_seconds"), float) for run in runs)
    # A real node carries fifteen keys; the fixture only the two this function
    # reads. So subset on the outer level, equality on the one whose key set was
    # invented.
    assert {"total_seconds", "runs"} <= set(fabricated)
    assert set(fabricated) <= set(real[0])
    assert set(fabricated["total_seconds"]) == set(real[0]["total_seconds"])
    assert set(fabricated["runs"][0]) <= set(real[0]["runs"][0])


def test_the_returned_dispersion_is_one_quantity_not_a_global_max(
    report_at: Any, tmp_path: Path
) -> None:
    """A pathological row elsewhere must not change what the sweep tables compare against.

    This is the regression that produced the "3% noise floor": a single number
    summarising every quantity in the run, taken as the worst of them, then used as
    the resolution of comparisons it was never measured on. Here the decode probe
    swings by 4x across rounds while the 512-token prefill is stable to 1%, and the
    returned value must be the prefill's.
    """
    model = "Qwen3-0.6B-slowq"
    directory = tmp_path / report_at.NOISE_DIR
    directory.mkdir(parents=True)
    for round_number in (1, 2):
        order = ("base", "opt") if round_number == 1 else ("opt", "base")
        for position, side in enumerate(order, start=1):
            # Prefill stable across rounds; decode deliberately wildly unstable.
            decode = 6.0 if side == "base" else 6.0 * (1 + 3.0 * (round_number - 1))
            payload = make_noise_payload(
                side, round_number, position,
                prefill_median=1000.0 if side == "base" else 700.0,
                decode_median=decode,
                workload_median=480.0 if side == "base" else 400.0,
            )
            (directory / f"{model}-r{round_number}-{side}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

    line: list[str] = []
    returned = report_at.paired_noise_section(line, [model])
    table = "\n".join(line)

    # The unstable decode row is present and called indeterminate, so the section
    # did not simply drop it to keep its headline small.
    assert report_at.INDETERMINATE in table
    assert "decode 短上下文" in table

    # With two rounds and identical prefill medians per side, the prefill's
    # between-process dispersion is zero — nowhere near the decode's 300%.
    assert returned is not None
    assert returned < 1.0, f"expected the prefill's own dispersion, got {returned}%"


def test_an_unmeasured_sweep_quantity_returns_none_rather_than_borrowing(
    report_at: Any, tmp_path: Path
) -> None:
    """If the rounds lack the quantity the sweep tables need, say so.

    Returning some other quantity's dispersion would be the borrowing this section
    exists to stop; None makes the callers print "no dispersion available" instead.
    """
    model = "Qwen3-0.6B-slowq"
    directory = tmp_path / report_at.NOISE_DIR
    directory.mkdir(parents=True)
    for round_number in (1, 2):
        order = ("base", "opt") if round_number == 1 else ("opt", "base")
        for position, side in enumerate(order, start=1):
            payload = {
                "label": f"noise-r{round_number}-{side}-pos{position}",
                "micro": {},
                "workloads": {
                    "zh-short": {
                        "total_seconds": {"median": 0.48, "samples": [0.48] * 7}
                    }
                },
            }
            (directory / f"{model}-r{round_number}-{side}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    line: list[str] = []
    assert report_at.paired_noise_section(line, [model]) is None
    assert "没有可引用的离散度" in "\n".join(line)


# --------------------------------------------------------------------------
# order balance recomputed over the rounds a row actually kept
# --------------------------------------------------------------------------
#
# The defect: ``paired_rounds`` computed AB/BA balance once per model, over the
# rounds as collected, and every (model, quantity) row of that model printed the
# same answer. But the contamination filter runs *per quantity*, so the round set a
# row reasons about is usually smaller than the set that was balanced. In
# artifacts/perf-round1-rerun/noise-floor six rows kept only r2/r3/r4 — 1 AB + 2 BA —
# and the report still printed 「顺序均衡：是」 for them, and five of the six still
# counted toward 「70/73 达到稳定收益判据」. The second process in a pair inherits a
# warm page cache, so an unbalanced remainder hands that advantage to one tree and
# the resulting ratio is not a measurement of the optimisation.


def noise_payload(
    round_number: int,
    side: str,
    position: Any,
    nodes: dict[str, tuple[float, Any]],
) -> dict[str, Any]:
    """A noise-floor payload carrying exactly the ``engine_prefill`` lengths named.

    ``nodes`` maps a length key to ``(median_ms, samples_ms or None)``. The median is
    what the gain is computed from and the samples are what the contamination filter
    judges, so a test can spike one round's samples while leaving its median — and
    therefore its gain — exactly where it wants it. That separation is the only way to
    reach "balanced as collected, unbalanced once the filter has run" without also
    moving the number under test.

    ``position=None`` omits the ``-pos`` marker entirely, which is how a run that
    never recorded where it sat is fabricated.
    """
    label = f"noise-r{round_number}-{side}"
    if position is not None:
        label += f"-pos{position}"
    micro: dict[str, Any] = {}
    for length, (median_ms, samples_ms) in nodes.items():
        samples = [median_ms] * 7 if samples_ms is None else list(samples_ms)
        micro[length] = {
            "median": median_ms / 1e3,
            "samples": [sample / 1e3 for sample in samples],
        }
    return {"label": label, "micro": {"engine_prefill": micro}}


def write_noise_rounds(
    directory: Path,
    model: str,
    spec: Sequence[tuple[int, str, Any, dict[str, tuple[float, Any]]]],
) -> None:
    """Fabricate a ``noise-floor`` directory from ``(round, side, position, nodes)``."""
    directory.mkdir(parents=True, exist_ok=True)
    for round_number, side, position, nodes in spec:
        payload = noise_payload(round_number, side, position, nodes)
        (directory / f"{model}-r{round_number}-{side}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def alternating_rounds(
    rounds: Sequence[int],
    *,
    base_ms: float = 1000.0,
    opt_ms: float = 700.0,
    length: str = "512",
    opt_samples: Any = None,
    position: Any = "alternating",
) -> list[tuple[int, str, Any, dict[str, tuple[float, Any]]]]:
    """A balanced AB/BA spec over ``rounds``: odd rounds run base first.

    ``opt_samples`` overrides the opt side's sample list for the *last* round only,
    which is how one round is made to fail the spike test without touching its
    median. ``position`` may be ``"alternating"`` (the default), ``None`` (no marker
    recorded) or a callable taking the round number.
    """
    spec: list[tuple[int, str, Any, dict[str, tuple[float, Any]]]] = []
    for number in rounds:
        base_first = number % 2 == 1
        order = (("base", 1), ("opt", 2)) if base_first else (("opt", 1), ("base", 2))
        for side, slot in order:
            if position == "alternating":
                slot_position: Any = slot
            elif callable(position):
                slot_position = position(number, side, slot)
            else:
                slot_position = position
            samples = None
            if side == "opt" and opt_samples is not None and number == rounds[-1]:
                samples = opt_samples
            median = base_ms if side == "base" else opt_ms
            spec.append((number, side, slot_position, {length: (median, samples)}))
    return spec


def rendered_rows(table: str) -> list[dict[str, str]]:
    """The paired table's data rows as dicts keyed by the header's own column names.

    Read out of the header rather than by position, so a column added in the middle
    fails the tests that index it instead of silently shifting every assertion by one.
    """
    lines = [line for line in table.splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    out = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == len(header):
            out.append(dict(zip(header, cells)))
    return out


def test_a_round_dropped_by_the_filter_breaks_a_balance_the_old_code_called_true(
    report_at: Any, tmp_path: Path
) -> None:
    """The reported defect, on data built so that only the filtering can cause it.

    Four rounds collected, 2 AB + 2 BA. One opt sample in r4 is 4.3x that round's
    median, so the spike rule drops r4 and leaves 1 AB + 2 BA. The gains are
    unaffected — r4's median stays where it was — so the four ratios still say 收益
    and the collected rounds still say balanced. Those are exactly the two numbers
    the old code read, and the row now has to say neither.
    """
    model = "Qwen3-0.6B-slowq"
    spiked = [700.0] * 6 + [3000.0]
    write_noise_rounds(
        tmp_path / report_at.NOISE_DIR,
        model,
        alternating_rounds((1, 2, 3, 4), opt_samples=spiked),
    )

    # What the old code read, from the functions it called: balance over all four
    # collected rounds, and a verdict over all four gains.
    collected = report_at.paired_rounds(
        model, lambda payload: report_at.paired_quantities(payload)
    )
    assert collected is not None and collected["order_balanced"] is True
    assert collected["round_count"] == 4
    gains = [1000.0 / 700.0] * 4
    assert report_at.gain_verdict(gains) == "收益"

    line: list[str] = []
    report_at.paired_noise_section(line, [model])
    table = "\n".join(line)
    row = rendered_rows(table)[0]

    assert row["弃用轮次"] == "r4", "the dropped round must still be named"
    assert row["有效轮次"] == "3/4"
    assert row["顺序均衡（保留轮次）"] == "**否（base 先跑 2/3）**"
    assert row["结论"] == report_at.UNORDERED
    assert "剔除异常轮次之后顺序不再均衡的行" in table
    assert report_at.PAIRED_SUMMARY["directional"] == 0
    assert report_at.PAIRED_SUMMARY["unbalanced_rows"] == 1


def test_rounds_unbalanced_as_collected_never_reach_a_direction(
    report_at: Any, tmp_path: Path
) -> None:
    """A driver that ran base first every round cannot produce a verdict either."""
    model = "Qwen3-0.6B-slowq"
    write_noise_rounds(
        tmp_path / report_at.NOISE_DIR,
        model,
        alternating_rounds(
            (1, 2, 3, 4), position=lambda number, side, slot: 1 if side == "base" else 2
        ),
    )
    line: list[str] = []
    report_at.paired_noise_section(line, [model])
    table = "\n".join(line)
    row = rendered_rows(table)[0]

    assert row["顺序均衡（保留轮次）"] == "**否（base 先跑 4/4）**"
    assert row["结论"] == report_at.UNORDERED
    assert row["弃用轮次"] == "—", "nothing was filtered here; the collection was wrong"
    assert report_at.PAIRED_SUMMARY["unbalanced"] == 1
    assert report_at.PAIRED_SUMMARY["directional"] == 0


def test_a_position_that_was_never_recorded_is_not_read_as_balanced(
    report_at: Any, tmp_path: Path
) -> None:
    """``None`` is neither 是 nor 否: "not recorded" must not become "balanced".

    Two shapes, because they fail differently. A label with no ``-pos`` marker at all
    is a harness that never wrote one; a marker whose value is not a digit is a label
    that was corrupted or reformatted. Both leave the balance question unanswerable,
    and guessing it from the filename order is what would hide a driver bug.
    """
    for model, position in (("Qwen3-0.6B-slowq", None), ("SmolLM2-360M-Instruct-slowq", "x")):
        write_noise_rounds(
            tmp_path / report_at.NOISE_DIR,
            model,
            alternating_rounds((1, 2), position=position),
        )
    line: list[str] = []
    report_at.paired_noise_section(line, ["Qwen3-0.6B-slowq", "SmolLM2-360M-Instruct-slowq"])
    rows = rendered_rows("\n".join(line))
    assert len(rows) == 2
    for row in rows:
        assert row["顺序均衡（保留轮次）"] == "**未记录**"
        assert row["结论"] == report_at.UNORDERED
    assert report_at.PAIRED_SUMMARY["directional"] == 0
    assert report_at.PAIRED_SUMMARY["unbalanced_rows"] == 2


def test_one_surviving_round_is_a_thin_sample_and_not_a_design_failure(
    report_at: Any, tmp_path: Path
) -> None:
    """Insufficient rounds get their own label, ahead of the balance test.

    One round is arithmetically unbalanced too — ``1 * 2 != 1`` — so asking about
    balance first would report a single surviving measurement as a driver bug. The
    order matters because the two labels send the reader to different remedies: add
    rounds, or fix the harness.
    """
    model = "Qwen3-0.6B-slowq"
    write_noise_rounds(
        tmp_path / report_at.NOISE_DIR,
        model,
        alternating_rounds((1,), position=None),
    )
    # A second model whose two collected rounds lose one to the spike rule, so the
    # row is left with exactly one round from a set that *was* balanced.
    other = "SmolLM2-360M-Instruct-slowq"
    write_noise_rounds(
        tmp_path / report_at.NOISE_DIR,
        other,
        alternating_rounds((1, 2), opt_samples=[700.0] * 6 + [3000.0]),
    )
    # A third model whose *every* round fails the same rule, so nothing is retained.
    # ``_order_balance`` on an empty set would otherwise answer True — ``0 * 2 == 0``
    # — and the row would print 「顺序均衡：是」 about rounds it does not have.
    stripped = "Qwen2.5-0.5B-Instruct-slowq"
    spiked_both = [
        (number, side, position, {length: (median, [median] * 6 + [median * 4.3])})
        for number, side, position, nodes in alternating_rounds((1, 2))
        for length, (median, _samples) in nodes.items()
    ]
    write_noise_rounds(tmp_path / report_at.NOISE_DIR, stripped, spiked_both)
    line: list[str] = []
    report_at.paired_noise_section(line, [model, other, stripped])
    rows = {row["模型"]: row for row in rendered_rows("\n".join(line))}

    assert rows["Qwen3-0.6B"]["有效轮次"] == "1/1"
    assert rows["Qwen3-0.6B"]["结论"] == report_at.INSUFFICIENT
    assert rows["SmolLM2-360M"]["有效轮次"] == "1/2"
    assert rows["SmolLM2-360M"]["结论"] == report_at.INSUFFICIENT
    assert rows["Qwen2.5-0.5B"]["有效轮次"] == "0/2"
    assert rows["Qwen2.5-0.5B"]["顺序均衡（保留轮次）"] == "**无保留轮次**"
    assert rows["Qwen2.5-0.5B"]["结论"] == report_at.INSUFFICIENT
    assert report_at.PAIRED_SUMMARY["insufficient_rows"] == 3
    # Neither is listed as an order problem: the remedy is more rounds, not a fix to
    # the harness, and saying otherwise sends the reader to the wrong one.
    assert report_at.PAIRED_SUMMARY["unbalanced_rows"] == 0
    assert report_at.PAIRED_SUMMARY["directional"] == 0


def test_a_balanced_row_still_reaches_its_direction(report_at: Any, tmp_path: Path) -> None:
    """The gate must not suppress the rows it was written to protect.

    Four rounds, alternating, nothing filtered: this is the case the paired
    experiment exists for, and a fix that made every row cautious would pass all the
    tests above while destroying the section's only useful output.
    """
    model = "Qwen3-0.6B-slowq"
    write_noise_rounds(
        tmp_path / report_at.NOISE_DIR, model, alternating_rounds((1, 2, 3, 4))
    )
    line: list[str] = []
    report_at.paired_noise_section(line, [model])
    table = "\n".join(line)
    row = rendered_rows(table)[0]

    assert row["有效轮次"] == "4/4"
    assert row["顺序均衡（保留轮次）"] == "是"
    assert row["结论"] == "收益"
    assert report_at.PAIRED_SUMMARY["directional"] == 1
    assert report_at.PAIRED_SUMMARY["unbalanced_rows"] == 0
    assert report_at.PAIRED_SUMMARY["insufficient_rows"] == 0
    assert "有限轮次方向一致" in table
    assert "测得稳定收益" not in table


def test_an_unbalanced_slowdown_is_excluded_the_same_way_a_speedup_is(
    report_at: Any, tmp_path: Path
) -> None:
    """Symmetric in the gain's direction, or the gate only ever blocks bad news.

    Two models over the same round structure: one where opt is faster, one where opt
    is slower. Both lose a round to the same spike rule, so both retain an unbalanced
    set. If the gate were written against ``gain > 1`` the 回退 row would still be
    printed as a direction while the 收益 row was not — and a rule that only ever
    removes favourable rows manufactures the regression it reports.
    """
    faster = "Qwen3-0.6B-slowq"
    slower = "SmolLM2-360M-Instruct-slowq"
    write_noise_rounds(
        tmp_path / report_at.NOISE_DIR,
        faster,
        alternating_rounds((1, 2, 3, 4), opt_samples=[700.0] * 6 + [3000.0]),
    )
    write_noise_rounds(
        tmp_path / report_at.NOISE_DIR,
        slower,
        alternating_rounds(
            (1, 2, 3, 4), base_ms=700.0, opt_ms=1000.0, opt_samples=[1000.0] * 6 + [4300.0]
        ),
    )
    line: list[str] = []
    report_at.paired_noise_section(line, [faster, slower])
    rows = {row["模型"]: row for row in rendered_rows("\n".join(line))}

    assert rows["Qwen3-0.6B"]["结论"] == report_at.UNORDERED
    assert rows["SmolLM2-360M"]["结论"] == report_at.UNORDERED
    assert report_at.PAIRED_SUMMARY["directional"] == 0
    assert report_at.PAIRED_SUMMARY["unbalanced_rows"] == 2
    assert report_at.PAIRED_SUMMARY["verdicts"].get("收益", 0) == 0
    assert report_at.PAIRED_SUMMARY["verdicts"].get("回退", 0) == 0

    # And the same two rows, balanced, do reach opposite directions: the gate is
    # about the round set, not about which way the ratio points.
    balanced_dir = tmp_path / "balanced"
    write_noise_rounds(
        balanced_dir / report_at.NOISE_DIR, faster, alternating_rounds((1, 2, 3, 4))
    )
    write_noise_rounds(
        balanced_dir / report_at.NOISE_DIR,
        slower,
        alternating_rounds((1, 2, 3, 4), base_ms=700.0, opt_ms=1000.0),
    )
    original = report_at.PAIRED_ROOT
    report_at.PAIRED_ROOT = balanced_dir
    try:
        line = []
        report_at.paired_noise_section(line, [faster, slower])
    finally:
        report_at.PAIRED_ROOT = original
    rows = {row["模型"]: row for row in rendered_rows("\n".join(line))}
    assert rows["Qwen3-0.6B"]["结论"] == "收益"
    assert rows["SmolLM2-360M"]["结论"] == "回退"
    assert report_at.PAIRED_SUMMARY["directional"] == 2


def test_the_summary_counts_are_the_verdicts_the_table_printed(
    report_at: Any, tmp_path: Path
) -> None:
    """§7.2 and §9.1 read ``PAIRED_SUMMARY``; the reader reads the table.

    Those have to be the same set of verdicts, which is what let 「70/73」 survive a
    table that no longer supported it. Every row of a mixed fabrication is counted
    back out of the rendered markdown and compared to the summary the prose quotes,
    so a future change that gates one and not the other fails here.
    """
    faster = "Qwen3-0.6B-slowq"
    slower = "SmolLM2-360M-Instruct-slowq"
    # r4's opt spikes, so both of the faster model's rows lose that round; one file
    # carries both lengths, because one file per (round, side) is what the harness
    # writes and a second spec for the same pair would overwrite the first.
    spike = [700.0] * 6 + [3000.0]
    two_quantity: list[tuple[int, str, Any, dict[str, tuple[float, Any]]]] = []
    for number, side, position, nodes in alternating_rounds(
        (1, 2, 3, 4), length="512", opt_samples=spike
    ):
        both = dict(nodes)
        both["32"] = nodes["512"]
        two_quantity.append((number, side, position, both))
    write_noise_rounds(tmp_path / report_at.NOISE_DIR, faster, two_quantity)
    write_noise_rounds(
        tmp_path / report_at.NOISE_DIR,
        slower,
        alternating_rounds((1, 2), base_ms=700.0, opt_ms=1000.0, length="512"),
    )
    line: list[str] = []
    report_at.paired_noise_section(line, [faster, slower])
    table = "\n".join(line)
    rows = rendered_rows(table)
    summary = report_at.PAIRED_SUMMARY

    counted: dict[str, int] = {}
    for row in rows:
        counted[row["结论"]] = counted.get(row["结论"], 0) + 1
    assert counted == summary["verdicts"], (counted, summary["verdicts"])
    assert sum(counted.values()) == summary["rows"] == len(rows)
    assert (
        counted.get("收益", 0) + counted.get("回退", 0) == summary["directional"]
    )
    assert counted.get(report_at.INDETERMINATE, 0) == summary["indeterminate"]
    assert counted.get(report_at.UNORDERED, 0) == summary["unbalanced_rows"]
    assert counted.get(report_at.INSUFFICIENT, 0) == summary["insufficient_rows"]
    # Self-consistency alone would also pass if every row were suppressed, so the
    # fabrication's expected shape is pinned too: two unbalanced remainders from the
    # model that lost r4, and one balanced 回退 from the model that did not.
    assert summary["rows"] == 3
    assert summary["directional"] == 1
    assert summary["unbalanced_rows"] == 2
    assert summary["insufficient_rows"] == 0
    assert summary["indeterminate"] == 0
    assert counted == {report_at.UNORDERED: 2, "回退": 1}


def test_a_re_measured_paired_run_is_read_from_its_own_directory_and_says_so(
    report_at: Any, tmp_path: Path
) -> None:
    """``--paired-data`` moves the paired table and nothing else.

    The drivers refuse to write into the published evidence directory, so a
    re-measurement has to land somewhere of its own — and a report generated from
    the old root still has to read the new one, and *name* it. A table that silently
    changed shape is indistinguishable from a table somebody edited.
    """
    published = tmp_path / "published"
    remeasured = tmp_path / "remeasured"
    model = "Qwen3-0.6B-slowq"
    # The same four rounds, one of them spiked in the re-measurement, so the two
    # directories give different verdicts and cannot be mistaken for one another.
    write_noise_rounds(
        published / report_at.NOISE_DIR, model, alternating_rounds((1, 2, 3, 4))
    )
    write_noise_rounds(
        remeasured / report_at.NOISE_DIR,
        model,
        alternating_rounds((1, 2, 3, 4), opt_samples=[700.0] * 6 + [3000.0]),
    )
    report_at.PERF = published
    report_at.PAIRED_ROOT = published
    assert report_at.paired_directory() == published / report_at.NOISE_DIR
    line: list[str] = []
    report_at.paired_noise_section(line, [model])
    table = "\n".join(line)
    assert rendered_rows(table)[0]["结论"] == "收益"
    assert "remeasured" not in table

    original = report_at.PAIRED_ROOT
    report_at.PAIRED_ROOT = remeasured
    try:
        assert report_at.paired_directory() == remeasured / report_at.NOISE_DIR
        line = []
        report_at.paired_noise_section(line, [model])
    finally:
        report_at.PAIRED_ROOT = original
    table = "\n".join(line)
    assert rendered_rows(table)[0]["结论"] == report_at.UNORDERED
    assert str(remeasured) in table, "the override has to name the directory it read"
    assert "不能替换它" in table, "the override has to say the old directory survives"


# --------------------------------------------------------------------------
# sigmoid attribution (tools/perf/sigmoid_check.py, ops.sigmoid's docstring)
# --------------------------------------------------------------------------
#
# The probe priced optimisation 2a against something it called "real gate
# activation" but computed as
#
#     rms_norm(embed_tokens(ids), input_layernorm.weight) @ gate_proj.weight.T
#
# A decoder layer does not do that. Between the embedding and ``gate_proj`` sit
# the whole attention block, a residual add, and a *second* norm —
# ``post_attention_layernorm``, whose weight is a different tensor from
# ``input_layernorm``'s. So the three headline numbers built on it (7.20x faster,
# 38.8% of values negative, 153% of the layer's matmul budget) described an
# activation this model never produces.
#
# The second false claim is cheaper to state and easier to check: "they come out
# of an RMSNorm, so they are centred by construction". RMSNorm divides by the
# root-mean-square. It never subtracts the mean. LayerNorm does; RMSNorm was
# written partly to stop doing exactly that.


@pytest.fixture(scope="module")
def sigmoid_probe() -> Any:
    return load_tool("sigmoid_check")


@pytest.fixture(scope="module")
def tiny_forward(tiny_model_dir: Path) -> Any:
    """A real forward through the tiny checkpoint, with ``silu`` under observation."""
    from slowq.model import CausalLMModel

    return CausalLMModel.from_directory(tiny_model_dir)


TINY_IDS = list(range(8))


def observed_silu_inputs(model: Any, ids: list[int]) -> list[Any]:
    """Every tensor ``silu`` is handed during one real forward, in layer order.

    Patched on the module, not on ``ops``: ``model.py`` does ``from .ops import
    silu``, so the name the layer actually calls is ``slowq.model.silu``. Patching
    ``ops.silu`` would observe nothing and the test would pass on an empty list if
    it were not for the length assertion in the caller.
    """
    import numpy as np

    import slowq.model as model_module

    seen: list[Any] = []
    real = model_module.silu

    def spy(x: Any) -> Any:
        seen.append(np.array(x, copy=True))
        return real(x)

    model_module.silu = spy
    try:
        model.forward(ids, model.new_cache(64))
    finally:
        model_module.silu = real
    return seen


def shortcut_gate_activation(model: Any, ids: list[int]) -> Any:
    """What the probe used to compute and call "real gate activation"."""
    import numpy as np

    from slowq import ops

    cfg = model.config
    weights = model.weights
    hidden = np.asarray(model.embed(np.asarray(ids, dtype=np.int64)), dtype=np.float32)
    normed = ops.rms_norm(
        hidden,
        weights.expect(f"{cfg.tensor_prefix}.layers.0.input_layernorm.weight", (cfg.hidden_size,)),
        cfg.rms_norm_eps,
    )
    return ops.linear(normed, weights.get(f"{cfg.tensor_prefix}.layers.0.mlp.gate_proj.weight"))


def test_rms_norm_does_not_subtract_the_mean() -> None:
    """The "centred by construction" claim, falsified on a constant vector.

    A constant row has mean 3 and standard deviation 0. RMSNorm divides by
    sqrt(mean(x^2)) = 3, so every element comes out as 1.0 — the mean survives,
    scaled. LayerNorm would have returned 0.0. If the output were centred the
    assertion below would be the other way round.
    """
    import numpy as np

    from slowq import ops

    x = np.full((4, 8), 3.0, dtype=np.float32)
    out = ops.rms_norm(x, np.ones(8, dtype=np.float32), 1e-6)
    assert abs(float(out.mean()) - 1.0) < 1e-3
    assert abs(float(out.mean())) > 0.5, "RMSNorm removed the mean, which would make it LayerNorm"


def test_a_shifted_input_keeps_its_shift_through_rms_norm() -> None:
    """Same point on a realistic vector: the shift is scaled down, not removed."""
    import numpy as np

    from slowq import ops

    rng = np.random.default_rng(20261026)
    x = (rng.standard_normal((16, 64)) * 0.1 + 1.0).astype(np.float32)
    out = ops.rms_norm(x, np.ones(64, dtype=np.float32), 1e-6)
    assert float(x.mean()) > 0.9
    assert float(out.mean()) > 0.9, "the mean should survive, divided by the RMS"


def test_the_shortcut_is_not_what_silu_receives(tiny_forward: Any) -> None:
    """The two tensors differ by orders of magnitude, so the old numbers priced
    an activation this model never produces.

    This is the evidence that the attribution was wrong; it holds whether or not
    the probe has been fixed, which is why it is separate from the test below.
    """
    import numpy as np

    truth = observed_silu_inputs(tiny_forward, TINY_IDS)[0]
    claimed = shortcut_gate_activation(tiny_forward, TINY_IDS)

    assert truth.shape == claimed.shape
    assert not np.allclose(truth, claimed, rtol=1e-3, atol=1e-4), (
        "the input_layernorm+gate_proj shortcut reproduced the real silu input; "
        "if that ever happens, this test's premise is wrong and the probe's old "
        "numbers were fine after all"
    )
    # Reported rather than merely asserted: the magnitude is the reason the
    # headline share was meaningless, not just technically inexact.
    relative = float(
        np.linalg.norm(truth - claimed) / max(float(np.linalg.norm(truth)), 1e-12)
    )
    assert relative > 0.1, f"shortcut is only {relative:.3f} off in rel_l2 — too close to matter"


def test_the_probe_reports_what_silu_really_receives(
    sigmoid_probe: Any, tiny_forward: Any, tiny_model_dir: Path
) -> None:
    """After the fix the probe's "real" activation *is* the observed one.

    Before the fix this fails: the function took a token count and a hardcoded
    model directory, and returned the shortcut above.
    """
    import numpy as np

    truth = observed_silu_inputs(tiny_forward, TINY_IDS)[0]
    claimed = sigmoid_probe.real_gate_activation(tiny_model_dir, np.asarray(TINY_IDS))
    assert np.array_equal(np.asarray(claimed), np.asarray(truth))


def test_the_probe_also_captures_the_mlp_input(
    sigmoid_probe: Any, tiny_forward: Any, tiny_model_dir: Path
) -> None:
    """The substitution measurement needs the tensor ``_mlp`` is called with.

    Pricing "old sigmoid vs new sigmoid inside the real MLP" requires the real
    MLP's input, which is the ``post_attention_layernorm`` output — not the gate
    activation, which is one matmul downstream of it.
    """
    import numpy as np

    import slowq.model as model_module

    seen: list[Any] = []
    real_mlp = model_module.DecoderModel._mlp

    def spy(self: Any, prefix: str, x: Any) -> Any:
        seen.append(np.array(x, copy=True))
        return real_mlp(self, prefix, x)

    model_module.DecoderModel._mlp = spy
    try:
        tiny_forward.forward(TINY_IDS, tiny_forward.new_cache(64))
    finally:
        model_module.DecoderModel._mlp = real_mlp

    captured = sigmoid_probe.capture_layer_zero(tiny_model_dir, np.asarray(TINY_IDS))
    assert np.array_equal(np.asarray(captured["mlp_input"]), seen[0])
    assert np.array_equal(np.asarray(captured["silu_input"]), observed_silu_inputs(tiny_forward, TINY_IDS)[0])


def test_the_probe_keeps_the_synthetic_inputs_but_says_what_they_are_for(
    sigmoid_probe: Any,
) -> None:
    """Synthetic distributions bound the operator; they are not attributions.

    The review asks for them to stay and to be labelled. What must not survive is
    the naming that let a synthetic row read like a measurement of this model.
    """
    labels = sigmoid_probe.SYNTHETIC_DISTRIBUTIONS
    assert set(labels) == {"standard normal", "all positive", "all negative"}
    assert "real gate activation" not in labels, (
        "the captured activation is not a synthetic distribution and must not be "
        "listed among them"
    )


def test_the_probe_separates_the_four_quantities(sigmoid_probe: Any) -> None:
    """Four different numbers were reported as if they were one.

    (1) the sigmoid operator alone, (2) the sum of a few warm matmul medians,
    (3) a real forward with one variable changed, (4) the end-to-end gain. The
    old "153%" was (1) divided by (2) and called a share of the layer. Each must
    carry its own name so a reader cannot mistake one for another.
    """
    assert sigmoid_probe.QUANTITIES == {
        "operator": "sigmoid 算子单独计时（新旧写法，同一份输入）",
        "matmul_sum": "该层 7 个 matmul 各自中位数之和（不是整层 profile）",
        "real_substitution": "真实前向里只替换 silu 的新旧写法（整层归因）",
        "end_to_end": "端到端收益，来自 ab/ 的 workloads，本脚本不测",
    }


# --------------------------------------------------------------------------
# reflow: an indented table row must survive byte-for-byte
# --------------------------------------------------------------------------


def test_reflow_leaves_an_indented_table_row_alone(report: Any) -> None:
    """A table nested in a list item is still a table.

    ``reflow``'s contract is that tables keep their exact bytes because wrapping
    them changes what renders. The guard tested ``item.startswith("|")``, so a row
    indented under a bullet — which is how a correction note inside an attribution
    list has to be written — fell through and got wrapped mid-cell. GFM then shows
    the second half of the row as prose below a truncated table.
    """
    row = "  | (3) | 真实前向里只替换 `silu` | 同一层、同一份输入，只改一个变量 | 能，这是唯一的整层归因 |"
    assert report.display_width(row) > 88, "the row has to be over-long to be a real test"
    assert report.reflow([row]) == [row]


def test_reflow_still_wraps_indented_prose(report: Any) -> None:
    """The fix must not turn reflow off for everything indented."""
    prose = ("  这一行是普通叙述文字，" * 8).rstrip()
    out = report.reflow([prose])
    assert len(out) > 1, "prose under an indent should still wrap"
    assert "".join(out).replace(" ", "") == prose.replace(" ", ""), "wrapping may not lose text"


def test_reflow_keeps_continuations_inside_the_indent(report: Any) -> None:
    """An indented paragraph stays one block after wrapping.

    A correction note under an attribution bullet has to be written as several
    indented strings, because only the first carries the ``-`` marker. Wrapping
    their continuations back to column 0 detaches them from the list item, so the
    rendered note falls out of the list it was written inside.
    """
    line = ("  已发布的比值被称为「占整层 matmul 预算的 153.2%」，那是把 (1) 除以 (2) "
            "再当成 (3) 来讲，这个说法撤回。")
    out = report.reflow([line])
    assert len(out) > 1, "the line has to be long enough to wrap for this to mean anything"
    assert all(piece.startswith("  ") for piece in out), out
    assert all(report.display_width(piece) <= 88 for piece in out), out


def test_reflow_leaves_a_list_marker_indent_alone(report: Any) -> None:
    """The marker case already worked; pin it so the new branch cannot break it."""
    line = "- **优化 2 拿走了其余部分。**「两条独立路线」那张表是这件事的交叉核对，它的相差列就是可信度。"
    out = report.reflow([line])
    assert out[0].startswith("- ")
    assert all(piece.startswith("  ") for piece in out[1:]), out


def test_the_shipped_docstring_does_not_repeat_the_withdrawn_attribution() -> None:
    """The false claim shipped in ``ops.py``, so it has to be corrected there too.

    A report can carry a correction and leave the source saying the old thing; a
    reader who only opens ``ops.py`` then learns the withdrawn version. These are
    the specific strings that were wrong, not a general style check:

    * "centred by construction" — RMSNorm divides by the root mean square and
      never subtracts the mean, so nothing about it centres anything.
    * 153% "of the layer's matmuls" — a ratio of an isolated operator to a sum of
      separately timed warm matmuls, presented as a share of a real layer.
    * 7.20x and 38.8% — measured on a tensor the model never produces.

    What replaces them is deliberately coarse, and coarser than I first wrote it.
    Three runs of the same probe gave layer shares of 30.8%, 31.4% and 27.3%, so a
    docstring quoting "~31%" was quoting one run and calling it a fact. The
    replacement quotes the range over every preserved run, and the drift guard below
    derives the required strings from those runs — so a fourth run that lands
    outside them fails the suite instead of silently making the source wrong.
    """
    from slowq import ops

    doc = squash(ops.sigmoid.__doc__ or "")
    for withdrawn in ("centred by construction", "天然居中", "153%", "7.20", "38.8"):
        assert withdrawn not in doc, f"ops.sigmoid still claims {withdrawn!r}"
    # And the replacement has to be there, or the docstring is merely silent.
    assert "of the real MLP" in doc, "the real-MLP attribution is missing"
    assert "of a real decoder layer" in doc, "the real-layer attribution is missing"
    # Deterministic, not a timing: reproduced exactly by every run.
    assert "84.5%" in doc, "the real activation's negative fraction is missing"


def squash(text: str) -> str:
    """Collapse whitespace so a rewrapped docstring does not change what this proves.

    Every assertion below is a substring check on prose that is also wrapped to a
    column limit. Without this, moving one word to the next line would read as the
    claim having been deleted.
    """
    return " ".join(text.split())


def decimals(quoted: str) -> int:
    """How many decimal places a range endpoint was written with.

    The resolution a sentence claims, used to round an observation before asking
    whether the sentence covers it. Without it, a measured 3.8996 fails a quoted
    "3.90-4.35x" and the only way out is to pad the band, which hides the real
    spread behind invented margin.
    """
    return len(quoted.split(".", 1)[1]) if "." in quoted else 0


def test_the_shipped_docstring_separates_the_operator_from_the_layer() -> None:
    """The docstring must say which of its numbers is an attribution.

    Both the ~4x operator speedup and the ~30% layer share are true; only one of
    them is "what the rewrite removes from a real layer". A reader who cannot tell
    them apart will quote the larger one, which is the exact failure this
    correction exists for.

    Checked against phrases with no markup in them: the disclaimer is written
    ``*not* a layer profile``, and asserting on "not a layer profile" would fail on
    the emphasis markers rather than on the claim.
    """
    from slowq import ops

    doc = squash(ops.sigmoid.__doc__ or "")
    assert "must not be read as a share of anything" in doc
    assert "a layer profile" in doc, "the matmul-sum denominator must be disclaimed"
    assert "Quote (3)" in doc, "the docstring must name which quantity to quote"
    assert "about 4x" in doc, "the operator figure must be labelled as the operator's own"
    assert "perf-round1-rerun/sigmoid/" in doc, "the reader must be sent to the raw data"


def probe_artifacts(subdir: str, pattern: str) -> list[Path]:
    """Every ``artifacts/*/<subdir>/<pattern>``, oldest first by mtime.

    One probe's artifacts do not live in one directory any more, and cannot: a
    re-measurement is refused entry to the directory the previous one is cited from
    (``tools/perf/out_guard.py``), which is the rule that keeps the old numbers
    auditable. So "the latest run" has to be *found* rather than pointed at. Pointing
    at a fixed path is what let the two meanings of "latest" come apart — the gate
    below read ``sigmoid-check.json`` from a directory that had stopped receiving
    anything, and called it the latest artifact in the repository.

    Sorted by mtime with the name as a tiebreak: two copies of the same run name can
    exist in two roots, and a stable order is what makes "the last element" mean
    something.
    """
    files = [
        path
        for directory in sorted((REPO_ROOT / "artifacts").glob(f"*/{subdir}"))
        for path in directory.glob(pattern)
        if path.is_file()
    ]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))


def artifact_label(path: Path) -> str:
    """``<run-root>/<subdir>/<file>`` — unique, and it says which run root it came from.

    ``<subdir>/<file>`` is not enough once the same probe has been re-measured into a
    second root: both roots hold a ``sigmoid-check.json``, and a failure message that
    names two different files identically is worse than one that names neither.
    """
    try:
        return path.relative_to(REPO_ROOT / "artifacts").as_posix()
    except ValueError:  # pragma: no cover - every artifact lives under artifacts/
        return path.as_posix()


SIGMOID_PATTERN = "sigmoid-check*.json"


def sigmoid_artifacts() -> list[Path]:
    return probe_artifacts("sigmoid", SIGMOID_PATTERN)


#: The newest sigmoid probe run anywhere under ``artifacts/``, and the one the
#: self-reference gate judges. ``CORRECTED_SIGMOID_DIR`` survives as the directory the
#: *correction* was first measured into — ``ops.sigmoid``'s docstring sends the reader
#: there — but it is no longer where "latest" comes from.
CORRECTED_SIGMOID_DIR = REPO_ROOT / "artifacts" / "perf-round1-rerun" / "sigmoid"
CORRECTED_SIGMOID_ARTIFACT = (sigmoid_artifacts() or [CORRECTED_SIGMOID_DIR / "sigmoid-check.json"])[-1]


def sigmoid_runs() -> list[tuple[str, dict]]:
    """Every preserved probe run across every run root, oldest first, latest last."""
    return [
        (artifact_label(p), json.loads(p.read_text(encoding="utf-8")))
        for p in sigmoid_artifacts()
    ]


#: ``(regex naming the two endpoints in the docstring, how to read a run's value)``.
#: A selector may return one number or a list; all of them are pooled before the
#: comparison, which is how the one-signed and within-run rows cover several
#: quantities with one assertion each.
DOCSTRING_RANGES: dict[str, tuple[str, Any]] = {
    "operator speedup on the real activation": (
        r"\*\*([\d.]+)-([\d.]+)x \(about 4x\)\*\*",
        lambda d: d["sigmoid_speedup_real_gate"],
    ),
    "matmul-sum total (ms)": (
        r"separately timed warm matmuls\*\*, ([\d.]+)-([\d.]+) ms",
        lambda d: d["matmul_total_seconds"] * 1000,
    ),
    "operator-over-matmul-sum ratio (%)": (
        r"yields \*\*([\d.]+)-([\d.]+)%\*\*",
        lambda d: d["old_sigmoid_share_of_matmuls"] * 100,
    ),
    "share of the real MLP removed (%)": (
        r"removes \*\*([\d.]+)-([\d.]+)% of the real MLP\*\*",
        lambda d: d["real_substitution"]["mlp_saved_share"] * 100,
    ),
    "share of a real decoder layer removed (%)": (
        r"and \*\*([\d.]+)-([\d.]+)% of a real decoder layer\*\*",
        lambda d: d["real_substitution"]["layer_saved_share"] * 100,
    ),
    "operator speedup on one-signed input": (
        r"within ([\d.]+)-([\d.]+)x of each other",
        lambda d: [
            d["sigmoid_rows"]["all positive"]["speedup"],
            d["sigmoid_rows"]["all negative"]["speedup"],
        ],
    ),
    "operator speedup on synthetic standard normal": (
        r"gives ([\d.]+)-([\d.]+)x",
        lambda d: d["sigmoid_rows"]["standard normal"]["speedup"],
    ),
    "within-run stdev/median (%)": (
        r"stdev/median of ([\d.]+)-([\d.]+)%",
        lambda d: [
            d["real_substitution"][k]["stdev"] / d["real_substitution"][k]["median"] * 100
            for k in ("layer_old", "layer_new", "mlp_old", "mlp_new")
        ],
    ),
}


@pytest.mark.skipif(
    not CORRECTED_SIGMOID_ARTIFACT.exists(),
    reason="the corrected artifact is gitignored; re-run tools/perf/sigmoid_check.py",
)
def test_the_docstring_ranges_cover_every_preserved_run() -> None:
    """The quoted range must contain every run, and must be a range.

    Cover semantics, not equality, and at the precision the docstring quotes. A
    point estimate would have been falsified by the very next run: three runs of the
    identical probe gave layer shares of 30.8%, 31.4% and 27.3%, so the docstring
    has to quote a band. Rounding the observation to the quoted endpoint's own
    precision before comparing is what keeps the band honest rather than padded —
    3.8996 satisfies "3.90-4.35x" because that is the resolution the sentence
    claims, while 4.50 would not.

    Covering also means the check fails in the direction that matters: a new run
    outside the band says the source now understates the spread, and someone has to
    widen it. Requiring ``low < high`` is what stops the band collapsing back to a
    single run after a cleanup, which would quietly reintroduce the original
    mistake.
    """
    from slowq import ops

    runs = sigmoid_runs()
    assert len(runs) >= 2, "a range over one run is a point estimate with extra steps"
    doc = squash(ops.sigmoid.__doc__ or "")

    for label, (pattern, select) in DOCSTRING_RANGES.items():
        match = re.search(pattern, doc)
        assert match, f"cannot find the {label} range in ops.sigmoid's docstring"
        low_text, high_text = match.group(1), match.group(2)
        low, high = float(low_text), float(high_text)
        assert low < high, f"{label}: {low_text}-{high_text} is not a range"
        precision = max(decimals(low_text), decimals(high_text))

        observed: list[float] = []
        for _, data in runs:
            value = select(data)
            observed.extend(value if isinstance(value, list) else [value])
        observed = [round(value, precision) for value in observed]
        assert min(observed) >= low, (
            f"{label}: a preserved run reached {min(observed):.{precision}f}, below "
            f"the {low_text} the docstring quotes"
        )
        assert max(observed) <= high, (
            f"{label}: a preserved run reached {max(observed):.{precision}f}, above "
            f"the {high_text} the docstring quotes"
        )


@pytest.mark.skipif(
    not CORRECTED_SIGMOID_ARTIFACT.exists(),
    reason="the corrected artifact is gitignored; re-run tools/perf/sigmoid_check.py",
)
def test_between_run_spread_exceeds_within_run_dispersion() -> None:
    """Pin the observation that makes the ranges above necessary.

    Within a run the 15 samples are tight, to a few percent of the median. That
    tightness is what made a single run look like a settled measurement. The point
    of this test is the comparison: the spread of the layer share *across* runs is
    several times the worst within-run relative dispersion, so the samples inside
    one process are not repeated measurements of the quantity being quoted. Same
    lesson as the decode-window one in ``core_ab.py``, reached from the other
    direction.

    The last assertion is the docstring's "differs by more than 4 points", which is
    monotone-safe: adding a run can only widen a min-max spread, so the sentence
    cannot be falsified by re-measuring — only by discovering the spread was always
    smaller than claimed.
    """
    from slowq import ops

    runs = sigmoid_runs()
    assert len(runs) >= 2, "one run cannot show a between-run spread"

    within = []
    for name, data in runs:
        for key in ("layer_old", "layer_new", "mlp_old", "mlp_new"):
            block = data["real_substitution"][key]
            within.append(block["stdev"] / block["median"])
    worst_within = max(within)

    layer_shares = [d["real_substitution"]["layer_saved_share"] * 100 for _, d in runs]
    spread_points = max(layer_shares) - min(layer_shares)
    relative_spread = spread_points / (sum(layer_shares) / len(layer_shares))

    assert worst_within < 0.05, worst_within
    assert relative_spread > 2 * worst_within, (
        f"between-run relative spread {relative_spread:.3f} is not clearly larger "
        f"than the worst within-run dispersion {worst_within:.3f}; the docstring's "
        "claim that the samples inside a run are not repeats would be unsupported"
    )
    assert spread_points > 4.0, spread_points
    assert "more than 4 points" in squash(ops.sigmoid.__doc__ or "")


@pytest.mark.skipif(
    not CORRECTED_SIGMOID_ARTIFACT.exists(),
    reason="the corrected artifact is gitignored; re-run tools/perf/sigmoid_check.py",
)
def test_the_deterministic_captures_are_identical_across_runs() -> None:
    """Separate what drifts from what cannot.

    The timings move between runs; the captured tensors must not, because they are
    a function of the weights and the input ids. If these ever differ, the probe is
    no longer measuring the same thing and every number in it is unrelated to the
    others — which is a much worse failure than noise.

    The second half checks that the docstring quotes those stable numbers at the
    precision the data supports, derived from the artifact rather than typed in:
    unlike the timings, these are allowed to be exact.
    """
    from slowq import ops

    runs = sigmoid_runs()
    _, first = runs[0]
    for name, data in runs[1:]:
        assert data["gate_activation"] == first["gate_activation"], name
        assert data["mlp_input"] == first["mlp_input"], name
        assert data["shortcut_vs_captured"] == first["shortcut_vs_captured"], name
        assert data["input_ids"] == first["input_ids"], name
        assert data["bit_equality_notes"] == first["bit_equality_notes"], name

    shortcut = first["shortcut_vs_captured"]
    assert shortcut["rel_l2"] > 1.0, (
        "the shortcut was expected to be worse than useless against the real "
        "activation; if this now holds, the premise of the correction changed"
    )

    gate = first["gate_activation"]
    doc = squash(ops.sigmoid.__doc__ or "")
    for needle in (
        f"{gate['fraction_negative'] * 100:.1f}%",
        f"mean {gate['mean']:.3f}",
        f"std {gate['std']:.3f}",
        f"rel_l2 {shortcut['rel_l2']:.2f}",
    ):
        assert needle in doc, f"the docstring owes the deterministic value {needle!r}"


@pytest.mark.skipif(
    not CORRECTED_SIGMOID_ARTIFACT.exists(),
    reason="the corrected artifact is gitignored; re-run tools/perf/sigmoid_check.py",
)
def test_the_latest_artifact_still_identifies_the_code_it_measured() -> None:
    """Why the probe records two hashes per file, and which one gates a re-run.

    The numbers measured here are quoted in ``ops.sigmoid``'s docstring, which lives
    inside one of the hashed files. So correcting the prose to match the measurement
    changes the file hash the measurement recorded — and if that hash were the gate,
    the only way to satisfy it would be to re-measure, which moves the numbers,
    which changes the prose again. ``*_code_sha256`` is the gate instead: it ignores
    docstrings and line numbers, so it changes exactly when the computation changes.

    ``sigmoid-check-run1.json`` is kept even though its ``ops_py_sha256`` names a
    file that no longer exists, because overwriting it would erase the evidence that
    one endpoint of the quoted range came from an earlier state of the source.

    The raw ``*_sha256`` values stay in the artifact and are deliberately not
    asserted here: the strictest question they answer ("is this byte-for-byte the
    file that ran?") has a no for any run whose prose was corrected afterwards, and
    a test that fails on that would train people to re-measure instead of to write
    down what they learned.
    """
    probe = load_tool("sigmoid_check")
    latest = json.loads(CORRECTED_SIGMOID_ARTIFACT.read_text())

    identity = latest["identity"]
    for key, path in (
        ("ops_py", REPO_ROOT / "src" / "slowq" / "ops.py"),
        ("model_py", REPO_ROOT / "src" / "slowq" / "model.py"),
        ("probe", Path(probe.__file__)),
    ):
        assert identity[f"{key}_code_sha256"] == probe.code_sha256(path), (
            f"{path.name} computes something different from what "
            f"{CORRECTED_SIGMOID_ARTIFACT.name} measured. Re-run the probe."
        )

    run1 = json.loads((CORRECTED_SIGMOID_DIR / "sigmoid-check-run1.json").read_text())
    assert run1["identity"]["ops_py_sha256"] != latest["identity"]["ops_py_sha256"], (
        "run1 is supposed to be an earlier state of the source; if its ops.py hash "
        "matches the latest, the two artifacts are duplicates and one is not evidence"
    )
    assert "ops_py_code_sha256" not in run1["identity"], (
        "run1 predates the code hash; if it has one, the archive was regenerated "
        "rather than preserved"
    )


def test_code_sha256_ignores_prose_and_line_numbers(
    sigmoid_probe: Any, tmp_path: Path
) -> None:
    """The mechanism the identity gate above rests on, tested directly.

    Four properties, each of which would silently break the gate if lost:

    * a docstring edit does not change the hash — this is the whole point;
    * a body edit does — otherwise the hash proves nothing about behaviour;
    * a comment edit does not change it — comments are not in the AST at all;
    * inserting a line above does not change it — ``ast.dump`` is called without
      attributes, so positions stay out of the hash. Rewrapping the docstring that
      quotes these numbers shifts every line below it, and that must not look like
      a code change.
    """
    source = tmp_path / "sample.py"
    original = '''"""Module prose."""


def f(x):
    """Function prose.

    Several lines of it.
    """
    # a comment
    return x * 2
'''
    source.write_text(original, encoding="utf-8")
    base = sigmoid_probe.code_sha256(source)

    source.write_text(original.replace("Function prose.", "Corrected prose."), encoding="utf-8")
    assert sigmoid_probe.code_sha256(source) == base, "a docstring edit must not look like a code change"

    source.write_text(original.replace('"""Module prose."""', '"""Rewritten."""'), encoding="utf-8")
    assert sigmoid_probe.code_sha256(source) == base

    source.write_text(original.replace("    # a comment", "    # a different comment"), encoding="utf-8")
    assert sigmoid_probe.code_sha256(source) == base, "comments are not code"

    source.write_text("\n\n" + original, encoding="utf-8")
    assert sigmoid_probe.code_sha256(source) == base, "line numbers must not enter the hash"

    source.write_text(original.replace("x * 2", "x * 3"), encoding="utf-8")
    assert sigmoid_probe.code_sha256(source) != base, "the hash must notice a behaviour change"

    source.write_text(original.replace("Function prose.", ""), encoding="utf-8")
    assert sigmoid_probe.code_sha256(source) == base, "removing a docstring is still prose"

    source.write_text(original + "\n\ndef g():\n    return 1\n", encoding="utf-8")
    assert sigmoid_probe.code_sha256(source) != base, "adding a function is a code change"



# --------------------------------------------------------------------------
# identity-check cost (tools/perf/identity_cost.py, VisionEngine's docstrings)
# --------------------------------------------------------------------------
#
# ``VisionEngine._model_identity_changed`` runs before *every* image cache lookup,
# and its docstring justified that with "0.028 ms median over 200 calls" — a number
# with no artifact behind it anywhere in the repository. The neighbouring
# ``_image_cache_key`` docstring quoted "15.6 ms for a 5-frame 15.0 MiB one ... at
# roughly 1 GB/s", which priced only ``pixel_values`` and silently left out the
# int64 attention mask the same call hashes: 40% of the bytes.
#
# Both are now measured by ``tools/perf/identity_cost.py`` into
# ``artifacts/perf-round1-rerun/identity/``, and the assertions below are the reason
# the probe records two hashes per file and quotes bands rather than points.

#: The directory the first set of identity runs was measured into. Kept as a name
#: because ``VisionEngine``'s docstrings cite it; it is no longer where "latest" comes
#: from, for the reason :func:`probe_artifacts` gives.
IDENTITY_DIR = REPO_ROOT / "artifacts" / "perf-round1-rerun" / "identity"
SMOLVLM_PATTERN = "SmolVLM-256M-Instruct-slowq-run*.json"
QWEN_LARGE_PATTERN = "Qwen3-1.7B-slowq-run*.json"

identity_artifacts_missing = pytest.mark.skipif(
    not probe_artifacts("identity", SMOLVLM_PATTERN),
    reason="the identity artifacts are gitignored; re-run tools/perf/identity_cost.py",
)


@pytest.fixture(scope="module")
def identity_probe() -> Any:
    return load_tool("identity_cost")


def identity_runs(pattern: str = SMOLVLM_PATTERN) -> list[tuple[str, dict]]:
    """Every preserved identity run matching ``pattern``, across every run root.

    Pooling runs made by *different versions of the probe* is legitimate here and only
    here: the versions differ in where they write, not in what they time. The gate
    below is the one that separates them, by probe hash.
    """
    return [
        (artifact_label(path), json.loads(path.read_text(encoding="utf-8")))
        for path in probe_artifacts("identity", pattern)
    ]


def test_the_identity_probe_imports_by_path_and_parses_no_argv_at_import(
    identity_probe: Any,
) -> None:
    """The load above *is* the assertion.

    ``identity_cost.py`` imports its sibling ``source_identity`` by putting
    ``tools/perf`` on ``sys.path`` itself rather than relying on the script
    directory, and keeps argument parsing inside ``main``. Get either wrong and a
    by-path load consumes pytest's own ``sys.argv`` and dies with "unrecognized
    arguments" — which is why the earlier ``ARGS = parse_args(sys.argv[1:])`` at
    module level had to go.
    """
    args = identity_probe.parse_args(["--model", "models/x", "--calls", "7"])
    assert (args.model, args.calls) == ("models/x", 7)
    assert not hasattr(identity_probe, "ARGS"), (
        "a module-level ARGS means importing this tool parses the caller's argv"
    )
    for name in ("measure_fingerprint", "measure_key", "measure_weight_hash",
                 "machine_state", "code_sha256", "sha256_file"):
        assert hasattr(identity_probe, name), name


def identity_layer_ms(data: dict, layer: str) -> Any:
    """One ``identity_layers`` median in ms, or ``None`` if that probe never timed it.

    ``None`` and not a default: the gate below skips a run that did not measure the
    quantity and asserts that at least one run did, so a band always describes runs
    that observed the thing it is quoting. Returning 0.0 here would make every old
    artifact "agree" with every band.
    """
    seconds = data.get("identity_layers", {}).get(layer, {}).get("seconds")
    return None if seconds is None else seconds["median"] * 1e3


def cache_path_seconds(
    data: dict, image: int, path: str, *, scale: float = 1.0
) -> Any:
    """One ``cache_paths`` wall-clock median, or ``None`` before that section existed."""
    rows = data.get("cache_paths")
    if not rows or len(rows) <= image:
        return None
    return rows[image]["paths"][path]["wall_seconds"]["median"] * scale


def derived_values(data: dict, key: str, *, scale: float = 1.0) -> Any:
    """A per-image list out of ``derived``, or ``None`` if that ratio was not derived."""
    values = data.get("derived", {}).get(key)
    return None if values is None else [value * scale for value in values]


#: ``(which docstring, regex naming the two quoted endpoints, how to read a run)``.
#: A selector may return one number or a list; all are pooled before the
#: comparison, which is how the within-run row covers six observations at once.
IDENTITY_RANGES: dict[str, tuple[str, str, Any]] = {
    "fingerprint median (ms)": (
        "_model_identity_changed",
        r"read a \*\*([\d.]+)-([\d.]+) ms\*\* median",
        lambda d: d["fingerprint"]["seconds"]["median"] * 1e3,
    ),
    "fingerprint within-run min and max (ms)": (
        "_model_identity_changed",
        r"one run of 200 calls \(([\d.]+)-([\d.]+) ms\)",
        lambda d: [
            d["fingerprint"]["seconds"]["min"] * 1e3,
            d["fingerprint"]["seconds"]["max"] * 1e3,
        ],
    ),
    "pixel key, 5 frames (ms)": (
        "_model_identity_changed",
        r"spends \*\*([\d.]+)-([\d.]+) ms\*\* hashing a 5-frame",
        lambda d: d["keys"][0]["seconds"]["median"] * 1e3,
    ),
    "pixel key, 9 frames (ms)": (
        "_model_identity_changed",
        r"and \*\*([\d.]+)-([\d.]+) ms\*\* on a 9-frame one",
        lambda d: d["keys"][1]["seconds"]["median"] * 1e3,
    ),
    "key-to-fingerprint ratio": (
        "_model_identity_changed",
        r"i\.e\. \*\*(\d+)-(\d+)x\*\* a bare ``fingerprint\(\)`` call",
        lambda d: d["derived"]["key_to_fingerprint"],
    ),
    "identity check median (ms)": (
        "_model_identity_changed",
        r"reads \*\*([\d.]+)-([\d.]+) ms\*\*, i\.e\.",
        lambda d: identity_layer_ms(d, "identity_check"),
    ),
    "identity check against the bare fingerprint": (
        "_model_identity_changed",
        r"i\.e\. \*\*([\d.]+)-([\d.]+)x\*\* the bare fingerprint",
        lambda d: d.get("derived", {}).get("identity_check_to_fingerprint"),
    ),
    "cache hit, 5 frames (ms)": (
        "_model_identity_changed",
        r"costs \*\*([\d.]+)-([\d.]+) ms\*\* for the 5-frame image",
        lambda d: cache_path_seconds(d, 0, "hit", scale=1e3),
    ),
    "cache hit, 9 frames (ms)": (
        "_model_identity_changed",
        r"and \*\*([\d.]+)-([\d.]+) ms\*\* for the 9-frame one",
        lambda d: cache_path_seconds(d, 1, "hit", scale=1e3),
    ),
    "identity check as a share of the hit (%)": (
        "_model_identity_changed",
        r"the identity check is \*\*([\d.]+)%-([\d.]+)%\*\* of that",
        lambda d: derived_values(d, "identity_share_of_hit", scale=100.0),
    ),
    "hit as a share of the miss (%)": (
        "_model_identity_changed",
        r"the hit is \*\*([\d.]+)%-([\d.]+)%\*\* of the miss it replaces",
        lambda d: derived_values(d, "hit_to_miss", scale=100.0),
    ),
    "cache-disabled request, 5 frames (s)": (
        "_model_identity_changed",
        r"costs \*\*([\d.]+)-([\d.]+) s\*\* per request instead of",
        lambda d: cache_path_seconds(d, 0, "disabled"),
    ),
    "cache miss, 5 frames (s)": (
        "_model_identity_changed",
        r"A miss costs \*\*([\d.]+)-([\d.]+) s\*\*",
        lambda d: cache_path_seconds(d, 0, "miss"),
    ),
    "miss against the disabled path": (
        "_model_identity_changed",
        r"which is \*\*([\d.]+)-([\d.]+)x\*\* the disabled path",
        lambda d: derived_values(d, "miss_to_disabled"),
    ),
    "weight sha256 on SmolVLM (s)": (
        "_model_identity_changed",
        r"measured \*\*([\d.]+)-([\d.]+) s\*\* on SmolVLM-256M",
        lambda d: d["weight_hash"]["seconds"],
    ),
    "weight sha256 as a share of the encode (%)": (
        "_model_identity_changed",
        r"\*\*([\d.]+)%-([\d.]+)% of the very encode",
        lambda d: d["derived"]["weight_hash_share_of_encode"] * 100,
    ),
    "pixel key, 5 frames, as quoted by _image_cache_key (ms)": (
        "_image_cache_key",
        r"\*\*([\d.]+)-([\d.]+) ms\*\* for a 5-frame one",
        lambda d: d["keys"][0]["seconds"]["median"] * 1e3,
    ),
    "pixel key, 9 frames, as quoted by _image_cache_key (ms)": (
        "_image_cache_key",
        r"\*\*([\d.]+)-([\d.]+) ms\*\* for a 9-frame one",
        lambda d: d["keys"][1]["seconds"]["median"] * 1e3,
    ),
    "key as a share of the vision encode (%)": (
        "_image_cache_key",
        r"That is \*\*([\d.]+)%-([\d.]+)%\*\* of the vision encode",
        lambda d: [share * 100 for share in d["derived"]["key_share_of_encode"]],
    ),
    "key throughput (GiB/s)": (
        "_image_cache_key",
        r"runs at ([\d.]+)-([\d.]+) GiB/s in every run",
        lambda d: [row["throughput_gib_per_second"] for row in d["keys"]],
    ),
    "vision encode (s)": (
        "_image_cache_key",
        r"a fraction of, ([\d.]+)-([\d.]+) s depending",
        lambda d: d["encode_image"]["seconds"],
    ),
}


@identity_artifacts_missing
def test_the_identity_docstring_bands_cover_every_run() -> None:
    """Cover semantics at the precision quoted, over every preserved run.

    The same rule the sigmoid section uses, for the same reason: a point estimate
    here was falsified by the second process. Three runs of an identical probe read
    0.0286, 0.0287 and 0.0295 ms, and the 5-frame key moved 15.41 to 16.33 ms —
    6%, which is larger than the quantity the docstring is arguing is negligible.
    Re-measuring into a second run root widened both again, to 0.0304 ms and
    16.62 ms, and a third root widened eleven of the twenty bands again — so the
    bands in the docstring are written over all nine processes and not over the
    three that were available when the rule was first needed.
    """
    from slowq.vlm import VisionEngine

    runs = identity_runs()
    assert len(runs) >= 2, "a range over one run is a point estimate with extra steps"

    for label, (attr, pattern, select) in IDENTITY_RANGES.items():
        doc = squash(getattr(VisionEngine, attr).__doc__ or "")
        match = re.search(pattern, doc)
        assert match, f"cannot find the {label} range in VisionEngine.{attr}'s docstring"
        low_text, high_text = match.group(1), match.group(2)
        low, high = float(low_text), float(high_text)
        assert low < high, f"{label}: {low_text}-{high_text} is not a range"
        precision = max(decimals(low_text), decimals(high_text))

        observed: list[float] = []
        for _, data in runs:
            value = select(data)
            if value is None:
                # A run collected before this quantity existed. Skipping it does not
                # loosen the band: the band covers the runs that measured the thing,
                # and the assertion below says at least one did. Requiring every run
                # to have every key would make an old artifact impossible to keep,
                # and keeping old artifacts is the rule this whole file enforces.
                continue
            observed.extend(value if isinstance(value, list) else [value])
        assert observed, f"{label}: no preserved run measured this"
        observed = [round(value, precision) for value in observed]
        assert min(observed) >= low, (
            f"{label}: a preserved run reached {min(observed):.{precision}f}, below "
            f"the {low_text} the docstring quotes"
        )
        assert max(observed) <= high, (
            f"{label}: a preserved run reached {max(observed):.{precision}f}, above "
            f"the {high_text} the docstring quotes"
        )


@identity_artifacts_missing
def test_the_large_model_figures_are_bands_over_every_preserved_qwen_run() -> None:
    """Qwen3-1.7B now has three runs, so its figures are bands and must say so.

    This test used to check two *scalars*, because only one run existed and a
    scalar was the honest form. The second run moved the fingerprint 0.0224 to
    0.0238 ms and the weight hash 7.91 to 8.49 s, and the third moved the
    fingerprint again, to 0.0240 ms, while leaving the weight hash band alone. The
    docstring widened with each, so what is checked here is now stricter than the
    cover rule above: each endpoint has to *equal* the observation it came from, at
    the precision written. Padding a band to make a re-measurement fit would pass a
    cover check and fail this one, which is the point — these endpoints are a record
    of the measured spread, not a tolerance around an expectation.
    """
    from slowq.vlm import VisionEngine

    doc = squash(VisionEngine._model_identity_changed.__doc__ or "")
    qwen = identity_runs(QWEN_LARGE_PATTERN)
    assert len(qwen) >= 2, (
        f"only {len(qwen)} {QWEN_LARGE_PATTERN} run(s); the docstring quotes these "
        "as ranges, and a range over one run is a point estimate with extra steps"
    )

    for label, pattern, select in (
        ("Qwen3-1.7B fingerprint (ms)",
         r"manifest reads \*\*([\d.]+)-([\d.]+) ms\*\*",
         lambda d: d["fingerprint"]["seconds"]["median"] * 1e3),
        ("Qwen3-1.7B weight hash (s)",
         r"and \*\*([\d.]+)-([\d.]+) s\*\* on Qwen3-1\.7B",
         lambda d: d["weight_hash"]["seconds"]),
    ):
        match = re.search(pattern, doc)
        assert match, f"cannot find the {label} range in the docstring"
        low_text, high_text = match.group(1), match.group(2)
        precision = max(decimals(low_text), decimals(high_text))
        observed = [round(select(data), precision) for _, data in qwen]
        assert min(observed) == float(low_text), (
            f"{label}: the runs reach {min(observed):.{precision}f}, the docstring "
            f"starts its range at {low_text}"
        )
        assert max(observed) == float(high_text), (
            f"{label}: the runs reach {max(observed):.{precision}f}, the docstring "
            f"ends its range at {high_text}"
        )


@identity_artifacts_missing
def test_fingerprint_cost_tracks_manifest_entries_not_tensor_bytes() -> None:
    """The property that makes a per-request identity check affordable at all.

    SmolVLM-256M has *more* manifest entries and 6.7x *fewer* tensor bytes than
    Qwen3-1.7B, and reads slower. If the cost tracked bytes the larger model would
    be hundreds of times slower and the design would be wrong; the ordering is what
    makes the docstring's "reads the manifest and never the tensors" a measured
    claim rather than a restatement of the implementation.
    """
    small = identity_runs()
    large = identity_runs(QWEN_LARGE_PATTERN)
    assert small and large

    little = small[0][1]["fingerprint"]
    big = large[0][1]["fingerprint"]
    assert big["tensor_bytes_logical"] > 6 * little["tensor_bytes_logical"], (
        "the two models must differ enough in size for the comparison to mean anything"
    )
    assert big["manifest_tensor_entries"] < little["manifest_tensor_entries"], (
        "the point needs the larger model to have *fewer* manifest entries"
    )
    assert big["seconds"]["median"] < little["seconds"]["median"], (
        "6.7x the bytes read slower, so the cost is not tracking the byte count"
    )
    # Closer to the entry-count ratio than to the byte-count ratio, which is the
    # quantitative form of the same statement.
    entry_ratio = little["manifest_tensor_entries"] / big["manifest_tensor_entries"]
    byte_ratio = big["tensor_bytes_logical"] / little["tensor_bytes_logical"]
    cost_ratio = little["seconds"]["median"] / big["seconds"]["median"]
    assert abs(cost_ratio - entry_ratio) < abs(cost_ratio - byte_ratio), (
        f"cost ratio {cost_ratio:.2f} sits nearer the byte ratio {byte_ratio:.2f} "
        f"than the entry ratio {entry_ratio:.2f}"
    )


@identity_artifacts_missing
def test_the_key_cost_is_priced_over_both_arrays() -> None:
    """The specific error the corrected docstring names.

    The old prose quoted 15.0 and 27.0 MiB at "roughly 1 GB/s". Those are the
    ``pixel_values`` sizes alone; the int64 attention mask hashed in the same call
    is 40% of the bytes. Pricing the operation over part of its input understated
    the rate by ~60% and made the cost look like a rounding error rather than a
    measured 1.5 GiB/s sweep.
    """
    for name, data in identity_runs():
        for row in data["keys"]:
            share = row["mask_bytes"] / row["hashed_bytes"]
            assert round(share, 2) == 0.40, f"{name}: mask share is {share:.3f}, not 40%"
            assert row["hashed_bytes"] == row["pixel_bytes"] + row["mask_bytes"]
            median = row["seconds"]["median"]
            assert row["throughput_gib_per_second"] == pytest.approx(
                row["hashed_bytes"] / median / (1 << 30), rel=1e-9
            ), f"{name}: throughput must be over everything hashed"
            pixel_only = row["pixel_bytes"] / median / (1 << 30)
            assert row["throughput_gib_per_second"] > 1.3 * pixel_only, (
                f"{name}: quoting pixel_values alone would read "
                f"{pixel_only:.2f} GiB/s against the recorded "
                f"{row['throughput_gib_per_second']:.2f}"
            )


@identity_artifacts_missing
def test_the_processes_agree_to_within_the_spread_the_docstring_quotes() -> None:
    """"Agreeing to within N%" is a claim about the runs, so it gets checked.

    Quoted as agreement between processes rather than as a within-run stdev,
    because those are different dispersions and the sigmoid section already
    documents what happens when they are conflated.

    The threshold is read out of the docstring and pinned to it rather than
    hardcoded here. It used to be ``<= 6.0``, which was true of the first three
    processes and became false when a re-measurement into a second run root added
    three more: the 5-frame key spans 15.41-16.62 ms across those six, i.e. 7.9%.
    Both shortcuts out of that failure are wrong — keeping 6.0 would report a real
    property of new data as a broken test, and simply relaxing the constant to 20
    would bury the disagreement under invented margin. So the sentence states a
    number, this asserts the pooled runs land inside it, *and* that it is the
    smallest whole percent containing them, which is what stops it being padded.
    The spelled-out process count is checked the same way, so a tenth run cannot
    hide inside a sentence about nine.

    A third re-measurement then added three more and the 8% survived them: the
    5-frame key still spans 15.41-16.62 ms across nine, i.e. 7.85%, so ``ceil`` is
    still 8 and only the count moved. That is the assertion earning its keep rather
    than being re-pinned after every run.
    """
    from slowq.vlm import VisionEngine

    doc = squash(VisionEngine._image_cache_key.__doc__ or "")
    match = re.search(
        r"(\w+) independent processes agreeing to within (\d+)%", doc
    )
    assert match, "the key docstring must say how many processes and how close"
    word, quoted_percent = match.group(1), int(match.group(2))
    runs = identity_runs()
    # Spelled out rather than counted from ``len(runs)``: the point is that the prose
    # says a number a reader can check. The table runs past the current count because
    # the provenance gate below forces a re-measurement every time the probe is edited,
    # and each one adds three runs — a dict that stops at today's count would make that
    # routine re-measurement look like a test failure.
    spelled = {
        "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    }
    assert word in spelled, f"unrecognised process count {word!r} in the docstring"
    assert spelled[word] == len(runs), (
        f"the docstring says {word} processes; {len(runs)} are preserved under "
        "artifacts/*/identity/"
    )

    worst = 0.0
    for index, label in enumerate(("5-frame", "9-frame")):
        medians = [d["keys"][index]["seconds"]["median"] for _, d in runs]
        assert len(medians) >= 2
        spread = (max(medians) - min(medians)) / min(medians) * 100
        worst = max(worst, spread)
        assert spread <= quoted_percent, (
            f"{label} key: the processes disagree by {spread:.1f}%, more than the "
            f"{quoted_percent}% the docstring claims for them"
        )
    assert math.ceil(worst) == quoted_percent, (
        f"the preserved runs agree to within {worst:.1f}%; the docstring quotes "
        f"{quoted_percent}%, which is padding rather than the measured spread"
    )


@identity_artifacts_missing
def test_the_identity_artifacts_still_identify_the_code_they_measured(
    identity_probe: Any,
) -> None:
    """The self-referential gate, applied to ``vlm.py`` this time.

    These numbers are quoted in ``VisionEngine``'s own docstrings, so writing them
    down changes the raw hash of the file they describe. ``*_code_sha256`` ignores
    docstrings and line numbers, which is what lets the prose cite the measurement
    without invalidating it — verified rather than assumed: the artifacts in
    ``perf-round1-rerun/identity/`` were produced before the docstrings were
    rewritten, and their raw ``vlm_py_sha256`` no longer matches the file while their
    ``vlm_py_code_sha256`` still does.

    The gate is *not* "every artifact matches the current source". That reading is
    unsatisfiable the moment a probe is edited: every artifact it ever made becomes
    historical at once, and the only ways to green the suite would be to rewrite the
    recorded hashes (fabricating provenance) or to delete the old runs (destroying
    evidence). Both are forbidden, so the artifacts are grouped by the probe that made
    them instead:

    * the newest artifact anywhere must have been made by the current probe — this is
      the half that forces a re-measurement rather than letting stale numbers ride;
    * every artifact in *that* group must also match the current ``vlm.py`` and
      ``weights.py``, which is the original claim;
    * older groups are kept and counted, not judged against source they predate.
    """
    paths = sorted(
        probe_artifacts("identity", SMOLVLM_PATTERN)
        + probe_artifacts("identity", QWEN_LARGE_PATTERN),
        key=lambda p: p.stat().st_mtime,
    )
    runs = [(artifact_label(p), json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    assert runs, "no identity artifacts at all"

    current_probe = identity_probe.code_sha256(Path(identity_probe.__file__))
    newest_name, newest = runs[-1]
    assert newest["identity"]["probe_code_sha256"] == current_probe, (
        f"the newest identity artifact ({newest_name}) was produced by an older "
        "identity_cost.py. Re-run tools/perf/identity_cost.py — the numbers quoted in "
        "vlm.py's docstrings must come from the probe that is in the tree."
    )

    sources = (
        ("vlm_py", REPO_ROOT / "src" / "slowq" / "vlm.py"),
        ("weights_py", REPO_ROOT / "src" / "slowq" / "weights.py"),
    )
    current = {key: identity_probe.code_sha256(path) for key, path in sources}
    historical: dict[str, int] = {}
    for name, data in runs:
        probe_hash = data["identity"]["probe_code_sha256"]
        if probe_hash != current_probe:
            historical[probe_hash] = historical.get(probe_hash, 0) + 1
            continue
        for key, path in sources:
            assert data["identity"][f"{key}_code_sha256"] == current[key], (
                f"{path.name} computes something different from what {name} measured. "
                "Re-run tools/perf/identity_cost.py."
            )

    # The other way to green a gate like this is to delete the artifacts it cannot
    # satisfy. Pinning the original run root's presence makes that a failure rather
    # than a cleanup: those four files are the evidence that the numbers quoted in
    # vlm.py's docstrings once came from an earlier state of the source.
    assert IDENTITY_DIR.is_dir(), f"{IDENTITY_DIR} was removed instead of superseded"
    preserved = sorted(p.name for p in IDENTITY_DIR.glob("*-run*.json"))
    assert len(preserved) >= 4, (
        f"the original identity run root lost artifacts: {preserved}"
    )
    assert historical, (
        "every artifact on disk was made by the current probe, so nothing is being "
        "superseded — either the older runs were deleted, or the probe has never "
        "changed and this grouping is untested"
    )


@identity_artifacts_missing
def test_machine_load_is_recorded_so_the_reading_can_be_judged() -> None:
    """A 0.028 ms measurement with no load average next to it cannot be judged.

    Both samples are required, and the load must have been readable: a probe that
    silently recorded nothing would still pass every numeric assertion above while
    leaving the reader no way to tell a busy machine from a slow one.
    """
    for name, data in identity_runs() + identity_runs(QWEN_LARGE_PATTERN):
        machine = data["machine"]
        for end in ("before", "after"):
            load = machine[end].get("load_average")
            assert load and len(load) == 3, f"{name}: no load average at {end}"
            assert machine[end].get("cpu_count_logical"), f"{name}: no cpu count at {end}"


# --------------------------------------------------------------------------
# run_reference.sh: the summary has to describe the run that produced it
# --------------------------------------------------------------------------
#
# ``validate_reference.py compare`` is read-only over the artifact directory, so
# ``artifacts/validation/<model>/compare.log`` exists only because *some* command
# redirected stdout into it. ``run_reference.sh`` redirected into
# ``$OUT/<model>-compare.log`` instead — and then parsed ``tensors=``,
# ``worst_numerics_rel_l2=`` and ``worst_internal_rel_l2=`` out of
# ``$dir/compare.log``, i.e. out of whatever generation happened to be lying
# there, and computed ``log=IDENTICAL`` by comparing that same file against
# ``$dir/compare-$TAG.log``: a copy the script had made from it a few lines
# earlier. A column that compares a file with its own copy cannot fail.
#
# It did not fail, and it was wrong: for SmolLM2-135M the run's own log differed
# from the stale one (``[compare] took 0.1 s`` → ``0.0 s``) while the summary
# printed ``log=IDENTICAL``. The numbers happened to agree because the numerics
# had not changed — which is precisely why the column cannot be left in place: it
# reports agreement it never measured.
REFERENCE_ROOT = REPO_ROOT / "artifacts" / "perf-round1-rerun"
REFERENCE_DIRS = sorted(
    (d for d in REFERENCE_ROOT.glob("reference-*") if (d / "summary.txt").is_file()),
    key=lambda d: d.stat().st_mtime,
)
reference_runs_missing = pytest.mark.skipif(
    not REFERENCE_DIRS,
    reason="the reference artifacts are gitignored; re-run tools/perf/run_reference.sh",
)
#: A summary written by the fixed driver names the generation it compared against.
#: Its absence is what marks a summary as produced by the self-comparing version.
SUMMARY_TAG_LINE = "# backup tag: "
#: ``[compare] took 0.1 s`` measures the machine at that instant, not the
#: numerics, so it is dropped from both sides before the log column is decided —
#: and the column is named ``log_no_timing`` so the filter is visible in the
#: header a reader actually looks at.
TIMING_LINE = re.compile(r"^\[compare\] took ")
NUMERICS_SECTION = re.compile(r"^\[numerics\]")
ANY_SECTION = re.compile(r"^\[")


def _summary_fields(row: str) -> tuple[str, dict[str, str]]:
    cells = row.split()
    return cells[0], dict(cell.split("=", 1) for cell in cells[1:])


def _worst_numerics(text: str) -> float | None:
    """The largest cross-implementation rel_L2 in the ``[numerics]`` table.

    ``[numerics]`` is a whitespace-aligned table whose 4th field is the rel L2 and
    which carries no ``rel_l2=`` prefix — the reason the old single ``grep -o``
    could not see it and printed the internal number under its name.
    """
    inside = False
    values = []
    for line in text.splitlines():
        if NUMERICS_SECTION.match(line):
            inside = True
            continue
        if ANY_SECTION.match(line):
            inside = False
            continue
        if not inside:
            continue
        cells = line.split()
        if len(cells) >= 4 and re.search(r"e[+-]", cells[3]):
            values.append(float(cells[3]))
    return max(values) if values else None


def _worst_internal(text: str) -> float | None:
    """The largest ``rel_l2=`` anywhere in the log: the chunk-vs-full self-check."""
    values = [float(v) for v in re.findall(r"rel_l2=([0-9.e+-]+)", text)]
    return max(values) if values else None


def _tensors(text: str) -> str:
    found = re.findall(r"^PASS: (\d+) tensors", text, re.M)
    return found[-1] if found else "NONE"


def test_the_reference_summary_is_derived_from_the_log_this_run_produced() -> None:
    """No column of ``summary.txt`` may be read out of a file this run did not write.

    The three numeric columns and the log column all used to come from
    ``$dir/compare.log``, which the driver never writes. Reading it is not a style
    problem: it makes the summary describe the previous generation while the
    ``npz=`` column next to it describes this one, and the two then look like a
    single consistent statement about one run.
    """
    source = (TOOL_DIR / "run_reference.sh").read_text(encoding="utf-8")
    lines = source.splitlines()
    stale = '"$dir/compare.log"'

    readers = [
        line.strip()
        for line in lines
        if stale in line
        and any(op in line for op in ("sed ", "awk ", "grep ", "cmp ", "diff "))
    ]
    assert not readers, (
        "run_reference.sh still reads $dir/compare.log, a file it does not write; "
        f"the summary would describe the previous generation: {readers}"
    )

    assert any(
        line.strip().startswith("cp ") and line.strip().endswith(stale)
        for line in lines
    ), (
        "run_reference.sh never writes $dir/compare.log, so docs/validation.md's "
        "'最近一次 compare 的完整输出' is false for every run this driver produces"
    )

    # The verdict names the log through the ``compare_log`` local, so the local has to
    # be pinned to the path this run writes — exactly once, or a later reassignment
    # could point it back at the stale file and every assertion below would still pass.
    assignments = [line.strip() for line in lines if line.strip().startswith("compare_log=")]
    assert assignments == ['compare_log="$OUT/$m-compare.log"'], (
        f"the compare log local must be assigned once, to this run's own log: {assignments}"
    )

    verdict = [line.strip() for line in lines if "same_log=IDENTICAL" in line]
    assert len(verdict) == 1, f"expected one log-verdict line, got {verdict}"
    assert '"$compare_log"' in verdict[0], (
        "the log column must compare the log *this run* produced, not the one that "
        f"was already on disk: {verdict[0]}"
    )
    assert '"$dir/compare-$TAG.log"' in verdict[0], (
        f"the log column must compare against the named backup generation: {verdict[0]}"
    )

    assert SUMMARY_TAG_LINE in source, (
        "summary.txt never says which generation IDENTICAL refers to, so the claim "
        "cannot be checked by anyone who did not watch the run"
    )
    assert "log_no_timing" in source, (
        "the log column drops the wall-clock line; its name has to say so, or a "
        "reader takes a filtered comparison for a byte-for-byte one"
    )


@reference_runs_missing
def test_a_tagged_reference_summary_reproduces_from_its_own_logs() -> None:
    """Every column of a tagged summary is recomputed here from the files it cites.

    Independent of the shell: the numbers come back out of the run's own compare
    log with Python, and the log column is re-decided by comparing that log against
    the backup generation the summary names. A driver that printed the previous
    generation's numbers, or that compared a file with its own copy, fails here even
    though nothing about its output *looks* wrong.
    """
    tagged = [
        d
        for d in REFERENCE_DIRS
        if SUMMARY_TAG_LINE in (d / "summary.txt").read_text(encoding="utf-8")
    ]
    assert tagged, (
        "no reference run on disk records the generation its summary compared "
        f"against; the newest is {REFERENCE_DIRS[-1].name}, whose `log=` column "
        "compared $dir/compare.log with a copy of itself"
    )
    for out in tagged:
        summary = (out / "summary.txt").read_text(encoding="utf-8")
        tag = next(
            line for line in summary.splitlines() if line.startswith(SUMMARY_TAG_LINE)
        ).split(SUMMARY_TAG_LINE, 1)[1].split()[0]
        rows = [
            line
            for line in summary.splitlines()
            if line and not line.startswith("#") and not line.startswith("OVERALL_RC")
        ]
        assert len(rows) == 8, f"{out.name}: expected 8 model rows, got {len(rows)}"
        for row in rows:
            model, fields = _summary_fields(row)
            fresh = out / f"{model}-compare.log"
            assert fresh.is_file(), f"{out.name}: {fresh.name} is missing"
            text = fresh.read_text(encoding="utf-8")

            assert fields["tensors"] == _tensors(text), (
                f"{out.name}/{model}: tensors= is not what this run's log says"
            )
            for column, recomputed in (
                ("worst_numerics_rel_l2", _worst_numerics(text)),
                ("worst_internal_rel_l2", _worst_internal(text)),
            ):
                if recomputed is None:
                    assert fields[column] == "n/a", f"{out.name}/{model}: {column}"
                else:
                    assert float(fields[column]) == pytest.approx(recomputed, rel=1e-9), (
                        f"{out.name}/{model}: {column}={fields[column]} but this run's "
                        f"log gives {recomputed}"
                    )

            backup = REPO_ROOT / "artifacts" / "validation" / model / f"compare-{tag}.log"
            assert backup.is_file(), f"{out.name}/{model}: no {backup.name} to compare against"
            expected = "IDENTICAL" if [
                line for line in text.splitlines() if not TIMING_LINE.match(line)
            ] == [
                line
                for line in backup.read_text(encoding="utf-8").splitlines()
                if not TIMING_LINE.match(line)
            ] else "DIFFERS"
            assert fields["log_no_timing"] == expected, (
                f"{out.name}/{model}: log_no_timing={fields['log_no_timing']} but "
                f"compare-{tag}.log against this run's log is {expected}"
            )


@reference_runs_missing
def test_a_superseded_reference_summary_is_kept_and_says_why() -> None:
    """An artifact whose column meant something else is labelled, not rewritten.

    The old summaries stay on disk byte for byte — they are the record of what was
    believed at the time, and ``npz=IDENTICAL`` in them is still real evidence. What
    has to be added is the note saying which column was not measuring anything, so a
    reader who opens ``reference-final/`` alone is not told a lie by omission.
    """
    summaries = {
        d: (d / "summary.txt").read_text(encoding="utf-8") for d in REFERENCE_DIRS
    }
    untagged = [d for d, text in summaries.items() if SUMMARY_TAG_LINE not in text]
    if not untagged:
        return
    tagged = [d for d, text in summaries.items() if SUMMARY_TAG_LINE in text]
    assert tagged, "a superseded summary needs something to point the reader at"
    for out in untagged:
        note = out / "LOG-COLUMN-CORRECTION.txt"
        assert note.is_file(), (
            f"{out.name}/summary.txt prints a `log=` column that compared "
            "$dir/compare.log against a copy of itself; without a note beside it the "
            "file still reads as a clean byte-for-byte reproduction"
        )
        body = note.read_text(encoding="utf-8")
        assert "log=" in body, f"{note.name} does not name the column it is correcting"
        assert any(d.name in body for d in tagged), (
            f"{note.name} does not point at the corrected run ({[d.name for d in tagged]})"
        )
        assert "log=IDENTICAL" in summaries[out], (
            f"{out.name}/summary.txt was edited instead of being left as the record "
            "of what was claimed"
        )


# --------------------------------------------------------------------------
# §2.2's byte-identity row: recomputed from the captures, not taken from prose
# --------------------------------------------------------------------------
VALIDATION_ROOT = REPO_ROOT / "artifacts" / "validation"
#: The backup taken before round 1 changed anything: the "before" side of the only
#: byte-level claim §2.2 makes about the reference artifacts.
PRE_ROUND1 = "pre-round1"


def _capture_dirs(root: Path = VALIDATION_ROOT) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if (d / "compare.log").is_file())


CAPTURE_DIRS = _capture_dirs()
validation_captures_missing = pytest.mark.skipif(
    not CAPTURE_DIRS,
    reason="the captured reference artifacts are gitignored; re-run run_reference.sh",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _drop_timing(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not TIMING_LINE.match(line))


def _identity_against(dirs: list[Path], backup: str) -> dict[str, dict[str, bool]]:
    """Per model, whether the current npz / log / timing-filtered log match a backup.

    Three columns and not one, because they answer three different questions: the npz
    is the bit-level claim the report actually rests on, the raw log includes a
    stopwatch the comparator prints about itself, and the filtered log is what is left
    once that stopwatch is set aside.
    """
    found: dict[str, dict[str, bool]] = {}
    for directory in dirs:
        older_npz = directory / f"slowq-{backup}.npz"
        older_log = directory / f"compare-{backup}.log"
        if not (older_npz.is_file() and older_log.is_file()):
            continue
        new = (directory / "compare.log").read_text(encoding="utf-8", errors="replace")
        old = older_log.read_text(encoding="utf-8", errors="replace")
        found[directory.name] = {
            "npz": _sha256(directory / "slowq.npz") == _sha256(older_npz),
            "log": new == old,
            "log_no_timing": _drop_timing(new) == _drop_timing(old),
        }
    return found


def _timing_only_difference(directory: Path, backup: str) -> bool:
    """Every differing line is the comparator's own wall clock — nothing numeric moved."""
    new = (directory / "compare.log").read_text(encoding="utf-8", errors="replace").splitlines()
    old = (directory / f"compare-{backup}.log").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    if len(new) != len(old):
        return False
    return all(
        a == b or (TIMING_LINE.match(a) and TIMING_LINE.match(b)) for a, b in zip(new, old)
    )


@validation_captures_missing
def test_the_pre_round1_identity_row_counts_the_models_it_actually_covers(
    report: Any, tmp_path: Path
) -> None:
    """§2.2's byte-identity row has to say how many of the eight it covers.

    The comparator prints its own wall clock into the log it writes, so
    "``compare.log`` 与 pre-round1 备份逐字节相同" is also a claim about the second
    each run happened to take. Measured: the npz matches for all eight, the raw log for
    seven, and the one that differs differs in exactly that line.

    Seven of eight is not a retraction, it is a count the row was missing — unlike the
    ``log=`` column, this claim is not vacuous, seven models really are identical. But
    an unqualified byte-level statement over a file that contains a stopwatch is the
    prose twin of that column, and §8 纪律 5 requires the filtered dimension to be
    named where the reader looks. So the row carries both counts and names the model
    that splits them.

    The counts are recomputed here rather than read out of the generator, which is the
    point: a number the document and its checker both take from the same function
    cannot disagree with the disk.
    """
    identity = _identity_against(CAPTURE_DIRS, PRE_ROUND1)
    assert identity, f"no `{PRE_ROUND1}` backups beside the captures in {VALIDATION_ROOT}"
    total = len(identity)
    counts = {key: [m for m, v in identity.items() if v[key]] for key in ("npz", "log", "log_no_timing")}
    differing = sorted(set(identity) - set(counts["log"]))
    for model in differing:
        assert _timing_only_difference(VALIDATION_ROOT / model, PRE_ROUND1), (
            f"{model}'s compare.log differs from its {PRE_ROUND1} backup in more than "
            "the wall-clock line, so the filtered count is a different claim rather "
            "than a rounding of the raw one"
        )

    doc = render_report(report, tmp_path, AB_DIRS[-1], tag="pre-round1").doc
    rows = [
        line
        for line in doc.read_text(encoding="utf-8").splitlines()
        if "8 模型官方参考验证" in line
    ]
    assert rows, "§2.2's reference-validation row is gone; this test guards nothing"
    row = " ".join(rows[0].split())

    # Per clause, not per row: "8/8 退出码 0" already contains the string "8/8", so a
    # whole-row search would let the exit-code count stand in for the npz count.
    clauses = [c.strip() for c in row.replace("|", "；").split("；") if c.strip()]

    def clause_naming(needle: str) -> str:
        hits = [c for c in clauses if needle in c]
        assert hits, f"no clause of the row names `{needle}`: {clauses}"
        return hits[0]

    assert f"{len(counts['npz'])}/{total}" in clause_naming("slowq.npz"), (
        f"the npz clause must state the count it measured ({len(counts['npz'])}/{total})"
    )
    log_clause = clause_naming("compare.log")
    assert f"{len(counts['log'])}/{total}" in log_clause, (
        f"the compare.log clause must state the raw count it measured "
        f"({len(counts['log'])}/{total}), not leave 'identical' unquantified"
    )
    assert f"{len(counts['log_no_timing'])}/{total}" in log_clause, (
        "the same clause must also give the count once the comparator's own wall clock "
        f"is dropped ({len(counts['log_no_timing'])}/{total}): that is the dimension the "
        "filter removed, and §8 纪律 5 requires it to be named where the reader looks"
    )
    for model in differing:
        assert model in log_clause, (
            f"{model} is what splits the raw count from the filtered one; a clause that "
            "prints both numbers without naming it sends the reader looking"
        )
    if differing:
        assert "备份**逐字节相同**" not in row, (
            f"unqualified byte-identity: {len(differing)} of {total} models differ by "
            "the comparator's own `[compare] took` line"
        )


# --------------------------------------------------------------------------- #
# driver exit codes: a run that failed must not report success
# --------------------------------------------------------------------------- #

#: Stand-in for ``.venv/bin/python``. Picks its exit code from ``$FAKE_RC_FILE``,
#: a file of ``<key>=<rc>`` lines keyed by ``$SLOWQ_MODEL_DIR`` when that is set
#: and by the first argument otherwise, so one fake interpreter serves both
#: ``run_integration.sh`` (which sets the model dir per pytest process) and
#: ``run_recollection.sh`` (which shells out to ``source_identity.py``).
FAKE_PYTHON = """\
#!/bin/bash
case "${1:-}" in
  tools/perf/out_guard.py) exec "@REAL_PY@" "$@" ;;
esac
key="${SLOWQ_MODEL_DIR:-}"
[ -n "$key" ] || key="${1:-}"
rc=0
if [ -n "${FAKE_RC_FILE:-}" ] && [ -f "${FAKE_RC_FILE}" ]; then
  while IFS='=' read -r k v; do
    [ "$k" = "$key" ] && rc="$v"
  done < "${FAKE_RC_FILE}"
fi
echo "fake python: ${key} -> rc=${rc}"
exit "${rc}"
"""

#: Stand-in for one chained measurement driver. Keys ``$FAKE_RC_FILE`` by its own
#: basename, so a test can fail exactly the step it means to fail.
FAKE_STEP = """\
#!/bin/bash
self="$(basename "$0" .sh)"
rc=0
if [ -n "${FAKE_RC_FILE:-}" ] && [ -f "${FAKE_RC_FILE}" ]; then
  while IFS='=' read -r k v; do
    [ "$k" = "$self" ] && rc="$v"
  done < "${FAKE_RC_FILE}"
fi
echo "fake ${self} -> rc=${rc}"
exit "${rc}"
"""


def fake_driver_root(tmp_path: Path, name: str) -> Path:
    """A throwaway repo holding a copy of one driver and the fakes it invokes.

    The drivers ``cd`` to ``$(dirname $0)/../..`` and then run ``.venv/bin/python``
    and sibling scripts by relative path, so a copy under ``<root>/tools/perf``
    behaves like the real thing while touching nothing real: no model directory is
    read, no pytest process starts, and ``artifacts/`` in the working tree is never
    named. Exit codes are the whole subject of these tests, and eight real models
    are not a way to check one.
    """
    root = tmp_path / f"driver-{name}"
    (root / "tools" / "perf").mkdir(parents=True)
    shutil.copy2(TOOL_DIR / f"{name}.sh", root / "tools" / "perf" / f"{name}.sh")
    shutil.copy2(TOOL_DIR / "out_guard.py", root / "tools" / "perf" / "out_guard.py")
    # Sourced by every driver, so it has to travel with the copy: the shell half
    # resolves the run root and the Python half refuses the write.
    shutil.copy2(TOOL_DIR / "out_guard.sh", root / "tools" / "perf" / "out_guard.sh")
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(
        FAKE_PYTHON.replace("@REAL_PY@", str(REPO_ROOT / ".venv" / "bin" / "python")),
        encoding="utf-8",
    )
    (venv / "python").chmod(0o755)
    for step in (
        "run_round1",
        "run_round1_longctx",
        "run_noise_floor",
        "run_projection",
    ):
        path = root / "tools" / "perf" / f"{step}.sh"
        path.write_text(FAKE_STEP, encoding="utf-8")
        path.chmod(0o755)
    return root


def run_driver(
    root: Path,
    name: str,
    args: list[str],
    rc_map: dict[str, int],
    extra_env: dict[str, str] | None = None,
) -> Any:
    """Run one copied driver with ``$FAKE_RC_FILE`` set from ``rc_map``.

    ``extra_env`` is how a test injects a conflict deterministically: the fakes read
    their instructions from the environment, so the injection happens at a named point
    inside the driver's own execution rather than in a race against it.
    """
    control = root / "rc.txt"
    control.write_text(
        "".join(f"{key}={value}\n" for key, value in rc_map.items()), encoding="utf-8"
    )
    return subprocess.run(
        ["bash", f"tools/perf/{name}.sh", *args],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "FAKE_RC_FILE": str(control), **(extra_env or {})},
    )


def test_run_integration_exits_zero_only_when_every_model_passed(tmp_path: Path) -> None:
    root = fake_driver_root(tmp_path, "run_integration")
    done = run_driver(root, "run_integration", ["logs"], {})
    assert done.returncode == 0, done.stderr
    assert "OVERALL_RC=0" in done.stdout


def test_run_integration_exits_nonzero_when_one_model_failed(tmp_path: Path) -> None:
    """The confirmed defect: the log said ``OVERALL_RC=1`` and the shell said 0.

    A driver that prints its own failure and then exits successfully cannot fail a
    pipeline, a ``&&`` chain or a human reading ``$?``. The status line was the only
    thing carrying the verdict, and nothing downstream reads a status line.
    """
    root = fake_driver_root(tmp_path, "run_integration")
    done = run_driver(
        root, "run_integration", ["logs"], {"models/Qwen3-0.6B-slowq": 1}
    )
    assert "OVERALL_RC=1" in done.stdout
    assert done.returncode != 0, (
        "run_integration.sh printed OVERALL_RC=1 and still exited 0"
    )


def test_run_integration_exits_nonzero_when_several_models_failed(tmp_path: Path) -> None:
    root = fake_driver_root(tmp_path, "run_integration")
    done = run_driver(
        root,
        "run_integration",
        ["logs"],
        {"models/Qwen3-0.6B-slowq": 1, "models/Qwen3-1.7B-slowq": 2},
    )
    assert done.returncode != 0, done.stderr


def test_run_integration_exits_nonzero_when_pytest_cannot_be_executed(tmp_path: Path) -> None:
    """A missing interpreter is a failure, not an empty run that passed."""
    root = fake_driver_root(tmp_path, "run_integration")
    (root / ".venv" / "bin" / "python").unlink()
    done = run_driver(root, "run_integration", ["logs"], {})
    assert done.returncode != 0, (
        "no interpreter to run, and the driver still reported success"
    )
    assert "OVERALL_RC=0" not in done.stdout


def test_run_integration_keeps_going_after_a_failure(tmp_path: Path) -> None:
    """Collecting the rest is the design; only the exit code was wrong.

    Stopping at the first failing model would trade one defect for another: the
    point of running all eight is to learn in one pass which of them fail, and a
    driver that aborts early makes the operator re-run it seven more times.
    """
    root = fake_driver_root(tmp_path, "run_integration")
    done = run_driver(
        root, "run_integration", ["logs"], {"models/Qwen3-0.6B-slowq": 1}
    )
    logs = sorted(p.name for p in (root / "logs").glob("*.log"))
    assert len(logs) == 8, f"the driver stopped early: {logs}"
    assert done.returncode != 0


def test_run_recollection_exits_nonzero_when_a_step_failed(tmp_path: Path) -> None:
    """The second confirmed defect: every step returned 7 and the driver returned 0."""
    root = fake_driver_root(tmp_path, "run_recollection")
    done = run_driver(
        root,
        "run_recollection",
        ["out"],
        {
            "run_round1": 7,
            "run_round1_longctx": 7,
            "run_noise_floor": 7,
            "run_projection": 7,
        },
    )
    assert done.returncode != 0, (
        "run_recollection.sh's four steps all failed and it still exited 0"
    )


def test_run_recollection_exits_nonzero_when_one_step_failed(tmp_path: Path) -> None:
    root = fake_driver_root(tmp_path, "run_recollection")
    done = run_driver(root, "run_recollection", ["out"], {"run_noise_floor": 3})
    assert done.returncode != 0, done.stderr
    assert "rc=3" in (root / "out" / "recollection-status.txt").read_text(
        encoding="utf-8"
    )


def test_run_recollection_exits_zero_when_every_step_passed(tmp_path: Path) -> None:
    root = fake_driver_root(tmp_path, "run_recollection")
    done = run_driver(root, "run_recollection", ["out"], {})
    assert done.returncode == 0, done.stderr
    status = (root / "out" / "recollection-status.txt").read_text(encoding="utf-8")
    assert "run_round1 rc=0" in status
    assert "OVERALL_RC=0" in status


def test_run_recollection_keeps_going_after_a_step_failed(tmp_path: Path) -> None:
    """``;`` between the steps is deliberate and must survive the exit-code fix.

    The header explains why: partial new evidence with an honest status file beats
    no evidence and a guess about where it stopped. Accumulating the failure and
    continuing are not in tension, and the test pins both halves.
    """
    root = fake_driver_root(tmp_path, "run_recollection")
    done = run_driver(root, "run_recollection", ["out"], {"run_round1": 1})
    status = (root / "out" / "recollection-status.txt").read_text(encoding="utf-8")
    for step in ("run_round1", "run_round1_longctx", "run_noise_floor", "run_projection"):
        assert f"{step} rc=" in status, f"{step} never ran: {status}"
    assert done.returncode != 0


def test_a_failed_recollection_does_not_announce_itself_as_done(tmp_path: Path) -> None:
    """The closing line is a status too, and it must not contradict the exit code.

    "re-collection done" printed over four failed steps is how a run gets filed as
    complete by whoever reads the tail of the log rather than ``$?``.
    """
    root = fake_driver_root(tmp_path, "run_recollection")
    done = run_driver(root, "run_recollection", ["out"], {"run_round1": 1})
    assert done.returncode != 0
    assert "re-collection done" not in done.stderr, (
        "a failed run still announced itself as done"
    )
    assert "OVERALL_RC=1" in (root / "out" / "recollection-status.txt").read_text(
        encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# overwrite protection: a reproduction run must not replace published evidence
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def guard() -> Any:
    return load_tool("out_guard")


#: Stand-in interpreter. Delegates ``out_guard.py`` to the real interpreter running
#: the *copied* script, so the guard under test resolves its repo root to the
#: throwaway tree rather than to the working one; ``baseline_tree.py`` gets a
#: constant answer; anything else is treated as a measurement program and writes
#: whatever ``--out`` names, with ``>`` — the silent overwrite under test.
FAKE_PYTHON_WITH_GUARD = """\
#!/bin/bash
REAL_PY="@REAL_PY@"
case "${1:-}" in
  tools/perf/out_guard.py) exec "$REAL_PY" "$@" ;;
  tools/perf/baseline_tree.py) echo "baseline-src"; exit 0 ;;
esac
out=""
prev=""
for a in "$@"; do
  [ "$prev" = "--out" ] && out="$a"
  prev="$a"
done
if [ -n "$out" ]; then
  mkdir -p "$(dirname "$out")"
  printf 'fake measurement\\n' > "$out"
fi
echo "fake measurement -> ${out:-<none>}"
exit 0
"""


def fake_repo(tmp_path: Path, tag: str, drivers: Sequence[str]) -> Path:
    """A throwaway repo whose ``tools/perf`` holds real copies of ``drivers``.

    Real copies, not stubs: the subject is what the drivers do before they measure,
    and a stub of the driver under test would be a test of the stub. The measurement
    programs are the only thing faked — via ``.venv/bin/python`` — because the real
    ones need a model and an hour.
    """
    root = tmp_path / f"repo-{tag}"
    (root / "tools" / "perf").mkdir(parents=True)
    for name in ("out_guard.py", "out_guard.sh", *drivers):
        shutil.copy2(TOOL_DIR / name, root / "tools" / "perf" / name)
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    interpreter = venv / "python"
    interpreter.write_text(
        FAKE_PYTHON_WITH_GUARD.replace(
            "@REAL_PY@", str(REPO_ROOT / ".venv" / "bin" / "python")
        ),
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    return root


def run_in(root: Path, command: list[str]) -> Any:
    return subprocess.run(command, cwd=root, capture_output=True, text=True)


ALL_DRIVERS = (
    "run_round1.sh",
    "run_round1_longctx.sh",
    "run_noise_floor.sh",
    "run_projection.sh",
    "run_sweep.sh",
    "run_ab.sh",
    "run_integration.sh",
    "run_reference.sh",
    "run_recollection.sh",
)


def test_reserve_creates_a_new_directory(guard: Any, tmp_path: Path) -> None:
    target = tmp_path / "fresh" / "ab"
    created = guard.reserve([target])
    assert created == [guard.canonical(target)]
    assert target.is_dir()


def test_reserve_refuses_an_existing_directory_and_changes_nothing(
    guard: Any, tmp_path: Path
) -> None:
    """The whole requirement in one test: nonzero exit, bytes untouched."""
    victim = tmp_path / "ab"
    victim.mkdir()
    (victim / "model-base.json").write_text("published evidence\n", encoding="utf-8")
    sibling = tmp_path / "sweep-opt"

    with pytest.raises(SystemExit) as raised:
        guard.reserve([sibling, victim])

    assert raised.value.code == 1
    assert not sibling.exists(), (
        "a conflict in one path must not leave the others half-created"
    )
    assert (victim / "model-base.json").read_text(encoding="utf-8") == (
        "published evidence\n"
    )


def test_reserve_accepts_an_existing_empty_directory(guard: Any, tmp_path: Path) -> None:
    """An orchestrator hands its own fresh root to a sub-driver; that is not a conflict.

    The alternative reading — refuse anything that exists — would make
    ``run_recollection.sh`` unable to call ``run_round1.sh`` at all, and the fix
    would be a shell that lies about which root it created.
    """
    empty = tmp_path / "root"
    empty.mkdir()
    assert guard.reserve([empty]) == [guard.canonical(empty)]


def test_reserve_refuses_a_trailing_slash_on_a_published_root(
    guard: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confirmed bypass: the old check compared ``$OUT_ROOT`` to a bare string.

    ``artifacts/perf-round1/`` is not ``artifacts/perf-round1``, so the comparison
    passed and the run wrote into the published evidence. Normalising first is the
    fix, and the trailing slash is the case that proves it.
    """
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    published = tmp_path / "artifacts" / "perf-round1" / "ab"
    published.mkdir(parents=True)
    (published / "Qwen3-0.6B-slowq-base.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        guard.reserve([f"{tmp_path}/artifacts/perf-round1/ab/"])
    assert raised.value.code == 1
    assert (published / "Qwen3-0.6B-slowq-base.json").read_text(encoding="utf-8") == (
        "{}\n"
    )


def test_reserve_refuses_a_relative_path_into_a_published_root(
    guard: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    published = tmp_path / "artifacts" / "perf-round1-rerun" / "noise-floor"
    published.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        guard.reserve(["artifacts/perf-round1-rerun/noise-floor"])


def test_reserve_refuses_dotdot_that_lands_in_a_published_root(
    guard: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    published = tmp_path / "artifacts" / "perf-round1" / "ab"
    published.mkdir(parents=True)
    scratch = tmp_path / "artifacts" / "scratch"
    scratch.mkdir(parents=True)
    with pytest.raises(SystemExit):
        guard.reserve([str(scratch / ".." / "perf-round1" / "ab")])


def test_reserve_refuses_a_symlink_whose_target_is_published(
    guard: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``realpath`` is what catches this: the name typed is outside, the bytes are inside."""
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    published = tmp_path / "artifacts" / "perf-round1" / "ab"
    published.mkdir(parents=True)
    (published / "keep.json").write_text("original\n", encoding="utf-8")
    link = tmp_path / "innocent-looking"
    link.symlink_to(tmp_path / "artifacts" / "perf-round1")

    with pytest.raises(SystemExit):
        guard.reserve([str(link / "ab")])
    assert (published / "keep.json").read_text(encoding="utf-8") == "original\n"


def test_reserve_refuses_a_dangling_symlink(guard: Any, tmp_path: Path) -> None:
    """``realpath`` reports the target, which does not exist — ``lexists`` sees the link."""
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "nowhere" / "ab")
    with pytest.raises(SystemExit):
        guard.reserve([str(link)])


def test_reserve_reports_the_same_path_twice_as_a_conflict(
    guard: Any, tmp_path: Path
) -> None:
    target = tmp_path / "ab"
    with pytest.raises(SystemExit):
        guard.reserve([str(target), str(target) + "/"])


def test_write_json_refuses_to_replace_a_measurement(guard: Any, tmp_path: Path) -> None:
    """The second layer: the pre-flight can pass and the file can still be there."""
    out = tmp_path / "model-base.json"
    out.write_text('{"label": "first run"}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        guard.write_json(out, {"label": "second run"})
    assert raised.value.code == 1
    assert json.loads(out.read_text(encoding="utf-8")) == {"label": "first run"}


def test_write_json_creates_a_new_measurement(guard: Any, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "model-base.json"
    written = guard.write_json(out, {"label": "opt"})
    assert written == guard.canonical(out)
    assert json.loads(out.read_text(encoding="utf-8")) == {"label": "opt"}


def test_write_json_refuses_a_published_directory(
    guard: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    published = tmp_path / "artifacts" / "perf-round1-rerun" / "projection"
    published.mkdir(parents=True)
    with pytest.raises(SystemExit):
        guard.write_json(published / "Qwen3-0.6B-slowq.json", {"a": 1})
    assert list(published.iterdir()) == []


def test_fresh_root_is_unique_and_printable(guard: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    first = guard.fresh_root("artifacts/perf-run")
    second = guard.fresh_root("artifacts/perf-run")
    assert first != second
    assert first.is_dir() and second.is_dir()
    assert guard.display(first) == f"artifacts/{first.name}"


def test_fresh_root_never_lands_in_published_evidence(
    guard: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    for name in guard.PUBLISHED_ROOTS:
        assert guard.published_conflict(guard.fresh_root("artifacts/perf-run")) is None
        assert guard.published_conflict(tmp_path / name / "ab") == name


def test_a_default_run_creates_a_new_directory_and_says_where(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No argument must not mean "the directory the published numbers are in"."""
    root = fake_repo(tmp_path, "default", ALL_DRIVERS)
    done = run_in(root, ["bash", "tools/perf/run_round1.sh"])
    assert done.returncode == 0, done.stderr
    created = sorted(p.name for p in (root / "artifacts").iterdir())
    assert len(created) == 1, created
    assert created[0] not in ("perf-round1", "perf-round1-rerun")
    assert created[0] in done.stderr, (
        "the driver created a run directory and did not tell the operator where"
    )
    assert (root / "artifacts" / created[0] / "ab").is_dir()


def test_running_the_same_driver_twice_refuses_the_second_run(tmp_path: Path) -> None:
    """The confirmed defect, end to end: the default was a directory already full of evidence."""
    root = fake_repo(tmp_path, "twice", ALL_DRIVERS)
    out = "artifacts/perf-round1-rerun"
    (root / out / "ab").mkdir(parents=True)
    victim = root / out / "ab" / "Qwen3-0.6B-slowq-base.json"
    victim.write_text('{"label": "published"}\n', encoding="utf-8")

    done = run_in(root, ["bash", "tools/perf/run_round1.sh", out])
    assert done.returncode != 0, done.stderr
    assert victim.read_text(encoding="utf-8") == '{"label": "published"}\n'


def test_an_explicit_conflicting_path_is_refused_before_measuring(tmp_path: Path) -> None:
    """Conflict detection has to be cheap and early, not a surprise after an hour."""
    root = fake_repo(tmp_path, "conflict", ALL_DRIVERS)
    out = root / "artifacts" / "scratch"
    (out / "noise-floor").mkdir(parents=True)
    (out / "noise-floor" / "old.json").write_text("kept\n", encoding="utf-8")

    done = run_in(
        root, ["bash", "tools/perf/run_noise_floor.sh", "artifacts/scratch", "2"]
    )
    assert done.returncode != 0, done.stderr
    assert (out / "noise-floor" / "old.json").read_text(encoding="utf-8") == "kept\n"
    assert not list((out / "noise-floor").glob("*-r1-*.json")), "measured anyway"


def test_a_trailing_slash_does_not_walk_past_the_published_check(tmp_path: Path) -> None:
    root = fake_repo(tmp_path, "slash", ALL_DRIVERS)
    published = root / "artifacts" / "perf-round1" / "ab"
    published.mkdir(parents=True)
    (published / "SmolLM2-135M-Instruct-slowq-base.json").write_text(
        '{"label": "published"}\n', encoding="utf-8"
    )

    done = run_in(root, ["bash", "tools/perf/run_round1.sh", "artifacts/perf-round1/"])
    assert done.returncode != 0, done.stderr
    assert (published / "SmolLM2-135M-Instruct-slowq-base.json").read_text(
        encoding="utf-8"
    ) == '{"label": "published"}\n'


def test_a_symlink_to_the_published_evidence_is_refused(tmp_path: Path) -> None:
    root = fake_repo(tmp_path, "symlink", ALL_DRIVERS)
    published = root / "artifacts" / "perf-round1" / "ab"
    published.mkdir(parents=True)
    (published / "keep.json").write_text("original\n", encoding="utf-8")
    (root / "rerun-alias").symlink_to(root / "artifacts" / "perf-round1")

    done = run_in(root, ["bash", "tools/perf/run_round1.sh", "rerun-alias"])
    assert done.returncode != 0, done.stderr
    assert (published / "keep.json").read_text(encoding="utf-8") == "original\n"


def test_an_absolute_path_is_normalised_the_same_way(tmp_path: Path) -> None:
    root = fake_repo(tmp_path, "absolute", ALL_DRIVERS)
    published = root / "artifacts" / "perf-round1" / "ab"
    published.mkdir(parents=True)
    done = run_in(
        root, ["bash", "tools/perf/run_round1.sh", str(root / "artifacts/perf-round1")]
    )
    assert done.returncode != 0, done.stderr


def test_a_sub_driver_shares_the_orchestrators_root_without_being_refused(
    tmp_path: Path,
) -> None:
    """The trap in the other direction: the root existing is normal, the product is not.

    ``run_recollection.sh`` creates the root, writes its own status and environment
    files into it, and then calls ``run_round1.sh`` with that same root. A guard that
    refused "the root already exists" would break the only documented way to run a
    whole re-collection, and the tempting workaround is to relax the guard.
    """
    root = fake_repo(tmp_path, "orchestrator", ALL_DRIVERS)
    done = run_in(root, ["bash", "tools/perf/run_recollection.sh"])
    assert done.returncode == 0, done.stderr
    created = sorted(p.name for p in (root / "artifacts").iterdir())
    assert len(created) == 1, created
    run_root = root / "artifacts" / created[0]
    for product in ("ab", "sweep-opt", "vision", "documents", "ab-longctx", "noise-floor", "projection"):
        assert (run_root / product).is_dir(), f"{product} was never produced"
    assert (run_root / "recollection-status.txt").is_file()


def test_run_recollection_refuses_before_step_one_when_a_later_step_would_conflict(
    tmp_path: Path,
) -> None:
    """Four hours of measurement must not end in "step 4 had nowhere to write".

    ``run_projection.sh`` is the last step and its directory is the one a second
    attempt is most likely to find occupied, so it is the case that matters.
    """
    root = fake_repo(tmp_path, "preflight", ALL_DRIVERS)
    out = root / "artifacts" / "scratch"
    (out / "projection").mkdir(parents=True)
    (out / "projection" / "old.json").write_text("kept\n", encoding="utf-8")

    done = run_in(root, ["bash", "tools/perf/run_recollection.sh", "artifacts/scratch"])
    assert done.returncode != 0, done.stderr
    assert not (out / "run_round1.log").exists(), (
        "step 1 ran even though step 4 had nowhere to write"
    )
    assert (out / "projection" / "old.json").read_text(encoding="utf-8") == "kept\n"


def test_run_projection_no_longer_moves_the_previous_attempt_aside(tmp_path: Path) -> None:
    """Archiving is still overwriting: the command that should have conflicted succeeded.

    The bytes survived under another name, but the run that was supposed to be
    refused reported success, and the directory the operator pointed at no longer
    holds what they put there.
    """
    source = (TOOL_DIR / "run_projection.sh").read_text(encoding="utf-8")
    assert "mv " not in source, "run_projection.sh still moves old evidence aside"

    root = fake_repo(tmp_path, "projection", ALL_DRIVERS)
    out = root / "artifacts" / "scratch"
    (out / "ab").mkdir(parents=True)
    (out / "projection").mkdir(parents=True)
    (out / "projection" / "Qwen3-0.6B-slowq.json").write_text("first attempt\n", encoding="utf-8")

    done = run_in(root, ["bash", "tools/perf/run_projection.sh", "artifacts/scratch"])
    assert done.returncode != 0, done.stderr
    assert (out / "projection" / "Qwen3-0.6B-slowq.json").read_text(encoding="utf-8") == (
        "first attempt\n"
    )
    assert sorted(p.name for p in out.iterdir()) == ["ab", "projection"], (
        "the previous attempt was archived rather than left where it was"
    )


# --------------------------------------------------------------------------- #
# round 3, problem 1a: a path that must hold a *file* is not free when a directory
#                      holds it, and a startup check is not a write
# --------------------------------------------------------------------------- #


def test_reserve_file_refuses_an_empty_directory_at_a_file_path(
    guard: Any, tmp_path: Path
) -> None:
    """The confirmed bypass behind reproduction A: the directory rule answered for files.

    ``run_recollection.sh`` pre-checked ``recollection-status.txt`` with the same
    ``_conflict`` it used for its product *directories*, and that function accepted any
    empty directory — a rule that is right for ``ab/`` and wrong for a status file. So
    a directory sitting on the status path passed the check, all four measurement steps
    ran, every status write failed with ``Is a directory``, and the driver still exited
    0 and printed "done".

    ``sibling`` is listed first on purpose: ``reserve`` promises to check every path
    before creating any, and an ``IsADirectoryError`` raised during the *creation* loop
    breaks that promise with the first file already on disk.
    """
    occupied = tmp_path / "recollection-status.txt"
    occupied.mkdir()
    sibling = tmp_path / "run_round1.log"

    with pytest.raises(SystemExit) as raised:
        guard.reserve([sibling, occupied], files=True)

    assert raised.value.code == 1
    assert not sibling.exists(), (
        "a conflict on the second path left the first one created"
    )
    assert occupied.is_dir() and not any(occupied.iterdir())


def test_reserve_file_still_creates_the_files_it_accepted(
    guard: Any, tmp_path: Path
) -> None:
    """The other direction: accepting an empty directory must not become refusing a free path."""
    first = tmp_path / "status.txt"
    second = tmp_path / "nested" / "run_round1.log"
    created = guard.reserve([first, second], files=True)
    assert created == [guard.canonical(first), guard.canonical(second)]
    assert first.is_file() and second.is_file()


def test_check_file_refuses_a_directory_where_a_file_must_go(
    guard: Any, tmp_path: Path
) -> None:
    """The rule has to be reachable from bash 3.2, which is what the drivers speak."""
    occupied = tmp_path / "recollection-status.txt"
    occupied.mkdir()
    free = tmp_path / "run_round1.log"
    with pytest.raises(SystemExit) as raised:
        guard.main(["check-file", str(occupied), str(free)])
    assert raised.value.code == 1
    assert not free.exists(), "check must create nothing"


def test_check_still_accepts_an_empty_directory_for_a_directory_path(
    guard: Any, tmp_path: Path
) -> None:
    """``check`` and ``check-file`` differ, and the difference is the whole fix.

    An orchestrator hands its own fresh root to a sub-driver, so "this directory
    exists and is empty" stays acceptable for a *directory* product; tightening that
    would break the only documented way to run a whole re-collection.
    """
    free = tmp_path / "ab"
    free.mkdir()
    assert guard.main(["check", str(free)]) == 0
    with pytest.raises(SystemExit):
        guard.main(["check-file", str(free)])


def test_preflight_accepts_a_free_path_and_creates_nothing(guard: Any, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "Qwen3-0.6B-slowq-opt.json"
    assert guard.preflight(out) == guard.canonical(out)
    assert not out.exists() and not out.parent.exists(), (
        "a preflight that creates the file it checked turns the write-time guard into "
        "a formality: open(x, 'x') can no longer fail on it"
    )


def test_preflight_of_no_output_is_not_a_conflict(guard: Any) -> None:
    """``--out`` is optional in all six measurement programs; omitting it stays legal."""
    assert guard.preflight(None) is None


def test_preflight_refuses_an_existing_file(guard: Any, tmp_path: Path) -> None:
    out = tmp_path / "opt.json"
    out.write_text('{"label": "first run"}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        guard.preflight(out)
    assert raised.value.code == 1
    assert json.loads(out.read_text(encoding="utf-8")) == {"label": "first run"}


def test_preflight_refuses_an_existing_directory(guard: Any, tmp_path: Path) -> None:
    out = tmp_path / "opt.json"
    out.mkdir()
    with pytest.raises(SystemExit):
        guard.preflight(out)
    assert out.is_dir()


def test_preflight_refuses_a_parent_that_is_not_a_directory(guard: Any, tmp_path: Path) -> None:
    """``--out a.json/b.json`` cannot be created, and finding that out after an hour is the defect."""
    blocker = tmp_path / "opt.json"
    blocker.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        guard.preflight(blocker / "model-base.json")
    assert raised.value.code == 1
    assert blocker.read_text(encoding="utf-8") == "not a directory\n"


def test_preflight_refuses_a_symlink_and_a_dangling_symlink(guard: Any, tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("kept\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(SystemExit):
        guard.preflight(link)
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "nowhere.json")
    with pytest.raises(SystemExit):
        guard.preflight(dangling)
    assert real.read_text(encoding="utf-8") == "kept\n"


def test_preflight_refuses_a_published_location(
    guard: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    published = tmp_path / "artifacts" / "perf-round1" / "ab" / "Qwen3-0.6B-slowq-opt.json"
    published.parent.mkdir(parents=True)
    with pytest.raises(SystemExit):
        guard.preflight(published)
    assert not published.exists()


def test_preflight_does_not_replace_the_write_it_precedes(
    guard: Any, tmp_path: Path
) -> None:
    """Both layers, in the order the reviewer asked for: preflight *and* exclusive write.

    A preflight that made ``write_json`` redundant would be a check performed earlier
    standing in for a check performed at the moment of writing — which is exactly the
    shape that let a conflict slip through between the two.
    """
    out = tmp_path / "model-base.json"
    guard.preflight(out)
    out.write_text('{"label": "created in between"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        guard.write_json(out, {"label": "mine"})
    assert json.loads(out.read_text(encoding="utf-8")) == {"label": "created in between"}


# --------------------------------------------------------------------------- #
# round 3, problem 1b: run_recollection.sh — status, environment and step logs
# --------------------------------------------------------------------------- #


def test_run_recollection_refuses_a_directory_sitting_on_its_status_file(
    tmp_path: Path,
) -> None:
    """Reproduction A, end to end: the steps must not start at all."""
    root = fake_driver_root(tmp_path, "run_recollection")
    out = root / "out"
    (out / "recollection-status.txt").mkdir(parents=True)

    done = run_driver(root, "run_recollection", ["out"], {})

    assert done.returncode != 0, done.stderr
    assert "re-collection done" not in done.stderr
    assert not list(out.glob("*.log")), (
        "the four measurement steps started even though the status file could never "
        "be written — this is the run that costs hours and produces no verdict"
    )
    assert not (out / "recollection-environment.txt").exists()
    assert (out / "recollection-status.txt").is_dir()


def test_run_recollection_refuses_a_directory_sitting_on_a_step_log(tmp_path: Path) -> None:
    root = fake_driver_root(tmp_path, "run_recollection")
    out = root / "out"
    (out / "run_noise_floor.log").mkdir(parents=True)

    done = run_driver(root, "run_recollection", ["out"], {})

    assert done.returncode != 0, done.stderr
    assert "re-collection done" not in done.stderr
    # The three siblings the test did not plant: their absence is what proves the driver
    # stopped at the pre-check rather than running the first two steps and failing on the
    # third. The planted one is a directory this test created, so a bare `glob("*.log")`
    # finds it and would make this assertion unsatisfiable.
    assert not (out / "run_round1.log").exists()
    assert not (out / "run_round1_longctx.log").exists()
    assert not (out / "run_projection.log").exists()
    assert (out / "run_noise_floor.log").is_dir()
    assert not list((out / "run_noise_floor.log").iterdir())


#: Stand-in step that creates the *next* step's log before the driver reaches it. The
#: pre-check cannot see this conflict: it does not exist yet when the driver starts.
#: Stopping it is the write's job, which in bash 3.2 means ``set -C``.
FAKE_STEP_POISON_NEXT = """\
#!/bin/bash
self="$(basename "$0" .sh)"
rc=0
if [ -n "${FAKE_RC_FILE:-}" ] && [ -f "${FAKE_RC_FILE}" ]; then
  while IFS='=' read -r k v; do
    [ "$k" = "$self" ] && rc="$v"
  done < "${FAKE_RC_FILE}"
fi
case "$self" in
  run_round1) printf 'pre-existing evidence\\n' > "$1/run_round1_longctx.log" ;;
esac
echo "fake ${self} -> rc=${rc}"
exit "${rc}"
"""


def install_steps(root: Path, script: str) -> None:
    for step in ("run_round1", "run_round1_longctx", "run_noise_floor", "run_projection"):
        path = root / "tools" / "perf" / f"{step}.sh"
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)


def test_a_log_conflict_appearing_after_the_precheck_keeps_its_bytes(
    tmp_path: Path,
) -> None:
    """Reproduction B: ``>`` truncated a log the driver did not create.

    The JSON products were already protected by ``open(..., "x")`` inside
    ``out_guard.write_json``; the logs were protected only by a check that had already
    run. Anything creating one of those paths in between — a second invocation pointed
    at the same root, an operator's own note — was silently replaced.
    """
    root = fake_driver_root(tmp_path, "run_recollection")
    install_steps(root, FAKE_STEP_POISON_NEXT)

    done = run_driver(root, "run_recollection", ["out"], {})
    victim = root / "out" / "run_round1_longctx.log"

    assert done.returncode != 0, done.stderr
    assert victim.read_text(encoding="utf-8") == "pre-existing evidence\n", (
        "the driver truncated a log it did not create"
    )
    assert "re-collection done" not in done.stderr


#: Stand-in step that seals the status file after every measurement has succeeded, so
#: only the evidence write fails. ``$1`` is the out-root: that is what the driver
#: passes each step.
FAKE_STEP_SEAL_STATUS = """\
#!/bin/bash
self="$(basename "$0" .sh)"
[ "$self" = "run_projection" ] && chmod 444 "$1/recollection-status.txt"
echo "fake ${self} -> rc=0"
exit 0
"""


def test_a_run_whose_status_write_failed_is_not_reported_as_complete(tmp_path: Path) -> None:
    """All four steps rc=0, the status record unwritable: the run is still incomplete.

    A status file is the thing read weeks later, and a run that could not finish
    writing it must not be findable as a finished one — neither through ``$?`` nor
    through the closing message. The partial status that *did* get written stays, and
    must not claim ``OVERALL_RC=0``.
    """
    root = fake_driver_root(tmp_path, "run_recollection")
    install_steps(root, FAKE_STEP_SEAL_STATUS)
    status = root / "out" / "recollection-status.txt"

    done = run_driver(root, "run_recollection", ["out"], {})
    try:
        assert done.returncode != 0, (
            "every measurement step returned 0, the status record could not be "
            "written, and the driver still reported success"
        )
        assert "re-collection done" not in done.stderr
        # Continuing is still the design: sealing the status file is not a reason to
        # skip the measurements that were left to run.
        logs = sorted(p.name for p in (root / "out").glob("*.log"))
        assert len(logs) == 4, f"a step was skipped: {logs}"
        text = status.read_text(encoding="utf-8")
        assert "run_round1 rc=0" in text
        assert "OVERALL_RC=0" not in text, (
            "a status file claiming OVERALL_RC=0 for a run that could not record its "
            "own outcome is worse than no status file"
        )
    finally:
        status.chmod(0o644)


def test_a_run_that_could_not_write_its_environment_record_is_not_complete(
    tmp_path: Path,
) -> None:
    """The environment file is the only record of the machine the numbers came from.

    Not a red-before-fix test — the step logs fail for the same reason and already
    dragged the exit code down. It is here because the fix must not make the verdict
    depend on the steps noticing: a driver whose steps all succeed and whose
    environment record is missing is still an incomplete run.
    """
    root = fake_driver_root(tmp_path, "run_recollection")
    out = root / "out"
    out.mkdir()
    out.chmod(0o555)
    try:
        done = run_driver(root, "run_recollection", ["out"], {})
        assert done.returncode != 0, done.stderr
        assert "re-collection done" not in done.stderr
        assert not (out / "recollection-environment.txt").exists()
    finally:
        out.chmod(0o755)


def test_the_status_file_and_the_exit_code_agree_when_everything_worked(
    tmp_path: Path,
) -> None:
    """The three verdicts — ``$?``, ``OVERALL_RC=``, the closing line — are one verdict."""
    root = fake_driver_root(tmp_path, "run_recollection")
    done = run_driver(root, "run_recollection", ["out"], {})
    status = (root / "out" / "recollection-status.txt").read_text(encoding="utf-8")
    assert done.returncode == 0, done.stderr
    assert "OVERALL_RC=0" in status
    assert "re-collection done" in done.stderr
    assert (root / "out" / "recollection-environment.txt").stat().st_size > 0, (
        "an empty environment record is not a record"
    )


def test_a_step_failure_is_in_both_the_status_file_and_the_exit_code(tmp_path: Path) -> None:
    root = fake_driver_root(tmp_path, "run_recollection")
    done = run_driver(root, "run_recollection", ["out"], {"run_noise_floor": 3})
    status = (root / "out" / "recollection-status.txt").read_text(encoding="utf-8")
    assert done.returncode != 0, done.stderr
    assert "run_noise_floor rc=3" in status
    assert "OVERALL_RC=1" in status
    assert "re-collection done" not in done.stderr


# --------------------------------------------------------------------------- #
# round 4, problem 1: a record file taken after the pre-check must not be appended to
# --------------------------------------------------------------------------- #


#: Stand-in interpreter that hands the driver a conflict no pre-check could have seen.
#:
#: ``out_guard.py check`` — the second of the driver's two pre-check calls, for the
#: seven product *directories* — is the last thing that runs before it creates its own
#: two record files. Creating ``$POISON_PATH`` on the way out of that call lands
#: strictly after every check has passed and strictly before the write, which is the
#: window the reproduction needs: no sleep, no second process, no race to win.
#:
#: ``$POISON_STAMP`` receives the victim's ``st_mtime_ns`` as the planter saw it, so a
#: test can assert the driver left the mtime alone instead of inferring that from the
#: bytes happening to match.
FAKE_PYTHON_POISON_AFTER_CHECK = """\
#!/bin/bash
case "${1:-}" in
  tools/perf/out_guard.py)
    "@REAL_PY@" "$@"
    rc=$?
    if [ "$rc" -eq 0 ] && [ "${2:-}" = "check" ] && [ -n "${POISON_PATH:-}" ]; then
      printf '%s\\n' "${POISON_CONTENT}" > "$POISON_PATH"
      "@REAL_PY@" -c 'import pathlib, sys; pathlib.Path(sys.argv[2]).write_text(str(pathlib.Path(sys.argv[1]).stat().st_mtime_ns))' "$POISON_PATH" "$POISON_STAMP"
    fi
    exit $rc
    ;;
esac
key="${SLOWQ_MODEL_DIR:-}"
[ -n "$key" ] || key="${1:-}"
rc=0
if [ -n "${FAKE_RC_FILE:-}" ] && [ -f "${FAKE_RC_FILE}" ]; then
  while IFS='=' read -r k v; do
    [ "$k" = "$key" ] && rc="$v"
  done < "${FAKE_RC_FILE}"
fi
echo "fake python: ${key} -> rc=${rc}"
exit "${rc}"
"""


def poison_after_precheck(root: Path, victim: Path, content: str) -> dict[str, str]:
    """Install the interpreter above, aimed at *victim*, and return its environment."""
    interpreter = root / ".venv" / "bin" / "python"
    interpreter.write_text(
        FAKE_PYTHON_POISON_AFTER_CHECK.replace(
            "@REAL_PY@", str(REPO_ROOT / ".venv" / "bin" / "python")
        ),
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    return {
        "POISON_PATH": str(victim),
        "POISON_STAMP": str(root / "victim-mtime.txt"),
        "POISON_CONTENT": content,
    }


def assert_untouched(victim: Path, stamp: Path, content: str) -> None:
    """The foreign file kept both its bytes and the mtime the planter recorded."""
    assert victim.read_text(encoding="utf-8") == content + "\n", (
        "the driver wrote into a record file it did not create"
    )
    assert victim.stat().st_mtime_ns == int(stamp.read_text(encoding="utf-8")), (
        "the foreign record's mtime moved, so something wrote to it after the pre-check"
    )


def test_an_environment_record_taken_after_the_precheck_is_never_appended_to(
    tmp_path: Path,
) -> None:
    """Reproduction 1: rc=0, "re-collection done", and a foreign file carrying the tail.

    ``set -C`` did refuse the ``>`` that would have created the record, so the block
    wrote nothing. But the check that followed was ``[ ! -s ]``, and a file another run
    had already filled *is* non-empty, so the failure was never recorded; the closing
    block then appended with ``>>``, which noclobber does not govern, and the run
    announced itself complete over somebody else's evidence.
    """
    root = fake_driver_root(tmp_path, "run_recollection")
    out = root / "out"
    out.mkdir()
    victim = out / "recollection-environment.txt"
    foreign = "another run's environment record"
    env = poison_after_precheck(root, victim, foreign)

    done = run_driver(root, "run_recollection", ["out"], {}, env)

    assert done.returncode != 0, done.stderr
    assert "re-collection done" not in done.stderr
    assert_untouched(victim, root / "victim-mtime.txt", foreign)
    # Continuing is still the design: losing the environment record is not a reason to
    # skip the measurements, and the four step logs are this run's own files.
    assert sorted(p.name for p in out.glob("*.log")) == [
        "run_noise_floor.log",
        "run_projection.log",
        "run_round1.log",
        "run_round1_longctx.log",
    ]


def test_a_status_file_taken_after_the_precheck_is_never_appended_to(tmp_path: Path) -> None:
    """Reproduction 2: the status half, which plants a verdict in a stranger's file.

    The first create *was* checked here — ``: >"$STATUS" || { EVIDENCE=1; ... }`` — so
    the run already exited nonzero before the fix. What it did not stop was the four
    step lines and the closing ``OVERALL_RC=`` going in anyway, so a run that never
    owned its status record sealed somebody else's with its own outcome.
    """
    root = fake_driver_root(tmp_path, "run_recollection")
    out = root / "out"
    out.mkdir()
    victim = out / "recollection-status.txt"
    foreign = "another run: OVERALL_RC=0"
    env = poison_after_precheck(root, victim, foreign)

    done = run_driver(root, "run_recollection", ["out"], {}, env)

    assert done.returncode != 0, done.stderr
    assert "re-collection done" not in done.stderr
    assert_untouched(victim, root / "victim-mtime.txt", foreign)
    assert len(list(out.glob("*.log"))) == 4, (
        "a driver that cannot record itself still has to run its steps"
    )


#: Stand-in step that seals the environment record once every measurement has succeeded,
#: so the only write left to fail is the closing block's append. ``$1`` is the out-root.
FAKE_STEP_SEAL_ENVIRONMENT = """\
#!/bin/bash
self="$(basename "$0" .sh)"
[ "$self" = "run_projection" ] && chmod 444 "$1/recollection-environment.txt"
echo "fake ${self} -> rc=0"
exit 0
"""


def test_an_environment_record_this_run_owns_but_cannot_finish_is_not_complete(
    tmp_path: Path,
) -> None:
    """Ownership is necessary, not sufficient: the closing append can still fail.

    Green before the fix too — the tail block already had its ``||``. It is here because
    the fix could have closed that branch by accident: an ownership flag that also
    silenced the ``||`` would turn "created the record, could not finish it" into a
    silent success, and that is the same missing evidence as never starting it.
    """
    root = fake_driver_root(tmp_path, "run_recollection")
    install_steps(root, FAKE_STEP_SEAL_ENVIRONMENT)
    environment = root / "out" / "recollection-environment.txt"

    done = run_driver(root, "run_recollection", ["out"], {})
    try:
        assert done.returncode != 0, done.stderr
        assert "re-collection done" not in done.stderr
        text = environment.read_text(encoding="utf-8")
        assert text.startswith("# round-1 re-collection\n"), (
            "this run created the record, so the head it wrote must still be there"
        )
        assert "## machine at the end" not in text
    finally:
        environment.chmod(0o644)


def test_a_run_that_owns_both_records_writes_the_head_and_the_tail(tmp_path: Path) -> None:
    """The fix moved the environment head from ``>`` to ``>>``, so both halves are pinned.

    Creating the file with an explicit ``: >`` and appending afterwards is what makes
    ownership decidable, and it would be easy to leave the head unwritten: ``-s`` would
    still pass because the tail filled the file, and a record whose first line is
    "## machine at the end" describes a machine nobody named a source identity for.
    """
    root = fake_driver_root(tmp_path, "run_recollection")
    done = run_driver(root, "run_recollection", ["out"], {})
    assert done.returncode == 0, done.stderr

    out = root / "out"
    text = (out / "recollection-environment.txt").read_text(encoding="utf-8")
    assert text.startswith("# round-1 re-collection\n")
    assert "## machine at the end" in text
    assert text.index("# round-1 re-collection") < text.index("## machine at the end")
    status = (out / "recollection-status.txt").read_text(encoding="utf-8")
    assert status.endswith("OVERALL_RC=0\n")


#: Stand-in interpreter that creates the log of the model *after* the one it is
#: running, so the conflict appears once ``out_guard_reserve`` has already accepted an
#: empty ``$OUT``. ``run_integration.sh`` ran with cwd=<repo root> and ``$OUT``
#: relative, so the fake resolves the same way.
FAKE_PYTHON_POISON_NEXT_LOG = """\
#!/bin/bash
case "${1:-}" in
  tools/perf/out_guard.py) exec "@REAL_PY@" "$@" ;;
esac
model="${SLOWQ_MODEL_DIR:-}"
if [ "$model" = "models/Qwen3-0.6B-slowq" ]; then
  printf 'a previous run\\n' > "logs/Qwen2.5-0.5B-Instruct-slowq.log"
fi
echo "fake pytest ${model}"
exit 0
"""


def test_run_integration_never_truncates_a_log_it_did_not_create(tmp_path: Path) -> None:
    """The same defect in the sibling driver: ``>`` on a path the reserve could not see."""
    root = fake_driver_root(tmp_path, "run_integration")
    interpreter = root / ".venv" / "bin" / "python"
    interpreter.write_text(
        FAKE_PYTHON_POISON_NEXT_LOG.replace(
            "@REAL_PY@", str(REPO_ROOT / ".venv" / "bin" / "python")
        ),
        encoding="utf-8",
    )
    interpreter.chmod(0o755)

    done = run_driver(root, "run_integration", ["logs"], {})
    victim = root / "logs" / "Qwen2.5-0.5B-Instruct-slowq.log"

    assert done.returncode != 0, done.stderr
    assert victim.read_text(encoding="utf-8") == "a previous run\n", (
        "run_integration.sh truncated a per-model log it did not create"
    )
    assert "OVERALL_RC=0" not in done.stdout


def test_run_reference_fails_when_its_source_identity_record_failed(tmp_path: Path) -> None:
    """``source-identity.txt`` is the record of *which tree* was validated.

    ``run_reference.sh`` redirected into it and looked only at the two
    ``validate_reference.py`` exit codes afterwards, so a source-identity run that
    failed — or whose redirect failed — left a summary reporting eight models validated
    against a tree nobody had named.
    """
    root = fake_driver_root(tmp_path, "run_reference")
    done = run_driver(
        root, "run_reference", ["logs", "pre-review"], {"tools/perf/source_identity.py": 3}
    )
    assert done.returncode != 0, done.stderr
    assert "OVERALL_RC=0" not in done.stdout, done.stdout


# --------------------------------------------------------------------------- #
# round 3, problem 2: an explicit --paired-data is the only source read
# --------------------------------------------------------------------------- #


BALANCED_FOUR = [(1, "base", 1), (1, "opt", 2), (2, "opt", 1), (2, "base", 2)]


def _prefill_median(payload: Any) -> Any:
    return payload["micro"]["engine_prefill"]["512"]["median"]


def test_an_explicit_paired_root_without_a_noise_floor_is_not_padded(
    report_at: Any, tmp_path: Path
) -> None:
    """The confirmed silent fallback: candidates were tried in order, not chosen.

    ``paired_directory()`` inserted ``PAIRED_ROOT/noise-floor`` ahead of
    ``PERF/noise-floor`` and returned the first one that existed. A typo'd or
    not-yet-collected ``--paired-data`` therefore rendered the *previous* experiment's
    verdicts, under a header that named the directory the operator had asked for.

    The old experiment has to sit at ``PERF`` itself, which is ``tmp_path`` under the
    ``report_at`` fixture: putting it in a sibling directory would leave nothing to
    fall back to and the test would pass without the fallback ever being exercised.
    """
    write_noise_runs(tmp_path / report_at.NOISE_DIR, "Qwen3-0.6B-slowq", BALANCED_FOUR)
    explicit = tmp_path / "new-run"
    explicit.mkdir()

    # The argument is refused outright, before a section is rendered, rather than
    # quietly re-pointed at the data root's own experiment.
    with pytest.raises(SystemExit):
        report_at.resolve_paired_root(str(explicit), tmp_path)

    # And a root that reached ``paired_directory()`` by any other route still does not
    # read the old one: the directory is derived from the explicit root alone, so the
    # absence is a gap in the report rather than somebody else's number.
    report_at.PAIRED_ROOT = explicit
    try:
        assert report_at.paired_directory() is None, (
            "an explicit root with no noise-floor/ fell back to the data root's"
        )
        assert report_at.paired_rounds("Qwen3-0.6B-slowq", _prefill_median) is None, (
            "read the data root's experiment even though an explicit root was set"
        )
    finally:
        report_at.PAIRED_ROOT = None


def test_a_model_missing_from_the_explicit_root_is_a_gap_not_the_old_number(
    report_at: Any, tmp_path: Path
) -> None:
    """Partial data must not be completed from the previous experiment, model by model.

    Green before the fix as well — the old fallback was per *directory*, not per
    model, so an explicit root that exists was already read exclusively. It is here
    because the fix moves the exclusivity into ``paired_directory()`` and a per-model
    fallback would be an easy way to reintroduce the padding while keeping the
    directory-level test above passing.
    """
    write_noise_runs(tmp_path / report_at.NOISE_DIR, "Qwen3-0.6B-slowq", BALANCED_FOUR)
    explicit = tmp_path / "new-run"
    write_noise_runs(
        explicit / report_at.NOISE_DIR, "SmolLM2-135M-Instruct-slowq", BALANCED_FOUR
    )

    report_at.PAIRED_ROOT = explicit
    try:
        assert report_at.paired_directory() == explicit / report_at.NOISE_DIR
        assert report_at.paired_rounds("Qwen3-0.6B-slowq", _prefill_median) is None, (
            "read from the old root even though --paired-data named a different one"
        )
        assert (
            report_at.paired_rounds("SmolLM2-135M-Instruct-slowq", _prefill_median)
            is not None
        )
    finally:
        report_at.PAIRED_ROOT = None


def test_without_an_explicit_root_the_data_roots_own_is_still_read(
    report_at: Any, tmp_path: Path
) -> None:
    """The default must not move: ``--data <root>`` alone still reads ``<root>/noise-floor``."""
    write_noise_runs(tmp_path / report_at.NOISE_DIR, "Qwen3-0.6B-slowq", BALANCED_FOUR)
    assert report_at.resolve_paired_root(None, tmp_path) == tmp_path
    assert report_at.paired_directory() == tmp_path / report_at.NOISE_DIR
    assert report_at.paired_rounds("Qwen3-0.6B-slowq", _prefill_median) is not None


def test_a_data_root_with_no_paired_experiment_resolves_to_nothing(
    report_at: Any, tmp_path: Path
) -> None:
    """The other half of the default: no experiment anywhere means "not run", not an error.

    Only an *explicit* ``--paired-data`` is refused when it cannot be read. Without
    one, "this round did not run the paired experiment" is a legitimate answer and
    §7.3 has said so all along.
    """
    assert report_at.resolve_paired_root(None, tmp_path) is None
    assert report_at.paired_directory() is None


def _unusable_paired_roots() -> list[tuple[str, Callable[[Path, Any], Path]]]:
    """``--paired-data`` values that must be refused before anything is written."""

    def missing(tmp_path: Path, report: Any) -> Path:
        return tmp_path / "typoed-run-root"

    def a_file(tmp_path: Path, report: Any) -> Path:
        path = tmp_path / "not-a-directory"
        path.write_text("x\n", encoding="utf-8")
        return path

    def no_noise_dir(tmp_path: Path, report: Any) -> Path:
        path = tmp_path / "run-without-noise-floor"
        path.mkdir()
        return path

    def empty_noise_dir(tmp_path: Path, report: Any) -> Path:
        path = tmp_path / "run-with-empty-noise-floor"
        (path / report.NOISE_DIR).mkdir(parents=True)
        return path

    def unparseable(tmp_path: Path, report: Any) -> Path:
        path = tmp_path / "run-with-junk"
        (path / report.NOISE_DIR).mkdir(parents=True)
        (path / report.NOISE_DIR / "Qwen3-0.6B-slowq-r1-base.json").write_text(
            "{ not json", encoding="utf-8"
        )
        return path

    return [
        ("不存在", missing),
        ("是一个文件", a_file),
        ("缺少 noise-floor 子目录", no_noise_dir),
        ("noise-floor 为空", empty_noise_dir),
        ("noise-floor 里读不出配对 JSON", unparseable),
    ]


@pytest.mark.parametrize(
    "build", [build for _, build in _unusable_paired_roots()],
    ids=[name for name, _ in _unusable_paired_roots()],
)
def test_an_unusable_explicit_paired_root_fails_before_the_report_is_written(
    report: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    build: Callable[[Path, Any], Path],
) -> None:
    """Rejected up front, not after the official document has been overwritten.

    The argument used to be resolved and never looked at, so every one of these five
    produced a complete, confident report from ``--data``'s own noise-floor. Failing
    late would be bad enough; failing *after* ``write_text`` means the argument error
    is discovered by a reader of the delivered document.

    The refusal has to name the paired root. Without that this test is satisfied by
    any ``SystemExit`` at all — including argparse rejecting an option it does not
    know, which is how it passed for the wrong reason the first time it was written.
    """
    data = tmp_path / "data"
    write_noise_runs(data / report.NOISE_DIR, "Qwen3-0.6B-slowq", BALANCED_FOUR)
    paired = build(tmp_path, report)
    tables = tmp_path / "tables.md"
    doc = tmp_path / "doc.md"

    saved = tuple(getattr(report, name) for name in _REPORT_GLOBALS)
    try:
        with pytest.raises(SystemExit) as raised:
            report.main(
                [
                    "--data", str(data),
                    "--out", str(tables),
                    "--doc", str(doc),
                    "--paired-data", str(paired),
                ]
            )
    finally:
        for name, value in zip(_REPORT_GLOBALS, saved):
            setattr(report, name, value)

    assert raised.value.code != 0
    stderr = capsys.readouterr().err
    assert "--paired-data" in stderr, f"the refusal does not name the argument: {stderr}"
    assert str(paired) in stderr, f"the refusal does not name the offending path: {stderr}"
    assert not tables.exists(), "the tables were written before the argument was judged"
    assert not doc.exists(), "the narrative document was written before the argument was judged"


def test_a_valid_explicit_paired_root_is_the_one_the_report_names(
    report: Any, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    explicit = tmp_path / "new-run"
    write_noise_runs(explicit / report.NOISE_DIR, "Qwen3-0.6B-slowq", BALANCED_FOUR)
    # The data root's own experiment is a different one, so a fallback is visible.
    write_noise_runs(data / report.NOISE_DIR, "Qwen3-0.6B-slowq", BALANCED_FOUR)

    rendered = render_report(report, tmp_path, data, paired=explicit, tag="explicit")

    header = rendered.tables.read_text(encoding="utf-8").splitlines()[3]
    assert str((explicit / report.NOISE_DIR).resolve()) in header, header
    assert str((data / report.NOISE_DIR).resolve()) not in header, (
        "the report names the data root's own experiment even though --paired-data "
        f"named another: {header}"
    )


# --------------------------------------------------------------------------- #
# round 3, problem 3: input data and output location are different arguments
# --------------------------------------------------------------------------- #


def test_the_generator_renders_from_a_read_only_data_root(report: Any, tmp_path: Path) -> None:
    """Input read-only, output writable — the two were the same directory before."""
    data = tmp_path / "data"
    write_noise_runs(data / report.NOISE_DIR, "Qwen3-0.6B-slowq", BALANCED_FOUR)
    data.chmod(0o555)
    try:
        rendered = render_report(report, tmp_path, data, tag="readonly")
    finally:
        data.chmod(0o755)
    assert rendered.tables.is_file() and rendered.doc.is_file()
    assert rendered.tables.read_text(encoding="utf-8").startswith(
        "<!-- 由 tools/perf/make_report.py 生成"
    )


def test_repeated_generation_from_one_input_is_byte_identical(
    report: Any, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    write_noise_runs(data / report.NOISE_DIR, "Qwen3-0.6B-slowq", BALANCED_FOUR)
    # The same tag both times, so both passes write the *same* two paths. Rendering to
    # two different names would compare §1.3's output-path rows rather than the report:
    # the document names the table its own invocation wrote, which is the fix for the
    # 65-vs-64 split and also the one line that legitimately differs between two
    # differently-named renders.
    first = render_report(report, tmp_path, data, tag="repeat")
    first_tables, first_doc = first.tables.read_bytes(), first.doc.read_bytes()
    second = render_report(report, tmp_path, data, tag="repeat")
    assert second.tables.read_bytes() == first_tables
    assert second.doc.read_bytes() == first_doc


def test_the_product_fingerprint_catches_a_rewrite_that_changes_no_bytes(
    tmp_path: Path,
) -> None:
    """A byte-identical regeneration still moves the mtime a reviewer reads.

    Regenerating the same table from the same input produces the same bytes, so a
    content-only guard would pass while the historical file's mtime moved to today --
    and "this derived table was written today" next to raw JSON from last week is the
    ambiguity the fixture exists to remove. ``st_mtime_ns`` is in the tuple for that
    reason; this is the test that keeps it there.
    """
    target = tmp_path / "report-tables.md"
    target.write_text("| 收益 |\n", encoding="utf-8")
    digest = product_fingerprint(target)
    assert digest is not None
    pinned = 1_600_000_000_000_000_000
    os.utime(target, ns=(pinned, pinned))
    assert product_fingerprint(target) == (digest[0], pinned, digest[2])

    target.write_bytes(target.read_bytes())
    rewritten = product_fingerprint(target)
    assert rewritten is not None
    assert rewritten[2] == digest[2], "content is unchanged, so only mtime can differ"
    assert rewritten[0] == digest[0]
    assert rewritten != (digest[0], pinned, digest[2])


@pytest.mark.parametrize("damage", ["deleted", "rewritten", "mtime moved"])
def test_the_product_guard_sees_all_three_ways_a_test_can_damage_a_product(
    tmp_path: Path, damage: str
) -> None:
    """The fixture's own comparison, driven against a fabricated product.

    Deliberately not the real four: two of them are derived tables under a gitignored
    ``artifacts/``, so testing the comparison against ``REPO_ROOT`` makes it run on one
    machine and fail on every checkout that did not also carry the historical evidence
    over. What is under test is a nested path beneath a root, and ``tmp_path`` has that
    shape without needing anything from ``artifacts/``.

    The three cases are the three the fingerprint tuple exists to tell apart, and each
    is a way a test can have written to a product while its own assertions passed:
    deleting it, changing it, and regenerating byte-identical content over it.
    """
    name = "artifacts/perf-round1/report-tables.md"
    target = tmp_path / name
    target.parent.mkdir(parents=True)
    target.write_text("| 收益 | 64 |\n", encoding="utf-8")

    before = fingerprint_all([name], tmp_path)
    assert changed_since(before, tmp_path) == [], "nothing has happened yet"

    if damage == "deleted":
        target.unlink()
    elif damage == "rewritten":
        target.write_text("| 收益 | 65 |\n", encoding="utf-8")
    else:
        pinned = 1_600_000_000_000_000_000
        os.utime(target, ns=(pinned, pinned))

    assert changed_since(before, tmp_path) == [name], damage


def test_a_product_that_never_existed_cannot_be_told_from_a_deleted_one(
    tmp_path: Path,
) -> None:
    """Why "the four files exist" is a separate gated test and not part of the fixture.

    ``product_fingerprint`` returns None for a missing path and ``None == None``, so a
    product absent before a test and absent after it is indistinguishable from one that
    was never on the list: the guard quietly stops guarding, and nothing inside the
    fixture can notice, because its only ground truth is the files themselves. Stating
    the gap is what keeps it from being read as coverage.
    """
    name = "artifacts/perf-round1/report-tables.md"
    before = fingerprint_all([name], tmp_path)
    assert before[name] is None
    assert changed_since(before, tmp_path) == []


@protected_products_missing
def test_every_protected_product_in_a_full_checkout_is_fingerprintable() -> None:
    """Release acceptance: with the evidence in place, all four are files the guard sees.

    This is the check the test above says cannot live inside the fixture. It belongs to
    a full checkout — the state the release flow produces and the state a reviewer reads
    the report in — and it is skipped, not deleted, wherever the historical artifacts
    were not carried over.
    """
    for name in PROTECTED_PRODUCTS:
        path = REPO_ROOT / name
        assert path.is_file(), f"{name} is on PROTECTED_PRODUCTS but does not exist"
        assert product_fingerprint(path) is not None, name


def test_every_delivered_first_round_document_is_on_the_protected_list() -> None:
    """Both paths a release writes have to be covered, and the list is not the source.

    The expectation comes off the filesystem rather than being restated, so dropping
    ``docs/推理加速-第1轮-测量表.md`` -- which only became an official product when §1.3
    started naming the table its own invocation wrote -- fails here instead of letting
    a test rewrite the delivered table and blame nobody. A third such file fails too.
    """
    delivered = sorted(
        path.name for path in (REPO_ROOT / "docs").glob("推理加速-第1轮*.md")
    )
    assert delivered, "no delivered first-round document found under docs/"
    for name in delivered:
        assert f"docs/{name}" in PROTECTED_PRODUCTS, name


def test_the_default_out_lands_beside_the_data_and_is_refused_when_published(
    report: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--out`` keeps its old default for a fresh root and refuses a published one.

    The default is what ``run_recollection.sh`` tells the operator to run next, so it
    has to keep working; the refusal is what stops a regeneration from replacing the
    historical derived table beside the raw JSON it was derived from.
    """
    data = tmp_path / "run"
    data.mkdir()
    doc = tmp_path / "doc.md"
    saved = tuple(getattr(report, name) for name in _REPORT_GLOBALS)
    try:
        report.main(["--data", str(data), "--doc", str(doc)])
        assert (data / "report-tables.md").is_file()

        monkeypatch.setattr(report, "ROOT", tmp_path)
        published = tmp_path / "artifacts" / "perf-round1"
        published.mkdir(parents=True)
        with pytest.raises(SystemExit) as raised:
            report.main(["--data", str(published), "--doc", str(doc)])
        assert raised.value.code != 0
        assert not (published / "report-tables.md").exists()
    finally:
        for name, value in zip(_REPORT_GLOBALS, saved):
            setattr(report, name, value)


PAIRED_HEADING = "### 跨进程离散度与逐轮配对增益"


def _paired_section(text: str) -> str:
    assert PAIRED_HEADING in text, "the paired section did not render; this test guards nothing"
    return text.split(PAIRED_HEADING, 1)[1].split("\n## ", 1)[0]


def _verdict_rows(section: str) -> list[tuple[str, ...]]:
    """``(模型, 量, 结论)`` for every row of the paired table.

    ``reflow`` keeps table lines byte-exact, so comparing parsed rows rather than raw
    text is a choice about the *message*, not about robustness: a diff of two cells
    says which verdict moved, a diff of two 40 KB documents does not.
    """
    rows = []
    for line in section.splitlines():
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in ("模型",):
            rows.append((cells[0], cells[1], cells[-1]))
    return rows


def test_the_document_and_its_tables_come_from_one_invocation(
    report: Any, tmp_path: Path
) -> None:
    """The 65-vs-64 split, made impossible rather than made equal.

    Two products of one report existed on disk and disagreed, because they were
    generated by two different commands reading two different paired experiments. The
    reviewer was right that this cannot be fixed by editing a number: the fix is that
    the document points at the table *this* invocation wrote, and that both halves are
    compared row by row here so a future split fails a test instead of a review.
    """
    data = tmp_path / "data"
    data.mkdir()
    explicit = tmp_path / "paired-run"
    write_noise_runs(explicit / report.NOISE_DIR, "Qwen3-0.6B-slowq", BALANCED_FOUR)

    rendered = render_report(report, tmp_path, data, paired=explicit, tag="oneshot")
    tables = rendered.tables.read_text(encoding="utf-8")
    doc = rendered.doc.read_text(encoding="utf-8")

    assert report.repo_relative(rendered.tables) in doc, (
        "the document does not name the table this invocation wrote, so a reader "
        "sent to check a row lands on some other run's table"
    )
    assert _verdict_rows(_paired_section(tables)) == _verdict_rows(_paired_section(doc))


# --------------------------------------------------------------------------- #
# round 3, problem 4: --out is checked before the model is loaded
# --------------------------------------------------------------------------- #


#: Every directly-invocable measurement program, with the smallest argument set that
#: reaches ``main()``. The model is deliberately absent: the point is that a conflict
#: on ``--out`` is found *before* anything tries to load one.
MEASUREMENT_ENTRY_POINTS = (
    ("core_ab.py", ["--src", ".", "--model", "models/absent-model"]),
    (
        "vision_cache_ab.py",
        ["--src", ".", "--model", "models/absent-model", "--image", "absent.png"],
    ),
    ("document_index_ab.py", ["--src", ".", "--model", "models/absent-model"]),
    ("identity_cost.py", ["--model", "models/absent-model"]),
    ("projection_probe.py", ["--model", "models/absent-model"]),
    ("sigmoid_check.py", ["--model", "models/absent-model"]),
)


def run_measurement(program: str, args: Sequence[str], out: Path) -> Any:
    return subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            f"tools/perf/{program}",
            *args,
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "program,args",
    MEASUREMENT_ENTRY_POINTS,
    ids=[program for program, _ in MEASUREMENT_ENTRY_POINTS],
)
def test_a_measurement_program_refuses_an_occupied_out_before_loading_anything(
    tmp_path: Path, program: str, args: Sequence[str]
) -> None:
    """The existing file survived, but only because the run wasted an hour first.

    ``out_guard.write_json`` was already exclusive, so nothing was ever overwritten;
    the defect was *when* the refusal happened — after the model load and the whole
    measurement suite, at the last line of ``main()``. The refusal has to be the first
    thing the program does, which is what asserting on stderr's first line checks.
    """
    out = tmp_path / "result.json"
    out.write_text('{"label": "an earlier run"}\n', encoding="utf-8")

    done = run_measurement(program, args, out)

    assert done.returncode != 0, done.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == {"label": "an earlier run"}
    first = done.stderr.lstrip().splitlines()[0] if done.stderr.strip() else ""
    assert first.startswith("out_guard: 拒绝写入"), (
        f"{program} did work before checking --out:\n{done.stderr[:600]}"
    )


@pytest.mark.parametrize(
    "program,args",
    MEASUREMENT_ENTRY_POINTS,
    ids=[program for program, _ in MEASUREMENT_ENTRY_POINTS],
)
def test_a_free_out_path_is_not_refused_by_the_preflight(
    tmp_path: Path, program: str, args: Sequence[str]
) -> None:
    """A preflight that blocks legitimate runs is worse than none.

    The model is absent, so the program still fails — but it fails *after* the
    preflight, on the load, which is the proof that the preflight let it through.
    """
    out = tmp_path / "nested" / "result.json"

    done = run_measurement(program, args, out)

    assert done.returncode != 0, done.stderr
    assert "out_guard: 拒绝写入" not in done.stderr, done.stderr[:600]
    assert not out.exists(), "the preflight created the file it was only meant to check"


@pytest.mark.parametrize(
    "program,args",
    MEASUREMENT_ENTRY_POINTS,
    ids=[program for program, _ in MEASUREMENT_ENTRY_POINTS],
)
def test_a_measurement_program_refuses_a_directory_at_its_out_path(
    tmp_path: Path, program: str, args: Sequence[str]
) -> None:
    """The problem-1 rule at the problem-4 call site: a directory is not a free file path."""
    out = tmp_path / "result.json"
    out.mkdir()

    done = run_measurement(program, args, out)

    assert done.returncode != 0, done.stderr
    assert out.is_dir() and not any(out.iterdir())
    assert "out_guard: 拒绝写入" in done.stderr


# --------------------------------------------------------------------------- #
# round 4, problem 3: a second main() in one process starts from nothing
# --------------------------------------------------------------------------- #


class Invocation(NamedTuple):
    """One ``main()``'s two products, plus the globals it left behind.

    The globals are read at the end of *this* call rather than at the start of the
    next one, because reading them after a second ``main()`` would show that second
    run's values and the leak would be invisible.
    """

    tables: Path
    doc: Path
    paired_summary: Any
    paired_between: dict[str, tuple[float, float]]
    sigmoid_correction: Path


@pytest.fixture
def consecutive_main(report: Any) -> Iterator[Callable[[Path, str], Invocation]]:
    """Run ``main()`` repeatedly in *one* process, restoring globals only at teardown.

    Deliberately not :func:`render_report`. That helper saves and restores
    ``_REPORT_GLOBALS`` around each call, which is right for the nine tests that use
    it and wrong here: the leak between two calls is the defect, so a helper that puts
    the state back in between hides it and the test would pass against the unfixed
    generator. Nothing reloads the module either — same object, same globals, the
    shape a notebook or a long-lived caller actually produces.

    The restore is at teardown, once, because ``report`` is module-scoped: a leaked
    ``PERF`` pointing into a deleted ``tmp_path`` would make every later test in this
    file describe a directory that no longer exists.
    """
    saved = tuple(getattr(report, name) for name in _REPORT_GLOBALS)

    def invoke(data: Path, tag: str) -> Invocation:
        tables = data.parent / f"{tag}-tables.md"
        doc = data.parent / f"{tag}-doc.md"
        report.main(
            ["--data", str(data), "--out", str(tables), "--doc", str(doc)]
        )
        return Invocation(
            tables,
            doc,
            report.PAIRED_SUMMARY,
            dict(report.PAIRED_BETWEEN),
            report.SIGMOID_CORRECTION,
        )

    try:
        yield invoke
    finally:
        for name, value in zip(_REPORT_GLOBALS, saved):
            setattr(report, name, value)


def synthetic_sigmoid_check(tag: str) -> dict[str, Any]:
    """A fabricated ``sigmoid-check.json`` that §5 can actually render.

    Minimal in the numbers, not in the keys: :func:`write_doc` formats
    ``shortcut_rel_l2`` with ``:.4f`` and the gate statistics through ``gate_pct``, so
    a payload missing them raises ``TypeError`` instead of reporting a gap — which
    would test the fixture rather than the generator. Everything is invented; the real
    probe's output sits under a gitignored ``artifacts/`` and this test must run
    without it.
    """

    def timed(seconds: float) -> dict[str, Any]:
        return {"median": seconds, "stdev": seconds * 0.02, "samples": [seconds] * 3}

    def row(old: float, new: float) -> dict[str, Any]:
        return {
            "old_seconds": old,
            "new_seconds": new,
            "speedup": old / new,
            "old": timed(old),
            "new": timed(new),
        }

    return {
        "model": "Qwen3-0.6B-slowq",
        "layer": 0,
        "tokens": 128,
        "blas": "accelerate",
        "identity": {"ops_py_code_sha256": "0" * 64},
        "bit_equality_failures": [],
        "bit_equality_notes": [f"{tag}: 逐位相同"],
        "gate_activation": {
            "mean": -0.5,
            "std": 0.6,
            "fraction_negative": 0.8,
            "absmax": 6.0,
        },
        "shortcut_vs_captured": {
            "rel_l2": 1.09,
            "shortcut_stats": {
                "mean": 0.06,
                "std": 0.24,
                "fraction_negative": 0.38,
                "absmax": 4.6,
            },
        },
        "matmul_total_seconds": 0.02,
        "old_sigmoid_share_of_matmuls": 0.14,
        "new_sigmoid_share_of_matmuls": 0.03,
        "sigmoid_rows": {
            "captured gate activation": row(0.0029, 0.00072),
            "standard normal": row(0.0052, 0.00071),
            "all positive": row(0.00081, 0.00072),
            "all negative": row(0.00081, 0.00072),
        },
        "real_substitution": {
            "mlp_old_seconds": 0.010,
            "mlp_new_seconds": 0.008,
            "mlp_saved_seconds": 0.002,
            "mlp_saved_share": 0.20,
            "layer_old_seconds": 0.030,
            "layer_new_seconds": 0.028,
            "layer_saved_seconds": 0.002,
            "layer_saved_share": 0.067,
            **{
                key: timed(value)
                for key, value in [
                    ("mlp_old", 0.010),
                    ("mlp_new", 0.008),
                    ("layer_old", 0.030),
                    ("layer_new", 0.028),
                ]
            },
        },
    }


def test_a_second_run_without_a_paired_experiment_does_not_quote_the_first(
    consecutive_main: Callable[[Path, str], Invocation], tmp_path: Path
) -> None:
    """The confirmed contradiction: §7.2 said 已经跑过 above a §7.3 that had no table.

    ``paired_noise_section`` returns early when the root holds no paired runs, so it
    never assigned ``PAIRED_SUMMARY`` — and ``main()`` never cleared it. Two thousand
    lines later ``write_doc`` read the *previous* root's summary and printed
    「N 个模型 × M 轮 …已经跑过」 into a document whose own tables came from
    ``incidental_noise_section``, the branch that runs precisely when there is no
    paired experiment. Both halves were true of a different run.

    The tables' own header comment is the control: ``main()`` derives it from
    ``paired_directory()``, which follows ``PAIRED_ROOT`` and was therefore correct
    even before the fix. A document that says 未运行 in its header and 已经跑过 in its
    prose is self-contradictory, which is the assertion below.
    """
    first = tmp_path / "with-paired"
    second = tmp_path / "without-paired"
    first.mkdir()
    second.mkdir()
    write_noise_runs(first / "noise-floor", "Qwen3-0.6B-slowq", BALANCED_FOUR)

    run_a = consecutive_main(first, "a")
    assert run_a.paired_summary is not None, "the first root has a paired experiment"

    run_b = consecutive_main(second, "b")

    assert run_b.paired_summary is None, (
        f"the second run inherited the first one's summary: {run_b.paired_summary}"
    )
    assert run_b.paired_between == {}
    assert "（未运行）" in run_b.tables.read_text(encoding="utf-8")
    doc = run_b.doc.read_text(encoding="utf-8")
    assert "本轮没有跑它" in doc, "§7.2 must say this round did not run the experiment"
    # 「已经跑过」 occurs once in the whole generator, in the branch that has a
    # summary to quote. Not "个模型 ×", which §1 uses for models × workloads.
    assert "已经跑过" not in doc, (
        "§7.2 quoted a model×round count from the previous --data root"
    )


def test_a_second_run_over_different_paired_data_keeps_only_its_own_models(
    consecutive_main: Callable[[Path, str], Invocation], tmp_path: Path
) -> None:
    """``PAIRED_BETWEEN`` is written per model and was never cleared.

    Both roots measure one model over two rounds, so ``PAIRED_SUMMARY`` agrees with
    itself either way and cannot catch this — the stale key can. §9.2 compares each
    model's optimisation-1 share against *that model's* own between-process spread,
    so a key left over from the previous root lets the comparison quote a model this
    experiment never measured.
    """
    first = tmp_path / "experiment-a"
    second = tmp_path / "experiment-b"
    write_noise_runs(first / "noise-floor", "Qwen3-0.6B-slowq", BALANCED_FOUR)
    write_noise_runs(second / "noise-floor", "SmolLM2-135M-Instruct-slowq", BALANCED_FOUR)

    run_a = consecutive_main(first, "a")
    assert set(run_a.paired_between) == {"Qwen3-0.6B"}

    run_b = consecutive_main(second, "b")

    assert set(run_b.paired_between) == {"SmolLM2-135M"}, (
        "the second run still carries the first experiment's model"
    )
    assert run_b.paired_summary is not None
    doc = run_b.doc.read_text(encoding="utf-8")
    assert "Qwen3-0.6B" not in doc.split("## 7.")[1].split("## 8.")[0], (
        "§7 of the second report names a model only the first experiment measured"
    )


def test_a_second_run_without_a_sigmoid_source_does_not_keep_the_first_ones(
    report: Any,
    consecutive_main: Callable[[Path, str], Invocation],
    tmp_path: Path,
) -> None:
    """``SIGMOID_CORRECTION`` is assigned only when ``<data>/sigmoid/`` exists.

    The rule is deliberate — a re-collected root carries its own probe and the
    published one falls back to the standalone correction directory — but "only assign
    when present" means a root *without* one kept whichever path the previous
    invocation had set. §5 then attributed this run's sigmoid cost to a correction
    measured against a different tree, and named that tree's file.

    Reset to ``DEFAULT_SIGMOID_CORRECTION``, which is what a fresh process resolves
    to, rather than to ``None``: the default is a real input for the published round,
    and clearing it would turn a correct fallback into a gap.
    """
    first = tmp_path / "with-sigmoid"
    second = tmp_path / "without-sigmoid"
    (first / "sigmoid").mkdir(parents=True)
    second.mkdir()
    correction = first / "sigmoid" / "sigmoid-check.json"
    correction.write_text(json.dumps(synthetic_sigmoid_check("a")), encoding="utf-8")

    run_a = consecutive_main(first, "a")
    assert run_a.sigmoid_correction == correction

    run_b = consecutive_main(second, "b")

    assert run_b.sigmoid_correction == report.DEFAULT_SIGMOID_CORRECTION, (
        "the second run kept a correction path instead of falling back to the default"
    )
    assert run_b.sigmoid_correction != correction, (
        "the second run still reads the first root's sigmoid correction"
    )
    for product in (
        run_b.tables.read_text(encoding="utf-8"),
        run_b.doc.read_text(encoding="utf-8"),
    ):
        assert "with-sigmoid" not in product, (
            "a product of the second run cites the first root's correction directory"
        )
