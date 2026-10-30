#!/usr/bin/env python3
"""Emit the markdown tables behind docs/推理加速-第1轮-2026-10-23.md.

Every number in the report's tables comes from this script reading the JSON that
the harnesses wrote, never from transcription. That matters more than it sounds:
the round has four separate A/B axes (base/opt tree, cache off/on, cold/warm
session, seven thread and chunk configurations) and a hand-copied table is where
a "before" silently becomes some other run's "after".

    .venv/bin/python tools/perf/make_report.py [--data artifacts/perf-round1]

Writes <data>/report-tables.md unless --out names somewhere else, and rewrites
docs/推理加速-第1轮-2026-10-23.md unless --doc names somewhere else. Sections whose
inputs are missing are emitted as an explicit MISSING line rather than dropped, so an
incomplete measurement run cannot produce a report that merely looks finished.

``--data`` exists because a re-measurement must not overwrite the run the
published report was derived from: the old JSON stays where it is, the new one
goes to its own directory, and this script is pointed at whichever of the two is
being turned into a report. Which one it read is printed into the report.

``--out`` and ``--doc`` exist because *reading* a data root and *writing* next to
it are two different permissions. A data root can be read-only evidence, and the
narrative document is the deliverable — a test that renders a report has no
business replacing either. Both default to the historical location, so the
command above still behaves as it always did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: One rule for "may this path be written", shared with the measurement programs and
#: the drivers rather than restated here. See tools/perf/out_guard.py.
import out_guard  # noqa: E402

DEFAULT_PERF = ROOT / "artifacts" / "perf-round1"
PERF = DEFAULT_PERF
OUT = PERF / "report-tables.md"
#: The narrative half of the report. Its filename is a constant rather than being
#: rebuilt in ``main`` so that the default and the module-level value cannot drift.
DEFAULT_DOC_NAME = "推理加速-第1轮-2026-10-23.md"
DOC = ROOT / "docs" / DEFAULT_DOC_NAME

#: The re-measured sigmoid attribution, read *in addition to* the published one.
#:
#: It cannot live under ``PERF``: ``artifacts/perf-round1/`` holds the raw evidence
#: behind the published report and the drivers refuse to write into it, so a
#: correction has to come from a directory of its own. That is also why both are
#: loaded — the report quotes the withdrawn numbers next to the ones that replace
#: them, and a correction that deleted the original would not be auditable.
DEFAULT_SIGMOID_CORRECTION = ROOT / "artifacts" / "perf-round1-rerun" / "sigmoid" / "sigmoid-check.json"
SIGMOID_CORRECTION = DEFAULT_SIGMOID_CORRECTION

#: Where ``baseline_tree.py`` materialises the base tree. Mirrors that script's
#: ``DEFAULT_DEST`` rather than following ``PERF``, and the report has to say so:
#: generated from the re-collected root, §1.1 names a directory under the
#: *published* root, and a reader who assumes every path in the report is relative
#: to ``--data`` goes looking for ``perf-round1-rerun/baseline-tree/`` and finds
#: nothing. The re-collection never rebuilt a baseline — rebuilding one means
#: checking out an older revision, and the point of re-collecting was to measure the
#: fixed source, not the old tree — so §6 says why the base column is empty there.
BASELINE_TREE = ROOT / "artifacts" / "perf-round1" / "baseline-tree"

#: The last completed re-collection *of the round this report describes*. §10's
#: commands no longer write here — they write into a fresh ``$RUN`` created by
#: ``out_guard.py fresh-root``, because a reproduction run that lands on the directory
#: the published numbers came from replaces evidence instead of adding to it. What this
#: path is still for is the *read* side, and only as a fallback: the 期望 lines quote
#: the counts the current tree produces, and once ``out_guard`` started refusing the
#: published roots a new count can only be persisted in a fresh one. Naming the source
#: of an expectation matters now that the command beside it writes somewhere else — an
#: unlabelled number next to a fresh directory reads as a prediction about that
#: directory, and it is not one.
REPRO_ROOT = ROOT / "artifacts" / "perf-round1-rerun"

#: ``out_guard.fresh_root``'s naming scheme: ``artifacts/perf-run-<UTC stamp>/``.
FRESH_ROOT_PATTERN = "perf-run-*"


def repro_roots() -> tuple[Path, ...]:
    """Every run root the 期望 lines may read, most recent first.

    Fresh roots before the published ones, and ``REPRO_ROOT`` kept as the fallback so a
    checkout with no fresh run still prints an expectation rather than nothing. Sorted
    by *name*, not mtime: the name is the fixed-width ``%Y%m%d-%H%M%S`` stamp the guard
    minted, so lexicographic order is chronological order, and a log somebody copied in
    afterwards cannot outrank the run that is actually newer.
    """
    fresh = sorted((ROOT / "artifacts").glob(FRESH_ROOT_PATTERN), reverse=True)
    return (*fresh, REPRO_ROOT, PERF, DEFAULT_PERF)


def withdrawn_sigmoid_path(perf: Path) -> Path:
    """Where the *withdrawn* sigmoid probe lives, given a data root.

    Unlike every other artifact this one does not follow ``PERF``. It is the
    published round's original probe — the object being corrected — so it has one
    fixed location, and a report generated from a re-collected round has to quote
    it just as much as one generated from the published round. Reading it as
    ``PERF / "sigmoid-check.json"`` makes §2.4's correction note silently lose its
    left-hand column on the re-collection root: the withdrawn 7.20× / 38.8% /
    153% render as 数据缺失, and a reader sees the replacement numbers with
    nothing to compare them against, which reads as "no prior claim existed"
    rather than as "the prior claim was wrong".
    """
    local = perf / "sigmoid-check.json"
    return local if local.exists() else DEFAULT_PERF / "sigmoid-check.json"

MODELS = [
    "SmolLM2-135M-Instruct-slowq",
    "SmolLM2-360M-Instruct-slowq",
    "Qwen2.5-0.5B-Instruct-slowq",
    "Qwen3-0.6B-slowq",
    "Qwen3-1.7B-slowq",
]

SHORT = {
    "SmolLM2-135M-Instruct-slowq": "SmolLM2-135M",
    "SmolLM2-360M-Instruct-slowq": "SmolLM2-360M",
    "Qwen2.5-0.5B-Instruct-slowq": "Qwen2.5-0.5B",
    "Qwen3-0.6B-slowq": "Qwen3-0.6B",
    "Qwen3-1.7B-slowq": "Qwen3-1.7B",
    "Qwen2.5-Coder-0.5B-Instruct-slowq": "Qwen2.5-Coder",
    "Qwen3-Embedding-0.6B-slowq": "Qwen3-Embedding",
    "SmolVLM-256M-Instruct-slowq": "SmolVLM-256M",
}

PREFILL_LENGTHS = ("8", "32", "128", "512")

# The two models the thread/chunk sweep covers, in both baseline/ and sweep-opt/.
# Kept at module level because the report's prose quotes the sweep's noise floor,
# and it has to be the floor of *these* runs rather than a recomputed lookalike.
SWEEP_MODELS = ["Qwen3-0.6B-slowq", "SmolLM2-360M-Instruct-slowq"]


def load(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(raw: str, root: Path) -> Path:
    """``raw`` as an absolute path under ``root``, with symlinks resolved.

    Every path argument this script takes goes through here, so "relative to what"
    has one answer instead of one per flag.
    """
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def repo_relative(path: Path) -> str:
    """``path`` as the reader will type it, not as this machine stores it.

    An absolute path in the doc names one person's home directory, and it is long
    enough that ``reflow`` wraps it mid-filename. A data root outside the checkout
    has no relative form, so it stays absolute rather than becoming a ``../`` run.
    """
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sigmoid_runs(correction: Path = DEFAULT_SIGMOID_CORRECTION) -> list[tuple[str, Any]]:
    """Every preserved run of the attribution probe, oldest first, latest last.

    One run is not a measurement of this quantity, and the report used to print one
    as if it were. The probe's own 15 samples agree to a few percent of their median,
    yet two runs of the identical probe disagree about the layer share by more than
    four points — so the number that can honestly be quoted is the band over the runs
    that were kept. They are kept rather than overwritten for exactly that reason,
    and reading only ``SIGMOID_CORRECTION`` would silently throw the rest away.
    """
    directory = correction.parent
    if not directory.is_dir():
        return []
    runs = [
        (path.name, load(path) or {})
        for path in sorted(directory.glob("sigmoid-check*.json"))
    ]
    return [(name, data) for name, data in runs if data]


def band(runs: list[tuple[str, Any]], select: Any) -> tuple[float, float] | None:
    """(min, max) of ``select`` over every run, or None when no run has it.

    ``select`` may return one number or a list of them; the lists are pooled before
    the extremes are taken, which is how one call covers both one-signed synthetic
    distributions and all four timed blocks of the substitution.
    """
    values: list[float] = []
    for _, data in runs:
        picked = select(data)
        for item in picked if isinstance(picked, list) else [picked]:
            if item is not None:
                values.append(item)
    return (min(values), max(values)) if values else None


def median_of(value: Any) -> float | None:
    """The median of a harness stat node, or the value itself if it is already a number.

    The harness stores most timings as ``{median, min, max, stdev, samples}`` and a
    few as bare floats. Normalising here keeps the table code from having to know
    which is which — but it always takes the *median*, never the min, so a table
    cannot quietly become a best-case table.
    """
    if isinstance(value, dict):
        value = value.get("median")
    return value if isinstance(value, (int, float)) else None


def ms(seconds: Any) -> str:
    value = median_of(seconds)
    return "—" if value is None else f"{value * 1e3:.1f}"


def ratio(base: Any, opt: Any) -> str:
    """base/opt, i.e. how many times faster the optimised side is."""
    left, right = median_of(base), median_of(opt)
    if not left or not right:
        return "—"
    return f"{left / right:.2f}×"


def spread(node: Any) -> str:
    """Median and full range from a ``{median,min,max,samples}`` node.

    The range is printed alongside the median on purpose: a median with no spread
    next to it invites reading a 30% swing as a speedup.
    """
    if not isinstance(node, dict):
        return "—"
    samples = node.get("samples") or []
    median = node.get("median")
    if median is None:
        return "—"
    if len(samples) < 2:
        return f"{median * 1e3:.1f} ms (n={len(samples)})"
    return f"{median * 1e3:.1f} ms [{min(samples) * 1e3:.1f}–{max(samples) * 1e3:.1f}]"


def range_only(node: Any) -> str:
    """Just the min–max, for tables whose previous column already shows the median."""
    if not isinstance(node, dict):
        return "—"
    samples = node.get("samples") or []
    if len(samples) < 2:
        return f"n={len(samples)}"
    return f"{min(samples) * 1e3:.1f}–{max(samples) * 1e3:.1f} ms"


def bounds(node: Any) -> tuple[float, float] | None:
    """The min–max pair in milliseconds, or None if there are fewer than two samples."""
    if not isinstance(node, dict):
        return None
    samples = node.get("samples") or []
    if len(samples) < 2:
        return None
    return (min(samples) * 1e3, max(samples) * 1e3)


def stat(node: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(node, dict):
        return None
    value = node.get(key)
    return value if isinstance(value, (int, float)) else None


def decode_self_check(payload: Any, workload: dict[str, Any]) -> float | None:
    """One tree's workload decode rate divided by its raw decode-probe rate.

    The harness times decode twice on the same tree: once as a raw single-token
    loop (``micro.raw_decode``, short context) and once inside each workload
    (``workloads[*].decode_seconds``). The ratio is *not* expected to be 1.0 — a
    workload with a long prompt decodes against a much longer KV than the probe
    does — but it is expected to be **the same on both trees**, because the
    context lengths match. Comparing the two trees' ratios on one row therefore
    cancels the context effect and leaves only machine state.

    Returns ``workload / micro``. The caller does the base-vs-opt comparison.
    """
    micro = stat((payload or {}).get("micro", {}).get("raw_decode"), "ms_per_token")
    generated = stat(workload, "generated_tokens")
    decode = stat(workload.get("decode_seconds"), "median")
    if not micro or not generated or decode is None:
        return None
    return (decode / generated) * 1e3 / micro


#: How far apart two self-check ratios may be before the row is called 待重测.
#:
#: Deliberately **not** the cross-process noise floor: this compares two
#: quantities measured inside the same process, where the only thing that can
#: differ is when each window ran. It is also deliberately not derived from a
#: per-row dispersion, because the rows it guards (decode long/short growth) come
#: from ratios of medians rather than from a sample set of their own. What it is:
#: the smallest multiple at which the two sides stop describing the same machine.
#: A direction-independent rule matters more than the exact value — see
#: :func:`ratio_disagreement`.
SELF_CHECK_TOLERANCE = 1.15


def ratio_disagreement(left: Any, right: Any) -> float | None:
    """``max/min`` of two positive ratios, i.e. how far apart they are.

    Symmetric by construction, which is the whole point. The rule this replaced
    was ``opt > base * 1.15`` **and** ``gain <= 1.0``: it only ever fired on rows
    that claimed a regression, so a row whose two sides disagreed by 2× in the
    *favourable* direction was printed as a clean speedup. Filtering anomalies in
    one direction does not remove noise from a table, it removes the noise that
    was inconvenient — and the surviving rows then look more consistent than the
    measurement ever was.
    """
    a, b = median_of(left), median_of(right)
    if not a or not b:
        return None
    return a / b if a >= b else b / a


def raw_prefill_self_check(
    base_micro: Any, opt_micro: Any, length: str
) -> tuple[float | None, float | None, float | None]:
    """``raw_prefill_full`` against ``engine_prefill`` on the same tree, per side.

    Returns ``(base_ratio, opt_ratio, disagreement)``. Both probes prefill the same
    ``length`` tokens and both project the vocabulary exactly once, so on one tree
    their medians have to land within a few percent of each other — the only real
    difference is one full-length attention versus four growing ones. That makes the
    *ratio* a machine-state sensor available for free inside one process, and the
    two sides' ratios have to agree with each other for the base/opt comparison of
    this row to mean anything.

    Empirically it separates cleanly: across both collected batches and both lengths
    (20 rows) every uncontaminated row lands in 1.004–1.114, while the one row whose
    base side got paged out sits at 11.6. ``SELF_CHECK_TOLERANCE`` (1.15) is the
    threshold already used by the decode and workload self-checks, so this adds no
    new magic number and loosens nothing.

    Symmetric for the same reason :func:`ratio_disagreement` is: a row whose base side
    thrashed prints an enormous *speedup*, and a one-directional rule keyed on
    regressions would wave it through.
    """
    ratios = []
    for micro in (base_micro, opt_micro):
        micro = micro if isinstance(micro, dict) else {}
        raw = stat((micro.get("raw_prefill_full") or {}).get(length), "median")
        engine = stat((micro.get("engine_prefill") or {}).get(length), "median")
        ratios.append(raw / engine if raw and engine else None)
    base_ratio, opt_ratio = ratios
    if base_ratio is None or opt_ratio is None:
        return base_ratio, opt_ratio, None
    disagreement = ratio_disagreement(base_ratio, opt_ratio)
    return base_ratio, opt_ratio, disagreement


#: What a set of paired gains is called when the rounds disagree about the sign.
#:
#: A module-level name rather than an inline string: the same verdict appears in
#: the workload table, the sweep tables and the summary, and the review requires
#: the exact wording 「尚不能确认收益或回退」 — not a per-site paraphrase that
#: quietly softens into "roughly flat" or hardens into "no regression".
INDETERMINATE = "尚不能确认收益或回退"

#: What a row is called when the rounds it *kept* cannot answer the question at all.
#:
#: Two labels rather than one, because "the gains straddle 1.00"
#: (:data:`INDETERMINATE`) and "this experiment cannot say" are different findings
#: and pooling them would let a design failure read as a measured tie. Both are
#: kept out of the 收益／回退 counts, and both say which rule failed so a reader can
#: tell a driver bug from a thin sample.
#:
#: ``UNORDERED`` fires when the retained rounds are not AB/BA balanced, *including*
#: when their labels carry no position at all: the second process in each pair
#: inherits a warm page cache, so a set that is not balanced hands that advantage to
#: one tree, and a set whose order was never recorded cannot show it did not.
UNORDERED = "顺序不均衡或未记录，尚不能确认收益或回退"

#: ``INSUFFICIENT`` fires when fewer than two paired gains survived, which is a
#: different test from :func:`gain_verdict`'s own: that one asks whether a
#: distribution straddles 1.0, this one asks whether there is a distribution.
INSUFFICIENT = "有效轮次不足，尚不能确认收益或回退"

#: The verdicts that mean "no direction claimed". A tuple so the membership tests
#: below cannot drift apart from the labels themselves.
UNSETTLED: tuple[str, ...] = (UNORDERED, INSUFFICIENT)

#: How many retained paired gains a row needs before any direction is claimed. Two
#: is the floor at which :func:`gain_verdict` can answer at all; naming it here
#: means the acceptance criterion is declared in one place, ahead of the data,
#: rather than being whatever the code happened to require.
MIN_RETAINED_ROUNDS = 2


def spread_ratio(values: Any) -> float | None:
    """``max/min`` of a positive sample set: dispersion as a multiple, not a percent.

    None when there are fewer than two usable numbers. A single observation has no
    dispersion, and reporting 1.00 for it would say "this did not vary" when the
    truth is "this was measured once".
    """
    positive = [v for v in (values or []) if isinstance(v, (int, float)) and v > 0]
    if len(positive) < 2:
        return None
    return max(positive) / min(positive)


def gain_verdict(gains: Any) -> str | None:
    """Classify paired per-round gains as 收益 / 回退 / :data:`INDETERMINATE`.

    The test is whether the gains *straddle* 1.0. Symmetric by construction — the
    function never looks at which side of 1.0 the point estimate sits on, so it
    cannot be more willing to call a slowdown indeterminate than a speedup or vice
    versa. That is the property the review asked to have tested, and the reason the
    rule is expressed on the distribution rather than on a threshold: a threshold
    ("ignore anything under 3%") has to be picked, and every pick is asymmetric in
    effect because the gains it filters are not symmetrically distributed.

    None when fewer than two rounds are available. One round is one observation,
    and no verdict — including a favourable one — can be derived from it.
    """
    values = [g for g in (gains or []) if isinstance(g, (int, float)) and g > 0]
    if len(values) < 2:
        return None
    low, high = min(values), max(values)
    if low <= 1.0 <= high:
        return INDETERMINATE
    return "收益" if low > 1.0 else "回退"


def sign_test(gains: list[float], threshold: float = 1.0) -> tuple[int, int, float]:
    """Exact two-sided sign test on whether paired gains sit above ``threshold``.

    Returns ``(usable, above, p)``. Gains exactly equal to the threshold are ties
    and are dropped, which is the standard sign-test convention and also the
    conservative one here: keeping them would count a dead heat as evidence.

    Exact rather than normal-approximated because the arithmetic is free —
    ``math.comb`` on integers plus a ``Fraction`` for the ratio, so no term ever
    passes through a float large enough to lose the tail. Worth doing, since the
    whole point of quoting a p-value is that a reader can check it.

    The p-value answers "could a coin flipped this many times land this lopsided".
    It does **not** answer "is every row faster": the rows are not independent
    (the same process contributes several quantities, and one machine contributes
    five models), so a pooled p-value overstates the evidence. That caveat is
    printed next to the number, not left to the reader.
    """
    usable = [g for g in gains if isinstance(g, (int, float)) and g != threshold]
    n = len(usable)
    if not n:
        return 0, 0, 1.0
    above = sum(1 for g in usable if g > threshold)
    tail = sum(math.comb(n, k) for k in range(above, n + 1))
    mirror = sum(math.comb(n, k) for k in range(0, n - above + 1))
    p = 2 * Fraction(min(tail, mirror), 2**n)
    return n, above, float(min(p, Fraction(1)))


def within_between(processes: Any) -> tuple[float | None, float | None]:
    """``(within-process spread, between-process spread)`` for one side.

    ``processes`` is a list with one entry per independent process, each entry the
    list of that process's own repeat samples. The two numbers answer different
    questions and are reported separately for that reason:

    * *within* — the median of each process's own ``max/min``. How reproducible one
      process is with itself.
    * *between* — the ``max/min`` of the per-process medians. How far apart two
      processes started minutes apart land on the same quantity.

    Collapsing these into a single "noise floor" is exactly what produced the
    round-1 report's 3% number: a within-process range of ±0.5% sits next to a
    between-process gap of 3%, and quoting one of them without the other lets a
    reader treat the tight one as the resolution of the comparison.
    """
    withins = [spread_ratio(samples) for samples in (processes or [])]
    withins = [value for value in withins if value is not None]
    medians = [
        statistics.median(samples)
        for samples in (processes or [])
        if isinstance(samples, (list, tuple)) and samples
    ]
    return (
        statistics.median(withins) if withins else None,
        spread_ratio(medians),
    )


#: Where ``run_noise_floor.sh`` writes, relative to the data root.
NOISE_DIR = "noise-floor"

#: The run root the paired experiment comes from, or ``None`` when there is no paired
#: experiment to read. Decided once, in :func:`resolve_paired_root`, before a single
#: section is rendered — :func:`paired_directory` only appends ``NOISE_DIR``.
#:
#: It cannot simply live under ``PERF``: ``artifacts/perf-round1{,-rerun}/`` hold the
#: raw evidence behind the published report and the drivers refuse to write into them,
#: so a re-measurement has to come from a directory of its own. Unlike the sigmoid
#: correction this one *replaces* rather than supplements — it re-runs every model at
#: the same configuration with more rounds, so there is no per-row question of which
#: of the two a number came from. The old directory stays on disk as the evidence
#: behind the withdrawn 「70/73」, and :func:`paired_directory`'s answer is printed
#: into §7.3, so the switch is stated rather than inferred from a table that changed
#: shape.
PAIRED_ROOT: Path | None = None


def paired_side_and_round(path: Path) -> tuple[int, str] | None:
    """``(round, side)`` read off a paired run's filename, or None if it is not one.

    Model names contain hyphens, so the round and side are split off from the
    *right*; splitting from the left would put part of the model name in the round.
    Shared by :func:`paired_rounds` and by the validation in
    :func:`resolve_paired_root` — two copies of this rule would let "is this
    directory a paired experiment" and "read this paired experiment" disagree, and
    the disagreement would surface as a directory that validates and then renders
    as MISSING.
    """
    head, _, tail = path.stem.rpartition("-r")
    if not head or "-" not in tail:
        return None
    round_text, side = tail.split("-", 1)
    if not round_text.isdigit() or side not in ("base", "opt"):
        return None
    return int(round_text), side


def paired_files(directory: Path) -> list[Path]:
    """Every file in ``directory`` whose name :func:`paired_rounds` would accept."""
    return [
        path
        for path in sorted(directory.glob("*-r*-*.json"))
        if paired_side_and_round(path) is not None
    ]


def paired_directory() -> Path | None:
    """The ``noise-floor/`` this report reads, or None if the experiment was never run.

    One directory, not a search. This used to try ``PAIRED_ROOT`` first and then
    ``PERF``'s own, returning whichever existed — so a mistyped or not-yet-collected
    ``--paired-data`` rendered the *previous* experiment's verdicts under a header
    naming the directory the operator had asked for. The candidates are now resolved
    once, in :func:`resolve_paired_root`, where an unusable explicit one is an error.

    ``PAIRED_ROOT`` is the *run root*, so ``NOISE_DIR`` is appended here and the
    result is existence-checked — but there is exactly one candidate. An explicit
    root that has no experiment under it reports "not run" rather than reaching for
    the data root's, which is the fallback that produced a report citing a directory
    it never opened.
    """
    if PAIRED_ROOT is None:
        return None
    directory = PAIRED_ROOT / NOISE_DIR
    return directory if directory.is_dir() else None


def resolve_paired_root(raw: str | None, data_root: Path) -> Path | None:
    """Which run root the paired experiment comes from, decided before anything renders.

    ``None`` comes back only when no paired experiment was asked for *and* the data
    root does not hold one. An explicit ``--paired-data`` never falls back to the
    data root's own: naming a directory is how an operator says which experiment a
    verdict came from, and quietly reading a different one produces a report that
    cites a run it never opened. That is the shape of the round-2 「70/73」 error,
    reached from the other direction.

    Judged here rather than at write time for the same reason: by the time the
    document is on disk, the previous good copy is already gone.
    """
    if not raw:
        return data_root if (data_root / NOISE_DIR).is_dir() else None

    requested = _resolve(raw, ROOT)
    problem = paired_root_problem(requested, requested / NOISE_DIR)
    if problem is None:
        return requested

    print(
        "make_report: --paired-data 指定的配对实验读不出来，本次不生成任何产物：",
        file=sys.stderr,
    )
    print(f"  - --paired-data={raw}（解析为 {requested}）：{problem}", file=sys.stderr)
    print(
        f"  不会退回 --data 下的 `{NOISE_DIR}/`：那会让正文引用一批它没有读过的数据。"
        "改成本次要判读的那个运行根，或者干脆不传 --paired-data。",
        file=sys.stderr,
    )
    raise SystemExit(1)


def paired_root_problem(requested: Path, directory: Path) -> str | None:
    """Why ``--paired-data requested`` cannot be read, or None when it can."""
    if not requested.exists():
        return "不存在（路径拼错，或这次实验还没有跑）"
    if not requested.is_dir():
        return "是一个文件，不是运行根目录"
    if not directory.is_dir():
        return f"下面没有 `{NOISE_DIR}/` 子目录，所以这不是 run_noise_floor.sh 的运行根"
    try:
        readable = any(load(path) for path in paired_files(directory))
    except (OSError, ValueError):
        return f"`{NOISE_DIR}/` 里有读不出来的 JSON（文件损坏，或不是本工具写的形状）"
    if not readable:
        return (
            f"`{NOISE_DIR}/` 里读不出任何配对 JSON——目录是空的，"
            "或者文件名不成 `<model>-r<round>-<base|opt>.json` 的形状"
        )
    return None


#: One sample inside a node blowing up past the node's own median. Catches the
#: "7 samples agree except one stalled for 100 s" shape, which a median hides and
#: which a whole-process wall clock dilutes below any sane threshold.
SPIKE_LIMIT = 2.0

#: A whole round sitting far above that quantity's other rounds. Needed because a
#: spike test cannot see it: when all seven samples of a round move up together,
#: ``max/median`` stays near 1. Measured against the median *of the round medians*
#: rather than the mean, so one contaminated round cannot drag the reference it is
#: being compared to.
LEVEL_LIMIT = 1.5

#: Filled by :func:`paired_noise_section` while the sweep tables are rendered, and
#: read by :func:`write_doc` afterwards. §7.2 and §9.1 both have to say whether the
#: paired experiment exists this round, and recomputing its verdicts a second time
#: would give the prose a way to disagree with the table it summarises — the same
#: reason ``noise_floor_percent`` is threaded through ``headline_numbers()``.
#: ``main()`` renders every section before calling ``write_doc``, so this is
#: populated by the time it is read; it stays ``None`` when the experiment did not
#: run, which is exactly the case the prose has to detect.
PAIRED_SUMMARY: dict[str, Any] | None = None

#: ``{short model name: (base-side %, opt-side %)}`` between-process dispersion for
#: the one quantity §9.2 has to compare optimisation 1's share against. Populated
#: alongside ``PAIRED_SUMMARY``. Per model on purpose: §9.2 used to compare the
#: band's global maximum against ``noise_floor_percent``, which is the *median of the
#: two sweep models* — a cross-model comparison that came out false for the very
#: model that set the maximum. Empty when the paired experiment did not run.
PAIRED_BETWEEN: dict[str, tuple[float, float]] = {}


def _reset_run_state() -> None:
    """Put every input-derived global back to its import-time value.

    ``main()`` reassigns the five names it declares ``global``, and that is enough for
    one process per report — the CLI's only shape. It is not enough for a second
    ``main()`` in the same process, because three of the names above are *not*
    reassigned by ``main()`` at all:

    * ``SIGMOID_CORRECTION`` is only overwritten when ``<data>/sigmoid/`` exists, so a
      root without one kept the previous root's correction.
    * ``PAIRED_SUMMARY`` is assigned only inside :func:`paired_noise_section`, which
      returns early when the root holds no paired runs. "This round did not run the
      experiment" therefore left the *previous* experiment's summary in place, and
      :func:`write_doc` — which reads it two thousand lines later — printed 「N 个模型 ×
      M 轮」 into §7.2 of a report whose own §7.3 table said 未运行.
    * ``PAIRED_BETWEEN`` is mutated per model and never cleared, so §9.2's comparison
      could quote a model the current experiment never measured.

    Reset rather than "assign the default at the top of ``main``", because two of the
    three defaults are not ``None`` and they live next to their declarations; a second
    copy of them in ``main`` is the drift this function exists to remove.
    """
    global PERF, OUT, DOC, SIGMOID_CORRECTION, PAIRED_ROOT
    global PAIRED_SUMMARY, PAIRED_BETWEEN
    PERF = DEFAULT_PERF
    OUT = PERF / "report-tables.md"
    DOC = ROOT / "docs" / DEFAULT_DOC_NAME
    SIGMOID_CORRECTION = DEFAULT_SIGMOID_CORRECTION
    PAIRED_ROOT = None
    PAIRED_SUMMARY = None
    PAIRED_BETWEEN = {}


def contaminated_rounds(
    per_round: dict[int, list[float]],
) -> dict[int, tuple[str, float]]:
    """Which rounds of one (quantity, side) series are unusable, and why.

    ``per_round`` maps a round number to that process's own repeat samples.
    Returns ``{round: (reason, value)}`` with reason ``"spike"`` or ``"level"``.

    Both tests are ratios of a quantity to its own scale, so neither needs a
    threshold in milliseconds and neither can be satisfied by a fast machine: they
    fire on shape, not on magnitude. They are also deliberately symmetric in base
    and opt — an outlier rule that only ever discards the slower side would
    manufacture the speedup it was supposed to be testing.

    Rounds where fewer than two samples survived, or where the median is not
    positive, are left unflagged: there is nothing to compare them against, and
    inventing a verdict for them is worse than reporting them as unusable upstream.
    """
    medians: dict[int, float] = {}
    for number, samples in per_round.items():
        usable = [s for s in (samples or []) if isinstance(s, (int, float)) and s > 0]
        if len(usable) >= 2:
            medians[number] = statistics.median(usable)
    if not medians:
        return {}
    reference = statistics.median(list(medians.values()))
    flagged: dict[int, tuple[str, float]] = {}
    for number, samples in per_round.items():
        usable = [s for s in (samples or []) if isinstance(s, (int, float)) and s > 0]
        if len(usable) < 2:
            continue
        median = statistics.median(usable)
        spike = max(usable) / median
        level = median / reference if reference > 0 else 0.0
        if spike > SPIKE_LIMIT:
            flagged[number] = ("spike", spike)
        elif level > LEVEL_LIMIT:
            flagged[number] = ("level", level)
    return flagged


def paired_rounds(model: str, extract: Any) -> dict[str, Any] | None:
    """The AB/BA paired runs for one model, grouped into rounds.

    ``extract`` maps one payload to the number under test, so the same reader
    serves the micro probes and every workload without knowing which of them it is
    being pointed at. Returns None when the directory does not exist — meaning this
    experiment has not been run — which callers must report as a gap rather than
    fill in from some other run.

    Order balance is checked from each payload's own ``label`` rather than trusted
    from the filename: the whole point of an AB/BA design is that neither tree
    systematically runs second, and a driver bug that broke that would otherwise be
    invisible in the output.
    """
    directory = paired_directory()
    if directory is None:
        return None
    rounds: dict[int, dict[str, Any]] = {}
    for path in sorted(directory.glob(f"{model}-r*-*.json")):
        parsed = paired_side_and_round(path)
        if parsed is None:
            continue
        number, side = parsed
        payload = load(path)
        if not payload:
            continue
        label = str(payload.get("label") or "")
        # rpartition yields "" when the marker is absent, so position is always a
        # string and the digit test below cannot be handed None. A label without
        # the marker means the run did not record where it sat in its round, which
        # is reported as "unknown" rather than guessed from the filename order.
        position = label.rpartition("-pos")[2]
        entry = rounds.setdefault(number, {})
        entry[side] = {
            "payload": payload,
            "value": extract(payload),
            "position": int(position) if position.isdigit() else None,
            "file": path.name,
        }
    complete = {
        number: sides
        for number, sides in sorted(rounds.items())
        if "base" in sides and "opt" in sides
    }
    if not complete:
        return None
    balanced, base_first = _order_balance(complete)
    return {
        "rounds": complete,
        "round_count": len(complete),
        "incomplete_rounds": sorted(set(rounds) - set(complete)),
        "base_first_rounds": base_first,
        "order_balanced": balanced,
    }


def _order_balance(rounds: dict[int, dict[str, Any]]) -> tuple[bool | None, int]:
    """How many rounds ran base first, and whether that is exactly half.

    ``None`` rather than ``False`` when the labels carry no position at all, so
    "not recorded" is never reported as "unbalanced" — and never as "balanced"
    either, which is the failure that would actually hide a driver bug.

    ``None`` for an empty round set for the same reason: ``0 * 2 == 0`` is True, so
    without this a row the contamination filter stripped of every round would print
    「顺序均衡：是」 about rounds it does not have.
    """
    if not rounds:
        return None, 0
    positions = {
        number: (sides["base"]["position"], sides["opt"]["position"])
        for number, sides in rounds.items()
    }
    if any(None in pair for pair in positions.values()):
        return None, 0
    # A round where both sides claim the same position means the labels do not
    # describe this design, so the balance question has no answer.
    if any(pair[0] == pair[1] for pair in positions.values()):
        return None, 0
    base_first = sum(1 for pair in positions.values() if pair[0] == 1)
    return base_first * 2 == len(rounds), base_first


def suspect_self_check(base_check: Any, opt_check: Any) -> bool:
    """Whether a row's two self-checks disagree too much for the row to be read.

    Takes only the two self-checks — **not** the gain. That signature is the fix:
    the predicate this replaced was
    ``gain <= 1.0 and opt_check > base_check * 1.15``, so a row had to claim a
    *slowdown* before its self-checks were ever looked at. A row claiming a 3×
    speedup on self-checks that disagreed by 2× was printed as a clean result.
    Passing the gain in would let that asymmetry come back; the caller has the gain
    available and deliberately does not supply it.

    A row with no self-check at all is suspect too: "cannot check" is not "passed
    the check", and reporting it as trustworthy would make a run that never
    collected micro data look better-covered than one that did.
    """
    disagreement = ratio_disagreement(base_check, opt_check)
    if disagreement is None:
        return True
    return disagreement > SELF_CHECK_TOLERANCE


def decode_median(node: Any) -> float | None:
    """The ms/token this row reports, as a number.

    Shared by the table cell and the ratio column, because those two must be the
    *same* quantity: a row may show a multi-window median while ``ms_per_token``
    still holds the legacy single-window scalar, and printing one next to a ratio
    computed from the other is the kind of mismatch that makes a table internally
    inconsistent in a way no reader can spot.
    """
    windows = node.get("ms_per_token_windows")
    samples = windows.get("samples") if isinstance(windows, dict) else None
    if samples:
        return statistics.median(samples)
    value = node.get("ms_per_token")
    return value if isinstance(value, (int, float)) else None


def decode_window_cell(node: Any) -> tuple[str, str, str]:
    """``(median ms/token, window range, window count)`` for a ``raw_decode`` node.

    Reads the ``ms_per_token_windows`` block that the repeating harness writes.
    Falls back to the legacy scalars, and in that case the count column reads
    ``1（单窗口）`` rather than ``1``: a bare 1 invites reading it as one repeat of
    a per-token sample, when it is one observation of the whole quantity with no
    dispersion attached at all.
    """
    median = decode_median(node)
    windows = node.get("ms_per_token_windows")
    samples = windows.get("samples") if isinstance(windows, dict) else None
    if median is None:
        return ("—", "—", "0")
    if samples:
        return (
            f"{median:.2f} ms/tok",
            f"{min(samples):.2f}–{max(samples):.2f} ms/tok",
            str(len(samples)),
        )
    return (f"{median:.2f} ms/tok", "无（单窗口）", "1（单窗口）")


def single_window_note(model: str, label: str, base_node: Any, opt_node: Any) -> str | None:
    """Name a decode row that was not measured over repeated windows on both sides."""
    lacking = [
        side
        for side, node in (("base", base_node), ("opt", opt_node))
        if not (isinstance(node.get("ms_per_token_windows"), dict)
                and node["ms_per_token_windows"].get("samples"))
    ]
    if not lacking:
        return None
    return (
        f"{SHORT[model]} {label}：{'／'.join(lacking)} 侧只有 1 个窗口，"
        "是**单窗口观察值**"
    )


def missing(line: list[str], what: str) -> None:
    line.append(f"\n**MISSING: {what}** — 这一节的输入还没有产生，不要在报告里填数。\n")


#: Every JSON this report reads, as paths relative to the data root. Listed
#: rather than globbed so that a stray file in the directory cannot quietly join
#: the identity table and make the run look better-covered than it is. The sweep
#: names follow run_sweep.sh's ``${model}-${label}-${tag}.json``, where the label
#: is ``default`` under ``baseline/`` and ``opt`` under ``sweep-opt/``.
SWEEP_TAGS = ("t-default", "t1", "t4", "t8", "c64", "c256", "c512")
IDENTITY_SOURCES: tuple[str, ...] = (
    *(f"ab/{model}-{side}.json" for model in MODELS for side in ("base", "opt")),
    *(f"ab-longctx/{model}-{side}.json" for model in MODELS for side in ("base", "opt")),
    *(f"baseline/{model}-default.json" for model in MODELS),
    *(
        f"baseline/{model}-default-{tag}.json"
        for model in SWEEP_MODELS
        for tag in SWEEP_TAGS
    ),
    *(
        f"sweep-opt/{model}-opt-{tag}.json"
        for model in SWEEP_MODELS
        for tag in SWEEP_TAGS
    ),
    *(f"projection/{model}.json" for model in MODELS),
    "vision/base.json",
    "vision/opt.json",
    "documents/base.json",
    "documents/opt.json",
    "sigmoid-check.json",
)


# --------------------------------------------------------------------------
# source identity
# --------------------------------------------------------------------------
def section_identity(line: list[str]) -> None:
    """Which code produced these numbers, per file, by hash.

    A timing is a claim about a particular ``src/``, and this round's optimised
    side is *uncommitted*, so there is no revision to name. ``source_identity``
    is the block :mod:`tools.perf.source_identity` writes into every harness
    payload for exactly that reason: per-file sha256 of the measured tree, of the
    harness that measured it, plus whatever git can say.

    JSON produced before that block existed has no identity, and this section
    says so per file instead of leaving the row out — an absent row reads as
    "not applicable", a named gap reads as "not verified".
    """
    line.append("## 源码身份\n")
    line.append(f"本节读取的数据根目录：`{repo_relative(PERF)}/`。\n")
    paired_source = paired_directory()
    if paired_source is not None and paired_source.parent != PERF:
        line.append(
            "**§7.3 的配对实验读的是另一个根**："
            f"`{repo_relative(paired_source.parent)}/`（`--paired-data`）。"
            f"`{repo_relative(PERF)}/{NOISE_DIR}/` 仍在磁盘上、未被改写，"
            "它是已撤回的那个计数背后的原始证据；重测加在它旁边，不替换它。\n"
        )

    # Before the branches, not after: this section has two early returns (no
    # measurements at all, and measurements that predate the identity block), and the
    # published round takes the second one. Appending the generator's own hash at the
    # end would have left exactly the renders that most need attribution — the ones
    # whose measurements cannot be hash-checked — without any.
    digest, script, script_lines = generator_identity()
    line.append(
        f"**生成本文的脚本自身**：`{script}`，{script_lines} 行，"
        f"sha256 `{digest}`。"
        "本节下面核对的是*被测*的树，这一行核对的是*写结论*的那一个文件："
        "本文每个数字和每句判定都由它从原始 JSON 算出，而它在审查期间被反复修改过，"
        "所以同一批 JSON 用不同版本的脚本重跑会给出不同措辞。"
        "它在工作区里、不在任何提交里，没有修订号可引，**这个哈希是它唯一的稳定标识**。"
        f"复核命令：`shasum -a 256 {script}`"
        "（与上面的值不符，说明你手里的本文不是这一版脚本生成的）。\n"
    )

    found: list[tuple[str, dict[str, Any]]] = []
    absent: list[str] = []
    for name in IDENTITY_SOURCES:
        payload = load(PERF / name)
        if payload is None:
            continue
        identity = payload.get("source_identity")
        if isinstance(identity, dict):
            found.append((name, identity))
        else:
            absent.append(name)

    if not found and not absent:
        missing(line, "任何带身份块的测量输出")
        return

    if not found:
        line.append(
            f"**这 {len(absent)} 份 JSON 都没有 `source_identity` 块**——它们产生于"
            "身份记录接入 harness 之前。可以核到的只有：基线树 "
            f"`{repo_relative(BASELINE_TREE)}/` 由 `tools/perf/baseline_tree.py` "
            "逐文件哈希核对为 `8968e39`；优化侧是当时的未提交工作区，**已无法事后重建**，"
            "因此没有它的哈希可列。这是这批数据的固有缺口，不是可以补测的项目："
            "补测只能产生**新的**一轮数据（见 §1.3 的独立输出目录），"
            "不能给已经跑完的这一轮补上它当时没记下的身份。\n"
        )
        # Grouped by directory rather than listed per file: 63 rows of "无身份块"
        # would bury the one fact they state, and a reader who needs the exact
        # file list has IDENTITY_SOURCES and the data directory.
        groups: dict[str, list[str]] = {}
        for name in absent:
            parent = Path(name).parent.as_posix()
            groups.setdefault("" if parent == "." else f"{parent}/", []).append(
                Path(name).name
            )
        line.append("| 目录 | 无身份块的 JSON |")
        line.append("| --- | --- |")
        for directory, names in sorted(groups.items()):
            shown = ", ".join(f"`{n}`" for n in names[:4])
            more = f" 等 {len(names)} 份" if len(names) > 4 else ""
            where = f"`{directory}`" if directory else "（数据根目录）"
            line.append(f"| {where} | {shown}{more} |")
        line.append("")
        return

    # One row per distinct measured tree. The digest covers src/ + tools/perf/
    # hashes only, so two runs of the same code on different days share a row —
    # which is the property that makes an A/B pair comparable at all.
    trees: dict[str, dict[str, Any]] = {}
    for name, identity in found:
        key = identity.get("digest") or "<no digest>"
        entry = trees.setdefault(
            key,
            {
                "roots": set(),
                "harnesses": set(),
                "harness_sha": {},
                "labels": set(),
                "heads": set(),
                "files": 0,
                "json": [],
            },
        )
        entry["roots"].add(identity.get("root") or "?")
        harness = identity.get("harness") or {}
        harness_name = Path(harness.get("path") or "?").name
        entry["harnesses"].add(harness_name)
        if harness.get("sha256"):
            entry["harness_sha"][harness_name] = harness["sha256"]
        if identity.get("revision_label"):
            entry["labels"].add(identity["revision_label"][:12])
        git = identity.get("git") or {}
        if git.get("head"):
            entry["heads"].add(git["head"][:12])
        entry["files"] = max(entry["files"], len(identity.get("source") or {}))
        entry["json"].append(name)

    line.append("| 被测树 digest（前 12 位） | 来源 | src 文件数 | harness | 覆盖 JSON |")
    line.append("| --- | --- | --- | --- | --- |")
    for key, entry in sorted(trees.items(), key=lambda kv: -len(kv[1]["json"])):
        origin = []
        if entry["labels"]:
            origin.append("revision " + "/".join(sorted(entry["labels"])))
        if entry["heads"]:
            origin.append("HEAD " + "/".join(sorted(entry["heads"])))
        if not origin:
            origin.append("未提交工作区（只有哈希可核）")
        line.append(
            f"| `{key[:12]}` | {'，'.join(origin)} | {entry['files']} | "
            f"{', '.join(sorted(entry['harnesses']))} | {len(entry['json'])} 份 |"
        )
    line.append("")

    # Classified from the identity block, not from the file name: the sweep
    # directories are called ``baseline/`` and ``sweep-opt/`` while the A/B files
    # are called ``-base``/``-opt``, so a name-based guess would mislabel the
    # whole sweep. What decides it is which tree was imported.
    def tree_role(identity: dict[str, Any]) -> str:
        git = identity.get("git") or {}
        root = identity.get("root") or ""
        if identity.get("revision_label") or root.endswith("baseline-tree"):
            return "baseline"
        return "working tree" if git.get("dirty") else "committed HEAD"

    roles: dict[str, set[str]] = {}
    for _name, identity in found:
        roles.setdefault((identity.get("digest") or "?")[:12], set()).add(tree_role(identity))
    line.append(
        f"{len(found)} 份 JSON 带身份块，归为 {len(trees)} 棵被测树，"
        f"角色分布：{', '.join(f'{k}={sorted(v)}' for k, v in sorted(roles.items()))}。"
        + (
            " **同一个 digest 同时被当作基线和优化侧使用，A/B 不成立。**"
            if any(len(v) > 1 for v in roles.values())
            else " 每个 digest 只对应一种角色，两侧测的确实是两棵不同的树。"
        )
        + "\n"
    )

    # Per tree, not per JSON: every file in one tree shares a digest by construction.
    # Which digests still reproduce is derived, not asserted. An earlier draft of this
    # paragraph claimed none of them did, and that was false: the baseline tree is a
    # ``git archive`` snapshot under ``artifacts/`` with an empty ``tooling`` half, so
    # this round's edits never touched it and its digest still hashes back. Only the
    # working-tree rows moved, and only in the half that renders prose.
    representative: dict[str, dict[str, Any]] = {}
    for _name, identity in found:
        representative.setdefault(identity.get("digest") or "<no digest>", identity)

    def half(identity: dict[str, Any], key: str, label: str) -> tuple[str, list[str]]:
        matched, changed = recorded_map_still_matches(identity, key)
        total = len(identity.get(key) or {})
        if not total:
            return f"{label} 那一半为空", []
        if not changed:
            return f"{label} {matched}/{total} 逐个相符", []
        shown = "、".join(f"`{c}`" for c in changed[:4])
        return (
            f"{label} **{len(changed)}/{total} 个文件已变**（{shown}"
            + ("…" if len(changed) > 4 else "")
            + "）",
            changed,
        )

    # Whether a tree's *own harness* moved is reported separately from whether any tool
    # moved, because the two answer different questions: a changed ``make_report.py``
    # only re-renders prose, but a changed harness means the timings came out of a
    # version of the collecting script the reader no longer has. An earlier draft named
    # one changed file and called it harmless ("不参与任何计时") without checking whether
    # it was a harness — exactly the conflation this split exists to prevent. ``src/`` is
    # file-by-file identical either way, so *what* was measured is unaffected; only the
    # reproducibility of the timing script is.
    #
    # Checked against every harness that collected a JSON in the tree, not just the
    # representative block's one: a tree groups JSONs from several harnesses (the table
    # above lists them), and the tooling-only branch's "采集它们的 harness 也一个字没变"
    # is a claim about all of them, so checking one would let a changed sibling through.
    reproducible: list[str] = []
    tooling_only: list[str] = []
    harness_moved: list[str] = []
    source_moved: list[str] = []
    for key, identity in sorted(representative.items()):
        src_text, src_changed = half(identity, "source", "`src/`")
        tool_text, tool_changed = half(identity, "tooling", "`tools/perf/`")
        tree = trees.get(key) or {}
        changed_names = {Path(c).name for c in tool_changed}
        moved = sorted(tree.get("harnesses", set()) & changed_names)
        entry = f"`{key[:12]}`（{src_text}；{tool_text}）"
        if src_changed:
            source_moved.append(entry)
        elif moved:
            cited = "、".join(
                f"`{name}`（当时那一版 sha256 "
                f"`{(tree.get('harness_sha', {}).get(name) or '?')[:16]}…`）"
                for name in moved
            )
            harness_moved.append(
                f"{entry}——**采集它的脚本本身在这之后被改过：{cited}**，"
                "记录还在，但工作区里已经不是它了"
            )
        elif tool_changed:
            tooling_only.append(entry)
        else:
            reproducible.append(entry)

    total_trees = len(representative)
    parts = []
    if reproducible:
        parts.append(
            f"**{len(reproducible)}/{total_trees} 棵树的 digest 现在仍能原样复现**"
            "（`source_identity.py --src <树> --digest-only` 逐棵给出上表的值）："
            + "；".join(reproducible)
            + "。"
        )
    if tooling_only:
        parts.append(
            f"**另 {len(tooling_only)} 棵的 digest 复现不出来了，变的是工具那一半——"
            "既不是被测代码，也不是采集它的那份脚本**："
            + "；".join(tooling_only)
            + "。原因是 digest 同时覆盖 `src/` 与 `tools/perf/`，而 "
            "`tools/perf/make_report.py`（生成本文的脚本）在审查期间被反复修改过。"
            "**计时数据仍然对得上它们当时测的那份 `src/`**，"
            "采集它们的 harness 也一个字没变。"
        )
    if harness_moved:
        parts.append(
            f"**还有 {len(harness_moved)} 棵，变的正是采集它的那份脚本**："
            + "；".join(harness_moved)
            + "。这不改变「测的是哪份被测代码」（`src/` 仍逐个相符），"
            "但意味着那一批计时出自读者手上已经没有的一版脚本，"
            "**要重跑必须先回到记录里那个 sha256 指认的版本**。"
        )
    if source_moved:
        parts.append(
            f"**{len(source_moved)} 棵树的被测代码本身已经和当时不一样了**："
            + "；".join(source_moved)
            + "。**这几棵树上的计时不能再用当前源码解释**，只能重测。"
        )
    if tooling_only or harness_moved or source_moved:
        parts.append(
            "要复现上面任何一个 digest，先把 `tools/perf/` 的改动搁置"
            "（`git stash push tools/perf/`），或者按 §10 第 3 步重跑一轮采集，"
            "让新一轮自己记下新的 digest。"
        )
    line.append("".join(parts) + "\n")
    if absent:
        line.append(
            f"另有 {len(absent)} 份 JSON 没有身份块：{', '.join(f'`{n}`' for n in absent[:6])}"
            + ("…" if len(absent) > 6 else "")
            + "。它们产生于身份记录接入之前，**不能用哈希核对**。\n"
        )
    line.append(
        "复核命令：`.venv/bin/python tools/perf/source_identity.py --src <树>`"
        "（打印全部逐文件哈希），加 `--digest-only` 只打印上表的 digest。"
        "基线树由 `.venv/bin/python tools/perf/baseline_tree.py` 从 `8968e39` "
        "重建并逐文件校验。\n"
    )


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def section_environment(line: list[str]) -> None:
    line.append("## 测量环境\n")
    sources = []
    for model in MODELS:
        for side in ("base", "opt"):
            payload = load(PERF / "ab" / f"{model}-{side}.json")
            if payload:
                sources.append((f"ab/{model}-{side}.json", payload))
    for name in ("vision/base.json", "vision/opt.json", "documents/base.json", "documents/opt.json"):
        payload = load(PERF / name)
        if payload:
            sources.append((name, payload))
    if not sources:
        missing(line, "任何 A/B 输出")
        return

    env = sources[0][1]["environment"]
    threads = sources[0][1]["threads"]
    line.append("| 项目 | 值 |")
    line.append("| --- | --- |")
    line.append(f"| 机器 | {env['cpu']}，{env['cpu_count_logical']} 逻辑核 |")
    line.append(f"| 系统 | {env['platform']} |")
    line.append(f"| Python / NumPy | {env['python']} / {env['numpy']} |")
    blas = env["blas"]
    line.append(f"| BLAS | {blas['name']}（version {blas['version']}） |")
    line.append(f"| 采样 | greedy |")

    requested = threads.get("requested")
    line.append(
        f"| 请求线程数 | {requested if requested is not None else '未指定（出厂默认）'} |"
    )
    line.append(
        f"| **有效线程数** | `{threads.get('effective_thread_count')}` —— "
        "**不可观测，记为 null 而不是猜** |"
    )
    exported = env.get("thread_env", {})
    unset = all(value is None for value in exported.values())
    line.append(
        f"| BLAS 线程环境变量 | {'全部未设置' if unset else json.dumps(exported)} |"
    )

    # Every payload must agree, or the table above describes only some runs.
    disagree = []
    for name, payload in sources:
        other = payload.get("environment", {})
        if (other.get("python"), other.get("numpy")) != (env["python"], env["numpy"]):
            disagree.append(name)
        if payload.get("threads", {}).get("effective_thread_count") is not None:
            disagree.append(f"{name} (claims an effective thread count)")
    if disagree:
        line.append(f"\n**环境不一致：{', '.join(sorted(set(disagree)))}**\n")
    else:
        line.append(
            f"\n以上取自 {len(sources)} 份输出，逐份核对 Python / NumPy 版本与"
            "「有效线程数不可观测」这一项，全部一致。\n"
        )


# --------------------------------------------------------------------------
# opt 1 + opt 2: micro benchmarks, base tree vs opt tree
# --------------------------------------------------------------------------
def ab_payloads(root: Path | None = None) -> dict[str, tuple[Any, Any]]:
    """The paired ``ab/`` payloads, keyed by model directory.

    Shared by §3's table and by §9.4's 待重测 row so both read the same cells; a
    row that quotes a ratio the table above it does not contain is worse than no
    row at all, because the reader has no way to tell which of the two is stale.
    """
    directory = (root or PERF) / "ab"
    payloads = {}
    for model in MODELS:
        base = load(directory / f"{model}-base.json")
        opt = load(directory / f"{model}-opt.json")
        if base and opt:
            payloads[model] = (base, opt)
    return payloads


def decode_long_self_check(payloads: dict[str, tuple[Any, Any]]) -> list[str]:
    """Long-context decode rows whose two sides disagree about the machine.

    The long-context row has a self-check available for free: the same process
    also measured the short-context decode, and a cache that has grown from ~96 to
    ~160 entries cannot double the per-token cost. The test is on the
    *inconsistency between the two sides' long/short ratios*, never on which way
    the gain points: a row whose two sides disagree about the machine is
    unreadable whether it claims a speedup or a regression, and flagging only the
    regressions would filter the table in one direction and leave the surviving
    rows looking cleaner than they are.
    """
    notes: list[str] = []
    for model, (base, opt) in payloads.items():
        b_short, b_long = base["micro"].get("raw_decode"), base["micro"].get("raw_decode_long")
        o_short, o_long = opt["micro"].get("raw_decode"), opt["micro"].get("raw_decode_long")
        if not (b_short and b_long and o_short and o_long):
            continue
        rates = {
            name: decode_median(node)
            for name, node in (
                ("b_short", b_short),
                ("b_long", b_long),
                ("o_short", o_short),
                ("o_long", o_long),
            )
        }
        if not all(value for value in rates.values()):
            continue
        b_growth = rates["b_long"] / rates["b_short"]
        o_growth = rates["o_long"] / rates["o_short"]
        speedup = rates["b_long"] / rates["o_long"]
        disagreement = ratio_disagreement(b_growth, o_growth)
        if disagreement is None or disagreement <= SELF_CHECK_TOLERANCE:
            continue
        notes.append(
            f"{SHORT[model]} decode（长上下文）：报出 {speedup:.2f}×"
            f"（{'即变慢' if speedup <= 1.0 else '即变快'}），"
            f"但同一进程里长/短上下文之比两侧对不上："
            f"base {b_growth:.2f}、opt {o_growth:.2f}"
            f"（相差 {disagreement:.2f}×，判据是 >{SELF_CHECK_TOLERANCE:.2f}×，"
            "与加速方向无关）。cache 从约 96 条涨到约 160 条不可能让每 token "
            "成本翻倍，所以这一行**待重测**。"
        )
    return notes


#: Where ``run_integration.sh`` has written its per-model pytest logs. The
#: re-collection nested one level deeper (``integration-fixed/logs/``) so that the
#: environment capture could sit beside it; the second review's re-validation used
#: ``integration-round2/``. Every shape has to be named or the newest observation
#: silently drops out of §9.4's spread and §10's expectation quotes an older run.
INTEGRATION_LOG_DIRS = (
    "integration-opt4",
    "integration-opt12",
    "integration-fixed/logs",
    "integration-round2",
)


def generator_identity() -> tuple[str, str, int]:
    """``(sha256, repo-relative path, line count)`` of the script rendering this document.

    §1.5 named the two measured trees and the harnesses that timed them, but not the
    generator. That gap mattered: every number and every verdict in this document is
    produced by ``write_doc``, which was edited repeatedly during the review, so two
    renders of the *same* JSON can disagree — and a reader checking a sentence against
    the artifacts had no way to tell which version of the prose produced it. The file
    is in the working tree rather than in a commit, so there is no revision to name;
    the hash is the only stable handle on it.
    """
    path = Path(__file__).resolve()
    raw = path.read_bytes()
    # ``splitlines`` rather than ``count(b"\n") + 1``: the file ends in a newline, so
    # the +1 form reports one line more than ``wc -l`` does, and the reader this line
    # is written for checks it with ``wc -l``.
    return hashlib.sha256(raw).hexdigest(), repo_relative(path), len(raw.decode().splitlines())


def partial_digest(identity: dict[str, Any], *keys: str) -> str:
    """``source_identity.digest`` restricted to some of its input maps.

    Redefined here rather than imported: ``tools/perf`` scripts are loaded by path in
    the tests, and a by-path load has no package context, so one tool importing its
    sibling would make it unloadable there. The duplication is pinned by
    ``test_the_partial_digest_mirrors_the_tool_it_copies``, which requires
    ``partial_digest(block, "source", "tooling")`` to equal the recorded ``digest`` on
    every artifact on disk — if ``source_identity.digest`` changes its payload, that
    test goes red instead of the two silently drifting.

    Splitting the maps is what makes §1.5 answerable *now*. The combined digest covers
    both halves, so editing this file alone moves the digest of every working-tree run
    even though the code they measured is untouched, and
    ``source_identity.py --digest-only`` stops reproducing those digests. A reader who
    follows §1.5's recheck command would then conclude the measurements cannot be tied
    to the source they measured. They can: the ``source`` half still hashes back file by
    file, so the timings still describe the measured code — only the ``tooling`` half
    moved. The baseline tree is the exception in the other direction: it is a
    ``git archive`` snapshot under ``artifacts/`` whose ``tooling`` half is empty, so
    nothing this round touched it and its combined digest still reproduces. Which trees
    fall in which bucket is derived in :func:`section_identity`, not asserted here.
    """
    payload = json.dumps(
        {key: identity.get(key) for key in keys}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def recorded_map_still_matches(identity: dict[str, Any], key: str) -> tuple[int, list[str]]:
    """``(matched, changed)`` for one recorded hash map — ``"source"`` or ``"tooling"``.

    Checked file by file rather than by re-running ``source_identity.py`` and
    comparing digests, because the question is not "is the tree identical" but "which
    half of the recorded identity moved". The two halves answer different questions
    and have to be reported separately: ``source`` is the code that was measured,
    ``tooling`` is the code that measured and renders it.

    Paths resolve against the block's own ``root``: the baseline row describes a tree
    materialised under ``artifacts/``, and hashing its relative paths against the
    working tree would report every base-vs-opt difference as a file that "changed
    since the run", which is the opposite of what the check is for.
    """
    root = Path(identity.get("root") or ROOT)
    matched = 0
    changed: list[str] = []
    for relative, recorded in sorted((identity.get(key) or {}).items()):
        path = root / relative
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == recorded:
            matched += 1
        else:
            changed.append(relative)
    return matched, changed


def integration_durations(model: str, roots: tuple[Path, ...] | None = None) -> list[tuple[str, float]]:
    """Every recorded pytest duration for one model's integration run.

    §9.4's row is a claim about *the machine*, not about the code: it says the same
    source took two very different amounts of wall time. That claim has to cover
    every duration on disk, not the two that existed when the sentence was typed —
    the re-collection added a third, and a spread quoted from a subset understates
    exactly the instability the row is about.

    Scans every root on disk rather than just ``PERF``, because the point is
    cross-window: the published round's two logs, the re-collection's one and the
    second review's one are four observations of the same phenomenon, and reading only
    ``PERF`` leaves the count at whatever the current ``--data`` happens to hold — two
    windows and 33× from the published root, three and 67× from the re-collected one,
    for the same round. ``repro_roots()`` is scanned rather than ``REPRO_ROOT`` alone because it is
    the same fixed-path argument one level up: a fresh ``perf-run-*`` root is not
    derived from ``PERF`` either, and the second review's re-validation wrote its eight
    logs there. Leaving it out would report a spread from a subset — the defect this
    docstring's first paragraph is about — and it would drop the one observation that
    landed in the slow band *after* the fix, which is the observation §9.4 and the
    environment note both need.
    """
    found: list[tuple[str, float]] = []
    seen: set[Path] = set()
    for root in roots or repro_roots():
        resolved = root.resolve()
        if resolved in seen:
            # Generating from the published root means DEFAULT_PERF *is* PERF; walking
            # it twice reports every duration twice and halves the spread's sample
            # count for no reason.
            continue
        seen.add(resolved)
        for relative in INTEGRATION_LOG_DIRS:
            path = resolved / relative / f"{model}.log"
            if not path.is_file():
                continue
            tail = path.read_text(encoding="utf-8", errors="replace")[-400:]
            match = re.search(r"\bin (\d+(?:\.\d+)?)s\b", tail)
            if match:
                found.append((repo_relative(path), float(match.group(1))))
    return found


#: How many per-model logs a complete integration run writes. ``run_integration.sh``
#: owns the list; this is only the count §10 needs to tell a complete set from a
#: partial one, so a reader is not told to expect a total summed over three runs.
INTEGRATION_MODELS = 8


def pytest_summary(path: Path) -> str | None:
    """The ``N passed, ...`` line a pytest log ends with, without its duration.

    The duration is dropped because it is the one part of the line that is
    machine-dependent, and §9.4 exists to say that wall time on this host varies by
    two orders of magnitude between windows. An expectation that included it would
    be an expectation nobody can reproduce.
    """
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.strip().splitlines()):
        match = re.search(r"(\d+ passed.*)", line)
        if match:
            # pytest writes either "in 8.34s" or "in 3828.08s (1:03:48)".
            return re.sub(r"\s+in\s+[\d:.]+\s*s?\s*(\([^)]*\))?\s*$", "", match.group(1)).strip()
    return None


def quick_suite_expectation(
    roots: tuple[Path, ...] | None = None,
) -> tuple[str, str] | None:
    """``(summary, source)`` for what the quick suite produced on the current tree.

    §10 tells a reader what to expect when they reproduce, so the number has to be
    the one the *current* tree produces. It was typed in as "826 passed" from the
    published round; the review added tests, and an expectation copied from the
    older log makes a correct run look like a failure. Read from the persisted log
    instead, so it moves when the suite moves.

    The source comes back with it because it is no longer a fixed path: once
    ``out_guard`` started refusing the published roots, a new count can only be
    persisted in a fresh ``perf-run-*`` root, so §10 has to print whichever log it
    actually read. A number cited next to a directory it did not come from is the
    defect this function exists to prevent, and naming ``REPRO_ROOT`` unconditionally
    would reintroduce it the first time the expectation moves.

    ``repro_roots()`` comes first because it holds the most recent complete run; see
    :func:`repro_roots` for why that is not ``PERF``.
    """
    seen: set[Path] = set()
    for root in roots or repro_roots():
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        for path in sorted(resolved.glob("quick*.log")):
            summary = pytest_summary(path)
            if summary:
                return summary, repo_relative(path)
    return None


def integration_totals(roots: tuple[Path, ...] | None = None) -> tuple[int, int, str] | None:
    """(passed, failed, directory) for one complete per-model integration run.

    Summed within a single directory, never across directories: the three log sets on
    disk are three runs of the same suite, and adding them would report 624 passed
    for a suite that has 208. ``repro_roots()`` is scanned first because it holds the
    most recent complete run; see :func:`repro_roots` for why that is not ``PERF``.
    """
    seen: set[Path] = set()
    for root in roots or repro_roots():
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        for relative in INTEGRATION_LOG_DIRS:
            directory = resolved / relative
            logs = sorted(directory.glob("*.log"))
            if len(logs) < INTEGRATION_MODELS:
                continue
            totals = {"passed": 0, "failed": 0}
            for path in logs:
                summary = pytest_summary(path) or ""
                for key in totals:
                    match = re.search(rf"(\d+) {key}", summary)
                    if match:
                        totals[key] += int(match.group(1))
            return totals["passed"], totals["failed"], repo_relative(directory)
    return None


VALIDATION_ROOT = ROOT / "artifacts" / "validation"
#: ``[compare] took 0.1 s`` is the comparator timing itself. It lands in the log it
#: writes, so a byte-level claim that ignores it is a claim about the second each run
#: happened to take. ``run_reference.sh`` makes the same distinction and names its
#: column ``log_no_timing=`` for the same reason.
TIMING_LINE = re.compile(r"^\[compare\] took ")
#: The backup taken before round 1 changed anything: the "before" side of §2.2's
#: byte-level claim.
PRE_ROUND1 = "pre-round1"


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _drop_timing(text: str) -> list[str]:
    return [line for line in text.splitlines() if not TIMING_LINE.match(line)]


def reference_identity(backup: str = PRE_ROUND1) -> dict[str, dict[str, bool]] | None:
    """Per model, whether the current npz / log / timing-filtered log match a backup.

    Three keys and not one, because they answer three different questions: the npz is
    the bit-level result the report rests on, the raw log includes a stopwatch, and the
    filtered log is what is left once that stopwatch is set aside.

    ``None`` when the captures are not on disk — ``artifacts/`` is gitignored — so the
    document can say the comparison was not re-measured here instead of printing a
    count nobody measured.
    """
    if not VALIDATION_ROOT.is_dir():
        return None
    found: dict[str, dict[str, bool]] = {}
    for directory in sorted(VALIDATION_ROOT.iterdir()):
        current_npz = directory / "slowq.npz"
        older_npz = directory / f"slowq-{backup}.npz"
        current_log = directory / "compare.log"
        older_log = directory / f"compare-{backup}.log"
        if not (current_npz.is_file() and older_npz.is_file()):
            continue
        if not (current_log.is_file() and older_log.is_file()):
            continue
        new = current_log.read_text(encoding="utf-8", errors="replace")
        old = older_log.read_text(encoding="utf-8", errors="replace")
        found[directory.name] = {
            "npz": _sha256_file(current_npz) == _sha256_file(older_npz),
            "log": new == old,
            "log_no_timing": _drop_timing(new) == _drop_timing(old),
        }
    return found or None


def timing_only_difference(directory: Path, backup: str) -> list[tuple[str, str]] | None:
    """The differing ``(before, after)`` log lines, or ``None`` if any is not a stopwatch.

    ``None`` also covers a length change: then the two logs differ by more than a timing
    value, and the filtered count would no longer be a rounding of the raw one but a
    different claim wearing its clothes.
    """
    new = (directory / "compare.log").read_text(encoding="utf-8", errors="replace").splitlines()
    old = (directory / f"compare-{backup}.log").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    if len(new) != len(old):
        return None
    pairs = [(before, after) for before, after in zip(old, new) if before != after]
    if not pairs or not all(TIMING_LINE.match(a) and TIMING_LINE.match(b) for a, b in pairs):
        return None
    return pairs


def reference_identity_cell(backup: str = PRE_ROUND1) -> str:
    """§2.2's byte-level result cell, derived from the captures rather than typed in.

    It used to read "新 ``compare.log`` 与 pre-round1 备份**逐字节相同**". That is the
    prose twin of the ``log=`` column defect in ``run_reference.sh``: unqualified, over
    a file that contains a stopwatch, and false for one of the eight. Unlike that column
    it is not vacuous — seven really are identical — so what it needs is a count and a
    name, not a retraction. Both are read off the disk here, which is also why the
    sentence cannot drift when the captures are refreshed.
    """
    identity = reference_identity(backup)
    if not identity:
        return (
            f"8/8 退出码 0，**860 个张量全部 PASS**，容差一项没动；"
            f"`artifacts/validation/` 不在这棵树上，逐字节比较**未在本地复算**"
            f"（复算命令见 §10）"
        )
    total = len(identity)
    matched = {
        key: [model for model, values in identity.items() if values[key]]
        for key in ("npz", "log", "log_no_timing")
    }
    cell = (
        f"8/8 退出码 0，**860 个张量全部 PASS**，容差一项没动；"
        f"`slowq.npz` 与 {backup} 备份 **{len(matched['npz'])}/{total} 逐字节相同**"
        f"（逐个 `cmp`，不读汇总表）；"
        f"`compare.log` **{len(matched['log'])}/{total} 逐字节相同**"
    )
    differing = sorted(set(identity) - set(matched["log"]))
    if differing:
        detail = []
        for model in differing:
            pairs = timing_only_difference(VALIDATION_ROOT / model, backup)
            if pairs is None:
                detail.append(f"{model} 的差异**不止计时行**")
                continue
            shown = "、".join(f"`{before}` ↔ `{after}`" for before, after in pairs)
            detail.append(f"{model} 差**恰好 {len(pairs)} 行**（{shown}）")
        cell += (
            "——" + "，".join(detail)
            + "，都是 `[compare] took` 那行比较器自己的墙钟，"
            f"滤掉该行后 **{len(matched['log_no_timing'])}/{total}**"
        )
    return cell


def section_micro(line: list[str]) -> None:
    line.append("\n## 优化 1 + 2：prefill 与 decode（两棵树，逐模型相邻交替）\n")
    payloads = ab_payloads()
    if not payloads:
        missing(line, "ab/*.json（run_round1.sh 第 1 步）")
        return

    line.append("### 引擎分块 prefill（`engine_prefill`）\n")
    line.append(
        "每一格是该长度下 7 次计时的中位数与最小–最大区间（2 次预热后）。"
        "括号里是**观测到的** decoder 调用次数与词表投影次数。\n"
    )
    header = ["模型", "长度"]
    for side in ("base", "opt"):
        header += [f"{side} 中位", f"{side} 区间", f"{side} 投影"]
    header.append("加速")
    line.append("| " + " | ".join(header) + " |")
    line.append("|" + " --- |" * len(header))
    outliers: list[str] = []
    suspect_prefill: list[str] = []
    for model, (base, opt) in payloads.items():
        for length in PREFILL_LENGTHS:
            b = base["micro"]["engine_prefill"].get(length)
            o = opt["micro"]["engine_prefill"].get(length)
            if not b or not o:
                continue
            row = [SHORT[model], f"{length} tok"]
            for side, node in (("base", b), ("opt", o)):
                samples = node.get("samples", [])
                median = statistics.median(samples) if samples else None
                lo = min(samples) if samples else None
                hi = max(samples) if samples else None
                row.append(ms(median))
                row.append("—" if lo is None else f"{lo * 1e3:.1f}–{hi * 1e3:.1f} ms")
                row.append(str(node.get("vocabulary_projections", "?")))
                # A sample an order of magnitude off the median is not spread, it
                # is an interruption. The median still describes the other six, so
                # the row stays in the table — but the reader has to be told that
                # this particular cell saw something else happen on the machine.
                if median and hi and hi > 3 * median:
                    outliers.append(
                        f"{SHORT[model]} {length} tok {side}：中位数 {median * 1e3:.1f} ms，"
                        f"但 7 次里最慢一次是 {hi * 1e3:.1f} ms（{hi / median:.0f} 倍）。"
                        f"中位数仍可用，区间上界不是。"
                    )
            row.append(ratio(stat(b, "median"), stat(o, "median")))
            line.append("| " + " | ".join(row) + " |")
    if outliers:
        line.append(f"\n**{len(outliers)} 格出现了数量级离群值：**\n")
        for item in outliers:
            line.append(f"- {item}")
        line.append("")

    line.append("\n### 整块 prefill\n")
    line.append(
        "`raw_prefill_full` 直接调一次 `model.forward(ids, cache)`，两棵树都只有 1 次词表投影，"
        "**所以这一行的加速全部来自优化 2，不是优化 1**——它是把两项改动分开归因的探针："
        "上面 `engine_prefill` 的加速是优化 1 + 优化 2 之和，这一行是优化 2 单独的值，"
        "两者相减就是优化 1 的贡献。若这一行接近 1.00× 而上面那几行明显更快，"
        "才说明收益主要来自跳过投影。\n"
    )
    line.append(
        "**`自检` = 同一棵树上 `raw_prefill_full` ÷ `engine_prefill`（同长度）。**"
        "两个探针 prefill 同样多的 token、同样只投影一次词表，真正的差别只有"
        "「一次全长 attention」对「四次逐渐变长的 attention」，所以在**同一棵树内**"
        "这个比值必然落在 1 附近；它是这一行**免费自带**的机器状态传感器。"
        "表里印的是**两侧这个比值之间的相差**，判据与 decode 自检同一个"
        f"（>{SELF_CHECK_TOLERANCE:.2f}×），而且**与加速方向无关**——"
        "base 侧被换出内存的那一行报出来的是一个巨大的**加速**，"
        "只盯回退的规则会放它过去。\n"
    )
    line.append(
        "| 模型 | 探针 | base 中位 | base 区间 | opt 中位 | opt 区间 | 加速 | "
        "样本数 base/opt | 自检（两侧相差） |"
    )
    line.append("|" + " --- |" * 9)
    probes = [
        ("raw_prefill_full", "512", "整块 prefill 512 tok"),
        ("raw_prefill_full", "128", "整块 prefill 128 tok"),
    ]
    for model, (base, opt) in payloads.items():
        for key, length, label in probes:
            b = base["micro"].get(key, {}).get(length)
            o = opt["micro"].get(key, {}).get(length)
            if not b or not o:
                continue
            base_ratio, opt_ratio, disagreement = raw_prefill_self_check(
                base.get("micro"), opt.get("micro"), length
            )
            suspect = disagreement is not None and disagreement > SELF_CHECK_TOLERANCE
            if suspect:
                suspect_prefill.append(
                    f"{SHORT[model]} {label}：报出 "
                    f"{ratio(stat(b, 'median'), stat(o, 'median'))}，"
                    f"但同一棵树上 `raw_prefill_full` ÷ `engine_prefill` 两侧对不上："
                    f"base {base_ratio:.2f}、opt {opt_ratio:.2f}"
                    f"（相差 {disagreement:.2f}×，判据是 >{SELF_CHECK_TOLERANCE:.2f}×，"
                    "与加速方向无关）。两个探针做的几乎是同一份工作，这个比值只能落在 1 附近，"
                    "所以是**其中一侧在测别的东西**（权重被换出内存后重新换入），"
                    "不是优化 2 带来的收益。该行判为「待重测」，"
                    "**不进本报告任何一处 base/opt 比较**——包括下面给优化 2 定价的两条路线。"
                )
            check_cell = (
                "**（数据缺失）**"
                if disagreement is None
                else f"{disagreement:.3f}×" + ("  **待重测**" if suspect else "")
            )
            line.append(
                f"| {SHORT[model]} | {label} | {ms(stat(b, 'median'))} | {range_only(b)} | "
                f"{ms(stat(o, 'median'))} | {range_only(o)} | "
                + ("**待重测**" if suspect else ratio(stat(b, "median"), stat(o, "median")))
                + f" | {len(b.get('samples') or [])} / {len(o.get('samples') or [])} "
                f"| {check_cell} |"
            )

    if suspect_prefill:
        line.append(f"\n**{len(suspect_prefill)} 行整块 prefill 数据待重测：**\n")
        for item in suspect_prefill:
            line.append(f"- {item}")
        line.append("")

    line.append("\n### decode：逐 token 前向\n")
    line.append(
        "`raw_decode` / `raw_decode_long` 是短 / 长上下文下的逐 token 前向，"
        "同样含优化 2 的两处改动（无分支 sigmoid + 单 token 时跳过全零 mask），"
        "这一轮没有把两者再拆开单独计时。\n"
    )
    line.append(
        "**decode 的重复单位是「窗口」，不是「token」。**一个窗口 = 用同一批 id 重新 "
        "prefill 一个新 cache，再跑固定步数的单 token 前向。每个窗口都从**相同的起始 "
        "token 序列、相同的 KV 长度、相同的上下文**开始，harness 还断言每个窗口结束时的 "
        "cache 长度完全一致（`cache_lengths_observed` 只有一个值）。计时边界只含那固定 "
        "步数的前向，**不含**建立起始状态的那次 prefill。一个窗口只产出一个样本"
        "（它的时长），所以「窗口数」就是样本数。\n\n"
        "同一窗口内连续的 token **不是独立重复实验**：第 30 步比第 1 步多注意 30 个 key，"
        "把它们拆成 30 个样本会用一个真实效应伪造出离散度，再把这个效应平均掉。\n"
    )
    line.append(
        "| 模型 | 探针 | base ms/tok | base 窗口区间 | opt ms/tok | opt 窗口区间 | 加速 | "
        "窗口数 base/opt |"
    )
    line.append("|" + " --- |" * 8)
    single_window_rows: list[str] = []
    for model, (base, opt) in payloads.items():
        for key, label in (
            ("raw_decode", "decode（短上下文）"),
            ("raw_decode_long", "decode（长上下文）"),
        ):
            b = base["micro"].get(key)
            o = opt["micro"].get(key)
            if not b or not o:
                continue
            cells = [SHORT[model], label]
            counts = []
            for node in (b, o):
                median, spread_text, windows = decode_window_cell(node)
                cells += [median, spread_text]
                counts.append(windows)
            cells.append(ratio(decode_median(b), decode_median(o)))
            cells.append(" / ".join(counts))
            line.append("| " + " | ".join(cells) + " |")
            note = single_window_note(model, label, b, o)
            if note:
                single_window_rows.append(note)
    suspect_decode = decode_long_self_check(payloads)
    if single_window_rows:
        line.append(f"\n**{len(single_window_rows)} 行是单窗口观察值：**\n")
        for item in single_window_rows:
            line.append(f"- {item}")
        line.append(
            "\n单窗口不是「测了一次所以数字小」，而是**没有自己的离散度**："
            "它无法回答「这个差是效应还是那一阵的机器」，所以既不写成收益也不写成回退。"
            "重跑 `tools/perf/core_ab.py --repeat N`（N ≥ 5）即可得到多窗口版本；"
            "旧的单窗口 JSON 保留在原目录，不被覆盖。\n"
        )
    if suspect_decode:
        line.append(f"\n**{len(suspect_decode)} 行 decode 数据待重测：**\n")
        for item in suspect_decode:
            line.append(f"- {item}")
        line.append("")


# --------------------------------------------------------------------------
# attributing the opt-1 + opt-2 delta to one change or the other
# --------------------------------------------------------------------------
def section_attribution(line: list[str]) -> None:
    """What one vocabulary projection costs, and therefore what optimisation 1 is worth.

    The engine_prefill speedups above are opt 1 + opt 2 combined, and the two are
    very unequal. Splitting them needs a measurement, not arithmetic on the
    end-to-end numbers: ``raw_prefill_full`` projects once on both trees, so its
    delta is opt 2 alone, but it is a *different path shape* from the engine's
    chunk loop (one full-length attention versus four growing ones), so its
    absolute milliseconds cannot be subtracted from the engine's.

    ``projection_probe.py`` supplies the missing term directly. ``forward`` is
    literally ``encode`` plus ``output_projection(hidden[-1])``, so paired timing
    of the two over the same ids isolates one projection with the weights, the
    memory state and the process all held fixed. Two routes to opt 2 then exist
    and are printed side by side — subtracting the probe's projection cost from
    the engine delta, and reading ``raw_prefill_full`` — because if they disagree
    by more than the noise floor, one of the three measurements is wrong.
    """
    line.append("\n## 优化 1 与优化 2 的归因\n")
    line.append(
        "`engine_prefill` 的加速是**优化 1 + 优化 2 之和**，两者极不均匀，"
        "所以必须拆开归因，不能把整个加速记在任何一项头上。\n\n"
        "拆不开的原因要说清楚：`raw_prefill_full` 两棵树都只投影 1 次，它的差值确实只含优化 2，"
        "但它与引擎的分块循环**不是同一条路径形状**（一次全长 attention vs 四次逐渐变长的），"
        "所以它的绝对毫秒数不能直接从引擎的绝对毫秒数里减掉——只有比值能比。\n\n"
        "`projection_probe.py` 直接补上缺的那一项：`forward` 就是 `encode` 加一次 "
        "`output_projection(hidden[-1])`（`model.py:390-398`），"
        "所以在同一棵树、同一批 id、同一个进程里**成对**计时再相减，"
        "得到的就是一次投影的净代价，权重、内存状态、进程全部固定。\n"
        "`output_projection` 在两棵树之间**逐字节相同**（已 diff 核对），"
        "所以在 opt 树上测到的投影代价可以直接用来给 base 树当初多做的那几次定价。\n"
    )

    probes: dict[str, Any] = {}
    for model in MODELS:
        payload = load(PERF / "projection" / f"{model}.json")
        if payload is not None:
            probes[model] = payload
    if not probes:
        missing(line, "projection/*.json（projection_probe.py 尚未运行）")
        return

    # Worst single paired difference anywhere in this batch, and whose it was.
    # Derived because the sentence below used to name a specific model, length and
    # millisecond value remembered from the published round; a re-collection that
    # thrashed differently would leave that name attached to the wrong number.
    worst_pair: tuple[float, str, str] | None = None
    negative_pairs = 0
    negatives_under_positive_median = 0
    total_pairs = 0
    models_with_negatives: set[str] = set()
    for model, payload in probes.items():
        for length, row in (payload.get("per_length") or {}).items():
            node = (row or {}).get("projection_seconds") or {}
            median = stat(node, "median")
            for sample in node.get("samples") or []:
                if not isinstance(sample, (int, float)):
                    continue
                total_pairs += 1
                if sample <= 0:
                    negative_pairs += 1
                    models_with_negatives.add(SHORT[model])
                    # "usable" in the probe's own validity field means *the median
                    # survived*, not "every pair was clean". Counted here so a reader
                    # who sees no 作废 row does not conclude the batch was clean.
                    if median is not None and median > 0:
                        negatives_under_positive_median += 1
                if worst_pair is None or sample < worst_pair[0]:
                    worst_pair = (sample, SHORT[model], length)
    line.append(
        "**探针自身的精度**：一次投影的代价与块长无关（永远是 `[1, hidden] × [hidden, vocab]`），"
        "所以 128 / 256 / 512 三行是**同一个量的三次独立测量**，它们的离散度就是这个探针的误差棒。"
        + (
            f"本批 {total_pairs} 对成对样本里有 **{negative_pairs} 对为负**"
            f"（{100 * negative_pairs / total_pairs:.1f}%，分布在 "
            f"{len(models_with_negatives)}/{len(probes)} 个模型上），"
            if total_pairs and negative_pairs
            else "本批没有出现负的成对样本，"
        )
        + (
            f"最差的一对是 {worst_pair[1]} 在 {worst_pair[2]} token 上的 "
            f"**{worst_pair[0] * 1e3:.0f} ms**。"
            if worst_pair and worst_pair[0] < 0
            else ""
        )
        + "而 `forward` 就是 `encode` 加一次真实存在的投影，**成对差值不可能为负**——"
        "出现负数只说明模型在成对计时的两半之间被换出了内存。"
        "所以**任何单次观测都不能用**，能用的是中位数；"
        "而中位数为负的那一行**整行作废**（下表标 `作废`，且不进「三者中位」）。\n"
    )
    if negatives_under_positive_median:
        line.append(
            f"\n**其中 {negatives_under_positive_median} 对负值落在中位数仍为正、"
            "因而没有被标 `作废` 的行里。**"
            "所以「没标作废」的准确含义是**这一格的 7 对里中位数为正**，"
            "不是「这一格测量干净」。这也是下面那个占比只能当区间读、"
            "且必须连着跨批次的重复性一起看的原因。\n"
        )

    line.append("### 一次词表投影的净代价\n")
    line.append(
        "| 模型 | vocab × hidden | 投影权重体积 | 128 tok | 256 tok | 512 tok | "
        "三者中位 | 三次测量离散度 |"
    )
    line.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    stable: dict[str, float] = {}
    invalid_models: list[str] = []
    #: Per-model spread of the three lengths that all price the *same* matmul, i.e.
    #: this probe's own error bar. Kept per model because the share conclusion below
    #: has to be judged against the resolution of the model that produced it.
    resolutions: dict[str, float] = {}
    for model, payload in probes.items():
        config = payload.get("config", {})
        per = payload.get("per_length", {})
        values = []
        cells = []
        for length in ("128", "256", "512"):
            median = stat((per.get(length) or {}).get("projection_seconds"), "median")
            # A negative median is not a small number, it is an invalid measurement:
            # forward = encode + one real projection. Leaving it in the same column
            # as the usable ones would let it drag "三者中位" with it.
            if median is not None and median <= 0:
                cells.append("**作废**")
                invalid_models.append(f"{SHORT[model]} {length} tok")
                continue
            cells.append("—" if median is None else f"{median * 1e3:.2f} ms")
            if median is not None:
                values.append(median)
        if not values:
            continue
        middle = statistics.median(values)
        stable[model] = middle
        if min(values) > 0 and len(values) > 1:
            resolutions[model] = 100 * (max(values) / min(values) - 1)
        weight_bytes = payload.get("projection_weight_bytes")
        line.append(
            "| "
            + " | ".join(
                [
                    SHORT[model],
                    f"{config.get('vocab_size')} × {config.get('hidden_size')}",
                    "—" if not weight_bytes else f"{weight_bytes / 2**20:.0f} MiB",
                    *cells,
                    f"**{middle * 1e3:.2f} ms**",
                    f"{(max(values) - min(values)) * 1e3:.2f} ms",
                ]
            )
            + " |"
        )
    line.append(
        "\n代价跟 `vocab_size × hidden_size` 走，不跟模型质量走："
        "从 SmolLM2 系列的 49152 词表到 Qwen 系列的 151936 词表差一个数量级，"
        "Qwen3-1.7B 的 `hidden_size` 又是 Qwen3-0.6B 的两倍。\n"
    )
    if invalid_models:
        # The cause is stated generically on purpose. An earlier version of this
        # paragraph blamed "最大那个模型的权重集放不进当时的可用内存", which is true
        # of the re-collection (all three invalid rows are Qwen3-1.7B) and false of
        # the published round, where the invalid cell is SmolLM2-360M at 512 tokens —
        # the *second smallest* model. Naming a mechanism the data does not support is
        # the same defect as naming a number it does not.
        line.append(
            f"\n**作废 {len(invalid_models)} 行：{'、'.join(invalid_models)}。**"
            "这些行的成对差值中位数为负，而 `forward` = `encode` + 一次真实投影，"
            "负值在算术上不可能，只能是**成对计时的两半之间机器状态变了**"
            "（权重被换出内存后重新换入、别的进程抢走内存带宽）。"
            "哪一次是哪一种，这批数据没有采集到能区分的指标，所以不指认。"
            "**它们没有被删掉，JSON 原样留在 `projection/` 里**，"
            "只是不进上面那张表的中位、也不进下面那张表的占比。"
            "缺了它们的模型在下面那张表里整行缺席，"
            "所以下面的占比区间**只覆盖采到有效数据的模型**。\n"
        )

    line.append("### 优化 1 在 512-token 分块 prefill 上值多少\n")
    line.append(
        "`chunk_size=128`，所以 512 token 分 4 块，优化 1 跳过其中 **3** 次投影"
        "（这个 3 是被 instrumented spy 数出来的，见上面 `base 投影` 列的 4 → 1，"
        "不是从 `length // chunk_size` 推的）。"
        "`占比` 用两种投影代价各算一遍：`512 tok 那次` 与 `三者中位`，"
        "两个数的差就是这项归因本身的不确定度。\n"
    )
    line.append(
        "| 模型 | base 引擎 prefill | 跳过次数 | 用 512 tok 代价 | 占比 | "
        "用三者中位 | 占比 | base→opt 实测省下 | 优化 1 占省下的 |"
    )
    line.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    shares: list[float] = []
    share_by_model: dict[str, float] = {}
    for model, payload in probes.items():
        per = (payload.get("per_length") or {}).get("512") or {}
        base_eng = stat(per, "engine_prefill_base")
        skipped = per.get("projections_skipped")
        direct = stat(per.get("projection_seconds"), "median")
        middle = stable.get(model)
        # ``direct`` can be negative while ``middle`` is positive (one bad length out
        # of three), and a negative cost times a skip count would *subtract* from the
        # measured saving and inflate optimisation 1's apparent share of it.
        if base_eng is None or skipped is None or middle is None:
            continue
        if direct is None or direct <= 0:
            continue
        opt_eng = None
        opt_payload = load(PERF / "ab" / f"{model}-opt.json")
        base_payload = load(PERF / "ab" / f"{model}-base.json")
        opt_eng = stat(
            ((opt_payload or {}).get("micro", {}).get("engine_prefill", {}) or {}).get("512"),
            "median",
        )
        saving = None if opt_eng is None else base_eng - opt_eng
        cells = [
            SHORT[model],
            f"{base_eng * 1e3:.1f} ms",
            str(skipped),
            f"{direct * skipped * 1e3:.1f} ms",
            f"{direct * skipped / base_eng * 100:.2f}%",
            f"{middle * skipped * 1e3:.1f} ms",
            f"{middle * skipped / base_eng * 100:.2f}%",
        ]
        shares.append(direct * skipped / base_eng * 100)
        shares.append(middle * skipped / base_eng * 100)
        # The larger of the two cost routes, per model: §9.2 has to say whether the
        # share is resolvable *for that model*, and quoting the band's global max
        # against a different model's dispersion is how the false "不大于" got written.
        share_by_model[SHORT[model]] = max(direct, middle) * skipped / base_eng * 100
        if saving is None or saving <= 0:
            cells += ["—", "—"]
        else:
            cells += [f"{saving * 1e3:.1f} ms", f"{direct * skipped / saving * 100:.1f}%"]
        line.append("| " + " | ".join(cells) + " |")

    if shares:
        worst_model, worst_share = max(share_by_model.items(), key=lambda kv: kv[1])
        line.append(
            f"\n**优化 1 的占比区间是 {min(shares):.2f}% ~ {max(shares):.2f}%**"
            f"（按模型取两种代价里较大的那个，最大的是 {worst_model} 的 {worst_share:.2f}%）。"
            "**结论：这一项省掉的工作是被直接观测到的（投影次数 4 → 1），"
            "但它不能当成实测加速来报。**理由不是「小到看不见」，是**拆不出来**：\n\n"
            "- 端到端上唯一测过这个量的实验是 §7.3 的配对实验，而它的 base／opt 两棵树"
            "**同时**差着优化 1 和优化 2，那几行的逐轮增益是两者之和。"
            "没有「只关优化 1」的配对探针，所以这个占比无论多大都分不出来。\n"
            "- 这个占比自己的误差棒也不小：上面那张表的 `三次测量离散度` 列，"
            "是同一个 `[1, hidden] × [hidden, vocab]` 矩阵乘在 128／256／512 三个长度上的极差，"
            f"相对幅度 {min(resolutions.values()):.0f}% ~ {max(resolutions.values()):.0f}%"
            "——比被估的量本身还大。\n\n"
            "**更正**：本节早先写的是「整个区间都不大于下面优化 5 那节观测到的跨进程离散度量级」，"
            "§9.2 那一行写的是「不大于约 1.5%」。两句都撤回。那个 1.5% 是"
            "**两个扫描模型**在 `引擎分块 prefill 512 tok` 一行 opt 侧进程间离散度的中位数，"
            "拿它去比**四个模型里最大的占比**，是跨模型比两列不同的东西；"
            f"而按模型各比一遍时，占比最大的 {worst_model} 自己那一行的 opt 侧进程间离散度"
            "**比它的占比还小**（数字在 §7.3 表里，不在此处复述），"
            "所以「不大于」按模型读也不成立。撤回这个比较不改变结论——"
            "结论靠的是上面第一条「拆不出来」，它不引用任何离散度。\n"
            "它保留在代码里的理由是免费且逐位相同，不是因为它快。\n"
        )

    line.append("### 两条独立路线给优化 2 定价，互相核对\n")
    line.append(
        "路线 A：`引擎实测省下 − 探针测到的 3 次投影` = 优化 2。\n"
        "路线 B：`raw_prefill_full` 的 base−opt 差值 = 优化 2（两棵树都只投影 1 次）。\n"
        "两条路线用的输入不同、路径形状不同、连测量时刻都不同，"
        "所以它们对得上才是真的对得上。`相差` 一列大到与下面优化 5 那节的"
        "跨进程离散度同量级，就说明三项测量里有一项是错的。\n\n"
        "**整行跳过的两种情况**（跳过而不是印一个数出来）：\n"
        "- 探针 512 tok 的成对差值中位数 ≤ 0 —— 路线 A 会**减去一个负数**，"
        "把优化 1 的功劳算到优化 2 头上，越脏的探针算出来优化 2 越大。\n"
        "- `raw_prefill_full` 的自检不过（上面那张表的 `自检` 列 >"
        f"{SELF_CHECK_TOLERANCE:.2f}×）—— 路线 B 整个是机器状态，不是代码。\n"
    )
    line.append(
        "| 模型 | 引擎 base→opt 省下 | 路线 A（减掉探针投影） | 路线 B（raw_prefill_full） | 相差 |"
    )
    line.append("| --- | --- | --- | --- | --- |")
    gaps: list[float] = []
    dropped: list[str] = []
    for model, payload in probes.items():
        per = (payload.get("per_length") or {}).get("512") or {}
        skipped_seconds = stat(per, "skipped_seconds")
        projection = stat(per.get("projection_seconds"), "median")
        base_payload = load(PERF / "ab" / f"{model}-base.json")
        opt_payload = load(PERF / "ab" / f"{model}-opt.json")
        if not base_payload or not opt_payload or skipped_seconds is None:
            continue
        b_micro, o_micro = base_payload.get("micro", {}), opt_payload.get("micro", {})
        b_eng = stat((b_micro.get("engine_prefill", {}) or {}).get("512"), "median")
        o_eng = stat((o_micro.get("engine_prefill", {}) or {}).get("512"), "median")
        b_raw = stat((b_micro.get("raw_prefill_full", {}) or {}).get("512"), "median")
        o_raw = stat((o_micro.get("raw_prefill_full", {}) or {}).get("512"), "median")
        if None in (b_eng, o_eng, b_raw, o_raw):
            continue
        if projection is not None and projection <= 0:
            dropped.append(
                f"{SHORT[model]}（探针 512 tok 中位数 {projection * 1e3:.2f} ms ≤ 0）"
            )
            continue
        _, _, disagreement = raw_prefill_self_check(b_micro, o_micro, "512")
        if disagreement is not None and disagreement > SELF_CHECK_TOLERANCE:
            dropped.append(f"{SHORT[model]}（`raw_prefill_full` 自检相差 {disagreement:.2f}×）")
            continue
        route_a = (b_eng - o_eng) - skipped_seconds
        route_b = b_raw - o_raw
        gap = abs(route_a - route_b) / route_b * 100 if route_b else float("nan")
        gaps.append(gap)
        line.append(
            f"| {SHORT[model]} | {(b_eng - o_eng) * 1e3:.1f} ms | {route_a * 1e3:.1f} ms "
            f"| {route_b * 1e3:.1f} ms | {gap:.1f}% |"
        )
    if dropped:
        line.append(
            f"\n**跳过 {len(dropped)} 个模型：{'；'.join(dropped)}。**"
            "被跳过不等于「相差很小所以省略」，而是**这张交叉核对表对它无法成立**——"
            "两条路线各有一个输入已经作废。上面「一次词表投影的净代价」表里"
            "标 `作废` 的行、「整块 prefill」表里标 `待重测` 的行，"
            "对应的就是这里的这几个模型；三张表少的是同一批行。\n"
        )
    if gaps:
        # The bound is ``SELF_CHECK_TOLERANCE`` expressed as a percentage, not a new
        # number: this column compares two independent measurements of the same
        # quantity, which is exactly what that constant already governs elsewhere in
        # the report. The conclusion below used to be asserted unconditionally —
        # "这个一致性足以支撑结论" — with the max gap hardcoded at 10.0%; a batch
        # whose routes disagree by more would have printed the same sentence.
        bound = 100 * (SELF_CHECK_TOLERANCE - 1)
        worst = max(gaps)
        agreeing = worst <= bound
        line.append(
            f"\n两条路线最大相差 **{worst:.1f}%**，最小 {min(gaps):.1f}%"
            f"（{len(gaps)} 个模型；判据是 ≤{bound:.0f}%，即报告其余各处"
            "「同一件事的两次独立测量」用的同一个门限）。"
            + (
                "考虑到路线 B 的路径形状与引擎不同（整块 attention vs 分块），"
                "这个一致性足以支撑结论：**512-token 分块 prefill 上省下的时间"
                "绝大部分来自优化 2（`sigmoid` 无分支化 + 单 token 跳过全零 mask），"
                "优化 1 只占个位数百分比。**\n"
                if agreeing
                else f"**交叉核对没有通过**：最差那 {worst:.1f}% 超过了 {bound:.0f}%，"
                "说明这三项测量（引擎差值、探针投影、`raw_prefill_full` 差值）里"
                "**至少有一项在这一批里不可用**。所以「省下的大部分来自优化 2」"
                "这一句在本批数据上**不能由这张表支撑**——它只能由上面那张"
                "「优化 1 占比」表支撑，而那张表给出的占比本来就在个位数百分比以下，"
                "两条路线一致与否并不改变那个量级。\n"
            )
        )


# --------------------------------------------------------------------------
# end-to-end workloads
# --------------------------------------------------------------------------
def section_workloads(line: list[str]) -> None:
    """The end-to-end table, from whichever run measured all five workloads.

    ``run_round1.sh`` step 1 uses the harness default ``--max-context 1024``,
    under which ``long-prompt-short-answer`` (a 1662-token prompt) is refused by
    both trees. ``run_round1_longctx.sh`` re-runs the whole suite at 2048 so that
    the prefill-dominated case — the shape optimisation 1 actually targets — is
    measured rather than skipped. That run is preferred when it exists, and the
    source is printed either way, because the two runs are not interchangeable:
    a bigger context window allocates a bigger cache.
    """
    line.append("\n## 端到端工作负载\n")
    payloads: dict[str, tuple[Any, Any]] = {}
    source = None
    for directory, note in (
        ("ab-longctx", "`--max-context 2048`（`run_round1_longctx.sh`），五个负载全部测到"),
        ("ab", "`--max-context 1024`（`run_round1.sh` 第 1 步），长 prompt 负载会被跳过"),
    ):
        found = {}
        for model in MODELS:
            base = load(PERF / directory / f"{model}-base.json")
            opt = load(PERF / directory / f"{model}-opt.json")
            if base and opt:
                found[model] = (base, opt)
        # Only a complete run is preferred: a half-finished ab-longctx must not
        # replace a complete ab and quietly drop models from the table.
        if len(found) == len(MODELS):
            payloads, source = found, (directory, note)
            break
        if not payloads and found:
            payloads, source = found, (directory, note)
    if not payloads:
        missing(line, "ab-longctx/*.json 或 ab/*.json（端到端工作负载）")
        return

    directory, note = source
    # ``PERF``, not a typed-in ``artifacts/perf-round1``: this sentence is the
    # provenance line for the whole end-to-end table, and printing the published
    # round's directory while reading the re-collection's would send anyone who
    # tries to check a number to a JSON that does not contain it.
    line.append(f"数据来源：`{repo_relative(PERF)}/{directory}/`，{note}。\n")
    if len(payloads) != len(MODELS):
        line.append(
            f"**只有 {len(payloads)}/{len(MODELS)} 个模型跑完："
            f"{', '.join(SHORT[m] for m in payloads)}。其余不在表里。**\n"
        )

    line.append(
        "每个工作负载 7 次取中位数（2 次预热后）。首 token = prefill + 第一个 token；"
        "总耗时是整次请求的墙钟。**输出 token 数不同的两行不能比**，"
        "所以表里带上生成数与停止原因，`输出一致` 一列比较两侧的 `text_sha`。\n"
    )
    line.append(
        "**`decode 自检` = 该工作负载自己记录的 decode 速率 ÷ 同一棵树上裸 decode 探针的速率**"
        "（`workloads[*].decode_seconds / generated_tokens` 除以 `micro.raw_decode.ms_per_token`）。"
        "它**不会等于 1.00**：长 prompt 负载的 KV 比探针长得多，逐 token 本来就更慢，"
        "所以 1.19、1.21 这种值是正常的。要看的是**同一行 base 和 opt 的自检是否一致**——"
        "上下文长度在两侧相同、会互相抵消，剩下的差异只能是那段时间机器的状态。\n"
    )
    line.append(
        "标记规则（写死在这里，不靠肉眼看表）：`base 自检` 与 `opt 自检` 相差超过 "
        f"{SELF_CHECK_TOLERANCE:.2f}×，该行判为「待重测」。判据是两侧之比，"
        "所以与加速方向无关：一行声称变快 3×、但两侧自检差 2×，同样待重测。\n\n"
        "这条规则原来是 `总加速 ≤ 1.00× 且 opt 自检 > base 自检 × 1.15`，"
        "两个条件合起来意味着**只有声称变慢的行才可能被标记**——"
        "声称变快的那些行无论自检多离谱都原样印成收益。那不是去噪，"
        "是只去掉碍事的噪声，剩下的行看起来比测量本身干净。\n"
    )
    line.append(
        "自检必须在**同一进程、同一时间窗**内采集才有意义：它比较的是「这棵树在那几分钟里"
        "的机器状态」。借几小时前另一次运行的 micro 来配对，得到的是**跨进程**比值，"
        "它混进了两次运行之间的机器漂移，而这个漂移正是自检本来要抓的东西。"
        f"所以没有同窗口 micro 的行自检列印 `—`，并按 **待重测（无同窗口自检）** 处理。\n"
    )
    line.append(
        "| 模型 | 工作负载 | 输入/生成 | base 首token | opt 首token | 首token 加速 | "
        "base 总 | opt 总 | 总加速 | base decode 自检 | opt decode 自检 | 输出一致 | 结论 |"
    )
    line.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    mismatched = []
    skipped: list[str] = []
    suspect: list[str] = []
    unchecked: list[str] = []
    rows_emitted = 0
    for model, (base, opt) in payloads.items():
        for name in base.get("workloads", {}):
            b = base["workloads"].get(name)
            o = opt["workloads"].get(name)
            if not b or not o:
                continue
            # A workload the harness refused to run must never collapse into a row
            # of em-dashes: that reads as "measured, no difference" rather than
            # "never measured". Both trees skip for the same reason (the prompt is
            # tokenised before either engine is touched), so it is not a regression
            # — but it is a hole in this round's coverage and is listed as one.
            if "skipped" in b or "skipped" in o:
                reason = b.get("skipped") or o.get("skipped")
                skipped.append(f"{SHORT[model]}/{name}: {reason}")
                continue
            b_ttft, o_ttft = stat(b.get("first_token_seconds"), "median"), stat(o.get("first_token_seconds"), "median")
            b_total, o_total = stat(b.get("total_seconds"), "median"), stat(o.get("total_seconds"), "median")
            same = "是" if b.get("text_sha") == o.get("text_sha") else "**否**"
            if b.get("text_sha") != o.get("text_sha"):
                mismatched.append(f"{model}/{name}")
            # Self-check against this payload's *own* micro block only. A payload
            # that ran just the workloads suite has no in-window decode probe, and
            # there is no substitute: the earlier run's micro came from a different
            # process at a different time, so pairing it with this row would report
            # machine drift between two runs as if it were drift within one.
            b_check = decode_self_check(base, b)
            o_check = decode_self_check(opt, o)
            gain = (b_total / o_total) if (b_total and o_total) else None
            disagreement = ratio_disagreement(b_check, o_check)
            # gain is deliberately not passed in: see suspect_self_check.
            suspect_row = suspect_self_check(b_check, o_check)
            if suspect_row:
                if disagreement is None:
                    unchecked.append(
                        f"{SHORT[model]}/{name}"
                        f"（总加速 {'—' if gain is None else f'{gain:.2f}×'}）"
                    )
                else:
                    suspect.append(
                        f"{SHORT[model]}/{name}：总加速 {gain:.2f}×"
                        f"（{'即变慢' if gain is not None and gain <= 1.0 else '即变快'}），"
                        f"但同一进程里两侧 decode 自检对不上："
                        f"base {b_check:.2f}、opt {o_check:.2f}"
                        f"（相差 {disagreement:.2f}×，判据是 >{SELF_CHECK_TOLERANCE:.2f}×，"
                        "与方向无关）→ **待重测**"
                    )
            line.append(
                f"| {SHORT[model]} | {name} | {b.get('prompt_tokens', '?')} / "
                f"{b.get('generated_tokens', '?')} ({b.get('stop_reason', '?')}) | "
                f"{ms(b_ttft)} | {ms(o_ttft)} | {ratio(b_ttft, o_ttft)} | "
                f"{ms(b_total)} | {ms(o_total)} | {ratio(b_total, o_total)} | "
                f"{'—' if b_check is None else f'{b_check:.2f}'} | "
                f"{'—' if o_check is None else f'{o_check:.2f}'} | {same} | "
                f"{'**待重测**' if suspect_row else '可信'} |"
            )
            rows_emitted += 1
    # Grouped rather than listed row by row: when a whole run executed only the
    # workloads suite, every single row fails for the identical reason, and 25
    # copies of that reason bury the rows that failed for a reason of their own.
    # Named individually only when the gap is partial — then which rows are
    # affected is the useful part. The verdict column already says 待重测 per row,
    # so enumerating a total failure repeats the table back at the reader.
    if unchecked:
        if len(unchecked) == rows_emitted:
            scope = f"上表全部 {len(unchecked)} 行"
        else:
            scope = f"{len(unchecked)}/{rows_emitted} 行：" + "、".join(unchecked)
        line.append(
            f"\n**{scope} 无同窗口自检，全部待重测。**\n\n"
            "这次运行只执行了 workloads 套件，进程里没有裸 decode 探针，"
            "所以自检列全是 `—`。表里的毫秒数是真实测量，但**无法判断这些差值里"
            "有多少是改动、有多少是那几分钟的机器状态**。`run_round1_longctx.sh` "
            "已改成 `--suite all --no-chat`，重跑即可在同窗口内采到自检；"
            "旧的无自检 JSON 保留在原目录，不覆盖。\n"
        )
    if suspect:
        line.append(
            f"\n**{len(suspect)} 行待重测（不能读成收益，也不能读成回退）：**\n"
        )
        for item in suspect:
            line.append(f"- {item}")
        line.append(
            "\n待重测不等于「不算回退」：它是说**这一行的数据本身不足以判断方向**。"
            "补跑更快也不能反过来说明原来那次不是回退——两次是不同的进程、"
            "不同的机器状态，旧结果既没被推翻也没被证实，原始记录保留在原目录。\n"
        )
    if mismatched:
        line.append(f"\n**输出不一致：{', '.join(mismatched)}**\n")
    else:
        line.append("\n全部工作负载两侧 `text_sha` 相同：greedy 输出逐 token 一致。\n")
    if skipped:
        line.append(f"**本轮未测到的工作负载（{len(skipped)} 项，两侧同样跳过，不是回退）：**\n")
        for item in skipped:
            line.append(f"- {item}")
        line.append("")
    rerun_contradiction(line, directory)


#: The three comparisons behind :func:`cross_check_values`, in the order they are
#: reported: ``(full label, short label, comparison)``. Named rather than positional
#: so the column can say *which* one fired: a row whose two batches agree about the
#: opt tree but whose ``ab/`` batch disagrees with itself is a different finding from
#: one where the batches describe different machines, and printing only the opt-side
#: ratios made the first read as the second.
CROSS_CHECKS = (
    ("`{first_dir}/` 内部 base↔opt", "`{first_dir}/` 内部",
     lambda a, b: ratio_disagreement(a[0], a[1])),
    ("`{second_dir}/` 内部 base↔opt", "`{second_dir}/` 内部",
     lambda a, b: ratio_disagreement(b[0], b[1])),
    ("跨遍 opt", "跨遍 opt", lambda a, b: ratio_disagreement(a[1], b[1])),
)

#: Label of the third comparison, so :func:`rerun_contradiction` can tell a row whose
#: two batches describe different machines from one where a single batch merely
#: disagrees with itself. Read out of ``CROSS_CHECKS`` rather than retyped: the two
#: have to keep matching for the split below to mean anything.
CROSS_BATCH_LABEL = CROSS_CHECKS[-1][0]


def cross_check_values(
    first: tuple[float | None, float | None] | None,
    second: tuple[float | None, float | None] | None,
) -> list[float] | None:
    """The three self-check disagreements for one row, in ``CROSS_CHECKS`` order.

    ``None`` when either batch lacks a same-window ``micro``, so "no evidence" stays
    distinguishable from "evidence that agrees".

    These are the numbers the verdict is computed from, which is why the column prints
    them rather than leaving the reader to divide the per-batch ratios: those are shown
    rounded, and 1.06 / 0.82 reads as 1.29 where the judgement used 1.31. A cell whose
    own arithmetic does not reproduce its own verdict invites the reader to distrust
    the verdict rather than the rounding.
    """
    if not first or not second:
        return None
    values = []
    for _, _, compare in CROSS_CHECKS:
        value = compare(first, second)
        if value is None:
            return None
        values.append(value)
    return values


def cross_check_failures(
    values: list[float] | None,
    first_dir: str = "ab",
    second_dir: str = "ab-longctx",
) -> list[str]:
    """Full labels of the comparisons above ``SELF_CHECK_TOLERANCE``.

    Empty both when every comparison agrees and when the row is unreadable, so
    callers must test ``cross_check_values(...) is None`` for readability first.
    """
    if not values:
        return []
    return [
        label.format(first_dir=first_dir, second_dir=second_dir)
        for (label, _, _), value in zip(CROSS_CHECKS, values)
        if value > SELF_CHECK_TOLERANCE
    ]


def cross_check_cell(
    first: tuple[float | None, float | None] | None,
    second: tuple[float | None, float | None] | None,
) -> str:
    """The cross-check column: what each batch measured, the three ratios, the verdict.

    All four measured ratios rather than the two opt-side ones, because the verdict can
    fire on an internal disagreement: showing only ``0.95`` and ``0.99`` next to
    "两遍对不上" reads as a contradiction in the report rather than as a finding about
    ``ab/``.
    """
    def side(pair: tuple[float | None, float | None] | None) -> str:
        if not pair or None in pair:
            return "无 `micro`"
        return f"base {pair[0]:.3f} / opt {pair[1]:.3f}"

    ratios = f"`ab/` {side(first)}；`ab-longctx/` {side(second)}"
    values = cross_check_values(first, second)
    if values is None:
        return f"{ratios} → 读不出来"
    measured = "；".join(
        f"{short.format(first_dir='ab', second_dir='ab-longctx')} {value:.2f}×"
        for (_, short, _), value in zip(CROSS_CHECKS, values)
    )
    cell = f"{ratios}；三项比值（判据 >{SELF_CHECK_TOLERANCE:.2f}×）{measured}"
    failures = cross_check_failures(values)
    if not failures:
        return f"{cell} → 三项全部一致"
    return f"{cell} → **对不上：{'；'.join(failures)}**"


def contradiction_rows(
    root: Path,
    primary: str = "ab-longctx",
    other: str = "ab",
    threshold: float = 0.10,
) -> list[tuple[str, str, float, float, Any, Any]]:
    """Rows where two runs of the same workload disagree by more than ``threshold``.

    ``(model, workload, other_gain, primary_gain, other_checks, primary_checks)``,
    where each ``checks`` pair is ``(base self-check, opt self-check)`` for that batch.

    Shared by §3's table and by :func:`headline_numbers`, which quotes the widest of
    these rows in §0. Deriving both from one function is what keeps §0 from naming a
    pair that §3's table does not contain — the discipline line used to be typed in
    from the published batch, so a report generated from the re-collection opened by
    quoting numbers its own evidence section had replaced.
    """
    rows = []
    for model in MODELS:
        pairs = {}
        for directory in (primary, other):
            base = load(root / directory / f"{model}-base.json")
            opt = load(root / directory / f"{model}-opt.json")
            if base and opt:
                pairs[directory] = (base, opt)
        if len(pairs) != 2:
            continue
        for name in pairs[primary][0].get("workloads", {}):
            gains = []
            checks: dict[str, tuple[float | None, float | None]] = {}
            for directory in (other, primary):
                base, opt = pairs[directory]
                b = base["workloads"].get(name)
                o = opt["workloads"].get(name)
                if not b or not o or "skipped" in b or "skipped" in o:
                    gains = []
                    break
                b_total = stat(b.get("total_seconds"), "median")
                o_total = stat(o.get("total_seconds"), "median")
                if not b_total or not o_total:
                    gains = []
                    break
                gains.append(b_total / o_total)
                checks[directory] = (decode_self_check(base, b), decode_self_check(opt, o))
            if len(gains) == 2 and abs(gains[0] - gains[1]) > threshold:
                rows.append((SHORT[model], name, gains[0], gains[1],
                             checks.get(other), checks.get(primary)))
    return rows


def batch_workloads(root: Path, directory: str) -> dict[str, list[str]]:
    """Workload names one batch measured on *both* trees, per model, not skipped.

    Both trees, because a workload only yields a base→opt gain when each side ran
    it; counting the opt payload alone would report a load as measured that the
    table below then has to print as missing.
    """
    per_model: dict[str, list[str]] = {}
    for model in MODELS:
        base = load(root / directory / f"{model}-base.json")
        opt = load(root / directory / f"{model}-opt.json")
        if not (base and opt):
            continue
        names = [
            name
            for name in (opt.get("workloads") or {})
            if all(
                isinstance((payload.get("workloads") or {}).get(name), dict)
                and "skipped" not in payload["workloads"][name]
                for payload in (base, opt)
            )
        ]
        if names:
            per_model[SHORT[model]] = names
    return per_model


def compared_workloads(
    root: Path, primary: str = "ab-longctx", other: str = "ab"
) -> dict[str, list[str]]:
    """Workload names both batches measured on both trees, per model.

    Derived because the sentence that quotes it used to be typed in as "同样四个
    工作负载". That was true of the published batch's rows and false of the
    re-collection's, and it cannot be made true by picking a different number: the
    count is *per model*, since ``long-prompt-short-answer`` is skipped on the two
    SmolLM2 models under ``ab/`` and so those two only ever have four comparable
    loads while the other three have five.
    """
    first = batch_workloads(root, primary)
    second = batch_workloads(root, other)
    return {
        short: [name for name in names if name in set(second.get(short, ()))]
        for short, names in first.items()
        if any(name in set(second.get(short, ())) for name in names)
    }


#: The two bare-decode probes, in the order §2.3's table prints them. Both batches
#: run them on both trees, which is what makes a cross-batch comparison possible at
#: all — unlike the workloads, whose ``ab-longctx/`` pass in the published round ran
#: ``--suite workloads`` only.
MICRO_DECODE_PROBES = (
    ("raw_decode", "decode 短上下文"),
    ("raw_decode_long", "decode 长上下文"),
)


def micro_decode_rows(
    root: Path, primary: str = "ab-longctx", other: str = "ab"
) -> list[dict[str, Any]]:
    """Per model × probe: what each batch's ``micro`` decode medians say.

    :func:`contradiction_rows` compares ``workloads[*].total_seconds`` and nothing
    else, so the disagreement it prints is an end-to-end one. The same two batches
    also each measured the bare decode probes on the same two trees, and those
    disagree more sharply — on the re-collection Qwen3-1.7B's base/opt gain is
    above 1.0 in ``ab/`` and below it in ``ab-longctx/``, i.e. **the two batches
    point in opposite directions about the same code**. Recording the per-side
    medians too is the point: the cleanest evidence there is not the gain but one
    side drifting against itself (same probe, same baseline tree, same day), which
    no ratio between the trees can show.
    """
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        payloads = {}
        for directory in (other, primary):
            base = load(root / directory / f"{model}-base.json")
            opt = load(root / directory / f"{model}-opt.json")
            if base and opt:
                payloads[directory] = (base, opt)
        if len(payloads) != 2:
            continue
        for key, label in MICRO_DECODE_PROBES:
            medians: dict[str, dict[str, float]] = {}
            windows: dict[str, int] = {}
            for directory, (base, opt) in payloads.items():
                sides: dict[str, float] = {}
                counts: list[int] = []
                for side, payload in (("base", base), ("opt", opt)):
                    node = (payload.get("micro") or {}).get(key)
                    value = decode_median(node) if isinstance(node, dict) else None
                    if value is None:
                        sides = {}
                        break
                    sides[side] = value
                    block = node.get("ms_per_token_windows")
                    samples = block.get("samples") if isinstance(block, dict) else None
                    counts.append(len(samples) if samples else 1)
                if len(sides) != 2:
                    medians = {}
                    break
                medians[directory] = sides
                windows[directory] = min(counts)
            if len(medians) != 2:
                continue
            gains = {d: v["base"] / v["opt"] for d, v in medians.items()}
            rows.append({
                "short": SHORT[model],
                "key": key,
                "label": label,
                "medians": medians,
                "windows": windows,
                "gains": gains,
                "disagreement": ratio_disagreement(gains[other], gains[primary]),
                "sign_flip": (gains[other] - 1) * (gains[primary] - 1) < 0,
                "base_drift": ratio_disagreement(
                    medians[other]["base"], medians[primary]["base"]
                ),
                "opt_drift": ratio_disagreement(
                    medians[other]["opt"], medians[primary]["opt"]
                ),
                "dirs": (other, primary),
            })
    return rows


def micro_decode_contradiction(line: list[str], primary: str) -> None:
    """Print the bare-decode probes' cross-batch disagreement next to the workloads'.

    The workloads table above this one is the only place the report used to look for
    "the two runs disagree", and it looks at ``total_seconds`` — a quantity that
    mixes prefill, decode and detokenising. The probes below isolate decode, so a
    disagreement there is harder to explain away, and a *sign flip* there means the
    two batches cannot both be describing the same machine.
    """
    if primary != "ab-longctx":
        return
    rows = micro_decode_rows(PERF)
    line.append("\n#### 裸 decode 探针：两遍互相比一次\n")
    if not rows:
        line.append(
            "**这一节读不出来**：两遍里至少有一遍的 `micro` 段是空的"
            "（`run_round1_longctx.sh` 当时只跑 `--suite workloads`），"
            "所以裸 decode 探针没有可比的第二个值。驱动已改成 `--suite all --no-chat`，"
            "重采的那一批就有；两批原始 JSON 都保留在原目录。\n"
        )
        return
    first, second = rows[0]["dirs"]
    line.append(
        f"两遍各自在**同一棵 base 树与同一棵 opt 树**上跑了 `{first}/` 与 "
        f"`{second}/` 的裸 decode 探针，代码一个字没改。下表把**每一侧自己的中位数**"
        "并排印出来，而不是只印增益：一侧对自己漂移，是比两棵树之间的比率更干净的证据。\n"
    )
    line.append(
        f"| 模型 | 探针 | base ms/tok（`{first}/` → `{second}/`） | "
        f"opt ms/tok（同） | base→opt（`{first}/`） | base→opt（`{second}/`） | "
        f"两批增益之差 | 窗口数 | 判读 |"
    )
    line.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    flagged: list[dict[str, Any]] = []
    flipped: list[dict[str, Any]] = []
    for row in rows:
        medians, gains, windows = row["medians"], row["gains"], row["windows"]
        disagreement = row["disagreement"]
        # A sign flip is flagged on its own, not only when it also clears the
        # magnitude threshold. Qwen3-1.7B's short-context decode lands at 1.146× —
        # *just* under 1.15 — while the two batches put its gain on opposite sides
        # of 1.00. Printing that as 两批一致 because a threshold did not fire would
        # bury the one row that says the two batches are describing different
        # machines, and would do it in the direction that flatters the change.
        if disagreement is None:
            verdict = "读不出来"
        elif row["sign_flip"]:
            flipped.append(row)
            flagged.append(row)
            verdict = "**尚不能确认收益或回退**；**两批方向相反**"
            if disagreement <= SELF_CHECK_TOLERANCE:
                verdict += (
                    f"（幅度 {disagreement:.3f}× 未过 "
                    f"{SELF_CHECK_TOLERANCE:.2f}× 门限，但方向相反本身就是判据）"
                )
        elif disagreement > SELF_CHECK_TOLERANCE:
            flagged.append(row)
            verdict = "**尚不能确认收益或回退**"
        else:
            verdict = "两批一致"
        def cell(side: str) -> str:
            return (f"{medians[first][side]:.2f} → {medians[second][side]:.2f}")
        gap = "—" if disagreement is None else f"{disagreement:.3f}×"
        line.append(
            f"| {row['short']} | {row['label']} | {cell('base')} | {cell('opt')} | "
            f"{gains[first]:.3f}× | {gains[second]:.3f}× | "
            f"{gap} | "
            f"{windows[first]} / {windows[second]} | {verdict} |"
        )
    line.append(
        f"\n幅度判据与上面那张表同一条：**两侧之比 >{SELF_CHECK_TOLERANCE:.2f}×，"
        "与方向无关**；方向相反的行**不看幅度**，一律判出去。"
        f"本批 {len(rows)} 行里 {len(flagged)} 行被判 `尚不能确认收益或回退`"
        + (
            f"，其中 {len(flipped)} 行**两批的增益落在 1.00 的两侧**——"
            "同一棵 base 树、同一棵 opt 树、同一天，一批读作变快、另一批读作变慢。"
            if flipped
            else "。"
        )
    )
    if flagged:
        widest = max(flagged, key=lambda r: r["base_drift"] or 0)
        line.append(
            "\n**最干净的证据在 base 侧自己身上**：同一探针、同一棵基线树、同一天，"
            f"{widest['short']} 的「{widest['label']}」两遍差 "
            f"{widest['base_drift']:.3f}×"
            f"（{widest['medians'][first]['base']:.2f} → "
            f"{widest['medians'][second]['base']:.2f} ms/tok）。"
            "基线树在两遍之间不可能变，所以这个差**只能是机器状态**——"
            "而它比被比较的那个改动还大时，同一行的 base→opt 增益就没有分辨力。\n"
        )
        line.append(
            "这些行一律 `尚不能确认收益或回退`：既不能说改动带来了收益，"
            "也不能说它带来了回退。要分开只能靠**多轮独立进程配对 + AB/BA 顺序均衡**"
            "（§7.3），不是靠在两遍里挑一遍。\n"
        )


def rerun_contradiction(line: list[str], primary: str) -> None:
    """Print where two runs of the same workload disagree, instead of picking one.

    The end-to-end suite ran twice this round: once inside ``run_round1.sh`` step 1
    at ``--max-context 1024`` and once at 2048 to recover the long-prompt workload.
    The workloads both passes measured are identical — same tree, same machine,
    hours apart — and on the 6.4 GiB model they do not agree. That disagreement is
    more informative than either table alone: it measures how much of an end-to-end
    number on this machine is the machine.

    Only shown when the primary source is the re-run, so the reader sees the table
    above and the contradiction below as a pair. The bare-decode subsection prints
    even when no *workload* row crossed the threshold: those are different
    quantities, and a batch whose end-to-end numbers happen to agree can still have
    decode probes that do not.
    """
    if primary != "ab-longctx":
        return
    rows = contradiction_rows(PERF)
    micro = micro_decode_rows(PERF)
    if not rows and not micro:
        return
    line.append("\n### 两次运行不一致的地方\n")
    if not rows:
        line.append(
            "**本批没有任何工作负载的两遍 `总加速` 相差超过 0.10。**"
            "下面那一节仍然印：裸 decode 探针是另一个量，"
            "端到端对得上不代表它也对得上。\n"
        )
        micro_decode_contradiction(line, primary)
        return
    loads = compared_workloads(PERF)
    counts = sorted({len(names) for names in loads.values()})
    if len(counts) == 1:
        load_text = f"两遍都测到的同样 {counts[0]} 个工作负载"
    else:
        load_text = "**两遍都测到的工作负载，逐模型不同**（" + "、".join(
            f"{short} {len(names)} 个" for short, names in sorted(loads.items())
        ) + "）"
    line.append(
        f"同一棵 opt 树、同一台机器、{load_text}，本轮跑了两遍"
        "（`ab/` 在 `run_round1.sh` 第 1 步，`ab-longctx/` 在几小时后补跑长 prompt 时）。"
        f"下表只列出两遍 `总加速` 相差超过 0.10 的行（本批 {len(rows)} 行，"
        f"涉及 {len({r[1] for r in rows})} 个负载、{len({r[0] for r in rows})} 个模型）。"
        "**这些差不是代码造成的，"
        "两遍之间代码一个字都没改：**\n"
    )
    line.append("| 模型 | 工作负载 | `ab/` 总加速 | `ab-longctx/` 总加速 | 两遍的 decode 自检 |")
    line.append("| --- | --- | --- | --- | --- |")
    for short, name, first, second, ab_checks, longctx_checks in rows:
        line.append(
            f"| {short} | {name} | {first:.2f}× | {second:.2f}× | "
            f"{cross_check_cell(ab_checks, longctx_checks)} |"
        )
    # Whether the two batches can cross-check each other is a property of the data,
    # not of the round. The published batch's ``ab-longctx/`` ran only the workloads
    # suite, so its ``micro`` is empty and its rows genuinely have no same-window
    # self-check. The re-collection's does not — the driver was changed to
    # ``--suite all --no-chat`` — and printing "拿不到自检" there both states
    # something false and discards the one measurement that could say whether the
    # disagreement below is the machine or the change.
    #
    # The three comparisons are not interchangeable and must not be collapsed into
    # 一致/对不上: a batch that disagrees with itself cannot corroborate anything,
    # which is a weaker finding than the two batches disagreeing with *each other*
    # about the same tree. Counting them apart is what keeps the prose below from
    # asserting a cross-batch contradiction for a row that only has an internal one.
    agreed: list[tuple] = []
    across: list[tuple] = []
    internal: list[tuple] = []
    for row in rows:
        values = cross_check_values(row[4], row[5])
        if values is None:
            continue
        failures = cross_check_failures(values)
        if not failures:
            agreed.append(row)
        elif any(item.startswith(CROSS_BATCH_LABEL) for item in failures):
            across.append(row)
        else:
            internal.append(row)
    readable = len(agreed) + len(across) + len(internal)
    line.append(
        "\n这就是为什么这张表的每一行都要配 `decode 自检` 列：在这台机器上，"
        "一个孤立的端到端数字不能单独作为结论。"
    )
    if not readable:
        line.append(
            f"**但这 {len(rows)} 行拿不到跨遍自检**——"
            "`ab-longctx/` 当时只跑了 `--suite workloads`，JSON 里 `micro` 为空，"
            "所以两遍都无法互相校验，上面那些行全部标 **待重测**。\n"
        )
    else:
        line.append(
            f"**这 {len(rows)} 行里有 {readable} 行两遍都带同窗口 `micro`，"
            "所以三项自检读得出来**（`run_round1_longctx.sh` 已改成 "
            "`--suite all --no-chat`）。"
            f"其中 {len(agreed)} 行三项全部一致；"
            f"{len(internal)} 行只是**某一遍自己内部**的 base↔opt 对不上"
            "——那一遍因此没法给另一遍作证，但两遍之间并不矛盾；"
            f"{len(across)} 行连**跨遍 opt** 都对不上，"
            "即同一棵树、同一个负载，两遍各自测出的 opt decode 自检比值互不相容，"
            "这才是「两遍在描述不同机器」的直接证据。"
            + (
                f"剩下 {len(rows) - readable} 行缺同窗口 `micro`，读不出来。"
                if readable < len(rows)
                else ""
            )
            + "**三项全部一致也只说明每一遍进程内部一致、且两遍 opt 侧的内部比值互相吻合，"
            "不说明两遍之间的总加速差是代码造成的**"
            "——代码一个字没改，那个差仍然是机器，上面那些行照旧全部标 **待重测**。\n"
        )
    line.append(
        "**本文早先在这里写过「`ab/` 那一遍里 Qwen3-1.7B 报出的 0.59× / 0.71× / 0.84× "
        "不是回退，补跑后同一批负载是 1.05× ~ 1.29×」——那句话撤回。**"
        "它只凭「补跑更快」就断言旧结果不是回退，而两次运行之间什么证据都没留下："
        "没有同窗口自检，没有多轮重复，没有系统指标。正确的说法是"
        "**尚不能确认收益或回退**：0.59× 既没被推翻也没被证实，"
        "两遍的原始 JSON 都保留在原目录。\n"
    )
    line.append(
        "早先这一段还把原因写成「它的工作集会被 macOS 压缩器换出」，并引 validation.md "
        "里那次 115 s 对 3828 s 的集成测试作依据。**那也撤回**：那是**另一次**测量窗口里的"
        "观察，本轮没有在这两个 A/B 窗口内采集任何系统证据（`vm_stat`、压缩器字节数、"
        "swap 用量、进程 RSS 时间序列都没有），所以内存压力在这里只是一个*假设*，"
        "不能写成原因。详见 §9.4 与 [测试与数值验证](validation.md)。\n"
    )
    primary_loads = batch_workloads(PERF, primary)
    other_loads = batch_workloads(PERF, "ab")
    line.append(
        f"**两遍都不是唯一真相；上面那张表用补跑的一遍，仅仅是因为它把负载测全了"
        f"（`{primary}/` 逐模型 "
        + "、".join(
            f"{short} {len(names)} 个" for short, names in sorted(primary_loads.items())
        )
        + f"，而 `ab/` 有模型只测到 "
        + str(min((len(names) for names in other_loads.values()), default=0))
        + " 个），不是因为它更快。**\n"
    )
    micro_decode_contradiction(line, primary)


# --------------------------------------------------------------------------
# opt 3: session reuse
# --------------------------------------------------------------------------
def section_chat(line: list[str]) -> None:
    line.append("\n## 优化 3：多轮 KV 复用（`chat` 套件）\n")
    rows = [
        (model, base.get("chat"), opt.get("chat"))
        for model, (base, opt) in ab_payloads().items()
    ]
    if not rows:
        missing(line, "ab/*.json 里的 chat 套件")
        return

    line.append("### 会话内逐轮复用\n")
    line.append(
        "`cold` 是每轮都新建引擎请求（第 1 轮之前的行为），`warm` 是同一个 `Session` 跑到底。"
        "**两轮在 cold 下的输出必须相同**，否则 reuse 改变了结果；"
        "`reused` 是引擎自己报告的复用 token 数，不是从 prompt 长度推的。\n"
    )
    for model, base_chat, opt_chat in rows:
        if not opt_chat:
            continue
        line.append(f"\n**{SHORT[model]}**（`session_supported`：base="
                    f"{(base_chat or {}).get('session_supported')}，opt={opt_chat.get('session_supported')}）\n")
        line.append("| 轮 | prompt | cold 复用 | cold prefill | warm 复用 | warm prefill | cold 首token | warm 首token | 首token 加速 | cold 总 | warm 总 |")
        line.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        cold = opt_chat.get("cold", {}).get("turns", {})
        warm = opt_chat.get("warm", {}).get("turns", {})
        for turn in sorted(cold, key=lambda t: int(t)):
            c = cold[turn]
            w = warm.get(turn)
            if not w:
                continue
            c_ttft = stat(c.get("first_token_seconds"), "median")
            w_ttft = stat(w.get("first_token_seconds"), "median")
            line.append(
                f"| {int(turn) + 1} | {c['prompt_tokens']} | {c['reused_tokens']} | "
                f"{c['prefill_tokens']} | **{w['reused_tokens']}** | **{w['prefill_tokens']}** | "
                f"{ms(c_ttft)} | {ms(w_ttft)} | {ratio(c_ttft, w_ttft)} | "
                f"{ms(stat(c.get('wall_seconds'), 'median'))} | "
                f"{ms(stat(w.get('wall_seconds'), 'median'))} |"
            )

    line.append("\n### 整段对话与一致性\n")
    line.append("| 模型 | cold 四轮合计 | warm 四轮合计 | 加速 | 比较轮数 | 比较轮次×轮 | 不一致 | 第二份 cache |")
    line.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for model, base_chat, opt_chat in rows:
        if not opt_chat:
            continue
        total = opt_chat.get("conversation_wall_seconds", {})
        agree = opt_chat.get("agreement", {})
        allocated = opt_chat.get("warm", {}).get("cache_allocated_bytes")
        line.append(
            f"| {SHORT[model]} | {ms(total.get('cold_sum_of_turn_medians'))} | "
            f"{ms(total.get('warm_sum_of_turn_medians'))} | "
            f"{total.get('speedup', float('nan')):.3f}× | {agree.get('compared_rounds', '?')} | "
            f"{agree.get('compared_turns', '?')} | {len(agree.get('mismatched', []))} | "
            f"{allocated / 2**20:.1f} MiB |" if allocated else
            f"| {SHORT[model]} | {ms(total.get('cold_sum_of_turn_medians'))} | "
            f"{ms(total.get('warm_sum_of_turn_medians'))} | "
            f"{total.get('speedup', float('nan')):.3f}× | {agree.get('compared_rounds', '?')} | "
            f"{agree.get('compared_turns', '?')} | {len(agree.get('mismatched', []))} | — |"
        )

    line.append("\n### base 树 cold 对 opt 树 cold：加 Session 有没有拖慢普通请求\n")
    line.append(
        "这一节回答的不是复用值不值，而是**新增会话代码有没有让原来那条独立请求路径变慢**。"
        "两侧都不复用 KV，差异只来自优化 1 与 2。\n"
    )
    line.append("| 模型 | 轮 | base cold 总 | opt cold 总 | 加速 | 输出一致 |")
    line.append("| --- | --- | --- | --- | --- | --- |")
    not_faster: list[str] = []
    mismatched: list[str] = []
    for model, base_chat, opt_chat in rows:
        if not base_chat or not opt_chat:
            continue
        b_turns = base_chat.get("cold", {}).get("turns", {})
        o_turns = opt_chat.get("cold", {}).get("turns", {})
        for turn in sorted(b_turns, key=lambda t: int(t)):
            b = b_turns[turn]
            o = o_turns.get(turn)
            if not o:
                continue
            b_wall = stat(b.get("wall_seconds"), "median")
            o_wall = stat(o.get("wall_seconds"), "median")
            same = b.get("text_sha") == o.get("text_sha")
            label = f"{SHORT[model]} 第 {int(turn) + 1} 轮"
            if not same:
                mismatched.append(label)
            elif b_wall and o_wall and o_wall > b_wall:
                not_faster.append(f"{label}（{b_wall * 1e3:.1f} → {o_wall * 1e3:.1f} ms）")
            line.append(
                f"| {SHORT[model]} | {int(turn) + 1} | {ms(b_wall)} | {ms(o_wall)} | "
                f"{ratio(b_wall, o_wall)} | {'是' if same else '**否**'} |"
            )

    # The output column is the one this table exists for: it asks whether adding
    # Session changed the standalone path, and that answer is timing-independent.
    # The timing column is still worth disowning where it reads as a regression,
    # because the chat suite only ran in step 1's window — the same window whose
    # Qwen3-1.7B numbers the workloads section proves were contaminated.
    if mismatched:
        line.append(
            f"\n**{len(mismatched)} 行输出不一致：{'、'.join(mismatched)}。**"
            "这才是真回退，与会话代码有关，必须修。\n"
        )
    else:
        line.append(
            "\n**全部行 `输出一致=是`**：独立请求路径的输出没有被新增的会话代码改变，"
            "这是这张表要回答的问题，且它与计时无关。\n"
        )
    if not_faster:
        line.append(
            f"\n**{len(not_faster)} 行的耗时读起来像回退，标记待重测：**"
            f"{'、'.join(not_faster)}。"
            "`chat` 套件只在第 1 步那个窗口跑过，`run_round1_longctx.sh` 补跑的只有 "
            "workloads 套件——也就是说这几行与上面「两次运行不一致的地方」表里被判为"
            "受污染的那批数字来自**同一个窗口**。"
            f"结论是 **{INDETERMINATE}**："
            "补跑那一侧更快**不能**用来证明这几行不是回退——两次测量不在同一窗口，"
            "谁快谁慢都可能是那一阵的机器，拿一次更快的补跑推翻一次更慢的原测，"
            "和拿原测断言回退是同一种错误，只是方向相反。"
            "**这几行的计时按不可用处理；`输出一致` 那一列不受影响，仍然成立。**"
            "要把它变成可用证据，需要在同一窗口内相邻交替重跑 `chat` 套件，本轮没有做。\n"
        )


# --------------------------------------------------------------------------
# opt 5: thread and chunk sweep
# --------------------------------------------------------------------------
def section_sweep(line: list[str]) -> None:
    line.append("\n## 优化 5：线程数与 prefill 分块参数\n")
    sweep_models = SWEEP_MODELS
    thread_tags = ["t-default", "t1", "t4", "t8"]
    chunk_tags = ["c64", "c256", "c512"]
    base_dir = PERF / "baseline"
    opt_dir = PERF / "sweep-opt"

    def collect(directory: Path, pattern: str) -> dict[str, Any]:
        found = {}
        for path in sorted(directory.glob(pattern)):
            found[path.stem] = load(path)
        return found

    base_files = collect(base_dir, "*-default-*.json")
    opt_files = collect(opt_dir, "*.json")
    if not opt_files:
        missing(line, "sweep-opt/*.json（run_round1.sh 第 2 步）")
        base_only = True
    else:
        base_only = False

    line.append(
        "每个配置一个**独立进程**。`t-default` 是完全不设线程环境变量，"
        "也就是出厂默认——它必须被测量，不能假设等于任何一个显式数字。"
        "`requested` 是请求的线程数，**有效工作线程数在 Python 里不可观测**，"
        "harness 记为 null，所以下面任何结论都只针对「请求了这个数」而不是「跑起来用了这么多线程」。\n"
    )
    if not base_files:
        # An em-dash in a base column is ambiguous on its own: it reads equally as
        # "not measured" and as "the report failed to find it". The two have
        # different consequences — the first means the sweep is opt-side only and
        # no base→opt comparison in this section is supported at all.
        line.append(
            f"**本轮没有 base 侧扫描数据**（`{repo_relative(base_dir)}/` 下没有 "
            "`*-default-*.json`）。已发布那一轮的 `baseline/` 是**本轮之前**那棵树跑的，"
            "重采时没有重建它——重建需要先检出一个更早的 revision，"
            "而重采的目标是用修复后的最终源码，不是用旧树。"
            "所以下面两张表的 base 列全部是 `—`，"
            "**本节不支持任何 base→opt 比较**；能说的只有 opt 侧各配置之间的相对关系，"
            "而且判读要用 §7 的跨进程离散度，不是同一进程内的 min–max 区间。\n"
        )
    floor = noise_floor(line, sweep_models, base_files, opt_files)

    for model in sweep_models:
        line.append(f"\n### {SHORT[model]}\n")
        line.append("**线程数**（`prefill_chunk_size=128`，512 token 分块 prefill / decode）\n")
        line.append(
            "| 配置 | 请求线程 | 环境变量在 import numpy 前设好 | base 512 prefill | "
            "opt 512 prefill | opt 区间 | base decode ms/tok | opt decode ms/tok | 有效线程 |"
        )
        line.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for tag in thread_tags:
            b = base_files.get(f"{model}-default-{tag}")
            o = opt_files.get(f"{model}-opt-{tag}")
            if not b and not o:
                continue
            threads = (b or o).get("threads", {})
            requested = threads.get("requested")
            effective = threads.get("effective_thread_count")
            before = threads.get("requested_before_numpy_import")
            b_pre = stat((b or {}).get("micro", {}).get("engine_prefill", {}).get("512"), "median")
            o_node = (o or {}).get("micro", {}).get("engine_prefill", {}).get("512")
            o_pre = stat(o_node, "median")
            b_dec = stat((b or {}).get("micro", {}).get("raw_decode"), "ms_per_token")
            o_dec = stat((o or {}).get("micro", {}).get("raw_decode"), "ms_per_token")
            line.append(
                f"| {tag} | {requested if requested is not None else '未设置'} | "
                f"{'是' if before else '**否**'} | "
                f"{ms(b_pre)} | {ms(o_pre)} | {range_only(o_node)} | "
                f"{'—' if b_dec is None else f'{b_dec:.2f}'} | "
                f"{'—' if o_dec is None else f'{o_dec:.2f}'} | `{effective}` |"
            )
        thread_summary(line, model, opt_files, thread_tags, floor)

        line.append(
            "\n**prefill 分块大小**（线程未设置；`c128` 即出厂默认，见上面 `t-default` 行）。"
            "`相对默认` 是 opt 侧对 `c128` 的百分比，负数表示更快；"
            "判断它是否可信要和**跨进程的重复测量**比，不是和同一进程内的区间比——"
            "下面每行给的是该配置自己那 7 次重复的观测区间，判读规则写在表后。\n"
        )
        line.append(
            "| 配置 | chunk | base 512 prefill | opt 512 prefill | opt 区间 | 相对默认 | opt 投影 | 峰值 RSS（opt） |"
        )
        line.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        default_b = base_files.get(f"{model}-default-t-default")
        default_o = opt_files.get(f"{model}-opt-t-default")
        configs = [("c128（默认）", "128", default_b, default_o)]
        for tag in chunk_tags:
            configs.append((tag, tag[1:], base_files.get(f"{model}-default-{tag}"), opt_files.get(f"{model}-opt-{tag}")))
        reference = stat((default_o or {}).get("micro", {}).get("engine_prefill", {}).get("512"), "median")
        for label, chunk, b, o in configs:
            if not b and not o:
                continue
            b_node = (b or {}).get("micro", {}).get("engine_prefill", {}).get("512")
            o_node = (o or {}).get("micro", {}).get("engine_prefill", {}).get("512")
            median = stat(o_node, "median")
            if not reference or not median:
                delta = "—"
            elif abs(median - reference) / reference < 1e-9:
                delta = "基准"
            else:
                delta = f"{100 * (median / reference - 1):+.1f}%"
            rss = (o or b or {}).get("rss_final_bytes")
            line.append(
                f"| {label} | {chunk} | {ms(stat(b_node, 'median'))} | {ms(median)} | "
                f"{range_only(o_node)} | {delta} | "
                f"{(o_node or {}).get('vocabulary_projections', '?')} | "
                f"{'—' if not rss else f'{rss / 2**30:.2f} GiB'} |"
            )
        chunk_summary(line, model, opt_files, (default_o or {}).get("micro", {}).get("engine_prefill", {}).get("512"), floor)
    if base_only:
        line.append("\n（上表 opt 列全为 `—`：优化后那一侧的扫描尚未执行。）\n")

    line.append(
        "\n**关于 base 侧没有 `投影` 列**：`baseline/` 那一批扫描是本轮早些时候跑的，"
        "当时的 harness 还没有 `entry_points` / `vocabulary_projections` 这两个字段"
        "（JSON 里确实没有这两个 key，不是读不出来）。base 树在 chunk 128、512 token 下"
        "做 4 次词表投影这一点，由后来同一棵 base 树、同一个 harness 跑出的 "
        "`ab/*-base.json` 记录，见上面「引擎分块 prefill」表的 `base 投影` 列。"
        "两批 base 测量的时间差也顺手做了核对：\n"
    )
    for model in sweep_models:
        late = base_files.get(f"{model}-default-t-default")
        early = load(PERF / "ab" / f"{model}-base.json")
        a = stat((late or {}).get("micro", {}).get("engine_prefill", {}).get("512"), "median")
        c = stat((early or {}).get("micro", {}).get("engine_prefill", {}).get("512"), "median")
        if a and c:
            line.append(
                f"- {SHORT[model]}：`baseline/` 扫描 {a * 1e3:.1f} ms vs `ab/` 相邻交替 "
                f"{c * 1e3:.1f} ms，相差 {100 * (a / c - 1):+.1f}%。"
            )
    line.append(
        "\n两批不是相邻交替跑的（`baseline/` 在前、`ab/` 在后，中间隔了几小时），"
        "所以上面这个差就是「不交替」的代价量级，读表时按这个精度读。\n"
    )


def noise_floor(
    line: list[str],
    sweep_models: list[str],
    base_files: dict[str, Any],
    opt_files: dict[str, Any],
) -> float | None:
    """How far apart independent processes land on the same measurement.

    Every table in this round prints a within-process min–max range, and those
    ranges are tight enough (often ±0.5%) to make a 3% difference between two
    configurations look decisive. It is not: the tight range only says the seven
    repeats inside one process agreed. Two processes started minutes apart on the
    same tree with the same arguments can land further apart than that.

    Prefers the AB/BA paired experiment ``run_noise_floor.sh`` produces, which
    measures this per quantity from several rounds. Falls back to the two incidental
    same-configuration runs this round happens to contain, and says so loudly —
    that fallback is where the published "3% noise floor" came from, and two data
    points cannot support a floor for every workload in the report.

    The choice is made on whether the experiment directory exists, not on whether
    the paired section managed to return a number: both sections write into
    ``line``, so falling back on a None return would print two contradictory
    accounts of the same question.

    The paired section is handed **every** model, not just the two the sweep tables
    cover. ``run_noise_floor.sh`` measures all five, and this section is what
    replaced the withdrawn "3% floor" as the judgement basis for §2–§6 as well as
    §7 — restricting it to ``sweep_models`` printed two models' rows and left the
    other three with no paired verdict on prefill, decode or end-to-end at all.
    The *return value* stays sweep-specific, because that is the only thing the two
    sweep tables below are allowed to quote.
    """
    if paired_directory() is not None:
        return paired_noise_section(line, MODELS)
    return incidental_noise_section(line, sweep_models, opt_files)


#: The probe the incidental fallback happens to have two independent runs of.
FALLBACK_PROBE = ("engine_prefill", "512")


def paired_quantities(payload: Any) -> dict[str, tuple[float, list[float]]]:
    """Every quantity in one payload that has both a median and a sample set.

    ``{name: (median, samples)}``, all in milliseconds. Covers the micro probes and
    each workload's own wall clock, so the paired analysis judges every row on that
    row's data. Borrowing one probe's spread for the whole report is the specific
    defect the "3% floor" had: it came from ``engine_prefill 512`` and was then
    applied to decode, to end-to-end workloads and to a thread sweep alike.
    """
    out: dict[str, tuple[float, list[float]]] = {}
    micro = payload.get("micro") or {}
    for key, prefix in (("engine_prefill", "引擎分块 prefill"), ("raw_prefill_full", "整块 prefill")):
        for length, node in sorted((micro.get(key) or {}).items()):
            median = stat(node, "median")
            samples = [s * 1e3 for s in (node.get("samples") or []) if isinstance(s, (int, float))]
            if median and samples:
                out[f"{prefix} {length} tok"] = (median * 1e3, samples)
    for key, label in (("raw_decode", "decode 短上下文"), ("raw_decode_long", "decode 长上下文")):
        node = micro.get(key)
        if not isinstance(node, dict):
            continue
        windows = node.get("ms_per_token_windows")
        samples = windows.get("samples") if isinstance(windows, dict) else None
        median = decode_median(node)
        if median is None:
            continue
        # A single-window node contributes one sample, which spread_ratio reports as
        # no dispersion at all rather than as zero dispersion — the distinction the
        # 单窗口观察值 labelling exists to keep visible.
        out[label] = (median, list(samples) if samples else [median])
    for name, node in sorted((payload.get("workloads") or {}).items()):
        if not isinstance(node, dict) or "skipped" in node:
            continue
        total = node.get("total_seconds")
        median = stat(total, "median")
        # A workload node records its repeats in ``runs``, one dict per run; its
        # ``total_seconds`` is the *summary* (median/min/max/stdev) and carries no
        # sample list. Reading samples off the summary silently yields [] and drops
        # every end-to-end row from the paired analysis, which is what this did
        # until it was caught: the section below claims to cover "each workload's
        # own wall clock" while receiving only the ten micro probes. Verified
        # against all 184 workload nodes in noise-floor/: median(min/max/stdev) of
        # the per-run values reproduce the recorded summary exactly.
        samples = [
            run.get("total_seconds") * 1e3
            for run in (node.get("runs") or [])
            if isinstance(run, dict) and isinstance(run.get("total_seconds"), (int, float))
        ]
        if median and samples:
            out[f"{name}（端到端）"] = (median * 1e3, samples)
    return out


def paired_noise_section(line: list[str], models: list[str]) -> float | None:
    """Per-quantity within/between dispersion and paired-gain verdicts.

    Returns None when the experiment does not exist, so the caller can fall back
    and label the fallback rather than silently presenting two runs as a
    distribution.

    Three separate numbers per row, because they answer three separate questions
    and quoting any one of them alone misleads:

    * *within* — how reproducible one process is with itself.
    * *between* — how far apart independent processes land on the same quantity.
      This is the resolution of a cross-process comparison, and it is the column
      the round-1 report did not have.
    * *paired gain* — base/opt recomputed **per round**, then classified by
      :func:`gain_verdict`. A row whose rounds straddle 1.0 is reported as
      尚不能确认收益或回退 no matter how good its median looks.
    """
    global PAIRED_SUMMARY, PAIRED_BETWEEN
    collected = [
        (model, paired_rounds(model, lambda payload: paired_quantities(payload)))
        for model in models
    ]
    collected = [(model, data) for model, data in collected if data]
    if not collected:
        return None

    line.append("\n### 跨进程离散度与逐轮配对增益（读下面两张表之前先看这个）\n")
    source = paired_directory()
    line.append(
        f"数据来自 `tools/perf/run_noise_floor.sh` 写入的 `{NOISE_DIR}/`，"
        f"**本表读的是 `{repo_relative(source.parent) if source else '（未运行）'}`**"
        + (
            "，不是 `--data` 那个根目录下的同名子目录。重测单独占一个目录，"
            "是因为已发布的那一份是撤回结论背后的原始证据，"
            "重测只能**加**在它旁边，不能替换它；两者都在磁盘上。\n"
            if source is not None and source.parent != PERF
            else "。\n"
        )
        + "每对 base／opt "
        "在独立进程里背靠背跑，多轮之间 **AB／BA 顺序交替**，所以「谁排在第二个进程」"
        "不会与「谁是优化版」混在一起——第二个进程总是受益于已经热起来的 page cache，"
        "顺序不均衡的话那份好处会被记成优化的收益。\n"
    )
    line.append(
        "**每一行用这一行自己的多轮数据判读**，不引用任何统一的噪声百分比："
        "`进程内` 是各进程自己 7 次重复的离散，`进程间` 是各轮中位数之间的离散，"
        "`逐轮增益` 是每一轮单独算一遍 base/opt 得到的那组数，"
        "`结论` 看这组数是否跨过 1.00。跨过了就是 "
        f"`{INDETERMINATE}`，与中位数落在哪一侧无关。\n"
    )
    line.append(
        f"`弃用轮次` 是被污染判据剔掉的轮次，判据有两条且**对 base／opt 同样严格**："
        f"**spike**（一个节点内 `max/median > {SPIKE_LIMIT}`，抓「7 个样本里有 1 个停了"
        f"几十秒」）与 **level**（该轮中位数 ÷ 各轮中位数的中位 `> {LEVEL_LIMIT}`，抓"
        "「整轮一起变慢」——spike 抓不到它，因为 7 个样本一起抬高时 `max/median` 仍然接近 1）。"
        "两条都是量与自身尺度之比，所以不含毫秒门限，换一台更快的机器也不会失效。"
        "**判据按「量 × 轮 × 侧」逐格生效，不按整份文件生效**：一个进程在 512-token "
        "prefill 上停过，它的 decode 窗口仍然是可用的，按文件弃用会把那些一起丢掉。\n"
    )
    line.append(
        f"`有效轮次` 与 `顺序均衡（保留轮次）` 两列都是**剔完之后重算的**，"
        "不是采集时的那个数。这一步是必须的：AB／BA 均衡是「谁排在第二个进程」"
        "的均衡，而第二个进程继承热起来的 page cache；剔掉一轮就可能把 2＋2 变成 "
        "2＋1，那时剩下的轮次里有一棵树系统性地占便宜，"
        "**用它算出的方向不是测量的方向**。所以本表把 `轮数` 换成 `有效轮次`"
        "（真正算出配对增益的轮数），把 `顺序均衡` 明确标注为按保留轮次计算，"
        "并且**只要这一列不是「是」，`结论` 就不给方向**——"
        f"改判 {UNORDERED}；有效轮次少于 {MIN_RETAINED_ROUNDS} 则改判 {INSUFFICIENT}。\n"
    )
    line.append(
        "| 模型 | 量 | 有效轮次 | 弃用轮次 | 顺序均衡（保留轮次） | 进程内离散 base/opt | "
        "进程间离散 base/opt | 逐轮增益 | 结论 |"
    )
    line.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    unbalanced: list[str] = []
    #: Rows whose *retained* rounds are not balanced, and rows left with too few of
    #: them. Separate from ``unbalanced`` because they are different findings about
    #: different round sets: one describes what was collected, the other what
    #: survived, and only the second one decides a verdict.
    unbalanced_rows: list[str] = []
    insufficient_rows: list[str] = []
    indeterminate: list[str] = []
    directional: list[tuple[str, str, str, list[float]]] = []
    row_count = 0
    verdict_counts: dict[str, int] = {}
    all_gains: list[float] = []
    #: ``[width, model, quantity, round, side]`` of the widest within-process range
    #: that survived both contamination criteria — the blind spot's worst case.
    diffuse_worst: list[Any] = [0.0, "", "", "", ""]
    excluded_cells = 0
    excluded_pairs = 0
    excluded_detail: list[str] = []
    # Collected for the one quantity the two sweep tables below actually compare
    # against, rather than as a max over every row: a single number summarising
    # decode, prefill and end-to-end workloads alike is precisely what the "3%
    # noise floor" was, and this section exists to stop doing that.
    sweep_quantity = f"引擎分块 prefill {FALLBACK_PROBE[1]} tok"
    sweep_opt_between: list[float] = []
    for model, data in collected:
        rounds = data["rounds"]
        # The balance of the rounds as *collected*. Kept because "this model's driver
        # ran base first every round" is worth saying on its own, but it is not what
        # any verdict below is allowed to read: a verdict reads the retained set,
        # which is recomputed per quantity inside the loop.
        raw_balanced = data["order_balanced"]
        if raw_balanced is not True:
            unbalanced.append(
                f"{SHORT[model]}：{data['round_count']} 轮里 base 先跑 "
                f"{data['base_first_rounds']} 轮"
                + ("（标签里没有位置信息，无法核对）" if raw_balanced is None else "（不均衡）")
            )
        # Only quantities present in every round on both sides are comparable;
        # a workload one round skipped cannot contribute a paired gain.
        common = set.intersection(
            *(set(sides[side]["value"]) for sides in rounds.values() for side in ("base", "opt"))
        ) if rounds else set()
        for name in sorted(common):
            # Contamination is judged per quantity, not per file. A process that
            # stalled on one 512-token prefill still produced usable decode
            # windows, and discarding the whole file throws those away too — which
            # for the noisiest model in this run would leave a single round and no
            # verdict at all. Discarding only the affected measurement is both
            # stricter where it matters and more informative everywhere else.
            bad: dict[str, dict[int, tuple[str, float]]] = {}
            for side in ("base", "opt"):
                bad[side] = contaminated_rounds(
                    {
                        number: sides[side]["value"][name][1]
                        for number, sides in rounds.items()
                        if name in sides[side]["value"]
                    }
                )
            usable = {
                number: sides
                for number, sides in rounds.items()
                if number not in bad["base"] and number not in bad["opt"]
            }
            dropped = sorted(set(rounds) - set(usable))
            for number in dropped:
                # Both sides can be bad in the same round, and each has its own
                # reason; reporting only the first would hide that the rule fired
                # twice and understate how much of the round was unusable.
                for side in ("base", "opt"):
                    hit = bad[side].get(number)
                    if not hit:
                        continue
                    reason, value = hit
                    excluded_cells += 1
                    excluded_detail.append(
                        f"{SHORT[model]} / {name} / r{number} 的 {side} 侧"
                        f"（{reason} {value:.2f}×，门限 "
                        f"{SPIKE_LIMIT if reason == 'spike' else LEVEL_LIMIT}）"
                    )
                excluded_pairs += 1
            # Balance recomputed over the rounds this row actually kept: same
            # function, same rule, different round set. That difference is the fix —
            # it used to be called once per model over the *collected* rounds, so a
            # row the filter had already cut from 2 AB + 2 BA down to 1 AB + 2 BA
            # still printed 「顺序均衡：是」 and still handed its direction to the
            # summary. Dropping a round cannot create balance, and the second process
            # in each pair is the one holding the warm page cache.
            retained_balanced, retained_base_first = _order_balance(usable)
            per_side = {}
            gains = []
            for number, sides in usable.items():
                b_median = sides["base"]["value"][name][0]
                o_median = sides["opt"]["value"][name][0]
                if b_median and o_median:
                    gains.append(b_median / o_median)
            all_gains.extend(gains)
            for side in ("base", "opt"):
                per_side[side] = within_between(
                    [sides[side]["value"][name][1] for sides in usable.values()]
                )
                # Widest within-process range among the rounds this row *kept*, so
                # the blind-spot disclosure below quotes a node that actually
                # survived the filter rather than one remembered from an earlier
                # analysis of the same directory.
                for number, sides in usable.items():
                    samples = [
                        s for s in sides[side]["value"][name][1]
                        if isinstance(s, (int, float)) and s > 0
                    ]
                    if len(samples) >= 2:
                        width = (max(samples) - min(samples)) / statistics.median(samples)
                        if width > diffuse_worst[0]:
                            diffuse_worst[:] = [width, SHORT[model], name, f"r{number}", side]
            betweens = [per_side[side][1] for side in ("base", "opt")]
            # Only the sweep models may contribute here: the two tables below are
            # about those two models, and widening the pool would let a third
            # model's dispersion set the resolution they quote.
            if name == sweep_quantity and betweens[1] and model in SWEEP_MODELS:
                sweep_opt_between.append(100 * (betweens[1] - 1))
            # §9.2's comparison is per model, so this one is *not* gated on
            # SWEEP_MODELS: it records the same quantity for every model the paired
            # experiment covered, and the prose picks out the row whose model also
            # produced the share being compared.
            if name == sweep_quantity and all(betweens):
                PAIRED_BETWEEN[SHORT[model]] = (
                    100 * (betweens[0] - 1),
                    100 * (betweens[1] - 1),
                )
            # Three tests, in this order. A row left with fewer than
            # MIN_RETAINED_ROUNDS usable gains is *also* arithmetically unbalanced
            # (1 * 2 != 1), so asking about balance first would report one surviving
            # round as a design failure instead of as a thin sample. Both stop at "no
            # direction claimed"; only the reason differs.
            effective = [g for g in gains if isinstance(g, (int, float)) and g > 0]
            verdict = gain_verdict(effective)
            if len(effective) < MIN_RETAINED_ROUNDS:
                verdict = INSUFFICIENT
            elif retained_balanced is not True:
                verdict = UNORDERED
            # Recorded after the verdict, and only for the rows that got it, so the
            # two lists are exactly the two counts in ``verdict_counts``. Listing a
            # one-round row as "unbalanced after filtering" would be arithmetically
            # true and still misleading: what is wrong with it is that it is one row.
            if verdict == INSUFFICIENT:
                insufficient_rows.append(
                    f"{SHORT[model]} / {name}：只算出 {len(effective)} 个有效配对增益"
                    f"（保留 {len(usable)}/{data['round_count']} 轮"
                    + (f"，弃用 {'、'.join(f'r{n}' for n in dropped)}" if dropped else "")
                    + "）"
                )
            elif verdict == UNORDERED:
                unbalanced_rows.append(
                    f"{SHORT[model]} / {name}：保留 {len(usable)}/{data['round_count']} 轮"
                    + (
                        "，标签里没有位置信息，无法核对"
                        if retained_balanced is None
                        else f"，其中 base 先跑 {retained_base_first} 轮（不均衡）"
                    )
                    + (f"；弃用了 {'、'.join(f'r{n}' for n in dropped)}" if dropped else "")
                )
            balance_text = (
                "**无保留轮次**"
                if not usable
                else "是"
                if retained_balanced
                else (
                    "**未记录**"
                    if retained_balanced is None
                    else f"**否（base 先跑 {retained_base_first}/{len(usable)}）**"
                )
            )
            row_count += 1
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            if verdict not in (INDETERMINATE, *UNSETTLED):
                directional.append((SHORT[model], name, verdict, gains))
            if verdict == INDETERMINATE:
                indeterminate.append(f"{SHORT[model]} / {name}")
            line.append(
                f"| {SHORT[model]} | {name} | {len(effective)}/{data['round_count']} | "
                + ("、".join(f"r{n}" for n in dropped) if dropped else "—")
                + f" | {balance_text} | "
                + " / ".join(
                    "—" if per_side[s][0] is None else f"{100 * (per_side[s][0] - 1):.1f}%"
                    for s in ("base", "opt")
                )
                + " | "
                + " / ".join(
                    "—" if per_side[s][1] is None else f"{100 * (per_side[s][1] - 1):.1f}%"
                    for s in ("base", "opt")
                )
                + f" | {'—' if not gains else f'{min(gains):.2f}–{max(gains):.2f}×'} | "
                f"{verdict} |"
            )
    if excluded_detail:
        line.append(
            f"\n**被污染判据弃用的测量：{excluded_pairs} 个（量 × 轮）配对，"
            f"涉及 {excluded_cells} 格（量 × 轮 × 侧）：**\n"
        )
        for item in excluded_detail:
            line.append(f"- {item}")
        line.append(
            "\n两个数不同，是因为同一轮的 base 与 opt 可能**都**不合格，"
            "那时一个配对被弃用但记两格。弃用是对称的：这些格子里 base 侧与 opt 侧都有，"
            "判据只看形状不看方向，所以它不能通过专挑慢的那一侧来制造加速。"
            "**但它也不是万能的**——spike 抓不到「7 个样本整体都很散」的节点"
            "（这种节点的 `max/median` 仍然小于门限），level 抓不到轮内弥散。"
            + (
                f"本轮**通过**了两条判据、进程内极差却最大的一格是 "
                f"{diffuse_worst[1]} / {diffuse_worst[2]} / {diffuse_worst[3]} 的 "
                f"{diffuse_worst[4]} 侧，极差 {100 * diffuse_worst[0]:.0f}%。"
                if diffuse_worst[0]
                else ""
            )
            + "要补第三条判据（进程内极差超过门限也弃用）；本轮没有加，"
            "因为下面那张表里凡是受它影响的行，结论已经落在"
            f"「{INDETERMINATE}」，多弃不改变判定；"
            "**但其余行若出现同类弥散，这条缺失会让坏数据混进来。**\n"
        )
    if unbalanced:
        line.append("\n**采集时顺序就不均衡的模型：**\n")
        for item in unbalanced:
            line.append(f"- {item}")
        line.append(
            "\n不均衡的轮次不能用来判断收益方向：排在第二个进程的那棵树系统性地"
            "占便宜，这份便宜会被算进它的加速里。\n"
        )
    if unbalanced_rows:
        line.append("\n**剔除异常轮次之后顺序不再均衡的行：**\n")
        for item in unbalanced_rows:
            line.append(f"- {item}")
        line.append(
            "\n这些行**采集时是均衡的**，是污染判据剔掉轮次之后才不均衡的，"
            "所以它们不出在上面那份名单里，却同样不能判方向。"
            "本表把顺序均衡按保留轮次重算就是为了这一类："
            "上一版只按采集到的轮次算一次，于是这几行仍然印着「顺序均衡：是」，"
            "并且它们的增益被计进了「达到稳定收益判据」的分子。"
            f"现在它们一律判 `{UNORDERED}`，不计入分子，"
            "弃用的轮次与理由仍逐条列在上面。\n"
        )
    if insufficient_rows:
        line.append(
            f"\n**有效轮次不足 {MIN_RETAINED_ROUNDS} 的行：**\n"
        )
        for item in insufficient_rows:
            line.append(f"- {item}")
        line.append(
            f"\n一轮是一次观察，不是分布：它既没有极差也没有方向可言，"
            f"所以判 `{INSUFFICIENT}`，不计入任何分子。"
            "注意这与「增益跨过 1.00」是两件事，下面那份名单是后者。\n"
        )
    if indeterminate:
        line.append(
            f"\n**{len(indeterminate)} 个量 {INDETERMINATE}：**\n\n"
            + "、".join(indeterminate)
            + "\n\n这些量的逐轮增益跨过了 1.00，所以这一轮的测量**不能**用来声称"
            "它们变快或变慢。中位数看着再好也一样：中位数是把跨过的两端平均掉了。\n"
        )
    if directional:
        by_verdict: dict[str, list[tuple[str, str, list[float]]]] = {}
        for model, name, verdict, gains in directional:
            by_verdict.setdefault(verdict, []).append((model, name, gains))
        line.append(
            f"\n**{len(directional)} 个量达到「有限轮次方向一致」判据：**\n"
        )
        for verdict in ("收益", "回退"):
            rows = by_verdict.get(verdict)
            if not rows:
                continue
            line.append(f"\n- `{verdict}`：**{len(rows)} 个量**")
            for model, name, gains in rows:
                line.append(
                    f"  - {model} / {name}：{min(gains):.2f}–{max(gains):.2f}×"
                    f"（{len(gains)} 轮全部同向）"
                )
        line.append(
            "\n**这是「有限轮次方向一致」，不是「稳定收益」；"
            "本表不再把后一个词当作一个可以判给的档位。**"
            "接受判据在测量之前定好，四条缺一不可："
            f"① 每一轮单独算一遍 base/opt；② 保留下来的有效配对 ≥ {MIN_RETAINED_ROUNDS}；"
            "③ 保留下来的轮次 AB／BA 顺序均衡（`顺序均衡（保留轮次）` 列为「是」）；"
            "④ 没有任何一轮跨过 1.00。"
            f"不满足 ② 判 `{INSUFFICIENT}`，不满足 ③ 判 `{UNORDERED}`，"
            f"不满足 ④ 判 `{INDETERMINATE}`，三者都不计入上面那个分子。\n"
        )
        line.append(
            f"\n**真实样本量与不确定性：**每行只有 "
            f"{min(len(g) for *_, g in directional)}–"
            f"{max(len(g) for *_, g in directional)} 轮独立进程，"
            "不是几十次重复。四条判据给的是**方向**，不是**大小**，"
            "也不是「统计上不可能反号」："
            f"{min(len(g) for *_, g in directional)} 轮全部同向在抛硬币下本就有可能"
            f"（p = {2 ** -min(len(g) for *_, g in directional):.3g}），"
            "所以本节把幅度交给 `进程间` 那一列，而它比这里的极差大。"
            "要说「稳定收益」，需要的是预先声明的轮数下限、"
            "幅度超过进程间离散度、以及跨机器复现，本轮三者都没有。\n"
        )
    sign_n, sign_above, sign_p = sign_test(all_gains)
    if sign_n:
        line.append(
            f"\n把全部 {row_count} 行的逐轮增益汇到一起做符号检验："
            f"**{sign_n} 个配对里 {sign_above} 个大于 1.00**，"
            f"双侧精确二项 p = {sign_p:.3g}。"
            "**这个 p 值只回答「一枚硬币抛这么多次会不会这么偏」，"
            "不回答「每一行都变快了吗」**：同一个进程贡献了好几个量、"
            "同一台机器贡献了 5 个模型，这些行不独立，"
            "汇总的 p 值因此**高估**证据强度。要按行看，看上面那张表的 `结论` 列。"
            "**这个 p 值也把上面那些不均衡、轮次不足的轮次一起算了进去**，"
            "所以它比逐行判据更宽松，不能拿来替代那一列。\n"
        )
    PAIRED_SUMMARY = {
        "models": len(collected),
        "rounds": max((data["round_count"] for _, data in collected), default=0),
        "rows": row_count,
        "verdicts": dict(verdict_counts),
        "directional": len(directional),
        "indeterminate": len(indeterminate),
        "unbalanced": len(unbalanced),
        #: The two counts that keep §7.2 and §9.1 from quoting a numerator the table
        #: does not support. ``unbalanced_rows`` is the one the old prose could not
        #: see at all: rows balanced as collected and unbalanced once the filter ran.
        "unbalanced_rows": len(unbalanced_rows),
        "insufficient_rows": len(insufficient_rows),
        "excluded_pairs": excluded_pairs,
        "excluded_cells": excluded_cells,
        "sign": (sign_n, sign_above, sign_p) if sign_n else None,
        #: The sample size §9.1 quotes, so it quotes the one §7.3 computed. Both
        #: mean "how many rounds survived the filter, over the rows that got a
        #: direction"; a range typed into the summary goes stale the first time the
        #: paired experiment is re-run with a different number of rounds, and then
        #: §9.1 contradicts the table two sections earlier.
        "round_span": (
            (
                min(len(g) for *_, g in directional),
                max(len(g) for *_, g in directional),
            )
            if directional
            else None
        ),
    }
    # One set of verdicts, checked against itself before anything downstream reads
    # it: the summary's counts have to add up to the rows the table printed, or the
    # prose is describing a different experiment from the one the reader just saw.
    assert sum(verdict_counts.values()) == row_count, verdict_counts
    for label, listed in (
        ("收益", None),
        ("回退", None),
        (INDETERMINATE, indeterminate),
        (UNORDERED, unbalanced_rows),
        (INSUFFICIENT, insufficient_rows),
    ):
        if listed is None:
            continue
        assert verdict_counts.get(label, 0) == len(listed), (label, verdict_counts)
    assert (
        verdict_counts.get("收益", 0) + verdict_counts.get("回退", 0) == len(directional)
    ), (verdict_counts, len(directional))
    assert len(directional) + len(indeterminate) + len(unbalanced_rows) + len(
        insufficient_rows
    ) == row_count, "every row must land in exactly one of the four lists"
    line.append(
        "\n进程内离散与进程间离散分开列，因为两者可以差一个量级，"
        "而**能用来判断「这个差是不是真的」的只有进程间那一列**：进程内区间再紧，"
        "也只说明那一个进程跟自己对得上。\n\n"
        "**但判方向用的不是进程间那一列，是 `逐轮增益`。**"
        "这两件事不能混：`进程间` 是**不配对**比较的分辨率——拿今天测的 base 去比"
        "昨天测的 opt，差值小于它就分不出来。而这里每一轮的 base 与 opt 是"
        "**背靠背两个独立进程、同一台机器、同一份权重**，"
        "机器状态的那部分噪声在相除时抵消掉了。\n\n"
        "**但抵消的前提是顺序均衡，而均衡只对保留下来的轮次成立。**"
        "排在第二个进程的那棵树继承热起来的 page cache，"
        "所以 `顺序均衡（保留轮次）` 为「否」的行里，那份便宜会被记成优化的收益——"
        f"这类行本表一律判 `{UNORDERED}`，不给方向。"
        "均衡为「是」的行才谈得上「`有效轮次` 那一列的轮数全部落在 1.00 同一侧"
        "是有信息的」，"
        "即使那几个比值本身的极差小于 `进程间` 那一列。"
        "反过来说：这个判据给的是**方向**，不是**大小**。"
        "一行的 `逐轮增益` 挤在 1.00 上方一点点（本表 Qwen3-0.6B 的两行 decode 就是），"
        "意思是「这几次配对都偏快」，"
        "**不是**「已经确定快了那么多」——要定价还得看 `进程间`。\n\n"
        "**这张表不提供任何可以套到别的量上的统一门限。**下面两张扫描表比的都是 "
        f"`{sweep_quantity}`，所以它们引用的离散度取自上表这一行的 opt 侧进程间值"
        + (
            f"（两个扫描模型的中位 {statistics.median(sweep_opt_between):.1f}%），"
            "是同一个量的同类测量，不是从别的量借来的常数。"
            "扫描表只覆盖这两个模型，所以这个中位数也只在这两个模型里取——"
            "把另外三个模型的离散度混进来，等于又造了一个统一门限。"
            if sweep_opt_between
            else "——但本轮没有采到这一行，所以那两张表**没有可引用的离散度**，"
            "其判读只能停在「方向未定」。"
        )
        + "\n"
    )
    return statistics.median(sweep_opt_between) if sweep_opt_between else None


def incidental_noise_section(
    line: list[str],
    sweep_models: list[str],
    opt_files: dict[str, Any],
) -> float | None:
    """The two same-configuration runs this round happens to contain.

    Not an experiment: ``ab/`` step 1 and ``sweep-opt/*-t-default`` are the same
    tree in two separate processes only by accident of what else was being
    measured. Two points have no distribution, and the max of two points is not a
    floor — it is one observation of a gap that happened to be large. Printed
    because it is better than asserting a number, and labelled so no reader takes
    it for the measured thing it is not.
    """
    probe, length = FALLBACK_PROBE
    rows = []
    for model in sweep_models:
        a = load(PERF / "ab" / f"{model}-opt.json")
        b = opt_files.get(f"{model}-opt-t-default")
        x = stat((a or {}).get("micro", {}).get(probe, {}).get(length), "median")
        y = stat((b or {}).get("micro", {}).get(probe, {}).get(length), "median")
        if x and y:
            rows.append((model, a, b, x, y))
    if not rows:
        return None
    line.append("\n### 跨进程离散度：**未做实验，只有两次偶然的同配置运行**\n")
    line.append(
        "`tools/perf/run_noise_floor.sh`（AB／BA 顺序均衡的多轮独立进程配对实验）"
        "**本轮没有跑**，所以这里没有每个负载自己的跨进程离散度。"
        "下面这两行是本轮恰好存在的、同一棵 opt 树在两个独立进程里的同一个探针"
        "（`ab/` 第 1 步 与 `sweep-opt/*-t-default`），"
        "它不是实验设计，只是两次测量的巧合：\n"
    )
    line.append(
        "| 模型 | `ab/` 中位 | `ab/` 区间 | `sweep-opt/` 中位 | `sweep-opt/` 区间 | 中位数相差 | 两个区间是否重叠 |"
    )
    line.append("| --- | --- | --- | --- | --- | --- | --- |")
    worst = 0.0
    disjoint = 0
    for model, a, b, x, y in rows:
        gap = 100 * abs(x / y - 1)
        worst = max(worst, gap)
        left = bounds(a["micro"][probe][length])
        right = bounds(b["micro"][probe][length])
        # Overlap is reported per row instead of asserted for the section: on one
        # model the two ranges do overlap, and claiming otherwise would be the
        # kind of tidy-but-false sentence this whole script exists to prevent.
        overlaps = None if not (left and right) else (left[0] <= right[1] and right[0] <= left[1])
        if overlaps is False:
            disjoint += 1
        line.append(
            f"| {SHORT[model]} | {x * 1e3:.1f} ms | {range_only(a['micro'][probe][length])} | "
            f"{y * 1e3:.1f} ms | {range_only(b['micro'][probe][length])} | {gap:.1f}% | "
            f"{'—' if overlaps is None else ('是' if overlaps else '**否**')} |"
        )
    gaps = [100 * abs(x / y - 1) for _, _, _, x, y in rows]
    # The spread across models is the argument, so it has to be computed. This
    # sentence carried a derived per-model list *and* a typed-in "一个 0.4% 一个
    # 3.0%" summary of it one line apart, which is the drift this script exists to
    # prevent: the two copies agree today and nothing checks that they agree
    # tomorrow.
    gap_ratio = max(gaps) / min(gaps) if gaps and min(gaps) else None
    line.append(
        f"\n**{disjoint}/{len(rows)} 个模型的进程内区间互不重叠，"
        f"另外 {len(rows) - disjoint} 个重叠。**"
        + (
            "报告里原来那句「两个模型的区间都不重叠」是错的。"
            if disjoint < len(rows)
            else ""
        )
        + "两行的中位数相差 "
        + "、".join(
            f"{SHORT[row[0]]} {gap:.1f}%" for row, gap in zip(rows, gaps)
        )
        + (
            f"——最大的是最小的 **{gap_ratio:.1f} 倍**，"
            if gap_ratio
            else "——"
        )
        + "**这正好说明它撑不起一个统一的底线**："
        "同一个探针在不同模型上的跨进程差就差了这么多，"
        "把它们的最大值当成所有负载的分辨率是没有依据的。\n"
    )
    line.append(
        f"所以下面两张表的判读规则改成：**不再引用任何统一的噪声百分比**，"
        "凡是差值落在这个量级以内的配置，结论一律写 "
        f"`{INDETERMINATE}`；要给出方向性结论，先跑 "
        "`bash tools/perf/run_noise_floor.sh <新的输出目录> 4`，"
        "让每个负载用自己的多轮数据说话。"
        f"这里的 {worst:.1f}% 只是**已知存在过这么大的跨进程差**，"
        "不是「小于它就是噪声、大于它就是收益」的门限。\n"
    )
    return worst


def thread_summary(
    line: list[str],
    model: str,
    opt_files: dict[str, Any],
    thread_tags: list[str],
    floor: float | None,
) -> None:
    """Say in numbers whether requesting threads changed anything.

    Reading four rows of a table and concluding "flat" is a judgement the reader
    cannot check. The extreme-to-extreme spread across the four configurations is
    the checkable form of the same statement, and putting the observed
    cross-process dispersion next to it turns "looks flat" into "flat relative to
    a gap this machine has actually produced".

    ``floor`` is that observed dispersion, *not* a threshold. The wording below is
    careful about the difference: it says the spread is the same order as a gap
    known to occur, therefore the direction cannot be resolved — it does not say
    "under 3% counts as noise", which would make a two-point coincidence into a
    decision rule for every configuration in the table.
    """
    prefill: list[float] = []
    decode: list[float] = []
    for tag in thread_tags:
        payload = opt_files.get(f"{model}-opt-{tag}") or {}
        value = stat(payload.get("micro", {}).get("engine_prefill", {}).get("512"), "median")
        if value:
            prefill.append(value)
        value = stat(payload.get("micro", {}).get("raw_decode"), "ms_per_token")
        if value:
            decode.append(value)
    if len(prefill) < 2:
        return

    def spread_percent(values: list[float]) -> float:
        return 100 * (max(values) / min(values) - 1)

    prefill_spread = spread_percent(prefill)
    decode_spread = spread_percent(decode) if decode else None
    text = f"\n四个线程配置之间，opt 侧 512-token 分块 prefill 的极差是 {prefill_spread:.1f}%"
    if decode_spread is not None:
        text += f"，decode 的极差是 {decode_spread:.1f}%"
    if floor:
        widest = max(prefill_spread, decode_spread or 0.0)
        multiple = widest / floor
        text += f"，与上面观测到的 {floor:.1f}% 跨进程离散度是同一量级（{multiple:.1f} 倍）"
        # A factor of two is the cut-off, stated here rather than hidden: below it
        # the spread and the observed gap are the same size, so "t-default was
        # fastest" and "t1 was fastest" are the same observation seen twice.
        if multiple < 2.0:
            text += (
                f"。**{INDETERMINATE}**：这一轮既没有测到"
                "「请求更多 BLAS 线程」带来的收益，也没有测到回退——请求值确实在 "
                "`import numpy` 之前进了环境（上表第三列），但耗时对它没有反应。"
                "至于是 Accelerate 忽略了这些变量、还是这些矩阵形状本来就受内存带宽限制，"
                "**本轮没有测，不做判断。**"
            )
        else:
            text += (
                "，比它大一倍以上，所以这一轮**看得出配置之间有差**；"
                "但这是同一进程内的极差，跨进程重复之后还在不在没有测，"
                "要改默认值得先跑 `run_noise_floor.sh` 拿到多轮配对数据。"
            )
    line.append(text + "\n")


def chunk_summary(
    line: list[str],
    model: str,
    opt_files: dict[str, Any],
    default_node: Any,
    floor: float | None,
) -> None:
    """Classify each non-default chunk size by whether the default is separable from it.

    The test is an overlap test on the observed samples, not a percentage cut-off:
    if the default configuration's *median* falls inside the candidate's measured
    [min, max], then the candidate produced that number in some run and the two
    cannot be told apart with one measurement each. Using the ranges keeps the
    verdict tied to data the harness already recorded, and avoids inventing a
    "X% counts as real" constant that would itself need justifying.
    """
    reference = median_of(default_node)
    span = bounds(default_node)
    if not reference or not floor:
        return
    better: list[str] = []
    worse: list[str] = []
    inside: list[str] = []
    for tag in ("c64", "c256", "c512"):
        payload = opt_files.get(f"{model}-opt-{tag}") or {}
        node = payload.get("micro", {}).get("engine_prefill", {}).get("512")
        median = median_of(node)
        if median is None:
            continue
        delta = 100 * (median / reference - 1)
        label = f"`{tag}`（{delta:+.1f}%，区间 {range_only(node)}）"
        candidate = bounds(node)
        # bounds() is in milliseconds while the medians are in seconds — the
        # comparison has to be done in one unit or every candidate looks covered.
        reference_ms = reference * 1e3
        # span is the *default's* range; the reference value compared against the
        # candidate's range is the same question from the other side, and both are
        # reported because either one alone can look decisive when it is not.
        covered = candidate is not None and candidate[0] <= reference_ms <= candidate[1]
        inside_default = span is not None and span[0] <= median * 1e3 <= span[1]
        if covered or inside_default:
            inside.append(label)
        elif delta < 0:
            better.append(label)
        else:
            worse.append(label)

    parts = []
    if better:
        parts.append(f"明确更快：{'、'.join(better)}")
    if worse:
        parts.append(f"明确更慢：{'、'.join(worse)}")
    if inside:
        parts.append(
            f"判不出来（默认值的中位数落在该配置的观测区间内，或反之）：{'、'.join(inside)}"
        )
    if not parts:
        return
    verdict = (
        "没有任何一个非默认 chunk 大小落在「明确更快」那一档，"
        "所以 `prefill_chunk_size` 的默认值 **128 不动**——改它需要的是稳定收益，"
        "而这里连方向都没测出来。"
        if not better
        else "**有配置明确更快**，需要重复测量确认适用范围后才能考虑改默认值。"
    )
    line.append("\n" + "；".join(parts) + "。" + verdict + "\n")


# --------------------------------------------------------------------------
# opt 4a: image feature cache
# --------------------------------------------------------------------------
def section_vision(line: list[str]) -> None:
    line.append("\n## 优化 4a：图像特征缓存（SmolVLM-256M-Instruct）\n")
    base = load(PERF / "vision" / "base.json")
    opt = load(PERF / "vision" / "opt.json")
    if not base or not opt:
        missing(line, "vision/{base,opt}.json")
        return

    line.append(
        f"`--warmup {base['args'].get('warmup')} --repeat {base['args'].get('repeat')} "
        f"--max-new-tokens {base['args'].get('max_new_tokens')}`，greedy。"
        f"base `cache_supported={base.get('cache_supported')}`（那棵树没有这个参数，"
        f"所以它的四条路径都退化成"
        f"「关缓存」），opt `cache_supported={opt.get('cache_supported')}`。\n"
    )
    line.append(
        "每条路径的 step 0 是冷的那次（必然 miss），step 1 是第二次请求。"
        "第二次请求命中还是不命中，正是这张表要区分的。\n"
    )

    def step_wall(payload: Any, path: str, index: int) -> float | None:
        node = (payload or {}).get("paths", {}).get(path, {}).get("steps", {}).get(str(index))
        return stat((node or {}).get("wall_seconds"), "median")

    def step_hit(payload: Any, path: str, index: int) -> Any:
        node = (payload or {}).get("paths", {}).get(path, {}).get("steps", {}).get(str(index))
        return (node or {}).get("vision_cache_hit")

    line.append("### 四条对照，各自回答一个不同的问题\n")
    line.append("| 对照 | 回答的问题 | before | after | 加速 | before 命中 | after 命中 |")
    line.append("| --- | --- | --- | --- | --- | --- | --- |")
    pairs = [
        ("base off-no-cache #1 → opt off-no-cache #1",
         "优化 1+2 在图文路径上单独值多少（**缓存不参与**）",
         step_wall(base, "off-no-cache", 1), step_wall(opt, "off-no-cache", 1),
         step_hit(base, "off-no-cache", 1), step_hit(opt, "off-no-cache", 1)),
        ("opt off-no-cache #1 → opt on-same-question #1",
         "**缓存本身**值多少，代码不变、同图同问",
         step_wall(opt, "off-no-cache", 1), step_wall(opt, "on-same-question", 1),
         step_hit(opt, "off-no-cache", 1), step_hit(opt, "on-same-question", 1)),
        ("base off-no-cache #1 → opt on-same-question #1",
         "升级并且问第二次，用户实际感受到的",
         step_wall(base, "off-no-cache", 1), step_wall(opt, "on-same-question", 1),
         step_hit(base, "off-no-cache", 1), step_hit(opt, "on-same-question", 1)),
        ("opt off-no-cache #2 → opt on-second-question #1",
         "同图**换一个问题**也命中——features 与问题无关",
         step_wall(opt, "off-no-cache", 2), step_wall(opt, "on-second-question", 1),
         step_hit(opt, "off-no-cache", 2), step_hit(opt, "on-second-question", 1)),
        ("opt off-no-cache #1 → opt on-other-image #1",
         "换一张图**必须**重算；这里 after 更慢是因为那张图更大（9 帧 vs 5 帧）",
         step_wall(opt, "off-no-cache", 1), step_wall(opt, "on-other-image", 1),
         step_hit(opt, "off-no-cache", 1), step_hit(opt, "on-other-image", 1)),
    ]
    for label, question, before, after, b_hit, o_hit in pairs:
        line.append(
            f"| {label} | {question} | {ms(before)} | {ms(after)} | {ratio(before, after)} | "
            f"{b_hit} | {o_hit} |"
        )

    line.append("\n### 两张图各自的规模\n")
    line.append("| 图 | 帧数 | image token | 像素字节 | features 字节 | encoder 输出字节 | 冷视觉编码 | 键哈希 |")
    line.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, image in (opt.get("images") or {}).items():
        line.append(
            f"| {key}（{Path(image['path']).name}） | {image.get('num_frames')} | "
            f"{image.get('num_image_tokens')} | {image.get('pixel_bytes', 0) / 2**20:.1f} MiB | "
            f"**{image.get('features_bytes', 0) / 2**20:.2f} MiB** | "
            f"{image.get('encoder_output_bytes', 0) / 2**20:.1f} MiB | "
            f"{image.get('vision_encode_seconds', 0) * 1e3:.0f} ms | "
            f"{ms(stat(image.get('key_seconds'), 'median'))} |"
        )
    line.append(
        "\n只缓存 `features` 而不是整个 `VisionOutput`：5 帧那张图的 encoder 输出是 "
        "features 的 20 倍大，缓存前者要花 20 倍内存去省同一次前向。"
        "键哈希那一列就是命中时 `vision_encode_seconds` 记的东西。\n"
    )

    line.append("### 命中时各计时字段的实际值\n")
    line.append(
        "`vision_encode_seconds` 在命中时是**本次请求自己**的像素哈希时间，"
        "不是当初那次前向的时长。所以视觉编码那一行的倍数要读作"
        "「一次完整视觉前向 ÷ 一次像素哈希」，**不是**「缓存把前向压到了 1/140」；"
        "请求墙钟那一行才是用户拿到的加速。\n"
    )
    headline = opt.get("headline", {}).get("same_question_again", {})
    without = headline.get("without_cache", {})
    with_cache = headline.get("with_cache", {})
    metric_labels = (
        ("wall_seconds", "请求墙钟"),
        ("first_token_seconds", "首 token"),
        ("vision_encode_seconds", "视觉编码字段"),
        ("prefill_seconds", "文本 prefill"),
        ("decode_seconds", "decode"),
    )
    # Computed once and used by both the table and the sentence under it. The
    # sentence used to type the same three ratios in as "约 14× / 约 142× / 约 7×",
    # and §5.4 typed them in again as "约 7× / 约 141×" — so one number had two
    # hardcoded copies that had already drifted apart by 1×, in a report whose
    # opening claim is that it contains no transcribed figures.
    metric_ratios = {
        key: ratio(without.get(key), with_cache.get(key)) for key, _ in metric_labels
    }
    line.append("| 指标 | 关缓存 | 同问再问（命中） | 倍数 |")
    line.append("| --- | --- | --- | --- |")
    for key, label in metric_labels:
        b_value, o_value = without.get(key), with_cache.get(key)
        line.append(f"| {label} | {ms(b_value)} | {ms(o_value)} | {metric_ratios[key]} |")

    def metric_x(key: str) -> str:
        """The table's own ratio cell, as prose, without re-deriving it."""
        b_value, o_value = without.get(key), with_cache.get(key)
        return "**（数据缺失）**" if not (b_value and o_value) else f"约 {b_value / o_value:.0f}×"

    line.append(
        f"\n命中侧 `vision_cache_hit={with_cache.get('vision_cache_hit')}`，"
        f"关缓存侧 `{without.get('vision_cache_hit')}`。"
        "`prefill_seconds` 与 `decode_seconds` 两边几乎相同——**缓存只动视觉那一段**，"
        f"文本塔该算的还是要算，所以首 token 的倍数（{metric_x('first_token_seconds')}）"
        f"远小于视觉字段的倍数（{metric_x('vision_encode_seconds')}），"
        f"而请求墙钟的倍数（{metric_x('wall_seconds')}）又小于首 token 的倍数："
        "decode 那一段没被优化。\n"
    )

    line.append("### 正确性与缓存流量\n")
    for label, payload in (("base", base), ("opt", opt)):
        agree = payload.get("agreement", {})
        line.append(
            f"- {label}：以 `{agree.get('reference_path')}` 为基准比较 "
            f"{agree.get('steps_compared')} 个生成步，不一致 "
            f"{len(agree.get('mismatched_steps', []))} 个，"
            f"`all_steps_identical={agree.get('all_steps_identical')}`。\n"
        )
    line.append(
        "- opt 四条路径的实测缓存流量与**从 step 列表推出来的期望**逐条相符"
        "（`matches_expectation`）；关缓存那条 `max_bytes=0`、hits/misses 全 0，"
        "换图那条只有 miss 没有 hit：\n"
    )
    for name, path in opt["paths"].items():
        actual = path.get("cache_traffic", {})
        line.append(
            f"  - `{name}`：hits={actual.get('hits')} misses={actual.get('misses')} "
            f"entries={actual.get('entries')} bytes={actual.get('bytes')} "
            f"evictions={actual.get('evictions')} → "
            f"`matches_expectation={actual.get('matches_expectation')}`\n"
        )
    off_rss = (opt["paths"].get("off-no-cache") or {}).get("rss_final_bytes")
    on_rss = (opt["paths"].get("on-same-question") or {}).get("rss_final_bytes")
    if off_rss and on_rss:
        line.append(
            f"\n峰值 RSS：关缓存 {off_rss / 2**30:.2f} GiB，开缓存 {on_rss / 2**30:.2f} GiB，"
            f"差额 {(on_rss - off_rss) / 2**20:.1f} MiB —— 缓存条目本身只有 0.70 MiB，"
            "差额主要是命中路径少了一次前向、因而峰值出现的位置不同，"
            "不能读成"
            "「缓存占了这么多内存」。\n"
        )


# --------------------------------------------------------------------------
# opt 4b: document index
# --------------------------------------------------------------------------
def section_documents(line: list[str]) -> None:
    line.append("\n## 优化 4b：预编码文档索引（Qwen3-Embedding-0.6B）\n")
    base = load(PERF / "documents" / "base.json")
    opt = load(PERF / "documents" / "opt.json")
    if not base or not opt:
        missing(line, "documents/{base,opt}.json")
        return

    corpus = opt.get("corpus", {})
    args = opt.get("args", {})
    line.append(
        f"语料 {corpus.get('documents')} 篇（合计 {corpus.get('tokens_total')} token），"
        f"查询 {corpus.get('queries')} 条，`--warmup {args.get('warmup')} "
        f"--repeat {args.get('repeat')}`，`max_length={args.get('max_length')}`、"
        f"`prefill_chunk_size={args.get('prefill_chunk_size')}`。"
        f"base `index_supported={base.get('index_supported')}`（那棵树没有 "
        f"`index_documents`），opt `index_supported={opt.get('index_supported')}`。\n"
    )

    line.append("### 每轮实际编码了多少次\n")
    line.append(
        "计数来自一个**贴在实例上的探针**，包住 `encode_document` / `encode_query`。"
        "`EmbeddingEngine.search`、`index_documents`、`DocumentIndex.search` 三处都经实例属性调用，"
        "所以计到的是真实调用次数，不是从公式推的。\n"
    )
    line.append("| 路径 | document 编码/轮 | query 编码/轮 | 每轮墙钟（中位数与全距） |")
    line.append("| --- | --- | --- | --- |")
    rows = [
        ("base `engine.search`", base.get("search")),
        ("opt `engine.search`（语义未改）", opt.get("search")),
        ("opt `index.search`（预编码）", opt.get("index")),
    ]
    for label, node in rows:
        if not node:
            continue
        counts = node.get("counts_per_round", {})
        line.append(
            f"| {label} | **{counts.get('document_encodes', '?')}** | "
            f"{counts.get('query_encodes', '?')} | {spread(node.get('round_wall_seconds'))} |"
        )

    line.append("\n### 建索引的一次性成本\n")
    build = (opt.get("index") or {}).get("build") or {}
    amortised = (opt.get("index") or {}).get("with_build_amortised") or {}
    line.append("| 项目 | 值 |")
    line.append("| --- | --- |")
    line.append(f"| 建索引墙钟（{corpus.get('documents')} 篇编码一遍） | {ms(build.get('wall_seconds'))} |")
    line.append(f"| 其中 `index.build_seconds`（引擎自己记的） | {ms(build.get('build_seconds'))} |")
    line.append(f"| 每次建索引的 document 编码次数 | {build.get('document_encodes_per_build', '?')} |")
    vectors_bytes = build.get("vectors_bytes")
    if vectors_bytes:
        line.append(f"| 索引向量占用 | {vectors_bytes / 2**20:.2f} MiB |")
    stats = build.get("stats")
    if stats:
        line.append(f"| `index.stats()` | `{json.dumps(stats, ensure_ascii=False)}` |")
    line.append(f"| 把重建摊进每轮之后的墙钟 | {spread(amortised.get('round_wall_seconds'))} |")

    line.append("\n### 四种读法，不要混用\n")
    headline = opt.get("headline", {})
    round_seconds = headline.get("round_seconds", {})
    speedup = headline.get("speedup", {})
    b_search = stat(base.get("search", {}).get("round_wall_seconds"), "median")
    o_search = stat(opt.get("search", {}).get("round_wall_seconds"), "median")
    o_index = stat(opt.get("index", {}).get("round_wall_seconds"), "median")
    line.append("| 读法 | 回答的问题 | 数值 |")
    line.append("| --- | --- | --- |")
    line.append(
        f"| base search → opt search | 优化 1+2 在一个**语义未改**的 API 上白拿的部分 | "
        f"{ms(b_search)} → {ms(o_search)} = **{ratio(b_search, o_search)}** |"
    )
    line.append(
        f"| opt search → opt index | **索引本身**值多少，代码不变 | "
        f"{ms(o_search)} → {ms(o_index)} = **{ratio(o_search, o_index)}**"
        f"（harness 记的 `queries_only` = {speedup.get('queries_only', float('nan')):.2f}×） |"
    )
    line.append(
        f"| base search → opt index | 升级并改用索引，用户实际感受到的 | "
        f"{ms(b_search)} → {ms(o_index)} = **{ratio(b_search, o_index)}** |"
    )
    line.append(
        f"| 每轮都重建索引 | **最保守**的读法：假设调用方一次都不复用索引 | "
        f"{round_seconds.get('index_including_a_rebuild_every_round', float('nan')) * 1e3:.1f} ms/轮 = "
        f"**{speedup.get('including_a_rebuild_every_round', float('nan')):.2f}×** |"
    )
    line.append(
        f"\n每 query 摊到的 document 编码次数：无索引 "
        f"{headline.get('documents_per_query_without_index')}，有索引 "
        f"{headline.get('documents_per_query_with_index')}。\n"
    )

    line.append("### 正确性\n")
    agree = opt.get("agreement", {})
    line.append(
        f"- 比较 {agree.get('queries_compared', '?')} 条查询 × "
        f"{len(agree.get('fields', []))} 个字段（{', '.join(agree.get('fields', []))}），"
        f"判据是 `{agree.get('comparison')}`，不一致 {len(agree.get('mismatches', []))} 个，"
        f"`all_identical={agree.get('all_identical')}`。\n"
    )
    line.append(
        "- 判据是**逐位相同**而不是 `allclose`：两条路径调的是同一个 "
        "`rank_documents` / `encode_document`，所以本该一模一样；"
        "用容差比较会把一个真实的回归放过去。\n"
    )
    line.append(
        f"- base 侧 `weights_fingerprint` = "
        f"`{json.dumps(base.get('engine', {}).get('weights_fingerprint'), ensure_ascii=False)}`"
        f"（那棵树没有 `Weights.fingerprint`，记 `null` 以保持两份输出同形状，"
        f"而不是让 base 直接崩掉），opt 侧是 "
        f"`{len(opt.get('engine', {}).get('weights_fingerprint') or [])}` 元组的前 6 项 + 张量表。\n"
    )
    b_rss = stat(base.get("search", {}).get("peak_rss_bytes"), "median")
    o_rss = stat(opt.get("search", {}).get("peak_rss_bytes"), "median")
    i_rss = stat(opt.get("index", {}).get("peak_rss_bytes"), "median")
    line.append(
        f"- 峰值 RSS：base search {b_rss / 2**30:.2f} GiB，opt search {o_rss / 2**30:.2f} GiB，"
        f"opt index {i_rss / 2**30:.2f} GiB。索引那条**不比 search 高**——"
        "省掉的是重复编码的中间激活，换来的只是那个向量矩阵。\n"
    )


def headline_numbers() -> dict[str, Any]:
    """The summary ranges the report's prose quotes, computed from the same JSON.

    A narrative that hardcodes "1.11×–2.14×" goes stale the moment the harness is
    re-run, and a stale summary sitting above a fresh table is worse than no
    summary: it is the part people read. So every range the prose mentions is
    derived here, and anything that cannot be derived is returned as ``None`` and
    rendered as an explicit gap rather than a plausible-looking number.
    """
    out: dict[str, Any] = {}

    def rng(values: list[float]) -> tuple[float, float] | None:
        finite = [v for v in values if v is not None and math.isfinite(v)]
        return (min(finite), max(finite)) if finite else None

    engine: list[float] = []
    prefill512: list[float] = []
    prefill512_excluded: list[str] = []
    decode_short: list[float] = []
    decode_windows: list[int] = []
    decode_single: list[str] = []
    for model in MODELS:
        base = load(PERF / "ab" / f"{model}-base.json") or {}
        opt = load(PERF / "ab" / f"{model}-opt.json") or {}
        b_micro, o_micro = base.get("micro", {}), opt.get("micro", {})
        for length in PREFILL_LENGTHS:
            b = stat((b_micro.get("engine_prefill", {}) or {}).get(length), "median")
            o = stat((o_micro.get("engine_prefill", {}) or {}).get(length), "median")
            if b and o:
                engine.append(b / o)
        b = stat((b_micro.get("raw_prefill_full", {}) or {}).get("512"), "median")
        o = stat((o_micro.get("raw_prefill_full", {}) or {}).get("512"), "median")
        # Same guard §2.2's table applies, and for the same reason: this feeds the
        # headline band, so one side that thrashed becomes the *maximum* of a range
        # quoted in the summary. Without it the re-collection would report
        # 整块 prefill up to 16.5× on the strength of a base side that spent 23 s
        # being paged back in.
        _, _, disagreement = raw_prefill_self_check(b_micro, o_micro, "512")
        if b and o and disagreement is not None and disagreement <= SELF_CHECK_TOLERANCE:
            prefill512.append(b / o)
        elif b and o:
            prefill512_excluded.append(SHORT[model])
        # ``decode_median`` rather than the legacy ``ms_per_token`` scalar: §2.3's
        # table prints the multi-window median, so the summary ratio above it has to
        # be formed from the same quantity or the two silently disagree. The counts
        # are recorded too, because §9.1 used to hardcode "decode 那一列是单窗口观察值"
        # and that stopped being true the moment the re-collection enabled --repeat.
        b_node = b_micro.get("raw_decode") or {}
        o_node = o_micro.get("raw_decode") or {}
        b = decode_median(b_node) if isinstance(b_node, dict) else None
        o = decode_median(o_node) if isinstance(o_node, dict) else None
        note = single_window_note(model, "decode 短上下文", b_node, o_node)
        if note:
            decode_single.append(note)
        for node in (b_node, o_node):
            windows = node.get("ms_per_token_windows") if isinstance(node, dict) else None
            samples = windows.get("samples") if isinstance(windows, dict) else None
            decode_windows.append(len(samples) if samples else 1)
        if b and o:
            decode_short.append(b / o)
    out["engine_prefill"] = rng(engine)
    out["raw_prefill_512"] = rng(prefill512)
    out["raw_prefill_512_excluded"] = prefill512_excluded
    out["raw_prefill_512_models"] = len(prefill512)
    out["raw_decode_short"] = rng(decode_short)
    out["decode_window_counts"] = rng([float(w) for w in decode_windows])
    out["decode_single_window_rows"] = decode_single

    # Workloads come from the re-run, which is the one that measured all five.
    first_token: list[float] = []
    total: list[float] = []
    source = PERF / "ab-longctx"
    if not source.exists():
        source = PERF / "ab"
    out["workload_source"] = source.name
    for model in MODELS:
        base = load(source / f"{model}-base.json") or {}
        opt = load(source / f"{model}-opt.json") or {}
        b_rows = base.get("workloads", {}) or {}
        for name, workload in (opt.get("workloads", {}) or {}).items():
            if "skipped" in workload:
                continue
            b = b_rows.get(name)
            if not isinstance(b, dict) or "skipped" in b:
                continue
            for key, bucket in (
                ("first_token_seconds", first_token),
                ("total_seconds", total),
            ):
                bv, ov = stat(b.get(key), "median"), stat(workload.get(key), "median")
                if bv and ov:
                    bucket.append(bv / ov)
    out["workload_first_token"] = rng(first_token)
    out["workload_total"] = rng(total)

    turn_last: list[float] = []
    conversation: list[float] = []
    payloads = ab_payloads()
    for model, (_, opt) in payloads.items():
        chat = (opt.get("chat") or {})
        cold = chat.get("cold", {}).get("turns", {})
        warm = chat.get("warm", {}).get("turns", {})
        last = max((int(t) for t in warm), default=None)
        if last is not None:
            c = stat((cold.get(str(last)) or {}).get("first_token_seconds"), "median")
            w = stat((warm.get(str(last)) or {}).get("first_token_seconds"), "median")
            if c and w:
                turn_last.append(c / w)
        value = chat.get("conversation_wall_seconds", {}).get("speedup")
        if isinstance(value, (int, float)) and math.isfinite(value):
            conversation.append(value)
    out["chat_turn4_first_token"] = rng(turn_last)
    out["chat_conversation"] = rng(conversation)

    # §9.4's 待重测 row quoted the published round's four cold-vs-cold ratios by
    # hand ("0.85× / 1.04× / 0.78× / 0.63×"). §4 prints that table from base/opt
    # ``chat.cold.turns``; recompute the same cells here so the row cannot claim a
    # regression pattern the table it points at does not contain — on the re-collected
    # root §4's own note reads "1 行读起来像回退（Qwen3-1.7B 第 4 轮）".
    chat_cold: list[tuple[str, int, float]] = []
    for model, (base, opt) in payloads.items():
        b_turns = (base.get("chat") or {}).get("cold", {}).get("turns", {})
        o_turns = (opt.get("chat") or {}).get("cold", {}).get("turns", {})
        for turn in sorted(b_turns, key=lambda t: int(t)):
            b_wall = stat(b_turns[turn].get("wall_seconds"), "median")
            o_wall = stat((o_turns.get(turn) or {}).get("wall_seconds"), "median")
            if b_wall and o_wall:
                chat_cold.append((SHORT[model], int(turn) + 1, b_wall / o_wall))
    out["chat_cold_rows"] = chat_cold
    out["chat_cold_slower"] = [row for row in chat_cold if row[2] < 1.0]

    # §9.4's other two rows were the same defect in a different section: "报出
    # 0.52×" is §3's long-context decode self-check, and "115.37 s / 3828.08 s /
    # 相差 33 倍" is the integration suite's wall time. Both are recomputed from
    # the sources those sections read.
    out["decode_long_suspects"] = decode_long_self_check(payloads)
    out["integration_durations"] = integration_durations("Qwen3-1.7B-slowq")

    # §0's discipline 2 quoted "0.59× 和 1.29×" as its example of an unusable
    # end-to-end number. That pair is in the *published* batch's two runs; generated
    # from the re-collected root, §3's table no longer contains it, so the report
    # opened with a number its own evidence section had replaced. Take the widest row
    # over every root on disk instead — both batches belong to this round and both
    # are preserved — and name the root so the reader can open the two JSONs.
    spread: tuple[str, str, float, float, float, str] | None = None
    for root in dict.fromkeys((REPRO_ROOT, PERF, DEFAULT_PERF)):
        for short, name, first, second, _, _ in contradiction_rows(root):
            low, high = sorted((first, second))
            if spread is None or high / low > spread[4]:
                spread = (short, name, low, high, high / low, repo_relative(root))
    out["end_to_end_spread"] = spread

    vision = (load(PERF / "vision" / "opt.json") or {}).get("headline", {})
    out["vision_same_question"] = stat(
        (vision.get("same_question_again") or {}).get("speedup"), "wall_seconds"
    )
    out["vision_other_question"] = stat(
        (vision.get("a_different_question") or {}).get("speedup"), "wall_seconds"
    )

    # §5.4's bullets. Each one restated a number §5.3's tables already derive, and
    # two of the copies had drifted: the other-image row reads 0.57× in the table
    # and 0.56× in the prose, and §5.3's own sentence quoted "视觉字段的倍数（约
    # 142×）" — which is the *different-question* headline's ratio, not the
    # same-question one the sentence is about (140.9×). Recomputed here from the
    # same nodes those tables read.
    vision_opt = load(PERF / "vision" / "opt.json") or {}
    same = vision.get("same_question_again") or {}
    vision_metrics: dict[str, Any] = {}
    for key in ("wall_seconds", "first_token_seconds", "vision_encode_seconds",
                "prefill_seconds", "decode_seconds"):
        before = (same.get("without_cache") or {}).get(key)
        after = (same.get("with_cache") or {}).get(key)
        vision_metrics[key] = before / after if before and after else None
    out["vision_metrics"] = vision_metrics
    hit_encode = (same.get("with_cache") or {}).get("vision_encode_seconds")
    out["vision_hit_encode_ms"] = hit_encode * 1e3 if hit_encode else None

    def vision_step_wall(path: str, index: int) -> float | None:
        node = (
            (vision_opt.get("paths") or {}).get(path, {}).get("steps", {}).get(str(index))
        )
        return stat((node or {}).get("wall_seconds"), "median")

    other_before = vision_step_wall("off-no-cache", 1)
    other_after = vision_step_wall("on-other-image", 1)
    out["vision_other_image_ratio"] = (
        other_before / other_after if other_before and other_after else None
    )
    out["vision_key_hash_ms"] = {
        key: {
            "num_frames": image.get("num_frames"),
            "ms": stat(image.get("key_seconds"), "median"),
        }
        for key, image in (vision_opt.get("images") or {}).items()
    }
    document_probe = load(PERF / "documents" / "opt.json") or {}
    documents = document_probe.get("headline", {})
    out["documents_queries_only"] = stat(documents.get("speedup"), "queries_only")
    out["documents_with_rebuild"] = stat(
        documents.get("speedup"), "including_a_rebuild_every_round"
    )

    # §6.4's "when is the index worth building" answer has to come out of the same
    # JSON as §6.3's table. It used to be a typed-in "实测 874.6 ms" plus a claim
    # that a single query does not pay for the build; the claim was false, because
    # one build (16 document encodes) and one query's worth of what ``search()``
    # repeats (16 document encodes) are the *same* work — so the first query
    # already breaks even and the second is pure gain. Deriving the count from the
    # payload is what makes that visible instead of plausible.
    index_node = (document_probe.get("index") or {})
    build_seconds = stat((index_node.get("build") or {}).get("build_seconds"), "median")
    queries_per_round = documents.get("queries_per_round")
    search_round = stat((document_probe.get("search") or {}).get("round_wall_seconds"), "median")
    index_round = stat(index_node.get("round_wall_seconds"), "median")
    per_query_saved = (
        (search_round - index_round) / queries_per_round
        if search_round and index_round and queries_per_round
        else None
    )
    out["documents_build_seconds"] = build_seconds
    out["documents_per_query_saved_seconds"] = per_query_saved
    out["documents_break_even_queries"] = (
        build_seconds / per_query_saved if build_seconds and per_query_saved else None
    )
    out["documents_corpus_size"] = (document_probe.get("corpus") or {}).get("documents")
    out["documents_queries_per_round"] = queries_per_round

    # Two independent estimates of optimisation 1's share, because the probe's own
    # precision (the spread across three lengths that must cost the same) is of
    # the same order as the quantity being estimated. Quoting one number would
    # hide that; quoting the union of both is the honest interval.
    shares: list[float] = []
    share_by_model: dict[str, float] = {}
    projection_costs: list[float] = []
    projection_resolutions: list[float] = []
    projection_voided: list[str] = []
    for model in MODELS:
        probe = load(PERF / "projection" / f"{model}.json") or {}
        per_length = probe.get("per_length") or {}
        per = per_length.get("512") or {}
        base_prefill = stat(per, "engine_prefill_base")
        skipped = per.get("projections_skipped")
        if not base_prefill or not skipped:
            continue
        raw_medians = [
            stat((per_length.get(length) or {}).get("projection_seconds"), "median")
            for length in ("128", "256", "512")
        ]
        # forward = encode + one real projection, so a non-positive median is an
        # invalid measurement (the model was paged out mid-pair), not a small one.
        medians = [m for m in raw_medians if m is not None and m > 0]
        direct = stat(per.get("projection_seconds"), "median")
        voided = [
            length
            for length, value in zip(("128", "256", "512"), raw_medians)
            if value is not None and value <= 0
        ]
        if voided or (direct is not None and direct <= 0):
            projection_voided.append(
                f"{SHORT[model]}（{'、'.join(voided) or '512'} tok 的成对差值中位数为负）"
            )
        # §2.4's second table drops the whole row when the 512-token paired median
        # is non-positive, because route A would then *subtract a negative* and hand
        # optimisation 1's credit to optimisation 2. This band has to obey the same
        # rule or it quotes a share no table in the report displays: computed from
        # the two surviving lengths it came out at 0.51% for SmolLM2-360M while the
        # table's own minimum was 0.55%, and a reader sent to check the lower bound
        # found nothing there. The *cost* band below keeps the model, because the
        # first table does print its 三者中位 — two quantities, two documented rules.
        row_voided = direct is not None and direct <= 0
        per_model: list[float] = []
        for cost in (None if row_voided else (direct if direct and direct > 0 else None),
                     None if row_voided else (statistics.median(medians) if medians else None)):
            if cost:
                shares.append(cost * skipped / base_prefill * 100)
                per_model.append(cost * skipped / base_prefill * 100)
        if per_model:
            share_by_model[SHORT[model]] = max(per_model)
        # §9.2 used to hardcode "0.86 ~ 11.04 ms" for the per-projection cost. Same
        # field §2.4's table prints, so read it from there.
        if medians:
            projection_costs.append(statistics.median(medians) * 1e3)
            # The probe's own resolution: 128/256/512 must all price the same
            # ``[1, hidden] × [hidden, vocab]`` matmul, so their spread is an error
            # bar, and it belongs next to the number it undermines.
            if min(medians) > 0:
                projection_resolutions.append(100 * (max(medians) / min(medians) - 1))
    out["opt1_share_percent"] = rng(shares)
    out["opt1_share_by_model"] = share_by_model
    out["projection_cost_ms"] = rng(projection_costs)
    out["projection_resolution_percent"] = rng(projection_resolutions)
    out["projection_models"] = len(projection_costs)
    out["projection_voided"] = projection_voided

    # Same floor the sweep section prints, from the same two runs, so the prose in
    # §0/§2/§7 cannot drift away from the table.
    opt_files = {path.stem: load(path) for path in sorted((PERF / "sweep-opt").glob("*.json"))}
    out["noise_floor_percent"] = noise_floor([], SWEEP_MODELS, {}, opt_files)

    # §7.1's "结论先说" quotes the sweep's spreads and chunk deltas. Those are the
    # same quantities thread_summary() and chunk_summary() print per model, so
    # they are recomputed here from the same JSON rather than typed in: a summary
    # section is where a stale number is least likely to be caught, because it
    # reads like prose and sits pages away from the table it summarises.
    def spread(values: list[float]) -> float | None:
        return 100 * (max(values) / min(values) - 1) if len(values) > 1 and min(values) else None

    threads: dict[str, dict[str, float | None]] = {}
    for model in SWEEP_MODELS:
        prefill, decode = [], []
        for tag in ("t-default", "t1", "t4", "t8"):
            payload = opt_files.get(f"{model}-opt-{tag}") or {}
            value = stat(payload.get("micro", {}).get("engine_prefill", {}).get("512"), "median")
            if value:
                prefill.append(value)
            value = stat(payload.get("micro", {}).get("raw_decode"), "ms_per_token")
            if value:
                decode.append(value)
        threads[SHORT[model]] = {"prefill": spread(prefill), "decode": spread(decode)}
    out["thread_spread_percent"] = threads

    chunks: dict[str, list[float]] = {}
    for tag in ("c64", "c256", "c512"):
        deltas = []
        for model in SWEEP_MODELS:
            default = opt_files.get(f"{model}-opt-t-default") or {}
            other = opt_files.get(f"{model}-opt-{tag}") or {}
            x = stat(default.get("micro", {}).get("engine_prefill", {}).get("512"), "median")
            y = stat(other.get("micro", {}).get("engine_prefill", {}).get("512"), "median")
            if x and y:
                deltas.append(100 * (y / x - 1))
        if deltas:
            chunks[tag] = deltas
    out["chunk_delta_percent"] = chunks

    floor_rows: dict[str, dict[str, float | None]] = {}
    for model in SWEEP_MODELS:
        a = load(PERF / "ab" / f"{model}-opt.json")
        b = opt_files.get(f"{model}-opt-t-default")
        x = stat((a or {}).get("micro", {}).get("engine_prefill", {}).get("512"), "median")
        y = stat((b or {}).get("micro", {}).get("engine_prefill", {}).get("512"), "median")
        if x and y:
            floor_rows[SHORT[model]] = {
                "ab": x * 1e3,
                "sweep": y * 1e3,
                "percent": abs(x - y) / min(x, y) * 100,
            }
    out["noise_floor_rows"] = floor_rows

    # §2.4 prices optimisation 2a from sigmoid-check.json. Those numbers were
    # typed into the prose once already and were wrong by the time the probe was
    # re-run (4.810 ms / 151.9% / 7.15× against an artifact saying 4.853 / 153.2 /
    # 7.20) — a ~1% cross-process difference, i.e. exactly the kind of gap that is
    # invisible when both copies are hardcoded and uncheckable by a reader.
    #
    # Reading the number out of the artifact fixed the transcription. It did not fix
    # what the number meant, and that is the larger error: the probe's "real gate
    # activation" was ``rms_norm(embed) -> input_layernorm -> gate_proj``, which
    # skips the whole attention block, the residual add, and swaps
    # ``post_attention_layernorm`` for ``input_layernorm``. The published 7.20×,
    # 38.8%-negative and 153%-of-the-layer therefore describe a tensor this model
    # never produces. Both artifacts are loaded so the report can print the
    # withdrawn figures beside the ones that replace them.
    withdrawn_path = withdrawn_sigmoid_path(PERF)
    sig = load(withdrawn_path) or {}
    rows = sig.get("sigmoid_rows") or {}
    real = rows.get("real gate activation") or {}
    one_sided = [
        (rows.get(label) or {}).get("speedup") for label in ("all positive", "all negative")
    ]
    one_sided = [v for v in one_sided if v]

    fixed = load(SIGMOID_CORRECTION) or {}
    fixed_rows = fixed.get("sigmoid_rows") or {}
    captured = fixed_rows.get("captured gate activation") or {}
    replaced = fixed.get("real_substitution") or {}
    shortcut = fixed.get("shortcut_vs_captured") or {}
    fixed_one_sided = [
        fixed_rows.get(label, {}).get("speedup") for label in ("all positive", "all negative")
    ]
    fixed_one_sided = [v for v in fixed_one_sided if v]

    runs = sigmoid_runs(SIGMOID_CORRECTION)

    def sub(key: str) -> Any:
        return lambda data: (data.get("real_substitution") or {}).get(key)

    def row_speedup(label: str) -> Any:
        return lambda data: (data.get("sigmoid_rows", {}).get(label) or {}).get("speedup")

    def one_signed(data: Any) -> list[Any]:
        return [
            (data.get("sigmoid_rows", {}).get(label) or {}).get("speedup")
            for label in ("all positive", "all negative")
        ]

    #: The quotable numbers, as bands over every preserved run. The latest run's own
    #: values stay in ``fixed`` above so the report can show what moved.
    captured_seconds = lambda which: (  # noqa: E731
        lambda data: (
            data.get("sigmoid_rows", {}).get("captured gate activation") or {}
        ).get(which)
    )
    spread = {
        "operator_speedup": band(runs, row_speedup("captured gate activation")),
        "operator_speedup_synthetic_normal": band(runs, row_speedup("standard normal")),
        "one_sided_speedup": band(runs, one_signed),
        "operator_old_seconds": band(runs, captured_seconds("old_seconds")),
        "operator_new_seconds": band(runs, captured_seconds("new_seconds")),
        "matmul_total_seconds": band(runs, lambda d: d.get("matmul_total_seconds")),
        "operator_over_matmul_sum": band(runs, lambda d: d.get("old_sigmoid_share_of_matmuls")),
        "mlp_old_seconds": band(runs, sub("mlp_old_seconds")),
        "mlp_new_seconds": band(runs, sub("mlp_new_seconds")),
        "mlp_saved_seconds": band(runs, sub("mlp_saved_seconds")),
        "mlp_saved_share": band(runs, sub("mlp_saved_share")),
        "layer_old_seconds": band(runs, sub("layer_old_seconds")),
        "layer_new_seconds": band(runs, sub("layer_new_seconds")),
        "layer_saved_seconds": band(runs, sub("layer_saved_seconds")),
        "layer_saved_share": band(runs, sub("layer_saved_share")),
        #: Relative dispersion *inside* one run, pooled over its four timed blocks.
        #: Comparing this with ``layer_saved_share`` above is the whole argument for
        #: quoting a band: the samples in one process are not repeat measurements.
        "within_run_rel_stdev": band(
            runs,
            lambda d: [
                (d.get("real_substitution") or {}).get(k, {}).get("stdev")
                / (d.get("real_substitution") or {}).get(k, {}).get("median")
                for k in ("layer_old", "layer_new", "mlp_old", "mlp_new")
                if (d.get("real_substitution") or {}).get(k, {}).get("median")
            ],
        ),
    }

    out["sigmoid"] = {
        # Both halves of the correction have to be traceable to a file. The
        # withdrawn one does not live under the data root when the report is
        # generated from a re-collection, and printing its real path is what stops
        # that from looking like a silently missing input.
        "withdrawn_path": repo_relative(withdrawn_path) if sig else None,
        "tokens": sig.get("tokens"),
        "old_ms": real.get("old_seconds"),
        "new_ms": sig.get("sigmoid_new_seconds"),
        "speedup": sig.get("sigmoid_speedup_real_gate"),
        "old_share": sig.get("old_sigmoid_share_of_matmuls"),
        "new_share": sig.get("new_sigmoid_share_of_matmuls"),
        "matmul_total_seconds": sig.get("matmul_total_seconds"),
        "one_sided_speedup": rng(one_sided),
        "gate": sig.get("gate_activation"),
        "blas": sig.get("blas"),
        # The corrected re-measurement, or None when it has not been run. None is
        # reported as a gap rather than filled in from the legacy artifact: the
        # legacy numbers are the thing being corrected, so borrowing them back
        # would make the correction a restatement of the error.
        "fixed_path": repo_relative(SIGMOID_CORRECTION) if fixed else None,
        "fixed": {
            "model": fixed.get("model"),
            "layer": fixed.get("layer"),
            "tokens": fixed.get("tokens"),
            "blas": fixed.get("blas"),
            "old_ms": captured.get("old_seconds"),
            "new_ms": captured.get("new_seconds"),
            "operator_speedup": captured.get("speedup"),
            "matmul_total_seconds": fixed.get("matmul_total_seconds"),
            "operator_over_matmul_sum": fixed.get("old_sigmoid_share_of_matmuls"),
            "gate": fixed.get("gate_activation"),
            "legacy_gate": sig.get("gate_activation"),
            "shortcut_rel_l2": shortcut.get("rel_l2"),
            "shortcut_stats": shortcut.get("shortcut_stats"),
            "mlp_old_ms": replaced.get("mlp_old_seconds"),
            "mlp_new_ms": replaced.get("mlp_new_seconds"),
            "mlp_saved_ms": replaced.get("mlp_saved_seconds"),
            "mlp_saved_share": replaced.get("mlp_saved_share"),
            "layer_old_ms": replaced.get("layer_old_seconds"),
            "layer_new_ms": replaced.get("layer_new_seconds"),
            "layer_saved_ms": replaced.get("layer_saved_seconds"),
            "layer_saved_share": replaced.get("layer_saved_share"),
            "one_sided_speedup": rng(fixed_one_sided),
            "notes": fixed.get("bit_equality_notes"),
            "failures": fixed.get("bit_equality_failures"),
            "repeat": (captured.get("old") or {}).get("repeat"),
            "identity": fixed.get("identity"),
            "runs": [
                {
                    "name": name,
                    "operator_speedup": (
                        data.get("sigmoid_rows", {}).get("captured gate activation") or {}
                    ).get("speedup"),
                    "matmul_total_seconds": data.get("matmul_total_seconds"),
                    "operator_over_matmul_sum": data.get("old_sigmoid_share_of_matmuls"),
                    "mlp_saved_share": sub("mlp_saved_share")(data),
                    "layer_saved_share": sub("layer_saved_share")(data),
                    "ops_code_sha256": (data.get("identity") or {}).get("ops_py_code_sha256"),
                }
                for name, data in runs
            ],
            "spread": spread,
            "run_count": len(runs),
        }
        if fixed
        else None,
    }

    # §4.3 quotes the session's second cache as a size range. The range is only
    # meaningful next to the max_context it was reserved for, because the
    # allocation is max_context-sized regardless of how much of it a conversation
    # reaches: the same code reserves 224.0 MiB at 1024 and 448 MiB at 2048, and
    # docs/architecture.md §6.1 quotes the 2048 figure. Both are right; neither
    # says so unless the context length travels with the number.
    cache_bytes: list[float] = []
    contexts: set[int] = set()
    for model in MODELS:
        opt = load(PERF / "ab" / f"{model}-opt.json") or {}
        allocated = (opt.get("chat", {}) or {}).get("warm", {}).get("cache_allocated_bytes")
        if allocated:
            cache_bytes.append(allocated / 2**20)
        context = (opt.get("args", {}) or {}).get("max_context")
        if context:
            contexts.add(int(context))
    out["session_cache"] = {
        "mib": rng(cache_bytes),
        "max_context": sorted(contexts),
    }
    return out


def span(pair: Any, unit: str = "×", digits: int = 2) -> str:
    """Render a (min, max) pair, or an explicit gap if it could not be derived."""
    if not pair:
        return "**（数据缺失）**"
    return f"{pair[0]:.{digits}f}{unit}–{pair[1]:.{digits}f}{unit}"


def solo(value: Any, unit: str = "×", digits: int = 2) -> str:
    return "**（数据缺失）**" if value is None else f"{value:.{digits}f}{unit}"


HEADING = re.compile(r"^#{2,6} ")


def demote(body: list[str]) -> list[str]:
    """Drop a generated section's own title and shift its headings down one level.

    The generator emits ``##``/``###`` for the standalone tables file. Every call
    site embeds one under a narrative heading that already names it — ``### 2.3
    实测`` under ``## 2. 优化 1 + 2``, ``### 4.4 实测`` under ``## 4. 优化 3``, and
    so on — so keeping the section's title reads the same section name twice and
    leaves an H3 sibling where an H4 child belongs.

    Items are split on embedded newlines before either step. Sections append
    their title as ``"\\n## 优化 3：…\\n"``, and an item that starts with a newline
    matches neither a heading test nor a level shift, which is how an un-demoted
    ``## 端到端工作负载`` ended up buried inside §3.
    """
    lines = [part for item in body for part in item.split("\n")]
    while lines and (not lines[0].strip() or HEADING.match(lines[0].lstrip())):
        lines.pop(0)
    return [f"#{line}" if HEADING.match(line) else line for line in lines]


BREAK_AFTER = "，。；：、！？）】》"
"""Characters a prose line may be wrapped *after*.

Deliberately narrow, and backed by ``PROTECTED`` below: markdown constructs that
must not be split are excluded explicitly rather than by luck. Emphasis spans
*can* contain these (``**跨进程…，小于…**``), and that is fine — CommonMark lets
emphasis span a soft line break within a paragraph.
"""

PROTECTED = re.compile(r"`[^`]*`|\[[^\]]*\]\([^)]*\)|\*\*[^*]+\*\*")
"""Inline code spans, links, and bold spans: avoid wrapping inside one.

Bold has to be in here even though CommonMark lets emphasis span a soft break,
because the closing ``**`` only counts as a delimiter when it is not preceded by
whitespace — a wrap that lands just before it leaves the ``**`` at the start of
the next line, where it closes nothing and renders literally.

Avoiding is not the same as forbidding: a bold span longer than the whole line
cannot be kept intact, so ``wrap_text`` falls back to breaking inside one and
``ATOMIC`` below is what is genuinely off limits.
"""

ATOMIC = re.compile(r"`[^`]*`|\[[^\]]*\]\([^)]*\)")
"""The subset of ``PROTECTED`` that must never be split at any cost.

A code span or a link broken across lines renders as literal backticks or a
literal ``[text](url)``, so there is no acceptable cut inside one. A bold span
broken at a punctuation mark still renders bold — CommonMark allows the soft
break — and that is strictly better than a cut that lands just *before* a
fullwidth comma, which puts the comma at the head of the next line where the
soft break renders as a visible space in front of it.
"""

LIST_PREFIX = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$", re.S)
QUOTE_PREFIX = re.compile(r"^((?:>\s?)+)(.*)$", re.S)


def display_width(text: str) -> int:
    """Columns ``text`` occupies, counting East Asian wide/fullwidth as two."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def splits_atomic(atomic: list[bool], index: int) -> bool:
    """True when cutting between ``index`` and ``index + 1`` splits an ``ATOMIC`` span."""
    if index < 0 or index + 1 >= len(atomic):
        return False
    return atomic[index] and atomic[index + 1]


def wrap_text(text: str, width: int) -> list[str]:
    """Greedy CJK-aware wrap. Returns the pieces without any prefix applied."""
    protected = [False] * len(text)
    for match in PROTECTED.finditer(text):
        for index in range(match.start(), match.end()):
            protected[index] = True
    atomic = [False] * len(text)
    for match in ATOMIC.finditer(text):
        for index in range(match.start(), match.end()):
            atomic[index] = True

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start
        used = 0
        while end < len(text):
            step = 2 if unicodedata.east_asian_width(text[end]) in ("W", "F") else 1
            if used + step > width and end > start:
                break
            used += step
            end += 1
        if end >= len(text):
            pieces.append(text[start:].rstrip())
            break

        cut = None
        for index in range(end - 1, start, -1):
            if protected[index] or protected[index - 1]:
                continue
            if text[index] in BREAK_AFTER or text[index] == " ":
                cut = index + 1
                break
        if cut is None:
            # Every punctuation mark in reach is inside a bold span. Breaking
            # after one still renders bold, and beats breaking before one: a
            # fullwidth comma at the head of a continuation line picks up the
            # soft break's space and reads as "分辨不出来 ，不能".
            for index in range(end - 1, start, -1):
                if text[index] not in BREAK_AFTER or splits_atomic(atomic, index):
                    continue
                if text[index + 1 : index + 3] == "**":
                    # The cut would put a closing delimiter at the head of the
                    # next line, where it is left-flanking and opens emphasis
                    # instead of closing it — the bug PROTECTED exists to avoid.
                    continue
                cut = index + 1
                break
        if cut is None:
            # No punctuation at all — fall back to any boundary that does not
            # split a code span or a link.
            for index in range(end, start, -1):
                if not protected[index] and not protected[index - 1]:
                    cut = index
                    break
            else:
                for index in range(end, start, -1):
                    if not splits_atomic(atomic, index - 1):
                        cut = index
                        break
        if cut is None:
            cut = end
        # Kinsoku push-out: a cut after ``）`` in ``（`x`）；`` leaves the ``；``
        # line-initial, and CJK punctuation may not start a line — CommonMark
        # renders the soft break as a space, so it reads "…（`x`） ；关缓存…".
        # Dragging the trailing run onto the current line overshoots ``width``
        # by a character or two, which costs nothing next to a broken sentence.
        while cut < len(text) and text[cut] in BREAK_AFTER:
            cut += 1
        pieces.append(text[start:cut].rstrip())
        start = cut
        while start < len(text) and text[start] == " ":
            start += 1
    return [piece for piece in pieces if piece]


def reflow(lines: list[str], width: int = 88) -> list[str]:
    """Wrap over-long prose so the doc matches the rest of docs/.

    Only paragraphs are touched, and the shape of each is preserved: a list item
    keeps its marker and its continuations are indented under the text, a
    blockquote keeps its ``>`` on every line. Fenced code, tables, headings and
    HTML comments keep their exact bytes, since reflowing those changes what
    renders rather than how it reads.

    Items are split on embedded newlines first. Several generated sections end a
    paragraph with ``"\\n"`` inside the string so that ``"\\n".join`` produces the
    blank line a following table or heading needs; wrapping would rstrip that
    away and the table would degrade into literal text, because GFM tables cannot
    interrupt a paragraph.
    """
    out: list[str] = []
    in_fence = False
    for item in [part for line in lines for part in line.split("\n")]:
        stripped = item.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(item)
            continue
        if (
            in_fence
            or not item.strip()
            # ``stripped``, not ``item``: a table nested inside a list item is
            # indented, and wrapping it mid-cell is exactly the "changes what
            # renders rather than how it reads" case this guard exists for. The
            # unindented test silently let those rows through.
            or stripped.startswith(("|", "#", "<!--"))
            or display_width(item) <= width
        ):
            out.append(item)
            continue

        prefix = ""
        body = item
        for pattern in (LIST_PREFIX, QUOTE_PREFIX):
            match = pattern.match(item)
            if match:
                prefix, body = match.group(1), match.group(2)
                break
        if not prefix:
            # No marker, but the author may still have indented the line to keep it
            # inside a list item's paragraph — which is how a correction note under
            # an attribution bullet has to be written. Wrapping those continuations
            # back to column 0 detaches them from the item they belong to, so the
            # leading whitespace becomes the continuation indent.
            leading = len(item) - len(item.lstrip(" "))
            if leading:
                prefix, body = item[:leading], item[leading:]
        indent = prefix if prefix.startswith(">") else " " * display_width(prefix)
        limit = width - display_width(prefix)
        if limit < 20:
            out.append(item)
            continue
        pieces = wrap_text(body, limit)
        if not pieces:
            out.append(item)
            continue
        out.append(prefix + pieces[0])
        out.extend(indent + piece for piece in pieces[1:])
    return squeeze_blank(out)


def squeeze_blank(lines: list[str]) -> list[str]:
    """Collapse runs of blank lines to one, except inside fenced code.

    ``demote`` splits every generated item on embedded newlines, so a section
    that appended ``"\\n## Title\\n"`` leaves its own trailing blank behind after
    the title is dropped and the narrative's blank joins it. Markdown renders
    either the same, but the doc is read as source too.
    """
    out: list[str] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and not line.strip() and out and not out[-1].strip():
            continue
        out.append(line)
    while out and not out[0].strip():
        out.pop(0)
    # The trailing blank is the mirror image of the leading one, and it survives
    # the loop above because there is no *second* blank for it to collapse with:
    # write_doc appends "\n" to the join, so one trailing "" already puts two
    # newlines at EOF. Harmless when rendered, but the doc is diffed as source.
    while out and not out[-1].strip():
        out.pop()
    return out


def write_doc(bodies: list[list[str]]) -> None:
    """Compose docs/推理加速-第1轮-2026-10-23.md from narrative + generated tables."""
    (
        identity,
        environment,
        micro,
        attribution,
        workloads,
        chat,
        vision,
        documents,
        sweep,
    ) = bodies
    n = headline_numbers()
    floor = n.get("noise_floor_percent")
    floor_text = "**（未测得）**" if floor is None else f"约 {floor:.1f}%"
    opt1_share = span(n.get("opt1_share_percent"), "%")
    # §9.2's "why unmeasurable" used to read "不大于本轮观测到的 约 1.5% 跨进程离散度".
    # That 1.5% is the *median of the two sweep models*, while the share band's upper
    # end comes from whichever model is largest — so the sentence compared one model's
    # share against two other models' dispersion, and came out false: the model that
    # sets the maximum has an opt-side dispersion *below* its own share. Compare per
    # model instead, print both sides, and say which way each one lands.
    share_by_model = n.get("opt1_share_by_model") or {}
    opt1_resolvable: list[str] = []
    opt1_unresolvable: list[str] = []
    for short, share in share_by_model.items():
        between = PAIRED_BETWEEN.get(short)
        if not between:
            continue
        (opt1_resolvable if share <= between[1] else opt1_unresolvable).append(
            f"{short} 占比 {share:.2f}% vs 自己那一行 base/opt 进程间 "
            f"{between[0]:.1f}%/{between[1]:.1f}%"
        )
    if opt1_resolvable or opt1_unresolvable:
        opt1_verdict_text = (
            "**按模型各比一遍**（占比取两种代价里较大的那个，离散度取该模型自己在 "
            "`引擎分块 prefill 512 tok` 一行的进程间值，都在 §7.3 表里）："
            + ("；".join(opt1_resolvable) + " —— 低于自身离散度，分不出来" if opt1_resolvable else "")
            + ("；**" if opt1_unresolvable else "")
            + ("；".join(opt1_unresolvable) + " —— 高于自身 opt 侧离散度**" if opt1_unresolvable else "")
            + "。所以「不大于跨进程离散度」这个说法**按模型读并不成立**，"
            "撤回它。结论不变，但换个不依赖离散度的理由：配对实验的 base／opt 两棵树"
            "**同时**差着优化 1 和优化 2，逐轮增益是两者之和，"
            "没有「只关优化 1」的配对探针，这个占比无论多大都拆不出来。"
        )
    else:
        opt1_verdict_text = (
            "**本轮没有可比对的逐模型进程间离散度**"
            "（配对实验未跑，或它没有覆盖 §2.4 那几张表的模型），"
            "所以这一格不下「分不出来」的结论；能说的只有配对实验的 base／opt 两棵树"
            "**同时**差着优化 1 和优化 2，没有「只关优化 1」的探针。"
        )
    # §9.2's per-projection cost and the probe's own resolution, both read out of
    # projection/*.json. When a re-collection did not run that probe — which is a
    # real possibility, since run_recollection.sh chains only the three drivers and
    # projection_probe.py is not one of them — the honest output is "本轮未采" plus
    # the reason, not a silently empty cell and not the previous round's numbers.
    cost_range = n.get("projection_cost_ms")
    resolution = n.get("projection_resolution_percent")
    if not cost_range:
        projection_cost_text = "**（本轮未采 `projection/*.json`）**"
        projection_resolution_text = ""
    else:
        projection_cost_text = (
            f"{cost_range[0]:.2f} ~ {cost_range[1]:.2f} ms"
            f"（{n.get('projection_models')} 个模型，随 `vocab × hidden` 走）"
        )
        projection_resolution_text = (
            ""
            if not resolution
            else f"；**探针自身分辨率 {resolution[0]:.0f}% ~ {resolution[1]:.0f}%**"
            "（128／256／512 三行测的是同一次投影，它们的极差就是这个探针的误差棒，"
            "它比被估的量还大）"
        )

    # §7.1/§7.2 quote the sweep. Every figure below is rendered from
    # headline_numbers(), which recomputes it from sweep-opt/*.json with the same
    # arithmetic thread_summary() and chunk_summary() use, so the summary cannot
    # disagree with the tables it summarises.
    thread_spread = n.get("thread_spread_percent") or {}
    chunk_delta = n.get("chunk_delta_percent") or {}
    floor_rows = n.get("noise_floor_rows") or {}

    def spread_text(key: str) -> str:
        parts = [
            f"{values[key]:.1f}%（{model}）"
            for model, values in thread_spread.items()
            if values.get(key) is not None
        ]
        return "、".join(parts) if parts else "**（数据缺失）**"

    def multiples_text() -> str:
        """Per-model widest spread as a multiple of the floor, ranged over models.

        Deliberately the same quantity ``thread_summary`` prints under each
        model's table — that one takes ``max(prefill, decode)`` for the model and
        divides by the floor. Taking the min over all four spreads instead would
        compare one model's decode spread against the floor and report a smaller,
        different number than the tables two sections down.
        """
        if not floor:
            return "**（数据缺失）**"
        widest = [
            max(v["prefill"], v["decode"])
            for v in thread_spread.values()
            if v.get("prefill") is not None and v.get("decode") is not None
        ]
        if not widest:
            return "**（数据缺失）**"
        return f"{min(widest) / floor:.1f} ~ {max(widest) / floor:.1f} 倍"

    thread_prefill_spread = spread_text("prefill")
    thread_decode_spread = spread_text("decode")
    thread_multiples = multiples_text()

    def chunk_text(tag: str) -> str:
        deltas = chunk_delta.get(tag)
        if not deltas:
            return "**（数据缺失）**"
        return " / ".join(f"{d:+.1f}%" for d in deltas)

    floor_sentence = "、".join(
        f"{model} {values['ab']:.1f} ms vs {values['sweep']:.1f} ms（{values['percent']:.1f}%）"
        for model, values in floor_rows.items()
    ) or "**（数据缺失）**"

    # §6.4's break-even, derived rather than asserted. ``ceil`` because a query
    # count is an integer and a fractional break-even still means the next whole
    # query is the first one that gains; the raw quotient is printed alongside so
    # "打平" (quotient ≤ 1) is distinguishable from "第 2 次才回本".
    build = n.get("documents_build_seconds")
    saved = n.get("documents_per_query_saved_seconds")
    quotient = n.get("documents_break_even_queries")
    build_text = "**（数据缺失）**" if not build else f"**{build * 1e3:.1f} ms**"
    saved_text = "**（数据缺失）**" if not saved else f"**{saved * 1e3:.1f} ms**"
    if quotient is None:
        break_even_text = "**（数据缺失）**"
        break_even_source = "数据缺失，无法推算"
    else:
        first = max(1, math.ceil(quotient))
        break_even_text = str(first)
        break_even_source = (
            f"建索引 {build * 1e3:.1f} ms ÷ 每次查询省下 {saved * 1e3:.1f} ms "
            f"= {quotient:.2f} 次"
        )
    document_corpus = n.get("documents_corpus_size")
    corpus_text = "**（数据缺失）**" if not document_corpus else str(document_corpus)
    queries_text = "**（数据缺失）**" if not n.get("documents_queries_per_round") else str(
        n.get("documents_queries_per_round")
    )

    # §7.2 and §9.1 both have to state whether the paired experiment exists, and
    # what it concluded. PAIRED_SUMMARY is filled by the section that prints the
    # table, so these sentences cannot describe a different experiment from the one
    # the reader just saw. The two branches differ in more than wording: without it,
    # every row in §9.1 is capped at "观察到加速".
    paired = PAIRED_SUMMARY
    if paired:
        sign = paired.get("sign")
        # "顺序均衡全部为「是」" is a claim about the *retained* rounds now, which is
        # the column the table prints. Gating it on the model-level count alone was
        # how §7.2 came to assert a balance the table's own rows contradicted.
        ordered = not paired["unbalanced"] and not paired["unbalanced_rows"]
        unsettled = paired["unbalanced_rows"] + paired["insufficient_rows"]
        paired_exists_text = (
            f"`bash tools/perf/run_noise_floor.sh <输出目录> {paired['rounds']}` 已经跑过："
            f"**{paired['models']} 个模型 × {paired['rounds']} 轮**，"
            "每对 base／opt 在独立进程里背靠背跑，AB／BA 顺序交替"
            + (
                "（表里 `顺序均衡（保留轮次）` 全部为「是」）"
                if ordered
                else (
                    f"（其中 **{paired['unbalanced_rows']} 个（模型 × 量）在剔除异常轮次之后"
                    "顺序不再均衡**，这些行不判方向，见 §7.3）"
                )
            )
            + f"，共判读 **{paired['rows']} 个（模型 × 量）**，"
            f"其中 **{paired['directional']} 个达到「有限轮次方向一致」判据**、"
            f"**{paired['indeterminate']} 个 {INDETERMINATE}**、"
            f"**{unsettled} 个因保留轮次不均衡或不足而不判方向**。"
            + (
                f"全部 {sign[0]} 个逐轮配对里 {sign[1]} 个大于 1.00"
                f"（双侧精确二项 p = {sign[2]:.2g}；行之间不独立，"
                "这个 p 值高估证据强度，也**没有**剔除不均衡的轮次，"
                "按行看表里的 `结论` 列）。"
                if sign
                else ""
            )
            + "进程内与进程间离散度分列，判据逐行用自己的多轮数据，"
            "不引用任何统一噪声百分比。详见 §7.3 的第一张表。"
        )
        paired_gain_text = (
            f"{paired['directional']}/{paired['rows']} 个（模型 × 量）达到"
            "**「有限轮次方向一致」**判据——每一轮独立进程各算一遍 base/opt、"
            f"保留的有效配对 ≥ {MIN_RETAINED_ROUNDS}、"
            "保留轮次 AB／BA 顺序均衡、且没有任何一轮跨过 1.00"
        )
        paired_indeterminate_text = (
            f"{paired['indeterminate']} 个 {INDETERMINATE}"
            if paired["indeterminate"]
            else "0 个方向未定"
        )
        paired_unsettled_text = (
            f"{paired['unbalanced_rows']} 个 {UNORDERED}、"
            f"{paired['insufficient_rows']} 个 {INSUFFICIENT}"
            if unsettled
            else "没有因轮次问题而不判方向的行"
        )
        round_span = paired["round_span"]
        paired_round_span_text = (
            f"{round_span[0]}–{round_span[1]} 轮独立进程"
            if round_span and round_span[0] != round_span[1]
            else f"{round_span[0]} 轮独立进程" if round_span else "有限轮次"
        )
    else:
        paired_exists_text = (
            "`bash tools/perf/run_noise_floor.sh <新的输出目录> 4` 会跑多轮独立进程配对、"
            "AB／BA 顺序均衡，并把进程内与进程间离散度分开记录，"
            "让每个负载用自己的多轮数据判断稳定性。**本轮没有跑它**，"
            "所以 §7.3 的第一张表只有那两行偶然同配置的运行，"
            "并明确标注为不是实验设计。"
        )
        paired_gain_text = "没有配对数据"
        paired_indeterminate_text = "**（无配对实验，全部方向未定）**"
        paired_unsettled_text = "**（无配对实验）**"
        paired_round_span_text = "**（无配对实验，轮数未知）**"

    # §9.1's decode column used to carry a hardcoded "（单窗口观察值）". Whether a
    # row is single-window is a property of the JSON, and §2.3 already prints it row
    # by row from the same field, so the summary has to read it too or the two
    # sections describe different data.
    window_counts = n.get("decode_window_counts")
    decode_single = n.get("decode_single_window_rows") or []
    if not window_counts:
        decode_window_text = "**（数据缺失）**"
    elif decode_single:
        decode_window_text = f"**其中 {len(decode_single)} 行是单窗口观察值**"
    else:
        low, high = window_counts
        decode_window_text = (
            f"每侧 {low:.0f} 个窗口" if low == high else f"每侧 {low:.0f}–{high:.0f} 个窗口"
        )

    # §9.1's 整块 prefill band, with the models the self-check threw out named.
    # A band over four models reads like a band over five unless the gap is stated,
    # and the excluded one is precisely the largest model — the one a reader is most
    # likely to go looking for.
    raw_prefill_excluded = n.get("raw_prefill_512_excluded") or []
    raw_prefill_models = n.get("raw_prefill_512_models")
    raw_prefill_text = span(n.get("raw_prefill_512")) + (
        f"（{raw_prefill_models} 个模型；{'、'.join(raw_prefill_excluded)} "
        "的 `raw_prefill_full` 自检不过，未计入，见 §2.2）"
        if raw_prefill_excluded
        else ""
    )

    # §2.4's optimisation-2a pricing, read out of the artifacts rather than typed
    # in. Every figure below is a field of one of two JSON files, so the prose
    # cannot survive the probe being re-run while still looking authoritative.
    sig = n.get("sigmoid") or {}
    gate = sig.get("gate") or {}
    fixed = sig.get("fixed") or {}
    fixed_gate = fixed.get("gate") or {}
    legacy_gate = fixed.get("legacy_gate") or gate

    def sig_ms(value: Any) -> str:
        return "**（数据缺失）**" if not value else f"{value * 1e3:.3f} ms"

    def sig_pct(value: Any) -> str:
        return "**（数据缺失）**" if not value else f"{value * 100:.1f}%"

    def sig_x(value: Any) -> str:
        return "**（数据缺失）**" if not value else f"{value:.2f}×"

    # Gate statistics are read off a dict that may be the *legacy* half of the
    # correction, so a missing key is a real possibility and not a zero. The rows
    # below used ``.get(k, 0)``, which prints "0.0%" / "+0.0000" — a value that
    # reads as measured, and in this table specifically as "the old probe found no
    # negative activations", the exact claim being withdrawn.
    def gate_pct(node: Any, key: str) -> str:
        value = (node or {}).get(key)
        return "**（数据缺失）**" if value is None else f"{value * 100:.1f}%"

    def gate_signed(node: Any, key: str) -> str:
        value = (node or {}).get(key)
        return "**（数据缺失）**" if value is None else f"{value:+.4f}"

    def gate_plain(node: Any, key: str) -> str:
        value = (node or {}).get(key)
        return "**（数据缺失）**" if value is None else f"{value:.4f}"

    def band_x(pair: Any) -> str:
        return span(pair, unit="×", digits=2)

    def band_pct(pair: Any) -> str:
        return (
            "**（数据缺失）**"
            if not pair
            else f"{pair[0] * 100:.1f}%–{pair[1] * 100:.1f}%"
        )

    def band_ms(pair: Any) -> str:
        return (
            "**（数据缺失）**"
            if not pair
            else f"{pair[0] * 1e3:.3f}–{pair[1] * 1e3:.3f} ms"
        )

    spread = fixed.get("spread") or {}
    run_rows = fixed.get("runs") or []
    run_count = fixed.get("run_count") or 0

    sigmoid_block: list[str] = [
        "- **`sigmoid` 改写（优化 2a）的归因：本文早先的版本是错的，下面先给更正，"
        "再给更正后的数字。**",
        "",
        "  四个量曾被当成一个量来引用，它们互不通用：",
        "",
        "  | # | 量 | 是什么 | 能不能叫「MLP 归因」 |",
        "  | --- | --- | --- | --- |",
        "  | (1) | 算子单独计时 | 新旧 `sigmoid` 在同一份输入上各自跑一遍 | 不能，它是算子比值 |",
        "  | (2) | 7 个 matmul 中位数之和 | 每个 matmul 在热循环里单独计时再相加 | 不能，它不是整层 profile |",
        "  | (3) | 真实前向里只替换 `silu` | 同一层、同一份输入，只改一个变量 | **能** |",
        "  | (4) | 端到端收益 | `ab/` 的 workloads，本探针不测 | 能，但那是另一份数据 |",
        "",
        "  已发布的 (1)/(2) 比值被称为「占整层 matmul 预算的 "
        f"{sig_pct(sig.get('old_share'))}」，那是把 (1) 除以 (2) 再当成 (3) 来讲。"
        "**这个说法撤回。** (2) 里没有 softmax、没有 norm、没有残差加、没有任何逐元素运算，"
        "而且它的每一项都是在热循环里单独测的——它不是一个层的执行时间。",
        "",
        "  同时撤回的还有三个具体数字的来源：它们建立在探针自称「真实 gate 激活」、"
        "实际却是 `rms_norm(embed_tokens(ids), input_layernorm.weight) @ gate_proj.weight.T` "
        "的张量上。真实的 decoder layer 在 embedding 与 `gate_proj` 之间还有整个 attention 块、"
        "一次残差加，以及**另一个** norm（`post_attention_layernorm`，权重与 "
        "`input_layernorm` 不是同一个张量）。",
        "",
        "  **「出自 RMSNorm，所以天然居中」这句话也是错的。** RMSNorm 除以均方根，"
        "它从不减均值；减均值的是 LayerNorm，而 RMSNorm 被写出来的一部分理由正是省掉那一步。"
        "实测也否证了它：真实捕获的激活均值是 "
        f"{fixed_gate.get('mean', float('nan')):+.4f}，不是 0。",
    ]

    if fixed:
        within = spread.get("within_run_rel_stdev")
        layer_band = spread.get("layer_saved_share")
        layer_points = (
            f"{(layer_band[1] - layer_band[0]) * 100:.1f}" if layer_band else "若干"
        )
        sigmoid_block += [
            "",
            f"  更正后的测量（`{fixed.get('model')}` layer {fixed.get('layer')}，"
            f"{fixed.get('tokens')} token，BLAS `{fixed.get('blas')}`，"
            f"每项 {fixed.get('repeat')} 次取中位数、5 次预热）。",
            "",
            f"  **下面每个数字都是 {run_count} 次独立运行的区间，不是某一次的读数。** "
            "这是重测过程中才发现的，而且它正是本文早先犯过的同一个错误："
            f"一次运行内部 {fixed.get('repeat')} 个样本的 stdev/median 只有 "
            f"{band_pct(within)}，看起来已经收敛；"
            f"但 {run_count} 次运行之间，真实整层占比相差 {layer_points} 个百分点"
            f"（{band_pct(layer_band)}）。"
            "**一次运行内的连续样本不是重复实验**——与 §3 里 decode 那条同一个道理，"
            "只是这次是从算子探针这一侧又出现了一遍。",
            "",
            f"  原始数据在 `{repo_relative(SIGMOID_CORRECTION.parent)}/`："
            "`sigmoid-check.json` 是最新一次，`sigmoid-check-run*.json` 是更早的，"
            "全部保留、未覆盖。**这个先后次序只能由文件 mtime 给出**"
            "（四份 JSON 的键集完全相同、内容互不相同，探针没有往输出里写时间戳），"
            "所以它比本文其它结论弱一档：一次 `cp -p` 或一次 checkout 就能把它抹掉。"
            "要让它变成可核对的证据，得给 `sigmoid_check.py` 加一个写入时刻字段并重跑。"
            f"最新一次记录的 `ops.py` **代码哈希** "
            f"`{(fixed.get('identity') or {}).get('ops_py_code_sha256', '（缺）')[:16]}…`"
            "（对文档字符串与行号不敏感的那一个，理由见本节末尾）。",
            "",
            f"  | 量 | 旧写法 | 新写法 | 结论（{run_count} 次区间） |",
            "  | --- | --- | --- | --- |",
            f"  | (1) 算子，**真实捕获**的 gate 激活 | "
            f"{band_ms(spread.get('operator_old_seconds'))} | "
            f"{band_ms(spread.get('operator_new_seconds'))} | "
            f"{band_x(spread.get('operator_speedup'))} 快 |",
            f"  | (1) 算子，全正 / 全负合成分布 | — | — | "
            f"只有 {band_x(spread.get('one_sided_speedup'))}，代价取决于是否跨 0 |",
            f"  | (3) **真实 MLP**，只替换 `silu` | "
            f"{band_ms(spread.get('mlp_old_seconds'))} | "
            f"{band_ms(spread.get('mlp_new_seconds'))} | "
            f"省下 {band_ms(spread.get('mlp_saved_seconds'))}"
            f" = 真实 MLP 的 **{band_pct(spread.get('mlp_saved_share'))}** |",
            f"  | (3) **真实 decoder layer**（含 attention / 两个 norm / 两处残差） | "
            f"{band_ms(spread.get('layer_old_seconds'))} | "
            f"{band_ms(spread.get('layer_new_seconds'))} | "
            f"省下 {band_ms(spread.get('layer_saved_seconds'))}"
            f" = 真实整层的 **{band_pct(spread.get('layer_saved_share'))}** |",
            "",
            "  区间是怎么来的，逐次读数全部列出（**每一行都是一次独立进程**）：",
            "",
            "  | 运行 | (1) 算子 | (2) 7 个 matmul 之和 | (1)/(2) | (3) 真实 MLP | (3) 真实整层 |",
            "  | --- | --- | --- | --- | --- | --- |",
            *[
                f"  | `{row.get('name')}` | {sig_x(row.get('operator_speedup'))} | "
                f"{sig_ms(row.get('matmul_total_seconds'))} | "
                f"{sig_pct(row.get('operator_over_matmul_sum'))} | "
                f"{sig_pct(row.get('mlp_saved_share'))} | "
                f"{sig_pct(row.get('layer_saved_share'))} |"
                for row in run_rows
            ],
            "",
            "  能站住的说法是上表最后一行给出的**区间**：**这次改写从一个真实 decoder "
            f"layer 里去掉了 {band_pct(spread.get('layer_saved_share'))}**。"
            f"(1) 那个 {band_x(spread.get('operator_speedup'))} 是算子自身的比值，"
            "不是 MLP 或整层变快了这么多——层里绝大部分不是 sigmoid。",
            "",
            "  读区间时有一件事不能做：**拿一列的下端去除另一列的上端**。"
            "同一列的两个端点来自不同的运行，跨列相除得到的是两次运行的混合，"
            "不对应任何一次测量。比值列是每次运行各自算好再取区间的。",
            "",
            "  这一节能说到哪一步，四个说法要分开：",
            "",
            "  | 说法 | 证据 | 结论 |",
            "  | --- | --- | --- |",
            "  | 功能正确 | 7 类输入的原始数组逐位比较（非 NaN 元素） | **成立** |",
            f"  | 省掉工作 | (1) 算子区间 {band_x(spread.get('operator_speedup'))}，"
            f"{run_count} 次同向 | **成立** |",
            f"  | 观察到加速（层内） | (3) 真实整层 {band_pct(layer_band)}，"
            f"{run_count} 次全部为正 | **成立** |",
            "  | 测得稳定收益（端到端） | 本探针不测端到端；那要 §3 的 A/B，"
            "而 §3 那 25 行本轮无同窗口自检、全部 `待重测` | "
            "**本节不能下这个结论** |",
            "",
            "  新旧激活差多远，探针直接算了出来："
            f"旧捷径与真实捕获值的 rel_l2 = **{fixed.get('shortcut_rel_l2', float('nan')):.4f}**，"
            "也就是差值比信号本身还大（这个值不随运行漂移，它是权重与输入 id 的函数）。"
            "三个被撤回的数字与更正后的对应值：",
            "",
            "  | | 已发布（错误激活） | 更正后（真实捕获） |",
            "  | --- | --- | --- |",
            f"  | 算子加速 | {solo(sig.get('speedup'))} | "
            f"**{band_x(spread.get('operator_speedup'))}** |",
            f"  | 负值占比 | {gate_pct(legacy_gate, 'fraction_negative')} | "
            f"**{gate_pct(fixed_gate, 'fraction_negative')}** |",
            f"  | 激活均值 | {gate_signed(legacy_gate, 'mean')} | "
            f"**{gate_signed(fixed_gate, 'mean')}** |",
            f"  | 激活标准差 | {gate_plain(legacy_gate, 'std')} | "
            f"**{gate_plain(fixed_gate, 'std')}** |",
            f"  | (1)/(2) 比值 | {sig_pct(sig.get('old_share'))} | "
            f"{band_pct(spread.get('operator_over_matmul_sum'))}"
            "（**仍然不是任何真实层的占比**） |",
            f"  | 真实整层归因 | 没有测过 | **{band_pct(layer_band)}** |",
            "",
            "  方向没变（改写仍然是净收益，而且收益很大），但**幅度变了，而且幅度本身"
            f"也只能给到区间**：算子加速从 {solo(sig.get('speedup'))} 降到 "
            f"{band_x(spread.get('operator_speedup'))}，"
            "「占整层 153% 量级」的说法换成「真实整层省下 "
            f"{band_pct(layer_band)}」。"
            f"旧数字全部保留在 `{sig.get('withdrawn_path') or repo_relative(withdrawn_sigmoid_path(PERF))}`，"
            "未删除、未改写；"
            "新数字在独立目录里，两份都在。",
            "",
            "  **为什么记两个哈希。** 上面这些区间被引用进 `ops.sigmoid` 的文档字符串，"
            "而那段字符串就在被测量的 `ops.py` 里面。于是「把结论写回源码」会改变"
            "源码的原始文件哈希，让刚刚记录的身份失效，进而要求重测，重测又移动数字——"
            "循环关不上。探针因此同时记原始文件哈希与**代码哈希**（把语法树里的文档"
            "字符串清空后再哈希，行号也不参与）：后者只在计算本身改变时改变。"
            "判定「这次测量还认不认当前源码」用的是后者，前者照样记录，用于最严格的"
            "「是不是逐字节那份文件」。`tests/test_perf_tooling.py` 同时钉住两件事："
            "代码哈希必须匹配，且**任何一次保留的运行落在上面区间之外就报错**。",
            "",
            f"  逐位相同的结论也跟着重测了：{len(fixed.get('failures') or [])} 处失败。"
            "探针自己打的 note："
            + " ".join(fixed.get("notes") or []),
            "",
            "  **注意这里区分了两件事**：NaN 的*位置*一致，与 NaN 的*位模式*一致，"
            "不是同一个断言。上面那条 note 说明的是前者；`nan` 有多种位模式，"
            "`nan / nan` 可以落在与 `exp(nan)` 不同的 payload 上，而两边同样都是 NaN。"
            "逐位结论只对非 NaN 元素成立，且是用原始数组比字节得到的，"
            "不是从日志打印的聚合误差推出来的。",
            "",
            "  合成分布（标准正态 / 全正 / 全负）**保留**，但它们的作用是给算子的"
            "输入敏感性划界，不是对本模型的测量；探针里它们单独列在 "
            "`SYNTHETIC_DISTRIBUTIONS` 中，捕获到的真实激活不在其中。"
            f"标准正态那一档的算子比值是 "
            f"{band_x(spread.get('operator_speedup_synthetic_normal'))}，"
            "比真实捕获的激活更高——**这恰好说明为什么不能拿合成分布当本模型的归因**。",
            "",
            "  复现：`.venv/bin/python tools/perf/sigmoid_check.py "
            "--out <新目录>/sigmoid-check.json`。要得到区间而不是单点，"
            "就在同一个目录下多跑几次并保留每一次的输出文件；"
            "报告读取的是整个目录。",
        ]
    else:
        sigmoid_block += [
            "",
            f"  **更正后的测量还没有跑**（期望在 "
            f"`{sig.get('fixed_path') or repo_relative(DEFAULT_SIGMOID_CORRECTION)}`）。"
            "在跑出来之前，这一项只能记为：**已发布的 "
            f"{solo(sig.get('speedup'))} / {sig_pct(sig.get('old_share'))} / "
            f"{gate_pct(gate, 'fraction_negative')} 三个数字全部撤回**，"
            "没有任何数字可以替代它们。已发布的那份 "
            f"`{repo_relative(withdrawn_sigmoid_path(PERF))}` 原样保留。",
            "",
            f"  已发布的算子计时（**建立在错误激活上，仅作记录**）：旧写法 "
            f"{sig_ms(sig.get('old_ms'))}、新写法 {sig_ms(sig.get('new_ms'))}、"
            f"7 个 matmul 中位数之和 {sig_ms(sig.get('matmul_total_seconds'))}、"
            f"全正 / 全负 {span(sig.get('one_sided_speedup'))}（BLAS `{sig.get('blas')}`）。",
        ]

    cache = n.get("session_cache") or {}
    cache_mib = span(cache.get("mib"), unit=" MiB", digits=1)
    contexts = cache.get("max_context") or []
    ctx = "/".join(str(c) for c in contexts) if contexts else "**（数据缺失）**"

    # §7's basis, and the reproduction recipe has to agree with it. The same
    # discriminator ``noise_floor()`` uses decides which section gets written, so a
    # data root that *ran* the paired experiment cannot also carry the sentence
    # saying it did not. That sentence is the one thing gating every cross-process
    # comparison in the report: leaving it hardcoded tells a reader to distrust
    # numbers while pointing away from the evidence sitting in the same directory.
    paired_basis = paired_directory() is not None
    # §0's first reading rule describes where the cross-process dispersion comes
    # from. That differs in kind between the two roots, not just in wording: one
    # round has a designed AB/BA paired experiment, the other has two runs that
    # happened to share a configuration. Asserting the weaker provenance while
    # generating from the stronger data tells the reader to distrust a number that
    # §7.3 derives from four independent paired rounds — and §0 is the section they
    # read first.
    if paired_basis:
        discipline_one = (
            f"1. **跨进程离散度是 {floor_text}，但它不是一个可以套到别的量上的门限。**"
            "这个数来自 `tools/perf/run_noise_floor.sh` 那个**设计过的**实验："
            "多轮独立进程、每对 base／opt 背靠背、AB／BA 顺序均衡，"
            "进程内与进程间离散度分列（见 §7.3）。"
            "**本文早先写的「两个进程内区间互不重叠」是错的**——那句建立在两次偶然"
            "同配置运行上，而且数据里也只有一个模型不重叠。"
            "现在改成：**不引用任何统一的噪声百分比**，每个量用自己的多轮数据判定，"
            f"判不出来的一律写 `{INDETERMINATE}`，既不写成收益也不写成回退。"
            "**判方向用的是 `逐轮增益`，不是进程间那一列**；"
            "进程间那一列是**不配对**比较的分辨率。见 §7。"
        )
    else:
        discipline_one = (
            f"1. **本轮观测到的跨进程离散度是 {floor_text}，但它不是一个门限。**"
            "这个数来自同一棵 opt 树、线程环境变量全不设、在两个独立进程里各测一次的"
            "**两次偶然同配置运行**——不是设计过的实验。"
            "**更正：本文早先写的「两个进程内区间互不重叠」是错的**——"
            "数据里两个模型只有一个不重叠，逐行的相差与重叠判定见 §7.2 那张表"
            "（本节不复述数字，免得两处漂移）。"
            "两行差了一个量级，正说明这两点撑不起一个统一底线："
            "把两者的最大值当成所有负载的分辨率是没有依据的。"
            "所以现在改成——**不引用任何统一的噪声百分比**，"
            "凡是差值落在这个量级以内的配置，结论一律写 "
            f"`{INDETERMINATE}`，既不写成收益也不写成回退；"
            "要给方向性结论，先跑 `bash tools/perf/run_noise_floor.sh <新目录> 4`"
            "（多轮独立进程配对、AB／BA 顺序均衡），让每个负载用自己的数据说话。"
            "见 §7。"
        )
    step4_note = (
        "# 4. 跨进程离散度实验（本文 §7 的结论依赖它）"
        if paired_basis
        else "# 4. 跨进程离散度实验（本文 §7 的结论依赖它；已发布那一轮没有跑）"
    )

    # §5.2/§5.4's vision figures, from the same nodes §5.3's tables read. Three of
    # them had a second, handwritten copy in the prose and two of the three had
    # already drifted: the other-image row read 0.57× in the table and 0.56× in the
    # bullet, and the pixel-hash time appeared as both "16.5 ms" and "16.4 ms".
    vision_metrics = n.get("vision_metrics") or {}

    def vision_x(key: str, digits: int = 2) -> str:
        value = vision_metrics.get(key)
        return "**（数据缺失）**" if not value else f"{value:.{digits}f}×"

    key_hash = n.get("vision_key_hash_ms") or {}
    key_hash_rows = sorted(
        (v for v in key_hash.values() if v.get("ms") and v.get("num_frames")),
        key=lambda v: v["num_frames"],
    )
    key_hash_text = (
        "、".join(f"{v['num_frames']} 帧 {v['ms'] * 1e3:.1f} ms" for v in key_hash_rows)
        or "**（数据缺失）**"
    )
    frames_text = (
        f"{key_hash_rows[-1]['num_frames']} 帧 vs {key_hash_rows[0]['num_frames']} 帧"
        if len(key_hash_rows) > 1
        else "帧数见 §5.3 那张表"
    )
    hit_encode = n.get("vision_hit_encode_ms")
    hit_encode_text = "**（数据缺失）**" if not hit_encode else f"实测 {hit_encode:.1f} ms"
    other_image = n.get("vision_other_image_ratio")
    other_image_text = "**（数据缺失）**" if not other_image else f"实测 {other_image:.2f}×"

    # §7.2's opening paragraph. Its subject is *where the dispersion number comes
    # from*, and that differs in kind between the roots: one has a designed paired
    # experiment, the other has two runs that happened to share a configuration and
    # whose per-row detail (``floor_rows``) the paired section does not fill — so
    # generating from the paired root printed "同一配置、两个独立进程：**（数据缺失）**"
    # under a sentence claiming there were only two points.
    sweep_quantity = f"引擎分块 prefill {FALLBACK_PROBE[1]} tok"
    if paired_basis and paired:
        noise_provenance = (
            f"**{floor_text} 不是一个「噪声底线」。**它是 §7.3 那张表里 "
            f"`{sweep_quantity}` 一行 opt 侧的**进程间**离散度"
            f"（跨 {paired['rounds']} 轮独立进程取中位），"
            "它只说明一件事：**不配对地**比较这个量，小于这个幅度分不出来。"
            "**更正**：本文早先把一个同类的数称为「噪声底线」并写成「小于它的差不算收益」，"
            "还写过「两个进程内区间互不重叠」。两句都撤回——"
            "前者把**一个量的观测**当成了**所有负载的门限**，"
            "后者在数据里只对一部分行成立（逐行判定见 §7.3）。"
            "所以现在的判读规则**不引用任何统一百分比**："
            "每个量用自己那几轮的**逐轮增益**判方向，"
            f"判不出来的一律写 `{INDETERMINATE}`。"
        )
    else:
        noise_provenance = (
            f"**{floor_text} 不是一个「噪声底线」，只是本轮观测到过的跨进程差。**"
            f"同一配置、两个独立进程：{floor_sentence}。"
            "**更正**：本文早先把这个数称为「噪声底线」并写成「小于它的差不算收益」，"
            "还写过「两个进程内区间互不重叠」——后半句是错的，两个模型里只有一个不重叠。"
            "两点数据没有分布，取其中较大者当成所有负载的门限更没有依据："
            "同一个探针在不同模型上的跨进程差就差了一个量级（逐行见 §7.2 那张表）。"
            "所以现在的判读规则不再引用任何统一百分比，"
            f"落在这个量级以内的一律写 `{INDETERMINATE}`。"
        )
    step1_noise_note = (
        "# 跨进程离散度实验（多轮、AB/BA 顺序均衡）。本文 §7 的判定就来自它"
        if paired_basis
        else "# 跨进程离散度实验（多轮、AB/BA 顺序均衡）。本轮**没有跑**，见 §7.2"
    )

    # §1.3's inventory is deliberately the published round's — those paths name the
    # evidence behind the already-released report and do not move with --data. The
    # sentence claiming the re-collection matches it is a different matter: it is
    # checkable against ``ls`` and was false. The re-collection has no baseline/,
    # and adds noise-floor/, identity/, reference-fixed/, integration-fixed/ and
    # sigmoid/. Derived from both trees so it cannot drift a third time.
    def _entries(root: Path) -> set[str]:
        # Directories and data files only. The run logs and status text files differ
        # between the two roots by construction — every round writes its own — and
        # listing all eleven of them buries the one entry that matters, which is
        # that the re-collection has no ``baseline/``.
        if not root.is_dir():
            return set()
        return {
            p.name
            for p in root.iterdir()
            if p.is_dir() or p.suffix not in {".log", ".txt"}
        }

    only_published = sorted(_entries(DEFAULT_PERF) - _entries(PERF))
    only_here = sorted(_entries(PERF) - _entries(DEFAULT_PERF))
    logs_here = sum(
        1 for p in PERF.iterdir() if p.is_file() and p.suffix in {".log", ".txt"}
    ) if PERF.is_dir() else 0
    if PERF.resolve() == DEFAULT_PERF.resolve():
        inventory_note = "本轮数据根就是 `artifacts/perf-round1/`，与下表一致。"
    else:
        parts = [f"本文的数据根是 `{repo_relative(PERF)}/`，**目录结构与下表不完全一致**："]
        if only_published:
            parts.append(
                "下表里的 " + "、".join(f"`{x}`" for x in only_published) + " 这一轮没有；"
            )
        if only_here:
            parts.append("这一轮另外多出 " + "、".join(f"`{x}`" for x in only_here) + "。")
        if logs_here:
            parts.append(
                f"（另有 {logs_here} 份本轮自己的运行日志与状态文件，逐份对应上面 §1.2 "
                "的命令，不在此列举。）"
            )
        inventory_note = "".join(parts)

    # §0's discipline 2 named "0.59× 和 1.29×" and "115 s 和 3828 s" as its two
    # examples of an end-to-end number this machine cannot support. Both were typed
    # in from the published batch. §3 prints the disagreement rows and §9.4 derives
    # the integration span, so a report generated from the re-collected root opened
    # by quoting numbers neither of those sections contained — the same defect as
    # §9.4's hand-written rows, one section earlier and harder to notice because
    # §0 reads as framing rather than as data.
    spread = n.get("end_to_end_spread")
    if not spread:
        spread_clause = "**（磁盘上没有两遍都测全的工作负载，举不出这样的例子）**"
    else:
        spread_model, spread_name, spread_low, spread_high, _, spread_root = spread
        spread_clause = (
            f"同一份代码、同一台机器，{spread_model} 的 `{spread_name}` 负载先后跑出过 "
            f"**{spread_low:.2f}× 和 {spread_high:.2f}×**"
            f"（`{spread_root}/` 的 `ab/` 与 `ab-longctx/`，两份原始 JSON 都还在磁盘上）。"
        )
    windows = n.get("integration_durations") or []
    if len(windows) < 2:
        integration_clause = ""
    else:
        seconds = sorted(value for _, value in windows)
        integration_clause = (
            "同一份代码的集成测试也先后跑出过 "
            + "、".join(f"{value:.2f} s" for value in seconds)
            + f"（{len(seconds)} 个窗口，相差 {seconds[-1] / seconds[0]:.0f} 倍）。"
        )

    cross_process_rule = (
        "- 跨进程比较前先读 §7：那里是 AB/BA 配对的独立进程实验，"
        "进程内与进程间离散度分开记，每个量各自判定；"
        "判不出来的一律记 "
        f"{INDETERMINATE}，不要从单次运行里读方向。"
        if paired_basis
        else (
            "- 跨进程比较前先读 §7：那里说明本轮**没有**做过配对实验，"
            "只有一个两次偶然同配置运行给出的离散度量级，它不是门限。"
            "要做方向性判断，先跑 `bash tools/perf/run_noise_floor.sh <新目录> 4`。"
        )
    )

    doc: list[str] = [
        "<!-- 表格部分由 tools/perf/make_report.py 生成，不要手改；",
        "     叙述部分也在同一脚本里（write_doc），改完重跑脚本即可。 -->",
        "",
        "# 推理加速第 1 轮实测（2026-10-23）",
        "",
        "[返回 README](../README.md) · [变更记录](changelog.md) · [架构说明](architecture.md) · "
        "[性能与内存](benchmarks.md) · [测试与数值验证](validation.md)",
        "",
        "> **本文的每个数字都是本轮实测的**，不是理论估算，也不是 v0.1 / v0.2 的历史基准。"
        "每一节都给出产生它的命令与原始 JSON 的位置。"
        "表格由 `tools/perf/make_report.py` 从 JSON 直接生成，"
        "叙述里引用的区间也由同一脚本从同一批 JSON 算出——**没有人工转录的数字**。",
        "",
        "本轮只做两件事：**删掉重复计算**，和**新增显式的复用接口**。"
        "不改数值路径，不引入 GPU 后端、量化、推测解码或并发服务；"
        "默认运行路径仍是 Python + NumPy、CPU、FP32、batch size 1。",
        "",
        "---",
        "",
        "## 0. 先说怎么读这份报告",
        "",
        "三条纪律，是本轮踩到坑之后写死在生成脚本里的，不是姿态：",
        "",
        discipline_one,
        "2. **一个孤立的端到端数字不能单独作为结论。**"
        f"{spread_clause}"
        "**这两个数都不能用**：本文早先写过「补跑后是 1.05× ~ 1.29×，所以 0.59× "
        "不是回退」，那句话已撤回——补跑更快并不能证明旧结果不是回退，"
        f"现在的结论是 **{INDETERMINATE}**，两次的原始记录都保留（见 §3）。"
        f"{integration_clause}"
        "所以每张端到端表都带一列**自检**（同一棵树用两种方式测同一个 decode 速率，"
        "两侧的比值应当一致），两遍运行的矛盾也**逐行公开**而不是只留好看的那一遍。"
        "见 §3。",
        "3. **没有收益的实验回退，不留半成品。**"
        "三项改动实测无收益或落在波动内，已从代码里撤掉，理由写在被影响函数的 docstring 里。"
        "见 §8。",
        "",
        "还有一条贯穿全文的区分：**「省掉的工作被直接观测到了」不等于「加速被测出来了」**。"
        "优化 1 属于前者而不属于后者，§2 会把这件事讲清楚，而不是把两者混成一个数字。",
        "",
        "---",
        "",
        "## 1. 环境、命令与原始结果位置",
        "",
        "### 1.1 两棵树",
        "",
        "| 树 | 怎么来 | 身份 |",
        "| --- | --- | --- |",
        "| `base` | `tools/perf/baseline_tree.py` 从 git 修订 **`8968e39`** 用 `git archive` 物化到 "
        f"`{repo_relative(BASELINE_TREE)}/`（这是该脚本的固定默认位置，"
        "**不随 `--data` 变**：重采那一轮没有重建基线，理由见 §6） | "
        "逐文件 sha256 与该修订核对；快照目录只是缓存，"
        "不是唯一一份证据 |",
        "| `opt` | 仓库当前工作区 `src/` | 逐文件 sha256。本轮改动**未提交**，没有修订号可引，"
        "哈希就是它的身份 |",
        "",
        "```bash",
        "# 物化基线（已存在则只做哈希校验），并把校验过的路径打到 stdout",
        "BASE_SRC=$(.venv/bin/python tools/perf/baseline_tree.py --print-dest)",
        "",
        "# 打印某棵树的全部逐文件哈希 / 只打印短 digest",
        ".venv/bin/python tools/perf/source_identity.py --src \"$BASE_SRC\"",
        ".venv/bin/python tools/perf/source_identity.py --src . --digest-only",
        "```",
        "",
        "`baseline_tree.py` **不覆盖已有快照**：目录非空而源码哈希对不上，它列出差异文件并"
        "以非零退出，而不是重新物化。两棵树的差异被逐文件核对过：`config.py` "
        "**逐字节相同**（默认值一个没动），`output_projection` **逐字节相同**"
        "（所以 §2 里在 opt 树上测到的投影代价可以直接给 base 树当初多做的那几次定价）。",
        "",
        "### 1.2 命令",
        "",
        "全部串行执行，**从头到尾没有两个基准同时跑过**——两个基准会争抢同一批核与"
        "同一条内存带宽，那样两个数都没有意义。",
        "",
        "```bash",
        "# 步骤 1~4：逐模型相邻交替的 A/B（micro + workloads + chat）、",
        "#          opt 树的线程/分块扫描、图像特征缓存 A/B、文档索引 A/B",
        "# 不带 out-root 时会在 artifacts/ 下**新建一个唯一运行目录**并把路径打到 stderr；",
        "# artifacts/perf-round1{,-rerun} 这两个已发布证据目录一律拒绝写入，",
        "# 尾斜杠 / 相对路径 / 绝对路径 / 符号链接都按 realpath 归一化后再判断，",
        "# 冲突在开始昂贵测量**之前**报出，并且不会留下半棵树。见 tools/perf/out_guard.py",
        "bash tools/perf/run_round1.sh [out-root]",
        "",
        "# 补跑：workloads 套件在 --max-context 2048 下重测五个模型",
        "bash tools/perf/run_round1_longctx.sh [out-root]",
        "",
        "# 下面三条单独调用时，先要一个新运行目录；已存在的 --out 文件会被拒绝而不是截断",
        'RUN=$(.venv/bin/python tools/perf/out_guard.py fresh-root); echo "$RUN"',
        "",
        "# 单独定价一次词表投影（优化 1 的归因），一个模型一个进程。",
        "# 这个探针是整轮里最不可重复的一项（答案是两个约 540 ms 中位数之差），",
        "# 所以第二次尝试绝不替换第一次：换目录，两份才可比",
        ".venv/bin/python tools/perf/projection_probe.py \\",
        "    --model models/Qwen3-0.6B-slowq \\",
        '    --out "$RUN/projection/Qwen3-0.6B-slowq.json"',
        "",
        "# 重新给 sigmoid 改写定价（优化 2 的归因）：捕获真实前向里 silu 收到的张量，",
        "# 并在真实 MLP / 真实 decoder layer 上只替换 silu 做新旧对比",
        ".venv/bin/python tools/perf/sigmoid_check.py \\",
        '    --out "$RUN/sigmoid/sigmoid-check.json"',
        "",
        step1_noise_note,
        'bash tools/perf/run_noise_floor.sh "$RUN" 4',
        "",
        "# 从上面所有 JSON 重新生成本文的表格部分",
        '.venv/bin/python tools/perf/make_report.py --data "$RUN"',
        "```",
        "",
        "`run_round1.sh` 的步骤 1 **在每个模型内部相邻交替**跑 base 和 opt，"
        "而不是先跑完五个 base 再跑五个 opt：这台机器会节流，页缓存只装得下约四份 "
        "Qwen3-1.7B 权重，隔一小时跑的两次看到的不是同一台机器，"
        "隔一分钟跑的两次基本上是。**比较的单位是那一对，所以那一对必须相邻。**",
        "",
        "### 1.3 原始结果位置",
        "",
        "`artifacts/` 被 `.gitignore` 忽略，**不随仓库分发**；下表是本机上的位置，"
        "用上面的命令可以重新产生。**工具本身在 `tools/perf/`，入版本管理**，"
        "用法与数据布局见 [`tools/perf/README.md`](../tools/perf/README.md)。",
        "",
        "下表是**已发布那一轮**的证据目录。`artifacts/perf-round1/` 与 "
        "`artifacts/perf-round1-rerun/` 都由 `tools/perf/out_guard.py` 保护："
        "**全部**驱动与测量程序都拒绝往里面写，路径按 realpath 归一化后判断，"
        "所以尾斜杠、相对路径、绝对路径和指向它们的符号链接都拦得住；"
        "不带参数调用驱动会新建一个唯一运行目录并把路径打到 stderr。"
        "重测产生的是新的证据，不是对旧证据的修订——覆盖掉再拿新数字当同一条序列，"
        "会让更正无法审计。",
        "",
        inventory_note,
        "",
        "**这一批 JSON 全部早于 `source_identity` 块**，所以没有一个能用哈希核对到源码；"
        "它们只能靠日期挂靠。重新采集的那一批带身份块，见 §9。",
        "",
        "| 目录 / 文件 | 内容 |",
        "| --- | --- |",
        "| `artifacts/perf-round1/ab/` | 步骤 1，5 模型 × {base, opt}，micro + workloads + chat |",
        "| `artifacts/perf-round1/ab-longctx/` | 补跑，同 5 模型 × 2 树，`--max-context 2048`。"
        "**当时只跑了 `--suite workloads`，JSON 里 `micro` 为空**，所以 §3 那 25 行"
        "拿不到窗口内自检，全部标 **待重测**；驱动已改成 `--suite all --no-chat` |",
        "| `artifacts/perf-round1/baseline/` | 本轮**之前**那棵树的线程 / 分块扫描（7 配置 × 2 模型） |",
        "| `artifacts/perf-round1/sweep-opt/` | 优化后那棵树的同一批扫描，列与 `baseline/` 对齐 |",
        "| `artifacts/perf-round1/projection/` | 一次词表投影的净代价，5 模型 |",
        "| `artifacts/perf-round1/sigmoid-check.json` | sigmoid 新旧写法计时 + 7 类输入的逐位相同检查。"
        "**其中「真实 gate 激活」那一行的归因已撤回**，见 §2.4；文件原样保留 |",
        "| `artifacts/perf-round1-rerun/sigmoid/sigmoid-check.json` | **更正后**的 sigmoid 归因："
        "真实前向捕获的 gate 激活 + 真实 MLP / 真实整层只替换 `silu` 的新旧对比 |",
        "| `artifacts/perf-round1/vision/` | 图像特征缓存 A/B（SmolVLM-256M-Instruct） |",
        "| `artifacts/perf-round1/documents/` | 预编码文档索引 A/B（Qwen3-Embedding-0.6B） |",
        "| `artifacts/perf-round1/integration-opt12/`、`integration-opt4/` | 8 模型真实权重集成测试日志，各 208 passed |",
        "| `artifacts/perf-round1/run_round1.log`、`run_round1_longctx.log` | 两次运行的完整 stderr |",
        "",
        # Every path below is derived, none of it typed in, because the claim this
        # table makes is "the same invocation wrote both halves" — a hardcoded path
        # makes that claim about whatever the *last* invocation happened to write. It
        # was wrong: the round-2 document pointed a reader at
        # ``artifacts/perf-round1/report-tables.md`` as 「与本文表格部分同源」 while its
        # own tables came from a different data root and a different paired experiment,
        # 65 收益 rows against the document's 64. Two commands, two experiments, and a
        # sentence asserting they were one.
        #
        # A table rather than a sentence for a reason that is not cosmetic: ``reflow``
        # wraps prose at 88 columns and a path long enough to overflow the line budget
        # gets cut mid-path even inside backticks, so the very thing a reader would
        # grep for is the thing that stops being greppable. Table rows keep their bytes.
        "**本次调用的输入与输出**（正文与表格由同一次调用写出，逐项一致，可以互相核对）：",
        "",
        "| 参数 | 本次实际使用的路径 |",
        "| --- | --- |",
        f"| `--data` | `{repo_relative(PERF)}` |",
        f"| `--paired-data` | `{repo_relative(paired_directory()) if paired_directory() else '（未传，且数据根下没有 ' + NOISE_DIR + '/）'}` |",
        f"| `--out`（表格部分） | `{repo_relative(OUT)}` |",
        f"| `--doc`（本文） | `{repo_relative(DOC)}` |",
        "",
        "`artifacts/perf-round1/report-tables.md` 与 "
        "`artifacts/perf-round1-rerun/report-tables.md` 是历史上另外的调用写出的派生表，"
        "**原样保留，不是本文表格的来源**：它们各自读的是自己那一次的数据根，"
        "行数与本文不同属于来源不同，不是某一处数字写错、也不该手改数字去对齐。",
        "",
        "### 1.4 测量环境",
        "",
    ]
    doc.extend(demote(environment))
    doc += ["", "### 1.5 源码身份", ""]
    doc.extend(demote(identity))
    doc += [
        "",
        "---",
        "",
        "## 2. 优化 1 + 2：删掉 prefill 与 decode 里的冗余计算",
        "",
        "### 2.1 各自省掉了什么工作",
        "",
        "**优化 1：非末 prefill chunk 不再做词表投影。**",
        "`CausalLMModel.prefill()` 原来对每个 chunk 都调 `forward()`，"
        "但生成只从最后一个 token 采样，所以只有末 chunk 的 logits 被读过。"
        "现在非末 chunk 走 `encode()`（decoder stack + KV 写入，不碰 `lm_head`），"
        "末 chunk 走 `forward()`。**被观测到的**投影次数从 4 降到 1"
        "（512 token / `chunk_size=128`），计数来自一个带深度守卫的 instrumented spy，"
        "同时统计 `forward` 与 `encode`，不是从 `length // chunk_size` 推的。",
        "",
        "**优化 2a：`sigmoid` 改写成 `exp(-|x|)` 的无分支形式。**",
        "原来的写法用两次布尔掩码索引，要在 MLP 的 `[T, intermediate_size]` 激活上做四次 gather。"
        "这一项是本轮 prefill 加速的**主要来源**，实测见 §2.3。",
        "",
        "**优化 2b：单 token decode 跳过恒为全零的 causal mask。**",
        "在「完整因果注意力 + 只读有效 KV 前缀」下，一 token 前向的因果三角已经覆盖全部缓存键，"
        "`causal_attention_mask` 返回的是全零 `[1, 1, 1, S]`；FP32 下加零是恒等操作，"
        "而这个 mask 建一次却被每一层消费一次。"
        "触发条件是**推导出来的**，不是假设的：掩码元素 `(i, j)` 为零当且仅当 "
        "`j <= offset + i` 对所有 `j < offset + num_tokens` 成立，化简即 `num_tokens <= 1`，"
        "所以它恰好在单 token 时触发，绝不会误伤还有自己三角的 chunk。",
        "",
        "### 2.2 正确性怎么验的",
        "",
        "| 改动 | 判据 | 结果 |",
        "| --- | --- | --- |",
        "| 优化 1 | `test_prefill_matches_projecting_every_chunk_exactly`："
        "拿「每个 chunk 都 `forward` 再取最后一个」和 `prefill()` 比 `np.array_equal` | "
        "**逐位相同**，不是近似 |",
        "| 优化 1 | 要求观测 `logits` 的 `DebugProbe` 恢复每 chunk 投影 | "
        "调试运行可以慢，不可以算出另一个结果 |",
        "| 优化 1 | `kv_cache.commit` 仍每 chunk 一次、仍在 `encode` 内部 | "
        "§4 的 cache 不变量未受影响 |",
        "| 优化 2a | `sigmoid_check.py` 在 7 类输入上比新旧两种写法（含 `-0.0`、`±inf`、`nan`） | "
        "`bit_equality_failures: []`；**非 NaN 元素逐位相同**，NaN 只保证位置一致 |",
        "| 优化 2b | 触发条件由掩码为零的充要条件推导（见 §2.1） | "
        "分块不变性测试仍 PASS（`rtol 2e-4 / atol 2e-5`） |",
        "| 两者 | 5 模型 × 5 负载的 greedy 输出 `text_sha` 两侧比较 | "
        "**25/25 相同**，逐 token 一致 |",
        "| 两者 | 快速测试 | **826 passed, 62 deselected** |",
        "| 两者 | 8 模型真实权重集成测试，重跑两次 | "
        "**两次都是 208 passed、0 failed**，逐目录一致 |",
        "| 两者 | 8 模型官方参考验证：重新抓取 SlowQ 侧再比 | "
        + reference_identity_cell()
        + " |",
        "",
        "上表四行「两者」记录的是**本轮当时**的执行结果，不是修复后的最终代码。"
        "审查之后源码又改过（见 §9），这些行要重新执行才成立，重跑记录见 §9。",
        "",
        "`sigmoid` 的比较里有两个坑，分开记：",
        "",
        "1. `nan` 有多种位模式，`nan / nan` 可以落在与 `exp(nan)` 不同的 payload 上，"
        "所以直接比字节会在两边同样都是 NaN 的地方报差异。做法是先用 "
        "`np.array_equal(..., equal_nan=True)` 比，再只在两边都非 NaN 的位置比字节。"
        "**因此「NaN 位置一致」与「NaN 位模式一致」是两个不同的断言**，"
        "本报告只声称前者；探针把它作为一条 note 单独打出来，不并入逐位结论。",
        "2. 逐位结论只能来自原始数组的比较。日志里打印的聚合误差（rel_l2、max_abs）"
        "两侧相同**不能**证明张量逐位相同——聚合量相等允许元素级差异互相抵消。",
        "",
        "`group_query_attention` **默认返回 pre-softmax scores** 的契约保留了，"
        "`softmax` 明确不得写入入参——否则调用方会在「pre-softmax」这个名字下拿到概率，"
        "而且能过所有数值检查。缓冲区所有权写死在 docstring 里。",
        "",
        "### 2.3 实测",
        "",
    ]
    doc.extend(demote(micro))
    doc += [
        "",
        # No narrative paragraph of my own here: the generated section opens by
        # saying the engine_prefill speedup is opt 1 + opt 2 combined and must be
        # split, which is exactly what a lead-in would say, and the measured range
        # is in the §2.3 table immediately above. A wrapper that paraphrases the
        # section it wraps is how this doc grew two copies of every conclusion.
        "### 2.4 两项改动的归因：prefill 的加速不是优化 1 带来的",
        "",
    ]
    doc.extend(demote(attribution))
    # Bullet 1 used to assert the share band's lower bound is negative, and to read
    # that as "the probe is telling us this quantity is below its own resolution".
    # Both halves were wrong and the second was the worse error: it turned a signal
    # that the measurement is *invalid* into a signal that it is merely *small*,
    # which invites continued use of the neighbouring cells — and those come from
    # the same contaminated paired timing. Since ``forward`` is ``encode`` plus one
    # real projection, a negative paired difference is arithmetically impossible;
    # such rows are voided, so whether the bound is negative at all is a property
    # of the batch in hand, not a fact to be asserted.
    share_band = n.get("opt1_share_percent")
    voided = n.get("projection_voided") or []
    voided_text = f"本批标 `作废` 的有：{'；'.join(voided)}。" if voided else "本批没有作废行。"
    why_voided = (
        "作废的理由**不是**「这个量低于探针自己的分辨率」："
        "`forward` = `encode` + 一次真实存在的投影，成对差值为负在算术上不可能，"
        "只能是**测量无效**（模型在成对计时的两半之间被换出了内存）。"
        "这个区别很要紧——读成「量太小」会让人继续引用同一格里相邻的那几个数，"
        "而它们来自同一次被污染的成对计时。"
    )
    if share_band and share_band[0] < 0:
        bound_bullet = (
            f"- 上面那个占比区间的**下界是负数**（{share_band[0]:.2f}%）。"
            "那不是说跳过投影会变快。" + why_voided + voided_text
        )
    else:
        lower = f"{share_band[0]:.2f}%" if share_band else "**（数据缺失）**"
        bound_bullet = (
            f"- 上面那个占比区间的下界是 **{lower}**，本批**不是负数**——"
            "成对差值中位数为负的行已整行 `作废`，不进这个区间。"
            + voided_text
            + why_voided
        )
    projection_band = n.get("projection_cost_ms")
    projection_text = (
        f"{projection_band[0]:.2f} ms 到 {projection_band[1]:.2f} ms"
        if projection_band
        else "**（数据缺失）**"
    )
    doc += [
        "",
        "**归因结论：**",
        "",
        # The range, the noise-floor comparison and the "work observed, speedup not
        # resolvable" verdict are printed a few lines above by
        # ``section_attribution``, computed from the same JSON, so restating them
        # here would just be a second copy to drift. This bullet carries only what
        # that section does not: how to read the bound.
        bound_bullet,
        "- **优化 2 拿走了其余部分。**「两条独立路线」那张表是这件事的交叉核对，"
        "它的相差列就是这次归因的可信度：两条路线的输入、路径形状、测量时刻都不同，"
        "所以对得上才是真的对得上，而不是同一次测量换两种写法。",
        *sigmoid_block,
        "- 优化 2b（单 token 跳过全零 mask）只作用于 decode，"
        f"而 decode 的实测加速只有 {span(n.get('raw_decode_short'))}，"
        "**本轮没有把 2a 与 2b 再拆开单独计时**，所以不能说这两者各占多少。",
        "",
        "### 2.5 适用条件",
        "",
        "- 优化 1 只在 **prompt 长于一个 chunk** 时触发：8 / 32 / 128 token 在默认 "
        "`chunk_size=128` 下都只有 1 个 chunk，投影次数本来就是 1，**这一项完全不参与**。"
        "表里 `base 投影` 列在这三行也是 1，不是 4。",
        "- 优化 1 的绝对收益跟 `vocab_size × hidden_size` 走：一次投影从 "
        f"{projection_text}（跨模型区间，取自上面那张表的「三者中位」列，"
        "作废行不计入）。小词表模型上它接近或低于这个探针自己的分辨率，"
        "分辨率见同表的「三次测量离散度」列。",
        "- 优化 2a 的收益跟 `chunk_size × intermediate_size` 走（sigmoid 的输入形状），"
        "与词表无关，所以在全部 5 个模型上都明显。",
        "- 优化 2b 只在 **decode**（单 token）路径触发，对 prefill 无影响。"
        "本轮**没有**把 2a 和 2b 再拆开单独计时——`raw_decode` 两行含两者之和。",
        "- 所有数字来自 greedy、FP32、batch size 1、Apple Accelerate BLAS。"
        "换 BLAS 后端（OpenBLAS / MKL）会改变 matmul 与 elementwise 的相对成本，"
        "因此**归因比例不能照搬到别的后端**，虽然两项改动在任何后端上都省掉同样的工作。",
        "",
        "---",
        "",
        "## 3. 端到端工作负载",
        "",
        "微基准说明算子层面省掉了什么，端到端说明用户感受到什么。"
        "5 个模型 × 5 个负载，每个 7 次取中位数（2 次预热后）。",
        "",
        f"**首 token 加速 {span(n.get('workload_first_token'))}，"
        f"整次请求墙钟加速 {span(n.get('workload_total'))}**"
        f"（数据源 `{n.get('workload_source')}/`）。"
        "墙钟的倍数明显低于首 token，因为**decode 那一段本轮没有被优化多少**"
        f"（`raw_decode` 只有 {span(n.get('raw_decode_short'))}）："
        "一个生成 64 token 的负载，decode 占掉大部分墙钟，prefill 省得再多也被摊薄。"
        "这正是 `multi-turn`（生成 11 token）和 `long-prompt-short-answer`（1662 进 / 32 出）"
        "的墙钟倍数最高的原因——它们的 prefill 占比最大。",
        "",
        "### 3.1 这一节为什么跑了两遍，以及两遍不一致的地方",
        "",
        "步骤 1 用 harness 默认的 `--max-context 1024`，"
        "而 `long-prompt-short-answer` 的 prompt 有 1662 token，"
        "**两棵树都因此拒绝执行**（不是 base 失败 opt 成功——分词发生在碰任何引擎之前）。"
        "被跳过的恰恰是唯一的 prefill 主导负载，也就是优化 1 针对的那个形状，"
        "缺了它这一轮最有力的证据就没测到，所以补跑了 `--max-context 2048`。",
        "",
        "补跑还暴露了一件更要紧的事：**同一份代码跑两遍，有四行结果相差超过 0.10×**。"
        "下面那张表把矛盾逐行公开，而不是只留好看的那一遍。",
        "",
    ]
    doc.extend(demote(workloads))
    doc += [
        "",
        "---",
        "",
        "## 4. 优化 3：显式的多轮 KV 复用",
        "",
        "### 4.1 省掉了什么工作",
        "",
        "本轮之前，CLI 的 `chat` 每轮都新建引擎请求：**第 n 轮要把前 n−1 轮的全部内容"
        "重新 prefill 一遍**。新增 `Engine.session()` → `Session` 之后，"
        "同一个会话内已经算过并写进 KV 的前缀不再重算。",
        "",
        f"实测第 4 轮首 token 加速 {span(n.get('chat_turn4_first_token'))}，"
        f"整段四轮对话 {span(n.get('chat_conversation'))}。"
        "第 1 轮必然是 1.00× 左右——没有东西可复用，"
        "如果第 1 轮也报出加速，那说明测的不是复用。",
        "",
        "复用边界按**实际 token IDs 的最长公共前缀**决定，不按文本："
        "BPE 会在轮次边界重新合并，同一段文字重新分词后可能得到不同的 id 序列。"
        "**少复用只是慢，多复用会让模型条件在一段它没见过的对话上。**"
        "表里的 `reused` 是引擎自己报告的复用 token 数，不是从 prompt 长度推的。",
        "",
        "### 4.2 正确性怎么验的",
        "",
        "- **cold 两遍的输出必须相同**，否则 reuse 改变了结果："
        "5 个模型 × 7 轮 × 4 轮次 = 28 次比较，**不一致 0 个**。",
        "- 四个边界情况都有测试（`tests/test_session.py`，新增 52 项）："
        "完全命中时仍要跑一个 token（生成采样的是最后一个位置的 **logits**，不是 hidden state）；"
        "失败或取消的请求会让 cache 比 `_committed` 长，所以 `_plan` 先 `del self._committed[length:]`；"
        "`_check_identity` 在模型或几何（层数、KV head 数、head dim、词表、`max_context`）变化时拒绝复用；"
        "图文会话直接拒绝。",
        "- **普通独立请求仍然互相隔离**：`stream_ids` 的独立路径照旧 reset，"
        "两个平铺请求不可能看到彼此的 KV。§4 的不变量 2 因此有了一个**写明的**例外。",
        "- `slowq chat` 默认走会话，`--no-reuse` 关掉，`/clear` 调 `session.reset()`，"
        "Ctrl-C 取消生成后会话仍可用。",
        "",
        "### 4.3 代价：第二份 cache",
        "",
        "复用要能在历史被编辑时回退，所以会话持有第二份 cache。"
        f"实测预留体积 {cache_mib}（随层数、KV head 数、`max_context` 走），"
        "开启时 CLI 会打印出来。**这是本轮唯一一处明确的内存换时间**，"
        "只发生在会话路径上，独立请求路径不付。",
        "",
        # Without this the 224.0 MiB in §4.4's table and the 448 MiB in
        # docs/architecture.md §6.1 read as a contradiction. They are the same
        # formula at two context lengths, and the allocation is reserved up front
        # rather than grown, so the number a user sees depends on a flag they set
        # at engine construction and not on the conversation they had.
        f"**那个区间是在 `max_context={ctx}` 下测的**（`tools/benchmark.py` 的默认值，"
        "见 §1.2 的命令行）。预留是**按 `max_context` 一次分配**的，不随对话变长而增长，"
        "所以 [架构说明 §4](architecture.md#4-kv-cache-语义) 的每 token 表在 1024 下给出的"
        "正是这些值；同一张表乘到 2048 就是 "
        "[架构说明 §6.1](architecture.md#61-多轮会话enginesession) 里 Qwen3-0.6B 的"
        "**约 448 MiB**。两个数都对，差别只在 `max_context`。",
        "",
        "### 4.4 实测",
        "",
    ]
    doc.extend(demote(chat))
    doc += [
        "",
        "### 4.5 适用条件",
        "",
        "- **只支持纯文本会话**，图文会话显式拒绝（不是静默退化）。",
        "- 收益随「共享前缀占新 prompt 的比例」上升：第 4 轮的复用率远高于第 1 轮。"
        "如果每轮的 prompt 几乎没有共同前缀（例如每次换一段无关的系统提示），"
        "复用会退化成 0，此时会话路径相对独立请求只多付一次前缀比较。",
        "- 历史被编辑、`/clear`、模型或几何变化都会让复用失效并重新 prefill，"
        "这是正确性要求，不是性能缺陷。",
        "",
        "---",
        "",
        "## 5. 优化 4a：图像特征缓存",
        "",
        "### 5.1 省掉了什么工作",
        "",
        "对同一张图问第二个问题时，视觉塔（SmolVLM-256M 上占一次请求的绝大部分）"
        "原来要**完整重跑一遍**。新增 `ImageFeatureCache` 之后，"
        "connector 之后的 `features` 被缓存，同图后续请求跳过整个视觉前向。",
        "",
        f"实测：同图同问第二次 {solo(n.get('vision_same_question'))}，"
        f"同图**换一个问题** {solo(n.get('vision_other_question'))}"
        "——features 与问题无关，所以换问题照样命中。",
        "",
        # The "only cache features, not the whole VisionOutput" rationale lives in
        # §5.3, next to the table of byte counts that makes it a measurement
        # rather than an assertion. Stating it here too meant two copies to drift.
        "### 5.2 正确性与失效",
        "",
        "- 缓存身份包含**像素内容与形状、attention mask、crop 排布、原图 / resize 尺寸、"
        "encoder 尺寸、processor 配置、`Weights.fingerprint()`**。"
        "换一张图必须重算——实测那条路径只有 miss 没有 hit。",
        "- `put` 拷入、`get` 拷出，两个请求不会持有同一个缓冲区，"
        "调用方改自己拿到的数组不会污染缓存。",
        "- 带 `DebugProbe` 时**双向绕过**缓存：调试路径必须走真实计算。",
        "- 生成结果：opt 侧 8 个生成步与关缓存基准比较，**不一致 0 个**，"
        "`all_steps_identical=True`；四条路径的实测缓存流量与从 step 列表推出的期望逐条相符。",
        "- 容量上限默认 128 MiB，`0` 关闭，负数报错。LRU 淘汰。",
        "- **计时诚实**：`vision_encode_seconds` 在命中时记的是**本次请求自己的**像素哈希时间"
        f"（{hit_encode_text}），不是当初那次前向的时长——"
        "把别人的耗时记在这次账上会正好掩盖缓存要证明的加速。",
        "- CLI 的两个图文入口显式传 `image_cache_bytes=0`：一问一答即退出，"
        "缓存只会在没有任何后续请求会用到它的情况下白付一次像素哈希。"
        "库路径默认开启，那才是「对同一张图问第二个问题」真正发生的地方。",
        "",
        "### 5.3 实测",
        "",
    ]
    doc.extend(demote(vision))
    doc += [
        "",
        "### 5.4 适用条件",
        "",
        "- 收益只在**同一张图被问第二次及以后**出现。第一次永远 miss，"
        f"并且要多付一次像素哈希（{key_hash_text}）。",
        f"- 换图必然重算，且换到更大的图会更慢（{frames_text}，{other_image_text}）——"
        "这一行不是回退，是那张图本身更大。倍数与 §5.3「四条对照」表里 "
        "`opt off-no-cache #1 → opt on-other-image #1` 那一行同源。",
        f"- 文本 prefill 与 decode **不被这个缓存影响**"
        f"（{vision_x('prefill_seconds')} / {vision_x('decode_seconds')}），"
        f"所以请求墙钟的倍数（约 {vision_x('wall_seconds', 0)}）"
        f"远小于视觉字段的倍数（约 {vision_x('vision_encode_seconds', 0)}）。",
        "",
        "---",
        "",
        "## 6. 优化 4b：预编码文档索引",
        "",
        "### 6.1 省掉了什么工作",
        "",
        "`EmbeddingEngine.search()` 每轮都把**全部文档重新编码一遍**："
        "16 篇语料 × 8 条查询 = 每轮 128 次 document 编码。"
        "新增 `index_documents()` → `DocumentIndex` 之后，"
        "`index.search(q)` 每轮只编码 query，document 编码次数为 **0**。",
        "",
        f"实测每轮墙钟：只查询 {solo(n.get('documents_queries_only'))}，"
        f"即使**每轮都重建索引**也有 {solo(n.get('documents_with_rebuild'))}。"
        "后一个数是刻意留的最保守读法——假设调用方一次都不复用索引。",
        "",
        "**原有 `search()` 一字未改、仍可用。**"
        "进程内、不落盘、无 ANN、无 RAG。",
        "",
        "### 6.2 正确性与失效",
        "",
        # The field list, the "why not allclose" rationale and the probe
        # description are printed by §6.3 next to the numbers they justify, and
        # that copy has to stay self-contained for report-tables.md. What belongs
        # here is the invalidation table, which §6.3 does not carry.
        "- 判据是**逐位相同**而不是 `allclose`，编码次数由贴在实例上的探针数出来"
        "而不是从公式推——两者的细节与实测结果见 §6.3。",
        "- 失效条件：模型对象、tokenizer 对象、`prefill_chunk_size`、"
        "`Weights.fingerprint()` 中任一变化即拒绝复用。"
        "`prefill_chunk_size` 在表里是因为分块只保证 2e-4 等价、**不保证逐位相同**；"
        "`max_length` 与 `dimension` 不在表里，因为它们不改变已编码向量的值。",
        "",
        "### 6.3 实测",
        "",
    ]
    doc.extend(demote(documents))
    doc += [
        "",
        "### 6.4 适用条件",
        "",
        f"- **第 {break_even_text} 次查询就打平，不是「只查一次白花」。**"
        f"建一次索引实测 {build_text}（16 篇文档各编码一次），"
        f"而 `search()` 每查一次都要把这 16 篇重新编码一遍，每次省下 {saved_text}。"
        "**这两件事是同一份工作**，所以查第 1 次时建索引的成本刚好等于省下的成本，"
        "第 2 次开始净赚，之后每多查一次就多赚一次。"
        f"（口径：{break_even_source}；语料 {corpus_text} 篇文档，每轮 {queries_text} 次查询。）",
        "- 语料变了就要重建。索引不检测文档内容是否被原地改掉——"
        "身份表里记的是**建索引时**的 fingerprint。",
        "",
        "---",
        "",
        "## 7. 优化 5：线程数与 prefill 分块参数——测了，默认值不动",
        "",
        "### 7.1 结论先说",
        "",
        "**两项默认值本轮都没有改动**，这是一个**有数据支撑的「不改」**，不是没做：",
        "",
        f"- **线程数**：四个配置（不设 / 1 / 4 / 8）之间，opt 侧 512-token 分块 prefill 的极差是 "
        f"{thread_prefill_spread}，decode 的极差是 {thread_decode_spread}。"
        f"每个模型取两者中较大的那个，与本轮观测到的 {floor_text} 跨进程离散度是 "
        f"{thread_multiples}。"
        f"**{INDETERMINATE}。**"
        "请求的线程数**确实**在 `import numpy` 之前进了环境（harness 记 "
        "`requested_before_numpy_import: true`），但耗时对它没有反应。"
        "至于是 Accelerate 忽略了这些变量、还是这些矩阵形状本来就受内存带宽限制，"
        "**本轮没有测，不做判断。**",
        f"- **`prefill_chunk_size`**：`c64`（{chunk_text('c64')}）和 "
        f"`c512`（{chunk_text('c512')}）**明确更慢**；"
        f"`c256`（{chunk_text('c256')}）按区间重叠判据属于**{INDETERMINATE}**。"
        "没有任何一个非默认值落在「明确更快」那一档，所以 **128 不动**。",
        "",
        "### 7.2 三个诚实声明",
        "",
        "**「请求的线程数」不是「实际工作线程数」。**"
        "有效 BLAS 线程数在 Python 里不可观测，harness 记为 `null` 而不是猜。"
        "所以下面任何结论都只针对「请求了这个数」。",
        "",
        noise_provenance,
        "",
        "**方向性结论现在有了，来自真正的配对实验。**"
        f"{paired_exists_text}",
        "",
        "### 7.3 实测",
        "",
    ]
    doc.extend(demote(sweep))

    # §9.4's three rows. Each one restated a number another section derives: the
    # chat ratios from §4's cold-vs-cold table, "报出 0.52×" from §3's decode
    # self-check, and the integration durations from the logs both roots keep.
    # Restating them by hand is how the published round's four ratios survived into
    # a report generated from a re-collection that measured one.
    slower = n.get("chat_cold_slower") or []
    chat_cold_rows = n.get("chat_cold_rows") or []
    if not chat_cold_rows:
        chat_cold_text = "**（数据缺失）**"
    elif not slower:
        chat_cold_text = (
            f"{len(chat_cold_rows)} 行 cold-vs-cold 计时**没有一行慢于 base**，"
            "所以这一项已不再待重测"
        )
    else:
        chat_cold_text = (
            f"{len(chat_cold_rows)} 行里 **{len(slower)} 行读起来像回退**（"
            + "、".join(f"{name} 第 {turn} 轮 {value:.2f}×" for name, turn, value in slower)
            + f"），与同一批被标记待重测的数字来自同一窗口，故**{INDETERMINATE}**："
            "既没被推翻也没被证实，原始记录保留。§4 那张表打印的是同一批单元格"
        )
    decode_suspects = n.get("decode_long_suspects") or []
    if not decode_suspects:
        decode_long_text = (
            "本批数据里**没有**触发长上下文自检的行"
            f"（判据：同一进程内长/短上下文之比两侧相差 >{SELF_CHECK_TOLERANCE:.2f}×，"
            "与加速方向无关），所以这一项不再待重测"
        )
    else:
        decode_long_text = (
            f"本批有 {len(decode_suspects)} 行触发长上下文自检（§3 逐行列出）："
            + "；".join(note.split("：", 1)[0] for note in decode_suspects)
            + f"，故**{INDETERMINATE}、标记待重测**"
        )
    durations = n.get("integration_durations") or []
    if len(durations) < 2:
        integration_text = "**（磁盘上不足两份集成测试日志，无法给出跨窗口离散度）**"
    else:
        seconds = [value for _, value in durations]
        integration_text = (
            f"同一份代码在这台机器上跑出过 "
            + "、".join(f"{value:.2f} s" for value in sorted(seconds))
            + f"，相差 **{max(seconds) / min(seconds):.0f} 倍**（{len(seconds)} 份日志："
            + "、".join(f"`{path}`" for path, _ in sorted(durations, key=lambda d: d[1]))
            + "）。"
        )

    # §10's "期望" lines. Both were typed in from the published round; the review
    # added tests, so a reader who reproduced correctly would have seen a different
    # number than the one the report told them to expect. Each one now names the file
    # it was read from rather than a root hard-coded here, because the two can come
    # from two different run roots and neither is ``REPRO_ROOT`` any more once a fresh
    # root holds a newer log — an expectation printed beside a directory it did not
    # come from is the same defect with a new number in it.
    quick = quick_suite_expectation()
    quick_expectation = (
        f"{quick[0]}（读自 `{quick[1]}`）" if quick else "**（本轮没有留下快速测试日志）**"
    )
    totals = integration_totals()
    integration_expectation = (
        f"{totals[0]} passed、{totals[1]} failed（`{totals[2]}/`，{INTEGRATION_MODELS} 份日志）"
        if totals
        else "**（磁盘上没有一套完整的集成测试日志）**"
    )
    doc += [
        "",
        "---",
        "",
        "## 8. 实验后回退 / 否决的改动",
        "",
        "按要求，没有收益或有回退的实验**回退实现并留下证据**，不留在代码里当半成品：",
        "",
        "| 实验 | 实测 | 处置 |",
        "| --- | --- | --- |",
        "| `softmax` 三个临时数组合并为单缓冲区 | ±1%，落在同一批数据的运行间波动内，"
        "即分配不是这个函数的成本 | **已回退**，理由写在 `softmax` 的 docstring 里，"
        "连同「不得写入入参」这条契约 |",
        "| `Weights.expect` 结果记忆化 | 无收益 | **否决** |",
        "| memmap 张量预先 `.T` | 无收益 | **否决** |",
        "",
        "另外有一项**没有回退但被降级**：优化 1 原本在 `model.py` 的 docstring 里被写成"
        "「被丢掉的三次投影是长 prefill 里最大的可避免开销」。"
        "§2.4 的直接测量显示它是 "
        f"{opt1_share}，而同一 prefill 上 `sigmoid` 改写拿走了绝大部分。"
        "**那句话是错的，已按实测重写**；改动本身保留（免费、逐位相同），"
        "但不再被宣称为可测加速。",
        "",
        "---",
        "",
        "## 9. 已完成 / 已回退 / 尚未验证",
        "",
        "### 9.1 已完成、并且**观察到加速**",
        "",
        "**下表里的倍数全部来自 `ab/`：每侧一个进程、进程内 7 次重复。**"
        "所以那一列**只能**支持「观察到加速」——7 个重复样本不是 7 次独立实验，"
        "它们共享同一个进程的 page cache、同一份已热起来的 BLAS 状态。"
        "这一点与 §2.4 sigmoid 探针那条是同一个道理，没有因为多跑了几个样本而改变。",
        "",
        f"- **decode 那一列**：{decode_window_text}"
        + (
            "（§2.3 已逐行标注："
            + "；".join(decode_single)
            + "）。"
            if decode_single
            else "，两侧都有多窗口离散度，不再是单窗口观察值。"
        ),
        f"- **「稳定收益」本轮没有测到；测到的是「有限轮次方向一致」，由 §7.3 的配对实验判：**"
        f"{paired_gain_text}；其余 {paired_indeterminate_text}"
        f"，另有 {paired_unsettled_text}。"
        "**这两列不能互相替代**：下表给的是幅度，配对表给的是方向一致性；"
        "一行可以幅度很大而方向不稳（Qwen3-1.7B 的 decode 就是），"
        "也可以方向一致而幅度小于进程间离散度（Qwen3-0.6B 的两行 decode 就是，"
        "幅度与它自己那行的 `进程间` 同一个量级）——"
        "后者能说「这几次配对都偏快」，不能说「已经确定快了那么多」，"
        "**更不能说「稳定收益」**："
        f"每行只有 {paired_round_span_text}，"
        "既没有预先声明的轮数下限，幅度也没有超过进程间离散度，更没有跨机器复现。",
        "",
        "| 改动 | 实测收益（`ab/`，单进程每侧） | 收益出现在什么条件下 |",
        "| --- | --- | --- |",
        "| 优化 2：sigmoid 无分支化 + 单 token 跳过全零 mask | "
        f"分块 prefill {span(n.get('engine_prefill'))}，"
        f"整块 prefill(512) {raw_prefill_text}，"
        f"decode {span(n.get('raw_decode_short'))}（{decode_window_text}） | "
        "所有 5 个模型、所有长度；prefill 上最大，decode 上最小。"
        "方向稳定性按模型分别判定，见 §7.3 配对表 |",
        "| 优化 3：多轮 KV 复用 | "
        f"第 4 轮首 token {span(n.get('chat_turn4_first_token'))}，"
        f"整段对话 {span(n.get('chat_conversation'))} | "
        "纯文本会话、共享前缀占比高时；第 1 轮无收益（应当如此） |",
        "| 优化 4a：图像特征缓存 | "
        f"同图第二次提问 {solo(n.get('vision_same_question'))}，"
        f"换问题 {solo(n.get('vision_other_question'))} | "
        "同一张图被问第二次及以后；首次必 miss |",
        "| 优化 4b：预编码文档索引 | "
        f"只查询 {solo(n.get('documents_queries_only'))}，"
        f"每轮重建也有 {solo(n.get('documents_with_rebuild'))} | "
        f"同一批文档被查询；**第 {break_even_text} 次查询就打平**"
        f"（建索引 {build_text} ≈ 一次查询省下的 {saved_text}），"
        "第 2 次开始净赚。语料变了要重建 |",
        "",
        "### 9.2 已完成但收益测不出来（保留，因为免费且逐位相同）",
        "",
        "| 改动 | 省掉的工作 | 为什么测不出来 |",
        "| --- | --- | --- |",
        "| 优化 1：非末 chunk 跳过词表投影 | "
        "投影次数 4 → 1（instrumented spy **直接观测**）；"
        f"一次投影的净代价 {projection_cost_text}{projection_resolution_text} | "
        f"占 512-token 分块 prefill 的 {opt1_share}；{opt1_verdict_text}"
        "且只在 prompt 长于一个 chunk 时触发 |",
        "| 优化 5：线程与分块参数 | 无（这是验证任务，不是优化） | "
        f"线程极差与观测到的跨进程离散度同量级（{thread_multiples}）；"
        "非默认 chunk 没有一个「明确更快」。**默认值不动** |",
        "",
        "默认值未改动这件事在源码里核对过，不是靠记忆："
        "`DEFAULT_PREFILL_CHUNK_SIZE = 128`（`embedding.py:54`、`engine.py:45`），"
        "`config.py` 与 baseline 树**逐字节相同**，"
        "`src/` 下没有任何 `os.environ[...] =` 写入。",
        "",
        "### 9.3 实验后回退",
        "",
        "见 §8：`softmax` 单缓冲区（已回退）、`Weights.expect` 记忆化（否决）、"
        "memmap 预先 `.T`（否决）。",
        "",
        "### 9.4 尚未验证",
        "",
        "下面每一条都给出「要补什么才能验证」，不含糊过去：",
        "",
        "| 项目 | 现状 | 要补什么 |",
        "| --- | --- | --- |",
        "| **Linux x86_64** | 全部验证都在 macOS arm64 / Apple M5 Pro 上执行。"
        "Linux 上只有 v0.1 的 Qwen3-0.6B 记录 | "
        "在那台机器上重跑 `run_round1.sh`；OpenBLAS/MKL 下归因比例很可能不同 |",
        "| **`chat` cold-vs-cold 计时** | "
        f"{chat_cold_text} | "
        "在同一窗口内相邻交替重跑 `chat` 套件（`--suite chat`），"
        "输出一致性那一列已经成立、不受影响 |",
        "| **`raw_decode_long`** | "
        f"{decode_long_text} | "
        "在空闲窗口单独重测该探针；`core_ab.py` 现已支持 `--repeat N` 多窗口，"
        "重跑即可得到该探针自己的离散度 |",
        "| **有效 BLAS 线程数** | Python 里不可观测，记为 `null` | "
        "需要从 C 层或 `Accelerate` 侧取；在那之前，"
        "「请求 N 线程 → 实际用 N 线程 → 耗时变化」这条因果链**没有建立** |",
        "| **优化 2a 与 2b 的分别定价** | `raw_decode` 两行含两者之和，未拆开 | "
        "加一个只关 mask 跳过的探针 |",
        "| **Qwen2.5-Coder-0.5B 的性能 A/B** | 只有集成测试（26 passed）与参考验证（PASS），"
        "未进本轮 A/B 的 5 模型清单 | 把它加进 `run_round1.sh` 的 `MODELS` 重跑 |",
        "| **图文会话的 KV 复用** | 优化 3 显式拒绝图文会话 | "
        "需要把 image token 的稳定身份纳入前缀比较，本轮未做 |",
        "| **缓存的跨进程 / 落盘持久化** | 本轮明确不做（进程内、不落盘） | 属于后续轮次的范围 |",
        "| **Qwen3-1.7B 的集成测试耗时** | "
        f"{integration_text}"
        "**原因未验证**：宿主机内存压缩／换页是一个*假设*，"
        "本轮没有在那几个测量窗口内采集任何系统证据"
        "（`vm_stat`、压缩器字节数、swap 用量、进程 RSS 时间序列都没有），"
        "所以不能把它写成原因。已知的只有「同一份代码、同一台机器、"
        "几次测量相差这么多」这个事实本身 | "
        "要么在测量窗口内同步采集上述系统指标再归因，"
        "要么**操纵**内存压力做受控实验；见 "
        "[测试与数值验证](validation.md)。"
        "**耗时不能用来判断代码快慢**这一条不依赖归因，正确性结论也不受影响 |",
        "",
        "---",
        "",
        "## 10. 复现",
        "",
        "```bash",
        "# 0. 一个全新的运行根目录，下面每一步都显式写进它。",
        "#    不带参数调用驱动时它们各自新建一个根，几步就散在几个目录里，",
        "#    合成一条序列反而更难核对，所以这里先建一个共用的。",
        "#    已存在的文件与目录一律不覆盖：out_guard 在开始昂贵测量之前报冲突并非零退出。",
        'RUN=$(.venv/bin/python tools/perf/out_guard.py fresh-root); echo "$RUN"',
        "",
        "# 1. 快速测试（离线，不下载权重）",
        '.venv/bin/python -m pytest -m "not integration" -q \\',
        '    2>&1 | tee "$RUN/quick-fixed.log"',
        f"#    期望值取自磁盘上最新那份快速测试日志（`$RUN` 是新建的，里面还没有日志，"
        f"所以这一格只能来自上一次运行）：{quick_expectation}",
        "",
        "# 2. 真实权重集成测试，逐模型单独设置 SLOWQ_MODEL_DIR。",
        "#    两个驱动都接受一个日志目录，已存在且非空的会被拒绝而不是截断。",
        'bash tools/perf/run_integration.sh "$RUN/integration-fixed/logs"',
        'bash tools/perf/run_reference.sh   "$RUN/reference-fixed"',
        f"#    期望：参考验证 8/8 退出码 0；集成合计（取自它自己点名的那个目录）"
        f"{integration_expectation}",
        "#    两个驱动都以**累计**失败作为自己的退出码：任一模型、任一步失败即非零，",
        "#    且不会在失败时打印「完成」。",
        "",
        "# 3. 全部性能测量（严格串行，耗时数小时）。",
        "#    这四步也可以一条命令跑完，并留下 recollection-status.txt：",
        '#      bash tools/perf/run_recollection.sh "$RUN"',
        'bash tools/perf/run_round1.sh "$RUN"',
        'bash tools/perf/run_round1_longctx.sh "$RUN"',
        "",
        step4_note,
        'bash tools/perf/run_noise_floor.sh "$RUN" 4',
        "",
        "# 5. 两项归因探针（机器空闲时单独跑）",
        ".venv/bin/python tools/perf/sigmoid_check.py \\",
        '    --out "$RUN/sigmoid/sigmoid-check.json"',
        "for m in SmolLM2-135M-Instruct SmolLM2-360M-Instruct \\",
        "         Qwen2.5-0.5B-Instruct Qwen3-0.6B Qwen3-1.7B; do",
        "  .venv/bin/python tools/perf/projection_probe.py \\",
        '      --model "models/${m}-slowq" \\',
        '      --out "$RUN/projection/${m}-slowq.json"',
        "done",
        "",
        "# 6. 重新生成本文的表格部分（--data 指向刚采集的那一轮）",
        '.venv/bin/python tools/perf/make_report.py --data "$RUN"',
        "```",
        "",
        "**测量纪律**（复现时若违反，数字不可比）：",
        "",
        "- 严格串行，任何时刻只有一个基准在跑。",
        "- A/B 的两棵树在**每个模型内部相邻交替**，不是先跑完一边。",
        "- 固定模型 revision、输入 token IDs（`probe_token_ids` 确定性生成）、输出长度、"
        "线程（全部不设）、设备（CPU）、精度（FP32）、采样（greedy）。",
        "- 预热 2 次后计时 7 次取中位数，并报告 min–max；"
        "首次前向与预热后稳态分开记（图像缓存那条路径的 step 0 必然 miss）。"
        "decode 的重复单位是**窗口**（每次重新 prefill 同样的 id 到全新 cache，"
        "再跑固定步数），同一窗口内连续的 token 不算独立重复。",
        "- 每份输出都记峰值 RSS、首 token、prefill、decode 与整次请求墙钟。",
        cross_process_rule,
        "",
    ]
    DOC.write_text("\n".join(reflow(doc)) + "\n", encoding="utf-8")
    print(f"wrote {DOC}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        default=str(DEFAULT_PERF),
        help="directory the harness JSON lives in (default: the round-1 evidence)",
    )
    parser.add_argument(
        "--paired-data",
        default=None,
        help=(
            "run root holding a re-measured AB/BA paired experiment; its "
            f"{NOISE_DIR}/ is read *instead of* --data's own, with no fallback — "
            "an unusable one fails the run rather than silently reading the old "
            "experiment (default: read --data's)"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="where to write the standalone tables (default: <data>/report-tables.md)",
    )
    parser.add_argument(
        "--doc",
        default=None,
        help=f"where to write the narrative document (default: docs/{DEFAULT_DOC_NAME})",
    )
    args = parser.parse_args(argv)

    # Before anything is resolved, so a second invocation in one process cannot inherit
    # the first one's conclusions. See :func:`_reset_run_state`.
    _reset_run_state()

    # Reassigned rather than threaded through as a parameter: ~60 call sites read
    # ``PERF`` and every one of them means "the data root this report describes".
    # A second module-level name for the same thing would be the kind of drift
    # that produces a report citing one directory's tables and another's prose.
    global PERF, OUT, DOC, SIGMOID_CORRECTION, PAIRED_ROOT
    PERF = _resolve(args.data, ROOT)

    # Judged before the first section is rendered, and therefore before either
    # output is touched. An argument that cannot be honoured has to fail while the
    # previous good copy is still on disk; the alternative is a run that replaces
    # the delivered document and *then* raises, which leaves neither the old report
    # nor a usable new one.
    PAIRED_ROOT = resolve_paired_root(args.paired_data, PERF)

    OUT = _resolve(args.out, ROOT) if args.out else PERF / "report-tables.md"
    DOC = _resolve(args.doc, ROOT) if args.doc else ROOT / "docs" / DEFAULT_DOC_NAME
    # Only the *default* is refused inside published evidence, and only because
    # nobody asked for that location: it follows ``--data``, so pointing the script
    # at a historical root would replace the derived table sitting beside the raw
    # JSON it was derived from. An explicit ``--out`` is a release action by the
    # person who typed it, so it is not second-guessed here.
    if not args.out:
        published = out_guard.published_conflict(OUT, ROOT)
        if published is not None:
            print(
                "make_report: 拒绝写入，表格的默认位置落在已发布证据目录内：",
                file=sys.stderr,
            )
            print(f"  - {out_guard.display(OUT)}：在 `{published}/` 内", file=sys.stderr)
            print(
                "  那份表格是已发布报告的派生证据，与它旁边的原始 JSON 一起原样保留。"
                "要给本轮生成表格，请用 --out 指定一个新位置。",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Same reasoning for the corrected attribution: it follows the data root when
    # that root carries one, so a report generated from a re-collected round quotes
    # that round's re-measurement. The published round has no ``sigmoid/`` of its own
    # (its probe output sits at the top level and is the thing being corrected), so
    # the default — the standalone correction directory — still applies to it.
    candidate = PERF / "sigmoid" / "sigmoid-check.json"
    if candidate.exists():
        SIGMOID_CORRECTION = candidate

    line: list[str] = [
        "<!-- 由 tools/perf/make_report.py 生成，不要手改。 -->",
        "<!-- 报告正文里的每个数字都来自这里读到的 JSON，不做人工转录。 -->",
        f"<!-- 数据根目录：{PERF} -->",
        f"<!-- 配对实验（§7.3）根目录：{paired_directory() or '（未运行）'} -->",
        # Which of the two arguments produced the line above. The path alone does not
        # say it: `<root>/noise-floor/` looks the same whether it was named or
        # inherited, and the round-2 report was wrong about exactly that.
        "<!-- 配对实验来源："
        f"{'--paired-data' if args.paired_data else f'--data 自己的 {NOISE_DIR}/'} -->",
        f"<!-- 正文（--doc）：{DOC} -->",
        "",
        "# 推理加速第 1 轮：测量表",
        "",
    ]
    sections = [
        section_identity,
        section_environment,
        section_micro,
        section_attribution,
        section_workloads,
        section_chat,
        section_vision,
        section_documents,
        section_sweep,
    ]
    bodies: list[list[str]] = []
    for section in sections:
        body: list[str] = []
        section(body)
        bodies.append(body)
        line.extend(body)
    OUT.write_text("\n".join(line) + "\n", encoding="utf-8")
    print(f"wrote {OUT}", file=sys.stderr)

    write_doc(bodies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
