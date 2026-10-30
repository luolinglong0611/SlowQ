#!/usr/bin/env python3
"""Price the model-identity check that guards the image feature cache.

    .venv/bin/python tools/perf/identity_cost.py \
        --model models/SmolVLM-256M-Instruct-slowq \
        --image artifacts/validation/SmolVLM-256M-Instruct-slowq/vision-grid-1x4.png \
        --image artifacts/validation/SmolVLM-256M-Instruct-slowq/vision-grid-2x4.png \
        --calls 200 \
        --out "$(.venv/bin/python tools/perf/out_guard.py fresh-root)/identity/SmolVLM-256M-Instruct-slowq.json"

``--out`` must name a path that does not exist yet; ``artifacts/perf-round1{,-rerun}``
are refused outright as published evidence. See ``tools/perf/out_guard.py``.

``VisionEngine._model_identity_changed`` runs before *every* image cache lookup.
That is a design decision with a cost attached, and the cost was asserted in a
docstring rather than measured: "0.028 ms median over 200 calls". This probe is
the artifact behind that sentence, and behind the sentence next to it — that
hashing the weights instead would cost more than the forward the cache skips.

The check covers more than the manifest fingerprint now: it also compares the model
object and the vision, preprocessing, token-id and shuffle config, because a
fingerprint describes the *manifest* and two models over one ``Weights`` with
different configs share one exactly. So the probe prices the layers separately —
``fingerprint`` inside ``identity_tuple`` inside ``identity_check`` — rather than
only the outermost one. A widening that was not measured is a widening nobody can
object to, which is how a per-request cost grows until it matters.

Six numbers, in the order they are collected:

``fingerprint``   :meth:`Weights.fingerprint` over ``--calls`` calls. It reads the
                  manifest and never a tensor, so its cost should track the
                  *entry count*, not the byte count; the JSON records both so a
                  reader can check that against a second model.
``identity``      ``_vision_identity`` and the whole of ``_model_identity_changed``
                  over the same ``--calls``, on an engine whose model has not moved.
                  Nested inside each other on purpose: the fingerprint is the first
                  of them, so the three together say what the config comparison and
                  the weak-reference check add on top of a bare manifest read, which
                  is the question the check's own docstring has to answer.
``key``           ``_image_cache_key`` on each ``--image``, i.e. the pixel hash
                  the identity check is being compared against. Measured here
                  rather than quoted from ``vision_cache_ab.py`` so that both
                  halves of the ratio come from one process and one memory state.
``encode_image``  The forward the cache exists to skip, for scale. Collected before
                  ``cache_paths``, which times that same forward through the two
                  paths that pay for it.
``cache_paths``   One image through each of the three ways it can reach the
                  connector — cache disabled, first miss, repeat hit — timed as the
                  wall clock of ``_vision_features`` and compared for bit-equality.
                  This is what a *request* pays, so it is the number that decides
                  whether the identity check is affordable; the ones above it only
                  explain where the cost goes. Collected after ``machine_after`` is
                  sampled, because running the vision tower moves the machine.
``weight_hash``   What the rejected alternative would cost: one sha256 pass over
                  every tensor's bytes. Collected **last** and only once, because
                  streaming hundreds of MiB through the page cache would perturb
                  everything above it.

Run one process per model, with nothing else resident. A probe reporting
hundredths of a millisecond must not be competing for bandwidth.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = Path(__file__).resolve().parent

# On sys.path before the measured tree, so this is unambiguously the harness's own
# tool and never a same-named module from a tree being timed. Inserted explicitly
# rather than relying on the script directory: tests/test_perf_tooling.py loads
# every tool in tools/perf by path, which gives a module no package context.
sys.path.insert(0, str(TOOL_DIR))

import out_guard  # noqa: E402
import source_identity  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import slowq  # noqa: E402
from slowq import Engine  # noqa: E402
from slowq.config import (  # noqa: E402
    TASK_IMAGE_TEXT_GENERATION,
    TASK_TEXT_EMBEDDING,
)
from slowq.embedding import EmbeddingEngine  # noqa: E402
from slowq.engine import recorded_task  # noqa: E402
from slowq.metrics import blas_info, environment_info, peak_rss_bytes  # noqa: E402
from slowq.vlm import VisionEngine, _vision_identity  # noqa: E402
from slowq.weights import Weights  # noqa: E402

assert slowq.__file__.startswith(str(ROOT / "src")), f"imported {slowq.__file__}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="price the pixel-hash side of the comparison too; repeatable. "
        "Ignored with a warning on a model whose engine has no image cache.",
    )
    parser.add_argument("--calls", type=int, default=200, help="fingerprint calls")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--key-calls", type=int, default=5, help="per image")
    parser.add_argument(
        "--path-calls",
        type=int,
        default=5,
        help="repeats per cache path (disabled / miss / hit), per image. These run "
        "the vision tower, so the count is a cost in seconds and not in samples.",
    )
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument(
        "--skip-weight-hash",
        action="store_true",
        help="do not stream every tensor through sha256 (slow on a large model)",
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_sha256(path: Path) -> str:
    """A hash of the *code* in ``path``, blind to its docstrings and comments.

    The numbers this probe reports land in ``vlm.py``'s own prose, which makes the
    identity self-referential: write the measured value into the docstring and the
    raw hash of the file the value describes changes, so the artifact appears to
    predate the source it was measured against. Hashing ``ast.dump`` with every
    docstring constant blanked breaks the loop — a comment-only edit keeps this
    hash, while any change to a statement does not.

    ``include_attributes=False`` matters: attributes would carry line numbers, and
    inserting a docstring line shifts every line below it.

    Duplicated from ``sigmoid_check.py`` rather than shared, deliberately: moving it
    into ``source_identity`` would change ``sigmoid_check``'s own code hash, and the
    artifacts that hash is recorded in are preserved evidence for numbers already
    published. Two twelve-line copies are cheaper than invalidating them.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body[0].value.value = ""
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def summarise(samples: Sequence[float]) -> dict[str, Any]:
    values = list(samples)
    out: dict[str, Any] = {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "calls": len(values),
    }
    if len(values) > 1:
        out["stdev"] = statistics.stdev(values)
        median = out["median"]
        # Dispersion against the median, so a reader can see whether the spread is
        # worth quoting as a band or is tight enough for a point estimate.
        out["spread_pct_of_median"] = (
            (max(values) - min(values)) / median * 100.0 if median else None
        )
    out["samples"] = values
    return out


def timed(fn: Callable[[], Any]) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def machine_state() -> dict[str, Any]:
    """Load and swap at the moment of measurement.

    ``environment_info`` describes the *software*; this describes how busy the
    machine was. The smallest number this probe reports is 0.028 ms, which is
    short enough that a background process can own it, so a reading without a
    load average next to it cannot be judged.

    Sampled at both ends of the timed section. The kernel refreshes its load
    averages only every few seconds and this section lasts well under one, so the
    two samples normally come back identical — that is the expected result, and
    its meaning is "the machine did not change underneath the measurement". A
    pair that *differs* is the signal worth reading: something else started or
    stopped mid-probe.

    Failures are recorded rather than raised: no reading is worse than "unknown".
    """
    state: dict[str, Any] = {}
    try:
        one, five, fifteen = os.getloadavg()
        state["load_average"] = [one, five, fifteen]
    except (OSError, AttributeError) as exc:
        state["load_average_error"] = str(exc)
    try:
        done = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
        )
        state["swap_usage"] = done.stdout.strip() if done.returncode == 0 else None
    except OSError as exc:
        state["swap_usage_error"] = str(exc)
    state["cpu_count_logical"] = os.cpu_count()
    return state


def load_engine(model_dir: Path, *, max_context: int, prefill_chunk_size: int) -> Any:
    """Whichever engine this directory's recorded task calls for.

    ``Engine.from_directory`` refuses a vision or embedding checkpoint by design,
    and the point of pricing the fingerprint on a *second* model is to show that
    the cost tracks the manifest's entry count rather than its byte count — which
    needs a large text model next to the small vision one.

    Keyword arguments are filtered against the signature instead of being listed
    per class: ``EmbeddingEngine`` takes ``max_length`` where the others take
    ``max_context``, and hardcoding that here is exactly the kind of detail that
    goes stale when a constructor grows a parameter.
    """
    task = recorded_task(model_dir)
    cls = {
        TASK_TEXT_EMBEDDING: EmbeddingEngine,
        TASK_IMAGE_TEXT_GENERATION: VisionEngine,
    }.get(task, Engine)
    wanted = {
        "max_context": max_context,
        "prefill_chunk_size": prefill_chunk_size,
    }
    accepted = inspect.signature(cls.from_directory).parameters
    kwargs = {name: value for name, value in wanted.items() if name in accepted}
    return cls.from_directory(model_dir, **kwargs)


def measure_fingerprint(weights: Weights, calls: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        weights.fingerprint()

    first = weights.fingerprint()
    samples = [timed(weights.fingerprint) for _ in range(calls)]

    # Hashability and stability are the two properties the cache key needs, and
    # both are cheap to assert here rather than to assume in a docstring.
    {first: True}
    stable = all(weights.fingerprint() == first for _ in range(5))
    tensor_table = first[-1]
    return {
        "seconds": summarise(samples),
        "stable_across_calls": stable,
        "hashable": True,
        "fingerprint_fields": len(first),
        "manifest_tensor_entries": len(tensor_table),
        # Both halves of the claim that the cost tracks the manifest's *entry*
        # count and not its byte count. Neither reads a tensor, so collecting them
        # here cannot perturb the timings above.
        "tensor_bytes_logical": weights.deduplicated_bytes(),
        "warmup": warmup,
    }


def measure_identity(engine: Any, calls: int, warmup: int) -> dict[str, Any]:
    """Price the identity check in layers, on an engine whose model has not moved.

    Three nested timings, each over the same ``calls``:

    ``fingerprint``      :meth:`Weights.fingerprint` alone — see
                         :func:`measure_fingerprint`, which collects it first and
                         is repeated here only as a same-process control.
    ``identity_tuple``   :func:`slowq.vlm._vision_identity`: the fingerprint plus
                         building the 5-tuple of frozen configs.
    ``identity_check``   the whole of ``_model_identity_changed``, i.e. that tuple
                         plus one weak-reference dereference and one comparison,
                         taking the ``return False`` branch every time.

    The difference between the last and the first is the entire cost of covering
    the config as well as the manifest, which is the number the fix has to justify.
    Quoting only ``identity_check`` would hide whether the widening was free or
    merely small, and quoting only ``fingerprint`` would understate what a request
    actually pays.

    ``changed`` counts how often the check reported a change. It must be zero: a
    check that fires on an unmoved model would clear the cache on every request,
    and the timings would then describe a cache that never hits — which is exactly
    the failure ``test_requests_without_a_swap_keep_hitting`` pins down in the
    suite, recorded here so the performance artifact cannot disagree with it.
    """
    for _ in range(warmup):
        _vision_identity(engine.model)
        engine._model_identity_changed()

    changed = 0

    def check() -> None:
        nonlocal changed
        if engine._model_identity_changed():
            changed += 1

    tuple_samples = [timed(lambda: _vision_identity(engine.model)) for _ in range(calls)]
    check_samples = [timed(check) for _ in range(calls)]
    return {
        # Wrapped in "seconds" to match measure_fingerprint's shape, so a reader of
        # the artifact navigates all three layers the same way.
        "identity_tuple": {"seconds": summarise(tuple_samples)},
        "identity_check": {"seconds": summarise(check_samples)},
        "changed_count": changed,
        "calls": calls,
        "warmup": warmup,
    }


def measure_cache_paths(
    engine: Any,
    model_dir: Path,
    images: Sequence[Path],
    *,
    calls: int,
    max_context: int,
    prefill_chunk_size: int,
) -> list[dict[str, Any]]:
    """One image through each of the three paths, timed and compared for equality.

    The paths differ in what they pay for, and the identity check is only two of
    them:

    ``disabled``  ``image_cache_bytes=0``, on a *separate* engine built that way.
                  Neither the identity check nor the pixel hash runs — the point of
                  the row is that turning the cache off is not free either, because
                  every request then pays for the tower.
    ``miss``      cold cache: identity check, pixel hash, the tower, and a copy in.
                  Cleared before each sample so every one is a genuine first
                  request; the clear is outside the timed region.
    ``hit``       warm cache: identity check, pixel hash, and a copy out. No tower.

    Two timings per sample, because they answer different questions. ``wall`` is
    ``_vision_features``'s own duration, which is what a request spends.
    ``reported`` is the seconds the call *returns*, and on the disabled and miss
    paths that is the tower's internal stopwatch rather than the wrapper's, so the
    gap between the two columns is precisely the overhead this probe exists to
    price. Reporting only one of them would either hide the overhead or charge the
    request for work the metrics do not claim.

    The features from all three paths, plus a direct ``encode_image``, must be
    bit-identical. That is the reason to measure them in one place: a cache that
    changed the answer would be a correctness bug wearing a speedup, and nothing in
    a timing table would show it. ``np.array_equal`` rather than a tolerance,
    because these are the same floats from the same weights in the same process —
    any difference at all is a difference in what was computed.
    """
    rows: list[dict[str, Any]] = []
    disabled_engine = VisionEngine.from_directory(
        model_dir,
        max_context=max_context,
        prefill_chunk_size=prefill_chunk_size,
        image_cache_bytes=0,
    )
    for image in images:
        prepared = engine.preprocessor.prepare(image)
        direct = engine.model.encode_image(prepared).features
        paths: dict[str, Any] = {}

        samples: list[float] = []
        reported: list[float] = []
        features = None
        for _ in range(calls):
            started = time.perf_counter()
            out, seconds, hit = disabled_engine._vision_features(prepared, probe=None)
            samples.append(time.perf_counter() - started)
            reported.append(seconds)
            features = out if features is None else features
            assert not hit
        paths["disabled"] = {
            "wall_seconds": summarise(samples),
            "reported_seconds": summarise(reported),
            "features_nbytes": int(np.asarray(features).nbytes),
        }

        samples, reported = [], []
        cached_features = None
        for _ in range(calls):
            engine.image_features.clear()
            started = time.perf_counter()
            out, seconds, hit = engine._vision_features(prepared, probe=None)
            samples.append(time.perf_counter() - started)
            reported.append(seconds)
            cached_features = out if cached_features is None else cached_features
            assert not hit
        paths["miss"] = {
            "wall_seconds": summarise(samples),
            "reported_seconds": summarise(reported),
            "features_nbytes": int(np.asarray(cached_features).nbytes),
        }

        samples, reported = [], []
        hit_features = None
        for _ in range(calls):
            assert len(engine.image_features) == 1, "the miss path left no entry"
            started = time.perf_counter()
            out, seconds, hit = engine._vision_features(prepared, probe=None)
            samples.append(time.perf_counter() - started)
            reported.append(seconds)
            hit_features = out if hit_features is None else hit_features
            assert hit
        paths["hit"] = {
            "wall_seconds": summarise(samples),
            "reported_seconds": summarise(reported),
            "features_nbytes": int(np.asarray(hit_features).nbytes),
        }

        # Bit-equality across every path and against the tower itself. Recorded
        # rather than only asserted, so the artifact says what was checked.
        comparisons = {
            "disabled_vs_direct": bool(np.array_equal(features, direct)),
            "miss_vs_direct": bool(np.array_equal(cached_features, direct)),
            "hit_vs_direct": bool(np.array_equal(hit_features, direct)),
            "hit_vs_miss": bool(np.array_equal(hit_features, cached_features)),
        }
        identical = all(comparisons.values())
        max_abs_diff = float(
            max(
                np.abs(np.asarray(a, dtype=np.float64) - np.asarray(direct, dtype=np.float64)).max()
                for a in (features, cached_features, hit_features)
            )
        )
        rows.append(
            {
                "path": str(image),
                "num_frames": prepared.num_frames,
                "features_shape": list(np.asarray(direct).shape),
                "paths": paths,
                "bit_identical": identical,
                "comparisons": comparisons,
                "max_abs_difference": max_abs_diff,
                "path_calls": calls,
            }
        )
        print(
            f"  cache_paths {image.name}  disabled "
            f"{paths['disabled']['wall_seconds']['median'] * 1e3:.1f} ms  miss "
            f"{paths['miss']['wall_seconds']['median'] * 1e3:.1f} ms  hit "
            f"{paths['hit']['wall_seconds']['median'] * 1e3:.2f} ms  "
            f"identical={identical}",
            file=sys.stderr,
        )
    return rows


def measure_key(engine: Any, path: Path, calls: int) -> dict[str, Any]:
    prepared = engine.preprocessor.prepare(path)
    samples = [timed(lambda: engine._image_cache_key(prepared)) for _ in range(calls)]
    pixel_values = np.asarray(prepared.pixel_values)
    mask = np.asarray(prepared.pixel_attention_mask)
    pixel_bytes = int(np.ascontiguousarray(pixel_values, dtype=np.float32).nbytes)
    mask_bytes = int(np.ascontiguousarray(mask).nbytes)
    hashed_bytes = pixel_bytes + mask_bytes
    median = statistics.median(samples)
    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "num_frames": prepared.num_frames,
        "rows": prepared.rows,
        "cols": prepared.cols,
        "pixel_shape": list(pixel_values.shape),
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        # Split, because a throughput quoted over only one of them is not the
        # throughput of the operation: the mask is int64 here and is two fifths
        # of what gets hashed.
        "pixel_bytes": pixel_bytes,
        "mask_bytes": mask_bytes,
        "hashed_bytes": hashed_bytes,
        "seconds": summarise(samples),
        "throughput_gib_per_second": (
            hashed_bytes / median / (1 << 30) if median else None
        ),
    }


def measure_weight_hash(weights: Weights) -> dict[str, Any]:
    """One sha256 pass over every tensor — the cost of the rejected alternative.

    Reads through the existing mmap rather than reopening files, so this prices
    what an identity check that actually verified contents would have to do.
    """
    total = 0
    digest = hashlib.sha256()

    def sweep() -> None:
        nonlocal total
        for name in weights.tensor_names:
            # One pass, not two: the byte count is accumulated while the pages are
            # already resident, because a second pass would be a cache-warm
            # re-read and would understate what a cold identity check costs.
            block = np.ascontiguousarray(np.asarray(weights.get(name)))
            total += int(block.nbytes)
            digest.update(block.tobytes())

    seconds = timed(sweep)
    return {
        "seconds": seconds,
        "tensor_bytes": total,
        "throughput_gib_per_second": total / seconds / (1 << 30) if seconds else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe.

    ``argv`` is a parameter rather than a module-level ``ARGS``, so importing this
    file has no side effects: ``tests/test_perf_tooling.py`` loads every tool in
    ``tools/perf`` by path, and a module that parsed ``sys.argv`` at import time
    would consume pytest's own arguments and fail to load at all.
    """
    args = parse_args(argv)
    # Before the model load; see out_guard.preflight.
    out_guard.preflight(args.out)
    started = time.perf_counter()
    model_dir = Path(args.model)
    engine = load_engine(
        model_dir,
        max_context=args.max_context,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    weights = engine.model.weights
    machine_before = machine_state()

    print(
        f"identity_cost: model={model_dir.name} calls={args.calls} "
        f"numpy={np.__version__} blas={blas_info().get('name')} "
        f"entries={len(weights.tensor_names)}",
        file=sys.stderr,
    )

    fingerprint = measure_fingerprint(weights, args.calls, args.warmup)
    median_ms = fingerprint["seconds"]["median"] * 1e3
    print(
        f"  fingerprint  median {median_ms:.4f} ms  "
        f"min {fingerprint['seconds']['min'] * 1e3:.4f} ms  "
        f"max {fingerprint['seconds']['max'] * 1e3:.4f} ms  "
        f"over {len(weights.tensor_names)} manifest entries",
        file=sys.stderr,
    )

    keys: list[dict[str, Any]] = []
    images = [Path(p) for p in args.image]
    if images and not hasattr(engine, "_image_cache_key"):
        print(
            f"  warning: {model_dir.name}'s engine has no image cache; "
            "--image ignored",
            file=sys.stderr,
        )
        images = []

    identity: dict[str, Any] | None = None
    if hasattr(engine, "_model_identity_changed"):
        identity = measure_identity(engine, args.calls, args.warmup)
        tuple_ms = identity["identity_tuple"]["seconds"]["median"] * 1e3
        check_ms = identity["identity_check"]["seconds"]["median"] * 1e3
        print(
            f"  identity  tuple median {tuple_ms:.4f} ms  check median "
            f"{check_ms:.4f} ms  ({check_ms / median_ms:.2f}x the bare fingerprint"
            f", changed={identity['changed_count']}/{args.calls})",
            file=sys.stderr,
        )
        assert identity["changed_count"] == 0, (
            "the identity check reported a change on an unmoved model, so every "
            "timing below describes a cache that never hits"
        )

    for image in images:
        row = measure_key(engine, image, args.key_calls)
        keys.append(row)
        ratio = row["seconds"]["median"] / fingerprint["seconds"]["median"]
        print(
            f"  key {image.name}  median {row['seconds']['median'] * 1e3:.2f} ms  "
            f"({row['hashed_bytes'] / (1 << 20):.1f} MiB hashed, "
            f"{ratio:.0f}x the manifest fingerprint)",
            file=sys.stderr,
        )

    # Sampled here, before the deliberately perturbing steps below: a 3.7 s vision
    # forward, ten of them per image in cache_paths, and a 1.5 s sweep of every
    # tensor all move the machine, and none of them may be allowed to stand in for
    # the state the timings above saw.
    machine_after = machine_state()

    encode: dict[str, Any] | None = None
    if images and hasattr(engine.model, "encode_image"):
        prepared = engine.preprocessor.prepare(images[0])
        output = engine.model.encode_image(prepared)
        encode = {
            "image": str(images[0]),
            "seconds": output.encode_seconds,
            "features_shape": list(np.asarray(output.features).shape),
        }
        print(
            f"  encode_image {encode['seconds'] * 1e3:.1f} ms — the forward the "
            "cache exists to skip",
            file=sys.stderr,
        )

    cache_paths: list[dict[str, Any]] = []
    if images and hasattr(engine, "_vision_features"):
        cache_paths = measure_cache_paths(
            engine,
            model_dir,
            images,
            calls=args.path_calls,
            max_context=args.max_context,
            prefill_chunk_size=args.prefill_chunk_size,
        )

    weight_hash = None if args.skip_weight_hash else measure_weight_hash(weights)
    if weight_hash is not None:
        print(
            f"  weight sha256  {weight_hash['seconds'] * 1e3:.1f} ms over "
            f"{weight_hash['tensor_bytes'] / (1 << 20):.1f} MiB — "
            f"{weight_hash['seconds'] / fingerprint['seconds']['median']:.0f}x "
            "the manifest fingerprint",
            file=sys.stderr,
        )

    vlm_py = ROOT / "src" / "slowq" / "vlm.py"
    weights_py = ROOT / "src" / "slowq" / "weights.py"

    # Every ratio the prose about this check quotes, computed here rather than by
    # hand: a hand-divided percentage in a docstring is the kind of number that
    # survives the measurement it was derived from going stale.
    fingerprint_median = fingerprint["seconds"]["median"]
    check_median = (
        identity["identity_check"]["seconds"]["median"] if identity else None
    )
    encode_seconds = encode["seconds"] if encode else None
    derived: dict[str, Any] = {
        "weight_hash_to_fingerprint": (
            weight_hash["seconds"] / fingerprint_median if weight_hash else None
        ),
        "weight_hash_share_of_encode": (
            weight_hash["seconds"] / encode_seconds
            if weight_hash and encode_seconds
            else None
        ),
        "key_to_fingerprint": [
            row["seconds"]["median"] / fingerprint_median for row in keys
        ],
        "key_to_identity_check": (
            [row["seconds"]["median"] / check_median for row in keys]
            if check_median
            else None
        ),
        "key_share_of_encode": [
            row["seconds"]["median"] / encode_seconds for row in keys
        ]
        if encode_seconds
        else None,
        # What the widening of the identity check from "the manifest" to "the
        # manifest and the config" costs, as a multiple of the bare fingerprint
        # read it replaced. Quoted so a reader can see whether it is free or
        # merely small; both are acceptable, and only one is honest to call free.
        "identity_check_to_fingerprint": (
            check_median / fingerprint_median if check_median else None
        ),
        # The identity check as a share of what a cache *hit* costs end to end. A
        # hit is identity check + pixel hash + copy out, so this is the share of the
        # cheapest path the check guards — the worst case for it, and the one that
        # decides whether running it per request is defensible.
        "identity_share_of_hit": [
            check_median / row["paths"]["hit"]["wall_seconds"]["median"]
            for row in cache_paths
        ]
        if check_median
        else None,
        "hit_to_miss": [
            row["paths"]["hit"]["wall_seconds"]["median"]
            / row["paths"]["miss"]["wall_seconds"]["median"]
            for row in cache_paths
        ],
        "miss_to_disabled": [
            row["paths"]["miss"]["wall_seconds"]["median"]
            / row["paths"]["disabled"]["wall_seconds"]["median"]
            for row in cache_paths
        ],
    }
    for index, row in enumerate(keys):
        shares = derived["key_share_of_encode"]
        print(
            f"  ratio {row['path'].split('/')[-1]}: key is "
            f"{derived['key_to_fingerprint'][index]:.0f}x the manifest fingerprint"
            + (
                f", {shares[index] * 100:.2f}% of the vision encode"
                if shares
                else ""
            ),
            file=sys.stderr,
        )
    if derived["identity_check_to_fingerprint"]:
        print(
            f"  ratio identity check: "
            f"{derived['identity_check_to_fingerprint']:.2f}x the manifest "
            f"fingerprint it wraps",
            file=sys.stderr,
        )
    for index, row in enumerate(cache_paths):
        shares = derived["identity_share_of_hit"]
        print(
            f"  ratio {Path(row['path']).name}: hit is "
            f"{derived['hit_to_miss'][index] * 100:.2f}% of a miss, miss is "
            f"{derived['miss_to_disabled'][index]:.2f}x cache-disabled"
            + (
                f", identity check is {shares[index] * 100:.2f}% of the hit"
                if shares
                else ""
            ),
            file=sys.stderr,
        )
    if derived["weight_hash_to_fingerprint"]:
        share = derived["weight_hash_share_of_encode"]
        print(
            f"  ratio weight sha256: "
            f"{derived['weight_hash_to_fingerprint']:.0f}x the manifest fingerprint"
            + (f", {share * 100:.1f}% of the vision encode" if share else ""),
            file=sys.stderr,
        )

    # Bit-equality across the three cache paths is a pass/fail property, not a
    # recorded curiosity: an artifact whose timings came from a cache that changed
    # the answer would be worse than no artifact, because it would be citable.
    diverged = [row["path"] for row in cache_paths if not row["bit_identical"]]
    if diverged:
        print(f"  FAILED: cache paths returned different features: {diverged}", file=sys.stderr)

    payload = {
        "model": str(model_dir),
        "task": recorded_task(model_dir),
        "engine_class": type(engine).__name__,
        "slowq_module": slowq.__file__,
        "config": {
            "calls": args.calls,
            "warmup": args.warmup,
            "key_calls": args.key_calls,
            "path_calls": args.path_calls,
            "max_context": args.max_context,
            "prefill_chunk_size": args.prefill_chunk_size,
        },
        "fingerprint": fingerprint,
        # Not "identity": that key already holds the source hashes below, and
        # tests/test_perf_tooling.py reads data["identity"]["probe_code_sha256"].
        "identity_layers": identity,
        "keys": keys,
        "encode_image": encode,
        "cache_paths": cache_paths,
        "weight_hash": weight_hash,
        "derived": derived,
        "machine": {"before": machine_before, "after": machine_after},
        "environment": environment_info(),
        "blas": blas_info(),
        "rss_after_load_bytes": peak_rss_bytes(),
        # Two hashes per file, not one: the raw one is the audit trail, and the
        # docstring-insensitive one is what survives writing these numbers into
        # vlm.py's own prose. See code_sha256.
        "identity": {
            "vlm_py_sha256": sha256_file(vlm_py),
            "weights_py_sha256": sha256_file(weights_py),
            "probe_sha256": sha256_file(Path(__file__).resolve()),
            "vlm_py_code_sha256": code_sha256(vlm_py),
            "weights_py_code_sha256": code_sha256(weights_py),
            "probe_code_sha256": code_sha256(Path(__file__).resolve()),
        },
        "source_identity": source_identity.record(ROOT, Path(__file__)),
        "wall_seconds": time.perf_counter() - started,
    }
    if args.out:
        # Refuses an existing path rather than truncating it; see out_guard.py.
        out = out_guard.write_json(args.out, payload)
        print(f"wrote {out_guard.display(out)}", file=sys.stderr)
    return 1 if diverged else 0


if __name__ == "__main__":
    sys.exit(main())
