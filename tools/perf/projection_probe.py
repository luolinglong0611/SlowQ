#!/usr/bin/env python3
"""Measure what one vocabulary projection costs, to price optimisation 1.

    .venv/bin/python tools/perf/projection_probe.py \
        --model models/Qwen3-0.6B-slowq \
        --out "$(.venv/bin/python tools/perf/out_guard.py fresh-root)/projection/Qwen3-0.6B-slowq.json"

``--out`` must name a path that does not exist yet. This probe is the least
repeatable measurement in the round — its answer is a difference between two ~540 ms
medians — so replacing a previous attempt destroys the only evidence that it moved.
See ``tools/perf/out_guard.py``.

``CausalLMModel.prefill`` now runs its non-final chunks through ``encode`` and
only the final one through ``forward``, so a 512-token prefill at
``chunk_size=128`` does one output projection instead of four. The A/B tables
show the projection *count* dropping 4 → 1 (observed, not inferred), but they
cannot show what that count was worth: the two trees also differ in ``sigmoid``
and in the single-token mask, so the end-to-end delta mixes all three.

This script isolates the one term. ``forward`` is literally ``encode`` plus
``output_projection(hidden[-1])`` (model.py:390-398), so timing both over the
same ids on the same tree and subtracting gives the cost of a single projection
with everything else held fixed — same weights, same memory state, same process.

Two properties make the subtraction trustworthy here:

* ``output_projection`` is byte-identical between the baseline tree and this one
  (verified by diff), so a cost measured on this tree prices the projections the
  baseline tree used to make.
* the pair is interleaved — ``forward`` then ``encode``, repeated — and the
  reported cost is the median of the *paired* differences, not the difference of
  two independently collected medians. A slow drift over the run then lands on
  both halves of a pair instead of on one side of the comparison.

Run one process per model. The 6.4 GiB Qwen3-1.7B gets compressed out of memory
if anything else is resident, and a probe that measures milliseconds must not be
competing for bandwidth with a second benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import slowq  # noqa: E402
from slowq import Engine  # noqa: E402
from slowq.metrics import blas_info, environment_info, peak_rss_bytes  # noqa: E402
import out_guard  # noqa: E402
import source_identity  # noqa: E402

assert slowq.__file__.startswith(str(ROOT / "src")), f"imported {slowq.__file__}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--lengths", default="128,256,512")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    # Defaults to the *out root's* own ab/, so a re-collection prices optimisation 1
    # against that re-collection's prefill instead of silently against the published
    # round's. Mixing rounds here would divide a fresh projection cost by a stale
    # prefill and call the quotient a share.
    parser.add_argument("--ab-dir", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


ARGS = parse_args(sys.argv[1:])


def resolve_ab_dir() -> Path:
    """Where to read ``engine_prefill`` from, given --ab-dir / --out."""
    if ARGS.ab_dir:
        path = Path(ARGS.ab_dir)
        return path if path.is_absolute() else ROOT / path
    if ARGS.out:
        out = Path(ARGS.out)
        root = out if out.is_absolute() else ROOT / out
        # .../<out-root>/projection/<model>.json -> <out-root>/ab
        return root.parent.parent / "ab"
    return ROOT / "artifacts" / "perf-round1" / "ab"


def probe_token_ids(length: int, vocab_size: int) -> list[int]:
    """Deterministic, in-range ids. Same rule as core_ab.py and tools/benchmark.py."""
    base = min(1000, max(vocab_size - 2, 1))
    span = max(vocab_size - base - 1, 1)
    return [base + (index % span) for index in range(length)]


def timed(fn: Callable[[], Any]) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def measure(model: Any, length: int, repeat: int, warmup: int) -> dict[str, Any]:
    """Paired forward/encode timings over ``length`` deterministic ids."""
    vocab = model.config.vocab_size
    ids = probe_token_ids(length, vocab)

    def run_forward() -> None:
        cache = model.new_cache(length + 1)
        model.forward(ids, cache)

    def run_encode() -> None:
        cache = model.new_cache(length + 1)
        model.encode(ids, cache)

    for _ in range(warmup):
        run_forward()
        run_encode()

    forwards: list[float] = []
    encodes: list[float] = []
    paired: list[float] = []
    for _ in range(repeat):
        # One pair back to back, difference taken inside the pair.
        forward_seconds = timed(run_forward)
        encode_seconds = timed(run_encode)
        forwards.append(forward_seconds)
        encodes.append(encode_seconds)
        paired.append(forward_seconds - encode_seconds)

    projection = statistics.median(paired)
    return {
        "prompt_tokens": length,
        "forward": {
            "median": statistics.median(forwards),
            "min": min(forwards),
            "max": max(forwards),
            "samples": forwards,
        },
        "encode": {
            "median": statistics.median(encodes),
            "min": min(encodes),
            "max": max(encodes),
            "samples": encodes,
        },
        "projection_seconds": {
            "median": projection,
            "min": min(paired),
            "max": max(paired),
            "samples": paired,
        },
        # Derived, so the reader does not have to trust a hand-computed ratio:
        # how many projections prefill() actually skips at this length, and what
        # they are worth against the measured engine prefill of the same length.
        "projections_skipped": -(-length // ARGS.prefill_chunk_size) - 1,
        "repeat": repeat,
        "warmup": warmup,
    }


def engine_prefill_median(ab_dir: Path, model_dir: str, length: int, side: str) -> float | None:
    """The measured engine prefill for this model/length/side, or None."""
    path = ab_dir / f"{model_dir}-{side}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    node = payload.get("micro", {}).get("engine_prefill", {}).get(str(length)) or {}
    value = node.get("median")
    return value if isinstance(value, (int, float)) else None


def main() -> int:
    # Before the model load; see out_guard.preflight.
    out_guard.preflight(ARGS.out)
    started = time.perf_counter()
    engine = Engine.from_directory(
        ARGS.model,
        max_context=ARGS.max_context,
        prefill_chunk_size=ARGS.prefill_chunk_size,
    )
    model = engine.model
    lengths = tuple(int(x) for x in ARGS.lengths.split(",") if x)
    ab_dir = resolve_ab_dir()
    model_dir = Path(ARGS.model).name

    print(
        f"projection_probe: model={model_dir} lengths={lengths} "
        f"chunk={ARGS.prefill_chunk_size} numpy={np.__version__} "
        f"blas={blas_info().get('name')} vocab={model.config.vocab_size} "
        f"hidden={model.config.hidden_size} ab_dir={ab_dir}",
        file=sys.stderr,
    )

    per_length: dict[str, Any] = {}
    negative_lengths: list[str] = []
    for length in lengths:
        row = measure(model, length, ARGS.repeat, ARGS.warmup)
        skipped = row["projections_skipped"]
        projection = row["projection_seconds"]["median"]
        row["skipped_seconds"] = projection * skipped
        # ``forward`` *is* ``encode`` plus one ``output_projection`` (model.py:390-398),
        # so the paired difference is real work and cannot be negative. A negative
        # median is therefore not a result, it is proof the timing was invalid — the
        # model got paged out between the two halves of the pair. Recording it per
        # row keeps the evidence (the review requires anomalies be marked, not
        # silently dropped) while making the number unusable downstream.
        negatives = sum(1 for s in row["projection_seconds"]["samples"] if s <= 0)
        row["invalid"] = projection <= 0
        row["negative_pairs"] = negatives
        if row["invalid"]:
            negative_lengths.append(str(length))
        for side in ("base", "opt"):
            row[f"engine_prefill_{side}"] = engine_prefill_median(ab_dir, model_dir, length, side)
        base_prefill = row["engine_prefill_base"]
        row["share_of_base_prefill"] = (
            row["skipped_seconds"] / base_prefill if base_prefill and not row["invalid"] else None
        )
        per_length[str(length)] = row
        share = row["share_of_base_prefill"]
        print(
            f"  {length:>4} tok  forward {row['forward']['median'] * 1e3:>8.2f} ms  "
            f"encode {row['encode']['median'] * 1e3:>8.2f} ms  "
            f"projection {projection * 1e3:>6.2f} ms  "
            f"x{skipped} skipped = {row['skipped_seconds'] * 1e3:>6.2f} ms"
            + (f"  = {share * 100:>5.2f}% of base engine prefill" if share else "")
            + (
                f"  **INVALID: {negatives}/{len(row['projection_seconds']['samples'])} "
                "pairs negative — model paged out mid-pair**"
                if row["invalid"]
                else ""
            ),
            file=sys.stderr,
        )

    # Three lengths price the same ``[1, hidden] × [hidden, vocab]`` matmul, so their
    # spread is this probe's error bar. Reported next to the numbers it undermines
    # rather than left for a reader to notice.
    valid_medians = [
        row["projection_seconds"]["median"]
        for row in per_length.values()
        if not row["invalid"] and row["projection_seconds"]["median"] > 0
    ]
    resolution = (
        max(valid_medians) / min(valid_medians) - 1 if len(valid_medians) > 1 else None
    )
    payload_validity = {
        "invalid_lengths": negative_lengths,
        "usable_lengths": [k for k, v in per_length.items() if not v["invalid"]],
        "across_length_spread": resolution,
        "verdict": (
            "unusable: forward − encode came out negative, so the model was paged out "
            "between the two halves of the pair"
            if len(negative_lengths) == len(lengths)
            else "partly unusable: some lengths have negative paired differences"
            if negative_lengths
            else "usable"
        ),
    }
    print(f"  validity: {payload_validity['verdict']}", file=sys.stderr)

    payload = {
        "model": ARGS.model,
        "slowq_module": slowq.__file__,
        # Same identity record the other harnesses write, so this probe's numbers
        # can be tied to the tree that produced them. Without it a re-collection
        # cannot tell its own projection costs from the published round's, and the
        # "new conclusions need new execution records" requirement is unverifiable
        # for the one probe that prices optimisation 1 directly.
        "source_identity": source_identity.record(ROOT, Path(__file__)),
        "args": vars(ARGS),
        # Recorded because the share below is a ratio across two different files:
        # this probe's projection cost over ab/'s engine prefill. If the two came
        # from different rounds the ratio is meaningless, and this is the field that
        # makes the question checkable.
        "ab_dir": str(ab_dir),
        "validity": payload_validity,
        "config": {
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "prefill_chunk_size": ARGS.prefill_chunk_size,
            "max_context": ARGS.max_context,
            "repeat": ARGS.repeat,
            "warmup": ARGS.warmup,
            "lengths": list(lengths),
        },
        "projection_weight_bytes": int(model.config.vocab_size)
        * int(model.config.hidden_size)
        * 4,
        "environment": environment_info(),
        "blas": blas_info(),
        "rss_after_load_bytes": peak_rss_bytes(),
        "per_length": per_length,
        "wall_seconds": time.perf_counter() - started,
    }
    if ARGS.out:
        # Refuses an existing path rather than truncating it; see out_guard.py.
        out = out_guard.write_json(ARGS.out, payload)
        print(f"wrote {out_guard.display(out)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
