#!/usr/bin/env python3
"""A/B harness for the image-feature cache (round-1 optimisation 4).

    .venv/bin/python tools/perf/vision_cache_ab.py \
        --src . --model models/SmolVLM-256M-Instruct-slowq \
        --image artifacts/validation/SmolVLM-256M-Instruct-slowq/vision-grid-1x4.png \
        --out "$(.venv/bin/python tools/perf/out_guard.py fresh-root)/vision/opt.json"

``--out`` must name a path that does not exist yet: the harness refuses to replace a
measurement that is already on disk, and ``artifacts/perf-round1{,-rerun}`` are
refused outright as published evidence. See ``tools/perf/out_guard.py``.

It runs unchanged against the baseline tree: where the working tree has
``image_cache_bytes``, ``engine.image_features`` and ``metrics.vision_cache_hit``,
the baseline has none of them, so the "cache off" path is the only one that can
run there and the cache-on paths report themselves unsupported. Comparing the
*same* path across the two JSONs is the before/after; comparing paths inside one
working-tree JSON is the hit/miss split.

Paths
-----
off-no-cache          ``image_cache_bytes=0`` — and on the baseline tree the only
                      state there is. Three requests over one image: a first
                      question, the same question again, and a different one. All
                      three run the tower, and they are the reference every
                      cache-on step is checked against.
on-same-question      the default cache, one question asked twice. Step 0 misses,
                      step 1 hits.
on-second-question    the default cache, two *different* questions about one
                      image — the case the cache exists for. The prompt, the
                      tokenisation and the answer all differ; only the pixels are
                      shared.
on-other-image        the default cache, two different images. Both must miss, or
                      the key is not discriminating.

Every step's token ids are compared against the ``off`` path's for the same
image and question, so a speedup that came from answering something else cannot
pass unnoticed.

One process, strictly sequential: these runs contend for the same cores and the
same memory bandwidth the vision tower needs.
"""

from __future__ import annotations

import os
import sys
import threading

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def requested_threads() -> int | None:
    for index, item in enumerate(sys.argv):
        if item == "--threads":
            value = sys.argv[index + 1] if index + 1 < len(sys.argv) else None
        elif item.startswith("--threads="):
            value = item.split("=", 1)[1]
        else:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


REQUESTED = requested_threads()
if REQUESTED is not None:
    for _name in THREAD_ENV_VARS:
        os.environ[_name] = str(REQUESTED)
BEFORE_NUMPY = "numpy" not in sys.modules

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Sequence  # noqa: E402


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="repo root whose src/ to import")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--image-other",
        default=None,
        help="a second, different image for the key-discrimination path",
    )
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=5, help="timed rounds per path")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--cache-bytes",
        type=int,
        default=128 * 1024 * 1024,
        help="budget for the cache-on paths; 0 makes them report unsupported",
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


ARGS = parse_args(sys.argv[1:])
SRC = str(Path(ARGS.src).resolve() / "src")
TREE = Path(ARGS.src).resolve()

# Imported before the measured tree goes on sys.path, so the identity recorded is
# unambiguously this harness's own tool and never a same-named module from the
# tree being timed.
import out_guard  # noqa: E402
import source_identity  # noqa: E402

sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

import slowq  # noqa: E402
from slowq.metrics import blas_info, environment_info, peak_rss_bytes  # noqa: E402
from slowq.sampling import SamplingParams  # noqa: E402
from slowq.vlm import VisionEngine  # noqa: E402

# Fail loudly rather than silently measuring the wrong tree (the dev venv has an
# editable install pointing at the main checkout).
assert slowq.__file__.startswith(SRC), f"imported {slowq.__file__}, expected under {SRC}"

#: Greedy, so two runs of one prompt have to produce the same token ids and any
#: difference is a bug rather than a sample.
SAMPLING = SamplingParams(temperature=0.0, top_k=0, top_p=1.0)

QUESTION_A = "Describe what you see in this picture."
QUESTION_B = "How many separate panels does this image have?"

#: ``path name -> (cache budget, [(image slot, question), ...])``. The image slot
#: is a key into the ``images`` dict built in ``main``. A budget of ``None`` means
#: "the engine's own default", which only exists on a tree that has the cache.
PATHS: dict[str, tuple[int | None, tuple[tuple[str, str], ...]]] = {
    # Every request runs the tower. Step 1 repeats the question (what a loop over
    # ``slowq vlm`` would do), step 2 asks a different one, so this path is the
    # reference both cache-on paths are compared against.
    "off-no-cache": (
        0,
        (("main", QUESTION_A), ("main", QUESTION_A), ("main", QUESTION_B)),
    ),
    "on-same-question": (None, (("main", QUESTION_A), ("main", QUESTION_A))),
    # The case the cache exists for: the prompt, the tokenisation and the answer
    # all differ, and only the pixels are shared.
    "on-second-question": (None, (("main", QUESTION_A), ("main", QUESTION_B))),
    "on-other-image": (None, (("main", QUESTION_A), ("other", QUESTION_A))),
}

PER_STEP_KEYS = (
    "wall_seconds",
    "vision_encode_seconds",
    "first_token_seconds",
    "prefill_seconds",
    "decode_seconds",
    "total_seconds",
    "tokenize_seconds",
    "peak_rss_bytes",
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
)


def summarise(values: Sequence[Any]) -> dict[str, float]:
    """median/min/max/stdev over the finite entries, NaN dropped not propagated."""
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return {"median": float("nan")}
    return {
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
        "stdev": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "samples": finite,
    }


def cache_supported() -> bool:
    """Whether this tree's ``VisionEngine`` can be given a feature-cache budget."""
    return "image_cache_bytes" in inspect.signature(
        VisionEngine.from_directory
    ).parameters


def load_engine(cache_bytes: int | None) -> Any:
    """A ``VisionEngine``, with ``cache_bytes`` only if this tree accepts it.

    ``None`` means "leave the engine's own default alone", which on the baseline
    tree is also the only option: its ``from_directory`` has no such parameter.
    """
    kwargs: dict[str, Any] = {
        "max_context": ARGS.max_context,
        "prefill_chunk_size": ARGS.prefill_chunk_size,
    }
    if cache_bytes is not None and cache_supported():
        kwargs["image_cache_bytes"] = cache_bytes
    return VisionEngine.from_directory(ARGS.model, **kwargs)


def cache_of(engine: Any) -> Any:
    return getattr(engine, "image_features", None)


def hit_of(result: Any) -> bool:
    return bool(getattr(result.metrics, "vision_cache_hit", False))


def run_step(engine: Any, image_path: Path, question: str) -> dict[str, Any]:
    """One ``generate_image`` call, timed from outside the engine.

    The wall clock is taken around the whole call, not read from
    ``metrics.total_seconds``: that one starts inside ``_stream`` and so excludes
    reading the file, resizing every crop and rendering the template — which is
    exactly the part that stays the same whether or not the tower runs.
    """
    started = time.perf_counter()
    result = engine.generate_image(
        image_path, question, sampling=SAMPLING, max_new_tokens=ARGS.max_new_tokens
    )
    wall = time.perf_counter() - started
    row = result.metrics.as_dict()
    row["wall_seconds"] = wall
    row["vision_cache_hit"] = hit_of(result)
    row["token_ids"] = list(result.token_ids)
    row["text_sha"] = hashlib.sha256(result.text.encode("utf-8")).hexdigest()[:16]
    row["text"] = result.text
    return row


def run_path(name: str, engine: Any, steps: tuple[tuple[str, str], ...],
             images: dict[str, Path]) -> dict[str, Any]:
    """Run one path on ``engine``: ``warmup + repeat`` rounds of its steps.

    Paths that share a cache budget share an engine, which is both how a real
    caller uses one and why the counters below are reported as deltas: they
    survive across paths, and a cumulative total would not describe this one.

    The cache is cleared between rounds so every round starts from a genuine
    miss. Leaving it warm would let step 0 of round 2 be served from round 1 — a
    hit rate no user sees when they ask about a picture for the first time.
    """
    cache = cache_of(engine)
    counted = cache.stats() if cache is not None else None
    out: dict[str, Any] = {
        "engine_cache_bytes": cache.max_bytes if cache is not None else 0,
        "round_order": ["cold"],
        "warmup_rounds": ARGS.warmup,
        "timed_rounds": ARGS.repeat,
    }

    def one_round() -> list[dict[str, Any]]:
        if cache is not None:
            cache.clear()
        return [run_step(engine, images[slot], question) for slot, question in steps]

    first_round = one_round()
    for _ in range(ARGS.warmup):
        one_round()
    rounds = [one_round() for _ in range(ARGS.repeat)]

    out["first_round"] = [
        {key: row[key] for key in (*PER_STEP_KEYS, "vision_cache_hit", "prompt_tokens",
                                    "generated_tokens", "stop_reason", "text_sha")}
        for row in first_round
    ]
    out["steps"] = {}
    for index in range(len(steps)):
        rows = [rnd[index] for rnd in rounds]
        entry: dict[str, Any] = {
            "question": steps[index][1],
            "image": steps[index][0],
            "prompt_tokens": rows[0]["prompt_tokens"],
            "generated_tokens": rows[0]["generated_tokens"],
            "stop_reason": rows[0]["stop_reason"],
            "vision_cache_hit": rows[0]["vision_cache_hit"],
            "vision_cache_hits_observed": sorted({r["vision_cache_hit"] for r in rows}),
            "text_sha": rows[0]["text_sha"],
            "identical_outputs": len({r["text_sha"] for r in rows}) == 1,
            "token_ids": rows[0]["token_ids"],
            "text": rows[0]["text"],
        }
        for key in PER_STEP_KEYS:
            entry[key] = summarise([row.get(key, float("nan")) for row in rows])
        out["steps"][str(index)] = entry

    # A step can only hit if an earlier step of the same round put the same image
    # in, so the counts follow from the step list rather than from what the
    # engine chose to report. Recorded next to the observed ones so a lying
    # ``vision_cache_hit`` is visible in the JSON. A disabled cache never looks
    # anything up, so its expectation is zero of both.
    rounds_run = 1 + ARGS.warmup + ARGS.repeat
    distinct_images = len({slot for slot, _ in steps})
    cache_on = cache is not None and cache.max_bytes > 0
    out["expected_cache_traffic"] = {
        "cache_enabled": cache_on,
        "rounds": rounds_run,
        "misses_per_round": distinct_images if cache_on else 0,
        "hits_per_round": (len(steps) - distinct_images) if cache_on else 0,
    }
    if cache is not None and counted is not None:
        after = cache.stats()
        observed = {
            "hits": after["hits"] - counted["hits"],
            "misses": after["misses"] - counted["misses"],
            "evictions": after["evictions"] - counted["evictions"],
        }
        expected = {
            "hits": out["expected_cache_traffic"]["hits_per_round"] * rounds_run,
            "misses": out["expected_cache_traffic"]["misses_per_round"] * rounds_run,
        }
        out["cache_traffic"] = {
            **observed,
            "entries": after["entries"],
            "bytes": after["bytes"],
            "max_bytes": after["max_bytes"],
            "matches_expectation": all(observed[k] == expected[k] for k in expected),
        }
        if not out["cache_traffic"]["matches_expectation"]:
            print(
                f"  {name}: cache traffic {observed} != expected {expected}",
                file=sys.stderr,
            )
    out["rss_final_bytes"] = peak_rss_bytes()
    print(
        f"  {name:<20} "
        + "  ".join(
            f"step{i} {'HIT ' if s['vision_cache_hit'] else 'miss'} "
            f"vision {s['vision_encode_seconds']['median'] * 1e3:>8.1f} ms "
            f"ttft {s['first_token_seconds']['median'] * 1e3:>8.1f} ms "
            f"wall {s['wall_seconds']['median'] * 1e3:>8.1f} ms"
            for i, s in out["steps"].items()
        ),
        file=sys.stderr,
    )
    return out


def image_facts(engine: Any, path: Path) -> dict[str, Any]:
    """What the preprocessor makes of one file: frames, crop grid, token count.

    Also the two costs the cache adds to a request — hashing the pixels into a
    key, and copying the features in and out — measured against the tower run
    they replace. Those three numbers are the whole argument for the cache, so
    they are recorded per image rather than summarised once.
    """
    prepared = engine.preprocessor.prepare(path)
    output = engine.model.encode_image(prepared)
    features = output.features
    facts: dict[str, Any] = {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "original_size": list(prepared.original_size),
        "resized_size": list(prepared.resized_size),
        "encoder_size": list(prepared.encoder_size),
        "rows": prepared.rows,
        "cols": prepared.cols,
        "num_frames": prepared.num_frames,
        "num_image_tokens": prepared.num_image_tokens,
        "pixel_bytes": int(np.asarray(prepared.pixel_values).nbytes),
        "features_shape": list(features.shape),
        "features_bytes": int(features.nbytes),
        "encoder_output_bytes": int(np.asarray(output.encoder_output).nbytes),
        "vision_encode_seconds": output.encode_seconds,
    }
    if hasattr(engine, "_image_cache_key"):
        key_seconds = []
        for _ in range(5):
            started = time.perf_counter()
            key = engine._image_cache_key(prepared)
            key_seconds.append(time.perf_counter() - started)
        facts["key_seconds"] = summarise(key_seconds)
        # The property the cache actually needs: the key works as a dict key.
        facts["key_usable_as_dict_key"] = bool({key: True}[key])
        facts["key_cost_share_of_encode"] = (
            facts["key_seconds"]["median"] / output.encode_seconds
        )
        copy_seconds = []
        for _ in range(5):
            started = time.perf_counter()
            np.array(features, dtype=np.float32, copy=True)
            copy_seconds.append(time.perf_counter() - started)
        facts["feature_copy_seconds"] = summarise(copy_seconds)
    return facts


def main() -> int:
    # Before the model load; see out_guard.preflight and core_ab.main.
    out_guard.preflight(ARGS.out)
    started = time.perf_counter()
    images: dict[str, Path] = {"main": Path(ARGS.image)}
    if ARGS.image_other:
        images["other"] = Path(ARGS.image_other)
    else:
        # The key-discrimination path needs two images; without a second one it
        # would compare an image with itself and prove nothing.
        PATHS.pop("on-other-image", None)

    probe_budget = ARGS.cache_bytes if cache_supported() else 0
    probe = load_engine(probe_budget)
    load_seconds = time.perf_counter() - started
    print(
        f"vision_cache_ab: src={SRC}\n         model={ARGS.model} threads={REQUESTED} "
        f"numpy={np.__version__} blas={blas_info().get('name')} "
        f"cache_supported={cache_supported()} load={load_seconds:.3f}s "
        f"active_threads={threading.active_count()}",
        file=sys.stderr,
    )

    payload: dict[str, Any] = {
        "label": ARGS.label,
        "src": SRC,
        "slowq_module": slowq.__file__,
        # Hashes of the tree that was imported plus of this harness, so the
        # numbers can be tied back to code rather than to a date. The optimised
        # side of this round is uncommitted, which is exactly the case where a
        # revision name does not exist and a hash is the only identity there is.
        "source_identity": source_identity.record(TREE, Path(__file__)),
        "model": ARGS.model,
        "args": {k: str(v) for k, v in vars(ARGS).items()},
        "threads": {
            "requested": REQUESTED,
            "requested_before_numpy_import": bool(BEFORE_NUMPY),
            "exported_env": {n: os.environ.get(n) for n in THREAD_ENV_VARS},
            "detected_backend": blas_info(),
            "logical_cpus": os.cpu_count(),
        },
        "environment": environment_info(),
        "sampling": SAMPLING.describe() if hasattr(SAMPLING, "describe") else {},
        "load_seconds": load_seconds,
        "rss_after_load_bytes": peak_rss_bytes(),
        "cache_supported": cache_supported(),
        "images": {slot: image_facts(probe, path) for slot, path in images.items()},
    }

    # One engine per cache budget, not per path: loading five engines to run five
    # paths would turn peak RSS into a measure of the harness rather than of the
    # model. The probe engine is the cache-on one, so nothing is loaded twice.
    engines: dict[int, Any] = {probe_budget: probe}
    payload["paths"] = {}
    for name, (budget, steps) in PATHS.items():
        print(f"[{name}]", file=sys.stderr)
        if budget is None and not cache_supported():
            payload["paths"][name] = {
                "unsupported": "this tree has no image_cache_bytes parameter"
            }
            continue
        if budget is None and ARGS.cache_bytes == 0:
            payload["paths"][name] = {"unsupported": "--cache-bytes 0 was requested"}
            continue
        wanted = ARGS.cache_bytes if budget is None else budget
        if wanted not in engines:
            engines[wanted] = load_engine(wanted)
        payload["paths"][name] = run_path(name, engines[wanted], steps, images)

    # Cross-path agreement: the same image and question must decode to the same
    # tokens whether the tower ran or not. The reference is the cache-off path,
    # which always does the full work, and it covers both questions.
    reference = payload["paths"].get("off-no-cache", {}).get("steps", {})
    baseline = {
        (step["image"], step["question"]): tuple(step["token_ids"])
        for step in reference.values()
    }
    mismatches = []
    compared = 0
    for path_name, path in payload["paths"].items():
        for step in path.get("steps", {}).values():
            expected = baseline.get((step["image"], step["question"]))
            if expected is None:
                continue
            compared += 1
            if tuple(step["token_ids"]) != expected:
                mismatches.append(
                    {
                        "path": path_name,
                        "image": step["image"],
                        "question": step["question"],
                        "vision_cache_hit": step["vision_cache_hit"],
                    }
                )
    payload["agreement"] = {
        "reference_path": "off-no-cache",
        "steps_compared": compared,
        "mismatched_steps": mismatches,
        "all_steps_identical": compared > 0 and not mismatches,
    }
    print(
        f"  agreement: {compared} steps compared against the cache-off path, "
        f"{len(mismatches)} mismatch(es)",
        file=sys.stderr,
    )

    # The headline: what a *repeat* request costs, with and without the cache.
    # Each pair is one request against one request, at the same step of the same
    # conversation shape, so nothing is being compared across workloads.
    headline: dict[str, Any] = {}
    for label, cached_path, cached_step, uncached_step in (
        ("same_question_again", "on-same-question", "1", "1"),
        ("a_different_question", "on-second-question", "1", "2"),
    ):
        without = reference.get(uncached_step)
        with_cache = (
            payload["paths"].get(cached_path, {}).get("steps", {}).get(cached_step)
        )
        if not without or not with_cache:
            continue
        entry: dict[str, Any] = {}
        for side, step in (("without_cache", without), ("with_cache", with_cache)):
            entry[side] = {
                "path": cached_path if side == "with_cache" else "off-no-cache",
                "vision_cache_hit": step["vision_cache_hit"],
                **{
                    key: step[key]["median"]
                    for key in ("wall_seconds", "vision_encode_seconds",
                                "first_token_seconds", "prefill_seconds",
                                "decode_seconds", "total_seconds", "peak_rss_bytes")
                },
            }
        entry["speedup"] = {
            key: (
                entry["without_cache"][key] / entry["with_cache"][key]
                if entry["with_cache"][key]
                else float("nan")
            )
            for key in ("wall_seconds", "vision_encode_seconds", "first_token_seconds")
        }
        entry["token_ids_identical"] = (
            without["token_ids"] == with_cache["token_ids"]
            if label == "same_question_again"
            else None  # different questions, so different answers are expected
        )
        headline[label] = entry
        print(
            f"  {label}: "
            + "  ".join(
                f"{key} {entry['without_cache'][key] * 1e3:.1f} -> "
                f"{entry['with_cache'][key] * 1e3:.1f} ms "
                f"({entry['speedup'][key]:.2f}x)"
                for key in ("vision_encode_seconds", "first_token_seconds", "wall_seconds")
            ),
            file=sys.stderr,
        )
    payload["headline"] = headline

    payload["rss_final_bytes"] = peak_rss_bytes()
    payload["wall_seconds"] = time.perf_counter() - started

    if ARGS.out:
        # Refuses an existing path rather than truncating it; see out_guard.py.
        out = out_guard.write_json(ARGS.out, payload)
        print(f"wrote {out_guard.display(out)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
