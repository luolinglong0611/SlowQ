#!/usr/bin/env python3
"""Measure what SlowQ actually costs, and record the machine it cost that on.

    python tools/benchmark.py --model models/Qwen3-0.6B-slowq
    python tools/benchmark.py --model ... --threads 4 --repeat 5 --json out.json
    python tools/benchmark.py --model ... --prompt "写一首关于慢炖的五言绝句"

There are no performance targets in this project and this tool does not pretend
to hit any. It reports three things and nothing else:

* **micro** — prefill and decode measured on their own, so a slowdown can be
  attributed to one of them. Prefill is measured twice per length: once as a
  single full forward, and once through the chunk loop the engine really runs.
  The two are different code paths and quoting one for the other is how a
  report ends up describing a run that never happened.
* **end to end** — real chat workloads through ``Engine.generate``, reported as
  the same intervals the CLI prints (see docs/architecture.md, "Timing
  intervals" for what each one includes and excludes).
* **environment** — CPU, Python, NumPy, BLAS and the memory numbers, because a
  tok/s figure without them is not a measurement. The ``--threads`` pin is
  reported next to them as a *setting*, kept apart from the detected backend:
  how many threads the BLAS really ran with is not observable from here.

Cold versus warm matters here and is reported separately: weights are mmap'ed,
so ``load_seconds`` covers mapping the files, while the *first* forward pays for
paging them in. Quoting a warm number for a cold run would overstate throughput
by several times.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Filled in by ``apply_thread_limit``; reported as a *setting*, never as a
#: measurement, because the effective BLAS thread count is not observable here.
REQUESTED_THREADS: int | None = None


def requested_threads(argv: Sequence[str]) -> int | None:
    """Read ``--threads`` from ``argv`` the same way argparse will.

    Both spellings have to be recognised. Accepting ``--threads 4`` but not
    ``--threads=4`` would export nothing, and the run would then report the
    BLAS default as though the pin had taken effect.
    """
    for index, item in enumerate(argv):
        if item == "--threads":
            value: str | None = argv[index + 1] if index + 1 < len(argv) else None
        elif item.startswith("--threads="):
            value = item.split("=", 1)[1]
        else:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None  # argparse reports the bad value with a better message
    return None


def apply_thread_limit() -> None:
    """Export ``--threads`` before NumPy loads its BLAS.

    Every BLAS reads these variables once, at library load time. Setting them
    after ``import numpy`` would silently change nothing, which is exactly the
    kind of measurement error this tool exists to avoid.
    """
    global REQUESTED_THREADS
    requested = requested_threads(sys.argv)
    if requested is None:
        return
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(requested)
    REQUESTED_THREADS = requested


apply_thread_limit()

#: Recorded here, before the imports below, so the report can state whether the
#: export really preceded NumPy. ``False`` would mean ``--threads`` did nothing.
THREAD_LIMIT_APPLIED_BEFORE_NUMPY = "numpy" not in sys.modules

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowq import Engine  # noqa: E402
from slowq.metrics import (  # noqa: E402
    blas_info,
    environment_info,
    human_bytes,
    peak_rss_bytes,
    stopwatch,
)
from slowq.model import Qwen3Model  # noqa: E402
from slowq.sampling import SamplingParams  # noqa: E402

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "benchmark"

#: Prefill lengths for the sweep. 8 and 32 sit inside one chunk at the default
#: chunk size, 512 forces several, so the sweep covers both paths.
PREFILL_LENGTHS = (8, 32, 128, 512)


@dataclass(frozen=True)
class Workload:
    name: str
    messages: tuple[dict[str, str], ...]
    max_new_tokens: int


#: Chat-shaped workloads rather than synthetic ones: the prompt lengths, the
#: chat template overhead and the stop-token behaviour are all part of the cost
#: a user actually sees.
WORKLOADS: tuple[Workload, ...] = (
    Workload(
        name="zh-short",
        messages=({"role": "user", "content": "你好，用一句话介绍你自己。"},),
        max_new_tokens=64,
    ),
    Workload(
        name="en-with-system",
        messages=(
            {
                "role": "system",
                "content": "You are a terse assistant. Answer in two sentences.",
            },
            {
                "role": "user",
                "content": "Why does a KV cache make autoregressive decoding "
                "linear in sequence length instead of quadratic?",
            },
        ),
        max_new_tokens=128,
    ),
    Workload(
        name="multi-turn",
        messages=(
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 12 * 8?"},
            {"role": "assistant", "content": "12 * 8 = 96."},
            {"role": "user", "content": "And what is that plus 4? Reply with just the number."},
        ),
        max_new_tokens=32,
    ),
    Workload(
        name="emoji-mixed-script",
        messages=(
            {
                "role": "user",
                "content": "用一个 emoji 表示开心 🎉，然后解释它的 Unicode 名称。",
            },
        ),
        max_new_tokens=64,
    ),
)


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def measure_load(model_dir: Path, max_context: int, prefill_chunk_size: int) -> tuple[Engine, float]:
    with stopwatch() as elapsed:
        engine = Engine.from_directory(
            model_dir, max_context=max_context, prefill_chunk_size=prefill_chunk_size
        )
    return engine, elapsed[0]


def probe_token_ids(length: int, vocab_size: int) -> list[int]:
    """Deterministic, always-in-range ids for the prefill probe.

    Real text would make the sweep depend on the tokenizer and on a vocabulary
    large enough to hold it; the tiny test model has 372 tokens, so a hard-coded
    1000-based ramp would have been out of range there.
    """
    base = min(1000, max(vocab_size - 2, 1))
    span = max(vocab_size - base - 1, 1)
    return [base + (index % span) for index in range(length)]


class ForwardSpy:
    """Count the ``model.forward`` calls a timing block really executed.

    The chunk count in the report has to be *observed*. Deriving it from
    ``length // chunk_size`` is what let an earlier version of this sweep claim
    four chunks while calling ``forward`` exactly once.
    """

    def __init__(self, model: Qwen3Model) -> None:
        self._model = model
        self._forward = model.forward
        self.token_counts: list[int] = []

    def __enter__(self) -> "ForwardSpy":
        spy = self

        def forward(input_ids, kv_cache, *args, **kwargs):
            spy.token_counts.append(len(input_ids))
            return spy._forward(input_ids, kv_cache, *args, **kwargs)

        self._model.forward = forward
        return self

    def __exit__(self, *exc_info) -> bool:
        self._model.forward = self._forward
        return False


def time_prefill(
    model: Qwen3Model, token_ids: Sequence[int], chunk_size: int | None, *, mode: str
) -> dict[str, Any]:
    """Measure one prefill, on a cache that starts empty.

    ``chunk_size=None`` feeds the whole block in a single forward; otherwise the
    block goes through the chunk loop the engine runs, appending to one growing
    cache. Only the forwards are inside the stopwatch: argmax, sampling and
    printing are not prefill and must not be billed to it.
    """
    blocks = (
        [list(token_ids)]
        if chunk_size is None
        else [
            list(token_ids)[start : start + chunk_size]
            for start in range(0, len(token_ids), chunk_size)
        ]
    )
    cache = model.new_cache(len(token_ids) + 1)
    with ForwardSpy(model) as spy, stopwatch() as elapsed:
        for block in blocks:
            model.forward(block, cache)
    seconds = elapsed[0]
    return {
        "mode": mode,
        "prompt_tokens": len(token_ids),
        "chunk_size": None if chunk_size is None else int(chunk_size),
        "forwards": len(spy.token_counts),
        "forward_token_counts": list(spy.token_counts),
        "prefill_seconds": seconds,
        "prefill_tokens_per_second": len(token_ids) / seconds if seconds > 0 else float("nan"),
        "cache_length": cache.length,
    }


def measure_prefill_sweep(engine: Engine, max_length: int) -> list[dict[str, Any]]:
    """Two rows per length: one full forward, and the engine's chunk loop.

    Both are prefill, but they are not the same measurement — one big matmul
    versus ``ceil(T / chunk)`` matmuls against a growing KV cache. Reporting the
    first under the second's label is how a report ends up describing a run that
    never happened.
    """
    model = engine.model
    rows: list[dict[str, Any]] = []
    for length in PREFILL_LENGTHS:
        if length > max_length:
            continue
        token_ids = probe_token_ids(length, model.config.vocab_size)
        for mode, chunk_size in (("full", None), ("chunked", engine.prefill_chunk_size)):
            row = time_prefill(model, token_ids, chunk_size, mode=mode)
            rows.append(row)
            label = "full" if mode == "full" else f"chunked@{row['chunk_size']}"
            print(
                f"  prefill {length:>4} tokens  {label:<13} "
                f"{row['prefill_seconds'] * 1e3:>9.1f} ms  "
                f"{row['prefill_tokens_per_second']:>8.1f} tok/s  "
                f"({row['forwards']} forward(s): {row['forward_token_counts']})",
                file=sys.stderr,
            )
    return rows


def measure_decode(engine: Engine, prompt_tokens: int, steps: int) -> dict[str, Any]:
    """Time ``steps`` single-token forwards after a short prefill."""
    model = engine.model
    token_ids = probe_token_ids(prompt_tokens, model.config.vocab_size)
    cache = model.new_cache(prompt_tokens + steps + 1)
    forward = model.forward(token_ids, cache)
    token_id = int(np.argmax(forward.logits))
    with stopwatch() as elapsed:
        for _ in range(steps):
            forward = model.forward([token_id], cache)
            token_id = int(np.argmax(forward.logits))
    seconds = elapsed[0]
    row = {
        "prompt_tokens": prompt_tokens,
        "decode_steps": steps,
        "decode_seconds": seconds,
        "decode_tokens_per_second": steps / seconds if seconds > 0 else float("nan"),
        "ms_per_token": seconds * 1e3 / steps if steps else float("nan"),
    }
    print(
        f"  decode  {steps:>4} steps   {seconds * 1e3:>9.1f} ms  "
        f"{row['decode_tokens_per_second']:>8.2f} tok/s  "
        f"({row['ms_per_token']:.1f} ms/token)",
        file=sys.stderr,
    )
    return row


def measure_workload(
    engine: Engine, workload: Workload, sampling: SamplingParams, repeat: int
) -> list[dict[str, Any]]:
    """Run one workload ``repeat`` times and keep every run."""
    runs: list[dict[str, Any]] = []
    for _ in range(repeat):
        result = engine.generate(
            workload.messages, sampling=sampling, max_new_tokens=workload.max_new_tokens
        )
        row = result.metrics.as_dict()
        row["text"] = result.text
        runs.append(row)
    return runs


def summarise(runs: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(run[key]) for run in runs if np.isfinite(run.get(key, float("nan")))]
    if not values:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def print_micro_report(prefill: list[dict[str, Any]], decode: dict[str, Any]) -> None:
    print("\n[micro] prefill sweep — 'full' is one forward over the whole block,")
    print("        'chunked' is the engine's chunk loop over one growing cache")
    print(f"  {'tokens':>7} {'mode':>13} {'ms':>10} {'tok/s':>10} {'forwards':>9}")
    for row in prefill:
        mode = "full" if row["mode"] == "full" else f"chunked@{row['chunk_size']}"
        print(
            f"  {row['prompt_tokens']:>7} {mode:>13} {row['prefill_seconds'] * 1e3:>10.1f} "
            f"{row['prefill_tokens_per_second']:>10.1f} {row['forwards']:>9}"
        )
    print("  (the forward count is observed calls, not length // chunk_size)")
    print("\n[micro] decode (single-token forwards, KV cache resident)")
    print(
        f"  {decode['decode_tokens_per_second']:.2f} tok/s "
        f"({decode['ms_per_token']:.2f} ms/token) after a "
        f"{decode['prompt_tokens']}-token prefill"
    )


def print_workload_report(name: str, runs: Sequence[dict[str, Any]]) -> None:
    first = runs[0]
    print(f"\n[{name}] {len(runs)} run(s), {first['prompt_tokens']} prompt tokens")
    print(f"  {'':<26} {'median':>10} {'min':>10} {'max':>10}")
    for key, label, scale in (
        ("first_token_seconds", "first token latency (ms)", 1e3),
        ("prefill_seconds", "prefill (ms)", 1e3),
        ("decode_seconds", "decode (ms)", 1e3),
        ("total_seconds", "total (ms)", 1e3),
    ):
        stats = summarise(runs, key)
        print(
            f"  {label:<26} {stats['median'] * scale:>10.1f} "
            f"{stats['min'] * scale:>10.1f} {stats['max'] * scale:>10.1f}"
        )
    for key, label in (
        ("decode_tokens_per_second", "decode tok/s"),
        ("prefill_tokens_per_second", "prefill tok/s"),
    ):
        stats = summarise(runs, key)
        print(
            f"  {label:<26} {stats['median']:>10.2f} "
            f"{stats['min']:>10.2f} {stats['max']:>10.2f}"
        )
    print(f"  {'generated tokens':<26} {first['generated_tokens']:>10}")
    print(f"  {'stop reason':<26} {first['stop_reason']:>10}")
    preview = runs[-1]["text"].replace("\n", " ")
    if len(preview) > 72:
        preview = preview[:72] + "..."
    print(f"  last answer: {preview}")


def print_environment(info: dict[str, Any]) -> None:
    print("\n[environment]")
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key:<24} {sub_value}")
        else:
            print(f"  {key:<28} {value}")


def thread_report() -> dict[str, Any]:
    """What was asked for, kept apart from what could be detected.

    The ``--threads`` pin is a *setting* and ``blas_info()`` is *library
    metadata*; neither is a measurement of the thread count NumPy used at
    runtime, which is not observable from here. Merging them into one dict under
    one name is how a setting ends up quoted as a result, so ``effective_thread_count``
    is reported as ``None`` rather than guessed from the request.
    """
    return {
        "requested": REQUESTED_THREADS,
        "requested_before_numpy_import": bool(THREAD_LIMIT_APPLIED_BEFORE_NUMPY),
        "exported_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "detected_backend": blas_info(),
        "logical_cpus": os.cpu_count(),
        "effective_thread_count": None,
        "note": (
            "'requested'/'exported_env' are settings and 'detected_backend' is "
            "library metadata; the effective BLAS thread count is not observable "
            "from this process and is reported as null"
        ),
    }


def print_threads(report: dict[str, Any]) -> None:
    print("\n[threads] requested setting vs detected backend — neither is a measured count")
    requested = report["requested"]
    print(f"  {'requested (--threads)':<28} {'(not given)' if requested is None else requested}")
    print(f"  {'exported before numpy':<28} {report['requested_before_numpy_import']}")
    for name, value in report["exported_env"].items():
        print(f"  {name:<28} {value}")
    backend = report["detected_backend"]
    detected = f"{backend.get('name', '?')} {backend.get('version', '')}".strip()
    print(f"  {'detected BLAS':<28} {detected}")
    print(f"  {'logical CPUs':<28} {report['logical_cpus']}")
    print(f"  {'effective thread count':<28} not observable")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SLOWQ_MODEL_DIR"),
        type=Path,
        help="converted model directory; defaults to $SLOWQ_MODEL_DIR",
    )
    parser.add_argument("--prompt", default=None, help="benchmark only this user prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument(
        "--decode-steps", type=int, default=32, help="single-token forwards in the decode probe"
    )
    parser.add_argument("--repeat", type=int, default=3, help="runs per workload")
    parser.add_argument(
        "--threads", type=int, default=None, help="pin the BLAS thread count for this run"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="0 means greedy, which is reproducible"
    )
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--json", type=Path, default=None, help="write the raw numbers here (default: artifacts/)"
    )
    parser.add_argument(
        "--no-micro", action="store_true", help="skip the prefill/decode probes"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model is None:
        print(
            "error: --model DIR is required (or set SLOWQ_MODEL_DIR).",
            file=sys.stderr,
        )
        return 2
    if args.repeat < 1:
        print("error: --repeat must be >= 1", file=sys.stderr)
        return 2

    sampling = SamplingParams(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed
    )

    print(f"慢炖 · SlowQ benchmark — {args.model}", file=sys.stderr)
    print(f"sampler: {sampling.describe()}", file=sys.stderr)
    engine, cold_load_seconds = measure_load(
        args.model, args.max_context, args.prefill_chunk_size
    )
    print(
        f"mapped weights in {cold_load_seconds:.3f} s "
        f"(mmap: pages are not resident until the first forward reads them)",
        file=sys.stderr,
    )
    rss_after_load = peak_rss_bytes()

    micro: dict[str, Any] = {}
    if not args.no_micro:
        budget = min(args.max_context - args.decode_steps - 1, max(PREFILL_LENGTHS))
        if budget < min(PREFILL_LENGTHS):
            print(
                f"\nskipping the micro probes: a {min(PREFILL_LENGTHS)}-token prefill "
                f"plus {args.decode_steps} decode steps does not fit in --max-context "
                f"{args.max_context}",
                file=sys.stderr,
            )
        else:
            print("\nprobing prefill and decode ...", file=sys.stderr)
            micro["prefill_sweep"] = measure_prefill_sweep(engine, budget)
            micro["decode"] = measure_decode(
                engine, prompt_tokens=min(64, budget), steps=args.decode_steps
            )
            # The first forward after mapping is the one that pages the weights
            # in, and it is the first row because 'full' is timed before 'chunked'.
            micro["cold_first_forward_seconds"] = micro["prefill_sweep"][0]["prefill_seconds"]
    rss_after_micro = peak_rss_bytes()

    if args.prompt is not None:
        workloads = (
            Workload(
                name="custom",
                messages=({"role": "user", "content": args.prompt},),
                max_new_tokens=args.max_new_tokens,
            ),
        )
    else:
        workloads = WORKLOADS

    results: dict[str, list[dict[str, Any]]] = {}
    for workload in workloads:
        print(f"\nrunning {workload.name} ...", file=sys.stderr)
        started = time.perf_counter()
        results[workload.name] = measure_workload(engine, workload, sampling, args.repeat)
        print(
            f"  {workload.name} took {time.perf_counter() - started:.1f} s "
            f"for {len(results[workload.name])} run(s)",
            file=sys.stderr,
        )

    _, warm_load_seconds = measure_load(
        args.model, args.max_context, args.prefill_chunk_size
    )
    rss_final = peak_rss_bytes()

    if micro:
        print_micro_report(micro["prefill_sweep"], micro["decode"])
    for name, runs in results.items():
        print_workload_report(name, runs)

    print("\n[model]")
    for label, value in engine.config.describe():
        print(f"  {label:<28} {value}")
    print(f"  {'weight bytes (logical)':<28} {human_bytes(engine.model.weights.deduplicated_bytes())}")
    print(f"  {'weight bytes (on disk)':<28} {human_bytes(engine.model.weights.file_bytes())}")
    print(f"  {'KV bytes per token':<28} {human_bytes(engine.config.kv_bytes_per_token())}")
    print(f"  {'KV allocated':<28} {human_bytes(engine.kv_cache.allocated_bytes())}")
    print(f"  {'max_context':<28} {engine.max_context}")
    print(f"  {'prefill_chunk_size':<28} {engine.prefill_chunk_size}")

    print("\n[memory] mmap'ed pages count towards RSS only once they are read")
    print(f"  {'peak RSS after mapping':<28} {human_bytes(rss_after_load)}")
    print(f"  {'peak RSS after probes':<28} {human_bytes(rss_after_micro)}")
    print(f"  {'peak RSS final':<28} {human_bytes(rss_final)}")

    print("\n[load]")
    print(f"  {'cold (first mapping)':<28} {cold_load_seconds * 1e3:.1f} ms")
    print(f"  {'warm (second mapping)':<28} {warm_load_seconds * 1e3:.1f} ms")

    info = environment_info()
    print_environment(info)

    threads = thread_report()
    print_threads(threads)

    payload = {
        "model_directory": str(args.model),
        "args": {key: str(value) for key, value in vars(args).items()},
        "sampler": sampling.describe(),
        "environment": info,
        "threads": threads,
        "load_seconds": {"cold": cold_load_seconds, "warm": warm_load_seconds},
        "peak_rss_bytes": {
            "after_load": rss_after_load,
            "after_micro": rss_after_micro,
            "final": rss_final,
        },
        "weight_logical_bytes": engine.model.weights.deduplicated_bytes(),
        "kv_allocated_bytes": engine.kv_cache.allocated_bytes(),
        "micro": micro,
        "workloads": results,
    }
    output = args.json or DEFAULT_ARTIFACTS / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
