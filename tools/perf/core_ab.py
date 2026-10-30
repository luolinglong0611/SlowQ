#!/usr/bin/env python3
"""A/B harness for the round-1 inference optimisations.

Deliberately uses *only* APIs that exist at the baseline commit, so the same file
runs unchanged against the pre-optimisation tree and the working tree (the one
exception is the ``chat`` suite, which asks ``hasattr(engine, "session")`` and
reports the warm path as unsupported when it is absent):

    .venv/bin/python tools/perf/core_ab.py \
        --src /path/to/slowq --model models/Qwen3-0.6B-slowq --suite all

Every number it reports is an observed wall-clock measurement from this process.
Nothing is derived from a token count divided by a chunk size.

Suites
------
micro      prefill through the *engine's* chunk loop (stream_ids with
           max_new_tokens=1), a raw single-forward prefill, and a raw
           single-token decode loop.
workloads  the same chat workloads tools/benchmark.py uses, through
           Engine.generate, reporting the engine's own metric fields.
chat       one fixed four-turn conversation, run turn by turn as standalone
           requests ("cold": every turn re-prefills the whole history) and, on a
           tree that has ``Engine.session``, through a Session ("warm": only the
           tokens past the shared prefix are prefilled). Every warm turn's token
           ids are compared against the cold turn's, so a speedup that came from
           doing less work than it should cannot pass unnoticed.

One process per (model, threads, suite) so that peak RSS is attributable.
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
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Sequence  # noqa: E402


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="repo root whose src/ to import")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--suite", default="all", choices=("micro", "workloads", "chat", "all")
    )
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=7, help="timed repeats after warmup")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--prefill-lengths", default="8,32,128,512")
    parser.add_argument(
        "--chat-max-new",
        type=int,
        default=32,
        help="tokens generated per turn of the chat suite's conversation",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="skip the chat suite even under --suite all: it costs ~80 extra "
        "generate calls per process and measures something the thread/chunk "
        "sweep does not vary",
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
from slowq import Engine  # noqa: E402
from slowq.errors import ContextOverflowError  # noqa: E402
from slowq.metrics import blas_info, environment_info, peak_rss_bytes  # noqa: E402
from slowq.sampling import SamplingParams  # noqa: E402

# Fail loudly rather than silently measuring the wrong tree (the dev venv has an
# editable install pointing at the main checkout).
assert slowq.__file__.startswith(SRC), f"imported {slowq.__file__}, expected under {SRC}"

PREFILL_LENGTHS = tuple(int(x) for x in ARGS.prefill_lengths.split(",") if x)


def probe_token_ids(length: int, vocab_size: int) -> list[int]:
    """Deterministic, in-range ids. Same rule as tools/benchmark.py."""
    base = min(1000, max(vocab_size - 2, 1))
    span = max(vocab_size - base - 1, 1)
    return [base + (index % span) for index in range(length)]


def timeit(fn: Callable[[], Any], repeat: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.fmean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples": samples,
        "repeat": repeat,
        "warmup": warmup,
    }


def summarise(values: Sequence[Any]) -> dict[str, float]:
    """median/min/max/stdev over the finite entries of ``values``.

    Non-finite entries are dropped rather than propagated: a workload that
    stopped before decoding anything has a ``decode_tokens_per_second`` of 0/0,
    and one NaN would otherwise make every statistic of that key NaN too.
    """
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return {"median": float("nan")}
    return {
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
        "stdev": statistics.stdev(finite) if len(finite) > 1 else 0.0,
    }


WORKLOADS: tuple[tuple[str, tuple[dict[str, str], ...], int], ...] = (
    (
        "zh-short",
        ({"role": "user", "content": "你好，用一句话介绍你自己。"},),
        64,
    ),
    (
        "en-with-system",
        (
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
        128,
    ),
    (
        "multi-turn",
        (
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 12 * 8?"},
            {"role": "assistant", "content": "12 * 8 = 96."},
            {
                "role": "user",
                "content": "And what is that plus 4? Reply with just the number.",
            },
        ),
        32,
    ),
    (
        "emoji-mixed-script",
        (
            {
                "role": "user",
                "content": "用一个 emoji 表示开心 🎉，然后解释它的 Unicode 名称。",
            },
        ),
        64,
    ),
    (
        # Long prompt, short answer: this is the workload where a per-chunk
        # vocabulary projection costs the most, because the prompt spans several
        # prefill chunks and only the last one's logits are ever sampled from.
        "long-prompt-short-answer",
        (
            {
                "role": "user",
                "content": (
                    "阅读下面这段材料，然后用一句话回答最后的问题。\n"
                    + (
                        "慢炖是一种把食材放在低温液体里长时间加热的烹调方式，"
                        "它让结缔组织慢慢转化成明胶，因此便宜的部位也能变得柔软。"
                        "推理引擎里的 KV cache 与之类似：把已经算过的中间结果留在锅里，"
                        "下一步只需要放进新的食材，而不必从头再煮一遍。"
                    )
                    * 8
                    + "\n问题：这段话用了什么比喻？"
                ),
            },
        ),
        32,
    ),
)


CHAT_SYSTEM = "You are a terse assistant. Answer in one or two sentences."

# A conversation that grows: each turn's prompt is the previous turn's prompt
# plus the previous reply plus one new question, so the share of the prompt that
# a Session can reuse rises turn by turn. The questions are related on purpose —
# an unrelated fourth turn would still share the whole prefix, and that is the
# point being measured.
CHAT_USER_TURNS: tuple[str, ...] = (
    "用一句话解释什么是 KV cache。",
    "它为什么能让自回归解码的复杂度变成线性？",
    "那么多轮对话里，前缀复用具体省掉了哪一部分计算？",
    "举一个会让这种复用失效的例子。",
)


# --------------------------------------------------------------------------- #


def decoder_call_spy(model: Any) -> Any:
    """Count the decoder-stack calls a block really makes, and which entry point.

    Both ``encode`` and ``forward`` have to be watched: ``DecoderModel.prefill``
    runs its non-final chunks through ``encode`` and only the last through
    ``forward``. Counting ``forward`` alone reports one call for a four-chunk
    prefill — a measurement that looks like evidence for the optimisation while
    actually hiding whether it happened. ``forward`` calls ``encode`` internally,
    so a depth guard keeps one chunk from being counted twice.
    """
    real_forward = model.forward
    real_encode = model.encode
    state: dict[str, Any] = {"counts": [], "entry_points": [], "depth": 0}

    def wrap(name: str, original: Any) -> Any:
        def wrapper(input_ids, kv_cache, *args, **kwargs):
            outer = state["depth"] == 0
            state["depth"] += 1
            try:
                if outer:
                    state["counts"].append(len(input_ids))
                    state["entry_points"].append(name)
                return original(input_ids, kv_cache, *args, **kwargs)
            finally:
                state["depth"] -= 1

        return wrapper

    def restore() -> None:
        model.forward = real_forward
        model.encode = real_encode

    model.forward = wrap("forward", real_forward)
    model.encode = wrap("encode", real_encode)
    state["restore"] = restore
    return state


def suite_micro(engine: Engine) -> dict[str, Any]:
    model = engine.model
    vocab = model.config.vocab_size
    out: dict[str, Any] = {}

    # (a) prefill through the engine's real chunk loop: stream_ids with
    #     max_new_tokens=1 does prefill + one sample and stops.
    out["engine_prefill"] = {}
    for length in PREFILL_LENGTHS:
        ids = probe_token_ids(length, vocab)

        def run(ids=ids) -> None:
            for _ in engine.stream_ids(ids, max_new_tokens=1):
                pass

        stats = timeit(run, ARGS.repeat, ARGS.warmup)
        # observed chunk count from one instrumented run
        spy = decoder_call_spy(model)
        try:
            for _ in engine.stream_ids(ids, max_new_tokens=1):
                pass
        finally:
            spy["restore"]()
        chunks = spy["counts"]
        stats["forward_token_counts"] = list(chunks)
        stats["forwards"] = len(chunks)
        stats["entry_points"] = list(spy["entry_points"])
        stats["vocabulary_projections"] = spy["entry_points"].count("forward")
        stats["prompt_tokens"] = length
        stats["tokens_per_second"] = length / stats["median"] if stats["median"] > 0 else float("nan")
        out["engine_prefill"][str(length)] = stats
        print(
            f"  engine prefill {length:>4} tok  {stats['median'] * 1e3:>9.2f} ms  "
            f"{stats['tokens_per_second']:>8.1f} tok/s  "
            f"({stats['forwards']} call(s): {chunks}, "
            f"{stats['vocabulary_projections']} vocab proj)",
            file=sys.stderr,
        )

    # (b) raw single forward over the whole block (no chunk loop, no lm_head skip
    #     possible: one forward == one logits row).
    out["raw_prefill_full"] = {}
    for length in PREFILL_LENGTHS:
        ids = probe_token_ids(length, vocab)

        def run(ids=ids) -> None:
            cache = model.new_cache(len(ids) + 1)
            model.forward(ids, cache)

        stats = timeit(run, ARGS.repeat, ARGS.warmup)
        stats["prompt_tokens"] = length
        stats["tokens_per_second"] = length / stats["median"] if stats["median"] > 0 else float("nan")
        out["raw_prefill_full"][str(length)] = stats
        print(
            f"  raw full prefill {length:>4} tok  {stats['median'] * 1e3:>9.2f} ms  "
            f"{stats['tokens_per_second']:>8.1f} tok/s",
            file=sys.stderr,
        )

    # (c) raw single-token decode loop on a resident cache.
    prompt_tokens = min(64, max(PREFILL_LENGTHS))
    ids = probe_token_ids(prompt_tokens, vocab)

    def decode_window(steps: int, warm_rounds: int, repeat: int) -> dict[str, Any]:
        """Time ``steps`` single-token forwards, ``repeat`` independent windows.

        A *window* is the unit of repetition, and it is the whole measurement:
        each one re-prefills the same ``ids`` into a fresh cache, so every window
        starts from the same token sequence, the same KV length and the same
        context, and ends at the same cache length. What varies between windows is
        only the machine.

        The consecutive tokens *inside* one window are not repeated experiments
        and are never reported as such. They are one trajectory through a growing
        cache — step 30 attends to 30 more keys than step 1 — so splitting a
        window into per-token samples would manufacture dispersion out of a real
        effect and then average the effect away. One window yields exactly one
        sample: its duration.

        The prefill that establishes the starting state is outside the timing
        boundary on purpose. This probe prices decode steps; a window that timed
        its own prefill would mix the two and change meaning with ``steps``.
        """
        # The prefill commits ``prompt_tokens`` slots and the first sampled token
        # is not fed back, so a window of ``steps`` decode forwards ends at
        # ``prompt_tokens + steps`` — the value the single-window harness printed
        # as ``cache_length_after``, and the one both trees must agree on.
        expected_length = prompt_tokens + steps

        def one_window() -> tuple[float, int]:
            cache = model.new_cache(prompt_tokens + steps + 8)
            tid = int(np.argmax(model.forward(ids, cache).logits))
            started = time.perf_counter()
            for _ in range(steps):
                tid = int(np.argmax(model.forward([tid], cache).logits))
            elapsed = time.perf_counter() - started
            return elapsed, cache.length

        for _ in range(warm_rounds):
            one_window()
        rounds = max(repeat, 1)
        results = [one_window() for _ in range(rounds)]
        seconds = [item[0] for item in results]
        lengths = sorted({item[1] for item in results})
        # Deterministic in the code, not in the machine: if this fires, the
        # windows are not starting from the same state and the median over them
        # is a median over different measurements.
        assert lengths == [expected_length], (
            f"decode windows ended at cache lengths {lengths}, expected "
            f"[{expected_length}]; the repeats are not re-establishing the same state"
        )

        node = {
            "median": statistics.median(seconds),
            "min": min(seconds),
            "max": max(seconds),
            "mean": statistics.fmean(seconds),
            "stdev": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
            "samples": seconds,
            "repeat": len(seconds),
            "warmup": warm_rounds,
        }
        per_token_ms = [sample * 1e3 / steps for sample in seconds]
        return {
            "prompt_tokens": prompt_tokens,
            "decode_steps": steps,
            # The three scalars below are the *median window*, in the same shape
            # and meaning the single-window harness used to report, so a table
            # built from them is a median-of-windows rather than one observation.
            "decode_seconds": node["median"],
            "decode_tokens_per_second": steps / node["median"] if node["median"] > 0 else float("nan"),
            "ms_per_token": node["median"] * 1e3 / steps,
            "cache_length_after": expected_length,
            "cache_lengths_observed": lengths,
            "warm_rounds": warm_rounds,
            "windows": node,
            "ms_per_token_windows": {
                **node,
                "samples": per_token_ms,
                "median": statistics.median(per_token_ms),
                "min": min(per_token_ms),
                "max": max(per_token_ms),
                "mean": statistics.fmean(per_token_ms),
                "stdev": statistics.stdev(per_token_ms) if len(per_token_ms) > 1 else 0.0,
            },
        }

    out["raw_decode"] = decode_window(ARGS.decode_steps, ARGS.warmup, ARGS.repeat)
    row = out["raw_decode"]
    window = row["windows"]
    print(
        f"  raw decode {row['decode_steps']:>3} steps  {row['decode_seconds'] * 1e3:>9.2f} ms  "
        f"{row['decode_tokens_per_second']:>8.2f} tok/s ({row['ms_per_token']:.2f} ms/token) "
        f"cache_length={row['cache_length_after']} "
        f"[{window['repeat']} window(s): {window['min'] * 1e3:.2f}-{window['max'] * 1e3:.2f} ms]",
        file=sys.stderr,
    )

    # (d) a second, longer decode window for a steadier rate.
    out["raw_decode_long"] = decode_window(4 * ARGS.decode_steps, ARGS.warmup, ARGS.repeat)
    row = out["raw_decode_long"]
    window = row["windows"]
    print(
        f"  raw decode {row['decode_steps']:>3} steps  {row['decode_seconds'] * 1e3:>9.2f} ms  "
        f"{row['decode_tokens_per_second']:>8.2f} tok/s "
        f"[{window['repeat']} window(s): {window['min'] * 1e3:.2f}-{window['max'] * 1e3:.2f} ms]",
        file=sys.stderr,
    )
    return out


def suite_workloads(engine: Engine) -> dict[str, Any]:
    sampling = SamplingParams(temperature=0.0, top_k=0, top_p=1.0)
    out: dict[str, Any] = {}
    for name, messages, max_new in WORKLOADS:
        runs: list[dict[str, Any]] = []
        try:
            for _ in range(ARGS.warmup):
                engine.generate(messages, sampling=sampling, max_new_tokens=max_new)
            for _ in range(ARGS.repeat):
                result = engine.generate(messages, sampling=sampling, max_new_tokens=max_new)
                row = result.metrics.as_dict()
                row["text_sha"] = hashlib.sha256(result.text.encode("utf-8")).hexdigest()[:16]
                row["token_ids"] = list(result.token_ids)
                runs.append(row)
        except ContextOverflowError as exc:
            # A small-vocabulary checkpoint tokenises the long-prompt workload to
            # more than --max-context tokens. SlowQ never truncates, so the honest
            # record is "this model cannot run this workload", not a shorter one.
            out[name] = {"skipped": str(exc)}
            print(f"  {name:<18} SKIPPED: {exc}", file=sys.stderr)
            continue

        def summ(key: str) -> dict[str, float]:
            return summarise([r.get(key, float("nan")) for r in runs])

        entry = {
            "prompt_tokens": runs[0]["prompt_tokens"],
            "generated_tokens": runs[0]["generated_tokens"],
            "prefill_chunks": runs[0]["prefill_chunks"],
            "stop_reason": runs[0]["stop_reason"],
            "text_sha": runs[-1]["text_sha"],
            "identical_outputs": len({r["text_sha"] for r in runs}) == 1,
            "first_token_seconds": summ("first_token_seconds"),
            "prefill_seconds": summ("prefill_seconds"),
            "decode_seconds": summ("decode_seconds"),
            "total_seconds": summ("total_seconds"),
            "tokenize_seconds": summ("tokenize_seconds"),
            "decode_tokens_per_second": summ("decode_tokens_per_second"),
            "prefill_tokens_per_second": summ("prefill_tokens_per_second"),
            "peak_rss_bytes": summ("peak_rss_bytes"),
            "runs": runs,
        }
        out[name] = entry
        print(
            f"  {name:<18} ttft {entry['first_token_seconds']['median'] * 1e3:>8.1f} ms  "
            f"prefill {entry['prefill_seconds']['median'] * 1e3:>8.1f} ms  "
            f"decode {entry['decode_tokens_per_second']['median']:>7.2f} tok/s  "
            f"total {entry['total_seconds']['median'] * 1e3:>8.1f} ms  "
            f"({entry['prompt_tokens']}p/{entry['generated_tokens']}g, "
            f"{entry['stop_reason']}, identical={entry['identical_outputs']})",
            file=sys.stderr,
        )
    return out


def conversation_round(runner: Callable[[list[dict[str, str]]], Any]) -> list[dict[str, Any]]:
    """Run the fixed conversation once, as one request per turn.

    ``runner`` takes the messages rendered so far and returns a
    ``GenerationResult``. Each reply is decoded to text and appended, so the next
    turn's prompt is re-rendered and re-tokenised from scratch. That is the path
    a real chat client takes, and it matters: the ids a reply was generated from
    need not survive a round trip through ``decode`` + ``encode``, which is why
    reuse is decided on token ids rather than on the message list.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM}]
    rows: list[dict[str, Any]] = []
    for turn, user in enumerate(CHAT_USER_TURNS):
        messages.append({"role": "user", "content": user})
        started = time.perf_counter()
        result = runner(messages)
        row = result.metrics.as_dict()
        row["turn"] = turn
        # Wall clock around the call, not the engine's own ``total_seconds``:
        # that one starts inside ``_stream`` and so excludes template rendering
        # and tokenisation of the whole growing history.
        row["wall_seconds"] = time.perf_counter() - started
        row["token_ids"] = list(result.token_ids)
        row["text_sha"] = hashlib.sha256(result.text.encode("utf-8")).hexdigest()[:16]
        rows.append(row)
        messages.append({"role": "assistant", "content": result.text})
    return rows


FIRST_ROUND_KEYS = (
    "turn",
    "prompt_tokens",
    "generated_tokens",
    "prefill_chunks",
    "stop_reason",
    "wall_seconds",
    "first_token_seconds",
    "prefill_seconds",
    "decode_seconds",
    "total_seconds",
    "text_sha",
)


def summarise_rounds(rounds: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Per-turn statistics over timed rounds that all ran the same conversation."""
    out: dict[str, Any] = {}
    for turn in range(len(CHAT_USER_TURNS)):
        rows = [row[turn] for row in rounds]
        # ``reused_tokens`` only exists on a tree with Sessions; on the baseline
        # the honest value is 0, because nothing was reused.
        reused = sorted({row.get("reused_tokens", 0) for row in rows})
        entry: dict[str, Any] = {
            "prompt_tokens": rows[0]["prompt_tokens"],
            "generated_tokens": rows[0]["generated_tokens"],
            "reused_tokens": rows[0].get("reused_tokens", 0),
            "reused_tokens_observed": reused,
            "prefill_tokens": rows[0].get("prefill_tokens", rows[0]["prompt_tokens"]),
            "prefill_chunks": rows[0]["prefill_chunks"],
            "stop_reason": rows[0]["stop_reason"],
            "text_sha": rows[0]["text_sha"],
            "identical_outputs": len({row["text_sha"] for row in rows}) == 1,
        }
        for key in (
            "wall_seconds",
            "first_token_seconds",
            "prefill_seconds",
            "decode_seconds",
            "total_seconds",
            "tokenize_seconds",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "peak_rss_bytes",
        ):
            entry[key] = summarise([row.get(key, float("nan")) for row in rows])
        out[str(turn)] = entry
    return out


def suite_chat(engine: Engine) -> dict[str, Any]:
    """Cold (standalone per turn) against warm (Session), on one conversation.

    The cold path runs on both trees and re-prefills the whole history every
    turn, which is what the CLI did before Sessions existed. The warm path needs
    ``Engine.session``; without it the suite still reports the cold numbers so
    the baseline JSON is not empty.

    Round order is cold-then-warm within one process, and the *first* round of
    each path is recorded separately and excluded from every median: the cold
    first round is also the process's first forward, so it pays for faulting the
    mmapped weights in. Reporting that as a prefill time would be reporting page
    faults, not arithmetic.
    """
    sampling = SamplingParams(temperature=0.0, top_k=0, top_p=1.0)
    max_new = ARGS.chat_max_new
    session = engine.session() if hasattr(engine, "session") else None
    out: dict[str, Any] = {
        "session_supported": session is not None,
        "turns": len(CHAT_USER_TURNS),
        "max_new_tokens": max_new,
        "round_order": ["cold", "warm"] if session is not None else ["cold"],
        "warmup_rounds": ARGS.warmup,
        "timed_rounds": ARGS.repeat,
    }

    def cold(messages: list[dict[str, str]]) -> Any:
        return engine.generate(messages, sampling=sampling, max_new_tokens=max_new)

    try:
        cold_first = conversation_round(cold)
        for _ in range(ARGS.warmup):
            conversation_round(cold)
        cold_rounds = [conversation_round(cold) for _ in range(ARGS.repeat)]
    except ContextOverflowError as exc:
        # Same rule as the workloads suite: a small-vocabulary checkpoint can
        # tokenise a four-turn Chinese conversation past --max-context, and
        # SlowQ refuses rather than truncating. Record the refusal.
        out["skipped"] = str(exc)
        print(f"  chat SKIPPED: {exc}", file=sys.stderr)
        return out

    out["cold"] = {
        "first_round": [{k: row[k] for k in FIRST_ROUND_KEYS} for row in cold_first],
        "turns": summarise_rounds(cold_rounds),
    }

    if session is None:
        out["warm"] = {"unsupported": "this tree has no Engine.session"}
    else:

        def warm(messages: list[dict[str, str]]) -> Any:
            return session.generate(messages, sampling=sampling, max_new_tokens=max_new)

        def warm_round() -> list[dict[str, Any]]:
            # Reset per round, so each round models a *fresh* conversation and
            # turn 0 is genuinely cold on both paths. Leaving the session warm
            # across rounds would let turn 0 reuse the previous round's entire
            # conversation — a hit rate no user ever sees.
            session.reset()
            return conversation_round(warm)

        warm_first = warm_round()
        for _ in range(ARGS.warmup):
            warm_round()
        warm_rounds = [warm_round() for _ in range(ARGS.repeat)]

        out["warm"] = {
            "first_round": [
                {**{k: row[k] for k in FIRST_ROUND_KEYS}, "reused_tokens": row.get("reused_tokens", 0)}
                for row in warm_first
            ],
            "turns": summarise_rounds(warm_rounds),
            "cache_allocated_bytes": session.cache.allocated_bytes(),
        }
        mismatched = [
            {"round": index, "turn": turn}
            for index, rnd in enumerate(warm_rounds)
            for turn, row in enumerate(rnd)
            if row["token_ids"] != cold_rounds[index][turn]["token_ids"]
        ]
        out["agreement"] = {
            "compared_rounds": len(warm_rounds),
            "compared_turns": len(warm_rounds) * len(CHAT_USER_TURNS),
            "mismatched": mismatched,
            "all_turns_identical": not mismatched,
        }
        cold_total = sum(t["wall_seconds"]["median"] for t in out["cold"]["turns"].values())
        warm_total = sum(t["wall_seconds"]["median"] for t in out["warm"]["turns"].values())
        out["conversation_wall_seconds"] = {
            # A sum of per-turn medians, not the median of per-conversation
            # totals. Stated because the two differ and the label should not lie.
            "cold_sum_of_turn_medians": cold_total,
            "warm_sum_of_turn_medians": warm_total,
            "speedup": cold_total / warm_total if warm_total > 0 else float("nan"),
        }

    for path in ("cold", "warm"):
        entry = out[path]
        if "turns" not in entry:
            print(f"  chat {path:<4} unsupported: {entry.get('unsupported')}", file=sys.stderr)
            continue
        for turn, row in entry["turns"].items():
            print(
                f"  chat {path:<4} turn {turn}  wall {row['wall_seconds']['median'] * 1e3:>8.1f} ms  "
                f"ttft {row['first_token_seconds']['median'] * 1e3:>8.1f} ms  "
                f"prefill {row['prefill_seconds']['median'] * 1e3:>8.1f} ms  "
                f"reuse {row['reused_tokens']:>4}/{row['prompt_tokens']:<4} "
                f"prefilled {row['prefill_tokens']:>4}  "
                f"({row['generated_tokens']}g, {row['stop_reason']}, "
                f"identical={row['identical_outputs']})",
                file=sys.stderr,
            )
    if "speedup" in out.get("conversation_wall_seconds", {}):
        total = out["conversation_wall_seconds"]
        print(
            f"  chat conversation  cold {total['cold_sum_of_turn_medians'] * 1e3:>8.1f} ms  "
            f"warm {total['warm_sum_of_turn_medians'] * 1e3:>8.1f} ms  "
            f"speedup {total['speedup']:.2f}x  "
            f"agreement={out['agreement']['all_turns_identical']} "
            f"({out['agreement']['compared_turns']} turns compared)",
            file=sys.stderr,
        )
    return out


def main() -> int:
    # Before the model load, not before the write. ``write_json`` at the end was
    # already exclusive, so an occupied ``--out`` never cost anybody their previous
    # results — it cost them this run: four suites of measurement, then a refusal.
    out_guard.preflight(ARGS.out)
    started = time.perf_counter()
    engine = Engine.from_directory(
        ARGS.model,
        max_context=ARGS.max_context,
        prefill_chunk_size=ARGS.prefill_chunk_size,
    )
    load_seconds = time.perf_counter() - started
    rss_after_load = peak_rss_bytes()
    print(
        f"core_ab: src={SRC}\n         model={ARGS.model} threads={REQUESTED} "
        f"numpy={np.__version__} blas={blas_info().get('name')} "
        f"load={load_seconds:.3f}s active_threads={threading.active_count()}",
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
            "effective_thread_count": None,
        },
        "environment": environment_info(),
        "load_seconds": load_seconds,
        "rss_after_load_bytes": rss_after_load,
    }
    if ARGS.suite in ("micro", "all"):
        print("[micro]", file=sys.stderr)
        payload["micro"] = suite_micro(engine)
    if ARGS.suite in ("workloads", "all"):
        print("[workloads]", file=sys.stderr)
        payload["workloads"] = suite_workloads(engine)
    # ``--no-chat`` trims ``all``; asking for the chat suite explicitly still
    # runs it, so the flag cannot silently produce an empty payload.
    if ARGS.suite == "chat" or (ARGS.suite == "all" and not ARGS.no_chat):
        print("[chat]", file=sys.stderr)
        payload["chat"] = suite_chat(engine)
    payload["rss_final_bytes"] = peak_rss_bytes()
    payload["wall_seconds"] = time.perf_counter() - started

    if ARGS.out:
        # Refuses an existing path rather than truncating it. This payload is the
        # evidence a row of the report is derived from, and whatever already sits at
        # that name was someone else's measurement. See tools/perf/out_guard.py.
        out = out_guard.write_json(ARGS.out, payload)
        print(f"wrote {out_guard.display(out)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
