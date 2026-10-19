"""Command line interface: ``generate``, ``chat``, ``inspect``.

Answers go to stdout; timing, memory and diagnostics go to stderr, so
``python -m slowq generate ... > out.txt`` yields clean text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

from . import __version__
from .engine import DEFAULT_MAX_CONTEXT, DEFAULT_MAX_NEW_TOKENS, DEFAULT_PREFILL_CHUNK_SIZE, Engine
from .errors import SlowQError
from .metrics import environment_info, human_bytes, peak_rss_bytes
from .sampling import SamplingParams
from .tokenizer import ChatMessage
from .weights import MANIFEST_FILENAME

CHAT_BANNER = """慢炖 · SlowQ chat  (Qwen3-0.6B, CPU, FP32, NumPy only)
Commands: /clear reset history · /history show history · /help · /exit (or Ctrl-D)
Ctrl-C during an answer cancels that turn and returns to the prompt.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slowq",
        description="慢炖 · SlowQ — slowly stew Qwen3-0.6B tokens on your CPU.",
    )
    parser.add_argument("--version", action="version", version=f"slowq {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model",
        default=os.environ.get("SLOWQ_MODEL_DIR"),
        metavar="DIR",
        help="converted model directory (the one containing "
        f"{MANIFEST_FILENAME}); defaults to $SLOWQ_MODEL_DIR",
    )
    common.add_argument(
        "--max-context",
        type=int,
        default=DEFAULT_MAX_CONTEXT,
        help="total token budget for prompt + generation (default: %(default)s)",
    )
    common.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="stop after this many generated tokens (default: %(default)s)",
    )
    common.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 means greedy decoding (default: %(default)s)",
    )
    common.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="keep the k most likely tokens; 0 disables (default: %(default)s)",
    )
    common.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="nucleus sampling threshold in (0, 1]; 1.0 disables (default: %(default)s)",
    )
    common.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for the sampler RNG; makes sampling reproducible",
    )
    common.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=DEFAULT_PREFILL_CHUNK_SIZE,
        help="tokens per prefill forward (default: %(default)s)",
    )

    generate = subparsers.add_parser(
        "generate",
        parents=[common],
        help="generate one answer for a prompt and exit",
    )
    generate.add_argument("--prompt", required=True, help="user message text")
    generate.add_argument("--system", default=None, help="optional system message")
    generate.add_argument(
        "--no-stream",
        action="store_true",
        help="print the answer once it is complete instead of token by token",
    )
    generate.add_argument(
        "--json",
        action="store_true",
        help="print the whole result (text, token ids, stop reason, metrics) as JSON",
    )
    generate.add_argument(
        "--quiet", action="store_true", help="suppress the statistics block on stderr"
    )
    generate.set_defaults(handler=_cmd_generate)

    chat = subparsers.add_parser(
        "chat",
        parents=[common],
        help="interactive multi-turn chat (history is re-rendered each turn)",
    )
    chat.add_argument("--system", default=None, help="optional system message")
    chat.add_argument(
        "--quiet", action="store_true", help="suppress per-turn statistics on stderr"
    )
    chat.set_defaults(handler=_cmd_chat)

    inspect = subparsers.add_parser(
        "inspect",
        parents=[common],
        help="show model config, weight storage and KV cache sizing",
    )
    inspect.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of text"
    )
    inspect.set_defaults(handler=_cmd_inspect)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model:
        print(
            "error: --model DIR is required (or set SLOWQ_MODEL_DIR).\n"
            "Convert a checkpoint first: python tools/convert_model.py "
            "--repo Qwen/Qwen3-0.6B --out models/Qwen3-0.6B-slowq",
            file=sys.stderr,
        )
        return 2
    try:
        return int(args.handler(args))
    except SlowQError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # Reaches here only from validating user-supplied flags; the model,
        # weight and sampling paths all raise SlowQError subclasses instead.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def _sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )


def _load_engine(args: argparse.Namespace) -> Engine:
    print(f"loading model from {args.model} ...", file=sys.stderr)
    engine = Engine.from_directory(
        args.model,
        max_context=args.max_context,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    print(
        f"loaded in {engine.load_seconds:.2f} s "
        f"(weights {human_bytes(engine.model.weights.deduplicated_bytes())} logical, "
        f"KV {human_bytes(engine.kv_cache.allocated_bytes())} allocated for "
        f"{engine.max_context} tokens)",
        file=sys.stderr,
    )
    return engine


def _print_stats(rows: Sequence[tuple[str, str]], header: str = "stats") -> None:
    print(f"[{header}]", file=sys.stderr)
    for label, value in rows:
        print(f"  {label:<24} {value}", file=sys.stderr)


# ---- subcommands ------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    messages: list[ChatMessage] = []
    if args.system:
        messages.append(ChatMessage(role="system", content=args.system))
    messages.append(ChatMessage(role="user", content=args.prompt))
    sampling = _sampling_params(args)
    print(f"[sampler] {sampling.describe()}", file=sys.stderr)

    if args.json:
        result = engine.generate(messages, sampling=sampling, max_new_tokens=args.max_new_tokens)
        json.dump(result.as_dict(), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.no_stream:
        result = engine.generate(messages, sampling=sampling, max_new_tokens=args.max_new_tokens)
        sys.stdout.write(result.text + "\n")
    else:
        result = None
        for step in engine.stream(
            messages, sampling=sampling, max_new_tokens=args.max_new_tokens
        ):
            sys.stdout.write(step.text_delta)
            sys.stdout.flush()
            if step.result is not None:
                result = step.result
        sys.stdout.write("\n")
        if result is None:  # pragma: no cover - stream always ends with a result
            raise AssertionError("generation produced no result")

    if not args.quiet:
        _print_stats(result.metrics.report())
        print(f"  {'peak RSS (final)':<24} {human_bytes(peak_rss_bytes())}", file=sys.stderr)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    sampling = _sampling_params(args)
    history: list[ChatMessage] = []
    if args.system:
        history.append(ChatMessage(role="system", content=args.system))

    print(CHAT_BANNER, file=sys.stderr)
    print(f"[sampler] {sampling.describe()}", file=sys.stderr)

    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye", file=sys.stderr)
            return 0
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            print("bye", file=sys.stderr)
            return 0
        if user_input == "/clear":
            system = [m for m in history if m.role == "system"]
            history = system
            print("[history cleared]", file=sys.stderr)
            continue
        if user_input == "/history":
            for message in history:
                print(f"  {message.role}: {message.content}", file=sys.stderr)
            continue
        if user_input == "/help":
            print(CHAT_BANNER, file=sys.stderr)
            continue

        history.append(ChatMessage(role="user", content=user_input))
        try:
            answer_parts: list[str] = []
            sys.stdout.write("SlowQ> ")
            result = None
            for step in engine.stream(
                history, sampling=sampling, max_new_tokens=args.max_new_tokens
            ):
                sys.stdout.write(step.text_delta)
                sys.stdout.flush()
                answer_parts.append(step.text_delta)
                if step.result is not None:
                    result = step.result
            sys.stdout.write("\n")
            sys.stdout.flush()
        except KeyboardInterrupt:
            # The next request resets the cache, so cancelling here is safe.
            print("\n[cancelled]", file=sys.stderr)
            history.pop()
            continue

        assert result is not None
        history.append(ChatMessage(role="assistant", content=result.text))
        if not args.quiet:
            _print_stats(result.metrics.report(), header="stats")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    model = engine.model
    weights = model.weights
    config = model.config

    kv_bytes_per_token = config.kv_bytes_per_token()
    payload: dict[str, Any] = {
        "slowq_version": __version__,
        "model_directory": str(args.model),
        "source": dict(weights.manifest.source),
        "manifest_format_version": weights.manifest.format_version,
        "model_config": {label: value for label, value in config.describe()},
        "weights": {
            "tensor count": len(weights.tensor_names),
            "dtypes on disk": list(weights.dtype_names),
            "logical size (shared weights counted once)": human_bytes(
                weights.deduplicated_bytes()
            ),
            "size of .npy files": human_bytes(weights.file_bytes()),
            "shared (tied) tensors": dict(weights.aliases),
        },
        "kv_cache": {
            "max_context": engine.max_context,
            "bytes per token": kv_bytes_per_token,
            "allocated for max_context": human_bytes(engine.kv_cache.allocated_bytes()),
            "layout": "[num_kv_heads, max_context, head_dim] per layer, K post-norm+RoPE",
        },
        "generation": {
            "stop token ids": sorted(engine.stop_ids),
            "repository defaults": {
                "temperature": engine.generation_config.temperature,
                "top_k": engine.generation_config.top_k,
                "top_p": engine.generation_config.top_p,
                "do_sample": engine.generation_config.do_sample,
            },
            "prefill_chunk_size": engine.prefill_chunk_size,
        },
        "tokenizer": {label: value for label, value in engine.tokenizer.describe()},
        "environment": environment_info(),
        "load_seconds": engine.load_seconds,
    }

    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    sections = [
        ("model", payload["model_config"]),
        ("source", payload["source"]),
        ("weights", payload["weights"]),
        ("KV cache", payload["kv_cache"]),
        ("generation", payload["generation"]),
        ("tokenizer", payload["tokenizer"]),
        ("environment", _flatten_environment(payload["environment"])),
    ]
    print(f"慢炖 · SlowQ {__version__} — inspect")
    for title, entries in sections:
        print(f"\n[{title}]")
        for label, value in entries.items():
            print(f"  {label:<44} {value}")
    print(
        f"\npeak RSS after load: {human_bytes(peak_rss_bytes())}"
        "  (mmap'ed weight pages are counted only once actually read)"
    )
    return 0


def _flatten_environment(info: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat
