"""Command line interface: ``generate``, ``chat``, ``embed``, ``vlm``, ``inspect``.

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
from .config import TASK_IMAGE_TEXT_GENERATION, TASK_TEXT_EMBEDDING
from .embedding import DEFAULT_QUERY_INSTRUCTION, EmbeddingEngine, similarity_matrix
from .engine import (
    DEFAULT_MAX_CONTEXT,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PREFILL_CHUNK_SIZE,
    Engine,
    recorded_task,
)
from .errors import SlowQError
from .metrics import environment_info, human_bytes, peak_rss_bytes, stopwatch
from .sampling import SamplingParams
from .tokenizer import ChatMessage
from .weights import MANIFEST_FILENAME

CHAT_BANNER = """慢炖 · SlowQ chat  (CPU, FP32, NumPy only)
Commands: /clear reset history · /history show history · /help · /exit (or Ctrl-D)
Ctrl-C during an answer cancels that turn and returns to the prompt.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slowq",
        description="慢炖 · SlowQ — slowly stew small-model tokens on your CPU.",
    )
    parser.add_argument("--version", action="version", version=f"slowq {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    with_model = argparse.ArgumentParser(add_help=False)
    with_model.add_argument(
        "--model",
        default=os.environ.get("SLOWQ_MODEL_DIR"),
        metavar="DIR",
        help="converted model directory (the one containing "
        f"{MANIFEST_FILENAME}); defaults to $SLOWQ_MODEL_DIR",
    )

    common = argparse.ArgumentParser(add_help=False, parents=[with_model])
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

    embed = subparsers.add_parser(
        "embed",
        parents=[with_model],
        help="encode text into L2-normalised vectors (needs an embedding checkpoint)",
        description=(
            "Encode text with a converted embedding checkpoint. Queries get the "
            "official 'Instruct: …\\nQuery:…' prefix, documents do not. Texts are "
            "encoded one at a time — SlowQ runs batch size 1 — and the combined "
            "result includes every query/document cosine similarity."
        ),
    )
    embed.add_argument(
        "--query",
        action="append",
        default=[],
        metavar="TEXT",
        help="a search query; repeatable. Gets the task-instruction prefix.",
    )
    embed.add_argument(
        "--document",
        action="append",
        default=[],
        metavar="TEXT",
        help="a passage to be retrieved; repeatable. Gets no prefix.",
    )
    embed.add_argument(
        "--instruction",
        default=DEFAULT_QUERY_INSTRUCTION,
        help="task description prepended to every --query (default: the official "
        "retrieval instruction)",
    )
    embed.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="refuse a text longer than this many tokens (default: %(default)s; "
        "SlowQ never truncates)",
    )
    embed.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=DEFAULT_PREFILL_CHUNK_SIZE,
        help="tokens per prefill forward (default: %(default)s)",
    )
    embed.add_argument(
        "--show-dims",
        type=int,
        default=8,
        help="how many vector components to print per text (default: %(default)s; "
        "--json always prints the whole vector)",
    )
    embed.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of text"
    )
    embed.set_defaults(handler=_cmd_embed)

    vlm = subparsers.add_parser(
        "vlm",
        parents=[common],
        help="answer a question about one local image (needs an image+text checkpoint)",
        description=(
            "Run one local image plus a text question through a converted "
            "image+text checkpoint. The image is cut into the official crops plus "
            "a global frame, encoded by the vision tower, and merged onto the "
            "prompt's <image> placeholders before prefill; decoding afterwards is "
            "the ordinary cached text loop. Reading the image needs Pillow "
            "(`pip install 'slowq[vision]'`)."
        ),
    )
    vlm.add_argument("--image", required=True, metavar="PATH", help="local image file")
    vlm.add_argument("--question", required=True, help="user message text")
    vlm.add_argument("--system", default=None, help="optional system message")
    vlm.add_argument(
        "--no-stream",
        action="store_true",
        help="print the answer once it is complete instead of token by token",
    )
    vlm.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the expanded prompt (crop tags and every <image> token) to stderr",
    )
    vlm.add_argument(
        "--json",
        action="store_true",
        help="print the whole result (text, token ids, stop reason, metrics) as JSON",
    )
    vlm.add_argument(
        "--quiet", action="store_true", help="suppress the statistics block on stderr"
    )
    vlm.set_defaults(handler=_cmd_vlm)

    inspect = subparsers.add_parser(
        "inspect",
        parents=[common],
        help="show model config, weight storage and KV cache sizing",
    )
    inspect.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of text"
    )
    inspect.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="only used when --model is an embedding directory (default: %(default)s)",
    )
    inspect.set_defaults(handler=_cmd_inspect)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model:
        print(
            "error: --model DIR is required (or set SLOWQ_MODEL_DIR).\n"
            "Convert a checkpoint first: python tools/convert_model.py "
            "--repo Qwen/Qwen3-0.6B --out models/Qwen3-0.6B-slowq\n"
            "See docs/models.md for every model SlowQ converts.",
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


def _one_line(text: str, limit: int = 60) -> str:
    """Collapse a prompt to a single displayable line."""
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else flattened[: limit - 1] + "…"


def _load_embedding_engine(args: argparse.Namespace) -> EmbeddingEngine:
    print(f"loading model from {args.model} ...", file=sys.stderr)
    engine = EmbeddingEngine.from_directory(
        args.model,
        max_length=args.max_length,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    print(
        f"loaded in {engine.load_seconds:.2f} s "
        f"(weights {human_bytes(engine.model.weights.deduplicated_bytes())} logical, "
        f"{engine.dimension}-dimensional vectors)",
        file=sys.stderr,
    )
    return engine


def _cmd_embed(args: argparse.Namespace) -> int:
    if not args.query and not args.document:
        print("error: pass at least one --query or --document to encode", file=sys.stderr)
        return 2
    if args.show_dims < 0:
        raise ValueError(f"--show-dims must be >= 0, got {args.show_dims}")

    engine = _load_embedding_engine(args)
    queries = [engine.encode_query(text, instruction=args.instruction) for text in args.query]
    documents = [engine.encode_document(text) for text in args.document]
    scores = (
        similarity_matrix([r.vector for r in queries], [r.vector for r in documents])
        if queries and documents
        else None
    )

    if args.json:
        json.dump(
            {
                "model_directory": str(args.model),
                "dimension": engine.dimension,
                "instruction": args.instruction,
                "queries": [result.as_dict() for result in queries],
                "documents": [result.as_dict() for result in documents],
                "similarity": (
                    None
                    if scores is None
                    else [[float(value) for value in row] for row in scores]
                ),
            },
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    for label, results in (("query", queries), ("document", documents)):
        for index, result in enumerate(results):
            shown = ", ".join(f"{value:+.6f}" for value in result.vector[: args.show_dims])
            ellipsis = ", …" if args.show_dims < result.dimension else ""
            print(f"[{label} {index}] {result.num_tokens} tokens · dim {result.dimension}")
            print(f"  text    {_one_line(result.text)}")
            print(f"  vector  {shown}{ellipsis}")

    if scores is not None:
        print("\n[cosine similarity]  (last-token pooled, L2-normalised, dot product)")
        for i, query in enumerate(args.query):
            print(f"  query {i}: {_one_line(query)}")
            ranked = sorted(
                ((j, float(scores[i][j])) for j in range(len(documents))),
                key=lambda item: item[1],
                reverse=True,
            )
            for position, (j, score) in enumerate(ranked):
                marker = "  ← best" if position == 0 else ""
                print(f"      {score:+.4f}  document {j}: {_one_line(args.document[j])}{marker}")

    print(
        f"\npeak RSS: {human_bytes(peak_rss_bytes())}  "
        f"({len(queries) + len(documents)} text(s) encoded one at a time)",
        file=sys.stderr,
    )
    return 0


def _load_vision_engine(args: argparse.Namespace) -> Any:
    """Load a converted image+text directory.

    ``slowq.vlm`` is imported here rather than at module scope so that the text
    commands keep working in an environment without Pillow — the module itself
    imports Pillow lazily, but there is no reason to load 1300 lines of vision
    code to answer a text prompt.
    """
    from .vlm import VisionEngine

    print(f"loading model from {args.model} ...", file=sys.stderr)
    engine = VisionEngine.from_directory(
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


def _image_rows(prompt: Any) -> list[tuple[str, Any]]:
    """What the official pipeline did to one image, before any model ran."""
    image = prompt.image
    grid = "no split (image fits one crop)" if not image.rows else f"{image.rows} x {image.cols} crops"
    return [
        ("original (HxW)", f"{image.original_size[0]}x{image.original_size[1]}"),
        ("resized (HxW)", f"{image.resized_size[0]}x{image.resized_size[1]}"),
        ("encoder (HxW)", f"{image.encoder_size[0]}x{image.encoder_size[1]}"),
        ("crop grid", grid),
        ("frames (crops + global)", str(image.num_frames)),
        ("image tokens", f"{image.num_image_tokens} ({image.num_frames} x {image.image_seq_len})"),
        ("prompt tokens", f"{len(prompt.token_ids)} total"),
    ]


def _cmd_vlm(args: argparse.Namespace) -> int:
    engine = _load_vision_engine(args)
    sampling = _sampling_params(args)

    with stopwatch() as prep:
        prompt = engine.build_prompt(args.image, args.question, system=args.system)
    _print_stats(
        [*_image_rows(prompt), ("image prep", f"{prep[0] * 1e3:.1f} ms")], header="image"
    )
    if args.show_prompt:
        print(f"[prompt]\n{prompt.text}", file=sys.stderr)
    print(f"[sampler] {sampling.describe()}", file=sys.stderr)

    if args.json:
        result = engine.generate_prompt(
            prompt, sampling=sampling, max_new_tokens=args.max_new_tokens
        )
        payload = result.as_dict()
        payload["image"] = dict(prompt.image.as_dict(), prep_seconds=prep[0])
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.no_stream:
        result = engine.generate_prompt(
            prompt, sampling=sampling, max_new_tokens=args.max_new_tokens
        )
        sys.stdout.write(result.text + "\n")
    else:
        result = None
        for step in engine.stream_prompt(
            prompt, sampling=sampling, max_new_tokens=args.max_new_tokens
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


def _inspect_embedding(args: argparse.Namespace) -> int:
    """``inspect`` for an embedding directory: no KV cache, no stop tokens."""
    engine = _load_embedding_engine(args)
    model = engine.model
    weights = model.weights
    config = model.config

    payload: dict[str, Any] = {
        "slowq_version": __version__,
        "model_directory": str(args.model),
        "source": dict(weights.manifest.source),
        "manifest_format_version": weights.manifest.format_version,
        "manifest_task": weights.manifest.task,
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
        "embedding": {
            "dimension": engine.dimension,
            "pooling": "last valid token of the final RMSNorm output",
            "normalisation": "L2, so a dot product is the cosine similarity",
            "query prefix": f"Instruct: {{task}}\\nQuery:{{query}}",
            "default instruction": DEFAULT_QUERY_INSTRUCTION,
            "max_length": engine.max_length,
            "prefill_chunk_size": engine.prefill_chunk_size,
            "output projection": "none (this checkpoint has no lm_head)",
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
        ("embedding", payload["embedding"]),
        ("tokenizer", payload["tokenizer"]),
        ("environment", _flatten_environment(payload["environment"])),
    ]
    print(f"慢炖 · SlowQ {__version__} — inspect")
    for title, entries in sections:
        print(f"\n[{title}]")
        for label, value in entries.items():
            print(f"  {label:<44} {value}")
    print(f"\npeak RSS after load: {human_bytes(peak_rss_bytes())}")
    return 0


def _inspect_vision(args: argparse.Namespace) -> int:
    """``inspect`` for an image+text directory: vision tower beside the decoder."""
    engine = _load_vision_engine(args)
    vlm_config = engine.vlm_config
    model = engine.model
    weights = model.weights
    config = model.config

    payload: dict[str, Any] = {
        "slowq_version": __version__,
        "model_directory": str(args.model),
        "source": dict(weights.manifest.source),
        "manifest_format_version": weights.manifest.format_version,
        "manifest_task": weights.manifest.task,
        "manifest_model_family": weights.manifest.model_family,
        "model_config": {label: value for label, value in vlm_config.describe()},
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
            "bytes per token": config.kv_bytes_per_token(),
            "allocated for max_context": human_bytes(engine.kv_cache.allocated_bytes()),
        },
        "generation": {
            "stop token ids": sorted(engine.stop_ids),
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
            print(f"  {str(label):<44} {value}")
    print(f"\npeak RSS after load: {human_bytes(peak_rss_bytes())}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    task = recorded_task(args.model)
    if task == TASK_TEXT_EMBEDDING:
        return _inspect_embedding(args)
    if task == TASK_IMAGE_TEXT_GENERATION:
        return _inspect_vision(args)
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
