#!/usr/bin/env python3
"""A/B harness for the pre-encoded document index (round-1 optimisation 4).

    .venv/bin/python tools/perf/document_index_ab.py \
        --src . --model models/Qwen3-Embedding-0.6B-slowq \
        --out "$(.venv/bin/python tools/perf/out_guard.py fresh-root)/documents/opt.json"

``--out`` must name a path that does not exist yet: the harness refuses to replace a
measurement already on disk, and ``artifacts/perf-round1{,-rerun}`` are refused
outright as published evidence. See ``tools/perf/out_guard.py``.

Two ways to answer the same N queries over the same M documents:

search  ``EmbeddingEngine.search(query, corpus)`` once per query. Every call
        re-encodes every document, so the corpus is encoded N times over. This
        is the only option at the baseline commit, and the harness runs it there
        unchanged.
index   ``EmbeddingEngine.index_documents(corpus)`` once, then
        ``DocumentIndex.search(query)`` per query: the corpus is encoded once and
        each query costs one encode plus a matrix multiply.

The number of ``encode_document`` calls is counted on both paths, so "the index
encodes the corpus once" is a measurement rather than a claim. Every query's
ranking and similarity row are compared between the paths, so a speedup that came
from scoring something else cannot pass unnoticed.

The corpus is fixed text held in this file. Past the built-in passages it cycles
them with a numeric tag, which makes the ranking less interesting and the encode
cost per document identical — and the encode cost is what is being measured.

One process, strictly sequential.
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
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Sequence  # noqa: E402


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="repo root whose src/ to import")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="the library default; nothing in the fixed corpus is near it, and "
        "SlowQ refuses over-long text rather than truncating, so it cannot "
        "change the measured work",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=16,
        help="documents to retrieve over; the 12 built-in passages, then tagged "
        "repeats of them, so the encode cost per document stays comparable",
    )
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=3, help="timed rounds per path")
    parser.add_argument("--warmup", type=int, default=1)
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
from slowq.embedding import EmbeddingEngine  # noqa: E402
from slowq.metrics import blas_info, environment_info, peak_rss_bytes  # noqa: E402

assert slowq.__file__.startswith(SRC), f"imported {slowq.__file__}, expected under {SRC}"

PASSAGES: tuple[str, ...] = (
    "慢炖是一个只依赖 NumPy 的 CPU 推理引擎，权重以只读 mmap 打开，全程 FP32。",
    "KV cache 把自回归解码的每一步压缩成一次单 token 前向，代价是显存或内存。",
    "Prefill 阶段一次处理整个提示词，是矩阵乘；decode 阶段一次一个 token，是矩阵向量乘。",
    "分块 prefill 让注意力分数矩阵的大小有上界，代价是浮点求和顺序的微小变化。",
    "Grouped-query attention 让多个查询头共享一组 key/value 头，减少了 KV cache 的体积。",
    "The vision tower of an image-text model runs once per image, not once per question.",
    "A read-only memory map is never written to, so two engines can share the same pages.",
    "Cosine similarity between two L2-normalised vectors is exactly their dot product.",
    "Last-token pooling takes the final valid position's hidden state as the sentence vector.",
    "Chunked prefill and single-shot prefill agree to within a relative L2 of 2e-4.",
    "Rounding modes and summation order are why two mathematically equal kernels differ.",
    "A cache that cannot say what it is keyed on is a source of wrong answers, not speed.",
)

QUERIES: tuple[str, ...] = (
    "慢炖是怎么加载权重的？",
    "KV cache 为什么能加速解码？",
    "How many times does the vision tower run per question?",
    "什么是 last-token pooling？",
    "Why do two equal kernels produce different numbers?",
    "What makes a cache key safe?",
    "分块 prefill 的代价是什么？",
    "How is cosine similarity computed here?",
)


def corpus_of(size: int) -> tuple[str, ...]:
    """``size`` documents, cycling the built-in passages with a numeric tag."""
    if size < 1:
        raise SystemExit("--corpus-size must be >= 1")
    out = []
    for index in range(size):
        passage = PASSAGES[index % len(PASSAGES)]
        round_number = index // len(PASSAGES)
        out.append(passage if round_number == 0 else f"[{round_number}] {passage}")
    return tuple(out)


def summarise(values: Sequence[Any]) -> dict[str, float]:
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


def index_supported() -> bool:
    return hasattr(EmbeddingEngine, "index_documents")


def weights_fingerprint(engine: Any) -> list[str] | None:
    """The checkpoint identity the index is keyed on, or ``None`` on a tree without it.

    ``Weights.fingerprint`` was added by this round's work, so the baseline tree
    cannot report one. Recording ``None`` keeps the two payloads the same shape and
    makes the difference readable, where calling it unconditionally would just
    crash the baseline run before it measured anything.
    """
    fingerprint = getattr(engine.model.weights, "fingerprint", None)
    if fingerprint is None:
        return None
    return [str(part) for part in fingerprint()[:-1]]


class Encoder:
    """Counts every encode the engine is asked for, passing each one through."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.documents = 0
        self.queries = 0
        self.document_seconds = 0.0
        self.query_seconds = 0.0
        self._real_document = engine.encode_document
        self._real_query = engine.encode_query
        engine.encode_document = self._document  # type: ignore[method-assign]
        engine.encode_query = self._query  # type: ignore[method-assign]

    def _document(self, text: str):
        started = time.perf_counter()
        result = self._real_document(text)
        self.document_seconds += time.perf_counter() - started
        self.documents += 1
        return result

    def _query(self, text: str, **kwargs):
        started = time.perf_counter()
        result = self._real_query(text, **kwargs)
        self.query_seconds += time.perf_counter() - started
        self.queries += 1
        return result

    def reset(self) -> None:
        self.documents = self.queries = 0
        self.document_seconds = self.query_seconds = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "document_encodes": self.documents,
            "query_encodes": self.queries,
            "document_encode_seconds": self.document_seconds,
            "query_encode_seconds": self.query_seconds,
        }


def run_search_path(engine: Any, counter: Encoder, corpus: tuple[str, ...],
                    queries: tuple[str, ...]) -> dict[str, Any]:
    """N queries, each re-encoding the whole corpus."""
    rounds = []
    for _ in range(1 + ARGS.warmup + ARGS.repeat):
        counter.reset()
        started = time.perf_counter()
        rows = []
        for query in queries:
            one = time.perf_counter()
            batch, ranking = engine.search(query, corpus)
            rows.append(
                {
                    "wall_seconds": time.perf_counter() - one,
                    "ranking": [(int(i), float(s)) for i, s in ranking],
                    "similarity": batch.similarity.tolist(),
                    "query_vector": batch.results[0].vector.tolist(),
                    "query_input_ids": list(batch.results[0].input_ids),
                }
            )
        rounds.append(
            {
                "wall_seconds": time.perf_counter() - started,
                "rows": rows,
                "counts": counter.snapshot(),
                "peak_rss_bytes": peak_rss_bytes(),
            }
        )
    return summarise_rounds(rounds, queries)


def run_index_path(engine: Any, counter: Encoder, corpus: tuple[str, ...],
                   queries: tuple[str, ...]) -> dict[str, Any]:
    """One build, then N queries against the stored vectors."""
    if not index_supported():
        return {"unsupported": "this tree has no EmbeddingEngine.index_documents"}

    builds = []
    rounds = []
    for _ in range(1 + ARGS.warmup + ARGS.repeat):
        counter.reset()
        build_started = time.perf_counter()
        index = engine.index_documents(corpus)
        build_wall = time.perf_counter() - build_started
        builds.append(
            {
                "wall_seconds": build_wall,
                "build_seconds": index.build_seconds,
                "counts": counter.snapshot(),
                "stats": index.stats(),
                "vectors_bytes": int(index.vectors.nbytes),
            }
        )

        counter.reset()
        started = time.perf_counter()
        rows = []
        for query in queries:
            one = time.perf_counter()
            batch, ranking = index.search(query)
            rows.append(
                {
                    "wall_seconds": time.perf_counter() - one,
                    "ranking": [(int(i), float(s)) for i, s in ranking],
                    "similarity": batch.similarity.tolist(),
                    "query_vector": batch.results[0].vector.tolist(),
                    "query_input_ids": list(batch.results[0].input_ids),
                }
            )
        rounds.append(
            {
                "wall_seconds": time.perf_counter() - started,
                "rows": rows,
                "counts": counter.snapshot(),
                "peak_rss_bytes": peak_rss_bytes(),
                "build": builds[-1],
            }
        )

    out = summarise_rounds(rounds, queries)
    out["build"] = {
        "wall_seconds": summarise([b["wall_seconds"] for b in builds]),
        "build_seconds": summarise([b["build_seconds"] for b in builds]),
        "vectors_bytes": builds[-1]["vectors_bytes"],
        "stats": builds[-1]["stats"],
        # The build is the corpus encode, and it happens once per round here only
        # because each round models a fresh process. A real caller builds once and
        # queries for the life of the index, so the per-query numbers below are
        # the ones to compare against the search path's.
        "document_encodes_per_build": builds[-1]["counts"]["document_encodes"],
    }
    out["with_build_amortised"] = {
        "round_wall_seconds": summarise(
            [r["wall_seconds"] + r["build"]["wall_seconds"] for r in rounds]
        ),
    }
    return out


def summarise_rounds(rounds: list[dict[str, Any]], queries: tuple[str, ...]) -> dict[str, Any]:
    """Drop the first round (it is also the process's first forward), keep the rest."""
    timed = rounds[1 + ARGS.warmup:]
    out: dict[str, Any] = {
        "rounds_total": len(rounds),
        "rounds_dropped_as_warmup": len(rounds) - len(timed),
        "round_wall_seconds": summarise([r["wall_seconds"] for r in timed]),
        "peak_rss_bytes": summarise([r["peak_rss_bytes"] for r in timed]),
        "counts_per_round": timed[-1]["counts"],
        "per_query_wall_seconds": {},
    }
    for index, query in enumerate(queries):
        out["per_query_wall_seconds"][str(index)] = {
            "query": query,
            **summarise([r["rows"][index]["wall_seconds"] for r in timed]),
        }
    out["rounds"] = [
        {
            "wall_seconds": r["wall_seconds"],
            "counts": r["counts"],
            "rows": r["rows"],
        }
        for r in rounds
    ]
    return out


def compare(search_out: dict[str, Any], index_out: dict[str, Any],
            queries: tuple[str, ...]) -> dict[str, Any]:
    """Rankings, scores and query vectors, compared between the two paths."""
    if "rounds" not in index_out or "rounds" not in search_out:
        return {"compared": False}
    mismatches = []
    for round_index, (a, b) in enumerate(
        zip(search_out["rounds"], index_out["rounds"])
    ):
        for query_index, query in enumerate(queries):
            left, right = a["rows"][query_index], b["rows"][query_index]
            for field in ("ranking", "similarity", "query_vector", "query_input_ids"):
                if left[field] != right[field]:
                    mismatches.append(
                        {"round": round_index, "query": query, "field": field}
                    )
    compared = len(search_out["rounds"]) * len(queries)
    return {
        "compared": True,
        "queries_compared": compared,
        "fields": ["ranking", "similarity", "query_vector", "query_input_ids"],
        "comparison": "exact equality on the JSON round-tripped values",
        "mismatches": mismatches,
        "all_identical": not mismatches,
    }


def main() -> int:
    # Before the corpus build and the model load; see out_guard.preflight.
    out_guard.preflight(ARGS.out)
    started = time.perf_counter()
    corpus = corpus_of(ARGS.corpus_size)
    queries = QUERIES[: ARGS.queries]
    if not queries:
        raise SystemExit("--queries must be >= 1")

    engine = EmbeddingEngine.from_directory(
        ARGS.model,
        max_length=ARGS.max_length,
        prefill_chunk_size=ARGS.prefill_chunk_size,
    )
    load_seconds = time.perf_counter() - started
    counter = Encoder(engine)
    tokens = [engine.encode_document(text).num_tokens for text in corpus]
    counter.reset()
    print(
        f"document_index_ab: src={SRC}\n         model={ARGS.model} "
        f"threads={REQUESTED} numpy={np.__version__} blas={blas_info().get('name')} "
        f"index_supported={index_supported()} load={load_seconds:.3f}s "
        f"corpus={len(corpus)} docs / {sum(tokens)} tokens queries={len(queries)} "
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
        "load_seconds": load_seconds,
        "rss_after_load_bytes": peak_rss_bytes(),
        "index_supported": index_supported(),
        "engine": {
            "dimension": engine.dimension,
            "max_length": engine.max_length,
            "prefill_chunk_size": engine.prefill_chunk_size,
            "weights_fingerprint": weights_fingerprint(engine),
        },
        "corpus": {
            "documents": len(corpus),
            "tokens_total": sum(tokens),
            "tokens_per_document": {
                "median": statistics.median(tokens),
                "min": min(tokens),
                "max": max(tokens),
            },
            "queries": list(queries),
        },
    }

    print("[search]", file=sys.stderr)
    payload["search"] = run_search_path(engine, counter, corpus, queries)
    print(
        f"  search   round {payload['search']['round_wall_seconds']['median'] * 1e3:>9.1f} ms  "
        f"document encodes/round {payload['search']['counts_per_round']['document_encodes']}",
        file=sys.stderr,
    )

    print("[index]", file=sys.stderr)
    payload["index"] = run_index_path(engine, counter, corpus, queries)
    if "rounds" in payload["index"]:
        print(
            f"  index    round {payload['index']['round_wall_seconds']['median'] * 1e3:>9.1f} ms  "
            f"document encodes/round "
            f"{payload['index']['counts_per_round']['document_encodes']}  "
            f"build {payload['index']['build']['wall_seconds']['median'] * 1e3:>8.1f} ms",
            file=sys.stderr,
        )

    payload["agreement"] = compare(payload["search"], payload["index"], queries)

    if "rounds" in payload["index"]:
        search_round = payload["search"]["round_wall_seconds"]["median"]
        index_round = payload["index"]["round_wall_seconds"]["median"]
        build_round = payload["index"]["build"]["wall_seconds"]["median"]
        payload["headline"] = {
            "queries_per_round": len(queries),
            "documents_per_query_without_index": payload["search"][
                "counts_per_round"
            ]["document_encodes"]
            / len(queries),
            "documents_per_query_with_index": payload["index"]["counts_per_round"][
                "document_encodes"
            ]
            / len(queries),
            "round_seconds": {
                "search": search_round,
                "index_queries_only": index_round,
                "index_including_a_rebuild_every_round": index_round + build_round,
            },
            "speedup": {
                "queries_only": search_round / index_round if index_round else float("nan"),
                # The pessimistic bound: even if a caller threw the index away and
                # rebuilt it for every round, this is what it would save.
                "including_a_rebuild_every_round": (
                    search_round / (index_round + build_round)
                    if index_round + build_round
                    else float("nan")
                ),
            },
            "agreement": payload["agreement"].get("all_identical"),
        }
        headline = payload["headline"]
        print(
            f"  {len(queries)} queries over {len(corpus)} documents: "
            f"search {search_round * 1e3:.1f} ms -> index {index_round * 1e3:.1f} ms "
            f"({headline['speedup']['queries_only']:.2f}x), "
            f"{build_round * 1e3:.1f} ms to build "
            f"({headline['speedup']['including_a_rebuild_every_round']:.2f}x even "
            "rebuilding every round), "
            f"document encodes "
            f"{payload['search']['counts_per_round']['document_encodes']} -> "
            f"{payload['index']['counts_per_round']['document_encodes']}, "
            f"identical={headline['agreement']}",
            file=sys.stderr,
        )

    payload["rss_final_bytes"] = peak_rss_bytes()
    payload["wall_seconds"] = time.perf_counter() - started

    if ARGS.out:
        # Refuses an existing path rather than truncating it; see out_guard.py.
        out = out_guard.write_json(ARGS.out, payload)
        print(f"wrote {out_guard.display(out)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
