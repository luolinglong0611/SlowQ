"""Timing, memory and token statistics.

Definitions used everywhere in SlowQ (also documented in docs/architecture.md):

``load_seconds``
    Building the engine from an already-converted directory: reading
    ``config.json``/``manifest.json``, memory-mapping the weights, compiling the
    chat template. Excludes downloading and converting the checkpoint. Because
    weights are mmap'ed, this also excludes paging them in — the first forward
    pays that cost, which is why ``prefill_seconds`` is reported separately.

``tokenize_seconds``
    Chat-template rendering plus BPE encoding of the prompt.

``vision_encode_seconds``
    Image+text requests only: running every frame of the image through the vision
    tower and the connector. It is part of time-to-first-token and is reported
    separately because it is not prefill — for a 17-frame image it is by far the
    largest single cost of the request. Resizing and normalising the frames
    happens earlier, in prompt construction, and is not included here.

``prefill_seconds``
    Wall time of all prefill forwards (every chunk), including the KV writes.

``first_token_seconds``
    Time-to-first-token: ``tokenize_seconds`` + ``prefill_seconds`` + the first
    sampling step, measured from the start of the request. Excludes model load.

``decode_seconds`` / ``decode_tokens``
    Everything after the first token. ``decode_tokens`` therefore excludes the
    first generated token, and ``decode_tokens_per_second`` is a steady-state
    rate that is not distorted by prefill.

``prompt_tokens`` counts the rendered chat template, not the raw user text, so
prefill throughput is measured against what the model actually consumed.

Memory is reported as three separate numbers because they mean different things:
``weight_logical_bytes`` is the size of the weight arrays, ``kv_allocated_bytes``
is the reserved cache, and ``peak_rss_bytes`` is what the OS says the process
really touched. Summing ``nbytes`` over mmap'ed arrays is *not* RSS: pages that
were never read are not resident.
"""

from __future__ import annotations

import os
import platform
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


def peak_rss_bytes() -> int:
    """Peak resident set size of this process.

    ``ru_maxrss`` is bytes on macOS and kibibytes on Linux; normalise both.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def human_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


@contextmanager
def stopwatch() -> Iterator[list[float]]:
    """``with stopwatch() as elapsed: ...`` -> ``elapsed[0]`` in seconds."""
    box: list[float] = []
    started = time.perf_counter()
    try:
        yield box
    finally:
        box.append(time.perf_counter() - started)


def thread_settings() -> dict[str, str | None]:
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    return {name: os.environ.get(name) for name in names}


def blas_info() -> dict[str, Any]:
    try:
        build = np.show_config("dicts")["Build Dependencies"]["blas"]
    except Exception:  # pragma: no cover - NumPy always ships this today
        return {"error": "unavailable"}
    return {key: build.get(key) for key in ("name", "version", "found", "lib directory")}


def cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            import subprocess

            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip() or platform.processor() or "unknown"
        except Exception:  # pragma: no cover
            return platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:  # pragma: no cover
        pass
    return platform.processor() or "unknown"


def environment_info() -> dict[str, Any]:
    return {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "blas": blas_info(),
        "thread_env": thread_settings(),
    }


@dataclass
class GenerationMetrics:
    """Per-request statistics. All durations are seconds from ``time.perf_counter``."""

    tokenize_seconds: float = 0.0
    vision_encode_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    first_token_seconds: float = 0.0
    total_seconds: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_chunks: int = 0
    stop_reason: str = ""
    weight_logical_bytes: int = 0
    kv_allocated_bytes: int = 0
    peak_rss_bytes: int = field(default_factory=peak_rss_bytes)

    @property
    def decode_tokens(self) -> int:
        """Tokens produced after the first one — the steady-state sample."""
        return max(self.generated_tokens - 1, 0)

    @property
    def decode_tokens_per_second(self) -> float:
        if self.decode_seconds <= 0 or self.decode_tokens == 0:
            return float("nan")
        return self.decode_tokens / self.decode_seconds

    @property
    def prefill_tokens_per_second(self) -> float:
        if self.prefill_seconds <= 0 or self.prompt_tokens == 0:
            return float("nan")
        return self.prompt_tokens / self.prefill_seconds

    def report(self) -> list[tuple[str, str]]:
        rows = [
            ("stop reason", self.stop_reason),
            ("prompt tokens", f"{self.prompt_tokens} ({self.prefill_chunks} chunk(s))"),
            ("generated tokens", str(self.generated_tokens)),
            ("tokenize", f"{self.tokenize_seconds * 1e3:.1f} ms"),
        ]
        if self.vision_encode_seconds > 0:
            # Only image+text requests have one, so a text run's report is
            # unchanged rather than gaining a permanent row of zeros.
            rows.append(("vision encode", f"{self.vision_encode_seconds * 1e3:.1f} ms"))
        rows.extend(
            [
                ("prefill", f"{self.prefill_seconds * 1e3:.1f} ms"),
                ("first token latency", f"{self.first_token_seconds * 1e3:.1f} ms"),
                ("decode", f"{self.decode_seconds * 1e3:.1f} ms"),
                ("decode speed", f"{self.decode_tokens_per_second:.2f} tok/s"),
                ("total", f"{self.total_seconds * 1e3:.1f} ms"),
                ("weights (logical)", human_bytes(self.weight_logical_bytes)),
                ("KV cache (allocated)", human_bytes(self.kv_allocated_bytes)),
                ("peak RSS", human_bytes(self.peak_rss_bytes)),
            ]
        )
        return rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokenize_seconds": self.tokenize_seconds,
            "vision_encode_seconds": self.vision_encode_seconds,
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "first_token_seconds": self.first_token_seconds,
            "total_seconds": self.total_seconds,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "decode_tokens": self.decode_tokens,
            "prefill_chunks": self.prefill_chunks,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "stop_reason": self.stop_reason,
            "weight_logical_bytes": self.weight_logical_bytes,
            "kv_allocated_bytes": self.kv_allocated_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
        }
