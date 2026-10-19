"""Sampling: greedy, temperature, top-k, top-p.

Filter order matches ``transformers``' logits-warper pipeline: temperature, then
top-k, then top-p. ``temperature == 0`` means greedy (argmax) and skips the rest,
which also keeps ``top_k``/``top_p`` from silently mattering in greedy mode.

The sampler owns a private ``numpy.random.Generator``; the global NumPy RNG is
never touched, so a fixed ``seed`` reproduces the same token sequence and other
parts of the program cannot shift it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import SamplingError
from .ops import F32, NEG_INF_F32, softmax


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    def validate(self) -> "SamplingParams":
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
            raise SamplingError(f"temperature must be a number, got {self.temperature!r}")
        if self.temperature < 0 or not np.isfinite(self.temperature):
            raise SamplingError(
                f"temperature must be 0 (greedy) or a positive finite number, "
                f"got {self.temperature!r}"
            )
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise SamplingError(f"top_k must be an int, got {self.top_k!r}")
        if self.top_k < 0:
            raise SamplingError(f"top_k must be >= 0 (0 disables it), got {self.top_k}")
        if not isinstance(self.top_p, (int, float)) or isinstance(self.top_p, bool):
            raise SamplingError(f"top_p must be a number, got {self.top_p!r}")
        if not (0.0 < self.top_p <= 1.0) or not np.isfinite(self.top_p):
            raise SamplingError(f"top_p must be in (0, 1], got {self.top_p!r}")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise SamplingError(f"seed must be an int or None, got {self.seed!r}")
        return self

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0

    def describe(self) -> str:
        mode = "greedy" if self.is_greedy else f"temperature={self.temperature:g}"
        if not self.is_greedy:
            if self.top_k:
                mode += f", top_k={self.top_k}"
            if self.top_p < 1.0:
                mode += f", top_p={self.top_p:g}"
            mode += f", seed={self.seed}"
        return mode


class Sampler:
    """Turns a ``[vocab_size]`` FP32 logit vector into one token id."""

    def __init__(self, params: SamplingParams | None = None) -> None:
        self.params = (params or SamplingParams()).validate()
        self._rng = self._build_rng(self.params.seed)

    @staticmethod
    def _build_rng(seed: int | None) -> np.random.Generator:
        # Explicit PCG64 so the stream does not change if NumPy's default changes.
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))

    def reseed(self, seed: int | None) -> None:
        """Restart the RNG, e.g. to replay the same conversation."""
        self._rng = self._build_rng(seed)

    @property
    def is_greedy(self) -> bool:
        return self.params.is_greedy

    def sample(self, logits: np.ndarray) -> int:
        array = np.asarray(logits)
        if array.ndim != 1:
            raise SamplingError(f"logits must be 1-D [vocab], got shape {array.shape}")
        array = array.astype(np.float32, copy=False)
        if array.size == 0:
            raise SamplingError("logits are empty; nothing to sample from")
        if not np.all(np.isfinite(array)):
            bad = int(np.count_nonzero(~np.isfinite(array)))
            raise SamplingError(
                f"{bad} of {array.size} logits are not finite (NaN/inf). Refusing to "
                "sample; this usually means the forward pass overflowed."
            )

        if self.params.is_greedy:
            # Ties go to the lowest token id (np.argmax returns the first maximum).
            return int(np.argmax(array))

        scaled = array / F32(self.params.temperature)
        scaled = self._apply_top_k(scaled, self.params.top_k)
        scaled = self._apply_top_p(scaled, self.params.top_p)
        probabilities = self._probabilities(scaled)

        cumulative = np.cumsum(probabilities, dtype=np.float64)
        cumulative /= cumulative[-1]
        draw = self._rng.random(dtype=np.float64)
        index = int(np.searchsorted(cumulative, draw, side="right"))
        return min(index, probabilities.shape[0] - 1)

    # ---- filters ------------------------------------------------------------

    @staticmethod
    def _apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
        if top_k == 0 or top_k >= logits.size:
            return logits
        threshold = np.partition(logits, logits.size - top_k)[logits.size - top_k]
        return np.where(logits < threshold, NEG_INF_F32, logits).astype(F32, copy=False)

    @staticmethod
    def _apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
        if top_p >= 1.0:
            return logits
        # Stable sort so the kept set is deterministic when logits tie.
        descending = np.argsort(logits, kind="stable")[::-1]
        sorted_logits = logits[descending]
        cumulative = np.cumsum(softmax(sorted_logits), dtype=np.float32)
        remove = cumulative > F32(top_p)
        # Keep the token that crosses the threshold, drop everything after it.
        remove[1:] = remove[:-1].copy()
        remove[0] = False
        if remove.all():
            raise SamplingError(
                f"top_p={top_p:g} removed every candidate token; raise top_p or top_k"
            )
        kept = descending[~remove]
        masked = np.full(logits.shape, NEG_INF_F32, dtype=np.float32)
        masked[kept] = logits[kept]
        return masked

    @staticmethod
    def _probabilities(logits: np.ndarray) -> np.ndarray:
        probabilities = softmax(logits)
        if not np.all(np.isfinite(probabilities)):
            raise SamplingError("sampling produced non-finite probabilities")
        total = probabilities.sum(dtype=np.float64)
        if not np.isfinite(total) or total <= 0.0:
            raise SamplingError(
                f"candidate probabilities sum to {total!r}; the candidate set is empty"
            )
        return probabilities
