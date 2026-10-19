"""Generation: chunked prefill, decode loop, stop conditions, streaming.

Layering: ``stream_ids`` is the only place that runs the loop. Every other entry
point (``generate``, ``generate_text``, ``stream``) resolves text into token IDs
and delegates, so there is exactly one implementation of "prefill, then decode
one token at a time".

The KV cache is owned by the engine and reset at the *start* of each request,
never at the end. A request that raises part-way through therefore cannot poison
the next one.

Timers cover model work only: ``decode_seconds`` accumulates forward + sample
time per step, so time a consumer spends printing between yields is not counted
as decoding.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .cache import KVCache
from .config import GenerationConfig, ModelConfig
from .errors import ContextOverflowError, UnsupportedInputError
from .metrics import GenerationMetrics, peak_rss_bytes, stopwatch
from .model import DebugProbe, Qwen3Model
from .sampling import Sampler, SamplingParams
from .tokenizer import ChatMessage, Tokenizer, stop_token_ids
from .weights import Weights

DEFAULT_MAX_CONTEXT = 2048
DEFAULT_PREFILL_CHUNK_SIZE = 128
DEFAULT_MAX_NEW_TOKENS = 128


class StopReason(str, Enum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"


@dataclass(frozen=True)
class GenerationResult:
    """One completed request.

    ``token_ids`` includes the terminating stop token when one was produced;
    ``text`` does not display it, because it is a special token.
    """

    text: str
    token_ids: tuple[int, ...]
    prompt_token_ids: tuple[int, ...]
    stop_reason: StopReason
    metrics: GenerationMetrics

    @property
    def stopped_on_eos(self) -> bool:
        return self.stop_reason is StopReason.EOS

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": list(self.token_ids),
            "prompt_token_ids": list(self.prompt_token_ids),
            "stop_reason": self.stop_reason.value,
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class GenerationStep:
    """One streamed increment. ``result`` is set only on the final step."""

    text_delta: str
    token_id: int
    index: int
    finished: bool
    result: GenerationResult | None = None


class Engine:
    """Tokenizer + model + generation loop."""

    def __init__(
        self,
        model: Qwen3Model,
        tokenizer: Tokenizer,
        generation_config: GenerationConfig,
        max_context: int = DEFAULT_MAX_CONTEXT,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        load_seconds: float = float("nan"),
    ) -> None:
        if prefill_chunk_size < 1:
            raise ValueError(f"prefill_chunk_size must be >= 1, got {prefill_chunk_size}")
        self.model = model
        self.config: ModelConfig = model.config
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.max_context = model.config.validate_max_context(max_context)
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.stop_ids = stop_token_ids(generation_config, tokenizer)
        self.load_seconds = load_seconds
        self._cache: KVCache = model.new_cache(self.max_context)

    @property
    def kv_cache(self) -> KVCache:
        """The engine's cache. Reset at the start of every request."""
        return self._cache

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        model_dir: str | Path,
        max_context: int = DEFAULT_MAX_CONTEXT,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
    ) -> "Engine":
        """Load a converted model. Fully offline: no network access happens here."""
        with stopwatch() as elapsed:
            root = Path(model_dir)
            config_path = root / "config.json"
            if not config_path.is_file():
                raise UnsupportedInputError(
                    f"{config_path} not found; run tools/convert_model.py first"
                )
            config = ModelConfig.from_hf_config(
                json.loads(config_path.read_text(encoding="utf-8"))
            )
            generation_config = GenerationConfig.from_json_file(
                root / "generation_config.json", config_path
            )
            model = Qwen3Model(config, Weights.from_directory(root))
            tokenizer = Tokenizer.from_directory(root)
        return cls(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            max_context=max_context,
            prefill_chunk_size=prefill_chunk_size,
            load_seconds=elapsed[0],
        )

    # ---- prompt preparation -------------------------------------------------

    def render(self, messages: Iterable[Mapping[str, Any] | ChatMessage]) -> str:
        return self.tokenizer.render_chat(messages)

    def encode_messages(
        self, messages: Iterable[Mapping[str, Any] | ChatMessage]
    ) -> list[int]:
        return self.tokenizer.encode_chat(messages)

    # ---- public API ---------------------------------------------------------

    def generate(
        self,
        messages: Iterable[Mapping[str, Any] | ChatMessage] | None = None,
        *,
        prompt: str | None = None,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> GenerationResult:
        """Non-streaming generation from chat messages (or a raw ``prompt``)."""
        return self._drain(
            self.stream(
                messages, prompt=prompt, sampling=sampling, max_new_tokens=max_new_tokens
            )
        )

    def generate_text(
        self,
        text: str,
        *,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> GenerationResult:
        """Complete ``text`` verbatim, without applying the chat template."""
        with stopwatch() as elapsed:
            prompt_ids = self.tokenizer.encode(text)
        return self._drain(
            self.stream_ids(
                prompt_ids,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
                tokenize_seconds=elapsed[0],
            )
        )

    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        probe: DebugProbe | None = None,
    ) -> GenerationResult:
        """Lowest-level API: token IDs in, result out. No tokenizer involved."""
        return self._drain(
            self.stream_ids(
                prompt_ids, sampling=sampling, max_new_tokens=max_new_tokens, probe=probe
            )
        )

    def stream(
        self,
        messages: Iterable[Mapping[str, Any] | ChatMessage] | None = None,
        *,
        prompt: str | None = None,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Iterator[GenerationStep]:
        """Streaming generation from chat messages (or a raw ``prompt``)."""
        prompt_ids, tokenize_seconds = self._prepare(messages, prompt)
        return self.stream_ids(
            prompt_ids,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            tokenize_seconds=tokenize_seconds,
        )

    def stream_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        tokenize_seconds: float = 0.0,
        probe: DebugProbe | None = None,
    ) -> Iterator[GenerationStep]:
        """The one real generation loop: chunked prefill, then token by token."""
        ids = [int(token_id) for token_id in prompt_ids]
        if not ids:
            raise UnsupportedInputError(
                "the prompt tokenized to zero tokens; nothing to condition on"
            )
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError(f"max_new_tokens must be an int, got {max_new_tokens!r}")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        required = len(ids) + max_new_tokens
        if required > self.max_context:
            raise ContextOverflowError(
                f"prompt ({len(ids)} tokens) + max_new_tokens ({max_new_tokens}) = "
                f"{required} exceeds max_context={self.max_context}. Raise "
                "--max-context or lower --max-new-tokens; SlowQ never truncates."
            )

        sampler = Sampler(sampling or SamplingParams())
        metrics = GenerationMetrics(
            tokenize_seconds=tokenize_seconds,
            prompt_tokens=len(ids),
            weight_logical_bytes=self.model.weights.deduplicated_bytes(),
            kv_allocated_bytes=self._cache.allocated_bytes(),
        )
        decoder = self.tokenizer.incremental_decoder()

        # Reset before use so an aborted previous request cannot be observed.
        self._cache.reset()
        request_start = time.perf_counter()

        forward = None
        with stopwatch() as prefill_elapsed:
            for start in range(0, len(ids), self.prefill_chunk_size):
                forward = self.model.forward(
                    ids[start : start + self.prefill_chunk_size],
                    self._cache,
                    probe=probe,
                )
                metrics.prefill_chunks += 1
        metrics.prefill_seconds = prefill_elapsed[0]
        assert forward is not None  # the loop above always runs at least once
        logits = forward.logits

        generated: list[int] = []
        text_parts: list[str] = []
        first_token = True

        while True:
            if first_token:
                with stopwatch() as sample_elapsed:
                    token_id = sampler.sample(logits)
                metrics.first_token_seconds = (
                    tokenize_seconds + metrics.prefill_seconds + sample_elapsed[0]
                )
                first_token = False
            else:
                with stopwatch() as step_elapsed:
                    logits = self.model.forward([token_id], self._cache, probe=probe).logits
                    token_id = sampler.sample(logits)
                metrics.decode_seconds += step_elapsed[0]

            generated.append(token_id)
            delta = decoder.push(token_id)
            text_parts.append(delta)

            is_stop_token = token_id in self.stop_ids
            if is_stop_token or len(generated) >= max_new_tokens:
                # Whatever the decoder is still holding back belongs to this last
                # token, so it goes out in the final delta as well as in the
                # result: a consumer that only reads the stream must see exactly
                # the text a blocking call returns.
                flushed = decoder.flush()
                text_parts.append(flushed)
                metrics.generated_tokens = len(generated)
                metrics.total_seconds = (
                    tokenize_seconds + time.perf_counter() - request_start
                )
                metrics.stop_reason = (
                    StopReason.EOS.value if is_stop_token else StopReason.MAX_NEW_TOKENS.value
                )
                metrics.peak_rss_bytes = peak_rss_bytes()
                result = GenerationResult(
                    text="".join(text_parts),
                    token_ids=tuple(generated),
                    prompt_token_ids=tuple(ids),
                    stop_reason=(
                        StopReason.EOS if is_stop_token else StopReason.MAX_NEW_TOKENS
                    ),
                    metrics=metrics,
                )
                yield GenerationStep(
                    text_delta=delta + flushed,
                    token_id=token_id,
                    index=len(generated) - 1,
                    finished=True,
                    result=result,
                )
                return

            yield GenerationStep(
                text_delta=delta,
                token_id=token_id,
                index=len(generated) - 1,
                finished=False,
                result=None,
            )

    # ---- internals ----------------------------------------------------------

    def _prepare(
        self,
        messages: Iterable[Mapping[str, Any] | ChatMessage] | None,
        prompt: str | None,
    ) -> tuple[list[int], float]:
        if (messages is None) == (prompt is None):
            raise ValueError("pass exactly one of 'messages' or 'prompt'")
        with stopwatch() as elapsed:
            if prompt is not None:
                if not isinstance(prompt, str):
                    raise UnsupportedInputError(
                        f"prompt must be a string, got {type(prompt).__name__}"
                    )
                messages = [ChatMessage(role="user", content=prompt)]
            prompt_ids = self.encode_messages(messages)  # type: ignore[arg-type]
        return prompt_ids, elapsed[0]

    @staticmethod
    def _drain(steps: Iterator[GenerationStep]) -> GenerationResult:
        result: GenerationResult | None = None
        for step in steps:
            if step.result is not None:
                result = step.result
        if result is None:
            raise AssertionError("generation loop ended without a result")
        return result
