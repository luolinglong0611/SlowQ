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

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .cache import KVCache
from .config import (
    TASK_TEXT_EMBEDDING,
    TASK_TEXT_GENERATION,
    GenerationConfig,
    ModelConfig,
)
from .errors import ConfigError, ContextOverflowError, UnsupportedInputError
from .metrics import GenerationMetrics, peak_rss_bytes, stopwatch
from .model import CausalLMModel, DebugProbe, load_config_and_weights
from .sampling import Sampler, SamplingParams
from .tokenizer import ChatMessage, Tokenizer, stop_token_ids
from .weights import Weights

DEFAULT_MAX_CONTEXT = 2048
DEFAULT_PREFILL_CHUNK_SIZE = 128
DEFAULT_MAX_NEW_TOKENS = 128


class StopReason(str, Enum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"


def recorded_task(model_dir: str | Path) -> str:
    """What a converted directory says it is for.

    Read from the manifest, not from ``config.json``: the config cannot answer
    it. ``Qwen/Qwen3-Embedding-0.6B`` publishes ``model_type: "qwen3"`` and
    ``architectures: ["Qwen3ForCausalLM"]``, the same task-relevant fields as the
    chat model, so a loader that guessed from them would build a generator out of
    an embedding checkpoint.
    """
    root = Path(model_dir)
    if not (root / "config.json").is_file():
        raise UnsupportedInputError(
            f"{root / 'config.json'} not found; run tools/convert_model.py first"
        )
    task = Weights.from_directory(root).manifest.task
    # A format_version 1 manifest carries no task key. Only SlowQ v0.1 wrote
    # those, and v0.1 converted nothing but Qwen3 causal LMs.
    return TASK_TEXT_GENERATION if task is None else task


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
        model: CausalLMModel,
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
        """Load a converted model. Fully offline: no network access happens here.

        The directory's recorded task decides whether this is the right engine to
        build. Refusing here, with the task named, is better than constructing a
        causal LM out of an embedding checkpoint and failing on a missing
        ``lm_head`` halfway through the first forward.
        """
        with stopwatch() as elapsed:
            root = Path(model_dir)
            task = recorded_task(root)
            if task == TASK_TEXT_EMBEDDING:
                raise ConfigError(
                    f"{root} is a converted text-embedding model (task={task!r}); it "
                    "has no output projection and cannot generate. Load it with "
                    "slowq.embedding.EmbeddingEngine, or from the CLI with "
                    "``slowq embed``."
                )
            if task != TASK_TEXT_GENERATION:
                raise ConfigError(
                    f"{root} is a converted {task!r} model; load it with "
                    "slowq.vlm.VisionEngine (CLI: ``slowq vlm``)."
                )
            config, weights = load_config_and_weights(root, task)
            generation_config = GenerationConfig.from_json_file(
                root / "generation_config.json", root / "config.json"
            )
            model = CausalLMModel(config, weights)
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

    def _validate_request(
        self, prompt_ids: Sequence[int], max_new_tokens: int
    ) -> list[int]:
        """Check what a request asks for before doing any of its work.

        Shared with ``slowq.vlm.VisionEngine.stream_prompt``, which has to run it
        *before* encoding the image: the vision tower is the most expensive part
        of that request, and a caller that asks for zero new tokens or for more
        than ``max_context`` should be refused without paying for it. Keeping the
        two paths on one implementation is what stops them drifting apart.

        Returns the prompt as a plain ``list[int]``, since every caller wants it.
        """
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
        return ids

    def stream_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        tokenize_seconds: float = 0.0,
        vision_encode_seconds: float = 0.0,
        probe: DebugProbe | None = None,
        prefill_embeds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Iterator[GenerationStep]:
        """The one real generation loop: chunked prefill, then token by token.

        ``prefill_embeds`` is ``(features, mask)``: rows that replace the embedded
        prompt tokens the boolean mask selects. Only ``slowq.vlm`` produces them —
        a text prompt embeds entirely from the table — and they are an argument
        rather than engine state so that a request cannot leak image features into
        the next one. ``vision_encode_seconds`` travels the same way: it belongs to
        time-to-first-token, and without it the vision path would report a TTFT
        that silently omits its largest cost.
        """
        ids = self._validate_request(prompt_ids, max_new_tokens)

        sampler = Sampler(sampling or SamplingParams())
        metrics = GenerationMetrics(
            tokenize_seconds=tokenize_seconds,
            vision_encode_seconds=vision_encode_seconds,
            prompt_tokens=len(ids),
            weight_logical_bytes=self.model.weights.deduplicated_bytes(),
            kv_allocated_bytes=self._cache.allocated_bytes(),
        )
        decoder = self.tokenizer.incremental_decoder()

        # Reset before use so an aborted previous request cannot be observed.
        self._cache.reset()
        request_start = time.perf_counter()

        forward = None
        embeds, embed_mask = (None, None) if prefill_embeds is None else prefill_embeds
        if embed_mask is not None:
            embed_mask = np.asarray(embed_mask, dtype=bool).reshape(-1)
            if embed_mask.shape[0] != len(ids):
                raise ValueError(
                    f"prefill_embeds covers {embed_mask.shape[0]} prompt positions "
                    f"but the prompt has {len(ids)} tokens"
                )
            if embeds is None and embed_mask.any():
                raise ValueError(
                    "prefill_embeds marks placeholder positions but supplies no "
                    "feature rows to put there"
                )
        with stopwatch() as prefill_elapsed:
            feature_cursor = 0
            for start in range(0, len(ids), self.prefill_chunk_size):
                chunk = ids[start : start + self.prefill_chunk_size]
                chunk_embeds = chunk_mask = None
                if embed_mask is not None:
                    # Placeholder tokens can straddle a chunk boundary, so each
                    # chunk takes its own slice of the feature rows in prompt
                    # order rather than assuming the image fits in one chunk.
                    chunk_mask = embed_mask[start : start + len(chunk)]
                    take = int(chunk_mask.sum())
                    if take:
                        chunk_embeds = embeds[feature_cursor : feature_cursor + take]
                    feature_cursor += take
                forward = self.model.forward(
                    chunk,
                    self._cache,
                    probe=probe,
                    inputs_embeds=chunk_embeds,
                    inputs_embed_mask=chunk_mask,
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
                    tokenize_seconds
                    + vision_encode_seconds
                    + metrics.prefill_seconds
                    + sample_elapsed[0]
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
                    tokenize_seconds
                    + vision_encode_seconds
                    + time.perf_counter()
                    - request_start
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
