"""Generation: chunked prefill, decode loop, stop conditions, streaming.

Layering: ``Engine._stream`` is the only place that runs the loop. Every other
entry point (``generate``, ``generate_text``, ``stream``, ``stream_ids`` and all
of ``Session``) resolves text into token IDs, picks a cache and a starting
position, and delegates, so there is exactly one implementation of "prefill, then
decode one token at a time".

The KV cache is owned by the engine and reset at the *start* of each request,
never at the end. A request that raises part-way through therefore cannot poison
the next one. A :class:`Session` opts out of that reset on purpose — it is the
one way to keep KV across requests — and pays for it with a second cache and an
explicit prefix match.

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
        vision_cache_hit: bool = False,
        probe: DebugProbe | None = None,
        prefill_embeds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Iterator[GenerationStep]:
        """The public generation entry point: chunked prefill, then token by token.

        ``prefill_embeds`` is ``(features, mask)``: rows that replace the embedded
        prompt tokens the boolean mask selects. Only ``slowq.vlm`` produces them —
        a text prompt embeds entirely from the table — and they are an argument
        rather than engine state so that a request cannot leak image features into
        the next one. ``vision_encode_seconds`` travels the same way: it belongs to
        time-to-first-token, and without it the vision path would report a TTFT
        that silently omits its largest cost. ``vision_cache_hit`` travels with it
        because a near-zero encode is ambiguous on its own — it could mean the
        image was tiny or that the tower never ran.

        Always starts from an empty cache, so two requests through an engine can
        never observe each other. Use :meth:`session` when that isolation is the
        wrong trade — a chat re-renders its whole history every turn.
        """
        ids = self._validate_request(prompt_ids, max_new_tokens)
        # Reset before use so an aborted previous request cannot be observed.
        self._cache.reset()
        yield from self._stream(
            ids,
            cache=self._cache,
            prefill_from=0,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            tokenize_seconds=tokenize_seconds,
            vision_encode_seconds=vision_encode_seconds,
            vision_cache_hit=vision_cache_hit,
            probe=probe,
            prefill_embeds=prefill_embeds,
        )

    def _stream(
        self,
        ids: list[int],
        *,
        cache: KVCache,
        prefill_from: int,
        sampling: SamplingParams | None,
        max_new_tokens: int,
        tokenize_seconds: float,
        vision_encode_seconds: float,
        probe: DebugProbe | None,
        prefill_embeds: tuple[np.ndarray, np.ndarray] | None,
        vision_cache_hit: bool = False,
        reused_tokens: int = 0,
        committed: list[int] | None = None,
    ) -> Iterator[GenerationStep]:
        """The one real loop, over a cache and a starting position the caller chose.

        ``stream_ids`` passes this engine's freshly reset cache with
        ``prefill_from=0``; :class:`Session` passes its own cache with the length of
        the prefix it recognised. Keeping both on this one function is what stops
        "prefill, then decode" from existing twice and drifting apart.

        Contract for callers:

        * ``ids`` has been through :meth:`_validate_request`.
        * ``cache.length == prefill_from``. Not re-checked here — ``prefill_from``
          is handed to ``DecoderModel.prefill`` as ``position_offset``, and
          ``encode`` already refuses an offset that disagrees with the cache,
          which is what keeps reused tokens on the right absolute positions.
        * ``committed``, when given, is the caller's record of which token ids the
          cache's committed slots hold. It is extended at exactly the points the
          cache is, so a request abandoned or failed mid-decode leaves it
          truthful rather than optimistically complete.
        """
        if prefill_from and prefill_embeds is not None:
            raise ValueError(
                "a request that reuses cached KV cannot also substitute image "
                "features: the reused prefix was embedded from the token table, "
                "and nothing here can confirm the cached slots still describe the "
                "same image. Sessions are text-only."
            )

        sampler = Sampler(sampling or SamplingParams())
        metrics = GenerationMetrics(
            tokenize_seconds=tokenize_seconds,
            vision_encode_seconds=vision_encode_seconds,
            vision_cache_hit=vision_cache_hit,
            prompt_tokens=len(ids),
            reused_tokens=reused_tokens,
            weight_logical_bytes=self.model.weights.deduplicated_bytes(),
            kv_allocated_bytes=cache.allocated_bytes(),
        )
        decoder = self.tokenizer.incremental_decoder()
        request_start = time.perf_counter()

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
            prefill = self.model.prefill(
                ids[prefill_from:],
                cache,
                chunk_size=self.prefill_chunk_size,
                probe=probe,
                position_offset=prefill_from,
                inputs_embeds=embeds,
                inputs_embed_mask=embed_mask,
            )
        metrics.prefill_seconds = prefill_elapsed[0]
        metrics.prefill_chunks = prefill.chunks
        logits = prefill.logits
        if committed is not None:
            committed.extend(ids[prefill_from:])

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
                    logits = self.model.forward([token_id], cache, probe=probe).logits
                    # Recorded before sampling, and with no yield in between: a
                    # sampler that raises, or a consumer that abandons the
                    # generator at the yield below, both leave ``committed``
                    # describing exactly the slots the cache holds.
                    if committed is not None:
                        committed.append(token_id)
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

    # ---- sessions -----------------------------------------------------------

    def session(self) -> "Session":
        """A multi-turn conversation that reuses its KV cache across requests.

        Allocates a *second* full-size cache rather than lending this engine's
        one; :class:`Session` explains why that has to be so, and what it costs.
        """
        return Session(self)

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


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Length of the longest run of token ids two sequences share from the start.

    Compared as ids, not as decoded text. A session decides what to reuse from the
    tokens it is about to feed the model, so a template that re-tokenises the same
    words into different ids — which BPE does at every turn boundary, because
    ``"hello"`` and ``"hello</s></tool_response>"`` are not the same merge — yields a shorter
    prefix instead of a wrong one. Under-reusing costs prefill time; over-reusing
    would condition the model on a conversation it was never shown.
    """
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


@dataclass(frozen=True)
class SessionReuse:
    """What one session request found in the cache, before it ran.

    Recorded at planning time rather than at completion, so a request that was
    cancelled or that raised still reports the decision it made — which is the
    part worth looking at when reuse stops happening.
    """

    prompt_tokens: int
    reused_tokens: int
    prefilled_tokens: int
    discarded_tokens: int
    cache_length_before: int

    @property
    def reuse_ratio(self) -> float:
        return self.reused_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def describe(self) -> str:
        return (
            f"reused {self.reused_tokens}/{self.prompt_tokens} prompt tokens "
            f"({self.reuse_ratio * 100:.0f}%), prefilled {self.prefilled_tokens}, "
            f"discarded {self.discarded_tokens}"
        )


class Session:
    """A multi-turn text conversation that reuses one KV cache across requests.

    Why this is not the default
    ---------------------------
    ``Engine.stream_ids`` resets the cache at the start of every request, so two
    requests through an engine can never observe each other. A chat does the
    opposite: it re-renders the whole history on every turn, so turn *n*'s prompt
    begins with turn *n-1*'s prompt plus the reply the model just wrote. That is
    work the cache already did. A session keeps the cache alive between turns and
    prefills only the part it does not recognise, which on a growing conversation
    turns an O(n²) prefill into an O(n) one.

    How the reuse boundary is chosen
    --------------------------------
    Each request renders and tokenises the *complete* history, then
    :func:`common_prefix_length` compares those ids against the ids the cache was
    actually built from. The match length is the reuse boundary. Nothing assumes
    the new prompt extends the old one, so editing or deleting an earlier message
    simply moves the boundary back; the cache is truncated to it and the rest is
    recomputed. A full match still runs one token, because generation samples from
    the last position's *logits* and the cache only holds its K/V.

    Cost
    ----
    A session allocates its **own** full-size cache. Sharing the engine's would
    mean any plain ``engine.generate()`` in between wiped the conversation, so the
    two are separate buffers: at ``max_context=2048`` on Qwen3-0.6B that is
    another ~448 MiB. ``reset()`` clears the prefix but keeps the allocation.

    Scope
    -----
    Text only. An image-text request replaces embedded prompt rows with features
    the vision tower produced, and a cached prefix carries no record of which
    image it came from, so ``VisionEngine`` keeps its own path and ``_stream``
    refuses the combination outright.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._model = engine.model
        self._identity = self._geometry(engine)
        self._cache: KVCache = engine.model.new_cache(engine.max_context)
        # Token ids the cache's committed slots hold, positionally. Kept in step
        # with ``cache.length`` by ``_stream`` at the points the cache advances,
        # so it stays truthful through a cancelled or failed request.
        self._committed: list[int] = []
        self._turns = 0
        self._last_reuse: SessionReuse | None = None

    @staticmethod
    def _geometry(engine: Engine) -> tuple[int, ...]:
        config = engine.model.config
        return (
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.vocab_size,
            engine.max_context,
        )

    # ---- state --------------------------------------------------------------

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def cache(self) -> KVCache:
        """This session's own cache — not the engine's."""
        return self._cache

    @property
    def cached_tokens(self) -> int:
        """Committed tokens currently reusable by the next request."""
        return self._cache.length

    @property
    def turns(self) -> int:
        """Requests whose final step reached the caller. A cancelled one is not a turn.

        Counted at delivery rather than at generator exhaustion, so a caller that
        stops reading once it has the result is still credited with the turn it
        completed. See :meth:`_run`.
        """
        return self._turns

    @property
    def last_reuse(self) -> SessionReuse | None:
        """The most recent request's reuse decision, or ``None`` before the first."""
        return self._last_reuse

    def reset(self) -> None:
        """Forget the conversation; keep the buffers.

        This is ``/clear``, not destruction: the next request starts from an empty
        cache but does not pay for another allocation.
        """
        self._cache.reset()
        self._committed.clear()
        self._turns = 0
        self._last_reuse = None

    # ---- requests -----------------------------------------------------------

    def stream(
        self,
        messages: Iterable[Mapping[str, Any] | ChatMessage] | None = None,
        *,
        prompt: str | None = None,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Iterator[GenerationStep]:
        """Streaming generation over a whole conversation, reusing what matches."""
        ids, tokenize_seconds = self._engine._prepare(messages, prompt)
        return self._run(
            ids,
            tokenize_seconds=tokenize_seconds,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            probe=None,
        )

    def generate(
        self,
        messages: Iterable[Mapping[str, Any] | ChatMessage] | None = None,
        *,
        prompt: str | None = None,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> GenerationResult:
        return Engine._drain(
            self.stream(
                messages, prompt=prompt, sampling=sampling, max_new_tokens=max_new_tokens
            )
        )

    def stream_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        probe: DebugProbe | None = None,
    ) -> Iterator[GenerationStep]:
        """Token ids in, no chat template. ``probe`` positions are block-relative.

        A caller that mixes this with :meth:`stream` gets reuse against whatever
        ids were actually committed, which is the point of comparing ids: there is
        no separate notion of "the conversation" to fall out of sync.
        """
        return self._run(
            list(prompt_ids),
            tokenize_seconds=0.0,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            probe=probe,
        )

    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        probe: DebugProbe | None = None,
    ) -> GenerationResult:
        return Engine._drain(
            self.stream_ids(
                prompt_ids, sampling=sampling, max_new_tokens=max_new_tokens, probe=probe
            )
        )

    def _run(
        self,
        ids: list[int],
        *,
        tokenize_seconds: float,
        sampling: SamplingParams | None,
        max_new_tokens: int,
        probe: DebugProbe | None,
    ) -> Iterator[GenerationStep]:
        """Stream one request, counting it as a turn when its result is delivered.

        Not after the generator finishes. ``yield from`` returns only if the
        consumer reads to the end, and a caller that stops as soon as it has the
        final step — which carries the complete :class:`GenerationResult`, so
        there is nothing left to read — closes the generator instead. Counting
        there reported ``turns=0`` for a request that had finished and been handed
        over.

        So the increment sits immediately before the final step's ``yield``: the
        caller has asked for a value and is about to get the result, which is the
        moment the request becomes a completed turn. Everything short of that
        counts as nothing. A ``close()`` or an exception at an earlier step
        resumes the generator nowhere, so the line is never reached; a stream that
        is exhausted normally reaches it once, because ``_stream`` yields exactly
        one ``finished`` step and then returns.
        """
        # Validated before planning: an over-long prompt must be refused without
        # truncating the cache first, or a bad request would cost the good prefix.
        ids = self._engine._validate_request(ids, max_new_tokens)
        start = self._plan(ids)
        counted = False
        for step in self._engine._stream(
            ids,
            cache=self._cache,
            prefill_from=start,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            tokenize_seconds=tokenize_seconds,
            vision_encode_seconds=0.0,
            probe=probe,
            prefill_embeds=None,
            reused_tokens=start,
            committed=self._committed,
        ):
            if step.finished and not counted:
                self._turns += 1
                counted = True
            yield step

    # ---- internals ----------------------------------------------------------

    def _check_identity(self) -> None:
        """Refuse to reuse a cache that no longer describes this engine's model.

        A session pins the engine it was built from, so this can only fire if the
        model or its geometry was swapped underneath it. The buffers were allocated
        from the *old* layer count and head shape, so continuing would write K/V
        into slots the current model reads as something else — wrong output with no
        error. A handful of integer comparisons per request is cheap insurance.
        """
        engine = self._engine
        if engine.model is not self._model:
            raise RuntimeError(
                "the engine's model was replaced after this session was created; "
                "the session's KV cache describes the previous one. Build a new "
                "session with engine.session()."
            )
        current = self._geometry(engine)
        if current != self._identity:
            raise RuntimeError(
                f"the engine's model geometry changed from {self._identity} to "
                f"{current} (layers, kv heads, head dim, vocab, max_context); the "
                "session's KV cache no longer matches it. Build a new session."
            )

    def _plan(self, ids: Sequence[int]) -> int:
        """Truncate the cache to the recognised prefix and return where to resume.

        The return value is the index into ``ids`` of the first token that is not
        already cached, so ``ids[result:]`` is exactly what gets prefilled.
        """
        self._check_identity()
        cache = self._cache
        length_before = cache.length

        # A request that failed mid-prefill commits the chunks it finished before
        # raising, so the cache can hold more tokens than this session recorded.
        # Dropping the unrecorded tail from ``_committed`` is safe, not lossy: the
        # reuse boundary can never exceed what is recorded, and every slot at or
        # past it is either overwritten by this prefill or left beyond ``length``,
        # where nothing reads it.
        del self._committed[length_before:]

        reuse = common_prefix_length(self._committed, ids)
        # A full match still needs one forward. The cache holds K/V for the last
        # prompt token, but generation samples from its logits, and those were
        # never computed: the token that would have produced them is the previous
        # reply's stop token, which the decode loop never feeds back. Re-running
        # that single position costs one decode step instead of a whole prefill.
        start = len(ids) - 1 if reuse == len(ids) else reuse

        cache.truncate(start)
        del self._committed[start:]
        self._last_reuse = SessionReuse(
            prompt_tokens=len(ids),
            reused_tokens=start,
            prefilled_tokens=len(ids) - start,
            discarded_tokens=length_before - start,
            cache_length_before=length_before,
        )
        return start

    def __repr__(self) -> str:
        return (
            f"Session(turns={self._turns}, cached_tokens={self._cache.length}, "
            f"max_context={self._cache.max_context}, "
            f"allocated={self._cache.allocated_bytes() / 2**20:.1f} MiB)"
        )
