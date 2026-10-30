"""Text embedding: last-token pooling, L2-normalised, and no generation at all.

``Qwen/Qwen3-Embedding-0.6B`` is a Qwen3 decoder used as an encoder. The official
recipe (the model card's "Sentence Transformers"-free snippet) is:

1. tokenise the text with the checkpoint's own post-processor — no chat
   template, and for this checkpoint that means a trailing ``<|fim_suffix|>``;
2. run the decoder once and take ``last_hidden_state``, which is the output of
   the final RMSNorm, not of ``lm_head``;
3. keep the row of the **last valid token** (``last_token_pool``);
4. L2-normalise it;
5. score two texts with a dot product, which on unit vectors is cosine.

Every one of those steps is a choice that is easy to get wrong in a way that
still produces plausible-looking numbers, so they are stated explicitly here:
the checkpoint's own causal attention is kept (not made bidirectional), pooling
is last-token (not mean), and the output is the hidden state (not the vocabulary
distribution). Nothing in this module calls ``output_projection`` or samples a
token.

Queries carry the official instruction prefix; documents do not::

    Instruct: {task description}\nQuery:{query}

Note there is no space after ``Query:``. That is what the model card's
``get_detailed_instruct`` returns, and adding one changes the tokenisation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import TASK_TEXT_EMBEDDING, ModelConfig
from .errors import ConfigError, ContextOverflowError, UnsupportedInputError
from .metrics import stopwatch
from .model import DebugProbe, DecoderModel, load_config_and_weights
from .ops import l2_normalize
from .tokenizer import Tokenizer
from .weights import Weights

#: The model card's example task description, used when a caller asks for a query
#: embedding without naming a task.
DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
#: ``max_length`` in the official snippet. SlowQ refuses longer input rather than
#: truncating it: a silently shortened document embeds to a different vector than
#: the same document would in a system that kept the tail.
DEFAULT_MAX_LENGTH = 8192
DEFAULT_PREFILL_CHUNK_SIZE = 128


def format_query(query: str, instruction: str | None = DEFAULT_QUERY_INSTRUCTION) -> str:
    """Wrap a query the way the official ``get_detailed_instruct`` does."""
    if instruction is None:
        return query
    return f"Instruct: {instruction}\nQuery:{query}"


def last_token_pool(hidden_states: np.ndarray, num_valid_tokens: int) -> np.ndarray:
    """The official ``last_token_pool`` for a single unpadded sequence.

    The reference picks ``last_hidden_state[:, -1]`` when the batch is
    left-padded and row ``attention_mask.sum() - 1`` otherwise. SlowQ encodes one
    sequence at a time with no padding, so the last valid token is the last row
    and the two branches agree — but the count is still taken from the caller
    rather than from ``hidden_states.shape[0]``, so a future padded path cannot
    silently pool a pad token.
    """
    hidden_states = np.asarray(hidden_states)
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be [num_tokens, hidden_size], got {hidden_states.shape}"
        )
    if not 1 <= num_valid_tokens <= hidden_states.shape[0]:
        raise ValueError(
            f"num_valid_tokens={num_valid_tokens} is outside 1.."
            f"{hidden_states.shape[0]}; there is no last valid token to pool"
        )
    return hidden_states[num_valid_tokens - 1]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Dot product of two vectors, which is cosine when both are unit-length.

    The normalisation is *not* repeated here on purpose: ``EmbeddingModel.encode``
    already produced unit vectors, and re-normalising would hide a caller passing
    in something that was not an embedding.
    """
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"dimension mismatch: {left.shape[0]} vs {right.shape[0]}")
    return float(np.dot(left, right))


def similarity_matrix(
    queries: Sequence[np.ndarray], documents: Sequence[np.ndarray]
) -> np.ndarray:
    """``[len(queries), len(documents)]`` cosine scores."""
    if not queries or not documents:
        raise ValueError("similarity_matrix needs at least one vector on each side")
    left = np.stack([np.asarray(q, dtype=np.float32).reshape(-1) for q in queries])
    right = np.stack([np.asarray(d, dtype=np.float32).reshape(-1) for d in documents])
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"dimension mismatch: {left.shape[1]} vs {right.shape[1]}"
        )
    return np.matmul(left, right.T)


def rank_documents(
    query: np.ndarray, documents: Sequence[EmbeddingResult]
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Score one query vector against encoded documents and rank them.

    Shared by :meth:`EmbeddingEngine.search` and :meth:`DocumentIndex.search` so
    that "the index gives a different answer than the plain search" is not a
    possible failure mode: there is one implementation of the ranking, and the
    only difference between the two entry points is where the document vectors
    came from.
    """
    scores = similarity_matrix([query], [result.vector for result in documents])
    ranking = sorted(
        ((index, float(score)) for index, score in enumerate(scores[0])),
        key=lambda item: item[1],
        reverse=True,
    )
    return scores, ranking


@dataclass(frozen=True)
class EmbeddingResult:
    """One encoded text."""

    vector: np.ndarray
    text: str
    input_ids: tuple[int, ...]
    num_tokens: int
    encode_seconds: float
    #: True when the official instruction prefix was applied to ``text``.
    is_query: bool = False

    @property
    def dimension(self) -> int:
        return int(self.vector.shape[-1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "vector": [float(value) for value in self.vector],
            "dimension": self.dimension,
            "input_ids": list(self.input_ids),
            "num_tokens": self.num_tokens,
            "is_query": self.is_query,
            "encode_seconds": self.encode_seconds,
        }


def detached(result: EmbeddingResult) -> EmbeddingResult:
    """``result`` with its own copy of the vector, sharing no buffer with it.

    ``EmbeddingResult`` is a frozen dataclass, and frozen freezes the *attribute
    binding*, not the array it points at: ``result.vector`` cannot be reassigned,
    but ``result.vector[:] = 0`` writes through happily. Anything that stores
    results and later hands them back therefore has to say which side owns the
    buffer, and :class:`DocumentIndex` answers by owning a private copy and
    detaching on the way out — otherwise a caller zeroing one document's vector
    silently re-scores the whole corpus for every later query, while
    :meth:`EmbeddingEngine.search` (which re-encodes) still returns the original
    answer. The scores keep looking like cosine similarities; they just stop
    meaning anything, which is the worst shape this bug can take.

    Only ``vector`` is copied. ``text`` and ``input_ids`` are a ``str`` and a
    ``tuple`` — immutable, so sharing them is not aliasing — and the rest are
    scalars.

    The copy is the whole cost of the guarantee: ``hidden_size`` float32 per
    document, so 4 MiB for a thousand Qwen3-Embedding-0.6B documents. That is
    the same order as the two ``np.stack`` calls :func:`similarity_matrix`
    already makes per search, and both are dwarfed by the query encode, which is
    a full prefill. Measured rather than asserted in
    ``tools/perf``'s index A/B, where the copy sits inside the timed window.
    """
    return replace(result, vector=np.array(result.vector, dtype=np.float32, copy=True))


@dataclass(frozen=True)
class EmbeddingBatch:
    """Several texts encoded one after another, with the scores between them.

    Sequential, not batched: SlowQ runs batch size 1 everywhere, and the official
    recipe pads a batch only to run it in one forward. Encoding each text on its
    own gives the same vectors without the padding, so ``similarity`` below is
    the same matrix the batched reference produces.
    """

    results: tuple[EmbeddingResult, ...]
    similarity: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterable[EmbeddingResult]:
        return iter(self.results)

    @property
    def vectors(self) -> np.ndarray:
        return np.stack([result.vector for result in self.results])

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "results": [result.as_dict() for result in self.results]
        }
        if self.similarity is not None:
            payload["similarity"] = [
                [float(value) for value in row] for row in self.similarity
            ]
        return payload


class EmbeddingModel:
    """The Qwen3 decoder used as an encoder: no ``lm_head``, no sampling."""

    def __init__(self, config: ModelConfig, weights: Weights) -> None:
        if config.task != TASK_TEXT_EMBEDDING:
            raise ConfigError(
                f"this directory's manifest records task={config.task!r}, not "
                f"{TASK_TEXT_EMBEDDING!r}. A text-generation checkpoint has an "
                "output projection this class will not use, and pooling its hidden "
                "states would produce a vector that means nothing."
            )
        self.config = config
        self.weights = weights
        self.decoder = DecoderModel(config, weights)

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "EmbeddingModel":
        config, weights = load_config_and_weights(model_dir)
        return cls(config, weights)

    @property
    def dimension(self) -> int:
        """The official full vector dimension. SlowQ does not truncate or resize it."""
        return self.config.hidden_size

    def encode_ids(
        self,
        input_ids: Sequence[int],
        *,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        probe: DebugProbe | None = None,
    ) -> np.ndarray:
        """Token IDs in, unit-norm ``[hidden_size]`` FP32 vector out.

        The cache is sized to this sequence rather than to a maximum context,
        because one encode is one prefill: this checkpoint's 224 KiB per token
        would make an 8192-token budget allocate 1792 MiB of K/V that a 20-token
        sentence (4.4 MiB) never touches. Long inputs are still processed in
        chunks so the attention score matrix stays bounded.
        """
        ids = [int(token_id) for token_id in input_ids]
        if not ids:
            raise UnsupportedInputError("the text tokenized to zero tokens; nothing to embed")
        if prefill_chunk_size < 1:
            raise ValueError(f"prefill_chunk_size must be >= 1, got {prefill_chunk_size}")

        cache = self.decoder.new_cache(len(ids))
        chunks = [
            ids[start : start + prefill_chunk_size]
            for start in range(0, len(ids), prefill_chunk_size)
        ]
        decoded = None
        for chunk in chunks:
            decoded = self.decoder.encode(chunk, cache, probe=probe)
        assert decoded is not None  # ``ids`` is non-empty, so there is a last chunk
        # The last valid token of the sequence is the last row of the *last
        # chunk*, so the count handed to the pool is that chunk's length, not
        # len(ids). Passing len(ids) would ask for a row the decoder never
        # returned whenever the text was longer than one chunk.
        pooled = last_token_pool(decoded.hidden_states, len(chunks[-1]))
        return l2_normalize(pooled)


class EmbeddingEngine:
    """Tokenizer + :class:`EmbeddingModel`."""

    def __init__(
        self,
        model: EmbeddingModel,
        tokenizer: Tokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        load_seconds: float = float("nan"),
    ) -> None:
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")
        if max_length > model.config.max_position_embeddings:
            raise ConfigError(
                f"max_length={max_length} exceeds max_position_embeddings="
                f"{model.config.max_position_embeddings}"
            )
        self.model = model
        self.config = model.config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefill_chunk_size = prefill_chunk_size
        self.load_seconds = load_seconds

    @property
    def dimension(self) -> int:
        return self.model.dimension

    @classmethod
    def from_directory(
        cls,
        model_dir: str | Path,
        max_length: int = DEFAULT_MAX_LENGTH,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
    ) -> "EmbeddingEngine":
        """Load a converted embedding model. Fully offline."""
        with stopwatch() as elapsed:
            root = Path(model_dir)
            config, weights = load_config_and_weights(root)
            if config.task != TASK_TEXT_EMBEDDING:
                raise ConfigError(
                    f"{root} is a converted {config.task!r} model, not "
                    f"{TASK_TEXT_EMBEDDING!r}. Load it with slowq.Engine "
                    "(CLI: ``slowq generate``) instead."
                )
            # No chat template is required: the official recipe tokenises the raw
            # text, and an embedding checkpoint that shipped without a template is
            # still perfectly usable.
            tokenizer = Tokenizer.from_directory(root, require_chat_template=False)
        return cls(
            model=EmbeddingModel(config, weights),
            tokenizer=tokenizer,
            max_length=max_length,
            prefill_chunk_size=prefill_chunk_size,
            load_seconds=elapsed[0],
        )

    # ---- public API ---------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        is_query: bool = False,
        instruction: str | None = DEFAULT_QUERY_INSTRUCTION,
        probe: DebugProbe | None = None,
    ) -> EmbeddingResult:
        """Encode one text.

        ``is_query=True`` applies the official instruction prefix; documents are
        encoded verbatim. The prefix is not applied to both sides, and no chat
        template is rendered for either.
        """
        if not isinstance(text, str):
            raise UnsupportedInputError(f"text must be a string, got {type(text).__name__}")
        payload = format_query(text, instruction) if is_query else text
        input_ids = self._tokenize(payload)
        with stopwatch() as elapsed:
            vector = self.model.encode_ids(
                input_ids, prefill_chunk_size=self.prefill_chunk_size, probe=probe
            )
        return EmbeddingResult(
            vector=vector,
            text=payload,
            input_ids=tuple(input_ids),
            num_tokens=len(input_ids),
            encode_seconds=elapsed[0],
            is_query=is_query,
        )

    def encode_query(
        self, query: str, *, instruction: str | None = DEFAULT_QUERY_INSTRUCTION
    ) -> EmbeddingResult:
        return self.encode(query, is_query=True, instruction=instruction)

    def encode_document(self, document: str) -> EmbeddingResult:
        """Encode a retrieval document. No instruction prefix is added."""
        return self.encode(document, is_query=False)

    def encode_many(
        self,
        texts: Sequence[str],
        *,
        is_query: bool = False,
        instruction: str | None = DEFAULT_QUERY_INSTRUCTION,
        with_similarity: bool = True,
    ) -> EmbeddingBatch:
        """Encode several texts in order and optionally score them against each other.

        Sequential by design. Nothing is padded and nothing is run as a batch, so
        the vectors are the ones a single-text encode would produce.
        """
        results = tuple(
            self.encode(text, is_query=is_query, instruction=instruction) for text in texts
        )
        similarity = (
            similarity_matrix([r.vector for r in results], [r.vector for r in results])
            if with_similarity and results
            else None
        )
        return EmbeddingBatch(results=results, similarity=similarity)

    def search(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str | None = DEFAULT_QUERY_INSTRUCTION,
    ) -> tuple[EmbeddingBatch, list[tuple[int, float]]]:
        """The minimal retrieval example: one query against a list of documents.

        Returns the embeddings and the document indices ranked by descending
        cosine score. This is the whole of SlowQ's retrieval support; there is no
        vector store and no RAG pipeline behind it.

        Every call re-encodes every document. For one query that is the honest
        cost of the answer; for several queries against the same corpus it is
        wasted work, and :meth:`index_documents` is what to use instead.
        """
        if not documents:
            raise UnsupportedInputError("search needs at least one document")
        encoded_query = self.encode_query(query, instruction=instruction)
        encoded_documents = [self.encode_document(text) for text in documents]
        scores, ranking = rank_documents(encoded_query.vector, encoded_documents)
        batch = EmbeddingBatch(results=(encoded_query, *encoded_documents), similarity=scores)
        return batch, ranking

    def index_documents(self, documents: Sequence[str]) -> "DocumentIndex":
        """Encode a corpus once, then query it repeatedly.

        ``search`` costs one encode per document per call, so N queries over M
        documents cost N*M encodes where M would do. An index pays the M up front
        and reduces each later query to a single encode plus a matrix multiply::

            index = engine.index_documents(corpus)
            batch, ranking = index.search("first question")
            batch, ranking = index.search("second question")

        The documents are encoded by exactly the same :meth:`encode_document`
        ``search`` uses — verbatim, no instruction prefix — and the ranking comes
        from the same :func:`rank_documents`, so ``index.search(q)`` and
        ``engine.search(q, corpus)`` return the same vectors and the same order.
        The index holds those vectors for as long as it is alive, which is the
        trade being offered: memory in exchange for not recomputing.
        """
        if not documents:
            raise UnsupportedInputError("index_documents needs at least one document")
        with stopwatch() as elapsed:
            results = tuple(self.encode_document(text) for text in documents)
        return DocumentIndex(self, results, build_seconds=elapsed[0])

    # ---- internals ----------------------------------------------------------

    def _tokenize(self, text: str) -> list[int]:
        """Token IDs for ``text``, with the checkpoint's own post-processor applied.

        ``add_special_tokens=True`` is what the official recipe does: it calls
        ``tokenizer(texts, padding=True, truncation=True, max_length=…)`` and
        that default runs ``tokenizer.json``'s post-processor. For
        ``Qwen/Qwen3-Embedding-0.6B`` the post-processor is a
        ``TemplateProcessing`` that appends ``<|fim_suffix|>`` (id 151643), so every
        official vector was pooled at that token — dropping it produces a
        different vector from a model that looks like it is doing the same
        thing. Measured, not assumed: the three generation checkpoints SlowQ
        supports add nothing either way, and this one adds exactly one token.

        The generation path deliberately encodes with ``False`` instead, because
        it renders a chat template first and ``apply_chat_template`` tokenises
        its own output that way. Same library, two official call sites, two
        answers; ``tools/validate_reference.py`` compares the IDs on both.
        """
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(input_ids) > self.max_length:
            raise ContextOverflowError(
                f"the text tokenized to {len(input_ids)} tokens, over "
                f"max_length={self.max_length}. SlowQ does not truncate: a shorter "
                "input embeds to a different vector, so passing a longer one and "
                "getting a truncated one back would be a silent wrong answer. "
                "Raise --max-length if the checkpoint's positions allow it."
            )
        return input_ids


class DocumentIndex:
    """A corpus encoded once and queried many times.

    Built by :meth:`EmbeddingEngine.index_documents`, which is where the
    motivation lives. The vectors are held for as long as the index is, so a
    thousand Qwen3-Embedding-0.6B documents cost about 4 MiB — against the
    minutes of re-encoding that ``search`` would spend on every additional query.

    Each :meth:`search` still encodes the query. That is deliberate: the query is
    what varies between calls, and caching it too would turn a retrieval API into
    a memo table whose hit rate depends on someone asking the same question twice.

    Stale-index protection mirrors :class:`slowq.engine.Session`. The model, the
    tokenizer and the encoding settings are recorded at build time and re-checked
    before every search, so an engine that has since been reloaded or
    reconfigured cannot silently score a fresh query against vectors a different
    configuration produced. The scores would still look like cosine similarities;
    they would just not mean anything.

    **Vector ownership.** The index owns a private copy of every document vector
    and detaches again on the way out, so :attr:`results` and :meth:`search` both
    return arrays the caller may write to — the same semantics
    :meth:`EmbeddingEngine.search` has, which is why the two stay
    interchangeable. :func:`detached` explains what goes wrong without this:
    ``EmbeddingResult`` is frozen, but frozen does not reach through to the
    ``ndarray``, and an index that shared its buffers let a caller re-score the
    corpus permanently.
    """

    #: Names for the entries of :meth:`_encoding_identity`, in order, so a changed
    #: setting can be reported by name instead of as two opaque tuples.
    _ENCODING_FIELDS = ("prefill_chunk_size", "weights")

    def __init__(
        self,
        engine: EmbeddingEngine,
        results: Sequence[EmbeddingResult],
        *,
        build_seconds: float = float("nan"),
    ) -> None:
        if not results:
            raise UnsupportedInputError("a DocumentIndex needs at least one document")
        self._engine = engine
        self._model = engine.model
        self._tokenizer = engine.tokenizer
        self._encoding = self._encoding_identity(engine)
        # Detached on the way in *and* on the way out, which is twice and not
        # once by design: the caller of ``__init__`` may keep writing to the
        # results it passed (that is route 1 of the leak), and the caller of
        # ``results``/``search`` may keep writing to what it gets back (routes 2
        # and 3). One copy at either end alone leaves the other route open.
        self._results: tuple[EmbeddingResult, ...] = tuple(
            detached(result) for result in results
        )
        #: Wall time of the encoding pass that built this index.
        self.build_seconds = build_seconds
        self._searches = 0

    # ---- what is in it ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._results)

    @property
    def results(self) -> tuple[EmbeddingResult, ...]:
        """The per-document results, in the order they were indexed.

        Freshly detached on every access, so the vectors are **writable and owned
        by the caller** — the same semantics :meth:`EmbeddingEngine.search` has,
        which re-encodes and therefore also hands back arrays nobody else holds.
        Writing to one changes nothing inside the index.
        """
        return tuple(detached(result) for result in self._results)

    @property
    def documents(self) -> tuple[str, ...]:
        """The indexed texts, in order. ``results[i].text`` is the same string."""
        return tuple(result.text for result in self._results)

    @property
    def vectors(self) -> np.ndarray:
        """``[num_documents, hidden_size]``, freshly stacked on every access.

        Stacked rather than stored so that the index never hands out a buffer it
        also scores against: a caller that normalises or edits the array it gets
        back changes only its own copy. Stacking a thousand unit vectors is
        microseconds next to the encoding that produced them.
        """
        return np.stack([result.vector for result in self._results])

    @property
    def dimension(self) -> int:
        return int(self._results[0].vector.shape[-1])

    @property
    def num_tokens(self) -> int:
        """Tokens across the whole corpus — what one rebuild would re-prefill."""
        return sum(result.num_tokens for result in self._results)

    # ---- querying -----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        instruction: str | None = DEFAULT_QUERY_INSTRUCTION,
    ) -> tuple[EmbeddingBatch, list[tuple[int, float]]]:
        """One query against the pre-encoded corpus.

        Returns exactly what ``EmbeddingEngine.search(query, documents)`` returns:
        an :class:`EmbeddingBatch` of ``(query, *documents)`` carrying the
        similarity row, and the document indices ranked by descending cosine
        score. Same shapes, same ranking function, same vectors — and, like that
        method, arrays the caller owns and may write to without reaching the
        index.

        Scoring reads the internal buffers and only the returned batch is
        detached, so the copy happens once per search rather than once per
        document per comparison.
        """
        self._check_identity()
        encoded_query = self._engine.encode_query(query, instruction=instruction)
        scores, ranking = rank_documents(encoded_query.vector, self._results)
        self._searches += 1
        batch = EmbeddingBatch(
            results=(encoded_query, *(detached(result) for result in self._results)),
            similarity=scores,
        )
        return batch, ranking

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self._results),
            "tokens": self.num_tokens,
            "dimension": self.dimension,
            "build_seconds": self.build_seconds,
            "searches": self._searches,
        }

    def __repr__(self) -> str:
        return (
            f"<DocumentIndex {len(self._results)} document(s), dim {self.dimension}, "
            f"{self._searches} search(es)>"
        )

    # ---- staleness ----------------------------------------------------------

    @staticmethod
    def _encoding_identity(engine: EmbeddingEngine) -> tuple[Any, ...]:
        """The settings a stored vector depends on, beyond model and tokenizer.

        Two entries, and both are there because a stored vector really would
        differ:

        * ``prefill_chunk_size``, even though chunking is meant to be
          transparent — it changes the order of the floating-point work in
          :meth:`EmbeddingModel.encode_ids`, and the project's own
          chunk-invariance tests allow a relative L2 difference of 2e-4 rather
          than asserting bit-equality. Vectors that differ by that much are
          still comparable, but they are not the vectors this index was built
          from, and guessing which side moved is not worth the risk;
        * the weight fingerprint, because the same engine pointed at a
          re-converted or different checkpoint encodes to different numbers.

        ``max_length`` is deliberately absent. It only bounds what gets
        *refused*: no vector in this index was produced by a truncated encode,
        since SlowQ raises :class:`ContextOverflowError` instead of truncating.
        Raising the limit afterwards changes nothing about the stored numbers,
        so refusing to search over it would be an annoyance with no correctness
        behind it. ``model.dimension`` is absent for the same kind of reason —
        a different dimension means a different model, which the identity check
        in :meth:`_check_identity` already catches by name.

        The tuple's order has to match :attr:`_ENCODING_FIELDS`, which is what
        turns a changed entry into a readable error instead of two opaque ones.
        """
        return (
            engine.prefill_chunk_size,
            engine.model.weights.fingerprint(),
        )

    def _check_identity(self) -> None:
        engine = self._engine
        if engine.model is not self._model:
            raise RuntimeError(
                "the engine's model was replaced after this index was built, so "
                "these vectors came from a different one. Build a new index."
            )
        if engine.tokenizer is not self._tokenizer:
            raise RuntimeError(
                "the engine's tokenizer was replaced after this index was built. "
                "A query tokenised by the new one would be scored against "
                "documents tokenised by the old one. Build a new index."
            )
        current = self._encoding_identity(engine)
        if current != self._encoding:
            changed = [
                name
                for name, before, after in zip(
                    self._ENCODING_FIELDS, self._encoding, current
                )
                if before != after
            ]
            raise RuntimeError(
                f"the engine's {' and '.join(changed)} changed after this index was "
                "built, so a query encoded now is not comparable with the document "
                "vectors it would be scored against. Build a new index."
            )
