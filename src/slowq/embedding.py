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

from dataclasses import dataclass
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
        """
        if not documents:
            raise UnsupportedInputError("search needs at least one document")
        encoded_query = self.encode_query(query, instruction=instruction)
        encoded_documents = [self.encode_document(text) for text in documents]
        scores = similarity_matrix(
            [encoded_query.vector], [result.vector for result in encoded_documents]
        )
        ranking = sorted(
            ((index, float(score)) for index, score in enumerate(scores[0])),
            key=lambda item: item[1],
            reverse=True,
        )
        batch = EmbeddingBatch(results=(encoded_query, *encoded_documents), similarity=scores)
        return batch, ranking

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
