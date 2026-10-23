"""Quick tests for the text-embedding path: input format, pooling, normalisation.

Everything here runs on the tiny converted directory from ``conftest``, so no
network and no real weights. What the tests pin down is the set of choices that
are easy to get wrong while still producing plausible-looking numbers:

* the query prefix is the official ``Instruct: …\\nQuery:…`` with no space after
  ``Query:``, and documents get no prefix at all;
* nothing renders the chat template and nothing adds a special token;
* pooling takes the **last** valid token's hidden state, not the mean and not
  the first;
* the output is L2-normalised, so a dot product is a cosine;
* the vector is ``hidden_size`` wide, never ``vocab_size`` wide — an encode that
  projected to the vocabulary would still return a unit vector;
* a text longer than ``max_length`` is refused rather than truncated;
* a multi-chunk encode lands on the same vector as a single-chunk one, **and**
  actually ran as several chunks — an equivalence assertion alone passes just as
  happily when nothing was split;

That last one is a regression: ``encode_ids`` used to pool at ``len(ids)`` while
``decoded.hidden_states`` held only the last chunk's rows, so any text longer
than ``prefill_chunk_size`` raised instead of embedding. Numerically comparing
the two chunkings against the official model is
``tools/validate_reference.py``'s job; this is the offline half.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.processors import TemplateProcessing

from slowq.config import ModelConfig
from slowq.embedding import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_PREFILL_CHUNK_SIZE,
    DEFAULT_QUERY_INSTRUCTION,
    EmbeddingBatch,
    EmbeddingEngine,
    EmbeddingModel,
    cosine_similarity,
    format_query,
    last_token_pool,
    similarity_matrix,
)
from slowq.errors import ConfigError, ContextOverflowError, UnsupportedInputError
from slowq.ops import l2_normalize
from slowq.weights import Weights

DOCUMENT = "慢炖用 NumPy 在 CPU 上跑小模型，一步一步把 token 炖出来。"
QUERY = "慢炖是怎么推理的？"
#: What Qwen3-Embedding's post-processor appends, and what the official vectors
#: were pooled at. Registered as a special token by ``conftest``.
EOS_TOKEN = "<|fim_middle|>"


@pytest.fixture
def engine(tiny_embedding_dir: Path) -> EmbeddingEngine:
    """A small chunk size so the chunked path is the one being tested."""
    return EmbeddingEngine.from_directory(
        tiny_embedding_dir, max_length=256, prefill_chunk_size=4
    )


@pytest.fixture
def eos_embedding_dir(tiny_embedding_dir: Path, tmp_path: Path) -> Path:
    """The tiny embedding directory, given Qwen3-Embedding's post-processor.

    The official checkpoint's ``tokenizer.json`` ends in a ``TemplateProcessing``
    that appends ``<|fim_suffix|>``, so calling the tokenizer with its default
    ``add_special_tokens`` yields one token more than calling it without — and
    the official recipe calls it with the default. The tiny tokenizer has no
    post-processor, so without this fixture nothing offline could tell the two
    encodings apart and the flag would be untested.
    """
    root = tmp_path / "embedding-with-eos"
    shutil.copytree(tiny_embedding_dir, root)
    path = root / "tokenizer.json"
    tokenizer = HFTokenizer.from_file(str(path))
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    assert eos_id is not None, "the tiny tokenizer must register the special token"
    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {EOS_TOKEN}",
        pair=f"$A $B {EOS_TOKEN}",
        special_tokens=[(EOS_TOKEN, eos_id)],
    )
    tokenizer.save(str(path))
    return root


# --------------------------------------------------------------------------- #
# 1. the input format: prefix, template, special tokens
# --------------------------------------------------------------------------- #


def test_the_query_prefix_is_the_official_one_down_to_the_missing_space():
    assert format_query("什么是慢炖") == (
        f"Instruct: {DEFAULT_QUERY_INSTRUCTION}\nQuery:什么是慢炖"
    )
    # ``Query:`` with no space is what get_detailed_instruct returns; a space
    # changes the tokenisation and therefore the vector.
    assert "Query: " not in format_query("什么是慢炖")


def test_a_custom_instruction_replaces_the_default():
    assert format_query("q", instruction="找相关段落") == "Instruct: 找相关段落\nQuery:q"


def test_instruction_none_leaves_the_query_alone():
    assert format_query("什么是慢炖", instruction=None) == "什么是慢炖"


def test_documents_are_encoded_verbatim_and_queries_are_not(engine):
    document = engine.encode_document(DOCUMENT)
    query = engine.encode_query(QUERY)
    assert document.text == DOCUMENT
    assert document.is_query is False
    assert query.text == f"Instruct: {DEFAULT_QUERY_INSTRUCTION}\nQuery:{QUERY}"
    assert query.is_query is True


def test_no_chat_template_is_rendered_for_either_side(engine):
    """The tiny template emits ``<|im_start|>``; the recipe must not reach it."""
    for result in (engine.encode_document(DOCUMENT), engine.encode_query(QUERY)):
        assert "<|im_start|>" not in result.text
        assert "<|im_end|>" not in result.text
        assert result.input_ids == tuple(
            engine.tokenizer.encode(result.text, add_special_tokens=True)
        )


def test_encode_defaults_to_the_document_side(engine):
    assert engine.encode(DOCUMENT).text == DOCUMENT


def test_the_checkpoint_post_processor_runs_and_its_token_is_what_gets_pooled(
    eos_embedding_dir: Path,
):
    """Qwen3-Embedding appends ``<|fim_middle|>``; the official vector is pooled there.

    ``transformers``' tokenizer defaults to ``add_special_tokens=True``, which
    runs ``tokenizer.json``'s post-processor. Encoding without it yields a
    plausible-looking unit vector from a model that is one token short of the
    official input, so the appended token has to be observable here.
    """
    engine = EmbeddingEngine.from_directory(
        eos_embedding_dir, max_length=256, prefill_chunk_size=4
    )
    eos_id = HFTokenizer.from_file(str(eos_embedding_dir / "tokenizer.json")).token_to_id(
        EOS_TOKEN
    )
    result = engine.encode_document(DOCUMENT)
    bare = engine.tokenizer.encode(DOCUMENT, add_special_tokens=False)

    assert result.input_ids[-1] == eos_id
    assert result.input_ids[:-1] == tuple(bare)
    assert result.num_tokens == len(bare) + 1


def test_a_non_string_input_is_refused(engine):
    with pytest.raises(UnsupportedInputError, match="must be a string"):
        engine.encode(["not", "a", "string"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 2. pooling: the last valid token, and nothing else
# --------------------------------------------------------------------------- #


def test_last_token_pool_picks_the_last_valid_row():
    rows = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert np.array_equal(last_token_pool(rows, 4), rows[3])
    assert np.array_equal(last_token_pool(rows, 2), rows[1])


def test_last_token_pool_refuses_a_count_the_rows_cannot_support():
    rows = np.zeros((3, 4), dtype=np.float32)
    for count in (0, -1, 4):
        with pytest.raises(ValueError, match="no last valid token"):
            last_token_pool(rows, count)


def test_last_token_pool_refuses_something_that_is_not_a_sequence_of_rows():
    with pytest.raises(ValueError, match=r"\[num_tokens, hidden_size\]"):
        last_token_pool(np.zeros(4, dtype=np.float32), 1)


def test_the_pooled_row_is_the_last_token_not_the_mean_or_the_first(engine):
    """Recomputed from the decoder directly, so this is not a restatement."""
    model = engine.model
    ids = engine.tokenizer.encode(DOCUMENT)
    assert len(ids) >= 3

    cache = model.decoder.new_cache(len(ids))
    rows = model.decoder.encode(ids, cache).hidden_states
    vector = model.encode_ids(ids, prefill_chunk_size=len(ids))

    np.testing.assert_allclose(vector, l2_normalize(rows[-1]), rtol=0, atol=1e-6)
    for other, label in ((rows[0], "first"), (rows.mean(axis=0), "mean")):
        assert not np.allclose(rows[-1], other, atol=1e-4), (
            f"the fixture cannot tell last-token pooling from {label}-token pooling"
        )
        assert not np.allclose(vector, l2_normalize(other), atol=1e-4)


def test_the_tiny_document_is_shorter_than_the_default_chunk_size(engine):
    """Why forwarding the chunk size is not a detail.

    At the default ``prefill_chunk_size`` this document is a *single* prefill, so
    an equivalence test that forgets to forward the engine's setting compares a
    vector with itself and still passes. That is exactly what the chunking test
    below used to do.
    """
    assert DEFAULT_PREFILL_CHUNK_SIZE == 128
    assert len(engine.tokenizer.encode(DOCUMENT)) < DEFAULT_PREFILL_CHUNK_SIZE


def _partition(num_tokens: int, chunk: int) -> list[int]:
    """The chunk lengths a real ``chunk``-sized prefill must produce."""
    return [min(chunk, num_tokens - start) for start in range(0, num_tokens, chunk)]


def _record_chunks(engine: EmbeddingEngine, monkeypatch) -> list[int]:
    """Every chunk length the decoder actually receives during one encode."""
    seen: list[int] = []
    real_encode = engine.model.decoder.encode

    def recording(chunk_ids, cache, **kwargs):
        seen.append(len(chunk_ids))
        return real_encode(chunk_ids, cache, **kwargs)

    monkeypatch.setattr(engine.model.decoder, "encode", recording)
    return seen


def test_a_multi_chunk_encode_matches_a_single_chunk_encode(
    tiny_embedding_dir: Path, monkeypatch
):
    """The regression: pooling used to be indexed by the whole sequence length.

    A text longer than ``prefill_chunk_size`` makes the decoder return only the
    last chunk's rows, so ``len(ids)`` was out of range and the encode raised.
    Chunking also has to be numerically invisible: same cache, same positions,
    same vector.

    Each case asserts the chunk lengths the decoder saw, not just the vector.
    Without that this test was vacuous: it built engines with a small
    ``prefill_chunk_size`` and then called ``model.encode_ids(ids)``, which falls
    back to the 128 default, so all seven cases ran as one 50-token chunk.
    """
    def load(chunk: int) -> EmbeddingEngine:
        return EmbeddingEngine.from_directory(
            tiny_embedding_dir, max_length=256, prefill_chunk_size=chunk
        )

    num_tokens = load(DEFAULT_PREFILL_CHUNK_SIZE).encode_document(DOCUMENT).num_tokens
    assert num_tokens > 3

    baseline_engine = load(num_tokens)
    baseline_chunks = _record_chunks(baseline_engine, monkeypatch)
    single = baseline_engine.encode_document(DOCUMENT).vector
    assert baseline_chunks == [num_tokens], "the baseline must be one prefill"
    assert len(single) == ModelConfig.from_json_file(
        tiny_embedding_dir / "config.json"
    ).hidden_size

    # 1 is the extreme; 3 and 7 do not divide 50, so the last chunk is short;
    # num_tokens - 1 leaves a one-token tail; the last two must collapse to one.
    for chunk in (1, 2, 3, 7, num_tokens - 1, num_tokens, num_tokens + 5):
        engine = load(chunk)
        seen = _record_chunks(engine, monkeypatch)
        vector = engine.encode_document(DOCUMENT).vector
        assert seen == _partition(num_tokens, chunk), (
            f"prefill_chunk_size={chunk} never reached the decoder; it saw {seen}"
        )
        if chunk < num_tokens:
            assert len(seen) > 1, f"prefill_chunk_size={chunk} did not split anything"
            assert seen[-1] == (num_tokens % chunk or chunk), "the tail chunk is wrong"
        np.testing.assert_allclose(
            vector, single, rtol=1e-5, atol=1e-6,
            err_msg=f"prefill_chunk_size={chunk} changed the vector",
        )


def test_encode_ids_honours_a_chunk_size_passed_to_it_directly(
    tiny_embedding_dir: Path, monkeypatch
):
    """The model-level parameter, independently of the engine that forwards it."""
    engine = EmbeddingEngine.from_directory(tiny_embedding_dir, max_length=256)
    ids = engine.tokenizer.encode(DOCUMENT)
    seen = _record_chunks(engine, monkeypatch)
    engine.model.encode_ids(ids, prefill_chunk_size=3)
    assert seen == _partition(len(ids), 3)


def test_the_default_chunk_size_is_smaller_than_a_real_document(engine):
    """Otherwise the chunked path above would never be the one that runs."""
    assert DEFAULT_PREFILL_CHUNK_SIZE == 128
    assert engine.prefill_chunk_size == 4
    assert len(engine.tokenizer.encode(DOCUMENT)) > engine.prefill_chunk_size


def test_encode_ids_refuses_an_empty_input(engine):
    with pytest.raises(UnsupportedInputError, match="zero tokens"):
        engine.model.encode_ids([])


def test_encode_ids_refuses_a_chunk_size_that_cannot_hold_a_token(engine):
    with pytest.raises(ValueError, match="prefill_chunk_size"):
        engine.model.encode_ids([1, 2, 3], prefill_chunk_size=0)


# --------------------------------------------------------------------------- #
# 3. the vector: normalised, hidden-sized, and not a vocabulary distribution
# --------------------------------------------------------------------------- #


def test_the_returned_vector_is_unit_norm(engine):
    for result in (engine.encode_document(DOCUMENT), engine.encode_query(QUERY)):
        assert result.vector.dtype == np.float32
        assert abs(float(np.linalg.norm(result.vector)) - 1.0) < 1e-6


def test_the_vector_is_hidden_size_wide_and_the_directory_has_no_head(
    engine, tiny_embedding_dir: Path
):
    """A full-vocab projection would still be unit-norm, so width is the check."""
    config = engine.config
    assert config.hidden_size != config.vocab_size
    assert engine.dimension == config.hidden_size
    assert engine.encode_document(DOCUMENT).dimension == config.hidden_size

    weights = Weights.from_directory(tiny_embedding_dir)
    assert not weights.has("lm_head.weight")
    manifest = json.loads((tiny_embedding_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"] == "text-embedding"
    assert manifest["aliases"] == {}


def test_cosine_similarity_of_unit_vectors_is_their_dot_product():
    left = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(left, left) == pytest.approx(1.0)
    assert cosine_similarity(left, right) == pytest.approx(0.0)
    assert cosine_similarity(left, -left) == pytest.approx(-1.0)


def test_cosine_similarity_does_not_renormalise_its_inputs():
    """Re-normalising here would hide a caller passing something unembedded."""
    two = np.array([2.0, 0.0], dtype=np.float32)
    assert cosine_similarity(two, two) == pytest.approx(4.0)


def test_cosine_similarity_refuses_a_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity(np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32))


def test_similarity_matrix_scores_every_query_against_every_document(engine):
    batch = engine.encode_many([DOCUMENT, QUERY, "another document"], with_similarity=True)
    assert isinstance(batch, EmbeddingBatch)
    assert batch.similarity is not None and batch.similarity.shape == (3, 3)
    np.testing.assert_allclose(
        np.diag(batch.similarity), np.ones(3), atol=1e-5,
        err_msg="a unit vector's similarity with itself must be 1",
    )
    np.testing.assert_allclose(batch.similarity, batch.similarity.T, atol=1e-6)
    assert np.abs(batch.similarity).max() <= 1.0 + 1e-5
    assert batch.vectors.shape == (3, engine.dimension)
    assert len(batch) == 3 and [result.text for result in batch]


def test_similarity_can_be_skipped_and_an_empty_side_is_refused(engine):
    assert engine.encode_many([DOCUMENT], with_similarity=False).similarity is None
    with pytest.raises(ValueError, match="at least one vector"):
        similarity_matrix([], [np.zeros(2, dtype=np.float32)])


# --------------------------------------------------------------------------- #
# 4. limits and refusals
# --------------------------------------------------------------------------- #


def test_a_text_over_max_length_is_refused_rather_than_truncated(tiny_embedding_dir: Path):
    engine = EmbeddingEngine.from_directory(tiny_embedding_dir, max_length=4)
    assert len(engine.tokenizer.encode(DOCUMENT)) > 4
    with pytest.raises(ContextOverflowError, match="does not truncate"):
        engine.encode_document(DOCUMENT)


def test_max_length_may_not_exceed_the_checkpoint_positions(tiny_embedding_dir: Path):
    positions = ModelConfig.from_json_file(tiny_embedding_dir / "config.json")
    with pytest.raises(ConfigError, match="max_position_embeddings"):
        EmbeddingEngine.from_directory(
            tiny_embedding_dir, max_length=positions.max_position_embeddings + 1
        )


def test_max_length_must_be_positive(tiny_embedding_dir: Path):
    with pytest.raises(ValueError, match="max_length"):
        EmbeddingEngine.from_directory(tiny_embedding_dir, max_length=0)


def test_the_default_max_length_is_the_official_one():
    assert DEFAULT_MAX_LENGTH == 8192


def test_a_text_generation_directory_is_refused_by_the_embedding_engine(
    tiny_model_dir: Path,
):
    """The two tasks read the same decoder, so the manifest has to decide."""
    with pytest.raises(ConfigError, match="not 'text-embedding'"):
        EmbeddingEngine.from_directory(tiny_model_dir)


def test_the_embedding_model_rejects_a_generation_config(tiny_model_dir: Path):
    config = ModelConfig.from_json_file(tiny_model_dir / "config.json")
    weights = Weights.from_directory(tiny_model_dir)
    assert config.task == "text-generation"
    with pytest.raises(ConfigError, match="not 'text-embedding'"):
        EmbeddingModel(config, weights)


def test_a_directory_without_a_chat_template_still_encodes(tmp_path: Path):
    """An embedding checkpoint that shipped no template is still usable."""
    root = tmp_path / "no-template"
    root.mkdir()
    from conftest import build_tiny_tokenizer, write_tiny_model
    from tokenizers import Tokenizer as HFTokenizer

    build_tiny_tokenizer(root)
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                **json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8")),
                "chat_template": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    vocab_size = HFTokenizer.from_file(str(root / "tokenizer.json")).get_vocab_size()
    write_tiny_model(root, vocab_size=vocab_size, task="text-embedding")

    engine = EmbeddingEngine.from_directory(root, max_length=64, prefill_chunk_size=4)
    assert engine.tokenizer.has_chat_template is False
    assert engine.encode_document(DOCUMENT).num_tokens > 0


# --------------------------------------------------------------------------- #
# 5. the minimal retrieval example
# --------------------------------------------------------------------------- #


def test_search_ranks_documents_by_descending_cosine(engine):
    documents = [DOCUMENT, QUERY, "a third unrelated passage about soup"]
    batch, ranking = engine.search(QUERY, documents)

    assert [result.text for result in batch.results] == [
        f"Instruct: {DEFAULT_QUERY_INSTRUCTION}\nQuery:{QUERY}",
        *documents,
    ]
    assert sorted(index for index, _ in ranking) == [0, 1, 2]
    scores = [score for _, score in ranking]
    assert scores == sorted(scores, reverse=True)
    assert batch.similarity is not None and batch.similarity.shape == (1, 3)
    for index, score in ranking:
        assert score == pytest.approx(float(batch.similarity[0][index]))
        assert -1.0 - 1e-5 <= score <= 1.0 + 1e-5


def test_search_without_an_instruction_leaves_the_query_bare(engine):
    batch, _ = engine.search(QUERY, [DOCUMENT], instruction=None)
    assert batch.results[0].text == QUERY


def test_search_needs_at_least_one_document(engine):
    with pytest.raises(UnsupportedInputError, match="at least one document"):
        engine.search(QUERY, [])
