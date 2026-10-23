"""Real-weight integration tests for the ``text-embedding`` task.

Enable them by naming a converted embedding directory::

    SLOWQ_MODEL_DIR=models/Qwen3-Embedding-0.6B-slowq python -m pytest -m integration

``tests/test_embedding.py`` already covers the recipe against a tiny synthetic
checkpoint. What only the real weights can show is that the recipe survives
contact with a model whose hidden state is 1024 wide and whose vocabulary is
151669 wide, whose tokenizer really does append ``<|fim_middle|>``, and whose
official instruction prefix really does change the tokenisation. So the
assertions here are about quantities that a plausible-looking but wrong
implementation would get wrong:

* the pooled vector is the *last* token's final hidden state — measured against
  the first token's and the mean over all tokens, on real activations;
* the vector is ``hidden_size`` wide, not ``vocab_size`` wide, and unit-norm;
* the runtime tokenises the recorded samples to exactly the IDs the official
  tokenizer produced when the reference was captured;
* nothing in the encode path generates a token or reaches the network.

The numeric agreement with the official implementation itself is graded by
``test_integration_reference.py``, which runs for every task; this module is
about the invariants around it.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import real_models
from slowq.config import ModelConfig
from slowq.embedding import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_QUERY_INSTRUCTION,
    EmbeddingEngine,
    format_query,
    similarity_matrix,
)
from slowq.engine import Engine
from slowq.errors import ConfigError, ContextOverflowError, UnsupportedInputError
from slowq.model import DebugProbe
from slowq.ops import l2_normalize

TASK = "text-embedding"

#: ``<|fim_middle|>``, which this checkpoint's own post-processor appends and at
#: which the official recipe pools. Asserted by ID rather than by name so a
#: tokenizer that quietly stopped adding it fails here.
TRAILING_SPECIAL_TOKEN_ID = 151643

#: Measured on the real checkpoint before being written down, for the query
#: "Which planet in the solar system has the most moons?" with the official
#: instruction prefix (31 tokens): cosine(last, mean) 0.382, cosine(last, first)
#: 0.124. Both poolings are therefore observably different vectors, not two
#: spellings of the same one, and the bounds below leave the measurement room to
#: move without becoming a test that only fails on a real regression.
MAX_COSINE_TO_MEAN_POOLING = 0.9
MAX_COSINE_TO_FIRST_TOKEN = 0.5

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        (reason := real_models.skip_reason(TASK)) is not None,
        reason=reason or "",
    ),
]


@pytest.fixture(scope="module")
def model_dir() -> Path:
    directory = real_models.selected_model_dir()
    assert directory is not None  # guaranteed by the module-level skip
    return directory


@pytest.fixture(scope="module")
def engine(model_dir: Path) -> EmbeddingEngine:
    """One engine for the whole module: the weights are memory-mapped once."""
    return EmbeddingEngine.from_directory(model_dir)


@pytest.fixture(scope="module")
def reference(model_dir: Path) -> dict:
    """The recorded official capture, or a skip when there is none.

    Only the tests that compare against what the official tokenizer produced
    need it; the rest run on any converted embedding directory.
    """
    if not real_models.reference_is_complete(model_dir):
        pytest.skip(
            f"no reference capture in {real_models.artifacts_dir(model_dir)}; run "
            "'python tools/validate_reference.py ref' first"
        )
    return real_models.read_json(real_models.artifacts_dir(model_dir) / "reference.json")


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    """The capture protocol's own embedding samples, not a copy of them.

    Taken from the validation tool so that changing a sample there changes what
    is asserted here, instead of leaving this module grading texts nobody
    captures any more.
    """
    return real_models.load_validation_tool().EMBEDDING_SAMPLES


# --------------------------------------------------------------------------- #
# the converted checkpoint
# --------------------------------------------------------------------------- #


def test_loading_never_reaches_the_network(model_dir: Path, monkeypatch):
    """A converted directory is complete; a download attempt is a bug."""
    real_models.block_network(monkeypatch)
    assert EmbeddingEngine.from_directory(model_dir).dimension > 0


def test_the_generation_engine_refuses_an_embedding_directory(model_dir: Path):
    """The task guard, on the checkpoint that most needs it.

    This directory publishes ``model_type: "qwen3"`` and
    ``architectures: ["Qwen3ForCausalLM"]`` — byte-for-byte what the chat model
    publishes. Only the manifest's recorded task tells the two apart, so loading
    it as a generator has to fail loudly rather than produce tokens from a
    checkpoint that was trained to produce vectors.
    """
    with pytest.raises(ConfigError, match="text-embedding"):
        Engine.from_directory(model_dir, max_context=128)


def test_the_config_is_the_one_the_checkpoint_declares(engine: EmbeddingEngine, model_dir: Path):
    """Every field, re-read from ``config.json``, not just the ones used below."""
    reread = ModelConfig.from_json_file(model_dir / "config.json", task=TASK)
    live = engine.config
    for field in fields(ModelConfig):
        if field.name == "raw":
            continue
        assert getattr(live, field.name) == getattr(reread, field.name), field.name


def test_the_vector_is_a_hidden_state_and_not_a_vocabulary_distribution(
    engine: EmbeddingEngine,
):
    """The dimension that distinguishes the two, on real shapes.

    A synthetic checkpoint can be built with ``hidden_size == vocab_size``, which
    would let an implementation that pooled the logits pass every test in
    ``tests/test_embedding.py``. Here the two differ by two orders of magnitude.
    """
    result = engine.encode_document("Saturn has 146 confirmed moons.")
    assert result.vector.shape == (engine.config.hidden_size,)
    assert result.vector.dtype == np.float32
    assert engine.dimension == engine.config.hidden_size
    assert engine.config.hidden_size != engine.config.vocab_size
    assert result.dimension == engine.config.hidden_size


# --------------------------------------------------------------------------- #
# the official recipe, on real activations
# --------------------------------------------------------------------------- #


def test_the_pooled_row_is_the_last_token_and_the_choice_is_observable(
    engine: EmbeddingEngine,
):
    """Last-token pooling, checked against the two plausible substitutes.

    ``final_norm`` is captured at *every* position of a single-chunk prefill, so
    the mean and the first token are computed from the same activations the
    recipe pools from rather than from a second forward. The returned vector must
    be exactly the normalised last row — bit for bit, since both come from the
    same FP32 array — and must be far from the other two.
    """
    text = format_query("Which planet in the solar system has the most moons?")
    ids = engine.tokenizer.encode(text, add_special_tokens=True)
    probe = DebugProbe.for_names({"final_norm"}, positions=tuple(range(len(ids))))
    vector = engine.model.encode_ids(ids, prefill_chunk_size=len(ids), probe=probe)

    rows = np.stack([probe.records[f"final_norm[{index}]"] for index in range(len(ids))])
    assert rows.shape == (len(ids), engine.config.hidden_size)
    assert np.array_equal(vector, l2_normalize(rows[-1]))

    mean_pooled = l2_normalize(rows.mean(axis=0))
    first_pooled = l2_normalize(rows[0])
    assert float(vector @ mean_pooled) < MAX_COSINE_TO_MEAN_POOLING, (
        "mean pooling produced (nearly) the official vector, so this test cannot "
        "tell the two recipes apart on this checkpoint"
    )
    assert float(vector @ first_pooled) < MAX_COSINE_TO_FIRST_TOKEN


def test_the_query_prefix_is_applied_to_queries_only(engine: EmbeddingEngine):
    """One text, two encodings, and the difference is the official prefix."""
    query = "Which planet in the solar system has the most moons?"
    prefixed = engine.encode_query(query)
    bare = engine.encode(query, is_query=False)

    assert prefixed.is_query and not bare.is_query
    assert prefixed.text == f"Instruct: {DEFAULT_QUERY_INSTRUCTION}\nQuery:{query}"
    assert bare.text == query
    # The prefix is 23 tokens of this tokenizer, so it is not a no-op: the two
    # vectors differ and the two token counts differ by the prefix's length.
    assert prefixed.num_tokens > bare.num_tokens
    assert not np.array_equal(prefixed.vector, bare.vector)


def test_encoding_is_deterministic(engine: EmbeddingEngine):
    first = engine.encode_document("Jupiter's diameter is about eleven times that of Earth.")
    second = engine.encode_document("Jupiter's diameter is about eleven times that of Earth.")
    assert np.array_equal(first.vector, second.vector)
    assert first.input_ids == second.input_ids


def test_a_chunked_encode_matches_a_single_prefill(engine: EmbeddingEngine):
    """Chunking changes FP32 reduction order and nothing else.

    The gate is the internal one (rel_l2 1e-5), which is 100x tighter than the
    cross-implementation gate the reference comparison uses.
    """
    ids = engine.tokenizer.encode(
        format_query("Which planet in the solar system has the most moons?"),
        add_special_tokens=True,
    )
    whole = engine.model.encode_ids(ids, prefill_chunk_size=len(ids))
    for chunk in (1, 3, 7):
        partial = engine.model.encode_ids(ids, prefill_chunk_size=chunk)
        real_models.assert_same_numerics(
            partial, whole, f"chunked encode ({chunk} tokens per forward)",
            check_argmax=False,
        )


def test_similarity_between_unit_vectors_is_their_dot_product(engine: EmbeddingEngine, samples):
    """The whole retrieval example, on real vectors.

    Asserted as invariants rather than as a quality judgement: the matrix is
    symmetric, its diagonal is 1 to within float32 rounding because a unit vector
    dots with itself, every score is a cosine, and the ranking is a permutation
    of the documents. The vectors ``encode_many`` returns must then be
    *bit-identical* to encoding one text at a time, which is the claim
    ``EmbeddingBatch``'s docstring makes.
    """
    sample = samples[0]
    query, documents = sample["query"], list(sample["documents"])

    batch, ranking = engine.search(query, documents)
    scores = batch.similarity
    assert scores is not None
    assert scores.shape == (1, len(documents))
    assert np.all(np.abs(scores) <= 1.0 + 1e-6)
    assert [index for index, _ in sorted(ranking)] == list(range(len(documents)))
    assert ranking[0][1] == pytest.approx(float(scores[0][ranking[0][0]]))
    assert np.array_equal(
        scores,
        similarity_matrix(
            [engine.encode_query(query).vector],
            [engine.encode_document(text).vector for text in documents],
        ),
    )

    texts = [query, *documents]
    encoded = engine.encode_many(texts)
    matrix = encoded.similarity
    assert matrix is not None
    assert matrix.shape == (len(texts), len(texts))
    one_at_a_time = np.stack([engine.encode_document(text).vector for text in texts])
    # The vectors are bit-identical, which is the claim under test: encoding in a
    # loop is encoding one at a time.
    assert np.array_equal(encoded.vectors, one_at_a_time)
    # The scores are not, because ``A @ A.T`` is free to reach a different BLAS
    # kernel than ``A @ B`` and sum the same 1024 products in a different order.
    # Recomputing the product by hand is still worth doing: it pins the matrix to
    # a dot product of the vectors above rather than to anything else.
    assert np.allclose(matrix, one_at_a_time @ one_at_a_time.T, rtol=0.0, atol=1e-6)
    assert np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-6)
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-5)
    assert encoded.vectors.shape == (len(texts), engine.config.hidden_size)

    # ``encode_many`` encodes the document side by default, so its first row is
    # the *bare* query while ``search`` prefixes it. The two disagreeing is the
    # instruction prefix doing its job, not an inconsistency: measured on this
    # checkpoint, the prefix moves the query's score for the first document from
    # 0.7284 to 0.7566 and the second one's from 0.2525 to 0.1242.
    assert not np.allclose(matrix[0][1:], scores[0])


def test_the_encode_path_never_projects_onto_the_vocabulary(
    engine: EmbeddingEngine, monkeypatch
):
    """No generation and no ``lm_head`` multiply, checked by making one fatal.

    The tied head this checkpoint stores is a real ``[151669, 1024]`` matrix, so
    a 1024-wide result cannot have come out of it — but "cannot" is an argument,
    not a test. Patching the projection to raise makes the claim executable: if
    any code path inside an encode reaches for the vocabulary, this fails.
    """

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("the encode path multiplied by lm_head")

    monkeypatch.setattr(engine.model.decoder, "output_projection", refuse)
    result = engine.encode_document("Saturn has 146 confirmed moons.")
    assert result.dimension == engine.config.hidden_size
    # The embedding model exposes a decoder and nothing that can sample a token.
    assert not hasattr(engine.model, "generate")
    assert not hasattr(engine.model, "forward")


def test_a_text_over_max_length_is_refused_rather_than_truncated(model_dir: Path):
    """Refusal, not truncation: a shortened text embeds to a different vector."""
    small = EmbeddingEngine.from_directory(model_dir, max_length=4)
    with pytest.raises(ContextOverflowError, match="does not truncate|max_length=4"):
        small.encode_document("Saturn has 146 confirmed moons, more than any other planet.")


def test_an_empty_text_is_refused(engine: EmbeddingEngine):
    """Whatever tokenises to zero tokens cannot be pooled."""
    with pytest.raises((UnsupportedInputError, ValueError)):
        engine.model.encode_ids([])


def test_the_official_max_length_is_the_documented_default():
    assert DEFAULT_MAX_LENGTH == 8192


# --------------------------------------------------------------------------- #
# against the recorded reference capture
# --------------------------------------------------------------------------- #


def test_the_runtime_tokenises_the_samples_to_the_recorded_ids(
    engine: EmbeddingEngine, reference: dict, samples
):
    """The IDs the official tokenizer produced, reproduced by this runtime.

    ``compare`` checks the same thing inside the validation tool; this checks it
    through the public API a caller actually uses, including the trailing
    ``<|fim_middle|>`` the checkpoint's post-processor adds and at which the
    official recipe pools.
    """
    recorded = {entry["name"]: entry for entry in reference["tokenizer"]}
    pooled = {
        (sample["name"], text["name"]): text
        for sample in reference["samples"]
        for text in sample["texts"]
    }

    for sample in samples:
        texts = [("query", sample["query"], True)] + [
            (f"doc{index}", document, False)
            for index, document in enumerate(sample["documents"])
        ]
        for text_name, text, is_query in texts:
            result = engine.encode(text, is_query=is_query, instruction=sample["instruction"])
            entry = recorded[f"{sample['name']}/{text_name}"]
            assert result.text == entry["text"]
            assert list(result.input_ids) == entry["ids"]
            assert result.input_ids[-1] == TRAILING_SPECIAL_TOKEN_ID

            metadata = pooled[(sample["name"], text_name)]
            assert result.num_tokens == metadata["num_tokens"]
            # The official pool index is the last valid token of an unpadded
            # sequence, which is what ``last_token_pool`` takes.
            assert metadata["pool_index"] == metadata["num_tokens"] - 1
            assert metadata["dimension"] == engine.config.hidden_size


def test_the_recorded_architecture_is_the_one_that_loaded(
    engine: EmbeddingEngine, reference: dict, model_dir: Path
):
    """The capture's own reading of ``config.json`` agrees with the live one."""
    tool = real_models.load_validation_tool()
    assert tool.architecture_fingerprint(engine.config) == reference["architecture"]
    manifest = real_models.read_json(model_dir / "manifest.json")
    assert manifest["task"] == TASK
    assert reference["capture"]["task"] == TASK
    assert reference["source"]["revision"] == manifest["source"]["revision"]


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #


def test_the_cli_encodes_offline(model_dir: Path, monkeypatch, capsys):
    """The whole user-facing path, with the network unplugged."""
    import json

    from slowq import cli

    real_models.block_network(monkeypatch)
    code = cli.main(
        [
            "embed",
            "--model",
            str(model_dir),
            "--query",
            "Which planet in the solar system has the most moons?",
            "--document",
            "Saturn has 146 confirmed moons, more than any other planet in the solar system.",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dimension"] == 1024
    assert len(payload["queries"]) == len(payload["documents"]) == 1
    assert len(payload["queries"][0]["vector"]) == payload["dimension"]
    assert payload["queries"][0]["is_query"] is True
    assert payload["documents"][0]["is_query"] is False
    assert payload["queries"][0]["input_ids"][-1] == TRAILING_SPECIAL_TOKEN_ID
    assert len(payload["similarity"]) == 1 and len(payload["similarity"][0]) == 1


def test_the_cli_refuses_to_encode_nothing(model_dir: Path, capsys):
    from slowq import cli

    assert cli.main(["embed", "--model", str(model_dir)]) == 2
    assert "at least one --query or --document" in capsys.readouterr().err
