"""Full-model forward: chunked prefill, cached decode, positions, and probes.

The central property under test is the one the spec calls out: a full-sequence
prefill, a chunked prefill, and token-by-token decode with a cache must produce
numerically identical results. Everything else here guards the invariants that
make that true.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer as HFTokenizer

from slowq import Engine
from slowq.cache import KVCache
from slowq.embedding import EmbeddingEngine
from slowq.errors import ConfigError, ContextOverflowError
from slowq.model import DebugProbe, CausalLMModel, load_config_and_weights

#: Patched in the mask tests below: ``encode`` looks these names up in the module
#: namespace at call time, so replacing them there is what observes the real path.
import slowq.model as model_module

from conftest import build_tiny_tokenizer, write_tiny_model

#: FP32 matmul reassociation between a long prefill and several short ones.
CHUNK_TOLERANCE = {"rtol": 2e-4, "atol": 2e-5}


@pytest.fixture
def model(tiny_model_dir):
    return CausalLMModel.from_directory(tiny_model_dir)


def run_chunks(model: CausalLMModel, ids: list[int], chunk: int):
    """Prefill ``ids`` in chunks of ``chunk`` and return the final result."""
    cache = model.new_cache(max_context=max(64, len(ids) + 8))
    result = None
    for start in range(0, len(ids), chunk):
        result = model.forward(ids[start : start + chunk], cache)
    return result, cache


# --------------------------------------------------------------------------- #
# shapes and bookkeeping
# --------------------------------------------------------------------------- #


def test_forward_projects_only_the_last_position(model: CausalLMModel):
    """No ``[T, vocab]`` logits tensor is ever materialised."""
    cache = model.new_cache(16)
    result = model.forward([1, 2, 3, 4, 5], cache)
    assert result.logits.shape == (model.config.vocab_size,)
    assert result.hidden_states.shape == (5, model.config.hidden_size)
    assert result.num_tokens == 5
    assert result.position_offset == 0


def test_cache_length_advances_once_per_forward_not_once_per_layer(model: CausalLMModel):
    cache = model.new_cache(16)
    model.forward([1, 2, 3], cache)
    # Every layer writes the same three slots; the length must not be multiplied.
    assert cache.length == 3
    assert cache.high_water_mark == 3


def test_forward_returns_the_cache_length_after_committing(model: CausalLMModel):
    cache = model.new_cache(16)
    result = model.forward([7, 8], cache)
    assert result.cache_length == 2


def test_second_forward_starts_at_the_cached_offset(model: CausalLMModel):
    cache = model.new_cache(16)
    model.forward([1, 2, 3], cache)
    second = model.forward([4], cache)
    assert second.position_offset == 3
    assert second.cache_length == 4


def test_forward_is_deterministic(model: CausalLMModel):
    first = model.forward([5, 6, 7], model.new_cache(16))
    second = model.forward([5, 6, 7], model.new_cache(16))
    np.testing.assert_array_equal(first.logits, second.logits)


def test_logits_stay_finite_and_in_float32(model: CausalLMModel):
    result = model.forward([1, 2, 3], model.new_cache(16))
    assert result.logits.dtype == np.float32
    assert np.all(np.isfinite(result.logits))
    assert np.all(np.isfinite(result.hidden_states))


def test_embed_rejects_out_of_vocabulary_ids(model: CausalLMModel):
    with pytest.raises(ValueError, match="outside the vocabulary"):
        model.embed([0, model.config.vocab_size])


def test_embed_rejects_a_negative_id(model: CausalLMModel):
    with pytest.raises(ValueError, match="outside the vocabulary"):
        model.embed([-1])


def test_embed_rejects_an_empty_input(model: CausalLMModel):
    with pytest.raises(ValueError, match="at least one token"):
        model.embed([])


def test_forward_rejects_an_empty_input(model: CausalLMModel):
    with pytest.raises(ValueError, match="at least one token"):
        model.forward([], model.new_cache(16))


# --------------------------------------------------------------------------- #
# the output projection is the tied embedding
# --------------------------------------------------------------------------- #


def test_output_projection_uses_the_tied_embedding_matrix(model: CausalLMModel):
    hidden = np.full(model.config.hidden_size, 0.25, dtype=np.float32)
    table = model.weights.get("model.embed_tokens.weight")
    np.testing.assert_allclose(model.output_projection(hidden), table @ hidden, rtol=1e-5)


def test_logits_are_the_final_normed_hidden_times_the_embedding(model: CausalLMModel):
    cache = model.new_cache(16)
    result = model.forward([3, 1, 4], cache)
    expected = model.output_projection(result.hidden_states[-1])
    np.testing.assert_allclose(result.logits, expected, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# full vs chunked vs cached decode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7])
def test_chunked_prefill_matches_a_single_prefill(model: CausalLMModel, chunk: int):
    ids = [11, 3, 7, 19, 2, 25, 8, 14, 1, 5, 30, 12]
    full, _ = run_chunks(model, ids, chunk=len(ids))
    chunked, cache = run_chunks(model, ids, chunk=chunk)
    np.testing.assert_allclose(full.logits, chunked.logits, **CHUNK_TOLERANCE)
    # ``hidden_states`` only covers the tokens of the current forward, so the
    # comparable slice is the last position of each.
    np.testing.assert_allclose(
        full.hidden_states[-1], chunked.hidden_states[-1], **CHUNK_TOLERANCE
    )
    assert cache.length == len(ids)


@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_chunk_boundaries_do_not_shift_positions(model: CausalLMModel, chunk: int):
    """Chunking must not restart RoPE positions at every chunk boundary."""
    ids = list(range(1, 10))
    reference, reference_cache = run_chunks(model, ids, chunk=len(ids))
    _, cache = run_chunks(model, ids, chunk=chunk)
    # Cached keys are stored after RoPE, so identical positions mean identical K.
    for layer in range(model.config.num_hidden_layers):
        np.testing.assert_allclose(
            cache.keys[layer], reference_cache.keys[layer], rtol=2e-4, atol=2e-5
        )
    assert reference.logits.shape == (model.config.vocab_size,)


def test_cached_token_by_token_decode_matches_one_long_prefill(model: CausalLMModel):
    """The property that makes incremental generation correct at all."""
    ids = [9, 4, 21, 2, 17, 6]

    prefill_cache = model.new_cache(32)
    prefill = model.forward(ids, prefill_cache)

    decode_cache = model.new_cache(32)
    step = None
    for token_id in ids:
        step = model.forward([token_id], decode_cache)
    assert step is not None
    np.testing.assert_allclose(prefill.logits, step.logits, **CHUNK_TOLERANCE)
    np.testing.assert_allclose(
        prefill.hidden_states[-1], step.hidden_states[-1], **CHUNK_TOLERANCE
    )


def test_prefill_then_decode_matches_a_longer_prefill(model: CausalLMModel):
    """Extend a sequence by one token two ways and compare the new logits."""
    prefix = [5, 12, 3, 27]
    extra = 8

    long_cache = model.new_cache(32)
    long = model.forward([*prefix, extra], long_cache)

    split_cache = model.new_cache(32)
    model.forward(prefix, split_cache)
    split = model.forward([extra], split_cache)

    np.testing.assert_allclose(long.logits, split.logits, **CHUNK_TOLERANCE)
    assert split_cache.length == long_cache.length == len(prefix) + 1


def test_a_cache_reset_makes_the_next_forward_reproducible(model: CausalLMModel):
    cache = model.new_cache(32)
    first = model.forward([1, 2, 3], cache).logits
    cache.reset()
    second = model.forward([1, 2, 3], cache).logits
    np.testing.assert_array_equal(first, second)


def test_two_caches_for_the_same_input_agree(model: CausalLMModel):
    ids = [2, 4, 6, 8, 10]
    a = model.forward(ids, model.new_cache(32)).logits
    b = model.forward(ids, model.new_cache(32)).logits
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# prefill(): the chunk loop with one vocabulary projection
#
# The optimisation is "stop projecting chunks whose logits are thrown away".
# That is only safe if what survives is *the same numbers*, so the identity
# tests below are exact rather than tolerant: a difference here would mean the
# chunking changed, not that FP32 reassociated.
# --------------------------------------------------------------------------- #


def run_prefill_loop(model: CausalLMModel, ids: list[int], chunk: int):
    """The pre-optimisation path: ``forward`` every chunk, keep the last."""
    cache = model.new_cache(max_context=max(64, len(ids) + 8))
    result = None
    for start in range(0, len(ids), chunk):
        result = model.forward(ids[start : start + chunk], cache)
    return result, cache


IDS = [11, 3, 7, 19, 2, 25, 8, 14, 1, 5, 30, 12]


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, len(IDS)])
def test_prefill_matches_projecting_every_chunk_exactly(model: CausalLMModel, chunk: int):
    loop, loop_cache = run_prefill_loop(model, IDS, chunk)
    cache = model.new_cache(max_context=max(64, len(IDS) + 8))
    result = model.prefill(IDS, cache, chunk_size=chunk)

    np.testing.assert_array_equal(result.logits, loop.logits)
    np.testing.assert_array_equal(result.hidden_states[-1], loop.hidden_states[-1])
    # The whole cache, not just its length: decode continues from these slots, so
    # a K/V difference here would show up as a divergent continuation.
    for layer in range(model.config.num_hidden_layers):
        np.testing.assert_array_equal(cache.keys[layer], loop_cache.keys[layer])
        np.testing.assert_array_equal(cache.values[layer], loop_cache.values[layer])
    assert cache.length == loop_cache.length == len(IDS)


@pytest.mark.parametrize("chunk,expected", [(1, 12), (3, 4), (5, 3), (7, 2), (12, 1), (99, 1)])
def test_prefill_reports_the_chunks_it_ran(model: CausalLMModel, chunk: int, expected: int):
    cache = model.new_cache(64)
    result = model.prefill(IDS, cache, chunk_size=chunk)
    assert result.chunks == expected
    assert result.num_tokens == len(IDS)
    assert result.position_offset == 0
    assert result.cache_length == len(IDS)


def test_prefill_projects_the_vocabulary_once_not_once_per_chunk(
    model: CausalLMModel, monkeypatch
):
    """The point of the change, counted rather than inferred from a timing."""
    calls: list[int] = []
    original = model.output_projection

    def counted(hidden):
        calls.append(np.asarray(hidden).reshape(-1, model.config.hidden_size).shape[0])
        return original(hidden)

    monkeypatch.setattr(model, "output_projection", counted)
    model.prefill(IDS, model.new_cache(64), chunk_size=3)
    assert calls == [1], f"four chunks projected {len(calls)} times: {calls}"


def test_a_probe_that_wants_logits_keeps_projecting_every_chunk(
    model: CausalLMModel, monkeypatch
):
    """A debug run may be slow; it may not compute something different."""
    calls: list[int] = []
    original = model.output_projection

    def counted(hidden):
        calls.append(1)
        return original(hidden)

    monkeypatch.setattr(model, "output_projection", counted)
    probe = DebugProbe.for_names({"logits"}, positions=(-1,))
    model.prefill(IDS, model.new_cache(64), chunk_size=3, probe=probe)
    assert len(calls) == 4, "one projection per chunk, as before the optimisation"
    # And the value it reports is unchanged: under chunking the last chunk's
    # record overwrites the earlier ones, so skipping them cannot move it.
    np.testing.assert_array_equal(
        probe.records["logits[-1]"],
        model.prefill(IDS, model.new_cache(64), chunk_size=3).logits,
    )


def test_a_probe_that_does_not_want_logits_does_not_force_the_projections(
    model: CausalLMModel, monkeypatch
):
    calls: list[int] = []
    original = model.output_projection

    def counted(hidden):
        calls.append(1)
        return original(hidden)

    monkeypatch.setattr(model, "output_projection", counted)
    probe = DebugProbe.for_names({"layer_output", "final_norm"}, positions=(-1,))
    model.prefill(IDS, model.new_cache(64), chunk_size=3, probe=probe)
    assert calls == [1]
    assert "layer0.layer_output[-1]" in probe.records


def test_prefill_hidden_states_cover_only_the_last_chunk(model: CausalLMModel):
    cache = model.new_cache(64)
    result = model.prefill(IDS, cache, chunk_size=5)
    assert result.chunks == 3
    # 12 tokens in chunks of 5 -> 5 + 5 + 2.
    assert result.hidden_states.shape == (2, model.config.hidden_size)
    assert result.hidden_states_offset == 10
    one_chunk = model.prefill(IDS, model.new_cache(64), chunk_size=len(IDS))
    np.testing.assert_allclose(
        result.hidden_states, one_chunk.hidden_states[-2:], **CHUNK_TOLERANCE
    )


def test_prefill_continues_from_a_cache_that_is_not_empty(model: CausalLMModel):
    """Positions come from the cache, so a resumed prefill must not restart them."""
    prefix, rest = IDS[:4], IDS[4:]

    whole = model.new_cache(64)
    expected = model.prefill([*prefix, *rest], whole, chunk_size=3)

    split = model.new_cache(64)
    model.prefill(prefix, split, chunk_size=3)
    resumed = model.prefill(rest, split, chunk_size=3)
    assert resumed.position_offset == len(prefix)
    assert resumed.cache_length == len(IDS)
    np.testing.assert_allclose(resumed.logits, expected.logits, **CHUNK_TOLERANCE)


def test_prefill_scatters_image_features_across_a_chunk_boundary(model: CausalLMModel):
    """Placeholder tokens straddle chunks, so rows are handed out in prompt order."""
    hidden = model.config.hidden_size
    features = np.arange(5 * hidden, dtype=np.float32).reshape(5, hidden) * 0.01
    mask = np.zeros(len(IDS), dtype=bool)
    mask[[1, 2, 6, 7, 8]] = True  # two in the first chunk of five, three in the second

    cache = model.new_cache(64)
    chunked = model.prefill(
        IDS, cache, chunk_size=5, inputs_embeds=features, inputs_embed_mask=mask
    )
    # The same block in one forward: identical inputs, identical merge.
    single = model.prefill(
        IDS, model.new_cache(64), chunk_size=len(IDS), inputs_embeds=features,
        inputs_embed_mask=mask,
    )
    np.testing.assert_allclose(chunked.logits, single.logits, **CHUNK_TOLERANCE)
    assert chunked.chunks == 3


def test_prefill_rejects_a_mask_that_does_not_cover_the_block(model: CausalLMModel):
    cache = model.new_cache(64)
    with pytest.raises(ValueError, match="covers 3 positions"):
        model.prefill(
            IDS,
            cache,
            chunk_size=4,
            inputs_embeds=np.zeros((1, model.config.hidden_size), dtype=np.float32),
            inputs_embed_mask=np.zeros(3, dtype=bool),
        )
    assert cache.length == 0


def test_prefill_rejects_placeholders_with_no_features_before_touching_the_cache(
    model: CausalLMModel,
):
    cache = model.new_cache(64)
    mask = np.zeros(len(IDS), dtype=bool)
    mask[0] = True
    with pytest.raises(ValueError, match="no inputs_embeds were supplied"):
        model.prefill(IDS, cache, chunk_size=4, inputs_embed_mask=mask)
    assert cache.length == 0, "a rejected prefill must not leave slots behind"


# ``prefill`` used to read ``inputs_embeds`` only inside its
# ``inputs_embed_mask is not None`` branch, and to hand each chunk a *slice* of
# the rows rather than checking the count against the mask. Two consequences:
# features passed without a mask were never looked at, and a row count that
# disagreed with the mask was reconciled by the slice, because ``features[0:1]``
# of a two-row array is one row and raises nothing. Both produced a completed
# prefill whose embeddings were silently missing or truncated — the same input
# given to ``forward`` was refused. The checks now run once over the whole block
# before the first chunk, through the code ``forward`` also uses.

#: Positions entirely inside the *third* chunk of four over ``IDS``. Illegal
#: features marked here are the interesting case: two legal chunks would run
#: first, so a check that fires per chunk rather than up front has already
#: committed K/V by the time it notices.
LATE_PLACEHOLDERS = (9, 10, 11)


def late_mask() -> np.ndarray:
    mask = np.zeros(len(IDS), dtype=bool)
    mask[list(LATE_PLACEHOLDERS)] = True
    return mask


def feature_rows(model: CausalLMModel, count: int, width: int | None = None) -> np.ndarray:
    """``[count, width]`` features with distinct values, so a dropped row shows."""
    width = model.config.hidden_size if width is None else width
    return np.arange(count * width, dtype=np.float32).reshape(count, width) * 0.01


def test_prefill_refuses_features_that_come_without_a_mask(model: CausalLMModel):
    """Silently ignoring them is the failure this exists to prevent.

    The features are well-formed and the prefill would have completed, so nothing
    downstream could tell the caller their embeddings never reached the decoder.
    """
    cache = model.new_cache(64)
    with pytest.raises(ValueError, match="inputs_embeds require an inputs_embed_mask"):
        model.prefill(
            IDS, cache, chunk_size=4, inputs_embeds=feature_rows(model, 2)
        )
    assert cache.length == 0


def test_prefill_refuses_more_feature_rows_than_the_mask_selects(model: CausalLMModel):
    """The review's repro: two rows, a mask selecting one token.

    The old cursor took ``features[0:1]`` and dropped the second row without a
    word, so a truncated image produced a confident answer.
    """
    mask = np.zeros(len(IDS), dtype=bool)
    mask[3] = True
    cache = model.new_cache(64)
    with pytest.raises(ValueError) as excinfo:
        model.prefill(
            IDS, cache, chunk_size=4, inputs_embeds=feature_rows(model, 2),
            inputs_embed_mask=mask,
        )
    message = str(excinfo.value)
    assert "expands to 1 placeholder tokens" in message
    assert "the encoder produced 2 feature rows" in message
    assert cache.length == 0


def test_prefill_refuses_fewer_rows_before_the_earlier_chunks_commit(
    model: CausalLMModel,
):
    """Too few rows used to raise too late to matter.

    The placeholders are all in the third chunk, so chunks one and two are legal
    and committed their K/V before chunk three's ``forward`` noticed. A caller
    catching that error was left holding a cache four slots longer than the
    tokens it had accounted for, and the next prefill would continue from there.
    """
    cache = model.new_cache(64)
    with pytest.raises(ValueError, match="the encoder produced 2 feature rows"):
        model.prefill(
            IDS, cache, chunk_size=4, inputs_embeds=feature_rows(model, 2),
            inputs_embed_mask=late_mask(),
        )
    assert cache.length == 0, "a rejected prefill left committed slots behind"


@pytest.mark.parametrize(
    "shape_id", ["one-dimensional", "wrong-hidden-size", "three-dimensional"]
)
def test_prefill_refuses_features_that_are_not_a_row_matrix(
    model: CausalLMModel, shape_id: str
):
    """Shape is checked against the model, not against whatever the slice yields.

    The placeholders are late, so before the up-front check these reached
    ``_merge_inputs_embeds`` only after two chunks had committed — and a
    one-dimensional array sliced as ``features[0:1]`` arrives there as shape
    ``(1,)``, which is a different complaint than the one the caller made.
    """
    hidden = model.config.hidden_size
    shapes = {
        "one-dimensional": (hidden,),
        "wrong-hidden-size": (len(LATE_PLACEHOLDERS), hidden + 1),
        "three-dimensional": (len(LATE_PLACEHOLDERS), 1, hidden),
    }
    cache = model.new_cache(64)
    with pytest.raises(ValueError) as excinfo:
        model.prefill(
            IDS, cache, chunk_size=4,
            inputs_embeds=np.zeros(shapes[shape_id], dtype=np.float32),
            inputs_embed_mask=late_mask(),
        )
    assert (
        f"inputs_embeds must be [n, {hidden}], got {shapes[shape_id]}"
        in str(excinfo.value)
    )
    assert cache.length == 0


def test_prefill_refuses_rows_nobody_asked_for(model: CausalLMModel):
    """An all-false mask selects zero positions, so one supplied row has nowhere to go."""
    cache = model.new_cache(64)
    with pytest.raises(ValueError, match="expands to 0 placeholder tokens"):
        model.prefill(
            IDS, cache, chunk_size=4, inputs_embeds=feature_rows(model, 1),
            inputs_embed_mask=np.zeros(len(IDS), dtype=bool),
        )
    assert cache.length == 0


@pytest.mark.parametrize(
    "kwargs_id", ["no-mask", "too-many-rows", "wrong-width", "rows-nobody-asked-for"]
)
def test_prefill_and_forward_refuse_the_same_block_identically(
    model: CausalLMModel, kwargs_id: str
):
    """One implementation of the rule, so the two entry points cannot drift.

    The mask-length message is excluded on purpose: it names the caller
    ("this forward" / "this prefill"), which is the one difference that should
    survive. Everything else has to be the same sentence, because it is the same
    check.
    """
    hidden = model.config.hidden_size
    one = np.zeros(len(IDS), dtype=bool)
    one[3] = True
    cases = {
        "no-mask": dict(inputs_embeds=feature_rows(model, 2)),
        "too-many-rows": dict(inputs_embeds=feature_rows(model, 2), inputs_embed_mask=one),
        "wrong-width": dict(
            inputs_embeds=feature_rows(model, 3, hidden + 1), inputs_embed_mask=late_mask()
        ),
        "rows-nobody-asked-for": dict(
            inputs_embeds=feature_rows(model, 1),
            inputs_embed_mask=np.zeros(len(IDS), dtype=bool),
        ),
    }
    kwargs = cases[kwargs_id]

    with pytest.raises(ValueError) as from_prefill:
        model.prefill(IDS, model.new_cache(64), chunk_size=4, **kwargs)
    with pytest.raises(ValueError) as from_forward:
        model.forward(IDS, model.new_cache(64), **kwargs)
    assert str(from_prefill.value) == str(from_forward.value)


def test_a_rejected_prefill_leaves_an_existing_cache_exactly_as_it_was(
    model: CausalLMModel,
):
    """Length *and* contents, over a cache that already holds a legal turn.

    The realistic shape of the bug is a second turn: the conversation so far is
    committed, an illegal image block arrives, and the caller catches the error
    and keeps talking. Whether that is safe depends on the refusal happening
    before any write, which ``cache.length == 0`` cannot express once the cache
    is not empty.
    """
    cache = model.new_cache(64)
    model.prefill(IDS[:4], cache, chunk_size=4)
    assert cache.length == 4
    before = [
        tuple(array.copy() for array in cache.read(layer, cache.length))
        for layer in range(model.config.num_hidden_layers)
    ]

    with pytest.raises(ValueError):
        model.prefill(
            IDS[4:], cache, chunk_size=2, inputs_embeds=feature_rows(model, 2),
            inputs_embed_mask=late_mask()[4:],
        )

    assert cache.length == 4
    for layer, (keys, values) in enumerate(before):
        now_keys, now_values = cache.read(layer, cache.length)
        np.testing.assert_array_equal(now_keys, keys)
        np.testing.assert_array_equal(now_values, values)


@pytest.mark.parametrize("chunk_size", [12, 7, 5, 4, 3, 1])
def test_a_late_image_span_still_matches_the_single_chunk_prefill(
    model: CausalLMModel, chunk_size: int
):
    """The up-front check must not change the legal path it now guards.

    ``test_prefill_scatters_image_features_across_a_chunk_boundary`` puts the
    placeholders in the first two chunks; these are all in the last one, which is
    the alignment the row cursor has to walk two empty chunks to reach.
    """
    features = feature_rows(model, len(LATE_PLACEHOLDERS))
    mask = late_mask()

    whole = model.prefill(
        IDS, model.new_cache(64), chunk_size=len(IDS), inputs_embeds=features,
        inputs_embed_mask=mask,
    )
    chunked = model.prefill(
        IDS, model.new_cache(64), chunk_size=chunk_size, inputs_embeds=features,
        inputs_embed_mask=mask,
    )
    np.testing.assert_allclose(chunked.logits, whole.logits, **CHUNK_TOLERANCE)
    # And the precondition: merging changed something, so the comparison above is
    # not two runs of the same embeddings-free prefill agreeing trivially.
    plain = model.prefill(IDS, model.new_cache(64), chunk_size=len(IDS))
    assert not np.allclose(whole.logits, plain.logits, **CHUNK_TOLERANCE)


def test_prefill_rejects_an_empty_block_and_a_zero_chunk_size(model: CausalLMModel):
    with pytest.raises(ValueError, match="at least one token"):
        model.prefill([], model.new_cache(64), chunk_size=4)
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        model.prefill(IDS, model.new_cache(64), chunk_size=0)


def test_prefill_rejects_an_offset_that_disagrees_with_the_cache(model: CausalLMModel):
    cache = model.new_cache(64)
    model.prefill(IDS[:3], cache, chunk_size=3)
    with pytest.raises(ValueError, match="disagrees with kv_cache.length"):
        model.prefill(IDS[3:], cache, chunk_size=3, position_offset=0)


def test_an_overflowing_prefill_commits_only_the_chunks_that_fit(model: CausalLMModel):
    """A chunked prefill cannot roll back, so the length must say what really ran.

    Claiming ``length == 0`` here would be a lie in the other direction: two
    chunks did complete a full forward and their K/V is in the slots.
    """
    cache = model.new_cache(8)
    with pytest.raises(ContextOverflowError):
        model.prefill(list(range(20)), cache, chunk_size=4)
    assert cache.length == 8  # two chunks committed, the third was refused
    assert cache.length <= cache.max_context

    refused_first = model.new_cache(4)
    with pytest.raises(ContextOverflowError):
        model.prefill(list(range(20)), refused_first, chunk_size=8)
    assert refused_first.length == 0


# --------------------------------------------------------------------------- #
# the decode step builds no causal mask
# --------------------------------------------------------------------------- #


def test_a_multi_token_block_builds_a_mask_and_a_single_token_one_does_not(
    model: CausalLMModel, monkeypatch
):
    """Observed by counting the calls, not inferred from a timing."""
    calls: list[tuple[int, int, int]] = []
    original = model_module.causal_attention_mask

    def counted(num_queries, num_keys, position_offset):
        calls.append((num_queries, num_keys, position_offset))
        return original(num_queries, num_keys, position_offset)

    monkeypatch.setattr(model_module, "causal_attention_mask", counted)

    cache = model.new_cache(32)
    model.forward([4, 9, 2], cache)
    assert calls == [(3, 3, 0)], "a three-token block still has a triangle"

    calls.clear()
    model.forward([7], cache)
    assert calls == [], "a decode step built the all-zero mask anyway"

    calls.clear()
    model.prefill([1, 2, 3, 4, 5], model.new_cache(32), chunk_size=2)
    # Chunks of 2 over five tokens: 2 + 2 + 1. Only the last is mask-free.
    assert calls == [(2, 2, 0), (2, 4, 2)]


def test_a_single_token_forward_is_unchanged_by_not_building_the_mask(
    model: CausalLMModel, monkeypatch
):
    """Put the zero mask back and check nothing moves.

    ``encode`` does not call ``causal_attention_mask`` at all for one token, so
    the comparison has to be made one level down: hand ``group_query_attention``
    the all-zero array it would have received, and require identical logits.
    """
    real = model_module.group_query_attention

    def with_zero_mask(q, k, v, num_kv_heads, scaling, mask=None):
        if mask is None:
            mask = np.zeros((1, 1, q.shape[1], k.shape[1]), dtype=np.float32)
        return real(q, k, v, num_kv_heads, scaling, mask)

    plain = model.new_cache(32)
    model.forward([4, 9, 2], plain)
    plain_logits = model.forward([7], plain).logits

    monkeypatch.setattr(model_module, "group_query_attention", with_zero_mask)
    masked = model.new_cache(32)
    model.forward([4, 9, 2], masked)
    masked_logits = model.forward([7], masked).logits

    np.testing.assert_array_equal(plain_logits, masked_logits)
    for layer in range(model.config.num_hidden_layers):
        np.testing.assert_array_equal(plain.keys[layer], masked.keys[layer])
        np.testing.assert_array_equal(plain.values[layer], masked.values[layer])


# --------------------------------------------------------------------------- #
# positions and capacity
# --------------------------------------------------------------------------- #


def test_a_position_offset_disagreeing_with_the_cache_is_rejected(model: CausalLMModel):
    cache = model.new_cache(32)
    model.forward([1, 2], cache)
    with pytest.raises(ValueError, match="disagrees with kv_cache.length"):
        model.forward([3], cache, position_offset=0)


def test_an_explicit_matching_offset_is_accepted(model: CausalLMModel):
    cache = model.new_cache(32)
    model.forward([1, 2], cache)
    assert model.forward([3], cache, position_offset=2).cache_length == 3


def test_exceeding_max_context_raises_instead_of_truncating(model: CausalLMModel):
    cache = model.new_cache(8)
    model.forward([1, 2, 3, 4, 5], cache)
    with pytest.raises(ContextOverflowError, match="max_context"):
        model.forward([6, 7, 8, 9], cache)


def test_an_overflowing_forward_leaves_the_cache_length_untouched(model: CausalLMModel):
    cache = model.new_cache(8)
    model.forward([1, 2, 3], cache)
    with pytest.raises(ContextOverflowError):
        model.forward(list(range(10)), cache)
    assert cache.length == 3


def test_exceeding_max_position_embeddings_raises(model: CausalLMModel):
    """Guards a caller who builds a KVCache larger than the model's position grid.

    ``new_cache`` already refuses such a capacity, so this constructs the cache
    directly to reach the check inside ``forward``.
    """
    limit = model.config.max_position_embeddings
    cache = KVCache(
        max_context=limit + 8,
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )
    cache.commit(limit - 1)
    with pytest.raises(ContextOverflowError, match="max_position_embeddings"):
        model.forward([1, 2], cache)


def test_new_cache_validates_max_context(model: CausalLMModel):
    with pytest.raises(Exception, match="max_position_embeddings"):
        model.new_cache(model.config.max_position_embeddings + 1)


def test_new_cache_is_sized_from_the_config(model: CausalLMModel):
    cache = model.new_cache(12)
    assert len(cache.keys) == model.config.num_hidden_layers
    assert cache.keys[0].shape == (
        model.config.num_key_value_heads,
        12,
        model.config.head_dim,
    )


# --------------------------------------------------------------------------- #
# debug probe
# --------------------------------------------------------------------------- #


def test_probing_does_not_change_the_numerics(model: CausalLMModel):
    plain = model.forward([1, 2, 3], model.new_cache(16)).logits
    probe = DebugProbe.for_names(DebugProbe.ALL_CAPTURES)
    probed = model.forward([1, 2, 3], model.new_cache(16), probe=probe).logits
    np.testing.assert_array_equal(plain, probed)


def test_probe_captures_the_requested_layers_only(model: CausalLMModel):
    probe = DebugProbe.for_names(
        {"layer_output"}, layer_indices={1}, positions=(0, -1)
    )
    model.forward([1, 2, 3, 4], model.new_cache(16), probe=probe)
    assert set(probe.records) == {"layer1.layer_output[0]", "layer1.layer_output[-1]"}
    assert probe.records["layer1.layer_output[-1]"].shape == (model.config.hidden_size,)


def test_probe_captures_every_requested_point(model: CausalLMModel):
    probe = DebugProbe.for_names(
        DebugProbe.ALL_CAPTURES, layer_indices={0}, positions=(-1,)
    )
    model.forward([1, 2], model.new_cache(16), probe=probe)
    keys = set(probe.records)
    assert "embed[-1]" in keys
    assert "layer0.attention_output[-1]" in keys
    assert "layer0.mlp_output[-1]" in keys
    assert "layer0.layer_output[-1]" in keys
    assert "final_norm[-1]" in keys
    assert "logits[-1]" in keys
    assert probe.records["logits[-1]"].shape == (model.config.vocab_size,)


def test_probe_with_no_layer_filter_captures_every_layer(model: CausalLMModel):
    probe = DebugProbe.for_names({"layer_output"})
    model.forward([1, 2], model.new_cache(16), probe=probe)
    captured = {int(k.split(".")[0].removeprefix("layer")) for k in probe.records}
    assert captured == set(range(model.config.num_hidden_layers))


def test_probe_records_are_detached_copies(model: CausalLMModel):
    probe = DebugProbe.for_names({"final_norm"})
    model.forward([1, 2, 3], model.new_cache(16), probe=probe)
    recorded = probe.records["final_norm[-1]"]
    assert isinstance(recorded, np.ndarray) and not isinstance(recorded, np.memmap)
    recorded[0] = 1e9  # mutating the capture must not corrupt the model
    assert model.forward([1, 2, 3], model.new_cache(16)).hidden_states[-1, 0] != 1e9


def test_probe_rejects_an_unknown_capture_point():
    with pytest.raises(ValueError, match="unknown capture points"):
        DebugProbe.for_names({"attention_scores"})


def test_a_layer_output_is_the_embedding_plus_both_residual_branches(model: CausalLMModel):
    """``out = x + Attention(norm(x))`` then ``out = out + MLP(norm(out))``."""
    probe = DebugProbe.for_names(
        {"embed", "attention_output", "mlp_output", "layer_output"}, layer_indices={0}
    )
    ids = [1, 2, 3]
    model.forward(ids, model.new_cache(16), probe=probe)
    embed = probe.records["embed[-1]"]
    attention = probe.records["layer0.attention_output[-1]"]
    mlp = probe.records["layer0.mlp_output[-1]"]
    np.testing.assert_allclose(
        probe.records["layer0.layer_output[-1]"], embed + attention + mlp, rtol=1e-5, atol=1e-6
    )


def test_the_second_layer_starts_where_the_first_one_ended(model: CausalLMModel):
    probe = DebugProbe.for_names({"layer_output"}, layer_indices={0, 1})
    model.forward([1, 2, 3], model.new_cache(16), probe=probe)
    # Layer 1's own residual chain must build on layer 0's output, so the two
    # cannot be equal (random weights) yet must both be finite and non-trivial.
    layer0 = probe.records["layer0.layer_output[-1]"]
    layer1 = probe.records["layer1.layer_output[-1]"]
    assert not np.allclose(layer0, layer1)
    assert np.all(np.isfinite(layer1)) and np.abs(layer1).max() > 0


# --------------------------------------------------------------------------- #
# probe logits at positions other than the last
# --------------------------------------------------------------------------- #


def assert_logits_match_their_own_hidden_state(model: CausalLMModel, probe: DebugProbe) -> None:
    """Every captured logit row must be the projection of the matching hidden state."""
    for position in probe.positions:
        np.testing.assert_allclose(
            probe.records[f"logits[{position}]"],
            model.output_projection(probe.records[f"final_norm[{position}]"]),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"logits[{position}] is not the projection of final_norm[{position}]",
        )


def test_probe_logits_at_every_requested_position_belong_to_that_position(model: CausalLMModel):
    probe = DebugProbe.for_names({"final_norm", "logits"}, positions=(0, 1, 2, -1))
    forward = model.forward([5, 6, 7, 8], model.new_cache(16), probe=probe)

    assert set(probe.records) == {
        f"{name}[{position}]"
        for name in ("final_norm", "logits")
        for position in (0, 1, 2, -1)
    }
    assert_logits_match_their_own_hidden_state(model, probe)
    # The last position is the one generation uses, and it must be unchanged.
    np.testing.assert_array_equal(probe.records["logits[-1]"], forward.logits)
    # Four distinct hidden states give four distinct distributions.
    rows = [probe.records[f"logits[{position}]"] for position in (0, 1, 2)]
    assert all(not np.allclose(rows[i], rows[j]) for i in range(3) for j in range(i + 1, 3))


def test_probe_logits_positions_count_from_the_start_of_this_forward_block(
    model: CausalLMModel,
):
    """Position 0 of a decode block is that block's first token, not the prompt's."""
    cache = model.new_cache(16)
    model.forward([5, 6, 7], cache)

    probe = DebugProbe.for_names({"final_norm", "logits"}, positions=(0, -1))
    model.forward([8, 9], cache, probe=probe)
    assert_logits_match_their_own_hidden_state(model, probe)

    prompt_probe = DebugProbe.for_names({"final_norm", "logits"}, positions=(0,))
    model.forward([5, 6, 7], model.new_cache(16), probe=prompt_probe)
    assert not np.allclose(
        probe.records["logits[0]"], prompt_probe.records["logits[0]"]
    ), "position 0 was resolved against the whole context instead of this block"


def test_a_single_token_forward_reports_its_only_position(model: CausalLMModel):
    cache = model.new_cache(16)
    model.forward([5, 6, 7], cache)
    probe = DebugProbe.for_names({"final_norm", "logits"}, positions=(0,))
    forward = model.forward([8], cache, probe=probe)
    np.testing.assert_array_equal(probe.records["logits[0]"], forward.logits)


def test_probing_a_position_the_block_does_not_have_is_an_error(model: CausalLMModel):
    for positions in ((4,), (1, 4), (-5,)):
        probe = DebugProbe.for_names({"logits"}, positions=positions)
        with pytest.raises(ValueError, match="outside this forward"):
            model.forward([1, 2, 3, 4], model.new_cache(16), probe=probe)

    probe = DebugProbe.for_names({"layer_output"}, positions=(9,))
    with pytest.raises(ValueError, match="outside this forward"):
        model.forward([1, 2], model.new_cache(16), probe=probe)


def test_positions_nobody_asked_for_are_never_projected(model: CausalLMModel, monkeypatch):
    """A full-vocab projection is the most expensive thing in the model."""
    calls: list[int] = []
    original = model.output_projection

    def counted(hidden):
        # Rows of the hidden state being projected: 1 means "the last position
        # only", anything more means extra full-vocab matmuls.
        calls.append(np.asarray(hidden).reshape(-1, model.config.hidden_size).shape[0])
        return original(hidden)

    monkeypatch.setattr(model, "output_projection", counted)

    model.forward([1, 2, 3, 4], model.new_cache(16))
    assert calls == [1], "an unprobed forward projected more than the last position"

    calls.clear()
    model.forward([1, 2, 3, 4], model.new_cache(16), probe=DebugProbe.for_names({"layer_output"}))
    assert calls == [1]

    calls.clear()
    probe = DebugProbe.for_names({"logits"}, positions=(-1,))
    model.forward([1, 2, 3, 4], model.new_cache(16), probe=probe)
    assert calls == [1], "the last position was projected twice"

    calls.clear()
    probe = DebugProbe.for_names({"logits"}, positions=(0, 1, -1))
    model.forward([1, 2, 3, 4], model.new_cache(16), probe=probe)
    assert calls == [1, 1, 1], "one projection per requested position, no more"


# --------------------------------------------------------------------------- #
# the directory has to agree with itself
#
# ``config.json`` says which rules decode the weights; a version-2 manifest also
# records the family the converter saw. Those are two independent claims about
# the same directory, so a hand-edited manifest or a swapped config used to load
# happily and then compute with the wrong family's rules -- no crash, just
# different numbers.
# --------------------------------------------------------------------------- #


def _tiny_dir(tmp_path: Path, name: str, *, task: str | None) -> Path:
    root = tmp_path / name
    root.mkdir()
    build_tiny_tokenizer(root)
    vocab_size = HFTokenizer.from_file(str(root / "tokenizer.json")).get_vocab_size()
    return write_tiny_model(root, vocab_size=vocab_size, task=task)


def _claiming_family(source: Path, tmp_path: Path, family: str) -> Path:
    """A copy whose manifest claims ``family``; config.json and tensors untouched."""
    root = tmp_path / f"{source.name}-claims-{family}"
    shutil.copytree(source, root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["model_family"] = family
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def test_a_manifest_that_disagrees_with_config_json_is_refused(tmp_path: Path):
    directory = _tiny_dir(tmp_path, "v2-generation", task="text-generation")
    config, _ = load_config_and_weights(directory)
    assert config.family == "qwen3"

    liar = _claiming_family(directory, tmp_path, "llama")
    with pytest.raises(ConfigError, match=r"model_family='llama'.*model_type='qwen3'"):
        load_config_and_weights(liar)
    # The public entry point refuses it too, before any weight is read.
    with pytest.raises(ConfigError, match="inconsistent"):
        Engine.from_directory(liar, max_context=64)


def test_the_embedding_entry_refuses_the_same_inconsistent_directory(tmp_path: Path):
    directory = _tiny_dir(tmp_path, "v2-embedding", task="text-embedding")
    assert EmbeddingEngine.from_directory(directory, max_length=64).dimension > 0

    liar = _claiming_family(directory, tmp_path, "qwen2")
    with pytest.raises(ConfigError, match=r"model_family='qwen2'.*model_type='qwen3'"):
        EmbeddingEngine.from_directory(liar, max_length=64)


def test_a_consistent_v2_directory_still_loads(tmp_path: Path):
    """The control: the check must not fire on what the converter really writes."""
    for task in ("text-generation", "text-embedding"):
        directory = _tiny_dir(tmp_path, f"v2-ok-{task}", task=task)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format_version"] == 2
        config, _ = load_config_and_weights(directory)
        assert config.family == manifest["model_family"] == "qwen3"


def test_a_v1_manifest_is_not_asked_for_a_field_it_never_had(tmp_path: Path):
    """v0.1 directories stay loadable: no ``model_family`` key, no new requirement."""
    directory = _tiny_dir(tmp_path, "v1-legacy", task=None)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert "model_family" not in manifest

    config, weights = load_config_and_weights(directory)
    assert weights.manifest.model_family is None
    assert config.family == "qwen3"
    assert Engine.from_directory(directory, max_context=64).generate(
        prompt="hi", max_new_tokens=1
    ).token_ids
