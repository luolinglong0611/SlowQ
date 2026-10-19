"""Full-model forward: chunked prefill, cached decode, positions, and probes.

The central property under test is the one the spec calls out: a full-sequence
prefill, a chunked prefill, and token-by-token decode with a cache must produce
numerically identical results. Everything else here guards the invariants that
make that true.
"""

from __future__ import annotations

import numpy as np
import pytest

from slowq.cache import KVCache
from slowq.errors import ContextOverflowError
from slowq.model import DebugProbe, Qwen3Model

#: FP32 matmul reassociation between a long prefill and several short ones.
CHUNK_TOLERANCE = {"rtol": 2e-4, "atol": 2e-5}


@pytest.fixture
def model(tiny_model_dir):
    return Qwen3Model.from_directory(tiny_model_dir)


def run_chunks(model: Qwen3Model, ids: list[int], chunk: int):
    """Prefill ``ids`` in chunks of ``chunk`` and return the final result."""
    cache = model.new_cache(max_context=max(64, len(ids) + 8))
    result = None
    for start in range(0, len(ids), chunk):
        result = model.forward(ids[start : start + chunk], cache)
    return result, cache


# --------------------------------------------------------------------------- #
# shapes and bookkeeping
# --------------------------------------------------------------------------- #


def test_forward_projects_only_the_last_position(model: Qwen3Model):
    """No ``[T, vocab]`` logits tensor is ever materialised."""
    cache = model.new_cache(16)
    result = model.forward([1, 2, 3, 4, 5], cache)
    assert result.logits.shape == (model.config.vocab_size,)
    assert result.hidden_states.shape == (5, model.config.hidden_size)
    assert result.num_tokens == 5
    assert result.position_offset == 0


def test_cache_length_advances_once_per_forward_not_once_per_layer(model: Qwen3Model):
    cache = model.new_cache(16)
    model.forward([1, 2, 3], cache)
    # Every layer writes the same three slots; the length must not be multiplied.
    assert cache.length == 3
    assert cache.high_water_mark == 3


def test_forward_returns_the_cache_length_after_committing(model: Qwen3Model):
    cache = model.new_cache(16)
    result = model.forward([7, 8], cache)
    assert result.cache_length == 2


def test_second_forward_starts_at_the_cached_offset(model: Qwen3Model):
    cache = model.new_cache(16)
    model.forward([1, 2, 3], cache)
    second = model.forward([4], cache)
    assert second.position_offset == 3
    assert second.cache_length == 4


def test_forward_is_deterministic(model: Qwen3Model):
    first = model.forward([5, 6, 7], model.new_cache(16))
    second = model.forward([5, 6, 7], model.new_cache(16))
    np.testing.assert_array_equal(first.logits, second.logits)


def test_logits_stay_finite_and_in_float32(model: Qwen3Model):
    result = model.forward([1, 2, 3], model.new_cache(16))
    assert result.logits.dtype == np.float32
    assert np.all(np.isfinite(result.logits))
    assert np.all(np.isfinite(result.hidden_states))


def test_embed_rejects_out_of_vocabulary_ids(model: Qwen3Model):
    with pytest.raises(ValueError, match="outside the vocabulary"):
        model.embed([0, model.config.vocab_size])


def test_embed_rejects_a_negative_id(model: Qwen3Model):
    with pytest.raises(ValueError, match="outside the vocabulary"):
        model.embed([-1])


def test_embed_rejects_an_empty_input(model: Qwen3Model):
    with pytest.raises(ValueError, match="at least one token"):
        model.embed([])


def test_forward_rejects_an_empty_input(model: Qwen3Model):
    with pytest.raises(ValueError, match="at least one token"):
        model.forward([], model.new_cache(16))


# --------------------------------------------------------------------------- #
# the output projection is the tied embedding
# --------------------------------------------------------------------------- #


def test_output_projection_uses_the_tied_embedding_matrix(model: Qwen3Model):
    hidden = np.full(model.config.hidden_size, 0.25, dtype=np.float32)
    table = model.weights.get("model.embed_tokens.weight")
    np.testing.assert_allclose(model.output_projection(hidden), table @ hidden, rtol=1e-5)


def test_logits_are_the_final_normed_hidden_times_the_embedding(model: Qwen3Model):
    cache = model.new_cache(16)
    result = model.forward([3, 1, 4], cache)
    expected = model.output_projection(result.hidden_states[-1])
    np.testing.assert_allclose(result.logits, expected, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# full vs chunked vs cached decode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7])
def test_chunked_prefill_matches_a_single_prefill(model: Qwen3Model, chunk: int):
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
def test_chunk_boundaries_do_not_shift_positions(model: Qwen3Model, chunk: int):
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


def test_cached_token_by_token_decode_matches_one_long_prefill(model: Qwen3Model):
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


def test_prefill_then_decode_matches_a_longer_prefill(model: Qwen3Model):
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


def test_a_cache_reset_makes_the_next_forward_reproducible(model: Qwen3Model):
    cache = model.new_cache(32)
    first = model.forward([1, 2, 3], cache).logits
    cache.reset()
    second = model.forward([1, 2, 3], cache).logits
    np.testing.assert_array_equal(first, second)


def test_two_caches_for_the_same_input_agree(model: Qwen3Model):
    ids = [2, 4, 6, 8, 10]
    a = model.forward(ids, model.new_cache(32)).logits
    b = model.forward(ids, model.new_cache(32)).logits
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# positions and capacity
# --------------------------------------------------------------------------- #


def test_a_position_offset_disagreeing_with_the_cache_is_rejected(model: Qwen3Model):
    cache = model.new_cache(32)
    model.forward([1, 2], cache)
    with pytest.raises(ValueError, match="disagrees with kv_cache.length"):
        model.forward([3], cache, position_offset=0)


def test_an_explicit_matching_offset_is_accepted(model: Qwen3Model):
    cache = model.new_cache(32)
    model.forward([1, 2], cache)
    assert model.forward([3], cache, position_offset=2).cache_length == 3


def test_exceeding_max_context_raises_instead_of_truncating(model: Qwen3Model):
    cache = model.new_cache(8)
    model.forward([1, 2, 3, 4, 5], cache)
    with pytest.raises(ContextOverflowError, match="max_context"):
        model.forward([6, 7, 8, 9], cache)


def test_an_overflowing_forward_leaves_the_cache_length_untouched(model: Qwen3Model):
    cache = model.new_cache(8)
    model.forward([1, 2, 3], cache)
    with pytest.raises(ContextOverflowError):
        model.forward(list(range(10)), cache)
    assert cache.length == 3


def test_exceeding_max_position_embeddings_raises(model: Qwen3Model):
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


def test_new_cache_validates_max_context(model: Qwen3Model):
    with pytest.raises(Exception, match="max_position_embeddings"):
        model.new_cache(model.config.max_position_embeddings + 1)


def test_new_cache_is_sized_from_the_config(model: Qwen3Model):
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


def test_probing_does_not_change_the_numerics(model: Qwen3Model):
    plain = model.forward([1, 2, 3], model.new_cache(16)).logits
    probe = DebugProbe.for_names(DebugProbe.ALL_CAPTURES)
    probed = model.forward([1, 2, 3], model.new_cache(16), probe=probe).logits
    np.testing.assert_array_equal(plain, probed)


def test_probe_captures_the_requested_layers_only(model: Qwen3Model):
    probe = DebugProbe.for_names(
        {"layer_output"}, layer_indices={1}, positions=(0, -1)
    )
    model.forward([1, 2, 3, 4], model.new_cache(16), probe=probe)
    assert set(probe.records) == {"layer1.layer_output[0]", "layer1.layer_output[-1]"}
    assert probe.records["layer1.layer_output[-1]"].shape == (model.config.hidden_size,)


def test_probe_captures_every_requested_point(model: Qwen3Model):
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


def test_probe_with_no_layer_filter_captures_every_layer(model: Qwen3Model):
    probe = DebugProbe.for_names({"layer_output"})
    model.forward([1, 2], model.new_cache(16), probe=probe)
    captured = {int(k.split(".")[0].removeprefix("layer")) for k in probe.records}
    assert captured == set(range(model.config.num_hidden_layers))


def test_probe_records_are_detached_copies(model: Qwen3Model):
    probe = DebugProbe.for_names({"final_norm"})
    model.forward([1, 2, 3], model.new_cache(16), probe=probe)
    recorded = probe.records["final_norm[-1]"]
    assert isinstance(recorded, np.ndarray) and not isinstance(recorded, np.memmap)
    recorded[0] = 1e9  # mutating the capture must not corrupt the model
    assert model.forward([1, 2, 3], model.new_cache(16)).hidden_states[-1, 0] != 1e9


def test_probe_rejects_an_unknown_capture_point():
    with pytest.raises(ValueError, match="unknown capture points"):
        DebugProbe.for_names({"attention_scores"})


def test_a_layer_output_is_the_embedding_plus_both_residual_branches(model: Qwen3Model):
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


def test_the_second_layer_starts_where_the_first_one_ended(model: Qwen3Model):
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


def assert_logits_match_their_own_hidden_state(model: Qwen3Model, probe: DebugProbe) -> None:
    """Every captured logit row must be the projection of the matching hidden state."""
    for position in probe.positions:
        np.testing.assert_allclose(
            probe.records[f"logits[{position}]"],
            model.output_projection(probe.records[f"final_norm[{position}]"]),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"logits[{position}] is not the projection of final_norm[{position}]",
        )


def test_probe_logits_at_every_requested_position_belong_to_that_position(model: Qwen3Model):
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
    model: Qwen3Model,
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


def test_a_single_token_forward_reports_its_only_position(model: Qwen3Model):
    cache = model.new_cache(16)
    model.forward([5, 6, 7], cache)
    probe = DebugProbe.for_names({"final_norm", "logits"}, positions=(0,))
    forward = model.forward([8], cache, probe=probe)
    np.testing.assert_array_equal(probe.records["logits[0]"], forward.logits)


def test_probing_a_position_the_block_does_not_have_is_an_error(model: Qwen3Model):
    for positions in ((4,), (1, 4), (-5,)):
        probe = DebugProbe.for_names({"logits"}, positions=positions)
        with pytest.raises(ValueError, match="outside this forward"):
            model.forward([1, 2, 3, 4], model.new_cache(16), probe=probe)

    probe = DebugProbe.for_names({"layer_output"}, positions=(9,))
    with pytest.raises(ValueError, match="outside this forward"):
        model.forward([1, 2], model.new_cache(16), probe=probe)


def test_positions_nobody_asked_for_are_never_projected(model: Qwen3Model, monkeypatch):
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
