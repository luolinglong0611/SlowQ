"""KV cache: capacity, position bookkeeping, reset, and request isolation."""

from __future__ import annotations

import numpy as np
import pytest

from slowq.cache import KVCache
from slowq.errors import ContextOverflowError

LAYERS, KV_HEADS, HEAD_DIM = 3, 2, 8


@pytest.fixture
def cache() -> KVCache:
    return KVCache(max_context=8, num_layers=LAYERS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM)


def slot_values(layer: int, token: int) -> np.ndarray:
    """A K/V vector that encodes its own (layer, token) so leaks are obvious."""
    return np.full((KV_HEADS, HEAD_DIM), layer * 100 + token, dtype=np.float32)


def fill(cache: KVCache, layer: int, start: int, count: int) -> None:
    keys = np.stack([slot_values(layer, start + i) for i in range(count)], axis=1)
    cache.write(layer, keys, keys * -1.0, position_offset=start)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_slots_are_preallocated_for_the_whole_context(cache: KVCache):
    assert len(cache.keys) == LAYERS and len(cache.values) == LAYERS
    for array in (*cache.keys, *cache.values):
        assert array.shape == (KV_HEADS, 8, HEAD_DIM)
        assert array.dtype == np.float32


def test_cache_stores_the_original_kv_head_count_not_the_gqa_expansion():
    """Qwen3-0.6B: 8 KV heads, not 16 repeated ones — the cache is 2x smaller."""
    cache = KVCache(max_context=4, num_layers=1, num_kv_heads=8, head_dim=128)
    assert cache.keys[0].shape[0] == 8


def test_allocated_bytes_counts_keys_and_values_for_every_layer(cache: KVCache):
    expected = 2 * LAYERS * KV_HEADS * 8 * HEAD_DIM * 4
    assert cache.allocated_bytes() == expected


@pytest.mark.parametrize("max_context", [0, -1])
def test_rejects_a_useless_max_context(max_context: int):
    with pytest.raises(ValueError, match="max_context"):
        KVCache(max_context, num_layers=1, num_kv_heads=1, head_dim=4)


def test_rejects_zero_layers():
    with pytest.raises(ValueError, match="num_layers"):
        KVCache(max_context=4, num_layers=0, num_kv_heads=1, head_dim=4)


# --------------------------------------------------------------------------- #
# write / read / commit
# --------------------------------------------------------------------------- #


def test_length_starts_at_zero_and_advances_only_on_commit(cache: KVCache):
    assert cache.length == 0
    fill(cache, layer=0, start=0, count=3)
    # Writing is not the same as completing a forward.
    assert cache.length == 0
    cache.commit(3)
    assert cache.length == 3


def test_commit_is_once_per_forward_not_once_per_layer(cache: KVCache):
    """The invariant that makes chunked prefill equal to one long prefill."""
    for layer in range(LAYERS):
        fill(cache, layer, start=0, count=4)
    assert cache.length == 0
    cache.commit(4)
    assert cache.length == 4
    assert cache.high_water_mark == 4


def test_write_places_values_at_absolute_positions(cache: KVCache):
    fill(cache, layer=1, start=2, count=3)
    keys, values = cache.read(1, total_length=5)
    assert keys.shape == (KV_HEADS, 5, HEAD_DIM)
    np.testing.assert_allclose(keys[:, 2, 0], np.full(KV_HEADS, 102.0, np.float32))
    np.testing.assert_allclose(keys[:, 4, 0], np.full(KV_HEADS, 104.0, np.float32))
    np.testing.assert_allclose(values[:, 3, 0], np.full(KV_HEADS, -103.0, np.float32))


def test_read_returns_views_not_copies(cache: KVCache):
    """Decode must not copy the cache; ``read`` slices the preallocated buffer."""
    keys, _ = cache.read(0, total_length=4)
    assert np.shares_memory(keys, cache.keys[0])


def test_reads_never_touch_slots_beyond_the_requested_length(cache: KVCache):
    cache.keys[0][:] = np.nan  # poison everything
    fill(cache, layer=0, start=0, count=2)
    keys, _ = cache.read(0, total_length=2)
    assert np.all(np.isfinite(keys))


def test_reset_clears_both_length_and_high_water_mark(cache: KVCache):
    cache.commit(5)
    cache.reset()
    assert cache.length == 0
    # The mark is per-request too: a new generation is measured on its own.
    assert cache.high_water_mark == 0


# --------------------------------------------------------------------------- #
# capacity
# --------------------------------------------------------------------------- #


def test_check_capacity_allows_an_exactly_full_cache(cache: KVCache):
    cache.check_capacity(8)


def test_check_capacity_raises_instead_of_truncating(cache: KVCache):
    with pytest.raises(ContextOverflowError, match="max_context is 8"):
        cache.check_capacity(9)


def test_write_past_the_end_raises(cache: KVCache):
    with pytest.raises(ContextOverflowError):
        fill(cache, layer=0, start=7, count=2)


def test_write_that_overflows_leaves_the_cache_untouched(cache: KVCache):
    fill(cache, layer=0, start=0, count=4)
    before = cache.keys[0].copy()
    with pytest.raises(ContextOverflowError):
        fill(cache, layer=0, start=4, count=8)
    np.testing.assert_array_equal(cache.keys[0], before)


def test_read_beyond_capacity_raises(cache: KVCache):
    with pytest.raises(ContextOverflowError):
        cache.read(0, total_length=9)


def test_remaining_shrinks_as_the_cache_fills(cache: KVCache):
    assert cache.remaining == 8
    cache.commit(3)
    assert cache.remaining == 5


@pytest.mark.parametrize("offset", [-1, -100])
def test_negative_position_offset_is_rejected(cache: KVCache, offset: int):
    keys = np.zeros((KV_HEADS, 1, HEAD_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="position_offset"):
        cache.write(0, keys, keys, position_offset=offset)


def test_negative_commit_is_rejected(cache: KVCache):
    with pytest.raises(ValueError, match="num_tokens"):
        cache.commit(-1)


# --------------------------------------------------------------------------- #
# shape validation
# --------------------------------------------------------------------------- #


def test_key_and_value_shape_mismatch_is_rejected(cache: KVCache):
    keys = np.zeros((KV_HEADS, 2, HEAD_DIM), dtype=np.float32)
    values = np.zeros((KV_HEADS, 3, HEAD_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        cache.write(0, keys, values, position_offset=0)


def test_wrong_head_count_is_rejected(cache: KVCache):
    """A GQA-expanded K (``num_attention_heads`` instead of ``num_key_value_heads``)."""
    wrong = np.zeros((4, 2, HEAD_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="expected K/V shape"):
        cache.write(0, wrong, wrong, position_offset=0)


def test_wrong_head_dim_is_rejected(cache: KVCache):
    wrong = np.zeros((KV_HEADS, 2, HEAD_DIM + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="expected K/V shape"):
        cache.write(0, wrong, wrong, position_offset=0)


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def test_reset_keeps_the_allocation_but_clears_the_length(cache: KVCache):
    fill(cache, layer=0, start=0, count=4)
    cache.commit(4)
    allocation = cache.keys[0]
    cache.reset()
    assert cache.length == 0
    assert cache.keys[0] is allocation  # no reallocation
    assert cache.remaining == 8


def test_an_aborted_request_cannot_leak_into_the_next_one(cache: KVCache):
    """Request A fills the cache then dies; request B must not see A's tokens."""
    fill(cache, layer=0, start=0, count=6)
    cache.commit(6)

    cache.reset()
    fill(cache, layer=0, start=0, count=2)
    cache.commit(2)

    keys, _ = cache.read(0, total_length=cache.length)
    # Only B's two tokens are readable; A's stale slots sit beyond ``length``.
    np.testing.assert_allclose(keys[:, 0, 0], np.full(KV_HEADS, 0.0, np.float32))
    np.testing.assert_allclose(keys[:, 1, 0], np.full(KV_HEADS, 1.0, np.float32))
    assert keys.shape[1] == 2


def test_reset_then_refill_overwrites_the_stale_slots(cache: KVCache):
    fill(cache, layer=0, start=0, count=4)
    cache.commit(4)
    cache.reset()
    new = np.full((KV_HEADS, 2, HEAD_DIM), -7.0, dtype=np.float32)
    cache.write(0, new, new, position_offset=0)
    cache.commit(2)
    keys, _ = cache.read(0, total_length=2)
    np.testing.assert_allclose(keys, np.full((KV_HEADS, 2, HEAD_DIM), -7.0, np.float32))


def test_repr_reports_the_important_numbers(cache: KVCache):
    cache.commit(3)
    text = repr(cache)
    assert "length=3" in text and "max_context=8" in text and "MiB" in text
