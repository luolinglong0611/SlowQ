"""Operator-level tests: RMSNorm, RoPE, softmax/SiLU, causal mask, GQA."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from slowq.ops import (
    F32,
    NEG_INF_F32,
    apply_rope,
    causal_attention_mask,
    group_query_attention,
    linear,
    merge_heads,
    rms_norm,
    rotate_half,
    rope_cos_sin,
    rope_inv_freq,
    sigmoid,
    silu,
    softmax,
    split_heads,
)


def naive_repeat_kv(x: np.ndarray, group: int) -> np.ndarray:
    """``[n_kv, T, d] -> [n_kv * group, T, d]`` in HF's ``repeat_kv`` order."""
    n_kv, num_tokens, head_dim = x.shape
    return (
        x[:, None, :, :]
        .reshape(n_kv, 1, num_tokens, head_dim)
        .repeat(group, axis=1)
        .reshape(n_kv * group, num_tokens, head_dim)
    )


def naive_attention(q, k, v, scaling, mask=None):
    """Per-head reference loop: slow, obvious, and independent of ops.py."""
    num_heads, num_queries, head_dim = q.shape
    total_keys = k.shape[1]
    out = np.zeros_like(q)
    scores_out = np.zeros((num_heads, num_queries, total_keys), dtype=np.float32)
    for head in range(num_heads):
        scores = (q[head] @ k[head].T) * np.float32(scaling)
        if mask is not None:
            scores = scores + mask[0, 0]
        probs = softmax(scores, axis=-1)
        out[head] = probs @ v[head]
        scores_out[head] = scores
    return out, scores_out


# --------------------------------------------------------------------------- #
# dtype discipline
# --------------------------------------------------------------------------- #


def test_every_operator_stays_in_float32():
    """A stray Python float must not promote activations to FP64 (NEP 50)."""
    x = np.ones((2, 4), dtype=np.float32)
    weight = np.full((4,), 0.5, dtype=np.float32)
    assert rms_norm(x, weight, 1e-6).dtype == np.float32
    assert silu(x).dtype == np.float32
    assert softmax(x).dtype == np.float32
    assert linear(x, np.ones((3, 4), dtype=np.float32)).dtype == np.float32
    cos, sin = rope_cos_sin(np.arange(2), rope_inv_freq(4, 10000.0))
    assert apply_rope(x, cos, sin).dtype == np.float32


def test_linear_matches_matmul_with_transposed_weight():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 7)).astype(F32)
    w = rng.normal(size=(3, 7)).astype(F32)
    np.testing.assert_allclose(linear(x, w), x @ w.T, rtol=1e-5)


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #


def test_rms_norm_matches_the_definition():
    x = np.array([[3.0, 4.0]], dtype=F32)
    weight = np.array([1.0, 2.0], dtype=F32)
    # rms = sqrt(mean([9, 16])) = sqrt(12.5)
    expected = np.array([[3.0, 8.0]], dtype=F32) / np.sqrt(np.float32(12.5) + np.float32(1e-6))
    np.testing.assert_allclose(rms_norm(x, weight, 1e-6), expected, rtol=1e-5)


def test_rms_norm_normalises_over_the_last_axis_only():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 4, 5)).astype(F32) * 10
    out = rms_norm(x, np.ones(5, dtype=F32), 1e-6)
    rms = np.sqrt(np.mean(out * out, axis=-1))
    np.testing.assert_allclose(rms, np.ones((3, 4), dtype=F32), rtol=1e-4)


def test_rms_norm_applies_the_weight():
    x = np.array([[1.0, 1.0]], dtype=F32)
    assert rms_norm(x, np.array([1.0, 1.0], dtype=F32), 1e-6)[0, 0] == pytest.approx(1.0, abs=1e-5)
    scaled = rms_norm(x, np.array([3.0, 1.0], dtype=F32), 1e-6)
    assert scaled[0, 0] == pytest.approx(3.0, abs=1e-5)


def test_rms_norm_of_zero_vector_is_zero_not_nan():
    x = np.zeros((1, 8), dtype=F32)
    out = rms_norm(x, np.ones(8, dtype=F32), 1e-6)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, np.zeros((1, 8), dtype=F32))


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #


def test_rope_inv_freq_matches_the_reference_formula():
    head_dim, theta = 8, 10000.0
    expected = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    got = rope_inv_freq(head_dim, theta)
    assert got.dtype == np.float32
    assert got.shape == (head_dim // 2,)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_rope_inv_freq_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="even"):
        rope_inv_freq(7, 10000.0)


def test_rope_cos_sin_duplicates_both_halves():
    inv_freq = rope_inv_freq(8, 1000000.0)
    cos, sin = rope_cos_sin(np.arange(4), inv_freq)
    assert cos.shape == (4, 8) and sin.shape == (4, 8)
    half = 4
    np.testing.assert_array_equal(cos[:, :half], cos[:, half:])
    np.testing.assert_array_equal(sin[:, :half], sin[:, half:])


def test_rope_position_zero_is_the_identity():
    inv_freq = rope_inv_freq(8, 1000000.0)
    cos, sin = rope_cos_sin(np.array([0]), inv_freq)
    np.testing.assert_allclose(cos, np.ones((1, 8), dtype=F32), atol=1e-6)
    np.testing.assert_allclose(sin, np.zeros((1, 8), dtype=F32), atol=1e-6)


def test_rotate_half_uses_the_official_half_split_pairing():
    """Qwen3/HF pairs index ``i`` with ``i + head_dim / 2``, not interleaved."""
    x = np.arange(8, dtype=F32).reshape(1, 8)
    np.testing.assert_array_equal(rotate_half(x)[0], [-4, -5, -6, -7, 0, 1, 2, 3])


def test_apply_rope_is_a_rotation_so_it_preserves_norm():
    inv_freq = rope_inv_freq(8, 1000000.0)
    cos, sin = rope_cos_sin(np.array([3]), inv_freq)
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=F32)
    rotated = apply_rope(x, cos, sin)
    # Each (i, i + half) pair is rotated by the same angle -> length preserved.
    for i in range(4):
        before = x[0, i] ** 2 + x[0, i + 4] ** 2
        after = rotated[0, i] ** 2 + rotated[0, i + 4] ** 2
        assert after == pytest.approx(before, rel=1e-5)


def test_apply_rope_matches_a_manual_two_d_rotation():
    inv_freq = rope_inv_freq(4, 100.0)
    position = 5
    cos, sin = rope_cos_sin(np.array([position]), inv_freq)
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=F32)
    got = apply_rope(x, cos, sin)[0]
    for i in range(2):
        angle = position * inv_freq[i]
        expected_i = x[0, i] * np.cos(angle) - x[0, i + 2] * np.sin(angle)
        expected_j = x[0, i + 2] * np.cos(angle) + x[0, i] * np.sin(angle)
        assert got[i] == pytest.approx(expected_i, rel=1e-5)
        assert got[i + 2] == pytest.approx(expected_j, rel=1e-5)


def test_apply_rope_broadcasts_over_heads():
    inv_freq = rope_inv_freq(8, 1000000.0)
    cos, sin = rope_cos_sin(np.arange(3), inv_freq)
    x = np.random.default_rng(2).normal(size=(4, 3, 8)).astype(F32)
    assert apply_rope(x, cos, sin).shape == (4, 3, 8)


# --------------------------------------------------------------------------- #
# softmax / SiLU
# --------------------------------------------------------------------------- #


def test_softmax_sums_to_one_and_matches_a_reference():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 11)).astype(F32) * 3
    out = softmax(x)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(6, dtype=F32), rtol=1e-5)
    shifted = x - x.max(axis=-1, keepdims=True)
    reference = np.exp(shifted, dtype=np.float64)
    reference /= reference.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(out, reference, rtol=1e-5, atol=1e-7)


def test_softmax_is_stable_for_large_magnitudes():
    """Unshifted exp would overflow to inf here and produce nan."""
    x = np.array([[1000.0, 1001.0, 999.0]], dtype=F32)
    out = softmax(x)
    assert np.all(np.isfinite(out))
    assert out.sum() == pytest.approx(1.0, abs=1e-5)
    assert int(np.argmax(out)) == 1


def test_softmax_of_the_hf_masking_constant_keeps_finite_rows():
    row = np.array([[0.0, NEG_INF_F32, NEG_INF_F32]], dtype=F32)
    out = softmax(row)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [[1.0, 0.0, 0.0]], atol=1e-6)


def test_sigmoid_does_not_overflow_or_warn_for_very_negative_input():
    x = np.array([-1e4, -200.0, -88.0, 0.0, 88.0, 200.0, 1e4], dtype=F32)
    # ``1 / (1 + exp(-x))`` would overflow for x < -88; any warning is a failure.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = sigmoid(x)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[-1] == 1.0
    assert out[3] == pytest.approx(0.5, abs=1e-6)


def test_silu_equals_x_times_sigmoid_x():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(size=100), [-500.0, -100.0, 100.0, 500.0]]).astype(F32)
    expected = x * (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60), dtype=np.float64)))
    np.testing.assert_allclose(silu(x), expected, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# causal mask
# --------------------------------------------------------------------------- #


def test_causal_mask_at_offset_zero_is_the_lower_triangle():
    mask = causal_attention_mask(4, 4, position_offset=0)
    assert mask.shape == (1, 1, 4, 4)
    assert mask.dtype == np.float32
    allowed = mask[0, 0] == 0.0
    np.testing.assert_array_equal(allowed, np.tril(np.ones((4, 4), dtype=bool)))


def test_causal_mask_uses_absolute_positions_for_a_chunk():
    """A chunk starting at position 2 must see the 2 cached keys entirely."""
    mask = causal_attention_mask(3, 5, position_offset=2)[0, 0]
    allowed = mask == 0.0
    expected = np.array(
        [
            [True, True, True, False, False],   # query at absolute position 2
            [True, True, True, True, False],    # position 3
            [True, True, True, True, True],     # position 4
        ]
    )
    np.testing.assert_array_equal(allowed, expected)


def test_causal_mask_never_uses_inf():
    """HF masks with ``finfo(float32).min``; -inf would make softmax nan."""
    mask = causal_attention_mask(2, 4, position_offset=0)
    assert np.all(np.isfinite(mask))
    assert mask.min() == NEG_INF_F32


# --------------------------------------------------------------------------- #
# split / merge heads
# --------------------------------------------------------------------------- #


def test_split_then_merge_heads_round_trips():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(7, 4 * 8)).astype(F32)
    heads = split_heads(x, num_heads=4, head_dim=8)
    assert heads.shape == (4, 7, 8)
    np.testing.assert_array_equal(merge_heads(heads, 4, 7, 8), x)


def test_split_heads_keeps_contiguous_head_blocks():
    """Head h owns columns ``[h * head_dim, (h + 1) * head_dim)``."""
    x = np.arange(2 * 3 * 4, dtype=F32).reshape(2, 12)
    heads = split_heads(x, num_heads=3, head_dim=4)
    np.testing.assert_array_equal(heads[1], x[:, 4:8])


# --------------------------------------------------------------------------- #
# grouped-query attention
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "num_heads,num_kv_heads", [(16, 8), (4, 2), (4, 4), (8, 1), (4, 1)]
)
def test_group_query_attention_matches_explicit_repeat_kv(num_heads, num_kv_heads):
    """The batch-folding trick must be numerically identical to repeat_kv."""
    rng = np.random.default_rng(6)
    head_dim, num_queries, total_keys = 8, 5, 9
    q = rng.normal(size=(num_heads, num_queries, head_dim)).astype(F32)
    k = rng.normal(size=(num_kv_heads, total_keys, head_dim)).astype(F32)
    v = rng.normal(size=(num_kv_heads, total_keys, head_dim)).astype(F32)
    mask = causal_attention_mask(num_queries, total_keys, position_offset=total_keys - num_queries)
    scaling = head_dim**-0.5

    context, scores = group_query_attention(q, k, v, num_kv_heads, scaling, mask)

    group = num_heads // num_kv_heads
    expected, expected_scores = naive_attention(
        q, naive_repeat_kv(k, group), naive_repeat_kv(v, group), scaling, mask
    )
    assert context.shape == (num_heads, num_queries, head_dim)
    assert scores.shape == (num_heads, num_queries, total_keys)
    np.testing.assert_allclose(context, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-5, atol=1e-6)


def test_group_query_attention_maps_head_h_to_kv_head_h_div_group():
    """Query head ``kv * group + r`` must read KV head ``kv`` (HF's ordering)."""
    head_dim, num_queries = 4, 3
    rng = np.random.default_rng(7)
    q = rng.normal(size=(4, num_queries, head_dim)).astype(F32)
    # Give each KV head a distinguishable, large-magnitude V so the winner is clear.
    v = np.zeros((2, num_queries, head_dim), dtype=F32)
    v[0, :, 0] = 1.0
    v[1, :, 0] = 100.0
    k = np.zeros((2, num_queries, head_dim), dtype=F32)

    context, _ = group_query_attention(q, k, v, num_kv_heads=2, scaling=1.0, mask=None)
    # group size 2: heads 0,1 -> kv 0 (value 1); heads 2,3 -> kv 1 (value 100).
    np.testing.assert_allclose(context[0, :, 0], np.full(num_queries, 1.0, F32), rtol=1e-4)
    np.testing.assert_allclose(context[1, :, 0], np.full(num_queries, 1.0, F32), rtol=1e-4)
    np.testing.assert_allclose(context[2, :, 0], np.full(num_queries, 100.0, F32), rtol=1e-4)
    np.testing.assert_allclose(context[3, :, 0], np.full(num_queries, 100.0, F32), rtol=1e-4)


def test_group_query_attention_rejects_indivisible_head_counts():
    q = np.zeros((5, 2, 4), dtype=F32)
    k = np.zeros((2, 2, 4), dtype=F32)
    with pytest.raises(ValueError, match="divisible"):
        group_query_attention(q, k, k, num_kv_heads=2, scaling=1.0)


def test_group_query_attention_without_a_mask_attends_to_every_key():
    head_dim = 4
    q = np.zeros((1, 1, head_dim), dtype=F32)
    k = np.zeros((1, 3, head_dim), dtype=F32)
    v = np.arange(3 * head_dim, dtype=F32).reshape(1, 3, head_dim)
    context, scores = group_query_attention(q, k, v, 1, 1.0, mask=None)
    # ``scores`` are pre-softmax; a zero query gives a uniform distribution.
    np.testing.assert_allclose(scores[0, 0], np.zeros(3, dtype=F32), atol=1e-6)
    np.testing.assert_allclose(context[0, 0], v[0].mean(axis=0), rtol=1e-5)


def test_group_query_attention_is_causal_when_masked():
    """With a causal mask, query 0 must not see values from later keys."""
    head_dim = 4
    num_tokens = 3
    q = np.zeros((1, num_tokens, head_dim), dtype=F32)
    k = np.zeros((1, num_tokens, head_dim), dtype=F32)
    v = np.zeros((1, num_tokens, head_dim), dtype=F32)
    v[0, :, 0] = np.arange(num_tokens, dtype=F32)  # key j carries value j
    mask = causal_attention_mask(num_tokens, num_tokens, position_offset=0)
    context, _ = group_query_attention(q, k, v, 1, 1.0, mask)
    # Uniform scores over the allowed prefix -> mean of 0..i
    np.testing.assert_allclose(
        context[0, :, 0], np.array([0.0, 0.5, 1.0], dtype=F32), atol=1e-6
    )
