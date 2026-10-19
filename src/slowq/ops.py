"""Numerical operators, written to mirror ``transformers`` Qwen3 semantics.

Every function takes and returns FP32. Two rules keep us out of FP64 by
accident: inputs are coerced with ``np.float32`` scalars rather than Python
floats, and no operator ever calls ``.astype(np.float64)``.

Reference (transformers 4.51.x, ``models/qwen3/modeling_qwen3.py``):

* ``Qwen3RMSNorm``   -- variance in fp32, ``weight * (x * rsqrt(var + eps))``
* ``Qwen3RotaryEmbedding`` -- ``inv_freq = 1 / theta ** (arange(0, d, 2) / d)``,
  ``emb = cat(freqs, freqs)``, ``cos/sin`` cast back to the activation dtype
* ``apply_rotary_pos_emb`` -- ``x * cos + rotate_half(x) * sin`` with
  ``rotate_half`` splitting the head into two halves (not interleaved pairs)
* attention scaling -- multiply by ``head_dim ** -0.5``
"""

from __future__ import annotations

import numpy as np

F32 = np.float32
#: Same value HF uses for masking instead of -inf: ``torch.finfo(float32).min``.
NEG_INF_F32 = np.finfo(np.float32).min


def as_f32(array: np.ndarray) -> np.ndarray:
    """View/cast to FP32 without copying when it already is."""
    return array if array.dtype == np.float32 else array.astype(np.float32, copy=False)


def linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``x @ W.T`` keeping HF's ``[out_features, in_features]`` weight layout.

    The transpose is a view, so a converted checkpoint is never re-transposed on
    disk nor copied at load time.
    """
    return np.matmul(as_f32(x), as_f32(weight).T)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm over the last axis, accumulated in FP32."""
    x = as_f32(x)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = np.float32(1.0) / np.sqrt(variance + F32(eps))
    return as_f32(weight) * (x * inv_rms)


def rope_inv_freq(head_dim: int, rope_theta: float) -> np.ndarray:
    """``[head_dim / 2]`` FP32 inverse frequencies."""
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    exponents = np.arange(0, head_dim, 2, dtype=np.float32) / F32(head_dim)
    inv_freq = np.float32(1.0) / (F32(rope_theta) ** exponents)
    return inv_freq.astype(np.float32, copy=False)


def rope_cos_sin(
    positions: np.ndarray, inv_freq: np.ndarray, attention_scaling: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Cos/sin tables of shape ``[T, head_dim]`` for the given absolute positions.

    Both halves are identical because ``rotate_half`` pairs index ``i`` with
    ``i + head_dim / 2``.
    """
    positions = as_f32(np.asarray(positions)).reshape(-1)
    freqs = np.outer(positions, as_f32(inv_freq))          # [T, head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)          # [T, head_dim]
    scale = F32(attention_scaling)
    return (np.cos(emb) * scale).astype(F32, copy=False), (np.sin(emb) * scale).astype(
        F32, copy=False
    )


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """``x * cos + rotate_half(x) * sin``; ``cos``/``sin`` broadcast over heads."""
    x = as_f32(x)
    return x * as_f32(cos) + rotate_half(x) * as_f32(sin)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-free logistic sigmoid.

    ``1 / (1 + exp(-x))`` overflows to ``inf`` for x < -88 and NumPy then warns
    on every MLP call, so the two branches are evaluated separately. This is the
    same ``x * sigmoid(x)`` HF computes, just without the spurious overflow.
    """
    x = as_f32(x)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = np.float32(1.0) / (np.float32(1.0) + np.exp(-x[positive], dtype=np.float32))
    exp_negative = np.exp(x[~positive], dtype=np.float32)
    out[~positive] = exp_negative / (np.float32(1.0) + exp_negative)
    return out


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish, i.e. ``x * sigmoid(x)`` (HF's ``act_fn`` for Qwen3)."""
    x = as_f32(x)
    return x * sigmoid(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Max-shifted softmax in FP32."""
    x = as_f32(x)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted, dtype=np.float32)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def causal_attention_mask(
    num_queries: int, num_keys: int, position_offset: int
) -> np.ndarray:
    """Additive ``[1, 1, num_queries, num_keys]`` FP32 mask.

    Query ``i`` sits at absolute position ``position_offset + i`` and may attend
    to key ``j`` iff ``j <= position_offset + i``. Passing the offset is what
    makes chunked prefill correct: a chunk starting at position 512 may see all
    512 cached keys, not just its own causal triangle.
    """
    query_positions = position_offset + np.arange(num_queries, dtype=np.int64)
    key_positions = np.arange(num_keys, dtype=np.int64)
    allowed = key_positions[None, :] <= query_positions[:, None]
    mask = np.zeros((1, 1, num_queries, num_keys), dtype=np.float32)
    mask[..., ~allowed] = NEG_INF_F32
    return mask


def group_query_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    num_kv_heads: int,
    scaling: float,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention with grouped-query heads.

    Shapes: ``q`` is ``[num_heads, T, head_dim]``, ``k``/``v`` are
    ``[num_kv_heads, T_total, head_dim]`` and ``mask`` broadcasts to
    ``[., ., T, T_total]``. Returns ``(context, scores)`` with shapes
    ``[num_heads, T, head_dim]`` and ``[num_heads, T, T_total]``.

    Instead of materialising ``repeat_kv(k)`` (which at T_total=2048 would copy
    16 MiB per layer per decoded token) the query heads are folded into the batch
    axis, so one batched matmul handles a whole GQA group. This is numerically
    identical to repeating K/V: query head ``kv * group + r`` attends to KV head
    ``kv`` in both formulations, in the same order HF's ``repeat_kv`` produces.
    """
    num_heads, num_queries, head_dim = q.shape
    if num_heads % num_kv_heads:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    kv_group_size = num_heads // num_kv_heads
    total_keys = k.shape[1]

    q_flat = q.reshape(num_kv_heads, kv_group_size * num_queries, head_dim)
    scores = np.matmul(q_flat, k.transpose(0, 2, 1))       # [nkv, group*T, T_total]
    scores *= F32(scaling)
    grouped_shape = (num_kv_heads, kv_group_size, num_queries, total_keys)
    if mask is not None:
        scores = (scores.reshape(grouped_shape) + mask).reshape(
            num_kv_heads, kv_group_size * num_queries, total_keys
        )
    probabilities = softmax(scores, axis=-1)
    context = np.matmul(probabilities, v)                  # [nkv, group*T, head_dim]
    return (
        context.reshape(num_heads, num_queries, head_dim),
        scores.reshape(num_heads, num_queries, total_keys),
    )


def split_heads(x: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    """``[T, num_heads * head_dim] -> [num_heads, T, head_dim]``."""
    num_tokens = x.shape[0]
    return x.reshape(num_tokens, num_heads, head_dim).transpose(1, 0, 2)


def merge_heads(x: np.ndarray, num_heads: int, num_queries: int, head_dim: int) -> np.ndarray:
    """``[num_heads, T, head_dim] -> [T, num_heads * head_dim]``."""
    return np.ascontiguousarray(x.transpose(1, 0, 2)).reshape(num_queries, num_heads * head_dim)
