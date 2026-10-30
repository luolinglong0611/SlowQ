"""Numerical operators, written to mirror ``transformers`` semantics.

Every function takes and returns FP32. Two rules keep us out of FP64 by
accident: inputs are coerced with ``np.float32`` scalars rather than Python
floats, and no operator ever calls ``.astype(np.float64)``.

Reference (transformers 4.51.x):

* ``Qwen3RMSNorm`` / ``LlamaRMSNorm`` -- variance in fp32,
  ``weight * (x * rsqrt(var + eps))``
* ``*RotaryEmbedding`` -- ``inv_freq = 1 / theta ** (arange(0, d, 2) / d)``,
  ``emb = cat(freqs, freqs)``, ``cos/sin`` cast back to the activation dtype.
  Identical in the qwen3, qwen2 and llama modules, and none of them reads
  ``rope_interleaved``, so one implementation covers all three families.
* ``apply_rotary_pos_emb`` -- ``x * cos + rotate_half(x) * sin`` with
  ``rotate_half`` splitting the head into two halves (not interleaved pairs)
* attention scaling -- multiply by ``head_dim ** -0.5``
* ``Idefics3VisionEncoderLayer`` -- LayerNorm with weight *and* bias, and
  ``gelu_pytorch_tanh``; the vision tower shares no normalisation or activation
  with the text decoder, so ``layer_norm`` and ``gelu_tanh`` live beside
  ``rms_norm`` and ``silu`` rather than replacing them.
"""

from __future__ import annotations

import numpy as np

F32 = np.float32
#: Same value HF uses for masking instead of -inf: ``torch.finfo(float32).min``.
NEG_INF_F32 = np.finfo(np.float32).min


def as_f32(array: np.ndarray) -> np.ndarray:
    """View/cast to FP32 without copying when it already is."""
    return array if array.dtype == np.float32 else array.astype(np.float32, copy=False)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """``x @ W.T (+ bias)``, keeping HF's ``[out_features, in_features]`` layout.

    The transpose is a view, so a converted checkpoint is never re-transposed on
    disk nor copied at load time. ``bias`` is ``None`` for every projection whose
    ``nn.Linear`` was declared with ``bias=False``, which is decided per family in
    :mod:`slowq.config` and not guessed here.
    """
    out = np.matmul(as_f32(x), as_f32(weight).T)
    return out if bias is None else out + as_f32(bias)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm over the last axis, accumulated in FP32."""
    x = as_f32(x)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = np.float32(1.0) / np.sqrt(variance + F32(eps))
    return as_f32(weight) * (x * inv_rms)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    """``nn.LayerNorm`` over the last axis: affine, biased variance, FP32."""
    x = as_f32(x)
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    normalised = (x - mean) / np.sqrt(variance + F32(eps))
    return as_f32(weight) * normalised + as_f32(bias)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """``gelu_pytorch_tanh``, the activation SmolVLM's vision MLP uses.

    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`` — the tanh
    approximation, not the erf-exact GELU. Using the wrong one is a small numeric
    difference that compounds through 12 encoder layers.
    """
    x = as_f32(x)
    inner = F32(0.7978845608028654) * (x + F32(0.044715) * x * x * x)
    return F32(0.5) * x * (F32(1.0) + np.tanh(inner))


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

    ``exp(-|x|)`` cannot overflow for any input, which lets both branches be
    written over the whole array instead of over a boolean mask: for ``x >= 0``
    the answer is ``1 / (1 + e)`` and for ``x < 0`` it is ``e / (1 + e)``, with
    the same ``e``. Those are the same two expressions the masked form computes,
    on the same values, so the result is bit-identical on every non-NaN element —
    including at ``-0.0`` (where ``x >= 0`` holds, as it did before) and at
    ``±inf``. NaN is the one exception, and it is an exception about *bit
    patterns*, not values: both forms return NaN in the same positions, but
    ``nan / nan`` can carry a different payload than ``exp(nan)`` did. Compare
    with ``np.array_equal(..., equal_nan=True)`` for the values and byte-compare
    only the non-NaN elements for the bitwise claim.

    What it replaces matters for speed, not just clarity — but the number that
    matters is not the operator's own speedup, and an earlier version of this
    docstring said so wrongly. Four quantities are in play and they are not
    interchangeable. Every figure below is a **range over every run preserved under
    ``artifacts/*/sigmoid/``**, never a single run of
    ``tools/perf/sigmoid_check.py`` (Qwen3-0.6B layer 0, 128-token prefill chunk,
    median of 15 with 5 warmups, Apple Accelerate / NumPy 2.5.4):

    1. **The operator alone**, old vs new on the captured activation: **3.90-4.35x
       (about 4x)**.
    2. **The sum of the layer's seven separately timed warm matmuls**, 3.17-3.36
       ms. Dividing (1)'s old time by it yields **88-96%**, which is a ratio of two
       unrelated measurements and must not be read as a share of anything: the
       denominator has no softmax, no norms, no residual adds and no elementwise
       work, so it is *not* a layer profile.
    3. **A real forward with only ``silu`` changed.** Same layer, same weights,
       same input tensor, same process, one cache object reused. This is the
       attribution: the rewrite removes **44-47% of the real MLP** and **27-31% of
       a real decoder layer** — attention, both norms and both residuals included.
    4. **The end-to-end gain**, which lives in the A/B workload data and is not
       measured here at all.

    Quote (3), and quote it as a range. Quoting (1) as an MLP or layer gain
    overstates it by however much of the layer is not sigmoid, which is most of
    it. Note how wide (3)'s range is compared with its own repeatability: within a
    run the 15 samples have stdev/median of 0.4-3.7%, yet the layer share differs
    by more than 4 points between runs. **A tight spread inside one process says
    nothing about how far two runs of the same probe can disagree** — which is why
    this docstring quotes ranges and why the artifacts keep every sample.

    The artifacts are in ``artifacts/perf-round1-rerun/sigmoid/`` (the four runs
    behind the correction), ``artifacts/probe-rerun-20261028/sigmoid/`` and
    ``artifacts/probe-rerun-20261029/sigmoid/`` (each the re-measurement owed after
    the probe itself was edited — the first to refuse overwriting evidence, the
    second to refuse it *before* the model load rather than after the measurement).
    ``out_guard`` will not let a new run land in a directory an old one is
    cited from, so one probe's artifacts accumulate across run roots rather than
    inside one, and "the latest" is the newest by mtime across all of them — which is
    how ``tests/test_perf_tooling.py`` finds it. Nothing is overwritten: within each
    root ``sigmoid-check-run*.json`` are the earlier runs and ``sigmoid-check.json``
    the last one taken there. Each records the probe's own hash and the
    hashes of the sources it measured, in two forms — the raw file hash, and one
    computed over the syntax tree with docstrings blanked out. The second is what
    decides whether a re-measurement is owed, because the numbers above live in
    ``ops.py``'s own prose: correcting that prose changes the raw hash without
    changing anything the probe executed. ``tests/test_perf_tooling.py`` reads every
    preserved run and fails if one falls outside the ranges quoted here, so those
    ranges cannot rot silently. Two bands have been widened by that gate rather than
    nudged by hand: the one-signed one 1.16 → 1.18, and the standard-normal one
    7.6 → 7.7.

    (1) depends strongly on where the input sits relative to zero, because the
    masked form only pays for gather/scatter when both halves are non-empty: on
    all-positive or all-negative input the two forms are within 1.06-1.18x of each
    other, while synthetic standard-normal input gives 7.3-7.7x. Real gate
    activations straddle zero — the captured activation from a real layer-0
    forward has mean -0.546, std 0.668 and **84.5% of its values negative**. Those
    three are not timings: every run reproduces them bit-for-bit, as it must,
    being a function of the weights and the input ids. Two notes on that:

    * The capture must come from a real forward. Recomputing it as
      ``rms_norm(embed_tokens(ids), input_layernorm.weight) @ gate_proj.weight.T``
      skips the whole attention block and the first residual, and uses
      ``input_layernorm`` where the MLP is fed by ``post_attention_layernorm``.
      That shortcut's output differs from the real activation by rel_l2 1.09 — the
      error is larger than the signal. An earlier probe used it, and every figure
      derived from it was withdrawn; the corrected numbers and the old ones are
      both in ``docs/推理加速-第1轮-2026-10-23.md`` §2.4.
    * The values straddle zero because that is what this model does, **not**
      because RMSNorm centres anything. RMSNorm divides by the root mean square;
      it never subtracts the mean. LayerNorm subtracts the mean. The captured
      mean of -0.546 is the disproof, on the real tensor.

    Reproduce all of it with ``tools/perf/sigmoid_check.py``, which captures the
    activation by observing the shipped forward rather than by re-deriving it, and
    which also checks bit equality on seven input families. Synthetic
    distributions stay in that script, labelled as bounds on the operator's input
    sensitivity — they are not measurements of this model.
    """
    x = as_f32(x)
    exp_neg_abs = np.exp(-np.abs(x), dtype=np.float32)
    denominator = exp_neg_abs + F32(1.0)
    return np.where(x >= 0, F32(1.0) / denominator, exp_neg_abs / denominator)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish, i.e. ``x * sigmoid(x)`` (HF's ``act_fn`` for the text MLP)."""
    x = as_f32(x)
    return x * sigmoid(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Max-shifted softmax in FP32.

    Must not write into ``x``. ``group_query_attention`` returns these same
    scores as its second value, documented as pre-softmax, so an in-place
    exponential here would hand the caller probabilities under that name while
    passing every numerical check.

    Folding the three temporaries into one buffer was tried and measured: ±1%
    against this version across decode- and prefill-shaped inputs, i.e. inside
    the run-to-run spread. The allocations are not what this function costs.
    """
    x = as_f32(x)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted, dtype=np.float32)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Unit-normalise the last axis, the pooling step of Qwen3-Embedding.

    HF's ``F.normalize`` divides by ``max(||x||, eps)``; the clamp only matters
    for an all-zero vector, where it yields zeros instead of NaNs.
    """
    x = as_f32(x)
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x / np.maximum(norm, F32(eps))


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
    With ``num_kv_heads == num_heads`` the group size is 1, which is exactly the
    multi-head attention the SmolVLM vision encoder needs, so it needs no second
    implementation.
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


def conv2d_patchify(
    image: np.ndarray, weight: np.ndarray, bias: np.ndarray, patch_size: int
) -> np.ndarray:
    """``nn.Conv2d(k=s=patch_size, padding="valid")`` over one ``[C, H, W]`` image.

    Returns ``[num_patches, out_channels]`` in row-major patch order, which is
    what ``Idefics3VisionEmbeddings`` produces with its
    ``permute(0, 2, 3, 1).reshape(B, -1, embed_dim)``.

    A stride-equal-kernel convolution is a gather plus a matmul, so no convolution
    is actually performed: the image is cut into non-overlapping ``[C, s, s]``
    blocks, each flattened in PyTorch's ``[out, in, kh, kw]`` order (channel-major,
    which is why the transpose below is ``(nh, nw, C, kh, kw)`` and not
    ``(nh, nw, kh, kw, C)``), and the whole grid multiplied by the reshaped
    kernel.
    """
    image = as_f32(image)
    channels, height, width = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image is {height}x{width}, which is not a whole number of "
            f"{patch_size}x{patch_size} patches"
        )
    rows, cols = height // patch_size, width // patch_size
    out_channels = weight.shape[0]
    blocks = image.reshape(channels, rows, patch_size, cols, patch_size)
    blocks = blocks.transpose(1, 3, 0, 2, 4).reshape(rows * cols, channels * patch_size * patch_size)
    kernel = as_f32(weight).reshape(out_channels, -1)
    return np.matmul(blocks, kernel.T) + as_f32(bias)


def pixel_shuffle(x: np.ndarray, scale_factor: int) -> np.ndarray:
    """``Idefics3Connector.pixel_shuffle``, reshape for reshape.

    ``[N, h*w, d] -> [N, (h/s)*(w/s), d*s*s]``. The two ``permute``/``reshape``
    pairs are not a no-op: they decide which neighbouring patches end up
    concatenated into one output feature, and therefore which image tokens the
    connector's projections see. Reproducing the sequence exactly is cheaper than
    reasoning about an equivalent one.
    """
    num_images, seq, embed_dim = x.shape
    height = width = int(seq**0.5)
    if height * width != seq:
        raise ValueError(f"pixel_shuffle expects a square patch grid, got seq={seq}")
    if height % scale_factor or width % scale_factor:
        raise ValueError(
            f"patch grid {height}x{width} is not divisible by scale_factor={scale_factor}"
        )
    x = x.reshape(num_images, height, width, embed_dim)
    x = x.reshape(num_images, height, width // scale_factor, embed_dim * scale_factor)
    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(
        num_images, width // scale_factor, height // scale_factor, embed_dim * scale_factor**2
    )
    x = x.transpose(0, 2, 1, 3)
    return np.ascontiguousarray(
        x.reshape(num_images, seq // scale_factor**2, embed_dim * scale_factor**2)
    )
