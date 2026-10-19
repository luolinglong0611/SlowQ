"""KV cache: pre-allocated slots, capacity checks, and position bookkeeping.

Cache semantics used everywhere in SlowQ
----------------------------------------
``length`` is *the number of tokens that have completed a full model forward*.
It is advanced once per forward, never once per layer. During a forward the
layers write into slots ``[position_offset, position_offset + T)`` and read
``[0, position_offset + T)``; ``length`` still reports the pre-forward value
until ``commit`` runs. That invariant is what makes chunked prefill and
single-token decode numerically identical to one long prefill.

Stored values
-------------
K is stored *after* Q/K RMSNorm and RoPE, so a cached key never needs rotating
again. V is stored as the raw projection output. Both keep the original
``num_key_value_heads`` axis: GQA expansion happens at attention time (see
``ops.group_query_attention``), so the cache is ``kv_group_size`` times smaller
than a naive repeated-KV cache.

Memory
------
Slots are allocated once for ``max_context`` and never grown, so decoding a
token is a strided slice assignment instead of ``np.concatenate`` over the whole
cache. For Qwen3-0.6B at ``max_context=2048`` that is
``2 * 28 layers * 8 kv heads * 2048 * 128 * 4 B ~= 448 MiB``; pass a smaller
``max_context`` to trade context length for memory.
"""

from __future__ import annotations

import numpy as np

from .errors import ContextOverflowError

COMPUTE_DTYPE = np.dtype(np.float32)


class KVCache:
    """Fixed-capacity key/value store for one sequence (batch size 1)."""

    def __init__(
        self,
        max_context: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        if max_context < 1:
            raise ValueError(f"max_context must be >= 1, got {max_context}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.max_context = int(max_context)
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.length = 0
        self.high_water_mark = 0

        layer_shape = (self.num_kv_heads, self.max_context, self.head_dim)
        # np.zeros -> calloc, so untouched pages are never resident.
        self.keys = [np.zeros(layer_shape, dtype=COMPUTE_DTYPE) for _ in range(num_layers)]
        self.values = [np.zeros(layer_shape, dtype=COMPUTE_DTYPE) for _ in range(num_layers)]

    # ---- capacity -----------------------------------------------------------

    @property
    def remaining(self) -> int:
        return self.max_context - self.length

    def check_capacity(self, total_length: int) -> None:
        """Raise rather than truncate or silently wrap around the buffer."""
        if total_length > self.max_context:
            raise ContextOverflowError(
                f"sequence would need {total_length} cache slots but max_context is "
                f"{self.max_context}. Re-run with a larger --max-context or a shorter "
                "prompt; SlowQ never truncates or overwrites cached tokens."
            )

    def allocated_bytes(self) -> int:
        return sum(a.nbytes for a in self.keys) + sum(a.nbytes for a in self.values)

    # ---- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Start a new sequence without reallocating.

        Slots beyond ``length`` are never read, so stale data from a previous
        (even aborted) request cannot leak into the next one.
        """
        self.length = 0
        self.high_water_mark = 0

    # ---- per-forward operations --------------------------------------------

    def write(
        self,
        layer_idx: int,
        new_keys: np.ndarray,
        new_values: np.ndarray,
        position_offset: int,
    ) -> None:
        """Store one layer's ``[num_kv_heads, T, head_dim]`` K/V at absolute positions."""
        if new_keys.shape != new_values.shape:
            raise ValueError(
                f"K/V shape mismatch: {new_keys.shape} vs {new_values.shape}"
            )
        num_new = new_keys.shape[1]
        end = position_offset + num_new
        if position_offset < 0:
            raise ValueError(f"position_offset must be >= 0, got {position_offset}")
        self.check_capacity(end)
        expected = (self.num_kv_heads, num_new, self.head_dim)
        if new_keys.shape != expected:
            raise ValueError(
                f"layer {layer_idx}: expected K/V shape {expected}, got {new_keys.shape}"
            )
        self.keys[layer_idx][:, position_offset:end, :] = new_keys
        self.values[layer_idx][:, position_offset:end, :] = new_values

    def read(self, layer_idx: int, total_length: int) -> tuple[np.ndarray, np.ndarray]:
        """Views of the first ``total_length`` cached positions for one layer."""
        if total_length > self.max_context:
            raise ContextOverflowError(
                f"cannot read {total_length} cache slots from a cache of size "
                f"{self.max_context}"
            )
        return (
            self.keys[layer_idx][:, :total_length, :],
            self.values[layer_idx][:, :total_length, :],
        )

    def commit(self, num_tokens: int) -> None:
        """Advance the global length after a complete forward over ``num_tokens``."""
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")
        self.length += num_tokens
        self.high_water_mark = max(self.high_water_mark, self.length)

    def __repr__(self) -> str:
        return (
            f"KVCache(length={self.length}, max_context={self.max_context}, "
            f"layers={self.num_layers}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, allocated={self.allocated_bytes() / 2**20:.1f} MiB)"
        )
