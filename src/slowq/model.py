"""Qwen3 forward pass: attention, MLP, decoder layer, full model.

Mirrors ``transformers`` 4.51.x ``models/qwen3/modeling_qwen3.py`` in eager
attention mode, batch size 1, no dropout. Layer structure::

    residual = x
    x = residual + Attention(input_layernorm(x))
    residual = x
    x = residual + MLP(post_attention_layernorm(x))

Attention structure (the parts that are easy to get wrong)::

    q = q_proj(x) -> [T, 16 * 128]   # 16 * 128 = 2048 != hidden_size 1024
    k = k_proj(x) -> [T,  8 * 128]
    v = v_proj(x) -> [T,  8 * 128]
    q = q_norm(q)   # RMSNorm per head over head_dim, not over hidden_size
    k = k_norm(k)   # v is neither normed nor rotated
    q, k = rope(q, k, positions = offset .. offset + T - 1)
    scores = q @ repeat_kv(k).T * head_dim ** -0.5 + causal_mask(offset)
    out = softmax(scores) @ repeat_kv(v) -> o_proj

See docs/architecture.md for the full shape table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .cache import KVCache
from .config import ModelConfig
from .errors import ContextOverflowError, SlowQError
from .ops import (
    apply_rope,
    causal_attention_mask,
    group_query_attention,
    linear,
    merge_heads,
    rms_norm,
    rope_cos_sin,
    rope_inv_freq,
    silu,
    split_heads,
)
from .weights import EMBEDDING_TENSOR_NAME, TIED_OUTPUT_EMBEDDING_ALIAS, Weights


def resolve_probe_position(position: int, num_tokens: int) -> int:
    """Map a probe position onto an index of this forward's input block.

    Positions are relative to the block, so ``0`` is the block's first token and
    ``-1`` its last — the same convention for every capture point, including in a
    decode step where the block is one token long. A position the block does not
    have is a caller error; clamping or wrapping it would silently report the
    wrong token's activations.
    """
    index = position + num_tokens if position < 0 else position
    if not 0 <= index < num_tokens:
        raise ValueError(
            f"probe position {position} is outside this forward's {num_tokens}-token "
            f"input block. Positions are relative to the block, so the valid ones "
            f"are 0..{num_tokens - 1} or -1..-{num_tokens}."
        )
    return index


@dataclass
class DebugProbe:
    """Opt-in capture of intermediate activations.

    Off by default: keeping every activation of a 2048-token prefill would need
    gigabytes. Ask for specific layers and positions instead, e.g.
    ``DebugProbe.for_names({"layer_output"}, layer_indices={0, 27}, positions=(-1,))``.
    """

    layer_indices: frozenset[int] | None = None
    positions: tuple[int, ...] = (-1,)
    captures: frozenset[str] = frozenset({"layer_output", "final_norm", "logits"})
    records: dict[str, np.ndarray] = field(default_factory=dict)

    ALL_CAPTURES = frozenset(
        {
            "embed",
            "attention_output",
            "mlp_output",
            "layer_output",
            "final_norm",
            "logits",
        }
    )

    @classmethod
    def for_names(
        cls,
        captures: Iterable[str],
        layer_indices: Iterable[int] | None = None,
        positions: Iterable[int] = (-1,),
    ) -> "DebugProbe":
        unknown = set(captures) - cls.ALL_CAPTURES
        if unknown:
            raise ValueError(
                f"unknown capture points {sorted(unknown)}; "
                f"choose from {sorted(cls.ALL_CAPTURES)}"
            )
        return cls(
            layer_indices=None if layer_indices is None else frozenset(layer_indices),
            positions=tuple(positions),
            captures=frozenset(captures),
        )

    def wants(self, name: str, layer_idx: int | None = None) -> bool:
        if name not in self.captures:
            return False
        return layer_idx is None or self.layer_indices is None or layer_idx in self.layer_indices

    def record(self, name: str, tensor: np.ndarray, layer_idx: int | None = None) -> None:
        """Store the requested positions of ``tensor`` (shape ``[T, ...]``)."""
        prefix = "" if layer_idx is None else f"layer{layer_idx}."
        length = int(np.asarray(tensor).shape[0])
        for position in self.positions:
            index = resolve_probe_position(position, length)
            self.records[f"{prefix}{name}[{position}]"] = np.array(
                tensor[index], dtype=np.float32
            )


@dataclass(frozen=True)
class ForwardResult:
    """Output of one full model forward.

    ``logits`` covers only the last position (``[vocab_size]``): projecting the
    whole sequence would build a ``[T, 151936]`` FP32 tensor and throw it away.
    """

    logits: np.ndarray
    hidden_states: np.ndarray
    position_offset: int
    num_tokens: int
    cache_length: int


class Qwen3Model:
    """The complete Qwen3 causal LM over a converted weight directory."""

    def __init__(self, config: ModelConfig, weights: Weights) -> None:
        self.config = config
        self.weights = weights
        weights.check_complete(config)
        self.inv_freq = rope_inv_freq(config.head_dim, config.rope_theta)

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "Qwen3Model":
        """Low-level loader. ``slowq.Engine.from_directory`` is the friendlier one."""
        root = Path(model_dir)
        config_path = root / "config.json"
        if not config_path.is_file():
            raise SlowQError(
                f"{config_path} not found; is this a converted SlowQ model directory?"
            )
        raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(ModelConfig.from_hf_config(raw), Weights.from_directory(root))

    def new_cache(self, max_context: int) -> KVCache:
        self.config.validate_max_context(max_context)
        return KVCache(
            max_context=max_context,
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
        )

    # ---- entry points -------------------------------------------------------

    def embed(self, input_ids: np.ndarray) -> np.ndarray:
        """``[T] int`` -> ``[T, hidden_size]`` FP32."""
        table = self.weights.expect(
            EMBEDDING_TENSOR_NAME, (self.config.vocab_size, self.config.hidden_size)
        )
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("input_ids must contain at least one token")
        bad = ids[(ids < 0) | (ids >= self.config.vocab_size)]
        if bad.size:
            raise ValueError(
                f"token id(s) {sorted(set(bad.tolist()))[:5]} are outside the "
                f"vocabulary [0, {self.config.vocab_size})"
            )
        return np.array(table[ids], dtype=np.float32)

    def output_projection(self, hidden_last: np.ndarray) -> np.ndarray:
        """``[hidden_size]`` -> ``[vocab_size]`` logits through the (tied) lm_head."""
        name = (
            EMBEDDING_TENSOR_NAME
            if self.config.tie_word_embeddings
            else TIED_OUTPUT_EMBEDDING_ALIAS
        )
        weight = self.weights.expect(name, (self.config.vocab_size, self.config.hidden_size))
        hidden = np.asarray(hidden_last, dtype=np.float32).reshape(1, -1)
        return np.matmul(hidden, weight.T)[0]

    def forward(
        self,
        input_ids: Sequence[int] | np.ndarray,
        kv_cache: KVCache,
        position_offset: int | None = None,
        probe: DebugProbe | None = None,
    ) -> ForwardResult:
        """Run one forward over ``input_ids`` and append their K/V to ``kv_cache``.

        ``position_offset`` defaults to ``kv_cache.length``. Passing it only lets
        a caller state the invariant they rely on; a value that disagrees with
        the cache is rejected rather than trusted, which is what keeps chunked
        prefill from drifting off the position grid.
        """
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        num_tokens = int(ids.shape[0])
        if num_tokens == 0:
            raise ValueError("forward() needs at least one token")

        offset = kv_cache.length if position_offset is None else int(position_offset)
        if offset != kv_cache.length:
            raise ValueError(
                f"position_offset={offset} disagrees with kv_cache.length="
                f"{kv_cache.length}. Positions are derived from the cache so "
                "prefill chunks and decode steps cannot drift."
            )
        total_length = offset + num_tokens
        kv_cache.check_capacity(total_length)
        if total_length > self.config.max_position_embeddings:
            raise ContextOverflowError(
                f"position {total_length - 1} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )

        positions = np.arange(offset, total_length, dtype=np.int64)
        cos, sin = rope_cos_sin(positions, self.inv_freq)
        mask = causal_attention_mask(num_tokens, total_length, offset)

        if probe is not None:
            # Checked up front: a bad position must be reported before the cache
            # has been written to, not 28 layers in.
            for position in probe.positions:
                resolve_probe_position(position, num_tokens)

        hidden = self.embed(ids)
        if probe is not None and probe.wants("embed"):
            probe.record("embed", hidden)

        for layer_idx in range(self.config.num_hidden_layers):
            hidden = self._decoder_layer(
                layer_idx, hidden, cos, sin, kv_cache, offset, total_length, mask, probe
            )

        hidden = rms_norm(
            hidden,
            self.weights.expect("model.norm.weight", (self.config.hidden_size,)),
            self.config.rms_norm_eps,
        )
        if probe is not None and probe.wants("final_norm"):
            probe.record("final_norm", hidden)

        logits = self.output_projection(hidden[-1])
        if probe is not None and probe.wants("logits"):
            self._record_probe_logits(probe, hidden, logits)

        # Advance the length once, after every layer has written its own slots.
        kv_cache.commit(num_tokens)

        return ForwardResult(
            logits=logits,
            hidden_states=hidden,
            position_offset=offset,
            num_tokens=num_tokens,
            cache_length=kv_cache.length,
        )

    # ---- internals ----------------------------------------------------------

    def _record_probe_logits(
        self, probe: DebugProbe, hidden: np.ndarray, last_logits: np.ndarray
    ) -> None:
        """Record one logit row per requested position of this block.

        ``ForwardResult.logits`` covers the last position only, so any other
        position needs an output projection of its own. Only the requested ones
        are computed: a full-vocab matmul is the most expensive single operation
        in the model, and positions nobody asked for must not cost anything. The
        last position reuses the projection generation already needs, which also
        keeps it bit-identical to ``ForwardResult.logits``.
        """
        num_tokens = int(hidden.shape[0])
        projected: dict[int, np.ndarray] = {num_tokens - 1: last_logits}
        for position in probe.positions:
            index = resolve_probe_position(position, num_tokens)
            if index not in projected:
                projected[index] = self.output_projection(hidden[index])
            probe.records[f"logits[{position}]"] = np.array(
                projected[index], dtype=np.float32
            )

    def _decoder_layer(
        self,
        layer_idx: int,
        hidden: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray,
        kv_cache: KVCache,
        position_offset: int,
        total_length: int,
        mask: np.ndarray,
        probe: DebugProbe | None,
    ) -> np.ndarray:
        cfg = self.config
        prefix = f"model.layers.{layer_idx}"

        residual = hidden
        normed = rms_norm(
            hidden,
            self.weights.expect(f"{prefix}.input_layernorm.weight", (cfg.hidden_size,)),
            cfg.rms_norm_eps,
        )
        attention = self._attention(
            layer_idx, prefix, normed, cos, sin, kv_cache, position_offset, total_length, mask
        )
        if probe is not None and probe.wants("attention_output", layer_idx):
            probe.record("attention_output", attention, layer_idx)
        hidden = residual + attention

        residual = hidden
        normed = rms_norm(
            hidden,
            self.weights.expect(
                f"{prefix}.post_attention_layernorm.weight", (cfg.hidden_size,)
            ),
            cfg.rms_norm_eps,
        )
        mlp = self._mlp(prefix, normed)
        if probe is not None and probe.wants("mlp_output", layer_idx):
            probe.record("mlp_output", mlp, layer_idx)
        hidden = residual + mlp

        if probe is not None and probe.wants("layer_output", layer_idx):
            probe.record("layer_output", hidden, layer_idx)
        return hidden

    def _attention(
        self,
        layer_idx: int,
        prefix: str,
        x: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray,
        kv_cache: KVCache,
        position_offset: int,
        total_length: int,
        mask: np.ndarray,
    ) -> np.ndarray:
        """``[T, hidden_size] -> [T, hidden_size]``; writes this layer's K/V."""
        cfg = self.config
        num_tokens = x.shape[0]
        w = self.weights

        q = split_heads(
            linear(x, w.expect(f"{prefix}.self_attn.q_proj.weight", (cfg.q_proj_out, cfg.hidden_size))),
            cfg.num_attention_heads,
            cfg.head_dim,
        )
        k = split_heads(
            linear(x, w.expect(f"{prefix}.self_attn.k_proj.weight", (cfg.kv_proj_out, cfg.hidden_size))),
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        v = split_heads(
            linear(x, w.expect(f"{prefix}.self_attn.v_proj.weight", (cfg.kv_proj_out, cfg.hidden_size))),
            cfg.num_key_value_heads,
            cfg.head_dim,
        )

        q = rms_norm(q, w.expect(f"{prefix}.self_attn.q_norm.weight", (cfg.head_dim,)), cfg.rms_norm_eps)
        k = rms_norm(k, w.expect(f"{prefix}.self_attn.k_norm.weight", (cfg.head_dim,)), cfg.rms_norm_eps)
        q = np.ascontiguousarray(apply_rope(q, cos, sin))
        k = np.ascontiguousarray(apply_rope(k, cos, sin))

        kv_cache.write(layer_idx, k, v, position_offset)
        cached_k, cached_v = kv_cache.read(layer_idx, total_length)

        context, _ = group_query_attention(
            q=q,
            k=cached_k,
            v=cached_v,
            num_kv_heads=cfg.num_key_value_heads,
            scaling=cfg.attention_scaling,
            mask=mask,
        )
        merged = merge_heads(context, cfg.num_attention_heads, num_tokens, cfg.head_dim)
        return linear(
            merged,
            w.expect(f"{prefix}.self_attn.o_proj.weight", (cfg.hidden_size, cfg.q_proj_out)),
        )

    def _mlp(self, prefix: str, x: np.ndarray) -> np.ndarray:
        """SwiGLU: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""
        cfg = self.config
        w = self.weights
        gate = linear(x, w.expect(f"{prefix}.mlp.gate_proj.weight", (cfg.intermediate_size, cfg.hidden_size)))
        up = linear(x, w.expect(f"{prefix}.mlp.up_proj.weight", (cfg.intermediate_size, cfg.hidden_size)))
        return linear(
            silu(gate) * up,
            w.expect(f"{prefix}.mlp.down_proj.weight", (cfg.hidden_size, cfg.intermediate_size)),
        )
