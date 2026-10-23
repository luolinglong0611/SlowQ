"""Text-decoder forward pass: attention, MLP, decoder layer, full model.

One implementation covers the qwen3, qwen2 and llama families, because after
checking each against its ``transformers`` 4.51.3 module the differences turn out
to be flags rather than structure::

    residual = x
    x = residual + Attention(input_layernorm(x))
    residual = x
    x = residual + MLP(post_attention_layernorm(x))

and the flags are:

=====================  ==============================  ====================  =================
                       qwen3                           qwen2                 llama
=====================  ==============================  ====================  =================
Q/K RMSNorm per head   yes                             no                    no
q/k/v bias             ``config.attention_bias``       always                ``config.attention_bias``
o_proj bias            ``config.attention_bias``       never                 ``config.attention_bias``
MLP bias               never                           never                 ``config.mlp_bias``
``head_dim``           explicit in config              ``hidden // heads``   ``hidden // heads``
=====================  ==============================  ====================  =================

Everything else — RMSNorm, RoPE, SwiGLU, GQA, the causal mask — is shared and was
already verified identical across the three modules.

Attention structure (the parts that are easy to get wrong)::

    q = q_proj(x) [+ bias] -> [T, num_attention_heads * head_dim]
    k = k_proj(x) [+ bias] -> [T, num_key_value_heads * head_dim]
    v = v_proj(x) [+ bias]
    q = q_norm(q)   # qwen3 only; RMSNorm per head over head_dim, not hidden_size
    k = k_norm(k)   # v is neither normed nor rotated
    q, k = rope(q, k, positions = offset .. offset + T - 1)
    scores = q @ repeat_kv(k).T * head_dim ** -0.5 + causal_mask(offset)
    out = softmax(scores) @ repeat_kv(v) -> o_proj [+ bias]

Note that ``head_dim * num_attention_heads`` need not equal ``hidden_size``:
Qwen3-0.6B has 16 * 128 = 2048 against a hidden_size of 1024, so the projection
output is reshaped by ``head_dim`` and never assumed to be ``hidden_size``.

See docs/architecture.md for the full shape table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .cache import KVCache
from .config import TASK_TEXT_EMBEDDING, TASK_TEXT_GENERATION, ModelConfig
from .errors import ConfigError, ContextOverflowError, SlowQError
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
from .weights import (
    Weights,
    embedding_tensor_name,
    output_embedding_name,
)


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
            "embed_sequence",
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

    def record_sequence(self, name: str, tensor: np.ndarray) -> None:
        """Store a whole ``[T, H]`` activation instead of selected positions.

        Only ``embed_sequence`` uses it: the merged image+text embedding matrix is
        the one activation where *which row got which value* is the thing under
        test, so sampling positions would leave most of the merge unobserved. It
        is a prefill-sized tensor, which is why it is opt-in by name.
        """
        self.records[name] = np.array(tensor, dtype=np.float32)


@dataclass(frozen=True)
class DecoderOutput:
    """Output of the decoder stack alone: final-normed hidden states, no logits."""

    hidden_states: np.ndarray
    position_offset: int
    num_tokens: int
    cache_length: int


@dataclass(frozen=True)
class ForwardResult:
    """Output of one full model forward.

    ``logits`` covers only the last position (``[vocab_size]``): projecting the
    whole sequence would build a ``[T, vocab_size]`` FP32 tensor and throw it away.
    """

    logits: np.ndarray
    hidden_states: np.ndarray
    position_offset: int
    num_tokens: int
    cache_length: int


class DecoderModel:
    """The family-generic text decoder over a converted weight directory.

    Runs the embedding table, every decoder layer and the final RMSNorm. The
    output projection is a separate step because a text-embedding checkpoint has
    no ``lm_head`` at all and must not be asked for one.
    """

    def __init__(self, config: ModelConfig, weights: Weights) -> None:
        self.config = config
        self.weights = weights
        weights.check_complete(config)
        self.inv_freq = rope_inv_freq(config.head_dim, config.rope_theta)

    @classmethod
    def from_directory(cls, model_dir: str | Path, task: str | None = None) -> "DecoderModel":
        """Low-level loader. ``slowq.Engine.from_directory`` is the friendlier one."""
        config, weights = load_config_and_weights(model_dir, task)
        return cls(config, weights)

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
            embedding_tensor_name(self.config),
            (self.config.vocab_size, self.config.hidden_size),
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
        """``[hidden_size]`` -> ``[vocab_size]`` logits through the lm_head."""
        if self.config.task == TASK_TEXT_EMBEDDING:
            raise SlowQError(
                f"task {self.config.task!r} produces a normalised hidden-state vector, "
                "not a vocabulary distribution; this checkpoint has no lm_head to "
                "project through. Use slowq.embedding.EmbeddingModel.encode instead."
            )
        name = output_embedding_name(self.config)
        weight = self.weights.expect(
            name, (self.config.vocab_size, self.config.hidden_size)
        )
        hidden = np.asarray(hidden_last, dtype=np.float32).reshape(1, -1)
        return np.matmul(hidden, weight.T)[0]

    def encode(
        self,
        input_ids: Sequence[int] | np.ndarray,
        kv_cache: KVCache,
        position_offset: int | None = None,
        probe: DebugProbe | None = None,
        inputs_embeds: np.ndarray | None = None,
        inputs_embed_mask: np.ndarray | None = None,
    ) -> DecoderOutput:
        """Run the decoder stack over ``input_ids``, appending their K/V to the cache.

        ``position_offset`` defaults to ``kv_cache.length``. Passing it only lets
        a caller state the invariant they rely on; a value that disagrees with
        the cache is rejected rather than trusted, which is what keeps chunked
        prefill from drifting off the position grid.
        """
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        num_tokens = int(ids.shape[0])
        if num_tokens == 0:
            raise ValueError("encode() needs at least one token")

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
        if self.config.sliding_window is not None and total_length > self.config.sliding_window:
            raise ContextOverflowError(
                f"position {total_length - 1} exceeds this checkpoint's "
                f"sliding_window={self.config.sliding_window}; SlowQ implements full "
                "attention only, and the reference would have masked those keys out."
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
        hidden = self._merge_inputs_embeds(hidden, inputs_embeds, inputs_embed_mask, num_tokens)
        if probe is not None and probe.wants("embed"):
            probe.record("embed", hidden)
        if probe is not None and probe.wants("embed_sequence"):
            probe.record_sequence("embed_sequence", hidden)

        for layer_idx in range(self.config.num_hidden_layers):
            hidden = self._decoder_layer(
                layer_idx, hidden, cos, sin, kv_cache, offset, total_length, mask, probe
            )

        hidden = rms_norm(
            hidden,
            self.weights.expect(
                f"{self.config.tensor_prefix}.norm.weight", (self.config.hidden_size,)
            ),
            self.config.rms_norm_eps,
        )
        if probe is not None and probe.wants("final_norm"):
            probe.record("final_norm", hidden)

        # Advance the length once, after every layer has written its own slots.
        kv_cache.commit(num_tokens)

        return DecoderOutput(
            hidden_states=hidden,
            position_offset=offset,
            num_tokens=num_tokens,
            cache_length=kv_cache.length,
        )

    def forward(
        self,
        input_ids: Sequence[int] | np.ndarray,
        kv_cache: KVCache,
        position_offset: int | None = None,
        probe: DebugProbe | None = None,
        inputs_embeds: np.ndarray | None = None,
        inputs_embed_mask: np.ndarray | None = None,
    ) -> ForwardResult:
        """``encode`` plus the output projection of the block's last position."""
        decoded = self.encode(
            input_ids,
            kv_cache,
            position_offset=position_offset,
            probe=probe,
            inputs_embeds=inputs_embeds,
            inputs_embed_mask=inputs_embed_mask,
        )
        logits = self.output_projection(decoded.hidden_states[-1])
        if probe is not None and probe.wants("logits"):
            self._record_probe_logits(probe, decoded.hidden_states, logits)
        return ForwardResult(
            logits=logits,
            hidden_states=decoded.hidden_states,
            position_offset=decoded.position_offset,
            num_tokens=decoded.num_tokens,
            cache_length=decoded.cache_length,
        )

    # ---- internals ----------------------------------------------------------

    def _merge_inputs_embeds(
        self,
        hidden: np.ndarray,
        inputs_embeds: np.ndarray | None,
        inputs_embed_mask: np.ndarray | None,
        num_tokens: int,
    ) -> np.ndarray:
        """Scatter pre-computed embeddings over the placeholder positions.

        The count is checked against the mask rather than trusted: a prompt whose
        ``<image>`` expansion disagrees with what the vision tower produced must
        fail here, loudly, instead of silently dropping image features or writing
        them onto text tokens.
        """
        if inputs_embeds is None:
            if inputs_embed_mask is not None and np.any(inputs_embed_mask):
                raise ValueError(
                    "inputs_embed_mask marks placeholder positions but no "
                    "inputs_embeds were supplied"
                )
            return hidden
        if inputs_embed_mask is None:
            raise ValueError("inputs_embeds require an inputs_embed_mask")
        mask = np.asarray(inputs_embed_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != num_tokens:
            raise ValueError(
                f"inputs_embed_mask covers {mask.shape[0]} positions but this "
                f"forward has {num_tokens} tokens"
            )
        features = np.asarray(inputs_embeds, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.config.hidden_size:
            raise ValueError(
                f"inputs_embeds must be [n, {self.config.hidden_size}], got "
                f"{features.shape}"
            )
        selected = int(mask.sum())
        if selected != features.shape[0]:
            raise ValueError(
                f"the prompt expands to {selected} placeholder tokens but the "
                f"encoder produced {features.shape[0]} feature rows; the image "
                "token count and the vision output disagree, so there is no "
                "correct way to line them up"
            )
        hidden = hidden.copy()
        hidden[mask] = features
        return hidden

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
        prefix = f"{cfg.tensor_prefix}.layers.{layer_idx}"

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
        attn = f"{prefix}.self_attn"

        def _bias(projection: str, size: int, enabled: bool) -> np.ndarray | None:
            return w.expect(f"{attn}.{projection}.bias", (size,)) if enabled else None

        q = split_heads(
            linear(
                x,
                w.expect(f"{attn}.q_proj.weight", (cfg.q_proj_out, cfg.hidden_size)),
                _bias("q_proj", cfg.q_proj_out, cfg.qkv_bias),
            ),
            cfg.num_attention_heads,
            cfg.head_dim,
        )
        k = split_heads(
            linear(
                x,
                w.expect(f"{attn}.k_proj.weight", (cfg.kv_proj_out, cfg.hidden_size)),
                _bias("k_proj", cfg.kv_proj_out, cfg.qkv_bias),
            ),
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        v = split_heads(
            linear(
                x,
                w.expect(f"{attn}.v_proj.weight", (cfg.kv_proj_out, cfg.hidden_size)),
                _bias("v_proj", cfg.kv_proj_out, cfg.qkv_bias),
            ),
            cfg.num_key_value_heads,
            cfg.head_dim,
        )

        if cfg.qk_norm:
            # transformers norms Q and K on [B, T, heads, head_dim] before the
            # head transpose; RMSNorm is over the last axis, so norming after
            # split_heads is the same numbers in a different axis order.
            q = rms_norm(
                q, w.expect(f"{attn}.q_norm.weight", (cfg.head_dim,)), cfg.rms_norm_eps
            )
            k = rms_norm(
                k, w.expect(f"{attn}.k_norm.weight", (cfg.head_dim,)), cfg.rms_norm_eps
            )
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
            w.expect(f"{attn}.o_proj.weight", (cfg.hidden_size, cfg.q_proj_out)),
            _bias("o_proj", cfg.hidden_size, cfg.o_proj_bias),
        )

    def _mlp(self, prefix: str, x: np.ndarray) -> np.ndarray:
        """SwiGLU: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""
        cfg = self.config
        w = self.weights
        mlp = f"{prefix}.mlp"

        def _bias(projection: str, size: int) -> np.ndarray | None:
            return w.expect(f"{mlp}.{projection}.bias", (size,)) if cfg.mlp_bias else None

        gate = linear(
            x,
            w.expect(f"{mlp}.gate_proj.weight", (cfg.intermediate_size, cfg.hidden_size)),
            _bias("gate_proj", cfg.intermediate_size),
        )
        up = linear(
            x,
            w.expect(f"{mlp}.up_proj.weight", (cfg.intermediate_size, cfg.hidden_size)),
            _bias("up_proj", cfg.intermediate_size),
        )
        return linear(
            silu(gate) * up,
            w.expect(f"{mlp}.down_proj.weight", (cfg.hidden_size, cfg.intermediate_size)),
            _bias("down_proj", cfg.hidden_size),
        )


class CausalLMModel(DecoderModel):
    """A decoder whose task is producing the next token.

    The only thing this adds over :class:`DecoderModel` is refusing to be built
    for a checkpoint that cannot do the job — an embedding directory has no
    ``lm_head``, and failing here names the reason instead of surfacing later as
    a missing-tensor error from inside a forward pass.
    """

    def __init__(self, config: ModelConfig, weights: Weights) -> None:
        if config.task == TASK_TEXT_EMBEDDING:
            raise SlowQError(
                f"the converted directory at hand is a {config.task!r} model, so it "
                "has no output projection. Load it with slowq.embedding."
                "EmbeddingModel (CLI: ``slowq embed``) instead of a causal LM."
            )
        super().__init__(config, weights)

    @classmethod
    def from_directory(cls, model_dir: str | Path, task: str | None = None) -> "CausalLMModel":
        config, weights = load_config_and_weights(model_dir, task)
        return cls(config, weights)


def load_config_and_weights(
    model_dir: str | Path, task: str | None = None
) -> tuple[ModelConfig, Weights]:
    """Read a converted directory's ``config.json`` and manifest together.

    The task is taken from the manifest, which is the only place it can come
    from: ``Qwen/Qwen3-Embedding-0.6B`` publishes ``model_type: "qwen3"`` and
    ``architectures: ["Qwen3ForCausalLM"]``, identical to the chat model, so no
    amount of looking at the config can tell the two apart. Passing ``task``
    overrides the manifest and is how the reference-validation tool loads a
    checkpoint it has just been told what to do with.

    The *family* is the other direction: it is read from ``config.json``, and a
    version-2 manifest also records the one the converter saw. The two must
    agree. They are independent claims about the same directory, and a directory
    where they disagree is refused here rather than loaded.

    What the disagreement would otherwise cost depends on which side was altered.
    Decoding always follows ``config.json``, so an edited manifest leaves the
    numbers untouched and misreports the directory instead — ``slowq inspect``
    would state a family the weights were never decoded with. A ``config.json``
    swapped for another family's is the dangerous direction: that would not
    crash, it would just compute different numbers.
    """
    root = Path(model_dir)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise SlowQError(
            f"{config_path} not found; is this a converted SlowQ model directory?"
        )
    weights = Weights.from_directory(root)
    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = task if task is not None else weights.manifest.task
    if resolved is None:
        # A version-1 manifest, written by SlowQ v0.1, which could only have
        # converted a Qwen3 causal LM.
        resolved = TASK_TEXT_GENERATION
    config = ModelConfig.from_hf_config(raw, task=resolved)
    recorded = weights.manifest.model_family
    if recorded is not None and recorded != config.family:
        # ``None`` means a version-1 manifest, which predates the key; those
        # directories are still loadable and are not asked to grow a field they
        # were written without.
        raise ConfigError(
            f"{root} is inconsistent: its manifest records "
            f"model_family={recorded!r} but its config.json declares "
            f"model_type={config.family!r}. Weights would be decoded with the "
            f"{config.family!r} rules while the directory claims to be "
            f"{recorded!r}. Re-convert it: python tools/convert_model.py --help"
        )
    return config, weights
