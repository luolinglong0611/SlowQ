"""Model configuration: architecture families, task types, support-range checks.

Everything numeric the model needs is read from the official ``config.json``;
nothing is derived from a hard-coded 0.6B constant.

Two independent axes are modelled here, and keeping them apart is the point:

``family``
    Which decoder implementation the checkpoint uses — ``qwen3``, ``qwen2`` or
    ``llama``. This decides the *forward pass* (Q/K RMSNorm, which projections
    carry a bias) and the *weight layout* (which tensors must exist). It is read
    from ``config.json``'s ``model_type``, which is exactly the key
    ``transformers``' own ``AutoModel`` dispatches on.

``task``
    What the checkpoint is *for* — text generation, text embedding, or
    image+text generation. This cannot be read from ``model_type`` or
    ``architectures``: ``Qwen/Qwen3-Embedding-0.6B`` ships
    ``model_type: "qwen3"`` and ``architectures: ["Qwen3ForCausalLM"]``, the
    same pair as the chat model. The task therefore comes from the converted
    directory's manifest, where the converter recorded the evidence it used.

Headroom for a fourth family, ``idefics3`` (SmolVLM), lives in :mod:`slowq.vlm`
because its config nests a text config and a vision config rather than being a
text config itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError

# ---- tasks -------------------------------------------------------------------

TASK_TEXT_GENERATION = "text-generation"
TASK_TEXT_EMBEDDING = "text-embedding"
TASK_IMAGE_TEXT_GENERATION = "image-text-generation"

#: Every task SlowQ can run. A converted directory must name one of these.
KNOWN_TASKS: tuple[str, ...] = (
    TASK_TEXT_GENERATION,
    TASK_TEXT_EMBEDDING,
    TASK_IMAGE_TEXT_GENERATION,
)

#: Tasks whose forward pass ends in a next-token distribution.
GENERATIVE_TASKS: tuple[str, ...] = (TASK_TEXT_GENERATION, TASK_IMAGE_TEXT_GENERATION)

# ---- families ----------------------------------------------------------------

FAMILY_QWEN3 = "qwen3"
FAMILY_QWEN2 = "qwen2"
FAMILY_LLAMA = "llama"
FAMILY_IDEFICS3 = "idefics3"

SUPPORTED_HIDDEN_ACT = "silu"
RUNTIME_DTYPE_NAME = "float32"

#: Bias rules below are either of these literals, or ``"config:<key>"``.
BIAS_ALWAYS_ON = "on"
BIAS_ALWAYS_OFF = "off"


@dataclass(frozen=True)
class TextFamily:
    """How one HF text-decoder family differs from the others.

    Every field is a fact read off the ``transformers`` 4.51.3 modelling code
    for that family, not a guess from a model's name; ``docs/architecture.md``
    lists the exact lines.
    """

    name: str
    #: ``config.json`` values of ``model_type`` that select this family.
    model_types: tuple[str, ...]
    #: ``config.json`` values of ``architectures`` accepted alongside them.
    architectures: tuple[str, ...]
    qkv_bias: str
    o_proj_bias: str
    mlp_bias: str
    #: Qwen3 normalises Q and K per head before RoPE; qwen2 and llama do not.
    qk_norm: bool
    #: Qwen3 sets ``head_dim`` independently of ``hidden_size / heads`` (0.6B:
    #: 16 * 128 = 2048 != 1024), so a missing value must not be inferred there.
    head_dim_must_be_explicit: bool
    #: qwen2 and qwen3 apply ``config.sliding_window`` in the model-level causal
    #: mask regardless of ``use_sliding_window``; llama has no sliding window.
    honours_config_sliding_window: bool


TEXT_FAMILIES: tuple[TextFamily, ...] = (
    TextFamily(
        name=FAMILY_QWEN3,
        model_types=("qwen3",),
        architectures=("Qwen3ForCausalLM", "Qwen3Model"),
        qkv_bias="config:attention_bias",
        o_proj_bias="config:attention_bias",
        mlp_bias=BIAS_ALWAYS_OFF,
        qk_norm=True,
        head_dim_must_be_explicit=True,
        honours_config_sliding_window=True,
    ),
    TextFamily(
        name=FAMILY_QWEN2,
        model_types=("qwen2",),
        architectures=("Qwen2ForCausalLM", "Qwen2Model"),
        # Hard-coded in Qwen2Attention.__init__, not read from the config:
        # q/k/v always have a bias, o_proj never does.
        qkv_bias=BIAS_ALWAYS_ON,
        o_proj_bias=BIAS_ALWAYS_OFF,
        mlp_bias=BIAS_ALWAYS_OFF,
        qk_norm=False,
        head_dim_must_be_explicit=False,
        honours_config_sliding_window=True,
    ),
    TextFamily(
        name=FAMILY_LLAMA,
        model_types=("llama",),
        # VLlama3ForCausalLM is SmolVLM's own name for the nested text config;
        # Idefics3Model builds it through AutoModel.from_config, which
        # dispatches on model_type == "llama" and ignores `architectures`.
        architectures=("LlamaForCausalLM", "LlamaModel", "VLlama3ForCausalLM"),
        qkv_bias="config:attention_bias",
        o_proj_bias="config:attention_bias",
        mlp_bias="config:mlp_bias",
        qk_norm=False,
        head_dim_must_be_explicit=False,
        honours_config_sliding_window=False,
    ),
)

FAMILY_BY_MODEL_TYPE: dict[str, TextFamily] = {
    model_type: family for family in TEXT_FAMILIES for model_type in family.model_types
}

SUPPORTED_MODEL_TYPES: tuple[str, ...] = tuple(sorted(FAMILY_BY_MODEL_TYPE))


def family_for(hf_config: Mapping[str, Any], where: str = "config.json") -> TextFamily:
    """Pick the family by ``model_type``, then confirm ``architectures`` agrees.

    Checking both is what stops a config that says ``llama`` but declares
    ``Qwen3ForCausalLM`` from being silently decoded with the wrong biases.
    """
    model_type = _require(hf_config, "model_type", where)
    family = FAMILY_BY_MODEL_TYPE.get(model_type)
    if family is None:
        raise ConfigError(
            f"{where}: model_type={model_type!r} is not supported. SlowQ implements "
            f"{', '.join(SUPPORTED_MODEL_TYPES)} (plus {FAMILY_IDEFICS3!r} for "
            "image+text models, which is loaded through slowq.vlm)."
        )
    architectures = tuple(hf_config.get("architectures") or ())
    if architectures and not set(architectures) & set(family.architectures):
        raise ConfigError(
            f"{where}: architectures={list(architectures)!r} does not match "
            f"model_type={model_type!r}; that family builds "
            f"{list(family.architectures)!r}. Refusing to guess which forward "
            "pass this checkpoint wants."
        )
    return family


def _bias(rule: str, hf_config: Mapping[str, Any], where: str, what: str) -> bool:
    if rule == BIAS_ALWAYS_ON:
        return True
    if rule == BIAS_ALWAYS_OFF:
        return False
    key = rule.split(":", 1)[1]
    value = hf_config.get(key, False)
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: {key} (used for {what}) must be a bool, got {value!r}")
    return value


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where} is missing required key {key!r}")
    return mapping[key]


def _require_false(mapping: Mapping[str, Any], key: str, where: str, feature: str) -> None:
    value = mapping.get(key)
    if value not in (None, False, 0, 0.0):
        raise ConfigError(f"{where}: {key}={value!r}. SlowQ does not implement {feature}.")


#: Structural parameter values of the original v0.1 target model. Used by
#: ``slowq inspect`` and by the reference-validation tool to state what was
#: actually loaded; it is deliberately *not* used to reject other checkpoints,
#: since every computation is parameterised by the config that was read.
QWEN3_06B_REFERENCE: dict[str, Any] = {
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
}


@dataclass(frozen=True)
class ModelConfig:
    """Text-decoder parameters, validated against the family's actual rules."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    family: str = FAMILY_QWEN3
    task: str = TASK_TEXT_GENERATION
    #: True when q/k/v projections carry a bias vector.
    qkv_bias: bool = False
    #: True when the output projection carries a bias vector.
    o_proj_bias: bool = False
    #: True when gate/up/down projections carry a bias vector (llama only).
    mlp_bias: bool = False
    #: True when Q and K are RMSNormed per head before RoPE (qwen3 only).
    qk_norm: bool = True
    #: ``config.sliding_window`` when the family's mask actually uses it.
    sliding_window: int | None = None
    #: Prefix of this decoder's tensors in the converted directory. ``model`` for
    #: a standalone text model; SmolVLM nests its decoder under
    #: ``model.text_model`` beside the vision tower.
    tensor_prefix: str = "model"
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    # ---- derived quantities -------------------------------------------------

    @property
    def kv_group_size(self) -> int:
        """Number of query heads sharing one KV head (GQA group)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_proj_out(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_out(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attention_scaling(self) -> float:
        return self.head_dim**-0.5

    @property
    def is_generative(self) -> bool:
        return self.task in GENERATIVE_TASKS

    def kv_bytes_per_token(self) -> int:
        """Bytes of K+V cache for one token, across all layers, in FP32."""
        return (
            2
            * self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * 4
        )

    @classmethod
    def from_hf_config(
        cls,
        hf_config: Mapping[str, Any],
        task: str = TASK_TEXT_GENERATION,
        tensor_prefix: str = "model",
        where: str = "config.json",
    ) -> "ModelConfig":
        """Read one text decoder's parameters out of an official ``config.json``.

        ``tensor_prefix`` and ``where`` exist for the nested decoder of an
        image+text model, whose weights live under ``model.text_model`` and whose
        config lives at ``config.json``'s ``text_config`` key: both are facts
        about *where* the decoder sits, not about how it computes, so they are
        arguments rather than a second class.
        """

        if task not in KNOWN_TASKS:
            raise ConfigError(
                f"task={task!r} is not one of {list(KNOWN_TASKS)}"
            )
        # TASK_IMAGE_TEXT_GENERATION is accepted here because an image+text
        # model's decoder *is* an ordinary text decoder: ``slowq.vlm`` reads
        # ``config["text_config"]`` and builds this from it, then adds the vision
        # tower beside it. Handing this method the *top-level* idefics3 config
        # instead fails in family_for(), which names slowq.vlm as the way in.

        family = family_for(hf_config, where)

        hidden_act = _require(hf_config, "hidden_act", where)
        if hidden_act != SUPPORTED_HIDDEN_ACT:
            raise ConfigError(
                f"{where}: hidden_act={hidden_act!r}; only {SUPPORTED_HIDDEN_ACT!r} "
                "is implemented."
            )

        # Features that would silently change numerics if we ignored them.
        # ``sliding_window`` is handled below rather than rejected: qwen2 and
        # qwen3 do apply it, and for the checkpoints SlowQ targets the window is
        # at least as wide as the context, which makes full attention provably
        # identical. That is checked in validate_max_context, not assumed.
        _require_false(hf_config, "use_sliding_window", where, "layer-level sliding-window attention")
        _require_false(hf_config, "rope_scaling", where, "RoPE scaling")
        _require_false(hf_config, "rope_parameters", where, "RoPE scaling")
        _require_false(hf_config, "rope_interleaved", where, "interleaved RoPE")

        sliding_window: int | None = None
        if family.honours_config_sliding_window:
            raw_window = hf_config.get("sliding_window")
            if raw_window is not None:
                if not isinstance(raw_window, int) or isinstance(raw_window, bool) or raw_window < 1:
                    raise ConfigError(
                        f"{where}: sliding_window must be a positive int or null, "
                        f"got {raw_window!r}"
                    )
                sliding_window = raw_window

        head_dim = _resolve_head_dim(hf_config, family, where)

        int_keys = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
        )
        values: dict[str, int] = {}
        for key in int_keys:
            value = _require(hf_config, key, where)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigError(f"{where}: {key} must be a positive int, got {value!r}")
            values[key] = value

        n_heads, n_kv = values["num_attention_heads"], values["num_key_value_heads"]
        if n_heads % n_kv != 0:
            raise ConfigError(
                f"{where}: num_attention_heads ({n_heads}) must be divisible by "
                f"num_key_value_heads ({n_kv}) for GQA."
            )

        rms_norm_eps = _require(hf_config, "rms_norm_eps", where)
        rope_theta = _require(hf_config, "rope_theta", where)
        if not (isinstance(rms_norm_eps, (int, float)) and rms_norm_eps > 0):
            raise ConfigError(f"{where}: rms_norm_eps must be > 0, got {rms_norm_eps!r}")
        if not (isinstance(rope_theta, (int, float)) and rope_theta > 0):
            raise ConfigError(f"{where}: rope_theta must be > 0, got {rope_theta!r}")

        max_position_embeddings = hf_config.get("max_position_embeddings")
        if not isinstance(max_position_embeddings, int) or max_position_embeddings <= 0:
            raise ConfigError(
                f"{where}: max_position_embeddings must be a positive int, "
                f"got {max_position_embeddings!r}"
            )

        return cls(
            hidden_size=values["hidden_size"],
            intermediate_size=values["intermediate_size"],
            num_hidden_layers=values["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
            vocab_size=values["vocab_size"],
            rms_norm_eps=float(rms_norm_eps),
            rope_theta=float(rope_theta),
            tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
            max_position_embeddings=max_position_embeddings,
            family=family.name,
            task=task,
            qkv_bias=_bias(family.qkv_bias, hf_config, where, "q/k/v projection bias"),
            o_proj_bias=_bias(family.o_proj_bias, hf_config, where, "o projection bias"),
            mlp_bias=_bias(family.mlp_bias, hf_config, where, "MLP bias"),
            qk_norm=family.qk_norm,
            sliding_window=sliding_window,
            tensor_prefix=tensor_prefix,
            raw=dict(hf_config),
        )

    @classmethod
    def from_json_file(
        cls, path: str | Path, task: str = TASK_TEXT_GENERATION
    ) -> "ModelConfig":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"model config not found: {path}")
        return cls.from_hf_config(json.loads(path.read_text(encoding="utf-8")), task=task)

    def validate_max_context(self, max_context: int) -> int:
        if not isinstance(max_context, int) or isinstance(max_context, bool):
            raise ConfigError(f"max_context must be an int, got {max_context!r}")
        if max_context < 1:
            raise ConfigError(f"max_context must be >= 1, got {max_context}")
        if max_context > self.max_position_embeddings:
            raise ConfigError(
                f"max_context={max_context} exceeds max_position_embeddings="
                f"{self.max_position_embeddings}. SlowQ does not implement RoPE "
                "scaling or position extrapolation."
            )
        if self.sliding_window is not None and max_context > self.sliding_window:
            raise ConfigError(
                f"max_context={max_context} exceeds this checkpoint's "
                f"sliding_window={self.sliding_window}. The {self.family} model-level "
                "causal mask applies that window even though use_sliding_window is "
                "false, and SlowQ implements full attention only. Up to "
                f"{self.sliding_window} tokens the two are identical; beyond it "
                "SlowQ would keep attending to keys the reference has masked out."
            )
        return max_context

    def describe(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [
            ("family", self.family),
            ("task", self.task),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("kv_group_size (GQA)", self.kv_group_size),
            ("head_dim", self.head_dim),
            ("q_proj out_features", self.q_proj_out),
            ("kv_proj out_features", self.kv_proj_out),
            ("vocab_size", self.vocab_size),
            ("rms_norm_eps", self.rms_norm_eps),
            ("rope_theta", self.rope_theta),
            ("tie_word_embeddings", self.tie_word_embeddings),
            ("q/k/v bias", self.qkv_bias),
            ("o_proj bias", self.o_proj_bias),
            ("mlp bias", self.mlp_bias),
            ("q/k norm (per head)", self.qk_norm),
            ("sliding_window", self.sliding_window),
            ("max_position_embeddings", self.max_position_embeddings),
            ("runtime compute dtype", RUNTIME_DTYPE_NAME),
        ]
        return rows


def _resolve_head_dim(
    hf_config: Mapping[str, Any], family: TextFamily, where: str
) -> int:
    """Read ``head_dim``, or derive it the way this family's modelling code does.

    ``transformers`` 4.51.3 uses ``getattr(config, "head_dim", hidden_size //
    num_attention_heads)`` in all three families, so the fallback is official —
    but only for families where released checkpoints actually satisfy it. Qwen3
    sets ``head_dim`` independently of ``hidden_size`` (0.6B: 16 heads x 128 =
    2048 != hidden_size 1024), so guessing there would produce a model that
    loads, runs, and is quietly wrong.
    """
    hidden_size = _require(hf_config, "hidden_size", where)
    n_heads = _require(hf_config, "num_attention_heads", where)
    if "head_dim" in hf_config:
        head_dim = hf_config["head_dim"]
        if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
            raise ConfigError(f"{where}: head_dim must be a positive int, got {head_dim!r}")
        return head_dim
    if family.head_dim_must_be_explicit:
        raise ConfigError(
            f"{where}: 'head_dim' must be present for model_type={family.name!r}. "
            f"transformers would fall back to hidden_size // num_attention_heads = "
            f"{hidden_size // n_heads}, but Qwen3 sets head_dim independently of "
            "hidden_size (0.6B: 16 x 128 = 2048 != 1024), so that fallback is "
            "wrong for every released Qwen3 checkpoint. SlowQ refuses to guess."
        )
    if not isinstance(hidden_size, int) or not isinstance(n_heads, int) or n_heads <= 0:
        raise ConfigError(
            f"{where}: hidden_size and num_attention_heads must be positive ints "
            f"to derive head_dim, got {hidden_size!r} and {n_heads!r}"
        )
    if hidden_size % n_heads != 0:
        raise ConfigError(
            f"{where}: no explicit head_dim, and hidden_size ({hidden_size}) is not "
            f"divisible by num_attention_heads ({n_heads}), so this {family.name} "
            "config does not determine a head_dim."
        )
    return hidden_size // n_heads


@dataclass(frozen=True)
class GenerationConfig:
    """Stop tokens and the repository's published sampling defaults."""

    eos_token_ids: tuple[int, ...]
    pad_token_id: int | None
    bos_token_id: int | None
    temperature: float
    top_k: int
    top_p: float
    do_sample: bool
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_hf_config(
        cls,
        generation_config: Mapping[str, Any] | None,
        model_config: Mapping[str, Any] | None = None,
    ) -> "GenerationConfig":
        source = dict(generation_config or {})
        # HF falls back to config.json for these two when generation_config.json
        # omits them; mirror that instead of guessing.
        fallback = dict(model_config or {})

        eos = source.get("eos_token_id", fallback.get("eos_token_id"))
        if eos is None:
            raise ConfigError(
                "no eos_token_id in generation_config.json or config.json; SlowQ "
                "will not guess a stop token."
            )
        eos_ids = tuple(int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]))

        def _int_or_none(key: str) -> int | None:
            value = source.get(key, fallback.get(key))
            return None if value is None else int(value)

        return cls(
            eos_token_ids=eos_ids,
            pad_token_id=_int_or_none("pad_token_id"),
            bos_token_id=_int_or_none("bos_token_id"),
            temperature=float(source.get("temperature", 1.0)),
            top_k=int(source.get("top_k", 0)),
            top_p=float(source.get("top_p", 1.0)),
            do_sample=bool(source.get("do_sample", False)),
            raw=source,
        )

    @classmethod
    def from_json_file(
        cls, path: str | Path, model_config_path: str | Path | None = None
    ) -> "GenerationConfig":
        path = Path(path)
        generation = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
        model = (
            json.loads(Path(model_config_path).read_text(encoding="utf-8"))
            if model_config_path and Path(model_config_path).is_file()
            else None
        )
        return cls.from_hf_config(generation, model)
