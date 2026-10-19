"""Model configuration, generation configuration, and v0.1 support-range checks.

Everything numeric the model needs is read from the official ``config.json``;
nothing is derived from a hard-coded 0.6B constant. In particular ``head_dim``
is read directly, because Qwen3 uses ``head_dim * num_attention_heads !=
hidden_size`` (16 * 128 = 2048 != 1024).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError

#: Structural parameter values of the v0.1 target model. Used by ``slowq inspect``
#: and by the reference-validation tool to state what was actually loaded; it is
#: deliberately *not* used to reject other Qwen3 sizes, since every computation
#: below is parameterised by the config that was read.
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

SUPPORTED_MODEL_TYPE = "qwen3"
SUPPORTED_ARCHITECTURE = "Qwen3ForCausalLM"
SUPPORTED_HIDDEN_ACT = "silu"
RUNTIME_DTYPE_NAME = "float32"


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where} is missing required key {key!r}")
    return mapping[key]


def _require_false(mapping: Mapping[str, Any], key: str, where: str, feature: str) -> None:
    value = mapping.get(key)
    if value not in (None, False, 0, 0.0):
        raise ConfigError(
            f"{where}: {key}={value!r}. SlowQ v0.1 does not implement {feature}."
        )


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters validated against the v0.1 support range."""

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
    attention_bias: bool = False
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
    def from_hf_config(cls, hf_config: Mapping[str, Any]) -> "ModelConfig":
        where = "config.json"

        model_type = _require(hf_config, "model_type", where)
        if model_type != SUPPORTED_MODEL_TYPE:
            raise ConfigError(
                f"{where}: model_type={model_type!r}, but SlowQ v0.1 only "
                f"supports {SUPPORTED_MODEL_TYPE!r}."
            )

        architectures = hf_config.get("architectures") or []
        if SUPPORTED_ARCHITECTURE not in architectures:
            raise ConfigError(
                f"{where}: architectures={list(architectures)!r} does not contain "
                f"{SUPPORTED_ARCHITECTURE!r}."
            )

        hidden_act = _require(hf_config, "hidden_act", where)
        if hidden_act != SUPPORTED_HIDDEN_ACT:
            raise ConfigError(
                f"{where}: hidden_act={hidden_act!r}; only {SUPPORTED_HIDDEN_ACT!r} "
                "is implemented."
            )

        # Features that would silently change numerics if we ignored them.
        _require_false(hf_config, "attention_bias", where, "attention bias")
        _require_false(hf_config, "use_sliding_window", where, "sliding-window attention")
        _require_false(hf_config, "sliding_window", where, "sliding-window attention")
        _require_false(hf_config, "rope_scaling", where, "RoPE scaling")
        _require_false(hf_config, "rope_parameters", where, "RoPE scaling")

        if "head_dim" not in hf_config:
            raise ConfigError(
                f"{where}: 'head_dim' must be present. SlowQ refuses to infer it "
                "from hidden_size / num_attention_heads because Qwen3 sets it "
                "independently (16 heads x 128 head_dim != hidden_size 1024)."
            )

        int_keys = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
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
            head_dim=values["head_dim"],
            vocab_size=values["vocab_size"],
            rms_norm_eps=float(rms_norm_eps),
            rope_theta=float(rope_theta),
            tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
            max_position_embeddings=max_position_embeddings,
            attention_bias=False,
            raw=dict(hf_config),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ModelConfig":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"model config not found: {path}")
        return cls.from_hf_config(json.loads(path.read_text(encoding="utf-8")))

    def validate_max_context(self, max_context: int) -> int:
        if not isinstance(max_context, int) or isinstance(max_context, bool):
            raise ConfigError(f"max_context must be an int, got {max_context!r}")
        if max_context < 1:
            raise ConfigError(f"max_context must be >= 1, got {max_context}")
        if max_context > self.max_position_embeddings:
            raise ConfigError(
                f"max_context={max_context} exceeds max_position_embeddings="
                f"{self.max_position_embeddings}. SlowQ v0.1 does not implement "
                "RoPE scaling or position extrapolation."
            )
        return max_context

    def describe(self) -> list[tuple[str, Any]]:
        return [
            ("model_type", SUPPORTED_MODEL_TYPE),
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
            ("attention_bias", self.attention_bias),
            ("max_position_embeddings", self.max_position_embeddings),
            ("runtime compute dtype", RUNTIME_DTYPE_NAME),
        ]


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
