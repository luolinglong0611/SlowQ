"""Config parsing and the v0.1 support-range checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slowq.config import (
    QWEN3_06B_REFERENCE,
    GenerationConfig,
    ModelConfig,
)
from slowq.errors import ConfigError


@pytest.fixture
def hf_config(tiny_model_dir: Path) -> dict:
    return json.loads((tiny_model_dir / "config.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #


def test_tiny_config_parses_with_the_right_derived_dimensions(hf_config: dict):
    config = ModelConfig.from_hf_config(hf_config)
    assert config.hidden_size == 16
    assert config.kv_group_size == 2
    assert config.q_proj_out == config.num_attention_heads * config.head_dim == 32
    assert config.kv_proj_out == config.num_key_value_heads * config.head_dim == 16
    assert config.tie_word_embeddings is True


def test_head_dim_is_read_not_derived_from_hidden_size(hf_config: dict):
    """The tiny config keeps Qwen3's property: head_dim * n_heads != hidden_size."""
    assert hf_config["head_dim"] * hf_config["num_attention_heads"] != hf_config["hidden_size"]
    config = ModelConfig.from_hf_config(hf_config)
    assert config.head_dim == hf_config["head_dim"]


def test_attention_scaling_is_one_over_sqrt_head_dim(hf_config: dict):
    config = ModelConfig.from_hf_config(hf_config)
    assert config.attention_scaling == pytest.approx(config.head_dim**-0.5)


def test_kv_bytes_per_token_matches_the_cache_allocation(tiny_model_dir: Path, tiny_config):
    from slowq.cache import KVCache

    max_context = 16
    cache = KVCache(
        max_context=max_context,
        num_layers=tiny_config.num_hidden_layers,
        num_kv_heads=tiny_config.num_key_value_heads,
        head_dim=tiny_config.head_dim,
    )
    assert cache.allocated_bytes() == tiny_config.kv_bytes_per_token() * max_context


def test_qwen3_06b_reference_table_is_consistent():
    """Sanity check on the documented 0.6B numbers (informational only)."""
    ref = QWEN3_06B_REFERENCE
    assert ref["num_attention_heads"] // ref["num_key_value_heads"] == 2
    assert ref["num_attention_heads"] * ref["head_dim"] != ref["hidden_size"]
    assert ref["num_attention_heads"] * ref["head_dim"] == 2048


def test_describe_lists_the_structural_parameters(hf_config: dict):
    described = dict(ModelConfig.from_hf_config(hf_config).describe())
    for key in ("hidden_size", "head_dim", "num_key_value_heads", "vocab_size"):
        assert key in described
    assert described["runtime compute dtype"] == "float32"


def test_from_json_file_reads_config_json(tiny_model_dir: Path):
    config = ModelConfig.from_json_file(tiny_model_dir / "config.json")
    assert config.num_hidden_layers == 2


def test_from_json_file_reports_a_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        ModelConfig.from_json_file(tmp_path / "config.json")


# --------------------------------------------------------------------------- #
# support range
# --------------------------------------------------------------------------- #


def test_rejects_a_missing_head_dim(hf_config: dict):
    del hf_config["head_dim"]
    with pytest.raises(ConfigError, match="head_dim"):
        ModelConfig.from_hf_config(hf_config)


def test_rejects_another_model_type(hf_config: dict):
    hf_config["model_type"] = "qwen2"
    with pytest.raises(ConfigError, match="only supports 'qwen3'"):
        ModelConfig.from_hf_config(hf_config)


def test_rejects_another_architecture(hf_config: dict):
    hf_config["architectures"] = ["Qwen2ForCausalLM"]
    with pytest.raises(ConfigError, match="Qwen3ForCausalLM"):
        ModelConfig.from_hf_config(hf_config)


def test_rejects_a_different_activation(hf_config: dict):
    hf_config["hidden_act"] = "gelu"
    with pytest.raises(ConfigError, match="hidden_act"):
        ModelConfig.from_hf_config(hf_config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("attention_bias", True),
        ("use_sliding_window", True),
        ("sliding_window", 128),
        ("rope_scaling", {"type": "linear", "factor": 2.0}),
        ("rope_parameters", {"rope_theta": 5000000.0}),
    ],
)
def test_rejects_unimplemented_features_that_would_change_numerics(
    hf_config: dict, key: str, value
):
    hf_config[key] = value
    with pytest.raises(ConfigError, match="does not implement"):
        ModelConfig.from_hf_config(hf_config)


def test_accepts_explicitly_disabled_optional_features(hf_config: dict):
    hf_config.update(attention_bias=False, use_sliding_window=False, sliding_window=None)
    assert ModelConfig.from_hf_config(hf_config).attention_bias is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("hidden_size", 0),
        ("hidden_size", -1),
        ("hidden_size", 16.0),
        ("num_hidden_layers", "2"),
        ("num_attention_heads", True),
    ],
)
def test_rejects_non_positive_or_non_int_dimensions(hf_config: dict, key: str, value):
    hf_config[key] = value
    with pytest.raises(ConfigError, match="positive int"):
        ModelConfig.from_hf_config(hf_config)


def test_rejects_a_missing_required_key(hf_config: dict):
    del hf_config["rope_theta"]
    with pytest.raises(ConfigError, match="missing required key 'rope_theta'"):
        ModelConfig.from_hf_config(hf_config)


def test_rejects_gqa_head_counts_that_do_not_divide(hf_config: dict):
    hf_config["num_key_value_heads"] = 3
    with pytest.raises(ConfigError, match="divisible"):
        ModelConfig.from_hf_config(hf_config)


@pytest.mark.parametrize("eps", [0, -1e-6, "1e-6"])
def test_rejects_a_bad_rms_norm_eps(hf_config: dict, eps):
    hf_config["rms_norm_eps"] = eps
    with pytest.raises(ConfigError, match="rms_norm_eps"):
        ModelConfig.from_hf_config(hf_config)


@pytest.mark.parametrize("theta", [0, -1.0, None])
def test_rejects_a_bad_rope_theta(hf_config: dict, theta):
    hf_config["rope_theta"] = theta
    with pytest.raises(ConfigError, match="rope_theta"):
        ModelConfig.from_hf_config(hf_config)


@pytest.mark.parametrize("positions", [None, 0, -8, "64"])
def test_rejects_a_bad_max_position_embeddings(hf_config: dict, positions):
    hf_config["max_position_embeddings"] = positions
    with pytest.raises(ConfigError, match="max_position_embeddings"):
        ModelConfig.from_hf_config(hf_config)


# --------------------------------------------------------------------------- #
# max_context
# --------------------------------------------------------------------------- #


def test_validate_max_context_accepts_a_smaller_capacity(hf_config: dict):
    config = ModelConfig.from_hf_config(hf_config)
    assert config.validate_max_context(8) == 8


def test_validate_max_context_rejects_extrapolation(hf_config: dict):
    config = ModelConfig.from_hf_config(hf_config)
    with pytest.raises(ConfigError, match="max_position_embeddings"):
        config.validate_max_context(hf_config["max_position_embeddings"] + 1)


@pytest.mark.parametrize("value", [0, -4, 8.0, "8", True])
def test_validate_max_context_rejects_non_positive_or_non_int(hf_config: dict, value):
    config = ModelConfig.from_hf_config(hf_config)
    with pytest.raises(ConfigError):
        config.validate_max_context(value)


# --------------------------------------------------------------------------- #
# generation config
# --------------------------------------------------------------------------- #


def test_generation_config_reads_the_official_stop_set(tiny_model_dir: Path):
    generation = GenerationConfig.from_json_file(
        tiny_model_dir / "generation_config.json", tiny_model_dir / "config.json"
    )
    assert len(generation.eos_token_ids) == 2
    assert generation.temperature == pytest.approx(0.7)
    assert generation.top_k == 20
    assert generation.top_p == pytest.approx(0.8)


def test_generation_config_falls_back_to_config_json_for_eos():
    generation = GenerationConfig.from_hf_config(
        {"temperature": 1.0}, {"eos_token_id": 151645, "pad_token_id": 151643}
    )
    assert generation.eos_token_ids == (151645,)
    assert generation.pad_token_id == 151643


def test_generation_config_accepts_a_list_of_eos_ids():
    generation = GenerationConfig.from_hf_config({"eos_token_id": [151645, 151643]})
    assert generation.eos_token_ids == (151645, 151643)


def test_generation_config_refuses_to_guess_a_stop_token():
    with pytest.raises(ConfigError, match="will not guess"):
        GenerationConfig.from_hf_config({"temperature": 0.7}, {"hidden_size": 16})


def test_generation_config_uses_hf_defaults_when_unspecified():
    generation = GenerationConfig.from_hf_config({"eos_token_id": 2})
    assert generation.temperature == 1.0
    assert generation.top_k == 0
    assert generation.top_p == 1.0
    assert generation.do_sample is False
    assert generation.bos_token_id is None
