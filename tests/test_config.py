"""Config parsing, per-family rules, and the support-range checks."""

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


def test_rejects_an_unknown_model_type(hf_config: dict):
    hf_config["model_type"] = "mistral"
    hf_config["architectures"] = ["MistralForCausalLM"]
    with pytest.raises(ConfigError, match="not supported"):
        ModelConfig.from_hf_config(hf_config)


def test_rejects_an_architecture_that_disagrees_with_the_model_type(hf_config: dict):
    """A qwen2 model_type wearing Qwen3's architecture name is a red flag, not a hint."""
    hf_config["model_type"] = "qwen2"
    with pytest.raises(ConfigError, match="does not match model_type"):
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
        ("use_sliding_window", True),
        ("rope_scaling", {"type": "linear", "factor": 2.0}),
        ("rope_parameters", {"rope_theta": 5000000.0}),
        ("rope_interleaved", True),
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
    config = ModelConfig.from_hf_config(hf_config)
    assert config.qkv_bias is False
    assert config.o_proj_bias is False
    assert config.sliding_window is None


# --------------------------------------------------------------------------- #
# per-family bias / norm / head_dim rules
# --------------------------------------------------------------------------- #


def test_qwen3_reads_its_attention_bias_from_the_config(hf_config: dict):
    """Qwen3Attention passes ``bias=config.attention_bias`` to all four Linears."""
    hf_config["attention_bias"] = True
    config = ModelConfig.from_hf_config(hf_config)
    assert config.family == "qwen3"
    assert config.qkv_bias is True
    assert config.o_proj_bias is True
    assert config.mlp_bias is False, "Qwen3MLP builds its Linears without a bias"
    assert config.qk_norm is True


def test_qwen3_refuses_to_derive_head_dim(hf_config: dict):
    """Qwen3 sets head_dim independently of hidden_size, so the fallback is wrong."""
    del hf_config["head_dim"]
    with pytest.raises(ConfigError, match="refuses to guess"):
        ModelConfig.from_hf_config(hf_config)


def qwen2_config(**overrides) -> dict:
    base = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 49408,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "use_sliding_window": False,
        "sliding_window": 32768,
    }
    base.update(overrides)
    return base


def llama_config(**overrides) -> dict:
    base = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "vocab_size": 49152,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 8192,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
    }
    base.update(overrides)
    return base


def test_qwen2_always_biases_qkv_and_never_biases_o_proj():
    """Qwen2Attention hard-codes ``bias=True`` for q/k/v and ``bias=False`` for o."""
    config = ModelConfig.from_hf_config(qwen2_config())
    assert config.family == "qwen2"
    assert config.qkv_bias is True
    assert config.o_proj_bias is False
    assert config.mlp_bias is False
    assert config.qk_norm is False, "qwen2 has no per-head Q/K RMSNorm"


def test_qwen2_ignores_an_attention_bias_key_that_would_change_nothing():
    """The config key does not exist for qwen2; a stray value must not flip anything."""
    config = ModelConfig.from_hf_config(qwen2_config(attention_bias=False))
    assert config.qkv_bias is True
    assert config.o_proj_bias is False


def test_qwen2_derives_head_dim_when_the_config_omits_it():
    config = ModelConfig.from_hf_config(qwen2_config())
    assert "head_dim" not in qwen2_config()
    assert config.head_dim == 896 // 14 == 64


def test_qwen2_keeps_a_sliding_window_the_mask_actually_applies():
    """Qwen2Model builds its causal mask from ``config.sliding_window`` whenever it
    is not None — independent of ``use_sliding_window``. Refusing the config would
    reject every Qwen2.5 checkpoint; ignoring the window would be wrong past it."""
    config = ModelConfig.from_hf_config(qwen2_config())
    assert config.sliding_window == 32768
    assert config.validate_max_context(32768) == 32768


def test_a_context_past_the_sliding_window_is_refused_even_though_it_fits_the_rope():
    """Qwen2.5 publishes sliding_window == max_position_embeddings, so the two
    checks coincide there. A narrower window must still be the binding one."""
    config = ModelConfig.from_hf_config(qwen2_config(sliding_window=1024))
    assert config.max_position_embeddings == 32768
    assert config.validate_max_context(1024) == 1024
    with pytest.raises(ConfigError, match="sliding_window"):
        config.validate_max_context(1025)


def test_llama_reads_both_bias_flags_from_the_config():
    config = ModelConfig.from_hf_config(llama_config())
    assert config.family == "llama"
    assert config.qkv_bias is False
    assert config.o_proj_bias is False
    assert config.mlp_bias is False
    assert config.qk_norm is False

    biased = ModelConfig.from_hf_config(llama_config(attention_bias=True, mlp_bias=True))
    assert biased.qkv_bias is True
    assert biased.o_proj_bias is True
    assert biased.mlp_bias is True


def test_llama_has_no_sliding_window_to_honour():
    """LlamaModel's mask never reads ``sliding_window``, so storing one would make
    validate_max_context reject a context the reference handles fine."""
    config = ModelConfig.from_hf_config(llama_config(sliding_window=64))
    assert config.sliding_window is None
    assert config.validate_max_context(8192) == 8192


def test_llama_derives_head_dim_and_checks_divisibility():
    assert ModelConfig.from_hf_config(llama_config()).head_dim == 576 // 9 == 64
    with pytest.raises(ConfigError, match="divisible"):
        ModelConfig.from_hf_config(llama_config(hidden_size=577))


def test_llama_keeps_an_explicit_head_dim():
    config = ModelConfig.from_hf_config(llama_config(head_dim=128))
    assert config.head_dim == 128
    assert config.q_proj_out == 9 * 128


def test_a_nested_decoder_is_accepted_for_the_image_text_task():
    """SmolVLM's ``text_config`` is an ordinary llama decoder, so the image+text
    task must not be rejected here: ``slowq.vlm`` hands this method the nested
    config and adds the vision tower beside it."""
    config = ModelConfig.from_hf_config(
        llama_config(),
        task="image-text-generation",
        tensor_prefix="model.text_model",
        where="config.json:text_config",
    )
    assert config.task == "image-text-generation"
    assert config.family == "llama"
    assert config.tensor_prefix == "model.text_model"


def test_the_top_level_idefics3_config_is_not_a_text_decoder():
    """What *is* refused: the wrapper config, whose ``model_type`` names no
    decoder family. Naming ``slowq.vlm`` in the message is the point."""
    wrapper = {
        "model_type": "idefics3",
        "architectures": ["Idefics3ForConditionalGeneration"],
        "image_token_id": 49190,
        "scale_factor": 4,
        "text_config": llama_config(),
        "vision_config": {"hidden_size": 768},
    }
    with pytest.raises(ConfigError, match="slowq.vlm"):
        ModelConfig.from_hf_config(wrapper, task="image-text-generation")


def test_the_embedding_task_is_recorded_on_the_config():
    config = ModelConfig.from_hf_config(qwen2_config(), task="text-embedding")
    assert config.task == "text-embedding"
    assert config.is_generative is False


def test_an_unknown_task_is_rejected():
    with pytest.raises(ConfigError, match="not one of"):
        ModelConfig.from_hf_config(qwen2_config(), task="text-classification")


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
