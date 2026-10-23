"""Weight loading: manifest validation, mmap, shape checks, tied embeddings."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from slowq.config import ModelConfig
from slowq.errors import WeightError
from slowq.weights import (
    EMBEDDING_TENSOR_NAME,
    TIED_OUTPUT_EMBEDDING_ALIAS,
    Weights,
    required_tensors,
)

VOCAB = 64


@pytest.fixture
def model_dir(tmp_path: Path, write_model) -> Path:
    return write_model(tmp_path / "weights", vocab_size=VOCAB)


@pytest.fixture
def config(model_dir: Path) -> ModelConfig:
    return ModelConfig.from_json_file(model_dir / "config.json")


def read_manifest(model_dir: Path) -> dict:
    return json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(model_dir: Path, manifest: dict) -> None:
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# required_tensors
# --------------------------------------------------------------------------- #


def test_required_tensors_covers_every_projection(config: ModelConfig):
    expected = required_tensors(config)
    per_layer = 11
    assert len(expected) == 2 + per_layer * config.num_hidden_layers
    assert expected[EMBEDDING_TENSOR_NAME] == (config.vocab_size, config.hidden_size)
    layer0 = "model.layers.0"
    assert expected[f"{layer0}.self_attn.q_proj.weight"] == (
        config.num_attention_heads * config.head_dim,
        config.hidden_size,
    )
    assert expected[f"{layer0}.self_attn.o_proj.weight"] == (
        config.hidden_size,
        config.num_attention_heads * config.head_dim,
    )
    assert expected[f"{layer0}.self_attn.q_norm.weight"] == (config.head_dim,)
    assert expected[f"{layer0}.mlp.down_proj.weight"] == (
        config.hidden_size,
        config.intermediate_size,
    )


def test_required_tensors_omits_a_tied_lm_head(config: ModelConfig):
    assert config.tie_word_embeddings
    assert TIED_OUTPUT_EMBEDDING_ALIAS not in required_tensors(config)


def test_required_tensors_includes_lm_head_when_not_tied(config: ModelConfig):
    untied = replace(config, tie_word_embeddings=False)
    assert TIED_OUTPUT_EMBEDDING_ALIAS in required_tensors(untied)


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #


def test_check_complete_passes_for_a_good_conversion(model_dir: Path, config: ModelConfig):
    Weights.from_directory(model_dir).check_complete(config)


def test_arrays_are_read_only_memory_maps(model_dir: Path):
    array = Weights.from_directory(model_dir).get(EMBEDDING_TENSOR_NAME)
    assert isinstance(array, np.memmap)
    assert array.flags.writeable is False
    with pytest.raises((ValueError, TypeError)):
        array[0, 0] = 1.0


def test_repeated_get_returns_the_same_mapping(model_dir: Path):
    weights = Weights.from_directory(model_dir)
    assert weights.get("model.norm.weight") is weights.get("model.norm.weight")


def test_get_matches_what_was_written(model_dir: Path, random_tensors):
    written = random_tensors(vocab_size=VOCAB)
    weights = Weights.from_directory(model_dir)
    for name, expected in written.items():
        np.testing.assert_array_equal(weights.get(name), expected)


def test_dtype_is_float32_everywhere(model_dir: Path):
    weights = Weights.from_directory(model_dir)
    assert weights.dtype_names == ("float32",)
    assert all(weights.get(name).dtype == np.float32 for name in weights.tensor_names)


# --------------------------------------------------------------------------- #
# tied embeddings
# --------------------------------------------------------------------------- #


def test_the_tied_lm_head_resolves_to_the_embedding(model_dir: Path):
    weights = Weights.from_directory(model_dir)
    assert weights.canonical_name(TIED_OUTPUT_EMBEDDING_ALIAS) == EMBEDDING_TENSOR_NAME
    assert weights.get(TIED_OUTPUT_EMBEDDING_ALIAS) is weights.get(EMBEDDING_TENSOR_NAME)


def test_the_tied_weight_is_stored_once(model_dir: Path):
    weights = Weights.from_directory(model_dir)
    files = [spec.file for spec in weights.iter_specs()]
    assert len(files) == len(set(files))  # no duplicate storage
    assert TIED_OUTPUT_EMBEDDING_ALIAS not in weights.tensor_names
    assert weights.aliases == {TIED_OUTPUT_EMBEDDING_ALIAS: EMBEDDING_TENSOR_NAME}


def test_deduplicated_bytes_counts_shared_weights_once(model_dir: Path, config: ModelConfig):
    weights = Weights.from_directory(model_dir)
    total = sum(int(np.prod(shape)) for shape in required_tensors(config).values())
    assert weights.deduplicated_bytes() == total * np.dtype(np.float32).itemsize


def test_npy_file_bytes_exceed_the_logical_size_because_of_headers(model_dir: Path):
    weights = Weights.from_directory(model_dir)
    assert weights.file_bytes() > weights.deduplicated_bytes()


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #


def test_missing_directory(tmp_path: Path):
    with pytest.raises(WeightError, match="model directory not found"):
        Weights.from_directory(tmp_path / "nope")


def test_missing_manifest(tmp_path: Path):
    (tmp_path / "bare").mkdir()
    with pytest.raises(WeightError, match="manifest.json"):
        Weights.from_directory(tmp_path / "bare")


def test_manifest_that_is_not_json(model_dir: Path):
    (model_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(WeightError, match="not valid JSON"):
        Weights.from_directory(model_dir)


def test_unsupported_manifest_format_version(model_dir: Path):
    manifest = read_manifest(model_dir)
    manifest["format_version"] = 99
    write_manifest(model_dir, manifest)
    with pytest.raises(WeightError, match="format_version"):
        Weights.from_directory(model_dir)


def test_manifest_declaring_a_different_compute_dtype(model_dir: Path):
    manifest = read_manifest(model_dir)
    manifest["compute_dtype"] = "bfloat16"
    write_manifest(model_dir, manifest)
    with pytest.raises(WeightError, match="only runs FP32"):
        Weights.from_directory(model_dir)


def test_manifest_without_a_source_is_rejected(model_dir: Path):
    manifest = read_manifest(model_dir)
    del manifest["source"]
    write_manifest(model_dir, manifest)
    with pytest.raises(WeightError, match="source"):
        Weights.from_directory(model_dir)


def test_manifest_listing_a_tensor_twice_is_rejected(model_dir: Path):
    manifest = read_manifest(model_dir)
    manifest["tensors"].append(dict(manifest["tensors"][0]))
    write_manifest(model_dir, manifest)
    with pytest.raises(WeightError, match="twice"):
        Weights.from_directory(model_dir)


def test_alias_pointing_at_an_unknown_tensor_is_rejected(model_dir: Path):
    manifest = read_manifest(model_dir)
    manifest["aliases"]["lm_head.weight"] = "model.does_not_exist.weight"
    write_manifest(model_dir, manifest)
    with pytest.raises(WeightError, match="unknown tensor"):
        Weights.from_directory(model_dir)


def test_manifest_entry_missing_a_key_is_rejected(model_dir: Path):
    manifest = read_manifest(model_dir)
    del manifest["tensors"][0]["shape"]
    write_manifest(model_dir, manifest)
    with pytest.raises(WeightError, match="lacks 'shape'"):
        Weights.from_directory(model_dir)


def test_missing_tensor_file(model_dir: Path):
    (model_dir / "tensors" / "model.norm.weight.npy").unlink()
    weights = Weights.from_directory(model_dir)
    with pytest.raises(WeightError, match="not found"):
        weights.get("model.norm.weight")


def test_unknown_tensor_name(model_dir: Path):
    weights = Weights.from_directory(model_dir)
    with pytest.raises(WeightError, match="missing from"):
        weights.get("model.layers.0.self_attn.q_proj.bias")


def test_check_complete_reports_a_missing_tensor(model_dir: Path, config: ModelConfig):
    manifest = read_manifest(model_dir)
    manifest["tensors"] = [
        entry
        for entry in manifest["tensors"]
        if entry["name"] != "model.layers.1.mlp.up_proj.weight"
    ]
    write_manifest(model_dir, manifest)
    with pytest.raises(WeightError, match="missing model.layers.1.mlp.up_proj.weight"):
        Weights.from_directory(model_dir).check_complete(config)


def test_check_complete_reports_a_shape_disagreement(model_dir: Path, config: ModelConfig):
    """A manifest shape that contradicts the config fails before any forward."""
    manifest = read_manifest(model_dir)
    for entry in manifest["tensors"]:
        if entry["name"] == "model.layers.0.self_attn.q_proj.weight":
            entry["shape"] = [config.hidden_size, config.hidden_size]
    write_manifest(model_dir, manifest)
    with pytest.raises(WeightError, match="manifest shape"):
        Weights.from_directory(model_dir).check_complete(config)


def test_disk_shape_contradicting_the_manifest(model_dir: Path):
    """The manifest is not trusted blindly: the file on disk is checked too."""
    path = model_dir / "tensors" / "model.norm.weight.npy"
    np.save(path, np.zeros(3, dtype=np.float32))
    with pytest.raises(WeightError, match="has shape"):
        Weights.from_directory(model_dir).get("model.norm.weight")


def test_a_non_float32_tensor_on_disk_is_rejected(model_dir: Path):
    path = model_dir / "tensors" / "model.norm.weight.npy"
    np.save(path, np.zeros(16, dtype=np.float64))
    with pytest.raises(WeightError, match="expected float32"):
        Weights.from_directory(model_dir).get("model.norm.weight")


def test_pickle_based_npy_files_are_refused(model_dir: Path):
    """``allow_pickle=False``: a checkpoint must never be able to run code."""
    path = model_dir / "tensors" / "model.norm.weight.npy"
    np.save(path, np.array([object()], dtype=object), allow_pickle=True)
    with pytest.raises(WeightError, match="memory-map|pickle"):
        Weights.from_directory(model_dir).get("model.norm.weight")


def test_expect_rejects_a_shape_that_disagrees_with_the_config(model_dir: Path):
    weights = Weights.from_directory(model_dir)
    with pytest.raises(WeightError, match="expected"):
        weights.expect("model.norm.weight", (999,))


def test_a_broken_conversion_fails_before_the_first_forward(model_dir: Path):
    """CausalLMModel validates completeness at construction, not 20 layers in."""
    from slowq.model import CausalLMModel

    (model_dir / "tensors" / "model.layers.1.mlp.down_proj.weight.npy").unlink()
    manifest = read_manifest(model_dir)
    manifest["tensors"] = [
        entry for entry in manifest["tensors"] if entry["name"] != "model.layers.1.mlp.down_proj.weight"
    ]
    write_manifest(model_dir, manifest)
    config = ModelConfig.from_json_file(model_dir / "config.json")
    with pytest.raises(WeightError):
        CausalLMModel(config, Weights.from_directory(model_dir))
