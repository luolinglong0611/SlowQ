"""Real-weight integration tests for the ``image-text-generation`` task.

Enable them by naming a converted image+text directory::

    SLOWQ_MODEL_DIR=models/SmolVLM-256M-Instruct-slowq python -m pytest -m integration

``tests/test_vlm.py`` covers the geometry, the preprocessing arithmetic and the
merge against a tiny synthetic checkpoint, where a 4x4 "image" is enough to
expose an off-by-one. What only the real checkpoint can show is that the whole
chain survives a real photograph's worth of pixels:

* the preprocessor reproduces the *official processor's* pixel tensor bit for
  bit, on the two recorded crop grids (1x4 and 2x4, 5 and 9 frames);
* the expanded prompt is the recorded one, token for token, with the image
  placeholders exactly where the capture says they are;
* the vision tower's attention really is bidirectional on a real frame — patch 0
  changes when the *last* patch's pixels change;
* greedy generation reproduces the recorded token sequence, and does *not*
  reproduce it when the image features are withheld from the very same prompt;
* the converted directory is self-sufficient: no network, and both towers' plus
  the processor's metadata on disk.

The numeric agreement with the official implementation, segment by segment, is
graded by ``test_integration_reference.py``; this module is about the invariants
around it, and about the parts a numeric tolerance would not catch.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest

import real_models
from slowq.embedding import EmbeddingEngine
from slowq.engine import Engine
from slowq.errors import ConfigError, ContextOverflowError, UnsupportedInputError
from slowq.vlm import (
    IMAGE_TOKEN,
    VisionEngine,
    expand_image_tokens,
    recorded_image_token_id,
    required_tensors,
    required_vision_tensors,
)

TASK = "image-text-generation"

#: The chunk size the reference capture was taken at. Reproducing it keeps the
#: generation below comparable with the recorded greedy sequence token for token.
PREFILL_CHUNK_SIZE = 100

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        (reason := real_models.skip_reason(TASK)) is not None,
        reason=reason or "",
    ),
]


@pytest.fixture(scope="module")
def model_dir() -> Path:
    directory = real_models.selected_model_dir()
    assert directory is not None  # guaranteed by the module-level skip
    return directory


@pytest.fixture(scope="module")
def engine(model_dir: Path) -> VisionEngine:
    """One engine for the whole module: the weights are memory-mapped once."""
    return VisionEngine.from_directory(
        model_dir,
        max_context=real_models.MAX_CONTEXT,
        prefill_chunk_size=PREFILL_CHUNK_SIZE,
    )


@pytest.fixture(scope="module")
def tool() -> object:
    return real_models.load_validation_tool()


@pytest.fixture(scope="module")
def artifacts(model_dir: Path) -> Path:
    """The directory holding the recorded capture and the sample PNGs, or a skip.

    The PNGs live there on purpose: both halves of the validation read the *same
    file*, so the pixel tensors compared below cannot differ because of the
    input. Regenerating them from the recorded seed would reintroduce exactly
    the doubt the digest is there to remove.
    """
    directory = real_models.artifacts_dir(model_dir)
    if not real_models.reference_is_complete(model_dir):
        pytest.skip(
            f"no reference capture in {directory}; run "
            "'python tools/validate_reference.py ref' first"
        )
    return directory


@pytest.fixture(scope="module")
def reference(artifacts: Path) -> dict:
    return real_models.read_json(artifacts / "reference.json")


@pytest.fixture(scope="module")
def cases(tool, reference: dict) -> list[dict]:
    """The capture protocol's samples joined with what was recorded for them."""
    recorded = {sample["name"]: sample for sample in reference["samples"]}
    return [dict(sample, recorded=recorded[sample["name"]]) for sample in tool.VISION_SAMPLES]


@pytest.fixture(scope="module")
def prompts(engine: VisionEngine, artifacts: Path, cases) -> dict:
    """Every recorded prompt, built once through the public API."""
    return {
        case["name"]: engine.build_prompt(
            artifacts / case["recorded"]["image"]["file"],
            case["question"],
            system=case["system"],
        )
        for case in cases
    }


# --------------------------------------------------------------------------- #
# the converted checkpoint
# --------------------------------------------------------------------------- #


def test_the_directory_is_self_sufficient_offline(model_dir: Path, engine: VisionEngine):
    """Config, both towers, the tokenizer *and* the processor metadata.

    The image preprocessing settings are part of the converted directory, not
    something fetched from the hub at request time — otherwise "offline" would
    only be true for the text half.
    """
    for name in ("config.json", "manifest.json", "preprocessor_config.json",
                 "processor_config.json", "chat_template.json", "tokenizer.json"):
        assert (model_dir / name).is_file(), name
    manifest = real_models.read_json(model_dir / "manifest.json")
    assert manifest["task"] == TASK
    assert manifest["model_family"] == "idefics3"
    assert manifest["processor_files"] == ["preprocessor_config.json", "processor_config.json"]
    assert engine.vlm_config.processor.image_mean is not None


def test_loading_never_reaches_the_network(model_dir: Path, monkeypatch):
    real_models.block_network(monkeypatch)
    assert VisionEngine.from_directory(model_dir, max_context=64).image_token_id > 0


def test_the_text_and_embedding_engines_refuse_an_image_text_directory(model_dir: Path):
    """Task guard, on the checkpoint whose ``model_type`` no text path implements.

    Both refusals matter and for different reasons: the generation engine would
    otherwise read ``config.json`` as a text config and miss every vision tensor,
    and the embedding engine would otherwise pool a decoder that has an
    ``lm_head`` and a vision tower bolted onto it.
    """
    with pytest.raises(ConfigError, match="image-text-generation"):
        Engine.from_directory(model_dir, max_context=64)
    with pytest.raises(ConfigError, match="idefics3"):
        EmbeddingEngine.from_directory(model_dir)


def test_the_image_token_id_is_the_tokenizers_own_placeholder(engine: VisionEngine, model_dir: Path):
    """Three independent readings of one number, on the real tokenizer."""
    recorded = recorded_image_token_id(model_dir)
    assert engine.image_token_id == recorded
    assert engine.tokenizer.encode(IMAGE_TOKEN) == [recorded]
    assert 0 <= recorded < engine.config.vocab_size


def test_both_towers_are_complete_and_share_nothing(engine: VisionEngine):
    """Every tensor the contract names is present, and no name serves twice.

    The two towers are deliberately unlike: the decoder is a Llama block with
    RMSNorm (weight only) and a SwiGLU MLP (gate/up/down, no bias), the vision
    tower a ViT block with LayerNorm (weight *and* bias) and a two-layer GELU MLP
    (fc1/fc2, with bias). Reusing the decoder's blocks in the vision branch — the
    cheapest way to get a plausible-looking image encoder — would show up here as
    missing names, and the shapes below as impossible ones.
    """
    config = engine.vlm_config
    weights = engine.model.weights
    weights.check_names(required_tensors(config))
    required = required_tensors(config)
    vision_required = required_vision_tensors(config)
    assert len(required) == len(weights.tensor_names)
    assert vision_required and set(vision_required) < set(required)

    text = {name for name in weights.tensor_names if ".text_model." in name}
    vision = {name for name in weights.tensor_names if ".vision_model." in name}
    connector = {name for name in weights.tensor_names if ".connector." in name}
    assert text and vision and connector
    assert not (text & vision) and not (text & connector) and not (vision & connector)
    assert not weights.aliases, "a vision checkpoint ties nothing; an alias here is a mistake"

    layer = "model.vision_model.encoder.layers.0"
    assert weights.get(f"{layer}.layer_norm1.bias").shape == (config.vision.hidden_size,)
    assert weights.get(f"{layer}.mlp.fc2.weight").shape == (
        config.vision.hidden_size,
        config.vision.intermediate_size,
    )
    assert weights.get(f"{layer}.self_attn.out_proj.bias").shape == (config.vision.hidden_size,)
    text_layer = f"{engine.config.tensor_prefix}.layers.0"
    for absent in (f"{text_layer}.input_layernorm.bias", f"{text_layer}.mlp.gate_proj.bias"):
        assert absent not in weights.tensor_names


# --------------------------------------------------------------------------- #
# the official preprocessing, bit for bit
# --------------------------------------------------------------------------- #


def test_the_sample_image_is_the_one_the_reference_read(
    engine: VisionEngine, artifacts: Path, cases, tool
):
    """Same bytes in, or the pixel comparison below would prove nothing."""
    for case in cases:
        recorded = case["recorded"]["image"]
        path = artifacts / recorded["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == recorded["sha256"], case["name"]
        assert tool.file_digest(path) == recorded["sha256"], case["name"]


def test_the_preprocessor_reproduces_the_recorded_pixel_tensors(
    engine: VisionEngine, artifacts: Path, cases, tool
):
    """RGB, resize, rescale, normalise, split, pad — against the official output.

    Every field of the recorded ``image`` block is compared, not just the pixel
    digest: a resize that lands on the same pixels by a different route would
    still show up in ``resized_size`` or ``encoder_size``, and a crop grid that
    came out one short would show up in ``num_frames``.
    """
    for case in cases:
        name = case["name"]
        recorded = case["recorded"]["image"]
        prepared = engine.preprocessor.prepare(artifacts / recorded["file"])

        assert prepared.rows == recorded["rows"], name
        assert prepared.cols == recorded["cols"], name
        assert prepared.num_frames == recorded["num_frames"], name
        assert prepared.num_image_tokens == recorded["num_image_tokens"], name
        assert list(prepared.original_size) == recorded["original_size"], name
        assert list(prepared.resized_size) == recorded["resized_size"], name
        assert list(prepared.encoder_size) == recorded["encoder_size"], name
        assert bool(prepared.pixel_attention_mask.all()) == recorded[
            "pixel_attention_mask_all_valid"
        ], name
        assert tool.tensor_digest(prepared.pixel_values) == recorded["pixel_values_digest"], name
        assert tool.tensor_digest(prepared.pixel_attention_mask.astype(np.float32)) == recorded[
            "pixel_attention_mask_digest"
        ], name


def test_the_image_token_count_follows_the_crop_grid(engine: VisionEngine, cases):
    """Crops row-major plus one global frame, each worth ``image_seq_len`` tokens.

    "One image" is not one frame: a 1200x300 picture becomes four 512x512 crops
    plus the resized whole, and skipping the crops would leave the model looking
    at a thumbnail it was never trained on.
    """
    config = engine.vlm_config
    image_seq_len = config.computed_image_seq_len
    for case in cases:
        recorded = case["recorded"]
        expected_frames = recorded["image"]["rows"] * recorded["image"]["cols"] + 1
        assert expected_frames == recorded["image"]["num_frames"], case["name"]
        assert (
            expected_frames * image_seq_len == recorded["image"]["num_image_tokens"]
        ), case["name"]
        assert recorded["image"]["num_image_tokens"] <= config.max_frames_per_image * image_seq_len


def test_the_expanded_prompt_is_the_recorded_one(
    engine: VisionEngine, prompts, reference: dict, cases
):
    """Text, token IDs and placeholder positions, against the official capture.

    The prompt is what the merge is indexed by: a placeholder one token earlier
    would put the first crop's features on a text token and still produce a
    grammatical sentence, which is why this is compared exactly rather than
    "looks right".
    """
    recorded_text = {entry["name"]: entry for entry in reference["tokenizer"]}
    for case in cases:
        name = case["name"]
        prompt = prompts[name]
        recorded = case["recorded"]

        assert prompt.text == recorded_text[name]["text"], name
        assert list(prompt.token_ids) == recorded_text[name]["ids"], name
        assert len(prompt.token_ids) == recorded["prompt_tokens"], name
        assert prompt.text.count(IMAGE_TOKEN) == recorded["image"]["num_image_tokens"], name
        assert len(prompt.image_positions) == recorded["image"]["num_image_tokens"], name
        assert prompt.image_positions[0] == recorded["first_image_position"], name
        assert prompt.image_positions[-1] == recorded["last_image_position"], name
        # Every marked position really is a placeholder, and every placeholder is
        # marked — the mask is what decides which rows get image features.
        assert all(
            prompt.token_ids[index] == engine.image_token_id for index in prompt.image_positions
        ), name
        assert sum(token == engine.image_token_id for token in prompt.token_ids) == len(
            prompt.image_positions
        ), name
        # The placeholders are *not* one contiguous block: the expansion is
        # ``<row_r_col_c>`` tag, then ``image_seq_len`` placeholders, per frame,
        # so the positions form exactly one run per frame and every run is full.
        # A missing crop, a duplicated global frame or a merge indexed by the
        # wrong run all break this while leaving the count above intact.
        runs: list[list[int]] = []
        for index in prompt.image_positions:
            if runs and index == runs[-1][-1] + 1:
                runs[-1].append(index)
            else:
                runs.append([index])
        assert len(runs) == prompt.image.num_frames, name
        assert [len(run) for run in runs] == [prompt.image.image_seq_len] * len(runs), name


def test_a_prompt_with_two_images_is_refused(engine: VisionEngine, prompts):
    """One image per request, stated as an error rather than a dropped second."""
    prepared = prompts["grid-1x4"].image
    with pytest.raises(UnsupportedInputError, match="exactly one image"):
        expand_image_tokens(f"{IMAGE_TOKEN} and {IMAGE_TOKEN}", prepared)


# --------------------------------------------------------------------------- #
# the vision tower, on real pixels
# --------------------------------------------------------------------------- #


def test_the_tower_attention_is_bidirectional_on_a_real_frame(
    engine: VisionEngine, prompts
):
    """Patch 0 reacts to the *last* patch's pixels, so the mask is not causal.

    The synthetic version of this test in ``tests/test_vlm.py`` can be passed by
    an implementation that happens to mix patches; here the perturbation is one
    16x16 patch in the bottom-right corner of a real 512x512 crop, and a causal
    mask would leave patch 0's output bit-identical.
    """
    prompt = prompts["grid-1x4"]
    baseline = engine.model.encode_image(prompt.image)

    patch = engine.vlm_config.vision.patch_size
    pixels = np.array(prompt.image.pixel_values, dtype=np.float32, copy=True)
    pixels[0, :, -patch:, -patch:] += 1.0
    perturbed = engine.model.encode_image(
        dataclasses.replace(prompt.image, pixel_values=pixels)
    )

    first = baseline.encoder_output[0][0]
    changed = perturbed.encoder_output[0][0]
    assert not np.allclose(first, changed), (
        "patch 0 did not notice a change in the last patch, so the vision tower "
        "is attending causally"
    )
    # The perturbed patch itself changed too, which is what makes the above
    # meaningful rather than a numerical accident.
    assert not np.allclose(baseline.encoder_output[0][-1], perturbed.encoder_output[0][-1])
    # A different frame's output is untouched: frames are encoded independently.
    assert np.array_equal(baseline.encoder_output[1], perturbed.encoder_output[1])
    assert baseline.encoder_output.shape == (
        prompt.image.num_frames,
        engine.vlm_config.vision.num_patches,
        engine.vlm_config.vision.hidden_size,
    )


# --------------------------------------------------------------------------- #
# generation, against the recorded reference
# --------------------------------------------------------------------------- #


def test_greedy_generation_reproduces_the_recorded_sequence(
    engine: VisionEngine, prompts, cases
):
    """The first four greedy tokens the official capture produced.

    Recorded for both samples as ``378, 2443, 5956, 282`` (" The image consists
    of"). A prefill that mis-aligned the image features, a connector with the
    wrong output width, or a cache that lost the image span would move one of
    these; "it wrote a sentence about the image" would not.
    """
    for case in cases:
        name = case["name"]
        recorded = case["recorded"]
        expected = (recorded["full_prefill_argmax"],) + tuple(
            step["argmax"] for step in recorded["decode_steps"]
        )
        result = engine.generate_prompt(prompts[name], max_new_tokens=len(expected))
        assert result.token_ids[: len(expected)] == expected, (
            f"{name}: generated {result.token_ids[: len(expected)]}, "
            f"the reference recorded {expected}"
        )
        assert result.text == engine.tokenizer.decode(list(result.token_ids)), name


def test_the_answer_depends_on_the_injected_image_features(
    engine: VisionEngine, prompts, cases
):
    """Same prompt, same positions, no merge — and a different continuation.

    The ablation keeps every token ID and therefore every position and the whole
    KV layout, and withholds only the connector's features, so the difference
    below is the image and nothing else. Measured on this checkpoint: with the
    features both samples continue ``378, 2443, 5956, 282``; without them
    grid-1x4 continues ``378, 2443, 314, 253`` (" The image is a") and grid-2x4
    ``378, 2443, 2744, 253`` (" The image shows a"). The two unprefilled runs
    differ from each other as well, which is what an unconditioned decoder does
    with two different position layouts.
    """
    for case in cases:
        name = case["name"]
        prompt = prompts[name]
        recorded = case["recorded"]
        expected = (recorded["full_prefill_argmax"],) + tuple(
            step["argmax"] for step in recorded["decode_steps"]
        )
        blind = engine.generate_ids(list(prompt.token_ids), max_new_tokens=len(expected))
        assert blind.token_ids[: len(expected)] != expected, (
            f"{name}: the answer did not change when the image features were "
            "withheld, so nothing here proves the image is used"
        )
        assert blind.prompt_token_ids == prompt.token_ids


def test_a_chunked_prefill_generates_the_same_tokens(engine: VisionEngine, model_dir: Path, cases, artifacts: Path):
    """Chunk boundaries inside the image span must not move the features.

    A second engine at a chunk size that cuts the 320-token image span into
    unequal pieces (37 does not divide 64), graded against the recorded sequence
    rather than against the first engine, so both are tied to the reference.
    """
    chunked = VisionEngine.from_directory(
        model_dir, max_context=real_models.MAX_CONTEXT, prefill_chunk_size=37
    )
    for case in cases:
        name = case["name"]
        recorded = case["recorded"]
        expected = (recorded["full_prefill_argmax"],) + tuple(
            step["argmax"] for step in recorded["decode_steps"]
        )
        prompt = chunked.build_prompt(
            artifacts / recorded["image"]["file"], case["question"], system=case["system"]
        )
        result = chunked.generate_prompt(prompt, max_new_tokens=len(expected))
        assert result.token_ids[: len(expected)] == expected, name


# --------------------------------------------------------------------------- #
# error paths
# --------------------------------------------------------------------------- #


def test_a_prompt_the_context_cannot_hold_is_refused_before_generating(
    model_dir: Path, cases, artifacts: Path
):
    """The *expanded* length is what has to fit, and crops are never dropped.

    grid-2x4 needs 612 tokens once its nine frames are expanded, so a 128-token
    budget has to refuse rather than quietly encode a smaller image.
    """
    small = VisionEngine.from_directory(model_dir, max_context=128)
    case = cases[-1]
    assert case["recorded"]["prompt_tokens"] > 128
    with pytest.raises(ContextOverflowError, match="never drops crops"):
        small.build_prompt(
            artifacts / case["recorded"]["image"]["file"], case["question"], system=case["system"]
        )


def test_a_missing_image_file_is_refused(engine: VisionEngine, artifacts: Path):
    with pytest.raises(UnsupportedInputError, match="image file not found"):
        engine.build_prompt(artifacts / "no-such-image.png", "What is this?")


def test_an_unreadable_image_file_is_refused(engine: VisionEngine, tmp_path: Path):
    """A file that is not an image must not become a blank one."""
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not a png at all")
    with pytest.raises(UnsupportedInputError, match="could not be read as an image"):
        engine.build_prompt(broken, "What is this?")


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #


def test_the_cli_answers_a_local_image_offline(
    model_dir: Path, artifacts: Path, cases, monkeypatch, capsys
):
    """The whole user-facing path — file path in, answer out — with no network."""
    import json

    from slowq import cli

    real_models.block_network(monkeypatch)
    case = cases[0]
    recorded = case["recorded"]
    argv = [
        "vlm",
        "--model",
        str(model_dir),
        "--image",
        str(artifacts / recorded["image"]["file"]),
        "--question",
        case["question"],
        "--max-new-tokens",
        "4",
        "--max-context",
        str(real_models.MAX_CONTEXT),
        "--prefill-chunk-size",
        str(PREFILL_CHUNK_SIZE),
        "--json",
        "--quiet",
    ]
    if case["system"] is not None:
        argv += ["--system", case["system"]]

    assert cli.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["image"]["num_frames"] == recorded["image"]["num_frames"]
    assert payload["image"]["num_image_tokens"] == recorded["image"]["num_image_tokens"]
    assert len(payload["prompt_token_ids"]) == recorded["prompt_tokens"]
    assert payload["token_ids"] == [
        recorded["full_prefill_argmax"],
        *(step["argmax"] for step in recorded["decode_steps"]),
    ]
    assert payload["text"]
