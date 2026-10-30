"""SmolVLM (``idefics3``): config, image preprocessing, vision tower, engine.

Nothing here touches the network or a real checkpoint. Two kinds of input are
used instead:

* the tiny ``idefics3`` directory from ``conftest``, whose 8px frames and 2px
  patches make every count in the pipeline — a 4x4 patch grid, ``scale_factor``
  2, 4 image tokens per frame, 5 frames for a 2x2 crop grid — small enough to
  work out by hand and assert exactly;
* hand-built :class:`PreparedImage` values, which let the vision tower and the
  merger be tested without Pillow. Only the tests that actually *read* an image
  skip when Pillow is absent, because ``slowq.vlm`` promises to be importable
  without it.

The last section covers :class:`ImageFeatureCache`, which is where "the vision
tower runs once per image" is turned into "once per *distinct* image". Its tests
count ``encode_image`` calls rather than timings, because a count is the claim
being made and a millisecond is not.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    TINY_VISION_ARCHITECTURE,
    TINY_VISION_PREPROCESSOR,
    TINY_VISION_PROCESSOR,
    TINY_VISION_TEXT_ARCHITECTURE,
    TINY_VISION_WRAPPER,
    build_tiny_vision_model,
    tiny_vision_tensors,
)
from slowq.errors import (
    ConfigError,
    ContextOverflowError,
    UnsupportedInputError,
    WeightError,
)
from slowq.model import CausalLMModel, DebugProbe
from slowq.ops import conv2d_patchify, gelu_tanh, layer_norm, pixel_shuffle, rms_norm
from slowq.vlm import (
    DEFAULT_IMAGE_CACHE_BYTES,
    FAKE_IMAGE_TOKEN,
    GLOBAL_IMAGE_TAG,
    IMAGE_TOKEN,
    ImageFeatureCache,
    ImagePreprocessor,
    ImageProcessorConfig,
    PreparedImage,
    SmolVLMConfig,
    SmolVLMModel,
    VisionConfig,
    VisionEngine,
    VisionPrompt,
    crop_grid,
    expand_image_tokens,
    image_prompt_string,
    patch_attention_mask,
    recorded_image_token_id,
    required_tensors,
    required_vision_tensors,
    resize_for_vision_encoder_size,
    resize_output_size_rescale_to_max_len,
    resize_output_size_scale_below_upper_bound,
    vision_position_ids,
)

# ---- the tiny geometry, read from the fixture rather than restated ----------- #

VISION_HIDDEN = TINY_VISION_ARCHITECTURE["hidden_size"]
VISION_LAYERS = TINY_VISION_ARCHITECTURE["num_hidden_layers"]
FRAME_SIDE = TINY_VISION_ARCHITECTURE["image_size"]
PATCH_SIDE = TINY_VISION_ARCHITECTURE["patch_size"]
PATCHES_PER_SIDE = FRAME_SIDE // PATCH_SIDE
SCALE_FACTOR = TINY_VISION_WRAPPER["scale_factor"]
TOKENS_PER_FRAME = TINY_VISION_PROCESSOR["image_seq_len"]
TEXT_HIDDEN = TINY_VISION_TEXT_ARCHITECTURE["hidden_size"]
CROP_SIDE = TINY_VISION_PREPROCESSOR["max_image_size"]["longest_edge"]
LONGEST_EDGE = TINY_VISION_PREPROCESSOR["size"]["longest_edge"]
MAX_FRAMES = (LONGEST_EDGE // CROP_SIDE) ** 2 + 1

#: 12x20 (HxW) resizes to 16x16, which is a 2x2 crop grid: 4 crops + the global
#: frame = 5 frames = 20 image tokens. 8x16 resizes to 8x16, a 1x2 grid, so it
#: exercises a non-square split.
WIDE_IMAGE = (12, 20)
NON_SQUARE_IMAGE = (8, 16)
WIDE_FRAMES = 5
WIDE_IMAGE_TOKENS = WIDE_FRAMES * TOKENS_PER_FRAME
NON_SQUARE_FRAMES = 3
NON_SQUARE_IMAGE_TOKENS = NON_SQUARE_FRAMES * TOKENS_PER_FRAME

QUESTION = "What is in this picture?"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def pil():
    """Pillow, or a skip. The config and tower tests must run without it."""
    return pytest.importorskip("PIL.Image", reason="reading an image needs Pillow")


def gradient_image(pil, height: int, width: int, *, mode: str = "RGB", alpha: int = 255):
    """A deterministic image in which every crop and every patch differs.

    A gradient rather than noise, so that "the features changed" can only mean
    "a different part of the image was read", and so that the same call always
    produces the same pixels.
    """
    rows, cols = np.mgrid[0:height, 0:width]
    channels = [
        (cols * 251 // max(width - 1, 1)).astype(np.uint8),
        (rows * 251 // max(height - 1, 1)).astype(np.uint8),
        ((cols + rows) * 7 % 251).astype(np.uint8),
    ]
    if mode == "RGBA":
        channels.append(np.full((height, width), alpha, dtype=np.uint8))
    return pil.fromarray(np.dstack(channels), mode)


def frames_of(*values: float, num_frames: int = 1) -> np.ndarray:
    """``[num_frames, 3, 8, 8]`` pixels, one constant value per frame."""
    pixels = np.zeros((num_frames, 3, FRAME_SIDE, FRAME_SIDE), dtype=np.float32)
    for index, value in enumerate(values):
        pixels[index] = value
    return pixels


def prepared(
    pixel_values: np.ndarray,
    *,
    image_seq_len: int = TOKENS_PER_FRAME,
    rows: int = 0,
    cols: int = 0,
    mask: np.ndarray | None = None,
) -> PreparedImage:
    """A ``PreparedImage`` built by hand, so no Pillow and no resizing involved."""
    num_frames, _, height, width = pixel_values.shape
    if mask is None:
        mask = np.ones((num_frames, height, width), dtype=np.int64)
    return PreparedImage(
        pixel_values=pixel_values,
        pixel_attention_mask=mask,
        rows=rows,
        cols=cols,
        original_size=(height, width),
        resized_size=(height, width),
        encoder_size=(height, width),
        image_seq_len=image_seq_len,
    )


def drain(steps) -> object:
    """Consume a stream and return its final ``GenerationResult``."""
    result = None
    for step in steps:
        if step.result is not None:
            result = step.result
    assert result is not None, "a generation stream always ends with a result"
    return result


def last_prompt_logits(engine, *, prompt=None, ids=None) -> np.ndarray:
    """The logits over the prompt's last position, via the debug probe."""
    probe = DebugProbe(positions=(-1,), captures=frozenset({"logits"}))
    steps = (
        engine.stream_prompt(prompt, max_new_tokens=1, probe=probe)
        if prompt is not None
        else engine.stream_ids(ids, max_new_tokens=1, probe=probe)
    )
    drain(steps)
    return probe.records["logits[-1]"]


def vision_model(directory: Path) -> SmolVLMModel:
    return SmolVLMModel.from_directory(directory)


def rewrite(directory: Path, name: str, payload: dict) -> None:
    (directory / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_json(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. the fixture's own arithmetic
# --------------------------------------------------------------------------- #


def test_the_tiny_geometry_is_the_one_the_assertions_assume():
    """Pin the fixture down, so a change to it fails here and not everywhere.

    These are the relations ``SmolVLMConfig`` enforces on a real checkpoint;
    restating them as arithmetic documents why 4 tokens per frame and 5 frames
    per wide image are the right expectations below.
    """
    assert PATCHES_PER_SIDE**2 // SCALE_FACTOR**2 == TOKENS_PER_FRAME == 4
    assert TINY_VISION_ARCHITECTURE["hidden_size"] * SCALE_FACTOR**2 == 32
    assert MAX_FRAMES == 5
    assert WIDE_IMAGE_TOKENS == 20
    assert NON_SQUARE_IMAGE_TOKENS == 12
    assert FRAME_SIDE == CROP_SIDE, "the encoder only accepts frames of the crop size"


# --------------------------------------------------------------------------- #
# 2. configuration
# --------------------------------------------------------------------------- #


def test_an_idefics3_directory_loads_as_a_wrapper_around_a_llama_decoder(
    tiny_vlm_dir: Path,
):
    config = SmolVLMConfig.from_directory(tiny_vlm_dir)
    assert config.family == "idefics3"
    assert config.task == "image-text-generation"
    assert config.scale_factor == SCALE_FACTOR
    assert config.connector_out_features == VISION_HIDDEN * SCALE_FACTOR**2
    assert config.max_frames_per_image == MAX_FRAMES
    # The decoder is an ordinary llama model that happens to live under a prefix.
    assert config.text.family == "llama"
    assert config.text.task == "image-text-generation"
    assert config.text.tensor_prefix == "model.text_model"
    assert config.text.qk_norm is False, "llama has no per-head Q/K RMSNorm"
    assert config.text.tie_word_embeddings is False
    assert config.vision.head_dim == VISION_HIDDEN // TINY_VISION_ARCHITECTURE[
        "num_attention_heads"
    ]


def test_image_token_id_is_the_id_the_tokenizer_gives_the_placeholder(
    tiny_vlm_dir: Path,
):
    """The one number the whole merge depends on must come from the vocabulary.

    ``config.json`` records an id and the prompt is searched for that id; if the
    two ever disagreed, no image feature would be placed and the model would
    answer from the text alone.
    """
    config = SmolVLMConfig.from_directory(tiny_vlm_dir)
    engine = VisionEngine.from_directory(tiny_vlm_dir, max_context=128)
    assert engine.tokenizer.id_to_token(config.image_token_id) == IMAGE_TOKEN
    assert engine.image_token_id == config.image_token_id
    assert recorded_image_token_id(tiny_vlm_dir) == config.image_token_id


def test_image_seq_len_is_cross_checked_against_the_vision_geometry(
    tmp_path: Path,
):
    """A recorded ``image_seq_len`` the geometry cannot produce is refused.

    The prompt would then contain a number of ``<image>`` placeholders the
    connector can never match, and the failure would surface as a merge error at
    request time — or worse, as an answer that quietly ignored the image.
    """
    root = tmp_path / "vlm"
    root.mkdir()
    build_tiny_vision_model(root, processor_overrides={"image_seq_len": TOKENS_PER_FRAME + 1})
    with pytest.raises(ConfigError, match="image_seq_len=5"):
        SmolVLMConfig.from_directory(root)


def test_the_vision_image_size_must_equal_the_preprocessor_crop_size(
    tmp_path: Path,
):
    """Frames are cut to ``max_image_size``, so the position embedding must fit them."""
    root = tmp_path / "vlm"
    root.mkdir()
    build_tiny_vision_model(
        root,
        preprocessor_overrides={"max_image_size": {"longest_edge": CROP_SIDE // 2}},
    )
    with pytest.raises(ConfigError, match="position embedding"):
        SmolVLMConfig.from_directory(root)


def test_image_token_id_must_be_inside_the_text_vocabulary(tmp_path: Path):
    root = tmp_path / "vlm"
    build_tiny_vision_model(root)
    config = read_json(root, "config.json")
    config["image_token_id"] = config["vocab_size"] + 5
    rewrite(root, "config.json", config)
    with pytest.raises(ConfigError, match="outside the text vocabulary"):
        SmolVLMConfig.from_directory(root)


def test_a_text_only_directory_is_not_an_image_text_model(tiny_model_dir: Path):
    """Task comes from the manifest, never from the config's shape."""
    with pytest.raises(ConfigError, match="was converted as task="):
        SmolVLMConfig.from_directory(tiny_model_dir)


def test_an_embedding_directory_is_not_an_image_text_model(tiny_embedding_dir: Path):
    with pytest.raises(ConfigError, match="was converted as task="):
        SmolVLMConfig.from_directory(tiny_embedding_dir)


def test_describe_keeps_the_wrapper_family_beside_the_decoder_family(
    tiny_vlm_dir: Path,
):
    """``describe()`` is flattened into a dict by ``slowq inspect``.

    The nested decoder also reports a ``family`` and a ``task``, and left
    unlabelled they would overwrite the wrapper's — so ``inspect`` would have
    called an idefics3 checkpoint a llama one.
    """
    described = dict(SmolVLMConfig.from_directory(tiny_vlm_dir).describe())
    assert described["family"] == "idefics3"
    assert described["task"] == "image-text-generation"
    assert described["decoder family"] == "llama"
    assert described["vision patches per frame"] == PATCHES_PER_SIDE**2
    assert described["max image tokens per image"] == MAX_FRAMES * TOKENS_PER_FRAME


# --------------------------------------------------------------------------- #
# 3. vision config validation
# --------------------------------------------------------------------------- #


def vision_config(**overrides) -> dict:
    return dict(TINY_VISION_ARCHITECTURE, **overrides)


def test_the_vision_tower_must_use_the_tanh_gelu():
    """``hidden_act`` is checked, not assumed: the decoder's SiLU would be wrong here."""
    with pytest.raises(ConfigError, match=r"hidden_act='silu'.*tanh approximation of GELU"):
        VisionConfig.from_hf_config(vision_config(hidden_act="silu"), "vision_config")
    with pytest.raises(ConfigError, match=r"hidden_act='gelu'.*tanh approximation of GELU"):
        VisionConfig.from_hf_config(vision_config(hidden_act="gelu"), "vision_config")


def test_vision_hidden_size_must_divide_by_the_head_count():
    with pytest.raises(ConfigError, match="divisible"):
        VisionConfig.from_hf_config(
            vision_config(hidden_size=VISION_HIDDEN, num_attention_heads=3), "vision_config"
        )


def test_the_frame_must_be_a_whole_number_of_patches():
    with pytest.raises(ConfigError, match="whole number"):
        VisionConfig.from_hf_config(vision_config(image_size=7), "vision_config")


def test_the_vision_tower_expects_rgb():
    with pytest.raises(ConfigError, match="RGB"):
        VisionConfig.from_hf_config(vision_config(num_channels=1), "vision_config")


@pytest.mark.parametrize(
    "key", ["hidden_size", "num_hidden_layers", "patch_size", "num_attention_heads"]
)
def test_a_non_positive_vision_dimension_is_refused(key: str):
    with pytest.raises(ConfigError, match=key):
        VisionConfig.from_hf_config(vision_config(**{key: 0}), "vision_config")


def test_a_missing_vision_dimension_is_refused_rather_than_defaulted():
    config = vision_config()
    del config["image_size"]
    with pytest.raises(ConfigError, match="image_size"):
        VisionConfig.from_hf_config(config, "vision_config")


# --------------------------------------------------------------------------- #
# 4. processor configuration
# --------------------------------------------------------------------------- #


def processor_config(directory: Path) -> ImageProcessorConfig:
    return ImageProcessorConfig.from_directory(directory)


def test_the_processor_settings_come_from_the_directory(tiny_vlm_dir: Path):
    processor = processor_config(tiny_vlm_dir)
    assert processor.size_longest_edge == LONGEST_EDGE
    assert processor.max_image_size_longest_edge == CROP_SIDE
    assert processor.image_seq_len == TOKENS_PER_FRAME
    assert processor.resample == 1
    assert processor.image_mean == (0.5, 0.5, 0.5)


@pytest.mark.parametrize(
    "flag",
    [
        "do_convert_rgb",
        "do_resize",
        "do_image_splitting",
        "do_rescale",
        "do_normalize",
        "do_pad",
    ],
)
def test_a_preprocessing_step_switched_off_is_refused(tmp_path: Path, flag: str):
    """SlowQ implements the pipeline with every step on; a checkpoint that turns
    one off needs a different path, and guessing which would change the pixels."""
    build_tiny_vision_model(tmp_path / "vlm", preprocessor_overrides={flag: False})
    with pytest.raises(ConfigError, match=flag):
        processor_config(tmp_path / "vlm")


def test_a_missing_preprocessor_config_is_refused(tmp_path: Path):
    root = tmp_path / "vlm"
    build_tiny_vision_model(root)
    (root / "preprocessor_config.json").unlink()
    with pytest.raises(ConfigError, match="preprocessor_config.json"):
        processor_config(root)


def test_a_missing_processor_config_is_refused(tmp_path: Path):
    """Without ``image_seq_len`` the prompt cannot be built to match the tower."""
    root = tmp_path / "vlm"
    build_tiny_vision_model(root)
    (root / "processor_config.json").unlink()
    with pytest.raises(ConfigError, match="image_seq_len"):
        processor_config(root)


def test_a_foreign_image_processor_type_is_refused(tmp_path: Path):
    build_tiny_vision_model(
        tmp_path / "vlm",
        preprocessor_overrides={"image_processor_type": "SiglipImageProcessor"},
    )
    with pytest.raises(
        ConfigError,
        match=r"image_processor_type='SiglipImageProcessor'.*Idefics3 pipeline",
    ):
        processor_config(tmp_path / "vlm")


def test_an_unknown_resampling_filter_is_refused(tmp_path: Path):
    build_tiny_vision_model(tmp_path / "vlm", preprocessor_overrides={"resample": 99})
    with pytest.raises(ConfigError, match="resample"):
        processor_config(tmp_path / "vlm")


def test_a_zero_normalisation_std_is_refused(tmp_path: Path):
    build_tiny_vision_model(
        tmp_path / "vlm", preprocessor_overrides={"image_std": [0.5, 0.0, 0.5]}
    )
    with pytest.raises(ConfigError, match="image_std"):
        processor_config(tmp_path / "vlm")


def test_a_wrong_length_normalisation_mean_is_refused(tmp_path: Path):
    build_tiny_vision_model(
        tmp_path / "vlm", preprocessor_overrides={"image_mean": [0.5, 0.5]}
    )
    with pytest.raises(ConfigError, match="image_mean"):
        processor_config(tmp_path / "vlm")


# --------------------------------------------------------------------------- #
# 5. weight rules
# --------------------------------------------------------------------------- #


def test_required_tensors_cover_both_towers_and_the_connector(tiny_vlm_dir: Path):
    config = SmolVLMConfig.from_directory(tiny_vlm_dir)
    names = set(required_tensors(config))
    shapes = required_vision_tensors(config)
    assert set(shapes) < names
    assert "model.text_model.embed_tokens.weight" in names
    # The head is untied, so it is a tensor of its own and not a manifest alias.
    assert "lm_head.weight" in names
    assert shapes["model.connector.modality_projection.proj.weight"] == (
        TEXT_HIDDEN,
        config.connector_out_features,
    )
    assert shapes["model.vision_model.embeddings.position_embedding.weight"] == (
        PATCHES_PER_SIDE**2,
        VISION_HIDDEN,
    )
    assert shapes["model.vision_model.embeddings.patch_embedding.weight"] == (
        VISION_HIDDEN,
        3,
        PATCH_SIDE,
        PATCH_SIDE,
    )
    assert not [name for name in names if "q_norm" in name or "k_norm" in name]


def test_the_vision_bias_vectors_are_part_of_the_contract(tiny_vlm_dir: Path):
    """LayerNorm and all four attention projections carry a bias here.

    These are exactly the tensors a decoder-shaped implementation would not ask
    for, so a converter that dropped them would produce a model that loads and
    computes the wrong thing.
    """
    names = set(required_vision_tensors(SmolVLMConfig.from_directory(tiny_vlm_dir)))
    for layer in range(VISION_LAYERS):
        prefix = f"model.vision_model.encoder.layers.{layer}"
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert f"{prefix}.self_attn.{projection}.bias" in names
        assert f"{prefix}.layer_norm1.bias" in names
        assert f"{prefix}.layer_norm2.bias" in names
    assert "model.vision_model.post_layernorm.bias" in names
    assert "model.connector.modality_projection.proj.bias" not in names


def test_a_missing_vision_tensor_fails_the_load(tmp_path: Path):
    root = tmp_path / "vlm"
    build_tiny_vision_model(root, omit=("model.vision_model.post_layernorm.bias",))
    with pytest.raises(WeightError, match="post_layernorm.bias"):
        vision_model(root)


def test_a_missing_text_tensor_fails_the_load(tmp_path: Path):
    root = tmp_path / "vlm"
    build_tiny_vision_model(root, omit=("model.text_model.layers.0.mlp.down_proj.weight",))
    with pytest.raises(WeightError, match="down_proj.weight"):
        vision_model(root)


# --------------------------------------------------------------------------- #
# 6. resize and split geometry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (12, 20, (10, 16)),  # width wins, int(16 / (20/12)) = 9, bumped to 10
        (20, 12, (16, 10)),  # the transposed case takes the other branch
        (16, 16, (16, 16)),  # already the target
        (4, 4, (16, 16)),  # upscaling is part of the official behaviour
        (1, 3, (6, 16)),  # extreme aspect: int(16/3)=5 truncated, then bumped to even
    ],
)
def test_the_longest_edge_resize_matches_the_official_arithmetic(
    height: int, width: int, expected: tuple[int, int]
):
    assert resize_output_size_rescale_to_max_len(height, width, 16) == expected


def test_the_odd_height_bump_is_part_of_the_official_arithmetic():
    """``if height % 2 != 0: height += 1`` changes the aspect ratio of the *next*
    resize, so it is not a cosmetic detail that could be dropped."""
    without_bump = (9, 16)
    assert resize_output_size_rescale_to_max_len(12, 20, 16) == (10, 16)
    assert without_bump != (10, 16)
    assert resize_for_vision_encoder_size(9, 16, 8) == (16, 16)
    assert resize_for_vision_encoder_size(10, 16, 8) == (16, 16)


def test_the_upper_bound_resize_only_fires_above_its_bound():
    assert resize_output_size_scale_below_upper_bound(16, 16) == (16, 16)
    assert resize_output_size_scale_below_upper_bound(10, 16) == (10, 16)
    height, width = resize_output_size_scale_below_upper_bound(100, 5000, max_len=4096)
    assert width == 4096
    assert height == int(4096 / 50)
    assert resize_output_size_scale_below_upper_bound(5000, 100, max_len=4096) == (
        4096,
        int(4096 * (100 / 5000)),
    )


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (10, 16, (16, 16)),  # 10 rounds up to two crops, so the ratio is not kept
        (8, 16, (8, 16)),  # already whole crops
        (16, 16, (16, 16)),
        (9, 9, (16, 16)),
        (16, 1, (16, 8)),  # the narrow side still gets a whole crop
    ],
)
def test_the_encoder_resize_rounds_both_sides_up_to_whole_crops(
    height: int, width: int, expected: tuple[int, int]
):
    assert resize_for_vision_encoder_size(height, width, CROP_SIDE) == expected


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (8, 8, (0, 0, 8, 8)),  # one crop covers it: no split, no row/col tags
        (16, 16, (2, 2, 8, 8)),
        (8, 16, (1, 2, 8, 8)),
        (16, 8, (2, 1, 8, 8)),
        (20, 12, (3, 2, 7, 6)),  # crops shrink so the grid comes out whole
    ],
)
def test_the_crop_grid_divides_the_frame_exactly(
    height: int, width: int, expected: tuple[int, ...]
):
    assert crop_grid(height, width, CROP_SIDE) == expected


def test_the_full_resize_chain_is_what_the_preprocessor_does(pil, tiny_vlm_dir: Path):
    """The three steps compose into the sizes ``prepare`` reports."""
    engine = VisionEngine.from_directory(tiny_vlm_dir, max_context=128)
    image = gradient_image(pil, *WIDE_IMAGE)
    result = engine.preprocessor.prepare(image)
    assert result.original_size == WIDE_IMAGE
    assert result.resized_size == resize_output_size_rescale_to_max_len(
        *WIDE_IMAGE, LONGEST_EDGE
    )
    assert result.encoder_size == resize_for_vision_encoder_size(
        *result.resized_size, CROP_SIDE
    )
    rows, cols, _crop_h, _crop_w = crop_grid(*result.encoder_size, CROP_SIDE)
    assert (result.rows, result.cols) == (rows, cols)


# --------------------------------------------------------------------------- #
# 7. the vision tower and the connector
# --------------------------------------------------------------------------- #


def test_the_connector_flattens_its_frames_in_frame_order(tiny_vlm_dir: Path):
    """``features`` is the exact layout the merge assigns from, so the flatten matters.

    The official ``inputs_merger`` reshapes the tower's output with
    ``view(-1, vision_hidden_size)`` over the frames, which is row-major: all of
    frame 0's tokens, then all of frame 1's. A token-major flatten would put
    frame 1's first token where frame 0's second belongs — every shape would stay
    right and every number after it would be wrong.
    """
    model = vision_model(tiny_vlm_dir)
    output = model.encode_image(prepared(frames_of(0.1, -0.4, 0.7, num_frames=3)))
    assert output.encoder_output.shape == (3, PATCHES_PER_SIDE**2, VISION_HIDDEN)
    assert output.per_frame.shape == (3, TOKENS_PER_FRAME, TEXT_HIDDEN)
    assert output.features.shape == (3 * TOKENS_PER_FRAME, TEXT_HIDDEN)
    assert np.array_equal(
        output.features, output.per_frame.reshape(3 * TOKENS_PER_FRAME, TEXT_HIDDEN)
    )


def test_a_frame_is_encoded_the_same_alone_as_in_a_stack(tiny_vlm_dir: Path):
    """Frames do not attend to each other, and the output order is the input order.

    The tower has no cache and no cross-frame attention, so encoding one frame on
    its own must give the row block the stack gives for it. Reversing the stack
    must reverse the blocks: that is what pins the ordering down rather than
    leaving it to whichever sort the reshape happened to apply.
    """
    model = vision_model(tiny_vlm_dir)
    pixels = frames_of(0.1, -0.4, 0.7, num_frames=3)
    stacked = model.encode_image(prepared(pixels))
    alone = model.encode_image(prepared(pixels[1:2]))
    assert np.allclose(alone.per_frame[0], stacked.per_frame[1], rtol=0, atol=1e-6)
    reversed_stack = model.encode_image(prepared(pixels[::-1]))
    assert np.allclose(
        reversed_stack.per_frame, stacked.per_frame[::-1], rtol=0, atol=1e-6
    )


def test_the_tower_attention_is_bidirectional(tiny_vlm_dir: Path):
    """Non-causal: the last patch of a frame changes the first patch's output.

    A decoder-shaped implementation would mask the vision attention causally and
    still produce a plausible tensor, so this is asserted rather than assumed.
    The change is applied to the bottom-right 2x2 pixels, which is the last patch
    of the 4x4 grid.
    """
    model = vision_model(tiny_vlm_dir)
    base = frames_of(0.3)
    changed = base.copy()
    changed[0, :, -PATCH_SIDE:, -PATCH_SIDE:] += 0.5
    before = model.encode_image(prepared(base)).encoder_output[0]
    after = model.encode_image(prepared(changed)).encoder_output[0]
    assert not np.allclose(before[0], after[0], rtol=0, atol=1e-6), (
        "the first patch did not see a change in the last one, so the vision "
        "attention is causal"
    )


def test_a_frame_the_position_embedding_cannot_cover_is_refused(tiny_vlm_dir: Path):
    """A 12px frame is a 6x6 patch grid; the table only has 4x4 rows."""
    model = vision_model(tiny_vlm_dir)
    oversized = np.zeros((1, 3, FRAME_SIDE + PATCH_SIDE * 2, FRAME_SIDE), dtype=np.float32)
    with pytest.raises(UnsupportedInputError, match="the position embedding only covers"):
        model.encode_image(prepared(oversized))


def test_a_partially_padded_patch_grid_is_refused(tiny_vlm_dir: Path):
    """SlowQ's single-image path pads nothing, so a hole means a different pipeline.

    A patch is valid if *any* of its pixels is, which is the official
    ``unfold``-and-sum rule — so one masked pixel inside a 2x2 patch is not a
    hole, and only a fully masked patch is.
    """
    model = vision_model(tiny_vlm_dir)
    one_pixel = np.ones((1, FRAME_SIDE, FRAME_SIDE), dtype=np.int64)
    one_pixel[0, 0, 0] = 0
    assert patch_attention_mask(one_pixel, PATCH_SIDE).all()

    whole_patch = np.ones((1, FRAME_SIDE, FRAME_SIDE), dtype=np.int64)
    whole_patch[0, :PATCH_SIDE, :PATCH_SIDE] = 0
    assert not patch_attention_mask(whole_patch, PATCH_SIDE).all()
    with pytest.raises(UnsupportedInputError, match="partially-padded patch grid"):
        model.encode_image(prepared(frames_of(0.5), mask=whole_patch))


# --------------------------------------------------------------------------- #
# 8. merging the image into the prompt, and generating from it
# --------------------------------------------------------------------------- #


def image_span(num_tokens: int) -> tuple[list[int], np.ndarray, list[int]]:
    """A prompt of ``num_tokens`` ids with a non-contiguous image span in it.

    Non-contiguous on purpose: the real expansion interleaves
    ``<fake_token_around_image>``, the ``<row_r_col_c>`` tags and newlines between
    the frames, so a test that only ever marks one contiguous run would not
    exercise the cursor the way the prompt does.
    """
    ids = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    assert len(ids) == num_tokens
    positions = [5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
    assert len(positions) == num_tokens - 8
    mask = np.zeros(num_tokens, dtype=bool)
    mask[positions] = True
    return ids, mask, positions


def fused_sequence(engine, ids, features, mask) -> np.ndarray:
    """The merged ``[T, H]`` matrix the decoder reads, via the debug probe."""
    probe = DebugProbe(positions=(-1,), captures=frozenset({"embed_sequence"}))
    engine.model.forward(list(ids), engine.model.new_cache(len(ids) + 1), probe=probe,
                         inputs_embeds=features, inputs_embed_mask=mask)
    return probe.records["embed_sequence"]


def test_the_image_rows_replace_exactly_the_marked_positions(tiny_vlm_engine):
    """Row by row over the whole prompt, not just at its two ends."""
    engine = tiny_vlm_engine
    ids, mask, positions = image_span(3 * TOKENS_PER_FRAME + 8)
    features = engine.model.encode_image(
        prepared(frames_of(0.1, -0.4, 0.7, num_frames=3))
    ).features
    fused = fused_sequence(engine, ids, features, mask)

    plain_probe = DebugProbe(positions=(-1,), captures=frozenset({"embed_sequence"}))
    engine.model.forward(
        list(ids), engine.model.new_cache(len(ids) + 1), probe=plain_probe
    )
    table = plain_probe.records["embed_sequence"]

    assert fused.shape == (len(ids), TEXT_HIDDEN)
    assert np.array_equal(fused[positions], features)
    text_rows = np.flatnonzero(~mask)
    assert np.array_equal(fused[text_rows], table[text_rows])
    # The precondition, stated rather than assumed: the merge changed something.
    assert not np.array_equal(fused[positions], table[positions])


def test_a_feature_count_that_disagrees_with_the_placeholders_is_refused(tiny_vlm_engine):
    """Loudly, and with both counts: there is no correct way to line them up."""
    engine = tiny_vlm_engine
    total = 3 * TOKENS_PER_FRAME + 8
    ids, mask, _ = image_span(total)
    features = engine.model.encode_image(
        prepared(frames_of(0.1, -0.4, 0.7, num_frames=3))
    ).features
    cache = engine.model.new_cache(total + 1)
    with pytest.raises(ValueError) as excinfo:
        engine.model.forward(
            list(ids), cache, inputs_embeds=features[:-1], inputs_embed_mask=mask
        )
    message = str(excinfo.value)
    assert f"expands to {mask.sum()} placeholder tokens" in message
    assert f"the encoder produced {len(features) - 1} feature rows" in message


def test_a_mask_that_does_not_cover_the_block_is_refused(tiny_vlm_engine):
    engine = tiny_vlm_engine
    ids, mask, _ = image_span(3 * TOKENS_PER_FRAME + 8)
    features = engine.model.encode_image(
        prepared(frames_of(0.1, -0.4, 0.7, num_frames=3))
    ).features
    with pytest.raises(ValueError, match="covers 19 positions but this forward has 20"):
        engine.model.forward(
            list(ids),
            engine.model.new_cache(len(ids) + 1),
            inputs_embeds=features,
            inputs_embed_mask=mask[:-1],
        )


def test_a_mask_without_features_is_refused(tiny_vlm_engine):
    engine = tiny_vlm_engine
    ids, mask, _ = image_span(3 * TOKENS_PER_FRAME + 8)
    with pytest.raises(ValueError, match="marks placeholder positions but no"):
        engine.model.forward(
            list(ids), engine.model.new_cache(len(ids) + 1), inputs_embed_mask=mask
        )


@pytest.mark.parametrize("chunk_size", [20, 11, 7, 6, 5, 3, 1])
def test_a_chunk_boundary_inside_the_image_span_does_not_move_the_features(
    tiny_vlm_dir: Path, chunk_size: int
):
    """The feature cursor in ``Engine.stream_ids``, driven rather than replicated.

    ``tools/validate_reference.py`` slices the connector rows per chunk the same
    way, but that is a copy of the logic; this is the loop generation actually
    runs. The chunk sizes cut the 12-token image span at every alignment — 7 and
    3 do not divide it, 6 divides it exactly, 1 puts one placeholder per forward,
    and 20 is the single-chunk run they are all compared against.
    """
    total = 3 * TOKENS_PER_FRAME + 8
    ids, mask, _ = image_span(total)

    def run(size: int):
        engine = VisionEngine.from_directory(
            tiny_vlm_dir, max_context=64, prefill_chunk_size=size
        )
        features = engine.model.encode_image(
            prepared(frames_of(0.1, -0.4, 0.7, num_frames=3))
        ).features
        probe = DebugProbe(positions=(-1,), captures=frozenset({"logits"}))
        result = drain(
            engine.stream_ids(
                ids,
                max_new_tokens=1,
                probe=probe,
                prefill_embeds=(features, mask),
            )
        )
        return result, probe.records["logits[-1]"]

    whole, whole_logits = run(4096)
    assert whole.metrics.prefill_chunks == 1
    chunked, logits = run(chunk_size)
    if chunk_size < total:
        assert chunked.metrics.prefill_chunks == math.ceil(total / chunk_size) > 1
    assert chunked.metrics.prompt_tokens == total
    # Chunking changes the order of the reductions, not the mathematics: measured
    # at 3e-8 max_abs on this fixture, so 1e-6 leaves two orders of headroom.
    # Replacing the cursor with a fixed slice of the feature rows — the bug this
    # exists for — moves the logits by 5e-2.
    difference = float(np.max(np.abs(logits - whole_logits)))
    assert difference <= 1e-6, f"max_abs {difference:.3e} at chunk size {chunk_size}"
    assert chunked.token_ids == whole.token_ids


def test_a_text_request_after_an_image_one_does_not_see_the_image(tiny_vlm_dir: Path):
    """``prefill_embeds`` is an argument, so it cannot survive into the next request.

    The KV cache is reset per request, and a text prompt supplies no features at
    all; if either leaked, the second request's logits would differ from a fresh
    engine's. Bit equality is the assertion because both runs are the same code
    over the same weights.
    """
    ids, mask, _ = image_span(3 * TOKENS_PER_FRAME + 8)
    engine = VisionEngine.from_directory(tiny_vlm_dir, max_context=64)
    features = engine.model.encode_image(
        prepared(frames_of(0.1, -0.4, 0.7, num_frames=3))
    ).features
    drain(engine.stream_ids(ids, max_new_tokens=2, prefill_embeds=(features, mask)))

    text_ids = [7, 8, 9, 10, 11]
    after = last_prompt_logits(engine, ids=text_ids)
    fresh = last_prompt_logits(
        VisionEngine.from_directory(tiny_vlm_dir, max_context=64), ids=text_ids
    )
    assert np.array_equal(after, fresh)


def test_a_prompt_that_is_not_exactly_one_image_is_refused():
    """Zero placeholders would silently drop the image; two would need two towers."""
    single = prepared(frames_of(0.5))
    with pytest.raises(UnsupportedInputError, match="contains 0 <image>"):
        expand_image_tokens("no placeholder here", single)
    with pytest.raises(UnsupportedInputError, match="contains 2 <image>"):
        expand_image_tokens(f"a {IMAGE_TOKEN} b {IMAGE_TOKEN} c", single)
    assert expand_image_tokens(f"a {IMAGE_TOKEN} b", single).count(IMAGE_TOKEN) == (
        TOKENS_PER_FRAME
    )


def test_the_expansion_names_every_crop_and_the_global_frame():
    """Row-major crop tags, 1-based, then the global frame — the official order."""
    text = image_prompt_string(2, 3, TOKENS_PER_FRAME)
    order = [
        "<row_1_col_1>", "<row_1_col_2>", "<row_1_col_3>",
        "<row_2_col_1>", "<row_2_col_2>", "<row_2_col_3>",
        GLOBAL_IMAGE_TAG,
    ]
    assert [text.index(tag) for tag in order] == sorted(text.index(tag) for tag in order)
    assert text.count(IMAGE_TOKEN) == 7 * TOKENS_PER_FRAME
    assert text.startswith(FAKE_IMAGE_TOKEN) and text.endswith(FAKE_IMAGE_TOKEN)
    # An unsplit image has no row or column tags at all.
    single = image_prompt_string(0, 0, TOKENS_PER_FRAME)
    assert "<row_" not in single
    assert single.count(IMAGE_TOKEN) == TOKENS_PER_FRAME


def test_a_prompt_the_context_cannot_hold_is_refused_before_generating(pil, tiny_vlm_dir: Path):
    """Counting the *expanded* prompt: 20 of its tokens are the image."""
    engine = VisionEngine.from_directory(tiny_vlm_dir, max_context=WIDE_IMAGE_TOKENS + 2)
    with pytest.raises(ContextOverflowError) as excinfo:
        engine.build_prompt(gradient_image(pil, *WIDE_IMAGE), QUESTION)
    message = str(excinfo.value)
    assert f"expands to {WIDE_IMAGE_TOKENS} of the prompt's" in message
    assert "never drops crops" in message


def test_build_prompt_marks_every_placeholder_and_nothing_else(pil, tiny_vlm_engine):
    engine = tiny_vlm_engine
    prompt = engine.build_prompt(gradient_image(pil, *WIDE_IMAGE), QUESTION)
    assert prompt.image.num_frames == WIDE_FRAMES
    assert len(prompt.image_positions) == WIDE_IMAGE_TOKENS
    assert all(prompt.token_ids[p] == engine.image_token_id for p in prompt.image_positions)
    # The positions are the whole set, not a prefix of it.
    assert sum(token == engine.image_token_id for token in prompt.token_ids) == (
        WIDE_IMAGE_TOKENS
    )
    assert prompt.text.count(IMAGE_TOKEN) == WIDE_IMAGE_TOKENS
    for tag in ("<row_1_col_1>", "<row_2_col_2>", GLOBAL_IMAGE_TAG, QUESTION):
        assert tag in prompt.text


def test_an_unsplit_image_builds_a_single_frame_prompt(pil, tiny_vlm_single_frame_dir: Path):
    engine = VisionEngine.from_directory(tiny_vlm_single_frame_dir, max_context=128)
    prompt = engine.build_prompt(gradient_image(pil, 8, 8), QUESTION)
    assert prompt.image.num_frames == 1
    assert (prompt.image.rows, prompt.image.cols) == (0, 0)
    assert len(prompt.image_positions) == TOKENS_PER_FRAME
    assert "<row_" not in prompt.text
    result = drain(engine.stream_prompt(prompt, max_new_tokens=3))
    assert result.metrics.prompt_tokens == len(prompt.token_ids)
    assert len(result.token_ids) >= 1


# --------------------------------------------------------------------------- #
# 9. an illegal request is refused before the image is encoded
# --------------------------------------------------------------------------- #


def hand_built_prompt(num_frames: int = 3) -> VisionPrompt:
    """A ``VisionPrompt`` assembled without Pillow, for engine-level tests.

    The token count and the placeholder positions come from :func:`image_span`, so
    the feature rows the tiny tower produces for ``num_frames`` frames line up with
    the mask exactly the way a real expansion would — the engine cannot tell this
    apart from ``build_prompt``'s output.
    """
    ids, mask, positions = image_span(num_frames * TOKENS_PER_FRAME + 8)
    pixels = frames_of(*np.linspace(0.1, -0.4, num_frames), num_frames=num_frames)
    prompt = VisionPrompt(
        text=IMAGE_TOKEN,
        token_ids=tuple(ids),
        image=prepared(pixels),
        image_positions=positions,
    )
    assert int(mask.sum()) == prompt.image.num_frames * TOKENS_PER_FRAME
    return prompt


def watch_encode_image(engine, monkeypatch) -> list[int]:
    """Record every ``encode_image`` call, passing each one through unchanged."""
    calls: list[int] = []
    real = engine.model.encode_image

    def counted(prepared_image):
        calls.append(prepared_image.num_frames)
        return real(prepared_image)

    monkeypatch.setattr(engine.model, "encode_image", counted)
    return calls


def model_over_same_weights(model: SmolVLMModel, **config_fields) -> SmolVLMModel:
    """A second model over *this model's own* ``Weights``, with replaced config fields.

    Sharing the ``Weights`` object is the point. ``Weights.fingerprint()`` reads the
    manifest and nothing else, and it is the same manifest, so a fingerprint-only
    identity check cannot see this swap at all — it returns the same tuple, decides
    nothing changed, and keeps serving. ``dataclasses.replace`` rather than a second
    directory for the same reason: nothing on disk moved either.

    The fields used by the callers below are all legal compute parameters, so no
    loader refuses them, no shape changes and no row count changes. That is what
    makes the failure silent rather than an exception.
    """
    config = model.vlm_config
    for field, value in config_fields.items():
        config = dataclasses.replace(config, **{field: value})
    return SmolVLMModel(config, model.weights)


#: ``(max_new_tokens, exception, a fragment of its message)`` for requests the
#: shared validation must refuse. ``True`` is listed because ``bool`` is an
#: ``int`` in Python, and ``max_new_tokens=True`` silently meaning "one token"
#: would be a bug nobody would notice.
ILLEGAL_TOKEN_BUDGETS = (
    (0, ValueError, "max_new_tokens must be >= 1, got 0"),
    (-3, ValueError, "max_new_tokens must be >= 1, got -3"),
    (True, ValueError, "max_new_tokens must be an int, got True"),
    (1.5, ValueError, "max_new_tokens must be an int, got 1.5"),
    ("8", ValueError, "max_new_tokens must be an int, got '8'"),
)


def test_an_illegal_token_budget_is_refused_before_the_vision_tower_runs(
    tiny_vlm_engine, monkeypatch
):
    """The whole point of the check being where it is.

    ``stream_prompt`` used to call ``encode_image`` first and delegate the
    validation to ``stream_ids``, which is a generator: so a request for zero new
    tokens ran the vision tower over every crop and only then raised — on the
    first ``next()``, not at the call. Both halves are asserted here, because
    either alone would pass with the other broken.
    """
    engine = tiny_vlm_engine
    calls = watch_encode_image(engine, monkeypatch)
    prompt = hand_built_prompt()

    for max_new_tokens, kind, fragment in ILLEGAL_TOKEN_BUDGETS:
        # Not ``pytest.raises``: the spy assertion has to run whether or not the
        # call raised, because "it encoded the image and then raised late" and
        # "it never raised at all" are two different ways to be wrong here.
        error: Exception | None = None
        try:
            engine.stream_prompt(prompt, max_new_tokens=max_new_tokens)
        except kind as exc:
            error = exc
        assert calls == [], (
            f"max_new_tokens={max_new_tokens!r} ran the vision tower "
            f"{len(calls)} time(s) before being refused"
        )
        assert error is not None, (
            f"max_new_tokens={max_new_tokens!r} was still accepted when "
            "stream_prompt returned"
        )
        assert fragment in str(error)

        with pytest.raises(kind, match=re.escape(fragment)):
            engine.generate_prompt(prompt, max_new_tokens=max_new_tokens)
        assert calls == [], f"max_new_tokens={max_new_tokens!r} encoded the image first"


def test_the_image_and_question_entry_point_refuses_it_too(
    pil, tiny_vlm_single_frame_dir: Path, monkeypatch
):
    """``stream_image`` is what the CLI calls, so it gets the same treatment.

    It has to render and tokenise the prompt before anything can be counted
    against the budget, but the vision tower — the expensive part — still must
    not run for a request that was never going to be served.
    """
    engine = VisionEngine.from_directory(tiny_vlm_single_frame_dir, max_context=128)
    calls = watch_encode_image(engine, monkeypatch)
    image = gradient_image(pil, 8, 8)

    with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
        engine.stream_image(image, QUESTION, max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
        engine.generate_image(image, QUESTION, max_new_tokens=0)
    assert calls == []

    result = engine.generate_image(image, QUESTION, max_new_tokens=2)
    assert len(result.token_ids) >= 1
    assert calls == [1], "an 8px image is not split, so exactly one frame"


def test_an_over_budget_image_request_is_refused_before_the_vision_tower_runs(
    tiny_vlm_dir: Path, monkeypatch
):
    """The context check counts the prompt, and the prompt is known up front."""
    prompt = hand_built_prompt()
    budget = len(prompt.token_ids) + 4
    engine = VisionEngine.from_directory(tiny_vlm_dir, max_context=budget)
    calls = watch_encode_image(engine, monkeypatch)

    # It fits exactly, so this is the boundary and not an off-by-one argument.
    drain(engine.stream_prompt(prompt, max_new_tokens=4))
    assert calls == [prompt.image.num_frames]

    calls.clear()
    with pytest.raises(ContextOverflowError) as excinfo:
        engine.stream_prompt(prompt, max_new_tokens=5)
    message = str(excinfo.value)
    assert f"prompt ({len(prompt.token_ids)} tokens) + max_new_tokens (5)" in message
    assert f"exceeds max_context={budget}" in message
    assert "never truncates" in message
    assert calls == []

    with pytest.raises(ContextOverflowError):
        engine.generate_prompt(prompt, max_new_tokens=5)
    assert calls == []


def test_the_vision_path_refuses_exactly_what_the_text_path_refuses(tiny_vlm_dir: Path):
    """One implementation of the rules, so the two engines cannot drift apart.

    Both sides are asked for the same thing over the same token ids and must
    produce the same exception *and the same message*: a caller who reads the
    text engine's wording into an error handler should not have to special-case
    the vision one.
    """
    engine = VisionEngine.from_directory(tiny_vlm_dir, max_context=24)
    prompt = hand_built_prompt()

    for max_new_tokens, kind, fragment in ILLEGAL_TOKEN_BUDGETS + (
        (8, ContextOverflowError, "exceeds max_context=24"),
    ):
        with pytest.raises(kind) as vision_error:
            engine.stream_prompt(prompt, max_new_tokens=max_new_tokens)
        with pytest.raises(kind) as text_error:
            drain(engine.stream_ids(prompt.token_ids, max_new_tokens=max_new_tokens))
        assert fragment in str(vision_error.value)
        assert str(vision_error.value) == str(text_error.value)


def test_a_legal_image_request_is_unchanged_by_the_earlier_check(
    tiny_vlm_engine, monkeypatch
):
    """Moving the check must not move anything a valid request observes.

    The image is still encoded exactly once, its features still reach the decoder
    (the tokens are bit-identical to the same request driven through
    ``stream_ids`` by hand), the stream still yields one step per token, the cache
    still holds prompt-plus-decoded at the end, and ``vision_encode_seconds`` is
    still part of time-to-first-token rather than dropped on the floor.
    """
    engine = tiny_vlm_engine
    calls = watch_encode_image(engine, monkeypatch)
    prompt = hand_built_prompt()

    steps = list(engine.stream_prompt(prompt, max_new_tokens=3))
    result = steps[-1].result
    assert result is not None
    assert calls == [prompt.image.num_frames], "the image must still be encoded once"

    # The features really were merged in: the same request built by hand.
    features = engine.model.encode_image(prompt.image)
    mask = np.zeros(len(prompt.token_ids), dtype=bool)
    mask[list(prompt.image_positions)] = True
    manual = drain(
        engine.stream_ids(
            prompt.token_ids,
            max_new_tokens=3,
            prefill_embeds=(features.features, mask),
            vision_encode_seconds=features.encode_seconds,
        )
    )
    assert result.token_ids == manual.token_ids
    assert result.prompt_token_ids == manual.prompt_token_ids == prompt.token_ids
    assert result.stop_reason == manual.stop_reason

    # Streaming: one step per generated token, indices in order, result at the end.
    assert [step.index for step in steps] == list(range(len(steps)))
    assert all(step.result is None for step in steps[:-1])
    assert all(step.finished is False for step in steps[:-1])
    assert steps[-1].finished is True
    assert "".join(step.text_delta for step in steps) == result.text
    assert len(steps) == len(result.token_ids)

    assert result.metrics.prompt_tokens == len(prompt.token_ids)
    assert result.metrics.prefill_chunks == 1
    assert result.metrics.vision_encode_seconds > 0.0
    assert result.metrics.first_token_seconds >= (
        result.metrics.vision_encode_seconds + result.metrics.prefill_seconds
    )
    assert engine.kv_cache.length == len(prompt.token_ids) + len(result.token_ids) - 1


def test_repeating_an_image_request_after_a_refused_one_still_generates(
    tiny_vlm_engine, monkeypatch
):
    """A refused request must leave nothing behind for the next one.

    The check runs before the cache is reset, so this is the case where an early
    refusal could plausibly have leaked state: refuse, then ask for something
    legal, and require the answer to be what a fresh engine gives.
    """
    engine = tiny_vlm_engine
    calls = watch_encode_image(engine, monkeypatch)
    prompt = hand_built_prompt()
    drain(engine.stream_prompt(prompt, max_new_tokens=3))
    first = drain(engine.stream_prompt(prompt, max_new_tokens=3))
    # One encode for two requests, not two: the second finds the same pixels in
    # the image-feature cache and never reaches the vision tower. Captured as a
    # count rather than assumed, so the refusal check below is about the refusal.
    encoded_before_refusal = len(calls)
    assert encoded_before_refusal == 1

    with pytest.raises(ValueError):
        engine.stream_prompt(prompt, max_new_tokens=0)
    assert len(calls) == encoded_before_refusal, (
        "the refused request must not have encoded anything"
    )

    after = drain(engine.stream_prompt(prompt, max_new_tokens=3))
    assert after.token_ids == first.token_ids
    assert engine.kv_cache.length == len(prompt.token_ids) + len(first.token_ids) - 1


# --------------------------------------------------------------------------- #
# 10. image feature reuse: one vision tower run per image, not per request
# --------------------------------------------------------------------------- #


def feature_block(width: int = 8, value: float = 0.0) -> np.ndarray:
    """A ``[4, width]`` float32 array, so its ``nbytes`` is exactly ``16 * width``.

    The shape is arbitrary; the byte count is not, because every budget assertion
    below is written in terms of it.
    """
    return np.full((4, width), value, dtype=np.float32)


def test_a_miss_then_a_hit_returns_the_same_numbers_and_counts_both():
    cache = ImageFeatureCache(max_bytes=1024)
    stored = feature_block()
    assert cache.get("k") is None
    assert (cache.hits, cache.misses, cache.evictions) == (0, 1, 0)

    assert cache.put("k", stored) is True
    got = cache.get("k")
    assert got is not None and np.array_equal(got, stored)
    assert got.dtype == np.float32
    assert (cache.hits, cache.misses) == (1, 1)
    assert len(cache) == 1 and "k" in cache and "other" not in cache
    assert cache.nbytes == stored.nbytes == 128

    stats = cache.stats()
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["entries"] == 1 and stats["bytes"] == 128
    assert stats["max_bytes"] == 1024
    assert "bytes=128/1024" in repr(cache)


def test_the_cache_owns_its_buffers_in_both_directions():
    """``put`` copies in, ``get`` copies out, so no two holders share an array.

    The engine hands the returned features to the decoder as the prefill's
    embedding rows. If the cache handed out the buffer it also scores the next
    request against, one caller writing into it would silently change what every
    later caller receives.
    """
    cache = ImageFeatureCache(max_bytes=1024)
    stored = feature_block(value=1.0)
    cache.put("k", stored)

    stored[:] = 2.0  # the caller kept writing after handing the array over
    first = cache.get("k")
    assert np.all(first == 1.0), "put must not keep a reference to the caller's array"

    first[:] = 3.0  # and a reader must not write through to the cache either
    second = cache.get("k")
    assert np.all(second == 1.0), "get must not hand out the stored array itself"
    assert second is not first


def test_the_budget_is_enforced_least_recently_used_first():
    entry = feature_block(8)  # 128 bytes each
    cache = ImageFeatureCache(max_bytes=2 * entry.nbytes)
    for key in ("a", "b", "c"):
        assert cache.put(key, entry) is True

    assert "a" not in cache, "the oldest entry is the one that has to go"
    assert "b" in cache and "c" in cache
    assert cache.evictions == 1
    assert cache.nbytes == 2 * entry.nbytes, "an eviction has to give the bytes back"


def test_reading_an_entry_makes_it_the_most_recently_used():
    """LRU, not FIFO: a picture that keeps getting asked about stays cached."""
    entry = feature_block(8)
    cache = ImageFeatureCache(max_bytes=2 * entry.nbytes)
    cache.put("a", entry)
    cache.put("b", entry)
    assert cache.get("a") is not None

    cache.put("c", entry)
    assert "b" not in cache, "the untouched entry should have been evicted, not 'a'"
    assert "a" in cache and "c" in cache


def test_replacing_a_key_does_not_double_count_its_bytes():
    cache = ImageFeatureCache(max_bytes=1024)
    cache.put("k", feature_block(8))
    assert cache.put("k", feature_block(4)) is True
    assert len(cache) == 1
    assert cache.nbytes == feature_block(4).nbytes == 64
    assert cache.evictions == 0


def test_an_entry_larger_than_the_whole_budget_is_skipped_not_thrashed():
    """Storing it would evict everything else and then not fit either."""
    cache = ImageFeatureCache(max_bytes=feature_block(8).nbytes)
    assert cache.put("small", feature_block(4)) is True

    assert cache.put("huge", feature_block(16)) is False
    assert "huge" not in cache
    assert cache.evictions == 0, "an entry that cannot fit must not evict to try"
    assert cache.get("small") is not None, "the usable entry survived"


def test_a_zero_budget_disables_the_cache():
    cache = ImageFeatureCache(max_bytes=0)
    assert cache.put("k", feature_block()) is False
    assert len(cache) == 0 and cache.nbytes == 0
    assert cache.get("k") is None
    assert (cache.hits, cache.misses) == (0, 1)
    assert cache.stats()["hit_rate"] == 0.0


def test_a_negative_budget_is_refused():
    with pytest.raises(ValueError, match=">= 0"):
        ImageFeatureCache(max_bytes=-1)


def test_clear_drops_the_entries_but_keeps_the_history():
    cache = ImageFeatureCache(max_bytes=1024)
    cache.put("k", feature_block())
    cache.get("k")
    cache.get("absent")
    cache.clear()

    assert len(cache) == 0 and cache.nbytes == 0
    # The counters describe this engine's lifetime, and freeing memory has not
    # un-happened the hit that was already served.
    assert (cache.hits, cache.misses) == (1, 1)


def test_two_requests_over_one_image_run_the_vision_tower_once(
    tiny_vlm_engine, monkeypatch
):
    """The claim, as a call count: a hit must not reach ``encode_image`` at all."""
    engine = tiny_vlm_engine
    calls = watch_encode_image(engine, monkeypatch)
    prompt = hand_built_prompt()

    first = drain(engine.stream_prompt(prompt, max_new_tokens=3))
    second = drain(engine.stream_prompt(prompt, max_new_tokens=3))

    assert calls == [prompt.image.num_frames], (
        f"the second request re-ran the tower: {calls}"
    )
    assert [step.metrics.vision_cache_hit for step in (first, second)] == [False, True]
    assert second.token_ids == first.token_ids
    assert second.text == first.text
    assert engine.image_features.stats()["entries"] == 1


def test_the_lookup_is_on_pixel_content_not_on_python_object_identity(
    tiny_vlm_engine, monkeypatch
):
    """No two requests ever share a ``PreparedImage``, so identity cannot be the key.

    ``build_prompt`` makes a new one per call, and ``ImagePreprocessor.prepare``
    allocates fresh pixel arrays, so an ``id()``-based key would have a hit rate
    of exactly zero while looking like a cache.
    """
    engine = tiny_vlm_engine
    calls = watch_encode_image(engine, monkeypatch)
    prompt = hand_built_prompt()
    drain(engine.stream_prompt(prompt, max_new_tokens=3))

    copy = dataclasses.replace(
        prompt,
        image=dataclasses.replace(
            prompt.image,
            pixel_values=np.array(prompt.image.pixel_values),
            pixel_attention_mask=np.array(prompt.image.pixel_attention_mask),
        ),
    )
    assert copy.image is not prompt.image
    assert copy.image.pixel_values is not prompt.image.pixel_values

    again = drain(engine.stream_prompt(copy, max_new_tokens=3))
    assert calls == [prompt.image.num_frames], f"a copy missed the cache: {calls}"
    assert again.metrics.vision_cache_hit is True


def test_one_changed_pixel_is_a_different_image(tiny_vlm_engine, monkeypatch):
    """The other direction: a near-miss must not be served as a hit."""
    engine = tiny_vlm_engine
    calls = watch_encode_image(engine, monkeypatch)
    prompt = hand_built_prompt()
    drain(engine.stream_prompt(prompt, max_new_tokens=3))

    pixels = np.array(prompt.image.pixel_values)
    pixels[0, 0, 0, 0] += 1e-3
    other = dataclasses.replace(prompt, image=prepared(pixels))
    changed = drain(engine.stream_prompt(other, max_new_tokens=3))

    assert calls == [3, 3], f"the modified image was served from cache: {calls}"
    assert changed.metrics.vision_cache_hit is False
    assert len(engine.image_features) == 2

    # The precondition, checked after the counts above so the two direct encodes
    # do not disturb them: these really are different images, so a hit would have
    # been a wrong answer rather than merely a fast one.
    assert not np.array_equal(
        engine.model.encode_image(prompt.image).features,
        engine.model.encode_image(other.image).features,
    )


def test_a_debug_probe_bypasses_the_cache(tiny_vlm_engine, monkeypatch):
    """A probe asks what *this* request computed, so it must not be served an old one."""
    engine = tiny_vlm_engine
    calls = watch_encode_image(engine, monkeypatch)
    prompt = hand_built_prompt()
    drain(engine.stream_prompt(prompt, max_new_tokens=3))
    assert len(engine.image_features) == 1

    probe = DebugProbe(positions=(-1,), captures=frozenset({"logits"}))
    watched = drain(engine.stream_prompt(prompt, max_new_tokens=3, probe=probe))
    assert calls == [3, 3], "the probed request was served from the cache"
    assert watched.metrics.vision_cache_hit is False
    assert "logits[-1]" in probe.records

    after = drain(engine.stream_prompt(prompt, max_new_tokens=3))
    assert after.metrics.vision_cache_hit is True
    # Still two encodes: the bypass neither reads nor writes, so the probed
    # request left the cache exactly as it found it and the third one hits.
    assert calls == [3, 3]
    assert after.token_ids == watched.token_ids, "the bypass must not change the answer"


def test_two_questions_about_one_picture_share_the_tower(
    pil, tiny_vlm_single_frame_dir: Path, monkeypatch
):
    """The user-visible case, through the entry point the CLI uses."""
    engine = VisionEngine.from_directory(tiny_vlm_single_frame_dir, max_context=128)
    calls = watch_encode_image(engine, monkeypatch)
    image = gradient_image(pil, 8, 8)

    first_prompt = engine.build_prompt(image, "What is in this picture?")
    second_prompt = engine.build_prompt(image, "Say more about it.")
    assert first_prompt.token_ids != second_prompt.token_ids, (
        "the two questions must produce different prompts, or this proves nothing"
    )

    first = drain(engine.stream_prompt(first_prompt, max_new_tokens=3))
    second = drain(engine.stream_prompt(second_prompt, max_new_tokens=3))
    assert calls == [1], f"the tower ran per question: {calls}"
    assert [step.metrics.vision_cache_hit for step in (first, second)] == [False, True]
    # One entry for two prompts: the question is not part of the key, and must
    # not be, or every rephrasing would pay for the tower again.
    assert engine.image_features.stats()["entries"] == 1


def test_the_cache_does_not_change_the_answer(tiny_vlm_dir: Path):
    """Cached, disabled and too-small-to-hold-anything all decode the same tokens."""
    prompt = hand_built_prompt()
    outputs = []
    for budget in (DEFAULT_IMAGE_CACHE_BYTES, 0, 1):
        engine = VisionEngine.from_directory(
            tiny_vlm_dir, max_context=128, image_cache_bytes=budget
        )
        drain(engine.stream_prompt(prompt, max_new_tokens=3))  # warm it, if it can
        result = drain(engine.stream_prompt(prompt, max_new_tokens=3))
        assert engine.image_features.max_bytes == budget
        outputs.append(result.token_ids)
    assert outputs[0] == outputs[1] == outputs[2]


def test_a_refused_request_neither_looks_up_nor_stores(tiny_vlm_engine):
    """Validation still runs first, so a bad request touches no cache state."""
    engine = tiny_vlm_engine
    prompt = hand_built_prompt()
    drain(engine.stream_prompt(prompt, max_new_tokens=3))
    drain(engine.stream_prompt(prompt, max_new_tokens=3))

    before = engine.image_features.stats()
    assert (before["hits"], before["misses"], before["entries"]) == (1, 1, 1), (
        "the precondition: one miss that stored, then one hit"
    )

    with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
        engine.stream_prompt(prompt, max_new_tokens=0)

    after = engine.image_features.stats()
    # Field by field rather than ``after == before``: ``hit_rate`` is NaN when
    # nothing has been looked up, and NaN != NaN would fail for the wrong reason.
    for key in before:
        assert after[key] == before[key], key


def test_a_hit_reports_the_time_this_request_spent_not_the_earlier_ones(
    tiny_vlm_engine, monkeypatch
):
    """A near-zero ``vision_encode_seconds`` is only meaningful if it really drops.

    The tiny fixture's tower is fast enough that a cached hit and a real encode
    are hard to tell apart by timing alone, so ``encode_image`` is wrapped in a
    fixed sleep: the miss has to be at least that slow, and the hit has to be far
    faster. Otherwise the cache could be reporting the *first* request's encode
    on every later one and nothing here would notice.
    """
    engine = tiny_vlm_engine
    real = engine.model.encode_image
    delay = 0.05

    def slow(prepared_image):
        time.sleep(delay)
        return real(prepared_image)

    monkeypatch.setattr(engine.model, "encode_image", slow)
    prompt = hand_built_prompt()
    miss = drain(engine.stream_prompt(prompt, max_new_tokens=3))
    hit = drain(engine.stream_prompt(prompt, max_new_tokens=3))

    assert (miss.metrics.vision_cache_hit, hit.metrics.vision_cache_hit) == (False, True)
    assert miss.metrics.vision_encode_seconds >= delay
    assert hit.metrics.vision_encode_seconds < delay / 5, (
        "a hit is still being charged for the tower it skipped"
    )
    assert hit.metrics.first_token_seconds < miss.metrics.first_token_seconds
    assert hit.token_ids == miss.token_ids

    rows = dict(hit.metrics.report())
    assert "features cached" in rows["vision encode"], rows["vision encode"]
    assert "features cached" not in dict(miss.metrics.report())["vision encode"]


def test_the_cache_key_covers_everything_the_features_depend_on(
    tiny_vlm_engine, other_weights_vlm_dir: Path
):
    """Entry by entry, because a key missing one is a wrong answer, not a slow one.

    Left out of the key: a resize setting, an ``image_seq_len``, a crop grid or a
    set of weights would all still produce *some* features of the right shape, so
    the failure would be silent and would look exactly like a fast request.

    The identity half goes through real model objects rather than through the
    engine's private fields. Writing ``engine._weights_fingerprint`` proved the key
    *reads* that attribute, which a test can arrange however it likes; it said
    nothing about whether the engine ever *updates* it, which is the half that was
    broken. Assigning ``engine.vlm_config`` has the same defect and was corrected
    here for the same reason: it is derived state that the next identity check
    overwrites, while the features come from ``engine.model.vlm_config``. A key that
    followed the derived copy could disagree with the model that computes it.
    """
    engine = tiny_vlm_engine
    image = hand_built_prompt().image
    base = engine._image_cache_key(image)
    assert base[-1] == engine._identity

    for replaced in (
        dataclasses.replace(image, image_seq_len=image.image_seq_len + 1),
        dataclasses.replace(image, rows=1, cols=1),
        dataclasses.replace(image, original_size=(64, 64)),
        dataclasses.replace(image, resized_size=(64, 64)),
        dataclasses.replace(image, encoder_size=(64, 64)),
    ):
        assert engine._image_cache_key(replaced) != base, replaced

    processor = engine.vlm_config.processor
    original = engine.model
    for field, value in (
        ("image_seq_len", processor.image_seq_len + 1),
        ("max_image_size_longest_edge", processor.max_image_size_longest_edge + 1),
        ("size_longest_edge", processor.size_longest_edge + 1),
        ("rescale_factor", processor.rescale_factor * 2),
        ("image_mean", tuple(v + 0.1 for v in processor.image_mean)),
        ("image_std", tuple(v * 2 for v in processor.image_std)),
        ("do_normalize", not processor.do_normalize),
        ("resample", processor.resample + 1),
    ):
        engine.model = model_over_same_weights(
            original,
            processor=dataclasses.replace(processor, **{field: value}),
        )
        engine._model_identity_changed()
        assert engine._image_cache_key(image) != base, field
    engine.model = original
    engine._model_identity_changed()

    # Same manifest, different vision config. ``layer_norm_eps`` is a legal compute
    # parameter: nothing refuses the swap, every shape still lines up, and the row
    # count still matches the mask. Only the key can tell the two towers apart.
    engine.model = model_over_same_weights(
        original,
        vision=dataclasses.replace(original.vlm_config.vision, layer_norm_eps=0.5),
    )
    assert engine._model_identity_changed()
    assert engine._image_cache_key(image) != base, "a changed vision config kept the key"

    # A different weight manifest is a different model, even in the same process.
    engine.model = SmolVLMModel.from_directory(other_weights_vlm_dir)
    assert engine._model_identity_changed()
    assert engine._image_cache_key(image) != base, "a changed checkpoint kept the key"


def test_two_engines_on_one_directory_agree_on_the_key(tiny_vlm_dir: Path):
    """The identity comes from the manifest's contents, not from object identity.

    Which is why the model object is tracked *separately* from :attr:`_identity`
    rather than folded into it: this equality is what lets two engines over one
    directory share a meaning for a key, and a per-object component would break it.
    """
    image = hand_built_prompt().image
    first = VisionEngine.from_directory(tiny_vlm_dir, max_context=128)
    second = VisionEngine.from_directory(tiny_vlm_dir, max_context=128)

    assert first is not second and first.model is not second.model
    assert first._identity == second._identity
    assert first._image_cache_key(image) == second._image_cache_key(image)


# --------------------------------------------------------------------------- #
# the model behind a live engine changing
# --------------------------------------------------------------------------- #

#: The one tensor overridden to make the second checkpoint. It is the whole of
#: the connector, so two directories differing only here produce different
#: features from identical pixels while agreeing on every shape.
CONNECTOR_WEIGHT = "model.connector.modality_projection.proj.weight"


@pytest.fixture
def other_weights_vlm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A second tiny VLM: same geometry, same processor, different connector.

    Deliberately *not* a different architecture. A geometry check would catch a
    swapped ``hidden_size``; the bug this fixture reproduces is invisible to one,
    because the shapes all still line up and the features still have the right
    row count. Only the weight identity distinguishes the two, which is why the
    cache key carries a fingerprint at all.

    ``vocab_size`` is irrelevant to the connector's shape, so the override can be
    generated without knowing the tokenizer this directory will get.
    """
    config = dict(
        TINY_VISION_WRAPPER,
        text_config=TINY_VISION_TEXT_ARCHITECTURE,
        vision_config=TINY_VISION_ARCHITECTURE,
    )
    other = tiny_vision_tensors(config, seed=987654321)[CONNECTOR_WEIGHT]
    root = tmp_path_factory.mktemp("tiny-vlm-other") / "tiny-smolvlm-other-weights"
    build_tiny_vision_model(root, overrides={CONNECTOR_WEIGHT: other})
    return root


def test_swapping_the_model_does_not_serve_the_previous_models_features(
    tiny_vlm_engine, other_weights_vlm_dir: Path
):
    """A real model swap, not a poked fingerprint: the cache must miss.

    Reproduced against the unfixed code. ``_weights_fingerprint`` was read once in
    ``__init__`` under the comment "the manifest never changes under a loaded
    engine", so after ``engine.model = other`` the key still carried the *old*
    fingerprint, the same image hit the same entry, and the engine fused the new
    decoder with the previous model's image features. Nothing raised: the features
    had the right shape and the right row count, so the prompt looked fine and the
    generation was confidently wrong.
    """
    engine = tiny_vlm_engine
    image = hand_built_prompt().image

    first, _, first_hit = engine._vision_features(image, probe=None)
    assert not first_hit
    cached, _, cached_hit = engine._vision_features(image, probe=None)
    assert cached_hit, "an unswapped engine must still hit — the fix is not 'never cache'"
    np.testing.assert_array_equal(cached, first)

    other = SmolVLMModel.from_directory(other_weights_vlm_dir)
    assert other.weights.fingerprint() != engine.model.weights.fingerprint()
    expected = other.encode_image(image).features
    assert not np.array_equal(expected, first), "the fixture must change the features"

    engine.model = other
    after, _, after_hit = engine._vision_features(image, probe=None)

    assert not after_hit, "the swapped model was served the previous model's features"
    assert not np.array_equal(after, first)
    np.testing.assert_array_equal(after, expected)


def test_a_swapped_engine_rewarms_and_then_hits_again(
    tiny_vlm_engine, other_weights_vlm_dir: Path
):
    """Invalidation costs one recompute, not the cache itself.

    If the response to a changed model were "stop caching", the second request
    would miss too and every later one with it. The contract is that the stale
    entries go away and the new model's features take their place.
    """
    engine = tiny_vlm_engine
    image = hand_built_prompt().image
    engine._vision_features(image, probe=None)

    engine.model = SmolVLMModel.from_directory(other_weights_vlm_dir)
    _, _, miss = engine._vision_features(image, probe=None)
    _, _, hit = engine._vision_features(image, probe=None)

    assert not miss and hit


def test_a_swap_drops_the_unreachable_entries(tiny_vlm_engine, other_weights_vlm_dir: Path):
    """Old entries must not linger and hold capacity the new model cannot use.

    A changed fingerprint makes every stored key unreachable, so leaving them in
    place would fill the budget with garbage and evict the new model's live
    entries — the cache would appear to work while hitting nothing. Two images
    are needed to see it: with one, "dropped the stale entry and stored a fresh
    one" and "kept the stale entry and stored nothing" both end at a length of
    one, and only the contents would tell them apart.
    """
    engine = tiny_vlm_engine
    first_image = hand_built_prompt(num_frames=3).image
    # A second *image*, not a second frame count: ``image_span`` builds exactly
    # 20 ids, so a 4-frame prompt has no token sequence to go with it. Same
    # geometry and different pixels is the case that matters here anyway — two
    # entries competing for the same budget.
    second_image = prepared(frames_of(0.5, 0.25, -0.5, num_frames=3))
    engine._vision_features(first_image, probe=None)
    engine._vision_features(second_image, probe=None)
    assert len(engine.image_features) == 2

    engine.model = SmolVLMModel.from_directory(other_weights_vlm_dir)
    engine._vision_features(first_image, probe=None)

    assert len(engine.image_features) == 1, "a stale entry survived the swap"


def test_the_identity_check_reads_the_manifest_and_no_tensor(
    tiny_vlm_engine, monkeypatch
):
    """Detecting a swap must not cost a scan of the weights it protects.

    Hashing the tensors to key a cache that exists to skip one forward would cost
    more than the forward it saves — the same reason ``Weights.fingerprint`` reads
    the manifest and never the arrays. This pins that down as behaviour rather
    than as a docstring promise: a cache **hit** must touch zero tensors, so the
    check cannot have quietly become a content hash. It passed before the fix too,
    where it was vacuous; it exists to constrain the fix, not to reproduce the
    bug.
    """
    engine = tiny_vlm_engine
    image = hand_built_prompt().image
    engine._vision_features(image, probe=None)

    reads: list[str] = []
    weights = engine.model.weights
    for name in ("get", "expect"):
        real = getattr(weights, name)

        def counted(*args, _real=real, _name=name, **kwargs):
            reads.append(f"{_name}{args[:1]}")
            return _real(*args, **kwargs)

        monkeypatch.setattr(weights, name, counted)

    _, _, hit = engine._vision_features(image, probe=None)
    assert hit
    assert reads == [], f"the cache-hit path read tensors: {reads}"


def test_a_probe_still_bypasses_the_cache_after_a_swap(
    tiny_vlm_engine, other_weights_vlm_dir: Path
):
    """The debug path is unchanged by identity handling, and must stay so.

    A probe means "show me what *this* request computed", which is why it neither
    reads nor writes the cache. A swap makes that more true, not less, so the two
    mechanisms have to compose rather than one short-circuiting the other.
    """
    engine = tiny_vlm_engine
    image = hand_built_prompt().image
    engine.model = SmolVLMModel.from_directory(other_weights_vlm_dir)

    probe = DebugProbe(captures=frozenset())
    features, _, hit = engine._vision_features(image, probe=probe)
    assert not hit
    np.testing.assert_array_equal(features, engine.model.encode_image(image).features)
    assert len(engine.image_features) == 0, "a probed request must not populate the cache"


# --------------------------------------------------------------------------- #
# the config behind a live engine changing, with the weights held still
#
# ``model_over_same_weights`` lives with the other helpers above; everything here
# uses it to hold the manifest fingerprint exactly still while the model changes.
# --------------------------------------------------------------------------- #


def test_a_swapped_vision_config_over_the_same_weights_does_not_serve_the_old_features(
    tiny_vlm_engine,
):
    """Same checkpoint, different ``layer_norm_eps``, and the cache must miss.

    Reproduced against the unfixed code, where ``_model_identity_changed`` compared
    ``weights.fingerprint()`` and returned early on a match. The manifest is the same
    manifest, so it matched: the old entry was served to the new tower's caller, the
    features differed from what ``other.encode_image`` produces by up to 0.163 per
    element, and ``engine.vlm_config`` was still the previous one. Every shape and
    every row count lined up, so the request completed and was confidently wrong.
    """
    engine = tiny_vlm_engine
    image = hand_built_prompt().image

    first, _, first_hit = engine._vision_features(image, probe=None)
    assert not first_hit
    _, _, warm = engine._vision_features(image, probe=None)
    assert warm, "the cache must be warm, or a miss below proves nothing"

    original = engine.model
    other = model_over_same_weights(
        original,
        vision=dataclasses.replace(original.vlm_config.vision, layer_norm_eps=0.5),
    )
    assert other.weights.fingerprint() == original.weights.fingerprint(), (
        "the fixture must hold the fingerprint still, or this is the weights test"
    )
    expected = other.encode_image(image).features
    assert not np.array_equal(expected, first), "the fixture must change the features"

    engine.model = other
    after, _, after_hit = engine._vision_features(image, probe=None)

    assert not after_hit, (
        "a same-weights/different-config swap was served the previous model's features"
    )
    np.testing.assert_array_equal(after, expected)
    assert engine.vlm_config is other.vlm_config, "the swap left the old config in place"


def test_a_swapped_preprocessor_config_refreshes_the_preprocessor_and_the_entry(
    tiny_vlm_engine,
):
    """The other half of the same bug, and it bites *before* the lookup.

    ``layer_norm_eps`` changes what a cached entry should hold. A normalization mean
    changes what the next request will produce, and it does so upstream of the cache:
    ``build_prompt`` crops with ``self.preprocessor``, so an engine still holding the
    previous model's settings feeds the new tower pixels it was never configured for.
    The key carries the processor config, which makes a stale one *self-consistent* —
    the same wrong settings on both sides of the comparison, so the entry hits.
    """
    engine = tiny_vlm_engine
    image = hand_built_prompt().image
    engine._vision_features(image, probe=None)
    _, _, warm = engine._vision_features(image, probe=None)
    assert warm

    original = engine.model
    processor = dataclasses.replace(
        original.vlm_config.processor,
        image_mean=tuple(v + 0.25 for v in original.vlm_config.processor.image_mean),
    )
    other = model_over_same_weights(original, processor=processor)
    assert other.weights.fingerprint() == original.weights.fingerprint()

    engine.model = other
    _, _, hit = engine._vision_features(image, probe=None)

    assert not hit, "a different preprocessing config reused the previous entry"
    assert engine.vlm_config is other.vlm_config
    assert engine.preprocessor.processor is processor, (
        "the engine is still cropping images to the previous model's settings"
    )


def test_build_prompt_preprocesses_with_the_swapped_models_settings(
    pil, tiny_vlm_engine
):
    """Through the entry point the CLI uses, with a real image and a real crop.

    The ordering is the guarantee: ``build_prompt`` re-reads the identity *before* it
    calls ``self.preprocessor.prepare``. Reversed, the first request after a swap
    would be cropped by the previous model and only the second would be right — a
    one-request window that no test counting cache hits would ever see.
    """
    engine = tiny_vlm_engine
    image = gradient_image(pil, *WIDE_IMAGE)
    engine.build_prompt(image, QUESTION)

    original = engine.model
    processor = dataclasses.replace(
        original.vlm_config.processor,
        image_mean=tuple(v + 0.25 for v in original.vlm_config.processor.image_mean),
    )
    engine.model = model_over_same_weights(original, processor=processor)

    prompt = engine.build_prompt(image, QUESTION)
    independent = ImagePreprocessor(processor).prepare(image)
    np.testing.assert_array_equal(prompt.image.pixel_values, independent.pixel_values)
    assert not np.array_equal(
        prompt.image.pixel_values,
        ImagePreprocessor(original.vlm_config.processor).prepare(image).pixel_values,
    ), "the fixture must change the pixels, or this asserts nothing"


def test_a_config_swap_rewarms_and_then_hits_again(tiny_vlm_engine):
    """Invalidation costs one recompute, not the cache.

    The config half of ``test_a_swapped_engine_rewarms_and_then_hits_again``: if the
    response to a changed config were "stop trusting the cache", every later request
    would miss too and the feature would be gone rather than corrected.
    """
    engine = tiny_vlm_engine
    image = hand_built_prompt().image
    engine._vision_features(image, probe=None)

    original = engine.model
    engine.model = model_over_same_weights(
        original,
        vision=dataclasses.replace(original.vlm_config.vision, layer_norm_eps=0.5),
    )
    _, _, miss = engine._vision_features(image, probe=None)
    _, _, hit = engine._vision_features(image, probe=None)

    assert not miss and hit


def test_requests_without_a_swap_keep_hitting(tiny_vlm_engine):
    """The identity check must not itself invalidate.

    A check that fired on every request would be invisible from the outside — same
    answers, same features — while paying for the tower each time, which is the one
    cost this cache exists to remove. So this counts hits rather than timings: a
    timing assertion would pass on a machine that was merely fast.
    """
    engine = tiny_vlm_engine
    image = hand_built_prompt().image

    _, _, first = engine._vision_features(image, probe=None)
    hits = [engine._vision_features(image, probe=None)[2] for _ in range(4)]

    assert not first and hits == [True, True, True, True]
    assert engine.image_features.stats()["entries"] == 1
    assert engine.vlm_config is engine.model.vlm_config


def test_a_text_only_model_is_refused_rather_than_served_from_the_cache(
    tiny_vlm_engine, tiny_model_dir: Path
):
    """An unsupported replacement raises instead of quietly returning stale features.

    ``VisionEngine.model`` is an ordinary attribute, so any object can be assigned to
    it. A text-only ``CausalLMModel`` has no ``vlm_config`` and cannot encode an image;
    the only honest responses are to refuse or to invent an answer. Inventing one here
    means serving the *previous* model's features for an image the new model has no
    tower for, which is the failure this whole section exists to prevent.

    ``TypeError`` and not a ``slowq.errors`` type: this is a caller binding the wrong
    kind of object, not a bad checkpoint or a bad request.
    """
    engine = tiny_vlm_engine
    image = hand_built_prompt().image
    engine._vision_features(image, probe=None)

    engine.model = CausalLMModel.from_directory(tiny_model_dir)
    with pytest.raises(TypeError) as excinfo:
        engine._vision_features(image, probe=None)
    assert "vlm_config" in str(excinfo.value)
