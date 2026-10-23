"""SmolVLM (``idefics3``) image+text generation: config, preprocessing, vision tower.

Everything here was read off the official checkpoint
``HuggingFaceTB/SmolVLM-256M-Instruct`` at commit
``7e3e67edbbed1bf9888184d9df282b700a323964`` and the ``transformers`` 4.51.3
``Idefics3*`` modules it dispatches to — not inferred from the name. The facts
that shape this file:

* The top-level config is ``model_type: "idefics3"`` and nests a **llama**
  ``text_config`` beside a SigLIP-derived ``vision_config``. The text decoder is
  therefore the same :class:`slowq.model.DecoderModel` SlowQ already runs, with
  ``tensor_prefix="model.text_model"``; only the vision tower and the connector
  are new code.
* The vision tower shares *nothing* numerically with the decoder: ``nn.LayerNorm``
  with weight **and** bias instead of RMSNorm, ``gelu_pytorch_tanh`` instead of
  SiLU, no RoPE (a learned per-patch position embedding), and **non-causal**
  attention where every patch sees every other patch.
* One image is not one frame. ``Idefics3ImageProcessor`` resizes the input so
  both sides are multiples of 512, cuts it into 512x512 crops row-major, and
  appends the whole image resized to 512x512 as a final "global" frame. Each
  frame becomes ``(512/16)**2 = 1024`` patches, and the connector's pixel shuffle
  folds ``scale_factor**2 = 16`` of them into one image token, so a frame is
  ``image_seq_len = 64`` tokens. A 4x4 split is therefore 17 frames and 1088
  image tokens, and skipping the crops would be silently throwing away most of
  the image.
* ``inputs_merger`` writes the connector's rows, flattened in frame order, onto
  the positions where ``input_ids == image_token_id``. Order matters and the
  count must match exactly; :meth:`slowq.model.DecoderModel._merge_inputs_embeds`
  refuses a mismatch instead of dropping features.

Pillow is an *optional* dependency (``pip install slowq[vision]``). It is imported
inside :func:`_pillow` so that importing this module — and the rest of SlowQ —
still works in a text-only environment, and so that a missing Pillow produces a
message naming the extra rather than a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import (
    FAMILY_IDEFICS3,
    TASK_IMAGE_TEXT_GENERATION,
    GenerationConfig,
    ModelConfig,
)
from .engine import DEFAULT_MAX_CONTEXT, Engine, GenerationResult, GenerationStep
from .errors import ConfigError, ContextOverflowError, UnsupportedInputError, WeightError
from .metrics import stopwatch
from .model import CausalLMModel, DebugProbe
from .ops import (
    conv2d_patchify,
    gelu_tanh,
    group_query_attention,
    layer_norm,
    linear,
    merge_heads,
    pixel_shuffle,
    split_heads,
)
from .sampling import SamplingParams
from .tokenizer import Tokenizer
from .weights import Weights

#: Where the nested decoder's tensors live. The checkpoint publishes the vision
#: tower and the connector as siblings under ``model``.
TEXT_TENSOR_PREFIX = "model.text_model"
VISION_TENSOR_PREFIX = "model.vision_model"
CONNECTOR_TENSOR_PREFIX = "model.connector"

#: ``Idefics3ForConditionalGeneration``'s own architecture name.
SUPPORTED_ARCHITECTURES = ("Idefics3ForConditionalGeneration",)

#: The processor's vocabulary of structural tags. ``<image>`` is the placeholder
#: that later carries a connector row; the rest are ordinary text tokens that
#: tell the model where a crop starts and that the global frame follows.
FAKE_IMAGE_TOKEN = "<fake_token_around_image>"
IMAGE_TOKEN = "<image>"
GLOBAL_IMAGE_TAG = "<global-img>"
END_OF_UTTERANCE_TOKEN = "<end_of_utterance>"

#: ``MAX_IMAGE_SIZE`` in ``image_processing_idefics3.py``. Only consulted after
#: the ``size["longest_edge"]`` resize, which for this checkpoint caps at 2048,
#: so it never fires — but reproducing it is what makes the resize a statement of
#: the official pipeline rather than an approximation that happens to agree.
VISION_UPPER_BOUND = 4096

#: ``preprocessor_config.json`` records ``resample: 1``, which is
#: ``PILImageResampling.LANCZOS``. SlowQ hands the value straight to Pillow, so
#: any filter Pillow knows is reproduced exactly; an unknown one is refused.
SUPPORTED_RESAMPLE = (0, 1, 2, 3, 4, 5)

#: Image modes SlowQ reads. ``convert_to_rgb`` in the official processor turns a
#: 3- or 4-channel uint8 array into RGB by alpha-compositing over white, which is
#: exactly what these two modes give. Palette, grayscale and 16-bit inputs take a
#: different route through ``to_pil_image`` that SlowQ has not verified against
#: the reference, so they are refused by name rather than guessed at.
SUPPORTED_IMAGE_MODES = ("RGB", "RGBA")


def _pillow() -> Any:
    """Import Pillow on first use, with an actionable message if it is absent."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise UnsupportedInputError(
            "reading an image needs Pillow, which is an optional dependency so "
            "that text-only installs stay small. Install it with "
            "`pip install 'slowq[vision]'`, or pass an already-loaded "
            "PIL.Image.Image."
        ) from exc
    return Image


# ---- configuration -----------------------------------------------------------


@dataclass(frozen=True)
class VisionConfig:
    """The SigLIP-derived vision tower, read from ``config.json["vision_config"]``."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    image_size: int
    patch_size: int
    layer_norm_eps: float
    hidden_act: str
    num_channels: int

    #: ``Idefics3VisionAttention`` derives this and raises if it does not divide.
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_per_side**2

    @property
    def attention_scaling(self) -> float:
        return self.head_dim**-0.5

    @classmethod
    def from_hf_config(cls, raw: Mapping[str, Any], where: str) -> "VisionConfig":
        if raw.get("hidden_act") != "gelu_pytorch_tanh":
            raise ConfigError(
                f"{where}: hidden_act={raw.get('hidden_act')!r}; the vision MLP uses "
                "the tanh approximation of GELU, and SlowQ implements only that one."
            )
        ints: dict[str, int] = {}
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "image_size",
            "patch_size",
            "num_channels",
        ):
            value = raw.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigError(f"{where}: {key} must be a positive int, got {value!r}")
            ints[key] = value
        eps = raw.get("layer_norm_eps")
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ConfigError(f"{where}: layer_norm_eps must be > 0, got {eps!r}")

        config = cls(
            hidden_size=ints["hidden_size"],
            intermediate_size=ints["intermediate_size"],
            num_hidden_layers=ints["num_hidden_layers"],
            num_attention_heads=ints["num_attention_heads"],
            image_size=ints["image_size"],
            patch_size=ints["patch_size"],
            layer_norm_eps=float(eps),
            hidden_act=str(raw["hidden_act"]),
            num_channels=ints["num_channels"],
        )
        if config.hidden_size % config.num_attention_heads:
            raise ConfigError(
                f"{where}: hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads}); "
                "Idefics3VisionAttention raises on the same condition."
            )
        if config.image_size % config.patch_size:
            raise ConfigError(
                f"{where}: image_size ({config.image_size}) must be a whole number of "
                f"patch_size ({config.patch_size}) patches."
            )
        if config.num_channels != 3:
            raise ConfigError(
                f"{where}: num_channels={config.num_channels}; SlowQ's image reader "
                "produces RGB, so a different channel count has no input to match."
            )
        return config


@dataclass(frozen=True)
class ImageProcessorConfig:
    """``preprocessor_config.json`` + ``processor_config.json``, validated.

    Every field is read from the converted directory. The defaults below are the
    values ``Idefics3ImageProcessor.__init__`` would fall back to, and they are
    *not* this checkpoint's: SmolVLM-256M records ``size.longest_edge=2048`` and
    ``max_image_size.longest_edge=512`` where the class defaults are 1456 and 364.
    Reading the file is what keeps a differently-configured checkpoint from being
    resized with numbers that belong to another one.
    """

    size_longest_edge: int = 4 * 364
    max_image_size_longest_edge: int = 364
    do_convert_rgb: bool = True
    do_resize: bool = True
    do_image_splitting: bool = True
    do_rescale: bool = True
    rescale_factor: float = 1 / 255
    do_normalize: bool = True
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5)
    do_pad: bool = True
    resample: int = 1
    #: ``processor_config.json``'s ``image_seq_len``: how many ``<image>`` tokens
    #: one frame expands to in the prompt.
    image_seq_len: int = 169

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "ImageProcessorConfig":
        root = Path(model_dir)
        preprocessor_path = root / "preprocessor_config.json"
        processor_path = root / "processor_config.json"
        if not preprocessor_path.is_file():
            raise ConfigError(
                f"{preprocessor_path} not found. An image+text model must carry its "
                "processor settings into the converted directory, or the pixel "
                "preprocessing cannot be reproduced offline."
            )
        raw = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        processor_raw = (
            json.loads(processor_path.read_text(encoding="utf-8"))
            if processor_path.is_file()
            else {}
        )
        where = "preprocessor_config.json"

        if raw.get("image_processor_type") != "Idefics3ImageProcessor":
            raise ConfigError(
                f"{where}: image_processor_type={raw.get('image_processor_type')!r}; "
                "SlowQ reproduces the Idefics3 pipeline only."
            )
        for key in ("do_convert_rgb", "do_resize", "do_image_splitting", "do_rescale",
                    "do_normalize", "do_pad"):
            value = raw.get(key, True)
            if value is not True:
                raise ConfigError(
                    f"{where}: {key}={value!r}. SlowQ implements the official "
                    "single-image pipeline with every step enabled; a checkpoint "
                    "that turns one off would need a different path, and guessing "
                    "which would silently change the pixels."
                )

        size = raw.get("size") or {}
        max_image_size = raw.get("max_image_size") or {}
        longest = size.get("longest_edge")
        patch_side = max_image_size.get("longest_edge")
        if not isinstance(longest, int) or longest <= 0:
            raise ConfigError(f"{where}: size.longest_edge must be a positive int, got {longest!r}")
        if not isinstance(patch_side, int) or patch_side <= 0:
            raise ConfigError(
                f"{where}: max_image_size.longest_edge must be a positive int, got {patch_side!r}"
            )

        resample = raw.get("resample", 1)
        if resample not in SUPPORTED_RESAMPLE:
            raise ConfigError(
                f"{where}: resample={resample!r} is not a Pillow resampling filter "
                f"SlowQ recognises ({list(SUPPORTED_RESAMPLE)})"
            )
        rescale_factor = raw.get("rescale_factor", 1 / 255)
        if not isinstance(rescale_factor, (int, float)) or rescale_factor <= 0:
            raise ConfigError(f"{where}: rescale_factor must be > 0, got {rescale_factor!r}")

        mean = raw.get("image_mean")
        std = raw.get("image_std")
        if not isinstance(mean, list) or len(mean) != 3:
            raise ConfigError(f"{where}: image_mean must be a 3-element list, got {mean!r}")
        if not isinstance(std, list) or len(std) != 3 or any(s == 0 for s in std):
            raise ConfigError(f"{where}: image_std must be 3 non-zero numbers, got {std!r}")

        image_seq_len = processor_raw.get("image_seq_len")
        if not isinstance(image_seq_len, int) or isinstance(image_seq_len, bool) or image_seq_len <= 0:
            raise ConfigError(
                "processor_config.json must record a positive int 'image_seq_len'. "
                "It is the number of <image> tokens one frame expands to, and "
                "without it the prompt cannot be built to match the connector's "
                "output."
            )

        return cls(
            size_longest_edge=int(longest),
            max_image_size_longest_edge=int(patch_side),
            do_convert_rgb=bool(raw["do_convert_rgb"]),
            do_resize=bool(raw["do_resize"]),
            do_image_splitting=bool(raw["do_image_splitting"]),
            do_rescale=bool(raw["do_rescale"]),
            rescale_factor=float(rescale_factor),
            do_normalize=bool(raw["do_normalize"]),
            image_mean=tuple(float(v) for v in mean),
            image_std=tuple(float(v) for v in std),
            do_pad=bool(raw["do_pad"]),
            resample=int(resample),
            image_seq_len=int(image_seq_len),
        )


@dataclass(frozen=True)
class SmolVLMConfig:
    """An ``idefics3`` checkpoint: nested text decoder + vision tower + processor."""

    text: ModelConfig
    vision: VisionConfig
    processor: ImageProcessorConfig
    image_token_id: int
    scale_factor: int
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def family(self) -> str:
        return FAMILY_IDEFICS3

    @property
    def task(self) -> str:
        return TASK_IMAGE_TEXT_GENERATION

    @property
    def computed_image_seq_len(self) -> int:
        """``Idefics3Model.__init__``'s own formula, used to cross-check the file."""
        return int(
            ((self.vision.image_size // self.vision.patch_size) ** 2)
            / (self.scale_factor**2)
        )

    @property
    def connector_out_features(self) -> int:
        return self.vision.hidden_size * self.scale_factor**2

    #: Frames = crops + one global frame, so a 4x4 split is 17 frames.
    @property
    def max_frames_per_image(self) -> int:
        side = self.processor.size_longest_edge // self.processor.max_image_size_longest_edge
        return side * side + 1

    @classmethod
    def from_hf_config(
        cls,
        raw: Mapping[str, Any],
        processor: ImageProcessorConfig,
        where: str = "config.json",
    ) -> "SmolVLMConfig":
        architectures = tuple(raw.get("architectures") or ())
        if raw.get("model_type") != FAMILY_IDEFICS3:
            raise ConfigError(
                f"{where}: model_type={raw.get('model_type')!r}; slowq.vlm loads "
                f"{FAMILY_IDEFICS3!r} checkpoints, which declare "
                f"{list(SUPPORTED_ARCHITECTURES)!r}."
            )
        if architectures and not set(architectures) & set(SUPPORTED_ARCHITECTURES):
            raise ConfigError(
                f"{where}: architectures={list(architectures)!r} does not include "
                f"{SUPPORTED_ARCHITECTURES[0]!r}; SlowQ implements that model's "
                "forward pass and will not assume another wrapper matches it."
            )

        image_token_id = raw.get("image_token_id")
        if not isinstance(image_token_id, int) or isinstance(image_token_id, bool):
            raise ConfigError(f"{where}: image_token_id must be an int, got {image_token_id!r}")
        scale_factor = raw.get("scale_factor")
        if not isinstance(scale_factor, int) or isinstance(scale_factor, bool) or scale_factor < 1:
            raise ConfigError(f"{where}: scale_factor must be a positive int, got {scale_factor!r}")

        text_raw = raw.get("text_config")
        vision_raw = raw.get("vision_config")
        if not isinstance(text_raw, Mapping):
            raise ConfigError(f"{where}: 'text_config' must be an object, got {text_raw!r}")
        if not isinstance(vision_raw, Mapping):
            raise ConfigError(f"{where}: 'vision_config' must be an object, got {vision_raw!r}")

        text = ModelConfig.from_hf_config(
            text_raw,
            task=TASK_IMAGE_TEXT_GENERATION,
            tensor_prefix=TEXT_TENSOR_PREFIX,
            where=f"{where}:text_config",
        )
        vision = VisionConfig.from_hf_config(vision_raw, f"{where}:vision_config")

        config = cls(
            text=text,
            vision=vision,
            processor=processor,
            image_token_id=int(image_token_id),
            scale_factor=int(scale_factor),
            raw=dict(raw),
        )

        # The connector's shapes are a function of both configs; a checkpoint
        # whose recorded image_seq_len disagrees with the geometry would build a
        # prompt the merger cannot satisfy, so it is refused at load time rather
        # than at the first request.
        if processor.image_seq_len != config.computed_image_seq_len:
            raise ConfigError(
                f"processor_config.json records image_seq_len="
                f"{processor.image_seq_len}, but ((image_size {vision.image_size} // "
                f"patch_size {vision.patch_size}) ** 2) / scale_factor**2 = "
                f"{config.computed_image_seq_len}. The prompt would contain the wrong "
                "number of <image> tokens for what the connector produces."
            )
        if vision.image_size != processor.max_image_size_longest_edge:
            raise ConfigError(
                f"vision_config.image_size={vision.image_size} but "
                f"preprocessor_config.json crops to "
                f"max_image_size.longest_edge={processor.max_image_size_longest_edge}. "
                "The learned position embedding is indexed by a "
                f"{vision.num_patches_per_side}x{vision.num_patches_per_side} grid, so "
                "a differently-sized frame has no valid position ids."
            )
        if not 0 <= config.image_token_id < text.vocab_size:
            raise ConfigError(
                f"image_token_id={config.image_token_id} is outside the text "
                f"vocabulary [0, {text.vocab_size})"
            )
        return config

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "SmolVLMConfig":
        root = Path(model_dir)
        config_path = root / "config.json"
        if not config_path.is_file():
            raise ConfigError(
                f"{config_path} not found; is this a converted SlowQ model directory?"
            )
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        weights = Weights.from_directory(root)
        if weights.manifest.task != TASK_IMAGE_TEXT_GENERATION:
            raise ConfigError(
                f"{root} was converted as task={weights.manifest.task!r}, not "
                f"{TASK_IMAGE_TEXT_GENERATION!r}; it has no vision tower to load."
            )
        if weights.manifest.model_family != FAMILY_IDEFICS3:
            raise ConfigError(
                f"{root} records model_family={weights.manifest.model_family!r}; "
                f"slowq.vlm loads {FAMILY_IDEFICS3!r} checkpoints."
            )
        return cls.from_hf_config(raw, ImageProcessorConfig.from_directory(root))

    def describe(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [
            ("family", self.family),
            ("task", self.task),
            ("image_token_id", self.image_token_id),
            ("scale_factor (pixel shuffle)", self.scale_factor),
            ("image_seq_len per frame", self.processor.image_seq_len),
            ("max frames per image", self.max_frames_per_image),
            ("max image tokens per image", self.max_frames_per_image * self.processor.image_seq_len),
            ("connector out_features", self.connector_out_features),
            ("--- vision tower", ""),
            ("vision hidden_size", self.vision.hidden_size),
            ("vision intermediate_size", self.vision.intermediate_size),
            ("vision num_hidden_layers", self.vision.num_hidden_layers),
            ("vision num_attention_heads", self.vision.num_attention_heads),
            ("vision head_dim", self.vision.head_dim),
            ("vision image_size", self.vision.image_size),
            ("vision patch_size", self.vision.patch_size),
            ("vision patches per frame", self.vision.num_patches),
            ("vision layer_norm_eps", self.vision.layer_norm_eps),
            ("vision hidden_act", self.vision.hidden_act),
            ("vision attention", "full (non-causal), no RoPE"),
            ("--- image preprocessing", ""),
            ("resize longest edge", self.processor.size_longest_edge),
            ("crop size", self.processor.max_image_size_longest_edge),
            ("resample (Pillow)", self.processor.resample),
            ("rescale_factor", self.processor.rescale_factor),
            ("image_mean", self.processor.image_mean),
            ("image_std", self.processor.image_std),
            ("--- text decoder", ""),
        ]
        # The nested decoder reports its own family and task, which are *llama*
        # and this same task. Left as they are, they would overwrite the wrapper's
        # "family: idefics3" in any flat rendering of these rows, so the two keys
        # that collide are named for what they describe.
        rows.extend(
            (f"decoder {label}" if label in ("family", "task") else label, value)
            for label, value in self.text.describe()
        )
        return rows


def required_vision_tensors(config: SmolVLMConfig) -> dict[str, tuple[int, ...]]:
    """Vision tower + connector tensors, with the shapes this config implies.

    Kept beside :func:`slowq.weights.required_text_tensors` rather than inside it
    because ``weights.py`` only ever sees a :class:`ModelConfig`, which knows
    nothing about a vision tower.
    """
    v = config.vision
    vp = VISION_TENSOR_PREFIX
    d = v.hidden_size
    expected: dict[str, tuple[int, ...]] = {
        f"{vp}.embeddings.patch_embedding.weight": (
            d,
            v.num_channels,
            v.patch_size,
            v.patch_size,
        ),
        f"{vp}.embeddings.patch_embedding.bias": (d,),
        f"{vp}.embeddings.position_embedding.weight": (v.num_patches, d),
        f"{vp}.post_layernorm.weight": (d,),
        f"{vp}.post_layernorm.bias": (d,),
        f"{CONNECTOR_TENSOR_PREFIX}.modality_projection.proj.weight": (
            config.text.hidden_size,
            config.connector_out_features,
        ),
    }
    for i in range(v.num_hidden_layers):
        layer = f"{vp}.encoder.layers.{i}"
        for norm in ("layer_norm1", "layer_norm2"):
            expected[f"{layer}.{norm}.weight"] = (d,)
            expected[f"{layer}.{norm}.bias"] = (d,)
        expected[f"{layer}.mlp.fc1.weight"] = (v.intermediate_size, d)
        expected[f"{layer}.mlp.fc1.bias"] = (v.intermediate_size,)
        expected[f"{layer}.mlp.fc2.weight"] = (d, v.intermediate_size)
        expected[f"{layer}.mlp.fc2.bias"] = (d,)
        attn = f"{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            # Idefics3VisionAttention declares nn.Linear(embed_dim, embed_dim)
            # for all four, so every one of them has a bias.
            expected[f"{attn}.{projection}.weight"] = (d, d)
            expected[f"{attn}.{projection}.bias"] = (d,)
    return expected


def required_tensors(config: SmolVLMConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor a converted ``idefics3`` directory must contain."""
    from .weights import required_tensors as text_required

    return {**text_required(config.text), **required_vision_tensors(config)}


# ---- image preprocessing -----------------------------------------------------


@dataclass(frozen=True)
class PreparedImage:
    """One image after the official pipeline: frames, mask, and the crop grid.

    ``pixel_values`` is ``[num_frames, 3, height, width]`` FP32 in the
    normalized space the vision tower expects — ``(uint8 / 255 - mean) / std`` —
    matching ``Idefics3ImageProcessor.preprocess``'s output with the batch axis
    squeezed off.
    """

    pixel_values: np.ndarray
    pixel_attention_mask: np.ndarray
    #: Crop grid. ``(0, 0)`` means the image needed no split and is one frame.
    rows: int
    cols: int
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    encoder_size: tuple[int, int]
    image_seq_len: int

    @property
    def num_frames(self) -> int:
        return int(self.pixel_values.shape[0])

    @property
    def num_image_tokens(self) -> int:
        return self.num_frames * self.image_seq_len

    def as_dict(self) -> dict[str, Any]:
        return {
            "pixel_values_shape": list(self.pixel_values.shape),
            "pixel_attention_mask_shape": list(self.pixel_attention_mask.shape),
            "pixel_attention_mask_all_valid": bool(self.pixel_attention_mask.all()),
            "rows": self.rows,
            "cols": self.cols,
            "num_frames": self.num_frames,
            "num_image_tokens": self.num_image_tokens,
            "original_size": list(self.original_size),
            "resized_size": list(self.resized_size),
            "encoder_size": list(self.encoder_size),
        }


def resize_output_size_rescale_to_max_len(height: int, width: int, max_len: int) -> tuple[int, int]:
    """``_resize_output_size_rescale_to_max_len``: longest edge becomes ``max_len``.

    The odd-looking ``if height % 2 != 0: height += 1`` is in the official code
    and is not cosmetic — it changes the aspect ratio of the next resize step, so
    reproducing the arithmetic (float division, ``int()`` truncation, then the
    bump) is the only way to land on the same dimensions.
    """
    aspect_ratio = width / height
    if width >= height:
        width = max_len
        height = int(width / aspect_ratio)
        if height % 2 != 0:
            height += 1
    else:
        height = max_len
        width = int(height * aspect_ratio)
        if width % 2 != 0:
            width += 1
    return max(height, 1), max(width, 1)


def resize_output_size_scale_below_upper_bound(
    height: int, width: int, max_len: int = VISION_UPPER_BOUND
) -> tuple[int, int]:
    """``_resize_output_size_scale_below_upper_bound``, applied after the above."""
    aspect_ratio = width / height
    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
    return max(height, 1), max(width, 1)


def resize_for_vision_encoder_size(height: int, width: int, patch_side: int) -> tuple[int, int]:
    """``resize_for_vision_encoder``'s target: both sides a multiple of 512.

    Aspect ratio is *not* preserved — ``int(width / aspect)`` is rounded up to the
    next multiple, which is exactly what lets the crop grid come out whole.
    """
    aspect_ratio = width / height
    if width >= height:
        width = math.ceil(width / patch_side) * patch_side
        height = int(width / aspect_ratio)
        height = math.ceil(height / patch_side) * patch_side
    else:
        height = math.ceil(height / patch_side) * patch_side
        width = int(height * aspect_ratio)
        width = math.ceil(width / patch_side) * patch_side
    return height, width


def crop_grid(height: int, width: int, patch_side: int) -> tuple[int, int, int, int]:
    """``split_image``'s grid and crop size: ``(rows, cols, crop_h, crop_w)``."""
    if height <= patch_side and width <= patch_side:
        return 0, 0, height, width
    rows = math.ceil(height / patch_side)
    cols = math.ceil(width / patch_side)
    return rows, cols, math.ceil(height / rows), math.ceil(width / cols)


def image_prompt_string(
    rows: int,
    cols: int,
    image_seq_len: int,
) -> str:
    """``get_image_prompt_string``: the text one ``<image>`` placeholder expands to.

    The order is not arbitrary — it is the order the frames come out of
    ``split_image`` (crops row-major, then the global frame), and
    ``inputs_merger`` fills ``<image>`` positions in increasing index order, so
    prompt order *is* frame order.
    """
    if rows == 0 and cols == 0:
        return (
            f"{FAKE_IMAGE_TOKEN}{GLOBAL_IMAGE_TAG}"
            + IMAGE_TOKEN * image_seq_len
            + FAKE_IMAGE_TOKEN
        )
    parts: list[str] = []
    for row in range(rows):
        for col in range(cols):
            parts.append(
                f"{FAKE_IMAGE_TOKEN}<row_{row + 1}_col_{col + 1}>"
                + IMAGE_TOKEN * image_seq_len
            )
        parts.append("\n")
    parts.append(
        f"\n{FAKE_IMAGE_TOKEN}{GLOBAL_IMAGE_TAG}"
        + IMAGE_TOKEN * image_seq_len
        + FAKE_IMAGE_TOKEN
    )
    return "".join(parts)


class ImagePreprocessor:
    """``Idefics3ImageProcessor`` for one image, in Pillow + NumPy.

    Steps, in the official order: resize the longest edge to
    ``size.longest_edge``, resize again so both sides are multiples of the crop
    size, cut into crops plus a global frame, convert each frame to RGB, rescale
    to ``[0, 1]``, normalize, pad, and transpose to channels-first.

    Resizing happens *before* the RGB conversion because that is the official
    order and it is observable: interpolating an RGBA image and then compositing
    over white is not the same as compositing and then interpolating.
    """

    def __init__(self, processor: ImageProcessorConfig) -> None:
        self.processor = processor

    # ---- entry point --------------------------------------------------------

    def prepare(self, image: Any) -> PreparedImage:
        """Run the full pipeline over one image and return its frames."""
        pil = self.load(image)
        original_size = (pil.height, pil.width)

        resized_size = self._resize_longest_edge(original_size)
        pil = self._pil_resize(pil, resized_size)

        encoder_size = resize_for_vision_encoder_size(
            pil.height, pil.width, self.processor.max_image_size_longest_edge
        )
        pil = self._pil_resize(pil, encoder_size)

        frames, rows, cols = self._split(pil)
        arrays = [self._normalize(self._to_rgb(frame)) for frame in frames]
        pixel_values, pixel_attention_mask = self._pad(arrays)

        return PreparedImage(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            rows=rows,
            cols=cols,
            original_size=original_size,
            resized_size=resized_size,
            encoder_size=encoder_size,
            image_seq_len=self.processor.image_seq_len,
        )

    # ---- steps --------------------------------------------------------------

    @staticmethod
    def load(image: Any) -> Any:
        """Accept a path or a PIL image; return an RGB/RGBA PIL image."""
        Image = _pillow()
        if isinstance(image, Image.Image):
            loaded = image
        else:
            path = Path(image) if not isinstance(image, Path) else image
            if not path.is_file():
                raise UnsupportedInputError(
                    f"image file not found: {path}. SlowQ does not fetch URLs; "
                    "download the image first and pass its path."
                )
            try:
                loaded = Image.open(path)
                loaded.load()
            except Exception as exc:
                raise UnsupportedInputError(
                    f"{path} could not be read as an image ({type(exc).__name__}: {exc})"
                ) from exc
        if loaded.mode not in SUPPORTED_IMAGE_MODES:
            raise UnsupportedInputError(
                f"image mode {loaded.mode!r} is not supported; SlowQ reads "
                f"{list(SUPPORTED_IMAGE_MODES)}. Convert the image to RGB first — "
                "palette, grayscale and 16-bit inputs take a different path through "
                "the official processor that SlowQ has not verified."
            )
        return loaded

    def _resize_longest_edge(self, size: tuple[int, int]) -> tuple[int, int]:
        height, width = size
        height, width = resize_output_size_rescale_to_max_len(
            height, width, self.processor.size_longest_edge
        )
        return resize_output_size_scale_below_upper_bound(height, width)

    def _pil_resize(self, image: Any, size: tuple[int, int]) -> Any:
        """PIL resize to exactly ``(height, width)`` — Pillow takes ``(w, h)``."""
        return image.resize((size[1], size[0]), self.processor.resample)

    def _split(self, image: Any) -> tuple[list[np.ndarray], int, int]:
        """``split_image``: row-major crops of a numpy array, then the global frame."""
        array = np.array(image)
        patch_side = self.processor.max_image_size_longest_edge
        height, width = array.shape[:2]
        rows, cols, crop_h, crop_w = crop_grid(height, width, patch_side)
        if rows == 0 and cols == 0:
            return [array], 0, 0

        frames = [
            array[r * crop_h : min(r * crop_h + crop_h, height),
                  c * crop_w : min(c * crop_w + crop_w, width)]
            for r in range(rows)
            for c in range(cols)
        ]
        # "For the global image at the end, we resize it to match the
        # max_image_size" — a real second resize of the whole image, not a copy.
        frames.append(np.array(self._pil_resize(image, (patch_side, patch_side))))
        return frames, rows, cols

    def _to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """``convert_to_rgb``: alpha-composite over white, then drop the alpha."""
        Image = _pillow()
        rgba = Image.fromarray(frame).convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255))
        return np.array(Image.alpha_composite(background, rgba).convert("RGB"))

    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        """``rescale`` then ``normalize``, matching their dtypes.

        ``rescale`` computes in float64 and stores float32; ``normalize`` keeps
        float32 and subtracts float32 constants. Doing both in float32 would move
        the rounding, and the pixel tensor is the one input the vision tower sees
        verbatim.
        """
        scaled = (frame.astype(np.float64) * self.processor.rescale_factor).astype(np.float32)
        mean = np.array(self.processor.image_mean, dtype=np.float32)
        std = np.array(self.processor.image_std, dtype=np.float32)
        return (scaled - mean) / std

    def _pad(self, frames: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Pad to the largest frame and build ``pixel_attention_mask``.

        Padding is with zeros *in the normalized space*, which is what the
        official ``pad`` does because it runs after ``normalize`` — a detail that
        matters to the model, since zero there is mid-grey, not black.

        For a single image every frame is exactly ``max_image_size`` square (the
        previous step made both sides a multiple of it, and the global frame is
        resized to it), so the padding is empty and the mask is all ones. The
        general path is kept because it is what defines the mask the vision tower
        checks against.
        """
        height = max(frame.shape[0] for frame in frames)
        width = max(frame.shape[1] for frame in frames)
        channels = frames[0].shape[2]
        values = np.zeros((len(frames), channels, height, width), dtype=np.float32)
        mask = np.zeros((len(frames), height, width), dtype=np.int64)
        for index, frame in enumerate(frames):
            frame_h, frame_w = frame.shape[:2]
            # (h, w, c) -> (c, h, w), the pipeline's final data_format=FIRST.
            values[index, :, :frame_h, :frame_w] = frame.transpose(2, 0, 1)
            mask[index, :frame_h, :frame_w] = 1
        return values, mask


def patch_attention_mask(pixel_attention_mask: np.ndarray, patch_size: int) -> np.ndarray:
    """``unfold``-ing the pixel mask into patch blocks: a patch is valid if any pixel is.

    ``[num_frames, H, W] -> [num_frames, H/patch, W/patch]`` bool.
    """
    mask = np.asarray(pixel_attention_mask)
    num_frames, height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"pixel mask is {height}x{width}, not a whole number of {patch_size}px patches"
        )
    blocks = mask.reshape(
        num_frames, height // patch_size, patch_size, width // patch_size, patch_size
    )
    return blocks.sum(axis=(2, 4)) > 0


def vision_position_ids(
    patches_h: int, patches_w: int, num_patches_per_side: int
) -> np.ndarray:
    """``Idefics3VisionEmbeddings``' bucketized position ids for one frame.

    ``boundaries = arange(1/n, 1.0, 1/n)`` (n-1 values) and
    ``bucketize(arange(0, 1-1e-6, 1/patches), boundaries, right=True)`` map a
    frame's patch grid onto the ``n x n`` grid the position embedding was trained
    on, so a 512x512 frame and a 1024x512 frame both index the same table.

    Computed in float32 because ``torch.arange`` with float arguments uses the
    default dtype, and for the full ``n x n`` grid this reduces exactly to
    ``arange(n * n)`` — the case SlowQ's single-image path always produces.
    """
    step = np.float32(1.0) / np.float32(num_patches_per_side)
    boundaries = np.arange(step, np.float32(1.0), step, dtype=np.float32)
    fractional_h = np.arange(0, 1 - 1e-6, 1 / patches_h, dtype=np.float32)
    fractional_w = np.arange(0, 1 - 1e-6, 1 / patches_w, dtype=np.float32)
    bucket_h = np.searchsorted(boundaries, fractional_h, side="right")
    bucket_w = np.searchsorted(boundaries, fractional_w, side="right")
    return (bucket_h[:, None] * num_patches_per_side + bucket_w[None, :]).reshape(-1)


# ---- vision tower ------------------------------------------------------------


@dataclass(frozen=True)
class VisionOutput:
    """Connector output for the frames of one image.

    ``features`` is ``[num_frames * image_seq_len, text_hidden_size]``, already
    flattened in frame order — the exact layout ``inputs_merger`` assigns onto the
    ``<image>`` positions.
    """

    features: np.ndarray
    #: ``[num_frames, image_seq_len, text_hidden_size]``, before the flatten.
    per_frame: np.ndarray
    #: ``[num_frames, num_patches, vision_hidden_size]``, after ``post_layernorm``.
    encoder_output: np.ndarray
    encode_seconds: float


class VisionModel:
    """The vision encoder and connector, over a converted weight directory."""

    def __init__(self, config: SmolVLMConfig, weights: Weights) -> None:
        self.config = config
        self.vision = config.vision
        self.weights = weights
        weights.check_names(required_vision_tensors(config))

    # ---- entry point --------------------------------------------------------

    def encode(self, prepared: PreparedImage) -> VisionOutput:
        """Run every frame of one image through the tower and the connector."""
        values = np.asarray(prepared.pixel_values, dtype=np.float32)
        if values.ndim != 4:
            raise ValueError(
                f"pixel_values must be [num_frames, C, H, W], got {values.shape}"
            )
        patch_mask = patch_attention_mask(prepared.pixel_attention_mask, self.vision.patch_size)
        if not patch_mask.all():
            raise UnsupportedInputError(
                "this image produced a partially-padded patch grid, and SlowQ's "
                "single-image pipeline should make that impossible: every frame is "
                f"{self.vision.image_size}px square, so the pixel mask is all ones. "
                f"Got a patch mask with {int((~patch_mask).sum())} invalid patches."
            )
        with stopwatch() as elapsed:
            encoder_output = np.stack(
                [self._encode_frame(values[index]) for index in range(values.shape[0])]
            )
            per_frame = self._connector(encoder_output)
        if per_frame.shape[1] != prepared.image_seq_len:
            raise WeightError(
                f"the connector produced {per_frame.shape[1]} tokens per frame but "
                f"processor_config.json records image_seq_len={prepared.image_seq_len}; "
                "the prompt would not line up with the features."
            )
        num_frames, seq, hidden = per_frame.shape
        return VisionOutput(
            features=per_frame.reshape(num_frames * seq, hidden),
            per_frame=per_frame,
            encoder_output=encoder_output,
            encode_seconds=elapsed[0],
        )

    # ---- internals ----------------------------------------------------------

    def _encode_frame(self, pixels: np.ndarray) -> np.ndarray:
        """``[3, H, W] -> [num_patches, vision_hidden_size]``, post_layernorm applied."""
        v = self.vision
        vp = VISION_TENSOR_PREFIX
        w = self.weights
        patches_h = pixels.shape[1] // v.patch_size
        patches_w = pixels.shape[2] // v.patch_size
        if patches_h != v.num_patches_per_side or patches_w != v.num_patches_per_side:
            raise UnsupportedInputError(
                f"frame is {pixels.shape[1]}x{pixels.shape[2]}px, which is a "
                f"{patches_h}x{patches_w} patch grid; the position embedding only "
                f"covers {v.num_patches_per_side}x{v.num_patches_per_side}. The "
                "preprocessor is supposed to make every frame exactly "
                f"{v.image_size}px square."
            )

        hidden = conv2d_patchify(
            pixels,
            w.expect(
                f"{vp}.embeddings.patch_embedding.weight",
                (v.hidden_size, v.num_channels, v.patch_size, v.patch_size),
            ),
            w.expect(f"{vp}.embeddings.patch_embedding.bias", (v.hidden_size,)),
            v.patch_size,
        )
        position_ids = vision_position_ids(patches_h, patches_w, v.num_patches_per_side)
        table = w.expect(
            f"{vp}.embeddings.position_embedding.weight", (v.num_patches, v.hidden_size)
        )
        hidden = hidden + np.asarray(table[position_ids], dtype=np.float32)

        for layer_idx in range(v.num_hidden_layers):
            hidden = self._encoder_layer(layer_idx, hidden)

        return layer_norm(
            hidden,
            w.expect(f"{vp}.post_layernorm.weight", (v.hidden_size,)),
            w.expect(f"{vp}.post_layernorm.bias", (v.hidden_size,)),
            v.layer_norm_eps,
        )

    def _encoder_layer(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        """``Idefics3EncoderLayer``: pre-LN attention, then pre-LN MLP.

        Non-causal and with no cache: a patch may attend to every other patch of
        its own frame, and the frame is processed in one shot.
        """
        v = self.vision
        prefix = f"{VISION_TENSOR_PREFIX}.encoder.layers.{layer_idx}"
        w = self.weights

        def _norm(name: str, x: np.ndarray) -> np.ndarray:
            return layer_norm(
                x,
                w.expect(f"{prefix}.{name}.weight", (v.hidden_size,)),
                w.expect(f"{prefix}.{name}.bias", (v.hidden_size,)),
                v.layer_norm_eps,
            )

        residual = hidden
        normed = _norm("layer_norm1", hidden)
        attention = self._attention(f"{prefix}.self_attn", normed)
        hidden = residual + attention

        residual = hidden
        normed = _norm("layer_norm2", hidden)
        mlp = linear(
            gelu_tanh(
                linear(
                    normed,
                    w.expect(f"{prefix}.mlp.fc1.weight", (v.intermediate_size, v.hidden_size)),
                    w.expect(f"{prefix}.mlp.fc1.bias", (v.intermediate_size,)),
                )
            ),
            w.expect(f"{prefix}.mlp.fc2.weight", (v.hidden_size, v.intermediate_size)),
            w.expect(f"{prefix}.mlp.fc2.bias", (v.hidden_size,)),
        )
        return residual + mlp

    def _attention(self, prefix: str, x: np.ndarray) -> np.ndarray:
        """``Idefics3VisionAttention``: full attention, all four projections biased."""
        v = self.vision
        w = self.weights
        num_patches = x.shape[0]

        def _project(name: str) -> np.ndarray:
            return split_heads(
                linear(
                    x,
                    w.expect(f"{prefix}.{name}.weight", (v.hidden_size, v.hidden_size)),
                    w.expect(f"{prefix}.{name}.bias", (v.hidden_size,)),
                ),
                v.num_attention_heads,
                v.head_dim,
            )

        q, k, value = _project("q_proj"), _project("k_proj"), _project("v_proj")
        context, _ = group_query_attention(
            q=q,
            k=k,
            v=value,
            num_kv_heads=v.num_attention_heads,
            scaling=v.attention_scaling,
            mask=None,
        )
        if context.shape[1] != num_patches:
            raise AssertionError("vision attention changed the sequence length")
        merged = merge_heads(context, v.num_attention_heads, num_patches, v.head_dim)
        return linear(
            merged,
            w.expect(f"{prefix}.out_proj.weight", (v.hidden_size, v.hidden_size)),
            w.expect(f"{prefix}.out_proj.bias", (v.hidden_size,)),
        )

    def _connector(self, encoder_output: np.ndarray) -> np.ndarray:
        """``Idefics3Connector``: pixel shuffle then the modality projection."""
        shuffled = pixel_shuffle(encoder_output, self.config.scale_factor)
        return linear(
            shuffled,
            self.weights.expect(
                f"{CONNECTOR_TENSOR_PREFIX}.modality_projection.proj.weight",
                (self.config.text.hidden_size, self.config.connector_out_features),
            ),
            # Idefics3SimpleMLP declares nn.Linear(..., bias=False).
            None,
        )


# ---- model and engine --------------------------------------------------------


class SmolVLMModel(CausalLMModel):
    """Text decoder + vision tower + connector, over one converted directory."""

    def __init__(self, config: SmolVLMConfig, weights: Weights) -> None:
        super().__init__(config.text, weights)
        self.vlm_config = config
        self.vision = VisionModel(config, weights)

    @classmethod
    def from_directory(cls, model_dir: str | Path, task: str | None = None) -> "SmolVLMModel":
        config = SmolVLMConfig.from_directory(model_dir)
        return cls(config, Weights.from_directory(model_dir))

    def encode_image(self, prepared: PreparedImage) -> VisionOutput:
        return self.vision.encode(prepared)


@dataclass(frozen=True)
class VisionPrompt:
    """A rendered image+text prompt, before generation."""

    text: str
    token_ids: tuple[int, ...]
    image: PreparedImage
    #: Positions of ``image_token_id`` in ``token_ids``.
    image_positions: tuple[int, ...]


class VisionEngine(Engine):
    """``Engine`` plus one image per request.

    The generation loop is inherited unchanged: :meth:`generate_image` prepares
    the pixels and the expanded prompt, computes the connector's features once,
    and hands both to ``stream_ids`` as the prefill's out-of-table embeddings.
    Decoding afterwards is the ordinary cached text loop, because
    ``Idefics3Model.forward`` only merges image features when
    ``past_seen_tokens == 0``.
    """

    def __init__(
        self,
        model: SmolVLMModel,
        tokenizer: Tokenizer,
        generation_config: GenerationConfig,
        max_context: int = DEFAULT_MAX_CONTEXT,
        prefill_chunk_size: int = 128,
        load_seconds: float = float("nan"),
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            max_context=max_context,
            prefill_chunk_size=prefill_chunk_size,
            load_seconds=load_seconds,
        )
        self.vlm_config = model.vlm_config
        self.preprocessor = ImagePreprocessor(model.vlm_config.processor)

    @property
    def image_token_id(self) -> int:
        return self.vlm_config.image_token_id

    @classmethod
    def from_directory(
        cls,
        model_dir: str | Path,
        *,
        max_context: int = DEFAULT_MAX_CONTEXT,
        prefill_chunk_size: int = 128,
    ) -> "VisionEngine":
        """Load a converted ``idefics3`` directory. Fully offline."""
        with stopwatch() as elapsed:
            root = Path(model_dir)
            model = SmolVLMModel.from_directory(root)
            tokenizer = Tokenizer.from_directory(root)
            generation_config = GenerationConfig.from_json_file(
                root / "generation_config.json", root / "config.json"
            )
        return cls(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            max_context=max_context,
            prefill_chunk_size=prefill_chunk_size,
            load_seconds=elapsed[0],
        )

    # ---- prompt construction ------------------------------------------------

    def build_prompt(
        self,
        image: Any,
        question: str,
        *,
        system: str | None = None,
        history: Sequence[Mapping[str, Any]] = (),
    ) -> VisionPrompt:
        """Preprocess the image, render the template, expand ``<image>``, tokenise.

        The template comes from the checkpoint and is rendered with structured
        content parts, because SmolVLM's template indexes
        ``message['content'][0]['type']`` — a plain-string message would raise
        inside Jinja rather than produce a prompt.
        """
        prepared = self.preprocessor.prepare(image)

        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.extend(dict(message) for message in history)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        )
        rendered = self.tokenizer.render_template(messages, add_generation_prompt=True)
        expanded = expand_image_tokens(rendered, prepared)
        token_ids = self.tokenizer.encode(expanded)

        positions = tuple(
            index for index, token_id in enumerate(token_ids) if token_id == self.image_token_id
        )
        if len(positions) != prepared.num_image_tokens:
            raise UnsupportedInputError(
                f"the expanded prompt tokenised to {len(positions)} {IMAGE_TOKEN} "
                f"tokens but the preprocessor produced {prepared.num_image_tokens} "
                f"({prepared.num_frames} frames x image_seq_len "
                f"{prepared.image_seq_len}). The two sides of the merge would not "
                "line up, so SlowQ refuses rather than dropping image features."
            )
        required = len(token_ids)
        if required > self.max_context:
            raise ContextOverflowError(
                f"this image expands to {prepared.num_image_tokens} of the prompt's "
                f"{required} tokens, which already exceeds max_context="
                f"{self.max_context}. A larger image or a wider --max-context is "
                "needed; SlowQ never drops crops to make it fit."
            )
        return VisionPrompt(
            text=expanded,
            token_ids=tuple(token_ids),
            image=prepared,
            image_positions=positions,
        )

    # ---- public API ---------------------------------------------------------

    def generate_image(
        self,
        image: Any,
        question: str,
        *,
        system: str | None = None,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = 64,
    ) -> GenerationResult:
        """Answer ``question`` about one local image."""
        return self._drain(
            self.stream_image(
                image,
                question,
                system=system,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
            )
        )

    def stream_image(
        self,
        image: Any,
        question: str,
        *,
        system: str | None = None,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = 64,
        probe: DebugProbe | None = None,
    ) -> Iterable[GenerationStep]:
        return self.stream_prompt(
            self.build_prompt(image, question, system=system),
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            probe=probe,
        )

    def generate_prompt(
        self,
        prompt: VisionPrompt,
        *,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = 64,
    ) -> GenerationResult:
        """Non-streaming generation from an already-built prompt."""
        return self._drain(
            self.stream_prompt(
                prompt, sampling=sampling, max_new_tokens=max_new_tokens
            )
        )

    def stream_prompt(
        self,
        prompt: VisionPrompt,
        *,
        sampling: SamplingParams | None = None,
        max_new_tokens: int = 64,
        probe: DebugProbe | None = None,
    ) -> Iterable[GenerationStep]:
        """Generate from an already-built prompt.

        Separate from :meth:`stream_image` because building a prompt resizes every
        frame of the image, and a caller that wants the crop/token statistics
        before generating should not pay for that twice.

        The request is validated before the image is encoded, using the same
        checks the text engine runs. :meth:`Engine.stream_ids` is a generator, so
        its own validation would not happen until the first ``next()`` — long
        after the vision tower had run over every crop. Refusing here costs
        nothing and makes an illegal image+text request as cheap to reject as a
        text one.
        """
        self._validate_request(prompt.token_ids, max_new_tokens)
        features = self.model.encode_image(prompt.image)
        mask = np.zeros(len(prompt.token_ids), dtype=bool)
        mask[list(prompt.image_positions)] = True
        return self.stream_ids(
            prompt.token_ids,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            probe=probe,
            prefill_embeds=(features.features, mask),
            vision_encode_seconds=features.encode_seconds,
        )

    def describe(self) -> list[tuple[str, Any]]:
        return self.vlm_config.describe()


def expand_image_tokens(text: str, prepared: PreparedImage) -> str:
    """Splice the expanded image string into every ``<image>`` of ``text``.

    ``Idefics3Processor.__call__`` splits on the placeholder and interleaves the
    per-image expansions; with one image there must be exactly one placeholder,
    and a prompt with none would silently drop the image.
    """
    parts = text.split(IMAGE_TOKEN)
    if len(parts) - 1 != 1:
        raise UnsupportedInputError(
            f"the rendered prompt contains {len(parts) - 1} {IMAGE_TOKEN} "
            "placeholders, and SlowQ's image+text path takes exactly one image. "
            "Multi-image prompts are out of scope for this version."
        )
    expansion = image_prompt_string(
        prepared.rows, prepared.cols, prepared.image_seq_len
    )
    return parts[0] + expansion + parts[1]


def load_config_and_weights(model_dir: str | Path) -> tuple[SmolVLMConfig, Weights]:
    """``slowq.model.load_config_and_weights`` for an ``idefics3`` directory."""
    return SmolVLMConfig.from_directory(model_dir), Weights.from_directory(model_dir)


def recorded_image_token_id(model_dir: str | Path) -> int:
    """``image_token_id`` from a converted directory, without loading weights."""
    root = Path(model_dir)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ConfigError(f"{config_path} not found; is this a converted model directory?")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    value = raw.get("image_token_id")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{config_path}: image_token_id must be an int, got {value!r}")
    return int(value)


__all__ = [
    "CONNECTOR_TENSOR_PREFIX",
    "END_OF_UTTERANCE_TOKEN",
    "FAKE_IMAGE_TOKEN",
    "GLOBAL_IMAGE_TAG",
    "IMAGE_TOKEN",
    "ImagePreprocessor",
    "ImageProcessorConfig",
    "PreparedImage",
    "SmolVLMConfig",
    "SmolVLMModel",
    "TEXT_TENSOR_PREFIX",
    "VISION_TENSOR_PREFIX",
    "VisionConfig",
    "VisionEngine",
    "VisionModel",
    "VisionOutput",
    "VisionPrompt",
    "crop_grid",
    "expand_image_tokens",
    "image_prompt_string",
    "load_config_and_weights",
    "patch_attention_mask",
    "recorded_image_token_id",
    "required_tensors",
    "required_vision_tensors",
    "resize_for_vision_encoder_size",
    "resize_output_size_rescale_to_max_len",
    "resize_output_size_scale_below_upper_bound",
    "vision_position_ids",
]
