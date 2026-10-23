"""Shared fixtures: a miniature but structurally faithful Qwen3 model directory.

The fast tests never download weights. Instead they build a ~40 KiB checkpoint on
disk with the exact layout ``tools/convert_model.py`` produces (``manifest.json``
plus one ``.npy`` per tensor) and a real byte-level BPE tokenizer trained on a
few Chinese/emoji/English sentences, so the loader, the mmap path, the chat
template and the incremental decoder are all exercised for real.

The tiny config keeps Qwen3's awkward dimension relationships rather than
simplifying them:

* ``head_dim * num_attention_heads != hidden_size`` (8 * 4 = 32 != 16), so any
  code that infers ``head_dim`` from ``hidden_size`` fails here.
* ``num_attention_heads / num_key_value_heads == 2``, the same GQA group size as
  Qwen3-0.6B.
* ``tie_word_embeddings = true``, so ``lm_head.weight`` exists only as an alias.

A second, equally miniature ``idefics3`` directory (see
:func:`build_tiny_vision_model`) does the same for the image+text task: a nested
llama decoder under ``model.text_model``, a SigLIP-shaped vision tower whose
geometry divides exactly, and SmolVLM's structural tags in the vocabulary. Its
pixel sizes are 8px rather than 512px so that a crop grid, a frame count and an
image-token count all stay small enough to work out by hand.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from tokenizers import AddedToken
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

#: Deliberately *not* the official Qwen3 template, but it exercises the same
#: features: string concatenation, the ``enable_thinking`` branch, and
#: ``add_generation_prompt``. Rendering fidelity against the real template is
#: checked by tools/validate_reference.py.
TINY_CHAT_TEMPLATE = """\
{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] }}
{%- if message['role'] == 'assistant' and enable_thinking is defined and enable_thinking is false %}
{{- '\\n<think>\\n\\n</tool_response>' }}
{%- endif %}
{{- '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- endif %}"""

TINY_CORPUS = [
    "你好，世界！SlowQ 用 CPU 慢慢炖出 token。",
    "hello world, the quick brown fox jumps over the lazy dog",
    "🐢🍲 慢炖 emoji test 😀🚀 中文测试",
] * 4

#: Every marker the template emits must be registered with ``special=True``;
#: that is what lets ``decode(skip_special_tokens=True)`` strip it, exactly as
#: the official tokenizer.json strips a trailing stop token. The chat markers are
#: scraped from the template itself so the two can never drift apart.
SPECIAL_TOKENS = [
    *sorted(set(re.findall(r"<\|[A-Za-z_]+\|>", TINY_CHAT_TEMPLATE))),
    "<|fim_middle|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "@Repository",
]

TINY_ARCHITECTURE = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 256,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
    "attention_bias": False,
    "use_sliding_window": False,
}

#: The ``idefics3`` wrapper: ``scale_factor=2`` over the 4x4 patch grid below
#: folds 2x2 patches into one image token, so a frame is 4 tokens and the
#: connector maps ``4 * 8 = 32`` vision features onto the decoder's 16
#: dimensions. Every one of those numbers is asserted in tests/test_vlm.py.
TINY_VISION_WRAPPER = {
    "model_type": "idefics3",
    "architectures": ["Idefics3ForConditionalGeneration"],
    "scale_factor": 2,
    "tie_word_embeddings": False,
}

#: SmolVLM's nested decoder is an ordinary llama config — including its own
#: ``VLlama3ForCausalLM`` architecture name and its explicit ``head_dim``, which
#: keeps ``num_attention_heads * head_dim != hidden_size`` true here too.
TINY_VISION_TEXT_ARCHITECTURE = {
    "model_type": "llama",
    "architectures": ["VLlama3ForCausalLM"],
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-5,
    "rope_theta": 100000.0,
    "rope_interleaved": False,
    "qk_layer_norms": False,
    "max_position_embeddings": 256,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "pixel_shuffle_factor": 2,
}

#: 8px frames of 2px patches: a 4x4 grid, 16 patches, ``head_dim = 8 // 2 = 4``.
TINY_VISION_ARCHITECTURE = {
    "model_type": "idefics3",
    "image_size": 8,
    "patch_size": 2,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_channels": 3,
    "layer_norm_eps": 1e-6,
    "hidden_act": "gelu_pytorch_tanh",
}

TINY_VISION_PREPROCESSOR = {
    "image_processor_type": "Idefics3ImageProcessor",
    "processor_class": "Idefics3Processor",
    "do_convert_rgb": True,
    "do_resize": True,
    "do_image_splitting": True,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "do_pad": True,
    "resample": 1,
    # 16 over an 8px crop is a 2x2 grid, so the widest image a test can build
    # gives 4 crops + 1 global frame = 5 frames = 20 image tokens.
    "size": {"longest_edge": 16},
    "max_image_size": {"longest_edge": 8},
}

#: Must equal ``((image_size // patch_size) ** 2) / scale_factor ** 2``; the
#: config loader refuses a directory where it does not.
TINY_VISION_PROCESSOR = {"image_seq_len": 4, "processor_class": "Idefics3Processor"}

#: SmolVLM's template indexes ``message['content'][0]['type']``, so content is a
#: list of parts and a plain string raises inside Jinja. Mirroring that is what
#: makes ``VisionEngine.build_prompt``'s structured parts a requirement under
#: test rather than a convention nobody checks.
TINY_VISION_CHAT_TEMPLATE = (
    "<|fim_prefix|>"
    "{% for message in messages %}"
    "{{ message['role'] | capitalize }}"
    "{% if message['content'][0]['type'] == 'image' %}{{ ':' }}{% else %}{{ ': ' }}{% endif %}"
    "{% for line in message['content'] %}"
    "{% if line['type'] == 'text' %}{{ line['text'] }}"
    "{% elif line['type'] == 'image' %}{{ '<image>' }}"
    "{% endif %}"
    "{% endfor %}"
    "<end_of_utterance>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)

#: ``<image>`` must be one special token: the vision path counts ids equal to
#: ``image_token_id``, and a placeholder that tokenised to byte pieces would
#: count as zero. The ``<row_r_col_c>`` tags cover every crop grid the tiny
#: geometry can produce, so they are single tokens too, as in the real vocab.
TINY_VISION_SPECIAL_TOKENS = [
    "<image>",
    "<fake_token_around_image>",
    "<global-img>",
    "<end_of_utterance>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    *(f"<row_{row}_col_{col}>" for row in (1, 2, 3) for col in (1, 2, 3)),
]


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #


def _write_json(root: Path, name: str, payload: Mapping) -> None:
    (root / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _train_tiny_bpe(root: Path, special_tokens: Sequence[str]) -> dict[str, int]:
    """Train a byte-level BPE over ``TINY_CORPUS``; return the special token ids.

    ``special=True`` matters twice over: it is what makes
    ``decode(skip_special_tokens=True)`` strip a chat marker from generated text,
    as the real tokenizers do, and what makes a structural tag such as ``<image>``
    encode to one id instead of a run of byte pieces.
    """
    tokenizer = HFTokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        show_progress=False,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=[
            AddedToken(token, special=True, normalized=False) for token in special_tokens
        ],
    )
    tokenizer.train_from_iterator(TINY_CORPUS, trainer)
    tokenizer.save(str(root / "tokenizer.json"))
    return {token: tokenizer.token_to_id(token) for token in special_tokens}


def build_tiny_tokenizer(root: Path) -> dict[str, int]:
    """Train a byte-level BPE tokenizer and write it plus its config files."""
    ids = _train_tiny_bpe(root, SPECIAL_TOKENS)
    _write_json(
        root,
        "tokenizer_config.json",
        {
            "chat_template": TINY_CHAT_TEMPLATE,
            "bos_token": None,
            # Qwen3 stops on <|fim_middle|>; <|fim_prefix|> is the other published eos id.
            "eos_token": "<|fim_middle|>",
            "pad_token": "<|fim_middle|>",
            "unk_token": None,
            "add_bos_token": False,
        },
    )
    _write_json(
        root,
        "generation_config.json",
        {
            # Mirrors Qwen3-0.6B, which publishes two stop ids.
            "eos_token_id": [ids["<|fim_middle|>"], ids["<|fim_prefix|>"]],
            "pad_token_id": ids["<|fim_middle|>"],
            "bos_token_id": None,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
            "do_sample": False,
        },
    )
    return ids


def build_tiny_vision_tokenizer(root: Path) -> dict[str, int]:
    """The same BPE with SmolVLM's structural tags, and its own stop token.

    SmolVLM ends a turn with ``<end_of_utterance>`` rather than ``<|fim_middle|>``, so
    both ``tokenizer_config.json`` and ``generation_config.json`` name that one.
    """
    ids = _train_tiny_bpe(root, TINY_VISION_SPECIAL_TOKENS)
    _write_json(
        root,
        "tokenizer_config.json",
        {
            "chat_template": TINY_VISION_CHAT_TEMPLATE,
            "bos_token": None,
            "eos_token": "<end_of_utterance>",
            "pad_token": "<|fim_middle|>",
            "unk_token": None,
            "add_bos_token": False,
            "additional_special_tokens": list(TINY_VISION_SPECIAL_TOKENS),
        },
    )
    _write_json(
        root,
        "generation_config.json",
        {
            "eos_token_id": ids["<end_of_utterance>"],
            "pad_token_id": ids["<|fim_middle|>"],
            "bos_token_id": None,
            "do_sample": False,
        },
    )
    return ids


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #


def tiny_tensors(
    vocab_size: int,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    seed: int = 20261018,
    prefix: str = "model",
    qk_norm: bool = True,
    untied_head: bool = False,
) -> dict[str, np.ndarray]:
    """Random FP32 tensors keyed by their canonical HF names.

    Projections use ``std=0.02`` like a freshly initialised transformer; norm
    weights use ``N(1, 0.02)`` rather than all-ones so a bug that ignores the
    norm weight cannot hide.

    ``prefix`` is where the decoder sits (``model`` normally, ``model.text_model``
    inside an image+text wrapper), ``qk_norm`` follows the family, and
    ``untied_head`` writes a separate ``lm_head.weight`` instead of leaving it to
    the manifest's alias.
    """
    rng = np.random.default_rng(seed)
    q_out = num_heads * head_dim
    kv_out = num_kv_heads * head_dim

    def matrix(out_features: int, in_features: int) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=(out_features, in_features)).astype(np.float32)

    def norm_weight(size: int) -> np.ndarray:
        return rng.normal(1.0, 0.02, size=(size,)).astype(np.float32)

    tensors = {
        f"{prefix}.embed_tokens.weight": matrix(vocab_size, hidden_size),
        f"{prefix}.norm.weight": norm_weight(hidden_size),
    }
    if untied_head:
        tensors["lm_head.weight"] = matrix(vocab_size, hidden_size)
    for i in range(num_layers):
        p = f"{prefix}.layers.{i}"
        tensors[f"{p}.self_attn.q_proj.weight"] = matrix(q_out, hidden_size)
        tensors[f"{p}.self_attn.k_proj.weight"] = matrix(kv_out, hidden_size)
        tensors[f"{p}.self_attn.v_proj.weight"] = matrix(kv_out, hidden_size)
        tensors[f"{p}.self_attn.o_proj.weight"] = matrix(hidden_size, q_out)
        if qk_norm:
            tensors[f"{p}.self_attn.q_norm.weight"] = norm_weight(head_dim)
            tensors[f"{p}.self_attn.k_norm.weight"] = norm_weight(head_dim)
        tensors[f"{p}.mlp.gate_proj.weight"] = matrix(intermediate_size, hidden_size)
        tensors[f"{p}.mlp.up_proj.weight"] = matrix(intermediate_size, hidden_size)
        tensors[f"{p}.mlp.down_proj.weight"] = matrix(hidden_size, intermediate_size)
        tensors[f"{p}.input_layernorm.weight"] = norm_weight(hidden_size)
        tensors[f"{p}.post_attention_layernorm.weight"] = norm_weight(hidden_size)
    return tensors


def _base_manifest(root: Path, *, format_version: int) -> dict:
    """The manifest fields every tiny directory shares, whatever its task."""
    return {
        "format_version": format_version,
        "created_at": "2026-10-18T00:00:00+0000",
        "tool": "tests/conftest.py",
        "compute_dtype": "float32",
        "source": {
            "repo_id": "tiny:local",
            "revision": "0" * 40,
            "local_path": str(root),
            "files": [{"name": "model.safetensors", "sha256": "0" * 64}],
        },
        "weight_layout": "[out_features, in_features]; runtime computes x @ W.T",
        "tensors": [],
        "aliases": {},
        "tokenizer_files": ["tokenizer.json", "tokenizer_config.json"],
    }


def _write_tensor_files(
    root: Path, tensors: Mapping[str, np.ndarray], omit: Sequence[str] = ()
) -> list[dict]:
    """Save one ``.npy`` per tensor and return their manifest entries."""
    (root / "tensors").mkdir(parents=True, exist_ok=True)
    entries = []
    for name in sorted(tensors):
        if name in omit:
            continue
        array = np.ascontiguousarray(tensors[name], dtype=np.float32)
        relative = f"{name}.npy"
        np.save(root / "tensors" / relative, array)
        entries.append(
            {
                "name": name,
                "file": relative,
                "shape": list(array.shape),
                "dtype": "float32",
                "source_dtype": "BF16",
            }
        )
    return entries


def _write_manifest(root: Path, manifest: Mapping) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_tiny_model(
    root: Path,
    *,
    vocab_size: int,
    omit: tuple[str, ...] = (),
    overrides: dict[str, np.ndarray] | None = None,
    config_overrides: dict | None = None,
    task: str | None = None,
) -> Path:
    """Write a complete converted-model directory and return its path.

    ``task=None`` reproduces the v0.1 directory: format_version 1, no ``task``
    and no ``model_family``, which is what an already-converted Qwen3-0.6B model
    on someone's disk looks like. Naming a task writes the version-2 manifest
    the converter produces today.
    """
    root = Path(root)
    (root / "tensors").mkdir(parents=True, exist_ok=True)
    config = dict(TINY_ARCHITECTURE, vocab_size=vocab_size)
    if config_overrides:
        config.update(config_overrides)
    _write_json(root, "config.json", config)

    tensors = tiny_tensors(vocab_size=vocab_size)
    if overrides:
        tensors.update(overrides)

    manifest = _base_manifest(root, format_version=1 if task is None else 2)
    manifest["tensors"] = _write_tensor_files(root, tensors, omit)
    if config["tie_word_embeddings"] and task != "text-embedding":
        # The converter stores a tied lm_head once and records the alias. An
        # embedding directory has no output projection at all, so there is
        # nothing to alias and requiring one would be a false failure.
        manifest["aliases"]["lm_head.weight"] = "model.embed_tokens.weight"
    if task is not None:
        manifest["task"] = task
        manifest["model_family"] = "qwen3"
    _write_manifest(root, manifest)
    return root


# --------------------------------------------------------------------------- #
# tiny image+text (idefics3) weights
# --------------------------------------------------------------------------- #


def tiny_vision_tensors(config: Mapping, seed: int = 20261019) -> dict[str, np.ndarray]:
    """Random FP32 tensors for the vision tower and connector ``config`` implies.

    Shapes come from the config rather than from defaults, so a test that
    overrides one dimension still gets tensors that match it — a mismatch then
    means something, instead of being fixture noise.

    LayerNorm weights sit near 1 and every bias near 0, which is both realistic
    and the only choice that keeps a two-layer random tower from overflowing:
    unlike the decoder, the vision blocks are pre-LN and un-normalised between
    the residual and the next layer.
    """
    vision = config["vision_config"]
    scale_factor = config["scale_factor"]
    text_hidden = config["text_config"]["hidden_size"]
    dim = vision["hidden_size"]
    patches_per_side = vision["image_size"] // vision["patch_size"]

    rng = np.random.default_rng(seed)

    def matrix(out_features: int, in_features: int) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=(out_features, in_features)).astype(np.float32)

    def norm_weight(size: int) -> np.ndarray:
        return rng.normal(1.0, 0.02, size=(size,)).astype(np.float32)

    def bias(size: int) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=(size,)).astype(np.float32)

    vp = "model.vision_model"
    tensors = {
        f"{vp}.embeddings.patch_embedding.weight": rng.normal(
            0.0,
            0.02,
            size=(dim, vision["num_channels"], vision["patch_size"], vision["patch_size"]),
        ).astype(np.float32),
        f"{vp}.embeddings.patch_embedding.bias": bias(dim),
        f"{vp}.embeddings.position_embedding.weight": matrix(patches_per_side**2, dim),
        f"{vp}.post_layernorm.weight": norm_weight(dim),
        f"{vp}.post_layernorm.bias": bias(dim),
        # Idefics3SimpleMLP is bias-free, so the connector has one weight only.
        "model.connector.modality_projection.proj.weight": matrix(
            text_hidden, dim * scale_factor**2
        ),
    }
    for i in range(vision["num_hidden_layers"]):
        layer = f"{vp}.encoder.layers.{i}"
        for norm in ("layer_norm1", "layer_norm2"):
            tensors[f"{layer}.{norm}.weight"] = norm_weight(dim)
            tensors[f"{layer}.{norm}.bias"] = bias(dim)
        tensors[f"{layer}.mlp.fc1.weight"] = matrix(vision["intermediate_size"], dim)
        tensors[f"{layer}.mlp.fc1.bias"] = bias(vision["intermediate_size"])
        tensors[f"{layer}.mlp.fc2.weight"] = matrix(dim, vision["intermediate_size"])
        tensors[f"{layer}.mlp.fc2.bias"] = bias(dim)
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            # All four are nn.Linear(dim, dim) with a bias in the official module.
            tensors[f"{layer}.self_attn.{projection}.weight"] = matrix(dim, dim)
            tensors[f"{layer}.self_attn.{projection}.bias"] = bias(dim)
    return tensors


def write_tiny_vision_model(
    root: Path,
    *,
    vocab_size: int,
    image_token_id: int,
    config_overrides: dict | None = None,
    text_overrides: dict | None = None,
    vision_overrides: dict | None = None,
    preprocessor_overrides: dict | None = None,
    processor_overrides: dict | None = None,
    omit: tuple[str, ...] = (),
    overrides: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a complete converted image+text directory and return its path.

    ``image_token_id`` is an argument because it has to be the id ``<image>``
    actually has in this directory's vocabulary: the whole vision path is driven
    by counting tokens equal to it, so a fixture that hard-coded a plausible
    number would quietly test nothing.
    """
    root = Path(root)
    text = dict(TINY_VISION_TEXT_ARCHITECTURE, vocab_size=vocab_size)
    if text_overrides:
        text.update(text_overrides)
    vision = dict(TINY_VISION_ARCHITECTURE)
    if vision_overrides:
        vision.update(vision_overrides)
    config = dict(
        TINY_VISION_WRAPPER,
        image_token_id=image_token_id,
        vocab_size=vocab_size,
        text_config=text,
        vision_config=vision,
    )
    if config_overrides:
        config.update(config_overrides)
    _write_json(root, "config.json", config)

    preprocessor = dict(TINY_VISION_PREPROCESSOR)
    if preprocessor_overrides:
        preprocessor.update(preprocessor_overrides)
    _write_json(root, "preprocessor_config.json", preprocessor)
    processor = dict(TINY_VISION_PROCESSOR)
    if processor_overrides:
        processor.update(processor_overrides)
    _write_json(root, "processor_config.json", processor)

    tensors = tiny_tensors(
        vocab_size=vocab_size,
        hidden_size=text["hidden_size"],
        intermediate_size=text["intermediate_size"],
        num_layers=text["num_hidden_layers"],
        num_heads=text["num_attention_heads"],
        num_kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        prefix="model.text_model",
        qk_norm=False,
        untied_head=True,
    )
    tensors.update(tiny_vision_tensors(config))
    if overrides:
        tensors.update(overrides)

    manifest = _base_manifest(root, format_version=2)
    manifest["task"] = "image-text-generation"
    manifest["model_family"] = "idefics3"
    manifest["tensors"] = _write_tensor_files(root, tensors, omit)
    manifest["processor_files"] = ["preprocessor_config.json", "processor_config.json"]
    _write_manifest(root, manifest)
    return root


def build_tiny_vision_model(root: Path, **kwargs) -> Path:
    """Tokenizer plus weights plus processor configs: one call, one directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ids = build_tiny_vision_tokenizer(root)
    vocab_size = HFTokenizer.from_file(str(root / "tokenizer.json")).get_vocab_size()
    return write_tiny_vision_model(
        root, vocab_size=vocab_size, image_token_id=ids["<image>"], **kwargs
    )


# --------------------------------------------------------------------------- #
# sampler stand-in
# --------------------------------------------------------------------------- #


def make_scripted_sampler(token_ids: Sequence[int]):
    """A ``Sampler`` stand-in that replays a fixed token sequence.

    The tiny model has random weights, so which token it would emit is not
    something a test can arrange. Scripting the sampler pins down the *loop*
    instead: stop conditions, streaming, text assembly, and CLI output routing.
    """

    class Scripted:
        def __init__(self, params=None) -> None:
            self.params = params
            self.calls = 0

        @property
        def is_greedy(self) -> bool:
            return True

        def sample(self, logits) -> int:
            token_id = token_ids[min(self.calls, len(token_ids) - 1)]
            self.calls += 1
            return int(token_id)

    return Scripted


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete tiny model directory, built once per test session."""
    root = tmp_path_factory.mktemp("tiny-model") / "tiny-qwen3"
    root.mkdir()
    build_tiny_tokenizer(root)
    vocab_size = HFTokenizer.from_file(str(root / "tokenizer.json")).get_vocab_size()
    write_tiny_model(root, vocab_size=vocab_size)
    return root


@pytest.fixture(scope="session")
def tiny_embedding_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same tiny decoder, converted as a text-embedding model.

    Same weights and same tokenizer, different manifest: ``task`` is what makes
    it an encoder, and the output projection it would never use is absent rather
    than merely unused.
    """
    root = tmp_path_factory.mktemp("tiny-embedding") / "tiny-qwen3-embedding"
    root.mkdir()
    build_tiny_tokenizer(root)
    vocab_size = HFTokenizer.from_file(str(root / "tokenizer.json")).get_vocab_size()
    write_tiny_model(root, vocab_size=vocab_size, task="text-embedding")
    return root


@pytest.fixture(scope="session")
def tiny_vlm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete tiny image+text (``idefics3``) directory.

    Its widest images split into a 2x2 crop grid, so one image is 4 crops plus
    the global frame = 5 frames = 20 image tokens: enough to make frame order,
    token counts and the context budget all observable.
    """
    root = tmp_path_factory.mktemp("tiny-vlm") / "tiny-smolvlm"
    root.mkdir()
    build_tiny_vision_model(root)
    return root


@pytest.fixture(scope="session")
def tiny_vlm_single_frame_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same tiny VLM, configured so no image is ever split.

    ``_resize_output_size_rescale_to_max_len`` makes the longest side exactly
    ``size.longest_edge``, so while that is 16 and the crop is 8px every image
    lands on a 2x2 grid. Setting the two equal is the only way to reach
    ``split_image``'s no-split branch — ``rows == cols == 0`` — which is what a
    small image does in the real pipeline and what ``image_prompt_string``
    formats differently (``<global-img>`` and no ``<row_r_col_c>`` tags).
    """
    root = tmp_path_factory.mktemp("tiny-vlm-single") / "tiny-smolvlm-single-frame"
    root.mkdir()
    build_tiny_vision_model(root, preprocessor_overrides={"size": {"longest_edge": 8}})
    return root


@pytest.fixture
def tiny_vlm_engine(tiny_vlm_dir: Path):
    """A fresh VisionEngine per test: the KV cache is shared state."""
    from slowq.vlm import VisionEngine

    return VisionEngine.from_directory(tiny_vlm_dir, max_context=128)


@pytest.fixture
def write_model():
    """Factory for standalone weight directories (no tokenizer, no fixtures)."""
    return write_tiny_model


@pytest.fixture
def random_tensors():
    """Factory for the random tensors ``write_tiny_model`` would have written."""
    return tiny_tensors


@pytest.fixture
def tiny_config(tiny_model_dir: Path):
    from slowq.config import ModelConfig

    return ModelConfig.from_json_file(tiny_model_dir / "config.json")


@pytest.fixture
def tiny_engine(tiny_model_dir: Path):
    """A fresh Engine per test: the KV cache is shared state, so never reuse one."""
    from slowq import Engine

    return Engine.from_directory(tiny_model_dir, max_context=64)


@pytest.fixture
def tiny_tokenizer(tiny_model_dir: Path):
    from slowq.tokenizer import Tokenizer

    return Tokenizer.from_directory(tiny_model_dir)


@pytest.fixture
def scripted_sampler():
    """Factory for a deterministic ``Sampler`` replacement (see conftest)."""
    return make_scripted_sampler
