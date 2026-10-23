#!/usr/bin/env python3
"""Validate SlowQ's numerics against the official Transformers reference.

The two implementations run in *separate processes* and exchange result files,
so a multi-GiB FP32 torch model and the NumPy model are never resident at the
same time::

    python tools/validate_reference.py numpy --model models/SmolLM2-135M-Instruct-slowq
    python tools/validate_reference.py ref   --hf-dir models/_cache/HuggingFaceTB--SmolLM2-135M-Instruct/<sha>
    python tools/validate_reference.py compare
    python tools/validate_reference.py all ...   # one process, both models (needs RAM)

Nothing here is specific to one model. What gets captured is *derived* from the
converted directory, so the same tool covers every family and task SlowQ
supports:

* **layers** — first, middle and last of the checkpoint's own
  ``num_hidden_layers``. A 30-layer SmolLM2 and a 28-layer Qwen3 are sampled at
  their own depths; no layer index is hard-coded.
* **weights** — chosen by the family's real rules: ``q_norm``/``k_norm`` only
  where the family has them, a projection bias only where one exists, and no
  ``lm_head`` for a checkpoint that ties it or has none at all.
* **architecture fields** — including the bias and norm flags, so a reference
  captured from a differently configured checkpoint is rejected before a single
  number is graded. The reference side restates those flags from the raw
  ``config.json`` using the official modelling code's expressions rather than
  asking SlowQ, which is what makes the agreement meaningful.
* **task** — text generation compares logits, hidden states and the greedy
  choice; text embedding compares the final hidden state at the pooled position,
  the pooled vector, its L2 normalisation and a similarity matrix. The reference
  side re-derives the task from the official directory with the converter's own
  evidence rules and refuses a disagreement, so a chat model cannot be validated
  as an embedding model or vice versa.

What is compared for text generation, using *identical token IDs* on both sides:

1. tokenizer      rendered chat text and token IDs vs ``AutoTokenizer``
2. weights        a sample of converted tensors must digest identically to the
                  FP32 upcast of the official BF16 checkpoint
3. full prefill   logits at the last position + selected layer hidden states
4. chunked prefill same quantities, produced by several forwards through a cache
5. cached decode  logits after each of ``--decode-steps`` forced decode steps
6. argmax         the greedy next token must agree at every compared position

Reference settings, fixed on purpose: ``torch_dtype=float32``, ``eval()``,
``attn_implementation="eager"`` (no SDPA/flash kernels), CPU, batch size 1.

Tolerances are declared *before* measuring, in ``TOLERANCES`` below.

**``compare`` grades no verdicts.** The reference process writes
``reference.json``: the text and token IDs its tokenizer produced, a sha256
digest of every sampled weight, the digests of the checkpoint files it loaded
and the capture configuration it ran under. The NumPy process writes the same
fields for *this* run. ``compare`` then recomputes every agreement itself, so a
passing boolean recorded by an earlier run cannot be carried forward — copying
one into the current results used to make a stale capture pass.

Completeness is derived, not discovered: ``expected_numeric_keys`` computes the
full numeric key set from the recorded capture configuration, and
``expected_sample_names`` / ``expected_weight_names`` do the same for the
tokenizer samples and the digested weights. A record missing from *either* file
fails the run — including one missing from *both*, which intersecting the two
file lists hides completely. That intersection is what let an empty ``ref.npz``
report ``PASS: 0 tensors compared``, an all-NaN tensor pass because
``nan > tolerance`` is False, and a sample neither side recorded pass because it
was never in the loop.

Both artifacts are small now (~11 MiB each): weights travel as digests rather
than as the 593 MiB embedding table, so the recorded reference can be checked
on another machine without moving it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowq.config import (  # noqa: E402
    FAMILY_IDEFICS3,
    FAMILY_LLAMA,
    FAMILY_QWEN2,
    FAMILY_QWEN3,
    TASK_IMAGE_TEXT_GENERATION,
    TASK_TEXT_EMBEDDING,
    TASK_TEXT_GENERATION,
)
from slowq.embedding import DEFAULT_QUERY_INSTRUCTION  # noqa: E402

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "validation"

#: Bumped whenever the recorded artifacts gain, lose or redefine a field. An
#: older capture is then rejected outright instead of passing checks it never
#: contained the data for. Version 3 is the multi-family, multi-task protocol:
#: the captured layers and digested weights are derived from the checkpoint
#: rather than being a fixed Qwen3 list, and the capture records which task it
#: covered.
ARTIFACT_FORMAT = 3

#: Written by the ``ref`` side only. The NumPy side must never produce it.
REFERENCE_FILENAME = "reference.json"

#: Declared up front so results cannot be graded against a threshold chosen
#: after seeing them.
#:
#: Rationale for the logits and hidden tolerances: FP32 unit round-off is 6e-8.
#: A deep residual network with 1024-3072 wide dot products and a softmax
#: amplifies that to roughly 1e-5 relative on the final logits, and the two
#: sides use different BLAS kernels (Accelerate vs torch's) with different
#: summation orders. We allow ~10x that.
#:
#: ``vector`` is an L2-normalised embedding, or a cosine similarity: its L2 norm
#: is exactly 1, so ``rel_l2`` *is* the distance between two unit vectors and
#: ``rel_l2 <= 1e-3`` means ``cos >= 1 - 5e-7``. Spreading that over 1024
#: components puts a single component's error near 3e-5; ``max_abs`` is declared
#: ~30x above it, the same margin the logits tolerances use.
TOLERANCES: dict[str, float] = {
    "rel_l2_logits": 1e-3,
    "max_abs_logits": 5e-2,
    "rel_l2_hidden": 1e-3,
    "max_abs_hidden": 5e-2,
    "rel_l2_vector": 1e-3,
    "max_abs_vector": 1e-3,
    "rel_l2_weights": 0.0,   # conversion must be exact: bf16 -> fp32 is a bit shift
    "max_abs_weights": 0.0,
}

ARGMAX_MUST_MATCH = True

#: Chat prompts, rendered by each checkpoint's own template. Deliberately
#: mixed-script and multi-turn: those are the cases where a tokenizer or a
#: template divergence shows up.
GENERATION_SAMPLES: list[dict[str, Any]] = [
    {
        "name": "zh-single-turn",
        "messages": [{"role": "user", "content": "你好，用一句话介绍你自己"}],
    },
    {
        "name": "en-with-system",
        "messages": [
            {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
            {"role": "user", "content": "Why is the sky blue?"},
        ],
    },
    {
        "name": "multi-turn",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 12 * 8?"},
            {"role": "assistant", "content": "12 * 8 = 96."},
            {"role": "user", "content": "And what is that plus 4? Reply with just the number."},
        ],
    },
    {
        "name": "emoji-and-mixed-script",
        "messages": [
            {
                "role": "user",
                "content": "用一个 emoji 表示开心 🎉，然后解释它的 Unicode 名称。",
            }
        ],
    },
]

#: Retrieval samples for the embedding task. Each encodes one query and several
#: documents; the two samples differ in whether the query carries the official
#: instruction prefix, because that prefix is the part of the recipe easiest to
#: get wrong (documents must *not* receive it).
EMBEDDING_SAMPLES: list[dict[str, Any]] = [
    {
        "name": "retrieval-with-instruction",
        "instruction": DEFAULT_QUERY_INSTRUCTION,
        "query": "Which planet in the solar system has the most moons?",
        "documents": [
            "Saturn has 146 confirmed moons, more than any other planet in the solar system.",
            "The Great Wall of China is a series of fortifications made of stone and brick.",
            "Jupiter's diameter is about eleven times that of Earth.",
        ],
    },
    {
        "name": "retrieval-without-instruction",
        "instruction": None,
        "query": "慢炖是一种什么样的烹饪方式？",
        "documents": [
            "慢炖是用低温长时间加热，让结缔组织慢慢转化为明胶的烹饪方法。",
            "向量检索通过比较嵌入之间的余弦相似度来排序候选文档。",
            "The boiling point of water at sea level is 100 degrees Celsius.",
        ],
    },
]

#: Every text of an embedding sample is captured under one of these names, in
#: this order. Recorded in the capture configuration so ``compare`` can rebuild
#: the key set without re-reading the samples.
EMBEDDING_TEXT_NAMES: dict[str, list[str]] = {
    sample["name"]: ["query", *(f"doc{index}" for index in range(len(sample["documents"])))]
    for sample in EMBEDDING_SAMPLES
}

#: The official Qwen3-Embedding retrieval input, restated on this side of the
#: comparison. A query is prefixed, a document is embedded verbatim, and there is
#: **no space** after ``Query:`` — the model card's own snippet, and the detail
#: easiest to get wrong.
#:
#: ``slowq.embedding.format_query`` builds the same string, and this format is
#: deliberately not allowed to call it. The reference side exists to grade
#: SlowQ; a reference that asks SlowQ how to spell its own input agrees with it
#: whatever it does, which is exactly what made the prefix untestable before.
REFERENCE_QUERY_FORMAT = "Instruct: {instruction}\nQuery:{query}"

#: Version of the embedding capture's input contract, recorded in the capture
#: block of both artifacts.
#:
#: Version 1 tokenised the text the NumPy side had already concatenated, so the
#: reference inherited SlowQ's prefix and a wrong one was compared against
#: itself. Version 2 rebuilds it here from the sample's raw query, instruction
#: and role.
#:
#: This is scoped to ``text-embedding`` rather than folded into
#: ``ARTIFACT_FORMAT`` because only that task's artifacts changed: the other two
#: tasks' reference text was already produced independently, by
#: ``apply_chat_template`` and by the official image processor. A version-3
#: generation or image+text reference stays valid; a version-1 embedding one is
#: refused, and re-capturing it needs the ``ref`` run only.
EMBEDDING_INPUT_FORMAT = 2


def reference_embedding_texts(sample_name: str) -> list[tuple[str, str, str, bool]]:
    """``(text_name, raw, text_to_embed, is_query)`` for one embedding sample.

    Taken from this tool's own :data:`EMBEDDING_SAMPLES`, never from the NumPy
    artifact's ``text`` field — that is the output under test. ``raw`` comes
    along so ``compare`` can tell "the two sides embedded different inputs" apart
    from "they embedded the same input with a different prefix".
    """
    sample = next(
        (entry for entry in EMBEDDING_SAMPLES if entry["name"] == sample_name), None
    )
    if sample is None:
        raise SystemExit(
            f"the NumPy artifact records an embedding sample {sample_name!r} that this "
            "version of the tool does not define; re-run 'validate_reference.py numpy'"
        )
    instruction = sample["instruction"]
    query = sample["query"]
    prefixed = (
        query
        if instruction is None
        else REFERENCE_QUERY_FORMAT.format(instruction=instruction, query=query)
    )
    return [("query", query, prefixed, True)] + [
        (f"doc{index}", document, document, False)
        for index, document in enumerate(sample["documents"])
    ]

#: Image+text samples for the ``image-text-generation`` task.
#:
#: The two sizes are chosen for the *crop grids* they produce, not for their
#: content. SmolVLM's preprocessor resizes the longest edge to
#: ``size.longest_edge`` (2048) and then rounds both sides up to a multiple of
#: ``max_image_size.longest_edge`` (512), so a 4:1 frame ends up 512x2048 — one
#: row of four crops — and a 2:1 frame ends up 1024x2048, two rows of four.
#: Together they cover a single-row grid, a multi-row grid, the ``<row_r_col_c>``
#: tags both produce, and the trailing global frame.
#:
#: Every image is *generated*, never downloaded or committed: the NumPy side
#: draws it and saves a PNG into the artifacts directory, the reference side
#: loads that same file and checks its digest, so both sides provably see
#: identical pixels and any difference in the pixel tensor is the preprocessing's
#: fault rather than the input's.
VISION_SAMPLES: list[dict[str, Any]] = [
    {
        "name": "grid-1x4",
        "width": 1200,
        "height": 300,
        "seed": 20261020,
        "question": "Describe the shapes you can see in this image.",
        "system": "You are a helpful assistant.",
    },
    {
        "name": "grid-2x4",
        "width": 800,
        "height": 400,
        "seed": 20261021,
        "question": "What colours appear in this picture?",
        "system": None,
    },
]


def build_vision_image(width: int, height: int, seed: int) -> Any:
    """Draw the sample image: a gradient with seeded shapes on top.

    Sharp edges and high-frequency colour on purpose. The official preprocessor
    resizes with Lanczos, which is exactly where a subtly different filter, a
    different rounding of the target size, or an off-by-one in the crop grid
    shows up — a smooth gradient would hide all three.
    """
    from PIL import Image, ImageDraw

    gradient = np.linspace(0.0, 255.0, width, dtype=np.float64)
    array = np.repeat(gradient[None, :, None], height, axis=0)
    array = np.repeat(array, 3, axis=2).astype(np.uint8)
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(seed)
    for index in range(6):
        colour = tuple(int(value) for value in rng.integers(0, 256, size=3))
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        side = max(8, min(width, height) // (index + 4))
        box = [x, y, min(width - 1, x + side), min(height - 1, y + side)]
        if index % 2:
            draw.ellipse(box, fill=colour)
        else:
            draw.rectangle(box, fill=colour)
    return image


def vision_image_filename(sample: Mapping[str, Any]) -> str:
    return f"vision-{sample['name']}.png"

#: Which layers to capture: first, middle and last *of this checkpoint*. Derived
#: from ``num_hidden_layers`` rather than fixed, because an error that only
#: appears after accumulation shows up at the last layer, and one present from
#: the start shows up at the first; a 30-layer SmolLM2 has no layer 27.
CAPTURED_POSITIONS = (0, -1)
#: Canonical name of the output projection. For a tied checkpoint this resolves
#: to the embedding matrix on both sides, which is exactly what the alias claims.
LM_HEAD_NAME = "lm_head.weight"


def capture_layers(num_hidden_layers: int) -> tuple[int, ...]:
    """First, middle and last layer index of a model with this many layers."""
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
    chosen = (0, num_hidden_layers // 2, num_hidden_layers - 1)
    # A 1- or 2-layer model has fewer distinct choices than three; deduplicate
    # rather than capture the same layer twice under two keys.
    return tuple(dict.fromkeys(chosen))


def samples_for(task: str) -> list[dict[str, Any]]:
    if task == TASK_TEXT_EMBEDDING:
        return EMBEDDING_SAMPLES
    if task == TASK_TEXT_GENERATION:
        return GENERATION_SAMPLES
    if task == TASK_IMAGE_TEXT_GENERATION:
        return VISION_SAMPLES
    raise ValueError(f"this tool has no capture protocol for task {task!r}")


#: Structural parameters both sides must agree on before any number is graded.
#: These are SlowQ's canonical names; the reference side restates each one from
#: the raw ``config.json`` using the official modelling code's own expressions,
#: so an agreement here means the two implementations read the checkpoint the
#: same way — not that SlowQ was asked twice.
ARCHITECTURE_FIELDS = (
    "family",
    "task",
    "tensor_prefix",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "rms_norm_eps",
    "rope_theta",
    "max_position_embeddings",
    "tie_word_embeddings",
    "qkv_bias",
    "o_proj_bias",
    "mlp_bias",
    "qk_norm",
    "sliding_window",
)

#: Extra fingerprint fields for an image+text checkpoint, describing the vision
#: tower and the connector rather than the decoder. Compared for the same reason
#: as the decoder fields: ``scale_factor`` decides how many patches the pixel
#: shuffle folds into one token, and ``image_seq_len`` decides how many ``<image>``
#: placeholders the prompt must contain, so a reference captured with either read
#: differently would be describing a different model.
VISION_ARCHITECTURE_FIELDS = (
    "decoder_family",
    "vision_hidden_size",
    "vision_intermediate_size",
    "vision_num_hidden_layers",
    "vision_num_attention_heads",
    "vision_head_dim",
    "vision_image_size",
    "vision_patch_size",
    "vision_hidden_act",
    "vision_layer_norm_eps",
    "vision_num_channels",
    "scale_factor",
    "image_seq_len",
    "image_token_id",
    "connector_out_features",
)


DIGEST_SPEC_VERSION = "slowq-weight-digest-v1"
DIGEST_BLOCK_BYTES = 1 << 22

#: Recorded next to every digest so the byte representation is stated, not
#: assumed. A digest is only comparable if both sides agree what was hashed.
WEIGHT_DIGEST_SPEC: dict[str, str] = {
    "version": DIGEST_SPEC_VERSION,
    "algorithm": "sha256",
    "preimage": (
        "the spec version, then the numpy dtype string, then the C-order shape, "
        "then the bytes of every element in C order"
    ),
    "dtype": "float32, IEEE 754 binary32 ('<f4')",
    "byte_order": "little-endian",
    "coverage": "the whole tensor: no striding, no sampling, no truncation",
}


class ComparisonError(ValueError):
    """A structural reason two arrays cannot be graded against each other.

    These are failures, never skips: the old comparator silently dropped both
    cases (shape mismatch after flattening, and non-finite values that compare
    False against every tolerance).
    """


@dataclass(frozen=True)
class Comparison:
    key: str
    sample: str
    kind: str
    max_abs: float
    mean_abs: float
    rel_l2: float
    argmax_match: bool | None = None


def compare_arrays(
    ours: np.ndarray, theirs: np.ndarray, key: str, *, sample: str = "", kind: str = "hidden"
) -> Comparison:
    """Compare two arrays, refusing to grade anything that is not comparable.

    ``kind`` is the caller's choice of tolerance row (``TOLERANCES``) and of
    whether an argmax is meaningful at all; only ``"logits"`` gets one.
    """
    ours = np.asarray(ours)
    theirs = np.asarray(theirs)
    if ours.shape != theirs.shape:
        raise ComparisonError(
            f"{key}: SlowQ shape {tuple(ours.shape)} vs reference shape {tuple(theirs.shape)}"
        )
    if ours.dtype != theirs.dtype:
        raise ComparisonError(f"{key}: dtype {ours.dtype} vs reference {theirs.dtype}")
    if ours.size == 0:
        raise ComparisonError(f"{key}: empty on both sides; nothing was captured")

    ours_bad = int(np.count_nonzero(~np.isfinite(ours)))
    theirs_bad = int(np.count_nonzero(~np.isfinite(theirs)))
    if ours_bad or theirs_bad:
        raise ComparisonError(
            f"{key}: not finite — {ours_bad} NaN/Inf value(s) on the SlowQ side, "
            f"{theirs_bad} on the reference side"
        )

    flat_ours = ours.astype(np.float64).reshape(-1)
    flat_theirs = theirs.astype(np.float64).reshape(-1)
    difference = flat_ours - flat_theirs
    numerator = float(np.linalg.norm(difference))
    denominator = float(np.linalg.norm(flat_theirs))
    if denominator == 0.0:
        # rel_l2 would be 0/0. An all-zero reference and an all-zero result do
        # agree; anything else is incomparable and must not be graded as 0.0.
        if numerator != 0.0:
            raise ComparisonError(
                f"{key}: the reference is all-zero but SlowQ is not; rel_l2 is undefined"
            )
        rel_l2 = 0.0
    else:
        rel_l2 = numerator / denominator

    # ``argmax`` only means something for a vocabulary distribution. Grading a
    # hidden state or a unit-norm embedding by its largest component would
    # report "greedy argmax disagrees" for a tensor that has no greedy choice,
    # and grading a similarity matrix by the flat index of its largest entry
    # would pass or fail on an artefact of the matrix being symmetric.
    argmax_match = None
    if kind == "logits" and flat_ours.size >= 2:
        argmax_match = bool(int(np.argmax(flat_ours)) == int(np.argmax(flat_theirs)))

    return Comparison(
        key=key,
        sample=sample,
        kind=kind,
        max_abs=float(np.max(np.abs(difference))),
        mean_abs=float(np.mean(np.abs(difference))),
        rel_l2=rel_l2,
        argmax_match=argmax_match,
    )


# ---- digests and fingerprints ------------------------------------------------


def tensor_digest(array: np.ndarray) -> dict[str, Any]:
    """sha256 over *every* element of one tensor, plus what that means.

    Hashing in blocks keeps the 593 MiB embedding table from being copied into
    RAM: the converted weights are mmap'ed, and ``memoryview`` walks the mapping
    in place.
    """
    contiguous = np.ascontiguousarray(np.asarray(array), dtype=np.float32)
    view = memoryview(contiguous).cast("B")
    hasher = hashlib.sha256()
    hasher.update(DIGEST_SPEC_VERSION.encode("ascii"))
    hasher.update(contiguous.dtype.str.encode("ascii"))
    hasher.update(repr(tuple(int(dim) for dim in contiguous.shape)).encode("ascii"))
    for start in range(0, len(view), DIGEST_BLOCK_BYTES):
        hasher.update(view[start : start + DIGEST_BLOCK_BYTES])
    return {
        "sha256": hasher.hexdigest(),
        "shape": [int(dim) for dim in contiguous.shape],
        "dtype": contiguous.dtype.str,
        "byte_order": "little" if contiguous.dtype.byteorder in ("<", "=") else "big",
        "elements": int(contiguous.size),
        "spec": DIGEST_SPEC_VERSION,
        "non_finite_elements": int(np.count_nonzero(~np.isfinite(contiguous))),
    }


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(DIGEST_BLOCK_BYTES), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _json_number(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def architecture_fingerprint(
    source: Any, extra_fields: Sequence[str] = ()
) -> dict[str, Any]:
    """Structural parameters, normalised so 1000000 and 1000000.0 agree.

    Both sides report these: a reference captured from a different revision, or
    from a differently configured checkpoint, is then rejected before a single
    number is graded. The bias and norm flags are included because they are the
    per-family facts most likely to be read differently by two implementations.

    ``extra_fields`` carries the vision tower's parameters for an image+text
    checkpoint; a text model passes nothing and its fingerprint stays unchanged.
    """
    fingerprint: dict[str, Any] = {}
    for name in (*ARCHITECTURE_FIELDS, *extra_fields):
        if isinstance(source, Mapping):
            if name not in source:
                raise ComparisonError(f"architecture fingerprint is missing {name!r}")
            value = source[name]
        else:
            value = getattr(source, name)
        fingerprint[name] = _json_number(value)
    return fingerprint


def reference_architecture(raw: Mapping[str, Any], task: str) -> dict[str, Any]:
    """The same fingerprint, restated from the official ``config.json``.

    Every value is computed here from the raw config using the expressions in
    ``transformers`` 4.51.3's modelling code, *not* read back from SlowQ:

    * ``head_dim`` — ``LlamaConfig`` sets it to ``hidden_size //
      num_attention_heads`` when absent, ``Qwen2Config`` does not define it at
      all (so ``getattr(config, "head_dim", hidden // heads)`` falls back), and
      ``Qwen3Config`` defaults it to 128. All three reduce to the same
      expression over the raw dict.
    * ``qkv_bias`` / ``o_proj_bias`` — ``Qwen2Attention`` hard-codes
      ``bias=True`` on q/k/v and ``bias=False`` on o_proj; qwen3 and llama read
      ``config.attention_bias`` for all four.
    * ``mlp_bias`` — only ``LlamaMLP`` reads ``config.mlp_bias``.
    * ``qk_norm`` — only qwen3 builds ``q_norm``/``k_norm``.
    * ``sliding_window`` — the qwen2 and qwen3 model-level causal mask applies
      ``config.sliding_window``; the string does not occur in
      ``modeling_llama.py`` at all.

    An ``idefics3`` checkpoint keeps its decoder under ``text_config`` and adds a
    ``vision_config`` beside it, so the decoder fields are restated from the
    nested dict and the vision fields from the top level — see
    :func:`_reference_vision_architecture`.

    ``tensor_prefix`` is the one field that is a SlowQ storage convention rather
    than an observed fact: a text decoder is always stored under ``model``,
    whatever the official checkpoint called it. The renames that follow from a
    different source layout are carried per tensor in ``source.weight_sources``
    and are verified by the digests, not by this string.
    """
    if raw.get("model_type") == FAMILY_IDEFICS3:
        return _reference_vision_architecture(raw, task)

    model_type = raw.get("model_type")
    family = {
        FAMILY_QWEN3: FAMILY_QWEN3,
        FAMILY_QWEN2: FAMILY_QWEN2,
        FAMILY_LLAMA: FAMILY_LLAMA,
    }.get(str(model_type))
    if family is None:
        raise SystemExit(f"{model_type!r} is not a text family this tool can fingerprint")

    text = _reference_decoder_fingerprint(raw, family, task, tensor_prefix="model")
    return architecture_fingerprint(text)


def _reference_decoder_fingerprint(
    raw: Mapping[str, Any], family: str, task: str, *, tensor_prefix: str
) -> dict[str, Any]:
    """The decoder half of a fingerprint, from one text-decoder config dict."""
    hidden_size = int(raw["hidden_size"])
    num_heads = int(raw["num_attention_heads"])
    # getattr(config, "head_dim", hidden_size // num_attention_heads)
    head_dim = raw.get("head_dim")
    head_dim = hidden_size // num_heads if head_dim is None else int(head_dim)

    if family == FAMILY_QWEN2:
        qkv_bias, o_proj_bias = True, False
    else:
        qkv_bias = o_proj_bias = bool(raw.get("attention_bias", False))
    mlp_bias = bool(raw.get("mlp_bias", False)) if family == FAMILY_LLAMA else False
    sliding_window = (
        raw.get("sliding_window") if family in (FAMILY_QWEN2, FAMILY_QWEN3) else None
    )

    return {
        "family": family,
        "task": task,
        "tensor_prefix": tensor_prefix,
        "hidden_size": hidden_size,
        "intermediate_size": int(raw["intermediate_size"]),
        "num_hidden_layers": int(raw["num_hidden_layers"]),
        "num_attention_heads": num_heads,
        "num_key_value_heads": int(raw["num_key_value_heads"]),
        "head_dim": head_dim,
        "vocab_size": int(raw["vocab_size"]),
        "rms_norm_eps": float(raw["rms_norm_eps"]),
        "rope_theta": float(raw["rope_theta"]),
        "max_position_embeddings": int(raw["max_position_embeddings"]),
        "tie_word_embeddings": bool(raw.get("tie_word_embeddings", False)),
        "qkv_bias": qkv_bias,
        "o_proj_bias": o_proj_bias,
        "mlp_bias": mlp_bias,
        "qk_norm": family == FAMILY_QWEN3,
        "sliding_window": None if sliding_window is None else int(sliding_window),
    }


def _reference_vision_architecture(raw: Mapping[str, Any], task: str) -> dict[str, Any]:
    """The fingerprint of an ``idefics3`` checkpoint, from its own config.

    ``Idefics3Model`` derives ``image_seq_len`` as
    ``((image_size // patch_size) ** 2) / scale_factor ** 2`` rather than reading
    ``processor_config.json``, and the connector's input width as
    ``vision_hidden_size * scale_factor ** 2``; both are restated here from the
    raw config so agreeing with SlowQ means the two implementations read the
    checkpoint the same way. The decoder half comes from ``text_config``'s own
    ``model_type`` through the same per-family rules a standalone text model gets,
    so SmolVLM's ``VLlama3ForCausalLM`` decoder is fingerprinted as the llama it
    is — ``attention_bias`` and ``mlp_bias`` off, no q/k norm — rather than being
    assumed from the wrapper's name.
    """
    text_config = raw.get("text_config")
    vision_config = raw.get("vision_config")
    if not isinstance(text_config, Mapping) or not isinstance(vision_config, Mapping):
        raise SystemExit(
            f"{FAMILY_IDEFICS3} config.json must nest text_config and vision_config"
        )
    decoder_family = str(text_config.get("model_type"))
    if decoder_family not in (FAMILY_QWEN3, FAMILY_QWEN2, FAMILY_LLAMA):
        raise SystemExit(
            f"{FAMILY_IDEFICS3}'s text_config declares model_type={decoder_family!r}, "
            "which this tool cannot restate the bias and norm rules for"
        )
    decoder = _reference_decoder_fingerprint(
        text_config, decoder_family, task, tensor_prefix="model.text_model"
    )

    image_size = int(vision_config["image_size"])
    patch_size = int(vision_config["patch_size"])
    scale_factor = int(raw["scale_factor"])
    vision_hidden_size = int(vision_config["hidden_size"])
    vision_heads = int(vision_config["num_attention_heads"])

    fields = {
        **decoder,
        # The wrapper's family is what SlowQ records in its manifest; which text
        # family the nested decoder happens to be travels beside it, because the
        # bias and norm rules above were restated from *that* name.
        "family": FAMILY_IDEFICS3,
        "decoder_family": decoder_family,
        "vision_hidden_size": vision_hidden_size,
        "vision_intermediate_size": int(vision_config["intermediate_size"]),
        "vision_num_hidden_layers": int(vision_config["num_hidden_layers"]),
        "vision_num_attention_heads": vision_heads,
        "vision_head_dim": vision_hidden_size // vision_heads,
        "vision_image_size": image_size,
        "vision_patch_size": patch_size,
        "vision_hidden_act": str(vision_config["hidden_act"]),
        "vision_layer_norm_eps": float(vision_config["layer_norm_eps"]),
        "vision_num_channels": int(vision_config["num_channels"]),
        "scale_factor": scale_factor,
        "image_seq_len": int(((image_size // patch_size) ** 2) / (scale_factor**2)),
        "image_token_id": int(raw["image_token_id"]),
        "connector_out_features": vision_hidden_size * scale_factor**2,
    }
    return architecture_fingerprint(fields, VISION_ARCHITECTURE_FIELDS)


def vision_architecture_fingerprint(vlm_config: Any) -> dict[str, Any]:
    """The same fingerprint, read out of SlowQ's :class:`SmolVLMConfig`.

    Mirrors :func:`_reference_vision_architecture` field for field. The two are
    built from different inputs — SlowQ's validated config object versus the raw
    official ``config.json`` — so their agreement is the check, not a restatement.
    """
    text, vision = vlm_config.text, vlm_config.vision
    fields = {
        **{name: getattr(text, name) for name in ARCHITECTURE_FIELDS},
        "family": FAMILY_IDEFICS3,
        "decoder_family": text.family,
        "vision_hidden_size": vision.hidden_size,
        "vision_intermediate_size": vision.intermediate_size,
        "vision_num_hidden_layers": vision.num_hidden_layers,
        "vision_num_attention_heads": vision.num_attention_heads,
        "vision_head_dim": vision.head_dim,
        "vision_image_size": vision.image_size,
        "vision_patch_size": vision.patch_size,
        "vision_hidden_act": vision.hidden_act,
        "vision_layer_norm_eps": vision.layer_norm_eps,
        "vision_num_channels": vision.num_channels,
        "scale_factor": vlm_config.scale_factor,
        "image_seq_len": vlm_config.computed_image_seq_len,
        "image_token_id": vlm_config.image_token_id,
        "connector_out_features": vlm_config.connector_out_features,
    }
    return architecture_fingerprint(fields, VISION_ARCHITECTURE_FIELDS)



def weight_check_sample(arch: Mapping[str, Any]) -> tuple[str, ...]:
    """Which converted tensors to digest, chosen by this family's real rules.

    Not a fixed list: ``q_norm``/``k_norm`` exist only for qwen3, a projection
    bias only where that family's ``nn.Linear`` declares one, and an embedding
    checkpoint has no output projection at all. A name this function does not
    return is a name the checkpoint legitimately does not have — requiring it
    would be a false failure, and skipping it silently would hide a real one.
    """
    prefix = str(arch["tensor_prefix"])
    layers = capture_layers(int(arch["num_hidden_layers"]))
    first, middle, last = layers[0], layers[len(layers) // 2], layers[-1]
    task = str(arch["task"])

    names = [f"{prefix}.embed_tokens.weight"]

    def attention(index: int) -> str:
        return f"{prefix}.layers.{index}.self_attn"

    names.append(f"{attention(first)}.q_proj.weight")
    if arch["qk_norm"]:
        names += [f"{attention(first)}.q_norm.weight", f"{attention(first)}.k_norm.weight"]
    if arch["qkv_bias"]:
        names.append(f"{attention(first)}.q_proj.bias")
    if arch["o_proj_bias"]:
        names.append(f"{attention(first)}.o_proj.bias")

    names.append(f"{attention(middle)}.k_proj.weight")
    names.append(f"{prefix}.layers.{middle}.mlp.gate_proj.weight")
    if arch["mlp_bias"]:
        names.append(f"{prefix}.layers.{middle}.mlp.gate_proj.bias")
    names.append(f"{prefix}.layers.{last}.mlp.down_proj.weight")
    names.append(f"{prefix}.norm.weight")

    if task != TASK_TEXT_EMBEDDING:
        # Digesting this even for a tied checkpoint is the point: SlowQ resolves
        # the name through the manifest alias and the reference through its own
        # tied ``lm_head``, so equal digests are what prove the alias is right.
        names.append(LM_HEAD_NAME)
    if task == TASK_IMAGE_TEXT_GENERATION:
        names += _vision_weight_check_sample(arch)
    return tuple(names)


def _vision_weight_check_sample(arch: Mapping[str, Any]) -> list[str]:
    """Vision-tower and connector tensors to digest, at the tower's own depths.

    The vision tower has its own layer count, so its first/middle/last are
    derived from ``vision_num_hidden_layers`` and are not the decoder's. The
    bias vectors are sampled alongside the weights because ``Idefics3VisionAttention``
    declares ``nn.Linear(embed_dim, embed_dim)`` for all four projections — every
    one biased, unlike the llama decoder sitting next to it — and a converter
    that dropped the vision biases would otherwise go unnoticed.
    """
    from slowq.vlm import CONNECTOR_TENSOR_PREFIX, VISION_TENSOR_PREFIX

    layers = capture_layers(int(arch["vision_num_hidden_layers"]))
    first, middle, last = layers[0], layers[len(layers) // 2], layers[-1]

    def layer(index: int) -> str:
        return f"{VISION_TENSOR_PREFIX}.encoder.layers.{index}"

    return [
        f"{VISION_TENSOR_PREFIX}.embeddings.patch_embedding.weight",
        f"{VISION_TENSOR_PREFIX}.embeddings.patch_embedding.bias",
        f"{VISION_TENSOR_PREFIX}.embeddings.position_embedding.weight",
        f"{layer(first)}.self_attn.q_proj.weight",
        f"{layer(first)}.self_attn.q_proj.bias",
        f"{layer(first)}.layer_norm1.weight",
        f"{layer(first)}.layer_norm1.bias",
        f"{layer(middle)}.self_attn.out_proj.weight",
        f"{layer(middle)}.mlp.fc1.weight",
        f"{layer(middle)}.mlp.fc1.bias",
        f"{layer(last)}.mlp.fc2.weight",
        f"{layer(last)}.layer_norm2.bias",
        f"{VISION_TENSOR_PREFIX}.post_layernorm.weight",
        f"{VISION_TENSOR_PREFIX}.post_layernorm.bias",
        f"{CONNECTOR_TENSOR_PREFIX}.modality_projection.proj.weight",
    ]



def capture_config(
    arch: Mapping[str, Any],
    *,
    prefill_chunk_size: int,
    decode_steps: int,
    vision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What a capture covers, recorded identically by both sides.

    ``compare`` recomputes this from the *current* rules and the architecture in
    ``samples.json``, so changing how layers or weights are chosen without
    re-capturing the reference fails instead of quietly comparing a subset.

    ``vision`` is the per-sample layout of an image+text capture — frame counts,
    image-token counts and the positions sampled in the fused sequence. It comes
    from tokenising the expanded prompt, which only the NumPy side can do with
    SlowQ's own tokenizer, so the reference side and ``compare`` both take it
    from the recorded capture exactly as they take ``prefill_chunk_size``.
    """
    task = str(arch["task"])
    samples = samples_for(task)
    config: dict[str, Any] = {
        "artifact_format": ARTIFACT_FORMAT,
        "digest_spec": WEIGHT_DIGEST_SPEC,
        "task": task,
        "family": str(arch["family"]),
        "sample_names": [sample["name"] for sample in samples],
        "layers": list(capture_layers(int(arch["num_hidden_layers"]))),
        "weight_names": list(weight_check_sample(arch)),
        "prefill_chunk_size": int(prefill_chunk_size),
    }
    if task == TASK_TEXT_EMBEDDING:
        # An embedding capture never decodes: it prefills each text once, at
        # full length and in chunks, and stops at the pooled vector.
        config["decode_steps"] = 0
        config["embedding_texts"] = {
            sample["name"]: list(EMBEDDING_TEXT_NAMES[sample["name"]]) for sample in samples
        }
        config["embedding_input_format"] = EMBEDDING_INPUT_FORMAT
    elif task == TASK_IMAGE_TEXT_GENERATION:
        config["decode_steps"] = int(decode_steps)
        config["positions_cached"] = [-1]
        config["vision_layers"] = list(capture_layers(int(arch["vision_num_hidden_layers"])))
        config["vision"] = {name: dict(value) for name, value in (vision or {}).items()}
    else:
        config["decode_steps"] = int(decode_steps)
        config["positions_full"] = [int(position) for position in CAPTURED_POSITIONS]
        config["positions_cached"] = [-1]
    return config


def expected_numeric_keys(capture: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """Every numeric key a complete capture must contain -> (sample, kind).

    Derived from the capture configuration, never from the intersection of the
    two files: an intersection only reports on what both sides happened to
    write, so a truncated or empty reference looks like a clean run.
    """
    task = str(capture["task"])
    if task == TASK_TEXT_EMBEDDING:
        return _expected_embedding_keys(capture)
    if task == TASK_IMAGE_TEXT_GENERATION:
        return _expected_vision_keys(capture)
    return _expected_generation_keys(capture)


def vision_capture_frames(num_frames: int) -> tuple[int, ...]:
    """Which frames of an image get their vision-tower output stored as arrays.

    First and last. The connector output is stored for *every* frame, so no
    frame's contribution to the text stream goes unchecked; these two are what
    separate "the tower is wrong" from "the connector is wrong", and they pin the
    frame ordering at both ends, which is where a row-major/global-frame mix-up
    lands. A single-frame image collapses to one index rather than recording the
    same tensor twice under two keys.
    """
    if num_frames < 1:
        raise ValueError(f"an image produced {num_frames} frames")
    return tuple(dict.fromkeys((0, num_frames - 1)))


def _expected_generation_keys(capture: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    expected: dict[str, tuple[str, str]] = {}
    decode_steps = int(capture["decode_steps"])
    for name in capture["sample_names"]:
        stages: list[tuple[str, Sequence[int]]] = [
            ("full_prefill", list(capture["positions_full"])),
            ("chunked_prefill", list(capture["positions_cached"])),
        ]
        stages += [
            (f"decode{step}", list(capture["positions_cached"])) for step in range(decode_steps)
        ]
        for stage, positions in stages:
            prefix = f"{name}.{stage}"
            expected[f"{prefix}.logits"] = (name, "logits")
            for position in positions:
                for layer in capture["layers"]:
                    expected[f"{prefix}.layer{layer}.layer_output[{position}]"] = (name, "hidden")
                expected[f"{prefix}.final_norm[{position}]"] = (name, "hidden")
    return expected


def _expected_embedding_keys(capture: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """Keys of an embedding capture: per text, plus one similarity matrix.

    ``final_norm[-1]`` and ``vector`` are both recorded, and the pooling
    position itself travels as metadata (``pool_index``) rather than as a
    tensor: the reference computes it as ``attention_mask.sum() - 1`` from its
    own tokenisation, so agreeing on the *index* is a real cross-check, while
    re-recording the row it selects would only restate it.
    """
    expected: dict[str, tuple[str, str]] = {}
    for name in capture["sample_names"]:
        for text in capture["embedding_texts"][name]:
            for stage in ("full_prefill", "chunked_prefill"):
                prefix = f"{name}.{text}.{stage}"
                for layer in capture["layers"]:
                    expected[f"{prefix}.layer{layer}.layer_output[-1]"] = (name, "hidden")
                expected[f"{prefix}.final_norm[-1]"] = (name, "hidden")
                expected[f"{prefix}.vector"] = (name, "vector")
        expected[f"{name}.similarity"] = (name, "vector")
    return expected


def _expected_vision_keys(capture: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """Keys of an image+text capture: the vision path, then the ordinary decode.

    Segment by segment, because a vision model has more places to be subtly
    wrong than a text one and one final-logit comparison cannot say which:

    1. ``vision.encoder_output[f]`` — the tower's post-LayerNorm output, for the
       frames :func:`vision_capture_frames` picks.
    2. ``vision.connector_output`` — pixel shuffle plus the modality projection,
       every frame, in the layout the merge assigns from.
    3. ``full_prefill.embed_sequence`` — the fused matrix the decoder actually
       reads: text rows from the embedding table, image rows from the connector.
    4. logits, selected layer outputs and the final norm, then the cached decode.

    The pixel tensors are *not* here: they travel as digests in ``samples.json``
    next to their shapes, because at 512x512x3 per frame they are the largest
    arrays in the run and a digest compares them exactly rather than within a
    tolerance.
    """
    expected: dict[str, tuple[str, str]] = {}
    decode_steps = int(capture["decode_steps"])
    vision = capture.get("vision") or {}
    for name in capture["sample_names"]:
        layout = vision.get(name)
        if not isinstance(layout, Mapping):
            raise ComparisonError(
                f"capture records no vision layout for sample {name!r}; these artifacts "
                "were written by an older version of this tool"
            )
        for frame in vision_capture_frames(int(layout["num_frames"])):
            expected[f"{name}.vision.encoder_output[{frame}]"] = (name, "hidden")
        expected[f"{name}.vision.connector_output"] = (name, "hidden")

        stages: list[tuple[str, Sequence[int]]] = [
            ("full_prefill", list(layout["positions_full"])),
            ("chunked_prefill", list(capture["positions_cached"])),
        ]
        stages += [
            (f"decode{step}", list(capture["positions_cached"])) for step in range(decode_steps)
        ]
        for stage, positions in stages:
            prefix = f"{name}.{stage}"
            expected[f"{prefix}.logits"] = (name, "logits")
            if stage == "full_prefill":
                # Only the full prefill sees the whole fused sequence at once;
                # a chunk's is that chunk's slice, which proves nothing about
                # where the image rows landed.
                expected[f"{prefix}.embed_sequence"] = (name, "hidden")
            for position in positions:
                for layer in capture["layers"]:
                    expected[f"{prefix}.layer{layer}.layer_output[{position}]"] = (name, "hidden")
                expected[f"{prefix}.final_norm[{position}]"] = (name, "hidden")
    return expected


def expected_sample_names(capture: Mapping[str, Any]) -> list[str]:
    """Every sample a complete capture must record.

    The same rule as ``expected_numeric_keys``: the capture configuration says
    what was supposed to be captured, so a sample *both* sides omitted is a
    missing record rather than one less thing to compare.

    Takes a capture, never ``None``: without one there is no agreed sample set,
    and guessing the current protocol's names is how an embedding capture came to
    be reported as missing four generation prompts. ``check_capture`` refuses the
    run and names the mismatch instead.
    """
    return [str(name) for name in capture["sample_names"]]


def expected_weight_names(arch: Mapping[str, Any]) -> tuple[str, ...]:
    """Every converted weight a complete capture must digest.

    Derived from the architecture this run actually loaded, not from the
    recorded capture: a capture that wrote down the wrong names is then caught
    rather than trusted.
    """
    return weight_check_sample(arch)



def snapshot_revision(directory: Path) -> str | None:
    """The commit a downloaded snapshot says it came from, if it says anything.

    Only a snapshot fetched by ``tools/convert_model.py`` carries the marker; a
    hand-made directory returns ``None`` and the file digests are then the only
    tie between the two sides.
    """
    tools_dir = str(Path(__file__).resolve().parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from convert_model import SNAPSHOT_MARKER

    marker = directory / SNAPSHOT_MARKER
    if not marker.is_file():
        return None
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    revision = recorded.get("revision")
    return str(revision) if isinstance(revision, str) else None


# ---- numpy side -------------------------------------------------------------


def _source_record(weights: Any, names: Sequence[str]) -> dict[str, Any]:
    """Provenance for the artifacts: which bytes this converted model claims.

    ``weight_sources`` maps a canonical tensor name to the name it had in the
    official checkpoint, for the tensors the converter renamed. The reference
    side needs it to find the same tensor in a checkpoint whose layout differs —
    the Qwen3-Embedding export has no ``model.`` prefix at all.
    """
    manifest = weights.manifest
    sources: dict[str, str] = {}
    for name in names:
        source_name = weights.spec(name).source_name
        if source_name is not None and source_name != name:
            sources[name] = source_name
    return {
        "repo_id": manifest.source.get("repo_id"),
        "revision": manifest.source_revision,
        "files": {
            str(entry["name"]): entry.get("sha256")
            for entry in manifest.source.get("files", [])
        },
        "weight_sources": sources,
    }


def _write_numpy_artifacts(
    artifacts: Path, payload: dict[str, Any], blobs: dict[str, np.ndarray]
) -> int:
    (artifacts / "samples.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(artifacts / "slowq.npz", **blobs)
    print(
        f"[numpy] wrote {artifacts / 'samples.json'} and {artifacts / 'slowq.npz'} "
        f"({len(blobs)} tensors)",
        file=sys.stderr,
    )
    return 0


def run_numpy_side(args: argparse.Namespace) -> int:
    """Capture this run's numbers, dispatching on the directory's recorded task.

    The task comes from the manifest, never from ``config.json``: the embedding
    checkpoint publishes ``model_type: "qwen3"`` and
    ``architectures: ["Qwen3ForCausalLM"]``, byte-for-byte what the chat model
    publishes, so guessing here would validate a generator against an encoder.
    """
    from slowq.engine import recorded_task

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    task = recorded_task(args.model)
    if task == TASK_TEXT_EMBEDDING:
        return _numpy_embedding_side(args, artifacts)
    if task == TASK_TEXT_GENERATION:
        return _numpy_generation_side(args, artifacts)
    if task == TASK_IMAGE_TEXT_GENERATION:
        return _numpy_vision_side(args, artifacts)
    raise SystemExit(f"this tool has no NumPy-side capture for task {task!r}")


def _numpy_generation_side(args: argparse.Namespace, artifacts: Path) -> int:
    from slowq.engine import Engine
    from slowq.model import DebugProbe
    from slowq.sampling import Sampler, SamplingParams

    engine = Engine.from_directory(
        args.model, max_context=args.max_context, prefill_chunk_size=args.prefill_chunk_size
    )
    greedy = Sampler(SamplingParams())
    captures = {"layer_output", "final_norm", "logits"}
    config = engine.config
    layers = capture_layers(config.num_hidden_layers)
    arch = architecture_fingerprint(config)

    payload: dict[str, Any] = {
        "samples": [],
        "model_directory": str(args.model),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "side": "numpy",
        "architecture": arch,
        "capture": capture_config(
            arch,
            prefill_chunk_size=args.prefill_chunk_size,
            decode_steps=args.decode_steps,
        ),
        "tokenizer": [],
        "weight_digests": {},
    }
    payload["source"] = _source_record(
        engine.model.weights, payload["capture"]["weight_names"]
    )
    blobs: dict[str, np.ndarray] = {}

    for sample in GENERATION_SAMPLES:
        name = sample["name"]
        messages = sample["messages"]
        text = engine.render(messages)
        prompt_ids = engine.tokenizer.encode(text)
        print(f"[numpy] {name}: {len(prompt_ids)} prompt tokens", file=sys.stderr)
        payload["tokenizer"].append(
            {"name": name, "text": text, "ids": [int(token) for token in prompt_ids]}
        )

        record: dict[str, Any] = {
            "name": name,
            "messages": messages,
            "rendered_text": text,
            "prompt_ids": prompt_ids,
            "prefill_chunk_size": args.prefill_chunk_size,
            "decode_steps": [],
        }

        budget = min(args.max_context, len(prompt_ids) + args.decode_steps + 8)

        # 1. full prefill in one forward
        cache = engine.model.new_cache(budget)
        probe = DebugProbe.for_names(captures, layers, CAPTURED_POSITIONS)
        full = engine.model.forward(prompt_ids, cache, probe=probe)
        blobs[f"{name}.full_prefill.logits"] = full.logits
        for key, value in probe.records.items():
            if key.startswith("layer") or key.startswith("final_norm"):
                blobs[f"{name}.full_prefill.{key}"] = value
        record["full_prefill_argmax"] = int(greedy.sample(full.logits))
        del cache

        # 2. chunked prefill: same tokens, several forwards through one cache
        cache = engine.model.new_cache(budget)
        probe = DebugProbe.for_names(captures, layers, (-1,))
        chunked = None
        for start in range(0, len(prompt_ids), args.prefill_chunk_size):
            chunked = engine.model.forward(
                prompt_ids[start : start + args.prefill_chunk_size], cache, probe=probe
            )
        assert chunked is not None
        blobs[f"{name}.chunked_prefill.logits"] = chunked.logits
        for key, value in probe.records.items():
            if key.startswith("layer") or key.startswith("final_norm"):
                blobs[f"{name}.chunked_prefill.{key}"] = value
        record["chunked_prefill_argmax"] = int(greedy.sample(chunked.logits))

        # 3. cached decode steps, recorded so the reference can replay the path
        logits = chunked.logits
        for step in range(args.decode_steps):
            token_id = int(greedy.sample(logits))
            step_probe = DebugProbe.for_names(captures, layers, (-1,))
            forward = engine.model.forward([token_id], cache, probe=step_probe)
            blobs[f"{name}.decode{step}.logits"] = forward.logits
            for key, value in step_probe.records.items():
                if key.startswith("layer") or key.startswith("final_norm"):
                    blobs[f"{name}.decode{step}.{key}"] = value
            record["decode_steps"].append(
                {
                    "input_token": token_id,
                    "cache_length_before": forward.position_offset,
                    "argmax": int(greedy.sample(forward.logits)),
                }
            )
            logits = forward.logits

        payload["samples"].append(record)
        print(
            f"[numpy]   full={record['full_prefill_argmax']} "
            f"chunked={record['chunked_prefill_argmax']} "
            f"decode={[s['argmax'] for s in record['decode_steps']]}",
            file=sys.stderr,
        )

    _digest_weights(engine.model.weights, payload)
    return _write_numpy_artifacts(artifacts, payload, blobs)


def _numpy_embedding_side(args: argparse.Namespace, artifacts: Path) -> int:
    from slowq.embedding import EmbeddingEngine, similarity_matrix
    from slowq.model import DebugProbe

    engine = EmbeddingEngine.from_directory(
        args.model,
        max_length=args.max_context,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    config = engine.config
    layers = capture_layers(config.num_hidden_layers)
    arch = architecture_fingerprint(config)
    captures = {"layer_output", "final_norm"}

    payload: dict[str, Any] = {
        "samples": [],
        "model_directory": str(args.model),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "side": "numpy",
        "architecture": arch,
        "capture": capture_config(
            arch,
            prefill_chunk_size=args.prefill_chunk_size,
            decode_steps=args.decode_steps,
        ),
        "tokenizer": [],
        "weight_digests": {},
    }
    payload["source"] = _source_record(
        engine.model.weights, payload["capture"]["weight_names"]
    )
    blobs: dict[str, np.ndarray] = {}

    for sample in EMBEDDING_SAMPLES:
        name = sample["name"]
        instruction = sample["instruction"]
        # The query carries the official instruction prefix; documents never do.
        texts = [("query", sample["query"], True)] + [
            (f"doc{index}", document, False)
            for index, document in enumerate(sample["documents"])
        ]
        record: dict[str, Any] = {
            "name": name,
            "instruction": instruction,
            "query": sample["query"],
            "documents": list(sample["documents"]),
            "texts": [],
        }
        vectors: list[np.ndarray] = []

        for text_name, raw_text, is_query in texts:
            # One call does the tokenisation and the chunked encode; the full
            # prefill then replays the *same* IDs in a single forward, so the two
            # stages differ only in how the sequence was fed in.
            probe = DebugProbe.for_names(captures, layers, (-1,))
            encoded = engine.encode(
                raw_text, is_query=is_query, instruction=instruction, probe=probe
            )
            prefix = f"{name}.{text_name}"
            for key, value in probe.records.items():
                blobs[f"{prefix}.chunked_prefill.{key}"] = value
            blobs[f"{prefix}.chunked_prefill.vector"] = encoded.vector

            full_probe = DebugProbe.for_names(captures, layers, (-1,))
            full_vector = engine.model.encode_ids(
                encoded.input_ids,
                prefill_chunk_size=len(encoded.input_ids),
                probe=full_probe,
            )
            for key, value in full_probe.records.items():
                blobs[f"{prefix}.full_prefill.{key}"] = value
            blobs[f"{prefix}.full_prefill.vector"] = full_vector

            record["texts"].append(
                {
                    "name": text_name,
                    "raw": raw_text,
                    "text": encoded.text,
                    "is_query": is_query,
                    "num_tokens": encoded.num_tokens,
                    "pool_index": encoded.num_tokens - 1,
                    "ids": [int(token) for token in encoded.input_ids],
                    "dimension": encoded.dimension,
                }
            )
            payload["tokenizer"].append(
                {
                    "name": f"{name}/{text_name}",
                    "text": encoded.text,
                    "ids": [int(token) for token in encoded.input_ids],
                }
            )
            vectors.append(encoded.vector)
            print(
                f"[numpy] {name}/{text_name}: {encoded.num_tokens} tokens -> "
                f"{encoded.dimension}-d",
                file=sys.stderr,
            )

        blobs[f"{name}.similarity"] = similarity_matrix(vectors, vectors)
        payload["samples"].append(record)

    _digest_weights(engine.model.weights, payload)
    return _write_numpy_artifacts(artifacts, payload, blobs)


def _numpy_vision_side(args: argparse.Namespace, artifacts: Path) -> int:
    """Capture one image+text request per sample, segment by segment.

    The prefill is driven through ``model.forward`` directly rather than through
    ``Engine.stream_ids`` — the same choice the text side makes — because a probe
    overwrites its records on every forward, so one probe cannot report the
    chunked prefill and the decode steps separately. The per-chunk slicing of the
    feature rows mirrors ``Engine.stream_ids``: placeholder tokens can straddle a
    chunk boundary, so each chunk takes its own slice in prompt order instead of
    assuming the image fits inside one.

    The image is written to a PNG in the artifacts directory and its digest
    recorded, so the reference side loads *these* pixels. Both sides then run
    their own preprocessor over an identical input, which is what makes the
    pixel-tensor digest a statement about the preprocessing rather than about the
    picture.
    """
    from slowq.model import DebugProbe
    from slowq.sampling import Sampler, SamplingParams
    from slowq.vlm import VisionEngine

    engine = VisionEngine.from_directory(
        args.model, max_context=args.max_context, prefill_chunk_size=args.prefill_chunk_size
    )
    greedy = Sampler(SamplingParams())
    config = engine.vlm_config
    layers = capture_layers(config.text.num_hidden_layers)
    arch = vision_architecture_fingerprint(config)
    # No "logits": asking the probe for logits costs one full-vocab projection per
    # requested position, and the logits that get compared come from
    # ``ForwardResult`` so they are bit-identical to what generation samples from.
    captures = {"layer_output", "final_norm", "embed_sequence"}

    payload: dict[str, Any] = {
        "samples": [],
        "model_directory": str(args.model),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "side": "numpy",
        "architecture": arch,
        "tokenizer": [],
        "weight_digests": {},
    }
    blobs: dict[str, np.ndarray] = {}

    vision_layout: dict[str, Any] = {}
    for sample in VISION_SAMPLES:
        name = sample["name"]
        image_path = artifacts / vision_image_filename(sample)
        build_vision_image(sample["width"], sample["height"], sample["seed"]).save(image_path)
        image_digest = file_digest(image_path)

        prompt = engine.build_prompt(
            image_path, sample["question"], system=sample["system"]
        )
        features = engine.model.encode_image(prompt.image)
        positions = prompt.image_positions
        mask = np.zeros(len(prompt.token_ids), dtype=bool)
        mask[list(positions)] = True

        first, last = positions[0], positions[-1]
        full_positions = [0, first, last, -1]
        vision_layout[name] = {
            "num_frames": prompt.image.num_frames,
            "num_image_tokens": prompt.image.num_image_tokens,
            "prompt_tokens": len(prompt.token_ids),
            "first_image_position": first,
            "last_image_position": last,
            "positions_full": full_positions,
        }
        payload["tokenizer"].append(
            {
                "name": name,
                "text": prompt.text,
                "ids": [int(token) for token in prompt.token_ids],
            }
        )
        record: dict[str, Any] = {
            "name": name,
            "question": sample["question"],
            "system": sample["system"],
            "image": {
                "file": image_path.name,
                "sha256": image_digest,
                "width": sample["width"],
                "height": sample["height"],
                "original_size": list(prompt.image.original_size),
                "resized_size": list(prompt.image.resized_size),
                "encoder_size": list(prompt.image.encoder_size),
                "rows": prompt.image.rows,
                "cols": prompt.image.cols,
                "num_frames": prompt.image.num_frames,
                "num_image_tokens": prompt.image.num_image_tokens,
                "pixel_values_digest": tensor_digest(prompt.image.pixel_values),
                "pixel_attention_mask_digest": tensor_digest(
                    prompt.image.pixel_attention_mask.astype(np.float32)
                ),
                "pixel_attention_mask_all_valid": bool(prompt.image.pixel_attention_mask.all()),
            },
            "prompt_tokens": len(prompt.token_ids),
            "first_image_position": first,
            "last_image_position": last,
            "prefill_chunk_size": args.prefill_chunk_size,
            "decode_steps": [],
        }

        for frame in vision_capture_frames(prompt.image.num_frames):
            blobs[f"{name}.vision.encoder_output[{frame}]"] = features.encoder_output[frame]
        blobs[f"{name}.vision.connector_output"] = features.features
        print(
            f"[numpy] {name}: {prompt.image.num_frames} frames "
            f"({prompt.image.rows}x{prompt.image.cols} crops + global), "
            f"{prompt.image.num_image_tokens} image tokens, "
            f"{len(prompt.token_ids)} prompt tokens",
            file=sys.stderr,
        )

        budget = min(args.max_context, len(prompt.token_ids) + args.decode_steps + 8)

        # 1. full prefill: every token, and every image row, in one forward.
        cache = engine.model.new_cache(budget)
        probe = DebugProbe.for_names(captures, layers, full_positions)
        full = engine.model.forward(
            list(prompt.token_ids),
            cache,
            probe=probe,
            inputs_embeds=features.features,
            inputs_embed_mask=mask,
        )
        _collect_probe(blobs, f"{name}.full_prefill", probe)
        blobs[f"{name}.full_prefill.logits"] = full.logits
        record["full_prefill_argmax"] = int(greedy.sample(full.logits))
        del cache

        # 2. chunked prefill: same tokens and same features, several forwards.
        cache = engine.model.new_cache(budget)
        probe = DebugProbe.for_names(captures - {"embed_sequence"}, layers, (-1,))
        chunked = None
        feature_cursor = 0
        ids = list(prompt.token_ids)
        for start in range(0, len(ids), args.prefill_chunk_size):
            chunk = ids[start : start + args.prefill_chunk_size]
            chunk_mask = mask[start : start + len(chunk)]
            take = int(chunk_mask.sum())
            chunk_embeds = (
                features.features[feature_cursor : feature_cursor + take] if take else None
            )
            feature_cursor += take
            chunked = engine.model.forward(
                chunk,
                cache,
                probe=probe,
                inputs_embeds=chunk_embeds,
                inputs_embed_mask=chunk_mask,
            )
        assert chunked is not None
        assert feature_cursor == features.features.shape[0]
        _collect_probe(blobs, f"{name}.chunked_prefill", probe)
        blobs[f"{name}.chunked_prefill.logits"] = chunked.logits
        record["chunked_prefill_argmax"] = int(greedy.sample(chunked.logits))

        # 3. cached decode. No image features: Idefics3Model merges only when
        # past_seen_tokens == 0, so these steps are the ordinary text loop.
        logits = chunked.logits
        for step in range(args.decode_steps):
            token_id = int(greedy.sample(logits))
            step_probe = DebugProbe.for_names(
                captures - {"embed_sequence"}, layers, (-1,)
            )
            forward = engine.model.forward([token_id], cache, probe=step_probe)
            _collect_probe(blobs, f"{name}.decode{step}", step_probe)
            blobs[f"{name}.decode{step}.logits"] = forward.logits
            record["decode_steps"].append(
                {
                    "input_token": token_id,
                    "cache_length_before": forward.position_offset,
                    "argmax": int(greedy.sample(forward.logits)),
                }
            )
            logits = forward.logits

        payload["samples"].append(record)
        print(
            f"[numpy]   full={record['full_prefill_argmax']} "
            f"chunked={record['chunked_prefill_argmax']} "
            f"decode={[s['argmax'] for s in record['decode_steps']]}",
            file=sys.stderr,
        )

    payload["capture"] = capture_config(
        arch,
        prefill_chunk_size=args.prefill_chunk_size,
        decode_steps=args.decode_steps,
        vision=vision_layout,
    )
    payload["source"] = _source_record(
        engine.model.weights, payload["capture"]["weight_names"]
    )
    _digest_weights(engine.model.weights, payload)
    return _write_numpy_artifacts(artifacts, payload, blobs)


def _collect_probe(blobs: dict[str, np.ndarray], prefix: str, probe: Any) -> None:
    """Copy one probe's records into the blob dict under a stage prefix.

    Filters by name rather than taking ``probe.records`` wholesale, so a capture
    point this protocol does not grade cannot slip into the artifact and trip the
    "unexpected tensor" check on the other side.
    """
    for key, value in probe.records.items():
        if key == "embed_sequence" or key.startswith("layer") or key.startswith("final_norm"):
            blobs[f"{prefix}.{key}"] = value


def _digest_weights(weights: Any, payload: dict[str, Any]) -> None:
    """Converted-weight provenance: digests of whole tensors.

    Digesting rather than shipping keeps the gate over every element without
    moving a multi-hundred-MiB embedding table around.
    """
    for tensor_name in payload["capture"]["weight_names"]:
        payload["weight_digests"][tensor_name] = tensor_digest(weights.get(tensor_name))
        print(f"[numpy] digested {tensor_name}", file=sys.stderr)



# ---- reference side ---------------------------------------------------------


def decoder_module(model: Any) -> tuple[str, Any]:
    """The submodule that owns ``embed_tokens``, ``layers`` and ``norm``.

    Found by walking the module tree rather than assumed, because it is not in
    the same place for every task: ``Qwen3ForCausalLM`` nests it under
    ``model``, the bare ``Qwen3Model`` the embedding checkpoint publishes *is*
    the root, and ``Idefics3ForConditionalGeneration`` puts it under
    ``model.text_model``. Hooking the wrong level would capture the residual
    stream of something that is not the decoder.
    """
    for name, module in model.named_modules():
        if all(
            hasattr(module, attribute)
            for attribute in ("embed_tokens", "layers", "norm")
        ):
            return name, module
    raise SystemExit(
        "no submodule of the reference model exposes embed_tokens/layers/norm; "
        "this tool does not know where to attach its hooks"
    )


def reference_task(hf_dir: Path, claimed: str) -> str:
    """Re-derive the task from the official directory and require agreement.

    Uses the converter's own evidence rules on the *source* checkpoint, so the
    reference side is not simply trusting ``samples.json``: if SlowQ converted a
    directory as an embedding model, the official export must carry the
    sentence-transformers markers that say so.
    """
    tools_dir = str(Path(__file__).resolve().parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from convert_model import TASK_AUTO, detect_task

    detected, evidence = detect_task(hf_dir, TASK_AUTO)
    if detected != claimed:
        raise SystemExit(
            f"the converted directory records task={claimed!r} but the official "
            f"checkpoint at {hf_dir} says {detected!r} ({evidence}). These two "
            "sides are not describing the same model."
        )
    print(f"[ref] task {detected} ({evidence})", file=sys.stderr)
    return detected


def resolve_reference_tensor(
    state_dict: Mapping[str, Any], name: str, sources: Mapping[str, str]
) -> tuple[str, Any]:
    """Find a canonical SlowQ tensor name in the official state dict.

    Three readings, tried in order: the canonical name, the name the converter
    recorded as the source (the embedding export has no ``model.`` prefix), and
    — for ``lm_head.weight`` of a tied checkpoint — the embedding matrix it is an
    alias of. Which one matched is returned so the report can show it.
    """
    candidates = [name]
    if name in sources:
        candidates.append(sources[name])
    if name == LM_HEAD_NAME:
        candidates += ["model.embed_tokens.weight", "embed_tokens.weight"]
    for candidate in candidates:
        tensor = state_dict.get(candidate)
        if tensor is not None:
            return candidate, tensor
    raise SystemExit(
        f"{name} is absent from the reference checkpoint (tried "
        f"{', '.join(repr(c) for c in candidates)})"
    )


def _digest_reference_weights(
    state_dict: Mapping[str, Any], names: Sequence[str], sources: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Digest the official checkpoint's tensors under SlowQ's canonical names.

    ``.to(torch.float32)`` is exact for a BF16 source — the same bit shift the
    converter performs — so equal digests mean equal bytes, not equal within a
    tolerance. That is why ``rel_l2_weights`` is declared as 0.0.
    """
    import torch

    digests: dict[str, Any] = {}
    resolved: dict[str, str] = {}
    for name in names:
        found, tensor = resolve_reference_tensor(state_dict, name, sources)
        array = tensor.detach().to(torch.float32).numpy()
        digests[name] = tensor_digest(array)
        resolved[name] = found
        print(f"[ref] digested {name} (as {found})", file=sys.stderr)
        del array
    return digests, resolved


def run_reference_side(args: argparse.Namespace) -> int:
    import torch  # noqa: F401  (imported first so a missing torch fails early)
    import transformers

    artifacts = Path(args.artifacts)
    samples_path = artifacts / "samples.json"
    if not samples_path.is_file():
        raise SystemExit(
            f"{samples_path} not found; run 'validate_reference.py numpy' first"
        )
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    if not payload.get("samples"):
        raise SystemExit(f"{samples_path} records no samples; re-run 'validate_reference.py numpy'")
    hf_dir = Path(args.hf_dir)

    claimed_task = str((payload.get("capture") or {}).get("task") or "")
    task = reference_task(hf_dir, claimed_task)

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    raw_config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))

    processor = None
    if task == TASK_IMAGE_TEXT_GENERATION:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # The processor is part of the checkpoint, and the image half of it is
        # the official preprocessor: the reference must preprocess with it rather
        # than with any SlowQ code, or the pixel digests would compare SlowQ
        # against itself.
        processor = AutoProcessor.from_pretrained(hf_dir)
        tokenizer = processor.tokenizer
        model = AutoModelForImageTextToText.from_pretrained(
            hf_dir, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
    elif task == TASK_TEXT_EMBEDDING:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_dir)
        # The base model, not the causal-LM wrapper: the official recipe reads
        # ``last_hidden_state``, and loading a projection SlowQ does not have
        # would only add weights to compare and a way to be silently wrong.
        model = AutoModel.from_pretrained(
            hf_dir, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_dir)
        model = AutoModelForCausalLM.from_pretrained(
            hf_dir, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()

    prefix, decoder = decoder_module(model)
    arch = reference_architecture(raw_config, task)
    if prefix and prefix != "model":
        # SmolVLM's decoder lives under model.text_model; the canonical prefix
        # SlowQ stores it under is then that same path, so say so explicitly.
        arch = {**arch, "tensor_prefix": prefix}
    # Both knobs come from the recorded capture rather than from samples[0]: an
    # embedding sample has no decode steps to count and no per-sample chunk size,
    # and the capture block is what ``compare`` checks the two sides against.
    recorded_capture = payload.get("capture")
    if not isinstance(recorded_capture, Mapping):
        raise SystemExit(
            f"{samples_path} records no capture configuration; re-run "
            "'validate_reference.py numpy' with this version of the tool"
        )
    capture = capture_config(
        arch,
        prefill_chunk_size=int(recorded_capture["prefill_chunk_size"]),
        decode_steps=int(recorded_capture["decode_steps"]),
        vision=recorded_capture.get("vision"),
    )
    print(
        f"[ref] transformers {transformers.__version__}, torch {torch.__version__}, "
        f"{arch['family']} family, {arch['num_hidden_layers']} layers, decoder at "
        f"{prefix or '<root>'}, eager attention, fp32",
        file=sys.stderr,
    )

    weight_digests, resolved_names = _digest_reference_weights(
        model.state_dict(), capture["weight_names"], (payload.get("source") or {}).get("weight_sources") or {}
    )

    # Which bytes the reference actually loaded. The converter records the same
    # digests in its manifest, so ``compare`` can tie the two together without a
    # network call and without trusting a revision string.
    checkpoint_files = {
        path.name: file_digest(path) for path in sorted(hf_dir.glob("*.safetensors"))
    }
    if not checkpoint_files:
        raise SystemExit(f"no *.safetensors under {hf_dir}; nothing to validate against")

    hooks = _LayerHooks(decoder, capture["layers"])

    if task == TASK_TEXT_EMBEDDING:
        blobs, tokenizer_reference, samples_reference = _reference_embedding_capture(
            model, tokenizer, payload, hooks
        )
    elif task == TASK_IMAGE_TEXT_GENERATION:
        blobs, tokenizer_reference, samples_reference = _reference_vision_capture(
            model, processor, decoder, payload, hooks, capture, artifacts
        )
    else:
        blobs, tokenizer_reference, samples_reference = _reference_generation_capture(
            model, tokenizer, payload, hooks
        )
    hooks.remove()

    reference = {
        "side": "reference",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hf_dir": str(hf_dir),
        "architecture": arch,
        "source": {
            "repo_id": None,
            "revision": snapshot_revision(hf_dir),
            "files": checkpoint_files,
            "weight_sources": resolved_names,
        },
        "capture": capture,
        "environment": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "attn_implementation": "eager",
            "dtype": "float32",
            "threads": args.threads,
            "device": str(next(model.parameters()).device),
        },
        "tokenizer": tokenizer_reference,
        "weight_digests": weight_digests,
        "samples": samples_reference,
    }
    (artifacts / REFERENCE_FILENAME).write_text(
        json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(artifacts / "ref.npz", **blobs)
    print(
        f"[ref] wrote {artifacts / REFERENCE_FILENAME} and {artifacts / 'ref.npz'} "
        f"({len(blobs)} tensors)",
        file=sys.stderr,
    )
    del model
    return 0


class _LayerHooks:
    """Forward hooks on the decoder's layers and final norm.

    ``output_hidden_states`` cannot be used for this comparison: HF appends the
    hidden state *before* each layer and then, after the final norm, appends the
    post-norm tensor. So the last layer's raw output is never exposed and
    ``hidden_states[i + 1]`` silently means "post-norm" for i == num_layers-1.
    Hooks capture the decoder layers' actual outputs instead.
    """

    def __init__(self, decoder: Any, layers: Sequence[int]) -> None:
        self.layers = tuple(int(index) for index in layers)
        self.captured: dict[str, np.ndarray] = {}
        self._handles = [
            decoder.layers[index].register_forward_hook(self._hook(f"layer{index}"))
            for index in self.layers
        ]
        self._handles.append(decoder.norm.register_forward_hook(self._hook("final_norm")))

    def _hook(self, name: str):
        def _run(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self.captured[name] = hidden.detach().float().numpy().copy()

        return _run

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _reference_generation_capture(
    model: Any, tokenizer: Any, payload: Mapping[str, Any], hooks: "_LayerHooks"
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    blobs: dict[str, np.ndarray] = {}

    # 1. tokenizer content — recorded as data, never as a verdict. ``compare``
    # decides whether it agrees with the current SlowQ output.
    tokenizer_reference: list[dict[str, Any]] = []
    samples_reference: list[dict[str, Any]] = []
    for sample in payload["samples"]:
        rendered = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        reference_ids = [
            int(token) for token in tokenizer(rendered, add_special_tokens=False).input_ids
        ]
        tokenizer_reference.append(
            {
                "name": sample["name"],
                "text": rendered,
                "ids": reference_ids,
                "decode_input_tokens": [
                    int(step["input_token"]) for step in sample.get("decode_steps", [])
                ],
            }
        )
        print(f"[ref] {sample['name']}: {len(reference_ids)} prompt tokens", file=sys.stderr)

    with torch.no_grad():
        for sample in payload["samples"]:
            name = sample["name"]
            prompt_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long)

            def record(stage_prefix: str, logits: Any, positions: Iterable[int]) -> None:
                blobs[f"{stage_prefix}.logits"] = logits[0, -1, :].detach().float().numpy()
                for position in positions:
                    for layer in hooks.layers:
                        tensor = hooks.captured[f"layer{layer}"]
                        blobs[f"{stage_prefix}.layer{layer}.layer_output[{position}]"] = (
                            tensor[0][position]
                        )
                    blobs[f"{stage_prefix}.final_norm[{position}]"] = hooks.captured[
                        "final_norm"
                    ][0][position]

            # 2. full prefill
            output = model(prompt_ids, use_cache=False, return_dict=True)
            record(f"{name}.full_prefill", output.logits, CAPTURED_POSITIONS)

            # 3. chunked prefill through a cache
            chunk = sample["prefill_chunk_size"]
            past = None
            for start in range(0, prompt_ids.shape[1], chunk):
                piece = prompt_ids[:, start : start + chunk]
                output = model(piece, past_key_values=past, use_cache=True, return_dict=True)
                past = output.past_key_values
            record(f"{name}.chunked_prefill", output.logits, (-1,))

            # 4. cached decode, replaying SlowQ's forced token path
            for step, step_record in enumerate(sample["decode_steps"]):
                token = torch.tensor([[step_record["input_token"]]], dtype=torch.long)
                output = model(token, past_key_values=past, use_cache=True, return_dict=True)
                past = output.past_key_values
                record(f"{name}.decode{step}", output.logits, (-1,))

            samples_reference.append({"name": name})
            print(f"[ref] {name}: done", file=sys.stderr)

    return blobs, tokenizer_reference, samples_reference


def _reference_embedding_capture(
    model: Any, tokenizer: Any, payload: Mapping[str, Any], hooks: "_LayerHooks"
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """The official recipe, run by torch: pool the last valid token, L2-normalise.

    ``last_hidden_state`` is the output of the decoder's final RMSNorm, so it is
    the same tensor the ``norm`` hook captures — but it is read here through the
    model's *public* output and pooled with the model card's own index
    (``attention_mask.sum() - 1``), which is the quantity SlowQ has to agree on.

    The text to embed is rebuilt here, by :func:`reference_embedding_texts`, from
    the sample's raw query, instruction and role — not read from the NumPy
    artifact's ``text`` field. That field is the output under test: taking it
    would let a SlowQ that prefixes a query wrongly hand the reference the same
    wrong string, and the two sides would agree perfectly on the wrong input.

    The tokenizer is called with its **default** ``add_special_tokens``, because
    that is what the official snippet does, and for this checkpoint the default
    matters: ``tokenizer.json``'s post-processor appends ``<|fim_middle|>``, so
    every official vector was pooled at that token. SlowQ's embedding path passes
    ``True`` for the same reason, and the two sides' recorded IDs are compared
    element by element, so a SlowQ that dropped the post-processor would be
    named here rather than merely producing a slightly different vector.

    One forward per text feeds both stages, because the reference has no chunked
    path of its own: SlowQ's full prefill and its chunk loop must each reproduce
    this single result.
    """
    import torch
    from torch.nn.functional import normalize

    blobs: dict[str, np.ndarray] = {}
    tokenizer_reference: list[dict[str, Any]] = []
    samples_reference: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample in payload["samples"]:
            name = sample["name"]
            texts = reference_embedding_texts(name)
            recorded = [entry["name"] for entry in sample["texts"]]
            if recorded != [text_name for text_name, _, _, _ in texts]:
                raise SystemExit(
                    f"the NumPy artifact recorded embedding texts {recorded} for sample "
                    f"{name!r} but this version of the tool defines "
                    f"{[text_name for text_name, _, _, _ in texts]}; re-run "
                    "'validate_reference.py numpy'"
                )
            vectors: list[np.ndarray] = []
            record: dict[str, Any] = {"name": name, "texts": []}

            for text_name, raw, text, is_query in texts:
                encoded = tokenizer(text)
                ids = [int(token) for token in encoded["input_ids"]]
                attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long)
                input_ids = torch.tensor([ids], dtype=torch.long)

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                pool_index = int(attention_mask.sum()) - 1
                pooled = output.last_hidden_state[0, pool_index]
                vector = normalize(pooled.unsqueeze(0), p=2, dim=1)[0].numpy()

                prefix = f"{name}.{text_name}"
                for stage in ("full_prefill", "chunked_prefill"):
                    for layer in hooks.layers:
                        tensor = hooks.captured[f"layer{layer}"]
                        blobs[f"{prefix}.{stage}.layer{layer}.layer_output[-1]"] = tensor[0][-1]
                    blobs[f"{prefix}.{stage}.final_norm[-1]"] = hooks.captured["final_norm"][0][-1]
                    blobs[f"{prefix}.{stage}.vector"] = vector

                tokenizer_reference.append(
                    {"name": f"{name}/{text_name}", "text": text, "ids": ids}
                )
                record["texts"].append(
                    {
                        "name": text_name,
                        "raw": raw,
                        "text": text,
                        "is_query": is_query,
                        "num_tokens": len(ids),
                        "pool_index": pool_index,
                        "dimension": int(vector.shape[0]),
                    }
                )
                vectors.append(vector)
                print(
                    f"[ref] {name}/{text_name}: {len(ids)} tokens, pool_index {pool_index}",
                    file=sys.stderr,
                )

            stacked = np.stack(vectors)
            blobs[f"{name}.similarity"] = (stacked @ stacked.T).astype(np.float32)
            samples_reference.append(record)

    return blobs, tokenizer_reference, samples_reference


def vision_modules(model: Any) -> tuple[Any, Any]:
    """The vision tower and the connector, found by walking rather than assumed.

    Same reason as :func:`decoder_module`: this tool already had to discover
    where the decoder lives, and a wrapper that nests the vision half somewhere
    else should be named here rather than raising an ``AttributeError`` from
    inside a hook after a multi-minute forward.
    """
    vision = connector = None
    for _name, module in model.named_modules():
        if vision is None and hasattr(module, "post_layernorm") and hasattr(module, "embeddings"):
            vision = module
        if connector is None and hasattr(module, "modality_projection"):
            connector = module
    if vision is None or connector is None:
        raise SystemExit(
            "no submodule of the reference model looks like the vision tower "
            "(post_layernorm + embeddings) or the connector (modality_projection); "
            "this tool does not know where to attach its vision hooks"
        )
    return vision, connector


class _VisionHooks:
    """The three places the official vision path is observable, segment by segment.

    ``post_layernorm``'s output *is* ``Idefics3VisionTransformer``'s
    ``last_hidden_state``, and the connector's output is the pixel-shuffled,
    text-space features ``inputs_merger`` assigns from. The merged
    ``inputs_embeds`` is different: it exists only as a local variable inside
    ``Idefics3Model.forward``, so it is read with a forward-*pre*-hook on the
    decoder — the last point at which the merge has happened and the first at
    which anything consumes it.

    These are removed before the chunked prefill. That phase calls the decoder
    directly with slices of the merged matrix, and leaving the pre-hook attached
    would overwrite the whole-sequence tensor with one chunk's.
    """

    def __init__(self, model: Any, decoder: Any) -> None:
        vision, connector = vision_modules(model)
        self.captured: dict[str, np.ndarray] = {}
        self._handles = [
            vision.post_layernorm.register_forward_hook(self._store("encoder_output")),
            connector.register_forward_hook(self._store("connector_output")),
            decoder.register_forward_pre_hook(self._store_merged, with_kwargs=True),
        ]

    def _store(self, name: str):
        def _run(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.captured[name] = tensor.detach().float().numpy().copy()

        return _run

    def _store_merged(self, _module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            raise SystemExit(
                "the reference decoder was called without inputs_embeds. "
                "Idefics3Model.forward always hands it the merged matrix that way, "
                "so this forward is not the one the hook was placed for and the "
                "fused sequence was not captured."
            )
        self.captured["merged"] = embeds.detach().float().numpy().copy()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _reference_image_sizes(
    image_processor: Any, image: Any
) -> tuple[list[int], list[int], list[int], int, int, int]:
    """The official pipeline's own intermediate sizes and crop grid.

    ``Idefics3ImageProcessor.preprocess`` returns only the finished frames, so
    these come from calling the same three official steps it calls — ``resize``,
    ``resize_for_vision_encoder`` and ``split_image`` — and reading each result's
    shape. That is why SlowQ's ``resized_size`` and ``encoder_size`` are
    comparable rather than merely similar: both sides are reporting the official
    functions' output, and a disagreement localises the failing step instead of
    leaving one opaque pixel-digest mismatch to explain.

    Sizes are ``(height, width)``, the order ``get_image_size`` returns and the
    order ``PreparedImage`` records.
    """
    from transformers.image_utils import (
        get_image_size,
        infer_channel_dimension_format,
        to_numpy_array,
    )

    array = to_numpy_array(image)
    data_format = infer_channel_dimension_format(array)
    resized = image_processor.resize(
        image=array,
        size=image_processor.size,
        resample=image_processor.resample,
        input_data_format=data_format,
    )
    encoder_input = image_processor.resize_for_vision_encoder(
        resized,
        image_processor.max_image_size["longest_edge"],
        resample=image_processor.resample,
        input_data_format=data_format,
    )
    frames, rows, cols = image_processor.split_image(
        encoder_input,
        max_image_size=image_processor.max_image_size,
        input_data_format=data_format,
    )
    return (
        list(get_image_size(array, data_format)),
        list(get_image_size(resized, data_format)),
        list(get_image_size(encoder_input, data_format)),
        int(rows),
        int(cols),
        len(frames),
    )


def _reference_vision_capture(
    model: Any,
    processor: Any,
    decoder: Any,
    payload: Mapping[str, Any],
    hooks: "_LayerHooks",
    capture: Mapping[str, Any],
    artifacts: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """The official image+text path, run by torch and captured segment by segment.

    Four stages per sample, in the order the protocol grades them:

    1. **preprocessing** — the official ``Idefics3ImageProcessor`` over the PNG
       the NumPy side wrote, whose digest is re-checked here first. Both sides
       read that one file, so a pixel-tensor difference afterwards is the
       preprocessing's fault and not the picture's.
    2. **prompt expansion** — ``apply_chat_template`` for the skeleton, then the
       official ``get_image_prompt_string`` for what one ``<image>`` becomes. The
       re-encoded expansion is *required* to reproduce the processor's own
       ``input_ids``; without that equality, comparing SlowQ's expanded text
       against a hand-built string would prove nothing.
    3. **vision tower and connector** — through :class:`_VisionHooks`.
    4. **decoder** — one full prefill through the wrapper, then the chunked
       prefill and the cached decode.

    The chunked prefill deliberately does not go through the wrapper.
    ``Idefics3Model.forward`` merges image features only when
    ``past_seen_tokens == 0`` *and* refuses ``inputs_embeds`` without
    ``input_ids`` on that first call, so feeding it chunk by chunk would embed
    every later chunk's ``<image>`` tokens from the table — and a reference that
    was wrong there would grade SlowQ's correct chunking as a failure. What the
    official forward does instead is merge once and then call the decoder, so
    that is what is reproduced: slices of the matrix the hook captured, into the
    decoder through a cache, with ``lm_head`` applied by hand to the last chunk,
    which is the wrapper's own final step.
    """
    import torch
    from PIL import Image
    from transformers.models.idefics3.processing_idefics3 import get_image_prompt_string

    blobs: dict[str, np.ndarray] = {}
    tokenizer_reference: list[dict[str, Any]] = []
    samples_reference: list[dict[str, Any]] = []
    image_processor = processor.image_processor
    image_token = processor.image_token.content
    image_token_id = int(model.config.image_token_id)
    image_seq_len = int(processor.image_seq_len)
    vision_layout = capture.get("vision") or {}

    with torch.no_grad():
        for sample in payload["samples"]:
            name = sample["name"]
            ours_image = sample.get("image")
            recorded_layout = vision_layout.get(name)
            if not isinstance(ours_image, Mapping) or not isinstance(recorded_layout, Mapping):
                raise SystemExit(
                    f"sample {name!r} records no image metadata or no vision layout; "
                    "these artifacts were written by an older version of this tool — "
                    "re-run 'validate_reference.py numpy'"
                )

            # 1. the same pixels both sides saw
            image_path = artifacts / str(ours_image["file"])
            if not image_path.is_file():
                raise SystemExit(
                    f"{image_path} is missing; the NumPy side writes the sample image "
                    "there so that both sides preprocess identical bytes — re-run "
                    "'validate_reference.py numpy'"
                )
            observed = file_digest(image_path)
            if observed != ours_image["sha256"]:
                raise SystemExit(
                    f"{image_path.name} has sha256 {observed} but the NumPy side recorded "
                    f"{ours_image['sha256']}. The two sides would be preprocessing "
                    "different pictures, so nothing downstream is comparable."
                )
            image = Image.open(image_path)
            image.load()

            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sample["system"]}],
                }
            ] if sample.get("system") is not None else []
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": sample["question"]},
                    ],
                }
            )

            # 2. the official prompt expansion
            template = processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            if template.count(image_token) != 1:
                raise SystemExit(
                    f"sample {name!r}: the rendered chat template contains "
                    f"{template.count(image_token)} {image_token!r} tokens, so there is "
                    "no single place to expand the image into"
                )
            sizes = _reference_image_sizes(image_processor, image)
            rows, cols, split_frames = sizes[3], sizes[4], sizes[5]
            expanded = template.replace(
                image_token,
                get_image_prompt_string(
                    rows,
                    cols,
                    image_seq_len,
                    fake_token_around_image=processor.fake_image_token.content,
                    image_token=image_token,
                    global_img_token=processor.global_image_tag,
                ),
            )

            encoded = processor(text=template, images=[image], return_tensors="pt")
            ids = [int(token) for token in encoded["input_ids"][0].tolist()]
            reencoded = [
                int(token) for token in processor.tokenizer(expanded)["input_ids"]
            ]
            if reencoded != ids:
                raise SystemExit(
                    f"sample {name!r}: re-encoding the expanded prompt gives "
                    f"{len(reencoded)} tokens but the official processor gives {len(ids)}. "
                    "This tool cannot reconstruct the expansion the processor used, so "
                    "the text comparison would be against the wrong string."
                )
            positions = [index for index, token in enumerate(ids) if token == image_token_id]

            pixel_values = encoded["pixel_values"].numpy()
            pixel_attention_mask = encoded["pixel_attention_mask"].numpy()
            num_frames = int(pixel_values.shape[1])
            # ``Idefics3Model.forward`` drops frames that are entirely zero before
            # the tower ever sees them. One dropped here would shift every frame
            # index the two sides compare, so it is counted and reported rather
            # than left to surface as a mysterious mismatch.
            squeezed = pixel_values.reshape(-1, *pixel_values.shape[2:])
            dropped = int((squeezed == 0.0).all(axis=(1, 2, 3)).sum())
            if dropped:
                print(
                    f"[ref] {name}: WARNING the official forward would drop {dropped} "
                    "all-zero frame(s); frame indices are not comparable",
                    file=sys.stderr,
                )

            if len(positions) != num_frames * image_seq_len:
                raise SystemExit(
                    f"sample {name!r}: the prompt has {len(positions)} {image_token!r} "
                    f"tokens but {num_frames} frames x image_seq_len {image_seq_len} = "
                    f"{num_frames * image_seq_len}. The merge cannot be described."
                )
            if num_frames != split_frames:
                raise SystemExit(
                    f"sample {name!r}: the processor padded the image to {num_frames} "
                    f"frames but split_image produced {split_frames}"
                )

            image_record = {
                "file": image_path.name,
                "sha256": observed,
                "width": int(image.width),
                "height": int(image.height),
                "original_size": sizes[0],
                "resized_size": sizes[1],
                "encoder_size": sizes[2],
                "rows": rows,
                "cols": cols,
                "num_frames": num_frames,
                "num_image_tokens": len(positions),
                "pixel_values_digest": tensor_digest(squeezed),
                "pixel_attention_mask_digest": tensor_digest(
                    pixel_attention_mask.reshape(-1, *pixel_attention_mask.shape[2:]).astype(
                        np.float32
                    )
                ),
                "pixel_attention_mask_all_valid": bool(pixel_attention_mask.all()),
                "dropped_padding_frames": dropped,
            }

            tokenizer_reference.append(
                {
                    "name": name,
                    "text": expanded,
                    "ids": ids,
                    "decode_input_tokens": [
                        int(step["input_token"]) for step in sample.get("decode_steps", [])
                    ],
                }
            )
            record: dict[str, Any] = {
                "name": name,
                "template_text": template,
                "image": image_record,
                "prompt_tokens": len(ids),
                "first_image_position": positions[0],
                "last_image_position": positions[-1],
                "decode_steps": [],
            }

            # 3 + 4. full prefill through the official wrapper
            vision_hooks = _VisionHooks(model, decoder)
            output = model(
                input_ids=torch.tensor([ids], dtype=torch.long),
                attention_mask=encoded["attention_mask"],
                pixel_values=encoded["pixel_values"],
                pixel_attention_mask=encoded["pixel_attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            encoder_output = vision_hooks.captured["encoder_output"]
            connector_output = vision_hooks.captured["connector_output"]
            merged = torch.from_numpy(vision_hooks.captured["merged"])
            vision_hooks.remove()

            for frame in vision_capture_frames(num_frames):
                blobs[f"{name}.vision.encoder_output[{frame}]"] = encoder_output[frame]
            # ``inputs_merger`` consumes ``image_hidden_states.view(-1, hidden)``,
            # so the flattened frame-major layout is what the merge assigns from
            # and what SlowQ's ``VisionOutput.features`` is compared against.
            blobs[f"{name}.vision.connector_output"] = connector_output.reshape(
                -1, connector_output.shape[-1]
            )
            blobs[f"{name}.full_prefill.embed_sequence"] = merged[0]

            def record_stage(stage: str, logits: Any, positions_to_store: Sequence[int]) -> None:
                blobs[f"{name}.{stage}.logits"] = logits[0, -1, :].detach().float().numpy()
                for position in positions_to_store:
                    for layer in hooks.layers:
                        blobs[f"{name}.{stage}.layer{layer}.layer_output[{position}]"] = (
                            hooks.captured[f"layer{layer}"][0][position]
                        )
                    blobs[f"{name}.{stage}.final_norm[{position}]"] = hooks.captured[
                        "final_norm"
                    ][0][position]

            record_stage("full_prefill", output.logits, recorded_layout["positions_full"])
            record["full_prefill_argmax"] = int(output.logits[0, -1, :].argmax())

            # chunked prefill: merge once, then the decoder in slices
            chunk = int(recorded_capture_chunk_size(payload, name))
            past = None
            result = None
            for start in range(0, merged.shape[1], chunk):
                result = decoder(
                    inputs_embeds=merged[:, start : start + chunk],
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = result.past_key_values
            assert result is not None
            chunked_logits = model.lm_head(result.last_hidden_state[:, -1:, :])
            record_stage("chunked_prefill", chunked_logits, capture["positions_cached"])
            record["chunked_prefill_argmax"] = int(chunked_logits[0, 0, :].argmax())

            # cached decode, replaying SlowQ's forced token path
            for step, step_record in enumerate(sample["decode_steps"]):
                token = torch.tensor([[step_record["input_token"]]], dtype=torch.long)
                output = model(
                    input_ids=token,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = output.past_key_values
                record_stage(f"decode{step}", output.logits, capture["positions_cached"])
                record["decode_steps"].append({"argmax": int(output.logits[0, -1, :].argmax())})

            samples_reference.append(record)
            print(
                f"[ref] {name}: {rows}x{cols} grid -> {num_frames} frames, "
                f"{len(positions)} image tokens, {len(ids)} prompt tokens, "
                f"full={record['full_prefill_argmax']} "
                f"chunked={record['chunked_prefill_argmax']} "
                f"decode={[s['argmax'] for s in record['decode_steps']]}",
                file=sys.stderr,
            )

    return blobs, tokenizer_reference, samples_reference


def recorded_capture_chunk_size(payload: Mapping[str, Any], name: str) -> int:
    """The chunk size the NumPy side used for this sample.

    Read from the sample rather than from the capture block because that is where
    the generation protocol records it too, and a capture that disagreed with the
    sample it describes should not silently pick one of the two.
    """
    for sample in payload["samples"]:
        if sample["name"] == name:
            return int(sample["prefill_chunk_size"])
    raise SystemExit(f"sample {name!r} is absent from samples.json; re-run 'numpy'")


# ---- comparison -------------------------------------------------------------


def load_metadata(path: Path, problems: list[str], hint: str) -> dict[str, Any] | None:
    if not path.is_file():
        problems.append(f"{path.name} not found — {hint}")
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"{path.name} is not valid JSON: {exc}")
        return None
    if not isinstance(loaded, dict):
        problems.append(f"{path.name} does not contain an object")
        return None
    return loaded


def load_blobs(path: Path, problems: list[str], hint: str) -> np.lib.npyio.NpzFile | None:
    if not path.is_file():
        problems.append(f"{path.name} not found — {hint}")
        return None
    try:
        return np.load(path)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        problems.append(f"{path.name} could not be read: {exc}")
        return None


def check_capture(
    ours: Mapping[str, Any], theirs: Mapping[str, Any], problems: list[str]
) -> dict[str, Any] | None:
    """Require the two runs to describe the same capture, and the code to agree.

    Returns the capture configuration to derive the expected keys from, or None
    if the artifacts cannot be trusted far enough to know what that set is.
    """
    print("\n[capture] does the recorded reference describe this run?")
    usable = True

    def row(label: str, ours_value: Any, theirs_value: Any, required: Any = None) -> None:
        nonlocal usable
        agree = ours_value == theirs_value
        current = True if required is None else ours_value == required
        status = "ok" if agree and current else "MISMATCH"
        print(f"  {label:<26} slowq={ours_value!r} ref={theirs_value!r} [{status}]")
        if not agree:
            problems.append(
                f"capture mismatch: {label} is {ours_value!r} on the SlowQ side but "
                f"{theirs_value!r} on the reference side — re-capture the reference"
            )
            usable = False
        if not current:
            problems.append(
                f"capture mismatch: {label} is {ours_value!r} in the artifacts but "
                f"{required!r} in this version of the tool — re-capture the reference"
            )
            usable = False

    ours_capture = ours.get("capture")
    theirs_capture = theirs.get("capture")
    if not isinstance(ours_capture, Mapping) or not isinstance(theirs_capture, Mapping):
        problems.append(
            "neither artifact records a capture configuration; these were written by an "
            "older version of this tool — re-run 'numpy' and re-capture 'ref'"
        )
        return None

    architecture = ours.get("architecture")
    if not isinstance(architecture, Mapping) or "num_hidden_layers" not in architecture:
        problems.append(
            "the SlowQ artifact records no architecture fingerprint; re-run 'numpy'"
        )
        return None
    try:
        required = capture_config(
            architecture,
            prefill_chunk_size=int(ours_capture.get("prefill_chunk_size", 0)),
            decode_steps=int(ours_capture.get("decode_steps", 0)),
            vision=ours_capture.get("vision"),
        )
    except (ValueError, KeyError, ComparisonError) as exc:
        problems.append(f"the recorded architecture cannot describe a capture: {exc}")
        return None
    rows = (
        ("artifact_format", "artifact_format"),
        ("task", "task"),
        ("family", "family"),
        ("sample_names", "sample_names"),
        ("layers", "layers"),
        ("weight_names", "weight_names"),
        ("prefill_chunk_size", "prefill_chunk_size"),
        ("decode_steps", "decode_steps"),
    )
    task = str(ours_capture.get("task"))
    if task == TASK_TEXT_EMBEDDING:
        rows += (
            ("embedding_texts", "embedding_texts"),
            ("embedding_input_format", "embedding_input_format"),
        )
    elif task == TASK_IMAGE_TEXT_GENERATION:
        # The per-sample layout travels in the capture block because only the
        # NumPy side can produce it with SlowQ's own tokenizer; both sides and
        # ``required`` then carry the same thing, and the row below is what makes
        # a reference captured from a different image or a different crop grid
        # fail instead of comparing misaligned frames.
        rows += (
            ("positions (cached)", "positions_cached"),
            ("vision_layers", "vision_layers"),
            ("vision layout", "vision"),
        )
    else:
        rows += (
            ("positions (full)", "positions_full"),
            ("positions (cached)", "positions_cached"),
        )
    for label, key in rows:
        row(label, ours_capture.get(key), theirs_capture.get(key), required[key])
    # Printed by version rather than by value: the full spec is a paragraph, and
    # it is recorded in both artifacts for anyone who wants to read it.
    ours_spec = ours_capture.get("digest_spec")
    theirs_spec = theirs_capture.get("digest_spec")
    if ours_spec == theirs_spec == WEIGHT_DIGEST_SPEC:
        print(
            f"  {'digest spec':<26} {WEIGHT_DIGEST_SPEC['version']} "
            f"({WEIGHT_DIGEST_SPEC['algorithm']} over the whole tensor) [ok]"
        )
    else:
        print(f"  {'digest spec':<26} MISMATCH")
        if ours_spec != theirs_spec:
            problems.append(
                "capture mismatch: the two sides digested weights under different specs "
                f"({ours_spec!r} vs {theirs_spec!r}) — re-capture the reference"
            )
        if ours_spec != WEIGHT_DIGEST_SPEC:
            problems.append(
                f"capture mismatch: the artifacts use weight digest spec {ours_spec!r} but "
                f"this version of the tool uses {WEIGHT_DIGEST_SPEC!r} — re-capture both sides"
            )
        usable = False

    ours_arch = ours.get("architecture")
    theirs_arch = theirs.get("architecture")
    if ours_arch != theirs_arch:
        print(f"  {'architecture':<26} MISMATCH")
        problems.append(
            f"the reference was captured from a different architecture: "
            f"{theirs_arch!r} vs the current model's {ours_arch!r}"
        )
        usable = False
    else:
        print(
            f"  {'architecture':<26} {ours_arch.get('family')} / {ours_arch.get('task')}, "
            f"{ours_arch.get('num_hidden_layers')} layers, vocab "
            f"{ours_arch.get('vocab_size')}, head_dim {ours_arch.get('head_dim')}, "
            f"qkv_bias={ours_arch.get('qkv_bias')} qk_norm={ours_arch.get('qk_norm')} [ok]"
        )
        if task == TASK_IMAGE_TEXT_GENERATION:
            print(
                f"  {'vision tower':<26} {ours_arch.get('decoder_family')} decoder, "
                f"{ours_arch.get('vision_num_hidden_layers')} vision layers, "
                f"image_seq_len={ours_arch.get('image_seq_len')} "
                f"(scale_factor {ours_arch.get('scale_factor')}), "
                f"connector_out={ours_arch.get('connector_out_features')}, "
                f"image_token_id={ours_arch.get('image_token_id')} [ok]"
            )

    if not usable:
        return None
    return dict(ours_capture)


def check_provenance(ours: Mapping[str, Any], theirs: Mapping[str, Any], problems: list[str]) -> None:
    """Tie the converted model to the bytes the reference actually loaded.

    The manifest records the sha256 the hub reported for each source file; the
    reference side hashes the files it read. Matching those two needs no network
    and no trust in a revision string.
    """
    print("\n[provenance] the checkpoint the reference read vs the one SlowQ claims")
    ours_files = (ours.get("source") or {}).get("files") or {}
    theirs_files = (theirs.get("source") or {}).get("files") or {}
    if not theirs_files:
        problems.append("the reference records no checkpoint file digests; re-capture it")
        return
    # Only shards are comparable: the manifest carries no digest for the small
    # non-LFS files (config.json and friends), and the reference does not read them.
    claimed_shards = {
        name: digest for name, digest in ours_files.items() if name.endswith(".safetensors")
    }
    if set(claimed_shards) != set(theirs_files):
        problems.append(
            f"the manifest claims shards {sorted(claimed_shards)} but the reference read "
            f"{sorted(theirs_files)} — the two sides are not looking at one checkpoint"
        )
    for name in sorted(set(claimed_shards) | set(theirs_files)):
        claimed = claimed_shards.get(name)
        observed = theirs_files.get(name)
        if claimed is None:
            status = "NOT CLAIMED BY THE MANIFEST"
            problems.append(
                f"{name}: the reference loaded this shard but the converted model's "
                f"manifest does not claim it"
            )
        elif observed is None:
            status = "NOT LOADED BY THE REFERENCE"
            problems.append(
                f"{name}: the manifest claims this shard but the reference never read it"
            )
        elif claimed != observed:
            status = "DIFFERENT BYTES"
            problems.append(
                f"{name}: the manifest claims sha256 {claimed} but the reference read "
                f"{observed} — the converted model did not come from these weights"
            )
        else:
            status = f"sha256 {observed[:16]}… ok"
        print(f"  {name:<40} {status}")

    ours_revision = (ours.get("source") or {}).get("revision")
    theirs_revision = (theirs.get("source") or {}).get("revision")
    if theirs_revision is None:
        print(f"  {'revision':<40} not recorded by the reference (no snapshot marker)")
        print("    the file digests above are what tie the two sides together")
    elif ours_revision != theirs_revision:
        print(f"  {'revision':<40} MISMATCH")
        problems.append(
            f"the reference was captured from revision {theirs_revision} but the "
            f"converted model is {ours_revision} — re-capture the reference"
        )
    else:
        print(f"  {'revision':<40} {ours_revision} [ok]")


def expected_tokenizer_names(capture: Mapping[str, Any] | None) -> list[str]:
    """Every tokenizer entry a complete capture must record.

    A generation capture renders one prompt per sample; an embedding capture
    tokenises one text per query *and* per document, so the entries are named
    ``<sample>/<text>`` and both sides must agree on that set.

    With no usable capture there is no set to require, so this returns nothing.
    The naming rule follows the task, and an unusable capture is one whose task
    cannot be trusted: falling back to the generation names there reported an
    embedding capture's four query/document entries as "missing from both sides"
    under prompt names it never used — four invented problems burying the real
    one. ``check_tokenizer`` still compares every entry both sides did record.
    """
    if not isinstance(capture, Mapping):
        return []
    if capture.get("task") == TASK_TEXT_EMBEDDING:
        texts = capture.get("embedding_texts") or {}
        return [
            f"{name}/{text}"
            for name in (capture.get("sample_names") or [])
            for text in texts.get(name, [])
        ]
    return expected_sample_names(capture)


def check_tokenizer(
    ours: Mapping[str, Any],
    theirs: Mapping[str, Any],
    problems: list[str],
    capture: Mapping[str, Any] | None = None,
) -> None:
    """Compare *content* recorded by the reference against this run's output."""
    task = str((capture or {}).get("task") or TASK_TEXT_GENERATION)
    print("\n[tokenizer] rendered text and token IDs vs AutoTokenizer")
    ours_entries = {entry["name"]: entry for entry in ours.get("tokenizer", [])}
    theirs_entries = {entry["name"]: entry for entry in theirs.get("tokenizer", [])}
    if not theirs_entries:
        problems.append(
            "the reference records no tokenizer content; these artifacts predate the "
            "content-based check — re-capture the reference"
        )
        return
    expected = set(expected_tokenizer_names(capture))
    for name in sorted(expected | set(ours_entries) | set(theirs_entries)):
        ours_entry = ours_entries.get(name)
        theirs_entry = theirs_entries.get(name)
        if ours_entry is None and theirs_entry is None:
            # Iterating the union of the two files alone hides this completely:
            # a sample nobody recorded is a sample nobody compared.
            problems.append(
                f"tokenizer sample {name!r} is missing from both sides: the capture "
                "declares it but neither artifact records it — re-run 'numpy' and "
                "re-capture 'ref'"
            )
            print(f"  {name:<24} MISSING from both sides")
            continue
        if ours_entry is None or theirs_entry is None:
            side = "SlowQ" if ours_entry is None else "reference"
            problems.append(f"tokenizer sample {name!r} is missing from the {side} side")
            print(f"  {name:<24} MISSING from {side}")
            continue
        text_ok = ours_entry["text"] == theirs_entry["text"]
        ids_ok = list(ours_entry["ids"]) == list(theirs_entry["ids"])
        status = "ok" if text_ok and ids_ok else "MISMATCH"
        print(
            f"  {name:<24} text={text_ok} ids={ids_ok} "
            f"tokens={len(theirs_entry['ids'])} [{status}]"
        )
        if not text_ok:
            kind = {
                TASK_TEXT_GENERATION: "chat text",
                TASK_IMAGE_TEXT_GENERATION: "expanded image+text prompt",
            }.get(task, "input text")
            problems.append(
                f"tokenizer sample {name!r}: SlowQ renders different {kind} than the "
                f"reference did (first difference at character "
                f"{_first_difference(ours_entry['text'], theirs_entry['text'])})"
            )
        if not ids_ok:
            problems.append(
                f"tokenizer sample {name!r}: SlowQ produced different token IDs than the "
                f"reference did ({len(ours_entry['ids'])} vs {len(theirs_entry['ids'])} tokens)"
            )

    # The decode path the reference replayed must be the one SlowQ just walked.
    ours_samples = {sample["name"]: sample for sample in ours.get("samples", [])}
    for name, theirs_entry in sorted(theirs_entries.items()):
        ours_sample = ours_samples.get(name)
        if ours_sample is None:
            continue
        replayed = list(theirs_entry.get("decode_input_tokens", []))
        walked = [int(step["input_token"]) for step in ours_sample.get("decode_steps", [])]
        if replayed != walked:
            problems.append(
                f"sample {name!r}: the reference replayed decode tokens {replayed} but "
                f"this run walked {walked} — the decode logits are not comparable"
            )
            print(f"  {name:<24} decode path MISMATCH {replayed} vs {walked}")


def _first_difference(ours: str, theirs: str) -> int:
    for index, (ours_char, theirs_char) in enumerate(zip(ours, theirs)):
        if ours_char != theirs_char:
            return index
    return min(len(ours), len(theirs))


def check_weights(ours: Mapping[str, Any], theirs: Mapping[str, Any], problems: list[str]) -> None:
    """Re-digest the current converted weights against the recorded digests.

    The expected name set is derived from this run's architecture fingerprint,
    not from the recorded capture: a capture that wrote down the wrong names is
    then caught here instead of being trusted, and a family's legitimate absence
    of ``q_norm`` or of a projection bias is a shorter list rather than a failure.
    """
    print("\n[weights] converted FP32 vs BF16 upcast of the official checkpoint")
    architecture = ours.get("architecture")
    if not isinstance(architecture, Mapping):
        problems.append("the SlowQ artifact records no architecture; cannot derive the weights to check")
        return
    try:
        expected_names = expected_weight_names(architecture)
    except (KeyError, ValueError) as exc:
        problems.append(f"the recorded architecture cannot select weights to digest: {exc}")
        return

    recorded = (ours.get("capture") or {}).get("weight_names")
    if recorded is not None and list(recorded) != list(expected_names):
        problems.append(
            "the capture recorded weight names that this version of the tool would "
            f"not choose for this architecture: {list(recorded)} vs {list(expected_names)} "
            "— re-run 'numpy' and re-capture 'ref'"
        )

    ours_digests = ours.get("weight_digests") or {}
    theirs_digests = theirs.get("weight_digests") or {}
    if not theirs_digests:
        problems.append(
            "the reference records no weight digests; these artifacts predate the "
            "digest-based check — re-capture the reference"
        )
        return
    resolved = (theirs.get("source") or {}).get("weight_sources") or {}
    expected = set(expected_names)
    for name in sorted(expected | set(ours_digests) | set(theirs_digests)):
        ours_entry = ours_digests.get(name)
        theirs_entry = theirs_digests.get(name)
        if ours_entry is None and theirs_entry is None:
            problems.append(
                f"weight {name}: missing from both sides — this protocol digests it but "
                "neither artifact records it; re-run 'numpy' and re-capture 'ref'"
            )
            print(f"  {name:<44} MISSING from both sides")
            continue
        if ours_entry is None or theirs_entry is None:
            side = "SlowQ" if ours_entry is None else "reference"
            problems.append(f"weight {name}: not digested on the {side} side")
            print(f"  {name:<44} MISSING from {side}")
            continue
        for field in ("non_finite_elements",):
            for label, entry in (("SlowQ", ours_entry), ("reference", theirs_entry)):
                if entry.get(field):
                    problems.append(
                        f"weight {name}: {entry[field]} non-finite value(s) on the {label} side"
                    )
        if ours_entry.get("spec") != DIGEST_SPEC_VERSION:
            problems.append(
                f"weight {name}: digested with spec {ours_entry.get('spec')!r}, this tool "
                f"uses {DIGEST_SPEC_VERSION!r}"
            )
        same_shape = ours_entry.get("shape") == theirs_entry.get("shape")
        same_dtype = ours_entry.get("dtype") == theirs_entry.get("dtype")
        same_bytes = ours_entry.get("sha256") == theirs_entry.get("sha256")
        status = "ok" if same_shape and same_dtype and same_bytes else "DIFFERS"
        found = resolved.get(name)
        alias = "" if found in (None, name) else f" as {found}"
        print(
            f"  {name:<44}{alias} shape={ours_entry.get('shape')} "
            f"dtype={ours_entry.get('dtype')} elements={ours_entry.get('elements')} [{status}]"
        )
        if not same_shape:
            problems.append(
                f"weight {name}: shape {ours_entry.get('shape')} vs reference "
                f"{theirs_entry.get('shape')}"
            )
        if not same_dtype:
            problems.append(
                f"weight {name}: dtype {ours_entry.get('dtype')} vs reference "
                f"{theirs_entry.get('dtype')}"
            )
        if not same_bytes:
            problems.append(
                f"weight {name}: sha256 {ours_entry.get('sha256')} differs from the "
                f"reference's {theirs_entry.get('sha256')} — the conversion is not exact"
            )


def check_numerics(
    ours: np.lib.npyio.NpzFile,
    theirs: np.lib.npyio.NpzFile,
    capture: Mapping[str, Any],
    problems: list[str],
) -> list[Comparison]:
    """Grade every expected tensor, and complain about any unexpected one."""
    expected = expected_numeric_keys(capture)
    ours_keys = set(ours.files)
    theirs_keys = set(theirs.files)

    for label, keys in (("SlowQ", ours_keys), ("reference", theirs_keys)):
        missing = sorted(expected.keys() - keys)
        if missing:
            shown = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
            problems.append(
                f"{len(missing)} expected tensor(s) missing from the {label} side: {shown}"
            )
        unexpected = sorted(keys - set(expected))
        if unexpected:
            shown = ", ".join(unexpected[:5]) + (" …" if len(unexpected) > 5 else "")
            problems.append(
                f"{len(unexpected)} unexpected tensor(s) in the {label} artifact: {shown} "
                "— these two files were not produced by the same capture"
            )

    print(f"\n[numerics] {len(expected)} expected tensors, SlowQ vs Transformers")
    comparisons: list[Comparison] = []
    for key in sorted(expected):
        sample, kind = expected[key]
        if key not in ours_keys or key not in theirs_keys:
            continue  # already reported above, per side, with the sample named
        try:
            comparison = compare_arrays(ours[key], theirs[key], key, sample=sample, kind=kind)
        except ComparisonError as exc:
            problems.append(f"sample {sample}: {exc}")
            continue
        comparisons.append(comparison)
        for metric in ("rel_l2", "max_abs"):
            limit = TOLERANCES[f"{metric}_{kind}"]
            value = getattr(comparison, metric)
            if not value <= limit:  # also catches NaN, which fails every '>' test
                problems.append(
                    f"sample {sample}, tensor {key}: {metric} {value:.3e} exceeds the "
                    f"declared {limit:.3e}"
                )
        if ARGMAX_MUST_MATCH and comparison.argmax_match is False:
            problems.append(
                f"sample {sample}, tensor {key}: greedy argmax disagrees with the reference"
            )
    return comparisons


def print_comparison_table(comparisons: Sequence[Comparison]) -> None:
    print(f"\n{'tensor':<58} {'max abs':>11} {'mean abs':>11} {'rel L2':>11}  argmax")
    for comparison in sorted(comparisons, key=lambda item: -item.rel_l2):
        argmax = (
            "-"
            if comparison.argmax_match is None
            else ("ok" if comparison.argmax_match else "FAIL")
        )
        print(
            f"{comparison.key:<58} {comparison.max_abs:>11.3e} "
            f"{comparison.mean_abs:>11.3e} {comparison.rel_l2:>11.3e}  {argmax}"
        )


def check_internal(
    ours: np.lib.npyio.NpzFile, capture: Mapping[str, Any], problems: list[str]
) -> None:
    """SlowQ against itself: one full prefill vs the chunk loop."""
    if str(capture["task"]) == TASK_TEXT_EMBEDDING:
        _check_internal_embedding(ours, capture, problems)
        return
    print("\n[internal] SlowQ full prefill vs chunked prefill")
    for name in capture["sample_names"]:
        full, chunked = f"{name}.full_prefill.logits", f"{name}.chunked_prefill.logits"
        if full not in ours.files or chunked not in ours.files:
            problems.append(f"sample {name}: cannot compare full vs chunked prefill, a side is missing")
            continue
        try:
            comparison = compare_arrays(
                ours[full], ours[chunked], f"{name} full-vs-chunked", kind="logits"
            )
        except ComparisonError as exc:
            problems.append(f"sample {name}: {exc}")
            continue
        print(
            f"  {name:<24} max_abs={comparison.max_abs:.3e} rel_l2={comparison.rel_l2:.3e} "
            f"argmax={'ok' if comparison.argmax_match else 'FAIL'}"
        )
        # Chunking changes reduction order, so bit equality is not guaranteed;
        # the declared logit tolerance still applies.
        if comparison.rel_l2 > TOLERANCES["rel_l2_logits"]:
            problems.append(
                f"sample {name}: chunked prefill drifted from the full prefill "
                f"(rel_l2 {comparison.rel_l2:.3e})"
            )
        if comparison.argmax_match is False:
            problems.append(f"sample {name}: chunking changed the greedy next token")

    if str(capture["task"]) == TASK_IMAGE_TEXT_GENERATION:
        _check_internal_vision_merge(ours, capture, problems)


def _check_internal_vision_merge(
    ours: np.lib.npyio.NpzFile, capture: Mapping[str, Any], problems: list[str]
) -> None:
    """SlowQ against itself: did the connector's rows land where the prompt says?

    ``check_numerics`` compares the fused matrix against the reference's, and
    that is the real gate. This check keeps its teeth in the case that gate
    cannot cover — both sides wrong the same way — because the merge is an
    *assignment*, not a computation: the row at the first ``<image>`` position
    must equal the connector's first feature row bit for bit, and the row at the
    last position its last. A SlowQ that ordered the frames differently, or
    offset the merge by one, fails here even if it agreed with a reference that
    made the same mistake.

    The two ends are checked rather than every position because the positions in
    between are not contiguous — the ``<fake_token_around_image>``,
    ``<row_r_col_c>`` and newline tokens sit between the frames — and the ends are
    where a frame-order or off-by-one error lands.
    """
    print("\n[internal] SlowQ image-feature placement in the fused sequence")
    layout = capture.get("vision") or {}
    for name in capture["sample_names"]:
        sequence_key = f"{name}.full_prefill.embed_sequence"
        connector_key = f"{name}.vision.connector_output"
        sample_layout = layout.get(name)
        if sequence_key not in ours.files or connector_key not in ours.files:
            problems.append(f"sample {name}: cannot check the image merge, a tensor is missing")
            print(f"  {name:<24} MISSING tensor")
            continue
        if not isinstance(sample_layout, Mapping):
            problems.append(f"sample {name}: the capture records no vision layout to check the merge against")
            continue
        sequence = ours[sequence_key]
        features = ours[connector_key]
        for field, feature_index, label in (
            ("first_image_position", 0, "first frame"),
            ("last_image_position", features.shape[0] - 1, "last frame"),
        ):
            if field not in sample_layout:
                problems.append(f"sample {name}: the capture records no {field}")
                continue
            position = int(sample_layout[field])
            if not 0 <= position < sequence.shape[0]:
                problems.append(
                    f"sample {name}: {field}={position} is outside the fused sequence "
                    f"of {sequence.shape[0]} rows"
                )
                continue
            row, target = sequence[position], features[feature_index]
            identical = bool(np.array_equal(row, target))
            max_abs = float(np.abs(row.astype(np.float64) - target.astype(np.float64)).max())
            print(
                f"  {name:<24} {label:<12} row {position} == feature row {feature_index} "
                f"max_abs={max_abs:.3e} [{'ok' if identical else 'DIFFERS'}]"
            )
            if not identical:
                problems.append(
                    f"sample {name}: the fused row at {field}={position} is not the "
                    f"connector's {label} output (max_abs {max_abs:.3e}). The merge is an "
                    "assignment, so any difference at all means the image features did "
                    "not land on the <image> positions in frame order."
                )


def _check_internal_embedding(
    ours: np.lib.npyio.NpzFile, capture: Mapping[str, Any], problems: list[str]
) -> None:
    """SlowQ against itself: chunked encode vs one full prefill, and unit norms.

    The norm assertion is separate from the reference comparison on purpose. It
    is the only check here that cannot be satisfied by both sides being wrong in
    the same way: a vector whose L2 norm is not 1 is not a cosine-similarity
    input, whatever the reference produced.
    """
    print("\n[internal] SlowQ chunked encode vs full prefill, and vector norms")
    for name in capture["sample_names"]:
        for text in capture["embedding_texts"][name]:
            full = f"{name}.{text}.full_prefill.vector"
            chunked = f"{name}.{text}.chunked_prefill.vector"
            if full not in ours.files or chunked not in ours.files:
                problems.append(f"{name}/{text}: a vector is missing from the SlowQ artifact")
                continue
            try:
                comparison = compare_arrays(
                    ours[full], ours[chunked], f"{name}/{text} full-vs-chunked", kind="vector"
                )
            except ComparisonError as exc:
                problems.append(f"{name}/{text}: {exc}")
                continue
            norm = float(np.linalg.norm(ours[chunked].astype(np.float64)))
            drift = abs(norm - 1.0)
            print(
                f"  {name}/{text:<6} max_abs={comparison.max_abs:.3e} "
                f"rel_l2={comparison.rel_l2:.3e} |v|={norm:.9f}"
            )
            if comparison.rel_l2 > TOLERANCES["rel_l2_vector"]:
                problems.append(
                    f"{name}/{text}: the chunked encode drifted from the full prefill "
                    f"(rel_l2 {comparison.rel_l2:.3e})"
                )
            if drift > 1e-6:
                problems.append(
                    f"{name}/{text}: the embedding is not unit-norm (|v| = {norm!r}); "
                    "dot products would not be cosines"
                )


#: The per-text fields that say *what was embedded*, as opposed to what came out
#: of it. Graded separately from the counts so that a failure can say which of
#: the two it is: the same text tokenised to a different length is a tokenizer
#: problem, and a different text is a prefix problem.
EMBEDDING_INPUT_FIELDS = ("raw", "is_query", "text")

#: What each of those fields must be, on both sides. ``is_query`` is spelled as
#: ``bool`` and checked with ``isinstance`` rather than by equality against
#: ``True``/``False``, because ``bool`` is a subclass of ``int`` in Python and a
#: JSON ``1`` would otherwise read as a truthy answer to "was this a query".
EMBEDDING_INPUT_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "raw": (str,),
    "is_query": (bool,),
    "text": (str,),
}


def _expected_type_words(field: str) -> str:
    """The type requirement in the words a reader can act on."""
    if field == "is_query":
        return "a JSON boolean (true or false, not 1 or 0)"
    return "a string"


def _embedding_input_field_issues(
    ours: Mapping[str, Any], theirs: Mapping[str, Any]
) -> list[tuple[str, str, str]]:
    """``(side, field, problem)`` for every input field that cannot be compared.

    Two records are only comparable if both actually carry the field, and carry
    it as the type the capture format specifies. Comparing them with
    ``ours.get(f) == theirs.get(f)`` cannot establish that: ``dict.get`` answers
    ``None`` for a key that was never there, so a capture where *both* sides
    dropped ``raw``/``is_query``/``text`` compared ``None`` against ``None``,
    agreed, and reported the input format as checked when nothing had been read.

    Absence and null are reported separately because they mean different things
    about how the artifact came to be: a capture from before the field existed
    has no key, and a serializer that wrote an unset value has one holding null.
    """
    issues: list[tuple[str, str, str]] = []
    for side, entry in (("SlowQ", ours), ("reference", theirs)):
        for field in EMBEDDING_INPUT_FIELDS:
            if field not in entry:
                issues.append(
                    (
                        side,
                        field,
                        f"the {side} record has no {field!r} field, so what it embedded "
                        f"cannot be compared; expected {_expected_type_words(field)}",
                    )
                )
                continue
            value = entry[field]
            if value is None:
                issues.append(
                    (
                        side,
                        field,
                        f"the {side} record's {field!r} is null, so what it embedded "
                        f"cannot be compared; expected {_expected_type_words(field)}",
                    )
                )
                continue
            if not isinstance(value, EMBEDDING_INPUT_FIELD_TYPES[field]):
                issues.append(
                    (
                        side,
                        field,
                        f"the {side} record's {field!r} is {value!r}, of type "
                        f"{type(value).__name__}, where the capture format requires "
                        f"{_expected_type_words(field)}",
                    )
                )
    return issues


def _report_embedding_input(
    name: str,
    text: str,
    ours: Mapping[str, Any],
    theirs: Mapping[str, Any],
    problems: list[str],
) -> None:
    """Name an input-format disagreement — the kind that used to be invisible."""
    where = f"{name}/{text}"
    if theirs.get("text") is None:
        problems.append(
            f"{where}: the reference records no input text of its own, so these "
            f"artifacts predate embedding input format {EMBEDDING_INPUT_FORMAT}. "
            "Before that the reference tokenised the text SlowQ had already "
            "prefixed, and so could not detect a wrong prefix; re-capture it."
        )
        return
    if ours.get("raw") != theirs.get("raw") or ours.get("is_query") != theirs.get(
        "is_query"
    ):
        problems.append(
            f"{where}: the two sides did not embed the same input. SlowQ records "
            f"raw={ours.get('raw')!r} is_query={ours.get('is_query')!r}; the "
            f"reference records raw={theirs.get('raw')!r} "
            f"is_query={theirs.get('is_query')!r}"
        )
        return
    problems.append(
        f"{where}: SlowQ embedded {ours.get('text')!r} where the official format is "
        f"{theirs.get('text')!r} — first difference at character "
        f"{_first_difference(str(ours.get('text')), str(theirs.get('text')))}. A query "
        "takes 'Instruct: {instruction}\\nQuery:{query}', with no space after the "
        "colon; a document takes no prefix at all."
    )


def check_embedding_metadata(
    ours: Mapping[str, Any], theirs: Mapping[str, Any], problems: list[str]
) -> None:
    """Input format, token counts, pooling indices and vector dimensions.

    This is where "took the last valid token" is actually verified rather than
    asserted: the reference derives its index from its own attention mask, so a
    SlowQ that pooled one row early — or that silently truncated the input — has
    a different ``pool_index`` and a different ``num_tokens``.

    It is also the only place the *input format* is graded, and it can only do
    that because the reference builds its own text: comparing SlowQ's prefixed
    string against a copy of itself would agree however the prefix was spelled.

    Before either side is compared, both are required to actually carry the
    fields — see :func:`_embedding_input_field_issues`. A record missing one is
    reported as unusable rather than as agreeing with the other side's equally
    missing one.
    """
    print("\n[embedding] input format, pooling position and vector dimension")
    ours_samples = {sample["name"]: sample for sample in ours.get("samples", [])}
    theirs_samples = {sample["name"]: sample for sample in theirs.get("samples", [])}
    if not theirs_samples:
        problems.append("the reference records no embedding samples; re-capture it")
        return
    for name in sorted(set(ours_samples) | set(theirs_samples)):
        ours_sample = ours_samples.get(name)
        theirs_sample = theirs_samples.get(name)
        if ours_sample is None or theirs_sample is None:
            side = "SlowQ" if ours_sample is None else "reference"
            problems.append(f"embedding sample {name!r} is missing from the {side} side")
            print(f"  {name:<32} MISSING from {side}")
            continue
        ours_texts = {entry["name"]: entry for entry in ours_sample.get("texts", [])}
        theirs_texts = {entry["name"]: entry for entry in theirs_sample.get("texts", [])}
        for text in sorted(set(ours_texts) | set(theirs_texts)):
            ours_entry = ours_texts.get(text)
            theirs_entry = theirs_texts.get(text)
            if ours_entry is None or theirs_entry is None:
                side = "SlowQ" if ours_entry is None else "reference"
                problems.append(f"{name}/{text}: missing from the {side} side")
                print(f"  {name}/{text:<22} MISSING from {side}")
                continue
            agrees = all(
                ours_entry.get(field) == theirs_entry.get(field)
                for field in ("num_tokens", "pool_index", "dimension")
            )
            issues = _embedding_input_field_issues(ours_entry, theirs_entry)
            for _side, _field, problem in issues:
                problems.append(f"{name}/{text}: {problem}")
            inputs_agree = not issues and all(
                ours_entry[field] == theirs_entry[field] for field in EMBEDDING_INPUT_FIELDS
            )
            print(
                f"  {name}/{text:<22} tokens={theirs_entry.get('num_tokens')} "
                f"pool_index={theirs_entry.get('pool_index')} "
                f"dim={theirs_entry.get('dimension')} "
                f"input={'ok' if inputs_agree else ('INVALID' if issues else 'MISMATCH')} "
                f"pooled={'ok' if agrees else 'MISMATCH'}"
            )
            if not inputs_agree:
                # A disagreement is only meaningful between two records that each
                # state their input; otherwise the message below would print
                # ``None`` against ``None`` and call it a prefix problem. The one
                # issue that is still worth explaining is a reference that recorded
                # no text of its own, which is a capture from before the input
                # format was versioned and has to be re-taken.
                if not issues or [(side, field) for side, field, _ in issues] == [
                    ("reference", "text")
                ]:
                    _report_embedding_input(name, text, ours_entry, theirs_entry, problems)
            if not agrees:
                problems.append(
                    f"{name}/{text}: SlowQ records tokens/pool_index/dimension "
                    f"{ours_entry.get('num_tokens')}/{ours_entry.get('pool_index')}/"
                    f"{ours_entry.get('dimension')} but the reference records "
                    f"{theirs_entry.get('num_tokens')}/{theirs_entry.get('pool_index')}/"
                    f"{theirs_entry.get('dimension')}"
                )


#: Every field of a sample's ``image`` record that both sides compute. Compared
#: field by field rather than as one dict so a failure names the step: the sizes
#: come from the official resizer, the grid from ``split_image``, and the digests
#: from the finished pixel tensors — a mismatch in ``encoder_size`` and one in
#: ``pixel_values_digest`` point at different code.
VISION_IMAGE_FIELDS = (
    "file",
    "sha256",
    "width",
    "height",
    "original_size",
    "resized_size",
    "encoder_size",
    "rows",
    "cols",
    "num_frames",
    "num_image_tokens",
    "pixel_values_digest",
    "pixel_attention_mask_digest",
    "pixel_attention_mask_all_valid",
)

#: Fields of the sample record itself that the vision protocol grades.
VISION_SAMPLE_FIELDS = (
    "prompt_tokens",
    "first_image_position",
    "last_image_position",
)


def check_vision_metadata(
    ours: Mapping[str, Any], theirs: Mapping[str, Any], problems: list[str]
) -> None:
    """Preprocessing, crop grid and image-token placement, both sides.

    This is where "SlowQ read the image the way the official processor does" is
    decided, and it is decided on data rather than on a plausible-looking answer:
    the same PNG file is preprocessed by both sides, then the resizer's
    intermediate sizes, the crop grid, the frame count, the pixel and mask
    digests and the positions of every ``<image>`` token are required to agree.

    A pixel-digest agreement is exact, not within a tolerance — ``tensor_digest``
    hashes the bytes, so this row passing means the two preprocessors produced
    bit-identical float32 tensors over the whole frame stack.

    The expanded prompt text is also split here into its two halves. ``compare``'s
    tokenizer check already compares the whole string, but on a prompt that is
    mostly ``<image>`` a single character offset is uninformative; knowing
    whether the *template* or the *expansion* differed says which of the two
    implementations to go and read.
    """
    print("\n[vision] preprocessing, crop grid and image-token placement")
    ours_samples = {sample["name"]: sample for sample in ours.get("samples", [])}
    theirs_samples = {sample["name"]: sample for sample in theirs.get("samples", [])}
    ours_texts = {entry["name"]: entry for entry in ours.get("tokenizer", [])}
    theirs_texts = {entry["name"]: entry for entry in theirs.get("tokenizer", [])}
    if not theirs_samples:
        problems.append("the reference records no image+text samples; re-capture it")
        return

    for name in sorted(set(ours_samples) | set(theirs_samples)):
        ours_sample = ours_samples.get(name)
        theirs_sample = theirs_samples.get(name)
        if ours_sample is None or theirs_sample is None:
            side = "SlowQ" if ours_sample is None else "reference"
            problems.append(f"vision sample {name!r} is missing from the {side} side")
            print(f"  {name:<24} MISSING from {side}")
            continue

        ours_image = ours_sample.get("image") or {}
        theirs_image = theirs_sample.get("image") or {}
        differing = [
            field
            for field in VISION_IMAGE_FIELDS
            if ours_image.get(field) != theirs_image.get(field)
        ]
        print(
            f"  {name:<24} grid={theirs_image.get('rows')}x{theirs_image.get('cols')} "
            f"frames={theirs_image.get('num_frames')} "
            f"image_tokens={theirs_image.get('num_image_tokens')} "
            f"encoder={theirs_image.get('encoder_size')} "
            f"pixels={'ok' if 'pixel_values_digest' not in differing else 'DIFFERS'} "
            f"mask={'ok' if 'pixel_attention_mask_digest' not in differing else 'DIFFERS'} "
            f"[{'ok' if not differing else 'MISMATCH: ' + ', '.join(differing)}]"
        )
        for field in differing:
            ours_value, theirs_value = ours_image.get(field), theirs_image.get(field)
            if field.endswith("_digest") and isinstance(ours_value, Mapping) and isinstance(
                theirs_value, Mapping
            ):
                problems.append(
                    f"vision sample {name}, {field}: shape {ours_value.get('shape')} vs "
                    f"reference {theirs_value.get('shape')}, sha256 "
                    f"{ours_value.get('sha256')} vs {theirs_value.get('sha256')} — the two "
                    "preprocessors did not produce identical tensors"
                )
            else:
                problems.append(
                    f"vision sample {name}, {field}: SlowQ records {ours_value!r} but the "
                    f"reference records {theirs_value!r}"
                )

        # The official forward drops frames that are entirely zero before the
        # tower sees them. SlowQ never drops one, so a non-zero count here means
        # the reference ran a different number of frames than SlowQ did and the
        # per-frame tensors below are not comparing like with like.
        dropped = int(theirs_image.get("dropped_padding_frames") or 0)
        if dropped:
            problems.append(
                f"vision sample {name}: the official forward dropped {dropped} all-zero "
                "frame(s) that SlowQ kept, so the frame indices are misaligned"
            )
        if not theirs_image.get("pixel_attention_mask_all_valid", False):
            problems.append(
                f"vision sample {name}: the official pixel attention mask is not all "
                "valid, which SlowQ's single-image path does not produce — the two "
                "sides padded differently"
            )

        for field in VISION_SAMPLE_FIELDS:
            if ours_sample.get(field) != theirs_sample.get(field):
                problems.append(
                    f"vision sample {name}, {field}: SlowQ records "
                    f"{ours_sample.get(field)!r} but the reference records "
                    f"{theirs_sample.get(field)!r}"
                )
        print(
            f"  {'':<24} prompt_tokens={theirs_sample.get('prompt_tokens')} "
            f"image positions {theirs_sample.get('first_image_position')}…"
            f"{theirs_sample.get('last_image_position')} "
            f"[{'ok' if all(ours_sample.get(f) == theirs_sample.get(f) for f in VISION_SAMPLE_FIELDS) else 'MISMATCH'}]"
        )

        ours_text = (ours_texts.get(name) or {}).get("text")
        template = theirs_sample.get("template_text")
        if ours_text is None or template is None:
            continue
        # ``check_tokenizer`` already compares the two expanded strings. On a
        # prompt that is mostly ``<image>``, a bare character offset does not say
        # which implementation to go and read, so the official template is split
        # at its single ``<image>`` and SlowQ's text is asked whether it has the
        # same head and the same tail: if it does, the template matched and the
        # difference is inside the expansion.
        head, _, tail = template.partition("<image>")
        template_ok = ours_text.startswith(head) and ours_text.endswith(tail)
        print(
            f"  {'':<24} chat template {'ok' if template_ok else 'DIFFERS'}, "
            f"image expansion {'ok' if template_ok and ours_text == (theirs_texts.get(name) or {}).get('text') else 'DIFFERS'}"
        )
        if not template_ok:
            problems.append(
                f"vision sample {name}: SlowQ's expanded prompt does not both start with "
                "the official chat template's head and end with its tail, so the "
                "difference is in the template rendering rather than in the image "
                f"expansion (head {head!r}, tail {tail!r})"
            )


def check_argmax(
    payload: Mapping[str, Any], theirs: np.lib.npyio.NpzFile, problems: list[str]
) -> None:
    """The strongest practical check: did both sides pick the same token?"""
    print("\n[argmax] greedy next-token choice, SlowQ vs reference")
    for sample in payload["samples"]:
        name = sample["name"]
        rows = [
            ("full_prefill", sample["full_prefill_argmax"], f"{name}.full_prefill.logits"),
            ("chunked_prefill", sample["chunked_prefill_argmax"], f"{name}.chunked_prefill.logits"),
        ]
        rows += [
            (f"decode{step}", record["argmax"], f"{name}.decode{step}.logits")
            for step, record in enumerate(sample["decode_steps"])
        ]
        for label, ours_choice, key in rows:
            if key not in theirs.files:
                problems.append(
                    f"sample {name}, {label}: the reference has no logits at {key}"
                )
                print(f"  {name:<24} {label:<16} MISSING from the reference")
                continue
            reference_choice = int(np.argmax(theirs[key]))
            match = int(ours_choice) == reference_choice
            print(
                f"  {name:<24} {label:<16} slowq={ours_choice:<7} ref={reference_choice:<7} "
                f"{'ok' if match else 'MISMATCH'}"
            )
            if not match:
                problems.append(
                    f"sample {name}, {label}: greedy choice {ours_choice} != reference "
                    f"{reference_choice}"
                )


def run_compare(args: argparse.Namespace) -> int:
    artifacts = Path(args.artifacts)
    problems: list[str] = []

    ours_meta = load_metadata(
        artifacts / "samples.json", problems, "run 'validate_reference.py numpy' first"
    )
    ref_meta = load_metadata(
        artifacts / REFERENCE_FILENAME,
        problems,
        "run 'validate_reference.py ref' (or re-capture it) first",
    )
    ours = load_blobs(
        artifacts / "slowq.npz", problems, "run 'validate_reference.py numpy' first"
    )
    theirs = load_blobs(
        artifacts / "ref.npz", problems, "run 'validate_reference.py ref' first"
    )
    if problems:
        return report(problems, [])

    assert ours_meta is not None and ref_meta is not None
    assert ours is not None and theirs is not None

    capture = check_capture(ours_meta, ref_meta, problems)
    check_provenance(ours_meta, ref_meta, problems)
    check_tokenizer(ours_meta, ref_meta, problems, capture)
    check_weights(ours_meta, ref_meta, problems)

    comparisons: list[Comparison] = []
    if capture is None:
        problems.append(
            "the numeric comparison was skipped: the two artifacts do not describe the "
            "same capture, so there is no agreed key set to grade"
        )
    else:
        comparisons = check_numerics(ours, theirs, capture, problems)
        print_comparison_table(comparisons)
        check_internal(ours, capture, problems)
        task = str(capture["task"])
        if task == TASK_TEXT_EMBEDDING:
            check_embedding_metadata(ours_meta, ref_meta, problems)
        elif task == TASK_IMAGE_TEXT_GENERATION:
            check_vision_metadata(ours_meta, ref_meta, problems)
            check_argmax(ours_meta, theirs, problems)
        else:
            check_argmax(ours_meta, theirs, problems)
        if not comparisons:
            problems.append(
                "no tensors were compared at all — an empty run is a failure, not a pass"
            )

    declared = ", ".join(f"{key}={value:g}" for key, value in TOLERANCES.items())
    print(f"\n[declared tolerances] {declared}")
    print(f"[reference] {ref_meta.get('environment')}")
    return report(problems, comparisons)


def report(problems: Sequence[str], comparisons: Sequence[Comparison]) -> int:
    if problems:
        print(f"\nFAIL ({len(problems)} problem(s)):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"\nPASS: {len(comparisons)} tensors compared within the declared tolerances."
    )
    return 0


# ---- entry point ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "side",
        choices=("numpy", "ref", "compare", "all"),
        help="which half to run; 'all' runs numpy, ref and compare in one process",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3-0.6B-slowq"),
        help="converted SlowQ model directory (numpy side); its recorded task "
        "decides which capture protocol runs",
    )
    parser.add_argument(
        "--hf-dir",
        type=Path,
        default=Path("models/Qwen3-0.6B-hf"),
        help="official checkpoint directory (ref side)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="capture directory; defaults to artifacts/validation/<model-name>, so "
        "validating a second model cannot silently overwrite the first one's reference",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=512,
        help="context budget for generation, and max_length for an embedding capture",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.artifacts is None:
        args.artifacts = DEFAULT_ARTIFACTS / Path(args.model).name
    started = time.perf_counter()
    if args.side == "numpy":
        code = run_numpy_side(args)
    elif args.side == "ref":
        code = run_reference_side(args)
    elif args.side == "compare":
        code = run_compare(args)
    else:
        code = run_numpy_side(args) or run_reference_side(args) or run_compare(args)
    print(f"\n[{args.side}] took {time.perf_counter() - started:.1f} s", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
