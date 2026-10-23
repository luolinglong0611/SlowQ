"""Loading of converted weights: manifest parsing, read-only mmap, shape checks.

On-disk layout produced by ``tools/convert_model.py``::

    <model_dir>/
      manifest.json
      config.json
      generation_config.json          (generative tasks only)
      tokenizer.json  tokenizer_config.json  ...
      tensors/<tensor-name>.npy

Every tensor is one ``.npy`` file so it can be inspected with ``np.load`` and a
text editor, and loaded with a read-only memory map: the OS pages weights in on
demand and shares them between processes, so we never copy the model at startup.

Which tensors *must* be there depends on the family and the task, so the rules
live in :func:`required_tensors` rather than in one hard-coded Qwen3 list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .config import (
    FAMILY_LLAMA,
    FAMILY_QWEN2,
    FAMILY_QWEN3,
    TASK_IMAGE_TEXT_GENERATION,
    TASK_TEXT_EMBEDDING,
    TASK_TEXT_GENERATION,
    ModelConfig,
)
from .errors import WeightError

MANIFEST_FILENAME = "manifest.json"
TENSOR_SUBDIR = "tensors"

#: Version 1 is what SlowQ v0.1 wrote: a Qwen3 causal LM, with no ``task`` or
#: ``model_family`` recorded because only one of each existed. Version 2 adds
#: those keys and is required for every other family and task. Both are read;
#: only version 2 is written.
SUPPORTED_FORMAT_VERSION = 2
ACCEPTED_FORMAT_VERSIONS = (1, 2)
COMPUTE_DTYPE = np.dtype(np.float32)

#: Default prefix of a text decoder's tensors. SmolVLM nests its decoder under
#: ``model.text_model`` instead, which its ``ModelConfig.tensor_prefix`` states.
TEXT_TENSOR_PREFIX = "model"
#: ``lm_head`` is a sibling of ``model`` in HF's causal-LM wrappers, so this name
#: carries no prefix in any family. It is also the alias the converter records
#: when ``lm_head.weight`` is tied to the embedding matrix and stored once.
TIED_OUTPUT_EMBEDDING_ALIAS = "lm_head.weight"
LM_HEAD_TENSOR_NAME = TIED_OUTPUT_EMBEDDING_ALIAS
EMBEDDING_TENSOR_NAME = f"{TEXT_TENSOR_PREFIX}.embed_tokens.weight"


def embedding_tensor_name(config: ModelConfig) -> str:
    return f"{config.tensor_prefix}.embed_tokens.weight"


def output_embedding_name(config: ModelConfig) -> str:
    """Where the output projection's weight actually lives.

    A tied checkpoint stores it once, as the embedding matrix, and the manifest
    maps ``lm_head.weight`` onto that; an untied one stores a separate
    ``lm_head.weight``.
    """
    return embedding_tensor_name(config) if config.tie_word_embeddings else LM_HEAD_TENSOR_NAME


def required_text_tensors(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Canonical logical tensor names -> expected shapes for one text decoder.

    The set is driven by the family's real rules, not by a Qwen3-shaped guess:
    ``q_norm``/``k_norm`` exist only for qwen3, the bias vectors only where that
    family's ``nn.Linear`` declares one, and ``lm_head.weight`` only when the
    embeddings are untied. An embedding model needs no output projection at all,
    so requiring one there would be a false failure rather than a real check.
    """
    if config.family not in (FAMILY_QWEN3, FAMILY_QWEN2, FAMILY_LLAMA):
        raise WeightError(f"no weight rules for family {config.family!r}")

    c = config
    p = c.tensor_prefix
    embed = f"{p}.embed_tokens.weight"
    expected: dict[str, tuple[int, ...]] = {
        embed: (c.vocab_size, c.hidden_size),
        f"{p}.norm.weight": (c.hidden_size,),
    }
    if c.task != TASK_TEXT_EMBEDDING and not c.tie_word_embeddings:
        expected[LM_HEAD_TENSOR_NAME] = (c.vocab_size, c.hidden_size)

    for i in range(c.num_hidden_layers):
        layer = f"{p}.layers.{i}"
        attn = f"{layer}.self_attn"
        expected[f"{attn}.q_proj.weight"] = (c.q_proj_out, c.hidden_size)
        expected[f"{attn}.k_proj.weight"] = (c.kv_proj_out, c.hidden_size)
        expected[f"{attn}.v_proj.weight"] = (c.kv_proj_out, c.hidden_size)
        expected[f"{attn}.o_proj.weight"] = (c.hidden_size, c.q_proj_out)
        if c.qkv_bias:
            expected[f"{attn}.q_proj.bias"] = (c.q_proj_out,)
            expected[f"{attn}.k_proj.bias"] = (c.kv_proj_out,)
            expected[f"{attn}.v_proj.bias"] = (c.kv_proj_out,)
        if c.o_proj_bias:
            expected[f"{attn}.o_proj.bias"] = (c.hidden_size,)
        if c.qk_norm:
            expected[f"{attn}.q_norm.weight"] = (c.head_dim,)
            expected[f"{attn}.k_norm.weight"] = (c.head_dim,)
        mlp = f"{layer}.mlp"
        expected[f"{mlp}.gate_proj.weight"] = (c.intermediate_size, c.hidden_size)
        expected[f"{mlp}.up_proj.weight"] = (c.intermediate_size, c.hidden_size)
        expected[f"{mlp}.down_proj.weight"] = (c.hidden_size, c.intermediate_size)
        if c.mlp_bias:
            expected[f"{mlp}.gate_proj.bias"] = (c.intermediate_size,)
            expected[f"{mlp}.up_proj.bias"] = (c.intermediate_size,)
            expected[f"{mlp}.down_proj.bias"] = (c.hidden_size,)
        expected[f"{layer}.input_layernorm.weight"] = (c.hidden_size,)
        expected[f"{layer}.post_attention_layernorm.weight"] = (c.hidden_size,)
    return expected


def required_tensors(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor ``config``'s *text decoder* needs, by task.

    Text generation and text embedding share one decoder; the difference is that
    an embedding model never projects to the vocabulary, so it must not be asked
    for an ``lm_head``.

    An image+text model's ``ModelConfig`` describes only the nested decoder
    (``model.text_model.*`` plus the top-level ``lm_head.weight``), which is why
    the vision tower's tensors are not listed here: ``slowq.vlm.required_tensors``
    adds them, and ``SmolVLMModel`` checks both sets.
    """
    if config.task not in (
        TASK_TEXT_GENERATION,
        TASK_TEXT_EMBEDDING,
        TASK_IMAGE_TEXT_GENERATION,
    ):
        raise WeightError(f"no weight rules for task {config.task!r}")
    return required_text_tensors(config)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    file: str
    shape: tuple[int, ...]
    dtype: str
    source_dtype: str | None = None
    #: Name in the source checkpoint, when the converter renamed it (the
    #: Qwen3-Embedding export has no ``model.`` prefix). ``None`` means unchanged.
    source_name: str | None = None


@dataclass(frozen=True)
class Manifest:
    format_version: int
    tensors: dict[str, TensorSpec]
    aliases: dict[str, str]
    source: Mapping[str, Any]
    compute_dtype: str
    #: Recorded by the converter; ``None`` for a version-1 manifest, which can
    #: only have come from the Qwen3 text-generation converter.
    task: str | None = None
    model_family: str | None = None

    @property
    def source_revision(self) -> str | None:
        return self.source.get("revision")


class Weights:
    """Lazy, read-only view over a converted model directory."""

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest
        self._cache: dict[str, np.ndarray] = {}

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "Weights":
        root = Path(model_dir)
        if not root.is_dir():
            raise WeightError(f"model directory not found: {root}")
        manifest_path = root / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise WeightError(
                f"{manifest_path} not found. Convert the official checkpoint "
                "first: python tools/convert_model.py --help"
            )
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WeightError(f"{manifest_path} is not valid JSON: {exc}") from exc
        return cls(root, _parse_manifest(raw))

    # ---- access -------------------------------------------------------------

    def canonical_name(self, name: str) -> str:
        return self.manifest.aliases.get(name, name)

    def has(self, name: str) -> bool:
        return self.canonical_name(name) in self.manifest.tensors

    def spec(self, name: str) -> TensorSpec:
        canonical = self.canonical_name(name)
        try:
            return self.manifest.tensors[canonical]
        except KeyError:
            raise WeightError(
                f"tensor {name!r} is missing from {self.manifest_path}"
            ) from None

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    def get(self, name: str) -> np.ndarray:
        """Return the tensor as a read-only FP32 memory-mapped array."""
        canonical = self.canonical_name(name)
        cached = self._cache.get(canonical)
        if cached is not None:
            return cached
        spec = self.spec(canonical)
        path = self.root / TENSOR_SUBDIR / spec.file
        if not path.is_file():
            raise WeightError(f"weight file for {canonical!r} not found: {path}")
        if spec.dtype != "float32":
            raise WeightError(
                f"tensor {canonical!r} is stored as {spec.dtype!r}; SlowQ only runs "
                "FP32. Re-run tools/convert_model.py."
            )
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except ValueError as exc:
            raise WeightError(
                f"failed to memory-map {path}: {exc}"
            ) from exc
        if array.dtype != COMPUTE_DTYPE:
            raise WeightError(
                f"tensor {canonical!r} has dtype {array.dtype} on disk, expected float32"
            )
        if tuple(array.shape) != tuple(spec.shape):
            raise WeightError(
                f"tensor {canonical!r} has shape {tuple(array.shape)} on disk but "
                f"the manifest declares {tuple(spec.shape)}"
            )
        self._cache[canonical] = array
        return array

    def expect(self, name: str, shape: Sequence[int]) -> np.ndarray:
        """``get`` plus a shape assertion against the model config."""
        array = self.get(name)
        expected = tuple(int(s) for s in shape)
        if tuple(array.shape) != expected:
            raise WeightError(
                f"tensor {name!r} has shape {tuple(array.shape)}, expected {expected}"
            )
        return array

    def check_names(self, expected: Mapping[str, tuple[int, ...]]) -> None:
        """Verify every listed tensor exists with the declared shape.

        Called at load time so a broken conversion fails before the first
        forward instead of 20 layers into one.
        """
        problems: list[str] = []
        for name, shape in expected.items():
            if not self.has(name):
                problems.append(f"missing {name} (expected shape {shape})")
                continue
            spec = self.spec(name)
            if tuple(spec.shape) != tuple(shape):
                problems.append(
                    f"{name}: manifest shape {tuple(spec.shape)}, expected {tuple(shape)}"
                )
        if problems:
            raise WeightError(
                "converted model is incomplete or inconsistent:\n  "
                + "\n  ".join(problems)
            )

    def check_complete(self, config: ModelConfig) -> None:
        self.check_names(required_tensors(config))

    # ---- introspection ------------------------------------------------------

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifest.tensors))

    @property
    def aliases(self) -> Mapping[str, str]:
        return self.manifest.aliases

    @property
    def dtype_names(self) -> tuple[str, ...]:
        return tuple(sorted({spec.dtype for spec in self.manifest.tensors.values()}))

    def deduplicated_bytes(self) -> int:
        """Logical size of the stored tensors, counting shared weights once."""
        return sum(
            int(np.prod(spec.shape)) * COMPUTE_DTYPE.itemsize
            for spec in self.manifest.tensors.values()
        )

    def file_bytes(self) -> int:
        total = 0
        for spec in self.manifest.tensors.values():
            path = self.root / TENSOR_SUBDIR / spec.file
            if path.is_file():
                total += path.stat().st_size
        return total

    def iter_specs(self) -> Iterator[TensorSpec]:
        yield from (self.manifest.tensors[name] for name in self.tensor_names)

    def close(self) -> None:
        self._cache.clear()


def _parse_manifest(raw: Mapping[str, Any]) -> Manifest:
    version = raw.get("format_version")
    if version not in ACCEPTED_FORMAT_VERSIONS:
        raise WeightError(
            f"unsupported manifest format_version {version!r}; this build of SlowQ "
            f"reads versions {list(ACCEPTED_FORMAT_VERSIONS)}"
        )
    compute_dtype = raw.get("compute_dtype")
    if compute_dtype != "float32":
        raise WeightError(
            f"manifest compute_dtype={compute_dtype!r}; SlowQ only runs FP32"
        )

    entries = raw.get("tensors")
    if not isinstance(entries, list) or not entries:
        raise WeightError("manifest has no 'tensors' list")

    tensors: dict[str, TensorSpec] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise WeightError(f"manifest tensor entry is not an object: {entry!r}")
        for key in ("name", "file", "shape", "dtype"):
            if key not in entry:
                raise WeightError(f"manifest tensor entry lacks {key!r}: {entry!r}")
        name = entry["name"]
        shape = tuple(int(s) for s in entry["shape"])
        if name in tensors:
            raise WeightError(f"manifest lists tensor {name!r} twice")
        source_name = entry.get("source_name")
        tensors[name] = TensorSpec(
            name=name,
            file=str(entry["file"]),
            shape=shape,
            dtype=str(entry["dtype"]),
            source_dtype=entry.get("source_dtype"),
            source_name=None if source_name is None else str(source_name),
        )

    aliases_raw = raw.get("aliases") or {}
    if not isinstance(aliases_raw, Mapping):
        raise WeightError("manifest 'aliases' must be an object")
    aliases: dict[str, str] = {}
    for alias, target in aliases_raw.items():
        if target not in tensors:
            raise WeightError(
                f"manifest alias {alias!r} points at unknown tensor {target!r}"
            )
        aliases[str(alias)] = str(target)

    source = raw.get("source")
    if not isinstance(source, Mapping):
        raise WeightError("manifest must record a 'source' object (repo id + revision)")

    task = raw.get("task")
    model_family = raw.get("model_family")
    if version == 1:
        # v0.1 only ever converted a Qwen3 causal LM, so a version-1 manifest
        # with no task key is unambiguous. Anything else would have been written
        # by version 2 and must say so.
        if task is not None or model_family is not None:
            raise WeightError(
                "a format_version 1 manifest must not carry 'task' or 'model_family'; "
                "those keys mean version 2"
            )
    else:
        if task is None or model_family is None:
            raise WeightError(
                "a format_version 2 manifest must record both 'task' and "
                "'model_family', so the converted directory states what it is "
                "without re-deriving it from the checkpoint's architecture name"
            )

    return Manifest(
        format_version=int(version),
        tensors=tensors,
        aliases=aliases,
        source=source,
        compute_dtype=str(compute_dtype),
        task=None if task is None else str(task),
        model_family=None if model_family is None else str(model_family),
    )


def load_manifest(model_dir: str | Path) -> Manifest:
    return Weights.from_directory(model_dir).manifest
