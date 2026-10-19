"""Loading of converted weights: manifest parsing, read-only mmap, shape checks.

On-disk layout produced by ``tools/convert_model.py``::

    <model_dir>/
      manifest.json
      config.json
      generation_config.json
      tokenizer.json  tokenizer_config.json  vocab.json  merges.txt
      tensors/<tensor-name>.npy

Every tensor is one ``.npy`` file so it can be inspected with ``np.load`` and a
text editor, and loaded with a read-only memory map: the OS pages weights in on
demand and shares them between processes, so we never copy the model at startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .config import ModelConfig
from .errors import WeightError

MANIFEST_FILENAME = "manifest.json"
TENSOR_SUBDIR = "tensors"
SUPPORTED_FORMAT_VERSION = 1
COMPUTE_DTYPE = np.dtype(np.float32)

#: ``lm_head.weight`` is tied to the embedding matrix for Qwen3-0.6B. The
#: converter stores it once; this maps the alias onto the stored tensor.
TIED_OUTPUT_EMBEDDING_ALIAS = "lm_head.weight"
EMBEDDING_TENSOR_NAME = "model.embed_tokens.weight"


def required_tensors(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Canonical logical tensor names -> expected shapes for a Qwen3 model."""
    c = config
    expected: dict[str, tuple[int, ...]] = {
        EMBEDDING_TENSOR_NAME: (c.vocab_size, c.hidden_size),
        "model.norm.weight": (c.hidden_size,),
    }
    if not c.tie_word_embeddings:
        expected[TIED_OUTPUT_EMBEDDING_ALIAS] = (c.vocab_size, c.hidden_size)

    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}"
        expected[f"{p}.self_attn.q_proj.weight"] = (c.q_proj_out, c.hidden_size)
        expected[f"{p}.self_attn.k_proj.weight"] = (c.kv_proj_out, c.hidden_size)
        expected[f"{p}.self_attn.v_proj.weight"] = (c.kv_proj_out, c.hidden_size)
        expected[f"{p}.self_attn.o_proj.weight"] = (c.hidden_size, c.q_proj_out)
        expected[f"{p}.self_attn.q_norm.weight"] = (c.head_dim,)
        expected[f"{p}.self_attn.k_norm.weight"] = (c.head_dim,)
        expected[f"{p}.mlp.gate_proj.weight"] = (c.intermediate_size, c.hidden_size)
        expected[f"{p}.mlp.up_proj.weight"] = (c.intermediate_size, c.hidden_size)
        expected[f"{p}.mlp.down_proj.weight"] = (c.hidden_size, c.intermediate_size)
        expected[f"{p}.input_layernorm.weight"] = (c.hidden_size,)
        expected[f"{p}.post_attention_layernorm.weight"] = (c.hidden_size,)
    return expected


@dataclass(frozen=True)
class TensorSpec:
    name: str
    file: str
    shape: tuple[int, ...]
    dtype: str
    source_dtype: str | None = None


@dataclass(frozen=True)
class Manifest:
    format_version: int
    tensors: dict[str, TensorSpec]
    aliases: dict[str, str]
    source: Mapping[str, Any]
    compute_dtype: str

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
                f"tensor {canonical!r} is stored as {spec.dtype!r}; SlowQ v0.1 "
                "only runs FP32. Re-run tools/convert_model.py."
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

    def check_complete(self, config: ModelConfig) -> None:
        """Verify every tensor the model needs exists with the right shape.

        Called at load time so a broken conversion fails before the first
        forward instead of 20 layers into one.
        """
        expected = required_tensors(config)
        problems: list[str] = []
        for name, shape in expected.items():
            if not self.has(name):
                problems.append(f"missing {name} (expected shape {shape})")
                continue
            spec = self.spec(name)
            if tuple(spec.shape) != shape:
                problems.append(
                    f"{name}: manifest shape {tuple(spec.shape)}, expected {shape}"
                )
        if problems:
            raise WeightError(
                "converted model is incomplete or inconsistent:\n  "
                + "\n  ".join(problems)
            )

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
    if version != SUPPORTED_FORMAT_VERSION:
        raise WeightError(
            f"unsupported manifest format_version {version!r}; this build of SlowQ "
            f"reads version {SUPPORTED_FORMAT_VERSION}"
        )
    compute_dtype = raw.get("compute_dtype")
    if compute_dtype != "float32":
        raise WeightError(
            f"manifest compute_dtype={compute_dtype!r}; SlowQ v0.1 only runs FP32"
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
        tensors[name] = TensorSpec(
            name=name,
            file=str(entry["file"]),
            shape=shape,
            dtype=str(entry["dtype"]),
            source_dtype=entry.get("source_dtype"),
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

    return Manifest(
        format_version=int(version),
        tensors=tensors,
        aliases=aliases,
        source=source,
        compute_dtype=str(compute_dtype),
    )


def load_manifest(model_dir: str | Path) -> Manifest:
    return Weights.from_directory(model_dir).manifest
