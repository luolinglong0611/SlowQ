"""Regression tests for ``tools/validate_reference.py``, offline and synthetic.

The comparator used to pass runs it had not compared. Two independent bugs:

* it iterated the *intersection* of the two artifact key sets, so an empty
  ``ref.npz`` produced ``PASS: 0 tensors compared`` and a missing reference
  logits tensor was skipped rather than reported;
* it graded with ``value > tolerance``, which is ``False`` for NaN, so a single
  NaN element anywhere in a tensor passed the whole run.

On top of that it trusted verdict booleans (``tokenizer_check``,
``weight_check``) written into ``samples.json`` by an earlier reference run, so
re-running only the NumPy half "verified" the tokenizer and the weights by
copying a historical PASS. Both now travel as *content* — the reference's
rendered text and token IDs, and sha256 digests of whole tensors — and
``run_compare`` recomputes every agreement itself.

Nothing here needs torch, a model, or a network: the artifacts are built for a
toy capture configuration and then broken in one way at a time.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A toy capture: two layers means ``capture_layers`` returns ``(0, 1)``, which
#: keeps the artifact small while still exercising per-layer key naming and a
#: first/middle/last selection that has to deduplicate.
NUM_LAYERS = 2
CHUNK = 4
DECODE_STEPS = 1
VOCAB = 32
HIDDEN = 8
SHARD = "model.safetensors"
SHARD_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

#: A toy vision tower, self-consistent with the fingerprint below: an 8x8 image
#: in 2x2 patches is 16 patches, and a ``scale_factor`` of 2 folds 2x2 of them
#: into one token, so ``image_seq_len`` is 4 and the connector reads 6*4 = 24
#: features wide. Keeping the arithmetic tied to the recorded config means the
#: tests below exercise the same relations the real protocol does, at 1/64th the
#: size.
VISION_HIDDEN = 6
VISION_PATCHES = 16
VISION_IMAGE_SEQ_LEN = 4
VISION_SCALE_FACTOR = 2
#: Text tokens the toy prompt carries before and after the image span. The real
#: expansion is not contiguous — ``<fake_token_around_image>`` and the
#: ``<row_r_col_c>`` tags sit between the frames — and only the two ends are
#: graded, so a toy with random text rows either side of them is faithful to
#: what the check reads.
VISION_LEAD_TOKENS = 4
#: rows, cols and frames per sample, the way the official splitter reports them.
VISION_GRIDS = {"grid-1x4": (1, 4, 5), "grid-2x4": (2, 4, 9)}
VISION_TEMPLATE = "<|im_start|>user\n<image>Describe this.<|im_end|>\n<|im_start|>assistant\n"


def toy_architecture(**overrides):
    """A Qwen3-shaped fingerprint, the family the toy artifacts were built for."""
    arch = {
        "family": "qwen3",
        "task": "text-generation",
        "tensor_prefix": "model",
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 2,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "max_position_embeddings": 256,
        "tie_word_embeddings": True,
        "qkv_bias": False,
        "o_proj_bias": False,
        "mlp_bias": False,
        "qk_norm": True,
        "sliding_window": None,
    }
    arch.update(overrides)
    return arch


def toy_vision_architecture(**overrides):
    """An ``idefics3`` fingerprint: a llama decoder wrapped around a vision tower.

    SmolVLM's decoder is a llama with no biases and no q/k norm, sitting inside a
    wrapper whose own name says nothing about that — which is why the fingerprint
    carries ``decoder_family`` beside ``family``.
    """
    arch = toy_architecture(
        family="idefics3",
        task="image-text-generation",
        tensor_prefix="model.text_model",
        qk_norm=False,
        tie_word_embeddings=False,
    )
    arch.update(
        {
            "decoder_family": "llama",
            "vision_hidden_size": VISION_HIDDEN,
            "vision_intermediate_size": VISION_HIDDEN * 2,
            "vision_num_hidden_layers": 2,
            "vision_num_attention_heads": 3,
            "vision_head_dim": 2,
            "vision_image_size": 8,
            "vision_patch_size": 2,
            "vision_hidden_act": "gelu_pytorch_tanh",
            "vision_layer_norm_eps": 1e-6,
            "vision_num_channels": 3,
            "scale_factor": VISION_SCALE_FACTOR,
            "image_seq_len": VISION_IMAGE_SEQ_LEN,
            "image_token_id": 27,
            "connector_out_features": VISION_HIDDEN * VISION_SCALE_FACTOR**2,
        }
    )
    arch.update(overrides)
    return arch


def toy_vision_layout(name: str) -> dict[str, object]:
    """One sample's frame grid, image-token span and sampled fused positions."""
    frames = VISION_GRIDS[name][2]
    features = frames * VISION_IMAGE_SEQ_LEN
    first = VISION_LEAD_TOKENS
    last = first + features - 1
    return {
        "num_frames": frames,
        "num_image_tokens": features,
        "prompt_tokens": features + 2 * VISION_LEAD_TOKENS,
        "first_image_position": first,
        "last_image_position": last,
        "positions_full": [0, first, last, -1],
    }


def load_validation_tool():
    """Import ``tools/validate_reference.py`` without making ``tools`` a package."""
    path = REPO_ROOT / "tools" / "validate_reference.py"
    spec = importlib.util.spec_from_file_location("slowq_validate_reference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before executing: the module uses ``from __future__ import
    # annotations``, and dataclasses resolves those strings via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return load_validation_tool()


class ArtifactSet:
    """A self-consistent pair of artifacts, plus handles for breaking it."""

    def __init__(
        self,
        tool,
        root: Path,
        arch: dict | None = None,
        *,
        numpy_prefixes_documents: bool = False,
    ) -> None:
        self.tool = tool
        self.root = root
        #: Fault injection for the embedding protocol: make the NumPy side apply
        #: the query prefix to documents as well, before the reference side is
        #: built. Injecting it afterwards would only prove the comparator can see
        #: a hand-edited artifact, not that it can see a broken SlowQ.
        self.numpy_prefixes_documents = numpy_prefixes_documents
        root.mkdir(parents=True, exist_ok=True)
        self.arch = arch if arch is not None else toy_architecture()
        self.vision_layout = (
            {
                sample["name"]: toy_vision_layout(sample["name"])
                for sample in tool.VISION_SAMPLES
            }
            if self.arch["task"] == "image-text-generation"
            else {}
        )
        self.capture = tool.capture_config(
            self.arch,
            prefill_chunk_size=CHUNK,
            decode_steps=DECODE_STEPS,
            vision=self.vision_layout or None,
        )
        self.arrays = self._make_arrays()
        self.ours_meta = self._make_ours_meta()
        self.ref_meta = self._make_ref_meta()
        self.write_metadata()
        self.write_blobs()

    # ---- construction -------------------------------------------------------

    def _shape_for(self, key: str, kind: str) -> tuple[int, ...]:
        """The shape of one captured tensor, from the key the protocol named it by.

        Only the three vision keys are matrices: the tower's output is one frame's
        patches, the connector's is the whole image span in text space, and the
        fused sequence is the prompt the decoder reads. Everything else is a row.
        """
        if kind == "logits":
            return (VOCAB,)
        if ".encoder_output[" in key:
            return (VISION_PATCHES, VISION_HIDDEN)
        layout = self.capture.get("vision", {}).get(key.split(".", 1)[0], {})
        if key.endswith(".connector_output"):
            return (int(layout["num_image_tokens"]), HIDDEN)
        if key.endswith(".embed_sequence"):
            return (int(layout["prompt_tokens"]), HIDDEN)
        return (HIDDEN,)

    def _make_arrays(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(20261019)
        keys = self.tool.expected_numeric_keys(self.capture)
        arrays = {
            key: rng.normal(size=self._shape_for(key, kind)).astype(np.float32)
            for key, (_, kind) in keys.items()
        }
        if self.capture["task"] == "text-embedding":
            for name in self.capture["sample_names"]:
                text_names = self.capture["embedding_texts"][name]
                for text in text_names:
                    key = f"{name}.{text}.full_prefill.vector"
                    # A vector that is not unit-norm fails the internal check, so
                    # the happy path normalises the ones graded as such.
                    unit = (arrays[key] / np.linalg.norm(arrays[key])).astype(np.float32)
                    arrays[key] = unit
                    # Chunking one encode must land on the same unit vector; that
                    # agreement is what the internal check measures.
                    arrays[f"{name}.{text}.chunked_prefill.vector"] = unit.copy()
                stacked = np.stack(
                    [arrays[f"{name}.{text}.chunked_prefill.vector"] for text in text_names]
                )
                arrays[f"{name}.similarity"] = (stacked @ stacked.T).astype(np.float32)
            return arrays
        if self.capture["task"] == "image-text-generation":
            # The merge is an assignment, so the happy path has to satisfy it
            # exactly: the fused row at each end of the image span *is* the
            # connector's row there, bit for bit. The rows in between stay random,
            # standing in for the text tokens and the inter-frame tags.
            for name, layout in self.capture["vision"].items():
                features = arrays[f"{name}.vision.connector_output"]
                sequence = arrays[f"{name}.full_prefill.embed_sequence"]
                sequence[layout["first_image_position"]] = features[0]
                sequence[layout["last_image_position"]] = features[-1]
        # The chunked and decode logits share the full prefill's argmax, which is
        # what the recorded sample claims; making them equal keeps the internal
        # full-vs-chunked check meaningful instead of incidentally failing.
        for name in self.capture["sample_names"]:
            full = arrays[f"{name}.full_prefill.logits"]
            for stage in ("chunked_prefill", *(f"decode{step}" for step in range(DECODE_STEPS))):
                arrays[f"{name}.{stage}.logits"] = full.copy()
        return arrays

    def _samples(self) -> list[dict[str, object]]:
        if self.capture["task"] == "text-embedding":
            return self._embedding_samples()
        if self.capture["task"] == "image-text-generation":
            return self._vision_samples()
        samples = []
        for index, name in enumerate(self.capture["sample_names"]):
            choice = int(np.argmax(self.arrays[f"{name}.full_prefill.logits"]))
            samples.append(
                {
                    "name": name,
                    "messages": [{"role": "user", "content": name}],
                    "rendered_text": f"<|im_start|>user\n{name}<|im_end|>\n",
                    "prompt_ids": [5, 6, 7, 8, 9, 10 + index],
                    "prefill_chunk_size": CHUNK,
                    "decode_steps": [
                        {
                            "input_token": 100 + index,
                            "cache_length_before": 6 + step,
                            "argmax": choice,
                        }
                        for step in range(DECODE_STEPS)
                    ],
                    "full_prefill_argmax": choice,
                    "chunked_prefill_argmax": choice,
                }
            )
        return samples

    def _embedding_samples(self) -> list[dict[str, object]]:
        """Sample records carrying everything ``check_embedding_metadata`` grades.

        The raw inputs are the tool's real ``EMBEDDING_SAMPLES`` and the prefixed
        text is built by the real ``slowq.embedding.format_query`` — the function
        under test. ``_make_ref_meta`` builds the reference side's text from
        ``reference_embedding_texts`` instead, so the two agree here for the same
        reason a real capture agrees, and stop agreeing for the same reason a real
        one would.

        ``format_query`` is imported inside the method on purpose: a
        module-level ``from … import format_query`` would bind the name before a
        test could monkeypatch it, and the fault injection would silently replace
        nothing.
        """
        from slowq.embedding import format_query

        samples = []
        for definition in self.tool.EMBEDDING_SAMPLES:
            name = definition["name"]
            instruction = definition["instruction"]
            raw_texts = [("query", definition["query"], True)] + [
                (f"doc{index}", document, False)
                for index, document in enumerate(definition["documents"])
            ]
            assert [text_name for text_name, _, _ in raw_texts] == (
                self.capture["embedding_texts"][name]
            )
            texts = []
            for index, (text_name, raw, is_query) in enumerate(raw_texts):
                tokens = 6 + index
                texts.append(
                    {
                        "name": text_name,
                        "raw": raw,
                        "text": (
                            format_query(raw, instruction)
                            if is_query or self.numpy_prefixes_documents
                            else raw
                        ),
                        "is_query": is_query,
                        "num_tokens": tokens,
                        "pool_index": tokens - 1,
                        "ids": [5, 6, 7, 8, 9, 10 + index][:tokens],
                        "dimension": HIDDEN,
                    }
                )
            samples.append(
                {
                    "name": name,
                    "instruction": instruction,
                    "query": definition["query"],
                    "documents": list(definition["documents"]),
                    "texts": texts,
                    "prefill_chunk_size": CHUNK,
                }
            )
        return samples

    def _reference_texts(self) -> dict[str, dict[str, object]]:
        """The reference side's own input records, keyed ``<sample>/<text>``.

        Mirrors ``_reference_embedding_capture``: the text comes from the tool's
        sample definitions through ``reference_embedding_texts``, never from the
        NumPy side's record. Empty for the other two tasks, whose real reference
        renders with the official chat template or image processor and which this
        fixture therefore echoes, as it did before.
        """
        if self.capture["task"] != "text-embedding":
            return {}
        return {
            f"{name}/{text_name}": {"raw": raw, "text": text, "is_query": is_query}
            for name in self.capture["sample_names"]
            for text_name, raw, text, is_query in self.tool.reference_embedding_texts(name)
        }

    def _vision_samples(self) -> list[dict[str, object]]:
        """Sample records carrying everything ``check_vision_metadata`` grades.

        The image fields are the ones the protocol compares field by field, so a
        toy that omitted one would leave that row of the check untested rather
        than passing it. The pixel and mask entries are real ``tensor_digest``
        records over small arrays: they are graded as *digests*, so the failure
        message has to be able to print a shape and a sha256 for both sides.
        """
        samples = []
        for name in self.capture["sample_names"]:
            rows, cols, frames = VISION_GRIDS[name]
            layout = self.capture["vision"][name]
            features = int(layout["num_image_tokens"])
            tokens = int(layout["prompt_tokens"])
            choice = int(np.argmax(self.arrays[f"{name}.full_prefill.logits"]))
            expansion = (
                "<fake_token_around_image><global-img><fake_token_around_image>"
                + "<image>" * features
                + "<fake_token_around_image>"
            )
            samples.append(
                {
                    "name": name,
                    "question": f"Describe {name}.",
                    "system": None,
                    "image": {
                        "file": f"vision-{name}.png",
                        "sha256": "a1" * 32,
                        "width": 4 * cols,
                        "height": 4 * rows,
                        "original_size": [4 * rows, 4 * cols],
                        "resized_size": [8 * rows, 8 * cols],
                        "encoder_size": [8 * rows, 8 * cols],
                        "rows": rows,
                        "cols": cols,
                        "num_frames": frames,
                        "num_image_tokens": features,
                        "pixel_values_digest": self.tool.tensor_digest(
                            np.zeros((frames, 3, 8, 8), dtype=np.float32)
                        ),
                        "pixel_attention_mask_digest": self.tool.tensor_digest(
                            np.ones((frames, 8, 8), dtype=np.float32)
                        ),
                        "pixel_attention_mask_all_valid": True,
                        "dropped_padding_frames": 0,
                    },
                    "template_text": VISION_TEMPLATE,
                    "rendered_text": VISION_TEMPLATE.replace("<image>", expansion),
                    "prompt_ids": list(range(tokens)),
                    "prompt_tokens": tokens,
                    "first_image_position": layout["first_image_position"],
                    "last_image_position": layout["last_image_position"],
                    "prefill_chunk_size": CHUNK,
                    "decode_steps": [
                        {
                            "input_token": 100,
                            "cache_length_before": tokens + step,
                            "argmax": choice,
                        }
                        for step in range(DECODE_STEPS)
                    ],
                    "full_prefill_argmax": choice,
                    "chunked_prefill_argmax": choice,
                }
            )
        return samples

    def _weight_arrays(self) -> dict[str, np.ndarray]:
        """One array per weight the protocol expects, so 'complete' means complete.

        ``check_weights`` derives its expected set from the recorded
        architecture, not from what the two files happen to contain, so a
        self-consistent toy artifact has to cover the same names a real capture
        of this family would.
        """
        rng = np.random.default_rng(11)
        arrays: dict[str, np.ndarray] = {}
        for name in self.tool.expected_weight_names(self.arch):
            if "embed_tokens" in name or name.startswith("lm_head"):
                shape: tuple[int, ...] = (VOCAB, HIDDEN)
            elif name.endswith("norm.weight"):
                shape = (HIDDEN,)
            else:
                shape = (HIDDEN, HIDDEN)
            arrays[name] = rng.normal(size=shape).astype(np.float32)
        return arrays

    def _make_ours_meta(self) -> dict:
        samples = self._samples()
        tokenizer_names = self.tool.expected_tokenizer_names(self.capture)
        text_by_name = {
            f"{sample['name']}/{text['name']}": text
            for sample in samples
            if isinstance(sample.get("texts"), list)
            for text in sample["texts"]
        }
        return {
            "samples": samples,
            "model_directory": "tiny:converted",
            "captured_at": "2026-10-19T00:00:00+0800",
            "side": "numpy",
            "architecture": dict(self.arch),
            "source": {
                "repo_id": "tiny:local",
                "revision": REVISION,
                # A non-shard entry with no digest must not upset the provenance
                # check: the reference only hashes *.safetensors.
                "files": {SHARD: SHARD_SHA256, "config.json": None},
                "weight_sources": {},
            },
            "capture": self.capture,
            "tokenizer": [
                {
                    "name": entry_name,
                    "text": text_by_name[entry_name]["text"]
                    if entry_name in text_by_name
                    else next(
                        sample["rendered_text"] for sample in samples
                        if sample["name"] == entry_name
                    ),
                    "ids": list(
                        text_by_name[entry_name]["ids"]
                        if entry_name in text_by_name
                        else next(
                            sample["prompt_ids"] for sample in samples
                            if sample["name"] == entry_name
                        )
                    ),
                }
                for entry_name in tokenizer_names
            ],
            "weight_digests": {
                name: self.tool.tensor_digest(array)
                for name, array in self._weight_arrays().items()
            },
        }

    def _make_ref_meta(self) -> dict:
        samples = self.ours_meta["samples"]
        reference_texts = self._reference_texts()
        ref_tokenizer = []
        for entry in self.ours_meta["tokenizer"]:
            record = {"name": entry["name"], "text": entry["text"], "ids": list(entry["ids"])}
            rebuilt = reference_texts.get(entry["name"])
            if rebuilt is not None:
                record["text"] = rebuilt["text"]
            sample = next((s for s in samples if s["name"] == entry["name"]), None)
            if sample is not None:
                record["decode_input_tokens"] = [
                    step["input_token"] for step in sample["decode_steps"]
                ]
            ref_tokenizer.append(record)
        ref_samples = json.loads(json.dumps(samples))
        for sample in ref_samples:
            for entry in sample.get("texts", []):
                rebuilt = reference_texts.get(f"{sample['name']}/{entry['name']}")
                if rebuilt is not None:
                    entry.update(rebuilt)
        return {
            "side": "reference",
            "captured_at": "2026-10-19T00:00:00+0800",
            "hf_dir": "tiny:hf",
            "architecture": dict(self.ours_meta["architecture"]),
            "source": {
                "repo_id": None,
                "revision": None,
                "files": {SHARD: SHARD_SHA256},
                "weight_sources": {
                    name: name for name in self.ours_meta["weight_digests"]
                },
            },
            "capture": json.loads(json.dumps(self.capture)),
            "environment": {
                "transformers": "0.0.test",
                "torch": "0.0.test",
                "attn_implementation": "eager",
                "dtype": "float32",
                "threads": 1,
                "device": "cpu",
            },
            "tokenizer": ref_tokenizer,
            "weight_digests": json.loads(json.dumps(self.ours_meta["weight_digests"])),
            "samples": ref_samples,
        }

    # ---- writing ------------------------------------------------------------

    def write_metadata(self) -> None:
        (self.root / "samples.json").write_text(
            json.dumps(self.ours_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.root / self.tool.REFERENCE_FILENAME).write_text(
            json.dumps(self.ref_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def write_blobs(
        self,
        *,
        ours: dict[str, np.ndarray] | None = None,
        ref: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Rewrite whichever side is named; both default to the built arrays."""
        if ours is None and ref is None:
            ours = ref = self.arrays
        if ours is not None:
            np.savez_compressed(self.root / "slowq.npz", **ours)
        if ref is not None:
            np.savez_compressed(self.root / "ref.npz", **ref)

    def blobs_without(self, *keys: str, side: str = "ref") -> None:
        """Rewrite one side with ``keys`` removed."""
        kept = {key: value for key, value in self.arrays.items() if key not in keys}
        self.write_blobs(**{side: kept})  # type: ignore[arg-type]

    def blobs_with(self, side: str, **replacements: np.ndarray) -> None:
        updated = dict(self.arrays)
        updated.update(replacements)
        self.write_blobs(**{side: updated})  # type: ignore[arg-type]

    def compare(self) -> int:
        args = self.tool.build_parser().parse_args(["compare", "--artifacts", str(self.root)])
        return int(self.tool.run_compare(args))

    def first_key(self, suffix: str = ".logits") -> str:
        return next(key for key in sorted(self.arrays) if key.endswith(suffix))


@pytest.fixture
def artifacts(tool, tmp_path: Path) -> ArtifactSet:
    return ArtifactSet(tool, tmp_path / "validation")


@pytest.fixture
def embedding_artifacts(tool, tmp_path: Path) -> ArtifactSet:
    """The same toy protocol, for a checkpoint whose recorded task is embedding."""
    return ArtifactSet(
        tool,
        tmp_path / "validation-embedding",
        arch=toy_architecture(task="text-embedding", tie_word_embeddings=False),
    )


@pytest.fixture
def vision_artifacts(tool, tmp_path: Path) -> ArtifactSet:
    """The same toy protocol, for a checkpoint whose recorded task is image+text."""
    return ArtifactSet(tool, tmp_path / "validation-vision", arch=toy_vision_architecture())


def failure_text(capsys) -> str:
    captured = capsys.readouterr()
    return captured.err + captured.out


# --------------------------------------------------------------------------- #
# the happy path, which every test below breaks in exactly one way
# --------------------------------------------------------------------------- #


def test_a_consistent_capture_passes_and_says_how_much_it_compared(artifacts, capsys):
    expected = len(artifacts.tool.expected_numeric_keys(artifacts.capture))
    # 4 samples x (7 full + 4 chunked + 4 decode); two layers, so both of
    # capture_layers(2) == (0, 1) contribute a layer_output at every position.
    assert expected == 60
    assert artifacts.compare() == 0
    assert f"PASS: {expected} tensors compared" in capsys.readouterr().out


def test_the_expected_key_set_is_derived_from_the_capture_configuration(tool):
    capture = tool.capture_config(
        toy_architecture(num_hidden_layers=28), prefill_chunk_size=8, decode_steps=3
    )
    assert capture["layers"] == [0, 14, 27]
    keys = tool.expected_numeric_keys(capture)
    assert len(keys) == 116
    assert "zh-single-turn.full_prefill.layer27.layer_output[-1]" in keys
    assert "zh-single-turn.full_prefill.final_norm[0]" in keys
    assert "emoji-and-mixed-script.decode2.logits" in keys
    # A cached stage has no position 0: only the last token is captured there.
    assert "zh-single-turn.decode0.final_norm[0]" not in keys
    assert all(kind in ("logits", "hidden") for _, kind in keys.values())


def test_the_captured_layers_follow_the_checkpoint_not_a_constant(tool):
    """First, middle and last of *this* model — a 30-layer SmolLM2 has no 27."""
    assert tool.capture_layers(28) == (0, 14, 27)
    assert tool.capture_layers(30) == (0, 15, 29)
    assert tool.capture_layers(32) == (0, 16, 31)
    assert tool.capture_layers(24) == (0, 12, 23)
    # Fewer than three distinct depths must deduplicate rather than repeat a key.
    assert tool.capture_layers(2) == (0, 1)
    assert tool.capture_layers(1) == (0,)
    with pytest.raises(ValueError, match="num_hidden_layers"):
        tool.capture_layers(0)


def test_the_digested_weights_follow_the_family_rules(tool):
    """``q_norm`` only where it exists; a bias only where one is declared."""
    qwen3 = tool.weight_check_sample(toy_architecture())
    assert "model.layers.0.self_attn.q_norm.weight" in qwen3
    assert "model.layers.0.self_attn.k_norm.weight" in qwen3
    assert not any(name.endswith(".bias") for name in qwen3)

    llama = tool.weight_check_sample(toy_architecture(family="llama", qk_norm=False))
    assert not any("q_norm" in name or "k_norm" in name for name in llama)
    assert "model.layers.1.mlp.down_proj.weight" in llama

    biased = tool.weight_check_sample(
        toy_architecture(family="qwen2", qk_norm=False, qkv_bias=True, mlp_bias=True)
    )
    assert "model.layers.0.self_attn.q_proj.bias" in biased
    assert "model.layers.1.mlp.gate_proj.bias" in biased
    # o_proj carries no bias in any family SlowQ supports unless it says so.
    assert "model.layers.0.self_attn.o_proj.bias" not in biased

    embedding = tool.weight_check_sample(toy_architecture(task="text-embedding"))
    assert "lm_head.weight" not in embedding, "an encoder has no output projection"
    assert "model.embed_tokens.weight" in embedding


def test_a_tied_checkpoint_still_digests_lm_head_through_the_alias(tool):
    """Equal digests for the alias and its target are what prove the alias."""
    tied = tool.weight_check_sample(toy_architecture(tie_word_embeddings=True))
    untied = tool.weight_check_sample(toy_architecture(tie_word_embeddings=False))
    assert "lm_head.weight" in tied
    assert tied == untied


def test_the_weight_names_follow_the_stored_prefix(tool):
    nested = tool.weight_check_sample(toy_architecture(tensor_prefix="model.text_model"))
    assert "model.text_model.embed_tokens.weight" in nested
    assert "model.text_model.layers.1.mlp.down_proj.weight" in nested


def test_the_reference_fingerprint_restates_the_official_rules(tool):
    """Read off the raw config, not asked of SlowQ — that is what makes it a check."""
    # Qwen2 hard-codes q/k/v bias on and o_proj bias off, whatever the config says.
    qwen2 = tool.reference_architecture(
        {
            "model_type": "qwen2",
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 49408,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 32768,
            "sliding_window": 32768,
            "attention_bias": False,
            "tie_word_embeddings": True,
        },
        "text-generation",
    )
    assert qwen2["family"] == "qwen2"
    assert qwen2["qkv_bias"] is True
    assert qwen2["o_proj_bias"] is False
    assert qwen2["mlp_bias"] is False
    assert qwen2["qk_norm"] is False
    # No explicit head_dim: derived the way Qwen2Attention's getattr does.
    assert qwen2["head_dim"] == 896 // 14
    assert qwen2["sliding_window"] == 32768

    llama = tool.reference_architecture(
        {
            "model_type": "llama",
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "vocab_size": 49152,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "max_position_embeddings": 8192,
            "attention_bias": False,
            "mlp_bias": False,
            "tie_word_embeddings": True,
        },
        "text-generation",
    )
    assert llama["head_dim"] == 64
    # modeling_llama.py never reads sliding_window, so a value in config.json
    # must not turn into one in the fingerprint.
    assert llama["sliding_window"] is None
    assert tool.reference_architecture(
        {
            "model_type": "llama",
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "vocab_size": 49152,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "max_position_embeddings": 8192,
            "sliding_window": 2048,
            "tie_word_embeddings": True,
        },
        "text-generation",
    )["sliding_window"] is None

    qwen3 = tool.reference_architecture(
        {
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 40960,
            "tie_word_embeddings": True,
        },
        "text-generation",
    )
    # Qwen3 sets head_dim independently of hidden_size: 16 x 128 != 1024.
    assert qwen3["head_dim"] == 128
    assert qwen3["qk_norm"] is True
    assert qwen3["qkv_bias"] is False


def test_an_unsupported_family_cannot_be_fingerprinted(tool):
    with pytest.raises(SystemExit, match="not a text family"):
        tool.reference_architecture({"model_type": "mistral"}, "text-generation")


# --------------------------------------------------------------------------- #
# the embedding protocol
# --------------------------------------------------------------------------- #


def test_an_embedding_capture_passes_and_names_what_it_compared(
    embedding_artifacts, capsys
):
    expected = len(
        embedding_artifacts.tool.expected_numeric_keys(embedding_artifacts.capture)
    )
    assert expected > 0
    assert embedding_artifacts.compare() == 0
    text = capsys.readouterr().out
    assert f"PASS: {expected} tensors compared" in text
    assert "pool_index" in text


def test_an_embedding_capture_never_records_a_logit_or_a_decode_step(
    embedding_artifacts
):
    capture = embedding_artifacts.capture
    assert capture["decode_steps"] == 0
    assert "positions_full" not in capture
    keys = embedding_artifacts.tool.expected_numeric_keys(capture)
    assert not any(key.endswith(".logits") for key in keys)
    assert "retrieval-with-instruction.query.full_prefill.vector" in keys
    assert "retrieval-with-instruction.doc2.chunked_prefill.vector" in keys
    assert "retrieval-without-instruction.similarity" in keys
    assert all(kind in ("hidden", "vector") for _, kind in keys.values())


def test_an_embedding_vector_that_is_not_unit_norm_fails(embedding_artifacts, capsys):
    key = "retrieval-with-instruction.query.chunked_prefill.vector"
    scaled = (embedding_artifacts.arrays[key] * 1.5).astype(np.float32)
    broken = dict(embedding_artifacts.arrays)
    broken[key] = scaled
    # The reference keeps the unit-norm vector, so this also fails the numeric
    # comparison; the point is that the *norm* is named as its own problem.
    embedding_artifacts.write_blobs(ours=broken)
    assert embedding_artifacts.compare() == 1
    assert "not unit-norm" in failure_text(capsys)


def test_a_pooling_index_the_reference_disagrees_with_fails(embedding_artifacts, capsys):
    embedding_artifacts.ours_meta["samples"][0]["texts"][0]["pool_index"] = 0
    embedding_artifacts.write_metadata()
    assert embedding_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "pool_index" in text or "SlowQ records tokens" in text


def test_an_embedding_token_count_the_reference_disagrees_with_fails(
    embedding_artifacts, capsys
):
    embedding_artifacts.ours_meta["samples"][0]["texts"][0]["num_tokens"] += 1
    embedding_artifacts.write_metadata()
    assert embedding_artifacts.compare() == 1
    assert "SlowQ records tokens" in failure_text(capsys)


def test_a_query_prefix_missing_from_one_side_is_named(embedding_artifacts, capsys):
    """The tokenizer entries of an embedding capture are named ``sample/text``."""
    name = "retrieval-with-instruction/query"
    for meta in (embedding_artifacts.ours_meta, embedding_artifacts.ref_meta):
        meta["tokenizer"] = [e for e in meta["tokenizer"] if e["name"] != name]
    embedding_artifacts.write_metadata()
    assert embedding_artifacts.compare() == 1
    text = failure_text(capsys)
    assert name in text
    assert "missing from both sides" in text


def test_the_reference_rebuilds_the_query_prefix_itself(tool, monkeypatch):
    """The reference must not ask SlowQ how to spell its own input.

    ``format_query`` is replaced with something obviously wrong and
    ``reference_embedding_texts`` has to be unaffected: it is the statement of the
    official format that SlowQ's version is graded against, so importing it would
    turn the comparison into SlowQ agreeing with itself.
    """
    import slowq.embedding as slowq_embedding

    monkeypatch.setattr(
        slowq_embedding, "format_query", lambda query, instruction=None: f"BROKEN {query}"
    )
    assert slowq_embedding.format_query("anything") == "BROKEN anything"

    rebuilt = {
        text_name: (raw, text, is_query)
        for text_name, raw, text, is_query in tool.reference_embedding_texts(
            "retrieval-with-instruction"
        )
    }
    assert sorted(rebuilt) == ["doc0", "doc1", "doc2", "query"]
    raw, query_text, is_query = rebuilt["query"]
    assert is_query is True
    assert raw == next(
        sample["query"]
        for sample in tool.EMBEDDING_SAMPLES
        if sample["name"] == "retrieval-with-instruction"
    )
    assert query_text == (
        f"Instruct: {tool.DEFAULT_QUERY_INSTRUCTION}\nQuery:{raw}"
    )
    # The detail this whole check exists for: no space after the colon.
    assert "\nQuery: " not in query_text
    assert "BROKEN" not in query_text
    for name in ("doc0", "doc1", "doc2"):
        raw, text, is_query = rebuilt[name]
        assert is_query is False
        assert text == raw, "a document is embedded verbatim, with no prefix"

    # An instruction of None means no prefix at all, not an empty one.
    bare = {
        text_name: text
        for text_name, _, text, _ in tool.reference_embedding_texts(
            "retrieval-without-instruction"
        )
    }
    assert bare["query"] == "慢炖是一种什么样的烹饪方式？"
    assert "Instruct:" not in bare["query"] and "Query:" not in bare["query"]

    with pytest.raises(SystemExit, match="does not define"):
        tool.reference_embedding_texts("no-such-sample")


def test_a_correct_embedding_input_format_is_graded_and_passes(
    embedding_artifacts, capsys
):
    """The control for the fault injections below: nothing there is fatal as such."""
    assert embedding_artifacts.compare() == 0
    out = capsys.readouterr().out
    names = [
        f"{sample}/{text}"
        for sample, texts in embedding_artifacts.capture["embedding_texts"].items()
        for text in texts
    ]
    assert len(names) == 8
    assert "input=MISMATCH" not in out
    assert out.count("input=ok") == len(names), names


def test_a_broken_query_prefix_fails_the_reference_comparison(
    tool, tmp_path: Path, monkeypatch, capsys
):
    """Fault injection: break only ``slowq.embedding.format_query``.

    Weights, token counts, pooling indices and every captured array stay exactly
    as they were, so this is the situation the old reference could not see — it
    tokenised the text the NumPy side had already prefixed, so both sides moved
    together and the run reported PASS on a query nobody would retrieve with.
    """
    import slowq.embedding as slowq_embedding

    real = slowq_embedding.format_query
    monkeypatch.setattr(
        slowq_embedding,
        "format_query",
        lambda query, instruction=None: real(query, instruction).replace(
            "\nQuery:", "\nQuery: "
        ),
    )
    broken = ArtifactSet(
        tool,
        tmp_path / "validation-broken-prefix",
        arch=toy_architecture(task="text-embedding", tie_word_embeddings=False),
    )

    ours = broken.ours_meta["samples"][0]["texts"][0]
    theirs = broken.ref_meta["samples"][0]["texts"][0]
    # The precondition, stated rather than assumed: the same raw query and role,
    # and a difference that is exactly the injected space.
    assert (ours["raw"], ours["is_query"]) == (theirs["raw"], theirs["is_query"])
    assert ours["text"] != theirs["text"], (
        "the reference echoed SlowQ's own text instead of rebuilding it, so the "
        "two sides moved together and this comparison cannot see the prefix at all"
    )
    assert ours["text"] == theirs["text"].replace("\nQuery:", "\nQuery: ")
    # Only the instructed query moved; the documents and the prefix-free sample
    # are untouched, so a check that flagged everything would be useless.
    assert broken.ours_meta["samples"][0]["texts"][1]["text"] == (
        broken.ref_meta["samples"][0]["texts"][1]["text"]
    )
    assert broken.ours_meta["samples"][1]["texts"][0]["text"] == (
        broken.ref_meta["samples"][1]["texts"][0]["text"]
    )

    assert broken.compare() == 1
    text = failure_text(capsys)
    assert "retrieval-with-instruction/query" in text
    assert "no space after the colon" in text
    assert "input=MISMATCH" in text
    # The tokenizer section names it too, with the character offset.
    assert "SlowQ renders different input text than the reference did" in text


def test_a_document_that_received_the_query_prefix_fails(
    tool, tmp_path: Path, capsys
):
    """The other half of the recipe: a document must stay verbatim.

    Injected while the NumPy side is being built rather than by editing the
    artifact afterwards, so this is a SlowQ that prefixes documents and not a
    file the comparator was handed already doctored.
    """
    artifacts = ArtifactSet(
        tool,
        tmp_path / "validation-prefixed-document",
        arch=toy_architecture(task="text-embedding", tie_word_embeddings=False),
        numpy_prefixes_documents=True,
    )
    document = artifacts.ours_meta["samples"][0]["texts"][1]
    assert document["name"] == "doc0" and document["is_query"] is False
    assert document["text"].startswith("Instruct: ")
    assert artifacts.ref_meta["samples"][0]["texts"][1]["text"] == document["raw"]

    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "retrieval-with-instruction/doc0" in text
    assert "no prefix at all" in text


def test_an_embedding_reference_captured_before_version_two_is_refused(
    embedding_artifacts, capsys
):
    """Compatibility is explicit: an older capture is refused, not trusted.

    A version-1 reference echoed SlowQ's own prefixed text, so it carries no
    evidence about the format at all. Grading against it would report PASS for the
    very thing it cannot see.
    """
    embedding_artifacts.ref_meta["capture"].pop("embedding_input_format")
    embedding_artifacts.write_metadata()
    assert embedding_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "embedding_input_format" in text
    assert "re-capture the reference" in text
    # And the numerics are not quietly graded anyway.
    assert "the numeric comparison was skipped" in text
    # One usable reason to re-capture, not four invented ones: an unusable capture
    # says nothing about which samples were required, so it must not fall back to
    # the generation protocol's names and report this capture's texts as missing.
    assert "missing from both sides" not in text
    assert "zh-single-turn" not in text
    assert text.count("  - ") == 2


def test_a_reference_with_no_input_text_of_its_own_is_named(embedding_artifacts, capsys):
    """The capture block claims version 2 but the records are version 1's."""
    for sample in embedding_artifacts.ref_meta["samples"]:
        for entry in sample["texts"]:
            entry.pop("text")
    embedding_artifacts.write_metadata()
    assert embedding_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "predate embedding input format 2" in text
    assert "re-capture it" in text


def test_the_two_sides_embedding_different_raw_text_is_named_separately(
    embedding_artifacts, capsys
):
    """A different input and a different prefix must not share one message.

    "SlowQ embedded a different string" is only actionable if the reader knows
    whether the underlying query differed too; if it did, the prefix is not the
    thing to go and look at.
    """
    entry = embedding_artifacts.ours_meta["samples"][0]["texts"][0]
    entry["raw"] = "a completely different question"
    entry["text"] = entry["raw"]
    for record in embedding_artifacts.ours_meta["tokenizer"]:
        if record["name"] == "retrieval-with-instruction/query":
            record["text"] = entry["text"]
    embedding_artifacts.write_metadata()
    assert embedding_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "did not embed the same input" in text
    assert "no space after the colon" not in text


# --------------------------------------------------------------------------- #
# the input fields have to be present and well-typed before they can be compared
# --------------------------------------------------------------------------- #

#: The two sides the way ``compare`` addresses them: the label it prints in a
#: failure, and the ``ArtifactSet`` attribute holding that side's records.
INPUT_SIDES = (("SlowQ", "ours_meta"), ("reference", "ref_meta"))

INPUT_FIELDS = ("raw", "is_query", "text")


def embedding_text_entry(
    artifacts: ArtifactSet, meta: str, sample: str, text: str
) -> dict:
    """One ``texts[]`` record of one side, addressed the way a failure names it."""
    record = next(s for s in getattr(artifacts, meta)["samples"] if s["name"] == sample)
    return next(entry for entry in record["texts"] if entry["name"] == text)


def input_row(output: str, sample: str, text: str) -> str:
    """The ``[embedding]`` table row for one text, whatever its padding."""
    wanted = f"{sample}/{text}"
    rows = [
        line
        for line in output.splitlines()
        if line.strip().startswith(wanted) and " input=" in line
    ]
    assert len(rows) == 1, (wanted, rows)
    return rows[0]


@pytest.mark.parametrize("side,meta", INPUT_SIDES)
@pytest.mark.parametrize("field", INPUT_FIELDS)
def test_an_input_field_missing_from_one_side_is_named_with_its_side(
    embedding_artifacts, capsys, side, meta, field
):
    """A one-sided loss has to say which side lost it, and which field.

    Grading the two sides with ``ours.get(f) == theirs.get(f)`` cannot: ``get``
    answers ``None`` for a key that was never written, so one side losing a field
    read as a disagreement with a message that named neither.
    """
    sample, text = "retrieval-with-instruction", "query"
    del embedding_text_entry(embedding_artifacts, meta, sample, text)[field]
    embedding_artifacts.write_metadata()

    assert embedding_artifacts.compare() == 1
    output = failure_text(capsys)
    assert f"{sample}/{text}: the {side} record has no {field!r} field" in output
    assert "input=ok" not in input_row(output, sample, text)


@pytest.mark.parametrize("field", INPUT_FIELDS)
def test_an_input_field_missing_from_both_sides_still_fails(
    embedding_artifacts, capsys, field
):
    """The case that used to pass: ``None == None``, so the two sides "agreed"."""
    sample, text = "retrieval-without-instruction", "doc1"
    for _side, meta in INPUT_SIDES:
        del embedding_text_entry(embedding_artifacts, meta, sample, text)[field]
    embedding_artifacts.write_metadata()

    assert embedding_artifacts.compare() == 1
    output = failure_text(capsys)
    for side, _meta in INPUT_SIDES:
        assert f"{sample}/{text}: the {side} record has no {field!r} field" in output
    assert "input=ok" not in input_row(output, sample, text)


def test_a_capture_where_both_sides_dropped_every_input_field_fails(
    embedding_artifacts, capsys
):
    """All three fields, both sides, every text: the reproduction from the review.

    On copies of the real ``Qwen3-Embedding-0.6B`` artifacts this exited 0 with
    ``PASS: 82 tensors compared`` and eight ``input=ok`` rows. The numerics here
    are genuinely unchanged too — pooling indices, token counts and every array
    still agree — so nothing else in the run has a reason to object, and this
    check is the only thing between that capture and a PASS.
    """
    samples = embedding_artifacts.capture["sample_names"]
    texts = embedding_artifacts.capture["embedding_texts"]
    for sample in samples:
        for text in texts[sample]:
            for _side, meta in INPUT_SIDES:
                entry = embedding_text_entry(embedding_artifacts, meta, sample, text)
                for field in INPUT_FIELDS:
                    del entry[field]
    embedding_artifacts.write_metadata()

    assert embedding_artifacts.compare() == 1
    output = failure_text(capsys)
    assert "PASS" not in output
    assert "input=ok" not in output
    assert output.count("input=INVALID") == 8, "every text is unusable, not some"
    # The counts still agree, which is exactly why the input fields need their own
    # check: nothing downstream of them objects to this capture.
    assert output.count("pooled=ok") == 8
    for sample in samples:
        for text in texts[sample]:
            for side, _meta in INPUT_SIDES:
                for field in INPUT_FIELDS:
                    assert (
                        f"{sample}/{text}: the {side} record has no {field!r} field"
                        in output
                    )


def test_a_null_input_field_is_reported_as_null_not_as_a_missing_key(
    embedding_artifacts, capsys
):
    """A serializer that wrote an unset value leaves the key there holding null.

    Both sides null is ``None == None`` again, so it has to fail too — but as a
    null, because "the key was never written" and "the key was written empty"
    point at different causes.
    """
    sample, text = "retrieval-with-instruction", "doc2"
    for _side, meta in INPUT_SIDES:
        embedding_text_entry(embedding_artifacts, meta, sample, text)["raw"] = None
    embedding_artifacts.write_metadata()

    assert embedding_artifacts.compare() == 1
    output = failure_text(capsys)
    for side, _meta in INPUT_SIDES:
        assert f"{sample}/{text}: the {side} record's 'raw' is null" in output
    assert "has no 'raw' field" not in output
    assert "input=ok" not in input_row(output, sample, text)


def test_an_is_query_written_as_an_integer_fails_though_both_sides_agree(
    embedding_artifacts, capsys
):
    """``bool`` subclasses ``int``, so no equality check can police this field.

    ``1 == 1`` holds and so does ``1 == True``, so a capture that recorded the
    role as a JSON number agreed with itself *and* with a correct capture. Only
    the type says whether the field means anything.
    """
    sample = "retrieval-with-instruction"
    for text, as_number in (("query", 1), ("doc0", 0)):
        for _side, meta in INPUT_SIDES:
            entry = embedding_text_entry(embedding_artifacts, meta, sample, text)
            assert entry["is_query"] == as_number, "equality already agrees: True == 1"
            entry["is_query"] = as_number
    embedding_artifacts.write_metadata()

    assert embedding_artifacts.compare() == 1
    output = failure_text(capsys)
    for text, as_number in (("query", 1), ("doc0", 0)):
        for side, _meta in INPUT_SIDES:
            assert (
                f"{sample}/{text}: the {side} record's 'is_query' is {as_number!r}, "
                f"of type int, where the capture format requires a JSON boolean"
            ) in output
        assert "input=ok" not in input_row(output, sample, text)


def test_an_input_text_of_the_wrong_type_is_refused_though_both_sides_match(
    embedding_artifacts, capsys
):
    """Two identical wrong types compare equal, so again only the type catches it."""
    sample, text = "retrieval-without-instruction", "query"
    wrong = ["慢炖是一种什么样的烹饪方式？"]
    for _side, meta in INPUT_SIDES:
        embedding_text_entry(embedding_artifacts, meta, sample, text)["text"] = list(wrong)
    embedding_artifacts.write_metadata()

    assert embedding_artifacts.compare() == 1
    output = failure_text(capsys)
    for side, _meta in INPUT_SIDES:
        assert (
            f"{sample}/{text}: the {side} record's 'text' is {wrong!r}, of type list, "
            f"where the capture format requires a string"
        ) in output
    assert "input=ok" not in input_row(output, sample, text)


def test_a_legal_capture_still_passes_the_field_check(embedding_artifacts, capsys):
    """The control, so the tests above are not green by demanding nothing.

    Every field is present and correctly typed on both sides, so the verdict is
    the same as it was before the check existed — for a capture that really does
    record what it embedded.
    """
    for side, meta in INPUT_SIDES:
        for sample in embedding_artifacts.capture["sample_names"]:
            for text in embedding_artifacts.capture["embedding_texts"][sample]:
                entry = embedding_text_entry(embedding_artifacts, meta, sample, text)
                assert isinstance(entry["raw"], str), side
                assert isinstance(entry["text"], str), side
                assert isinstance(entry["is_query"], bool), side

    assert embedding_artifacts.compare() == 0
    output = failure_text(capsys)
    assert "PASS" in output
    rows = [line for line in output.splitlines() if " input=" in line]
    assert len(rows) == 8
    assert all("input=ok" in row for row in rows)
    assert "INVALID" not in output


def test_a_legal_capture_whose_inputs_disagree_still_fails(embedding_artifacts, capsys):
    """Presence and type are not the whole check — the values still have to agree.

    A fix that only demanded the fields exist would let a wrong query prefix back
    through, so the original case stays covered next to the new one.
    """
    sample, text = "retrieval-with-instruction", "query"
    entry = embedding_text_entry(embedding_artifacts, "ours_meta", sample, text)
    entry["text"] = entry["text"].replace("\nQuery:", "\nQuery: ")
    for record in embedding_artifacts.ours_meta["tokenizer"]:
        if record["name"] == f"{sample}/{text}":
            record["text"] = entry["text"]
    embedding_artifacts.write_metadata()

    assert embedding_artifacts.compare() == 1
    output = failure_text(capsys)
    assert "no space after the colon" in output
    assert "input=MISMATCH" in input_row(output, sample, text)
    assert "input=INVALID" not in output


def test_the_cli_refuses_a_capture_that_dropped_the_input_fields(embedding_artifacts):
    """Through the real entry point, so the exit code is the one a caller gets."""
    for sample in embedding_artifacts.capture["sample_names"]:
        for text in embedding_artifacts.capture["embedding_texts"][sample]:
            for _side, meta in INPUT_SIDES:
                entry = embedding_text_entry(embedding_artifacts, meta, sample, text)
                for field in INPUT_FIELDS:
                    del entry[field]
    embedding_artifacts.write_metadata()

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "validate_reference.py"),
            "compare",
            "--artifacts",
            str(embedding_artifacts.root),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert completed.returncode == 1
    assert "retrieval-with-instruction/query" in completed.stderr
    for side, _meta in INPUT_SIDES:
        assert f"the {side} record has no 'is_query' field" in completed.stderr
    assert "PASS" not in completed.stdout
    assert "input=ok" not in completed.stdout
    assert completed.stdout.count("input=INVALID") == 8


# --------------------------------------------------------------------------- #
# the production reference capture, executed rather than restated in a fixture
# --------------------------------------------------------------------------- #


class _FakeTorchTensor:
    """The slice of the torch tensor API ``_reference_embedding_capture`` touches.

    Only ``__getitem__``, ``sum``, ``unsqueeze``, ``shape`` and ``numpy`` — enough
    for the pooling index and the L2 normalisation. What is *not* faked is which
    string gets embedded, and that is the code under test.
    """

    def __init__(self, array) -> None:
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, key):
        return _FakeTorchTensor(self.array[key])

    def sum(self):
        return self.array.sum()

    def unsqueeze(self, axis):
        return _FakeTorchTensor(np.expand_dims(self.array, axis))

    def numpy(self):
        return self.array


def _install_fake_torch(monkeypatch):
    """Put a stand-in ``torch`` in ``sys.modules`` for one test.

    The real reference side runs in a separate environment that has torch; this
    repository deliberately does not, and the capture imports it inside the
    function. ``monkeypatch.setitem`` removes the entries again on teardown, so
    nothing here leaks into the dependency-boundary tests.
    """
    torch = types.ModuleType("torch")
    torch.long = "int64"
    torch.no_grad = contextlib.nullcontext
    torch.tensor = lambda obj, dtype=None: _FakeTorchTensor(np.asarray(obj))

    functional = types.ModuleType("torch.nn.functional")

    def normalize(value, p=2, dim=-1):
        array = value.array.astype(np.float64)
        denominator = np.linalg.norm(array, axis=dim, keepdims=True)
        return _FakeTorchTensor((array / denominator).astype(np.float32))

    functional.normalize = normalize
    nn = types.ModuleType("torch.nn")
    nn.functional = functional
    torch.nn = nn
    for name, module in (
        ("torch", torch),
        ("torch.nn", nn),
        ("torch.nn.functional", functional),
    ):
        monkeypatch.setitem(sys.modules, name, module)


class _FakeLayerHooks:
    """``_LayerHooks`` without a decoder to attach to: the same two attributes."""

    def __init__(self, layers) -> None:
        self.layers = tuple(int(index) for index in layers)
        self.captured: dict[str, np.ndarray] = {}


class _FakeReferenceModel:
    """One forward per call, filling the hooks the way real forward hooks do.

    The numbers are meaningless and only have to be the right shape; the capture
    reads ``last_hidden_state`` and the hook dictionary, and both are here.
    """

    def __init__(self, hidden: int, hooks: _FakeLayerHooks) -> None:
        self.hidden = hidden
        self.hooks = hooks
        self.forwards = 0

    def __call__(self, input_ids=None, attention_mask=None, use_cache=False, return_dict=True):
        assert use_cache is False and return_dict is True
        rows = int(input_ids.array.shape[1])
        self.forwards += 1
        state = np.random.default_rng(rows).normal(size=(1, rows, self.hidden))
        state = state.astype(np.float32)
        for layer in self.hooks.layers:
            self.hooks.captured[f"layer{layer}"] = state
        self.hooks.captured["final_norm"] = state
        # (batch, sequence, hidden): the capture pools it as ``[0, pool_index]``.
        return types.SimpleNamespace(last_hidden_state=_FakeTorchTensor(state))


class _SpyTokenizer:
    """Records every string the capture hands it, and how it was handed over."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        num_ids = 6 + len(text) % 7
        return {"input_ids": list(range(1, num_ids + 1)), "attention_mask": [1] * num_ids}

    @property
    def texts(self) -> list[str]:
        return [text for text, _ in self.calls]


def slowq_embedding_payload(tool, *, broken_prefix: bool = False) -> dict:
    """The ``payload`` the NumPy side hands the reference run.

    This is SlowQ's record, not a reimplementation of the capture: the only field
    of it that matters here is ``text``, which is precisely the field the
    reference must ignore. ``broken_prefix`` spells the query prefix with an extra
    space after the colon — the one-character mistake the whole input-format
    check exists to catch.
    """
    from slowq.embedding import format_query

    def prefixed(query, instruction):
        text = format_query(query, instruction)
        return text.replace("\nQuery:", "\nQuery: ") if broken_prefix else text

    samples = []
    for definition in tool.EMBEDDING_SAMPLES:
        instruction = definition["instruction"]
        texts = [
            {
                "name": "query",
                "raw": definition["query"],
                "text": prefixed(definition["query"], instruction),
                "is_query": True,
            }
        ]
        texts += [
            {"name": f"doc{index}", "raw": document, "text": document, "is_query": False}
            for index, document in enumerate(definition["documents"])
        ]
        samples.append({"name": definition["name"], "instruction": instruction, "texts": texts})
    return {"samples": samples}


def run_reference_capture(tool, payload, monkeypatch):
    """Execute the production capture against the stand-ins above."""
    _install_fake_torch(monkeypatch)
    hooks = _FakeLayerHooks((0, 1))
    model = _FakeReferenceModel(HIDDEN, hooks)
    tokenizer = _SpyTokenizer()
    blobs, tokenizer_reference, samples_reference = tool._reference_embedding_capture(
        model, tokenizer, payload, hooks
    )
    return blobs, tokenizer_reference, samples_reference, tokenizer, model


def official_embedding_texts(tool) -> list[str]:
    """The eight strings the official recipe embeds, spelled out literally.

    Deliberately *not* ``reference_embedding_texts``: comparing the capture
    against the function it calls would pass even if both were wrong. This is the
    model card's format restated a third time, in a test.
    """
    instructed, uninstructed = tool.EMBEDDING_SAMPLES
    assert instructed["name"] == "retrieval-with-instruction"
    assert uninstructed["name"] == "retrieval-without-instruction"
    assert uninstructed["instruction"] is None
    return [
        f"Instruct: {instructed['instruction']}\nQuery:{instructed['query']}",
        *instructed["documents"],
        uninstructed["query"],
        *uninstructed["documents"],
    ]


def test_the_production_reference_capture_embeds_its_own_text(tool, monkeypatch):
    """Run the real ``_reference_embedding_capture`` and watch what it tokenises.

    The previous version of this suite built an ``ArtifactSet`` and rebuilt the
    reference text inside a test helper, so the production function was never
    called — replacing its entire body with ``raise AssertionError`` left all 91
    tests passing. Here it runs, on a payload whose ``text`` fields are already
    wrong, and the spy shows it embedded the official strings anyway.
    """
    payload = slowq_embedding_payload(tool, broken_prefix=True)
    corrupted = [entry["text"] for sample in payload["samples"] for entry in sample["texts"]]
    assert corrupted[0].count("\nQuery: ") == 1, "the payload really is broken"

    _blobs, _tokenizer_reference, samples_reference, tokenizer, model = run_reference_capture(
        tool, payload, monkeypatch
    )

    # The whole test: exactly the official eight, in the capture's order, and not
    # one of the strings SlowQ handed over.
    expected = official_embedding_texts(tool)
    assert tokenizer.texts == expected, (
        "the reference tokenised something other than the official eight strings; "
        "if the difference is SlowQ's spacing, it inherited SlowQ's own text field"
    )
    assert corrupted[0] not in tokenizer.texts, (
        "the reference tokenised the text SlowQ had already prefixed, so a wrong "
        "prefix is compared against itself and can never be reported"
    )
    assert all(kwargs == {} for _, kwargs in tokenizer.calls), (
        "the official snippet calls the tokenizer with its default "
        "add_special_tokens, and this checkpoint's post-processor matters"
    )

    # The stand-ins really were exercised, so a failure above is a finding about
    # the capture and not about a mock that never ran.
    assert model.forwards == len(expected)
    assert len(samples_reference) == 2

    records = {
        f"{sample['name']}/{entry['name']}": entry
        for sample in samples_reference
        for entry in sample["texts"]
    }
    query = records["retrieval-with-instruction/query"]
    assert query["text"] == expected[0]
    assert query["raw"] == tool.EMBEDDING_SAMPLES[0]["query"]
    assert query["is_query"] is True
    bare = records["retrieval-without-instruction/query"]
    assert bare["text"] == bare["raw"] and bare["is_query"] is True
    assert "Query:" not in bare["text"], "no instruction means no prefix at all"
    for name, entry in records.items():
        if not name.endswith("/query"):
            assert entry["text"] == entry["raw"], f"{name} took a prefix"
            assert entry["is_query"] is False
        # The capture writes the three fields the comparator now refuses to do
        # without, and writes them with the types it demands.
        assert isinstance(entry["raw"], str) and isinstance(entry["text"], str)
        assert isinstance(entry["is_query"], bool)
        assert isinstance(entry["num_tokens"], int) and isinstance(entry["pool_index"], int)
        assert entry["dimension"] == HIDDEN


def test_the_reference_records_do_not_move_when_slowqs_payload_does(
    tool, monkeypatch
):
    """The independence proof, stated as an equality rather than as a message.

    Capture twice from two payloads that differ only in SlowQ's ``text`` field.
    If the reference read that field, its own records would move with it and the
    two sides would agree on whatever SlowQ said.
    """
    honest = slowq_embedding_payload(tool, broken_prefix=False)
    broken = slowq_embedding_payload(tool, broken_prefix=True)
    assert [e["text"] for e in honest["samples"][0]["texts"]] != [
        e["text"] for e in broken["samples"][0]["texts"]
    ], "the two payloads must actually differ"

    _, honest_tokenizer, honest_samples, _, _ = run_reference_capture(tool, honest, monkeypatch)
    _, broken_tokenizer, broken_samples, _, _ = run_reference_capture(tool, broken, monkeypatch)

    assert honest_samples == broken_samples, (
        "the reference's own records moved when SlowQ's payload did, so it is "
        "reading SlowQ's text field and grading it against itself"
    )
    assert honest_tokenizer == broken_tokenizer, "the rendered texts moved too"
    assert [e["text"] for e in honest_tokenizer] == official_embedding_texts(tool)


def test_the_reference_capture_names_a_payload_that_disagrees_about_its_texts(
    tool, monkeypatch
):
    """The guard that runs before any text is read: names, not contents.

    It compares the recorded *names* against this version of the tool, so a
    payload from a capture with a different text set is refused instead of being
    silently re-keyed.
    """
    payload = slowq_embedding_payload(tool)
    payload["samples"][0]["texts"].pop()
    with pytest.raises(SystemExit, match="re-run 'validate_reference.py numpy'"):
        run_reference_capture(tool, payload, monkeypatch)


def test_a_broken_prefix_is_caught_when_the_reference_is_re_captured_by_production(
    tool, tmp_path: Path, monkeypatch, capsys
):
    """End to end: SlowQ broken, reference re-taken by the real capture, ``compare``.

    Only the reference side's *input records* are replaced, by what
    ``_reference_embedding_capture`` returned for the same broken payload; the
    token counts and every numeric array stay as the fixture built them. So the
    counts still agree and the only thing left to fail is the input format, which
    is what this is testing.
    """
    import slowq.embedding as slowq_embedding

    real = slowq_embedding.format_query
    monkeypatch.setattr(
        slowq_embedding,
        "format_query",
        lambda query, instruction=None: real(query, instruction).replace(
            "\nQuery:", "\nQuery: "
        ),
    )
    artifacts = ArtifactSet(
        tool,
        tmp_path / "validation-recaptured",
        arch=toy_architecture(task="text-embedding", tie_word_embeddings=False),
    )
    _blobs, tokenizer_reference, samples_reference, _tokenizer, _model = run_reference_capture(
        tool, slowq_embedding_payload(tool), monkeypatch
    )

    captured = {f"{s['name']}/{e['name']}": e for s in samples_reference for e in s["texts"]}
    rendered = {entry["name"]: entry["text"] for entry in tokenizer_reference}
    for sample in artifacts.ref_meta["samples"]:
        for entry in sample["texts"]:
            source = captured[f"{sample['name']}/{entry['name']}"]
            for field in INPUT_FIELDS:
                entry[field] = source[field]
    for entry in artifacts.ref_meta["tokenizer"]:
        entry["text"] = rendered[entry["name"]]
    artifacts.write_metadata()

    ours = artifacts.ours_meta["samples"][0]["texts"][0]
    theirs = artifacts.ref_meta["samples"][0]["texts"][0]
    assert ours["text"] == theirs["text"].replace("\nQuery:", "\nQuery: "), (
        "the production capture returned the same broken text SlowQ handed it, so "
        "there is no disagreement left for compare to find"
    )
    assert ours["num_tokens"] == theirs["num_tokens"], "only the input text differs"

    assert artifacts.compare() == 1
    output = failure_text(capsys)
    assert "retrieval-with-instruction/query" in output
    assert "no space after the colon" in output
    assert "input=MISMATCH" in input_row(output, "retrieval-with-instruction", "query")
    assert "input=INVALID" not in output


def test_a_similarity_matrix_is_not_graded_by_a_next_token_argmax(
    embedding_artifacts, capsys
):
    """A cosine-similarity matrix has no greedy choice for the two sides to share.

    Both sides here differ by 1e-4 at two entries — well inside the declared
    ``max_abs_vector`` of 1e-3 — but that is enough to move which entry is the
    largest. Grading an argmax over it would fail a run whose numbers agree, and
    would make the verdict depend on which of two nearly tied documents sorted
    first.
    """
    key = "retrieval-with-instruction.similarity"
    base = embedding_artifacts.arrays[key]
    top = int(np.argmax(base))
    second = int(np.argsort(base.reshape(-1))[-2])
    assert top != second

    lead = np.float32(float(np.max(base)) + 1.0)
    trail = np.float32(lead - np.float32(1e-4))
    reference = base.copy()
    ours = base.copy()
    reference.flat[top], reference.flat[second] = lead, trail
    ours.flat[top], ours.flat[second] = trail, lead
    # The precondition, stated rather than assumed: the two sides genuinely
    # disagree about the flat maximum while agreeing to within tolerance.
    assert int(np.argmax(ours)) == second != int(np.argmax(reference))
    declared = embedding_artifacts.tool.TOLERANCES["max_abs_vector"]
    assert float(np.max(np.abs(ours - reference))) < declared

    embedding_artifacts.write_blobs(
        ours={**embedding_artifacts.arrays, key: ours},
        ref={**embedding_artifacts.arrays, key: reference},
    )
    assert embedding_artifacts.compare() == 0
    assert "greedy argmax" not in failure_text(capsys)



# --------------------------------------------------------------------------- #
# the image+text protocol
# --------------------------------------------------------------------------- #


def smolvlm_raw_config(**text_overrides) -> dict:
    """SmolVLM-256M-Instruct's ``config.json``, the numbers the checkpoint ships.

    Reproduced rather than read from a download so the fingerprint tests stay
    offline; these are the values recorded for revision
    ``7e3e67edbbed1bf9888184d9df282b700a323964``.
    """
    text_config = {
        "model_type": "llama",
        "architectures": ["VLlama3ForCausalLM"],
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "head_dim": 64,
        "vocab_size": 49280,
        "rms_norm_eps": 1e-5,
        "rope_theta": 100000,
        "max_position_embeddings": 8192,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": False,
    }
    text_config.update(text_overrides)
    return {
        "model_type": "idefics3",
        "architectures": ["Idefics3ForConditionalGeneration"],
        "image_token_id": 49190,
        "scale_factor": 4,
        "tie_word_embeddings": False,
        "vocab_size": 49280,
        "text_config": text_config,
        "vision_config": {
            "model_type": "idefics3",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "image_size": 512,
            "patch_size": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "num_channels": 3,
        },
    }


def test_a_vision_capture_passes_and_says_what_it_looked_at(vision_artifacts, capsys):
    expected = len(vision_artifacts.tool.expected_numeric_keys(vision_artifacts.capture))
    # 2 samples x (2 encoder frames + 1 connector + 14 full + 4 chunked + 4 decode).
    assert expected == 50
    assert vision_artifacts.compare() == 0
    text = capsys.readouterr().out
    assert f"PASS: {expected} tensors compared" in text
    # The vision tower's own layout is printed, so a reader can see which model
    # was graded rather than only that something was.
    assert "vision tower" in text
    assert "image_seq_len=4" in text and "connector_out=24" in text
    assert "grid=1x4 frames=5 image_tokens=20" in text
    assert "pixels=ok mask=ok" in text


def test_a_vision_capture_records_no_pooled_vector_and_one_fused_sequence(
    vision_artifacts
):
    """``embed_sequence`` exists only for the full prefill: a chunk's proves nothing.

    A chunk holds a slice of the fused matrix, so comparing it would restate the
    logits comparison at a different offset instead of saying where the image rows
    landed in the prompt.
    """
    keys = vision_artifacts.tool.expected_numeric_keys(vision_artifacts.capture)
    assert not any(key.endswith(".vector") or key.endswith(".similarity") for key in keys)
    fused = [key for key in keys if key.endswith(".embed_sequence")]
    assert fused == ["grid-1x4.full_prefill.embed_sequence", "grid-2x4.full_prefill.embed_sequence"]
    assert all(kind in ("logits", "hidden") for _, kind in keys.values())


def test_the_vision_key_set_follows_the_frame_grid(vision_artifacts):
    """First and last frame of *this* image, and every frame's connector row."""
    keys = vision_artifacts.tool.expected_numeric_keys(vision_artifacts.capture)
    assert "grid-2x4.vision.encoder_output[0]" in keys
    assert "grid-2x4.vision.encoder_output[8]" in keys
    # 9 frames, 2 of them stored: the middle seven are covered by the connector
    # output, which is recorded as one matrix over the whole image span.
    assert not any(
        key.startswith("grid-2x4.vision.encoder_output[") and key.endswith("4]")
        for key in keys
    )
    assert keys["grid-1x4.vision.connector_output"] == ("grid-1x4", "hidden")
    # The sampled fused positions come from the per-sample layout, not from a
    # capture-wide constant: a 5-frame image and a 9-frame one have different spans.
    assert "grid-1x4.full_prefill.final_norm[23]" in keys
    assert "grid-2x4.full_prefill.final_norm[39]" in keys
    assert "grid-1x4.full_prefill.final_norm[39]" not in keys


def test_vision_capture_frames_picks_the_two_ends_of_the_frame_stack(tool):
    assert tool.vision_capture_frames(9) == (0, 8)
    assert tool.vision_capture_frames(5) == (0, 4)
    assert tool.vision_capture_frames(2) == (0, 1)
    # A single-frame image collapses rather than storing one tensor under two keys.
    assert tool.vision_capture_frames(1) == (0,)
    with pytest.raises(ValueError, match="produced 0 frames"):
        tool.vision_capture_frames(0)


def test_a_capture_that_records_no_vision_layout_is_rejected(vision_artifacts, tool):
    """Without the layout the expected key set cannot be built at all."""
    capture = json.loads(json.dumps(vision_artifacts.capture))
    capture["vision"] = {}
    with pytest.raises(tool.ComparisonError, match="records no vision layout"):
        tool.expected_numeric_keys(capture)


def test_the_reference_fingerprint_restates_the_vision_tower_from_the_raw_config(tool):
    """Derived from the nested configs, not read back from SlowQ."""
    arch = tool.reference_architecture(smolvlm_raw_config(), "image-text-generation")
    assert arch["family"] == "idefics3"
    # The wrapper is called Idefics3, but the bias and norm rules were restated
    # from the decoder's own model_type — a llama, so all three off.
    assert arch["decoder_family"] == "llama"
    assert arch["tensor_prefix"] == "model.text_model"
    assert arch["qkv_bias"] is False and arch["o_proj_bias"] is False
    assert arch["mlp_bias"] is False and arch["qk_norm"] is False
    assert arch["sliding_window"] is None
    # Idefics3Model computes image_seq_len itself; processor_config.json is not
    # read, so restating the expression is what makes the agreement a check.
    assert arch["image_seq_len"] == 64
    assert arch["connector_out_features"] == 768 * 16
    assert arch["vision_head_dim"] == 64
    assert arch["vision_num_hidden_layers"] == 12
    assert arch["image_token_id"] == 49190
    assert arch["head_dim"] == 64
    assert set(tool.VISION_ARCHITECTURE_FIELDS) <= set(arch)


def test_an_explicit_decoder_head_dim_survives_the_vision_wrapper(tool):
    """576 / 9 is also 64, so SmolVLM alone cannot tell the two readings apart."""
    derived = tool.reference_architecture(
        smolvlm_raw_config(head_dim=None), "image-text-generation"
    )["head_dim"]
    assert derived == 576 // 9
    explicit = tool.reference_architecture(smolvlm_raw_config(head_dim=32), "image-text-generation")
    assert explicit["head_dim"] == 32, "an explicit head_dim must not be recomputed"


def test_an_idefics3_config_that_does_not_nest_its_two_halves_is_rejected(tool):
    raw = smolvlm_raw_config()
    del raw["vision_config"]
    with pytest.raises(SystemExit, match="must nest text_config and vision_config"):
        tool.reference_architecture(raw, "image-text-generation")


def test_an_idefics3_decoder_family_without_restatable_rules_is_rejected(tool):
    """Guessing mistral's bias rules from the wrapper's name is what this refuses."""
    with pytest.raises(SystemExit, match="cannot restate the bias and norm rules"):
        tool.reference_architecture(
            smolvlm_raw_config(model_type="mistral"), "image-text-generation"
        )


def test_the_digested_vision_weights_follow_the_tower_depths(tool):
    """The tower has its own layer count, and every one of its projections is biased."""
    arch = toy_vision_architecture(num_hidden_layers=2, vision_num_hidden_layers=12)
    names = tool.weight_check_sample(arch)
    assert "model.vision_model.embeddings.patch_embedding.bias" in names
    assert "model.vision_model.embeddings.position_embedding.weight" in names
    assert "model.vision_model.encoder.layers.0.self_attn.q_proj.bias" in names
    assert "model.vision_model.encoder.layers.0.layer_norm1.bias" in names
    assert "model.vision_model.encoder.layers.6.self_attn.out_proj.weight" in names
    assert "model.vision_model.encoder.layers.11.mlp.fc2.weight" in names
    assert "model.vision_model.post_layernorm.bias" in names
    assert "model.connector.modality_projection.proj.weight" in names
    # First/middle/last of the *tower*: 0, 6 and 11, not the decoder's 0, 1, 1.
    depths = {
        int(name.split(".encoder.layers.")[1].split(".")[0])
        for name in names
        if ".encoder.layers." in name
    }
    assert depths == {0, 6, 11}
    # The decoder half is still digested, under the prefix the wrapper stores it at.
    assert "model.text_model.embed_tokens.weight" in names
    assert "model.text_model.norm.weight" in names
    assert "lm_head.weight" in names
    # A text capture of the same decoder must not be asked for any of it.
    text_only = tool.weight_check_sample(toy_vision_architecture(task="text-generation"))
    assert not any(name.startswith("model.vision_model") for name in text_only)


def test_a_pixel_tensor_the_two_preprocessors_disagree_about_is_named_with_its_digest(
    vision_artifacts, capsys
):
    """Exact, not within a tolerance: a digest is over the bytes of every frame."""
    image = vision_artifacts.ours_meta["samples"][0]["image"]
    image["pixel_values_digest"] = vision_artifacts.tool.tensor_digest(
        np.zeros((5, 3, 8, 8), dtype=np.float32) + 0.5
    )
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "pixels=DIFFERS" in text and "mask=ok" in text
    assert "MISMATCH: pixel_values_digest" in text
    assert "the two preprocessors did not produce identical tensors" in text
    # Both shapes and both hashes are printed, so the reader can tell a resized
    # frame stack from a differently filtered one of the same shape.
    assert "shape [5, 3, 8, 8] vs reference [5, 3, 8, 8]" in text
    ours_sha = image["pixel_values_digest"]["sha256"]
    theirs_sha = vision_artifacts.ref_meta["samples"][0]["image"]["pixel_values_digest"]["sha256"]
    assert ours_sha != theirs_sha
    assert ours_sha in text and theirs_sha in text


def test_a_crop_grid_the_reference_disagrees_about_fails(vision_artifacts, capsys):
    """A wrong grid changes the frame count, the prompt and every tensor after it."""
    vision_artifacts.ours_meta["samples"][1]["image"]["rows"] = 1
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "vision sample grid-2x4, rows: SlowQ records 1 but the reference records 2" in text


def test_an_image_token_count_the_reference_disagrees_about_fails(vision_artifacts, capsys):
    vision_artifacts.ours_meta["samples"][0]["image"]["num_image_tokens"] -= 1
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "num_image_tokens: SlowQ records 19 but the reference records 20" in text
    assert "image_tokens=20" in text


def test_an_image_token_position_the_reference_disagrees_about_fails(vision_artifacts, capsys):
    """The count can agree while the span sits one token over."""
    vision_artifacts.ours_meta["samples"][0]["first_image_position"] += 1
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "first_image_position: SlowQ records 5 but the reference records 4" in text
    assert "image positions 4…23" in text


def test_a_pixel_attention_mask_that_is_not_all_valid_fails(vision_artifacts, capsys):
    """Raised even when both sides agree, because SlowQ's path cannot produce it."""
    for meta in (vision_artifacts.ours_meta, vision_artifacts.ref_meta):
        meta["samples"][0]["image"]["pixel_attention_mask_all_valid"] = False
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "pixel attention mask is not all valid" in text
    assert "the two sides padded differently" in text


def test_a_reference_that_dropped_padding_frames_is_not_comparable(vision_artifacts, capsys):
    """``Idefics3Model.forward`` skips all-zero frames; SlowQ never does.

    If the reference ran fewer frames, the per-frame tensors below are indexed
    against different stacks and every comparison after it is meaningless.
    """
    vision_artifacts.ref_meta["samples"][0]["image"]["dropped_padding_frames"] = 1
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "dropped 1 all-zero frame(s) that SlowQ kept" in text
    assert "the frame indices are misaligned" in text


def test_a_vision_sample_missing_from_one_side_is_named(vision_artifacts, capsys):
    vision_artifacts.ours_meta["samples"] = [
        sample for sample in vision_artifacts.ours_meta["samples"]
        if sample["name"] != "grid-2x4"
    ]
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "vision sample 'grid-2x4' is missing from the SlowQ side" in text
    assert "MISSING from SlowQ" in text


def test_a_capture_config_that_disagrees_about_the_vision_layout_fails(
    vision_artifacts, capsys
):
    """A reference captured from a different image or a different crop grid."""
    vision_artifacts.ref_meta["capture"]["vision"]["grid-2x4"]["num_frames"] = 5
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "capture mismatch: vision layout" in text
    assert "the numeric comparison was skipped" in text


def test_a_prompt_that_differs_inside_the_expansion_blames_the_expansion(
    vision_artifacts, capsys
):
    """The template's head and tail are intact, so the chat template is not at fault."""
    entry = next(
        e for e in vision_artifacts.ours_meta["tokenizer"] if e["name"] == "grid-1x4"
    )
    entry["text"] = entry["text"].replace("<global-img>", "<global_img>", 1)
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "chat template ok, image expansion DIFFERS" in text
    assert "renders different expanded image+text prompt" in text
    assert "template rendering" not in text


def test_a_prompt_that_differs_in_the_template_blames_the_template(vision_artifacts, capsys):
    entry = next(
        e for e in vision_artifacts.ours_meta["tokenizer"] if e["name"] == "grid-1x4"
    )
    entry["text"] = entry["text"].replace("Describe this.", "Describe the image.", 1)
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "chat template DIFFERS" in text
    assert "the difference is in the template rendering rather than in the image expansion" in text


# --------------------------------------------------------------------------- #
# the image merge: the one check a wrong reference cannot rescue
# --------------------------------------------------------------------------- #


def test_an_image_feature_that_missed_its_position_is_named(vision_artifacts, capsys):
    name = "grid-1x4"
    key = f"{name}.full_prefill.embed_sequence"
    broken = vision_artifacts.arrays[key].copy()
    broken[vision_artifacts.capture["vision"][name]["first_image_position"]][0] += 1.0
    vision_artifacts.write_blobs(ours={**vision_artifacts.arrays, key: broken})
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "the fused row at first_image_position=4 is not the connector's first frame output" in text
    assert "max_abs 1.000e+00" in text


def test_a_merge_both_sides_got_wrong_the_same_way_still_fails(vision_artifacts, capsys):
    """The case the reference comparison cannot cover, which is why this check exists.

    Both artifacts carry the same off-by-one — every image row shifted one feature
    along — so ``check_numerics`` compares two identical matrices and passes. The
    merge is an assignment rather than a computation, so SlowQ against itself is
    the only comparison that can see it.
    """
    shifted = {}
    for name, layout in vision_artifacts.capture["vision"].items():
        features = vision_artifacts.arrays[f"{name}.vision.connector_output"]
        sequence = vision_artifacts.arrays[f"{name}.full_prefill.embed_sequence"].copy()
        sequence[layout["first_image_position"]] = features[1]
        sequence[layout["last_image_position"]] = features[-2]
        shifted[f"{name}.full_prefill.embed_sequence"] = sequence
    vision_artifacts.write_blobs(
        ours={**vision_artifacts.arrays, **shifted},
        ref={**vision_artifacts.arrays, **shifted},
    )
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "did not land on the <image> positions in frame order" in text
    assert "grid-1x4" in text and "grid-2x4" in text
    # The numeric half genuinely passed: identical bytes on both sides.
    assert "exceeds the declared" not in text
    assert "first frame  row 4 == feature row 0 max_abs=" in text
    assert "[DIFFERS]" in text


def test_an_image_position_outside_the_fused_sequence_is_reported(vision_artifacts, capsys):
    """An out-of-range index must be named, not raise or silently skip."""
    for meta in (vision_artifacts.ours_meta, vision_artifacts.ref_meta):
        meta["capture"]["vision"]["grid-1x4"]["last_image_position"] = 999
        meta["samples"][0]["last_image_position"] = 999
    vision_artifacts.write_metadata()
    assert vision_artifacts.compare() == 1
    text = failure_text(capsys)
    assert "last_image_position=999 is outside the fused sequence of 28 rows" in text


# --------------------------------------------------------------------------- #
# missing tensors: the intersection bug
# --------------------------------------------------------------------------- #


def test_an_empty_reference_fails_instead_of_comparing_nothing(artifacts, capsys):
    artifacts.write_blobs(ref={})
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "60 expected tensor(s) missing from the reference side" in text
    assert "PASS" not in text


def test_a_tensor_missing_from_the_reference_is_named(artifacts, capsys):
    key = "zh-single-turn.full_prefill.logits"
    artifacts.blobs_without(key)
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "missing from the reference side" in text
    assert key in text
    assert "zh-single-turn" in text


def test_a_tensor_missing_from_the_slowq_side_is_named(artifacts, capsys):
    key = "multi-turn.decode0.final_norm[-1]"
    artifacts.blobs_without(key, side="ours")
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "missing from the SlowQ side" in text
    assert key in text


def test_a_tensor_missing_from_both_sides_still_fails(artifacts, capsys):
    key = "en-with-system.chunked_prefill.logits"
    artifacts.blobs_without(key, side="ours")
    artifacts.blobs_without(key, side="ref")
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "missing from the SlowQ side" in text
    assert "missing from the reference side" in text


def test_leftovers_from_another_capture_are_rejected_not_skipped(artifacts, capsys):
    """The old comparator skipped ``weights.*``; an unexplained key means the two
    files did not come from one run."""
    stale = dict(artifacts.arrays)
    stale["weights.model.norm.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    artifacts.write_blobs(ours=stale)
    assert artifacts.compare() == 1
    assert "unexpected tensor(s) in the SlowQ artifact" in failure_text(capsys)


# --------------------------------------------------------------------------- #
# non-finite values and shapes
# --------------------------------------------------------------------------- #


def test_a_single_nan_fails_the_run(artifacts, capsys):
    key = artifacts.first_key()
    broken = artifacts.arrays[key].copy()
    broken[broken.size // 2] = np.nan
    artifacts.blobs_with("ours", **{key: broken})
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "not finite" in text and key in text


def test_an_inf_on_the_reference_side_fails_the_run(artifacts, capsys):
    key = "zh-single-turn.full_prefill.layer0.layer_output[-1]"
    broken = artifacts.arrays[key].copy()
    broken[0] = np.inf
    artifacts.blobs_with("ref", **{key: broken})
    assert artifacts.compare() == 1
    assert "not finite" in failure_text(capsys)


def test_an_all_nan_tensor_is_not_rescued_by_the_tolerance_comparison(artifacts, capsys):
    """``nan > tolerance`` is False, so the old check graded NaN as acceptable."""
    key = artifacts.first_key()
    artifacts.blobs_with("ours", **{key: np.full_like(artifacts.arrays[key], np.nan)})
    assert artifacts.compare() == 1
    assert "not finite" in failure_text(capsys)


def test_a_raw_shape_difference_is_caught_before_flattening(artifacts):
    """Same element count and the same flattened bytes, different shape.

    The old comparator reshaped both sides to 1-D first, so a ``[4, 8]`` tensor
    and its ``[8, 4]`` reinterpretation compared equal.
    """
    matrix = np.arange(32, dtype=np.float32)
    with pytest.raises(artifacts.tool.ComparisonError) as excinfo:
        artifacts.tool.compare_arrays(matrix.reshape(4, 8), matrix.reshape(8, 4), "matrix")
    assert "shape (4, 8) vs reference shape (8, 4)" in str(excinfo.value)


def test_an_empty_tensor_is_a_failure(artifacts):
    empty = np.zeros(0, dtype=np.float32)
    with pytest.raises(artifacts.tool.ComparisonError, match="empty on both sides"):
        artifacts.tool.compare_arrays(empty, empty, "nothing")


def test_a_dtype_difference_is_reported(artifacts):
    values = np.arange(4, dtype=np.float32)
    with pytest.raises(artifacts.tool.ComparisonError, match="dtype"):
        artifacts.tool.compare_arrays(values, values.astype(np.float64), "mixed")


def test_an_all_zero_reference_is_comparable_only_against_zeros(artifacts):
    zeros = np.zeros(4, dtype=np.float32)
    comparison = artifacts.tool.compare_arrays(zeros, zeros, "zeros")
    assert comparison.rel_l2 == 0.0 and comparison.max_abs == 0.0
    with pytest.raises(artifacts.tool.ComparisonError, match="rel_l2 is undefined"):
        artifacts.tool.compare_arrays(np.ones(4, dtype=np.float32), zeros, "zeros")


# --------------------------------------------------------------------------- #
# tolerances
# --------------------------------------------------------------------------- #


def test_a_difference_beyond_the_declared_tolerance_fails(artifacts, capsys):
    key = "zh-single-turn.full_prefill.logits"
    reference = artifacts.arrays[key]
    # A 5% relative perturbation: far past rel_l2 1e-3, and its max_abs also
    # exceeds 5e-2 for a standard-normal logits vector.
    artifacts.blobs_with("ours", **{key: (reference * 1.05).astype(np.float32)})
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert key in text
    assert "exceeds the declared" in text


def test_the_declared_tolerances_were_not_loosened(artifacts):
    assert artifacts.tool.TOLERANCES == {
        "rel_l2_logits": 1e-3,
        "max_abs_logits": 5e-2,
        "rel_l2_hidden": 1e-3,
        "max_abs_hidden": 5e-2,
        # A unit-norm vector or a cosine: rel_l2 1e-3 means cos >= 1 - 5e-7.
        "rel_l2_vector": 1e-3,
        "max_abs_vector": 1e-3,
        "rel_l2_weights": 0.0,
        "max_abs_weights": 0.0,
    }


def test_a_flipped_greedy_choice_is_reported(artifacts, capsys):
    key = "zh-single-turn.full_prefill.logits"
    swapped = artifacts.arrays[key].copy()
    best = int(np.argmax(swapped))
    second = int(np.argsort(swapped)[-2])
    swapped[best], swapped[second] = swapped[second], swapped[best]
    artifacts.blobs_with("ours", **{key: swapped})
    assert artifacts.compare() == 1
    assert "greedy argmax disagrees" in failure_text(capsys)


def test_an_argmax_is_only_graded_for_a_vocabulary_distribution(artifacts):
    """``argmax_match`` is ``None`` — not ``False`` — for anything but logits.

    ``None`` is what the comparison table prints as ``-`` and what
    ``check_numerics`` declines to grade, so a hidden state or a unit-norm
    embedding cannot be reported as a greedy next-token disagreement.
    """
    one = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    for kind in ("hidden", "vector"):
        assert artifacts.tool.compare_arrays(one, other, "x", kind=kind).argmax_match is None
    assert artifacts.tool.compare_arrays(one, other, "x", kind="logits").argmax_match is False
    assert artifacts.tool.compare_arrays(one, one, "x", kind="logits").argmax_match is True
    # A single element offers no choice to disagree about, logits or not.
    single = np.array([1.0], dtype=np.float32)
    assert artifacts.tool.compare_arrays(single, single, "x", kind="logits").argmax_match is None


# --------------------------------------------------------------------------- #
# metadata: a stale or partial capture must not pass
# --------------------------------------------------------------------------- #


def test_a_missing_reference_metadata_file_fails(artifacts, capsys):
    (artifacts.root / artifacts.tool.REFERENCE_FILENAME).unlink()
    assert artifacts.compare() == 1
    assert "reference.json not found" in failure_text(capsys)


def test_a_missing_blob_file_fails(artifacts, capsys):
    (artifacts.root / "ref.npz").unlink()
    assert artifacts.compare() == 1
    assert "ref.npz not found" in failure_text(capsys)


def test_artifacts_without_a_capture_configuration_are_rejected(artifacts, capsys):
    """What an old capture looks like: no ``capture`` key at all."""
    del artifacts.ours_meta["capture"]
    del artifacts.ref_meta["capture"]
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "capture configuration" in text
    assert "the numeric comparison was skipped" in text


def test_a_capture_config_that_disagrees_between_the_sides_fails(artifacts, capsys):
    artifacts.ref_meta["capture"]["decode_steps"] = DECODE_STEPS + 1
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "capture mismatch: decode_steps" in failure_text(capsys)


def test_a_capture_config_that_disagrees_with_the_code_fails(artifacts, tool, capsys, monkeypatch):
    """Changing which layers are captured invalidates an existing reference."""
    monkeypatch.setattr(tool, "CAPTURED_POSITIONS", (-1,))
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "in this version of the tool" in failure_text(capsys)


def test_an_older_artifact_format_is_rejected(artifacts, capsys):
    artifacts.ref_meta["capture"]["artifact_format"] = 1
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "capture mismatch: artifact_format" in failure_text(capsys)


def test_a_different_architecture_is_rejected(artifacts, capsys):
    artifacts.ref_meta["architecture"]["num_hidden_layers"] = 28
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "different architecture" in failure_text(capsys)


def test_missing_weight_digests_are_a_failure_not_a_skip(artifacts, capsys):
    del artifacts.ref_meta["weight_digests"]
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "records no weight digests" in failure_text(capsys)


def test_missing_tokenizer_content_is_a_failure_not_a_skip(artifacts, capsys):
    del artifacts.ref_meta["tokenizer"]
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "records no tokenizer content" in failure_text(capsys)


# --------------------------------------------------------------------------- #
# metadata coverage: a record *both* sides omitted is still a missing record
# --------------------------------------------------------------------------- #


def drop_tokenizer_sample(meta: dict, name: str) -> None:
    meta["tokenizer"] = [entry for entry in meta["tokenizer"] if entry["name"] != name]


def test_a_complete_artifact_covers_every_sample_and_weight_the_protocol_expects(
    artifacts, tool
):
    """The tests below only mean something if 'complete' is defined by the protocol."""
    expected_samples = set(tool.expected_tokenizer_names(artifacts.capture))
    expected_weights = set(tool.expected_weight_names(artifacts.arch))
    for meta in (artifacts.ours_meta, artifacts.ref_meta):
        assert {entry["name"] for entry in meta["tokenizer"]} == expected_samples
        assert set(meta["weight_digests"]) == expected_weights
    # With no usable capture there is no sample set to require. Inventing one is
    # worse than requiring none: it used to report an embedding capture as
    # missing four *generation* prompts, under names it never used.
    assert tool.expected_tokenizer_names(None) == []


def test_a_complete_embedding_artifact_covers_every_text_of_every_sample(
    embedding_artifacts, tool
):
    expected = {
        f"{name}/{text}"
        for name, texts in embedding_artifacts.capture["embedding_texts"].items()
        for text in texts
    }
    assert expected == {
        "retrieval-with-instruction/query",
        "retrieval-with-instruction/doc0",
        "retrieval-with-instruction/doc1",
        "retrieval-with-instruction/doc2",
        "retrieval-without-instruction/query",
        "retrieval-without-instruction/doc0",
        "retrieval-without-instruction/doc1",
        "retrieval-without-instruction/doc2",
    }
    for meta in (embedding_artifacts.ours_meta, embedding_artifacts.ref_meta):
        assert {entry["name"] for entry in meta["tokenizer"]} == expected
        assert set(meta["weight_digests"]) == set(
            tool.expected_weight_names(embedding_artifacts.arch)
        )


def test_a_tokenizer_sample_missing_from_both_sides_still_fails(artifacts, capsys):
    """The loop used to run over the union of the two files.

    A sample neither side recorded was therefore never named, and the run could
    report PASS having compared three of the four declared samples.
    """
    name = "multi-turn"
    for meta in (artifacts.ours_meta, artifacts.ref_meta):
        drop_tokenizer_sample(meta, name)
    artifacts.write_metadata()

    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert name in text
    assert "missing from both sides" in text
    assert "PASS" not in text


def test_a_weight_digest_missing_from_both_sides_still_fails(artifacts, capsys):
    name = "model.norm.weight"
    for meta in (artifacts.ours_meta, artifacts.ref_meta):
        del meta["weight_digests"][name]
    artifacts.write_metadata()

    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert name in text
    assert "missing from both sides" in text
    assert "PASS" not in text


def test_a_tokenizer_sample_missing_from_one_side_is_named(artifacts, capsys):
    name = "en-with-system"
    drop_tokenizer_sample(artifacts.ours_meta, name)
    artifacts.write_metadata()

    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert name in text
    assert "missing from the SlowQ side" in text


def test_a_weight_digest_missing_from_one_side_is_named(artifacts, capsys):
    name = next(
        weight
        for weight in artifacts.capture["weight_names"]
        if weight.endswith("k_proj.weight")
    )
    del artifacts.ref_meta["weight_digests"][name]
    artifacts.write_metadata()

    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert name in text
    assert "not digested on the reference side" in text


def test_a_sample_the_capture_declares_but_neither_side_recorded_is_named(
    artifacts, tool, capsys, monkeypatch
):
    """The expected set is the capture configuration's, not the files' union."""
    monkeypatch.setattr(
        tool, "GENERATION_SAMPLES", [*tool.GENERATION_SAMPLES, {"name": "extra-sample", "messages": []}]
    )
    for meta in (artifacts.ours_meta, artifacts.ref_meta):
        meta["capture"]["sample_names"] = [*meta["capture"]["sample_names"], "extra-sample"]
    artifacts.write_metadata()

    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "extra-sample" in text
    assert "missing from both sides" in text
    assert "capture mismatch" not in text, "this should fail on coverage, not on the capture"


def test_a_weight_the_protocol_expects_but_neither_side_digested_is_named(
    artifacts, tool, capsys, monkeypatch
):
    """Adding a weight to the protocol invalidates a capture that lacks it."""
    original = tool.weight_check_sample
    monkeypatch.setattr(
        tool,
        "weight_check_sample",
        lambda arch: (*original(arch), "model.layers.1.mlp.up_proj.weight"),
    )
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "model.layers.1.mlp.up_proj.weight" in text
    assert "missing from both sides" in text
    # The recorded capture no longer describes what this version of the tool
    # would select, which is reported too rather than silently compared short.
    assert "capture mismatch: weight_names" in text


def test_the_cli_exits_non_zero_when_both_sides_omit_a_record(artifacts):
    """Through the real entry point, not just ``run_compare`` in-process."""
    drop_tokenizer_sample(artifacts.ours_meta, "zh-single-turn")
    drop_tokenizer_sample(artifacts.ref_meta, "zh-single-turn")
    for meta in (artifacts.ours_meta, artifacts.ref_meta):
        del meta["weight_digests"]["model.norm.weight"]
    artifacts.write_metadata()

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "validate_reference.py"),
            "compare",
            "--artifacts",
            str(artifacts.root),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert completed.returncode == 1
    assert "zh-single-turn" in completed.stderr
    assert "model.norm.weight" in completed.stderr
    assert "PASS" not in completed.stdout


# --------------------------------------------------------------------------- #
# the current run must be re-verified, not inherited
# --------------------------------------------------------------------------- #


def test_a_copied_historical_verdict_does_not_make_a_drifted_tokenizer_pass(
    artifacts, capsys
):
    """Exactly what the integration test used to do: carry the old booleans over.

    The current render differs from the reference's, so the run must fail no
    matter which verdicts are present in ``samples.json``.
    """
    artifacts.ours_meta["tokenizer"][0]["text"] += " "
    artifacts.ours_meta["tokenizer_check"] = [
        {"name": entry["name"], "text_match": True, "ids_match": True}
        for entry in artifacts.ref_meta["tokenizer"]
    ]
    artifacts.ours_meta["weight_check"] = [
        {"name": name, "status": "ok", "bitwise_identical": True}
        for name in artifacts.ref_meta["weight_digests"]
    ]
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "renders different chat text" in text
    assert artifacts.ours_meta["samples"][0]["name"] in text


def test_a_changed_token_id_fails(artifacts, capsys):
    artifacts.ours_meta["tokenizer"][1]["ids"] = [
        token + 1 for token in artifacts.ours_meta["tokenizer"][1]["ids"]
    ]
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "different token IDs" in failure_text(capsys)


def test_a_mutated_converted_weight_fails(artifacts, capsys):
    """One element of one tensor, changed after the digest was taken."""
    name = "model.norm.weight"
    mutated = artifacts.tool.tensor_digest(np.zeros(HIDDEN, dtype=np.float32))
    artifacts.ours_meta["weight_digests"][name] = mutated
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert name in text
    assert "the conversion is not exact" in text


def test_a_weight_digest_that_does_not_state_its_spec_fails(artifacts, capsys):
    artifacts.ours_meta["weight_digests"]["model.norm.weight"]["spec"] = "some-other-digest"
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "digested with spec" in failure_text(capsys)


def test_a_non_finite_converted_weight_fails(artifacts, capsys):
    digest = artifacts.ours_meta["weight_digests"]["model.norm.weight"]
    digest["non_finite_elements"] = 3
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "3 non-finite value(s) on the SlowQ side" in failure_text(capsys)


def test_the_digest_record_states_the_byte_representation(artifacts):
    digest = artifacts.ours_meta["weight_digests"]["model.embed_tokens.weight"]
    assert digest["dtype"] == "<f4"
    assert digest["byte_order"] == "little"
    assert digest["shape"] == [VOCAB, HIDDEN]
    assert digest["elements"] == VOCAB * HIDDEN
    assert digest["spec"] == artifacts.tool.DIGEST_SPEC_VERSION
    spec = artifacts.tool.WEIGHT_DIGEST_SPEC
    for required in ("algorithm", "preimage", "dtype", "byte_order", "coverage"):
        assert spec[required]
    assert "whole tensor" in spec["coverage"]


def test_the_digest_covers_every_element_not_a_stride(artifacts):
    """Flipping the last element of a large tensor must change the digest."""
    size = 4096
    base = np.zeros(size, dtype=np.float32)
    changed = base.copy()
    changed[-1] = 1.0
    tool = artifacts.tool
    assert tool.tensor_digest(base)["sha256"] != tool.tensor_digest(changed)["sha256"]
    # ... and a transposed view of the same numbers is a different tensor.
    matrix = np.arange(16, dtype=np.float32)
    assert tool.tensor_digest(matrix.reshape(4, 4))["sha256"] != (
        tool.tensor_digest(matrix.reshape(4, 4).T)["sha256"]
    )


def test_a_drifted_decode_path_fails(artifacts, capsys):
    artifacts.ref_meta["tokenizer"][0]["decode_input_tokens"] = [999]
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "replayed decode tokens" in failure_text(capsys)


def test_weights_that_are_not_the_reference_bytes_fail(artifacts, capsys):
    """The provenance tie: the manifest's claim vs what the reference hashed."""
    artifacts.ours_meta["source"]["files"][SHARD] = "0" * 64
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "did not come from these weights" in failure_text(capsys)


def test_a_shard_the_reference_never_read_fails(artifacts, capsys):
    artifacts.ours_meta["source"]["files"]["model-00002-of-00002.safetensors"] = "1" * 64
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "not looking at one checkpoint" in failure_text(capsys)


def test_a_revision_the_reference_recorded_must_match(artifacts, capsys):
    artifacts.ref_meta["source"]["revision"] = "d" * 40
    artifacts.write_metadata()
    assert artifacts.compare() == 1
    assert "re-capture the reference" in failure_text(capsys)


def test_a_reference_without_a_revision_still_checks_the_file_digests(artifacts, capsys):
    assert artifacts.ref_meta["source"]["revision"] is None
    assert artifacts.compare() == 0
    assert "not recorded by the reference" in capsys.readouterr().out
