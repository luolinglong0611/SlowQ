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

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A toy capture: two layers means only layer 0 of ``CAPTURED_LAYERS`` survives,
#: which keeps the artifact small while still exercising per-layer key naming.
NUM_LAYERS = 2
CHUNK = 4
DECODE_STEPS = 1
VOCAB = 32
HIDDEN = 8
SHARD = "model.safetensors"
SHARD_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


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

    def __init__(self, tool, root: Path) -> None:
        self.tool = tool
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.capture = tool.capture_config(
            NUM_LAYERS, prefill_chunk_size=CHUNK, decode_steps=DECODE_STEPS
        )
        self.arrays = self._make_arrays()
        self.ours_meta = self._make_ours_meta()
        self.ref_meta = self._make_ref_meta()
        self.write_metadata()
        self.write_blobs()

    # ---- construction -------------------------------------------------------

    def _make_arrays(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(20261019)
        keys = self.tool.expected_numeric_keys(self.capture)
        arrays = {
            key: rng.normal(size=(VOCAB if kind == "logits" else HIDDEN,)).astype(np.float32)
            for key, (_, kind) in keys.items()
        }
        # The chunked and decode logits share the full prefill's argmax, which is
        # what the recorded sample claims; making them equal keeps the internal
        # full-vs-chunked check meaningful instead of incidentally failing.
        for name in self.capture["sample_names"]:
            full = arrays[f"{name}.full_prefill.logits"]
            for stage in ("chunked_prefill", *(f"decode{step}" for step in range(DECODE_STEPS))):
                arrays[f"{name}.{stage}.logits"] = full.copy()
        return arrays

    def _architecture(self) -> dict[str, object]:
        return {
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
        }

    def _samples(self) -> list[dict[str, object]]:
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

    def _weight_arrays(self) -> dict[str, np.ndarray]:
        """One array per weight the protocol expects, so 'complete' means complete.

        ``check_weights`` derives its expected set from ``WEIGHT_CHECK_SAMPLE``,
        not from what the two files happen to contain, so a self-consistent toy
        artifact has to cover the same names the real capture does.
        """
        rng = np.random.default_rng(11)
        arrays: dict[str, np.ndarray] = {}
        for name in self.tool.WEIGHT_CHECK_SAMPLE:
            if "embed_tokens" in name:
                shape: tuple[int, ...] = (VOCAB, HIDDEN)
            elif name.endswith("norm.weight"):
                shape = (HIDDEN,)
            else:
                shape = (HIDDEN, HIDDEN)
            arrays[name] = rng.normal(size=shape).astype(np.float32)
        return arrays

    def _make_ours_meta(self) -> dict:
        samples = self._samples()
        return {
            "samples": samples,
            "model_directory": "tiny:converted",
            "captured_at": "2026-10-19T00:00:00+0800",
            "side": "numpy",
            "architecture": self._architecture(),
            "source": {
                "repo_id": "tiny:local",
                "revision": REVISION,
                # A non-shard entry with no digest must not upset the provenance
                # check: the reference only hashes *.safetensors.
                "files": {SHARD: SHARD_SHA256, "config.json": None},
            },
            "capture": self.capture,
            "tokenizer": [
                {
                    "name": sample["name"],
                    "text": sample["rendered_text"],
                    "ids": list(sample["prompt_ids"]),
                }
                for sample in samples
            ],
            "weight_digests": {
                name: self.tool.tensor_digest(array)
                for name, array in self._weight_arrays().items()
            },
        }

    def _make_ref_meta(self) -> dict:
        samples = self.ours_meta["samples"]
        return {
            "side": "reference",
            "captured_at": "2026-10-19T00:00:00+0800",
            "hf_dir": "tiny:hf",
            "architecture": dict(self.ours_meta["architecture"]),
            "source": {"repo_id": None, "revision": None, "files": {SHARD: SHARD_SHA256}},
            "capture": json.loads(json.dumps(self.capture)),
            "environment": {
                "transformers": "0.0.test",
                "torch": "0.0.test",
                "attn_implementation": "eager",
                "dtype": "float32",
                "threads": 1,
            },
            "tokenizer": [
                {
                    "name": sample["name"],
                    "text": sample["rendered_text"],
                    "ids": list(sample["prompt_ids"]),
                    "decode_input_tokens": [
                        step["input_token"] for step in sample["decode_steps"]
                    ],
                }
                for sample in samples
            ],
            "weight_digests": json.loads(json.dumps(self.ours_meta["weight_digests"])),
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


def failure_text(capsys) -> str:
    captured = capsys.readouterr()
    return captured.err + captured.out


# --------------------------------------------------------------------------- #
# the happy path, which every test below breaks in exactly one way
# --------------------------------------------------------------------------- #


def test_a_consistent_capture_passes_and_says_how_much_it_compared(artifacts, capsys):
    expected = len(artifacts.tool.expected_numeric_keys(artifacts.capture))
    assert expected == 44  # 4 samples x (5 full + 3 chunked + 3 decode)
    assert artifacts.compare() == 0
    assert f"PASS: {expected} tensors compared" in capsys.readouterr().out


def test_the_expected_key_set_is_derived_from_the_capture_configuration(tool):
    capture = tool.capture_config(28, prefill_chunk_size=8, decode_steps=3)
    keys = tool.expected_numeric_keys(capture)
    assert len(keys) == 116
    assert "zh-single-turn.full_prefill.layer27.layer_output[-1]" in keys
    assert "zh-single-turn.full_prefill.final_norm[0]" in keys
    assert "emoji-and-mixed-script.decode2.logits" in keys
    # A cached stage has no position 0: only the last token is captured there.
    assert "zh-single-turn.decode0.final_norm[0]" not in keys
    assert all(kind in ("logits", "hidden") for _, kind in keys.values())


# --------------------------------------------------------------------------- #
# missing tensors: the intersection bug
# --------------------------------------------------------------------------- #


def test_an_empty_reference_fails_instead_of_comparing_nothing(artifacts, capsys):
    artifacts.write_blobs(ref={})
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "44 expected tensor(s) missing from the reference side" in text
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
    expected_samples = set(tool.expected_sample_names(artifacts.capture))
    for meta in (artifacts.ours_meta, artifacts.ref_meta):
        assert {entry["name"] for entry in meta["tokenizer"]} == expected_samples
        assert set(meta["weight_digests"]) == set(tool.expected_weight_names())
    # A run that already failed the capture check still says which samples are
    # missing, instead of comparing the nothing that is left.
    assert tool.expected_sample_names(None) == [
        sample["name"] for sample in tool.SAMPLES
    ]


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
    name = "model.layers.13.self_attn.k_proj.weight"
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
        tool, "SAMPLES", [*tool.SAMPLES, {"name": "extra-sample", "messages": []}]
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
    monkeypatch.setattr(
        tool,
        "WEIGHT_CHECK_SAMPLE",
        (*tool.WEIGHT_CHECK_SAMPLE, "model.layers.1.mlp.up_proj.weight"),
    )
    assert artifacts.compare() == 1
    text = failure_text(capsys)
    assert "model.layers.1.mlp.up_proj.weight" in text
    assert "missing from both sides" in text


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
