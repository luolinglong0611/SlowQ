"""The reference regression, for whichever converted directory is named.

    SLOWQ_MODEL_DIR=models/Qwen3-Embedding-0.6B-slowq python -m pytest -m integration

Unlike the per-task modules this one has no task filter: ``numpy`` and
``compare`` dispatch on the manifest themselves, so the same seven tests grade a
chat model, an embedding model and an image+text model against the official
capture that was recorded for it. What they assert is that the *current* code
still reproduces the numbers that were signed off, and that the comparison
cannot be fooled.

The reference capture needs torch and the official checkpoint, so it is made once
by ``tools/validate_reference.py ref`` and stored under
``artifacts/validation/<model-name>/``. Everything here regenerates only the
SlowQ half, which is why the torch half never runs inside pytest and these tests
stay usable where torch is not installed.

Nothing is carried forward from the recorded run: the tokenizer text, the token
IDs and the weight digests compared here are the ones *this* run produced, and
neither artifact file contains a verdict to inherit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import real_models

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        (reason := real_models.skip_reason_for_any_task()) is not None,
        reason=reason or "",
    ),
]


@pytest.fixture(scope="module")
def model_dir() -> Path:
    directory = real_models.selected_model_dir()
    assert directory is not None  # guaranteed by the module-level skip
    return directory


@pytest.fixture(scope="module")
def fresh_numpy_run(model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the NumPy half once, against the real model, next to the recorded reference.

    Module-scoped because this is the slow part (every sample through the real
    FP32 model at the reference's own chunk size). The tests below copy the
    artifacts and break one thing each, which is cheap: without the weight blobs
    the two NPZ files are a few MiB.
    """
    return real_models.fresh_numpy_run(model_dir, tmp_path_factory.mktemp("validation"))


def test_the_current_code_still_matches_the_recorded_reference(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    directory = real_models.copied_artifacts(fresh_numpy_run, tmp_path)
    tool = real_models.load_validation_tool()
    expected = real_models.expected_numeric_count(directory)

    assert real_models.compare(directory) == 0, (
        "SlowQ no longer matches the Transformers reference within the declared "
        f"tolerances: {tool.TOLERANCES}"
    )
    assert f"PASS: {expected} tensors compared" in capsys.readouterr().out


def test_this_run_records_its_own_evidence_and_inherits_no_verdicts(
    fresh_numpy_run: Path, model_dir: Path
):
    """The artifact a fresh run writes must contain content, not conclusions."""
    tool = real_models.load_validation_tool()
    payload = real_models.read_json(fresh_numpy_run / "samples.json")
    reference = real_models.read_json(fresh_numpy_run / "reference.json")

    # The fields the old test copied forward do not exist any more, on either side.
    for side in (payload, reference):
        assert "tokenizer_check" not in side
        assert "weight_check" not in side

    manifest = real_models.read_json(model_dir / "manifest.json")
    assert payload["source"]["revision"] == manifest["source"]["revision"]
    assert payload["architecture"]["task"] == real_models.recorded_task(model_dir)
    assert payload["architecture"]["num_hidden_layers"] >= 1
    assert payload["architecture"]["vocab_size"] >= 1
    assert payload["capture"]["artifact_format"] == tool.ARTIFACT_FORMAT
    assert payload["capture"] == reference["capture"]
    assert payload["architecture"] == reference["architecture"]
    # Which weights get digested is derived from this checkpoint's own layer
    # count, family and task, so a 30-layer Llama samples different tensors than
    # a 28-layer Qwen3 and a vision model adds both towers; neither is compared
    # against a hard-coded list.
    assert set(payload["weight_digests"]) == set(payload["capture"]["weight_names"])
    assert set(reference["weight_digests"]) == set(payload["weight_digests"])
    assert {entry["name"] for entry in payload["tokenizer"]} == set(
        tool.expected_tokenizer_names(payload["capture"])
    )
    # Every digest states the byte representation it was taken over.
    for digest in payload["weight_digests"].values():
        assert digest["dtype"] == "<f4"
        assert digest["byte_order"] == "little"
        assert digest["spec"] == tool.DIGEST_SPEC_VERSION
        assert digest["non_finite_elements"] == 0
    # The reference side hashed the checkpoint it loaded; the manifest claims the
    # same bytes, which is what ties the two captures to one model.
    assert set(reference["source"]["files"]) <= set(payload["source"]["files"])
    assert reference["source"]["files"]


def test_a_drifted_tokenizer_output_fails_against_the_recorded_reference(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    """One token ID changed on the SlowQ side is a failed validation.

    This is the case the copied ``tokenizer_check`` booleans used to hide: the
    reference's own IDs are stored as data, so the current output is compared
    against them every run.
    """
    directory = real_models.copied_artifacts(fresh_numpy_run, tmp_path)

    def drift(payload: dict) -> None:
        payload["tokenizer"][0]["ids"] = [
            token + 1 for token in payload["tokenizer"][0]["ids"]
        ]

    real_models.rewrite(directory, drift)
    assert real_models.compare(directory) == 1
    assert "different token IDs" in capsys.readouterr().err


def test_a_drifted_render_fails_against_the_recorded_reference(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    """The prompt text is compared too, not just the IDs it tokenised to."""
    directory = real_models.copied_artifacts(fresh_numpy_run, tmp_path)

    def drift(payload: dict) -> None:
        payload["tokenizer"][0]["text"] += " "

    real_models.rewrite(directory, drift)
    assert real_models.compare(directory) == 1
    # What was rendered differs by task — chat text, an expanded image+text
    # prompt, or a bare embedding input — so the assertion stops at the part
    # every task shares.
    assert "renders different" in capsys.readouterr().err


def test_a_drifted_converted_weight_fails_against_the_recorded_reference(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    """The weights are compared by digest, so a changed tensor cannot slip through."""
    directory = real_models.copied_artifacts(fresh_numpy_run, tmp_path)
    tool = real_models.load_validation_tool()
    # Whichever tensor both sides digested: the name differs per family and task
    # (a Llama has no q_norm, a vision model digests both towers) so it cannot be
    # hard-coded.
    name = real_models.digest_names_present_on_both_sides(directory)[0]

    def drift(payload: dict) -> None:
        payload["weight_digests"][name] = tool.tensor_digest(
            np.zeros(payload["weight_digests"][name]["shape"], dtype=np.float32)
        )

    real_models.rewrite(directory, drift)
    assert real_models.compare(directory) == 1
    assert "the conversion is not exact" in capsys.readouterr().err


def test_a_dropped_tensor_fails_against_the_recorded_reference(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    """A capture that quietly compares fewer tensors is not a pass."""
    directory = real_models.copied_artifacts(fresh_numpy_run, tmp_path)
    blobs = dict(np.load(directory / "slowq.npz"))
    dropped = sorted(blobs)[-1]
    del blobs[dropped]
    np.savez_compressed(directory / "slowq.npz", **blobs)
    assert real_models.compare(directory) == 1
    err = capsys.readouterr().err
    assert "missing from the SlowQ side" in err
    assert dropped in err


def test_a_reference_capture_that_is_incomplete_prompts_a_recapture(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    """Old or partial artifacts must ask for a new capture, never pass."""
    directory = real_models.copied_artifacts(fresh_numpy_run, tmp_path)
    (directory / "reference.json").unlink()
    assert real_models.compare(directory) == 1
    assert "re-capture" in capsys.readouterr().err


def test_the_cli_reports_the_directory_offline(model_dir: Path, monkeypatch, capsys):
    """``slowq inspect``, the one command all three tasks share.

    Lives here rather than in a per-task module so that every converted directory
    gets an offline CLI check, including the ones whose task-specific command is
    exercised elsewhere. ``inspect`` dispatches on the recorded task, so this is
    also the check that the dispatch itself works on a real directory of each
    kind.
    """
    import json

    from slowq import __version__
    from slowq import cli

    real_models.block_network(monkeypatch)
    argv = ["inspect", "--model", str(model_dir), "--json", "--max-context", "128"]
    assert cli.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)

    manifest = real_models.read_json(model_dir / "manifest.json")
    assert payload["model_directory"] == str(model_dir)
    assert payload["slowq_version"] == __version__
    assert payload["manifest_format_version"] == manifest["format_version"]
    # The provenance the CLI reports is the provenance the converter recorded:
    # repo, revision and the digests of the files it read.
    assert payload["source"] == manifest["source"]
    assert payload["source"]["revision"]
    assert payload["model_config"]
    assert payload["tokenizer"]
    assert payload["weights"]["tensor count"] == len(manifest["tensors"])
    assert payload["environment"]
