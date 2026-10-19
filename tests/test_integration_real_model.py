"""Real Qwen3-0.6B integration tests. Skipped unless explicitly enabled.

Enable them with::

    SLOWQ_MODEL_DIR=models/Qwen3-0.6B-slowq python -m pytest -m integration

They need a converted checkpoint (``tools/convert_model.py``) and, for the
reference comparison, a capture made beforehand by ``tools/validate_reference.py
ref`` — both ``artifacts/validation/ref.npz`` (the numbers) and
``artifacts/validation/reference.json`` (the reference tokenizer's text and IDs,
weight digests, capture configuration). The torch half never runs inside pytest,
so these tests stay usable in an environment where torch is not installed.

Everything here is deliberately *not* about whether the answer reads well. The
spec is explicit that "it produces Chinese" is not evidence of correctness, so
the assertions are about invariants that a numerical bug would break:
determinism, chunked-vs-full prefill agreement, cached-vs-uncached agreement,
explicit refusal on overflow, and agreement with the Transformers reference
inside tolerances that were declared before they were measured.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from slowq.config import QWEN3_06B_REFERENCE
from slowq.engine import Engine
from slowq.errors import ContextOverflowError
from slowq.sampling import SamplingParams

MODEL_DIR_ENV = "SLOWQ_MODEL_DIR"
REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDED_ARTIFACTS = REPO_ROOT / "artifacts" / "validation"

#: Declared before measuring. Rationale: chunking and caching only change FP32
#: reduction order *within one implementation*, so the gate has to be tighter than
#: the cross-implementation one in tools/validate_reference.py (rel_l2 1e-3,
#: max_abs 5e-2 for the same quantities), not looser.
#:
#: Measured afterwards on Qwen3-0.6B (21-token prompt, chunk sizes 1/3/7/16, plus
#: cached decode against one long prefill): worst logits rel_l2 1.64e-06, worst
#: hidden rel_l2 2.28e-06, worst max_abs 6.58e-05 — i.e. 4x inside the rel_l2
#: gate and 15x inside the max_abs gate. Both gates passed as declared; neither
#: was widened to make a failure go away.
INTERNAL_REL_L2_TOLERANCE = 1e-5
INTERNAL_MAX_ABS_TOLERANCE = 1e-3

MAX_NEW_TOKENS = 48

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get(MODEL_DIR_ENV),
        reason=f"set {MODEL_DIR_ENV} to a converted model directory to run these",
    ),
]


def load_validation_tool():
    """Import ``tools/validate_reference.py`` without making ``tools`` a package."""
    path = REPO_ROOT / "tools" / "validate_reference.py"
    spec = importlib.util.spec_from_file_location("slowq_validate_reference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registering before execution is not optional: the tool's dataclasses use
    # ``from __future__ import annotations``, and dataclasses resolves those
    # strings through ``sys.modules[cls.__module__]``.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model_dir() -> Path:
    return Path(os.environ[MODEL_DIR_ENV]).expanduser()


@pytest.fixture(scope="module")
def engine(model_dir: Path) -> Engine:
    """One engine for the whole module.

    Sharing it is safe because the engine resets the KV cache at the *start* of
    every request, and loading 2.4 GiB of weights once instead of per test is
    the difference between a usable and an unbearable test run.
    """
    return Engine.from_directory(model_dir, max_context=512, prefill_chunk_size=64)


def assert_same_numerics(
    ours: np.ndarray, theirs: np.ndarray, what: str, *, check_argmax: bool = True
) -> None:
    """Assert two arrays agree to within the declared internal tolerances.

    Chunking and caching change FP32 reduction order and nothing else, so the
    greedy choice must survive too — a difference large enough to flip an argmax
    is a bug, not noise.

    Both a relative and an absolute gate are needed. A pure relative criterion is
    meaningless for a hidden-state vector, whose components span 0.0008 to 35:
    the small ones carry a large relative error while contributing nothing to the
    result. A pure absolute criterion would in turn let a whole vector drift.
    """
    ours = np.asarray(ours, dtype=np.float64).reshape(-1)
    theirs = np.asarray(theirs, dtype=np.float64).reshape(-1)
    assert ours.shape == theirs.shape, f"{what}: {ours.shape} vs {theirs.shape}"

    difference = np.linalg.norm(ours - theirs)
    denominator = np.linalg.norm(theirs)
    rel_l2 = float(difference / denominator) if denominator else 0.0
    max_abs = float(np.max(np.abs(ours - theirs))) if ours.size else 0.0

    assert rel_l2 < INTERNAL_REL_L2_TOLERANCE, (
        f"{what}: rel_l2 {rel_l2:.3e} exceeds the declared "
        f"{INTERNAL_REL_L2_TOLERANCE:.3e} (max_abs {max_abs:.3e})"
    )
    assert max_abs < INTERNAL_MAX_ABS_TOLERANCE, (
        f"{what}: max_abs {max_abs:.3e} exceeds the declared "
        f"{INTERNAL_MAX_ABS_TOLERANCE:.3e} (rel_l2 {rel_l2:.3e})"
    )
    if check_argmax:
        assert int(np.argmax(ours)) == int(np.argmax(theirs)), (
            f"{what}: the greedy next token differs "
            f"({int(np.argmax(ours))} vs {int(np.argmax(theirs))})"
        )


# --------------------------------------------------------------------------- #
# the converted checkpoint
# --------------------------------------------------------------------------- #


def test_the_checkpoint_is_the_published_qwen3_06b_architecture(engine: Engine):
    for field, expected in QWEN3_06B_REFERENCE.items():
        assert getattr(engine.config, field) == expected, field


def test_the_weights_are_complete_and_the_tied_head_is_stored_once(engine: Engine):
    weights = engine.model.weights
    weights.check_complete(engine.config)
    assert weights.aliases == {"lm_head.weight": "model.embed_tokens.weight"}
    assert set(weights.dtype_names) == {"float32"}
    # 0.6B parameters at 4 bytes each, minus the tied lm_head that is not stored
    # twice: ~2.4 GiB, and the embedding table is ~0.6 GiB of that.
    assert 2_300_000_000 < weights.deduplicated_bytes() < 2_500_000_000


def test_the_kv_cache_is_sized_from_the_real_config(engine: Engine):
    per_token = engine.config.kv_bytes_per_token()
    assert per_token == 2 * 28 * 8 * 128 * 4
    assert engine.kv_cache.allocated_bytes() == per_token * engine.max_context


def test_loading_never_reaches_the_network(model_dir: Path, monkeypatch):
    """``from_directory`` must work offline; a socket use would be a bug."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("the runtime opened a network connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    assert Engine.from_directory(model_dir, max_context=64) is not None


# --------------------------------------------------------------------------- #
# generation on the real model
# --------------------------------------------------------------------------- #


def test_greedy_generation_is_deterministic(engine: Engine):
    messages = [{"role": "user", "content": "用一句话解释什么是 KV cache。"}]
    first = engine.generate(messages, max_new_tokens=MAX_NEW_TOKENS)
    second = engine.generate(messages, max_new_tokens=MAX_NEW_TOKENS)
    assert first.token_ids == second.token_ids
    assert first.text == second.text


def test_a_fixed_seed_reproduces_a_sampled_answer(engine: Engine):
    messages = [{"role": "user", "content": "Give me three random colour names."}]
    sampling = SamplingParams(temperature=0.8, top_k=20, top_p=0.9, seed=20261018)
    first = engine.generate(messages, sampling=sampling, max_new_tokens=MAX_NEW_TOKENS)
    second = engine.generate(messages, sampling=sampling, max_new_tokens=MAX_NEW_TOKENS)
    assert first.token_ids == second.token_ids


def test_an_answer_is_complete_text_not_a_stream_of_replacement_characters(engine: Engine):
    """Not a quality check — a check that incremental decoding is lossless.

    A model that produced fluent Chinese with a broken byte-level decoder would
    still fail here, and a correct decoder that produced nonsense would pass.
    """
    for messages in (
        [{"role": "user", "content": "用一个 emoji 表示开心 🎉，然后解释它的 Unicode 名称。"}],
        [{"role": "user", "content": "Reply in English: what does the turtle emoji mean?"}],
    ):
        result = engine.generate(messages, max_new_tokens=MAX_NEW_TOKENS)
        assert result.text.strip(), "the model produced no visible text"
        assert "\ufffd" not in result.text
        # The stop token terminates generation but must never be rendered.
        if result.stopped_on_eos:
            assert result.token_ids[-1] in engine.stop_ids
            marker = engine.tokenizer.id_to_token(result.token_ids[-1])
            assert marker not in result.text


def test_streaming_yields_the_same_text_as_the_blocking_call(engine: Engine):
    messages = [{"role": "user", "content": "What is 12 * 8? Reply with just the number."}]
    steps = list(engine.stream(messages, max_new_tokens=MAX_NEW_TOKENS))
    assert steps[-1].finished and steps[-1].result is not None
    assert "".join(step.text_delta for step in steps) == steps[-1].result.text
    assert [step.index for step in steps] == list(range(len(steps)))
    assert steps[-1].result.text == engine.generate(
        messages, max_new_tokens=MAX_NEW_TOKENS
    ).text


def test_a_multi_turn_conversation_grows_the_prompt(engine: Engine):
    history = [{"role": "user", "content": "My favourite number is 7. Remember it."}]
    first = engine.generate(history, max_new_tokens=16)
    history += [
        {"role": "assistant", "content": first.text},
        {"role": "user", "content": "What is it?"},
    ]
    second = engine.generate(history, max_new_tokens=16)
    assert len(second.prompt_token_ids) > len(first.prompt_token_ids)
    # The previous answer must actually be in the second prompt, not dropped.
    rendered = engine.tokenizer.decode(list(second.prompt_token_ids))
    assert first.text in rendered


def test_a_request_that_does_not_fit_is_refused_not_truncated(model_dir: Path):
    small = Engine.from_directory(model_dir, max_context=32)
    long_prompt = [{"role": "user", "content": "请把下面这句话重复很多遍：" + "慢炖" * 40}]
    with pytest.raises(ContextOverflowError, match="never truncates"):
        small.generate(long_prompt, max_new_tokens=8)


# --------------------------------------------------------------------------- #
# numerical invariants on the real model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk", [1, 3, 7, 16])
def test_chunked_prefill_matches_a_single_prefill(engine: Engine, chunk: int):
    """Odd chunk sizes matter: the last chunk is then shorter than the others."""
    model = engine.model
    prompt_ids = engine.encode_messages(
        [{"role": "user", "content": "Explain grouped-query attention in one sentence."}]
    )
    assert len(prompt_ids) > chunk, "the test prompt must span several chunks"

    one_shot = model.new_cache(len(prompt_ids) + 4)
    expected = model.forward(prompt_ids, one_shot)

    chunked = model.new_cache(len(prompt_ids) + 4)
    for start in range(0, len(prompt_ids), chunk):
        result = model.forward(prompt_ids[start : start + chunk], chunked)

    assert chunked.length == one_shot.length == len(prompt_ids)
    assert_same_numerics(result.logits, expected.logits, f"chunk={chunk} logits")
    assert_same_numerics(
        result.hidden_states[-1],
        expected.hidden_states[-1],
        f"chunk={chunk} final hidden state",
        check_argmax=False,
    )


def test_cached_decode_matches_one_long_prefill(engine: Engine):
    """The KV cache must be exactly equivalent to recomputing from scratch."""
    model = engine.model
    prompt_ids = engine.encode_messages([{"role": "user", "content": "什么是注意力机制？"}])

    cached = model.new_cache(len(prompt_ids) + 8)
    logits = model.forward(prompt_ids, cached).logits
    generated: list[int] = []
    for _ in range(3):
        token_id = int(np.argmax(logits))
        generated.append(token_id)
        logits = model.forward([token_id], cached).logits

    recomputed = model.new_cache(len(prompt_ids) + 8)
    expected = model.forward([*prompt_ids, *generated], recomputed).logits

    assert_same_numerics(logits, expected, "cached decode vs one long prefill")


def test_repeating_a_request_reuses_a_reset_cache(engine: Engine):
    messages = [{"role": "user", "content": "Say hello."}]
    first = engine.generate(messages, max_new_tokens=8)
    assert engine.kv_cache.high_water_mark > 0
    second = engine.generate(messages, max_new_tokens=8)
    assert second.token_ids == first.token_ids
    assert engine.kv_cache.length == len(first.prompt_token_ids) + len(first.token_ids) - 1


# --------------------------------------------------------------------------- #
# against the Transformers reference
# --------------------------------------------------------------------------- #

#: The reference half writes both files. ``ref.npz`` holds the numbers and
#: ``reference.json`` holds the *content* the numbers are graded against: the
#: reference tokenizer's rendered text and token IDs, sha256 digests of whole
#: weight tensors, the digests of the checkpoint files it read, and the capture
#: configuration. Neither file contains a verdict, so there is nothing to copy
#: forward from an older run.
RECORDED_REFERENCE_FILES = ("ref.npz", "reference.json")


def recorded_reference_is_complete() -> bool:
    return all((RECORDED_ARTIFACTS / name).is_file() for name in RECORDED_REFERENCE_FILES)


@pytest.fixture(scope="module")
def fresh_numpy_run(model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the NumPy half once, against the real model, next to the recorded reference.

    Module-scoped because this is the slow part (four samples through a 0.6B FP32
    model with a chunk size of 8). The tests below copy the artifacts and break
    one thing each, which is cheap: without the weight blobs ``slowq.npz`` is a
    few MiB.
    """
    if not recorded_reference_is_complete():
        pytest.skip(
            f"no complete reference capture in {RECORDED_ARTIFACTS} (need "
            f"{', '.join(RECORDED_REFERENCE_FILES)}); run "
            "'python tools/validate_reference.py ref' first"
        )
    work = tmp_path_factory.mktemp("validation")
    for name in RECORDED_REFERENCE_FILES:
        shutil.copyfile(RECORDED_ARTIFACTS / name, work / name)
    tool = load_validation_tool()
    argv = [
        "numpy",
        "--model",
        str(model_dir),
        "--artifacts",
        str(work),
        "--max-context",
        "512",
        "--prefill-chunk-size",
        "8",
        "--decode-steps",
        "3",
    ]
    assert tool.run_numpy_side(tool.build_parser().parse_args(argv)) == 0
    return work


def copied_artifacts(source: Path, destination: Path) -> Path:
    for name in ("samples.json", "slowq.npz", *RECORDED_REFERENCE_FILES):
        shutil.copyfile(source / name, destination / name)
    return destination


def compare(directory: Path) -> int:
    tool = load_validation_tool()
    return tool.run_compare(
        tool.build_parser().parse_args(["compare", "--artifacts", str(directory)])
    )


def rewrite(directory: Path, mutate) -> dict:
    path = directory / "samples.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def test_the_current_code_still_matches_the_recorded_reference(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    """Re-run the NumPy side and compare it against the captured torch tensors.

    The reference capture needs torch and 2.4 GiB of FP32 weights, so it is done
    once by ``tools/validate_reference.py ref`` and stored. This test regenerates
    only the SlowQ half, which means it fails if the runtime drifts away from the
    numbers that were signed off. Nothing is carried over from the recorded run:
    the tokenizer text, the token IDs and the weight digests compared here are
    the ones *this* run produced.
    """
    directory = copied_artifacts(fresh_numpy_run, tmp_path)
    tool = load_validation_tool()
    expected = len(tool.expected_numeric_keys(json.loads(
        (directory / "samples.json").read_text(encoding="utf-8")
    )["capture"]))

    assert compare(directory) == 0, (
        "SlowQ no longer matches the Transformers reference within the declared "
        f"tolerances: {tool.TOLERANCES}"
    )
    assert f"PASS: {expected} tensors compared" in capsys.readouterr().out


def test_this_run_records_its_own_evidence_and_inherits_no_verdicts(
    fresh_numpy_run: Path, model_dir: Path, engine: Engine
):
    """The artifact a fresh run writes must contain content, not conclusions."""
    tool = load_validation_tool()
    payload = json.loads((fresh_numpy_run / "samples.json").read_text(encoding="utf-8"))
    reference = json.loads((fresh_numpy_run / "reference.json").read_text(encoding="utf-8"))

    # The fields the old test copied forward do not exist any more, on either side.
    for side in (payload, reference):
        assert "tokenizer_check" not in side
        assert "weight_check" not in side

    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["source"]["revision"] == manifest["source"]["revision"]
    assert payload["architecture"]["num_hidden_layers"] == engine.config.num_hidden_layers
    assert payload["architecture"]["vocab_size"] == engine.config.vocab_size
    assert payload["capture"]["artifact_format"] == tool.ARTIFACT_FORMAT
    assert payload["capture"] == reference["capture"]
    assert payload["architecture"] == reference["architecture"]
    assert set(payload["weight_digests"]) == set(tool.WEIGHT_CHECK_SAMPLE)
    assert {entry["name"] for entry in payload["tokenizer"]} == set(
        payload["capture"]["sample_names"]
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
    directory = copied_artifacts(fresh_numpy_run, tmp_path)

    def drift(payload: dict) -> None:
        payload["tokenizer"][0]["ids"] = [
            token + 1 for token in payload["tokenizer"][0]["ids"]
        ]

    rewrite(directory, drift)
    assert compare(directory) == 1
    assert "different token IDs" in capsys.readouterr().err


def test_a_drifted_render_fails_against_the_recorded_reference(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    directory = copied_artifacts(fresh_numpy_run, tmp_path)

    def drift(payload: dict) -> None:
        payload["tokenizer"][0]["text"] += " "

    rewrite(directory, drift)
    assert compare(directory) == 1
    assert "renders different chat text" in capsys.readouterr().err


def test_a_drifted_converted_weight_fails_against_the_recorded_reference(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    """The weights are compared by digest, so a changed tensor cannot slip through."""
    directory = copied_artifacts(fresh_numpy_run, tmp_path)
    tool = load_validation_tool()

    def drift(payload: dict) -> None:
        payload["weight_digests"]["model.norm.weight"] = tool.tensor_digest(
            np.zeros(payload["weight_digests"]["model.norm.weight"]["shape"], dtype=np.float32)
        )

    rewrite(directory, drift)
    assert compare(directory) == 1
    assert "the conversion is not exact" in capsys.readouterr().err


def test_a_dropped_tensor_fails_against_the_recorded_reference(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    directory = copied_artifacts(fresh_numpy_run, tmp_path)
    blobs = dict(np.load(directory / "slowq.npz"))
    dropped = sorted(blobs)[-1]
    del blobs[dropped]
    np.savez_compressed(directory / "slowq.npz", **blobs)
    assert compare(directory) == 1
    err = capsys.readouterr().err
    assert "missing from the SlowQ side" in err
    assert dropped in err


def test_a_reference_capture_that_is_incomplete_prompts_a_recapture(
    fresh_numpy_run: Path, tmp_path: Path, capsys
):
    """Old or partial artifacts must ask for a new capture, never pass."""
    directory = copied_artifacts(fresh_numpy_run, tmp_path)
    (directory / "reference.json").unlink()
    assert compare(directory) == 1
    assert "re-capture" in capsys.readouterr().err
