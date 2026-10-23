"""Shared plumbing for the real-weight integration tests.

One environment variable names the directory under test::

    SLOWQ_MODEL_DIR=models/SmolVLM-256M-Instruct-slowq python -m pytest -m integration

The *task* recorded in that directory's manifest then decides which integration
modules run and which skip themselves, so the same command validates a chat
model, an embedding model or a vision model without a second switch to keep in
sync. The reference regression in ``test_integration_reference.py`` has no task
filter at all and runs for every directory. Nothing here guesses the task from
``config.json``: Qwen3-Embedding-0.6B publishes ``model_type: "qwen3"`` and
``architectures: ["Qwen3ForCausalLM"]``, byte-for-byte what the chat model
publishes.

The reference artifacts live in ``artifacts/validation/<model-name>/``, the same
default ``tools/validate_reference.py`` picks, so validating a second model
cannot overwrite the first one's capture.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

MODEL_DIR_ENV = "SLOWQ_MODEL_DIR"
REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "validation"

#: The reference half writes both. ``ref.npz`` holds the numbers and
#: ``reference.json`` holds the *content* they are graded against: the reference
#: tokenizer's rendered text and token IDs, sha256 digests of whole weight
#: tensors, the digests of the checkpoint files it read, and the capture
#: configuration. Neither file contains a verdict, so there is nothing to copy
#: forward from an older run.
REFERENCE_FILES = ("ref.npz", "reference.json")

#: Context budget for a fresh NumPy capture. Only big enough to matter for the
#: vision samples, whose prompts run to 612 tokens once one image has been
#: expanded into 576 placeholders.
MAX_CONTEXT = 1024

#: Declared before measuring, and unchanged since v0.1. Rationale: chunking and
#: caching only change FP32 reduction order *within one implementation*, so the
#: gate has to be tighter than the cross-implementation one in
#: tools/validate_reference.py (rel_l2 1e-3, max_abs 5e-2 for the same
#: quantities), not looser.
#:
#: Measured afterwards, worst case over all six converted chat checkpoints
#: (Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-0.5B-Instruct, Qwen2.5-Coder-0.5B-Instruct,
#: SmolLM2-135M-Instruct, SmolLM2-360M-Instruct; 21-39 token prompt depending on
#: the tokenizer, chunk sizes 1/3/7/16, plus cached decode against one long
#: prefill): worst logits rel_l2 3.916e-06, worst hidden rel_l2 2.940e-06, worst
#: max_abs 1.755e-04 — i.e. 2.5x inside the rel_l2 gate and 5.7x inside the
#: max_abs gate. Both gates passed as declared on every model; neither was
#: widened to make a failure go away.
INTERNAL_REL_L2_TOLERANCE = 1e-5
INTERNAL_MAX_ABS_TOLERANCE = 1e-3


def assert_same_numerics(
    ours: Any, theirs: Any, what: str, *, check_argmax: bool = True
) -> None:
    """Assert two arrays agree to within the declared internal tolerances.

    Chunking and caching change FP32 reduction order and nothing else, so the
    greedy choice must survive too — a difference large enough to flip an argmax
    is a bug, not noise.

    Both a relative and an absolute gate are needed. A pure relative criterion is
    meaningless for a hidden-state vector, whose components span several orders
    of magnitude: the small ones carry a large relative error while contributing
    nothing to the result. A pure absolute criterion would in turn let a whole
    vector drift.
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


def selected_model_dir() -> Path | None:
    raw = os.environ.get(MODEL_DIR_ENV)
    return Path(raw).expanduser() if raw else None


def recorded_task(model_dir: Path) -> str:
    from slowq.engine import recorded_task as _recorded_task

    return _recorded_task(model_dir)


def skip_reason_for_any_task() -> str | None:
    """Why the task-agnostic modules cannot run, or ``None``.

    ``validate_reference.py`` dispatches on the manifest itself, so the reference
    regression needs only *some* converted directory and no particular task.
    """
    model_dir = selected_model_dir()
    if model_dir is None:
        return f"set {MODEL_DIR_ENV} to a converted model directory to run these"
    if not (model_dir / "config.json").is_file():
        return f"{model_dir} has no config.json; run tools/convert_model.py first"
    return None


def skip_reason(task: str) -> str | None:
    """Why this module cannot run against ``$SLOWQ_MODEL_DIR``, or ``None``."""
    wrong_directory = skip_reason_for_any_task()
    if wrong_directory is not None:
        return wrong_directory
    model_dir = selected_model_dir()
    assert model_dir is not None
    found = recorded_task(model_dir)
    if found != task:
        return (
            f"{model_dir} is a converted {found} checkpoint, and this module "
            f"tests {task}; point {MODEL_DIR_ENV} at one of those to run it"
        )
    return None


def artifacts_dir(model_dir: Path) -> Path:
    return ARTIFACT_ROOT / Path(model_dir).name


def load_validation_tool() -> Any:
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


def reference_is_complete(model_dir: Path) -> bool:
    directory = artifacts_dir(model_dir)
    return all((directory / name).is_file() for name in REFERENCE_FILES)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def capture_settings(model_dir: Path) -> tuple[int, int]:
    """``(prefill_chunk_size, decode_steps)`` of the recorded reference.

    Taken from the capture rather than passed in: ``compare`` refuses two runs
    that describe different captures, so a fresh NumPy side has to reproduce the
    reference's chunking and step count exactly.
    """
    capture = read_json(artifacts_dir(model_dir) / "reference.json")["capture"]
    return int(capture["prefill_chunk_size"]), int(capture["decode_steps"])


def fresh_numpy_run(
    model_dir: Path, work: Path, *, extra_args: tuple[str, ...] = ()
) -> Path:
    """Re-run the NumPy half next to a copy of the recorded reference.

    This is the slow part of the reference regression (every sample through the
    real FP32 model), which is why each module calls it from a module-scoped
    fixture; the tests that break one thing each then copy the result, which is
    cheap because ``samples.json`` and the two NPZ files are a few MiB without
    the weight blobs.
    """
    if not reference_is_complete(model_dir):
        pytest.skip(
            f"no complete reference capture in {artifacts_dir(model_dir)} (need "
            f"{', '.join(REFERENCE_FILES)}); run "
            "'python tools/validate_reference.py ref' first"
        )
    work.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_FILES:
        shutil.copyfile(artifacts_dir(model_dir) / name, work / name)
    chunk_size, decode_steps = capture_settings(model_dir)
    tool = load_validation_tool()
    argv = [
        "numpy",
        "--model",
        str(model_dir),
        "--artifacts",
        str(work),
        "--max-context",
        str(MAX_CONTEXT),
        "--prefill-chunk-size",
        str(chunk_size),
        "--decode-steps",
        str(decode_steps),
        *extra_args,
    ]
    assert tool.run_numpy_side(tool.build_parser().parse_args(argv)) == 0
    return work


def compare(directory: Path) -> int:
    tool = load_validation_tool()
    return tool.run_compare(
        tool.build_parser().parse_args(["compare", "--artifacts", str(directory)])
    )


def copied_artifacts(source: Path, destination: Path) -> Path:
    for name in ("samples.json", "slowq.npz", *REFERENCE_FILES):
        shutil.copyfile(source / name, destination / name)
    return destination


def rewrite(directory: Path, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    path = directory / "samples.json"
    payload = read_json(path)
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def expected_numeric_count(directory: Path) -> int:
    tool = load_validation_tool()
    return len(tool.expected_numeric_keys(read_json(directory / "samples.json")["capture"]))


def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any socket use by the runtime a test failure.

    A converted directory has everything it needs on disk, so reaching the
    network is a bug rather than a fallback. Patched at the ``socket`` module
    because that is what both ``urllib`` and ``http.client`` go through.
    """
    import socket

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the runtime opened a network connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


def digest_names_present_on_both_sides(directory: Path) -> list[str]:
    """Weight names both captures digested, in the SlowQ side's order."""
    ours = read_json(directory / "samples.json")["weight_digests"]
    theirs = read_json(directory / "reference.json")["weight_digests"]
    return [name for name in ours if name in theirs]
