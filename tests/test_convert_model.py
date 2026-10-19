"""Regression tests for ``tools/convert_model.py``, offline and on tiny inputs.

The converter is the one part of SlowQ where a bug destroys data instead of
producing a wrong answer, so most of this file is about what happens when a
conversion *fails*:

* an existing, working model directory must survive byte for byte;
* a failed first conversion must not leave a directory that looks like a model;
* a tied ``lm_head`` living in another shard must be read from *its own* file;
* the download cache must be keyed by the resolved commit, not by a file name.

Nothing here touches the network or a real checkpoint. The synthetic checkpoints
use the actual safetensors container format (8-byte little-endian header length,
JSON header, flat data blob), so the parser is exercised for real.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer as HFTokenizer

from conftest import TINY_ARCHITECTURE, build_tiny_tokenizer, tiny_tensors
from slowq.config import ModelConfig
from slowq.engine import Engine
from slowq.errors import SlowQError
from slowq.weights import Weights

REPO_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING = "model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
SHARD_1 = "model-00001-of-00002.safetensors"
SHARD_2 = "model-00002-of-00002.safetensors"
#: Big enough for the 24-tensor shard's header, and identical for both shards.
PADDED_HEADER_BYTES = 8192


def load_convert_tool():
    """Import ``tools/convert_model.py`` without making ``tools`` a package."""
    path = REPO_ROOT / "tools" / "convert_model.py"
    spec = importlib.util.spec_from_file_location("slowq_convert_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before executing: the module uses ``from __future__ import
    # annotations``, and dataclasses resolves those strings via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return load_convert_tool()


# --------------------------------------------------------------------------- #
# synthetic checkpoints
# --------------------------------------------------------------------------- #


def to_bf16_bits(array: np.ndarray) -> np.ndarray:
    """FP32 -> the raw BF16 bits an official checkpoint stores (truncating)."""
    packed = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    return (packed >> 16).astype(np.uint16)


def from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def encode_safetensors(
    arrays: Mapping[str, tuple[str, np.ndarray]], *, header_bytes: int | None = None
) -> bytes:
    """Serialise ``{name: (dtype, array)}`` into safetensors bytes.

    ``header_bytes`` pads ``__metadata__`` so the JSON header is exactly that
    long, which pins the data blob's offset. That is how a test can make two
    shards put different tensors at the same byte range.
    """
    header: dict[str, object] = {}
    blob = bytearray()
    for name, (dtype, array) in arrays.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(blob), len(blob) + len(raw)],
        }
        blob.extend(raw)

    def render(metadata: Mapping[str, str] | None) -> bytes:
        payload = dict(header)
        if metadata is not None:
            payload["__metadata__"] = dict(metadata)
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    encoded = render(None)
    if header_bytes is not None:
        pad = header_bytes - len(render({"pad": ""}))
        assert pad >= 0, f"the header is already longer than {header_bytes} bytes"
        while len(render({"pad": "x" * pad})) != header_bytes:
            pad += header_bytes - len(render({"pad": "x" * pad}))
        encoded = render({"pad": "x" * pad})
    return len(encoded).to_bytes(8, "little") + encoded + bytes(blob)


@pytest.fixture(scope="module")
def checkpoint_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tokenizer + config files shared by every synthetic checkpoint."""
    root = tmp_path_factory.mktemp("checkpoint-template")
    build_tiny_tokenizer(root)
    vocab_size = HFTokenizer.from_file(str(root / "tokenizer.json")).get_vocab_size()
    (root / "config.json").write_text(
        json.dumps(dict(TINY_ARCHITECTURE, vocab_size=vocab_size), indent=2), encoding="utf-8"
    )
    (root / "vocab_size.txt").write_text(str(vocab_size), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def vocab_size(checkpoint_template: Path) -> int:
    return int((checkpoint_template / "vocab_size.txt").read_text(encoding="utf-8"))


def write_checkpoint(
    root: Path,
    template: Path,
    tensors: Mapping[str, np.ndarray],
    *,
    shards: Sequence[Mapping[str, np.ndarray]] | None = None,
    dtype: str = "BF16",
    header_bytes: Mapping[str, int] | None = None,
) -> Path:
    """Write a checkpoint directory: config + tokenizer + one or more shards."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        shutil.copyfile(template / name, root / name)

    def encoded(name: str) -> tuple[str, np.ndarray]:
        array = tensors[name]
        return (dtype, to_bf16_bits(array) if dtype == "BF16" else array)

    if shards is None:
        payloads: list[tuple[str, dict[str, tuple[str, np.ndarray]]]] = [
            ("model.safetensors", {name: encoded(name) for name in sorted(tensors)})
        ]
    else:
        payloads = [
            (
                f"model-{index + 1:05d}-of-{len(shards):05d}.safetensors",
                {name: encoded(name) for name in sorted(shard)},
            )
            for index, shard in enumerate(shards)
        ]

    weight_map: dict[str, str] = {}
    for filename, arrays in payloads:
        (root / filename).write_bytes(
            encode_safetensors(arrays, header_bytes=(header_bytes or {}).get(filename))
        )
        weight_map.update({name: filename for name in arrays})

    if len(payloads) > 1:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )
    return root


@pytest.fixture
def make_checkpoint(checkpoint_template: Path):
    """Factory: ``make_checkpoint(root, seed=..., tensors=..., broken=...)``."""

    def factory(
        root: Path,
        *,
        seed: int = 20261018,
        tensors: Mapping[str, np.ndarray] | None = None,
        broken: str | None = None,
        **kwargs,
    ) -> Path:
        vocab = int((checkpoint_template / "vocab_size.txt").read_text(encoding="utf-8"))
        weights = dict(tiny_tensors(vocab_size=vocab, seed=seed)) if tensors is None else dict(tensors)
        # The official Qwen3-0.6B checkpoint materialises the tied head anyway,
        # so the synthetic one does too: that is the path the alias logic covers.
        weights.setdefault(LM_HEAD, weights[EMBEDDING].copy())
        if broken == "shape":
            # Sorted last, so a single-shard conversion writes most of the model
            # before it hits this.
            weights["model.norm.weight"] = np.concatenate(
                [weights["model.norm.weight"], np.zeros(1, dtype=np.float32)]
            )
        return write_checkpoint(root, checkpoint_template, weights, **kwargs)

    return factory


def snapshot_tree(root: Path) -> dict[str, str]:
    """Every file under ``root``, mapped to its sha256."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def assert_still_usable(model_dir: Path) -> None:
    """The directory must load *and* run, not merely pass a manifest check."""
    config = ModelConfig.from_json_file(model_dir / "config.json")
    weights = Weights.from_directory(model_dir)
    weights.check_complete(config)
    engine = Engine.from_directory(model_dir, max_context=32)
    assert engine.generate_ids([1, 2, 3], max_new_tokens=2).token_ids


# --------------------------------------------------------------------------- #
# 1. a failed conversion must not damage an existing model
# --------------------------------------------------------------------------- #


def test_a_conversion_that_fails_part_way_leaves_the_existing_model_intact(
    tool, make_checkpoint, tmp_path: Path, capsys
):
    good = make_checkpoint(tmp_path / "ckpt-good", seed=1)
    broken = make_checkpoint(tmp_path / "ckpt-broken", seed=2, broken="shape")
    model_dir = tmp_path / "model"

    assert tool.main(["--source", str(good), "--out", str(model_dir)]) == 0
    before = snapshot_tree(model_dir)
    manifest_before = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert_still_usable(model_dir)

    capsys.readouterr()
    # --overwrite is the only way a second conversion can touch the directory at
    # all, so it is the path where a mid-conversion failure used to be fatal.
    assert tool.main(["--source", str(broken), "--out", str(model_dir), "--overwrite"]) == 1
    assert "shape" in capsys.readouterr().err

    assert snapshot_tree(model_dir) == before, "a failed conversion modified an existing model"
    manifest_after = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_after["source"] == manifest_before["source"], "provenance changed"
    assert manifest_after["created_at"] == manifest_before["created_at"]
    assert_still_usable(model_dir)


def test_a_failed_first_conversion_leaves_nothing_that_looks_like_a_model(
    tool, make_checkpoint, tmp_path: Path
):
    broken = make_checkpoint(tmp_path / "ckpt-broken", broken="shape")
    out = tmp_path / "model"

    assert tool.main(["--source", str(broken), "--out", str(out)]) == 1

    assert not out.exists(), f"{out} was created by a conversion that failed"
    assert [path.name for path in tmp_path.iterdir()] == ["ckpt-broken"], (
        "a failed conversion left a staging directory behind"
    )
    with pytest.raises(Exception):
        Weights.from_directory(out)


def test_a_non_empty_output_directory_is_refused_unless_overwrite_is_requested(
    tool, make_checkpoint, tmp_path: Path, capsys
):
    first = make_checkpoint(tmp_path / "ckpt-a", seed=3)
    second = make_checkpoint(tmp_path / "ckpt-b", seed=4)
    model_dir = tmp_path / "model"
    assert tool.main(["--source", str(first), "--out", str(model_dir)]) == 0
    before = snapshot_tree(model_dir)

    capsys.readouterr()
    assert tool.main(["--source", str(second), "--out", str(model_dir)]) == 1
    err = capsys.readouterr().err
    assert "--overwrite" in err and "not empty" in err
    assert snapshot_tree(model_dir) == before


def test_overwrite_keeps_the_old_model_until_the_new_one_validates(
    tool, make_checkpoint, tmp_path: Path
):
    good = make_checkpoint(tmp_path / "ckpt-good", seed=5)
    broken = make_checkpoint(tmp_path / "ckpt-broken", seed=6, broken="shape")
    newer = make_checkpoint(tmp_path / "ckpt-newer", seed=7)
    model_dir = tmp_path / "model"
    assert tool.main(["--source", str(good), "--out", str(model_dir)]) == 0
    before = snapshot_tree(model_dir)
    expected_entries = ["ckpt-broken", "ckpt-good", "ckpt-newer", "model"]

    # The replacement fails: the old model must be exactly what it was.
    assert tool.main(["--source", str(broken), "--out", str(model_dir), "--overwrite"]) == 1
    assert snapshot_tree(model_dir) == before
    assert_still_usable(model_dir)
    assert sorted(path.name for path in tmp_path.iterdir()) == expected_entries, (
        "a failed --overwrite left a staging or backup directory behind"
    )

    # A valid replacement installs, and the provenance points at the new source.
    assert tool.main(["--source", str(newer), "--out", str(model_dir), "--overwrite"]) == 0
    assert snapshot_tree(model_dir) != before
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["local_path"] == str(newer)
    assert_still_usable(model_dir)
    assert sorted(path.name for path in tmp_path.iterdir()) == expected_entries


def test_a_successful_conversion_is_complete_and_exact(tool, make_checkpoint, tmp_path: Path):
    """The happy path, so the failure tests cannot pass by breaking everything."""
    source = make_checkpoint(tmp_path / "ckpt", seed=8)
    out = tmp_path / "model"
    assert tool.main(["--source", str(source), "--out", str(out), "--revision", "ab" * 20]) == 0

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    config = ModelConfig.from_json_file(out / "config.json")
    weights = Weights.from_directory(out)
    assert manifest["aliases"] == {LM_HEAD: EMBEDDING}
    assert manifest["source"]["revision"] == "ab" * 20
    assert {entry["name"] for entry in manifest["tensors"]} == set(weights.manifest.tensors)
    assert not (out / "tensors" / f"{LM_HEAD}.npy").exists()

    weights.check_complete(config)
    for name, expected in tiny_tensors(vocab_size=config.vocab_size, seed=8).items():
        # BF16 truncates, so compare against the truncation of the same values.
        assert np.array_equal(
            np.asarray(weights.get(name)), from_bf16_bits(to_bf16_bits(expected))
        ), name


# --------------------------------------------------------------------------- #
# 1b. a shard that was not fully downloaded
# --------------------------------------------------------------------------- #


def truncate_shard(tool, checkpoint: Path, *, keep_bytes_after_header: int) -> Path:
    """Cut ``model.safetensors`` short, leaving its header intact.

    An interrupted download looks exactly like this: the header is complete, so
    the tensor list parses and every byte range looks plausible, but the ranges
    run past the end of the file.
    """
    shard = checkpoint / "model.safetensors"
    _, data_offset = tool.read_safetensors_header(shard)
    with shard.open("r+b") as handle:
        handle.truncate(data_offset + keep_bytes_after_header)
    return checkpoint


def test_a_truncated_shard_names_the_file_instead_of_raising_a_numpy_error(
    tool, make_checkpoint, tmp_path: Path, capsys
):
    """``memoryview`` slicing clamps, so the short read must be caught here.

    Left to NumPy, the failure surfaces as a reshape error that names neither
    the shard nor the tensor — and, because the traceback pins the frame that
    sliced the mapping, closing the mapping then raises ``BufferError`` *over*
    it. Both hide which file was actually bad.
    """
    source = truncate_shard(
        tool, make_checkpoint(tmp_path / "ckpt", seed=21), keep_bytes_after_header=4
    )
    out = tmp_path / "model"

    assert tool.main(["--source", str(source), "--out", str(out)]) == 1

    err = capsys.readouterr().err
    assert "model.safetensors" in err and "truncated" in err
    assert "Traceback" not in err, "an internal error escaped instead of the tool's own message"
    assert "BufferError" not in err, "closing the mapping replaced the real error"
    assert not out.exists()


def test_a_range_past_the_end_of_the_shard_is_rejected_before_numpy_sees_it(
    tool, make_checkpoint, tmp_path: Path
):
    source = truncate_shard(
        tool, make_checkpoint(tmp_path / "ckpt", seed=22), keep_bytes_after_header=4
    )
    entries = {entry.name: entry for entry in tool.iter_safetensors_entries(source)}
    with tool.mapped(source / "model.safetensors") as blob:
        for name, entry in entries.items():
            if entry.end <= len(blob):
                continue
            with pytest.raises(SlowQError) as excinfo:
                tool.convert_tensor(entry, blob)
            assert name in str(excinfo.value) and "truncated" in str(excinfo.value)
            break
        else:  # pragma: no cover - the truncation would have had no effect
            pytest.fail("every tensor range still fitted inside the truncated shard")


def truncate_after(tool, checkpoint: Path, *keep: str) -> Path:
    """Truncate the shard just past the last of ``keep``, so those stay readable.

    Used to get *past* the tied-head check and fail inside the tensor loop —
    the point where most of a model has already been written.
    """
    shard = checkpoint / "model.safetensors"
    entries = {entry.name: entry for entry in tool.iter_safetensors_entries(checkpoint)}
    limit = max(entries[name].end for name in keep)
    with shard.open("r+b") as handle:
        handle.truncate(limit + 4)
    return checkpoint


def test_a_truncated_replacement_leaves_the_existing_model_intact(
    tool, make_checkpoint, tmp_path: Path, capsys
):
    good = make_checkpoint(tmp_path / "ckpt-good", seed=23)
    truncated = truncate_after(
        tool, make_checkpoint(tmp_path / "ckpt-bad", seed=24), LM_HEAD, EMBEDDING
    )
    model_dir = tmp_path / "model"
    assert tool.main(["--source", str(good), "--out", str(model_dir)]) == 0
    before = snapshot_tree(model_dir)

    capsys.readouterr()
    assert tool.main(["--source", str(truncated), "--out", str(model_dir), "--overwrite"]) == 1
    err = capsys.readouterr().err

    assert "truncated" in err, f"the real cause was not reported: {err[-400:]}"
    assert "Traceback" not in err and "BufferError" not in err
    assert snapshot_tree(model_dir) == before, "a truncated replacement damaged the old model"
    assert_still_usable(model_dir)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ckpt-bad", "ckpt-good", "model"], (
        "a failed --overwrite left a staging or backup directory behind"
    )


# --------------------------------------------------------------------------- #
# 1c. an output directory that already exists and is empty
# --------------------------------------------------------------------------- #


def test_a_pre_existing_empty_output_directory_is_accepted_without_overwrite(
    tool, make_checkpoint, tmp_path: Path
):
    """``check_output_is_installable`` allowed it, ``install`` then refused it.

    Pre-creating ``--out`` (or letting an editor create it) used to fail *after*
    every tensor had been converted, with a message blaming the directory for
    having "appeared while converting".
    """
    source = make_checkpoint(tmp_path / "ckpt", seed=31)
    out = tmp_path / "model"
    out.mkdir()

    assert tool.main(["--source", str(source), "--out", str(out)]) == 0

    assert_still_usable(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["local_path"] == str(source)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ckpt", "model"], (
        "the staging directory was left behind"
    )


def test_a_directory_holding_only_a_hidden_file_is_not_an_empty_output_directory(
    tool, make_checkpoint, tmp_path: Path, capsys
):
    source = make_checkpoint(tmp_path / "ckpt", seed=32)
    out = tmp_path / "model"
    out.mkdir()
    (out / ".DS_Store").write_bytes(b"junk")

    capsys.readouterr()
    assert tool.main(["--source", str(source), "--out", str(out)]) == 1

    err = capsys.readouterr().err
    assert "--overwrite" in err and ".DS_Store" in err
    assert (out / ".DS_Store").read_bytes() == b"junk"
    assert sorted(path.name for path in out.iterdir()) == [".DS_Store"], (
        "a refused conversion touched the directory it was not allowed to write"
    )


def test_a_directory_that_becomes_non_empty_while_converting_is_not_overwritten(
    tool, make_checkpoint, tmp_path: Path, monkeypatch, capsys
):
    """The check at the end must still be a check.

    Accepting an empty ``--out`` up front is only safe if the install step
    refuses a directory that gained a file in the meantime; otherwise a
    conversion would swallow whatever the user put there.
    """
    source = make_checkpoint(tmp_path / "ckpt", seed=33)
    out = tmp_path / "model"
    validate = tool.validate_converted_model

    def a_file_appears(directory: Path, config) -> None:
        validate(directory, config)
        out.mkdir(exist_ok=True)
        (out / "notes.txt").write_text("do not touch", encoding="utf-8")

    monkeypatch.setattr(tool, "validate_converted_model", a_file_appears)
    capsys.readouterr()
    assert tool.main(["--source", str(source), "--out", str(out)]) == 1

    err = capsys.readouterr().err
    assert "not empty" in err, err
    assert (out / "notes.txt").read_text(encoding="utf-8") == "do not touch"
    assert sorted(path.name for path in out.iterdir()) == ["notes.txt"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ckpt", "model"], (
        "a refused install left a staging directory behind"
    )


# --------------------------------------------------------------------------- #
# 2. tied weights that live in different shards
# --------------------------------------------------------------------------- #


def tied_two_shard_checkpoint(make_checkpoint, root: Path, vocab: int, *, identical: bool, pad: bool):
    """``lm_head`` in shard 2, ``embed_tokens`` in shard 1, equal or not."""
    tensors = tiny_tensors(vocab_size=vocab, seed=9)
    tensors[LM_HEAD] = tensors[EMBEDDING].copy() if identical else tensors[EMBEDDING] + 1.0
    first = {EMBEDDING: tensors[EMBEDDING]}
    second = {name: value for name, value in tensors.items() if name != EMBEDDING}
    return make_checkpoint(
        root,
        tensors=tensors,
        shards=[first, second],
        header_bytes={SHARD_1: PADDED_HEADER_BYTES, SHARD_2: PADDED_HEADER_BYTES} if pad else None,
    )


def test_a_tied_head_in_another_shard_is_accepted_when_the_bytes_really_match(
    tool, make_checkpoint, vocab_size: int, tmp_path: Path
):
    source = tied_two_shard_checkpoint(
        make_checkpoint, tmp_path / "ckpt", vocab_size, identical=True, pad=False
    )
    out = tmp_path / "model"
    assert tool.main(["--source", str(source), "--out", str(out)]) == 0

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["aliases"] == {LM_HEAD: EMBEDDING}
    assert not (out / "tensors" / f"{LM_HEAD}.npy").exists()
    weights = Weights.from_directory(out)
    assert np.array_equal(np.asarray(weights.get(LM_HEAD)), np.asarray(weights.get(EMBEDDING)))


def test_a_tied_head_that_differs_in_another_shard_is_rejected_even_when_offsets_coincide(
    tool, make_checkpoint, vocab_size: int, tmp_path: Path, capsys
):
    """The adversarial case: same offsets, different bytes, different files.

    Reading both tensors through the head's shard compares ``lm_head`` with
    itself, and reports a legal tie for a checkpoint that does not have one.
    """
    source = tied_two_shard_checkpoint(
        make_checkpoint, tmp_path / "ckpt", vocab_size, identical=False, pad=True
    )
    head = source / SHARD_2
    embedding = source / SHARD_1
    assert head.is_file() and embedding.is_file()

    capsys.readouterr()
    assert tool.main(["--source", str(source), "--out", str(tmp_path / "model")]) == 1
    assert "tie_word_embeddings" in capsys.readouterr().err
    assert not (tmp_path / "model").exists()


def test_a_tied_head_that_differs_in_the_same_shard_is_rejected(
    tool, make_checkpoint, vocab_size: int, tmp_path: Path, capsys
):
    tensors = tiny_tensors(vocab_size=vocab_size, seed=10)
    tensors[LM_HEAD] = tensors[EMBEDDING] + 1.0
    source = make_checkpoint(tmp_path / "ckpt", tensors=tensors)

    capsys.readouterr()
    assert tool.main(["--source", str(source), "--out", str(tmp_path / "model")]) == 1
    assert "tie_word_embeddings" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# 3. the download cache is keyed by repo + resolved commit
# --------------------------------------------------------------------------- #


REVISION_1 = "a" * 40
REVISION_2 = "b" * 40


class FakeHub:
    """An in-memory Hugging Face endpoint serving two revisions of one repo."""

    endpoint = "https://hub.test"
    branches = {"main": REVISION_1}

    def __init__(self, revisions: Mapping[str, Path]) -> None:
        self.revisions = {name: Path(root) for name, root in revisions.items()}
        self.downloads: list[str] = []

    def blob(self, revision: str, name: str) -> bytes:
        return (self.revisions[revision] / name).read_bytes()

    def names(self, revision: str) -> list[str]:
        return sorted(path.name for path in self.revisions[revision].iterdir())

    def http_get(self, url: str, timeout: float = 60.0) -> bytes:
        assert url.startswith(self.endpoint), url
        path = url[len(self.endpoint) :]
        if "/tree/" in path:
            revision = path.split("/tree/", 1)[1]
            return json.dumps(
                [
                    {
                        "type": "file",
                        "path": name,
                        "size": len(self.blob(revision, name)),
                        # Only LFS files publish a sha256; small JSON files do not.
                        "lfs": {"oid": hashlib.sha256(self.blob(revision, name)).hexdigest()}
                        if name.endswith(".safetensors")
                        else None,
                    }
                    for name in self.names(revision)
                ]
            ).encode("utf-8")
        if "/api/models/" in path:
            branch = path.rsplit("/revision/", 1)[1] if "/revision/" in path else "main"
            return json.dumps({"sha": self.branches[branch]}).encode("utf-8")
        raise AssertionError(f"unexpected URL {url}")

    def http_download(self, url: str, destination: Path, timeout: float = 60.0) -> None:
        assert url.startswith(f"{self.endpoint}/"), url
        revision, name = url.split("/resolve/", 1)[1].split("/", 1)
        self.downloads.append(f"{revision}:{name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.blob(revision, name))


@pytest.fixture
def fake_hub(monkeypatch, tool, make_checkpoint, tmp_path: Path) -> FakeHub:
    hub = FakeHub(
        {
            REVISION_1: make_checkpoint(tmp_path / "hub-rev1", seed=11),
            REVISION_2: make_checkpoint(tmp_path / "hub-rev2", seed=12),
        }
    )
    monkeypatch.setenv("HF_ENDPOINT", hub.endpoint)
    monkeypatch.setattr(tool, "http_get", hub.http_get)
    monkeypatch.setattr(tool, "http_download", hub.http_download)
    return hub


def test_the_same_commit_reuses_its_snapshot_and_a_new_commit_gets_a_new_one(
    tool, fake_hub: FakeHub, tmp_path: Path
):
    cache = tmp_path / "cache"
    first = tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    assert first == cache / "Tiny--Qwen3" / REVISION_1
    assert (first / "model.safetensors").is_file()
    assert (first / tool.SNAPSHOT_MARKER).is_file()
    downloaded_first = list(fake_hub.downloads)
    assert downloaded_first, "nothing was downloaded"

    tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    assert fake_hub.downloads == downloaded_first, "a cached snapshot was downloaded again"

    second = tool.download_snapshot("Tiny/Qwen3", REVISION_2, cache)
    assert second == cache / "Tiny--Qwen3" / REVISION_2
    assert len(fake_hub.downloads) > len(downloaded_first)
    assert (
        hashlib.sha256((first / "model.safetensors").read_bytes()).hexdigest()
        != hashlib.sha256((second / "model.safetensors").read_bytes()).hexdigest()
    ), "two revisions share one snapshot directory"


def test_a_cached_file_that_does_not_match_the_listing_is_downloaded_again(
    tool, fake_hub: FakeHub, tmp_path: Path
):
    cache = tmp_path / "cache"
    snapshot = tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    before = len(fake_hub.downloads)

    truncated = snapshot / "model.safetensors"
    truncated.write_bytes(truncated.read_bytes()[:-16])
    tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    assert f"{REVISION_1}:model.safetensors" in fake_hub.downloads[before:]


def test_a_snapshot_directory_from_another_source_is_refused(
    tool, fake_hub: FakeHub, tmp_path: Path
):
    cache = tmp_path / "cache"
    snapshot = tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)

    marker = snapshot / tool.SNAPSHOT_MARKER
    marker.write_text(
        json.dumps(
            {"repo_id": "Tiny/Qwen3", "revision": REVISION_2, "endpoint": fake_hub.endpoint}
        ),
        encoding="utf-8",
    )
    with pytest.raises(tool.SlowQError, match="revision"):
        tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)

    # A hand-made directory with no marker is not a verified snapshot.
    marker.unlink()
    with pytest.raises(tool.SlowQError, match=tool.SNAPSHOT_MARKER):
        tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)


def test_the_manifest_records_the_commit_the_bytes_actually_came_from(
    tool, fake_hub: FakeHub, tmp_path: Path
):
    cache = tmp_path / "cache"
    out = tmp_path / "model"
    argv = ["--repo", "Tiny/Qwen3", "--download-dir", str(cache), "--out", str(out)]
    assert tool.main([*argv, "--revision", "main"]) == 0

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["repo_id"] == "Tiny/Qwen3"
    assert manifest["source"]["revision"] == REVISION_1
    assert manifest["source"]["local_path"] == str(cache / "Tiny--Qwen3" / REVISION_1)
    first_embedding = np.array(Weights.from_directory(out).get(EMBEDDING), dtype=np.float32)

    # Same output, new revision: the weights must change with the recorded commit.
    assert tool.main([*argv, "--revision", REVISION_2, "--overwrite"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["revision"] == REVISION_2
    second_embedding = np.array(Weights.from_directory(out).get(EMBEDDING), dtype=np.float32)
    assert not np.array_equal(first_embedding, second_embedding), (
        "the revision changed but the weights did not: a stale snapshot was reused"
    )


def test_skip_download_needs_a_full_commit_and_a_verified_snapshot(
    tool, fake_hub: FakeHub, tmp_path: Path, capsys
):
    cache = tmp_path / "cache"
    argv = ["--repo", "Tiny/Qwen3", "--download-dir", str(cache), "--skip-download"]
    capsys.readouterr()
    assert tool.main([*argv, "--revision", "main", "--out", str(tmp_path / "model")]) == 1
    assert "commit sha" in capsys.readouterr().err
    assert fake_hub.downloads == [], "--skip-download contacted the network"

    assert tool.main([*argv, "--revision", REVISION_2, "--out", str(tmp_path / "m1")]) == 1
    assert "does not exist" in capsys.readouterr().err

    tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    downloads = list(fake_hub.downloads)
    assert tool.main([*argv, "--revision", REVISION_1, "--out", str(tmp_path / "m2")]) == 0
    assert fake_hub.downloads == downloads
    manifest = json.loads((tmp_path / "m2" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["revision"] == REVISION_1


def test_an_unverified_download_directory_is_refused_by_skip_download(
    tool, fake_hub: FakeHub, tmp_path: Path
):
    cache = tmp_path / "cache"
    tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    (cache / "Tiny--Qwen3" / REVISION_1 / tool.SNAPSHOT_MARKER).unlink()
    before = list(fake_hub.downloads)
    assert (
        tool.main(
            [
                "--repo",
                "Tiny/Qwen3",
                "--revision",
                REVISION_1,
                "--download-dir",
                str(cache),
                "--out",
                str(tmp_path / "model"),
                "--skip-download",
            ]
        )
        == 1
    )
    assert fake_hub.downloads == before


# --------------------------------------------------------------------------- #
# 3b. an interrupted download must be resumable
# --------------------------------------------------------------------------- #


def snapshot_marker(tool, root: Path) -> dict:
    return json.loads((root / tool.SNAPSHOT_MARKER).read_text(encoding="utf-8"))


def interrupt_downloads(
    tool, monkeypatch, hub: FakeHub, *, after: int, corrupt: bool = False
) -> None:
    """Let ``after`` files land, then fail the way a killed connection does.

    ``corrupt`` also leaves a short file at the *final* path, which is what a
    download that died between two writes looks like once nothing is left to
    distinguish it from a finished one. Any exception leaves the same cache
    state; ``SlowQError`` keeps these assertions about exit codes instead of
    tracebacks.
    """
    delegate = hub.http_download
    landed = {"count": 0}

    def flaky(url: str, destination: Path, timeout: float = 60.0) -> None:
        if landed["count"] >= after:
            if corrupt:
                revision, name = url.split("/resolve/", 1)[1].split("/", 1)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(hub.blob(revision, name)[:7])
            raise SlowQError("interrupted by a test")
        landed["count"] += 1
        delegate(url, destination, timeout)

    monkeypatch.setattr(tool, "http_download", flaky)


def serve_a_corrupt_file(tool, monkeypatch, hub: FakeHub, name: str) -> None:
    """Serve ``name`` short forever, so no number of retries can complete it."""
    delegate = hub.http_download

    def short(url: str, destination: Path, timeout: float = 60.0) -> None:
        if url.rsplit("/", 1)[1] == name:
            revision, _ = url.split("/resolve/", 1)[1].split("/", 1)
            hub.downloads.append(f"{revision}:{name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(hub.blob(revision, name)[:-8])
            return
        delegate(url, destination, timeout)

    monkeypatch.setattr(tool, "http_download", short)


def test_an_interrupted_first_download_is_finished_by_rerunning_the_same_command(
    tool, fake_hub: FakeHub, monkeypatch, tmp_path: Path, capsys
):
    """One failed download used to make the cache permanently unusable.

    The marker was written only after the last file arrived, so an interruption
    left a directory with nothing to prove SlowQ had created it — and the next
    identical command refused that directory before calling the downloader once.
    """
    cache = tmp_path / "cache"
    root = cache / "Tiny--Qwen3" / REVISION_1
    interrupt_downloads(tool, monkeypatch, fake_hub, after=2)

    with pytest.raises(SlowQError, match="interrupted"):
        tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)

    assert root.is_dir(), "the interrupted run left nothing to resume from"
    assert snapshot_marker(tool, root)["complete"] is False, (
        "an unfinished download was recorded as a complete snapshot"
    )
    assert len(fake_hub.downloads) == 2
    landed = [entry.split(":", 1)[1] for entry in fake_hub.downloads]

    monkeypatch.setattr(tool, "http_download", fake_hub.http_download)
    capsys.readouterr()
    assert tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache) == root

    assert snapshot_marker(tool, root)["complete"] is True
    for name in landed:
        assert sum(entry.endswith(f":{name}") for entry in fake_hub.downloads) == 1, (
            f"{name} was fetched again even though the cached copy matched the listing"
        )
    # The resumed snapshot is a real snapshot: it converts, and --skip-download
    # accepts it afterwards without contacting the network.
    downloads = list(fake_hub.downloads)
    out = tmp_path / "model"
    assert (
        tool.main(
            [
                "--repo",
                "Tiny/Qwen3",
                "--revision",
                REVISION_1,
                "--download-dir",
                str(cache),
                "--out",
                str(out),
                "--skip-download",
            ]
        )
        == 0
    )
    assert fake_hub.downloads == downloads
    assert_still_usable(out)


def test_a_half_written_file_is_not_accepted_as_a_cache_hit(
    tool, fake_hub: FakeHub, monkeypatch, tmp_path: Path
):
    cache = tmp_path / "cache"
    root = cache / "Tiny--Qwen3" / REVISION_1
    interrupt_downloads(tool, monkeypatch, fake_hub, after=2, corrupt=True)

    with pytest.raises(SlowQError, match="interrupted"):
        tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)

    truncated = root / "model.safetensors"
    wanted = fake_hub.blob(REVISION_1, "model.safetensors")
    assert truncated.is_file()
    assert truncated.stat().st_size < len(wanted)
    assert snapshot_marker(tool, root)["complete"] is False

    monkeypatch.setattr(tool, "http_download", fake_hub.http_download)
    tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)

    assert truncated.read_bytes() == wanted, "the truncated file was kept as a cache hit"
    assert snapshot_marker(tool, root)["complete"] is True


def test_a_snapshot_whose_files_never_match_is_never_marked_complete(
    tool, fake_hub: FakeHub, monkeypatch, tmp_path: Path, capsys
):
    """Completeness has to be established, not assumed once the loop finishes."""
    cache = tmp_path / "cache"
    root = cache / "Tiny--Qwen3" / REVISION_1
    serve_a_corrupt_file(tool, monkeypatch, fake_hub, "model.safetensors")

    capsys.readouterr()
    with pytest.raises(SlowQError) as excinfo:
        tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    assert "model.safetensors" in str(excinfo.value)
    assert snapshot_marker(tool, root)["complete"] is False

    capsys.readouterr()
    assert (
        tool.main(
            [
                "--repo",
                "Tiny/Qwen3",
                "--revision",
                REVISION_1,
                "--download-dir",
                str(cache),
                "--out",
                str(tmp_path / "model"),
                "--skip-download",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "--skip-download" in err, err


def test_a_complete_snapshot_is_reused_without_rewriting_its_marker(
    tool, fake_hub: FakeHub, tmp_path: Path
):
    cache = tmp_path / "cache"
    root = tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    marker = snapshot_marker(tool, root)
    assert marker["complete"] is True
    assert marker["repo_id"] == "Tiny/Qwen3"
    assert marker["revision"] == REVISION_1
    downloads = list(fake_hub.downloads)

    tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)

    assert fake_hub.downloads == downloads, "a complete snapshot was downloaded again"
    assert snapshot_marker(tool, root) == marker, "a cache hit rewrote the marker"


def test_a_directory_slowq_cannot_prove_it_created_is_refused_with_a_way_out(
    tool, fake_hub: FakeHub, tmp_path: Path
):
    """No marker means no ownership: refuse, but say how to recover safely."""
    cache = tmp_path / "cache"
    root = cache / "Tiny--Qwen3" / REVISION_1
    root.mkdir(parents=True)
    for name in ("model.safetensors", "config.json"):
        (root / name).write_bytes(fake_hub.blob(REVISION_1, name))
    before = snapshot_tree(root)

    with pytest.raises(SlowQError) as excinfo:
        tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)

    message = str(excinfo.value)
    assert fake_hub.downloads == [], "a directory of unknown origin was trusted"
    assert snapshot_tree(root) == before, "SlowQ deleted or rewrote files it does not own"
    # Refusing is only useful if the message says what to do next.
    assert tool.SNAPSHOT_MARKER in message
    assert str(root) in message and "--download-dir" in message and "--source" in message


def test_a_marker_that_cannot_be_read_is_refused_and_not_replaced(
    tool, fake_hub: FakeHub, tmp_path: Path
):
    cache = tmp_path / "cache"
    root = tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    marker = root / tool.SNAPSHOT_MARKER
    marker.write_text("{not json", encoding="utf-8")
    before = snapshot_tree(root)
    downloads = list(fake_hub.downloads)

    with pytest.raises(SlowQError, match="not valid JSON"):
        tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)

    assert fake_hub.downloads == downloads
    assert snapshot_tree(root) == before


def test_a_marker_naming_another_repo_or_endpoint_is_refused(
    tool, fake_hub: FakeHub, tmp_path: Path
):
    cache = tmp_path / "cache"
    root = tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    marker = root / tool.SNAPSHOT_MARKER
    downloads = list(fake_hub.downloads)

    recorded = snapshot_marker(tool, root)
    for key, wrong in (("repo_id", "Other/Qwen3"), ("endpoint", "https://mirror.test")):
        marker.write_text(
            json.dumps({**recorded, key: wrong}), encoding="utf-8"
        )
        with pytest.raises(SlowQError, match=key):
            tool.download_snapshot("Tiny/Qwen3", REVISION_1, cache)
    assert fake_hub.downloads == downloads, "a snapshot from elsewhere was reused"
