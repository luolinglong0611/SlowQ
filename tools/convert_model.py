#!/usr/bin/env python3
"""Convert an official Qwen3 checkpoint into SlowQ's offline FP32 format.

    python tools/convert_model.py --repo Qwen/Qwen3-0.6B --out models/Qwen3-0.6B-slowq
    python tools/convert_model.py --source models/Qwen3-0.6B-hf --out models/Qwen3-0.6B-slowq

Output layout (see src/slowq/weights.py)::

    <out>/manifest.json  config.json  generation_config.json  tokenizer files
    <out>/tensors/<tensor-name>.npy

Design notes:

* **No torch.** ``safetensors`` is an 8-byte little-endian header length, a JSON
  header, then a flat data blob. We mmap that blob and convert each tensor in
  isolation, so peak memory is one tensor, not one model.
* **BF16 -> FP32 is exact.** A BF16 value is the top 16 bits of an IEEE-754
  single, so ``uint32(bits) << 16`` reinterpreted as float32 is the same number,
  with no rounding and no denormal/NaN surprises.
* **Weights keep HF's ``[out_features, in_features]`` layout.** The runtime
  computes ``x @ W.T``, so neither conversion nor inference transposes data.
* **Tied embeddings are stored once.** Qwen3-0.6B ties ``lm_head.weight`` to
  ``model.embed_tokens.weight``; we verify the two byte ranges are identical and
  record an alias instead of duplicating 600 MiB.
* The source revision is pinned and written into the manifest, so a conversion
  is reproducible and inference afterwards is fully offline.
* **A failed conversion cannot damage a model.** Everything is written into a
  staging directory next to ``--out`` and only installed once it loads back
  through the runtime's own loader, so an existing directory is never a mixture
  of old and new weights behind a still-valid manifest. ``--out`` may be absent
  or empty; a non-empty one needs ``--overwrite``, checked both before the
  conversion and again at the install step.
* **An interrupted download can be finished.** The snapshot marker is written
  before the first file and flipped to ``complete`` only after every downloaded
  file has been checked against the revision's listing, so re-running the same
  command resumes instead of refusing the directory it already started. A
  directory SlowQ cannot account for is still refused — and never deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowq.config import ModelConfig  # noqa: E402
from slowq.errors import ConfigError, SlowQError  # noqa: E402
from slowq.weights import (  # noqa: E402
    EMBEDDING_TENSOR_NAME,
    MANIFEST_FILENAME,
    SUPPORTED_FORMAT_VERSION,
    TENSOR_SUBDIR,
    TIED_OUTPUT_EMBEDDING_ALIAS,
    Weights,
    required_tensors,
)

DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_REPO = "Qwen/Qwen3-0.6B"
#: Pinned so a re-run months from now still produces the same bytes.
DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CONFIG_FILE = "config.json"
GENERATION_CONFIG_FILE = "generation_config.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
SAFETENSORS_FILE = "model.safetensors"
CHUNK_BYTES = 1 << 20
#: Written into every snapshot directory we create, *before* the first download
#: as well as after the last one. It is what proves a directory is ours and which
#: repo+commit it is meant to hold, and it records whether the download finished,
#: so an interrupted one can be resumed instead of being refused forever.
SNAPSHOT_MARKER = ".slowq-snapshot.json"
COMMIT_SHA_LENGTH = 40


NUMPY_DTYPE_BY_SAFETENSORS = {
    "F64": np.dtype(np.float64),
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "BF16": None,  # handled by the bit-shift path; numpy has no bf16 dtype
    "I64": np.dtype(np.int64),
    "I32": np.dtype(np.int32),
    "I16": np.dtype(np.int16),
    "I8": np.dtype(np.int8),
    "U8": np.dtype(np.uint8),
    "BOOL": np.dtype(np.bool_),
}


@dataclass(frozen=True)
class SafeTensorEntry:
    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


# ---- safetensors parsing ----------------------------------------------------


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    """Return ``(header, data_offset)`` without loading tensor data."""
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise SlowQError(f"{path} is too small to be a safetensors file")
        header_length = int.from_bytes(raw_length, "little")
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise SlowQError(f"{path} has a truncated JSON header")
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise SlowQError(f"{path} header is not valid JSON: {exc}") from exc
    return header, 8 + header_length


def iter_safetensors_entries(root: Path) -> Iterator[SafeTensorEntry]:
    """Yield every tensor in the checkpoint with offsets absolute within its file.

    The header stores offsets relative to the start of the data blob, so the
    blob's own offset (``8 + len(header_json)``) has to be added here; forgetting
    it silently reads the JSON header as weights.
    """
    index_path = root / SAFETENSORS_INDEX_FILE
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise SlowQError(f"{index_path} has no weight_map")
        files = sorted(set(weight_map.values()))
    else:
        single = root / SAFETENSORS_FILE
        if not single.is_file():
            raise SlowQError(
                f"no {SAFETENSORS_FILE} or {SAFETENSORS_INDEX_FILE} under {root}"
            )
        files = [SAFETENSORS_FILE]

    for filename in files:
        path = root / filename
        if not path.is_file():
            raise SlowQError(f"checkpoint references a missing file: {path}")
        header, data_offset = read_safetensors_header(path)
        for name, meta in sorted(header.items()):
            if name == "__metadata__":
                continue
            offsets = meta.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise SlowQError(f"{filename}: tensor {name!r} has no data_offsets")
            yield SafeTensorEntry(
                name=name,
                file=filename,
                dtype=str(meta.get("dtype")),
                shape=tuple(int(d) for d in meta.get("shape", [])),
                begin=data_offset + int(offsets[0]),
                end=data_offset + int(offsets[1]),
            )


def tensor_view(entry: SafeTensorEntry, blob: mmap.mmap) -> memoryview:
    """This tensor's bytes, checked against the shard's *actual* length.

    ``memoryview`` slicing clamps the way ``bytes`` slicing does, so a truncated
    shard would otherwise reach NumPy as a silently short buffer and surface as a
    reshape error naming neither the file nor the tensor.
    """
    view = memoryview(blob)[entry.begin : entry.end]
    if len(view) != entry.end - entry.begin:
        raise SlowQError(
            f"tensor {entry.name!r} claims bytes {entry.begin}..{entry.end} of "
            f"{entry.file}, but that file is only {len(blob)} bytes long: the "
            "checkpoint is truncated or was not fully downloaded"
        )
    return view


def convert_tensor(entry: SafeTensorEntry, blob: mmap.mmap) -> np.ndarray:
    """Convert one tensor to FP32. Only this tensor is ever resident."""
    length = entry.end - entry.begin
    if length <= 0:
        raise SlowQError(f"tensor {entry.name!r} has an empty data range")
    view = tensor_view(entry, blob)

    if entry.dtype == "BF16":
        if length != entry.numel * 2:
            raise SlowQError(
                f"tensor {entry.name!r}: {length} bytes do not match "
                f"{entry.numel} BF16 elements"
            )
        raw = np.frombuffer(view, dtype=np.uint16)
        # A BF16 value is the top half of an IEEE-754 single: shift left 16 bits
        # and reinterpret. Exact, with no rounding and correct NaN/inf handling.
        bits = raw.astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32).reshape(entry.shape).copy()

    source_dtype = NUMPY_DTYPE_BY_SAFETENSORS.get(entry.dtype)
    if source_dtype is None:
        raise SlowQError(
            f"tensor {entry.name!r} has unsupported safetensors dtype {entry.dtype!r}"
        )
    if length != entry.numel * source_dtype.itemsize:
        raise SlowQError(
            f"tensor {entry.name!r}: {length} bytes do not match {entry.numel} "
            f"{entry.dtype} elements"
        )
    return np.frombuffer(view, dtype=source_dtype).reshape(entry.shape).astype(
        np.float32, copy=True
    )


def tensor_sha256(entry: SafeTensorEntry, blob: mmap.mmap) -> str:
    digest = hashlib.sha256()
    view = tensor_view(entry, blob)
    for offset in range(0, len(view), CHUNK_BYTES):
        digest.update(view[offset : offset + CHUNK_BYTES])
    return digest.hexdigest()


# ---- downloading ------------------------------------------------------------


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


def http_get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "slowq-convert/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def http_download(url: str, destination: Path, timeout: float = 60.0) -> None:
    """Stream a file to disk, reporting progress on stderr."""
    request = urllib.request.Request(url, headers={"User-Agent": "slowq-convert/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length") or 0)
        temporary = destination.with_suffix(destination.suffix + ".part")
        received = 0
        next_report = 0.0
        with temporary.open("wb") as handle:
            while True:
                block = response.read(CHUNK_BYTES)
                if not block:
                    break
                handle.write(block)
                received += len(block)
                if time.monotonic() >= next_report:
                    next_report = time.monotonic() + 2.0
                    percent = f"{100 * received / total:.0f}%" if total else "?"
                    print(
                        f"  {destination.name}: {received / 2**20:.1f} MiB / "
                        f"{total / 2**20:.1f} MiB ({percent})",
                        file=sys.stderr,
                        flush=True,
                    )
        temporary.replace(destination)


def resolve_revision(repo: str, revision: str | None) -> str:
    if is_commit_sha(revision):
        return str(revision)
    url = f"{endpoint()}/api/models/{repo}"
    if revision:
        url += f"/revision/{revision}"
    try:
        info = json.loads(http_get(url))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        raise SlowQError(
            f"could not resolve revision for {repo} via {url}: {exc}. "
            "Pass --revision explicitly or use --source with a local snapshot."
        ) from exc
    sha = info.get("sha")
    if not isinstance(sha, str):
        raise SlowQError(f"{url} did not return a commit sha")
    return sha


def is_commit_sha(revision: str | None) -> bool:
    """True for a full 40-hex commit sha, i.e. something that needs no network."""
    return bool(
        revision
        and len(revision) == COMMIT_SHA_LENGTH
        and all(character in "0123456789abcdef" for character in revision)
    )


def repo_slug(repo: str) -> str:
    return repo.replace("/", "--")


def snapshot_dir(cache_root: Path, repo: str, revision: str) -> Path:
    """Where exactly one commit of exactly one repo lives.

    Keying the cache by ``repo`` *and* the resolved commit is what makes reuse
    safe: a branch name moves, and a directory named after it would silently
    serve yesterday's weights while the manifest records today's revision.
    """
    return Path(cache_root) / repo_slug(repo) / revision


@dataclass(frozen=True)
class SnapshotFile:
    """One file of a repo revision, as listed by the Hub API."""

    path: str
    size: int | None = None
    sha256: str | None = None


def list_repo_files(repo: str, revision: str) -> list[SnapshotFile]:
    try:
        listing = json.loads(http_get(f"{endpoint()}/api/models/{repo}/tree/{revision}"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        raise SlowQError(f"could not list files of {repo}@{revision}: {exc}") from exc
    files: list[SnapshotFile] = []
    for item in listing:
        if item.get("type") != "file":
            continue
        lfs = item.get("lfs") or {}
        # For LFS-tracked files the Hub publishes the sha256 of the file bytes;
        # for small files it publishes a git blob sha1, which is not comparable.
        sha256 = lfs.get("oid") if isinstance(lfs, dict) else None
        files.append(
            SnapshotFile(
                path=str(item["path"]),
                size=item.get("size"),
                sha256=sha256 if isinstance(sha256, str) and len(sha256) == 64 else None,
            )
        )
    return files


def wanted_snapshot_files(repo: str, revision: str) -> list[SnapshotFile]:
    return [
        entry
        for entry in list_repo_files(repo, revision)
        if entry.path.endswith(".safetensors")
        or entry.path in (CONFIG_FILE, GENERATION_CONFIG_FILE, SAFETENSORS_INDEX_FILE)
        or entry.path in TOKENIZER_FILES
    ]


def verify_snapshot(
    root: Path, repo: str, revision: str, *, require_complete: bool = True
) -> dict[str, Any]:
    """Refuse a snapshot directory whose marker does not name this repo+commit.

    Returns the marker's contents. ``require_complete`` is what ``--skip-download``
    asks for: a directory whose download never finished is not a snapshot, and
    with the network switched off there is no way to finish it.
    """
    marker = root / SNAPSHOT_MARKER
    if not marker.is_file():
        raise SlowQError(
            f"{root} has no {SNAPSHOT_MARKER}, so nothing confirms it holds "
            f"{repo}@{revision[:12]}. SlowQ neither reuses nor deletes a directory it "
            "cannot account for. If this is what an interrupted download by an older "
            "version of this tool left behind, remove that one directory and re-run "
            "the same command; otherwise point --download-dir somewhere fresh, or pass "
            "--source with a directory you prepared yourself."
        )
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SlowQError(f"{marker} is not valid JSON: {exc}") from exc
    if not isinstance(recorded, dict):
        raise SlowQError(f"{marker} does not contain an object")
    for key, expected in (("repo_id", repo), ("revision", revision), ("endpoint", endpoint())):
        found = recorded.get(key)
        if found != expected:
            raise SlowQError(
                f"{marker} records {key}={found!r} but {expected!r} was requested. "
                "This snapshot came from somewhere else; not reusing it."
            )
    if require_complete and not marker_is_complete(recorded):
        raise SlowQError(
            f"{root} is an unfinished download of {repo}@{revision[:12]}: {marker.name} "
            "records it as incomplete. Re-run the same command without --skip-download "
            "to finish it."
        )
    return recorded


def marker_is_complete(recorded: Mapping[str, Any]) -> bool:
    """Does the marker claim every listed file arrived and was checked?

    A marker written before this field existed was only ever written after the
    last download had succeeded, so its absence means complete — an existing
    cache keeps working instead of being fetched all over again.
    """
    return recorded.get("complete", True) is True


def write_snapshot_marker(
    root: Path,
    repo: str,
    revision: str,
    wanted: Sequence[SnapshotFile],
    *,
    complete: bool,
    created_at: str | None = None,
) -> None:
    """Record what this directory is meant to hold, and whether it holds it yet.

    Written *before* the first download as well as after it. The marker is the
    only evidence that a directory is ours, so a run that was interrupted
    without one left a cache that could never be finished: the next identical
    command refused the directory it had started. ``complete=True`` is only ever
    written once every downloaded file has been checked against the listing.
    """
    payload: dict[str, Any] = {
        "repo_id": repo,
        "revision": revision,
        "endpoint": endpoint(),
        "created_at": created_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "complete": complete,
        "files": [
            {"path": entry.path, "size": entry.size, "sha256": entry.sha256}
            for entry in wanted
        ],
    }
    if complete:
        payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    marker = root / SNAPSHOT_MARKER
    # ``.part`` so an interrupted write of the marker itself is cleaned up by
    # reject_stale_files() on the next run, and os.replace so a reader never
    # sees half a JSON document.
    temporary = root / (SNAPSHOT_MARKER + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(marker)


def matches_listing(target: Path, entry: SnapshotFile) -> bool:
    """Is the cached file the one the listing describes?

    Size catches a truncated or half-written file, the LFS sha256 catches a file
    that has the right length and the wrong bytes.
    """
    if not target.is_file():
        return False
    if entry.size is not None and target.stat().st_size != entry.size:
        return False
    return entry.sha256 is None or sha256_file(target) == entry.sha256


def confirm_snapshot_files(root: Path, downloaded: Sequence[SnapshotFile]) -> None:
    """Re-check the files just fetched before the snapshot may call itself complete.

    A download loop that ends without raising is not proof of anything: the
    server can send fewer bytes than the listing promised, and nothing else
    looks. This is the point where "complete" becomes known instead of assumed.
    """
    bad = [entry.path for entry in downloaded if not matches_listing(root / entry.path, entry)]
    if bad:
        shown = ", ".join(bad[:5]) + (" …" if len(bad) > 5 else "")
        raise SlowQError(
            f"{len(bad)} file(s) under {root} still do not match the revision's listing "
            f"after downloading: {shown}. The snapshot stays marked incomplete, so "
            "re-running the same command fetches them again."
        )


def download_snapshot(repo: str, revision: str, cache_root: Path) -> Path:
    """Make sure ``cache_root/<repo>/<revision>`` holds this exact commit.

    Returns the snapshot directory, which is what the conversion then reads and
    what the manifest records as its source.

    An interrupted download is *resumed*, not refused: the marker written before
    the first file says whose directory this is and that it is not finished, and
    every file is checked against the listing on the way in, so a short or
    half-written one is fetched again rather than trusted.
    """
    if not is_commit_sha(revision):
        raise SlowQError(
            f"refusing to cache {repo}@{revision!r}: only a full commit sha "
            "identifies a snapshot unambiguously. Resolve it first."
        )
    root = snapshot_dir(cache_root, repo, revision)
    wanted = wanted_snapshot_files(repo, revision)
    if not wanted:
        raise SlowQError(f"{repo}@{revision[:12]} lists no files SlowQ needs")

    recorded: dict[str, Any] | None = None
    if root.is_dir():
        recorded = verify_snapshot(root, repo, revision, require_complete=False)
        reject_stale_files(root, wanted)
    else:
        root.mkdir(parents=True, exist_ok=True)

    pending = [entry for entry in wanted if not matches_listing(root / entry.path, entry)]
    if not pending and recorded is not None and marker_is_complete(recorded):
        print(
            f"snapshot {repo}@{revision[:12]} -> {root} ({len(wanted)} files, complete)",
            file=sys.stderr,
        )
        return root

    created_at = str(recorded["created_at"]) if recorded and recorded.get("created_at") else None
    print(
        f"snapshot {repo}@{revision[:12]} -> {root} ({len(wanted)} files, "
        f"{'resuming' if recorded is not None else 'downloading'} {len(pending)})",
        file=sys.stderr,
    )
    write_snapshot_marker(
        root, repo, revision, wanted, complete=False, created_at=created_at
    )
    pending_paths = {entry.path for entry in pending}
    for entry in wanted:
        if entry.path not in pending_paths:
            print(f"  {entry.path}: cached", file=sys.stderr)
    for entry in pending:
        target = root / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        http_download(f"{endpoint()}/{repo}/resolve/{revision}/{entry.path}", target)

    confirm_snapshot_files(root, pending)
    write_snapshot_marker(
        root, repo, revision, wanted, complete=True, created_at=created_at
    )
    return root


def reject_stale_files(root: Path, wanted: Sequence[SnapshotFile]) -> None:
    """A leftover shard or index from another revision would be read as truth."""
    expected = {entry.path for entry in wanted} | {SNAPSHOT_MARKER}
    stale = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and str(path.relative_to(root)) not in expected
        and not path.name.endswith(".part")
        and (
            path.suffix == ".safetensors"
            or path.name in (SAFETENSORS_INDEX_FILE, CONFIG_FILE, *TOKENIZER_FILES)
        )
    )
    for path in root.rglob("*.part"):
        path.unlink()  # our own interrupted download, never a user file
    if stale:
        raise SlowQError(
            f"{root} contains file(s) this revision does not list: "
            f"{', '.join(stale[:5])}. They would be read as checkpoint data. "
            "Remove them or use a different --download-dir."
        )



# ---- conversion -------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def mapped(path: Path) -> Iterator[mmap.mmap]:
    """Read-only memory map of one checkpoint shard."""
    with path.open("rb") as handle:
        blob = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        yield blob
    finally:
        try:
            blob.close()
        except BufferError:
            # Only reachable when the body raised: the traceback pins the frame
            # that sliced the mapping, so NumPy is still exporting a buffer from
            # it. Re-raising here would replace the real error with this one, and
            # the mapping is released when the process exits anyway.
            pass


def tensor_sha256_in(source: Path, entry: SafeTensorEntry) -> str:
    """Digest one tensor, read from the shard that actually contains it."""
    with mapped(source / entry.file) as blob:
        return tensor_sha256(entry, blob)


def tied_head_matches_embedding(source: Path, head: SafeTensorEntry, embedding: SafeTensorEntry) -> bool:
    """Are the tied head and the embedding the same tensor?

    Each is read from *its own* shard: in a sharded checkpoint the two usually
    live in different files, and their byte ranges mean nothing outside the file
    they were listed from. Comparing two ranges of one mapping can reject a legal
    checkpoint, or — when the offsets happen to coincide — compare a tensor with
    itself and accept an illegal one.
    """
    if head.shape != embedding.shape or head.dtype != embedding.dtype:
        return False
    return tensor_sha256_in(source, head) == tensor_sha256_in(source, embedding)


def inspect_source(source: Path) -> tuple[ModelConfig, dict[str, SafeTensorEntry], dict[str, tuple[int, ...]]]:
    """Read the checkpoint's config and tensor listing, and check it is complete."""
    config_path = source / CONFIG_FILE
    if not config_path.is_file():
        raise SlowQError(f"{config_path} not found; is {source} an official checkpoint?")
    hf_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = ModelConfig.from_hf_config(hf_config)

    entries = list(iter_safetensors_entries(source))
    by_name = {entry.name: entry for entry in entries}
    if len(by_name) != len(entries):
        raise SlowQError("checkpoint lists the same tensor name in two files")

    # required_tensors() already omits lm_head.weight when the model is tied.
    expected = required_tensors(config)
    missing = [name for name in expected if name not in by_name]
    if missing:
        raise SlowQError(
            "checkpoint is missing required tensor(s): " + ", ".join(missing[:8])
        )
    return config, by_name, expected


def convert(
    source: Path,
    output: Path,
    *,
    repo: str | None = None,
    revision: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the SlowQ model directory and return the manifest.

    The conversion happens in a staging directory beside ``output`` and is
    installed only after it has been read back through the runtime's own loader.
    A failure part-way through therefore leaves ``output`` exactly as it was, and
    a first conversion that fails leaves no directory at all.
    """
    output = Path(output)
    check_output_is_installable(output, overwrite=overwrite)
    config, by_name, expected = inspect_source(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=str(output.parent))
    )
    installed = False
    try:
        manifest = convert_into(
            source,
            staging,
            config=config,
            by_name=by_name,
            expected=expected,
            repo=repo,
            revision=revision,
        )
        validate_converted_model(staging, config)
        install(staging, output, overwrite=overwrite)
        installed = True
    finally:
        if not installed:
            shutil.rmtree(staging, ignore_errors=True)
    return manifest


def check_output_is_installable(output: Path, *, overwrite: bool) -> None:
    """Fail before converting, not after, if ``output`` is not ours to write.

    An absent path and an existing *empty* directory are both writable:
    ``install`` renames the staging directory over either one, and refuses a
    directory that gained a file in between. Only contents are protected.
    """
    if not output.exists():
        return
    if not output.is_dir():
        raise SlowQError(f"--out {output} exists and is not a directory")
    entries = sorted(path.name for path in output.iterdir())
    if not entries or overwrite:
        return
    raise SlowQError(
        f"{output} is not empty ({len(entries)} entries: "
        f"{', '.join(entries[:4])}{'…' if len(entries) > 4 else ''}). A conversion "
        "that fails part-way through would leave a model mixing old and new "
        "weights behind a still-valid manifest, so SlowQ will not write into it. "
        "Pass --overwrite to replace it in one step, or choose a fresh --out."
    )


def convert_into(
    source: Path,
    destination: Path,
    *,
    config: ModelConfig,
    by_name: dict[str, SafeTensorEntry],
    expected: dict[str, tuple[int, ...]],
    repo: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Write a complete, self-consistent model directory into ``destination``."""
    tensor_dir = destination / TENSOR_SUBDIR
    tensor_dir.mkdir(parents=True, exist_ok=True)

    aliases: dict[str, str] = {}
    if config.tie_word_embeddings and TIED_OUTPUT_EMBEDDING_ALIAS in by_name:
        # The checkpoint materialised the tied head anyway. Store one copy and
        # record the alias — but only after proving the bytes really are equal.
        if not tied_head_matches_embedding(
            source, by_name[TIED_OUTPUT_EMBEDDING_ALIAS], by_name[EMBEDDING_TENSOR_NAME]
        ):
            raise ConfigError(
                f"config.json sets tie_word_embeddings=true but "
                f"{TIED_OUTPUT_EMBEDDING_ALIAS} differs from {EMBEDDING_TENSOR_NAME}. "
                "Refusing to alias weights that are not actually shared."
            )
        aliases[TIED_OUTPUT_EMBEDDING_ALIAS] = EMBEDDING_TENSOR_NAME
        print(
            f"  {TIED_OUTPUT_EMBEDDING_ALIAS} == {EMBEDDING_TENSOR_NAME} "
            "(tied; storing one copy)",
            file=sys.stderr,
        )

    stored_names = {name for name in expected if name in by_name}
    written: list[dict[str, Any]] = []
    source_files: dict[str, str] = {}

    grouped: dict[str, list[SafeTensorEntry]] = {}
    for entry in by_name.values():
        grouped.setdefault(entry.file, []).append(entry)

    total = len(stored_names)
    done = 0
    for filename in sorted(grouped):
        path = source / filename
        source_files[filename] = sha256_file(path)
        with mapped(path) as blob:
            for entry in sorted(grouped[filename], key=lambda item: item.name):
                if entry.name not in stored_names:
                    continue
                array = convert_tensor(entry, blob)
                if tuple(array.shape) != expected[entry.name]:
                    raise SlowQError(
                        f"tensor {entry.name!r} has shape {tuple(array.shape)}, "
                        f"expected {expected[entry.name]}"
                    )
                relative = f"{entry.name}.npy"
                np.save(tensor_dir / relative, array)
                written.append(
                    {
                        "name": entry.name,
                        "file": relative,
                        "shape": list(array.shape),
                        "dtype": "float32",
                        "source_dtype": entry.dtype,
                    }
                )
                del array
                done += 1
                print(
                    f"  [{done}/{total}] {entry.name} {tuple(entry.shape)} "
                    f"{entry.dtype} -> float32",
                    file=sys.stderr,
                    flush=True,
                )

    written.sort(key=lambda item: item["name"])

    manifest = {
        "format_version": SUPPORTED_FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "tools/convert_model.py",
        "compute_dtype": "float32",
        "source": {
            "repo_id": repo or f"local:{source}",
            "revision": revision,
            "local_path": str(source),
            "files": [
                {"name": name, "sha256": digest} for name, digest in sorted(source_files.items())
            ],
        },
        "weight_layout": "[out_features, in_features]; runtime computes x @ W.T",
        "tensors": written,
        "aliases": aliases,
        "tokenizer_files": [
            name for name in TOKENIZER_FILES if (source / name).is_file()
        ],
    }

    for name in (CONFIG_FILE, GENERATION_CONFIG_FILE, *TOKENIZER_FILES):
        source_file = source / name
        if source_file.is_file():
            shutil.copyfile(source_file, destination / name)
        elif name in (CONFIG_FILE, "tokenizer.json", "tokenizer_config.json"):
            raise SlowQError(f"{source_file} is required but missing")

    (destination / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_converted_model(directory: Path, config: ModelConfig) -> None:
    """Load the directory back the way inference will, before it becomes a model.

    ``check_complete`` alone is not enough: it only reads the manifest, so a
    directory whose ``.npy`` files are missing, truncated or mis-shaped would
    pass. Mapping every tensor here is cheap (only the headers are read) and
    catches exactly that.
    """
    from slowq.tokenizer import Tokenizer

    weights = Weights.from_directory(directory)
    weights.check_complete(config)
    for spec in weights.iter_specs():
        weights.get(spec.name)
    on_disk = {path.name for path in (directory / TENSOR_SUBDIR).iterdir()}
    declared = {spec.file for spec in weights.iter_specs()}
    if on_disk - declared:
        raise SlowQError(
            f"{directory}/tensors holds files the manifest does not declare: "
            f"{', '.join(sorted(on_disk - declared)[:4])}"
        )
    Tokenizer.from_directory(directory)


def install(staging: Path, output: Path, *, overwrite: bool) -> None:
    """Move the validated staging directory into place.

    Both steps are renames inside one directory, so the model directory is
    either the old one or the new one — never a mixture, and never absent for
    longer than one rename.
    """
    if not overwrite:
        # Let the rename itself be the check: it refuses a non-empty directory,
        # so this cannot drift away from check_output_is_installable() the way an
        # exists() test did. An --out that was already there but empty is still
        # ours to take; one that gained a file while converting is not, and the
        # failure is atomic, so nothing in it is touched.
        try:
            staging.replace(output)
        except OSError as exc:
            entries = (
                sorted(path.name for path in output.iterdir()) if output.is_dir() else []
            )
            listing = (
                f" ({len(entries)} entries: {', '.join(entries[:4])}"
                f"{'…' if len(entries) > 4 else ''})"
                if entries
                else ""
            )
            raise SlowQError(
                f"{output} is not empty{listing} and --overwrite was not given, so the "
                f"converted model was not installed ({exc.strerror}). Nothing in it was "
                "changed. Pass --overwrite to replace it in one step, or choose a fresh "
                "--out."
            ) from exc
        return
    if not output.exists():
        staging.replace(output)
        return
    backup = output.with_name(f"{output.name}.replaced-{os.getpid()}")
    output.replace(backup)
    try:
        staging.replace(output)
    except BaseException:
        backup.replace(output)  # put the previous model back
        raise
    shutil.rmtree(backup, ignore_errors=True)



def summarise(manifest: dict[str, Any], output: Path) -> str:
    tensors = manifest["tensors"]
    elements = sum(int(np.prod(t["shape"])) for t in tensors)
    return (
        f"wrote {output}\n"
        f"  tensors      {len(tensors)} stored, {len(manifest['aliases'])} shared via alias\n"
        f"  parameters   {elements:,} unique\n"
        f"  fp32 size    {elements * 4 / 2**30:.2f} GiB logical\n"
        f"  revision     {manifest['source'].get('revision')}\n"
        f"  aliases      {manifest['aliases']}"
    )


# ---- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an official Qwen3 checkpoint to SlowQ's offline FP32 format.",
    )
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument(
        "--repo",
        default=None,
        help=f"Hugging Face repo id to download (default: {DEFAULT_REPO})",
    )
    origin.add_argument(
        "--source",
        type=Path,
        default=None,
        help="local directory that already holds the official files",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help=f"pin a commit sha (default: {DEFAULT_REVISION[:12]}… for the default repo)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="cache root for downloaded snapshots; the snapshot itself lands in "
        "<cache>/<repo>/<commit> (default: <out>-hf)",
    )
    parser.add_argument("--out", type=Path, required=True, help="output model directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing --out directory. The new model is built and "
        "validated beside it first, so a failure keeps the old one usable.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="reuse an existing snapshot without contacting the network",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output: Path = args.out

    try:
        if args.source is not None:
            source: Path = args.source
            repo, revision = None, args.revision
            if not source.is_dir():
                raise SlowQError(f"--source {source} is not a directory")
        else:
            repo = args.repo or DEFAULT_REPO
            revision = args.revision or (
                DEFAULT_REVISION if repo == DEFAULT_REPO else None
            )
            cache_root = args.download_dir or output.parent / f"{output.name}-hf"
            if args.skip_download:
                if not is_commit_sha(revision):
                    raise SlowQError(
                        "--skip-download cannot guess where a snapshot lives: pass "
                        f"--revision as a full {COMMIT_SHA_LENGTH}-character commit sha"
                    )
                source = snapshot_dir(cache_root, repo, str(revision))
                if not source.is_dir():
                    raise SlowQError(
                        f"{source} does not exist; drop --skip-download to fetch it"
                    )
                verify_snapshot(source, repo, str(revision))
            else:
                # Pin before downloading: the manifest must record the exact
                # commit the bytes came from, not a floating branch name.
                if not is_commit_sha(revision):
                    revision = resolve_revision(repo, revision)
                source = download_snapshot(repo, revision, cache_root)

        started = time.perf_counter()
        print(f"converting {source} -> {output}", file=sys.stderr)
        manifest = convert(
            source, output, repo=repo, revision=revision, overwrite=args.overwrite
        )
        elapsed = time.perf_counter() - started
        print(summarise(manifest, output), file=sys.stderr)
        print(f"  took {elapsed:.1f} s", file=sys.stderr)
        print(
            "\nnext: python -m slowq inspect --model "
            f"{output}\n      python -m slowq generate --model {output} --prompt 你好",
            file=sys.stderr,
        )
    except SlowQError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
