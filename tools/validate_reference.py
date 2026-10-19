#!/usr/bin/env python3
"""Validate SlowQ's numerics against the Transformers Qwen3 reference.

The two implementations run in *separate processes* and exchange result files,
so a 2.4 GiB FP32 torch model and the NumPy model are never resident at the
same time::

    python tools/validate_reference.py numpy    --model models/Qwen3-0.6B-slowq
    python tools/validate_reference.py ref      --hf-dir models/Qwen3-0.6B-hf
    python tools/validate_reference.py compare
    python tools/validate_reference.py all ...   # one process, both models (needs RAM)

What is compared, using *identical token IDs* on both sides:

1. tokenizer      rendered chat text and token IDs vs ``AutoTokenizer``
2. weights        a sample of converted tensors must digest identically to the
                  FP32 upcast of the official BF16 checkpoint
3. full prefill   logits at the last position + selected layer hidden states
4. chunked prefill same quantities, produced by two forwards through a cache
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

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "validation"

#: Bumped whenever the recorded artifacts gain, lose or redefine a field. An
#: older capture is then rejected outright instead of passing checks it never
#: contained the data for.
ARTIFACT_FORMAT = 2

#: Written by the ``ref`` side only. The NumPy side must never produce it.
REFERENCE_FILENAME = "reference.json"

#: Declared up front so results cannot be graded against a threshold chosen
#: after seeing them.
#:
#: Rationale: FP32 unit round-off is 6e-8. A 28-layer residual network with
#: 1024-3072 wide dot products and a softmax amplifies that to roughly 1e-5
#: relative on the final logits, and the two sides use different BLAS kernels
#: (Accelerate vs torch's) with different summation orders. We allow ~10x that.
TOLERANCES: dict[str, float] = {
    "rel_l2_logits": 1e-3,
    "max_abs_logits": 5e-2,
    "rel_l2_hidden": 1e-3,
    "max_abs_hidden": 5e-2,
    "rel_l2_weights": 0.0,   # conversion must be exact: bf16 -> fp32 is a bit shift
    "max_abs_weights": 0.0,
}

ARGMAX_MUST_MATCH = True

SAMPLES: list[dict[str, Any]] = [
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

#: Which layers to capture. First, middle and last catch errors that only show
#: up after accumulation as well as errors present from layer 0.
CAPTURED_LAYERS = (0, 13, 27)
CAPTURED_POSITIONS = (0, -1)
WEIGHT_CHECK_SAMPLE = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.13.self_attn.k_proj.weight",
    "model.layers.27.mlp.down_proj.weight",
    "model.norm.weight",
)

#: Structural parameters both sides must agree on before any number is graded.
ARCHITECTURE_FIELDS = (
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
    """Compare two arrays, refusing to grade anything that is not comparable."""
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

    return Comparison(
        key=key,
        sample=sample,
        kind=kind,
        max_abs=float(np.max(np.abs(difference))),
        mean_abs=float(np.mean(np.abs(difference))),
        rel_l2=rel_l2,
        argmax_match=(
            None
            if flat_ours.size < 2
            else bool(int(np.argmax(flat_ours)) == int(np.argmax(flat_theirs)))
        ),
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


def architecture_fingerprint(source: Any) -> dict[str, Any]:
    """Structural parameters, normalised so 1000000 and 1000000.0 agree.

    Both sides report these: a reference captured from a different revision, or
    from a differently configured checkpoint, is then rejected before a single
    number is graded.
    """
    fingerprint: dict[str, Any] = {}
    for name in ARCHITECTURE_FIELDS:
        value = source[name] if isinstance(source, Mapping) else getattr(source, name)
        fingerprint[name] = _json_number(value)
    return fingerprint


def capture_config(
    num_hidden_layers: int, *, prefill_chunk_size: int, decode_steps: int
) -> dict[str, Any]:
    """What a capture covers, recorded identically by both sides.

    ``compare`` recomputes this from the *current* constants, so changing
    ``CAPTURED_LAYERS`` in the code without re-capturing the reference fails
    instead of quietly comparing a subset.
    """
    return {
        "artifact_format": ARTIFACT_FORMAT,
        "digest_spec": WEIGHT_DIGEST_SPEC,
        "sample_names": [sample["name"] for sample in SAMPLES],
        "layers": [layer for layer in CAPTURED_LAYERS if layer < num_hidden_layers],
        "positions_full": [int(position) for position in CAPTURED_POSITIONS],
        "positions_cached": [-1],
        "prefill_chunk_size": int(prefill_chunk_size),
        "decode_steps": int(decode_steps),
    }


def expected_numeric_keys(capture: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """Every numeric key a complete capture must contain -> (sample, kind).

    Derived from the capture configuration, never from the intersection of the
    two files: an intersection only reports on what both sides happened to
    write, so a truncated or empty reference looks like a clean run.
    """
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


def expected_sample_names(capture: Mapping[str, Any] | None) -> list[str]:
    """Every tokenizer sample a complete capture must record.

    The same rule as ``expected_numeric_keys``: the capture configuration says
    what was supposed to be captured, so a sample *both* sides omitted is a
    missing record rather than one less thing to compare. Falls back to the
    current protocol when the artifacts do not describe a capture at all — that
    run has already failed, but it should still say what is missing.
    """
    if isinstance(capture, Mapping) and capture.get("sample_names"):
        return [str(name) for name in capture["sample_names"]]
    return [str(sample["name"]) for sample in SAMPLES]


def expected_weight_names() -> tuple[str, ...]:
    """Every converted weight a complete capture must digest.

    Both capture sides walk ``WEIGHT_CHECK_SAMPLE``, so anything absent from
    either artifact means a weight this protocol requires went unverified.
    """
    return tuple(WEIGHT_CHECK_SAMPLE)


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


def run_numpy_side(args: argparse.Namespace) -> int:
    from slowq.engine import Engine
    from slowq.model import DebugProbe
    from slowq.sampling import Sampler, SamplingParams

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    engine = Engine.from_directory(
        args.model, max_context=args.max_context, prefill_chunk_size=args.prefill_chunk_size
    )
    greedy = Sampler(SamplingParams())
    captures = {"layer_output", "final_norm", "logits"}
    config = engine.config

    payload: dict[str, Any] = {
        "samples": [],
        "model_directory": str(args.model),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "side": "numpy",
        "architecture": architecture_fingerprint(config),
        "source": {
            "repo_id": engine.model.weights.manifest.source.get("repo_id"),
            "revision": engine.model.weights.manifest.source_revision,
            "files": {
                str(entry["name"]): entry.get("sha256")
                for entry in engine.model.weights.manifest.source.get("files", [])
            },
        },
        "capture": capture_config(
            config.num_hidden_layers,
            prefill_chunk_size=args.prefill_chunk_size,
            decode_steps=args.decode_steps,
        ),
        "tokenizer": [],
        "weight_digests": {},
    }
    blobs: dict[str, np.ndarray] = {}

    for sample in SAMPLES:
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

        # 1. full prefill in one forward
        cache = engine.model.new_cache(min(args.max_context, len(prompt_ids) + args.decode_steps + 8))
        probe = DebugProbe.for_names(captures, CAPTURED_LAYERS, CAPTURED_POSITIONS)
        full = engine.model.forward(prompt_ids, cache, probe=probe)
        blobs[f"{name}.full_prefill.logits"] = full.logits
        for key, value in probe.records.items():
            if key.startswith("layer") or key.startswith("final_norm"):
                blobs[f"{name}.full_prefill.{key}"] = value
        record["full_prefill_argmax"] = int(greedy.sample(full.logits))
        del cache

        # 2. chunked prefill: same tokens, several forwards through one cache
        cache = engine.model.new_cache(min(args.max_context, len(prompt_ids) + args.decode_steps + 8))
        probe = DebugProbe.for_names(captures, CAPTURED_LAYERS, (-1,))
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
            step_probe = DebugProbe.for_names(captures, CAPTURED_LAYERS, (-1,))
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

    # 4. converted-weight provenance. Digests of whole tensors, so the gate
    # covers every element without shipping 593 MiB of embedding table around.
    for tensor_name in WEIGHT_CHECK_SAMPLE:
        array = engine.model.weights.get(tensor_name)
        payload["weight_digests"][tensor_name] = tensor_digest(array)
        print(f"[numpy] digested {tensor_name}", file=sys.stderr)

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


# ---- reference side ---------------------------------------------------------


def run_reference_side(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    config = model.config
    print(
        f"[ref] transformers {transformers.__version__}, torch {torch.__version__}, "
        f"{config.num_hidden_layers} layers, eager attention, fp32",
        file=sys.stderr,
    )

    blobs: dict[str, np.ndarray] = {}

    # 1. tokenizer content — recorded as data, never as a verdict. ``compare``
    # decides whether it agrees with the current SlowQ output.
    tokenizer_reference: list[dict[str, Any]] = []
    for sample in payload["samples"]:
        rendered = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        reference_ids = [int(token) for token in tokenizer(rendered, add_special_tokens=False).input_ids]
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

    # 2. weight digests of the official checkpoint, taken over the same byte
    # representation the converted tensors are digested with.
    weight_digests: dict[str, Any] = {}
    state_dict = model.state_dict()
    for tensor_name in WEIGHT_CHECK_SAMPLE:
        reference_tensor = state_dict.get(tensor_name)
        if reference_tensor is None and tensor_name == "lm_head.weight":
            reference_tensor = state_dict.get("model.embed_tokens.weight")
        if reference_tensor is None:
            raise SystemExit(
                f"{tensor_name} is absent from the reference checkpoint at {hf_dir}"
            )
        array = reference_tensor.detach().to(torch.float32).numpy()
        weight_digests[tensor_name] = tensor_digest(array)
        print(f"[ref] digested {tensor_name}", file=sys.stderr)
        del array
    del state_dict

    # 3. which bytes the reference actually loaded. The converter records the
    # same digests in its manifest, so ``compare`` can tie the two together
    # without a network call and without trusting a revision string.
    checkpoint_files = {
        path.name: file_digest(path) for path in sorted(hf_dir.glob("*.safetensors"))
    }
    if not checkpoint_files:
        raise SystemExit(f"no *.safetensors under {hf_dir}; nothing to validate against")

    layer_indices = [i for i in CAPTURED_LAYERS if i < config.num_hidden_layers]

    # ``output_hidden_states`` cannot be used for this comparison: HF appends the
    # hidden state *before* each layer and then, after the final norm, appends the
    # post-norm tensor. So the last layer's raw output is never exposed and
    # ``hidden_states[i + 1]`` silently means "post-norm" for i == num_layers-1.
    # Forward hooks capture the decoder layers' actual outputs instead.
    captured: dict[str, np.ndarray] = {}

    def hook(name: str):
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[name] = hidden.detach().to(torch.float32).numpy().copy()

        return _hook

    handles = [
        model.model.layers[layer].register_forward_hook(hook(f"layer{layer}"))
        for layer in layer_indices
    ]
    handles.append(model.model.norm.register_forward_hook(hook("final_norm")))

    def record_positions(prefix: str, logits: torch.Tensor, positions: Iterable[int]) -> None:
        blobs[f"{prefix}.logits"] = logits[0, -1, :].detach().to(torch.float32).numpy()
        for position in positions:
            for layer in layer_indices:
                tensor = captured[f"layer{layer}"]
                blobs[f"{prefix}.layer{layer}.layer_output[{position}]"] = tensor[0][position]
            blobs[f"{prefix}.final_norm[{position}]"] = captured["final_norm"][0][position]

    with torch.no_grad():
        for sample in payload["samples"]:
            name = sample["name"]
            prompt_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long)

            # 4. full prefill
            output = model(prompt_ids, use_cache=False, return_dict=True)
            record_positions(f"{name}.full_prefill", output.logits, CAPTURED_POSITIONS)

            # 5. chunked prefill through a cache
            chunk = sample["prefill_chunk_size"]
            past = None
            for start in range(0, prompt_ids.shape[1], chunk):
                piece = prompt_ids[:, start : start + chunk]
                output = model(piece, past_key_values=past, use_cache=True, return_dict=True)
                past = output.past_key_values
            record_positions(f"{name}.chunked_prefill", output.logits, (-1,))

            # 6. cached decode, replaying SlowQ's forced token path
            for step, step_record in enumerate(sample["decode_steps"]):
                token = torch.tensor([[step_record["input_token"]]], dtype=torch.long)
                output = model(token, past_key_values=past, use_cache=True, return_dict=True)
                past = output.past_key_values
                record_positions(f"{name}.decode{step}", output.logits, (-1,))

            print(f"[ref] {name}: done", file=sys.stderr)

    for handle in handles:
        handle.remove()

    reference = {
        "side": "reference",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hf_dir": str(hf_dir),
        "architecture": architecture_fingerprint(config),
        "source": {
            "repo_id": None,
            "revision": snapshot_revision(hf_dir),
            "files": checkpoint_files,
        },
        "capture": capture_config(
            int(config.num_hidden_layers),
            prefill_chunk_size=payload["samples"][0]["prefill_chunk_size"],
            decode_steps=len(payload["samples"][0]["decode_steps"]),
        ),
        "environment": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "attn_implementation": "eager",
            "dtype": "float32",
            "threads": args.threads,
        },
        "tokenizer": tokenizer_reference,
        "weight_digests": weight_digests,
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
    return 0


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
    required = capture_config(
        int(architecture["num_hidden_layers"]),
        prefill_chunk_size=int(ours_capture.get("prefill_chunk_size", 0)),
        decode_steps=int(ours_capture.get("decode_steps", 0)),
    )
    for label, key in (
        ("artifact_format", "artifact_format"),
        ("sample_names", "sample_names"),
        ("layers", "layers"),
        ("positions (full)", "positions_full"),
        ("positions (cached)", "positions_cached"),
        ("prefill_chunk_size", "prefill_chunk_size"),
        ("decode_steps", "decode_steps"),
    ):
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
        print(f"  {'architecture':<26} {ours_arch.get('num_hidden_layers')} layers, "
              f"vocab {ours_arch.get('vocab_size')} [ok]")

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


def check_tokenizer(
    ours: Mapping[str, Any],
    theirs: Mapping[str, Any],
    problems: list[str],
    capture: Mapping[str, Any] | None = None,
) -> None:
    """Compare *content* recorded by the reference against this run's output."""
    print("\n[tokenizer] rendered text and token IDs vs AutoTokenizer")
    ours_entries = {entry["name"]: entry for entry in ours.get("tokenizer", [])}
    theirs_entries = {entry["name"]: entry for entry in theirs.get("tokenizer", [])}
    if not theirs_entries:
        problems.append(
            "the reference records no tokenizer content; these artifacts predate the "
            "content-based check — re-capture the reference"
        )
        return
    expected = set(expected_sample_names(capture))
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
            problems.append(
                f"tokenizer sample {name!r}: SlowQ renders different chat text than the "
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
    """Re-digest the current converted weights against the recorded digests."""
    print("\n[weights] converted FP32 vs BF16 upcast of the official checkpoint")
    ours_digests = ours.get("weight_digests") or {}
    theirs_digests = theirs.get("weight_digests") or {}
    if not theirs_digests:
        problems.append(
            "the reference records no weight digests; these artifacts predate the "
            "digest-based check — re-capture the reference"
        )
        return
    expected = set(expected_weight_names())
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
        print(
            f"  {name:<44} shape={ours_entry.get('shape')} dtype={ours_entry.get('dtype')} "
            f"elements={ours_entry.get('elements')} [{status}]"
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
    print("\n[internal] SlowQ full prefill vs chunked prefill")
    for name in capture["sample_names"]:
        full, chunked = f"{name}.full_prefill.logits", f"{name}.chunked_prefill.logits"
        if full not in ours.files or chunked not in ours.files:
            problems.append(f"sample {name}: cannot compare full vs chunked prefill, a side is missing")
            continue
        try:
            comparison = compare_arrays(ours[full], ours[chunked], f"{name} full-vs-chunked")
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
        help="converted SlowQ model directory (numpy side)",
    )
    parser.add_argument(
        "--hf-dir",
        type=Path,
        default=Path("models/Qwen3-0.6B-hf"),
        help="official checkpoint directory (ref side)",
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--prefill-chunk-size", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
