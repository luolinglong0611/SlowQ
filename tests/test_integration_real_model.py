"""Real-weight integration tests for the ``text-generation`` task.

Enable them by naming a converted directory::

    SLOWQ_MODEL_DIR=models/SmolLM2-135M-Instruct-slowq python -m pytest -m integration

Every converted chat model runs the same module: nothing here is specific to one
checkpoint, so the assertions hold for Qwen3, Qwen2.5 and Llama alike and a new
family only has to survive them rather than be written into them. The module
skips itself when the directory's manifest records a different task — an
embedding or an image+text checkpoint has its own module, because its invariants
are different.

They need a converted checkpoint (``tools/convert_model.py``) and, for the
reference comparison, a capture made beforehand by ``tools/validate_reference.py
ref`` — both ``artifacts/validation/<model-name>/ref.npz`` (the numbers) and
``.../reference.json`` (the reference tokenizer's text and IDs, weight digests,
capture configuration). The torch half never runs inside pytest, so these tests
stay usable in an environment where torch is not installed.

Everything here is deliberately *not* about whether the answer reads well. The
spec is explicit that "it produces Chinese" is not evidence of correctness, so
the assertions are about invariants that a numerical bug would break:
determinism, chunked-vs-full prefill agreement, cached-vs-uncached agreement,
explicit refusal on overflow, and agreement with the Transformers reference
inside tolerances that were declared before they were measured.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import real_models
from slowq.config import QWEN3_06B_REFERENCE, ModelConfig
from slowq.engine import Engine
from slowq.errors import ContextOverflowError
from slowq.sampling import SamplingParams
from slowq.weights import (
    LM_HEAD_TENSOR_NAME,
    embedding_tensor_name,
    output_embedding_name,
)

MAX_NEW_TOKENS = 48

TASK = "text-generation"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        (reason := real_models.skip_reason(TASK)) is not None,
        reason=reason or "",
    ),
]


@pytest.fixture(scope="module")
def model_dir() -> Path:
    directory = real_models.selected_model_dir()
    assert directory is not None  # guaranteed by the module-level skip
    return directory


@pytest.fixture(scope="module")
def engine(model_dir: Path) -> Engine:
    """One engine for the whole module.

    Sharing it is safe because the engine resets the KV cache at the *start* of
    every request, and memory-mapping gigabytes of FP32 weights once instead of
    per test is the difference between a usable and an unbearable test run.
    """
    return Engine.from_directory(model_dir, max_context=512, prefill_chunk_size=64)


# --------------------------------------------------------------------------- #
# the converted checkpoint
# --------------------------------------------------------------------------- #


def test_the_loaded_config_is_the_one_the_checkpoint_declares(
    engine: Engine, model_dir: Path
):
    """Re-read ``config.json`` and require every field to survive the round trip.

    Not a tautology: the engine's copy went through the manifest's recorded task,
    the family lookup and ``_resolve_head_dim``, each of which could have fallen
    back to a Qwen3-shaped default instead of reading the file.
    """
    config = engine.config
    reread = ModelConfig.from_json_file(model_dir / "config.json", task=config.task)
    for field in fields(ModelConfig):
        if field.name == "raw":
            continue
        assert getattr(reread, field.name) == getattr(config, field.name), field.name


def test_the_live_config_agrees_with_the_reference_reading_of_the_same_file(
    engine: Engine, model_dir: Path
):
    """Two independent readings of one ``config.json`` must describe one model.

    ``architecture_fingerprint`` restates what SlowQ loaded; the reference side
    derived its copy from the official checkpoint using the modelling code's own
    expressions (``config.attention_bias`` for Llama, the hard-coded bias of
    ``Qwen2Attention``, ``head_dim`` versus ``hidden_size // heads``). They agree
    only if the family's rules were applied rather than defaulted, which is the
    check that would catch a Qwen2.5 loaded as a Qwen3 or a Llama given a Q/K
    norm it does not have.
    """
    recorded = real_models.artifacts_dir(model_dir) / "reference.json"
    if not recorded.is_file():
        pytest.skip(f"no recorded reference architecture in {recorded}")
    tool = real_models.load_validation_tool()
    assert tool.architecture_fingerprint(engine.config) == real_models.read_json(
        recorded
    )["architecture"]


def test_qwen3_06b_still_has_its_v01_parameters(engine: Engine, model_dir: Path):
    """The v0.1 target model must not have drifted while families were added."""
    source = real_models.read_json(model_dir / "manifest.json").get("source") or {}
    identity = " ".join(str(source.get(key) or "") for key in ("repo_id", "local_path"))
    if "Qwen3-0.6B" not in identity:
        pytest.skip(f"this checkpoint is not Qwen/Qwen3-0.6B (source: {identity!r})")
    for field, expected in QWEN3_06B_REFERENCE.items():
        assert getattr(engine.config, field) == expected, field


def test_the_weights_are_complete_and_a_tied_head_is_stored_once(
    engine: Engine, model_dir: Path
):
    config = engine.config
    weights = engine.model.weights
    weights.check_complete(config)
    manifest = real_models.read_json(model_dir / "manifest.json")
    assert dict(weights.aliases) == dict(manifest["aliases"])
    assert set(weights.dtype_names) == {"float32"}

    embedding = embedding_tensor_name(config)
    head = LM_HEAD_TENSOR_NAME
    if config.tie_word_embeddings:
        # Tied: the head is an alias, so its numbers exist exactly once on disk.
        assert output_embedding_name(config) == embedding
        assert weights.aliases[head] == embedding
        assert head not in weights.manifest.tensors
    else:
        assert output_embedding_name(config) == head
        assert head in weights.manifest.tensors
        assert head not in weights.aliases
        assert not np.array_equal(weights.get(head), weights.get(embedding))

    # The manifest's shapes must describe the bytes that were actually written;
    # ``deduplicated_bytes`` is what the metrics report as the model's size.
    tensor_dir = model_dir / "tensors"
    on_disk = sum(
        np.load(tensor_dir / spec.file, mmap_mode="r").nbytes
        for spec in weights.manifest.tensors.values()
    )
    assert on_disk == weights.deduplicated_bytes()


def test_the_kv_cache_is_sized_from_the_real_config(engine: Engine):
    """``head_dim`` has to be the checkpoint's own, not ``hidden_size // heads``.

    The cache is sized from ``num_key_value_heads * head_dim``, so a derived
    head_dim would silently allocate the wrong number of bytes on every model
    where the two differ — Qwen3-0.6B (16 * 128 != 1024) and SmolVLM's decoder
    (9 * 64 != 576) among them.
    """
    config = engine.config
    weights = engine.model.weights
    prefix = f"{config.tensor_prefix}.layers.0.self_attn"
    assert weights.get(f"{prefix}.q_proj.weight").shape == (
        config.num_attention_heads * config.head_dim,
        config.hidden_size,
    )
    assert weights.get(f"{prefix}.k_proj.weight").shape == (
        config.num_key_value_heads * config.head_dim,
        config.hidden_size,
    )

    per_token = config.kv_bytes_per_token()
    assert per_token == (
        2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 4
    )
    assert engine.kv_cache.allocated_bytes() == per_token * engine.max_context


def test_loading_never_reaches_the_network(model_dir: Path, monkeypatch):
    """``from_directory`` must work offline; a socket use would be a bug."""
    real_models.block_network(monkeypatch)
    assert Engine.from_directory(model_dir, max_context=64) is not None


def test_the_cli_generates_offline(model_dir: Path, monkeypatch, capsys):
    """The whole user-facing path, with the network unplugged."""
    from slowq import cli

    real_models.block_network(monkeypatch)
    code = cli.main(
        [
            "generate",
            "--model",
            str(model_dir),
            "--prompt",
            "Reply with the single word: offline",
            "--max-new-tokens",
            "8",
            "--max-context",
            "128",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.strip()


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


def test_an_answer_is_the_tokenizers_own_decoding_of_the_returned_ids(engine: Engine):
    """Not a quality check — a check that incremental decoding is lossless.

    A model that produced fluent Chinese with a broken byte-level decoder would
    still fail here, and a correct decoder that produced nonsense would pass.

    What must *not* be asserted is the absence of U+FFFD. Generation stops at
    ``max_new_tokens``, and stopping in the middle of a multi-byte character
    leaves the final token's bytes incomplete: Qwen3-1.7B answers the emoji
    question with a 3-byte lead plus one continuation byte at token 48 of 48, and
    a one-shot decode of the same IDs renders the same replacement character.
    That is the model's output, not a decoding bug, so the invariant is agreement
    between the two decoders rather than a character blacklist.
    """
    for messages in (
        [{"role": "user", "content": "用一个 emoji 表示开心 🎉，然后解释它的 Unicode 名称。"}],
        [{"role": "user", "content": "Reply in English: what does the turtle emoji mean?"}],
    ):
        result = engine.generate(messages, max_new_tokens=MAX_NEW_TOKENS)
        assert result.text.strip(), "the model produced no visible text"
        assert result.text == engine.tokenizer.decode(list(result.token_ids))
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
    real_models.assert_same_numerics(result.logits, expected.logits, f"chunk={chunk} logits")
    real_models.assert_same_numerics(
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

    real_models.assert_same_numerics(logits, expected, "cached decode vs one long prefill")


def test_repeating_a_request_reuses_a_reset_cache(engine: Engine):
    messages = [{"role": "user", "content": "Say hello."}]
    first = engine.generate(messages, max_new_tokens=8)
    assert engine.kv_cache.high_water_mark > 0
    second = engine.generate(messages, max_new_tokens=8)
    assert second.token_ids == first.token_ids
    assert engine.kv_cache.length == len(first.prompt_token_ids) + len(first.token_ids) - 1
