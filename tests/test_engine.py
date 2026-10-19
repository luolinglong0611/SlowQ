"""Engine: the generation loop, stop conditions, streaming, and request isolation."""

from __future__ import annotations

import pytest

from slowq import Engine
from slowq.engine import DEFAULT_MAX_NEW_TOKENS, StopReason
from slowq.errors import ContextOverflowError, UnsupportedInputError
from slowq.sampling import SamplingParams

USER_TURN = [{"role": "user", "content": "你好🐢"}]


@pytest.fixture
def stop_ids(tiny_engine: Engine) -> list[int]:
    return sorted(tiny_engine.stop_ids)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_from_directory_loads_everything_offline(tiny_model_dir):
    engine = Engine.from_directory(tiny_model_dir, max_context=32)
    assert engine.config.num_hidden_layers == 2
    assert engine.max_context == 32
    assert engine.load_seconds == engine.load_seconds  # not NaN
    assert engine.load_seconds >= 0.0
    assert engine.stop_ids


def test_max_context_is_validated_against_the_model(tiny_model_dir):
    with pytest.raises(Exception, match="max_position_embeddings"):
        Engine.from_directory(tiny_model_dir, max_context=10_000)


def test_a_smaller_max_context_shrinks_the_kv_cache(tiny_model_dir):
    small = Engine.from_directory(tiny_model_dir, max_context=16)
    large = Engine.from_directory(tiny_model_dir, max_context=64)
    assert large.kv_cache.allocated_bytes() == 4 * small.kv_cache.allocated_bytes()


def test_prefill_chunk_size_must_be_positive(tiny_model_dir):
    with pytest.raises(ValueError, match="prefill_chunk_size"):
        Engine.from_directory(tiny_model_dir, max_context=32, prefill_chunk_size=0)


def test_a_missing_model_directory_is_reported(tmp_path):
    with pytest.raises(UnsupportedInputError, match="config.json"):
        Engine.from_directory(tmp_path / "nothing")


# --------------------------------------------------------------------------- #
# prompt preparation
# --------------------------------------------------------------------------- #


def test_generate_accepts_chat_messages(tiny_engine: Engine):
    result = tiny_engine.generate(USER_TURN, max_new_tokens=3)
    assert result.prompt_token_ids == tuple(tiny_engine.encode_messages(USER_TURN))
    assert len(result.token_ids) == 3


def test_generate_accepts_a_raw_prompt(tiny_engine: Engine):
    result = tiny_engine.generate(prompt="hi", max_new_tokens=2)
    expected = tiny_engine.encode_messages([{"role": "user", "content": "hi"}])
    assert list(result.prompt_token_ids) == expected


def test_generate_requires_exactly_one_of_messages_or_prompt(tiny_engine: Engine):
    with pytest.raises(ValueError, match="exactly one"):
        tiny_engine.generate(max_new_tokens=1)
    with pytest.raises(ValueError, match="exactly one"):
        tiny_engine.generate(USER_TURN, prompt="hi", max_new_tokens=1)


def test_generate_rejects_a_non_string_prompt(tiny_engine: Engine):
    with pytest.raises(UnsupportedInputError, match="must be a string"):
        tiny_engine.generate(prompt=42, max_new_tokens=1)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]}],
        [{"role": "user", "content": [{"type": "image_url", "url": "http://x"}]}],
        [{"role": "tool", "content": "result"}],
    ],
)
def test_generate_rejects_input_v01_cannot_represent(tiny_engine: Engine, messages):
    with pytest.raises(UnsupportedInputError):
        tiny_engine.generate(messages, max_new_tokens=1)


def test_generate_text_skips_the_chat_template(tiny_engine: Engine):
    result = tiny_engine.generate_text("raw text", max_new_tokens=1)
    assert list(result.prompt_token_ids) == tiny_engine.tokenizer.encode("raw text")


def test_generate_ids_never_touches_the_tokenizer(tiny_engine: Engine):
    result = tiny_engine.generate_ids([1, 2, 3, 4], max_new_tokens=2)
    assert result.prompt_token_ids == (1, 2, 3, 4)
    assert len(result.token_ids) == 2


def test_render_and_encode_messages_agree(tiny_engine: Engine):
    assert tiny_engine.encode_messages(USER_TURN) == tiny_engine.tokenizer.encode(
        tiny_engine.render(USER_TURN)
    )


# --------------------------------------------------------------------------- #
# stop conditions
# --------------------------------------------------------------------------- #


def test_generation_stops_on_max_new_tokens(tiny_engine: Engine):
    result = tiny_engine.generate(USER_TURN, max_new_tokens=5)
    assert result.stop_reason is StopReason.MAX_NEW_TOKENS
    assert not result.stopped_on_eos
    assert len(result.token_ids) == 5


def test_generation_stops_on_eos(
    tiny_engine: Engine, stop_ids, scripted_sampler, monkeypatch
):
    eos = stop_ids[0]
    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler([7, 8, eos, 9, 10]))
    result = tiny_engine.generate(USER_TURN, max_new_tokens=10)
    assert result.stop_reason is StopReason.EOS
    assert result.stopped_on_eos
    # The stop token is part of the id sequence but is never shown as text.
    assert result.token_ids[-1] == eos
    assert len(result.token_ids) == 3


def test_the_stop_token_is_not_rendered_into_the_text(
    tiny_engine: Engine, stop_ids, scripted_sampler, monkeypatch
):
    eos = stop_ids[0]
    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler([7, eos]))
    result = tiny_engine.generate(USER_TURN, max_new_tokens=10)
    marker = tiny_engine.tokenizer.id_to_token(eos)
    assert marker not in result.text
    assert result.text == tiny_engine.tokenizer.decode([7])


def test_an_eos_as_the_very_first_token_stops_immediately(
    tiny_engine: Engine, stop_ids, scripted_sampler, monkeypatch
):
    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler([stop_ids[0]]))
    result = tiny_engine.generate(USER_TURN, max_new_tokens=10)
    assert result.token_ids == (stop_ids[0],)
    assert result.stop_reason is StopReason.EOS


def test_the_second_published_stop_id_also_stops(
    tiny_engine: Engine, stop_ids, scripted_sampler, monkeypatch
):
    """Qwen3 publishes two eos ids; both must terminate generation."""
    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler([5, stop_ids[1]]))
    assert tiny_engine.generate(USER_TURN, max_new_tokens=10).stop_reason is StopReason.EOS


def test_max_new_tokens_must_be_at_least_one(tiny_engine: Engine):
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_new_tokens"):
            tiny_engine.generate(USER_TURN, max_new_tokens=bad)


@pytest.mark.parametrize("bad", [1.5, "4", True, None])
def test_a_non_integer_max_new_tokens_is_rejected(tiny_engine: Engine, bad):
    with pytest.raises(ValueError, match="max_new_tokens"):
        tiny_engine.generate(USER_TURN, max_new_tokens=bad)


def test_an_empty_prompt_is_rejected(tiny_engine: Engine):
    with pytest.raises(UnsupportedInputError, match="zero tokens"):
        tiny_engine.generate_ids([], max_new_tokens=1)


# --------------------------------------------------------------------------- #
# context capacity
# --------------------------------------------------------------------------- #


def test_prompt_plus_max_new_tokens_must_fit_the_context(tiny_engine: Engine):
    prompt_ids = tiny_engine.encode_messages(USER_TURN)
    with pytest.raises(ContextOverflowError, match="exceeds max_context"):
        tiny_engine.generate_ids(prompt_ids, max_new_tokens=tiny_engine.max_context)


def test_a_prompt_that_exactly_fills_the_budget_is_allowed(
    tiny_engine: Engine, scripted_sampler, monkeypatch
):
    prompt_ids = tiny_engine.encode_messages(USER_TURN)
    budget = tiny_engine.max_context - len(prompt_ids)
    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler([7] * budget))
    result = tiny_engine.generate_ids(prompt_ids, max_new_tokens=budget)
    assert len(result.token_ids) == budget
    # The final sampled token is returned to the caller but never fed back
    # through the model, so it does not occupy a cache slot.
    assert tiny_engine.kv_cache.length == len(prompt_ids) + budget - 1


def test_a_long_prompt_is_refused_rather_than_truncated(tiny_model_dir):
    engine = Engine.from_directory(tiny_model_dir, max_context=16)
    with pytest.raises(ContextOverflowError):
        engine.generate_ids(list(range(1, 30)), max_new_tokens=1)


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #


def test_stream_yields_one_step_per_token(tiny_engine: Engine):
    steps = list(tiny_engine.stream(USER_TURN, max_new_tokens=4))
    assert len(steps) == 4
    assert [step.index for step in steps] == [0, 1, 2, 3]
    assert [step.finished for step in steps] == [False, False, False, True]
    assert [step.result is not None for step in steps] == [False, False, False, True]


def test_streamed_deltas_concatenate_to_the_final_text(tiny_engine: Engine):
    steps = list(tiny_engine.stream(USER_TURN, max_new_tokens=6))
    assert "".join(step.text_delta for step in steps) == steps[-1].result.text


def test_streamed_token_ids_match_the_result(tiny_engine: Engine):
    steps = list(tiny_engine.stream(USER_TURN, max_new_tokens=5))
    assert tuple(step.token_id for step in steps) == steps[-1].result.token_ids


def test_stream_can_be_abandoned_without_breaking_the_next_request(tiny_engine: Engine):
    """Breaking out of the generator mid-answer must not poison the cache."""
    for step in tiny_engine.stream(USER_TURN, max_new_tokens=8):
        if step.index == 1:
            break
    assert tiny_engine.kv_cache.length > 0

    again = tiny_engine.generate(USER_TURN, max_new_tokens=3)
    reference = Engine.from_directory(
        tiny_engine.model.weights.root, max_context=tiny_engine.max_context
    ).generate(USER_TURN, max_new_tokens=3)
    assert again.token_ids == reference.token_ids


# --------------------------------------------------------------------------- #
# the decoder's buffered tail
# --------------------------------------------------------------------------- #


def find_incomplete_byte_token(engine: Engine) -> int:
    """A token the incremental decoder has to hold back.

    Byte-level BPE splits multi-byte characters across tokens, so decoding such a
    token on its own yields U+FFFD and the decoder buffers it instead of emitting
    it. Generation that ends while the buffer is non-empty is exactly the case
    where ``flush()`` still has something to say — which is what truncating a
    multi-byte character at ``max_new_tokens`` does.
    """
    for token_id in range(engine.config.vocab_size):
        decoder = engine.tokenizer.incremental_decoder()
        if decoder.push(token_id) == "" and decoder.flush() != "":
            return token_id
    raise AssertionError("the vocabulary has no incomplete-byte token")


def test_a_character_truncated_by_max_new_tokens_reaches_the_stream(
    tiny_engine: Engine, scripted_sampler, monkeypatch
):
    partial = find_incomplete_byte_token(tiny_engine)
    tail = tiny_engine.tokenizer.decode([partial])
    assert tail, "this test needs a token whose flush is non-empty"

    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler([7, partial]))
    steps = list(tiny_engine.stream(USER_TURN, max_new_tokens=2))
    result = steps[-1].result
    assert result.stop_reason is StopReason.MAX_NEW_TOKENS

    # The tail is part of the answer, and it must arrive through the stream too:
    # a consumer that only reads deltas has to see what a blocking call returns.
    assert result.text.endswith(tail)
    assert "".join(step.text_delta for step in steps) == result.text
    assert steps[-1].text_delta.endswith(tail)


def test_a_buffered_character_is_flushed_on_the_eos_path(
    tiny_engine: Engine, stop_ids, scripted_sampler, monkeypatch
):
    partial = find_incomplete_byte_token(tiny_engine)
    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler([partial, stop_ids[0]]))
    steps = list(tiny_engine.stream(USER_TURN, max_new_tokens=8))
    result = steps[-1].result

    assert result.stop_reason is StopReason.EOS
    assert result.text == tiny_engine.tokenizer.decode([partial])
    assert "".join(step.text_delta for step in steps) == result.text


def test_flushing_neither_duplicates_nor_drops_complete_text(
    tiny_engine: Engine, scripted_sampler, monkeypatch
):
    """Chinese and emoji span several tokens; nothing may be lost or repeated."""
    expected = "你好🐢慢炖"
    token_ids = tiny_engine.tokenizer.encode(expected)

    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler(token_ids))
    steps = list(tiny_engine.stream(USER_TURN, max_new_tokens=len(token_ids)))
    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler(token_ids))
    blocking = tiny_engine.generate(USER_TURN, max_new_tokens=len(token_ids))

    assert steps[-1].result.text == expected == blocking.text
    assert "".join(step.text_delta for step in steps) == blocking.text



def test_a_raising_request_leaves_the_cache_reusable(tiny_engine: Engine):
    tiny_engine.generate(USER_TURN, max_new_tokens=2)
    with pytest.raises(ContextOverflowError):
        tiny_engine.generate_ids([1] * 200, max_new_tokens=1)
    # The next request resets the cache itself, so this must still work.
    assert len(tiny_engine.generate(USER_TURN, max_new_tokens=2).token_ids) == 2


def test_two_identical_requests_give_identical_results(tiny_engine: Engine):
    first = tiny_engine.generate(USER_TURN, max_new_tokens=4)
    second = tiny_engine.generate(USER_TURN, max_new_tokens=4)
    assert first.token_ids == second.token_ids
    assert first.text == second.text


def test_the_cache_is_reset_between_requests(tiny_engine: Engine):
    tiny_engine.generate(USER_TURN, max_new_tokens=3)
    long_length = tiny_engine.kv_cache.length
    assert long_length > 3
    tiny_engine.generate_ids([1, 2], max_new_tokens=1)
    # Only the second request's two prompt tokens remain; nothing from the first.
    assert tiny_engine.kv_cache.length == 2


# --------------------------------------------------------------------------- #
# multi-turn
# --------------------------------------------------------------------------- #


def test_a_multi_turn_history_is_re_rendered_whole(tiny_model_dir):
    """v0.1 re-encodes the full history each turn instead of reusing the cache."""
    engine = Engine.from_directory(tiny_model_dir, max_context=256)
    history = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
    ]
    result = engine.generate(history, max_new_tokens=2)
    assert list(result.prompt_token_ids) == engine.encode_messages(history)
    short = engine.generate(history[:2], max_new_tokens=2)
    assert len(result.prompt_token_ids) > len(short.prompt_token_ids)
    # A re-rendered history starts from an empty cache every turn.
    assert engine.kv_cache.length == len(short.prompt_token_ids) + 1


def test_a_follow_up_turn_extends_the_previous_prompt(tiny_engine: Engine):
    first = tiny_engine.encode_messages([{"role": "user", "content": "hi"}])
    second = tiny_engine.encode_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "again"},
        ]
    )
    assert second[: len(first)] == first


# --------------------------------------------------------------------------- #
# chunked prefill
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk", [1, 2, 3, 7])
def test_the_prefill_chunk_size_does_not_change_the_output(tiny_model_dir, chunk: int):
    engine = Engine.from_directory(tiny_model_dir, max_context=64, prefill_chunk_size=chunk)
    result = engine.generate(USER_TURN, max_new_tokens=3)
    baseline = Engine.from_directory(tiny_model_dir, max_context=64).generate(
        USER_TURN, max_new_tokens=3
    )
    assert result.token_ids == baseline.token_ids


def test_the_number_of_prefill_chunks_is_reported(tiny_model_dir):
    engine = Engine.from_directory(tiny_model_dir, max_context=64, prefill_chunk_size=3)
    prompt_ids = list(range(1, 11))  # 10 tokens -> chunks of 3,3,3,1
    metrics = engine.generate_ids(prompt_ids, max_new_tokens=1).metrics
    assert metrics.prefill_chunks == 4
    assert metrics.prompt_tokens == 10


def test_one_chunk_is_used_when_the_prompt_is_short(tiny_engine: Engine):
    assert tiny_engine.generate_ids([1, 2, 3], max_new_tokens=1).metrics.prefill_chunks == 1


# --------------------------------------------------------------------------- #
# sampling pass-through and metrics
# --------------------------------------------------------------------------- #


def test_greedy_generation_is_reproducible(tiny_engine: Engine):
    params = SamplingParams(temperature=0.0)
    a = tiny_engine.generate(USER_TURN, sampling=params, max_new_tokens=6).token_ids
    b = tiny_engine.generate(USER_TURN, sampling=params, max_new_tokens=6).token_ids
    assert a == b


def test_a_seeded_sampler_reproduces_the_same_text(tiny_engine: Engine):
    params = SamplingParams(temperature=0.9, top_k=10, top_p=0.9, seed=2026)
    a = tiny_engine.generate(USER_TURN, sampling=params, max_new_tokens=8)
    b = tiny_engine.generate(USER_TURN, sampling=params, max_new_tokens=8)
    assert a.token_ids == b.token_ids and a.text == b.text


def test_invalid_sampling_parameters_surface_before_any_computation(tiny_engine: Engine):
    from slowq.errors import SamplingError

    with pytest.raises(SamplingError):
        tiny_engine.generate(USER_TURN, sampling=SamplingParams(top_p=2.0), max_new_tokens=1)


def test_metrics_describe_the_request(tiny_engine: Engine):
    result = tiny_engine.generate(USER_TURN, max_new_tokens=4)
    metrics = result.metrics
    assert metrics.prompt_tokens == len(result.prompt_token_ids)
    assert metrics.generated_tokens == 4
    assert metrics.decode_tokens == 3  # the first token belongs to prefill timing
    assert metrics.stop_reason == StopReason.MAX_NEW_TOKENS.value
    assert metrics.prefill_seconds > 0
    assert metrics.first_token_seconds > 0
    assert metrics.total_seconds > 0
    assert metrics.decode_seconds >= 0


def test_metrics_report_weight_and_kv_sizes_separately(tiny_engine: Engine):
    metrics = tiny_engine.generate(USER_TURN, max_new_tokens=1).metrics
    assert metrics.weight_logical_bytes == tiny_engine.model.weights.deduplicated_bytes()
    assert metrics.kv_allocated_bytes == tiny_engine.kv_cache.allocated_bytes()
    assert metrics.peak_rss_bytes > metrics.kv_allocated_bytes


def test_metrics_are_json_serialisable(tiny_engine: Engine):
    payload = tiny_engine.generate(USER_TURN, max_new_tokens=2).as_dict()
    import json

    text = json.dumps(payload, ensure_ascii=False)
    assert payload["stop_reason"] == "max_new_tokens"
    assert len(payload["token_ids"]) == 2
    assert "decode_tokens_per_second" in json.loads(text)["metrics"]


def test_the_report_lists_every_documented_interval(tiny_engine: Engine):
    report = dict(tiny_engine.generate(USER_TURN, max_new_tokens=2).metrics.report())
    for key in (
        "stop reason",
        "prompt tokens",
        "generated tokens",
        "tokenize",
        "prefill",
        "first token latency",
        "decode",
        "decode speed",
        "total",
        "weights (logical)",
        "KV cache (allocated)",
        "peak RSS",
    ):
        assert key in report


def test_the_documented_default_max_new_tokens_is_applied(tiny_engine: Engine):
    """The tiny model's context (64) is smaller than the default, so the budget
    check must name it in the error rather than silently clamping."""
    with pytest.raises(ContextOverflowError, match=str(DEFAULT_MAX_NEW_TOKENS)):
        tiny_engine.generate(USER_TURN)
