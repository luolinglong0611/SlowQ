"""Tokenizer, chat-template rendering, and incremental (streaming) decoding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slowq.config import GenerationConfig
from slowq.errors import ConfigError, UnsupportedInputError
from slowq.tokenizer import (
    ChatMessage,
    IncrementalTextDecoder,
    Tokenizer,
    normalize_messages,
    stop_token_ids,
)

EMOJI = "🐢"
#: Both of these split across several byte-level tokens in the tiny vocabulary
#: (see test_the_tiny_tokenizer_really_does_split_multibyte_characters).
CHINESE = "中文"
MIXED = f"你好，{EMOJI} SlowQ {CHINESE} abc 😀🚀"


# --------------------------------------------------------------------------- #
# message validation
# --------------------------------------------------------------------------- #


def test_normalize_accepts_plain_mappings():
    messages = normalize_messages(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[1].content == "u"


def test_normalize_accepts_chat_message_objects():
    messages = normalize_messages([ChatMessage(role="user", content="hi")])
    assert messages == [ChatMessage(role="user", content="hi")]


def test_normalize_accepts_an_assistant_turn_for_history():
    roles = [m.role for m in normalize_messages(
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )]
    assert roles == ["user", "assistant"]


def test_normalize_rejects_an_empty_conversation():
    with pytest.raises(UnsupportedInputError, match="must not be empty"):
        normalize_messages([])


@pytest.mark.parametrize("role", ["tool", "function", "developer", "USER", ""])
def test_normalize_rejects_unsupported_roles(role: str):
    with pytest.raises(UnsupportedInputError, match="is not supported"):
        normalize_messages([{"role": role, "content": "x"}])


def test_normalize_rejects_tool_calls():
    with pytest.raises(UnsupportedInputError, match="tool_calls"):
        normalize_messages(
            [{"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}]
        )


def test_normalize_rejects_function_call():
    with pytest.raises(UnsupportedInputError, match="function_call"):
        normalize_messages([{"role": "assistant", "content": "", "function_call": {"name": "f"}}])


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": "hi"}, {"type": "image_url", "url": "x"}],
        {"type": "image"},
        None,
        42,
        b"bytes",
    ],
)
def test_normalize_rejects_non_string_content(content):
    """The text path is text-only: multi-part content must be refused, not flattened."""
    with pytest.raises(UnsupportedInputError, match="text-only|plain string"):
        normalize_messages([{"role": "user", "content": content}])


def test_normalize_rejects_unknown_fields():
    with pytest.raises(UnsupportedInputError, match="unsupported field"):
        normalize_messages([{"role": "user", "content": "x", "name": "n"}])


def test_normalize_rejects_a_non_mapping_message():
    with pytest.raises(UnsupportedInputError, match="mapping or ChatMessage"):
        normalize_messages(["just a string"])


def test_normalize_reports_the_offending_index():
    with pytest.raises(UnsupportedInputError, match=r"messages\[1\]"):
        normalize_messages([{"role": "user", "content": "ok"}, {"role": "tool", "content": "x"}])


# --------------------------------------------------------------------------- #
# chat template
# --------------------------------------------------------------------------- #


def test_render_chat_wraps_every_message(tiny_tokenizer: Tokenizer):
    rendered = tiny_tokenizer.render_chat([{"role": "user", "content": "你好"}])
    assert "<|im_start|>user\n你好<|im_end|>\n" in rendered


def test_render_chat_appends_a_generation_prompt_by_default(tiny_tokenizer: Tokenizer):
    rendered = tiny_tokenizer.render_chat([{"role": "user", "content": "hi"}])
    assert rendered.endswith("<|im_start|>assistant\n")


def test_render_chat_can_omit_the_generation_prompt(tiny_tokenizer: Tokenizer):
    rendered = tiny_tokenizer.render_chat(
        [{"role": "user", "content": "hi"}], add_generation_prompt=False
    )
    assert not rendered.endswith("<|im_start|>assistant\n")
    assert rendered.endswith("<|im_end|>\n")


def test_enable_thinking_false_emits_an_empty_think_block(tiny_tokenizer: Tokenizer):
    """The v0.1 default is non-thinking mode, which the template marks explicitly."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    rendered = tiny_tokenizer.render_chat(messages, enable_thinking=False)
    assert "<think>\n\n</tool_response>" in rendered


def test_enable_thinking_true_omits_the_empty_think_block(tiny_tokenizer: Tokenizer):
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    rendered = tiny_tokenizer.render_chat(messages, enable_thinking=True)
    assert "<think>" not in rendered


def test_the_default_renders_non_thinking_mode(tiny_tokenizer: Tokenizer):
    messages = [{"role": "assistant", "content": "a"}]
    assert tiny_tokenizer.render_chat(messages) == tiny_tokenizer.render_chat(
        messages, enable_thinking=False
    )


def test_rendered_history_is_preserved_in_order(tiny_tokenizer: Tokenizer):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    rendered = tiny_tokenizer.render_chat(messages)
    assert rendered.index("sys") < rendered.index("u1") < rendered.index("a1") < rendered.index("u2")


# --------------------------------------------------------------------------- #
# encode / decode
# --------------------------------------------------------------------------- #


def test_encode_decode_round_trips_chinese_and_emoji(tiny_tokenizer: Tokenizer):
    ids = tiny_tokenizer.encode(MIXED)
    assert ids and all(isinstance(token, int) for token in ids)
    assert tiny_tokenizer.decode(ids) == MIXED


def test_add_special_tokens_makes_no_difference_for_a_byte_level_tokenizer(
    tiny_tokenizer: Tokenizer,
):
    """Qwen's post-processor adds nothing; the template emits every marker."""
    assert tiny_tokenizer.encode("abc", add_special_tokens=False) == tiny_tokenizer.encode(
        "abc", add_special_tokens=True
    )


def test_encode_chat_matches_encoding_the_rendered_text(tiny_tokenizer: Tokenizer):
    messages = [{"role": "user", "content": "你好🐢"}]
    assert tiny_tokenizer.encode_chat(messages) == tiny_tokenizer.encode(
        tiny_tokenizer.render_chat(messages)
    )


def test_a_longer_prompt_shares_the_prefix_of_a_shorter_one(tiny_tokenizer: Tokenizer):
    """BPE merges at a boundary can break this; the chat template must not."""
    short = tiny_tokenizer.encode_chat([{"role": "user", "content": "hi"}])
    long = tiny_tokenizer.encode_chat(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    )
    assert long[: len(short)] == short


def test_decode_can_show_special_tokens(tiny_tokenizer: Tokenizer):
    rendered = tiny_tokenizer.render_chat([{"role": "user", "content": "x"}])
    ids = tiny_tokenizer.encode(rendered)
    assert "<|im_start|>" not in tiny_tokenizer.decode(ids, skip_special_tokens=True)
    assert "<|im_start|>" in tiny_tokenizer.decode(ids, skip_special_tokens=False)


def test_special_token_ids_are_resolved_from_the_vocabulary(tiny_tokenizer: Tokenizer):
    ids = tiny_tokenizer.special_token_ids
    assert ids["bos"] is None
    assert isinstance(ids["eos"], int)
    assert tiny_tokenizer.id_to_token(ids["eos"]) == "<|fim_middle|>"
    assert tiny_tokenizer.token_to_id("<|fim_prefix|>") is not None
    assert tiny_tokenizer.token_to_id("definitely-not-a-token") is None


def test_vocab_size_is_positive_and_covers_the_special_tokens(tiny_tokenizer: Tokenizer):
    assert tiny_tokenizer.vocab_size > 256  # at least the byte-level alphabet
    assert tiny_tokenizer.special_token_ids["eos"] < tiny_tokenizer.vocab_size


def test_describe_mentions_the_backend_and_template(tiny_tokenizer: Tokenizer):
    described = dict(tiny_tokenizer.describe())
    assert described["backend"].startswith("tokenizers ")
    assert "chat template" in described


# --------------------------------------------------------------------------- #
# incremental decoding
# --------------------------------------------------------------------------- #


def test_the_tiny_tokenizer_really_does_split_multibyte_characters(
    tiny_tokenizer: Tokenizer,
):
    """Precondition for the streaming tests below: without it they prove nothing."""
    assert len(tiny_tokenizer.encode(EMOJI)) > 1
    assert len(tiny_tokenizer.encode(CHINESE)) > 1


def test_decoding_a_partial_multibyte_token_yields_a_replacement_char(
    tiny_tokenizer: Tokenizer,
):
    ids = tiny_tokenizer.encode(EMOJI)
    assert tiny_tokenizer.decode(ids[:1]).endswith("\ufffd")


def test_incremental_decode_reconstructs_the_exact_text(tiny_tokenizer: Tokenizer):
    ids = tiny_tokenizer.encode(MIXED)
    decoder = tiny_tokenizer.incremental_decoder()
    emitted = "".join(decoder.push(token) for token in ids) + decoder.flush()
    assert emitted == MIXED


def test_incremental_decode_never_emits_a_replacement_character(
    tiny_tokenizer: Tokenizer,
):
    """The 锟斤拷 bug: emitting a half-decoded character to the user's terminal."""
    decoder = tiny_tokenizer.incremental_decoder()
    for token in tiny_tokenizer.encode(MIXED):
        chunk = decoder.push(token)
        assert "\ufffd" not in chunk
    assert "\ufffd" not in decoder.flush()


def test_naive_per_token_decoding_would_have_been_wrong(tiny_tokenizer: Tokenizer):
    ids = tiny_tokenizer.encode(MIXED)
    naive = "".join(tiny_tokenizer.decode([token]) for token in ids)
    assert naive != MIXED


def test_incremental_decode_holds_back_until_a_character_is_complete(
    tiny_tokenizer: Tokenizer,
):
    ids = tiny_tokenizer.encode(EMOJI)
    decoder = tiny_tokenizer.incremental_decoder()
    chunks = [decoder.push(token) for token in ids]
    assert chunks[:-1] == [""] * (len(ids) - 1)  # nothing emitted mid-character
    assert decoder.flush() == ""
    assert chunks[-1] == EMOJI


@pytest.mark.parametrize("text", [EMOJI, CHINESE, "😀", MIXED])
def test_incremental_decode_streams_a_split_character_exactly_once(
    tiny_tokenizer: Tokenizer, text: str
):
    """Each multi-byte character arrives in one chunk, whole, and only once."""
    ids = tiny_tokenizer.encode(text)
    decoder = tiny_tokenizer.incremental_decoder()
    chunks = [decoder.push(token) for token in ids]
    chunks.append(decoder.flush())
    assert "".join(chunks) == text
    assert all("\ufffd" not in chunk for chunk in chunks)
    assert [chunk for chunk in chunks if chunk] != []


def test_push_many_matches_pushing_one_at_a_time(tiny_tokenizer: Tokenizer):
    ids = tiny_tokenizer.encode(MIXED)
    one_at_a_time = tiny_tokenizer.incremental_decoder()
    text = "".join(one_at_a_time.push(token) for token in ids) + one_at_a_time.flush()

    batched = tiny_tokenizer.incremental_decoder()
    assert batched.push_many(ids) + batched.flush() == text == MIXED


def test_flush_emits_a_buffered_incomplete_sequence():
    class AlwaysIncomplete:
        def decode(self, token_ids, skip_special_tokens=True):
            return "\ufffd"

    decoder = IncrementalTextDecoder(AlwaysIncomplete())
    assert decoder.push(1) == ""
    assert decoder.push(2) == ""
    assert decoder.flush() == "\ufffd"  # the safety valve releases it at the end


def test_the_buffer_is_released_after_max_held_tokens():
    """A genuinely invalid byte sequence must not pin the buffer forever."""

    class AlwaysIncomplete:
        def decode(self, token_ids, skip_special_tokens=True):
            return "\ufffd"

    decoder = IncrementalTextDecoder(AlwaysIncomplete())
    emitted = [decoder.push(token) for token in range(IncrementalTextDecoder.MAX_HELD_TOKENS + 2)]
    assert emitted[: IncrementalTextDecoder.MAX_HELD_TOKENS - 1] == [""] * (
        IncrementalTextDecoder.MAX_HELD_TOKENS - 1
    )
    assert any(chunk for chunk in emitted)  # eventually forced out


def test_a_fresh_decoder_has_an_empty_buffer(tiny_tokenizer: Tokenizer):
    decoder = tiny_tokenizer.incremental_decoder()
    assert decoder.flush() == ""


def test_two_decoders_are_independent(tiny_tokenizer: Tokenizer):
    ids = tiny_tokenizer.encode(MIXED)
    first, second = tiny_tokenizer.incremental_decoder(), tiny_tokenizer.incremental_decoder()
    first.push(ids[0])
    assert second.push_many(ids) + second.flush() == MIXED


# --------------------------------------------------------------------------- #
# stop tokens
# --------------------------------------------------------------------------- #


def test_stop_token_ids_come_from_the_generation_config(tiny_model_dir: Path):
    generation = GenerationConfig.from_json_file(
        tiny_model_dir / "generation_config.json", tiny_model_dir / "config.json"
    )
    tokenizer = Tokenizer.from_directory(tiny_model_dir)
    ids = stop_token_ids(generation, tokenizer)
    assert ids == frozenset(generation.eos_token_ids)
    assert tokenizer.special_token_ids["eos"] in ids


def test_stop_token_ids_rejects_an_out_of_range_eos(tiny_tokenizer: Tokenizer):
    generation = GenerationConfig.from_hf_config({"eos_token_id": [tiny_tokenizer.vocab_size + 5]})
    with pytest.raises(ConfigError, match="outside the tokenizer vocabulary"):
        stop_token_ids(generation, tiny_tokenizer)


def test_stop_token_ids_rejects_an_empty_stop_set(tiny_tokenizer: Tokenizer):
    generation = GenerationConfig(eos_token_ids=(), pad_token_id=None, bos_token_id=None,
                                  temperature=1.0, top_k=0, top_p=1.0, do_sample=False)
    with pytest.raises(ConfigError, match="no eos_token_id"):
        stop_token_ids(generation, tiny_tokenizer)


# --------------------------------------------------------------------------- #
# loading failures
# --------------------------------------------------------------------------- #


def test_a_missing_tokenizer_file_is_reported(tmp_path: Path):
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError, match="tokenizer.json"):
        Tokenizer.from_directory(tmp_path)


def test_a_missing_tokenizer_config_is_reported(tiny_model_dir: Path, tmp_path: Path):
    import shutil

    shutil.copyfile(tiny_model_dir / "tokenizer.json", tmp_path / "tokenizer.json")
    with pytest.raises(ConfigError, match="tokenizer_config.json"):
        Tokenizer.from_directory(tmp_path)


def test_a_missing_chat_template_is_reported(tiny_model_dir: Path, tmp_path: Path):
    import shutil

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(tiny_model_dir / name, tmp_path / name)
    config = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    del config["chat_template"]
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="will not\n?.*guess|chat_template"):
        Tokenizer.from_directory(tmp_path)


def test_a_special_token_absent_from_the_vocabulary_is_reported(
    tiny_model_dir: Path, tmp_path: Path
):
    import shutil

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(tiny_model_dir / name, tmp_path / name)
    config = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    config["bos_token"] = "<|fim_prefix|>not-in-vocab"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="not in the vocabulary"):
        Tokenizer.from_directory(tmp_path)
