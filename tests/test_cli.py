"""CLI: subcommand wiring, the stdout/stderr split, and exit codes.

Every test drives the real ``main`` with a real (tiny) model directory, so
argument parsing, engine construction, streaming and the stats block are all
exercised as a user would. Exit codes follow the usual convention: 0 success,
1 a ``SlowQError`` the user can act on, 2 a usage error, 130 interrupted.
"""

from __future__ import annotations

import builtins
import json
import re

import pytest

from slowq import __version__
from slowq.cli import main

PROMPT = "你好🐢"
ANSWER = "慢炖"

#: The tiny checkpoint declares ``max_position_embeddings = 256``, and the
#: engine refuses to extrapolate beyond it, so CLI tests pin a smaller budget.
DEFAULT_MAX_CONTEXT = 128


class Raise:
    """A chat input line that raises instead of returning text."""

    def __init__(self, exception: BaseException) -> None:
        self.exception = exception

    def __call__(self) -> str:
        raise self.exception


def interrupting_sampler(after: int, token_id: int = 7):
    """A ``Sampler`` that raises ``KeyboardInterrupt`` after ``after`` draws.

    Ctrl-C during decode arrives as a ``KeyboardInterrupt`` from inside the
    generation loop; this reproduces that without sending a real signal.
    """

    class Interrupting:
        def __init__(self, params=None) -> None:
            self.params = params
            self.calls = 0

        @property
        def is_greedy(self) -> bool:
            return True

        def sample(self, logits) -> int:
            self.calls += 1
            if self.calls > after:
                raise KeyboardInterrupt
            return token_id

    return Interrupting


def prompt_token_counts(stderr: str) -> list[int]:
    """The ``prompt tokens`` value of every stats block, in order."""
    return [int(match) for match in re.findall(r"prompt tokens\s+(\d+)", stderr)]


def last_line(stderr: str) -> str:
    """The final stderr line, which is where the CLI puts an error message."""
    return stderr.strip().splitlines()[-1]


@pytest.fixture(autouse=True)
def clear_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own $SLOWQ_MODEL_DIR must not leak into these tests."""
    monkeypatch.delenv("SLOWQ_MODEL_DIR", raising=False)


@pytest.fixture
def run(capsys: pytest.CaptureFixture, tiny_model_dir):
    """``run(["generate", ...]) -> (exit_code, stdout, stderr)``.

    ``--model`` and ``--max-context`` are injected unless the test supplies
    them, so each test only spells out what it is actually about.
    """

    def invoke(argv: list[str], *, model=tiny_model_dir):
        full = list(argv)
        injected: list[str] = []
        if model is not None and "--model" not in full:
            injected += ["--model", str(model)]
        if "--max-context" not in full:
            injected += ["--max-context", str(DEFAULT_MAX_CONTEXT)]
        code = main([full[0], *injected, *full[1:]])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return invoke


@pytest.fixture
def answer(tiny_tokenizer, scripted_sampler, monkeypatch: pytest.MonkeyPatch) -> str:
    """Force a known one-token answer, and return the text the CLI should print.

    The tiny model has random weights, so its greedy output is not something a
    test can arrange — and it happens to decode to whitespace, which would make
    every stdout assertion vacuous. One token keeps ``--max-new-tokens 1`` and
    the expected stdout the same thing.
    """
    token_id = tiny_tokenizer.encode(ANSWER)[0]
    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler([token_id]))
    return tiny_tokenizer.decode([token_id])


@pytest.fixture
def feed(monkeypatch: pytest.MonkeyPatch):
    """Script the lines ``chat`` reads from stdin; EOF once they run out."""

    def install(lines: list) -> None:
        remaining = iter(lines)

        def fake_input(prompt: str = "") -> str:
            try:
                line = next(remaining)
            except StopIteration:
                raise EOFError from None
            return line() if callable(line) else line

        monkeypatch.setattr(builtins, "input", fake_input)

    return install


# --------------------------------------------------------------------------- #
# parser level
# --------------------------------------------------------------------------- #


def test_the_version_flag_reports_the_package_version(capsys: pytest.CaptureFixture):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_an_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main(["brew"])
    assert excinfo.value.code == 2


def test_a_missing_model_directory_is_a_usage_error(capsys: pytest.CaptureFixture):
    assert main(["generate", "--prompt", PROMPT]) == 2
    stderr = capsys.readouterr().err
    assert "--model" in stderr
    # The message must say how to get one, not just that something is missing.
    assert "convert_model.py" in stderr


def test_the_model_directory_can_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tiny_model_dir
):
    monkeypatch.setenv("SLOWQ_MODEL_DIR", str(tiny_model_dir))
    assert main(["inspect", "--max-context", str(DEFAULT_MAX_CONTEXT)]) == 0


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def test_generate_puts_the_answer_on_stdout_and_the_stats_on_stderr(run, answer):
    code, out, err = run(["generate", "--prompt", PROMPT, "--max-new-tokens", "1"])
    assert code == 0
    assert out == answer + "\n"
    assert "[stats]" in err
    assert "decode speed" in err


def test_stdout_stays_clean_enough_to_redirect(run, answer):
    """``slowq generate ... > out.txt`` must not capture diagnostics."""
    _, out, err = run(["generate", "--prompt", PROMPT, "--max-new-tokens", "1"])
    for diagnostic in ("[stats]", "[sampler]", "loading model", "loaded in"):
        assert diagnostic not in out
        assert diagnostic in err


def test_streaming_and_no_stream_produce_the_same_text(run, answer):
    _, streamed, _ = run(["generate", "--prompt", PROMPT, "--max-new-tokens", "1"])
    _, batched, _ = run(
        ["generate", "--prompt", PROMPT, "--max-new-tokens", "1", "--no-stream"]
    )
    assert streamed == batched == answer + "\n"


def test_generate_json_emits_the_whole_result(run):
    code, out, _ = run(["generate", "--prompt", PROMPT, "--max-new-tokens", "3", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["stop_reason"] == "max_new_tokens"
    assert len(payload["token_ids"]) == 3
    assert payload["prompt_token_ids"]
    metrics = payload["metrics"]
    for key in ("prefill_seconds", "first_token_seconds", "decode_seconds", "peak_rss_bytes"):
        assert key in metrics


def test_the_system_flag_becomes_part_of_the_prompt(run, tiny_tokenizer):
    _, out, _ = run(
        ["generate", "--prompt", PROMPT, "--system", "你是慢炖", "--max-new-tokens", "1", "--json"]
    )
    expected = tiny_tokenizer.encode_chat(
        [{"role": "system", "content": "你是慢炖"}, {"role": "user", "content": PROMPT}]
    )
    assert json.loads(out)["prompt_token_ids"] == expected


def test_without_a_system_flag_the_prompt_is_a_single_user_turn(run, tiny_tokenizer):
    _, out, _ = run(["generate", "--prompt", PROMPT, "--max-new-tokens", "1", "--json"])
    expected = tiny_tokenizer.encode_chat([{"role": "user", "content": PROMPT}])
    assert json.loads(out)["prompt_token_ids"] == expected


def test_greedy_decoding_is_the_default(run):
    _, _, err = run(["generate", "--prompt", PROMPT, "--max-new-tokens", "1"])
    assert "[sampler] greedy" in err


def test_the_sampling_flags_reach_the_sampler(run):
    _, _, err = run(
        [
            "generate",
            "--prompt",
            PROMPT,
            "--max-new-tokens",
            "1",
            "--temperature",
            "0.8",
            "--top-k",
            "20",
            "--top-p",
            "0.9",
            "--seed",
            "7",
        ]
    )
    assert "[sampler] temperature=0.8, top_k=20, top_p=0.9, seed=7" in err


def test_a_fixed_seed_reproduces_the_same_tokens(run):
    sampling = ["--temperature", "0.9", "--top-k", "20", "--seed", "1234", "--json"]
    argv = ["generate", "--prompt", PROMPT, "--max-new-tokens", "6", *sampling]
    _, first, _ = run(argv)
    _, second, _ = run(argv)
    assert json.loads(first)["token_ids"] == json.loads(second)["token_ids"]


def test_invalid_sampling_parameters_are_reported_not_traced(run):
    code, _, err = run(["generate", "--prompt", PROMPT, "--max-new-tokens", "1", "--top-p", "0"])
    assert code == 1
    assert last_line(err).startswith("error: ")
    assert "top_p" in err


def test_max_new_tokens_bounds_the_answer(run):
    for budget in (1, 5):
        _, out, _ = run(["generate", "--prompt", PROMPT, "--max-new-tokens", str(budget), "--json"])
        assert len(json.loads(out)["token_ids"]) == budget


def test_the_prefill_chunk_size_flag_controls_chunking(run):
    _, out, _ = run(
        ["generate", "--prompt", PROMPT, "--max-new-tokens", "1", "--prefill-chunk-size", "3", "--json"]
    )
    payload = json.loads(out)
    prompt_tokens = len(payload["prompt_token_ids"])
    assert payload["metrics"]["prefill_chunks"] == -(-prompt_tokens // 3)


def test_the_prompt_is_never_truncated_to_fit_the_context(run):
    code, out, err = run(
        ["generate", "--prompt", PROMPT, "--max-new-tokens", "1", "--max-context", "8"]
    )
    assert code == 1
    assert out == ""
    assert "exceeds max_context" in err


def test_a_broken_model_directory_exits_with_an_error(run, tmp_path):
    code, _, err = run(["inspect"], model=tmp_path)
    assert code == 1
    assert last_line(err).startswith("error: ")
    assert "convert_model.py" in err


@pytest.mark.parametrize(
    "flag", ["--max-new-tokens", "--prefill-chunk-size"],
)
def test_a_rejected_flag_value_is_a_usage_error(run, flag):
    code, _, err = run(["generate", "--prompt", PROMPT, flag, "0"])
    assert code == 2
    assert last_line(err).startswith("error: ")


def test_quiet_drops_the_stats_block_but_keeps_the_load_line(run, answer):
    _, out, err = run(["generate", "--prompt", PROMPT, "--max-new-tokens", "1", "--quiet"])
    assert out == answer + "\n"
    assert "[stats]" not in err
    assert "loading model" in err


def test_ctrl_c_during_decoding_exits_130(run, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("slowq.engine.Sampler", interrupting_sampler(after=0))
    code, _, err = run(["generate", "--prompt", PROMPT, "--max-new-tokens", "4"])
    assert code == 130
    assert "interrupted" in err


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #


def test_inspect_prints_every_observability_section(run):
    code, out, err = run(["inspect"])
    assert code == 0
    for section in ("[model]", "[source]", "[weights]", "[KV cache]", "[generation]", "[tokenizer]"):
        assert section in out
    assert "peak RSS after load" in out
    assert "loading model from" in err


def test_inspect_reports_the_tied_embedding_as_shared_storage(run):
    _, out, _ = run(["inspect", "--json"])
    payload = json.loads(out)
    assert payload["weights"]["shared (tied) tensors"] == {
        "lm_head.weight": "model.embed_tokens.weight"
    }
    assert payload["weights"]["dtypes on disk"] == ["float32"]


def test_inspect_json_matches_what_the_engine_actually_allocated(run, tiny_model_dir):
    from slowq import Engine

    engine = Engine.from_directory(tiny_model_dir, max_context=DEFAULT_MAX_CONTEXT)
    _, out, _ = run(["inspect", "--json"])
    payload = json.loads(out)

    assert payload["model_directory"] == str(tiny_model_dir)
    assert payload["slowq_version"] == __version__
    assert payload["manifest_format_version"] == engine.model.weights.manifest.format_version
    assert payload["kv_cache"]["max_context"] == engine.max_context
    assert payload["kv_cache"]["bytes per token"] == engine.config.kv_bytes_per_token()
    assert payload["generation"]["stop token ids"] == sorted(engine.stop_ids)
    assert payload["generation"]["prefill_chunk_size"] == engine.prefill_chunk_size
    assert set(payload["model_config"]) == {label for label, _ in engine.config.describe()}


def test_inspect_json_records_the_benchmark_environment(run):
    _, out, _ = run(["inspect", "--json"])
    environment = json.loads(out)["environment"]
    for key in ("platform", "cpu", "cpu_count_logical", "python", "numpy", "blas", "thread_env"):
        assert key in environment


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #


def test_chat_answers_a_turn_and_exits_on_eof(run, feed, answer):
    feed([PROMPT])
    code, out, err = run(["chat", "--max-new-tokens", "1"])
    assert code == 0
    assert out == f"SlowQ> {answer}\n"
    assert "bye" in err


def test_chat_prints_the_banner_and_the_sampler_before_the_first_prompt(run, feed):
    feed([])
    _, _, err = run(["chat"])
    assert "Commands:" in err
    assert "[sampler] greedy" in err


def test_chat_exit_and_quit_both_end_the_session(run, feed):
    for command in ("/exit", "/quit"):
        feed([command])
        assert run(["chat"])[0] == 0


def test_chat_treats_ctrl_d_as_a_clean_exit(run, feed):
    feed([Raise(EOFError())])
    code, _, err = run(["chat"])
    assert code == 0
    assert "bye" in err


def test_chat_treats_ctrl_c_at_the_prompt_as_a_clean_exit(run, feed):
    feed([Raise(KeyboardInterrupt())])
    code, _, err = run(["chat"])
    assert code == 0
    assert "bye" in err


def test_chat_help_reprints_the_banner(run, feed):
    feed(["/help"])
    _, _, err = run(["chat"])
    assert err.count("Commands:") == 2


def test_chat_skips_blank_input_without_generating(run, feed):
    feed(["", "   ", "/exit"])
    code, out, _ = run(["chat"])
    assert code == 0
    assert out == ""


def test_chat_history_lists_the_turns_so_far(run, feed, answer):
    feed([PROMPT, "/history", "/exit"])
    _, _, err = run(["chat", "--max-new-tokens", "1"])
    assert f"  user: {PROMPT}" in err
    assert f"  assistant: {answer}" in err


def test_chat_clear_drops_the_turns_but_keeps_the_system_message(run, feed):
    feed([PROMPT, "/clear", "/history", "/exit"])
    _, _, err = run(["chat", "--system", "你是慢炖", "--max-new-tokens", "1"])
    after_clear = err[err.index("[history cleared]") :]
    assert "user:" not in after_clear
    assert "assistant:" not in after_clear
    assert "system: 你是慢炖" in after_clear


def test_chat_re_renders_the_whole_history_each_turn(run, feed):
    """Turn two must be conditioned on turn one, not start from scratch."""
    feed([PROMPT, PROMPT, "/exit"])
    code, _, err = run(["chat", "--max-new-tokens", "1"])
    assert code == 0
    counts = prompt_token_counts(err)
    assert len(counts) == 2
    assert counts[1] > counts[0]


def test_chat_ctrl_c_cancels_only_the_current_turn(run, feed, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("slowq.engine.Sampler", interrupting_sampler(after=0))
    feed([PROMPT, "/history", "/exit"])
    code, out, err = run(["chat", "--max-new-tokens", "4"])
    assert code == 0
    assert "[cancelled]" in err
    # The interrupted user turn is dropped, so nothing is listed as history.
    assert "user:" not in err[err.index("[cancelled]") :]
    assert "SlowQ>" in out


def test_chat_quiet_suppresses_the_per_turn_stats(run, feed, answer):
    feed([PROMPT, "/exit"])
    _, out, err = run(["chat", "--max-new-tokens", "1", "--quiet"])
    assert out == f"SlowQ> {answer}\n"
    assert "[stats]" not in err


def test_chat_honours_the_sampling_flags(run, feed):
    feed(["/exit"])
    _, _, err = run(["chat", "--temperature", "0.7", "--top-p", "0.8", "--seed", "5"])
    assert "[sampler] temperature=0.7, top_p=0.8, seed=5" in err
