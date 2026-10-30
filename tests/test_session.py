"""Sessions: multi-turn KV reuse, and everything that can go wrong with it.

The load-bearing claim these tests make is not "reuse is faster" — it is that
reuse does not change the answer. Every correctness test here therefore compares
a session against a standalone ``Engine`` request over the *same* history and
demands the same token ids, not merely similar ones. Speed is a benchmark's job.

The tiny model has random weights, so a conversation's text is not something a
test can arrange; what it can arrange is that two paths through the same model
agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from slowq import ChatMessage, Engine, Session
from slowq.engine import SessionReuse, common_prefix_length
from slowq.errors import ContextOverflowError
from slowq.sampling import SamplingParams

GREEDY = SamplingParams(temperature=0.0, seed=0)

#: A reused prefix was produced by a different-shaped matmul than a cold one
#: (``[17, hidden] @ W.T`` rather than ``[35, hidden] @ W.T``), so the KV agrees
#: to FP32 reassociation rather than bit for bit. This is the same tolerance
#: tests/test_model.py uses for chunked-versus-whole prefill, for the same reason.
CHUNK_TOLERANCE = {"rtol": 2e-4, "atol": 2e-5}

QUESTION = "Q{}: name a prime."


def turn(index: int) -> ChatMessage:
    return ChatMessage(role="user", content=QUESTION.format(index))


def conversation(turns: int, *, system: str = "terse") -> list[ChatMessage]:
    """A conversation of ``turns`` questions.

    Kept short on purpose: the tiny tokenizer's vocabulary is 372 tokens, so the
    chat template alone costs ~20 tokens a turn and ``max_position_embeddings``
    is 256. Long prose here would test the context limit instead of reuse.
    """
    head = [ChatMessage(role="system", content=system)] if system else []
    return [*head, *(turn(i) for i in range(turns))]


@pytest.fixture
def engine(tiny_model_dir) -> Engine:
    """A fresh engine per test. Its cache is shared state, so never reuse one."""
    return Engine.from_directory(tiny_model_dir, max_context=256)


@pytest.fixture
def session(engine: Engine) -> Session:
    return engine.session()


def append_reply(history: list[ChatMessage], result) -> list[ChatMessage]:
    """What a chat does between turns: record the answer it was given."""
    history.append(ChatMessage(role="assistant", content=result.text))
    return history


def non_stop_ids(engine: Engine, count: int) -> list[int]:
    """``count`` token ids that will not end a request.

    A test that abandons a stream part-way through has to be sure the stream was
    still running when it let go; a model that argmaxes straight onto EOS would
    otherwise make "cancelled after two steps" mean "finished after one".
    """
    out = []
    for token_id in range(engine.tokenizer.vocab_size):
        if token_id not in engine.stop_ids:
            out.append(token_id)
        if len(out) == count:
            return out
    raise AssertionError(f"the vocabulary has fewer than {count} non-stop tokens")


def count_decoder_tokens(model) -> tuple[list[int], list[int]]:
    """Token counts handed to ``encode`` and ``forward``, outermost call only.

    Both have to be watched because ``DecoderModel.prefill`` runs its non-final
    chunks through ``encode`` and only the last through ``forward``; counting
    ``forward`` alone reports one call for a four-chunk prefill. ``forward`` calls
    ``encode`` internally, so a depth guard keeps one chunk counted once.
    """
    real_encode, real_forward = model.encode, model.forward
    encodes: list[int] = []
    forwards: list[int] = []
    depth = [0]

    def wrap(sink: list[int], original):
        def wrapper(input_ids, kv_cache, *args, **kwargs):
            outer = depth[0] == 0
            depth[0] += 1
            try:
                if outer:
                    sink.append(len(input_ids))
                return original(input_ids, kv_cache, *args, **kwargs)
            finally:
                depth[0] -= 1

        return wrapper

    model.encode = wrap(encodes, real_encode)
    model.forward = wrap(forwards, real_forward)
    return encodes, forwards


def restore_decoder(model) -> None:
    """Undo :func:`count_decoder_tokens` by rebinding the real methods."""
    for name in ("encode", "forward"):
        # ``model.encode`` is a bound method off the class, so dropping the
        # instance attribute is the whole undo.
        model.__dict__.pop(name, None)


# --------------------------------------------------------------------------- #
# common_prefix_length
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([], [], 0),
        ([], [1, 2], 0),
        ([1, 2], [], 0),
        ([1, 2, 3], [1, 2, 3], 3),
        ([1, 2, 3], [1, 2], 2),
        ([1, 2], [1, 2, 3], 2),
        ([1, 2, 9], [1, 2, 3], 2),
        ([9, 2, 3], [1, 2, 3], 0),
        ([1, 1, 1], [1, 1, 2, 1], 2),
    ],
)
def test_common_prefix_length(left, right, expected):
    assert common_prefix_length(left, right) == expected


def test_common_prefix_length_compares_ids_not_text():
    """The same words can tokenise differently at a turn boundary; ids decide.

    ``[7, 8]`` and ``[7, 9]`` decode to one string under a tokenizer that merges
    them, and reusing on that basis would condition the model on tokens it was
    never shown. Comparing ids cannot make that mistake.
    """
    assert common_prefix_length([7, 8], [7, 9]) == 1


# --------------------------------------------------------------------------- #
# construction and isolation
# --------------------------------------------------------------------------- #


def test_a_session_starts_empty(engine: Engine, session: Session):
    assert session.cached_tokens == 0
    assert session.turns == 0
    assert session.last_reuse is None
    assert session.engine is engine


def test_a_session_allocates_its_own_cache(engine: Engine, session: Session):
    """Not the engine's: a plain request in between must not wipe a conversation.

    The cost is a second full-size allocation, which is why ``--no-reuse`` exists
    and why this is a session's decision rather than the engine's default.
    """
    assert session.cache is not engine.kv_cache
    assert session.cache.max_context == engine.max_context
    assert session.cache.allocated_bytes() == engine.kv_cache.allocated_bytes()


def test_two_sessions_do_not_share_a_cache(engine: Engine):
    first, second = engine.session(), engine.session()
    assert first.cache is not second.cache
    first.generate_ids([1, 2, 3], sampling=GREEDY, max_new_tokens=1)
    assert second.cached_tokens == 0


def test_a_plain_request_between_turns_does_not_disturb_the_session(engine: Engine):
    """The isolation guarantee the other direction: sessions do not leak either."""
    session = engine.session()
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=4)
    cached_after_first = session.cached_tokens

    # An unrelated request resets the *engine's* cache, not the session's.
    engine.generate(conversation(1), sampling=GREEDY, max_new_tokens=4)

    assert session.cached_tokens == cached_after_first
    grown = append_reply(list(history), first) + [turn(1)]
    second = session.generate(grown, sampling=GREEDY, max_new_tokens=4)
    assert session.last_reuse.reused_tokens > 0
    assert second.token_ids == engine.generate(grown, sampling=GREEDY, max_new_tokens=4).token_ids


def test_a_session_leaves_the_engine_cache_alone(engine: Engine, session: Session):
    session.generate(conversation(1), sampling=GREEDY, max_new_tokens=3)
    assert engine.kv_cache.length == 0


# --------------------------------------------------------------------------- #
# reuse
# --------------------------------------------------------------------------- #


def test_the_first_turn_reuses_nothing(engine: Engine, session: Session):
    history = conversation(1)
    result = session.generate(history, sampling=GREEDY, max_new_tokens=4)
    reuse = session.last_reuse
    assert reuse.reused_tokens == 0
    assert reuse.prefilled_tokens == len(result.prompt_token_ids)
    assert reuse.discarded_tokens == 0
    assert result.metrics.reused_tokens == 0


def test_the_second_turn_reuses_the_first_turns_prefix(engine: Engine, session: Session):
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=6)
    grown = append_reply(list(history), first) + [turn(1)]

    second = session.generate(grown, sampling=GREEDY, max_new_tokens=6)

    reuse = session.last_reuse
    assert reuse.reused_tokens > 0
    # Turn two's prompt is turn one's prompt plus the reply plus the new question,
    # so most of it is already in the cache — but not all of it, because the
    # template's turn boundary re-tokenises.
    assert reuse.prefilled_tokens < reuse.prompt_tokens
    assert second.metrics.prompt_tokens == len(second.prompt_token_ids)
    assert second.metrics.reused_tokens == reuse.reused_tokens
    assert second.metrics.prefill_tokens == reuse.prefilled_tokens


def test_a_reused_turn_matches_a_standalone_request_token_for_token(engine: Engine):
    """The correctness claim every other test here rests on."""
    session = engine.session()
    history = conversation(1)
    for index in range(3):
        history.append(turn(index))
        via_session = session.generate(history, sampling=GREEDY, max_new_tokens=8)
        standalone = engine.generate(history, sampling=GREEDY, max_new_tokens=8)
        assert via_session.token_ids == standalone.token_ids
        assert via_session.text == standalone.text
        assert via_session.prompt_token_ids == standalone.prompt_token_ids
        # Turn one has nothing to reuse; every turn after it does.
        if index:
            assert session.last_reuse.reused_tokens > 0
        append_reply(history, via_session)


def test_prompt_tokens_still_counts_the_whole_rendered_history(engine: Engine, session: Session):
    """Reuse changes what was computed, not what the model was conditioned on.

    ``prompt_tokens`` shrinking on a reused turn would make prefill throughput
    incomparable between a chat and a one-shot request, and would hide the fact
    that the answer still depends on every token in the history.
    """
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=4)
    grown = append_reply(list(history), first) + [turn(1)]
    second = session.generate(grown, sampling=GREEDY, max_new_tokens=4)
    assert second.metrics.prompt_tokens == len(second.prompt_token_ids)
    assert second.metrics.prompt_tokens == len(engine.encode_messages(grown))


def test_a_reused_prefix_holds_the_same_kv_as_a_cold_prefill(engine: Engine):
    """Reuse changes how the cache got filled, not what is in it."""
    session = engine.session()
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=4)
    grown = append_reply(list(history), first) + [turn(1)]
    session.generate(grown, sampling=GREEDY, max_new_tokens=4)
    reused = session.last_reuse.reused_tokens
    warm = session.cache

    engine.generate(grown, sampling=GREEDY, max_new_tokens=4)
    cold = engine.kv_cache

    assert reused > 0
    assert warm.length == cold.length
    for layer in range(warm.num_layers):
        for warm_array, cold_array in (
            (warm.keys[layer], cold.keys[layer]),
            (warm.values[layer], cold.values[layer]),
        ):
            np.testing.assert_allclose(
                warm_array[:, : cold.length, :],
                cold_array[:, : cold.length, :],
                **CHUNK_TOLERANCE,
            )


def test_a_reused_turn_runs_fewer_decoder_tokens_than_a_cold_one(engine: Engine):
    """Reuse has to show up as work not done, not just as a number in a report."""
    session = engine.session()
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=6)
    grown = append_reply(list(history), first) + [turn(1)]

    encodes, forwards = count_decoder_tokens(engine.model)
    try:
        warm_result = session.generate(grown, sampling=GREEDY, max_new_tokens=6)
        warm = sum(encodes) + sum(forwards)
        reused = session.last_reuse.reused_tokens
        encodes.clear()
        forwards.clear()
        cold_result = engine.generate(grown, sampling=GREEDY, max_new_tokens=6)
        cold = sum(encodes) + sum(forwards)
    finally:
        restore_decoder(engine.model)

    assert warm < cold
    # Both requests answer the same prompt with the same sampler, so the entire
    # difference in decoder tokens is prefill the warm one did not have to do.
    assert cold_result.token_ids == warm_result.token_ids
    assert cold - warm == reused


def test_reuse_survives_a_multi_chunk_prefill(tiny_model_dir):
    """The reuse boundary need not land on a chunk edge."""
    engine = Engine.from_directory(tiny_model_dir, max_context=256, prefill_chunk_size=3)
    session = engine.session()
    history = conversation(1)
    for index in range(3):
        history.append(turn(index))
        via_session = session.generate(history, sampling=GREEDY, max_new_tokens=5)
        standalone = engine.generate(history, sampling=GREEDY, max_new_tokens=5)
        assert via_session.token_ids == standalone.token_ids
        assert via_session.metrics.prefill_chunks > 1
        append_reply(history, via_session)


# --------------------------------------------------------------------------- #
# the full-hit case
# --------------------------------------------------------------------------- #


def test_an_entirely_cached_prompt_still_runs_one_token(engine: Engine, session: Session):
    """The cache holds K/V for the last position; generation needs its logits.

    The token that would have produced them is the previous reply's stop token,
    which the decode loop never feeds back. So a full match re-runs exactly one
    position rather than reporting a zero-token prefill and sampling from a
    distribution belonging to some earlier token.
    """
    ids = engine.encode_messages(conversation(1))
    first = session.generate_ids(ids, sampling=GREEDY, max_new_tokens=5)
    assert session.last_reuse.reused_tokens == 0

    second = session.generate_ids(ids, sampling=GREEDY, max_new_tokens=5)

    reuse = session.last_reuse
    assert reuse.reused_tokens == len(ids) - 1
    assert reuse.prefilled_tokens == 1
    # The reply's tokens are gone: they described a continuation this request is
    # not asking for. All of them, because the cache never held the last one —
    # the decode loop samples a token and stops without feeding it back.
    assert reuse.discarded_tokens == len(first.token_ids)
    assert second.metrics.prefill_tokens == 1
    assert second.token_ids == first.token_ids


def test_a_full_hit_costs_one_forward_not_a_whole_prefill(engine: Engine, session: Session):
    ids = engine.encode_messages(conversation(1))
    session.generate_ids(ids, sampling=GREEDY, max_new_tokens=3)

    encodes, forwards = count_decoder_tokens(engine.model)
    try:
        session.generate_ids(ids, sampling=GREEDY, max_new_tokens=1)
        tokens = sum(encodes) + sum(forwards)
    finally:
        restore_decoder(engine.model)

    # One prefill of the single unmatched position; max_new_tokens=1 means the
    # first sampled token ends the request, so no decode forward runs at all.
    assert tokens == 1
    assert forwards == [1]


def test_a_prompt_shorter_than_the_cache_reuses_and_discards(engine: Engine, session: Session):
    """Asking an earlier question again must not read the later turn's KV."""
    short_ids = engine.encode_messages(conversation(1))
    long_ids = short_ids + [7, 11, 13]
    session.generate_ids(long_ids, sampling=GREEDY, max_new_tokens=3)
    assert session.cached_tokens > len(short_ids)

    result = session.generate_ids(short_ids, sampling=GREEDY, max_new_tokens=3)

    reuse = session.last_reuse
    assert reuse.reused_tokens == len(short_ids) - 1
    assert reuse.prefilled_tokens == 1
    assert reuse.discarded_tokens > 0
    assert result.token_ids == engine.generate_ids(
        short_ids, sampling=GREEDY, max_new_tokens=3
    ).token_ids


def test_a_template_need_not_render_a_prefix_of_itself(engine: Engine):
    """Why the reuse boundary is decided on ids and never on the message list.

    Two back-to-back user turns do not render as an extension of one, so the
    shorter conversation is *not* a token prefix of the longer one. A session that
    reasoned about messages ("turn one is unchanged, so reuse it") would hand the
    model KV computed for a different token sequence — wrong output with no error.
    Comparing ids under-reuses here instead, which costs prefill time and nothing
    else.
    """
    short = engine.encode_messages(conversation(1))
    long = engine.encode_messages(conversation(2))
    shared = common_prefix_length(short, long)
    assert short[:1] == long[:1]  # they do start together
    assert 0 < shared < len(short)  # and part way through they stop agreeing

    session = engine.session()
    session.generate_ids(long, sampling=GREEDY, max_new_tokens=1)
    session.generate_ids(short, sampling=GREEDY, max_new_tokens=1)

    reuse = session.last_reuse
    assert reuse.reused_tokens == shared
    assert reuse.prefilled_tokens == len(short) - shared


# --------------------------------------------------------------------------- #
# divergence: edited history, /clear
# --------------------------------------------------------------------------- #


def test_editing_an_earlier_turn_discards_the_cache_past_the_edit(engine: Engine, session: Session):
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=6)
    grown = append_reply(list(history), first) + [turn(1)]
    session.generate(grown, sampling=GREEDY, max_new_tokens=6)
    cached_before = session.cached_tokens

    edited = list(grown)
    edited[1] = ChatMessage(role="user", content="Different Q0.")
    result = session.generate(edited, sampling=GREEDY, max_new_tokens=6)

    reuse = session.last_reuse
    # The system message still matches; everything from the edited turn on is gone.
    assert 0 < reuse.reused_tokens < cached_before
    assert reuse.discarded_tokens > 0
    assert result.token_ids == engine.generate(edited, sampling=GREEDY, max_new_tokens=6).token_ids


def test_deleting_a_turn_shrinks_the_reused_prefix(engine: Engine, session: Session):
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=5)
    grown = append_reply(list(history), first) + [turn(1)]
    session.generate(grown, sampling=GREEDY, max_new_tokens=5)

    result = session.generate(history, sampling=GREEDY, max_new_tokens=5)
    assert result.token_ids == first.token_ids
    assert session.last_reuse.reused_tokens > 0


def test_reset_clears_the_prefix_but_keeps_the_allocation(engine: Engine, session: Session):
    session.generate(conversation(1), sampling=GREEDY, max_new_tokens=4)
    buffers = session.cache.keys[0]

    session.reset()

    assert session.cached_tokens == 0
    assert session.turns == 0
    assert session.last_reuse is None
    assert session.cache.keys[0] is buffers  # /clear is not a reallocation
    result = session.generate(conversation(1), sampling=GREEDY, max_new_tokens=4)
    assert session.last_reuse.reused_tokens == 0
    assert result.token_ids == engine.generate(
        conversation(1), sampling=GREEDY, max_new_tokens=4
    ).token_ids


def test_a_reset_session_reuses_nothing_from_the_conversation_before_it(engine: Engine):
    session = engine.session()
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=4)
    grown = append_reply(list(history), first) + [turn(1)]
    session.reset()
    session.generate(grown, sampling=GREEDY, max_new_tokens=4)
    assert session.last_reuse.reused_tokens == 0


# --------------------------------------------------------------------------- #
# cancellation and failure
# --------------------------------------------------------------------------- #


def test_abandoning_a_stream_mid_decode_leaves_the_cache_consistent(
    engine: Engine, session: Session, monkeypatch: pytest.MonkeyPatch, scripted_sampler
):
    """Ctrl-C in the CLI drops the generator; the cache must still make sense.

    ``_stream`` records each commit where the decoder makes it, with no yield in
    between, so an abandoned request leaves the session describing exactly the
    slots that exist — not an optimistic count that includes a half-written turn.

    The sampler is scripted to emit non-stop tokens so that "consumed two steps"
    means "one decode forward ran", rather than depending on what a random-weight
    model happens to argmax.
    """
    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler(non_stop_ids(engine, 8)))
    history = conversation(1)
    prompt_tokens = len(engine.encode_messages(history))

    stream = session.stream(history, sampling=GREEDY, max_new_tokens=8)
    seen = [next(stream), next(stream)]
    assert not seen[0].finished and not seen[1].finished
    stream.close()

    # Prefill committed the prompt; the second step ran one decode forward.
    assert session.cached_tokens == prompt_tokens + 1
    assert session.turns == 0  # a cancelled turn is not a completed one

    monkeypatch.undo()
    # And the session still works, and still agrees with a cold request.
    result = session.generate(history, sampling=GREEDY, max_new_tokens=8)
    assert result.token_ids == engine.generate(history, sampling=GREEDY, max_new_tokens=8).token_ids


def test_a_cancelled_turn_does_not_poison_the_next_one(engine: Engine, session: Session):
    """The CLI's Ctrl-C path: pop the user message, ask something else instead."""
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=5)
    grown = append_reply(list(history), first) + [turn(1)]

    abandoned = session.stream(grown, sampling=GREEDY, max_new_tokens=8)
    next(abandoned)
    abandoned.close()

    # The caller rolled its history back, so the next request does not contain the
    # cancelled question at all.
    result = session.generate(grown[:-1], sampling=GREEDY, max_new_tokens=5)
    assert result.token_ids == engine.generate(
        grown[:-1], sampling=GREEDY, max_new_tokens=5
    ).token_ids


def test_a_prefill_that_raises_after_committing_leaves_the_session_usable(engine: Engine):
    """``prefill`` commits chunk by chunk, so a failure can outrun the record.

    The session must not then believe its list of committed ids describes the
    whole cache. Recovering by ignoring the unrecorded tail is safe rather than
    lossy: the reuse boundary can never exceed what was recorded, and every slot
    at or past it is about to be overwritten or left beyond ``length``.
    """
    engine.prefill_chunk_size = 2
    session = engine.session()
    ids = engine.encode_messages(conversation(1))
    real_prefill = engine.model.prefill

    def exploding_prefill(input_ids, kv_cache, **kwargs):
        result = real_prefill(input_ids, kv_cache, **kwargs)
        raise RuntimeError("committed every chunk, then failed")

    engine.model.prefill = exploding_prefill
    try:
        with pytest.raises(RuntimeError, match="committed every chunk"):
            session.generate_ids(ids, sampling=GREEDY, max_new_tokens=3)
        # The cache really did advance past what the session recorded.
        assert session.cache.length == len(ids)
        assert session.cached_tokens == len(ids)
    finally:
        del engine.model.__dict__["prefill"]

    result = session.generate_ids(ids, sampling=GREEDY, max_new_tokens=3)
    assert result.token_ids == engine.generate_ids(ids, sampling=GREEDY, max_new_tokens=3).token_ids
    # Nothing was reusable: the session could not vouch for any of those slots.
    assert session.last_reuse.reused_tokens == 0


def test_a_request_that_overflows_is_refused_without_touching_the_cache(
    engine: Engine, session: Session
):
    history = conversation(1)
    session.generate(history, sampling=GREEDY, max_new_tokens=4)
    cached = session.cached_tokens
    reuse_before = session.last_reuse

    with pytest.raises(ContextOverflowError):
        session.generate(history, sampling=GREEDY, max_new_tokens=engine.max_context)

    # Validation runs before planning, so a bad request costs nothing: the good
    # prefix is still there and still reusable.
    assert session.cached_tokens == cached
    assert session.last_reuse is reuse_before


def test_a_prompt_longer_than_the_context_is_refused(engine: Engine, session: Session):
    ids = list(range(engine.max_context + 1))
    with pytest.raises(ContextOverflowError):
        session.generate_ids(ids, sampling=GREEDY, max_new_tokens=1)
    assert session.cached_tokens == 0


def test_an_empty_prompt_is_refused(engine: Engine, session: Session):
    from slowq.errors import UnsupportedInputError

    with pytest.raises(UnsupportedInputError, match="zero tokens"):
        session.generate_ids([], sampling=GREEDY, max_new_tokens=1)


# --------------------------------------------------------------------------- #
# model identity
# --------------------------------------------------------------------------- #


def test_swapping_the_engines_model_invalidates_the_session(engine: Engine, tiny_model_dir):
    session = engine.session()
    session.generate(conversation(1), sampling=GREEDY, max_new_tokens=3)

    engine.model = Engine.from_directory(tiny_model_dir, max_context=256).model

    with pytest.raises(RuntimeError, match="model was replaced"):
        session.generate(conversation(1), sampling=GREEDY, max_new_tokens=3)


def test_growing_max_context_under_a_session_is_refused(engine: Engine, session: Session):
    """The buffers were allocated for the old budget; a longer prompt would not fit.

    Without this the failure surfaces as a ``ContextOverflowError`` from inside a
    prefill chunk, blaming the prompt for a limit the session's cache never had.
    """
    session.generate(conversation(1), sampling=GREEDY, max_new_tokens=3)
    engine.max_context = engine.max_context * 2

    with pytest.raises(RuntimeError, match="geometry changed"):
        session.generate(conversation(1), sampling=GREEDY, max_new_tokens=3)


# --------------------------------------------------------------------------- #
# scope: text only
# --------------------------------------------------------------------------- #


def test_a_request_cannot_reuse_kv_and_substitute_image_features(engine: Engine, session: Session):
    """The guard lives in ``_stream`` because that is what owns the invariant.

    A cached prefix was embedded from the token table. Nothing in the cache
    records which image a substituted row came from, so reusing one against a
    different picture would be silently wrong rather than merely stale.
    """
    ids = engine.encode_messages(conversation(1))
    session.generate_ids(ids, sampling=GREEDY, max_new_tokens=2)
    features = np.zeros((1, engine.config.hidden_size), dtype=np.float32)
    mask = np.zeros(len(ids), dtype=bool)
    mask[0] = True

    steps = engine._stream(
        ids,
        cache=session.cache,
        prefill_from=1,
        sampling=GREEDY,
        max_new_tokens=2,
        tokenize_seconds=0.0,
        vision_encode_seconds=0.0,
        probe=None,
        prefill_embeds=(features, mask),
    )
    with pytest.raises(ValueError, match="text-only"):
        next(steps)


def test_a_session_stream_ids_accepts_a_probe(engine: Engine, session: Session):
    """Debug observation is not something reuse gets to drop.

    A probe's positions are relative to the block being forwarded, and a reused
    turn's block starts at the reuse boundary — so what a session captures is the
    newly prefilled tail, which is the part that has not already been looked at.
    """
    from slowq.model import DebugProbe

    ids = engine.encode_messages(conversation(1))
    session.generate_ids(ids, sampling=GREEDY, max_new_tokens=2)
    probe = DebugProbe.for_names({"logits"})
    session.generate_ids(ids, sampling=GREEDY, max_new_tokens=2, probe=probe)
    assert probe.records
    assert any(key.endswith("logits[-1]") for key in probe.records)


def test_a_probe_that_wants_logits_still_gets_them_on_a_reused_turn(
    engine: Engine, session: Session
):
    """``prefill`` projects every chunk for a logits probe; reuse must not undo that."""
    from slowq.model import DebugProbe

    engine.prefill_chunk_size = 2
    ids = engine.encode_messages(conversation(1))
    session.generate_ids(ids, sampling=GREEDY, max_new_tokens=2)

    probe = DebugProbe.for_names({"logits"})
    session.generate_ids(ids, sampling=GREEDY, max_new_tokens=1, probe=probe)
    assert session.last_reuse.prefilled_tokens == 1
    assert probe.records


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_the_report_shows_reuse_only_when_there_is_some(engine: Engine, session: Session):
    history = conversation(1)
    cold = session.generate(history, sampling=GREEDY, max_new_tokens=4)
    assert cold.metrics.reused_tokens == 0
    assert "KV reuse" not in dict(cold.metrics.report())
    # A standalone request never reuses, so its report is unchanged by this feature.
    standalone = engine.generate(history, sampling=GREEDY, max_new_tokens=4)
    assert "KV reuse" not in dict(standalone.metrics.report())

    grown = append_reply(list(history), cold) + [turn(1)]
    warm = session.generate(grown, sampling=GREEDY, max_new_tokens=4)
    assert warm.metrics.reused_tokens > 0
    row = dict(warm.metrics.report())["KV reuse"]
    assert f"{warm.metrics.reused_tokens}/{warm.metrics.prompt_tokens}" in row
    assert "prefilled" in row


def test_prefill_throughput_is_measured_against_the_tokens_actually_prefilled(
    engine: Engine, session: Session
):
    """Otherwise a reused turn reports a rate for work it skipped."""
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=4)
    grown = append_reply(list(history), first) + [turn(1)]
    metrics = session.generate(grown, sampling=GREEDY, max_new_tokens=4).metrics

    assert metrics.reused_tokens > 0
    assert metrics.prefill_tokens == metrics.prompt_tokens - metrics.reused_tokens
    assert metrics.prefill_tokens_per_second == pytest.approx(
        metrics.prefill_tokens / metrics.prefill_seconds
    )
    # The rate is derived from the tokens actually pushed through the stack, so a
    # reused turn cannot report throughput for prefill it skipped. What it must
    # not do is silently keep the old ``prompt_tokens`` formula: the two differ by
    # exactly the reuse, which is asserted rather than timed, because a wall-clock
    # inequality on a 400-microsecond prefill is noise, not evidence.
    assert metrics.prefill_tokens < metrics.prompt_tokens


def test_a_cold_requests_prefill_throughput_is_unchanged_by_this_feature(engine: Engine):
    """``reused_tokens`` defaults to 0, so the pre-existing formula still holds."""
    metrics = engine.generate(conversation(1), sampling=GREEDY, max_new_tokens=3).metrics
    assert metrics.reused_tokens == 0
    assert metrics.prefill_tokens == metrics.prompt_tokens
    assert metrics.prefill_tokens_per_second == pytest.approx(
        metrics.prompt_tokens / metrics.prefill_seconds
    )


def test_as_dict_carries_the_reuse_fields(engine: Engine, session: Session):
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=3)
    grown = append_reply(list(history), first) + [turn(1)]
    payload = session.generate(grown, sampling=GREEDY, max_new_tokens=3).metrics.as_dict()
    for key in ("reused_tokens", "prefill_tokens", "prompt_tokens"):
        assert key in payload
    assert payload["reused_tokens"] > 0
    assert payload["prefill_tokens"] == payload["prompt_tokens"] - payload["reused_tokens"]


def test_session_reuse_describes_itself():
    reuse = SessionReuse(
        prompt_tokens=80,
        reused_tokens=60,
        prefilled_tokens=20,
        discarded_tokens=7,
        cache_length_before=67,
    )
    assert reuse.reuse_ratio == pytest.approx(0.75)
    text = reuse.describe()
    assert "60/80" in text and "prefilled 20" in text and "discarded 7" in text


def test_session_reuse_ratio_is_zero_for_an_empty_prompt():
    assert SessionReuse(0, 0, 0, 0, 0).reuse_ratio == 0.0


def test_the_session_repr_reports_what_it_holds(engine: Engine, session: Session):
    session.generate(conversation(1), sampling=GREEDY, max_new_tokens=3)
    text = repr(session)
    assert "turns=1" in text and "cached_tokens=" in text and "MiB" in text


def test_the_completed_turn_count_ignores_cancelled_requests(
    engine: Engine, session: Session, monkeypatch: pytest.MonkeyPatch, scripted_sampler
):
    monkeypatch.setattr("slowq.engine.Sampler", scripted_sampler(non_stop_ids(engine, 8)))
    stream = session.stream(conversation(1), sampling=GREEDY, max_new_tokens=8)
    assert not next(stream).finished
    stream.close()
    assert session.turns == 0

    monkeypatch.undo()
    session.generate(conversation(1), sampling=GREEDY, max_new_tokens=2)
    assert session.turns == 1


# ``_turns`` used to be incremented *after* ``yield from`` returned. A generator
# that is closed never returns from a yield, so a caller that stopped reading at
# the final step — the natural thing to do once ``result`` is in hand — left the
# session reporting zero turns for a request that had completed and been handed
# over. Completion is therefore recorded as the final step is delivered, which
# also means a stream that dies or is abandoned on the way there counts nothing.


def test_a_stream_closed_on_its_final_step_still_counts_the_turn(
    engine: Engine, session: Session
):
    """One token of budget: the first step *is* the final one, and it is closed."""
    stream = session.stream(conversation(1), sampling=GREEDY, max_new_tokens=1)
    step = next(stream)
    assert step.finished and step.result is not None
    stream.close()

    assert session.turns == 1
    assert "turns=1" in repr(session)


def test_a_stream_closed_after_an_eos_step_still_counts_the_turn(
    engine: Engine, session: Session, monkeypatch: pytest.MonkeyPatch, scripted_sampler
):
    """The same shape reached by a stop token rather than by the token budget.

    The scripted sampler keeps "closed after the third step" meaning "closed
    after the *final* step": with random weights the model could argmax onto EOS
    at any point, and a test whose subject is which step was last cannot leave
    that to chance.
    """
    stop = sorted(engine.stop_ids)[0]
    monkeypatch.setattr(
        "slowq.engine.Sampler", scripted_sampler([*non_stop_ids(engine, 2), stop])
    )
    stream = session.stream(conversation(1), sampling=GREEDY, max_new_tokens=8)
    steps = [next(stream), next(stream), next(stream)]
    assert [step.finished for step in steps] == [False, False, True]
    assert steps[-1].token_id == stop
    stream.close()

    assert session.turns == 1


def test_exhausting_a_stream_counts_the_turn_exactly_once(engine: Engine, session: Session):
    """The delivery-based count must not also count the generator's own finish."""
    steps = list(session.stream(conversation(1), sampling=GREEDY, max_new_tokens=3))
    assert sum(step.finished for step in steps) == 1, "one final step per request"
    assert session.turns == 1

    session.generate(conversation(1), sampling=GREEDY, max_new_tokens=3)
    assert session.turns == 2


def test_a_decode_that_raises_mid_stream_does_not_count_the_turn(
    engine: Engine, session: Session
):
    """An exception is not a completion, and the session has to survive it.

    The prefill is let through whatever its chunking, and the second single-token
    decode fails, so the caller has already been given one step when the stream
    dies. That is the case where counting at delivery and counting at the end
    genuinely differ, and the one where counting would be wrong.
    """
    ids = engine.encode_messages(conversation(1))
    real_forward = engine.model.forward
    decode_calls = [0]

    def failing_forward(input_ids, kv_cache, *args, **kwargs):
        if len(input_ids) == 1:
            decode_calls[0] += 1
            if decode_calls[0] > 1:
                raise RuntimeError("decoder failed mid-stream")
        return real_forward(input_ids, kv_cache, *args, **kwargs)

    engine.model.forward = failing_forward
    try:
        with pytest.raises(RuntimeError, match="decoder failed mid-stream"):
            list(session.stream_ids(ids, sampling=GREEDY, max_new_tokens=6))
    finally:
        del engine.model.__dict__["forward"]

    assert session.turns == 0
    result = session.generate_ids(ids, sampling=GREEDY, max_new_tokens=3)
    assert result.token_ids == engine.generate_ids(
        ids, sampling=GREEDY, max_new_tokens=3
    ).token_ids
    assert session.turns == 1


def test_a_turn_counted_at_delivery_does_not_outrun_the_reuse_it_reports(
    engine: Engine, session: Session
):
    """``turns`` and ``last_reuse`` describe the same request, or neither does.

    Counting completion early would be a small lie on its own; paired with a
    ``last_reuse`` recorded at planning time it would be two numbers that a
    reader reasonably expects to belong together.
    """
    history = conversation(1)
    stream = session.stream(history, sampling=GREEDY, max_new_tokens=1)
    final = next(stream)
    stream.close()

    assert session.turns == 1
    assert session.last_reuse is not None
    assert session.last_reuse.reused_tokens == 0  # the first turn has nothing to reuse

    grown = append_reply(list(history), final.result) + [turn(1)]
    session.generate(grown, sampling=GREEDY, max_new_tokens=1)
    assert session.turns == 2
    assert session.last_reuse.reused_tokens > 0


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #


def test_a_streamed_session_turn_yields_the_same_text_as_a_blocking_one(
    engine: Engine, session: Session
):
    history = conversation(1)
    first = session.generate(history, sampling=GREEDY, max_new_tokens=6)
    grown = append_reply(list(history), first) + [turn(1)]

    deltas: list[str] = []
    result = None
    for step in session.stream(grown, sampling=GREEDY, max_new_tokens=6):
        deltas.append(step.text_delta)
        if step.result is not None:
            result = step.result
    assert result is not None
    assert "".join(deltas) == result.text
    assert result.metrics.reused_tokens > 0


def test_a_session_accepts_a_raw_prompt(engine: Engine, session: Session):
    result = session.generate(prompt="hi", sampling=GREEDY, max_new_tokens=3)
    assert result.token_ids == engine.generate(
        prompt="hi", sampling=GREEDY, max_new_tokens=3
    ).token_ids


def test_a_session_needs_exactly_one_of_messages_or_prompt(engine: Engine, session: Session):
    with pytest.raises(ValueError, match="exactly one"):
        session.generate(sampling=GREEDY, max_new_tokens=2)
    with pytest.raises(ValueError, match="exactly one"):
        session.generate(conversation(1), prompt="hi", sampling=GREEDY, max_new_tokens=2)
