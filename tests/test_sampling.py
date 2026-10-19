"""Sampling: greedy, temperature, top-k, top-p, seeds, and parameter validation."""

from __future__ import annotations

import numpy as np
import pytest

from slowq.errors import SamplingError
from slowq.sampling import Sampler, SamplingParams

VOCAB = 64


def logits_with(values: dict[int, float], size: int = VOCAB, base: float = -10.0) -> np.ndarray:
    vector = np.full(size, base, dtype=np.float32)
    for index, value in values.items():
        vector[index] = value
    return vector


def draw(sampler: Sampler, logits: np.ndarray, count: int = 400) -> list[int]:
    return [sampler.sample(logits) for _ in range(count)]


# --------------------------------------------------------------------------- #
# greedy
# --------------------------------------------------------------------------- #


def test_greedy_returns_the_argmax():
    sampler = Sampler(SamplingParams(temperature=0.0))
    assert sampler.sample(logits_with({7: 5.0, 3: 2.0})) == 7


def test_greedy_breaks_ties_towards_the_lowest_token_id():
    sampler = Sampler(SamplingParams(temperature=0.0))
    assert sampler.sample(logits_with({9: 1.0, 4: 1.0, 30: 1.0})) == 4


def test_temperature_zero_means_greedy_and_ignores_top_k_top_p():
    """A ``temperature=0`` request must not be silently filtered by top_p."""
    sampler = Sampler(SamplingParams(temperature=0.0, top_k=1, top_p=0.01, seed=123))
    assert sampler.is_greedy
    logits = logits_with({20: 1.0, 5: 5.0})
    assert draw(sampler, logits, count=20) == [5] * 20


def test_default_params_are_greedy():
    assert Sampler().is_greedy
    assert Sampler().sample(logits_with({2: 0.5})) == 2


def test_greedy_ignores_the_seed_entirely():
    logits = logits_with({3: 1.0, 4: 0.5})
    for seed in (None, 0, 1, 2026):
        assert Sampler(SamplingParams(temperature=0.0, seed=seed)).sample(logits) == 3


# --------------------------------------------------------------------------- #
# parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=-1.0),
        SamplingParams(temperature=float("nan")),
        SamplingParams(temperature=float("inf")),
        SamplingParams(temperature="0.7"),
        SamplingParams(top_k=-1),
        SamplingParams(top_k=1.5),
        SamplingParams(top_k=True),
        SamplingParams(top_p=0.0),
        SamplingParams(top_p=-0.5),
        SamplingParams(top_p=1.5),
        SamplingParams(top_p=float("nan")),
        SamplingParams(seed="abc"),
        SamplingParams(seed=1.5),
    ],
)
def test_invalid_parameters_are_rejected(params: SamplingParams):
    with pytest.raises(SamplingError):
        Sampler(params)


def test_valid_boundary_parameters_are_accepted():
    Sampler(SamplingParams(temperature=0.0))
    Sampler(SamplingParams(temperature=1e-6))
    Sampler(SamplingParams(temperature=0.8, top_k=0, top_p=1.0))
    Sampler(SamplingParams(temperature=0.8, top_k=1, top_p=1e-6, seed=0))


def test_describe_reports_the_mode():
    assert "greedy" in SamplingParams().describe()
    text = SamplingParams(temperature=0.7, top_k=20, top_p=0.8, seed=3).describe()
    for fragment in ("temperature=0.7", "top_k=20", "top_p=0.8", "seed=3"):
        assert fragment in text


# --------------------------------------------------------------------------- #
# logits validation
# --------------------------------------------------------------------------- #


def test_non_finite_logits_are_refused():
    sampler = Sampler(SamplingParams(temperature=0.8, seed=1))
    bad = logits_with({0: 1.0})
    bad[5] = np.nan
    with pytest.raises(SamplingError, match="not finite"):
        sampler.sample(bad)


def test_infinite_logits_are_refused_even_in_greedy_mode():
    bad = logits_with({0: 1.0})
    bad[5] = np.inf
    with pytest.raises(SamplingError, match="not finite"):
        Sampler().sample(bad)


def test_empty_logits_are_refused():
    with pytest.raises(SamplingError, match="empty"):
        Sampler().sample(np.zeros(0, dtype=np.float32))


def test_two_dimensional_logits_are_refused():
    with pytest.raises(SamplingError, match="1-D"):
        Sampler().sample(np.zeros((1, VOCAB), dtype=np.float32))


def test_sampling_does_not_mutate_the_callers_logits():
    """The engine feeds the same array shape every step; in-place edits would rot."""
    sampler = Sampler(SamplingParams(temperature=1.0, top_k=4, top_p=0.9, seed=7))
    logits = logits_with({1: 3.0, 2: 2.0, 3: 1.0, 4: 0.5, 5: 0.0})
    original = logits.copy()
    draw(sampler, logits, count=10)
    np.testing.assert_array_equal(logits, original)


# --------------------------------------------------------------------------- #
# temperature
# --------------------------------------------------------------------------- #


def test_a_low_temperature_sharpens_the_distribution():
    logits = logits_with({10: 2.0, 11: 1.0, 12: 0.0})
    cold = draw(Sampler(SamplingParams(temperature=0.01, seed=5)), logits, 300)
    hot = draw(Sampler(SamplingParams(temperature=20.0, seed=5)), logits, 300)
    assert cold.count(10) / len(cold) > hot.count(10) / len(hot)
    assert cold.count(10) == len(cold)  # effectively greedy at T -> 0


def test_a_high_temperature_approaches_uniform_over_candidates():
    logits = logits_with({0: 3.0, 1: 2.0, 2: 1.0}, base=-3.0)
    draws = draw(Sampler(SamplingParams(temperature=1000.0, seed=11)), logits, 900)
    fractions = [draws.count(index) / len(draws) for index in range(3)]
    assert max(fractions) - min(fractions) < 0.12


# --------------------------------------------------------------------------- #
# top-k
# --------------------------------------------------------------------------- #


def test_top_k_never_samples_outside_the_k_best_tokens():
    logits = logits_with({1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.0})
    draws = draw(Sampler(SamplingParams(temperature=1.0, top_k=2, seed=3)), logits, 500)
    assert set(draws) <= {1, 2}


def test_top_k_of_one_is_greedy_even_at_a_high_temperature():
    logits = logits_with({6: 4.0, 7: 3.0})
    draws = draw(Sampler(SamplingParams(temperature=5.0, top_k=1, seed=9)), logits, 50)
    assert set(draws) == {6}


def test_top_k_zero_disables_the_filter():
    logits = logits_with({1: 5.0, 2: 4.9}, base=4.8)
    draws = draw(Sampler(SamplingParams(temperature=1.0, top_k=0, seed=4)), logits, 600)
    assert len(set(draws)) > 2


def test_top_k_larger_than_the_vocabulary_disables_the_filter():
    logits = logits_with({1: 5.0, 2: 4.0, 3: 3.0})
    draws = draw(Sampler(SamplingParams(temperature=1.0, top_k=VOCAB + 5, seed=4)), logits, 400)
    assert set(draws) == {1, 2, 3}


def test_top_k_keeps_ties_at_the_threshold():
    """``logits < threshold`` is masked, so equal logits all survive."""
    logits = logits_with({1: 2.0, 2: 2.0, 3: 2.0, 4: 0.0})
    draws = draw(Sampler(SamplingParams(temperature=1.0, top_k=2, seed=13)), logits, 600)
    assert {1, 2, 3} <= set(draws)
    assert 4 not in draws


# --------------------------------------------------------------------------- #
# top-p
# --------------------------------------------------------------------------- #


def test_top_p_drops_the_low_probability_tail():
    # p ~ [0.86, 0.12, 0.02] after softmax; top_p=0.9 keeps the first two.
    logits = logits_with({1: 4.0, 2: 2.0, 3: 0.0})
    draws = draw(Sampler(SamplingParams(temperature=1.0, top_p=0.9, seed=17)), logits, 500)
    assert set(draws) <= {1, 2}


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    """HF keeps the crossing token instead of dropping the mass below top_p.

    p ~= [0.665, 0.245, 0.090]; at top_p=0.7 the cumulative reaches 0.910 on
    token 2, so token 2 is the one that crosses and must stay.
    """
    logits = logits_with({1: 4.0, 2: 3.0, 3: 2.0})
    draws = draw(Sampler(SamplingParams(temperature=1.0, top_p=0.7, seed=19)), logits, 400)
    assert 1 in draws and 2 in draws
    assert 3 not in draws


def test_top_p_of_one_disables_the_filter():
    logits = logits_with({1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0})
    draws = draw(Sampler(SamplingParams(temperature=1.0, top_p=1.0, seed=23)), logits, 800)
    assert len(set(draws)) >= 3


def test_a_very_small_top_p_still_leaves_one_candidate():
    logits = logits_with({1: 5.0, 2: 0.0, 3: -5.0})
    draws = draw(Sampler(SamplingParams(temperature=1.0, top_p=1e-6, seed=29)), logits, 20)
    assert set(draws) == {1}


def test_top_k_and_top_p_combine_in_the_hf_order():
    """temperature -> top-k -> top-p: top-p chooses from the top-k survivors."""
    logits = logits_with({1: 6.0, 2: 5.0, 3: 4.0, 4: 3.0})
    draws = draw(
        Sampler(SamplingParams(temperature=1.0, top_k=3, top_p=0.7, seed=31)), logits, 500
    )
    # top-k=3 drops token 4; top-p=0.7 over {1,2,3} keeps exactly {1,2}.
    assert 1 in draws and 2 in draws
    assert not ({3, 4} & set(draws))


# --------------------------------------------------------------------------- #
# seeds
# --------------------------------------------------------------------------- #


def test_a_fixed_seed_reproduces_the_same_token_sequence():
    params = SamplingParams(temperature=1.0, top_p=0.95, seed=2026)
    logits = np.linspace(-2.0, 2.0, VOCAB, dtype=np.float32)
    first = draw(Sampler(params), logits, 60)
    second = draw(Sampler(params), logits, 60)
    assert first == second


def test_reseed_replays_the_same_stream():
    sampler = Sampler(SamplingParams(temperature=1.0, seed=99))
    logits = np.linspace(-2.0, 2.0, VOCAB, dtype=np.float32)
    first = draw(sampler, logits, 40)
    sampler.reseed(99)
    assert draw(sampler, logits, 40) == first


def test_different_seeds_give_different_sequences():
    logits = np.linspace(-1.0, 1.0, VOCAB, dtype=np.float32)
    a = draw(Sampler(SamplingParams(temperature=1.0, seed=1)), logits, 40)
    b = draw(Sampler(SamplingParams(temperature=1.0, seed=2)), logits, 40)
    assert a != b


def test_an_unseeded_sampler_is_not_reproducible_but_is_valid():
    logits = np.linspace(-1.0, 1.0, VOCAB, dtype=np.float32)
    a = draw(Sampler(SamplingParams(temperature=1.0)), logits, 40)
    b = draw(Sampler(SamplingParams(temperature=1.0)), logits, 40)
    assert all(0 <= token < VOCAB for token in a + b)
    assert a != b


def test_the_sampler_leaves_the_global_numpy_rng_alone():
    """A library that reseeds ``np.random`` would break the caller's own code."""
    np.random.seed(1234)
    expected = np.random.random(3)
    np.random.seed(1234)
    sampler = Sampler(SamplingParams(temperature=1.0, seed=7))
    draw(sampler, np.linspace(-1.0, 1.0, VOCAB, dtype=np.float32), 50)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_the_sampling_distribution_matches_the_softmax_of_the_logits():
    logits = np.array([2.0, 1.0, 0.0] + [-30.0] * (VOCAB - 3), dtype=np.float32)
    shifted = logits[:3] - logits[:3].max()
    expected = np.exp(shifted, dtype=np.float64)
    expected /= expected.sum()
    draws = draw(Sampler(SamplingParams(temperature=1.0, seed=77)), logits, 6000)
    for index in range(3):
        observed = draws.count(index) / len(draws)
        assert observed == pytest.approx(expected[index], abs=0.03)
    assert not any(token >= 3 for token in draws)
