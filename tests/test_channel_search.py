"""Choosing scoring channels by how two-moded their signal is.

The premise, measured on session8 rather than assumed: a channel's bimodality
is label-free, and it ranks channels the same way their actual REM-from-NREM
separation does (+0.87 rank correlation). That is what makes a search possible
before anything has been scored.

These fixtures build a probe where only some channels carry a state-dependent
signal, so a search that works has to find them and a search that does not has
nowhere to hide.
"""

import numpy as np
import pytest

from spikeshpc.states import (
    bimodality_score,
    pick_bimodal_channels,
    pick_channels,
    rank_channels_by_bimodality,
    sampled_spectra,
)

FS = 250.0
DURATION = 900.0
N_CHANNELS = 16
# the channels given a real state-dependent signal
THETA_CHANNELS = [5, 6, 7]
SW_CHANNELS = [11, 12, 13]

CONFIG = {
    "window_s": 4.0,
    "freq_range": [1.0, 100.0],
    "n_freqs": 60,
    "theta_band": [5.0, 10.0],
    "theta_ref_band": [2.0, 16.0],
    "slow_wave_max_hz": 32.0,
}


def make_recording():
    """Half the session in a theta state, half in a slow-wave state.

    Only THETA_CHANNELS get the 7 Hz bout and only SW_CHANNELS get the 2 Hz
    one; every other channel is pink-ish noise with the same variance, so a
    selector that picks by amplitude or by position cannot pass.
    """
    import spikeinterface.full as si

    rng = np.random.default_rng(0)
    n = int(DURATION * FS)
    t = np.arange(n) / FS

    # long alternating bouts, so windows land cleanly inside one state or other
    state = (np.floor(t / 60.0).astype(int) % 2).astype(bool)   # 60 s blocks

    traces = rng.normal(0, 20.0, (n, N_CHANNELS)).astype(np.float32)
    for c in THETA_CHANNELS:
        traces[:, c] += (60.0 * state * np.sin(2 * np.pi * 7.0 * t)).astype(np.float32)
    for c in SW_CHANNELS:
        # 1.4 Hz, deliberately below the theta ratio's 2-16 Hz denominator. At
        # 2 Hz this fixture had the slow-wave channels winning the *theta*
        # search: a big state-dependent oscillation inside the reference band
        # swings the ratio through its denominator, so those channels are
        # genuinely bimodal in theta ratio without carrying any theta. Real
        # probes can do this too -- it is a reason to read the chosen channels
        # rather than trust them blindly.
        traces[:, c] += (
            80.0 * (~state) * np.sin(2 * np.pi * 1.4 * t)
        ).astype(np.float32)

    rec = si.NumpyRecording([traces], sampling_frequency=FS)
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)
    rec.set_dummy_probe_from_locations(
        np.column_stack([np.zeros(N_CHANNELS), np.arange(N_CHANNELS) * 20.0])
    )
    return rec


@pytest.fixture(scope="module")
def rec():
    return make_recording()


# ── the score itself ────────────────────────────────────────────────────
def test_two_separated_modes_score_higher_than_one():
    rng = np.random.default_rng(0)
    one = rng.normal(0, 1, 2000)
    two = np.r_[rng.normal(-3, 1, 1000), rng.normal(3, 1, 1000)]
    assert bimodality_score(two) > 3 * bimodality_score(one)


def test_the_unimodal_floor_is_not_zero():
    """A two-component fit to one Gaussian still separates its two components.

    So the scale has a floor near 1.5, not 0, and a score of 2 means "about as
    two-moded as noise". Only differences between channels mean anything.
    """
    rng = np.random.default_rng(3)
    floors = [bimodality_score(rng.normal(0, 1, 3000)) for _ in range(5)]
    assert 1.0 < np.median(floors) < 2.2, floors


def test_the_score_grows_with_separation():
    rng = np.random.default_rng(1)
    scores = [
        bimodality_score(np.r_[rng.normal(-d, 1, 1000), rng.normal(d, 1, 1000)])
        for d in (0.5, 1.5, 3.0)
    ]
    assert scores[0] < scores[1] < scores[2]


def test_a_degenerate_signal_scores_nan():
    assert np.isnan(bimodality_score(np.ones(500)))
    assert np.isnan(bimodality_score(np.arange(10)))       # too few samples


# ── sampling ────────────────────────────────────────────────────────────
def test_the_spectra_are_on_the_scoring_frequency_grid(rec):
    freqs, spec = sampled_spectra(rec, rec.channel_ids[:4], CONFIG, n_windows=20)
    assert spec.shape == (4, CONFIG["n_freqs"], 20)
    expected = np.logspace(np.log10(1.0), np.log10(100.0), CONFIG["n_freqs"])
    np.testing.assert_allclose(freqs, expected)


def test_sampling_is_deterministic(rec):
    a = sampled_spectra(rec, rec.channel_ids[:3], CONFIG, n_windows=20)[1]
    b = sampled_spectra(rec, rec.channel_ids[:3], CONFIG, n_windows=20, seed=7)[1]
    np.testing.assert_array_equal(a, b)


def test_too_short_a_recording_is_refused(rec):
    with pytest.raises(ValueError, match="too short"):
        sampled_spectra(rec, rec.channel_ids[:2], {**CONFIG, "window_s": 10000.0})


# ── the ranking ─────────────────────────────────────────────────────────
def test_the_theta_search_finds_the_theta_channels(rec):
    ranked = rank_channels_by_bimodality(rec, rec.channel_ids, CONFIG, "theta",
                                         n_windows=120)
    top = [int(c) for c, _ in ranked[:3]]
    assert set(top) == set(THETA_CHANNELS), ranked[:6]


def test_the_slow_wave_search_finds_the_slow_wave_channels(rec):
    ranked = rank_channels_by_bimodality(rec, rec.channel_ids, CONFIG, "slow_wave",
                                         n_windows=120)
    top = [int(c) for c, _ in ranked[:3]]
    assert set(top) == set(SW_CHANNELS), ranked[:6]


def test_the_two_searches_disagree(rec):
    """They must, or the search is picking channels on something generic."""
    theta = {int(c) for c, _ in rank_channels_by_bimodality(
        rec, rec.channel_ids, CONFIG, "theta", n_windows=120)[:3]}
    sw = {int(c) for c, _ in rank_channels_by_bimodality(
        rec, rec.channel_ids, CONFIG, "slow_wave", n_windows=120)[:3]}
    assert not (theta & sw)


def test_ranking_is_best_first_with_nan_last(rec):
    ranked = rank_channels_by_bimodality(rec, rec.channel_ids, CONFIG, "theta",
                                         n_windows=120)
    scores = np.array([s for _, s in ranked])
    finite = scores[np.isfinite(scores)]
    assert list(finite) == sorted(finite, reverse=True)
    # every NaN sits after every real score, so [:n] is always the best n
    assert np.all(np.isfinite(scores[: len(finite)]))


def test_an_unknown_signal_is_refused(rec):
    with pytest.raises(ValueError, match="slow_wave.*theta"):
        rank_channels_by_bimodality(rec, rec.channel_ids[:2], CONFIG, "gamma")


# ── picking ─────────────────────────────────────────────────────────────
def test_picking_returns_the_best_channels_in_probe_order(rec):
    chosen = pick_bimodal_channels(rec, 3, CONFIG, "theta", candidate_step=1,
                                   n_windows=120, verbose=False)
    assert [int(c) for c in chosen] == sorted(THETA_CHANNELS)


def test_it_beats_spreading_evenly_on_this_probe(rec):
    """The whole justification: evenly spaced can miss the signal entirely."""
    searched = {int(c) for c in pick_bimodal_channels(
        rec, 3, CONFIG, "theta", candidate_step=1, n_windows=120, verbose=False)}
    spread = {int(c) for c in pick_channels(rec, 3)}
    assert len(searched & set(THETA_CHANNELS)) == 3
    assert len(spread & set(THETA_CHANNELS)) < 3


def test_excluded_channels_are_never_chosen(rec):
    chosen = pick_bimodal_channels(rec, 3, CONFIG, "theta", exclude=["5", "6"],
                                   candidate_step=1, n_windows=120, verbose=False)
    assert "5" not in [str(c) for c in chosen]
    assert "6" not in [str(c) for c in chosen]


def test_asking_for_everything_returns_everything(rec):
    chosen = pick_bimodal_channels(rec, N_CHANNELS + 5, CONFIG, "theta",
                                   n_windows=40, verbose=False)
    assert len(chosen) == N_CHANNELS


def test_excluding_everything_is_refused(rec):
    with pytest.raises(ValueError, match="No channels left"):
        pick_bimodal_channels(rec, 2, CONFIG, "theta",
                              exclude=[str(c) for c in rec.channel_ids])


def test_the_candidate_step_reduces_the_search(rec):
    """Neighbouring contacts see the same field; scoring all of them is waste."""
    chosen = pick_bimodal_channels(rec, 2, CONFIG, "theta", candidate_step=8,
                                   n_windows=60, verbose=False)
    assert len(chosen) == 2
    # with step 8 the candidates are 0 and 8, so it cannot find 5/6/7 -- the
    # point is that it still returns a valid, in-order answer
    assert list(chosen) == sorted(chosen, key=lambda c: int(c))


def test_it_reports_what_it_searched(rec, capsys):
    pick_bimodal_channels(rec, 3, CONFIG, "theta", candidate_step=1, n_windows=120)
    printed = capsys.readouterr().out
    assert "theta: searched" in printed and "bimodality" in printed
