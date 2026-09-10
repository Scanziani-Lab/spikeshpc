"""Bayesian head-direction decoding: encoding model, decode, shuffle control.

The synthetic population below is a ring attractor that actually works: 24 von
Mises cells spread evenly over the circle, firing Poisson against a heading
that random-walks. It walks rather than sweeps on purpose -- a periodic sweep
comes back into register under a circular shift, so the shuffle null would come
out as good as the data and the control would prove nothing.

During "REM" the tracked head is parked while the population is driven by a
second, internal heading. That is the situation the whole module exists for:
recovering a heading that no camera can see.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from spikeshpc.decoder import (
    DecoderData,
    circular_correlation,
    circular_difference,
    decode,
    decoding_metrics,
    fit_encoding_model,
    movement_variance,
    poisson_log_likelihood,
    prepare_decoder_data,
    ring_transition,
    run_decoder,
    shuffle_test,
    split_train_test,
    state_interval_mask,
)

RATE = 120.0  # camera frames per second
N_UNITS = 24
PEAK_HZ = 30.0
BASELINE_HZ = 0.5
KAPPA = 4.0
WALK_STEP_DEG = 3.0  # per camera frame
BIN_FRAMES = 6  # 50 ms at 120 fps


def block_mean_step_var(bin_frames, step_deg=WALK_STEP_DEG):
    """Variance of the step between consecutive block means of a random walk.

    Derived from bin_frames rather than hard-coded, so a test that does not
    itself choose bin_s cannot be broken by someone changing the default.
    """
    k = bin_frames
    return k * step_deg**2 * (2 * k**2 + 1) / (3 * k**2)


BLOCK_MEAN_STEP_VAR = block_mean_step_var(BIN_FRAMES)


class FakeSorting:
    def __init__(self, spike_times_by_unit):
        self.unit_ids = np.array(sorted(spike_times_by_unit))
        self._spikes = {k: np.sort(v) for k, v in spike_times_by_unit.items()}

    def get_unit_spike_train(self, unit_id, return_times=False):
        return self._spikes[unit_id]


class FakeAnalyzer:
    def __init__(self, sorting):
        self.sorting = sorting


def von_mises_rates(heading_deg, preferred_deg):
    """(n_frames, n_units) firing rate in Hz for each heading."""
    offset = np.deg2rad(heading_deg[:, None] - preferred_deg[None, :])
    return BASELINE_HZ + PEAK_HZ * np.exp(KAPPA * (np.cos(offset) - 1.0))


def poisson_spike_times(rates_hz, frame_times, rng):
    """Spike times drawn per frame from `rates_hz`, jittered inside the frame."""
    dt = np.diff(frame_times)
    counts = rng.poisson(rates_hz[:-1] * dt[:, None])
    spikes = {}
    for unit in range(counts.shape[1]):
        frame_idx = np.repeat(np.arange(len(dt)), counts[:, unit])
        spikes[unit] = frame_times[frame_idx] + rng.uniform(
            0, dt[frame_idx], frame_idx.size
        )
    return spikes


def random_walk_heading(n, rng, step_deg=WALK_STEP_DEG):
    return np.cumsum(rng.normal(0, step_deg, n)) % 360.0


@pytest.fixture(scope="module")
def session():
    """600 s wake, 120 s NREM, 200 s REM, in that order.

    Wake: the population follows the tracked heading. NREM: the head is parked
    and the units fire slowly and without structure. REM: the head is still
    parked, but the population follows an *internal* heading of its own.
    """
    rng = np.random.default_rng(0)
    n_wake, n_nrem, n_rem = int(600 * RATE), int(120 * RATE), int(200 * RATE)
    n_total = n_wake + n_nrem + n_rem
    frame_times = np.arange(n_total) / RATE

    preferred = np.linspace(0, 360, N_UNITS, endpoint=False)

    wake_heading = random_walk_heading(n_wake, rng)
    internal_rem = random_walk_heading(n_rem, rng)
    # the tracked head does not move once the animal is asleep
    tracked = np.r_[wake_heading, np.full(n_nrem + n_rem, 270.0)]

    rates = np.vstack(
        [
            von_mises_rates(wake_heading, preferred),
            np.full((n_nrem, N_UNITS), 1.0),  # unstructured slow firing
            von_mises_rates(internal_rem, preferred),
        ]
    )
    sorting = FakeSorting(poisson_spike_times(rates, frame_times, rng))

    wake_stop = float(frame_times[n_wake])
    nrem_stop = float(frame_times[n_wake + n_nrem])
    intervals = {
        "WAKE": [[0.0, wake_stop]],
        "NREM": [[wake_stop, nrem_stop]],
        "REM": [[nrem_stop, float(frame_times[-1])]],
    }
    return {
        "sorting": sorting,
        "analyzer": FakeAnalyzer(sorting),
        "unit_ids": sorting.unit_ids,
        "heading_deg": tracked,
        "frame_times": frame_times,
        "intervals": intervals,
        "preferred": preferred,
        "internal_rem": internal_rem,
        "n_wake": n_wake,
        "n_rem": n_rem,
    }


@pytest.fixture(scope="module")
def data(session):
    return prepare_decoder_data(
        session["sorting"],
        session["unit_ids"],
        session["heading_deg"],
        session["frame_times"],
        bin_s=0.05,
    )


@pytest.fixture(scope="module")
def split(session, data):
    wake = state_interval_mask(data, session["intervals"], "WAKE")
    return split_train_test(data, wake, test_fraction=0.3, block_s=30.0, seed=1)


@pytest.fixture(scope="module")
def model(data, split):
    return fit_encoding_model(data, split[0], n_angle_bins=180)


# ── angles ───────────────────────────────────────────────────────────────
def test_circular_difference_takes_the_short_way_round():
    assert circular_difference(10.0, 350.0) == pytest.approx(20.0)
    assert circular_difference(350.0, 10.0) == pytest.approx(-20.0)
    assert abs(circular_difference(0.0, 180.0)) == pytest.approx(180.0)


def test_circular_correlation_rates_a_noisy_decode_highly():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 360, 2000)
    noisy = (truth + rng.normal(0, 5, 2000)) % 360.0
    assert circular_correlation(truth, noisy) > 0.95


def test_a_constant_offset_does_not_count_against_the_decode():
    """A miscalibrated head frame is a rotation, not a decoding failure.

    Pearson's r on the raw angles is what this exists to avoid: a quarter of
    the points cross the 0/360 seam under a 90 degree offset, and it reports a
    perfectly informative decode as slightly anti-correlated.
    """
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 360, 4000)
    offset = (truth + 90.0) % 360.0

    assert circular_correlation(truth, offset) == pytest.approx(1.0)
    assert np.corrcoef(truth, offset)[0, 1] < 0.0


def test_circular_correlation_of_unrelated_angles_is_near_zero():
    rng = np.random.default_rng(1)
    a, b = rng.uniform(0, 360, 5000), rng.uniform(0, 360, 5000)
    assert abs(circular_correlation(a, b)) < 0.1


# ── binning ──────────────────────────────────────────────────────────────
def test_bins_are_whole_camera_frames_on_measured_timestamps(session, data):
    assert data.bin_frames == 6  # 50 ms at 120 fps
    assert data.bin_s == pytest.approx(0.05, rel=1e-6)
    assert np.isin(data.edges, session["frame_times"]).all()
    assert data.counts.shape == (data.n_bins, N_UNITS)
    assert len(data.time_s) == len(data.duration_s) == data.n_bins


def test_one_frame_per_bin_keeps_the_tuning_curve_convention(session):
    """At bin_frames=1 the bin heading is the heading at the interval start."""
    fine = prepare_decoder_data(
        session["sorting"],
        session["unit_ids"],
        session["heading_deg"],
        session["frame_times"],
        bin_s=1 / RATE,
    )
    assert fine.bin_frames == 1
    np.testing.assert_allclose(
        fine.heading_deg, session["heading_deg"][: fine.n_bins], atol=1e-9
    )


def test_every_spike_is_counted_once(session, data):
    within = 0
    for unit in session["unit_ids"]:
        spikes = session["sorting"].get_unit_spike_train(unit)
        within += np.count_nonzero(
            (spikes >= data.edges[0]) & (spikes < data.edges[-1])
        )
    assert data.counts.sum() == within


@pytest.mark.parametrize(
    "camera_hz,bin_s,expected",
    [
        (120.0, 1 / 60, 2),        # exactly two frames
        (119.99, 1 / 60, 2),       # a slow camera must not halve the bin
        (120.008, 1 / 60, 2),      # or a fast one lengthen it
        (59.9989, 2 / 60, 2),      # this rig's other camera
        (120.0, 0.05, 6),
        (120.0, 1 / 120, 1),
        (120.0, 1 / 240, 1),       # asking for less than a frame still gets one
        (120.0, 0.0125, 1),        # 1.5 frames: a real request, rounded down
    ],
)
def test_a_bin_is_the_number_of_frames_you_meant(camera_hz, bin_s, expected):
    """Truncating the ratio halves bins exactly where it is most often used.

    A whole number of frames is the natural thing to ask for, and it is
    precisely where a camera off its nominal rate flips the truncation.
    """
    from spikeshpc.decoder import _frames_per_bin

    assert _frames_per_bin(bin_s, 1.0 / camera_hz) == expected


def test_a_slightly_slow_camera_does_not_halve_the_bin(session):
    """End to end, on frame times that are not on an exact grid."""
    frame_times = np.arange(len(session["frame_times"])) / 119.99
    data = prepare_decoder_data(
        session["sorting"], session["unit_ids"], session["heading_deg"],
        frame_times, bin_s=1 / 60,
    )
    assert data.bin_frames == 2
    assert data.bin_s == pytest.approx(2 / 119.99, rel=1e-6)


def test_mismatched_heading_and_frame_times_is_rejected(session):
    with pytest.raises(ValueError, match="same length"):
        prepare_decoder_data(
            session["sorting"],
            session["unit_ids"],
            session["heading_deg"][:-5],
            session["frame_times"],
        )


def test_no_units_is_rejected(session):
    with pytest.raises(ValueError, match="no units"):
        prepare_decoder_data(
            session["sorting"], [], session["heading_deg"], session["frame_times"]
        )


def test_an_analyzer_works_as_well_as_a_sorting(session, data):
    from_analyzer = prepare_decoder_data(
        session["analyzer"],
        session["unit_ids"],
        session["heading_deg"],
        session["frame_times"],
        bin_s=0.05,
    )
    np.testing.assert_array_equal(from_analyzer.counts, data.counts)


def test_state_mask_keeps_only_bins_wholly_inside_the_state(session, data):
    wake = state_interval_mask(data, session["intervals"], "WAKE")
    rem = state_interval_mask(data, session["intervals"], "REM")

    assert not (wake & rem).any()
    assert data.duration_s[wake].sum() == pytest.approx(600.0, abs=0.2)
    assert data.duration_s[rem].sum() == pytest.approx(200.0, abs=0.2)
    # nothing from the NREM block sneaks in
    assert data.time_s[wake].max() < session["intervals"]["NREM"][0][0]


# ── train / test split ───────────────────────────────────────────────────
def test_the_split_is_disjoint_and_stays_inside_the_mask(data, split):
    train, test = split
    assert not (train & test).any()
    assert data.duration_s[test].sum() / data.duration_s[train | test].sum() == (
        pytest.approx(0.3, abs=0.05)
    )


def test_contiguous_mode_holds_out_the_end(data, session):
    wake = state_interval_mask(data, session["intervals"], "WAKE")
    train, test = split_train_test(data, wake, 0.25, mode="contiguous")
    assert data.time_s[train].max() <= data.time_s[test].min()
    assert data.duration_s[test].sum() == pytest.approx(150.0, abs=1.0)


def test_blocks_mode_samples_the_whole_session(data, split):
    """The point of blocks: the test set is not all from one end."""
    train, test = split
    assert data.time_s[test].min() < 120.0
    assert data.time_s[test].max() > 480.0


def test_a_bigger_test_fraction_adds_blocks_rather_than_reshuffling(data, session):
    """Why two split sizes can plot the same window.

    The block order depends on the seed alone, and the test set is a prefix of
    it, so the sets are nested. A larger fraction therefore often keeps the
    same earliest block -- and plot_decoded starts at the first decoded bin.
    """
    wake = state_interval_mask(data, session["intervals"], "WAKE")
    small = np.flatnonzero(split_train_test(data, wake, 0.2, block_s=30.0, seed=4)[1])
    large = np.flatnonzero(split_train_test(data, wake, 0.6, block_s=30.0, seed=4)[1])

    assert set(small) < set(large), "the smaller test set should be nested inside"
    assert data.time_s[large].min() <= data.time_s[small].min()


def test_the_same_window_still_holds_a_different_decode(data, split, session):
    """Identical-looking is not identical: the model behind it changed."""
    wake = state_interval_mask(data, session["intervals"], "WAKE")
    a_train, a_test = split_train_test(data, wake, 0.3, block_s=30.0, seed=7)
    b_train, b_test = split_train_test(data, wake, 0.6, block_s=30.0, seed=7)

    a = decode(data, fit_encoding_model(data, a_train), a_test)
    b = decode(data, fit_encoding_model(data, b_train), b_test)

    shared = np.intersect1d(a.bin_index, b.bin_index)
    assert shared.size > 100, "no overlap to compare"
    in_a = np.isin(a.bin_index, shared)
    in_b = np.isin(b.bin_index, shared)

    # same bins, so the ground truth is identical -- that is the green trace
    np.testing.assert_allclose(a.actual_deg[in_a], b.actual_deg[in_b])
    # but the encoding models differ, so the decode does too
    assert not np.array_equal(a.decoded_deg[in_a], b.decoded_deg[in_b])


def test_a_bad_test_fraction_is_rejected(data, session):
    wake = state_interval_mask(data, session["intervals"], "WAKE")
    with pytest.raises(ValueError, match="test_fraction must be in"):
        split_train_test(data, wake, test_fraction=1.5)


def test_a_wrongly_sized_mask_is_rejected(data, session):
    with pytest.raises(ValueError, match="entries but there are"):
        split_train_test(data, np.ones(data.n_bins + 3, dtype=bool))


def test_an_unknown_split_mode_is_rejected(data, session):
    wake = state_interval_mask(data, session["intervals"], "WAKE")
    with pytest.raises(ValueError, match="must be 'blocks' or 'contiguous'"):
        split_train_test(data, wake, mode="every-other-tuesday")


# ── encoding model ───────────────────────────────────────────────────────
def test_the_model_recovers_each_unit_s_preferred_direction(session, model):
    error = circular_difference(model.preferred_deg, session["preferred"])
    assert np.abs(error).max() < 10.0, error
    assert (model.mean_vector_length > 0.3).all()


def test_rates_are_floored_so_one_unit_cannot_veto_a_heading(data, model):
    assert model.rate_hz.min() >= model.min_rate_hz
    log_likelihood = poisson_log_likelihood(
        data.counts[:100], data.duration_s[:100], model.rate_hz
    )
    assert np.isfinite(log_likelihood).all()


def test_the_model_reports_where_the_animal_barely_looked(data, split):
    train, _ = split
    model = fit_encoding_model(data, train, n_angle_bins=180)
    assert model.occupancy_s.sum() == pytest.approx(
        data.duration_s[train].sum(), rel=1e-6
    )


def test_a_thin_angle_bin_is_warned_about(data, split):
    """Far more angle bins than the animal can have visited."""
    with pytest.warns(UserWarning, match="training occupancy"):
        fit_encoding_model(data, split[0], n_angle_bins=3600, min_occupancy_s=1.0)


def test_training_on_nothing_is_rejected(data):
    with pytest.raises(ValueError, match="keeps no bins"):
        fit_encoding_model(data, np.zeros(data.n_bins, dtype=bool))


# ── dynamics ─────────────────────────────────────────────────────────────
def test_the_transition_is_row_stochastic_and_symmetric():
    transition = ring_transition(180, movement_var_deg2=25.0)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0)
    np.testing.assert_allclose(transition, transition.T, atol=1e-12)


def test_the_transition_wraps_at_north():
    """Bin 0's neighbours include the last bin, or the ring is not a ring."""
    transition = ring_transition(36, movement_var_deg2=200.0)
    assert transition[0, -1] == pytest.approx(transition[0, 1])
    assert transition[0, -1] > transition[0, len(transition) // 2]


def test_the_transition_never_assigns_exactly_zero():
    """A Gaussian tail underflows to zero; a probability of zero is forever."""
    bare = gaussian_filter1d(np.eye(180), sigma=1.6, axis=1, mode="wrap")
    assert (bare == 0).any(), "the fixture is not exercising the underflow"

    transition = ring_transition(180, movement_var_deg2=4.0)
    assert (transition > 0).all()
    np.testing.assert_allclose(transition.sum(axis=1), 1.0)


def test_a_confident_decode_recovers_from_a_teleport():
    """The head jumps; a decoder locked out of the truth never comes back.

    The likelihood is made overwhelming on purpose, so the posterior commits
    hard before the jump. Without the uniform leak in the transition, the prior
    at the new heading is exactly zero and the decode stays at the old one for
    the rest of the run.
    """
    from spikeshpc.decoder import _forward_backward

    n_angle, before, after = 180, 20, 120
    distance = np.abs(circular_difference(np.arange(n_angle) * 2.0, 0.0))
    log_likelihood = np.vstack(
        [
            np.tile(-2.0 * np.roll(distance, before), (100, 1)),
            np.tile(-2.0 * np.roll(distance, after), (100, 1)),
        ]
    )
    transition = ring_transition(n_angle, movement_var_deg2=4.0)
    causal, _ = _forward_backward(log_likelihood, transition, acausal=False)

    assert causal[99].argmax() == before
    assert causal[-1].argmax() == after
    # and it gets there quickly rather than crawling round the ring
    caught_up = np.flatnonzero(causal[100:].argmax(axis=1) == after)
    assert caught_up.size and caught_up[0] < 20


def test_infinite_variance_is_a_uniform_transition():
    transition = ring_transition(20, movement_var_deg2=np.inf)
    np.testing.assert_allclose(transition, 1.0 / 20)


def test_too_tight_a_walk_for_the_angle_bins_warns(data):
    with pytest.warns(UserWarning, match="under-diffuse"):
        ring_transition(360, movement_var_deg2=0.01)


def test_the_default_movement_variance_scales_with_bin_width(data):
    assert movement_variance(data) == pytest.approx(2.0 * data.bin_s * 100.0)


def test_the_measured_movement_variance_matches_the_real_walk(data, split):
    """The heading walks at 3 deg per frame; a 6-frame bin diffuses less.

    Not 6 * 3^2 = 54: a bin's heading is the mean over its frames, and
    averaging a random walk within the bin smooths part of the step away. For
    block means of a random walk the variance of the step between consecutive
    blocks is k*sigma^2*(2k^2+1)/(3k^2), i.e. 36.5 here. That is the right
    number to want, because it is the diffusion of the quantity the decoder
    actually carries from bin to bin.
    """
    assert movement_variance(data, split[0]) == pytest.approx(
        BLOCK_MEAN_STEP_VAR, rel=0.15
    )


# ── decoding ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def decoded(data, model, split):
    return decode(data, model, split[1], label="test")


def test_the_decoder_recovers_held_out_wake_heading(decoded):
    assert decoded.metrics["median_abs_error_deg"] < 10.0, decoded.metrics
    assert decoded.metrics["circular_correlation"] > 0.95
    assert decoded.metrics["frac_within_deg"] > 0.9


def test_the_decode_covers_the_masked_bins_and_only_those(data, split, decoded):
    _, test = split
    assert set(decoded.bin_index) <= set(np.flatnonzero(test))
    assert decoded.n_decoded == pytest.approx(test.sum(), rel=0.02)


def test_the_posterior_is_a_distribution_over_headings(decoded, model):
    assert decoded.posterior.shape == (decoded.n_decoded, model.n_angle_bins)
    np.testing.assert_allclose(decoded.posterior.sum(axis=1), 1.0, atol=1e-4)
    assert (decoded.entropy_bits >= 0).all()
    assert decoded.entropy_bits.max() <= np.log2(model.n_angle_bins) + 1e-6


def test_confidence_tracks_accuracy(decoded):
    """Bins the decoder was sure about should be the ones it got right."""
    sure = decoded.posterior_max > np.median(decoded.posterior_max)
    assert np.abs(decoded.error_deg[sure]).mean() < np.abs(
        decoded.error_deg[~sure]
    ).mean()


def test_the_acausal_posterior_beats_the_causal_one(data, model, split):
    causal = decode(data, model, split[1], acausal=False)
    acausal = decode(data, model, split[1], acausal=True)
    assert (
        acausal.metrics["median_abs_error_deg"]
        <= causal.metrics["median_abs_error_deg"]
    )


def test_dropping_the_dynamics_prior_still_decodes_but_worse(data, model, split):
    """Without a random walk each bin is decoded from its own spikes alone."""
    with_prior = decode(data, model, split[1])
    without = decode(data, model, split[1], movement_var_deg2=np.inf)
    assert without.metrics["median_abs_error_deg"] < 45.0
    assert (
        without.metrics["median_abs_error_deg"]
        > with_prior.metrics["median_abs_error_deg"]
    )


def test_belief_is_not_carried_across_a_gap(data, model, split):
    """A run starts from a uniform prior, so its first bin is spikes only."""
    _, test = split
    result = decode(data, model, test, acausal=False)
    assert result.run_index.max() > 0, "the blocked split should give many runs"

    log_likelihood = poisson_log_likelihood(
        data.counts, data.duration_s, model.rate_hz
    )
    first = np.flatnonzero(np.r_[True, np.diff(result.run_index) != 0])
    from_spikes_alone = log_likelihood[result.bin_index[first]].argmax(axis=1)
    np.testing.assert_array_equal(
        result.decoded_deg[first], model.bin_centers_deg[from_spikes_alone]
    )


def test_a_stretched_bin_is_dropped_as_a_tracking_hole(data, model, split):
    _, test = split
    stretched = DecoderData(
        counts=data.counts,
        heading_deg=data.heading_deg,
        duration_s=data.duration_s.copy(),
        time_s=data.time_s,
        edges=data.edges,
        unit_ids=data.unit_ids,
        bin_frames=data.bin_frames,
    )
    victim = int(np.flatnonzero(test)[len(np.flatnonzero(test)) // 2])
    stretched.duration_s[victim] = 5.0

    result = decode(stretched, model, test, max_gap_s=1.0)
    assert victim not in set(result.bin_index)


def test_a_model_from_other_units_is_refused(session, model):
    other = prepare_decoder_data(
        FakeSorting({u: np.array([1.0, 2.0]) for u in range(3)}),
        [0, 1, 2],
        session["heading_deg"],
        session["frame_times"],
        bin_s=0.05,
    )
    with pytest.raises(ValueError, match="different units"):
        decode(other, model, np.ones(other.n_bins, dtype=bool))


def test_a_mask_with_no_usable_run_is_rejected(data, model):
    lonely = np.zeros(data.n_bins, dtype=bool)
    lonely[::100] = True  # every run is one bin long
    with pytest.raises(ValueError, match="min_run_s"):
        decode(data, model, lonely, min_run_s=5.0)


def test_metrics_of_a_perfect_decode():
    angles = np.linspace(0, 360, 500, endpoint=False)
    metrics = decoding_metrics(angles, angles)
    assert metrics["median_abs_error_deg"] == 0.0
    assert metrics["rmse_deg"] == 0.0
    assert metrics["frac_within_deg"] == 1.0
    assert metrics["circular_correlation"] == pytest.approx(1.0)


# ── the control ──────────────────────────────────────────────────────────
def test_the_shifted_shuffle_is_far_worse_than_the_decode(data, model, split):
    result = shuffle_test(
        data, model, split[1], kind="shift", n_shuffles=20,
        min_shift_s=30.0, seed=0,
    )
    assert result.better == "lower"
    assert result.p_value == pytest.approx(1 / 21)  # the floor: nothing beat it
    assert result.observed < result.null.min() / 3
    assert result.z_score < -3


def test_the_shuffle_null_is_what_chance_looks_like(data, model, split):
    """Guessing at random on a circle averages 90 deg of absolute error."""
    result = shuffle_test(
        data, model, split[1], kind="shift", n_shuffles=20, min_shift_s=30.0
    )
    assert 45.0 < result.null.mean() < 110.0


def test_the_unit_shuffle_breaks_the_population_code(data, model, split):
    result = shuffle_test(
        data, model, split[1], kind="units",
        metric="mean_posterior_max", n_shuffles=20, seed=0,
    )
    assert result.better == "higher"
    assert result.observed > result.null.max()
    assert result.p_value == pytest.approx(1 / 21)


def test_an_unknown_shuffle_kind_is_rejected(data, model, split):
    with pytest.raises(ValueError, match="must be 'shift' or 'units'"):
        shuffle_test(data, model, split[1], kind="jumble", n_shuffles=2)


def test_an_unknown_metric_is_rejected(data, model, split):
    with pytest.raises(ValueError, match="is not one of"):
        shuffle_test(data, model, split[1], metric="vibes", n_shuffles=2)


# ── REM ──────────────────────────────────────────────────────────────────
def test_rem_decoding_recovers_the_internal_heading(session, data, model):
    """The whole point: a heading the camera cannot see.

    The tracked head is parked at 270 deg throughout, so the decode is judged
    against the internal heading the REM spikes were generated from.
    """
    rem_mask = state_interval_mask(data, session["intervals"], "REM")
    result = decode(data, model, rem_mask, with_metrics=False, label="REM")

    # the internal heading, on the decoder's own bins
    n_wake_bins = session["n_wake"] // data.bin_frames
    rem_start = session["n_wake"] + int(120 * RATE)
    frame_of_bin = result.bin_index * data.bin_frames - rem_start
    truth = session["internal_rem"][np.clip(frame_of_bin, 0, session["n_rem"] - 1)]

    assert result.n_decoded > 0.9 * (200.0 / data.bin_s)
    assert np.median(np.abs(circular_difference(result.decoded_deg, truth))) < 15.0
    assert circular_correlation(result.decoded_deg, truth) > 0.9

    # and it is nothing like the parked camera heading
    assert np.median(np.abs(circular_difference(result.decoded_deg, 270.0))) > 45.0
    assert n_wake_bins > 0


def test_rem_metrics_are_left_empty_because_there_is_no_ground_truth(
    session, data, model
):
    rem_mask = state_interval_mask(data, session["intervals"], "REM")
    result = decode(data, model, rem_mask, with_metrics=False)
    assert "median_abs_error_deg" not in result.metrics
    # confidence is still reported: it needs no heading to compare against
    assert result.metrics["mean_posterior_max"] > 0


# ── end to end ───────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def full_run(session):
    return run_decoder(
        session["analyzer"],
        session["unit_ids"],
        session["heading_deg"],
        session["frame_times"],
        session["intervals"],
        test_fraction=0.3,
        block_s=30.0,
        n_shuffles=20,
        seed=1,
        verbose=False,
    )


def test_run_decoder_trains_tests_shuffles_and_decodes_rem(full_run):
    assert full_run.test.metrics["median_abs_error_deg"] < 10.0
    assert full_run.test_shuffle.p_value == pytest.approx(1 / 21)
    assert full_run.rem is not None
    assert full_run.rem_shuffle.p_value == pytest.approx(1 / 21)
    assert not (full_run.train_mask & full_run.test_mask).any()


def test_the_summary_says_what_happened(full_run):
    text = full_run.summary()
    assert "wake test" in text
    assert "REM" in text
    assert "median_abs_error_deg" in text


def test_estimating_the_movement_variance_works_end_to_end(session):
    run = run_decoder(
        session["sorting"],
        session["unit_ids"],
        session["heading_deg"],
        session["frame_times"],
        session["intervals"],
        movement_var_deg2="estimate",
        n_shuffles=0,
        decode_rem=False,
        verbose=False,
    )
    assert run.movement_var_deg2 == pytest.approx(
        block_mean_step_var(run.data.bin_frames), rel=0.15
    )
    assert run.test.metrics["median_abs_error_deg"] < 10.0


def test_a_session_with_no_rem_warns_rather_than_failing(session):
    wake_only = {"WAKE": session["intervals"]["WAKE"]}
    with pytest.warns(UserWarning, match="no decoder bin falls inside a REM"):
        run = run_decoder(
            session["sorting"],
            session["unit_ids"],
            session["heading_deg"],
            session["frame_times"],
            wake_only,
            n_shuffles=0,
            verbose=False,
        )
    assert run.rem is None


def test_a_session_with_no_wake_is_refused(session):
    with pytest.raises(ValueError, match="no decoder bin falls inside a WAKE"):
        run_decoder(
            session["sorting"],
            session["unit_ids"],
            session["heading_deg"],
            session["frame_times"],
            {"NREM": session["intervals"]["NREM"]},
            n_shuffles=0,
            verbose=False,
        )


# ── plots ────────────────────────────────────────────────────────────────
def test_the_plots_draw(full_run):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spikeshpc.decoder import (
        plot_decoded,
        plot_encoding_model,
        plot_error,
        plot_shuffle,
    )

    axis = plot_decoded(full_run.test, window_s=30.0)
    title = axis.get_title()
    # the figure has to say which run it is, or two splits look the same
    assert "decoded bins" in title and "median" in title
    plt.close(axis.figure)

    for make in (
        lambda: plot_encoding_model(full_run.model),
        lambda: plot_decoded(full_run.test, window_s=30.0),
        lambda: plot_decoded(full_run.rem, window_s=30.0),
        lambda: plot_error(full_run.test),
        lambda: plot_shuffle(full_run.test_shuffle),
    ):
        made = make()
        axis = made[0] if isinstance(made, np.ndarray) else made
        assert axis.figure is not None
        plt.close(axis.figure)
