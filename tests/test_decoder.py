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

import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from spikeshpc.decoder import (
    WAKE_SHUFFLE_METRICS,
    DecoderData,
    apply_decoder,
    bins_in_interval_mask,
    circular_correlation,
    circular_difference,
    decode,
    decoding_metrics,
    fit_encoding_model,
    metrics_by_group,
    movement_variance,
    poisson_log_likelihood,
    prepare_decoder_data,
    reference_decode,
    restrict_data,
    restrict_model,
    ring_transition,
    run_decoder,
    shuffle_test,
    split_train_test,
    state_interval_mask,
)
from spikeshpc.states import frames_in_states, interval_mask_to_spans

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


def test_a_bin_needs_every_one_of_its_frame_intervals(session, data):
    """A frame excluded from the middle of a bin sinks that bin, and only it."""
    keep = np.ones(len(session["frame_times"]) - 1, dtype=bool)
    keep[3 * data.bin_frames + 2] = False  # inside the fourth bin

    bins = bins_in_interval_mask(data, keep)
    assert list(np.flatnonzero(~bins)) == [3]
    # testing only a bin's edges, as state masks do, would have kept it
    spans = {"KEEP": interval_mask_to_spans(session["frame_times"], keep)}
    assert state_interval_mask(data, spans, "KEEP")[3]


def test_a_bin_mask_wants_intervals_not_bins(data):
    with pytest.raises(ValueError, match="masks intervals, not frames or bins"):
        bins_in_interval_mask(data, np.ones(data.n_bins, dtype=bool))


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


@pytest.mark.parametrize(
    "n_angle, variance, fast",
    [
        (360, 4.98, True),  # the band notebook 3 runs with
        (180, 25.0, True),
        (90, 1.75, True),  # under half a bin: one bin each way
        (36, 200.0, False),  # too wide to gain: kept as a matrix
        (20, np.inf, True),  # uniform
        (24, 0.0, True),  # stay put, plus the leak
    ],
)
def test_the_ring_step_is_the_matrix_product(n_angle, variance, fast):
    from spikeshpc.decoder import _RingStep

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transition = ring_transition(n_angle, movement_var_deg2=variance)
    step = _RingStep(transition)
    assert step.fast == fast
    rows = np.random.default_rng(0).random((4, n_angle))
    np.testing.assert_allclose(step.forward(rows), rows @ transition, rtol=1e-12)
    np.testing.assert_allclose(step.backward(rows), (transition @ rows.T).T, rtol=1e-12)


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


def test_the_likelihood_run_by_run_is_the_likelihood_up_front(data, model, split, decoded):
    """Computing it per run saves memory and must change nothing else."""
    from spikeshpc.decoder import _forward_backward, _runs

    whole = poisson_log_likelihood(data.counts, data.duration_s, model.rate_hz)
    transition = ring_transition(model.n_angle_bins, movement_variance(data))
    expected = np.concatenate([
        _forward_backward(whole[a:b], transition)[1]
        for a, b in _runs(split[1], data.duration_s, max_gap_s=1.0, min_run_s=1.0)
    ])
    np.testing.assert_allclose(decoded.posterior, expected, rtol=1e-5, atol=1e-7)


def test_metrics_of_a_perfect_decode():
    angles = np.linspace(0, 360, 500, endpoint=False)
    metrics = decoding_metrics(angles, angles)
    assert metrics["median_abs_error_deg"] == 0.0
    assert metrics["rmse_deg"] == 0.0
    assert metrics["frac_within_deg"] == 1.0
    assert metrics["circular_correlation"] == pytest.approx(1.0)
    assert metrics["offset_deg"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["median_abs_error_corrected_deg"] == pytest.approx(0.0, abs=1e-9)


def test_a_constant_offset_is_measured_and_can_be_removed():
    """A population that turned as a whole, read with the old curves."""
    rng = np.random.default_rng(2)
    truth = rng.uniform(0, 360, 3000)
    decoded = (truth + 40.0 + rng.normal(0, 3, 3000)) % 360.0
    metrics = decoding_metrics(decoded, truth, tolerance_deg=10.0)
    assert metrics["offset_deg"] == pytest.approx(40.0, abs=1.0)
    assert metrics["median_abs_error_deg"] == pytest.approx(40.0, abs=1.0)
    assert metrics["median_abs_error_corrected_deg"] < 3.0
    assert metrics["frac_within_deg"] == 0.0
    assert metrics["frac_within_corrected_deg"] > 0.99


def test_each_group_is_scored_on_its_own_bins(decoded):
    labels = np.where(np.arange(decoded.n_decoded) < decoded.n_decoded // 3, "a", "b")
    index = np.arange(decoded.n_decoded, dtype=float)
    table = metrics_by_group(decoded, labels, extra={"index": index})

    assert list(table) == ["a", "b"]
    assert sum(t["n_bins"] for t in table.values()) == decoded.n_decoded
    for name, metrics in table.items():
        keep = labels == name
        expected = decoding_metrics(
            decoded.decoded_deg[keep], decoded.actual_deg[keep],
            decoded.metrics["tolerance_deg"],
        )
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value), key
        assert metrics["time_s"] == pytest.approx(decoded.duration_s[keep].sum())
        assert metrics["mean_posterior_max"] == pytest.approx(
            decoded.posterior_max[keep].mean()
        )
        assert metrics["population_rate_hz"] == pytest.approx(
            decoded.n_spikes[keep].sum() / decoded.duration_s[keep].sum()
        )
        assert metrics["index"] == pytest.approx(index[keep].mean())
        assert (metrics["q25_abs_error_deg"] <= metrics["median_abs_error_deg"]
                <= metrics["q75_abs_error_deg"])


def test_groups_come_back_in_the_order_asked_and_empty_ones_are_dropped(decoded):
    labels = np.full(decoded.n_decoded, "late", dtype=object)
    labels[:50] = "early"  # as a fixed-width <U4 array this would be "earl"
    table = metrics_by_group(decoded, labels, order=["late", "never", "early"])
    assert list(table) == ["late", "early"]


def test_labels_must_cover_every_decoded_bin(decoded):
    with pytest.raises(ValueError, match="labels has"):
        metrics_by_group(decoded, np.zeros(3))
    with pytest.raises(ValueError, match="extra"):
        metrics_by_group(
            decoded, np.zeros(decoded.n_decoded), extra={"speed": np.zeros(3)}
        )


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
    with pytest.raises(ValueError, match="is not one of"):
        shuffle_test(data, model, split[1], metric=["median_abs_error_deg", "vibes"],
                     n_shuffles=2)


def test_several_metrics_are_judged_on_one_set_of_shuffles(data, model, split):
    both = shuffle_test(
        data, model, split[1], metric=WAKE_SHUFFLE_METRICS, n_shuffles=5, seed=2
    )
    alone = shuffle_test(
        data, model, split[1], metric="median_abs_error_deg", n_shuffles=5, seed=2
    )
    assert list(both) == list(WAKE_SHUFFLE_METRICS)
    np.testing.assert_allclose(both["median_abs_error_deg"].null, alone.null)
    assert all(test.better == "lower" for test in both.values())


def one_at_a_time_null(data, model, mask, kind, metrics, n_shuffles, seed,
                       min_shift_s=30.0, **decode_kwargs):
    """The null as shuffle_test used to make it: a decode() per surrogate."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_shuffles):
        if kind == "shift":
            min_shift = max(1, round(min_shift_s / data.bin_s))
            shift = int(rng.integers(min_shift, data.n_bins - min_shift))
            surrogate = replace(data, counts=np.roll(data.counts, shift, axis=0))
            shuffled = model
        else:
            surrogate = data
            order = rng.permutation(len(model.unit_ids))
            shuffled = replace(model, rate_hz=model.rate_hz[order])
        run = decode(surrogate, shuffled, mask, keep_posterior=False, **decode_kwargs)
        null.append([run.metrics[name] for name in metrics])
    return np.array(null).T


ALL_WAKE_METRICS = [*WAKE_SHUFFLE_METRICS, "mean_posterior_max", "mean_entropy_bits"]


@pytest.mark.parametrize("acausal", [True, False])
@pytest.mark.parametrize("kind", ["shift", "units"])
def test_batched_shuffles_decode_each_surrogate_as_decode_would(
    data, model, split, monkeypatch, kind, acausal
):
    """Batches split mid-way and likelihood chunks cut across runs change nothing."""
    import spikeshpc.decoder as decoder

    monkeypatch.setattr(decoder, "_CHUNK_BINS", 37)
    tests = shuffle_test(
        data, model, split[1], kind=kind, metric=ALL_WAKE_METRICS, n_shuffles=5,
        seed=3, acausal=acausal, tolerance_deg=10.0, n_jobs=1,
        max_batch_bytes=2 * 8 * model.n_angle_bins * 400,  # 2 shuffles over 400 bins
    )
    expected = one_at_a_time_null(
        data, model, split[1], kind, ALL_WAKE_METRICS, 5, seed=3,
        acausal=acausal, tolerance_deg=10.0,
    )
    for k, name in enumerate(ALL_WAKE_METRICS):
        np.testing.assert_allclose(tests[name].null, expected[k], rtol=1e-12, err_msg=name)


def test_rem_unit_shuffles_match_one_at_a_time_decodes(session, data, model):
    rem_mask = state_interval_mask(data, session["intervals"], "REM")
    test = shuffle_test(
        data, model, rem_mask, kind="units", metric="mean_posterior_max",
        n_shuffles=6, seed=4, with_metrics=False, n_jobs=1,
    )
    expected = one_at_a_time_null(
        data, model, rem_mask, "units", ["mean_posterior_max"], 6, seed=4,
        with_metrics=False,
    )
    np.testing.assert_allclose(test.null, expected[0], rtol=1e-12)


def test_the_null_does_not_depend_on_how_many_workers_share_it(data, model, split):
    kwargs = dict(metric=WAKE_SHUFFLE_METRICS, n_shuffles=6, seed=5, tolerance_deg=10.0)
    here = shuffle_test(data, model, split[1], n_jobs=1, **kwargs)
    pooled = shuffle_test(data, model, split[1], n_jobs=3, **kwargs)
    for name in WAKE_SHUFFLE_METRICS:
        np.testing.assert_allclose(pooled[name].null, here[name].null, rtol=1e-12)
        assert pooled[name].p_value == here[name].p_value


def test_no_shuffles_still_scores_the_observed_decode(data, model, split):
    test = shuffle_test(data, model, split[1], n_shuffles=0)
    assert test.null.shape == (0,)
    assert np.isfinite(test.observed)


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


@pytest.fixture(scope="module")
def nrem_run(session):
    """`full_run` with NREM decoded too."""
    return run_decoder(
        session["analyzer"],
        session["unit_ids"],
        session["heading_deg"],
        session["frame_times"],
        session["intervals"],
        test_fraction=0.3,
        block_s=30.0,
        n_shuffles=10,
        decode_nrem=True,
        seed=1,
        verbose=False,
    )


def test_nrem_is_decoded_only_when_asked(session, full_run, nrem_run):
    assert full_run.nrem is None and full_run.nrem_mask is None
    start, stop = session["intervals"]["NREM"][0]
    assert nrem_run.nrem.duration_s.sum() == pytest.approx(stop - start, rel=0.02)
    # no heading to score against, as in REM; confidence is still reported
    assert "median_abs_error_deg" not in nrem_run.nrem.metrics
    assert nrem_run.nrem.metrics["mean_posterior_max"] > 0
    assert nrem_run.nrem_shuffle.kind == "units"
    assert len(nrem_run.nrem_shuffle.null) == 10
    # and the REM decode is the one it would have been without it
    np.testing.assert_array_equal(nrem_run.rem.decoded_deg, full_run.rem.decoded_deg)


def test_unstructured_nrem_is_less_confident_than_rem(nrem_run):
    """The synthetic NREM fires flat and slow: there is no heading in it to be sure of."""
    nrem = nrem_run.nrem.metrics["mean_posterior_max"]
    assert nrem < nrem_run.rem.metrics["mean_posterior_max"]
    assert nrem < nrem_run.test.metrics["mean_posterior_max"]
    # scrambling which curve is whose costs the REM population its coherence,
    # and the unstructured NREM nothing like as much
    assert nrem_run.rem_shuffle.z_score > nrem_run.nrem_shuffle.z_score


def test_a_session_with_no_nrem_warns_rather_than_failing(session):
    no_nrem = {k: v for k, v in session["intervals"].items() if k != "NREM"}
    with pytest.warns(UserWarning, match="no decoder bin falls inside a NREM"):
        run = run_decoder(
            session["sorting"],
            session["unit_ids"],
            session["heading_deg"],
            session["frame_times"],
            no_nrem,
            n_shuffles=0,
            decode_rem=False,
            decode_nrem=True,
            verbose=False,
        )
    assert run.nrem is None and run.nrem_mask is None


def test_the_summary_reports_nrem_when_it_was_decoded(full_run, nrem_run):
    assert "NREM" in nrem_run.summary()
    assert "NREM" not in full_run.summary()


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


def test_within_keeps_training_and_testing_inside_the_spans(session, full_run):
    """Movement bouts, say: both sides of the split come from them alone."""
    wake_stop = session["intervals"]["WAKE"][0][1]
    spans = [[0.0, 250.0], [300.0, 600.0]]
    run = run_decoder(
        session["sorting"],
        session["unit_ids"],
        session["heading_deg"],
        session["frame_times"],
        session["intervals"],
        n_shuffles=0,
        verbose=False,
        within=np.array(spans),  # an array works as well as a list
    )
    used = run.train_mask | run.test_mask
    inside = ((run.data.edges[:-1] >= 0.0) & (run.data.edges[1:] <= 250.0)) | (
        (run.data.edges[:-1] >= 300.0) & (run.data.edges[1:] <= 600.0)
    )
    assert used.any() and not (used & ~inside).any()
    assert run.data.duration_s[used].sum() == pytest.approx(550.0, rel=0.01)
    assert run.data.time_s[used].max() < wake_stop
    assert run.test.metrics["median_abs_error_deg"] < 10.0
    # REM is not wake, so `within` does not reach it
    assert run.rem.n_decoded == full_run.rem.n_decoded


def test_within_spans_that_miss_wake_are_refused(session):
    with pytest.raises(ValueError, match="inside the `within` spans"):
        run_decoder(
            session["sorting"],
            session["unit_ids"],
            session["heading_deg"],
            session["frame_times"],
            session["intervals"],
            n_shuffles=0,
            decode_rem=False,
            verbose=False,
            within=session["intervals"]["NREM"],
        )


def test_interval_mask_keeps_training_and_testing_inside_it(session, full_run):
    """A tuning's interval mask, handed on: both sides of the split come from it."""
    frame_times = session["frame_times"]
    _, keep = frames_in_states(
        frame_times, {"KEEP": [[0.0, 250.0], [300.0, 600.0]]}, "KEEP"
    )
    run = run_decoder(
        session["sorting"],
        session["unit_ids"],
        session["heading_deg"],
        frame_times,
        session["intervals"],
        n_shuffles=0,
        verbose=False,
        interval_mask=keep,
    )
    used = run.train_mask | run.test_mask
    inside = ((run.data.edges[:-1] >= 0.0) & (run.data.edges[1:] <= 250.0)) | (
        (run.data.edges[:-1] >= 300.0) & (run.data.edges[1:] <= 600.0)
    )
    assert used.any() and not (used & ~inside).any()
    assert run.data.duration_s[used].sum() == pytest.approx(550.0, rel=0.01)
    assert run.test.metrics["median_abs_error_deg"] < 10.0
    # REM is not wake, so the mask does not reach it
    assert run.rem.n_decoded == full_run.rem.n_decoded


def test_an_interval_mask_must_be_over_intervals_and_meet_wake(session):
    frame_times = session["frame_times"]
    inputs = (
        session["sorting"],
        session["unit_ids"],
        session["heading_deg"],
        frame_times,
        session["intervals"],
    )
    quiet = dict(n_shuffles=0, decode_rem=False, verbose=False)

    with pytest.raises(ValueError, match="masks intervals, not"):
        run_decoder(*inputs, interval_mask=np.ones(len(frame_times), dtype=bool), **quiet)
    _, nrem = frames_in_states(frame_times, session["intervals"], "NREM")
    with pytest.raises(ValueError, match="wholly inside `interval_mask`"):
        run_decoder(*inputs, interval_mask=nrem, **quiet)


# ── another recording ────────────────────────────────────────────────────
def renamed(sorting, offset=100):
    """The same spikes under other unit ids, as a later recording's sort numbers them."""
    return FakeSorting({u + offset: sorting.get_unit_spike_train(u) for u in sorting.unit_ids})


@pytest.fixture(scope="module")
def turned(session):
    """The same animal later on: every preferred direction turned +60 deg, units renumbered."""
    rng = np.random.default_rng(5)
    rates = von_mises_rates(session["heading_deg"], session["preferred"] + 60.0)
    spikes = poisson_spike_times(rates, session["frame_times"], rng)
    return FakeSorting({u + 100: t for u, t in spikes.items()})


def transfer_to(session, sorting, model, data, **kwargs):
    kwargs = {"bin_s": 0.05, "movement_var_deg2": movement_variance(data), "n_shuffles": 0,
              "verbose": False, **kwargs}
    return apply_decoder(
        model, sorting, {u: u + 100 for u in session["unit_ids"]}, session["heading_deg"],
        session["frame_times"], session["intervals"], **kwargs,
    )


def test_a_restricted_model_keeps_the_named_units_in_order(model):
    sub = restrict_model(model, [5, 2, 7])
    assert sub.unit_ids.tolist() == [5, 2, 7]
    np.testing.assert_array_equal(sub.rate_hz, model.rate_hz[[5, 2, 7]])
    with pytest.raises(ValueError, match="are not among"):
        restrict_model(model, [5, 99])


def test_restricting_the_model_is_fitting_those_units_alone(data, split, model):
    units = [3, 11, 17]
    alone = fit_encoding_model(restrict_data(data, units), split[0], n_angle_bins=180)
    np.testing.assert_allclose(restrict_model(model, units).rate_hz, alone.rate_hz)


def test_restricted_data_keeps_the_named_columns(data):
    sub = restrict_data(data, [4, 1])
    assert sub.unit_ids.tolist() == [4, 1]
    np.testing.assert_array_equal(sub.counts, data.counts[:, [4, 1]])
    assert sub.n_bins == data.n_bins


def test_decoding_partners_is_decoding_the_units_they_stand_for(session, data, model):
    """The same spikes under other ids decode exactly as the originals do."""
    transfer = transfer_to(session, renamed(session["sorting"]), model, data, decode_rem=False)
    wake = state_interval_mask(data, session["intervals"], "WAKE")
    direct = decode(data, model, wake, movement_var_deg2=movement_variance(data),
                    tolerance_deg=10.0)

    np.testing.assert_array_equal(transfer.wake.decoded_deg, direct.decoded_deg)
    assert transfer.unit_ids.tolist() == list(session["unit_ids"])
    assert transfer.partner_ids.tolist() == [u + 100 for u in session["unit_ids"]]
    assert transfer.rem is None


def test_a_turned_population_is_read_at_a_constant_offset(session, data, model, turned):
    """Every PD turned +60, so each unit now fires 60 deg past where the model expects.

    The decoder therefore reports the head 60 deg short of where it is: the
    offset is minus the rotation. Removed, the code is as good as ever.
    """
    transfer = transfer_to(session, turned, model, data, decode_rem=False)
    metrics = transfer.wake.metrics
    assert transfer.offset_deg == pytest.approx(-60.0, abs=5.0)
    assert metrics["median_abs_error_deg"] == pytest.approx(60.0, abs=8.0)
    assert metrics["median_abs_error_corrected_deg"] < 10.0
    assert metrics["circular_correlation"] > 0.9


def test_one_set_of_shuffles_judges_raw_and_corrected_error(session, data, model, turned):
    transfer = transfer_to(session, turned, model, data, n_shuffles=5, decode_rem=False)
    assert list(transfer.wake_shuffles) == list(WAKE_SHUFFLE_METRICS)
    for test in transfer.wake_shuffles.values():
        assert len(test.null) == 5
        assert test.p_value == pytest.approx(1 / 6)  # nothing shuffled beat it


def test_the_wake_of_another_recording_is_all_of_its_masked_wake(session, data, model):
    frame_times = session["frame_times"]
    _, keep = frames_in_states(frame_times, {"KEEP": [[0.0, 250.0], [300.0, 600.0]]}, "KEEP")
    transfer = transfer_to(session, renamed(session["sorting"]), model, data,
                           interval_mask=keep, decode_rem=False)
    assert transfer.wake.duration_s.sum() == pytest.approx(550.0, rel=0.02)
    with pytest.raises(ValueError, match="masks intervals, not"):
        transfer_to(session, renamed(session["sorting"]), model, data,
                    interval_mask=np.ones(len(frame_times), dtype=bool))


def test_a_transfer_needs_distinct_matched_units(session, model):
    inputs = (session["heading_deg"], session["frame_times"], session["intervals"])
    with pytest.raises(ValueError, match="empty"):
        apply_decoder(model, session["sorting"], {}, *inputs, verbose=False)
    with pytest.raises(ValueError, match="same partner"):
        apply_decoder(model, session["sorting"], {0: 5, 1: 5}, *inputs, verbose=False)


def test_a_reference_decodes_the_baseline_s_own_held_out_bins(full_run):
    units = list(full_run.model.unit_ids[:10])
    reference = reference_decode(full_run, units, n_shuffles=0, verbose=False)
    expected = decode(
        restrict_data(full_run.data, units), restrict_model(full_run.model, units),
        full_run.test_mask, movement_var_deg2=full_run.movement_var_deg2, tolerance_deg=10.0,
    )
    assert set(reference.wake.bin_index) <= set(np.flatnonzero(full_run.test_mask))
    np.testing.assert_array_equal(reference.wake.decoded_deg, expected.decoded_deg)
    assert reference.rem.n_decoded == full_run.rem.n_decoded


def test_a_transfer_decodes_nrem_when_asked(session, data, model, turned):
    transfer = transfer_to(session, turned, model, data, decode_rem=False, decode_nrem=True)
    start, stop = session["intervals"]["NREM"][0]
    assert transfer.rem is None
    assert transfer.nrem.duration_s.sum() == pytest.approx(stop - start, rel=0.02)
    assert transfer.nrem.label == "transfer NREM"
    row = transfer.as_row()
    assert row["nrem_s"] == pytest.approx(transfer.nrem.duration_s.sum())
    assert row["rem_s"] == 0.0
    assert "NREM" in transfer.summary()
    assert transfer_to(session, turned, model, data, decode_rem=False).nrem is None


def test_a_reference_decodes_nrem_if_the_run_did(full_run, nrem_run):
    units = list(nrem_run.model.unit_ids[:10])
    reference = reference_decode(nrem_run, units, n_shuffles=0, verbose=False)
    assert reference.nrem.n_decoded == nrem_run.nrem.n_decoded
    assert reference_decode(full_run, units, n_shuffles=0, verbose=False).nrem is None


def test_the_transfer_summary_gets_an_nrem_panel_when_needed(full_run, nrem_run):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spikeshpc.decoder import plot_transfer_summary

    units = list(nrem_run.model.unit_ids)
    with_nrem = reference_decode(nrem_run, units, n_shuffles=3, verbose=False)
    without = reference_decode(full_run, units, n_shuffles=0, verbose=False)

    axes = plot_transfer_summary({"a": with_nrem, "b": without}, references={"b": with_nrem})
    assert len(axes) == 4
    assert axes[3].get_ylabel() == "NREM posterior max"
    plt.close(axes[0].figure)

    axes = plot_transfer_summary({"b": without})
    assert len(axes) == 3
    plt.close(axes[0].figure)

    _, three = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="4 panels"):
        plot_transfer_summary({"a": with_nrem}, axes=three)
    plt.close("all")


def test_a_transfer_summarises_tabulates_and_draws(session, data, model, turned, full_run):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spikeshpc.decoder import plot_transfer_summary

    transfer = transfer_to(session, turned, model, data, n_shuffles=3)
    text = transfer.summary()
    assert "offset" in text and "REM" in text

    row = transfer.as_row()
    assert row["n_units"] == N_UNITS
    assert row["wake_offset_deg"] == pytest.approx(transfer.offset_deg)
    assert {"wake_error_p", "wake_error_corrected_p", "rem_p"} <= set(row)

    reference = reference_decode(full_run, list(full_run.model.unit_ids), n_shuffles=3,
                                 verbose=False)
    axes = plot_transfer_summary(
        {"baseline": reference, "turned": transfer}, references={"turned": reference}
    )
    assert len(axes) == 3
    assert [t.get_text() for t in axes[0].get_xticklabels()][1].startswith("turned")
    plt.close(axes[0].figure)


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


def test_grouped_metrics_draw_as_bars(full_run):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spikeshpc.decoder import plot_metrics_by_group

    test = full_run.test
    labels = np.where(np.arange(test.n_decoded) % 2 == 0, "even", "odd")
    table = metrics_by_group(test, labels)

    axes = plot_metrics_by_group(table)
    assert len(axes) == 3
    # chance on the error panel, and each group's time on its tick
    assert any(np.allclose(line.get_ydata(), 90.0) for line in axes[0].lines)
    ticks = [t.get_text() for t in axes[0].get_xticklabels()]
    assert ticks[0].startswith("even") and ticks[0].endswith("s")
    plt.close(axes[0].figure)

    both = plot_metrics_by_group(
        {"model a": table, "model b": table}, metrics=("median_abs_error_deg",)
    )
    assert len(both[0].patches) == 4  # two series x two groups
    plt.close(both[0].figure)
