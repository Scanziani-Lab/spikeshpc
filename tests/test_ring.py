"""The ring from correlations: pair metric, embedding, screen, decode, comparisons.

The synthetic population is 24 von Mises head-direction units around a heading
that random-walks quickly enough to decorrelate within the 5 s before the
shoulder starts, plus 16 untuned units firing steadily. Every unit's rate is
scaled by one slow common gain (a 30 s drift, like arousal), which is what the
baseline subtraction is there to remove. As in the decoder's tests, the
tracked head is parked during NREM and REM; in NREM everything fires flat,
and in REM the head-direction units follow an internal heading of their own.
"""

import warnings

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from spikeshpc.decoder import (
    DecoderData,
    circular_correlation,
    circular_difference,
    metrics_by_group,
    prepare_decoder_data,
    run_decoder,
    state_interval_mask,
)
from spikeshpc.optitrack.tuning import HDTuningStats
from spikeshpc.ring import (
    NO_PARTNERS,
    ON_RING,
    SILENT,
    _embed,
    align_angles,
    circular_r,
    compare_angles,
    compare_rings,
    decode_ring,
    fit_ring,
    pair_correlogram,
    pair_decodes,
    pairwise_correlations,
    ring_vs_tuning,
    run_ring,
)

RATE = 60.0  # camera frames per second, as the shutter-aligned frames are
N_HD = 24
N_FLAT = 16
WALK_STEP_DEG = 6.0  # per frame: the heading decorrelates within a few seconds
WAKE_S, NREM_S, REM_S = 600.0, 120.0, 200.0


class FakeSorting:
    def __init__(self, spike_times_by_unit):
        self.unit_ids = np.array(sorted(spike_times_by_unit))
        self._spikes = {k: np.sort(v) for k, v in spike_times_by_unit.items()}

    def get_unit_spike_train(self, unit_id, return_times=False):
        return self._spikes[unit_id]


class FakeAnalyzer:
    def __init__(self, sorting):
        self.sorting = sorting


def von_mises_rates(heading_deg, preferred_deg, peak_hz=30.0, kappa=4.0):
    offset = np.deg2rad(heading_deg[:, None] - preferred_deg[None, :])
    return 0.5 + peak_hz * np.exp(kappa * (np.cos(offset) - 1.0))


def poisson_spike_times(rates_hz, frame_times, rng):
    dt = np.diff(frame_times)
    counts = rng.poisson(rates_hz[:-1] * dt[:, None])
    spikes = {}
    for unit in range(counts.shape[1]):
        frame = np.repeat(np.arange(len(dt)), counts[:, unit])
        spikes[unit] = frame_times[frame] + rng.uniform(0, dt[frame], frame.size)
    return spikes


def random_walk(n, rng):
    return np.cumsum(rng.normal(0, WALK_STEP_DEG, n)) % 360.0


@pytest.fixture(scope="module")
def session():
    rng = np.random.default_rng(0)
    n_wake, n_nrem, n_rem = (int(s * RATE) for s in (WAKE_S, NREM_S, REM_S))
    n = n_wake + n_nrem + n_rem
    frame_times = np.arange(n) / RATE
    preferred = np.linspace(0, 360, N_HD, endpoint=False)

    wake_heading = random_walk(n_wake, rng)
    internal_rem = random_walk(n_rem, rng)
    tracked = np.r_[wake_heading, np.full(n_nrem + n_rem, 270.0)]
    head_direction = np.vstack(
        [
            von_mises_rates(wake_heading, preferred),
            np.full((n_nrem, N_HD), 1.0),
            von_mises_rates(internal_rem, preferred),
        ]
    )
    slow = gaussian_filter1d(rng.normal(size=n), 30 * RATE)
    gain = np.clip(1 + 0.4 * slow / slow.std(), 0.2, None)
    rates = np.hstack([head_direction, np.full((n, N_FLAT), 5.0)]) * gain[:, None]
    spikes = poisson_spike_times(rates, frame_times, rng)
    # a unit that fires only in NREM: silent in every wake bin
    nrem_frames = frame_times[n_wake : n_wake + n_nrem]
    spikes[N_HD + N_FLAT] = rng.choice(nrem_frames, 500)
    sorting = FakeSorting(spikes)

    wake_stop = float(frame_times[n_wake])
    nrem_stop = float(frame_times[n_wake + n_nrem])
    return {
        "sorting": sorting,
        "analyzer": FakeAnalyzer(sorting),
        "unit_ids": sorting.unit_ids,
        "heading_deg": tracked,
        "frame_times": frame_times,
        "intervals": {
            "WAKE": [[0.0, wake_stop]],
            "NREM": [[wake_stop, nrem_stop]],
            "REM": [[nrem_stop, float(frame_times[-1])]],
        },
        "preferred": preferred,
        "internal_rem": internal_rem,
        "n_wake": n_wake,
        "n_nrem": n_nrem,
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
def wake(session, data):
    return state_interval_mask(data, session["intervals"], "WAKE")


@pytest.fixture(scope="module")
def ring(data, wake):
    return fit_ring(data, wake, verbose=False)


@pytest.fixture(scope="module")
def run(session):
    return run_ring(
        session["analyzer"],
        session["unit_ids"],
        session["heading_deg"],
        session["frame_times"],
        session["intervals"],
        test_fraction=0.3,
        block_s=30.0,
        n_shuffles=50,
        decode_nrem=True,
        seed=1,
        verbose=False,
    )


def tuning_stats(session):
    """What the tuning notebook would say: the HD units tuned at their true PD."""
    rng = np.random.default_rng(9)
    stats = {}
    for unit in session["unit_ids"].tolist():
        tuned = unit < N_HD
        stats[unit] = HDTuningStats(
            mean_vector_length=rng.uniform(0.4, 0.8) if tuned else rng.uniform(0.0, 0.05),
            preferred_direction_deg=float(session["preferred"][unit])
            if tuned
            else float(rng.uniform(0, 360)),
            peak_rate_hz=30.0,
            mean_rate_hz=5.0,
            n_spikes=1000,
            p_value=0.001 if tuned else 0.5,
            mvl_threshold=0.1,
            significant=tuned,
        )
    return stats


# ── angles ───────────────────────────────────────────────────────────────
def test_circular_r_is_one_for_a_rotation_and_minus_one_for_a_mirror():
    """Units spread evenly round a ring have no mean direction, which is fine here.

    It is not for the Jammalamadaka coefficient, which needs one: for a perfect
    rotation of evenly spread angles it can come out anywhere.
    """
    a = np.linspace(0, 360, 24, endpoint=False)
    assert circular_r(a, (a + 77.0) % 360) == pytest.approx(1.0)
    assert circular_r(a, (200.0 - a) % 360) == pytest.approx(-1.0)
    assert abs(circular_correlation(a, (a + 77.0) % 360)) < 0.9  # why it is not used

    rng = np.random.default_rng(3)
    unrelated = circular_r(rng.uniform(0, 360, 500), rng.uniform(0, 360, 500))
    assert abs(unrelated) < 0.1


def test_circular_r_is_fisher_and_lee_over_pairs():
    rng = np.random.default_rng(4)
    a, b = rng.uniform(0, 360, 30), rng.uniform(0, 360, 30)
    ar, br = np.deg2rad(a), np.deg2rad(b)
    upper = np.triu_indices(30, 1)
    sa = np.sin(ar[:, None] - ar[None, :])[upper]
    sb = np.sin(br[:, None] - br[None, :])[upper]
    expected = np.sum(sa * sb) / np.sqrt(np.sum(sa**2) * np.sum(sb**2))
    assert circular_r(a, b) == pytest.approx(expected)


def test_alignment_recovers_a_reflection_and_rotation_and_undoes_them():
    rng = np.random.default_rng(5)
    a = rng.uniform(0, 360, 200)
    for flip, offset in ((False, 40.0), (True, 300.0)):
        b = ((-a if flip else a) + offset + rng.normal(0, 5, a.size)) % 360
        fitted = align_angles(a, b)
        assert fitted.flip == flip
        assert abs(circular_difference(fitted.offset_deg, offset)) < 2.0
        assert fitted.resultant > 0.95 > 0.2 > fitted.resultant_other
        np.testing.assert_allclose(fitted.invert(fitted.apply(a)), a % 360, atol=1e-9)


def test_two_alignments_compose_into_one():
    """A ring onto another ring, then that ring onto heading, in one step."""
    rng = np.random.default_rng(10)
    x = rng.uniform(0, 360, 50)
    for flip_a, flip_b in ((False, False), (False, True), (True, False), (True, True)):
        a = align_angles(x, ((-x if flip_a else x) + 25.0) % 360)
        b = align_angles(x, ((-x if flip_b else x) + 300.0) % 360)
        both = a.then(b)
        np.testing.assert_allclose(
            circular_difference(both.apply(x), b.apply(a.apply(x))), 0.0, atol=1e-9
        )
        assert both.flip == (flip_a != flip_b)
        assert "composed" in str(both)


def test_compare_angles_judges_a_pairing_against_permuted_ones():
    rng = np.random.default_rng(6)
    a = np.linspace(0, 360, 20, endpoint=False)
    b = (150.0 - a + rng.normal(0, 8, a.size)) % 360
    result = compare_angles(a, b, n_shuffles=200)
    assert result.alignment.flip
    assert result.circular_r > 0.95
    assert result.median_abs_error_deg < 10.0
    assert result.shuffle.p_value == pytest.approx(1 / 201)
    with pytest.raises(ValueError, match="need at least 3"):
        compare_angles([1.0, 2.0], [3.0, 4.0])


# ── the pair metric ──────────────────────────────────────────────────────
def toy_data(counts, bin_s=0.05):
    edges = np.arange(len(counts) + 1) * bin_s
    return DecoderData(
        counts=np.asarray(counts, dtype=np.int32),
        heading_deg=np.zeros(len(counts)),
        duration_s=np.diff(edges),
        time_s=(edges[:-1] + edges[1:]) / 2,
        edges=edges,
        unit_ids=np.arange(np.shape(counts)[1]),
        bin_frames=3,
    )


def brute_force(rates, mask, lags):
    """Correlation of every pair at each lag, from explicit sums over bin pairs."""
    index = np.flatnonzero(mask)
    x = rates - rates[index].mean(axis=0)
    sd = np.sqrt((x[index] ** 2).mean(axis=0))
    n_units = rates.shape[1]
    r = np.full((len(lags), n_units, n_units), np.nan)
    for k, lag in enumerate(lags):
        t = index[(index + lag >= 0) & (index + lag < len(mask))]
        t = t[mask[t + lag]]
        if t.size:
            r[k] = (x[t + lag].T @ x[t]) / t.size / np.outer(sd, sd)
    return r


@pytest.mark.parametrize("smooth_sigma_s", [0.0, 0.1])
def test_masked_lagged_correlations_match_a_brute_force_sum(smooth_sigma_s):
    """Each lag divided by its own number of in-mask pairs; gaps add nothing."""
    rng = np.random.default_rng(7)
    counts = rng.poisson(2.0, size=(1200, 4))
    counts[:, 1] += counts[:, 0]  # some shared spikes, so the pairs differ
    data = toy_data(counts)
    mask = np.ones(1200, dtype=bool)
    mask[200:260] = False
    mask[500:900:7] = False
    mask[1000:1100] = False

    result = pairwise_correlations(data, mask, smooth_sigma_s=smooth_sigma_s,
                                   max_lag_s=1.0, baseline_start_s=0.5)
    rates = data.counts / data.duration_s[:, None]
    if smooth_sigma_s:
        rates = gaussian_filter1d(rates, smooth_sigma_s / 0.05, axis=0, mode="constant",
                                  truncate=5.0)
    lags = np.arange(-20, 21)
    r = brute_force(rates, mask, lags)
    shoulder = np.abs(lags) >= 10
    np.testing.assert_allclose(result.zero_lag, r[20], atol=1e-6)
    np.testing.assert_allclose(result.baseline, np.nanmean(r[shoulder], axis=0), atol=1e-6)
    assert result.n_lags == shoulder.sum()


def test_the_correlogram_of_a_pair_is_the_metric_s_two_terms(data, wake):
    correlations = pairwise_correlations(data, wake, [0, 1])
    lags, r = pair_correlogram(data, wake, 0, 1)
    assert r[lags.size // 2] == pytest.approx(correlations.zero_lag[0, 1], abs=1e-6)
    shoulder = np.abs(lags) >= 5.0 - 1e-9
    assert r[shoulder].mean() == pytest.approx(correlations.baseline[0, 1], abs=1e-6)


def test_the_baseline_cancels_slow_comodulation_but_keeps_fast_synchrony():
    rng = np.random.default_rng(8)
    n = 72_000  # an hour of 50 ms bins
    slow = gaussian_filter1d(rng.normal(size=n), 600)  # a 30 s drift
    slow = np.clip(1 + 0.5 * slow / slow.std(), 0.1, None)
    a = rng.poisson(0.5 * slow)
    b = rng.poisson(0.5 * slow)  # shares only the drift with a
    c = rng.binomial(a, 0.6) + rng.poisson(0.2, n)  # fires with a, bin for bin
    data = toy_data(np.column_stack([a, b, c]))
    result = pairwise_correlations(data, np.ones(n, dtype=bool))
    metric = result.metric
    assert result.zero_lag[0, 1] > 0.05  # the drift correlates them...
    assert abs(metric[0, 1]) < 0.02  # ...and the baseline takes it away
    assert metric[0, 2] > 0.3  # fast synchrony survives


def test_a_shoulder_beyond_the_mask_is_refused():
    data = toy_data(np.random.default_rng(0).poisson(2.0, size=(400, 3)))
    mask = np.zeros(400, dtype=bool)
    mask[:50] = True  # 2.5 s, shorter than the shoulder's inner edge
    with pytest.raises(ValueError, match="no baseline"):
        pairwise_correlations(data, mask)
    with pytest.raises(ValueError, match="baseline_start_s <= max_lag_s"):
        pairwise_correlations(data, np.ones(400, dtype=bool), max_lag_s=1.0,
                              baseline_start_s=2.0)


# ── the ring ─────────────────────────────────────────────────────────────
def test_ring_positions_are_the_preferred_directions(session, ring):
    preferred = session["preferred"][ring.unit_ids]
    result = compare_angles(ring.angle_deg, preferred, n_shuffles=200)
    assert result.circular_r > 0.95
    assert result.median_abs_error_deg < 15.0
    assert result.shuffle.p_value == pytest.approx(1 / 201)


def test_a_ring_looks_like_one(ring):
    assert ring.eigenvalue_ratio > 0.5
    assert ring.eigenvalues[2] < 0.5 * ring.eigenvalues[1]
    assert ring.consistency_r > 0.7
    assert ring.split_half_r > 0.8
    assert ring.n_components == 1


def test_units_without_partners_are_screened_off_the_ring(ring):
    screen = ring.screen
    assert sorted(ring.unit_ids.tolist()) == list(range(N_HD))
    assert (screen.loc[list(range(N_HD)), "status"] == ON_RING).all()
    assert (screen.loc[list(range(N_HD, N_HD + N_FLAT)), "status"] == NO_PARTNERS).all()
    assert screen.loc[N_HD + N_FLAT, "status"] == SILENT
    assert set(ring.excluded_unit_ids.tolist()) == set(range(N_HD, N_HD + N_FLAT + 1))
    # the screen's margin: weakest partner of the ring vs strongest of the rest
    assert screen.loc[list(range(N_HD)), "partner_snr"].min() > 8.0
    assert screen.loc[list(range(N_HD, N_HD + N_FLAT)), "partner_snr"].max() < 4.0
    assert sorted(ring.members().tolist()) == list(range(N_HD))


def test_without_the_screen_every_firing_unit_is_placed_and_coupling_tells_them_apart(
    data, wake
):
    everyone = fit_ring(data, wake, min_partner_snr=None, verbose=False)
    assert everyone.n_units == N_HD + N_FLAT  # all but the silent one
    head_direction = everyone.unit_ids < N_HD
    coupling = everyone.coupling
    assert coupling[head_direction].min() > 0.5 > coupling[~head_direction].max()
    assert sorted(everyone.members().tolist()) == list(range(N_HD))


def test_the_ring_is_ordered_by_angle(ring):
    order = ring.order
    np.testing.assert_array_equal(np.sort(ring.angle_of(order)), ring.angle_of(order))
    assert ring.angle_of([ring.unit_ids[3]])[0] == ring.angle_deg[3]


def test_a_disconnected_graph_is_patched_with_a_warning():
    rng = np.random.default_rng(1)
    metric = np.zeros((20, 20))
    for block in (slice(0, 10), slice(10, 20)):
        metric[block, block] = rng.uniform(0.2, 0.6, (10, 10))
    metric = (metric + metric.T) / 2
    np.fill_diagonal(metric, np.nan)
    with pytest.warns(UserWarning, match="falls into 2 pieces"):
        result = _embed(metric, k_neighbors=3)
    assert result["n_components"] == 2
    assert np.isfinite(result["geodesic"]).all()


def test_too_few_units_or_a_bad_mask_is_refused(data, wake):
    with pytest.raises(ValueError, match="bins"):
        fit_ring(data, wake[:-1], verbose=False)
    with pytest.raises(ValueError, match="too few"):
        fit_ring(data, wake, unit_ids=list(range(8)), verbose=False)
    with pytest.raises(ValueError, match="keeps no bins"):
        fit_ring(data, np.zeros(data.n_bins, dtype=bool), verbose=False)


# ── decoding ─────────────────────────────────────────────────────────────
def test_run_ring_holds_out_exactly_the_bins_run_decoder_does(session, run):
    decoder = run_decoder(
        session["sorting"],
        list(range(N_HD)),
        session["heading_deg"],
        session["frame_times"],
        session["intervals"],
        bin_s=0.05,
        test_fraction=0.3,
        block_s=30.0,
        seed=1,
        n_shuffles=0,
        decode_rem=False,
        verbose=False,
    )
    np.testing.assert_array_equal(decoder.train_mask, run.train_mask)
    np.testing.assert_array_equal(decoder.test_mask, run.test_mask)
    # and both decode the same held-out bins
    np.testing.assert_array_equal(decoder.test.bin_index, run.test.bin_index)


def test_the_ring_decodes_held_out_wake(run):
    metrics = run.test.metrics
    assert metrics["median_abs_error_deg"] < 15.0
    assert metrics["median_abs_error_corrected_deg"] < 15.0
    assert run.test_shuffle.p_value == pytest.approx(1 / 51)
    assert run.test_shuffle.null.mean() > 60.0
    assert run.alignment.margin > 0.5
    # a unit's place on the ring, in heading coordinates, is its preferred direction
    errors = circular_difference(
        run.aligned_angle_deg, np.linspace(0, 360, N_HD, endpoint=False)[run.ring.unit_ids]
    )
    assert np.median(np.abs(errors)) < 15.0


def test_a_mirrored_heading_mirrors_the_alignment_and_nothing_else(session, run):
    mirrored = run_ring(
        session["sorting"],
        session["unit_ids"],
        (360.0 - session["heading_deg"]) % 360.0,
        session["frame_times"],
        session["intervals"],
        test_fraction=0.3,
        block_s=30.0,
        n_shuffles=0,
        decode_rem=False,
        seed=1,
        verbose=False,
    )
    assert mirrored.alignment.flip != run.alignment.flip
    np.testing.assert_allclose(mirrored.ring.angle_deg, run.ring.angle_deg)
    assert mirrored.test.metrics["median_abs_error_deg"] == pytest.approx(
        run.test.metrics["median_abs_error_deg"], abs=1.0
    )


def test_rem_decoding_follows_the_internal_heading(session, run):
    rem = run.rem
    frame = rem.bin_index * run.data.bin_frames - (session["n_wake"] + session["n_nrem"])
    truth = session["internal_rem"][np.clip(frame, 0, session["n_rem"] - 1)]
    assert rem.n_decoded > 0.9 * REM_S / run.data.bin_s
    assert np.median(np.abs(circular_difference(rem.decoded_deg, truth))) < 15.0
    assert circular_r(rem.decoded_deg, truth) > 0.9
    # and nothing like the parked camera
    assert np.median(np.abs(circular_difference(rem.decoded_deg, 270.0))) > 45.0
    assert run.rem_shuffle.p_value == pytest.approx(1 / 51)
    assert "median_abs_error_deg" not in rem.metrics


def test_unstructured_nrem_is_not_coherent(run):
    assert run.nrem.metrics["mean_posterior_max"] < run.rem.metrics["mean_posterior_max"]
    assert run.nrem_shuffle.z_score < 3.0 < run.rem_shuffle.z_score


def test_a_session_without_rem_warns(session):
    no_rem = {k: v for k, v in session["intervals"].items() if k != "REM"}
    with pytest.warns(UserWarning, match="no decoder bin falls inside a REM"):
        result = run_ring(
            session["sorting"],
            session["unit_ids"],
            session["heading_deg"],
            session["frame_times"],
            no_rem,
            n_shuffles=0,
            verbose=False,
        )
    assert result.rem is None


def test_weights_change_the_vote_not_the_bins(data, ring, run):
    test = run.test_mask
    default = decode_ring(data, ring, test, run.alignment)
    uniform = decode_ring(data, ring, test, run.alignment, weights="uniform")
    np.testing.assert_array_equal(default.bin_index, uniform.bin_index)
    assert uniform.metrics["median_abs_error_deg"] < 20.0
    with pytest.raises(ValueError, match="weights"):
        decode_ring(data, ring, test, weights="loudest")
    with pytest.raises(ValueError, match="ring units"):
        decode_ring(data, ring, test, weights=np.ones(3))


def test_a_ring_decode_works_with_the_decoder_s_tools(run):
    test = run.test
    labels = np.where(test.posterior_max > np.median(test.posterior_max), "long", "short")
    table = metrics_by_group(test, labels, order=["short", "long"])
    # a longer population vector is a more confident one
    assert table["long"]["median_abs_error_deg"] < table["short"]["median_abs_error_deg"]
    assert test.posterior is None and np.isnan(test.entropy_bits).all()
    assert test.n_spikes.sum() > 0


def test_decodes_on_different_grids_pair_bin_by_bin(session, run):
    fine = prepare_decoder_data(
        session["sorting"], session["unit_ids"], session["heading_deg"],
        session["frame_times"], bin_s=1 / 60,
    )
    fine_rem = decode_ring(
        fine, run.ring, state_interval_mask(fine, session["intervals"], "REM"),
        run.alignment, with_metrics=False,
    )
    index_fine, index_coarse = pair_decodes(fine_rem, run.rem)
    assert index_fine.size > 0.95 * fine_rem.n_decoded
    centre = fine_rem.time_s[index_fine]
    start = run.rem.time_s[index_coarse] - run.rem.duration_s[index_coarse] / 2
    assert np.all((centre >= start) & (centre < start + run.rem.duration_s[index_coarse]))
    agree = circular_difference(
        fine_rem.decoded_deg[index_fine], run.rem.decoded_deg[index_coarse]
    )
    assert np.median(np.abs(agree)) < 10.0


# ── comparisons ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def halves(data, wake):
    """The same population fitted twice, on each half of wake."""
    first = wake & (data.time_s < WAKE_S / 2)
    second = wake & (data.time_s >= WAKE_S / 2)
    return fit_ring(data, first, verbose=False), fit_ring(data, second, verbose=False)


def test_two_halves_of_one_session_agree(halves):
    comparison = compare_rings(*halves, n_shuffles=200)
    assert comparison.n == N_HD
    assert comparison.angles.circular_r > 0.9
    assert comparison.angles.shuffle.p_value == pytest.approx(1 / 201)
    assert comparison.structure_r > 0.8
    assert comparison.structure.p_value == pytest.approx(1 / 201)
    row = comparison.as_row()
    assert row["n_pairs"] == N_HD and row["structure_p"] == pytest.approx(1 / 201)


def test_a_scrambled_matching_does_not(halves):
    first, second = halves
    units = first.unit_ids.tolist()
    scrambled = dict(zip(units, np.random.default_rng(2).permutation(units).tolist()))
    comparison = compare_rings(first, second, unit_map=scrambled, n_shuffles=200)
    assert comparison.angles.shuffle.p_value > 0.05
    assert comparison.structure.p_value > 0.05
    assert comparison.structure_r < 0.3


def test_a_unit_map_is_followed_and_missing_units_are_skipped(halves):
    first, second = halves
    units = first.unit_ids.tolist()
    mapping = {u: u for u in units[:10]}
    mapping[999] = units[0]  # not on the first ring
    comparison = compare_rings(first, second, unit_map=mapping, n_shuffles=0)
    assert comparison.units_a.tolist() == units[:10]
    assert comparison.structure is None and comparison.angles.shuffle is None
    with pytest.raises(ValueError, match="at least 4"):
        compare_rings(first, second, unit_map={units[0]: units[0]})


def test_ring_vs_tuning_finds_the_tuned_units(session, ring):
    result = ring_vs_tuning(ring, tuning_stats(session), n_shuffles=200)
    crosstab = result.crosstab()
    assert crosstab.loc["tuned", "member"] == N_HD
    assert crosstab.loc["untuned", NO_PARTNERS] == N_FLAT
    assert result.auc_partner_snr == pytest.approx(1.0)
    assert result.angles.circular_r > 0.95
    assert result.angles.shuffle.p_value == pytest.approx(1 / 201)
    # the pair metric falls with tuning difference, embedding or no embedding
    assert result.structure_r > 0.7
    curve = result.curve
    assert curve["mean_z"].iloc[0] > 0 > curve["mean_z"].iloc[-1]
    assert np.isnan(result.auc_coupling)  # no untuned unit made it onto the ring
    text = result.summary()
    assert f"{N_HD} of {N_HD} tuned" in text and "n/a" in text


def test_the_summary_and_the_row_say_what_happened(run):
    text = run.summary()
    for phrase in ("ring wake test", "onto heading", "REM", "NREM", "without partners"):
        assert phrase in text
    row = run.as_row()
    assert row["n_units"] == N_HD and row["n_candidates"] == N_HD + N_FLAT + 1
    assert row["test_p"] == pytest.approx(1 / 51)
    assert row["rem_s"] > 0 and row["nrem_s"] > 0


# ── plots ────────────────────────────────────────────────────────────────
def test_the_plots_draw(session, data, wake, ring, halves, run):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spikeshpc.decoder import plot_decoded, plot_error
    from spikeshpc.ring import (
        plot_correlograms,
        plot_ring,
        plot_ring_comparison,
        plot_ring_vs_tuning,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to say about any of them
        preferred = dict(zip(range(N_HD), session["preferred"]))
        tuning = ring_vs_tuning(ring, tuning_stats(session), n_shuffles=20)
        comparison = compare_rings(*halves, n_shuffles=20)
        for make, n_axes in (
            (lambda: plot_ring(ring), 4),
            (lambda: plot_ring(ring, color=preferred), 4),
            (lambda: plot_correlograms(data, ring, wake), 3),
            (lambda: plot_ring_vs_tuning(tuning), 3),
            (lambda: plot_ring_comparison(comparison, ("1st", "2nd")), 4),
        ):
            axes = make()
            assert len(axes) == n_axes
            plt.close(axes[0].figure)
    for make in (
        lambda: plot_decoded(run.test, window_s=20.0),
        lambda: plot_error(run.test),
    ):
        made = make()
        axis = made[0] if isinstance(made, np.ndarray) else made
        plt.close(axis.figure)
