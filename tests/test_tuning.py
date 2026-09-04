from __future__ import annotations

import numpy as np

from spikeshpc.optitrack.tuning import (
    compute_hd_tuning_curve,
    compute_hd_tuning_significance,
    compute_mean_vector_length,
)


def _angular_distance(a, b):
    return np.minimum(np.abs(a - b), 360.0 - np.abs(a - b))


def _bump_firing_rate(heading_deg, preferred_deg, width_deg=15.0, peak_hz=20.0, base_hz=1.0):
    in_bump = _angular_distance(heading_deg, preferred_deg) < width_deg
    return np.where(in_bump, peak_hz, base_hz)


def test_tuning_curve_peaks_at_preferred_heading():
    heading_deg = np.arange(3600) % 360.0
    firing_rate = _bump_firing_rate(heading_deg, preferred_deg=90.0)

    bin_centers, rate = compute_hd_tuning_curve(
        heading_deg, firing_rate, n_bins=36, smooth_sigma_deg=5.0
    )

    assert abs(bin_centers[np.argmax(rate)] - 90.0) <= 10.0


def test_tuning_curve_smoothing_wraps_across_0_360():
    heading_deg = np.arange(3600) % 360.0
    firing_rate = _bump_firing_rate(heading_deg, preferred_deg=350.0)

    bin_centers, rate = compute_hd_tuning_curve(
        heading_deg, firing_rate, n_bins=36, smooth_sigma_deg=5.0
    )

    peak = bin_centers[np.argmax(rate)]
    assert _angular_distance(peak, 350.0) <= 10.0


def test_mean_vector_length_flat_vs_concentrated():
    bin_centers = np.arange(5.0, 360.0, 10.0)

    flat_mvl, _ = compute_mean_vector_length(bin_centers, np.ones(36))
    assert flat_mvl < 1e-9

    single_bin = np.zeros(36)
    single_bin[9] = 10.0  # bin centered on 95 deg
    peaked_mvl, preferred = compute_mean_vector_length(bin_centers, single_bin)
    assert peaked_mvl == 1.0
    assert preferred == 95.0

    silent_mvl, silent_preferred = compute_mean_vector_length(bin_centers, np.zeros(36))
    assert np.isnan(silent_mvl) and np.isnan(silent_preferred)


class _FakeSorting:
    """Minimal stand-in for a spikeinterface sorting: spike times per unit."""

    def __init__(self, spike_times_by_unit):
        self._spike_times = spike_times_by_unit
        self.unit_ids = np.array(list(spike_times_by_unit))

    def get_unit_spike_train(self, unit_id, return_times=True):
        return self._spike_times[unit_id]


class _FakeAnalyzer:
    def __init__(self, sorting):
        self.sorting = sorting


def _simulated_session(duration_s=200.0, fps=120.0, turn_rate_deg_s=30.0, seed=1):
    """Frame times, heading (a steady turn sweeping all directions), and spikes.

    Unit "tuned" fires only near 90 deg; "untuned" fires at a constant rate
    independent of heading; "silent" never fires.
    """
    rng = np.random.default_rng(seed)
    frame_times = np.arange(0.0, duration_s, 1 / fps)
    heading_deg = (frame_times * turn_rate_deg_s) % 360.0

    in_bump = _angular_distance(heading_deg[:-1], 90.0) < 20.0
    tuned_p = np.where(in_bump, 0.15, 0.002)
    tuned = frame_times[:-1][rng.random(len(tuned_p)) < tuned_p]
    untuned = frame_times[:-1][rng.random(len(tuned_p)) < 0.02]

    sorting = _FakeSorting(
        {"tuned": tuned, "untuned": untuned, "silent": np.array([])}
    )
    return _FakeAnalyzer(sorting), heading_deg, frame_times


def test_hd_significance_separates_tuned_from_untuned_units():
    analyzer, heading_deg, frame_times = _simulated_session()

    stats = compute_hd_tuning_significance(
        analyzer, heading_deg, frame_times, n_shuffles=100, alpha=0.01
    )

    tuned = stats["tuned"]
    assert tuned.significant
    assert tuned.p_value <= 0.01
    assert tuned.mean_vector_length > tuned.mvl_threshold
    assert _angular_distance(tuned.preferred_direction_deg, 90.0) <= 10.0

    untuned = stats["untuned"]
    assert not untuned.significant
    assert untuned.mean_vector_length < untuned.mvl_threshold


def test_hd_significance_handles_silent_unit():
    analyzer, heading_deg, frame_times = _simulated_session()

    stats = compute_hd_tuning_significance(
        analyzer, heading_deg, frame_times, unit_ids=["silent"], n_shuffles=100
    )

    silent = stats["silent"]
    assert silent.n_spikes == 0
    assert np.isnan(silent.mean_vector_length)
    assert silent.p_value == 1.0
    assert not silent.significant
