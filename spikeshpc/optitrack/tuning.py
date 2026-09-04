"""Head-direction tuning curves: unit firing rate vs. heading angle.

Firing rate is computed per inter-frame interval (the gaps between
consecutive shutter-closure timestamps), then binned by the heading at the
start of each interval into a circular, occupancy-normalized tuning curve.
Whether a curve is more directional than chance is decided by a shuffle test
(:func:`compute_hd_tuning_significance`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _frame_spike_counts(sorting, unit_id, frame_times: np.ndarray) -> np.ndarray:
    """Spike count in each inter-frame interval of ``frame_times``."""
    spike_times = sorting.get_unit_spike_train(unit_id, return_times=True)
    counts, _ = np.histogram(spike_times, bins=frame_times)
    return counts.astype(float)


def compute_frame_firing_rates(sorting, unit_id, frame_times: np.ndarray) -> np.ndarray:
    """Firing rate (Hz) in each inter-frame interval of ``frame_times``.

    ``frame_times`` must be on the same clock as the sorting's spike times --
    i.e. the (aligned) shutter-closure timestamps, not the OptiTrack take's
    own clock. Returns an array of length ``len(frame_times) - 1``.
    """
    return _frame_spike_counts(sorting, unit_id, frame_times) / np.diff(frame_times)


def _bin_headings(
    heading_deg: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bin centers (deg) and the bin index of each entry of ``heading_deg``."""
    bin_edges = np.linspace(0.0, 360.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(heading_deg, bin_edges) - 1, 0, n_bins - 1)
    return bin_centers, bin_idx


def _occupancy_normalized_rate(
    bin_idx: np.ndarray,
    spike_counts: np.ndarray,
    summed_occupancy: np.ndarray,
    n_bins: int,
    smooth_sigma_deg: float,
) -> np.ndarray:
    """Smoothed rate (Hz) per heading bin, given pre-binned headings.

    ``summed_occupancy`` is the time spent in each bin, i.e. the denominator
    that :func:`compute_hd_tuning_curve` builds; it is fixed across the shuffles
    of :func:`compute_hd_tuning_significance`, so it is computed once by the caller.
    """
    summed_spikes = np.bincount(bin_idx, weights=spike_counts, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(summed_occupancy > 0, summed_spikes / summed_occupancy, 0.0)

    sigma_bins = smooth_sigma_deg / (360.0 / n_bins)
    return gaussian_filter1d(rate, sigma=sigma_bins, mode="wrap")


def compute_hd_tuning_curve(
    heading_deg: np.ndarray,
    firing_rate: np.ndarray,
    occupancy_time: np.ndarray | None = None,
    n_bins: int = 360,
    smooth_sigma_deg: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Circular, occupancy-normalized tuning curve: rate (Hz) vs. heading bin.

    ``heading_deg`` and ``firing_rate`` are one value per interval (same
    length). Occupancy-weighting (spike count summed per bin / time summed
    per bin, rather than a plain mean of per-interval rates) matters when the
    animal spends unequal time at different headings; pass ``occupancy_time``
    (interval durations) for that -- otherwise every interval is weighted
    equally. Smoothing wraps at 0/360 degrees.
    """
    if occupancy_time is None:
        occupancy_time = np.ones_like(firing_rate)

    bin_centers, bin_idx = _bin_headings(heading_deg, n_bins)
    summed_occupancy = np.bincount(bin_idx, weights=occupancy_time, minlength=n_bins)
    smoothed = _occupancy_normalized_rate(
        bin_idx,
        firing_rate * occupancy_time,
        summed_occupancy,
        n_bins,
        smooth_sigma_deg,
    )
    return bin_centers, smoothed


def compute_mean_vector_length(
    bin_centers_deg: np.ndarray, rate: np.ndarray
) -> tuple[float, float]:
    """Directionality of a tuning curve: (mean vector length, preferred direction).

    The mean vector length (a.k.a. the Rayleigh vector length) is the rate-
    weighted circular mean of the heading bins, normalized by the summed rate:
    0 for a flat curve, 1 for all firing in a single bin. The preferred
    direction is that vector's angle, in [0, 360) degrees.

    Returns ``(nan, nan)`` for a silent unit (no firing in any bin).
    """
    total = rate.sum()
    if not total > 0:
        return float("nan"), float("nan")

    resultant = np.sum(rate * np.exp(1j * np.deg2rad(bin_centers_deg))) / total
    return float(np.abs(resultant)), float(np.degrees(np.angle(resultant)) % 360.0)


@dataclass
class HDTuningStats:
    """How directional a unit's tuning curve is, and how likely that is by chance.

    ``p_value`` is the fraction of shuffles whose mean vector length reached the
    observed one (see :func:`compute_hd_tuning_significance`), so it is bounded
    below by ``1 / (n_shuffles + 1)``. ``mvl_threshold`` is the corresponding
    critical value: the ``100 * (1 - alpha)``th percentile of this unit's own
    null distribution.
    """

    mean_vector_length: float
    preferred_direction_deg: float
    peak_rate_hz: float
    mean_rate_hz: float
    n_spikes: int
    p_value: float
    mvl_threshold: float
    significant: bool

    def __str__(self) -> str:
        return (
            f"MVL={self.mean_vector_length:.3f} (chance {self.mvl_threshold:.3f}), "
            f"preferred {self.preferred_direction_deg:.1f} deg, "
            f"peak {self.peak_rate_hz:.1f} Hz, mean {self.mean_rate_hz:.1f} Hz, "
            f"p={self.p_value:.4f}"
            f"{' *' if self.significant else ''}"
        )


def compute_hd_tuning_significance(
    analyzer,
    heading_deg: np.ndarray,
    frame_times: np.ndarray,
    unit_ids=None,
    n_bins: int = 36,
    smooth_sigma_deg: float = 10.0,
    n_shuffles: int = 500,
    min_shift_s: float = 20.0,
    alpha: float = 0.01,
    seed: int = 0,
) -> dict:
    """Test each unit's tuning curve against a shifted-spike-train null.

    Returns ``{unit_id: HDTuningStats}``. Arguments shared with
    :func:`compute_all_units_tuning_curves` mean the same thing there, and
    should be given the same values so the tested curves are the plotted ones.

    The statistic is the tuning curve's mean vector length
    (:func:`compute_mean_vector_length`). The null distribution comes from
    circularly shifting the unit's per-interval spike counts against the
    heading by a random offset of at least ``min_shift_s`` seconds and
    recomputing the curve. Shifting rather than permuting keeps the spike
    train's own temporal structure (bursting, slow rate drift) and the
    animal's occupancy intact, and only destroys their alignment -- an
    analytic Rayleigh test would instead assume independent samples and
    uniform sampling of heading, and would call almost every unit tuned.

    ``p_value`` is ``(1 + #{shuffled MVL >= observed}) / (n_shuffles + 1)``,
    and ``significant`` is ``p_value <= alpha``. Note this is a per-unit
    threshold: across many units, correct for multiple comparisons (or read
    ``p_value`` yourself) rather than trusting the flag on its own.
    """
    if len(heading_deg) != len(frame_times):
        raise ValueError(
            f"heading_deg ({len(heading_deg)}) and frame_times ({len(frame_times)}) "
            "must be the same length"
        )

    sorting = analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    occupancy_time = np.diff(frame_times)
    n_intervals = len(occupancy_time)
    bin_centers, bin_idx = _bin_headings(heading_deg[:-1], n_bins)
    summed_occupancy = np.bincount(bin_idx, weights=occupancy_time, minlength=n_bins)

    mean_interval = occupancy_time.mean()
    min_shift = int(round(min_shift_s / mean_interval))
    if 2 * min_shift >= n_intervals:
        raise ValueError(
            f"min_shift_s={min_shift_s} leaves no room to shift a recording of "
            f"{n_intervals * mean_interval:.1f} s"
        )
    shifts = np.random.default_rng(seed).integers(
        min_shift, n_intervals - min_shift, size=n_shuffles
    )

    stats = {}
    for unit_id in unit_ids:
        spike_counts = _frame_spike_counts(sorting, unit_id, frame_times)
        rate = _occupancy_normalized_rate(
            bin_idx, spike_counts, summed_occupancy, n_bins, smooth_sigma_deg
        )
        mvl, preferred_deg = compute_mean_vector_length(bin_centers, rate)

        if np.isnan(mvl):  # silent unit: no curve to test
            p_value, threshold, significant = 1.0, float("nan"), False
        else:
            # Slices of the doubled counts are the circular shifts, as views: no
            # per-shuffle copy, and the occupancy denominator stays put with the
            # heading, which is what the shifted train is being tested against.
            doubled = np.concatenate([spike_counts, spike_counts])
            null_mvl = np.array(
                [
                    compute_mean_vector_length(
                        bin_centers,
                        _occupancy_normalized_rate(
                            bin_idx,
                            doubled[shift : shift + n_intervals],
                            summed_occupancy,
                            n_bins,
                            smooth_sigma_deg,
                        ),
                    )[0]
                    for shift in shifts
                ]
            )
            p_value = (1 + np.count_nonzero(null_mvl >= mvl)) / (n_shuffles + 1)
            threshold = float(np.percentile(null_mvl, 100 * (1 - alpha)))
            significant = p_value <= alpha

        stats[unit_id] = HDTuningStats(
            mean_vector_length=mvl,
            preferred_direction_deg=preferred_deg,
            peak_rate_hz=float(rate.max()),
            mean_rate_hz=float(spike_counts.sum() / occupancy_time.sum()),
            n_spikes=int(spike_counts.sum()),
            p_value=float(p_value),
            mvl_threshold=threshold,
            significant=bool(significant),
        )
    return stats


def get_unit_depths(analyzer, unit_ids=None) -> dict:
    """Probe depth (the unit_locations y-coordinate, in the probe's native units) per unit.

    Requires the analyzer to have a computed ``"unit_locations"`` extension.
    Pass ``unit_ids`` to restrict/order the result (e.g. the same ids used for
    :func:`compute_all_units_tuning_curves`); default is every unit.
    """
    locations = analyzer.get_extension("unit_locations").get_data()
    depth_by_unit = dict(zip(analyzer.sorting.unit_ids, locations[:, 1]))
    if unit_ids is None:
        return depth_by_unit
    return {unit_id: depth_by_unit[unit_id] for unit_id in unit_ids}


def compute_all_units_tuning_curves(
    analyzer,
    heading_deg: np.ndarray,
    frame_times: np.ndarray,
    unit_ids=None,
    n_bins: int = 360,
    smooth_sigma_deg: float = 10.0,
) -> dict:
    """Tuning curve for each unit in ``unit_ids`` (default: every unit in ``analyzer.sorting``).

    Pass ``unit_ids`` (e.g. the ones labeled "good") to skip the rest rather
    than computing and discarding their tuning curves.

    ``heading_deg`` and ``frame_times`` must be the same length -- i.e.
    ``heading_deg`` already indexed down to the frames returned by
    :func:`optitrack.sync.align_frames_to_shutter_events`, matched 1:1 with
    the (aligned) ``frame_times`` (the shutter-closure timestamps). The last
    entry of each has no following interval and is dropped internally.
    """
    sorting = analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids
    occupancy_time = np.diff(frame_times)
    interval_heading = heading_deg[:-1]

    curves = {}
    for unit_id in unit_ids:
        firing_rate = compute_frame_firing_rates(sorting, unit_id, frame_times)
        curves[unit_id] = compute_hd_tuning_curve(
            interval_heading,
            firing_rate,
            occupancy_time=occupancy_time,
            n_bins=n_bins,
            smooth_sigma_deg=smooth_sigma_deg,
        )
    return curves
