"""Head-direction tuning curves: unit firing rate vs. heading angle.

Firing rate is computed per inter-frame interval (the gaps between
consecutive shutter-closure timestamps), then binned by the heading at the
start of each interval into a circular, occupancy-normalized tuning curve.
Whether a curve is more directional than chance is decided by a shuffle test
(:func:`compute_hd_tuning_significance`), optionally followed by a population-
level cut on how directional (:func:`apply_mvl_cutoff`).
"""

from __future__ import annotations

import warnings
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


def _check_interval_mask(interval_mask, n_intervals: int):
    """Validate an interval mask, or None. Returns a boolean array or None."""
    if interval_mask is None:
        return None
    keep = np.asarray(interval_mask, dtype=bool)
    if keep.shape != (n_intervals,):
        raise ValueError(
            f"interval_mask has {keep.shape} entries but there are "
            f"{n_intervals} inter-frame intervals. It masks intervals, not "
            "frames -- see spikeshpc.states.frames_in_states."
        )
    if not keep.any():
        raise ValueError("interval_mask keeps no intervals at all.")
    return keep


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

    ``weakly_tuned`` marks a unit that passed the shuffle test and the rate
    floor but fell below the population MVL cut, ``mvl_cutoff`` (NaN when no
    cut was applied); see :func:`apply_mvl_cutoff`. Such a unit is not
    ``significant``.
    """

    mean_vector_length: float
    preferred_direction_deg: float
    peak_rate_hz: float
    mean_rate_hz: float
    n_spikes: int
    p_value: float
    mvl_threshold: float
    significant: bool
    too_quiet: bool = False
    weakly_tuned: bool = False
    mvl_cutoff: float = float("nan")
    null_band: np.ndarray | None = None  # (n_percentiles, n_bins), Hz
    null_bin_centers_deg: np.ndarray | None = None
    null_percentiles: tuple = ()

    def __str__(self) -> str:
        return (
            f"MVL={self.mean_vector_length:.3f} (chance {self.mvl_threshold:.3f}), "
            f"preferred {self.preferred_direction_deg:.1f} deg, "
            f"peak {self.peak_rate_hz:.1f} Hz, mean {self.mean_rate_hz:.1f} Hz, "
            f"p={self.p_value:.4f}"
            f"{' (too quiet)' if self.too_quiet else ''}"
            f"{' (weakly tuned)' if self.weakly_tuned else ''}"
            f"{' *' if self.significant else ''}"
        )


def find_bimodal_threshold(values) -> float:
    """The cut that best splits ``values`` into a low group and a high group.

    Otsu's method, done exactly: every split between consecutive sorted values
    is tried and the one maximizing the between-group variance wins, which is
    the same as the best two-cluster k-means. The threshold is the midpoint of
    the gap it picks. No histogram binning, so it behaves with a few dozen
    values, which is what a probe's worth of tuned units amounts to.

    It always returns a split, bimodal or not -- look at the distribution
    before trusting it on a new dataset.
    """
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if len(v) < 2 or v[0] == v[-1]:
        raise ValueError(f"need at least two distinct finite values, got {len(v)}")

    n = len(v)
    k = np.arange(1, n)  # size of the low group
    csum = np.cumsum(v)
    mean_low = csum[:-1] / k
    mean_high = (csum[-1] - csum[:-1]) / (n - k)
    between = (k / n) * (1 - k / n) * (mean_low - mean_high) ** 2
    between[v[1:] == v[:-1]] = -np.inf  # a tie cannot be split
    best = int(np.argmax(between))
    return float((v[best] + v[best + 1]) / 2)


def apply_mvl_cutoff(stats: dict, min_mvl) -> float:
    """Drop significant units whose mean vector length is below a cut. In place.

    ``min_mvl`` is None (no cut), a float (the cut itself), or ``"bimodal"``
    (the cut :func:`find_bimodal_threshold` picks from the MVLs of the units
    that passed the shuffle test and the rate floor). Units below the cut are
    flagged ``weakly_tuned`` and lose ``significant``; everything else is left
    as it was. Returns the cut used, NaN for none.

    The shuffle test only asks whether a curve beats chance, and with enough
    spikes a barely-directional curve does. Among tuned units the MVLs tend to
    split into a weak group just clearing the shuffle and a strong group of
    clear head-direction cells; this keeps the second.

    ``"bimodal"`` is population-level, so the cut depends on which units are in
    ``stats``. It can be re-run on stats that were already cut (e.g. a loaded
    ``HDTuning.stats``) to try a different ``min_mvl`` without redoing the
    shuffles: the candidates are the units that were significant *before* any
    earlier cut, and ``None`` restores them.
    """
    candidates = [s for s in stats.values() if s.significant or s.weakly_tuned]

    if min_mvl is None:
        cutoff = float("nan")
    elif isinstance(min_mvl, str):
        if min_mvl != "bimodal":
            raise ValueError(f"min_mvl must be None, 'bimodal' or a float, not {min_mvl!r}")
        mvls = [s.mean_vector_length for s in candidates]
        if len(set(mvls)) < 2:
            warnings.warn(
                f"min_mvl='bimodal' needs at least two tuned units to split, got "
                f"{len(candidates)}; no MVL cut applied."
            )
            cutoff = float("nan")
        else:
            cutoff = find_bimodal_threshold(mvls)
    else:
        cutoff = float(min_mvl)

    for s in stats.values():
        s.mvl_cutoff = cutoff
    for s in candidates:
        s.weakly_tuned = bool(s.mean_vector_length < cutoff)  # False for a NaN cutoff
        s.significant = not s.weakly_tuned
    return cutoff


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
    interval_mask=None,
    min_peak_rate_hz: float = 0.0,
    null_percentiles=(2.5, 50.0, 97.5),
    min_mvl=None,
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

    ``min_peak_rate_hz`` additionally requires the curve to reach that rate
    somewhere before the unit counts as tuned, and flags the rest as
    ``too_quiet``. The shuffle test asks whether a unit's firing is more
    concentrated in heading than chance, which a unit firing a handful of
    spikes can satisfy on sparsity alone -- the shifted null is just as sparse,
    so a couple of spikes that happen to land together beat it. Such a curve is
    not wrong, it is simply an estimate from almost nothing, and it carries
    nearly no information into a decoder's likelihood. Default 0.0 keeps every
    unit the test passes; 1 Hz is a reasonable floor for decoding.

    ``min_mvl`` then cuts, among the units still significant, the ones whose
    mean vector length is low: None (default) for no cut, a float to use as
    the cut, or ``"bimodal"`` to pick it from the tuned units' MVL
    distribution. They are flagged ``weakly_tuned``; the cut used is on every
    unit's ``mvl_cutoff``. See :func:`apply_mvl_cutoff`, which can also re-cut
    saved stats without recomputing them.

    ``null_percentiles`` keeps the shuffled *curves* as well as their summary
    statistic, as a per-bin envelope on ``HDTuningStats.null_band`` -- what a
    chance curve looks like for this unit's own spike count and the animal's
    own occupancy. Drawing it under the real curve
    (:func:`optitrack.widgets.show_hd_tuning_widget`) turns "p = 0.002" back
    into something that can be looked at, which matters most for the sparse
    units where a p-value is least intuitive. Pass ``()`` to skip it.

    Read that band as pointwise, not simultaneous: across 36 bins, a real
    curve poking above the 97.5th percentile in one of them is unremarkable.
    The MVL p-value is still the test; the band is for seeing what it tested.
    """
    if len(heading_deg) != len(frame_times):
        raise ValueError(
            f"heading_deg ({len(heading_deg)}) and frame_times ({len(frame_times)}) "
            "must be the same length"
        )
    if isinstance(min_mvl, str) and min_mvl != "bimodal":  # before the slow part
        raise ValueError(f"min_mvl must be None, 'bimodal' or a float, not {min_mvl!r}")

    sorting = analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    occupancy_time = np.diff(frame_times)
    interval_heading = heading_deg[:-1]
    keep = _check_interval_mask(interval_mask, len(occupancy_time))
    if keep is not None:
        occupancy_time = occupancy_time[keep]
        interval_heading = interval_heading[keep]
    n_intervals = len(occupancy_time)
    bin_centers, bin_idx = _bin_headings(interval_heading, n_bins)
    summed_occupancy = np.bincount(bin_idx, weights=occupancy_time, minlength=n_bins)

    mean_interval = occupancy_time.mean()
    min_shift = round(min_shift_s / mean_interval)
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
        if keep is not None:
            spike_counts = spike_counts[keep]
        rate = _occupancy_normalized_rate(
            bin_idx, spike_counts, summed_occupancy, n_bins, smooth_sigma_deg
        )
        mvl, preferred_deg = compute_mean_vector_length(bin_centers, rate)

        band = None
        if np.isnan(mvl):  # silent unit: no curve to test
            p_value, threshold, significant = 1.0, float("nan"), False
        else:
            # Slices of the doubled counts are the circular shifts, as views: no
            # per-shuffle copy, and the occupancy denominator stays put with the
            # heading, which is what the shifted train is being tested against.
            doubled = np.concatenate([spike_counts, spike_counts])
            null_curves = np.array(
                [
                    _occupancy_normalized_rate(
                        bin_idx,
                        doubled[shift : shift + n_intervals],
                        summed_occupancy,
                        n_bins,
                        smooth_sigma_deg,
                    )
                    for shift in shifts
                ]
            )
            null_mvl = np.array(
                [compute_mean_vector_length(bin_centers, c)[0] for c in null_curves]
            )
            p_value = (1 + np.count_nonzero(null_mvl >= mvl)) / (n_shuffles + 1)
            threshold = float(np.percentile(null_mvl, 100 * (1 - alpha)))
            significant = p_value <= alpha
            if null_percentiles:
                # only the envelope is kept: the curves themselves are
                # n_shuffles x n_bins per unit, which for a whole probe is
                # hundreds of megabytes to describe a shaded region
                band = np.percentile(null_curves, list(null_percentiles), axis=0)

        too_quiet = bool(rate.max() < min_peak_rate_hz)
        significant = bool(significant and not too_quiet)

        stats[unit_id] = HDTuningStats(
            mean_vector_length=mvl,
            preferred_direction_deg=preferred_deg,
            peak_rate_hz=float(rate.max()),
            mean_rate_hz=float(spike_counts.sum() / occupancy_time.sum()),
            n_spikes=int(spike_counts.sum()),
            p_value=float(p_value),
            mvl_threshold=threshold,
            significant=significant,
            too_quiet=too_quiet,
            null_band=band,
            null_bin_centers_deg=bin_centers if band is not None else None,
            null_percentiles=tuple(null_percentiles) if band is not None else (),
        )
    if min_mvl is not None:
        apply_mvl_cutoff(stats, min_mvl)
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


def get_kilosort_unit_locations(ks_path, unit_ids=None) -> dict:
    """``{unit_id: (x, y)}`` in probe coordinates, from kilosort's own spike positions.

    kilosort4 estimates an (x, y) for every spike (``spike_positions.npy``)
    but saves no per-cluster location, so a unit's is taken here as the median
    over its spikes -- robust to the few spikes localized off onto a
    neighbouring column. This is kilosort's estimate, not the analyzer's
    center-of-mass one that :func:`get_unit_depths` reads.

    Clusters come from ``spike_clusters.npy``, i.e. after any curation merges,
    matching the unit ids ``si.read_kilosort`` gives. ``unit_ids`` restricts
    and orders the result (default: every cluster with spikes).
    """
    from pathlib import Path

    ks_path = Path(ks_path)
    positions = np.load(ks_path / "spike_positions.npy")
    clusters = np.load(ks_path / "spike_clusters.npy").ravel()
    if len(positions) != len(clusters):
        raise ValueError(
            f"{len(positions)} spike positions but {len(clusters)} spike clusters in {ks_path}"
        )
    order = np.argsort(clusters, kind="stable")
    ids, starts = np.unique(clusters[order], return_index=True)
    groups = np.split(positions[order], starts[1:])
    location_by_unit = {
        int(unit): tuple(float(v) for v in np.median(group, axis=0))
        for unit, group in zip(ids, groups)
    }
    if unit_ids is None:
        return location_by_unit
    missing = [u for u in unit_ids if int(u) not in location_by_unit]
    if missing:
        raise KeyError(f"no kilosort spikes for units {missing[:10]}")
    return {u: location_by_unit[int(u)] for u in unit_ids}


def compute_all_units_tuning_curves(
    analyzer,
    heading_deg: np.ndarray,
    frame_times: np.ndarray,
    unit_ids=None,
    n_bins: int = 360,
    smooth_sigma_deg: float = 10.0,
    interval_mask=None,
) -> dict:
    """Tuning curve for each unit in ``unit_ids`` (default: every unit in ``analyzer.sorting``).

    Pass ``unit_ids`` (e.g. the ones labeled "good") to skip the rest rather
    than computing and discarding their tuning curves.

    ``heading_deg`` and ``frame_times`` must be the same length -- i.e.
    ``heading_deg`` already indexed down to the frames returned by
    :func:`optitrack.sync.align_frames_to_shutter_events`, matched 1:1 with
    the (aligned) ``frame_times`` (the shutter-closure timestamps). The last
    entry of each has no following interval and is dropped internally.

    ``interval_mask`` (length ``len(frame_times) - 1``) restricts the curve to
    a subset of inter-frame intervals -- the wake epochs, say. Pass the *full*
    arrays alongside it rather than pre-filtering them: a filtered
    ``frame_times`` splices the removed span into one enormous interval that
    absorbs every spike fired during it. See
    :func:`spikeshpc.states.frames_in_states`.
    """
    sorting = analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids
    occupancy_time = np.diff(frame_times)
    interval_heading = heading_deg[:-1]
    keep = _check_interval_mask(interval_mask, len(occupancy_time))
    if keep is not None:
        occupancy_time = occupancy_time[keep]
        interval_heading = interval_heading[keep]

    curves = {}
    for unit_id in unit_ids:
        firing_rate = compute_frame_firing_rates(sorting, unit_id, frame_times)
        if keep is not None:
            firing_rate = firing_rate[keep]
        curves[unit_id] = compute_hd_tuning_curve(
            interval_heading,
            firing_rate,
            occupancy_time=occupancy_time,
            n_bins=n_bins,
            smooth_sigma_deg=smooth_sigma_deg,
        )
    return curves


def plot_hd_tuning_population(
    hd_tuning,
    significant_only: bool = True,
    normalized: bool = True,
    cmap="hsv",
    n_hist_bins: int = 36,
    figsize=(10.0, 4.5),
):
    """Where the population's tuning peaks: a histogram and every curve overlaid.

    ``hd_tuning`` is an :class:`optitrack.store.HDTuning` (from
    ``load_hd_tuning``). ``significant_only`` restricts to its ``tuned_ids``.

    Left: how many units peak in each of ``n_hist_bins`` heading bins, where a
    unit's peak is the argmax of its saved curve -- the direction of maximal
    firing, not the MVL's preferred direction, which a skewed curve pulls off
    the peak. Each unit is its own block in the stack, in its own color.
    Right: every curve on a polar axis. With ``normalized`` (the default) each
    is divided by its own peak so all of them reach 1, which lines up shapes
    and directions whatever the rate. Without it they are drawn in Hz on one
    shared scale, out to a round number past the highest peak, so how hard
    each unit fires shows too -- at the cost of one fast unit flattening the
    quiet ones.

    Colors are ``cmap`` sampled evenly across the units in order of peak
    direction, so each is distinct and the same unit has the same color in both
    panels. ``cmap`` is anything ``matplotlib.colormaps`` takes; the default
    ``"hsv"`` is cyclic, like heading. Units that never fire have no peak and
    are left out.

    Returns ``(fig, (ax_hist, ax_polar))``.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    unit_ids = hd_tuning.tuned_ids if significant_only else hd_tuning.unit_ids
    bin_centers = np.asarray(hd_tuning.bin_centers_deg, dtype=float)
    curves = np.array([hd_tuning.curve(u)[1] for u in unit_ids], dtype=float)
    if curves.size:
        firing = curves.max(axis=1) > 0
        unit_ids, curves = np.asarray(unit_ids)[firing], curves[firing]
    if len(unit_ids) == 0:
        raise ValueError(
            "no units to plot" + (" -- none are significantly tuned" if significant_only else "")
        )

    peak_deg = bin_centers[np.argmax(curves, axis=1)]
    order = np.argsort(peak_deg, kind="stable")
    unit_ids, curves, peak_deg = unit_ids[order], curves[order], peak_deg[order]
    radii = curves / curves.max(axis=1, keepdims=True) if normalized else curves
    n = len(unit_ids)
    # bin centers, not 0..1 inclusive: a cyclic map's two ends are one color
    colors = matplotlib.colormaps.get_cmap(cmap)((np.arange(n) + 0.5) / n)

    fig = plt.figure(figsize=figsize)
    ax_hist = fig.add_subplot(1, 2, 1)
    ax_polar = fig.add_subplot(1, 2, 2, projection="polar")

    edges = np.linspace(0.0, 360.0, n_hist_bins + 1)
    hist_idx = np.clip(np.digitize(peak_deg, edges) - 1, 0, n_hist_bins - 1)
    stack = np.zeros(n_hist_bins)
    for i, color in zip(hist_idx, colors):
        ax_hist.bar(
            edges[i], 1, width=edges[1] - edges[0], bottom=stack[i], align="edge",
            color=color, edgecolor="white", linewidth=0.5,
        )
        stack[i] += 1
    ax_hist.set_xlim(0, 360)
    ax_hist.set_xticks(np.arange(0, 361, 90))
    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_hist.set_xlabel("Heading of peak firing (degrees)")
    ax_hist.set_ylabel("Units")
    ax_hist.spines[["top", "right"]].set_visible(False)

    theta = np.deg2rad(np.r_[bin_centers, bin_centers[0]])  # closed loop
    for rate, color, unit in zip(radii, colors, unit_ids):
        ax_polar.plot(theta, np.r_[rate, rate[0]], color=color, lw=1.3, label=str(unit))
    spokes = np.arange(0, 360, 30)
    ax_polar.set_thetagrids(spokes, labels=[f"{a}" if a % 90 == 0 else "" for a in spokes])
    if normalized:
        ax_polar.set_ylim(0, 1.0)
        ax_polar.set_yticks([0.5, 1.0])
        ax_polar.set_yticklabels(["", "1"], fontsize=8, color="gray")
        ax_polar.set_title("Normalized firing rate", fontsize=10, pad=14)
    else:
        # the locator's last tick is at or past the highest peak, so the rim is
        # a labeled ring and no curve is clipped
        rings = MaxNLocator(nbins=4).tick_values(0, curves.max())
        rings = rings[rings > 0]
        ax_polar.set_ylim(0, rings[-1])
        ax_polar.set_yticks(rings)
        ax_polar.set_yticklabels([f"{r:g}" for r in rings], fontsize=8, color="gray")
        ax_polar.set_title("Firing rate (Hz)", fontsize=10, pad=14)
    ax_polar.set_rlabel_position(45)
    ax_polar.grid(color="0.85", lw=0.6)

    which = "significantly tuned units" if significant_only else "units"
    fig.suptitle(f"{hd_tuning.session}: {n} {which}", fontsize=11)
    fig.tight_layout()
    return fig, (ax_hist, ax_polar)


def plot_tuning_comparison(
    curve_sets: dict,
    unit_ids=None,
    ncols: int = 6,
    colors=None,
    panel_size=(2.4, 1.9),
):
    """Each unit's tuning curve under several conditions, one panel per unit.

    ``curve_sets`` is ``{condition: curves}``, each ``curves`` the dict
    :func:`compute_all_units_tuning_curves` returns -- the same units during
    moving and still wake, say, or the first and second half of a session. A
    unit whose curve keeps its peak and shape across conditions codes heading
    the same way in both; one whose curve flattens or moves does not.

    Curves are in Hz, each panel scaled to its own unit, and each title gives
    the conditions' mean vector lengths in legend order. Only units with a
    curve in every condition are drawn; ``unit_ids`` picks and orders them
    (default: the first condition's order).

    Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    if not curve_sets:
        raise ValueError("no conditions to compare")
    names = list(curve_sets)
    if unit_ids is None:
        unit_ids = list(curve_sets[names[0]])
    unit_ids = [u for u in unit_ids if all(u in curve_sets[n] for n in names)]
    if not unit_ids:
        raise ValueError("no unit has a curve in every condition")
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ncols = max(1, min(ncols, len(unit_ids)))
    nrows = int(np.ceil(len(unit_ids) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )
    for ax, unit in zip(axes.flat, unit_ids):
        mvls = []
        for k, name in enumerate(names):
            centers, rate = (np.asarray(a, dtype=float) for a in curve_sets[name][unit])
            ax.plot(centers, rate, color=colors[k % len(colors)], lw=1.2, label=str(name))
            mvls.append(compute_mean_vector_length(centers, rate)[0])
        ax.set_title(
            f"unit {unit}  MVL " + " / ".join(f"{m:.2f}" for m in mvls), fontsize=8
        )
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 90))
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(unit_ids):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Hz", fontsize=8)
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.supxlabel("heading (deg)", fontsize=9)
    fig.tight_layout()
    return fig, axes


def _draw_direction_wheel(ax, cmap, n: int = 360):
    """A ring colored by ``cmap`` around the heading circle, as the color key."""
    theta = np.linspace(0.0, 2 * np.pi, n + 1)
    ax.pcolormesh(
        theta, [0.6, 1.0], (theta[:-1] / (2 * np.pi))[None, :],
        cmap=cmap, vmin=0.0, vmax=1.0, shading="flat",
    )
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0", "90", "180", "270"], fontsize=7)
    ax.tick_params(pad=0)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title("Preferred\ndirection (deg)", fontsize=8, pad=10)


def plot_hd_tuning_on_probe(
    hd_tuning,
    unit_locations: dict,
    probe,
    significant_only: bool = True,
    show_untuned: bool = True,
    inset_depth_um=(1800.0, 2400.0),
    cmap="hsv",
    size_per_mvl: float = 600.0,
    legend_mvls=(0.4, 0.6, 0.8),
    figsize=(8.0, 10.0),
):
    """Where the head-direction units sit on the probe: one bubble per unit.

    Each bubble is drawn at the unit's ``unit_locations`` (``{unit_id: (x, y)}``
    in probe coordinates, e.g. :func:`get_kilosort_unit_locations`), over the
    probe outline and contacts as :func:`probeinterface.plotting.plot_probe`
    draws them. Its *area* is proportional to the unit's mean vector length
    (``size_per_mvl`` points^2 per unit MVL, a fixed scale so sessions can be
    compared) and its color is the MVL's preferred direction on the cyclic
    ``cmap``, keyed by the color wheel.

    ``hd_tuning`` is an :class:`optitrack.store.HDTuning`. ``significant_only``
    bubbles just its ``tuned_ids``; with ``show_untuned`` every other unit in
    it is marked as a small grey dot, so the tuned ones can be read against
    where units were recorded at all. ``probe`` is a probeinterface ``Probe``
    or a path to a probeinterface JSON (the pipeline's ``probe.json``; its
    first probe is used).

    Left: the whole probe. Right: ``inset_depth_um`` (``(low, high)`` in the
    probe's y coordinates, or None for no inset) enlarged, its span boxed on
    the left panel. Depth is the probe's y, measured up from the tip. The
    whole-probe bubbles are drawn at a third of the inset's size, since that
    panel is compressed several-fold; the MVL legend is at the inset's scale.

    Returns ``(fig, axes)``, ``axes`` a dict with ``"probe"``, ``"inset"``
    (absent without one) and ``"wheel"``.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch, Rectangle
    from probeinterface import read_probeinterface
    from probeinterface.plotting import plot_probe

    if not hasattr(probe, "contact_positions"):
        probe = read_probeinterface(probe).probes[0]
    cmap = matplotlib.colormaps.get_cmap(cmap)

    all_ids = list(hd_tuning.unit_ids)
    shown = list(hd_tuning.tuned_ids) if significant_only else all_ids
    if not shown:
        raise ValueError(
            "no units to plot" + (" -- none are significantly tuned" if significant_only else "")
        )
    missing = [u for u in shown if u not in unit_locations]
    if missing:
        raise KeyError(f"no location for units {missing[:10]}")
    # biggest first, so a small bubble is never hidden under a large one
    mvl = np.array([hd_tuning.stats[u].mean_vector_length for u in shown])
    order = np.argsort(-mvl, kind="stable")
    shown = [shown[i] for i in order]
    mvl = mvl[order]
    pref = np.array([hd_tuning.stats[u].preferred_direction_deg for u in shown]) % 360.0
    xy = np.array([unit_locations[u] for u in shown], dtype=float)
    shown_set = set(shown)
    background = [u for u in all_ids if u not in shown_set and u in unit_locations]
    bg_xy = np.array([unit_locations[u] for u in background], dtype=float).reshape(-1, 2)
    show_bg = show_untuned and len(bg_xy) > 0

    contacts = probe.contact_positions
    x_lo, x_hi = contacts[:, 0].min(), contacts[:, 0].max()
    pad_x = max(40.0, 0.6 * (x_hi - x_lo))
    xlims = (x_lo - pad_x, x_hi + pad_x)
    full_ylims = (contacts[:, 1].min() - 150.0, contacts[:, 1].max() + 100.0)

    def draw(ax, ylims, bubble_scale):
        plot_probe(
            probe, ax=ax, title=False, xlims=xlims, ylims=ylims,
            contacts_colors="0.8",
            contact_kwargs=dict(alpha=1.0, edgecolor="none", lw=0, zorder=2),
            # plot_probe adds the outline after the contacts; keep it underneath
            probe_shape_kwargs=dict(facecolor="0.95", edgecolor="0.6", lw=0.8, alpha=1.0, zorder=1),
        )
        ax.set_aspect("auto")
        if show_bg:
            ax.scatter(bg_xy[:, 0], bg_xy[:, 1], s=8 * bubble_scale, c="0.45",
                       lw=0, alpha=0.7, zorder=3)
        ax.scatter(
            xy[:, 0], xy[:, 1], s=size_per_mvl * mvl * bubble_scale,
            c=cmap(pref / 360.0), edgecolors="k", linewidths=0.5, alpha=0.85, zorder=4,
        )
        ax.set_xlabel("x (µm)", fontsize=9)
        ax.set_ylabel("Depth from tip (µm)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig = plt.figure(figsize=figsize)
    has_inset = inset_depth_um is not None
    if has_inset:
        grid = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.35,
                                left=0.1, right=0.76, bottom=0.07, top=0.92)
        ax_probe = fig.add_subplot(grid[0])
        ax_inset = fig.add_subplot(grid[1])
    else:
        ax_probe = fig.add_axes([0.15, 0.07, 0.5, 0.85])
    draw(ax_probe, full_ylims, 1 / 3 if has_inset else 1.0)
    ax_probe.set_title("Whole probe", fontsize=10)
    axes = {"probe": ax_probe}

    if has_inset:
        lo, hi = sorted(inset_depth_um)
        draw(ax_inset, (lo, hi), 1.0)
        n_in = int(np.sum((xy[:, 1] >= lo) & (xy[:, 1] <= hi)))
        ax_inset.set_title(f"{lo:g}–{hi:g} µm ({n_in} of {len(shown)} units)", fontsize=10)
        ax_probe.add_patch(Rectangle(
            (xlims[0], lo), xlims[1] - xlims[0], hi - lo,
            fill=False, edgecolor="k", lw=1.0, ls="--", zorder=5,
        ))
        for y in (lo, hi):
            fig.add_artist(ConnectionPatch(
                xyA=(xlims[1], y), coordsA=ax_probe.transData,
                xyB=(xlims[0], y), coordsB=ax_inset.transData,
                color="0.4", lw=0.8, ls="--",
            ))
        axes["inset"] = ax_inset

    ax_wheel = fig.add_axes([0.81, 0.72, 0.14, 0.14], projection="polar")
    _draw_direction_wheel(ax_wheel, cmap)
    axes["wheel"] = ax_wheel

    handles = [
        Line2D([], [], ls="", marker="o", markersize=np.sqrt(size_per_mvl * m),
               markerfacecolor="0.7", markeredgecolor="k", markeredgewidth=0.5,
               label=f"{m:g}")
        for m in legend_mvls
    ]
    if show_bg:
        handles.append(Line2D([], [], ls="", marker="o", markersize=np.sqrt(8),
                              markerfacecolor="0.45", markeredgecolor="none",
                              label="untuned" if significant_only else "no location"))
    fig.legend(
        handles=handles, title="MVL" + ("\n(inset scale)" if has_inset else ""),
        loc="upper left", bbox_to_anchor=(0.79, 0.62),
        frameon=False, fontsize=8, title_fontsize=8, labelspacing=1.6, borderpad=1.0,
    )

    which = "significantly tuned units" if significant_only else "units"
    fig.suptitle(f"{hd_tuning.session}: {len(shown)} {which} on the probe", fontsize=11)
    return fig, axes
