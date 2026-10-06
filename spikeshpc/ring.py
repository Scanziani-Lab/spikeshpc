"""A head-direction ring from the correlation structure of spiking, without heading.

Head-direction cells behave as a ring attractor: cells that prefer nearby
directions fire together, and cells that prefer opposite ones do not. That
structure is in the spikes themselves -- it survives sleep, when nothing turns
the head (Peyrache et al. 2015) -- so it can be read without any reference to
heading. This module places units on a ring by how they co-fire, then decodes
heading from where on that ring the population is active. A unit's place on
the ring is a property of the network rather than of its tuning to the head,
which is what makes it usable where tuning has been disrupted: a unit followed
correctly from one recording to the next should keep its place on the ring.

Ported from the lab's ``HD_CCH_to_ring.m``, with ideas from SPUD (Chaudhuri et
al. 2019, https://doi.org/10.1038/s41593-019-0460-x). For each pair of units:

1. the zero-lag correlation of their Gaussian-smoothed rates, minus their mean
   correlation over the "shoulder" lags (5-10 s). The shoulder holds what the
   pair shares slowly -- arousal, brain state, a common drift in rate -- so
   what is left is fast co-firing;
2. z-scored across pairs, and turned into a distance, ``max(z) - z``.

Then Isomap: a k-nearest-neighbour graph on those distances, geodesic
distances through it, and classical MDS to two dimensions. Units of a ring
come out on a circle, and each unit's angle there is its position on the
ring. The decode is a population vector: each bin's z-scored rates summed
around the ring, each unit pointing at its own angle.

What differs from the MATLAB, on purpose:

- Masked time is handled exactly. The MATLAB zero-fills everything outside
  the mobility epochs and correlates across the gaps, relying on the baseline
  to cancel the common drop in rate that creates. Here a gap contributes
  nothing, and each lag's covariance is divided by its own number of in-mask
  bin pairs, which matters for fragmented masks such as moving-only wake.
- The shoulder is one FFT filter per unit and one matrix product, rather than
  a cross-correlogram per pair.
- Units are screened before the embedding. A unit whose strongest partners are
  no more correlated than noise allows (``partner_snr``, against noise
  measured from the two halves of the data) has no place on any ring, and
  left in, such units wreck Isomap: each links a few random units into
  shortcuts across the ring, and patched as the MATLAB patches a disconnected
  graph, their huge distances claim the leading MDS dimensions. On synthetic
  data with 24 ring units among 200 independent ones, the MATLAB pipeline
  finds no ring at all; screened, it places the 24 within 7 degrees.
- Membership is ``coupling``, how strongly a unit's correlations rise towards
  its ring neighbours and fall towards the units opposite it, not the radius.
  Units off the ring land inside it and outside it alike, so radius alone
  does not separate them (AUC 0.45-0.7 on the synthetic mixes; coupling 1.0).
- The population vector uses rates z-scored per unit and weighted by
  coupling. On raw rates with equal weights, the fastest-firing units and the
  units off the ring decide the decoded angle.
- The ring is fitted on training bins only, and its arbitrary rotation and
  handedness are fitted to heading on those same bins, so the held-out decode
  is scored on data that set nothing.
- Ring positions are compared with Fisher and Lee's circular correlation.
  The Jammalamadaka coefficient that :func:`spikeshpc.decoder.circular_correlation`
  computes needs each variable's mean direction, and units spread evenly
  round a ring have none: it can come out anywhere in [-1, 1] for a perfect
  match.

Bins, masks and the train/test split are the decoder's
(:func:`spikeshpc.decoder.prepare_decoder_data`,
:func:`~spikeshpc.decoder.split_train_test`), so with the same settings
:func:`run_ring` holds out exactly the bins
:func:`~spikeshpc.decoder.run_decoder` does. Decodes come back as
:class:`~spikeshpc.decoder.Decoded`, for ``show_decoded``, ``plot_error``,
``metrics_by_group`` and ``find_turns``; their ``posterior_max`` holds the
population vector's normalized length, the ring's analogue of confidence.

Typical use, with the GOOD and MUA units in the depth band of the
head-direction structure::

    ring_run = run_ring(analyzer, band_unit_ids, heading_deg, shutter_close_times,
                        scoring.intervals, interval_mask=hd.interval_mask)
    print(ring_run.summary())
    plot_ring(ring_run.ring)
    print(ring_vs_tuning(ring_run.ring, hd).summary())

Two caveats. Units with partners of their own -- an assembly, another
structure the probe crosses -- pass the screen and take the embedding over, so
give the ring the units of the head-direction structure, by depth. On
session7 a ring of all 285 GOOD and MUA units decoded wake at 70 degrees and
made a member of 1 of the 33 tuned units; the 51 units in the band the tuned
units occupy decoded at 18 degrees, with 28 of its 30 tuned units members.
The eigenvalues tell the two apart: a ring gives two nearly equal leading ones
well clear of the third, noise a flat run of them. And the shoulder holds
heading persistence as well as slow drift: while the animal holds one heading
for longer than the shoulder, the baseline approaches the peak, so pairs are
scored mostly from movement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.csgraph import connected_components, shortest_path

from .decoder import (
    Decoded,
    DecoderData,
    ShuffleTest,
    _judge,
    _runs,
    _unit_positions,
    _wake_mask,
    circular_difference,
    decoding_metrics,
    prepare_decoder_data,
    split_train_test,
    state_interval_mask,
)

__all__ = [
    "AngleComparison",
    "Correlations",
    "RingAlignment",
    "RingComparison",
    "RingModel",
    "RingRun",
    "RingTuning",
    "align_angles",
    "circular_r",
    "compare_angles",
    "compare_rings",
    "control_band",
    "decode_ring",
    "fit_ring",
    "pair_correlogram",
    "pair_decodes",
    "pairwise_correlations",
    "plot_correlograms",
    "plot_ring",
    "plot_ring_comparison",
    "plot_ring_vs_tuning",
    "ring_vs_tuning",
    "run_ring",
]

_CHUNK_UNITS = 32  # units smoothed or filtered at a time, to bound memory
_GRAM_ROWS = 1 << 16  # bins per float64 block of a matrix product

# what fit_ring calls each candidate unit, in ring.screen
ON_RING = "ring"
NO_PARTNERS = "no partners"
SILENT = "silent"


# ── helpers ──────────────────────────────────────────────────────────────
def _check_mask(data: DecoderData, mask, name: str = "mask") -> np.ndarray:
    """Check that a mask has one entry per bin.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    mask : array-like of bool
        Mask to check.
    name : str, default "mask"
        Name of the argument, for messages.

    Returns
    -------
    numpy.ndarray of bool
        The mask.

    Raises
    ------
    ValueError
        If the mask is the wrong shape.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (data.n_bins,):
        raise ValueError(
            f"{name} has {mask.shape} entries but there are {data.n_bins} bins"
        )
    return mask


def _span(index: np.ndarray, margin: int, n: int) -> tuple[int, int]:
    """Find the rows holding `index` with `margin` bins to spare on either side.

    Parameters
    ----------
    index : numpy.ndarray
        Sorted bin indices.
    margin : int
        Bins to keep either side.
    n : int
        Number of bins.

    Returns
    -------
    lo, hi : int
        Rows ``lo:hi``, clipped to ``0:n``.
    """
    return max(int(index[0]) - margin, 0), min(int(index[-1]) + margin + 1, n)


def _smoothed_rates(data: DecoderData, mask, sigma_s: float, columns) -> np.ndarray:
    """Gaussian-smoothed firing rate of some units, in the masked bins.

    The smoothing runs over the full grid and only then is the mask applied,
    so a bin at the edge of the mask is smoothed with the spikes on both sides
    of the edge rather than with zeros.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    mask : numpy.ndarray of bool
        Bins wanted.
    sigma_s : float
        Width of the Gaussian, in seconds; 0 leaves the rates unsmoothed.
    columns : array-like of int
        Columns of ``data.counts`` wanted.

    Returns
    -------
    numpy.ndarray
        Rate in Hz, shape (masked bins, len(columns)), float32.
    """
    index = np.flatnonzero(mask)
    columns = np.asarray(columns, dtype=int)
    rates = np.empty((index.size, columns.size), dtype=np.float32)
    if index.size == 0:
        return rates
    sigma_bins = sigma_s / data.bin_s
    margin = int(np.ceil(5.0 * sigma_bins)) + 1 if sigma_bins > 0 else 0
    lo, hi = _span(index, margin, data.n_bins)
    rows = index - lo
    duration = data.duration_s[lo:hi, None]
    for start in range(0, columns.size, _CHUNK_UNITS):
        cols = columns[start : start + _CHUNK_UNITS]
        rate = data.counts[lo:hi, cols] / duration
        if sigma_bins > 0:
            rate = gaussian_filter1d(
                rate, sigma_bins, axis=0, mode="constant", truncate=5.0
            )
        rates[:, start : start + cols.size] = rate[rows]
    return rates


def _gram(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute ``a.T @ b``, summed in float64 a block of rows at a time.

    Parameters
    ----------
    a, b : numpy.ndarray
        Arrays with the same number of rows.

    Returns
    -------
    numpy.ndarray
        Shape (a.shape[1], b.shape[1]).
    """
    out = np.zeros((a.shape[1], b.shape[1]))
    for start in range(0, a.shape[0], _GRAM_ROWS):
        rows = slice(start, start + _GRAM_ROWS)
        block = np.asarray(a[rows], dtype=np.float64)
        out += block.T @ np.asarray(b[rows], dtype=np.float64)
    return out


def _pair_counts(mask: np.ndarray, max_lag: int) -> np.ndarray:
    """Count the pairs of masked bins that lie each lag apart.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        Mask over consecutive bins.
    max_lag : int
        Largest lag, in bins.

    Returns
    -------
    numpy.ndarray
        Count at each lag from ``-max_lag`` to ``max_lag``.
    """
    m = np.asarray(mask, dtype=float)
    n = m.size
    full = signal.fftconvolve(m, m[::-1], mode="full")  # lag 0 at index n - 1
    lags = np.arange(-max_lag, max_lag + 1)
    counts = np.zeros(lags.size)
    reachable = np.abs(lags) < n
    counts[reachable] = np.rint(full[n - 1 + lags[reachable]])
    return counts


def _pearson(x, y) -> float:
    """Pearson's r over the entries where both are finite.

    Parameters
    ----------
    x, y : array-like
        Values to correlate.

    Returns
    -------
    float
        r, or NaN with fewer than three finite pairs or no spread.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    x = x[finite] - x[finite].mean()
    y = y[finite] - y[finite].mean()
    denominator = np.sqrt(np.sum(x**2) * np.sum(y**2))
    return float(np.sum(x * y) / denominator) if denominator > 0 else float("nan")


def _auc(positive, negative) -> float:
    """Area under the ROC curve: how often a positive outranks a negative.

    Parameters
    ----------
    positive, negative : array-like
        Scores of the two groups.

    Returns
    -------
    float
        AUC in [0, 1], ties counting half; NaN if a group is empty.
    """
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not (positive.size and negative.size):
        return float("nan")
    ranks = stats.rankdata(np.r_[positive, negative])
    u = ranks[: positive.size].sum() - positive.size * (positive.size + 1) / 2
    return float(u / (positive.size * negative.size))


def _number(value: float) -> str:
    """Format a statistic, or say it could not be computed.

    Parameters
    ----------
    value : float
        The statistic.

    Returns
    -------
    str
        Two decimals, or "n/a" for NaN (an AUC with an empty group, say).
    """
    return f"{value:.2f}" if np.isfinite(value) else "n/a"


def circular_r(a_deg, b_deg) -> float:
    """Fisher and Lee's circular correlation, in [-1, 1].

    The mean over pairs of observations of ``sin(a_i - a_j) sin(b_i - b_j)``,
    normalized: +1 when ``b`` is ``a`` turned by any angle, -1 when it is
    ``a`` mirrored and turned, near 0 when unrelated. Built from differences
    within each variable, it needs neither variable's mean direction, so --
    unlike :func:`spikeshpc.decoder.circular_correlation` -- it is well
    defined for angles spread evenly round the circle, which units on a ring
    are.

    Parameters
    ----------
    a_deg, b_deg : array-like
        Paired angles, in degrees; pairs with a NaN are dropped.

    Returns
    -------
    float
        The coefficient; NaN with fewer than three pairs or no spread.
    """
    a = np.deg2rad(np.asarray(a_deg, dtype=float))
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    a, b = a[finite], b[finite]
    n = a.size
    # sum over all ordered pairs of sin(a_i - a_j) sin(b_i - b_j), expanded
    # (Fisher 1993, eq. 6.36), so no n x n matrix is formed
    cross = 4.0 * (
        np.sum(np.cos(a) * np.cos(b)) * np.sum(np.sin(a) * np.sin(b))
        - np.sum(np.cos(a) * np.sin(b)) * np.sum(np.sin(a) * np.cos(b))
    )
    spread_a = n**2 - np.abs(np.exp(2j * a).sum()) ** 2
    spread_b = n**2 - np.abs(np.exp(2j * b).sum()) ** 2
    denominator = np.sqrt(spread_a * spread_b)
    return float(cross / denominator) if denominator > 0 else float("nan")


# ── pairwise correlations ────────────────────────────────────────────────
@dataclass
class Correlations:
    """Correlations of smoothed rates between every pair of units, within a mask.

    ``metric`` is what the ring is built from: the zero-lag correlation minus
    the mean correlation over the shoulder lags.

    Attributes
    ----------
    unit_ids : numpy.ndarray
        Units, in the order of the rows and columns.
    zero_lag : numpy.ndarray
        Pearson r at lag 0, shape (n_units, n_units).
    baseline : numpy.ndarray
        Mean r over the shoulder lags, same shape.
    rate_mean_hz, rate_sd_hz : numpy.ndarray
        Mean and SD of each unit's smoothed rate over the masked bins.
    time_s : float
        Masked time, in seconds.
    n_lags : int
        Shoulder lags averaged into `baseline`.
    """

    unit_ids: np.ndarray
    zero_lag: np.ndarray
    baseline: np.ndarray
    rate_mean_hz: np.ndarray
    rate_sd_hz: np.ndarray
    time_s: float
    n_lags: int

    @property
    def metric(self) -> np.ndarray:
        return self.zero_lag - self.baseline


def pairwise_correlations(
    data: DecoderData,
    mask,
    unit_ids=None,
    smooth_sigma_s: float = 0.05,
    max_lag_s: float = 10.0,
    baseline_start_s: float = 5.0,
) -> Correlations:
    """Correlate the smoothed rates of every pair of units, at lag 0 and on the shoulder.

    Rates are binned, smoothed with a Gaussian and centred on their mean over
    the masked bins. The correlation at lag l is the covariance of unit i at
    time t + l with unit j at time t, summed over the pairs of bins that are
    both masked and divided by how many such pairs there are, then scaled by
    the two units' standard deviations. A gap in the mask contributes nothing
    to any lag. ``baseline`` averages that correlation over the lags with
    ``baseline_start_s <= |l| <= max_lag_s`` -- the MATLAB's shoulders --
    computed for all pairs at once as one FFT filter per unit and one matrix
    product.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    mask : array-like of bool
        Bins to correlate within.
    unit_ids : sequence, optional
        Units to correlate; all of `data`'s by default.
    smooth_sigma_s : float, default 0.05
        Width of the Gaussian smoothing, in seconds.
    max_lag_s : float, default 10.0
        Outer edge of the shoulder, in seconds.
    baseline_start_s : float, default 5.0
        Inner edge of the shoulder, in seconds.

    Returns
    -------
    Correlations
        The matrices. A unit that never fires near the masked bins has no
        variance, and NaN in its row and column.

    Raises
    ------
    ValueError
        If the mask keeps no bins, the shoulder is empty, or no two masked
        bins are a shoulder lag apart.
    """
    mask = _check_mask(data, mask)
    if not mask.any():
        raise ValueError("mask keeps no bins to correlate")
    unit_ids = np.asarray(data.unit_ids if unit_ids is None else list(unit_ids))
    columns = _unit_positions(data.unit_ids, unit_ids)
    max_lag = int(round(max_lag_s / data.bin_s))
    start = int(round(baseline_start_s / data.bin_s))
    if not 0 < start <= max_lag:
        raise ValueError(
            f"need 0 < baseline_start_s <= max_lag_s, got {baseline_start_s} and "
            f"{max_lag_s} s ({start} and {max_lag} bins of {data.bin_s:.4f} s)"
        )

    x = _smoothed_rates(data, mask, smooth_sigma_s, columns)
    mean = x.mean(axis=0, dtype=np.float64)
    x -= mean.astype(np.float32)
    zero_cov = _gram(x, x) / x.shape[0]

    index = np.flatnonzero(mask)
    lo, hi = _span(index, max_lag, data.n_bins)
    local = index - lo
    counts = _pair_counts(mask[lo:hi], max_lag)
    lags = np.arange(-max_lag, max_lag + 1)
    shoulder = (np.abs(lags) >= start) & (counts > 0)
    if not shoulder.any():
        raise ValueError(
            f"no two masked bins are {baseline_start_s}-{max_lag_s} s apart, so "
            "there is no baseline to subtract"
        )
    # each lag's sum divided by its own number of pairs, then averaged over lags
    kernel = np.zeros(lags.size)
    kernel[shoulder] = 1.0 / (counts[shoulder] * shoulder.sum())

    base_cov = np.empty_like(zero_cov)
    for first in range(0, columns.size, _CHUNK_UNITS):
        cols = slice(first, first + _CHUNK_UNITS)
        block = np.zeros((hi - lo, x[:, cols].shape[1]))
        block[local] = x[:, cols]
        # out[t] = sum_l kernel(l) x(t + l): the kernel is symmetric, so
        # convolving and correlating are the same thing
        filtered = signal.oaconvolve(block, kernel[:, None], mode="same", axes=0)
        base_cov[:, cols] = _gram(x, filtered[local])
    base_cov = (base_cov + base_cov.T) / 2

    sd = np.sqrt(np.clip(np.diag(zero_cov), 0.0, None))
    scale = np.outer(sd, sd)
    with np.errstate(invalid="ignore", divide="ignore"):
        zero_lag = np.where(scale > 0, zero_cov / scale, np.nan)
        baseline = np.where(scale > 0, base_cov / scale, np.nan)
    return Correlations(
        unit_ids=unit_ids,
        zero_lag=zero_lag,
        baseline=baseline,
        rate_mean_hz=mean,
        rate_sd_hz=sd,
        time_s=float(data.duration_s[mask].sum()),
        n_lags=int(shoulder.sum()),
    )


def pair_correlogram(
    data: DecoderData,
    mask,
    unit_a,
    unit_b,
    smooth_sigma_s: float = 0.05,
    max_lag_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlate one pair of units at every lag, as :func:`pairwise_correlations` does.

    For looking at what the baseline is subtracting: MATLAB's Fig 1.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    mask : array-like of bool
        Bins to correlate within.
    unit_a, unit_b
        The pair; positive lags are `unit_a` later than `unit_b`.
    smooth_sigma_s : float, default 0.05
        Width of the Gaussian smoothing, in seconds.
    max_lag_s : float, default 10.0
        Largest lag, in seconds.

    Returns
    -------
    lags_s : numpy.ndarray
        Lags, in seconds.
    r : numpy.ndarray
        Correlation at each lag; NaN where no two masked bins are that far
        apart.
    """
    mask = _check_mask(data, mask)
    columns = _unit_positions(data.unit_ids, [unit_a, unit_b])
    x = _smoothed_rates(data, mask, smooth_sigma_s, columns).astype(np.float64)
    x -= x.mean(axis=0)
    sd = np.sqrt((x**2).mean(axis=0))
    max_lag = int(round(max_lag_s / data.bin_s))
    index = np.flatnonzero(mask)
    lo, hi = _span(index, max_lag, data.n_bins)
    full = np.zeros((hi - lo, 2))
    full[index - lo] = x
    n = hi - lo
    cross = signal.fftconvolve(full[:, 0], full[::-1, 1], mode="full")
    lags = np.arange(-max_lag, max_lag + 1)
    counts = _pair_counts(mask[lo:hi], max_lag)
    r = np.full(lags.size, np.nan)
    ok = (counts > 0) & (np.abs(lags) < n)
    if sd.prod() > 0:
        r[ok] = cross[n - 1 + lags[ok]] / counts[ok] / sd.prod()
    return lags * data.bin_s, r


# ── the ring ─────────────────────────────────────────────────────────────
@dataclass
class RingModel:
    """Units placed on a ring by their correlations; see :func:`fit_ring`.

    Angles are in the ring's own frame, whose rotation and handedness are
    arbitrary: :func:`align_angles` maps them onto heading or onto another
    ring.

    Attributes
    ----------
    unit_ids : numpy.ndarray
        Units on the ring, in the order of every per-unit array.
    angle_deg : numpy.ndarray
        Each unit's position on the ring, in [0, 360).
    radius : numpy.ndarray
        Its distance from the centre of the embedding -- the MATLAB's "tuning
        strength". Units off the ring land outside it as well as inside, so it
        is not a measure of membership; `coupling` is.
    coupling : numpy.ndarray
        How strongly the unit's correlations (z) rise towards its neighbours
        on the ring and fall towards the units opposite: the slope of its row
        of `corr_z` on the cosine of the others' angular distance from it.
        Near 1 or more for a unit of the ring, near 0 for one that is not.
    embedding : numpy.ndarray
        The 2-D Isomap coordinates, centred, shape (n_units, 2).
    corr : numpy.ndarray
        Zero-lag minus shoulder correlation, NaN on the diagonal.
    corr_z : numpy.ndarray
        `corr` z-scored over the ring's pairs, NaN on the diagonal.
    zero_lag, baseline : numpy.ndarray
        The two terms of `corr`.
    geodesic : numpy.ndarray
        Graph distances the embedding was made from.
    eigenvalues : numpy.ndarray
        Of the MDS Gram matrix, largest first. A ring gives two nearly equal
        leading ones (a cosine and a sine) well clear of the rest; units with
        no common structure give a flat spectrum, so the second alone being
        close to the first proves nothing.
    stress : float
        Mismatch between the graph distances and the 2-D ones, 0 to 1.
    n_components : int
        Pieces the neighbour graph fell into; 1 is connected.
    consistency_r : float
        Pearson r over all pairs between `corr_z` and the cosine of their
        angular distance (MATLAB's Fig 1d as one number).
    split_half_r : float
        Pearson r over the ring's pairs between `corr` from the first and the
        second half of the fitting time: how reproducible the structure is.
    partner_snr : numpy.ndarray
        Mean of the unit's `k_neighbors` strongest `corr` above the median
        pair, in units of `noise_sd`. Every unit on the ring passed the
        screen on it.
    noise_sd : float
        Standard deviation of a pair's `corr` from sampling alone, from the
        two halves.
    rate_mean_hz, rate_sd_hz : numpy.ndarray
        Each unit's smoothed rate over the fitting bins.
    fit_time_s : float
        Fitting time, in seconds.
    screen : pandas.DataFrame
        Every candidate unit, with ``rate_hz``, ``partner_snr`` and
        ``status``: "ring", "no partners" (screened out) or "silent".
    params : dict
        Settings the ring was fitted with.
    """

    unit_ids: np.ndarray
    angle_deg: np.ndarray
    radius: np.ndarray
    coupling: np.ndarray
    embedding: np.ndarray
    corr: np.ndarray
    corr_z: np.ndarray
    zero_lag: np.ndarray
    baseline: np.ndarray
    geodesic: np.ndarray
    eigenvalues: np.ndarray
    stress: float
    n_components: int
    consistency_r: float
    split_half_r: float
    partner_snr: np.ndarray
    noise_sd: float
    rate_mean_hz: np.ndarray
    rate_sd_hz: np.ndarray
    fit_time_s: float
    screen: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: dict = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def radius_z(self) -> np.ndarray:
        spread = self.radius.std()
        if not spread > 0:
            return np.zeros_like(self.radius)
        return (self.radius - self.radius.mean()) / spread

    @property
    def eigenvalue_ratio(self) -> float:
        """Get the second eigenvalue over the first: near 1 for a ring.

        Returns
        -------
        float
            The ratio.
        """
        return float(self.eigenvalues[1] / self.eigenvalues[0])

    @property
    def excluded_unit_ids(self) -> np.ndarray:
        """Get the candidates left off the ring, silent or without partners.

        Returns
        -------
        numpy.ndarray
            Unit ids.
        """
        if self.screen.empty:
            return np.array([])
        return self.screen.index[self.screen["status"] != ON_RING].to_numpy()

    @property
    def order(self) -> np.ndarray:
        """Get the units in order around the ring.

        Returns
        -------
        numpy.ndarray
            Unit ids sorted by `angle_deg`.
        """
        return self.unit_ids[np.argsort(self.angle_deg, kind="stable")]

    def angle_of(self, unit_ids) -> np.ndarray:
        """Get some units' positions on the ring.

        Parameters
        ----------
        unit_ids : sequence
            Units to look up.

        Returns
        -------
        numpy.ndarray
            Angles, in degrees.
        """
        return self.angle_deg[_unit_positions(self.unit_ids, list(unit_ids))]

    def members(self, min_coupling: float = 0.5) -> np.ndarray:
        """Get the units whose co-firing is organised by the ring.

        Parameters
        ----------
        min_coupling : float, default 0.5
            Smallest `coupling` that counts.

        Returns
        -------
        numpy.ndarray
            Unit ids, in ring order.
        """
        keep = np.nan_to_num(self.coupling, nan=-np.inf) >= min_coupling
        units = self.unit_ids[keep]
        return units[np.argsort(self.angle_deg[keep], kind="stable")]

    def __repr__(self) -> str:
        return (
            f"RingModel({self.n_units} units, eigenvalue ratio "
            f"{self.eigenvalue_ratio:.2f}, stress {self.stress:.2f}, consistency r "
            f"{self.consistency_r:.2f}, split-half r {self.split_half_r:.2f}, "
            f"fitted on {self.fit_time_s:.0f}s)"
        )


def _classical_mds(
    distance: np.ndarray, n_dims: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Embed points from their distances by classical MDS (MATLAB's cmdscale).

    Parameters
    ----------
    distance : numpy.ndarray
        Symmetric distances, shape (n, n).
    n_dims : int, default 2
        Dimensions kept.

    Returns
    -------
    coordinates : numpy.ndarray
        Shape (n, n_dims). Each axis is signed so that its largest coordinate
        is positive, which makes the result deterministic.
    eigenvalues : numpy.ndarray
        Of the double-centred Gram matrix, largest first.

    Raises
    ------
    ValueError
        If fewer than `n_dims` eigenvalues are positive.
    """
    n = len(distance)
    centering = np.eye(n) - np.full((n, n), 1.0 / n)
    gram = -0.5 * centering @ (distance**2) @ centering
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    signs = np.sign(vectors[np.abs(vectors).argmax(axis=0), np.arange(n)])
    vectors = vectors * np.where(signs == 0, 1.0, signs)
    if not (values[:n_dims] > 0).all():
        raise ValueError(
            f"fewer than {n_dims} positive eigenvalues: the distances have no "
            f"{n_dims}-D structure to embed"
        )
    return vectors[:, :n_dims] * np.sqrt(values[:n_dims]), values


def _ring_coupling(
    corr_z: np.ndarray, angle_deg: np.ndarray
) -> tuple[np.ndarray, float]:
    """Measure how much co-firing is organised around the ring, per unit and overall.

    Parameters
    ----------
    corr_z : numpy.ndarray
        z-scored pair metric, NaN on the diagonal.
    angle_deg : numpy.ndarray
        Positions on the ring.

    Returns
    -------
    coupling : numpy.ndarray
        Per unit, the least-squares slope of its row of `corr_z` on the cosine
        of the others' angular distance from it.
    consistency_r : float
        Pearson r between `corr_z` and that cosine over every pair at once.
    """
    n = len(angle_deg)
    cosine = np.cos(np.deg2rad(angle_deg[:, None] - angle_deg[None, :]))
    coupling = np.full(n, np.nan)
    for i in range(n):
        others = np.arange(n) != i
        x = cosine[i, others] - cosine[i, others].mean()
        y = corr_z[i, others]
        if np.sum(x**2) > 0:
            coupling[i] = np.sum(x * (y - y.mean())) / np.sum(x**2)
    upper = np.triu_indices(n, 1)
    return coupling, _pearson(corr_z[upper], cosine[upper])


def _embed(metric: np.ndarray, k_neighbors: int, stacklevel: int = 3) -> dict:
    """Put units on a ring from their pair metric: z-score, distance, graph, Isomap.

    Parameters
    ----------
    metric : numpy.ndarray
        Pair metric, shape (n, n), finite off the diagonal.
    k_neighbors : int
        Neighbours per unit in the graph; two units are linked if either is
        among the other's nearest, as in the MATLAB.
    stacklevel : int, default 3
        Of the warning about a disconnected graph.

    Returns
    -------
    dict
        ``corr_z``, ``geodesic``, ``embedding``, ``eigenvalues``,
        ``angle_deg``, ``radius``, ``stress``, ``n_components``,
        ``coupling``, ``consistency_r``.

    Raises
    ------
    ValueError
        If the metric is not finite or has no spread, or the units are too
        few for `k_neighbors`.
    """
    n = metric.shape[0]
    if not 0 < k_neighbors < n:
        raise ValueError(
            f"k_neighbors={k_neighbors} needs more than {k_neighbors} units; there are {n}"
        )
    off = ~np.eye(n, dtype=bool)
    values = metric[off]
    if not np.isfinite(values).all():
        raise ValueError("the pair metric is not finite for every pair")
    spread = values.std(ddof=1)
    if not spread > 0:
        raise ValueError(
            "every pair is equally correlated: there is no structure to embed"
        )

    corr_z = (metric - values.mean()) / spread
    np.fill_diagonal(corr_z, np.nan)
    distance = np.nanmax(corr_z) - corr_z
    np.fill_diagonal(distance, 0.0)
    # the most correlated pair is at distance 0, and a graph cannot hold a
    # zero-length edge
    distance[off] = np.maximum(distance[off], 1e-9 * max(float(distance.max()), 1.0))

    nearest_idx = np.argsort(np.where(off, distance, np.inf), axis=1, kind="stable")
    nearest = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k_neighbors)
    nearest[rows, nearest_idx[:, :k_neighbors].ravel()] = True
    graph = np.where(nearest | nearest.T, distance, 0.0)
    n_components, labels = connected_components(graph, directed=False)
    geodesic = shortest_path(graph, method="D", directed=False)
    if n_components > 1:
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        warnings.warn(
            f"the {k_neighbors}-nearest-neighbour graph falls into {n_components} "
            f"pieces (sizes {sizes}); distances between pieces are set to twice "
            "the longest within one, as the MATLAB does. Raise k_neighbors to "
            "join them.",
            stacklevel=stacklevel,
        )
        finite = np.isfinite(geodesic)
        geodesic[~finite] = 2.0 * geodesic[finite].max()

    embedding, eigenvalues = _classical_mds(geodesic, 2)
    embedding = embedding - embedding.mean(axis=0)
    angle_deg = np.rad2deg(np.arctan2(embedding[:, 1], embedding[:, 0])) % 360.0
    radius = np.hypot(embedding[:, 0], embedding[:, 1])
    flat = np.sqrt(((embedding[:, None, :] - embedding[None, :, :]) ** 2).sum(axis=-1))
    stress = float(np.sqrt(np.sum((geodesic - flat) ** 2) / np.sum(geodesic**2)))
    coupling, consistency_r = _ring_coupling(corr_z, angle_deg)
    return {
        "corr_z": corr_z,
        "geodesic": geodesic,
        "embedding": embedding,
        "eigenvalues": eigenvalues,
        "angle_deg": angle_deg,
        "radius": radius,
        "stress": stress,
        "n_components": int(n_components),
        "coupling": coupling,
        "consistency_r": consistency_r,
    }


def _masked_rates(
    data: DecoderData, mask: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    """Mean firing rate of some units over the masked bins.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    mask : numpy.ndarray of bool
        Bins to count over.
    columns : numpy.ndarray of int
        Columns of ``data.counts``.

    Returns
    -------
    numpy.ndarray
        Rate in Hz per column.
    """
    index = np.flatnonzero(mask)
    totals = np.zeros(columns.size)
    for start in range(0, index.size, _GRAM_ROWS):
        totals += data.counts[index[start : start + _GRAM_ROWS]][:, columns].sum(axis=0)
    return totals / data.duration_s[index].sum()


def _partner_snr(metric: np.ndarray, noise_sd: float, k: int) -> np.ndarray:
    """Score each unit by how far its strongest partners stand above noise.

    Parameters
    ----------
    metric : numpy.ndarray
        Pair metric, NaN on the diagonal.
    noise_sd : float
        Standard deviation of a pair's metric from sampling alone.
    k : int
        Partners averaged.

    Returns
    -------
    numpy.ndarray
        Mean of each unit's `k` largest values above the median pair, over
        `noise_sd`.
    """
    off = ~np.eye(len(metric), dtype=bool)
    centre = np.nanmedian(metric[off])
    values = np.where(off, np.nan_to_num(metric, nan=-np.inf), -np.inf)
    strongest = -np.sort(-values, axis=1)
    top = strongest[:, : min(k, len(metric) - 1)].mean(axis=1)
    return (top - centre) / noise_sd if noise_sd > 0 else np.full(len(metric), np.nan)


def fit_ring(
    data: DecoderData,
    mask,
    unit_ids=None,
    smooth_sigma_s: float = 0.05,
    max_lag_s: float = 10.0,
    baseline_start_s: float = 5.0,
    k_neighbors: int = 8,
    min_rate_hz: float = 0.1,
    min_partner_snr: float | None = 4.0,
    verbose: bool = True,
) -> RingModel:
    """Place units on a ring by their correlations in the bins under `mask`.

    No heading is used: only the spikes. The pair metric is computed three
    times -- on all of the fitting time and on each half of it -- and the
    halves give both the noise the screen measures against and
    ``split_half_r``. See the module docstring for the steps.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    mask : array-like of bool
        Bins to fit on: training wake, say, or REM.
    unit_ids : sequence, optional
        Candidate units; all of `data`'s by default.
    smooth_sigma_s : float, default 0.05
        Width of the Gaussian smoothing of the rates, in seconds.
    max_lag_s : float, default 10.0
        Outer edge of the shoulder, in seconds. The MATLAB uses 5 for sleep.
    baseline_start_s : float, default 5.0
        Inner edge of the shoulder, in seconds. The MATLAB uses 2.5 for sleep.
    k_neighbors : int, default 8
        Neighbours per unit in the Isomap graph, and partners averaged into
        ``partner_snr``.
    min_rate_hz : float, default 0.1
        Units firing less than this over the fitting bins are left off as
        silent.
    min_partner_snr : float or None, default 4.0
        Units whose ``partner_snr`` falls below this are left off before the
        embedding. On synthetic data, units with no partners score under 4
        even among 200 of them, and weak head-direction units (3 Hz peaks)
        over 5. None places every unit that fires, as the MATLAB does.
    verbose : bool, default True
        Print a summary.

    Returns
    -------
    RingModel
        The ring.

    Raises
    ------
    ValueError
        If the mask keeps no bins, too few units are left to place, or their
        correlations have no structure to embed.
    """
    mask = _check_mask(data, mask)
    if not mask.any():
        raise ValueError("mask keeps no bins to fit the ring on")
    candidates = np.asarray(data.unit_ids if unit_ids is None else list(unit_ids))
    rate = _masked_rates(data, mask, _unit_positions(data.unit_ids, candidates))
    active = candidates[rate >= min_rate_hz]
    status = np.where(rate >= min_rate_hz, ON_RING, SILENT).astype(object)
    if active.size < k_neighbors + 2:
        raise ValueError(
            f"{active.size} of {candidates.size} units fire at {min_rate_hz} Hz or more "
            f"in the fitting bins, too few for k_neighbors={k_neighbors}: need at least "
            f"{k_neighbors + 2}"
        )

    settings = dict(
        smooth_sigma_s=smooth_sigma_s,
        max_lag_s=max_lag_s,
        baseline_start_s=baseline_start_s,
    )
    correlations = pairwise_correlations(data, mask, active, **settings)
    metric = correlations.metric
    np.fill_diagonal(metric, np.nan)

    # the two halves of the fitting time: their difference is the noise
    index = np.flatnonzero(mask)
    elapsed = np.cumsum(data.duration_s[index])
    first = np.zeros(data.n_bins, dtype=bool)
    first[index[elapsed <= elapsed[-1] / 2]] = True
    halves = None
    try:
        halves = [
            pairwise_correlations(data, half, active, **settings).metric
            for half in (first, mask & ~first)
        ]
    except ValueError as error:
        warnings.warn(
            f"the halves of the fitting time could not be correlated ({error}), so "
            "there is no noise to screen against and no split-half reliability",
            stacklevel=2,
        )
    noise_sd = float("nan")
    snr = np.full(active.size, np.nan)
    if halves is not None:
        off = ~np.eye(active.size, dtype=bool)
        difference = (halves[0] - halves[1])[off]
        difference = difference[np.isfinite(difference)]
        # robust SD of a half's noise, halved: the whole is the mean of two halves
        noise_sd = float(
            1.4826 * np.median(np.abs(difference - np.median(difference))) / 2
        )
        snr = _partner_snr(metric, noise_sd, k_neighbors)

    on_ring = np.ones(active.size, dtype=bool)
    if min_partner_snr is not None and np.isfinite(snr).all():
        on_ring = snr >= min_partner_snr
    status[np.isin(candidates, active[~on_ring])] = NO_PARTNERS
    ring_units = active[on_ring]
    if ring_units.size < k_neighbors + 2:
        raise ValueError(
            f"{ring_units.size} units have partners beyond noise (partner SNR >= "
            f"{min_partner_snr}), too few for k_neighbors={k_neighbors}: need at "
            f"least {k_neighbors + 2}. Lower min_partner_snr, or fit on more time."
        )
    keep = np.flatnonzero(on_ring)
    sub = np.ix_(keep, keep)
    ring = _embed(metric[sub], k_neighbors)

    split_half_r = float("nan")
    if halves is not None:
        upper = np.triu_indices(keep.size, 1)
        split_half_r = _pearson(halves[0][sub][upper], halves[1][sub][upper])

    snr_of = dict(zip(active.tolist(), snr.tolist()))
    screen = pd.DataFrame(
        {
            "rate_hz": rate,
            "partner_snr": [snr_of.get(u, np.nan) for u in candidates.tolist()],
            "status": status,
        },
        index=pd.Index(candidates, name="unit_id"),
    )
    model = RingModel(
        unit_ids=ring_units,
        angle_deg=ring["angle_deg"],
        radius=ring["radius"],
        coupling=ring["coupling"],
        embedding=ring["embedding"],
        corr=metric[sub],
        corr_z=ring["corr_z"],
        zero_lag=correlations.zero_lag[sub],
        baseline=correlations.baseline[sub],
        geodesic=ring["geodesic"],
        eigenvalues=ring["eigenvalues"],
        stress=ring["stress"],
        n_components=ring["n_components"],
        consistency_r=ring["consistency_r"],
        split_half_r=split_half_r,
        partner_snr=snr[keep],
        noise_sd=noise_sd,
        rate_mean_hz=correlations.rate_mean_hz[keep],
        rate_sd_hz=correlations.rate_sd_hz[keep],
        fit_time_s=correlations.time_s,
        screen=screen,
        params=dict(
            settings,
            k_neighbors=k_neighbors,
            min_rate_hz=min_rate_hz,
            min_partner_snr=min_partner_snr,
            bin_s=data.bin_s,
        ),
    )
    if verbose:
        n_silent = int((status == SILENT).sum())
        n_alone = int((status == NO_PARTNERS).sum())
        top = model.eigenvalues[:3] / model.eigenvalues[0]
        print(
            f"  ring: {model.n_units} of {candidates.size} units, fitted on "
            f"{model.fit_time_s:.0f}s ({n_silent} silent; {n_alone} without partners "
            f"beyond noise)"
        )
        print(
            f"    eigenvalues 1 : {top[1]:.2f} : {top[2]:.2f}, stress {model.stress:.2f}, "
            f"consistency r {model.consistency_r:.2f}, split-half r {split_half_r:.2f}, "
            f"{len(model.members())} members"
        )
    return model


# ── alignment ────────────────────────────────────────────────────────────
@dataclass
class RingAlignment:
    """A reflection and a rotation taking angles onto a reference.

    ``aligned = (-angle if flip else angle) + offset_deg``, wrapped to [0, 360).

    Attributes
    ----------
    flip : bool
        Whether the angles are mirrored first.
    offset_deg : float
        Rotation then added, in degrees.
    resultant : float
        Mean resultant length of the differences after alignment: 1 when the
        angles match up to this transform exactly, near 0 when unrelated.
    resultant_other : float
        The same for the other handedness: how clearly the choice was made.
    n : int
        Angles it was fitted on.
    """

    flip: bool
    offset_deg: float
    resultant: float
    resultant_other: float
    n: int

    @property
    def margin(self) -> float:
        """Get how much better the chosen handedness fitted than the other.

        Returns
        -------
        float
            Difference of the two resultant lengths.
        """
        return self.resultant - self.resultant_other

    def apply(self, angle_deg) -> np.ndarray:
        """Map angles onto the reference.

        Parameters
        ----------
        angle_deg : array-like
            Angles in the original frame, in degrees.

        Returns
        -------
        numpy.ndarray
            Aligned angles, in [0, 360).
        """
        sign = -1.0 if self.flip else 1.0
        return (sign * np.asarray(angle_deg, dtype=float) + self.offset_deg) % 360.0

    def invert(self, angle_deg) -> np.ndarray:
        """Map aligned angles back to the original frame.

        Parameters
        ----------
        angle_deg : array-like
            Aligned angles, in degrees.

        Returns
        -------
        numpy.ndarray
            Original angles, in [0, 360).
        """
        sign = -1.0 if self.flip else 1.0
        return (sign * (np.asarray(angle_deg, dtype=float) - self.offset_deg)) % 360.0

    def then(self, other: RingAlignment) -> RingAlignment:
        """Compose this alignment with another applied after it.

        A ring aligned onto another ring, and that ring onto heading, reads as
        heading in one step: ``a.then(b).apply(x)`` is ``b.apply(a.apply(x))``.

        Parameters
        ----------
        other : RingAlignment
            The alignment applied second.

        Returns
        -------
        RingAlignment
            The composition. It was not fitted to anything, so its resultants
            are NaN.
        """
        sign = -1.0 if other.flip else 1.0
        return RingAlignment(
            flip=self.flip != other.flip,
            offset_deg=float((sign * self.offset_deg + other.offset_deg) % 360.0),
            resultant=float("nan"),
            resultant_other=float("nan"),
            n=0,
        )

    def __str__(self) -> str:
        turn = f"{'mirrored, then ' if self.flip else ''}turned {self.offset_deg:+.1f} deg"
        if not np.isfinite(self.resultant):
            return f"{turn} (composed, not fitted)"
        return (
            f"{turn} (resultant {self.resultant:.2f}; {self.resultant_other:.2f} the "
            f"other way round, n={self.n})"
        )


def align_angles(angle_deg, reference_deg) -> RingAlignment:
    """Fit the reflection and rotation that best take angles onto a reference.

    The handedness whose differences have the larger mean resultant length
    wins, and the rotation is their circular mean -- the MATLAB's flip check,
    and SPUD's shift-and-flip. Pairs with a NaN are skipped.

    Parameters
    ----------
    angle_deg, reference_deg : array-like
        Paired angles, in degrees.

    Returns
    -------
    RingAlignment
        The transform.

    Raises
    ------
    ValueError
        If the inputs differ in length or fewer than two pairs are finite.
    """
    a = np.deg2rad(np.asarray(angle_deg, dtype=float))
    b = np.deg2rad(np.asarray(reference_deg, dtype=float))
    if a.shape != b.shape:
        raise ValueError(f"{a.shape} angles but {b.shape} reference angles")
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 2:
        raise ValueError("fewer than two finite pairs of angles to align")
    a, b = a[finite], b[finite]
    kept = np.exp(1j * (b - a)).mean()
    mirrored = np.exp(1j * (b + a)).mean()
    flip = bool(np.abs(mirrored) > np.abs(kept))
    chosen, other = (mirrored, kept) if flip else (kept, mirrored)
    return RingAlignment(
        flip=flip,
        offset_deg=float(np.rad2deg(np.angle(chosen)) % 360.0),
        resultant=float(np.abs(chosen)),
        resultant_other=float(np.abs(other)),
        n=int(finite.sum()),
    )


# ── decoding ─────────────────────────────────────────────────────────────
def _unit_weights(ring: RingModel, weights) -> np.ndarray:
    """Turn a weighting choice into one weight per ring unit.

    Parameters
    ----------
    ring : RingModel
        The ring.
    weights : {"coupling", "uniform", "radius"} or array-like
        ``"coupling"`` scales each unit by its coupling over the largest,
        negatives as 0; ``"radius"`` by its radius over the largest; or one
        weight per unit.

    Returns
    -------
    numpy.ndarray
        Weights, float32.

    Raises
    ------
    ValueError
        If the choice is unknown or the wrong length.
    """
    if isinstance(weights, str):
        if weights == "coupling":
            values = np.clip(np.nan_to_num(ring.coupling, nan=0.0), 0.0, None)
        elif weights == "uniform":
            values = np.ones(ring.n_units)
        elif weights == "radius":
            values = ring.radius.astype(float)
        else:
            raise ValueError(
                "weights must be 'coupling', 'uniform', 'radius' or an array, "
                f"not {weights!r}"
            )
        if values.max() > 0:
            values = values / values.max()
    else:
        values = np.asarray(weights, dtype=float)
        if values.shape != (ring.n_units,):
            raise ValueError(f"{values.shape} weights for {ring.n_units} ring units")
    return values.astype(np.float32)


@dataclass
class _Frame:
    """A population-vector decode in the ring's own frame, before alignment."""

    bin_index: np.ndarray
    run_index: np.ndarray
    ring_deg: np.ndarray
    length: np.ndarray
    z: np.ndarray  # (bins, units) z-scored smoothed rates
    weights: np.ndarray


def _population_vector(z: np.ndarray, angle_deg: np.ndarray, weights: np.ndarray):
    """Sum the units around the ring, each pointing at its own angle.

    Parameters
    ----------
    z : numpy.ndarray
        z-scored rates, shape (bins, units).
    angle_deg : numpy.ndarray
        Each unit's angle.
    weights : numpy.ndarray
        Each unit's weight.

    Returns
    -------
    angle_deg : numpy.ndarray
        Direction of the vector per bin, in [0, 360).
    length : numpy.ndarray
        Its length over the summed weighted |z|, in [0, 1]: 1 when every
        active unit points the same way.
    """
    theta = np.deg2rad(np.asarray(angle_deg, dtype=float))
    real = z @ (weights * np.cos(theta)).astype(np.float32)
    imag = z @ (weights * np.sin(theta)).astype(np.float32)
    norm = np.abs(z) @ weights
    with np.errstate(invalid="ignore", divide="ignore"):
        length = np.where(norm > 0, np.hypot(real, imag) / norm, 0.0)
    return np.rad2deg(np.arctan2(imag, real)) % 360.0, length.astype(float)


def _ring_frame(data, ring, mask, weights, max_gap_s, min_run_s) -> _Frame:
    """Decode the masked bins in the ring's own frame.

    Parameters
    ----------
    data : DecoderData
        Binned data holding the ring's units.
    ring : RingModel
        The ring.
    mask : numpy.ndarray of bool
        Bins to decode.
    weights : str or array-like
        As for :func:`decode_ring`.
    max_gap_s, min_run_s : float
        As for :func:`spikeshpc.decoder.decode`.

    Returns
    -------
    _Frame
        The decode.

    Raises
    ------
    ValueError
        If no run of masked bins is long enough.
    """
    runs = _runs(mask, data.duration_s, max_gap_s, min_run_s)
    if not runs:
        raise ValueError(
            f"no run of masked bins reaches min_run_s={min_run_s} "
            f"(after dropping bins longer than max_gap_s={max_gap_s})"
        )
    bin_index = np.concatenate([np.arange(a, b) for a, b in runs])
    run_index = np.concatenate([np.full(b - a, i) for i, (a, b) in enumerate(runs)])
    decoded = np.zeros(data.n_bins, dtype=bool)
    decoded[bin_index] = True

    columns = _unit_positions(data.unit_ids, ring.unit_ids)
    z = _smoothed_rates(data, decoded, ring.params.get("smooth_sigma_s", 0.05), columns)
    # each unit against its own mean and spread over the bins being decoded
    mean = z.mean(axis=0, dtype=np.float64)
    sd = z.std(axis=0, dtype=np.float64)
    z -= mean.astype(np.float32)
    z /= np.where(sd > 0, sd, np.inf).astype(np.float32)
    w = _unit_weights(ring, weights)
    ring_deg, length = _population_vector(z, ring.angle_deg, w)
    return _Frame(bin_index, run_index, ring_deg, length, z, w)


def _as_decoded(
    data, ring, frame, alignment, with_metrics, tolerance_deg, label
) -> Decoded:
    """Package a ring-frame decode as a :class:`~spikeshpc.decoder.Decoded`.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    ring : RingModel
        The ring.
    frame : _Frame
        The decode.
    alignment : RingAlignment or None
        Mapping onto heading; None leaves the ring's frame.
    with_metrics : bool
        Score against the measured heading.
    tolerance_deg : float
        Tolerance for ``frac_within_deg``.
    label : str
        Name of the decode.

    Returns
    -------
    Decoded
        With ``posterior_max`` holding the population vector's length.
    """
    decoded = frame.ring_deg if alignment is None else alignment.apply(frame.ring_deg)
    actual = data.heading_deg[frame.bin_index]
    metrics = decoding_metrics(decoded, actual, tolerance_deg) if with_metrics else {}
    metrics["mean_posterior_max"] = float(frame.length.mean())

    columns = _unit_positions(data.unit_ids, ring.unit_ids)
    n_spikes = np.empty(frame.bin_index.size, dtype=np.int64)
    for start in range(0, frame.bin_index.size, _GRAM_ROWS):
        rows = slice(start, start + _GRAM_ROWS)
        n_spikes[rows] = data.counts[frame.bin_index[rows]][:, columns].sum(axis=1)
    return Decoded(
        label=label,
        time_s=data.time_s[frame.bin_index],
        duration_s=data.duration_s[frame.bin_index],
        decoded_deg=decoded,
        actual_deg=actual,
        error_deg=circular_difference(decoded, actual),
        posterior_max=frame.length,
        entropy_bits=np.full(frame.bin_index.size, np.nan),
        n_spikes=n_spikes,
        run_index=frame.run_index,
        bin_index=frame.bin_index,
        bin_centers_deg=np.empty(0),
        posterior=None,
        metrics=metrics,
    )


def decode_ring(
    data: DecoderData,
    ring: RingModel,
    mask,
    alignment: RingAlignment | None = None,
    weights="coupling",
    max_gap_s: float = 1.0,
    min_run_s: float = 1.0,
    with_metrics: bool = True,
    tolerance_deg: float = 10.0,
    label: str = "ring",
) -> Decoded:
    """Decode the bins under `mask` from where on the ring the population is active.

    Each unit's smoothed rate is z-scored over the decoded bins, so the vector
    points at the units firing above their own mean. Stretches are found as
    :func:`spikeshpc.decoder.decode` finds them, so on the same mask both
    decode the same bins.

    Parameters
    ----------
    data : DecoderData
        Binned data holding the ring's units.
    ring : RingModel
        The ring.
    mask : array-like of bool
        Bins to decode.
    alignment : RingAlignment, optional
        Mapping from the ring's frame onto heading, from :func:`align_angles`
        on training bins. Without it the angles stay in the ring's arbitrary
        frame, where only ``median_abs_error_corrected_deg`` means anything,
        and that only if the ring is not mirrored.
    weights : {"coupling", "uniform", "radius"} or array-like, default "coupling"
        How much each unit counts.
    max_gap_s : float, default 1.0
        Largest bin that still joins a stretch.
    min_run_s : float, default 1.0
        Shortest stretch decoded.
    with_metrics : bool, default True
        Score against the measured heading.
    tolerance_deg : float, default 10.0
        Tolerance for ``frac_within_deg``.
    label : str, default "ring"
        Name of the decode.

    Returns
    -------
    Decoded
        The decode. ``posterior_max`` holds the population vector's normalized
        length, ``entropy_bits`` is NaN and there is no posterior.

    Raises
    ------
    ValueError
        If the mask is the wrong shape or no stretch is long enough.
    """
    mask = _check_mask(data, mask)
    frame = _ring_frame(data, ring, mask, weights, max_gap_s, min_run_s)
    return _as_decoded(data, ring, frame, alignment, with_metrics, tolerance_deg, label)


def _shift_null(
    data, train, test, observed, n_shuffles, min_shift_s, seed
) -> ShuffleTest:
    """Judge a wake decode against heading shifted in time, alignment refitted each time.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    train, test : _Frame
        Ring-frame decodes of the training and held-out bins.
    observed : float
        The real decode's median absolute error.
    n_shuffles : int
        Number of shifts.
    min_shift_s : float
        Smallest shift, in seconds.
    seed : int
        Random seed.

    Returns
    -------
    ShuffleTest
        The test of ``median_abs_error_deg``.

    Raises
    ------
    ValueError
        If the data are too short for the shift.
    """
    n = data.n_bins
    min_shift = max(1, round(min_shift_s / data.bin_s))
    if 2 * min_shift >= n:
        raise ValueError(
            f"min_shift_s={min_shift_s} leaves no room to shift "
            f"{data.duration_s.sum():.0f}s of data"
        )
    rng = np.random.default_rng(seed)
    shifts = [int(rng.integers(min_shift, n - min_shift)) for _ in range(n_shuffles)]
    null = np.empty(n_shuffles)
    for k, shift in enumerate(shifts):
        # np.roll(heading, shift)[i] is heading[(i - shift) % n]
        train_heading = data.heading_deg[(train.bin_index - shift) % n]
        test_heading = data.heading_deg[(test.bin_index - shift) % n]
        alignment = align_angles(train.ring_deg, train_heading)
        null[k] = np.median(
            np.abs(circular_difference(alignment.apply(test.ring_deg), test_heading))
        )
    return _judge("median_abs_error_deg", observed, null, "shift")


def _unit_null(frame: _Frame, angle_deg, n_shuffles: int, seed: int) -> ShuffleTest:
    """Judge a sleep decode's coherence against the ring with its angles scrambled.

    Rates, synchrony and each unit's weight survive; which unit sits where on
    the ring does not.

    Parameters
    ----------
    frame : _Frame
        The decode.
    angle_deg : numpy.ndarray
        The ring's angles.
    n_shuffles : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    ShuffleTest
        The test of ``mean_posterior_max``, the mean vector length.
    """
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(angle_deg)) for _ in range(n_shuffles)]
    theta = np.deg2rad(np.asarray(angle_deg, dtype=float))
    norm = np.abs(frame.z) @ frame.weights
    inverse_norm = np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0)
    batch = max(1, min(64, int(2**25 // max(frame.z.shape[0], 1))))
    null = np.empty(n_shuffles)
    for start in range(0, n_shuffles, batch):
        chunk = orders[start : start + batch]
        cos = np.stack([frame.weights * np.cos(theta[o]) for o in chunk], axis=1)
        sin = np.stack([frame.weights * np.sin(theta[o]) for o in chunk], axis=1)
        real = frame.z @ cos.astype(np.float32)
        length = np.hypot(real, frame.z @ sin.astype(np.float32))
        null[start : start + len(chunk)] = (length * inverse_norm[:, None]).mean(axis=0)
    return _judge("mean_posterior_max", float(frame.length.mean()), null, "units")


# ── the whole thing ──────────────────────────────────────────────────────
@dataclass
class RingRun:
    """Everything one call to :func:`run_ring` produced.

    Attributes
    ----------
    data : DecoderData
        Binned data.
    ring : RingModel
        The ring, fitted on `train_mask`.
    alignment : RingAlignment
        Ring frame onto heading, fitted on `train_mask`.
    train_mask, test_mask : numpy.ndarray of bool
        Training and held-out bins.
    test : Decoded
        Decode of the held-out wake.
    test_shuffle : ShuffleTest or None
        Time-shift null for `test`.
    rem, nrem : Decoded or None
        Decodes of REM and NREM, in heading coordinates through `alignment`.
    rem_shuffle, nrem_shuffle : ShuffleTest or None
        Their angle-permutation nulls.
    rem_mask, nrem_mask : numpy.ndarray of bool or None
        Bins decoded.
    weights : str or numpy.ndarray
        How the population vector weighted the units.
    """

    data: DecoderData
    ring: RingModel
    alignment: RingAlignment
    train_mask: np.ndarray
    test_mask: np.ndarray
    test: Decoded
    test_shuffle: ShuffleTest | None = None
    rem: Decoded | None = None
    rem_shuffle: ShuffleTest | None = None
    rem_mask: np.ndarray | None = None
    nrem: Decoded | None = None
    nrem_shuffle: ShuffleTest | None = None
    nrem_mask: np.ndarray | None = None
    weights: object = "coupling"

    @property
    def aligned_angle_deg(self) -> np.ndarray:
        """Get each ring unit's position in heading coordinates.

        The heading at which the decode points at that unit: comparable with
        its preferred direction, and what to sort and color a raster by.

        Returns
        -------
        numpy.ndarray
            Angles, in `ring.unit_ids` order.
        """
        return self.alignment.apply(self.ring.angle_deg)

    def summary(self) -> str:
        screen = self.ring.screen
        lines = [
            f"{self.ring}",
            (
                f"  {self.ring.n_units} of {len(screen)} units on the ring "
                f"({int((screen['status'] == SILENT).sum())} silent, "
                f"{int((screen['status'] == NO_PARTNERS).sum())} without partners), "
                f"{len(self.ring.members())} members"
            ),
            (
                f"  train {self.data.duration_s[self.train_mask].sum():.0f}s / "
                f"test {self.data.duration_s[self.test_mask].sum():.0f}s"
            ),
            f"  onto heading: {self.alignment}",
            f"  test: {self.test}",
        ]
        for key in (
            "median_abs_error_deg",
            "median_abs_error_corrected_deg",
            "frac_within_deg",
            "circular_correlation",
        ):
            if key in self.test.metrics:
                lines.append(f"    {key} = {self.test.metrics[key]:.3f}")
        if self.test_shuffle is not None:
            lines.append(f"    vs shuffle: {self.test_shuffle}")
        for state, decoded, shuffle in (
            ("REM", self.rem, self.rem_shuffle),
            ("NREM", self.nrem, self.nrem_shuffle),
        ):
            if decoded is None:
                continue
            lines.append(f"  {state}: {decoded}")
            lines.append(
                f"    mean vector length = {decoded.metrics['mean_posterior_max']:.3f} "
                f"(wake test {self.test.metrics['mean_posterior_max']:.3f})"
            )
            if shuffle is not None:
                lines.append(f"    vs shuffle: {shuffle}")
        return "\n".join(lines)

    def as_row(self) -> dict:
        """Collect the headline numbers as one flat row, for a table across recordings.

        Returns
        -------
        dict
            The row.
        """
        ring, test = self.ring, self.test.metrics
        row = {
            "n_candidates": len(ring.screen),
            "n_units": ring.n_units,
            "n_members": len(ring.members()),
            "fit_s": ring.fit_time_s,
            "eigenvalue_ratio": ring.eigenvalue_ratio,
            "stress": ring.stress,
            "consistency_r": ring.consistency_r,
            "split_half_r": ring.split_half_r,
            "flipped": self.alignment.flip,
            "offset_deg": self.alignment.offset_deg,
            "alignment_resultant": self.alignment.resultant,
            "alignment_margin": self.alignment.margin,
            "test_s": float(self.test.duration_s.sum()),
            "test_median_abs_error_deg": test.get("median_abs_error_deg", np.nan),
            "test_median_abs_error_corrected_deg": test.get(
                "median_abs_error_corrected_deg", np.nan
            ),
            "test_frac_within_deg": test.get("frac_within_deg", np.nan),
            "test_circular_correlation": test.get("circular_correlation", np.nan),
            "test_mean_length": test.get("mean_posterior_max", np.nan),
        }
        if self.test_shuffle is not None:
            row["test_null_mean"] = float(self.test_shuffle.null.mean())
            row["test_p"] = self.test_shuffle.p_value
            row["test_z"] = self.test_shuffle.z_score
        for key, decoded, shuffle in (
            ("rem", self.rem, self.rem_shuffle),
            ("nrem", self.nrem, self.nrem_shuffle),
        ):
            row[f"{key}_s"] = (
                float(decoded.duration_s.sum()) if decoded is not None else 0.0
            )
            row[f"{key}_mean_length"] = (
                decoded.metrics["mean_posterior_max"] if decoded is not None else np.nan
            )
            if shuffle is not None:
                row[f"{key}_null_mean"] = float(shuffle.null.mean())
                row[f"{key}_p"] = shuffle.p_value
                row[f"{key}_z"] = shuffle.z_score
        return row

    def __repr__(self) -> str:
        return self.summary()


def run_ring(
    sorting,
    unit_ids,
    heading_deg,
    frame_times,
    intervals,
    bin_s: float = 0.05,
    smooth_sigma_s: float = 0.05,
    max_lag_s: float = 10.0,
    baseline_start_s: float = 5.0,
    k_neighbors: int = 8,
    min_rate_hz: float = 0.1,
    min_partner_snr: float | None = 4.0,
    weights="coupling",
    test_fraction: float = 0.3,
    split_mode: str = "blocks",
    block_s: float = 60.0,
    tolerance_deg: float = 10.0,
    n_shuffles: int = 100,
    min_shift_s: float = 30.0,
    decode_rem: bool = True,
    decode_nrem: bool = False,
    seed: int = 0,
    within=None,
    interval_mask=None,
    verbose: bool = True,
) -> RingRun:
    """Fit a ring on training wake, then decode held-out wake, REM and NREM from it.

    The counterpart of :func:`spikeshpc.decoder.run_decoder`, with the ring in
    place of tuning curves. Wake is masked and split exactly as there --
    ``within``, ``interval_mask``, ``test_fraction``, ``split_mode``,
    ``block_s`` and ``seed`` mean the same, and with the same ``bin_s`` the
    two hold out the same bins. Then:

    1. the ring is fitted on the training bins, from spikes alone;
    2. its frame is aligned to heading on the training bins (a reflection and
       a rotation, :func:`align_angles`) -- the only use of heading;
    3. the held-out wake is decoded and scored, against a null that shifts
       heading in time and refits the alignment for every shift;
    4. REM and NREM are decoded through the same alignment and judged by
       their coherence (mean vector length) against the ring with its angles
       permuted across units.

    Parameters
    ----------
    sorting : spikeinterface BaseSorting or SortingAnalyzer
        Source of the spike trains.
    unit_ids : sequence
        Candidate units: the GOOD and MUA units of the head-direction
        structure -- a depth band -- not only the tuned ones. The whole probe
        lets other structures take the embedding over.
    heading_deg, frame_times : array-like
        The shutter-aligned pair.
    intervals : dict
        ``scoring.intervals`` from :func:`spikeshpc.load_states`.
    bin_s : float, default 0.05
        Bin width, rounded down to whole camera frames (3 frames at 60 Hz;
        the MATLAB uses 40 ms). The decoder's ``bin_s`` gives bins that pair
        one to one with its decodes, at three times the memory: the counts are
        ``n_bins x n_units`` int32.
    smooth_sigma_s, max_lag_s, baseline_start_s, k_neighbors, min_rate_hz, min_partner_snr
        As for :func:`fit_ring`.
    weights : {"coupling", "uniform", "radius"} or array-like, default "coupling"
        As for :func:`decode_ring`.
    test_fraction, split_mode, block_s, seed, within, interval_mask
        As for :func:`spikeshpc.decoder.run_decoder`.
    tolerance_deg : float, default 10.0
        Tolerance for ``frac_within_deg``.
    n_shuffles : int, default 100
        Shuffles per null; 0 skips them. Neither re-decodes anything, so they
        are cheap.
    min_shift_s : float, default 30.0
        Smallest shift of heading against the decode, in seconds.
    decode_rem, decode_nrem : bool, default True, False
        Also decode REM, NREM.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    RingRun
        The ring, the alignment, the train/test masks and the decodes with
        their nulls.

    Raises
    ------
    ValueError
        If no WAKE bin remains, or the ring cannot be fitted.
    """
    data = prepare_decoder_data(sorting, unit_ids, heading_deg, frame_times, bin_s)
    if verbose:
        print(f"  {data}")
    wake_mask = _wake_mask(data, intervals, frame_times, within, interval_mask, verbose)
    train_mask, test_mask = split_train_test(
        data, wake_mask, test_fraction, split_mode, block_s, seed
    )
    if verbose:
        print(
            f"  wake {data.duration_s[wake_mask].sum():.0f}s -> "
            f"train {data.duration_s[train_mask].sum():.0f}s, "
            f"test {data.duration_s[test_mask].sum():.0f}s ({split_mode})"
        )

    ring = fit_ring(
        data,
        train_mask,
        smooth_sigma_s=smooth_sigma_s,
        max_lag_s=max_lag_s,
        baseline_start_s=baseline_start_s,
        k_neighbors=k_neighbors,
        min_rate_hz=min_rate_hz,
        min_partner_snr=min_partner_snr,
        verbose=verbose,
    )
    train = _ring_frame(data, ring, train_mask, weights, 1.0, 1.0)
    alignment = align_angles(train.ring_deg, data.heading_deg[train.bin_index])
    if verbose:
        print(f"  onto heading: {alignment}")

    test_frame = _ring_frame(data, ring, test_mask, weights, 1.0, 1.0)
    test = _as_decoded(
        data, ring, test_frame, alignment, True, tolerance_deg, "ring wake test"
    )
    if verbose:
        print(f"  {test}")
    test_shuffle = None
    if n_shuffles:
        test_shuffle = _shift_null(
            data,
            train,
            test_frame,
            test.metrics["median_abs_error_deg"],
            n_shuffles,
            min_shift_s,
            seed,
        )
        if verbose:
            print(f"    {test_shuffle}")

    sleep = {"REM": (None, None, None), "NREM": (None, None, None)}
    for state, wanted in (("REM", decode_rem), ("NREM", decode_nrem)):
        if not wanted:
            continue
        mask = state_interval_mask(data, intervals, state)
        if not mask.any():
            warnings.warn(
                f"no decoder bin falls inside a {state} interval", stacklevel=2
            )
            continue
        try:
            frame = _ring_frame(data, ring, mask, weights, 1.0, 1.0)
        except ValueError as error:
            warnings.warn(f"{state} not decoded: {error}", stacklevel=2)
            continue
        decoded = _as_decoded(
            data, ring, frame, alignment, False, tolerance_deg, f"ring {state}"
        )
        if verbose:
            print(f"  {decoded}")
        shuffle = None
        if n_shuffles:
            shuffle = _unit_null(frame, ring.angle_deg, n_shuffles, seed)
        if verbose and shuffle is not None:
            print(f"    {shuffle}")
        sleep[state] = (decoded, shuffle, mask)

    return RingRun(
        data=data,
        ring=ring,
        alignment=alignment,
        train_mask=train_mask,
        test_mask=test_mask,
        test=test,
        test_shuffle=test_shuffle,
        rem=sleep["REM"][0],
        rem_shuffle=sleep["REM"][1],
        rem_mask=sleep["REM"][2],
        nrem=sleep["NREM"][0],
        nrem_shuffle=sleep["NREM"][1],
        nrem_mask=sleep["NREM"][2],
        weights=weights,
    )


def pair_decodes(a: Decoded, b: Decoded) -> tuple[np.ndarray, np.ndarray]:
    """Pair each bin of one decode with the bin of another that contains its centre.

    For comparing two decoders bin by bin, on the same grid or not: on the
    same grid each pair is one bin.

    Parameters
    ----------
    a, b : Decoded
        Two decodes; `b`'s bins in time order, as every decode's are.

    Returns
    -------
    index_a, index_b : numpy.ndarray of int
        Positions in `a` and in `b` of each pair.
    """
    start = b.time_s - b.duration_s / 2
    stop = b.time_s + b.duration_s / 2
    j = np.searchsorted(start, a.time_s, side="right") - 1
    inside = j >= 0
    inside[inside] = a.time_s[inside] < stop[j[inside]]
    return np.flatnonzero(inside), j[inside]


def control_band(
    unit_ids,
    depths_um,
    band_um: tuple[float, float],
    n_units: int | None = None,
    gap_um: float = 0.0,
    side: str = "either",
) -> tuple[tuple[float, float], np.ndarray]:
    """A depth band clear of `band_um`, about as wide and holding as many units.

    The negative control for a ring: :func:`run_ring` on as many units, from
    tissue the head-direction structure does not reach, should find no ring.
    Equal numbers matter because the screen, the Isomap graph and the
    population vector all change with the number of units.

    Every run of `n_units` units consecutive in depth, all more than `gap_um`
    outside `band_um`, is a candidate. A run that fits within `band_um`'s
    width with no other unit inside gets exactly that width; one that does
    not gets as close as it can -- its own span if it is wider, out to its
    neighbours if it is narrower. A band never reaches past the outermost
    unit, where the probe may have no channels, or into the gap. Of the runs
    that come closest to the width, the furthest from `band_um` wins, as the
    least likely to hold head-direction cells that straggle past it.

    Parameters
    ----------
    unit_ids : sequence
        Candidate units, such as ``hd.unit_ids``.
    depths_um : array-like
        Their depths, in the same order, such as ``hd.depths``.
    band_um : (float, float)
        The ring's band, ends included, as the notebook's ``ring_depth_um``.
    n_units : int, optional
        Units the control band should hold. Default: as many as `band_um`
        holds, which is ``len(ring_units)``.
    gap_um : float, default 0
        Clearance between the two bands.
    side : {"either", "below", "above"}
        Smaller or larger depth values than `band_um`.

    Returns
    -------
    band_um : (float, float)
        The control band, ends included.
    unit_ids : numpy.ndarray
        Its units, in the order given.

    Warns
    -----
    UserWarning
        When the band's width is more than 10% off `band_um`'s: the units
        outside `band_um` are denser or sparser than inside it.
    """
    unit_ids = np.asarray(unit_ids)
    depths = np.asarray(depths_um, dtype=float)
    if depths.shape != unit_ids.shape:
        raise ValueError(
            f"{depths.size} depths for {unit_ids.size} units; give one per unit"
        )
    if side not in ("either", "below", "above"):
        raise ValueError(f"side must be 'either', 'below' or 'above', not {side!r}")
    low, high = sorted(float(edge) for edge in band_um)
    width = high - low
    if n_units is None:
        n_units = int(np.sum((depths >= low) & (depths <= high)))
    if n_units < 1:
        raise ValueError("the control band needs at least 1 unit")

    tol = 1e-6  # um: an edge short of the neighbouring unit, not on it
    best = None
    for name, inner in (("below", low - gap_um), ("above", high + gap_um)):
        if side not in ("either", name):
            continue
        outside = depths < inner if name == "below" else depths > inner
        d = np.sort(depths[outside])
        for i in range(d.size - n_units + 1):
            first, last = d[i], d[i + n_units - 1]
            if i > 0:
                floor = d[i - 1] + tol
            else:
                floor = first if name == "below" else inner + tol
            if i + n_units < d.size:
                ceiling = d[i + n_units] - tol
            else:
                ceiling = inner - tol if name == "below" else last
            if floor > first or ceiling < last:
                continue  # a unit at the same depth as the run's end
            w = float(np.clip(width, last - first, ceiling - floor))
            # centred on the run, as far as its neighbours allow
            start = float(
                np.clip(
                    (first + last - w) / 2,
                    max(floor, last - w),
                    min(first, ceiling - w),
                )
            )
            distance = low - (start + w) if name == "below" else start - high
            key = (abs(w - width), -distance)
            if best is None or key < best[0]:
                best = (key, (start, start + w))

    if best is None:
        where = "" if side == "either" else f", {side} it"
        raise ValueError(
            f"fewer than {n_units} units, or none consecutive, lie more than "
            f"{gap_um:g} um outside {low:.0f}-{high:.0f} um{where}"
        )
    band = best[1]
    if best[0][0] > 0.1 * width:
        warnings.warn(
            f"the control band is {band[1] - band[0]:.0f} um wide, against "
            f"{width:.0f} um: {n_units} units are spread differently outside "
            f"{low:.0f}-{high:.0f} um",
            stacklevel=2,
        )
    inside = (depths >= band[0]) & (depths <= band[1])
    return band, unit_ids[inside]


# ── comparisons ──────────────────────────────────────────────────────────
@dataclass
class AngleComparison:
    """Two sets of paired angles, aligned and compared; see :func:`compare_angles`.

    Attributes
    ----------
    alignment : RingAlignment
        Reflection and rotation taking the angles onto the reference.
    aligned_deg, reference_deg : numpy.ndarray
        The aligned angles and the reference.
    circular_r : float
        Fisher and Lee's circular correlation of the aligned angles with the
        reference (:func:`circular_r`): 1 for a perfect match.
    median_abs_error_deg : float
        Median |aligned - reference|.
    shuffle : ShuffleTest or None
        The median error against the reference permuted, the alignment
        refitted for each permutation.
    """

    alignment: RingAlignment
    aligned_deg: np.ndarray
    reference_deg: np.ndarray
    circular_r: float
    median_abs_error_deg: float
    shuffle: ShuffleTest | None = None

    @property
    def n(self) -> int:
        return len(self.aligned_deg)

    def __str__(self) -> str:
        text = (
            f"circular r {self.circular_r:.2f}, median |error| "
            f"{self.median_abs_error_deg:.1f} deg over {self.n} after "
            f"{'mirroring and ' if self.alignment.flip else ''}turning "
            f"{self.alignment.offset_deg:+.1f} deg"
        )
        if self.shuffle is not None:
            text += (
                f"; vs permuted {self.shuffle.null.mean():.1f} deg: "
                f"p = {self.shuffle.p_value:.4f}"
            )
        return text


def compare_angles(
    angle_deg, reference_deg, n_shuffles: int = 1000, seed: int = 0
) -> AngleComparison:
    """Ask whether paired angles agree up to a reflection and a rotation.

    A ring's frame is arbitrary, so this is the only sense in which a ring
    position can match a preferred direction, or another ring's position. The
    null permutes which reference belongs to which angle and refits the
    alignment for each permutation, so it is judged on the same terms as the
    real pairing.

    Parameters
    ----------
    angle_deg, reference_deg : array-like
        Paired angles, in degrees; pairs with a NaN are dropped.
    n_shuffles : int, default 1000
        Permutations; 0 skips the null.
    seed : int, default 0
        Random seed.

    Returns
    -------
    AngleComparison
        The comparison.

    Raises
    ------
    ValueError
        If fewer than three pairs are finite.
    """
    a = np.asarray(angle_deg, dtype=float)
    b = np.asarray(reference_deg, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        raise ValueError(f"{finite.sum()} finite pairs of angles: need at least 3")
    a, b = a[finite], b[finite]

    alignment = align_angles(a, b)
    aligned = alignment.apply(a)
    median = float(np.median(np.abs(circular_difference(aligned, b))))
    shuffle = None
    if n_shuffles:
        rng = np.random.default_rng(seed)
        null = np.empty(n_shuffles)
        for k in range(n_shuffles):
            permuted = b[rng.permutation(b.size)]
            fitted = align_angles(a, permuted)
            null[k] = np.median(np.abs(circular_difference(fitted.apply(a), permuted)))
        shuffle = _judge("median_abs_error_deg", median, null, "permutation")
    return AngleComparison(
        alignment=alignment,
        aligned_deg=aligned,
        reference_deg=b,
        circular_r=circular_r(aligned, b),
        median_abs_error_deg=median,
        shuffle=shuffle,
    )


def _structure_test(x: np.ndarray, y: np.ndarray, n_shuffles: int, seed: int):
    """Correlate two pair-by-pair matrices over their upper triangles (a Mantel test).

    Parameters
    ----------
    x, y : numpy.ndarray
        Square matrices over the same units, in the same order.
    n_shuffles : int
        Permutations of `y`'s units, rows and columns together; 0 skips them.
    seed : int
        Random seed.

    Returns
    -------
    r : float
        Pearson r over the pairs.
    test : ShuffleTest or None
        r against the permutations, as ``structure_r``.
    """
    n = x.shape[0]
    upper = np.triu_indices(n, 1)
    r = _pearson(x[upper], y[upper])
    if not n_shuffles:
        return r, None
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffles)
    for k in range(n_shuffles):
        order = rng.permutation(n)
        null[k] = _pearson(x[upper], y[np.ix_(order, order)][upper])
    return r, _judge("structure_r", r, null, "permutation")


@dataclass
class RingTuning:
    """How a ring lines up with the units' measured tuning; see :func:`ring_vs_tuning`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per candidate unit of the ring: ``status`` and
        ``partner_snr`` from the screen; ``coupling``, ``member``,
        ``radius`` and ``ring_angle_deg`` (aligned to the preferred
        directions) for the units on the ring; ``preferred_deg``, ``mvl``
        and ``tuned`` from the tuning; ``error_deg`` for tuned units on the
        ring.
    angles : AngleComparison or None
        Ring position against preferred direction, tuned units on the ring.
    auc_partner_snr : float
        How often a tuned candidate has stronger partners than an untuned one.
    auc_coupling, auc_radius : float
        The same for coupling and radius, among the units on the ring.
    radius_mvl_rho, coupling_mvl_rho : float
        Spearman's rho of radius and of coupling with mean vector length,
        units on the ring.
    structure_r : float
        Pearson r over tuned pairs on the ring between the pair metric (z) and
        the cosine of their preferred directions' difference -- with no
        embedding in it.
    structure : ShuffleTest or None
        `structure_r` against the preferred directions permuted.
    curve : pandas.DataFrame
        Mean and SEM of the pair metric (z) by preferred-direction difference,
        tuned pairs (MATLAB's Fig 3c).
    min_coupling : float
        The coupling that made a unit a member.
    """

    table: pd.DataFrame
    angles: AngleComparison | None
    auc_partner_snr: float
    auc_coupling: float
    auc_radius: float
    radius_mvl_rho: float
    coupling_mvl_rho: float
    structure_r: float
    structure: ShuffleTest | None
    curve: pd.DataFrame
    min_coupling: float = 0.5

    def crosstab(self) -> pd.DataFrame:
        """Count tuned and untuned candidates by what the ring made of them.

        Returns
        -------
        pandas.DataFrame
            Rows tuned / untuned; columns "member", "on ring, not member" and
            each reason a unit was left off.
        """
        status = self.table["status"]
        where = np.where(
            self.table["member"],
            "member",
            np.where(status == ON_RING, "on ring, not member", status),
        )
        table = pd.crosstab(
            np.where(self.table["tuned"], "tuned", "untuned"), where
        ).rename_axis(index=None, columns=None)
        columns = ("member", "on ring, not member", NO_PARTNERS, SILENT)
        order = [c for c in columns if c in table]
        return table[order]

    def summary(self) -> str:
        tuned = self.table["tuned"]
        member = self.table["member"]
        lines = [
            f"ring vs tuning: {len(self.table)} candidates, {int(tuned.sum())} tuned",
            (
                f"  members (coupling >= {self.min_coupling:g}): "
                f"{int((member & tuned).sum())} of {int(tuned.sum())} tuned units, "
                f"{int((member & ~tuned).sum())} of {int((~tuned).sum())} untuned"
            ),
            (
                f"  tuned vs untuned AUC: partner SNR {_number(self.auc_partner_snr)}; "
                f"on the ring, coupling {_number(self.auc_coupling)}, radius "
                f"{_number(self.auc_radius)}"
            ),
            (
                f"  with MVL (Spearman), on the ring: coupling "
                f"{_number(self.coupling_mvl_rho)}, radius {_number(self.radius_mvl_rho)}"
            ),
        ]
        if self.angles is not None:
            lines.append(f"  ring position vs preferred direction: {self.angles}")
        text = f"  pair metric vs tuning similarity (tuned pairs): r {self.structure_r:.2f}"
        if self.structure is not None:
            text += f", p = {self.structure.p_value:.4f}"
        lines.append(text)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def ring_vs_tuning(
    ring: RingModel,
    hd,
    min_coupling: float = 0.5,
    n_shuffles: int = 1000,
    seed: int = 0,
    bin_deg: float = 15.0,
) -> RingTuning:
    """Ask whether the ring found the head-direction cells, and put them in order.

    Separate questions, each with its own answer:

    - are the tuned candidates the ones the ring took in -- stronger partners
      (``partner_snr``), more ring-organised co-firing (``coupling``)?
    - do the tuned units' ring positions match their preferred directions,
      up to a reflection and a rotation (:func:`compare_angles`)?
    - is the pair metric itself organised by tuning -- pairs with similar
      preferred directions more correlated? This one involves no embedding,
      so if it fails, no ring can succeed.

    Parameters
    ----------
    ring : RingModel
        The ring.
    hd : spikeshpc.optitrack.HDTuning
        The same recording's tuning; anything with a ``stats`` dict of
        ``unit -> HDTuningStats`` (or that dict itself) works. Candidates it
        does not cover count as untuned.
    min_coupling : float, default 0.5
        Coupling that makes a unit on the ring a member.
    n_shuffles : int, default 1000
        Permutations for the two nulls; 0 skips them.
    seed : int, default 0
        Random seed.
    bin_deg : float, default 15.0
        Width of the preferred-difference bins of `curve`.

    Returns
    -------
    RingTuning
        The answers.
    """
    stats_of = hd.stats if hasattr(hd, "stats") else hd
    screen = ring.screen
    if screen.empty:
        screen = pd.DataFrame(
            {"partner_snr": ring.partner_snr, "status": ON_RING},
            index=pd.Index(ring.unit_ids, name="unit_id"),
        )
    candidates = screen.index.to_numpy()

    def tuning(unit):
        unit_stats = stats_of.get(unit)
        if unit_stats is None:
            return np.nan, np.nan, False
        return (
            unit_stats.preferred_direction_deg,
            unit_stats.mean_vector_length,
            bool(unit_stats.significant),
        )

    measured = [tuning(u) for u in candidates.tolist()]
    table = pd.DataFrame(
        {
            "status": screen["status"].to_numpy(),
            "partner_snr": screen["partner_snr"].to_numpy(dtype=float),
            "preferred_deg": [m[0] for m in measured],
            "mvl": [m[1] for m in measured],
            "tuned": [m[2] for m in measured],
        },
        index=pd.Index(candidates, name="unit_id"),
    )
    table["tuned"] &= np.isfinite(table["preferred_deg"].to_numpy(dtype=float))
    on = pd.Series(ring.unit_ids)
    table["coupling"] = pd.Series(ring.coupling, index=on).reindex(table.index)
    table["radius"] = pd.Series(ring.radius, index=on).reindex(table.index)
    table["member"] = table["coupling"].fillna(-np.inf).to_numpy() >= min_coupling

    # the ring's own units, in its order, with their tuning
    preferred = table["preferred_deg"].reindex(ring.unit_ids).to_numpy(dtype=float)
    mvl = table["mvl"].reindex(ring.unit_ids).to_numpy(dtype=float)
    tuned = table["tuned"].reindex(ring.unit_ids).to_numpy(dtype=bool)

    angles = None
    aligned = np.full(ring.n_units, np.nan)
    structure_r, structure = float("nan"), None
    curve = pd.DataFrame(columns=["difference_deg", "mean_z", "sem_z", "n_pairs"])
    if tuned.sum() >= 3:
        angles = compare_angles(ring.angle_deg[tuned], preferred[tuned], n_shuffles, seed)
        aligned = angles.alignment.apply(ring.angle_deg)

        index = np.flatnonzero(tuned)
        corr = ring.corr_z[np.ix_(index, index)]
        cosine = np.cos(np.deg2rad(preferred[index][:, None] - preferred[index][None, :]))
        structure_r, structure = _structure_test(cosine, corr, n_shuffles, seed)
        upper = np.triu_indices(index.size, 1)
        difference = np.abs(
            circular_difference(preferred[index][:, None], preferred[index][None, :])
        )[upper]
        z = corr[upper]
        edges = np.arange(0.0, 180.0 + bin_deg, bin_deg)
        which = np.clip(np.digitize(difference, edges) - 1, 0, len(edges) - 2)
        rows = []
        for k in range(len(edges) - 1):
            values = z[which == k]
            if values.size:
                rows.append(
                    {
                        "difference_deg": (edges[k] + edges[k + 1]) / 2,
                        "mean_z": float(values.mean()),
                        "sem_z": float(values.std(ddof=1) / np.sqrt(values.size))
                        if values.size > 1
                        else float("nan"),
                        "n_pairs": int(values.size),
                    }
                )
        curve = pd.DataFrame(rows, columns=curve.columns)
    table["ring_angle_deg"] = pd.Series(aligned, index=on).reindex(table.index)
    table["error_deg"] = np.where(
        table["tuned"] & table["ring_angle_deg"].notna(),
        circular_difference(table["ring_angle_deg"], table["preferred_deg"]),
        np.nan,
    )

    def rho(x, y):
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return float("nan")
        return float(stats.spearmanr(x, y)[0])

    is_tuned = table["tuned"].to_numpy(dtype=bool)
    snr = table["partner_snr"].to_numpy(dtype=float)
    return RingTuning(
        table=table,
        angles=angles,
        auc_partner_snr=_auc(snr[is_tuned], snr[~is_tuned]),
        auc_coupling=_auc(ring.coupling[tuned], ring.coupling[~tuned]),
        auc_radius=_auc(ring.radius[tuned], ring.radius[~tuned]),
        radius_mvl_rho=rho(ring.radius, mvl),
        coupling_mvl_rho=rho(ring.coupling, mvl),
        structure_r=structure_r,
        structure=structure,
        curve=curve,
        min_coupling=min_coupling,
    )


@dataclass
class RingComparison:
    """Two rings compared on the units they share; see :func:`compare_rings`.

    Attributes
    ----------
    units_a, units_b : numpy.ndarray
        The paired units, in each ring's own ids.
    angles : AngleComparison
        Ring positions of the pairs, `b` aligned onto `a`.
    corr_a, corr_b : numpy.ndarray
        Pair metric (z) of every two paired units, in each ring, in the same
        order.
    structure_r : float
        Pearson r between `corr_a` and `corr_b`.
    structure : ShuffleTest or None
        `structure_r` against `b`'s units permuted.
    split_half_r_a, split_half_r_b : float
        Each ring's own split-half reliability: about as high as
        `structure_r` can be.
    """

    units_a: np.ndarray
    units_b: np.ndarray
    angles: AngleComparison
    corr_a: np.ndarray
    corr_b: np.ndarray
    structure_r: float
    structure: ShuffleTest | None
    split_half_r_a: float
    split_half_r_b: float

    @property
    def n(self) -> int:
        return len(self.units_a)

    def summary(self) -> str:
        text = f"  pair metric: r {self.structure_r:.2f}"
        if self.structure is not None:
            text += (
                f" vs permuted {self.structure.null.mean():.2f} +/- "
                f"{self.structure.null.std():.2f}, p = {self.structure.p_value:.4f}"
            )
        text += (
            f" (each ring's split-half r: {self.split_half_r_a:.2f}, "
            f"{self.split_half_r_b:.2f})"
        )
        return "\n".join(
            [f"{self.n} paired units", f"  ring positions: {self.angles}", text]
        )

    def as_row(self) -> dict:
        """Collect the headline numbers as one flat row.

        Returns
        -------
        dict
            The row.
        """
        row = {
            "n_pairs": self.n,
            "angle_circular_r": self.angles.circular_r,
            "angle_median_abs_error_deg": self.angles.median_abs_error_deg,
            "angle_flipped": self.angles.alignment.flip,
            "angle_offset_deg": self.angles.alignment.offset_deg,
            "structure_r": self.structure_r,
            "split_half_r_a": self.split_half_r_a,
            "split_half_r_b": self.split_half_r_b,
        }
        if self.angles.shuffle is not None:
            row["angle_null_mean"] = float(self.angles.shuffle.null.mean())
            row["angle_p"] = self.angles.shuffle.p_value
        if self.structure is not None:
            row["structure_null_mean"] = float(self.structure.null.mean())
            row["structure_p"] = self.structure.p_value
        return row

    def __repr__(self) -> str:
        return self.summary()


def compare_rings(
    ring_a: RingModel,
    ring_b: RingModel,
    unit_map: dict | None = None,
    n_shuffles: int = 1000,
    seed: int = 0,
) -> RingComparison:
    """Ask whether two rings agree on the units they share.

    Two tests, neither of which uses tuning:

    - positions: do paired units sit at the same places on both rings, up to
      a reflection and a rotation?
    - structure: are paired units correlated alike in both -- a Mantel test of
      the pair metric over the paired units? This one needs no embedding, so
      it still answers where one ring is too degraded to read.

    Each against a null that re-pairs the units at random. Between two
    recordings, with `unit_map` from the matching, that asks whether the
    matches are the same cells; within one recording -- wake against REM, say
    -- whether the structure is the same in both states.

    Parameters
    ----------
    ring_a, ring_b : RingModel
        The rings.
    unit_map : dict, optional
        ``{unit of a: unit of b}``. Pairs with a unit missing from its ring
        are skipped. By default each unit the two rings share is paired with
        itself.
    n_shuffles : int, default 1000
        Permutations per null; 0 skips them.
    seed : int, default 0
        Random seed.

    Returns
    -------
    RingComparison
        The comparison.

    Raises
    ------
    ValueError
        If fewer than four pairs are on both rings.
    """
    on_a = set(ring_a.unit_ids.tolist())
    on_b = set(ring_b.unit_ids.tolist())
    if unit_map is None:
        pairs = [(u, u) for u in ring_a.unit_ids.tolist() if u in on_b]
    else:
        pairs = [(a, b) for a, b in unit_map.items() if a in on_a and b in on_b]
    if len(pairs) < 4:
        raise ValueError(f"{len(pairs)} pairs of units are on both rings: need at least 4")
    units_a = np.asarray([a for a, _ in pairs])
    units_b = np.asarray([b for _, b in pairs])
    index_a = _unit_positions(ring_a.unit_ids, units_a.tolist())
    index_b = _unit_positions(ring_b.unit_ids, units_b.tolist())

    angles = compare_angles(
        ring_b.angle_deg[index_b], ring_a.angle_deg[index_a], n_shuffles, seed
    )
    corr_a = ring_a.corr_z[np.ix_(index_a, index_a)]
    corr_b = ring_b.corr_z[np.ix_(index_b, index_b)]
    structure_r, structure = _structure_test(corr_a, corr_b, n_shuffles, seed)
    upper = np.triu_indices(len(pairs), 1)
    return RingComparison(
        units_a=units_a,
        units_b=units_b,
        angles=angles,
        corr_a=corr_a[upper],
        corr_b=corr_b[upper],
        structure_r=structure_r,
        structure=structure,
        split_half_r_a=ring_a.split_half_r,
        split_half_r_b=ring_b.split_half_r,
    )


# ── looking at it ────────────────────────────────────────────────────────
def _binned_mean(x, y, edges):
    """Mean of `y` in bins of `x`.

    Parameters
    ----------
    x, y : numpy.ndarray
        Values.
    edges : numpy.ndarray
        Bin edges.

    Returns
    -------
    centers, means : numpy.ndarray
        Bins with at least one value.
    """
    which = np.digitize(x, edges) - 1
    centers, means = [], []
    for k in range(len(edges) - 1):
        values = y[(which == k) & np.isfinite(y)]
        if values.size:
            centers.append((edges[k] + edges[k + 1]) / 2)
            means.append(values.mean())
    return np.asarray(centers), np.asarray(means)


def _identity_lines(ax):
    """Draw y = x on a 0-360 square, wrapped round both edges.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    """
    for shift in (-360.0, 0.0, 360.0):
        ax.plot([0, 360], [shift, 360 + shift], color="0.5", ls="--", lw=0.8)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.set_yticks(np.arange(0, 361, 90))


def plot_ring(ring: RingModel, color=None, cmap="hsv", axes=None):
    """Plot the ring, its sorted correlations, its consistency and its eigenvalues.

    MATLAB's Figs 1d and 2, plus the eigenvalue check:

    - the units in the 2-D embedding, colored by ring angle (or by `color`),
      members ringed in black;
    - the pair metric (z), sorted by ring angle: a ring is a bright band
      along the diagonal that wraps round at the corners;
    - the pair metric against distance on the ring, with its binned mean:
      high for neighbours, low for opposites;
    - the leading MDS eigenvalues: a ring gives two nearly equal ones well
      clear of the third, noise a flat run of them.

    Parameters
    ----------
    ring : RingModel
        The ring.
    color : dict or array-like, optional
        A value per unit (``{unit: value}`` or in ring order) to color the
        embedding by -- preferred direction, say. Default the ring angle.
    cmap : str, default "hsv"
        Colormap of the embedding.
    axes : sequence of matplotlib.axes.Axes, optional
        Four axes to draw on.

    Returns
    -------
    sequence of matplotlib.axes.Axes
        The four axes.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(17.0, 4.0))
    embedding_ax, matrix_ax, consistency_ax, eigen_ax = axes

    if color is None:
        values, label = ring.angle_deg, "ring angle (deg)"
    elif isinstance(color, dict):
        values = np.array(
            [color.get(u, np.nan) for u in ring.unit_ids.tolist()], dtype=float
        )
        label = ""
    else:
        values, label = np.asarray(color, dtype=float), ""
    member = np.isin(ring.unit_ids, ring.members())
    points = embedding_ax.scatter(
        ring.embedding[:, 0],
        ring.embedding[:, 1],
        c=values,
        cmap=cmap,
        s=np.where(member, 40, 18),
        edgecolors=np.where(member, "k", "none"),
        linewidths=0.6,
        plotnonfinite=True,
    )
    embedding_ax.figure.colorbar(points, ax=embedding_ax, label=label, shrink=0.8)
    embedding_ax.set_aspect("equal", adjustable="datalim")
    embedding_ax.set_title(
        f"{ring.n_units} units ({member.sum()} members), stress {ring.stress:.2f}",
        fontsize=10,
    )
    embedding_ax.set_xlabel("Isomap 1")
    embedding_ax.set_ylabel("Isomap 2")

    order = np.argsort(ring.angle_deg, kind="stable")
    matrix = ring.corr_z[np.ix_(order, order)]
    limit = float(np.nanpercentile(np.abs(matrix), 98)) or 1.0
    colormap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="0.85")
    image = matrix_ax.imshow(
        matrix, cmap=colormap, vmin=-limit, vmax=limit, interpolation="nearest"
    )
    matrix_ax.figure.colorbar(image, ax=matrix_ax, label="pair metric (z)", shrink=0.8)
    matrix_ax.set_title("sorted by ring angle", fontsize=10)
    matrix_ax.set_xlabel("unit")
    matrix_ax.set_ylabel("unit")

    upper = np.triu_indices(ring.n_units, 1)
    distance = np.abs(
        circular_difference(ring.angle_deg[:, None], ring.angle_deg[None, :])
    )[upper]
    z = ring.corr_z[upper]
    consistency_ax.scatter(
        distance, z, s=4, color="k", alpha=min(1.0, 300.0 / max(z.size, 1)), lw=0
    )
    centers, means = _binned_mean(distance, z, np.arange(0.0, 181.0, 15.0))
    consistency_ax.plot(centers, means, color="#e41a1c", lw=2)
    consistency_ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    consistency_ax.set_xlim(0, 180)
    consistency_ax.set_xticks(np.arange(0, 181, 45))
    consistency_ax.set_xlabel("distance on the ring (deg)")
    consistency_ax.set_ylabel("pair metric (z)")
    consistency_ax.set_title(f"consistency r = {ring.consistency_r:.2f}", fontsize=10)

    shown = ring.eigenvalues[: min(8, ring.eigenvalues.size)] / ring.eigenvalues[0]
    eigen_ax.bar(
        np.arange(1, shown.size + 1),
        shown,
        color=["#377eb8"] * 2 + ["0.6"] * (shown.size - 2),
    )
    eigen_ax.axhline(0.0, color="0.5", lw=0.8)
    eigen_ax.set_xticks(np.arange(1, shown.size + 1))
    eigen_ax.set_xlabel("MDS dimension")
    eigen_ax.set_ylabel("eigenvalue / first")
    eigen_ax.set_title(f"second / first = {ring.eigenvalue_ratio:.2f}", fontsize=10)
    for ax in (consistency_ax, eigen_ax):
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].figure.tight_layout()
    return axes


def plot_correlograms(data: DecoderData, ring: RingModel, mask, pairs=None, axes=None):
    """Plot the lagged correlation of a few pairs, with the baseline the metric subtracts.

    MATLAB's Fig 1a-c: by default the most correlated pair, one nearest zero
    and the most anti-correlated, by the ring's own metric. The shaded lags
    are the shoulder and its mean the dashed line; the metric is the dot at 0
    minus the line.

    Parameters
    ----------
    data : DecoderData
        Binned data holding the pairs.
    ring : RingModel
        The ring, for its settings and to choose pairs.
    mask : array-like of bool
        Bins to correlate within -- the ring's fitting bins, to see what it saw.
    pairs : sequence of (unit, unit), optional
        Pairs to draw, from the ring's units.
    axes : sequence of matplotlib.axes.Axes, optional
        One axes per pair.

    Returns
    -------
    sequence of matplotlib.axes.Axes
        The axes.
    """
    import matplotlib.pyplot as plt

    z = ring.corr_z
    if pairs is None:
        upper = np.triu_indices(ring.n_units, 1)
        values = z[upper]
        picks = [np.nanargmax(values), np.nanargmin(np.abs(values)), np.nanargmin(values)]
        pairs = [(ring.unit_ids[upper[0][k]], ring.unit_ids[upper[1][k]]) for k in picks]
    if axes is None:
        _, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    start = ring.params.get("baseline_start_s", 5.0)
    max_lag = ring.params.get("max_lag_s", 10.0)
    for ax, (a, b) in zip(axes, pairs):
        lags, r = pair_correlogram(
            data, mask, a, b, ring.params.get("smooth_sigma_s", 0.05), max_lag
        )
        shoulder = np.abs(lags) >= start - 1e-9
        baseline = float(np.nanmean(r[shoulder]))
        ax.axvspan(-max_lag, -start, color="0.92", lw=0)
        ax.axvspan(start, max_lag, color="0.92", lw=0)
        ax.plot(lags, r, color="k", lw=1.2)
        ax.axhline(baseline, color="#e41a1c", ls="--", lw=1, label="baseline")
        ax.plot(0.0, r[lags.size // 2], "o", color="#e41a1c")
        ax.axvline(0.0, color="0.6", ls=":", lw=0.8)
        i, j = _unit_positions(ring.unit_ids, [a, b])
        ax.set_title(f"units {a} & {b}: z = {z[i, j]:.1f}", fontsize=10)
        ax.set_xlabel("lag (s)")
        ax.set_xlim(-max_lag, max_lag)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("correlation (r)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].figure.tight_layout()
    return axes


def plot_ring_vs_tuning(result: RingTuning, axes=None):
    """Plot how the ring lines up with tuning: positions, coupling, the pair metric.

    MATLAB's Fig 3, with coupling in place of radius.

    Parameters
    ----------
    result : RingTuning
        From :func:`ring_vs_tuning`.
    axes : sequence of matplotlib.axes.Axes, optional
        Three axes to draw on.

    Returns
    -------
    sequence of matplotlib.axes.Axes
        The three axes.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    angle_ax, coupling_ax, curve_ax = axes
    table = result.table
    shown = table[table["tuned"] & table["ring_angle_deg"].notna()]

    points = angle_ax.scatter(
        shown["preferred_deg"],
        shown["ring_angle_deg"],
        c=shown["mvl"],
        cmap="viridis",
        s=30,
        edgecolors="k",
        linewidths=0.4,
    )
    angle_ax.figure.colorbar(points, ax=angle_ax, label="MVL", shrink=0.8)
    _identity_lines(angle_ax)
    angle_ax.set_xlabel("preferred direction (deg)")
    angle_ax.set_ylabel("ring position, aligned (deg)")
    if result.angles is not None:
        title = (
            f"circular r {result.angles.circular_r:.2f}, median |error| "
            f"{result.angles.median_abs_error_deg:.0f} deg"
        )
        if result.angles.shuffle is not None:
            title += f", p = {result.angles.shuffle.p_value:.4f}"
        angle_ax.set_title(title, fontsize=10)

    on_ring = table[table["coupling"].notna()]
    for flag, color, name in ((False, "0.6", "untuned"), (True, "#e41a1c", "tuned")):
        group = on_ring[on_ring["tuned"] == flag]
        coupling_ax.scatter(
            group["mvl"], group["coupling"], s=24, color=color, label=name, lw=0
        )
    coupling_ax.axhline(
        result.min_coupling, color="0.4", ls="--", lw=0.8, label="member cut"
    )
    coupling_ax.set_xlabel("mean vector length (tuning)")
    coupling_ax.set_ylabel("coupling to the ring")
    coupling_ax.set_title(
        f"coupling AUC {result.auc_coupling:.2f}, partner-SNR AUC "
        f"{result.auc_partner_snr:.2f}",
        fontsize=10,
    )
    coupling_ax.legend(fontsize=8, frameon=False)

    curve = result.curve
    if len(curve):
        curve_ax.errorbar(
            curve["difference_deg"],
            curve["mean_z"],
            yerr=curve["sem_z"],
            color="k",
            marker="o",
            ms=4,
            capsize=2,
        )
    curve_ax.axhline(0.0, color="#e41a1c", ls="--", lw=0.8)
    curve_ax.set_xlim(0, 180)
    curve_ax.set_xticks(np.arange(0, 181, 45))
    curve_ax.set_xlabel("preferred-direction difference (deg)")
    curve_ax.set_ylabel("pair metric (z)")
    title = f"tuned pairs: r {result.structure_r:.2f}"
    if result.structure is not None:
        title += f", p = {result.structure.p_value:.4f}"
    curve_ax.set_title(title, fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].figure.tight_layout()
    return axes


def plot_ring_comparison(comparison: RingComparison, labels=("a", "b"), axes=None):
    """Plot two rings' agreement on their paired units, each test beside its null.

    Parameters
    ----------
    comparison : RingComparison
        From :func:`compare_rings`.
    labels : tuple of str, default ("a", "b")
        Names of the two rings.
    axes : sequence of matplotlib.axes.Axes, optional
        Four axes to draw on.

    Returns
    -------
    sequence of matplotlib.axes.Axes
        The four axes.
    """
    import matplotlib.pyplot as plt

    from .decoder import plot_shuffle

    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(17.0, 3.9))
    angle_ax, angle_null_ax, corr_ax, corr_null_ax = axes
    a, b = labels
    angles = comparison.angles

    angle_ax.scatter(angles.reference_deg, angles.aligned_deg, s=28, color="k", lw=0)
    _identity_lines(angle_ax)
    angle_ax.set_xlabel(f"ring position in {a} (deg)")
    angle_ax.set_ylabel(f"in {b}, aligned (deg)")
    angle_ax.set_title(
        f"{comparison.n} pairs: circular r {angles.circular_r:.2f}", fontsize=10
    )
    if angles.shuffle is not None:
        plot_shuffle(angles.shuffle, ax=angle_null_ax)
    else:
        angle_null_ax.set_axis_off()

    corr_ax.scatter(comparison.corr_a, comparison.corr_b, s=10, color="k", alpha=0.5, lw=0)
    corr_ax.set_xlabel(f"pair metric in {a} (z)")
    corr_ax.set_ylabel(f"pair metric in {b} (z)")
    corr_ax.set_title(
        f"r {comparison.structure_r:.2f} (split-half {comparison.split_half_r_a:.2f}, "
        f"{comparison.split_half_r_b:.2f})",
        fontsize=10,
    )
    if comparison.structure is not None:
        plot_shuffle(comparison.structure, ax=corr_null_ax)
    else:
        corr_null_ax.set_axis_off()
    for ax in (angle_ax, corr_ax):
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].figure.tight_layout()
    return axes
