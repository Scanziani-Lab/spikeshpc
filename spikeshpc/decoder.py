"""Bayesian decoding of head direction from head-direction-tuned units.

Taken from Moritz's ``run_decoder.py``. a sorted-spikes point process decoder
on a ring state space. Per-unit tuning curves are the encoding model, the
likelihood of a time bin is Poisson given those curves, and a Gaussian random
walk on the ring supplies the dynamics that carry belief from one bin to the next.
The readout is the MAP of the posterior.

It is reimplemented here rather than delegated to
``replay_trajectory_classification``, which supplied the decoder there. RTC is
unmaintained, does not install on this environment's Python, and its ring
"track graph" is scaffolding for a general linearized-track machinery that head
direction does not need -- a circle is already one-dimensional and periodic.
The maths below is the same; what it drops is a dependency and the
interpolation layer RTC's fixed-rate time grid demanded.

Typical use, from a notebook that already has the objects
``1_calculate_HD_tuning.ipynb`` builds::

    run = run_decoder(
        analyzer, tuned_unit_ids, heading_deg, shutter_close_times,
        scoring.intervals, test_fraction=0.3,
    )
    print(run.test.metrics)
    plot_shuffle(run.test_shuffle)
    plot_decoded(run.test)
    plot_decoded(run.rem)          # only meaningful if the test looked good
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import correlate1d, gaussian_filter1d

from .optitrack.tuning import (
    _bin_headings,
    compute_hd_tuning_curve,
    compute_mean_vector_length,
)
from .states import frames_in_states

__all__ = [
    "Decoded",
    "DecoderData",
    "DecoderRun",
    "EncodingModel",
    "ShuffleTest",
    "TransferRun",
    "apply_decoder",
    "bins_in_interval_mask",
    "decode",
    "fit_encoding_model",
    "metrics_by_group",
    "plot_decoded",
    "plot_encoding_model",
    "plot_error",
    "plot_metrics_by_group",
    "plot_shuffle",
    "plot_transfer_summary",
    "prepare_decoder_data",
    "reference_decode",
    "restrict_data",
    "restrict_model",
    "ring_transition",
    "run_decoder",
    "shuffle_test",
    "split_train_test",
    "state_interval_mask",
]


# ── angles ───────────────────────────────────────────────────────────────
def circular_difference(a_deg, b_deg) -> np.ndarray:
    """Compute the signed difference a - b, wrapped into (-180, 180].

    Parameters
    ----------
    a_deg, b_deg : array-like
        Angles in degrees.

    Returns
    -------
    numpy.ndarray
        Difference in degrees.
    """
    return (np.asarray(a_deg, float) - np.asarray(b_deg, float) + 180.0) % 360.0 - 180.0


def circular_mean(angles_deg, axis=None) -> np.ndarray:
    """Compute the mean direction from the resultant vector.

    Parameters
    ----------
    angles_deg : array-like
        Angles in degrees.
    axis : int, optional
        Axis to average over; all by default.

    Returns
    -------
    numpy.ndarray
        Mean direction in [0, 360), in degrees.
    """
    radians = np.deg2rad(np.asarray(angles_deg, dtype=float))
    resultant = np.exp(1j * radians).mean(axis=axis)
    return np.rad2deg(np.angle(resultant)) % 360.0


def circular_correlation(a_deg, b_deg) -> float:
    """Jammalamadaka's circular correlation coefficient, in [-1, 1].

    The circular analogue of Pearson's r: sine deviations from each variable's
    own mean direction, rather than linear deviations from its mean. Using
    Pearson's r on raw angles instead would be wrecked by the 0/360 wrap --
    a decoder that is perfect except for tracking across north would score
    near zero.

    Parameters
    ----------
    a_deg, b_deg : array-like
        Angles in degrees, of equal length.

    Returns
    -------
    float
        Coefficient in [-1, 1]; NaN if undefined.
    """
    a = np.deg2rad(np.asarray(a_deg, dtype=float))
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if a.size < 2:
        return float("nan")

    da = np.sin(a - np.angle(np.exp(1j * a).mean()))
    db = np.sin(b - np.angle(np.exp(1j * b).mean()))
    denominator = np.sqrt(np.sum(da**2) * np.sum(db**2))
    if not denominator > 0:
        return float("nan")
    return float(np.sum(da * db) / denominator)


# ── binned inputs ────────────────────────────────────────────────────────
@dataclass
class DecoderData:
    """Spike counts and heading on the decoder's own time grid.

    One row per decoder bin. A bin spans ``bin_frames`` camera frames, so
    ``edges`` are real shutter-closure timestamps and ``duration_s`` is
    measured rather than assumed -- there is no interpolation anywhere between
    the camera and the decoder.

    ``heading_deg`` is the circular mean of the headings of the frames inside
    the bin (identical to the tuning-curve convention of heading-at-interval-
    start when ``bin_frames == 1``).

    Attributes
    ----------
    counts : numpy.ndarray
        Spike counts, shape (n_bins, n_units).
    heading_deg : numpy.ndarray
        Heading per bin, in degrees.
    duration_s : numpy.ndarray
        Measured duration of each bin, in seconds.
    time_s : numpy.ndarray
        Start time of each bin, in seconds.
    edges : numpy.ndarray
        Bin edges, in seconds.
    unit_ids : numpy.ndarray
        Units, in the order of the columns of `counts`.
    bin_frames : int
        Camera frames per bin.
    """

    counts: np.ndarray  # (n_bins, n_units), integer spike counts
    heading_deg: np.ndarray  # (n_bins,)
    duration_s: np.ndarray  # (n_bins,)
    time_s: np.ndarray  # (n_bins,) bin centre on the recording clock
    edges: np.ndarray  # (n_bins + 1,) shutter-closure timestamps
    unit_ids: np.ndarray
    bin_frames: int

    @property
    def n_bins(self) -> int:
        return len(self.duration_s)

    @property
    def n_units(self) -> int:
        return self.counts.shape[1]

    @property
    def bin_s(self) -> float:
        """Get the nominal bin width -- the median, since frame intervals wobble.

        Returns
        -------
        float
            Bin width, in seconds.
        """
        return float(np.median(self.duration_s))

    def __repr__(self) -> str:
        return (
            f"DecoderData({self.n_bins} bins of {1e3 * self.bin_s:.1f} ms, "
            f"{self.n_units} units, {self.duration_s.sum():.0f}s)"
        )


def _as_sorting(obj):
    """Accept either a sorting or a sorting analyzer.

    Parameters
    ----------
    obj : BaseSorting or SortingAnalyzer
        Object to unwrap.

    Returns
    -------
    BaseSorting
        The sorting.
    """
    return obj.sorting if hasattr(obj, "sorting") else obj


def _unit_positions(have, want) -> np.ndarray:
    """Find where each wanted unit sits in a list of available ones.

    Parameters
    ----------
    have : sequence
        Available unit ids.
    want : sequence
        Unit ids to locate.

    Returns
    -------
    numpy.ndarray of int
        Position of each of `want` in `have`.

    Raises
    ------
    ValueError
        If any unit is not in `have`.
    """
    position = {u: i for i, u in enumerate(np.asarray(have).tolist())}
    missing = [u for u in want if u not in position]
    if missing:
        raise ValueError(
            f"units {missing[:10]} are not among the {len(position)} held here"
        )
    return np.array([position[u] for u in want], dtype=int)


def _frames_per_bin(bin_s: float, frame_s: float, tolerance: float = 0.01) -> int:
    """Count how many camera frames fit in a bin, forgiving a near-miss.

    Truncating the ratio is the obvious thing and it is wrong exactly where it
    is most often used. Asking for a bin of a whole number of frames -- 1/60 s
    on a 120 fps camera, say -- puts the ratio on a knife edge, and a camera
    that actually runs at 119.99 Hz drops it to 1.9998, which truncates to one
    frame and silently halves every bin. Real cameras are never on their
    nominal rate: this rig's are 120.008 and 59.9989 Hz.

    So a ratio within ``tolerance`` of a whole number is taken as that whole
    number, and anything else still rounds down. Asking for 1.5 frames still
    gets one; asking for what you thought was two gets two; asking for less
    than a frame gets one, since a bin has to hold something.

    Parameters
    ----------
    bin_s : float
        Target bin width, in seconds.
    frame_s : float
        Camera frame interval, in seconds.
    tolerance : float, default 0.01
        How close to a whole number the ratio may be.

    Returns
    -------
    int
        Frames per bin, at least 1.
    """
    ratio = bin_s / frame_s
    nearest = round(ratio)
    if nearest >= 1 and abs(ratio - nearest) <= tolerance * nearest:
        return nearest
    return max(1, int(ratio))


def prepare_decoder_data(
    sorting,
    unit_ids,
    heading_deg,
    frame_times,
    bin_s: float = 0.05,
) -> DecoderData:
    """Bin spikes and heading onto a grid of whole camera frames.

    ``heading_deg`` and ``frame_times`` are the shutter-aligned arrays from
    ``1_calculate_HD_tuning.ipynb``: same length, one entry per shutter-closure
    event, on the recording's clock. ``unit_ids`` are the units to decode from
    -- the head-direction-tuned ones -- and their order is preserved
    throughout.

    Parameters
    ----------
    sorting : spikeinterface BaseSorting or SortingAnalyzer
        Source of the spike trains.
    unit_ids : sequence
        Units to decode from -- the head-direction-tuned ones. Their order is
        preserved throughout.
    heading_deg, frame_times : array-like
        The shutter-aligned arrays from ``1_calculate_HD_tuning.ipynb``: same
        length, one entry per shutter-closure event, on the recording's clock.
    bin_s : float, default 0.05
        A *target*: the grid is the largest whole number of camera frames not
        exceeding it, so bin edges stay on measured timestamps. At 120 fps the
        50 ms default is 6 frames.

    Returns
    -------
    DecoderData
        Counts and heading on the decoder's grid.

    Raises
    ------
    ValueError
        If there are no units or the inputs are inconsistent.
    """
    sorting = _as_sorting(sorting)
    heading_deg = np.asarray(heading_deg, dtype=float)
    frame_times = np.asarray(frame_times, dtype=float)
    unit_ids = np.asarray(list(unit_ids))

    if heading_deg.shape != frame_times.shape:
        raise ValueError(
            f"heading_deg ({heading_deg.shape}) and frame_times "
            f"({frame_times.shape}) must be the same length -- both are indexed "
            "by shutter-closure event"
        )
    if len(unit_ids) == 0:
        raise ValueError("no units to decode from")

    frame_s = float(np.median(np.diff(frame_times)))
    bin_frames = _frames_per_bin(bin_s, frame_s)
    n_bins = (len(frame_times) - 1) // bin_frames
    if n_bins < 2:
        raise ValueError(
            f"bin_s={bin_s} over {len(frame_times)} frames leaves {n_bins} bins"
        )

    edges = frame_times[: n_bins * bin_frames + 1 : bin_frames]
    duration_s = np.diff(edges)
    time_s = (edges[:-1] + edges[1:]) / 2

    # heading of each bin: the circular mean of the frames it spans. Frames
    # past the last whole bin are dropped rather than folded into a short one.
    used = heading_deg[: n_bins * bin_frames].reshape(n_bins, bin_frames)
    binned_heading = used[:, 0] if bin_frames == 1 else circular_mean(used, axis=1)

    counts = np.empty((n_bins, len(unit_ids)), dtype=np.int32)
    for j, unit_id in enumerate(unit_ids):
        spike_times = sorting.get_unit_spike_train(unit_id, return_times=True)
        counts[:, j], _ = np.histogram(spike_times, bins=edges)

    return DecoderData(
        counts=counts,
        heading_deg=binned_heading,
        duration_s=duration_s,
        time_s=time_s,
        edges=edges,
        unit_ids=unit_ids,
        bin_frames=bin_frames,
    )


def state_interval_mask(data: DecoderData, intervals, states=("WAKE",)) -> np.ndarray:
    """Find which decoder bins lie wholly inside the given states.

    A bin survives only if both of its edges do, which is the same rule
    :func:`spikeshpc.states.intervals_between_frames` applies to inter-frame
    intervals: a bin straddling a state boundary belongs to neither state.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    intervals : dict
        ``{state: [[start, stop], ...]}``.
    states : tuple of str, default ("WAKE",)
        States to keep.

    Returns
    -------
    numpy.ndarray of bool
        One value per bin.
    """
    _, keep = frames_in_states(data.edges, intervals, states)
    return keep


def bins_in_interval_mask(data: DecoderData, interval_mask) -> np.ndarray:
    """Find which decoder bins have every one of their frame intervals in a mask.

    A bin spans ``bin_frames`` of the inter-frame intervals and survives only
    if all of them do. :func:`state_interval_mask` can only test a bin's two
    edges, which is enough for state epochs that last minutes but lets through
    a frame excluded from the middle of a bin.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    interval_mask : array-like of bool
        Mask over the inter-frame intervals of the ``frame_times`` the data
        was prepared from -- the mask the tuning functions take, e.g. from
        :func:`spikeshpc.states.behavior_interval_mask`.

    Returns
    -------
    numpy.ndarray of bool
        One value per bin.

    Raises
    ------
    ValueError
        If the mask does not match the data.
    """
    interval_mask = np.asarray(interval_mask, dtype=bool)
    n_used = data.n_bins * data.bin_frames
    # prepare_decoder_data drops the frames past the last whole bin, so the
    # full mask is up to bin_frames - 1 intervals longer than the bins use
    if (
        interval_mask.ndim != 1
        or not n_used <= len(interval_mask) < n_used + data.bin_frames
    ):
        raise ValueError(
            f"interval_mask has {interval_mask.shape} entries, which is not the "
            f"inter-frame intervals of the frames behind {data.n_bins} bins of "
            f"{data.bin_frames} frames. It masks intervals, not frames or bins."
        )
    return interval_mask[:n_used].reshape(data.n_bins, data.bin_frames).all(axis=1)


def restrict_data(data: DecoderData, unit_ids) -> DecoderData:
    """Restrict the data to some units, keeping the same bins.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    unit_ids : sequence
        Units to keep, in the order wanted.

    Returns
    -------
    DecoderData
        Data holding only those units' counts, in that order.
    """
    unit_ids = list(unit_ids)
    index = _unit_positions(data.unit_ids, unit_ids)
    return replace(
        data, counts=data.counts[:, index], unit_ids=np.asarray(data.unit_ids)[index]
    )


# ── train / test split ───────────────────────────────────────────────────
def split_train_test(
    data: DecoderData,
    mask: np.ndarray,
    test_fraction: float = 0.3,
    mode: str = "blocks",
    block_s: float = 60.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide the bins under `mask` into training and held-out testing halves.

    ``mode="blocks"`` (default) cuts the masked time into contiguous blocks of
    ``block_s`` and assigns whole blocks at random. Blocks, not individual
    bins: neighbouring bins share the same head direction and often the same
    burst, so a bin-wise split would let the decoder be tested on data it had
    effectively already seen, and every score would come out flattering. A
    minute is far longer than the animal holds a heading, so the blocks
    themselves are near-independent while still sampling the whole session --
    which matters because tuning drifts over hours, and a decoder trained only
    on the first part of a session is being asked a harder question than the
    one you meant to ask.

    ``mode="contiguous"`` instead holds out the last ``test_fraction`` of the
    masked time in one piece. That *is* the harder question -- it measures
    whether the encoding model still holds later on -- so it is the honest
    choice when you care about applying the decoder to a later block (REM after
    a wake session, say). Expect it to score worse.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    mask : numpy.ndarray of bool
        Bins to divide.
    test_fraction : float, default 0.3
        Fraction of the masked time held out, in (0, 1).
    mode : {"blocks", "contiguous"}, default "blocks"
        How to carve the held-out part.
    block_s : float, default 60.0
        Block length for ``mode="blocks"``, in seconds.
    seed : int, default 0
        Seed for assigning blocks.

    Returns
    -------
    train, test : numpy.ndarray of bool
        Arrays over all bins; both are subsets of `mask` and they do not
        overlap.

    Raises
    ------
    ValueError
        If `test_fraction` or `mode` is invalid, or `mask` keeps no bins.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (data.n_bins,):
        raise ValueError(
            f"mask has {mask.shape} entries but there are {data.n_bins} bins"
        )
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if not mask.any():
        raise ValueError("mask keeps no bins at all")

    index = np.flatnonzero(mask)
    elapsed = np.cumsum(data.duration_s[index]) - data.duration_s[index]
    total = float(data.duration_s[index].sum())

    train = np.zeros(data.n_bins, dtype=bool)
    test = np.zeros(data.n_bins, dtype=bool)

    if mode == "contiguous":
        is_test = elapsed >= total * (1.0 - test_fraction)
    elif mode == "blocks":
        block = (elapsed // block_s).astype(int)
        n_blocks = int(block.max()) + 1
        if n_blocks < 4:
            warnings.warn(
                f"{n_blocks} block(s) of {block_s}s in {total:.0f}s of data: the "
                "split is coarse. Shorten block_s or use mode='contiguous'.",
                stacklevel=2,
            )
        order = np.random.default_rng(seed).permutation(n_blocks)
        n_test = max(1, round(test_fraction * n_blocks))
        is_test = np.isin(block, order[:n_test])
    else:
        raise ValueError(f"mode must be 'blocks' or 'contiguous', got {mode!r}")

    test[index[is_test]] = True
    train[index[~is_test]] = True

    if not train.any() or not test.any():
        raise ValueError(
            f"test_fraction={test_fraction} left one side empty "
            f"({train.sum()} train, {test.sum()} test bins)"
        )
    return train, test


# ── encoding model ───────────────────────────────────────────────────────
@dataclass
class EncodingModel:
    """Per-unit firing rate as a function of head direction: p(spikes | angle).

    ``rate_hz`` is ``(n_units, n_angle_bins)``, the same occupancy-normalized,
    circularly smoothed tuning curve that
    :func:`spikeshpc.optitrack.tuning.compute_hd_tuning_curve` builds and that
    the units were selected on -- so what the decoder believes about a unit is
    exactly the curve you looked at.

    Attributes
    ----------
    bin_centers_deg : numpy.ndarray
        Centres of the heading bins.
    rate_hz : numpy.ndarray
        Firing rate, shape (n_units, n_angle_bins).
    unit_ids : numpy.ndarray
        Units, in row order.
    occupancy_s : numpy.ndarray
        Training time spent in each heading bin, in seconds.
    train_time_s : float
        Total training time, in seconds.
    min_rate_hz : float
        Floor applied to the curves.
    """

    bin_centers_deg: np.ndarray
    rate_hz: np.ndarray
    unit_ids: np.ndarray
    occupancy_s: np.ndarray
    train_time_s: float
    min_rate_hz: float

    @property
    def n_angle_bins(self) -> int:
        return len(self.bin_centers_deg)

    @property
    def preferred_deg(self) -> np.ndarray:
        return np.array(
            [
                compute_mean_vector_length(self.bin_centers_deg, rate)[1]
                for rate in self.rate_hz
            ]
        )

    @property
    def mean_vector_length(self) -> np.ndarray:
        return np.array(
            [
                compute_mean_vector_length(self.bin_centers_deg, rate)[0]
                for rate in self.rate_hz
            ]
        )

    def __repr__(self) -> str:
        return (
            f"EncodingModel({len(self.unit_ids)} units x {self.n_angle_bins} "
            f"angle bins, trained on {self.train_time_s:.0f}s)"
        )


def fit_encoding_model(
    data: DecoderData,
    mask: np.ndarray,
    n_angle_bins: int = 180,
    smooth_sigma_deg: float = 10.0,
    min_rate_hz: float = 0.1,
    min_occupancy_s: float = 1.0,
) -> EncodingModel:
    """Fit tuning curves for every unit, from the bins under `mask` only.

    ``min_rate_hz`` floors the curves. Without it an angle a unit happened
    never to fire at has rate zero, its log-likelihood is -inf, and a single
    spike there vetoes that heading outright however much the rest of the
    population likes it -- one unit's sampling gap becoming a hard constraint.
    The floor makes a surprising spike merely expensive.

    Angle bins the animal barely visited during training are reported in
    ``occupancy_s``; a warning names them, because a curve there is an estimate
    from almost nothing and the decoder has no way to know that.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    mask : numpy.ndarray of bool
        Training bins.
    n_angle_bins : int, default 180
        Number of heading bins.
    smooth_sigma_deg : float, default 10.0
        Width of the circular smoothing, in degrees.
    min_rate_hz : float, default 0.1
        Floor under the curves.
    min_occupancy_s : float, default 1.0
        Heading bins visited for less than this are reported in a warning.

    Returns
    -------
    EncodingModel
        The fitted model.

    Raises
    ------
    ValueError
        If `mask` keeps no bins or the inputs are inconsistent.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (data.n_bins,):
        raise ValueError(
            f"mask has {mask.shape} entries but there are {data.n_bins} bins"
        )
    if not mask.any():
        raise ValueError("mask keeps no bins to train on")

    heading = data.heading_deg[mask]
    duration = data.duration_s[mask]
    counts = data.counts[mask]

    _, bin_idx = _bin_headings(heading, n_angle_bins)
    occupancy_s = np.bincount(bin_idx, weights=duration, minlength=n_angle_bins)

    rates = np.empty((data.n_units, n_angle_bins))
    for j in range(data.n_units):
        with np.errstate(invalid="ignore", divide="ignore"):
            firing_rate = np.where(duration > 0, counts[:, j] / duration, 0.0)
        bin_centers_deg, rates[j] = compute_hd_tuning_curve(
            heading,
            firing_rate,
            occupancy_time=duration,
            n_bins=n_angle_bins,
            smooth_sigma_deg=smooth_sigma_deg,
        )

    thin = int(np.count_nonzero(occupancy_s < min_occupancy_s))
    if thin:
        warnings.warn(
            f"{thin}/{n_angle_bins} head-direction bins have under "
            f"{min_occupancy_s}s of training occupancy; the decoder can still "
            "report those headings but has almost no evidence about them.",
            stacklevel=2,
        )

    return EncodingModel(
        bin_centers_deg=bin_centers_deg,
        rate_hz=np.maximum(rates, min_rate_hz),
        unit_ids=data.unit_ids,
        occupancy_s=occupancy_s,
        train_time_s=float(duration.sum()),
        min_rate_hz=min_rate_hz,
    )


def restrict_model(model: EncodingModel, unit_ids) -> EncodingModel:
    """Restrict an encoding model to some units.

    Each unit's curve is fitted from its own spikes alone, so this is exactly
    the model that fitting those units by themselves would have produced --
    which is what lets one baseline fit serve every subset of it.

    Parameters
    ----------
    model : EncodingModel
        Model to restrict.
    unit_ids : sequence
        Units to keep, in the order wanted.

    Returns
    -------
    EncodingModel
        The model with only those units.
    """
    unit_ids = list(unit_ids)
    index = _unit_positions(model.unit_ids, unit_ids)
    return replace(
        model, rate_hz=model.rate_hz[index], unit_ids=np.asarray(model.unit_ids)[index]
    )


# ── dynamics ─────────────────────────────────────────────────────────────
def movement_variance(
    data: DecoderData,
    mask: np.ndarray | None = None,
    per_100hz: float = 2.0,
) -> float:
    """Compute the random-walk variance in deg^2 per decoder bin.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    mask : numpy.ndarray of bool, optional
        If given, the variance is measured: the variance of the frame-to-frame
        change in heading, scaled to the bin width. If None it is the
        `per_100hz` convention carried over from ``run_decoder.py``.
    per_100hz : float, default 2.0
        deg^2 per 10 ms bin, scaled linearly with bin duration as diffusion
        requires.

    Returns
    -------
    float
        Variance in deg^2 per bin.

    Raises
    ------
    ValueError
        If there are not enough adjacent bins under `mask` to measure it.
    """
    if mask is None:
        return float(per_100hz) * data.bin_s * 100.0

    steps = circular_difference(data.heading_deg[mask][1:], data.heading_deg[mask][:-1])
    # only steps between adjacent bins are one bin's worth of movement
    adjacent = np.diff(np.flatnonzero(mask)) == 1
    steps = steps[adjacent]
    if steps.size < 2:
        raise ValueError("not enough adjacent bins under the mask to measure movement")
    return float(np.var(steps))


def ring_transition(
    n_angle_bins: int, movement_var_deg2: float, leak: float = 1e-12
) -> np.ndarray:
    """Build p(angle now | angle one bin ago) as a wrapped Gaussian random walk.

    Row-stochastic and symmetric, since the ring is uniform: row ``i`` is the
    kernel centred on bin ``i``, wrapped at 0/360 so that north has neighbours
    on both sides.

    ``leak`` mixes in a whisker of uniform. A Gaussian kernel a hundred bins
    from its centre is not merely small but exactly zero in floating point, so
    without the leak a confident posterior can end up assigning literally zero
    probability to the truth and never recover from it -- one tracking glitch
    and the decode is locked out for the rest of the run. The leak says the
    head can teleport, very rarely, which is both numerically safe and a fair
    description of what a lost frame looks like to the decoder.

    Pass ``movement_var_deg2 = inf`` for a uniform transition -- no dynamics at
    all, every bin decoded independently from its own spikes. That is the
    control worth running when a decoded trajectory looks suspiciously smooth:
    a strong random-walk prior can manufacture smoothness out of noise, and if
    the uniform decode still tracks the animal then the smoothness was real.

    Parameters
    ----------
    n_angle_bins : int
        Number of heading bins, at least 2.
    movement_var_deg2 : float
        Variance of the random walk, in deg^2 per bin; ``inf`` gives a uniform
        transition.
    leak : float, default 1e-12
        Weight of the uniform component.

    Returns
    -------
    numpy.ndarray
        Row-stochastic matrix, shape (n_angle_bins, n_angle_bins).

    Raises
    ------
    ValueError
        If there are fewer than 2 angle bins.
    """
    if n_angle_bins < 2:
        raise ValueError("need at least 2 angle bins")
    if not np.isfinite(movement_var_deg2):
        return np.full((n_angle_bins, n_angle_bins), 1.0 / n_angle_bins)
    if movement_var_deg2 <= 0:
        transition = np.eye(n_angle_bins)
    else:
        bin_width_deg = 360.0 / n_angle_bins
        sigma_bins = np.sqrt(movement_var_deg2) / bin_width_deg
        if sigma_bins < 0.5:
            warnings.warn(
                f"random-walk sd is {sigma_bins:.2f} angle bins "
                f"({np.sqrt(movement_var_deg2):.2f} deg vs {bin_width_deg:.2f} "
                "deg bins): the discretized walk barely leaves its bin per step "
                "and will under-diffuse. Use fewer angle bins or a longer bin_s.",
                stacklevel=2,
            )
        transition = gaussian_filter1d(
            np.eye(n_angle_bins), sigma=sigma_bins, axis=1, mode="wrap"
        )

    transition /= transition.sum(axis=1, keepdims=True)
    if leak:
        transition = (1.0 - leak) * transition + leak / n_angle_bins
    return transition


# ── the decode itself ────────────────────────────────────────────────────
def poisson_log_likelihood(
    counts: np.ndarray, duration_s: np.ndarray, rate_hz: np.ndarray
) -> np.ndarray:
    """Compute log p(spikes in bin | head direction), up to a constant, per (bin, angle).

    Units are taken to be conditionally independent given head direction and
    Poisson within a bin, so the population log-likelihood is the sum over
    units of ``n log(lambda dt) - lambda dt``. Terms that do not depend on the
    angle -- ``log(dt)`` and ``log(n!)`` -- are dropped: they shift every
    column of a row by the same amount and vanish in the normalization.

    Parameters
    ----------
    counts : numpy.ndarray
        Spike counts, shape (n_bins, n_units).
    duration_s : numpy.ndarray
        Duration of each bin, in seconds.
    rate_hz : numpy.ndarray
        Tuning curves, shape (n_units, n_angle_bins).

    Returns
    -------
    numpy.ndarray
        Log-likelihood, shape (n_bins, n_angle_bins).
    """
    counts = np.asarray(counts, dtype=float)
    rate_hz = np.asarray(rate_hz, dtype=float)
    expected = np.asarray(duration_s, dtype=float)[:, None] * rate_hz.sum(axis=0)
    return counts @ np.log(rate_hz) - expected


def _runs(mask: np.ndarray, duration_s: np.ndarray, max_gap_s: float, min_run_s: float):
    """Find the contiguous stretches of a mask.

    Belief propagates along a run and is reset at each new one: the prior says
    the head has not moved far since the previous bin, which is a claim about
    an adjacent bin and nothing else. Carrying it across the hour between two
    wake epochs would assert the animal ended where it began.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        Bins to decode.
    duration_s : numpy.ndarray
        Duration of each bin, in seconds.
    max_gap_s : float
        Largest gap, in seconds, that still joins two bins into one run.
    min_run_s : float
        Shortest run kept.

    Returns
    -------
    list of tuple of slice or int
        ``(start, stop)`` into all bins, one per stretch.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    mask &= duration_s <= max_gap_s  # a stretched bin is a hole in the tracking

    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [
        (int(a), int(b))
        for a, b in zip(starts, stops)
        if duration_s[a:b].sum() >= min_run_s
    ]


class _RingStep:
    """``p @ transition`` and ``transition @ v`` for rows of posteriors.

    A :func:`ring_transition` is circulant: every row is the first one turned.
    And it is a narrow Gaussian band over a uniform floor (the leak), so a
    product with it is a short wrapped filter plus the floor times the row's
    total -- a few dozen multiply-adds per angle bin where the dense product
    streams the whole matrix through memory at every step. Any other
    transition, or a band too wide to gain from this, is used as a matrix.

    Rows are independent: a row of a batch comes out exactly as it would on
    its own, which is what lets shuffles be decoded in batches.

    Parameters
    ----------
    transition : numpy.ndarray
        Circulant transition matrix from :func:`ring_transition`.
    """

    def __init__(self, transition):
        transition = np.asarray(transition, dtype=float)
        n = transition.shape[0]
        self.n = n
        self.dense = transition
        self.dense_t = np.ascontiguousarray(transition.T)
        self.uniform = np.full((1, n), 1.0 / n)
        self.fast = False
        row = transition[0]
        turned = np.stack([np.roll(row, i) for i in range(n)])
        if not np.allclose(turned, transition, rtol=1e-12, atol=0):
            return
        floor = row.min()
        band = row - floor
        offsets = (np.flatnonzero(band > 0) + n // 2) % n - n // 2
        reach = int(np.abs(offsets).max()) if offsets.size else 0
        if 2 * reach + 1 > n // 4:
            return
        d = np.arange(-reach, reach + 1)
        # correlate1d: out[j] = sum_k w[k] x[j + k - reach]
        self.forward_weights = band[(-d) % n]  # sum_i p[i] T[i, j]
        self.backward_weights = band[d % n]  # sum_j T[i, j] v[j]
        self.floor = floor
        self.fast = True

    def forward(self, p):
        if not self.fast:
            return p @ self.dense
        return correlate1d(
            p, self.forward_weights, axis=-1, mode="wrap"
        ) + self.floor * p.sum(axis=-1, keepdims=True)

    def backward(self, v):
        if not self.fast:
            return v @ self.dense_t
        return correlate1d(
            v, self.backward_weights, axis=-1, mode="wrap"
        ) + self.floor * v.sum(axis=-1, keepdims=True)


def _normalize_rows(p, uniform):
    """Normalise each row; a row with nothing in it becomes uniform.

    Parameters
    ----------
    p : numpy.ndarray
        Non-negative rows.
    uniform : numpy.ndarray
        Row substituted for an empty one.

    Returns
    -------
    numpy.ndarray
        Rows summing to 1.
    """
    total = p.sum(axis=1, keepdims=True)
    if (total > 0).all():
        return p / total
    empty = ~(total[:, 0] > 0)
    p = p / np.where(empty[:, None], 1.0, total)
    p[empty] = uniform
    return p


def _forward_backward(log_likelihood, transition, acausal=True):
    """Compute causal and acausal posteriors over one run, by the HMM forward-backward.

    The causal posterior at bin t uses spikes up to t only, so it is what a
    decoder running live would report. The acausal one also uses everything
    after t; it is better, and it is the right choice for offline analysis, but
    it cannot be read as a prediction.

    ``transition`` is a matrix or a :class:`_RingStep`. Each step works on a
    one-row batch, exactly as :func:`_shuffle_null` steps its batches.

    Parameters
    ----------
    log_likelihood : numpy.ndarray
        Shape (n_bins, n_angle_bins), from :func:`poisson_log_likelihood`.
    transition : numpy.ndarray or _RingStep
        Transition model.
    acausal : bool, default True
        Also compute the smoothed posterior.

    Returns
    -------
    causal : numpy.ndarray
        Posterior using spikes up to each bin only.
    smoothed : numpy.ndarray or None
        Posterior using all spikes; None if not `acausal`.
    """
    step = transition if isinstance(transition, _RingStep) else _RingStep(transition)
    n_t, n_x = log_likelihood.shape
    # Subtracting each row's max before exponentiating keeps the likelihood
    # away from underflow; it is a per-row constant, so the normalized
    # posterior is unchanged.
    likelihood = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
    uniform = step.uniform

    causal = np.empty((n_t, n_x))
    posterior = uniform
    for t in range(n_t):
        prior = posterior if t == 0 else step.forward(posterior)
        posterior = _normalize_rows(prior * likelihood[t : t + 1], uniform)
        causal[t] = posterior[0]

    if not acausal:
        return causal, None

    smoothed = np.empty((n_t, n_x))
    smoothed[-1] = causal[-1]
    backward = np.ones((1, n_x))
    for t in range(n_t - 2, -1, -1):
        backward = _normalize_rows(
            step.backward(likelihood[t + 1 : t + 2] * backward), uniform
        )
        smoothed[t] = _normalize_rows(causal[t : t + 1] * backward, uniform)[0]
    return causal, smoothed


@dataclass
class Decoded:
    """One decoded stretch: the MAP angle per bin and how sure it was.

    ``actual_deg`` is the measured heading. During REM it is whatever the
    OptiTrack rigid body was pointing at while the animal slept, so it is not
    ground truth for anything and ``metrics`` is left empty.

    Attributes
    ----------
    label : str
        Name of the decode.
    time_s, duration_s : numpy.ndarray
        Start time and duration of each decoded bin, in seconds.
    decoded_deg : numpy.ndarray
        MAP heading per bin, in degrees.
    actual_deg : numpy.ndarray
        Measured heading per bin, in degrees.
    error_deg : numpy.ndarray
        Signed circular error per bin, in degrees.
    posterior_max : numpy.ndarray
        Peak of the posterior per bin: how sure the decoder was.
    entropy_bits : numpy.ndarray
        Entropy of the posterior per bin, in bits.
    n_spikes : numpy.ndarray
        Spikes per bin.
    run_index : numpy.ndarray
        Which contiguous stretch each bin belongs to.
    bin_index : numpy.ndarray
        Index of each bin in the full grid.
    bin_centers_deg : numpy.ndarray
        Centres of the heading bins.
    posterior : numpy.ndarray or None
        Full posterior, if kept.
    metrics : dict
        Output of :func:`decoding_metrics`.
    """

    label: str
    time_s: np.ndarray
    duration_s: np.ndarray
    decoded_deg: np.ndarray
    actual_deg: np.ndarray
    error_deg: np.ndarray  # signed, decoded - actual, in (-180, 180]
    posterior_max: np.ndarray  # probability at the MAP bin
    entropy_bits: np.ndarray  # of the full posterior; low = confident
    n_spikes: np.ndarray  # summed over units, per bin
    run_index: np.ndarray
    bin_index: np.ndarray  # back into the DecoderData
    bin_centers_deg: np.ndarray
    posterior: np.ndarray | None = None  # (n_bins, n_angle_bins), float32
    metrics: dict = field(default_factory=dict)

    @property
    def n_decoded(self) -> int:
        return len(self.time_s)

    def __repr__(self) -> str:
        head = (
            f"Decoded({self.label}: {self.n_decoded} bins, {self.duration_s.sum():.0f}s"
        )
        if "median_abs_error_deg" in self.metrics:
            head += f", median error {self.metrics['median_abs_error_deg']:.1f} deg"
        return head + ")"


def decoding_metrics(decoded_deg, actual_deg, tolerance_deg: float = 30.0) -> dict:
    """Score how close the decode came, by four measures that fail differently.

    The median absolute error is the headline because it is what a reader
    pictures and because it survives the occasional bin where the population
    falls silent and the posterior wanders. The mean and the RMSE are given
    alongside precisely because they do not: the gap between median and RMSE is
    the size of the tail. ``frac_within_deg`` is the same tail question asked
    directly, and the circular correlation is the only one of the four that a
    constant offset -- a miscalibrated head frame -- would not punish.

    ``offset_deg`` is that offset: the circular mean of the error. The
    ``*_corrected_deg`` pair asks the median and tolerance questions again with
    it removed. A head-direction population that turned as a whole between two
    recordings is read by a decoder fitted on the first at a constant offset,
    which the raw error scores as failure and the corrected error does not. On
    a model's own held-out data the offset is near zero and the two agree.

    Parameters
    ----------
    decoded_deg, actual_deg : array-like
        Decoded and measured heading, in degrees.
    tolerance_deg : float, default 30.0
        Error counted as within tolerance.

    Returns
    -------
    dict
        Median, mean and RMS absolute error, ``frac_within_deg``, the circular
        correlation, ``offset_deg`` and the ``*_corrected_deg`` pair; empty if
        there is nothing finite to score.
    """
    error = circular_difference(decoded_deg, actual_deg)
    error = error[np.isfinite(error)]
    if error.size == 0:
        return {}
    absolute = np.abs(error)
    offset = float(circular_difference(circular_mean(error), 0.0))
    corrected = np.abs(circular_difference(error, offset))
    return {
        "median_abs_error_deg": float(np.median(absolute)),
        "mean_abs_error_deg": float(absolute.mean()),
        "rmse_deg": float(np.sqrt(np.mean(error**2))),
        "frac_within_deg": float(np.mean(absolute <= tolerance_deg)),
        "tolerance_deg": float(tolerance_deg),
        "circular_correlation": circular_correlation(decoded_deg, actual_deg),
        "offset_deg": offset,
        "median_abs_error_corrected_deg": float(np.median(corrected)),
        "frac_within_corrected_deg": float(np.mean(corrected <= tolerance_deg)),
        "n_bins": int(error.size),
    }


def decode(
    data: DecoderData,
    model: EncodingModel,
    mask: np.ndarray,
    movement_var_deg2: float | None = None,
    acausal: bool = True,
    max_gap_s: float = 1.0,
    min_run_s: float = 1.0,
    keep_posterior: bool = True,
    with_metrics: bool = True,
    tolerance_deg: float = 30.0,
    label: str = "decode",
) -> Decoded:
    """Decode head direction in the bins under `mask`.

    ``movement_var_deg2`` defaults to the ``run_decoder.py`` convention for
    this bin width (see :func:`movement_variance`); pass ``np.inf`` to drop the
    dynamics prior and decode each bin from its own spikes alone.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    model : EncodingModel
        Fitted model.
    mask : numpy.ndarray of bool
        Bins to decode.
    movement_var_deg2 : float, optional
        Random-walk variance, in deg^2 per bin. Defaults to the
        ``run_decoder.py`` convention for this bin width (see
        :func:`movement_variance`); ``np.inf`` drops the dynamics prior.
    acausal : bool, default True
        Use spikes after each bin as well as before.
    max_gap_s : float, default 1.0
        Largest gap that still joins two bins into one stretch.
    min_run_s : float, default 1.0
        Shortest stretch decoded.
    keep_posterior : bool, default True
        Keep the full posterior in the result.
    with_metrics : bool, default True
        Score the decode against the measured heading.
    tolerance_deg : float, default 30.0
        Tolerance for ``frac_within_deg``.
    label : str, default "decode"
        Name of the decode.

    Returns
    -------
    Decoded
        The decoded stretch.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or no bin can be decoded.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (data.n_bins,):
        raise ValueError(
            f"mask has {mask.shape} entries but there are {data.n_bins} bins"
        )
    if not np.array_equal(np.asarray(model.unit_ids), np.asarray(data.unit_ids)):
        raise ValueError(
            "the model was fitted on different units (or a different order) "
            "than this data holds"
        )

    if movement_var_deg2 is None:
        movement_var_deg2 = movement_variance(data)
    transition = ring_transition(model.n_angle_bins, movement_var_deg2)

    runs = _runs(mask, data.duration_s, max_gap_s, min_run_s)
    if not runs:
        raise ValueError(
            f"no run of masked bins reaches min_run_s={min_run_s} "
            f"(after dropping bins longer than max_gap_s={max_gap_s})"
        )

    posteriors, indices, run_ids = [], [], []
    for run_id, (a, b) in enumerate(runs):
        # per run rather than for every bin up front: a decode only visits the
        # masked bins, and the whole recording's likelihood is n_bins x
        # n_angle_bins floats -- gigabytes for a night of 60 Hz bins
        log_likelihood = poisson_log_likelihood(
            data.counts[a:b], data.duration_s[a:b], model.rate_hz
        )
        causal, smoothed = _forward_backward(
            log_likelihood, transition, acausal=acausal
        )
        posteriors.append(smoothed if acausal else causal)
        indices.append(np.arange(a, b))
        run_ids.append(np.full(b - a, run_id))

    posterior = np.concatenate(posteriors)
    bin_index = np.concatenate(indices)
    run_index = np.concatenate(run_ids)

    best = posterior.argmax(axis=1)
    decoded_deg = model.bin_centers_deg[best]
    actual_deg = data.heading_deg[bin_index]
    entropy = _entropy_bits(posterior)

    result = Decoded(
        label=label,
        time_s=data.time_s[bin_index],
        duration_s=data.duration_s[bin_index],
        decoded_deg=decoded_deg,
        actual_deg=actual_deg,
        error_deg=circular_difference(decoded_deg, actual_deg),
        posterior_max=posterior[np.arange(len(best)), best],
        entropy_bits=entropy,
        n_spikes=data.counts[bin_index].sum(axis=1),
        run_index=run_index,
        bin_index=bin_index,
        bin_centers_deg=model.bin_centers_deg,
        posterior=posterior.astype(np.float32) if keep_posterior else None,
    )
    result.metrics = _decode_metrics(
        decoded_deg,
        actual_deg,
        result.posterior_max,
        entropy,
        with_metrics,
        tolerance_deg,
    )
    return result


def _entropy_bits(posterior: np.ndarray) -> np.ndarray:
    """Compute the entropy of each posterior.

    Parameters
    ----------
    posterior : numpy.ndarray
        Probabilities along the last axis.

    Returns
    -------
    numpy.ndarray
        Entropy in bits.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(posterior > 0, posterior * np.log2(posterior), 0.0)
    return -terms.sum(axis=-1)


def _decode_metrics(
    decoded_deg, actual_deg, posterior_max, entropy_bits, with_metrics, tolerance_deg
) -> dict:
    """Compute a decode's ``metrics``, shared by :func:`decode` and the shuffle workers.

    Parameters
    ----------
    decoded_deg, actual_deg : numpy.ndarray
        Decoded and measured heading, in degrees.
    posterior_max, entropy_bits : numpy.ndarray
        Per-bin confidence measures.
    with_metrics : bool
        Score against the measured heading.
    tolerance_deg : float
        Tolerance for ``frac_within_deg``.

    Returns
    -------
    dict
        The metrics.
    """
    metrics = (
        decoding_metrics(decoded_deg, actual_deg, tolerance_deg) if with_metrics else {}
    )
    metrics["mean_posterior_max"] = float(np.mean(posterior_max))
    if entropy_bits is not None:
        metrics["mean_entropy_bits"] = float(np.mean(entropy_bits))
    return metrics


def metrics_by_group(
    decoded: Decoded,
    labels,
    order=None,
    extra: dict | None = None,
    tolerance_deg: float | None = None,
) -> dict:
    """Score the decode separately for each group of its bins.

    The question is *where* a decode fails rather than how often, and which
    explanation the pattern fits. A group that is wrong but confident is a
    population reporting some other heading; one that is wrong and unsure had
    too little to go on, and ``population_rate_hz`` says whether that is
    because the units went quiet.

    Parameters
    ----------
    decoded : Decoded
        The decode.
    labels : array-like
        One entry per decoded bin: "moving"/"still", say, or a binned
        :func:`spikeshpc.states.seconds_since`.
    order : sequence, optional
        Order of the groups; defaults to the sorted labels.
    extra : dict, optional
        Per-bin arrays whose group means are added to the metrics.
    tolerance_deg : float, optional
        Defaults to the one `decoded` was scored with, so the groups'
        ``frac_within_deg``, weighted by ``n_bins``, average back to the whole
        decode's.

    Returns
    -------
    dict
        ``{label: metrics}`` in `order`; labels with no bins are left out.
        Each ``metrics`` is :func:`decoding_metrics` on that group's bins plus:

        - ``time_s``: how much decoded time the group holds
        - ``q25_abs_error_deg``, ``q75_abs_error_deg``: the spread around the
          median, for error bars
        - ``mean_posterior_max``: how sure the decoder was
        - ``population_rate_hz``: summed spikes over time, the evidence it had
        - ``<name>``: the group mean of each per-bin array in `extra`

    Raises
    ------
    ValueError
        If `labels` or `extra` do not match the decoded bins.
    """
    labels = np.asarray(labels)
    if labels.shape != (decoded.n_decoded,):
        raise ValueError(
            f"labels has {labels.shape} entries but {decoded.label!r} has "
            f"{decoded.n_decoded} decoded bins"
        )
    extra = {name: np.asarray(v, dtype=float) for name, v in (extra or {}).items()}
    for name, values in extra.items():
        if values.shape != (decoded.n_decoded,):
            raise ValueError(
                f"extra[{name!r}] has {values.shape} entries but {decoded.label!r} "
                f"has {decoded.n_decoded} decoded bins"
            )
    if tolerance_deg is None:
        tolerance_deg = decoded.metrics.get("tolerance_deg", 30.0)
    if order is None:
        order = sorted(set(labels.tolist()))

    table = {}
    for label in order:
        keep = labels == label
        if not keep.any():
            continue
        time_s = float(decoded.duration_s[keep].sum())
        metrics = decoding_metrics(
            decoded.decoded_deg[keep], decoded.actual_deg[keep], tolerance_deg
        )
        absolute = np.abs(decoded.error_deg[keep])
        absolute = absolute[np.isfinite(absolute)]
        if absolute.size:
            q25, q75 = np.percentile(absolute, [25, 75])
            metrics["q25_abs_error_deg"] = float(q25)
            metrics["q75_abs_error_deg"] = float(q75)
        metrics["time_s"] = time_s
        metrics["mean_posterior_max"] = float(decoded.posterior_max[keep].mean())
        metrics["population_rate_hz"] = (
            float(decoded.n_spikes[keep].sum() / time_s) if time_s > 0 else float("nan")
        )
        for name, values in extra.items():
            finite = values[keep][np.isfinite(values[keep])]
            metrics[name] = float(finite.mean()) if finite.size else float("nan")
        table[label] = metrics
    return table


# ── the control ──────────────────────────────────────────────────────────
@dataclass
class ShuffleTest:
    """An observed statistic against the null distribution it is judged by.

    Attributes
    ----------
    metric : str
        Name of the statistic.
    observed : float
        Its observed value.
    null : numpy.ndarray
        Its value under each shuffle.
    p_value : float
        Fraction of the null at least as good as the observed value.
    better : {"lower", "higher"}
        Which direction counts as better.
    kind : str
        Kind of shuffle: "shift" or "units".
    """

    metric: str
    observed: float
    null: np.ndarray
    p_value: float
    better: str  # "lower" or "higher": which direction counts as success
    kind: str  # how the null was generated

    @property
    def z_score(self) -> float:
        spread = self.null.std()
        if not spread > 0:
            return float("nan")
        return float((self.observed - self.null.mean()) / spread)

    def __str__(self) -> str:
        return (
            f"{self.metric} = {self.observed:.3f} vs null "
            f"{self.null.mean():.3f} +/- {self.null.std():.3f} "
            f"(n={len(self.null)}, {self.better} is better), "
            f"p = {self.p_value:.4f}, z = {self.z_score:+.1f}"
        )


def shuffle_test(
    data: DecoderData,
    model: EncodingModel,
    mask: np.ndarray,
    kind: str = "shift",
    metric: str = "median_abs_error_deg",
    n_shuffles: int = 100,
    min_shift_s: float = 30.0,
    seed: int = 0,
    progress: bool = False,
    n_jobs: int | None = None,
    max_batch_bytes: int = 2**30,
    movement_var_deg2: float | None = None,
    acausal: bool = True,
    max_gap_s: float = 1.0,
    min_run_s: float = 1.0,
    with_metrics: bool = True,
    tolerance_deg: float = 30.0,
) -> ShuffleTest:
    """Re-decode many times with the coding destroyed, for comparison.

    Each shuffle is decoded exactly as :func:`decode` would decode it. Two
    things make that fast.

    First, `n_jobs` worker processes split the shuffles between them. Every
    shift or permutation is drawn up front, in the order the one-at-a-time loop
    drew them, so the null does not depend on `n_jobs`.

    Second, within a worker, shuffles are decoded in batches, stepping every
    one of them through the forward-backward together: neither kind of shuffle
    changes which bins are decoded, so one step is a single (batch x angle) @
    (angle x angle) product rather than a vector product per shuffle. A batch
    holds each shuffle's causal posterior over a run, ``8 x run bins x angle
    bins`` bytes, and `max_batch_bytes` caps that per worker, so the batches
    are smaller over long runs. Batching changes the order of floating-point
    sums, so a null can differ from a one-at-a-time decode's in the last
    digits, and where two headings tie, in the MAP bin.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    model : EncodingModel
        Fitted model.
    mask : numpy.ndarray of bool
        Bins decoded for the observed statistic.
    kind : {"shift", "units"}, default "shift"
        ``"shift"`` circularly shifts the spike data against the heading by at
        least `min_shift_s`. Every unit moves together, so firing rates, burst
        structure, population synchrony and the animal's occupancy all survive
        intact and only their alignment in time is broken. That is the null
        for "does this decoder track the animal": use it wherever there is a
        measured heading to be wrong about, i.e. on the wake test set.
        ``"units"`` instead permutes which tuning curve belongs to which unit.
        Rates and synchrony again survive; what breaks is the population code
        itself. This is the null to use on REM, where there is no heading to
        misalign against and the question is whether the posterior is sharper
        and more coherent than a scrambled population would make it -- so pair
        it with a metric like ``mean_posterior_max``.
    metric : str or sequence of str, default "median_abs_error_deg"
        Statistic to judge. A sequence judges several statistics on the same
        shuffled decodes -- the raw and the offset-corrected error, say --
        without paying for the decodes twice.
    n_shuffles : int, default 100
        Number of shuffles.
    min_shift_s : float, default 30.0
        Smallest shift, in seconds.
    seed : int, default 0
        Random seed.
    progress : bool, default False
        Show a progress bar.
    n_jobs : int, optional
        Worker processes. Default is one per CPU core, capped by the number of
        shuffles; 1 runs them here.
    max_batch_bytes : int, default 2**30
        Memory cap per worker for one batch.
    movement_var_deg2, acausal, max_gap_s, min_run_s, with_metrics, tolerance_deg
        As for :func:`decode`.

    Returns
    -------
    ShuffleTest or dict of str to ShuffleTest
        One test, or ``{metric: ShuffleTest}`` in order when `metric` is a
        sequence.

    Raises
    ------
    ValueError
        If `kind` is unknown, no metric is given, or the inputs are
        inconsistent.
    """
    if kind not in ("shift", "units"):
        raise ValueError(f"kind must be 'shift' or 'units', got {kind!r}")
    metrics = [metric] if isinstance(metric, str) else list(metric)
    if not metrics:
        raise ValueError("no metric to test")

    decode_kwargs = dict(
        movement_var_deg2=movement_var_deg2,
        acausal=acausal,
        max_gap_s=max_gap_s,
        min_run_s=min_run_s,
        with_metrics=with_metrics,
        tolerance_deg=tolerance_deg,
    )
    observed = decode(
        data, model, mask, keep_posterior=False, label="observed", **decode_kwargs
    )
    for name in metrics:
        if name not in observed.metrics:
            raise ValueError(
                f"metric {name!r} is not one of {sorted(observed.metrics)}"
            )

    # every draw up front, in the order the one-at-a-time loop made them
    rng = np.random.default_rng(seed)
    identity = np.arange(len(model.unit_ids))
    if kind == "shift":
        min_shift = max(1, round(min_shift_s / data.bin_s))
        if n_shuffles and 2 * min_shift >= data.n_bins:
            raise ValueError(
                f"min_shift_s={min_shift_s} leaves no room to shift "
                f"{data.duration_s.sum():.0f}s of data"
            )
        shifts = [
            int(rng.integers(min_shift, data.n_bins - min_shift))
            for _ in range(n_shuffles)
        ]
        orders = [identity] * n_shuffles
    else:
        shifts = [0] * n_shuffles
        orders = [rng.permutation(len(model.unit_ids)) for _ in range(n_shuffles)]

    if movement_var_deg2 is None:
        movement_var_deg2 = movement_variance(data)
    runs = _runs(mask, data.duration_s, max_gap_s, min_run_s)
    bin_index = np.concatenate([np.arange(a, b) for a, b in runs])
    job = dict(
        counts=data.counts,
        duration_s=data.duration_s,
        actual_deg=data.heading_deg[bin_index],
        runs=runs,
        rate_hz=model.rate_hz,
        bin_centers_deg=model.bin_centers_deg,
        transition=ring_transition(model.n_angle_bins, movement_var_deg2),
        acausal=acausal,
        metrics=metrics,
        with_metrics=with_metrics,
        tolerance_deg=tolerance_deg,
        max_batch_bytes=max_batch_bytes,
    )

    n_workers = min(n_shuffles, _default_jobs() if n_jobs is None else max(1, n_jobs))
    chunks = np.array_split(np.arange(n_shuffles), max(1, n_workers))
    null = np.empty((len(metrics), n_shuffles))
    if n_workers <= 1:
        if n_shuffles:
            null[:] = _shuffle_null(shifts=shifts, orders=orders, **job)
    else:
        with _single_threaded_children(), ProcessPoolExecutor(n_workers) as pool:
            futures = {
                pool.submit(
                    _shuffle_null,
                    shifts=[shifts[i] for i in chunk],
                    orders=[orders[i] for i in chunk],
                    **job,
                ): chunk
                for chunk in chunks
            }
            done = 0
            for future in as_completed(futures):
                null[:, futures[future]] = future.result()
                done += len(futures[future])
                if progress:
                    print(f"      shuffle {done}/{n_shuffles}", end="\r")

    tests = {
        name: _judge(name, observed.metrics[name], null[k], kind)
        for k, name in enumerate(metrics)
    }
    return tests[metric] if isinstance(metric, str) else tests


def _default_jobs() -> int:
    """Count the CPU cores this process may use.

    Returns
    -------
    int
        Worker count, at least 1.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # Windows, macOS
        return max(1, os.cpu_count() or 1)


_CHUNK_BINS = 2048  # bins of likelihood a shuffle worker holds at a time

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def _single_threaded_children():
    """Start worker processes with one BLAS thread each.

    Every worker already has a core to itself; a BLAS that also spreads each
    small product over every core just has the workers fight over them. A
    child reads these when it imports numpy, so they are set only while the
    pool starts its workers, and this process's own BLAS is untouched.

    Yields
    ------
    None
        Control, while the environment variables are set.
    """
    saved = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    os.environ.update({name: "1" for name in _THREAD_VARIABLES})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _shuffle_null(
    counts,
    duration_s,
    actual_deg,
    runs,
    rate_hz,
    bin_centers_deg,
    transition,
    acausal,
    metrics,
    with_metrics,
    tolerance_deg,
    max_batch_bytes,
    shifts,
    orders,
) -> np.ndarray:
    """Score each shuffle's decode.

    Shuffle ``k`` decodes ``np.roll(counts, shifts[k])`` with the tuning
    curves reordered by ``orders[k]``, over the same runs as the observed
    decode, and is scored as :func:`decode` scores it. Module level, so worker
    processes can import it.

    Parameters
    ----------
    counts : numpy.ndarray
        Spike counts, shape (n_bins, n_units).
    duration_s : numpy.ndarray
        Duration of each bin, in seconds.
    actual_deg : numpy.ndarray
        Measured heading per bin, in degrees.
    runs : list
        Stretches decoded, from :func:`_runs`.
    rate_hz : numpy.ndarray
        Tuning curves, shape (n_units, n_angle_bins).
    bin_centers_deg : numpy.ndarray
        Centres of the heading bins.
    transition : numpy.ndarray or _RingStep
        Transition model.
    acausal : bool
        Use the acausal posterior.
    metrics : sequence of str
        Names of the statistics to compute.
    with_metrics : bool
        Score against the measured heading.
    tolerance_deg : float
        Tolerance for ``frac_within_deg``.
    max_batch_bytes : int
        Memory cap for one batch.
    shifts : sequence of int
        Roll applied to `counts` in each shuffle.
    orders : sequence of numpy.ndarray
        Unit permutation applied to the curves in each shuffle.

    Returns
    -------
    numpy.ndarray
        Statistic of each shuffle, shape (n_metrics, n_shuffles).
    """
    n_shuffles = len(shifts)
    n_bins = len(duration_s)
    n_angles = transition.shape[0]
    n_decoded = sum(b - a for a, b in runs)
    rates = [rate_hz[order] for order in orders]
    want_entropy = "mean_entropy_bits" in metrics
    best = np.empty((n_shuffles, n_decoded), dtype=np.intp)
    peak = np.empty((n_shuffles, n_decoded))
    entropy = np.empty((n_shuffles, n_decoded)) if want_entropy else None
    step = _RingStep(transition)
    uniform = step.uniform

    def likelihood(ks, a, b):
        """Compute exp(log-likelihood - row max) of bins a:b, for shuffles `ks`.

        Parameters
        ----------
        ks : sequence of int
            Shuffles to compute.
        a, b : int
            Range of bins.

        Returns
        -------
        numpy.ndarray
            Scaled likelihoods.
        """
        out = np.empty((len(ks), b - a, n_angles))
        for j, k in enumerate(ks):
            rows = (np.arange(a, b) - shifts[k]) % n_bins  # np.roll(counts, shift)[a:b]
            log_likelihood = poisson_log_likelihood(
                counts[rows], duration_s[a:b], rates[k]
            )
            out[j] = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
        return out

    def record(batch, position, posterior):
        best[batch, position] = which = posterior.argmax(axis=1)
        peak[batch, position] = posterior[np.arange(len(which)), which]
        if want_entropy:
            entropy[batch, position] = _entropy_bits(posterior)

    chunk = _CHUNK_BINS
    start = 0
    for a, b in runs:
        n_t = b - a
        # the causal posteriors the backward pass reads back
        per_shuffle = 8 * n_angles * (n_t if acausal else min(n_t, chunk))
        size = int(min(n_shuffles, max(1, max_batch_bytes // per_shuffle)))
        for first in range(0, n_shuffles, size):
            batch = slice(first, min(first + size, n_shuffles))
            ks = range(n_shuffles)[batch]
            causal = np.empty((len(ks), n_t, n_angles)) if acausal else None
            posterior = None
            for c in range(a, b, chunk):
                lik = likelihood(ks, c, min(c + chunk, b))
                for i in range(lik.shape[1]):
                    t = c - a + i
                    prior = uniform if t == 0 else step.forward(posterior)
                    posterior = _normalize_rows(prior * lik[:, i], uniform)
                    if acausal:
                        causal[:, t] = posterior
                    else:
                        record(batch, start + t, posterior)
            if acausal:
                record(batch, start + n_t - 1, causal[:, -1])
                backward = np.ones((len(ks), n_angles))
                following = None  # likelihood of bin t + 1
                for c in reversed(range(a, b, chunk)):
                    lik = likelihood(ks, c, min(c + chunk, b))
                    for i in reversed(range(lik.shape[1])):
                        t = c - a + i
                        if t < n_t - 1:
                            backward = _normalize_rows(
                                step.backward(following * backward), uniform
                            )
                            record(
                                batch,
                                start + t,
                                _normalize_rows(causal[:, t] * backward, uniform),
                            )
                        following = lik[:, i]
        start += n_t

    decoded_deg = bin_centers_deg[best]
    null = np.empty((len(metrics), n_shuffles))
    for k in range(n_shuffles):
        values = _decode_metrics(
            decoded_deg[k],
            actual_deg,
            peak[k],
            entropy[k] if want_entropy else None,
            with_metrics,
            tolerance_deg,
        )
        null[:, k] = [values[name] for name in metrics]
    return null


def _judge(metric: str, observed: float, null: np.ndarray, kind: str) -> ShuffleTest:
    """Judge one observed statistic against its null, the right way up.

    Parameters
    ----------
    metric : str
        Name of the statistic.
    observed : float
        Observed value.
    null : numpy.ndarray
        Value under each shuffle.
    kind : str
        Kind of shuffle.

    Returns
    -------
    ShuffleTest
        The test.
    """
    # error-like metrics are better when small; confidence-like ones when large
    better = "lower" if "error" in metric or "rmse" in metric else "higher"
    hits = (
        np.count_nonzero(null <= observed)
        if better == "lower"
        else np.count_nonzero(null >= observed)
    )
    return ShuffleTest(
        metric=metric,
        observed=float(observed),
        null=null,
        p_value=float((1 + hits) / (len(null) + 1)),
        better=better,
        kind=kind,
    )


# ── the whole thing ──────────────────────────────────────────────────────
@dataclass
class DecoderRun:
    """Everything one call to :func:`run_decoder` produced.

    Attributes
    ----------
    data : DecoderData
        Binned data.
    model : EncodingModel
        The fitted model.
    train_mask, test_mask : numpy.ndarray of bool
        Training and held-out bins.
    test : Decoded
        Decode of the held-out wake.
    test_shuffle : ShuffleTest or None
        Time-shift null for `test`.
    rem : Decoded or None
        Decode of REM.
    rem_shuffle : ShuffleTest or None
        Unit-permutation null for `rem`.
    rem_mask : numpy.ndarray of bool or None
        REM bins decoded.
    nrem : Decoded or None
        Decode of NREM, if asked for.
    nrem_shuffle : ShuffleTest or None
        Unit-permutation null for `nrem`.
    nrem_mask : numpy.ndarray of bool or None
        NREM bins decoded.
    movement_var_deg2 : float
        Random-walk variance used, in deg^2 per bin.
    """

    data: DecoderData
    model: EncodingModel
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
    movement_var_deg2: float = float("nan")

    def summary(self) -> str:
        lines = [
            f"{self.model}",
            (
                f"  train {self.data.duration_s[self.train_mask].sum():.0f}s / "
                f"test {self.data.duration_s[self.test_mask].sum():.0f}s"
            ),
            f"  test: {self.test}",
        ]
        for key in (
            "median_abs_error_deg",
            "rmse_deg",
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
                f"    mean posterior max = "
                f"{decoded.metrics['mean_posterior_max']:.3f} "
                f"(wake test {self.test.metrics['mean_posterior_max']:.3f})"
            )
            if shuffle is not None:
                lines.append(f"    vs shuffle: {shuffle}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _wake_mask(
    data, intervals, frame_times, within=None, interval_mask=None, verbose=True
):
    """Find the WAKE bins to train and test on, inside `within` and `interval_mask`.

    Shared by :func:`run_decoder` and :func:`spikeshpc.ring.run_ring`, so that
    with the same settings both split exactly the same bins.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    intervals : dict
        ``{state: [[start, stop], ...]}``.
    frame_times : array-like
        The shutter-aligned frame times `data` was prepared from.
    within : list of list of float, optional
        ``[start, stop]`` spans; only WAKE bins lying wholly inside one are kept.
    interval_mask : array-like of bool, optional
        Mask over the inter-frame intervals of `frame_times`; only WAKE bins
        whose every frame interval it keeps are kept.
    verbose : bool, default True
        Print how much wake each restriction keeps.

    Returns
    -------
    numpy.ndarray of bool
        One value per bin.

    Raises
    ------
    ValueError
        If no WAKE bin remains, or `interval_mask` is not over the intervals.
    """
    wake_mask = state_interval_mask(data, intervals, "WAKE")
    if not wake_mask.any():
        raise ValueError("no decoder bin falls inside a WAKE interval")
    if within is not None:
        wake_s = data.duration_s[wake_mask].sum()
        spans = np.asarray(within, dtype=float).reshape(-1, 2).tolist()
        wake_mask &= state_interval_mask(data, {"WITHIN": spans}, "WITHIN")
        if not wake_mask.any():
            raise ValueError("no WAKE decoder bin lies inside the `within` spans")
        if verbose:
            print(
                f"  within the given spans: {data.duration_s[wake_mask].sum():.0f}s "
                f"of {wake_s:.0f}s wake"
            )
    if interval_mask is not None:
        interval_mask = np.asarray(interval_mask, dtype=bool)
        n_intervals = len(frame_times) - 1
        if interval_mask.shape != (n_intervals,):
            raise ValueError(
                f"interval_mask has {interval_mask.shape} entries but there are "
                f"{n_intervals} inter-frame intervals. It masks intervals, not "
                "frames -- see spikeshpc.states.frames_in_states."
            )
        wake_s = data.duration_s[wake_mask].sum()
        wake_mask &= bins_in_interval_mask(data, interval_mask)
        if not wake_mask.any():
            raise ValueError("no WAKE decoder bin lies wholly inside `interval_mask`")
        if verbose:
            print(
                f"  within interval_mask: {data.duration_s[wake_mask].sum():.0f}s "
                f"of {wake_s:.0f}s wake"
            )
    return wake_mask


def _decode_state(
    data,
    model,
    mask,
    state,
    label,
    shared,
    n_shuffles,
    seed,
    n_jobs,
    keep_posterior,
    verbose,
    warn_prefix="",
    stacklevel=3,
):
    """Decode one sleep state and judge it against the unit-permutation null.

    There is no heading to be wrong about in sleep, so the null has to break
    the population code rather than its alignment in time; the statistic is
    how concentrated the posterior is.

    Parameters
    ----------
    data : DecoderData
        Binned data.
    model : EncodingModel
        Fitted model.
    mask : numpy.ndarray of bool or None
        The state's bins; None if it was not asked for.
    state : str
        State name, for the warning.
    label : str
        Name of the decode.
    shared : dict
        ``movement_var_deg2``, ``acausal`` and ``tolerance_deg``.
    n_shuffles, seed, n_jobs
        As for :func:`shuffle_test`; no shuffles if `n_shuffles` is 0.
    keep_posterior : bool
        Keep the full posterior.
    verbose : bool
        Print progress.
    warn_prefix : str, default ""
        Put before the warning.
    stacklevel : int, default 3
        Of the warning.

    Returns
    -------
    decoded : Decoded or None
        The decode; None if `mask` is None or keeps no bin (a warning says so).
    shuffle : ShuffleTest or None
        The null of ``mean_posterior_max``.
    mask : numpy.ndarray of bool or None
        `mask`, or None where there was nothing to decode.
    """
    if mask is None:
        return None, None, None
    if not mask.any():
        warnings.warn(
            f"{warn_prefix}no decoder bin falls inside a {state} interval",
            stacklevel=stacklevel,
        )
        return None, None, None
    decoded = decode(
        data,
        model,
        mask,
        keep_posterior=keep_posterior,
        with_metrics=False,
        label=label,
        **shared,
    )
    if verbose:
        print(f"  {decoded}")
    shuffle = None
    if n_shuffles:
        shuffle = shuffle_test(
            data,
            model,
            mask,
            kind="units",
            metric="mean_posterior_max",
            n_shuffles=n_shuffles,
            seed=seed,
            with_metrics=False,
            n_jobs=n_jobs,
            **shared,
        )
        if verbose:
            print(f"    {shuffle}")
    return decoded, shuffle, mask


def run_decoder(
    sorting,
    unit_ids,
    heading_deg,
    frame_times,
    intervals,
    bin_s: float = (1 / 60),
    n_angle_bins: int = 180,
    smooth_sigma_deg: float = 10.0,
    min_rate_hz: float = 0.1,
    test_fraction: float = 0.3,
    split_mode: str = "blocks",
    block_s: float = 60.0,
    movement_var_deg2: float | str | None = None,
    acausal: bool = True,
    tolerance_deg: float = 10.0,
    n_shuffles: int = 100,
    min_shift_s: float = 30.0,
    decode_rem: bool = True,
    decode_nrem: bool = False,
    keep_posterior: bool = True,
    seed: int = 0,
    verbose: bool = True,
    within=None,
    interval_mask=None,
    n_jobs: int | None = None,
) -> DecoderRun:
    """
    Train on wake, test on held-out wake against a shuffle, then decode REM (and NREM).

    Parameters
    ----------
    sorting : spikeinterface BaseSorting or SortingAnalyzer
        Source of the spike trains.
    unit_ids : sequence
        The head-direction-tuned units you selected.
    heading_deg, frame_times : array-like
        The shutter-aligned pair.
    intervals : dict
        ``scoring.intervals`` from :func:`spikeshpc.load_states`.
    bin_s : float, default 1/60
        Bin size in seconds. To set to camera frame rate, use
        ``1 / frame_rate``.
    n_angle_bins : int, default 180
        Number of bins to divide heading space into.
    smooth_sigma_deg : float, default 10.0
        How much to smooth tuning curves, in degrees.
    min_rate_hz : float, default 0.1
        Floor applied to tuning curves to avoid overweighting random spikes.
    test_fraction : float, default 0.3
        Test/train split.
    split_mode : {"blocks", "contiguous"}, default "blocks"
        How to split the wake data into train/test chunks.
    block_s : float, default 60.0
        If "blocks" mode is used, how long in seconds the blocks should be.
    movement_var_deg2 : float or "estimate" or None, default None
        Variance of the random walk on heading, in deg^2 per bin. ``None``
        uses the convention of 2 deg^2 per 10 ms bin, scaled to `bin_s`;
        ``"estimate"`` estimates it from the actual heading; ``np.inf`` removes
        the limit completely; a number fixes it at that value.
    acausal : bool, default True
        Use the acausal decoder rather than the causal one.
    tolerance_deg : float, default 10.0
        For evaluating the model: acceptable deviation from the true heading.
    n_shuffles : int, default 100
        Number of shuffles for the control analysis.
    min_shift_s : float, default 30.0
        Amount to shift spike trains against heading in the shuffle analysis.
    decode_rem : bool, default True
        Whether to run the decoder on REM data. Set False for fine-tuning the
        model.
    decode_nrem : bool, default False
        Whether to run the decoder on NREM data too, judged against the same
        unit-permutation null as REM.
    keep_posterior : bool, default True
        Whether to save posteriors for REM and NREM analysis.
    seed : int, default 0
        Seed for generating the random train/test split.
    verbose : bool, default True
        Print progress on model generation.
    within : list of list of float, optional
        ``[start, stop]`` spans on the recording clock, e.g. ``[[0, 7200]]``
        for the first two hours. Only WAKE bins lying wholly inside one are
        trained and tested on, so both sides of the split come from those
        periods. REM and NREM are decoded as before. Default is all of wake.
    interval_mask : array-like of bool, optional
        Mask over the inter-frame intervals of `frame_times` (length
        ``len(frame_times) - 1``) -- the one the tuning functions take, e.g.
        from :func:`spikeshpc.behavior_interval_mask`. Only WAKE bins whose
        every frame interval it keeps are trained and tested on, and it is
        applied before the train/test split, so both sides come from the kept
        time. REM and NREM are decoded as before. Default is all of wake.
    n_jobs : int, optional
        Worker processes for the shuffles. Default is one per CPU core; 1 runs
        them in this process.

    Returns
    -------
    DecoderRun
        The model, the train/test masks, and the decoded test, REM and NREM
        stretches with their shuffle tests.

    Raises
    ------
    ValueError
        If no WAKE bin remains inside the requested intervals.

    Notes
    -----
    **Tuning guide.**
    The factor that matters most is not in this list: which units you pass in.
    A decoder is a weighted vote of tuning curves, so adding a unit with a
    weak or unstable curve costs more than any parameter here will win back.
    Start there, then:

    **bin_s** (0.05) -- decoder time bin, rounded *down* to a whole number of
      camera frames, so at 120 fps 0.05 becomes exactly 6 frames. Longer bins
      collect more spikes and give a sharper posterior, at the price of
      smearing fast head turns: the animal's heading has to be roughly constant
      within a bin for the Poisson likelihood to mean anything. Shorter bins
      track turns but lean harder on the random-walk prior to fill in. If the
      decode looks noisy, try 0.1 before anything else; if it lags real turns,
      try 0.025. Cost is linear in 1/bin_s.

    **n_angle_bins** (180) -- resolution of the state space, so 2 degrees per
      bin. Rarely worth changing: below the tracking noise it buys nothing and
      costs time quadratically in the transition matrix. It does interact with
      ``movement_var_deg2`` -- see the warning :func:`ring_transition` raises
      when the random walk is too tight for the bins to represent.

    **smooth_sigma_deg** (10.0) -- circular smoothing of the tuning curves that
      form the encoding model. This is a bias/variance dial on the model
      itself: too small and each curve carries the sampling noise of its own
      training bins into every decode; too large and genuinely sharp cells are
      flattened and stop discriminating. Raise it when training data is thin.

    **min_rate_hz** (0.1) -- floor under the tuning curves. Without it, a
      heading a unit never happened to fire at has rate zero, its
      log-likelihood is -inf, and one spike there vetoes that heading outright
      however much the rest of the population likes it. Raise it to make the
      population more forgiving of surprising spikes, lower it to let confident
      units veto more strongly. Sensitive to your unit count: with few units, a
      higher floor is safer.

    **test_fraction** (0.3) and **split_mode** ("blocks") / **block_s** (60.0)
      -- how much wake is held out and how it is carved. ``"blocks"`` scatters
      whole 60 s blocks across the session; ``"contiguous"`` holds out the tail
      in one piece. See :func:`split_train_test` for why blocks, and why
      contiguous is the harder and more honest question if you mean to apply
      the model to REM later. Expect contiguous to score worse; if it scores
      *much* worse, the tuning is drifting over the session and that is worth
      knowing before you trust a REM decode from the same model.

    **movement_var_deg2** (None) -- how far the head is assumed to move per
      bin, as the variance of a random walk on the ring. This is the strongest
      prior in the model and the easiest one to fool yourself with. A number
      sets it directly; ``"estimate"`` measures it from the training heading,
      which is the principled choice; ``None`` uses the convention carried over
      from ``run_decoder.py`` (2 deg^2 per 10 ms bin, scaled to ``bin_s``);
      ``np.inf`` removes the prior entirely and decodes each bin from its own
      spikes alone.

      Too small and the posterior becomes sticky -- it will lag real turns and
      produce smooth trajectories whatever the spikes say, which looks like a
      good decode and is not. Run ``np.inf`` once as a control: if the decode
      still tracks the animal without any dynamics, the smoothness was real.

    **acausal** (True) -- use spikes from after each bin as well as before.
      Strictly better offline and the right default. Set False when the decoded
      trajectory has to be readable as a prediction, or to check that a result
      does not depend on the smoother.

    **tolerance_deg** (30.0) -- only affects the reported
      ``frac_within_deg``; it changes no decode.

    **n_shuffles** (100) and **min_shift_s** (30.0) -- the control. The
      p-value's floor is ``1 / (n_shuffles + 1)``, so 100 shuffles cannot report
      better than p = 0.0099; raise it if you need a smaller number, set 0 to
      skip the controls while tinkering. ``min_shift_s`` keeps a shift from
      landing back near register.

    **decode_rem** (True) / **keep_posterior** (True) -- set ``decode_rem``
      False while tuning on wake, since REM costs a second decode and its own
      shuffles. ``keep_posterior`` False drops the full posterior from the
      result, which is what :func:`plot_decoded` shades; it is the memory the
      run holds, roughly ``n_bins x n_angle_bins x 4`` bytes.

    **decode_nrem** (False) -- NREM is usually hours long, so it costs the
      longest decode of the run and shuffles to match, and its posterior is
      the largest thing ``keep_posterior`` keeps. Read it knowing that the
      random walk was fitted to waking head movement, while the internal
      heading moves several times faster in NREM than in wake (Peyrache et al.
      2015): the prior can smooth over real jumps. Decode it once more with
      ``decode(run.data, run.model, run.nrem_mask, movement_var_deg2=np.inf)``
      as the control.

    **seed** (0) -- fixes both the train/test split and the shuffles. Vary it
      to check a result is not an artefact of one particular split; if the
      metrics move a lot across seeds, the test set is too small.
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

    model = fit_encoding_model(
        data, train_mask, n_angle_bins, smooth_sigma_deg, min_rate_hz
    )

    if movement_var_deg2 == "estimate":
        movement_var_deg2 = movement_variance(data, train_mask)
    elif movement_var_deg2 is None:
        movement_var_deg2 = movement_variance(data)
    movement_var_deg2 = float(movement_var_deg2)
    if verbose:
        print(
            f"  random walk: {np.sqrt(movement_var_deg2):.2f} deg sd per "
            f"{1e3 * data.bin_s:.0f} ms bin"
        )

    shared = {
        "movement_var_deg2": movement_var_deg2,
        "acausal": acausal,
        "tolerance_deg": tolerance_deg,
    }
    test = decode(
        data,
        model,
        test_mask,
        keep_posterior=keep_posterior,
        label="wake test",
        **shared,
    )
    if verbose:
        print(f"  {test}")
        # decode() skips stretches under min_run_s, which a fragmented mask is
        # made of; say so rather than let the test set shrink unannounced
        test_s = data.duration_s[test_mask].sum()
        skipped_s = test_s - test.duration_s.sum()
        if skipped_s > 0.01 * test_s:
            print(
                f"    {skipped_s:.0f}s of the {test_s:.0f}s test set was not "
                "decoded: decode() skips stretches under 1 s"
            )

    test_shuffle = None
    if n_shuffles:
        if verbose:
            print(f"  {n_shuffles} shifted-spike shuffles...")
        test_shuffle = shuffle_test(
            data,
            model,
            test_mask,
            kind="shift",
            metric="median_abs_error_deg",
            n_shuffles=n_shuffles,
            min_shift_s=min_shift_s,
            seed=seed,
            n_jobs=n_jobs,
            **shared,
        )
        if verbose:
            print(f"    {test_shuffle}")

    # no heading to be wrong about in sleep, so each state's null breaks the
    # population code rather than its alignment in time
    sleep = {}
    for state, wanted in (("REM", decode_rem), ("NREM", decode_nrem)):
        sleep[state] = _decode_state(
            data,
            model,
            state_interval_mask(data, intervals, state) if wanted else None,
            state,
            label=state,
            shared=shared,
            n_shuffles=n_shuffles,
            seed=seed,
            n_jobs=n_jobs,
            keep_posterior=keep_posterior,
            verbose=verbose,
        )
    rem, rem_shuffle, rem_mask = sleep["REM"]
    nrem, nrem_shuffle, nrem_mask = sleep["NREM"]

    return DecoderRun(
        data=data,
        model=model,
        train_mask=train_mask,
        test_mask=test_mask,
        test=test,
        test_shuffle=test_shuffle,
        rem=rem,
        rem_shuffle=rem_shuffle,
        rem_mask=rem_mask,
        nrem=nrem,
        nrem_shuffle=nrem_shuffle,
        nrem_mask=nrem_mask,
        movement_var_deg2=movement_var_deg2,
    )


# ── carrying a model to another recording ────────────────────────────────
# the wake statistics tested against the time-shift null, from one set of shuffles
WAKE_SHUFFLE_METRICS = ("median_abs_error_deg", "median_abs_error_corrected_deg")


@dataclass
class TransferRun:
    """An encoding model read out on spikes it was not fitted on.

    Either another recording's, through :func:`apply_decoder`, or the fitting
    recording's own held-out wake, REM and NREM through
    :func:`reference_decode`. ``unit_ids`` are the model's units and
    ``partner_ids`` the units whose spikes stood in for them, in the same order
    -- the same ids, for a reference. ``wake_shuffles`` maps each of
    :data:`WAKE_SHUFFLE_METRICS` to its :class:`ShuffleTest` against the
    time-shift null; ``rem_shuffle`` and ``nrem_shuffle`` are the
    unit-permutation nulls of the sleep posteriors' confidence.

    Attributes
    ----------
    label : str
        Name of the run.
    unit_ids, partner_ids : numpy.ndarray
        The model's units, and the units whose spikes stood in for them.
    data : DecoderData
        Binned data.
    wake, rem, nrem : Decoded
        Decodes of wake, REM and NREM (`rem` and `nrem` may be None).
    wake_mask, rem_mask, nrem_mask : numpy.ndarray of bool
        Bins decoded.
    wake_shuffles : dict
        Each of :data:`WAKE_SHUFFLE_METRICS` mapped to its
        :class:`ShuffleTest`.
    rem_shuffle, nrem_shuffle : ShuffleTest or None
        Unit-permutation nulls of the REM and NREM posteriors' confidence.
    """

    label: str
    unit_ids: np.ndarray
    partner_ids: np.ndarray
    data: DecoderData
    wake: Decoded
    wake_mask: np.ndarray
    wake_shuffles: dict = field(default_factory=dict)
    rem: Decoded | None = None
    rem_mask: np.ndarray | None = None
    rem_shuffle: ShuffleTest | None = None
    nrem: Decoded | None = None
    nrem_mask: np.ndarray | None = None
    nrem_shuffle: ShuffleTest | None = None

    @property
    def offset_deg(self) -> float:
        """Get the wake decode's mean error: how far the population reads turned.

        Returns
        -------
        float
            Offset in degrees; NaN if unavailable.
        """
        return float(self.wake.metrics.get("offset_deg", np.nan))

    def as_row(self) -> dict:
        """Collect the headline numbers as one flat row, for a table across recordings.

        Returns
        -------
        dict
            The row.
        """
        wake = self.wake.metrics
        row = {
            "label": self.label,
            "n_units": len(self.unit_ids),
            "wake_s": float(self.wake.duration_s.sum()),
            "wake_median_abs_error_deg": wake.get("median_abs_error_deg", np.nan),
            "wake_median_abs_error_corrected_deg": wake.get(
                "median_abs_error_corrected_deg", np.nan
            ),
            "wake_frac_within_deg": wake.get("frac_within_deg", np.nan),
            "wake_frac_within_corrected_deg": wake.get(
                "frac_within_corrected_deg", np.nan
            ),
            "wake_circular_correlation": wake.get("circular_correlation", np.nan),
            "wake_offset_deg": self.offset_deg,
            "wake_mean_posterior_max": wake.get("mean_posterior_max", np.nan),
        }
        for metric, test in self.wake_shuffles.items():
            short = metric.replace("median_abs_error_", "error_").replace("_deg", "")
            row[f"wake_{short}_null_mean"] = float(test.null.mean())
            row[f"wake_{short}_p"] = test.p_value
            row[f"wake_{short}_z"] = test.z_score
        for key, decoded, shuffle in (
            ("rem", self.rem, self.rem_shuffle),
            ("nrem", self.nrem, self.nrem_shuffle),
        ):
            row[f"{key}_s"] = (
                float(decoded.duration_s.sum()) if decoded is not None else 0.0
            )
            row[f"{key}_mean_posterior_max"] = (
                decoded.metrics["mean_posterior_max"] if decoded is not None else np.nan
            )
            if shuffle is not None:
                row[f"{key}_null_mean"] = float(shuffle.null.mean())
                row[f"{key}_p"] = shuffle.p_value
                row[f"{key}_z"] = shuffle.z_score
        return row

    def summary(self) -> str:
        wake = self.wake.metrics
        lines = [
            f"{self.label}: {len(self.unit_ids)} units",
            f"  wake: {self.wake}",
            (
                f"    median |error| {wake['median_abs_error_deg']:.1f} deg, "
                f"{wake['median_abs_error_corrected_deg']:.1f} deg after removing a "
                f"{wake['offset_deg']:+.1f} deg offset; circular r = "
                f"{wake['circular_correlation']:.3f}"
            ),
        ]
        for test in self.wake_shuffles.values():
            lines.append(f"    vs shuffle: {test}")
        for state, decoded, shuffle in (
            ("REM", self.rem, self.rem_shuffle),
            ("NREM", self.nrem, self.nrem_shuffle),
        ):
            if decoded is None:
                continue
            lines.append(f"  {state}: {decoded}")
            lines.append(
                f"    mean posterior max = {decoded.metrics['mean_posterior_max']:.3f} "
                f"(wake {wake['mean_posterior_max']:.3f})"
            )
            if shuffle is not None:
                lines.append(f"    vs shuffle: {shuffle}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def apply_decoder(
    model: EncodingModel,
    sorting,
    unit_map: dict,
    heading_deg,
    frame_times,
    intervals,
    interval_mask=None,
    movement_var_deg2: float | None = None,
    bin_s: float = (1 / 60),
    acausal: bool = True,
    tolerance_deg: float = 10.0,
    n_shuffles: int = 100,
    min_shift_s: float = 30.0,
    seed: int = 0,
    decode_rem: bool = True,
    decode_nrem: bool = False,
    keep_posterior: bool = False,
    label: str = "transfer",
    verbose: bool = True,
    n_jobs: int | None = None,
) -> TransferRun:
    """Decode another recording with a model fitted on the baseline.

    ``unit_map`` is ``{model unit: this recording's unit}``, from matching the
    two recordings. The model is restricted to those units and each column is
    filled with its partner's spikes here, so the question is: read with the
    baseline's tuning curves, does this recording's activity still say where
    the head points?

    Nothing here was trained on, so wake is every WAKE bin -- inside
    ``interval_mask``, the mask this recording's own tuning was computed on,
    as the baseline's split is inside its own -- scored raw and with its
    constant offset removed (see :func:`decoding_metrics`), each against the
    time-shift null. REM, and NREM if asked for, are decoded and judged
    against the unit-permutation null exactly as :func:`run_decoder` does.
    ``movement_var_deg2`` should be the baseline run's, so the prior is the
    one the model was scored with, and the remaining settings should match
    that run's too.

    Parameters
    ----------
    model : EncodingModel
        Model fitted on the baseline.
    sorting : spikeinterface BaseSorting or SortingAnalyzer
        This recording's sorting.
    unit_map : dict
        ``{model unit: this recording's unit}``, from matching the two
        recordings. The model is restricted to those units and each column is
        filled with its partner's spikes here.
    heading_deg, frame_times : array-like
        This recording's shutter-aligned pair.
    intervals : dict
        This recording's state intervals.
    interval_mask : array-like of bool, optional
        The mask this recording's own tuning was computed on.
    movement_var_deg2 : float, optional
        Should be the baseline run's, so the prior is the one the model was
        scored with.
    bin_s : float, default 1/60
        Bin size in seconds.
    acausal : bool, default True
        Use the acausal decoder.
    tolerance_deg : float, default 10.0
        Tolerance for ``frac_within_deg``.
    n_shuffles : int, default 100
        Number of shuffles.
    min_shift_s : float, default 30.0
        Smallest shift, in seconds.
    seed : int, default 0
        Random seed.
    decode_rem : bool, default True
        Also decode REM.
    decode_nrem : bool, default False
        Also decode NREM.
    keep_posterior : bool, default False
        Keep the full posterior.
    label : str, default "transfer"
        Name of the run.
    verbose : bool, default True
        Print progress.
    n_jobs : int, optional
        As for :func:`shuffle_test`.

    Returns
    -------
    TransferRun
        The read-out.

    Raises
    ------
    ValueError
        If `unit_map` is empty or not one-to-one, or no bin falls inside the
        requested states.
    """
    model_units = list(unit_map)
    if not model_units:
        raise ValueError("unit_map is empty: there are no matched units to decode from")
    partners = [unit_map[u] for u in model_units]
    if len(set(partners)) != len(partners):
        raise ValueError("unit_map gives two model units the same partner")

    model = restrict_model(model, model_units)
    data = prepare_decoder_data(sorting, partners, heading_deg, frame_times, bin_s)
    # named by the model's units, so decode() matches its columns to the curves
    data = replace(data, unit_ids=np.asarray(model.unit_ids))
    if verbose:
        print(f"  {label}: {data}")

    wake_mask = state_interval_mask(data, intervals, "WAKE")
    if not wake_mask.any():
        raise ValueError(f"{label}: no decoder bin falls inside a WAKE interval")
    if interval_mask is not None:
        interval_mask = np.asarray(interval_mask, dtype=bool)
        if interval_mask.shape != (len(frame_times) - 1,):
            raise ValueError(
                f"interval_mask has {interval_mask.shape} entries but there are "
                f"{len(frame_times) - 1} inter-frame intervals. It masks intervals, "
                "not frames -- see spikeshpc.states.frames_in_states."
            )
        wake_mask &= bins_in_interval_mask(data, interval_mask)
        if not wake_mask.any():
            raise ValueError(
                f"{label}: no WAKE decoder bin lies wholly inside `interval_mask`"
            )
    rem_mask = state_interval_mask(data, intervals, "REM") if decode_rem else None
    nrem_mask = state_interval_mask(data, intervals, "NREM") if decode_nrem else None

    return _read_out(
        data,
        model,
        np.asarray(partners),
        wake_mask,
        rem_mask,
        nrem_mask=nrem_mask,
        movement_var_deg2=movement_var_deg2,
        acausal=acausal,
        tolerance_deg=tolerance_deg,
        n_shuffles=n_shuffles,
        min_shift_s=min_shift_s,
        seed=seed,
        keep_posterior=keep_posterior,
        label=label,
        verbose=verbose,
        n_jobs=n_jobs,
    )


def reference_decode(
    run: DecoderRun,
    unit_ids,
    acausal: bool = True,
    tolerance_deg: float = 10.0,
    n_shuffles: int = 100,
    min_shift_s: float = 30.0,
    seed: int = 0,
    keep_posterior: bool = False,
    label: str = "baseline",
    verbose: bool = True,
    n_jobs: int | None = None,
) -> TransferRun:
    """Decode the fitting recording's own held-out wake and REM with only some units.

    What :func:`apply_decoder` on another recording should be compared with:
    the same units and the same curves, on data the model never saw. Without
    it, a recording where fewer units were found looks worse for having fewer
    units rather than for coding worse. NREM is decoded too if the run decoded
    it.

    Parameters
    ----------
    run : DecoderRun
        The baseline run.
    unit_ids : sequence
        Units to decode with.
    acausal : bool, default True
        Use the acausal decoder.
    tolerance_deg : float, default 10.0
        Tolerance for ``frac_within_deg``.
    n_shuffles : int, default 100
        Number of shuffles.
    min_shift_s : float, default 30.0
        Smallest shift, in seconds.
    seed : int, default 0
        Random seed.
    keep_posterior : bool, default False
        Keep the full posterior.
    label : str, default "baseline"
        Name of the run.
    verbose : bool, default True
        Print progress.
    n_jobs : int, optional
        As for :func:`shuffle_test`.

    Returns
    -------
    TransferRun
        The read-out.
    """
    unit_ids = list(unit_ids)
    model = restrict_model(run.model, unit_ids)
    data = restrict_data(run.data, unit_ids)
    return _read_out(
        data,
        model,
        np.asarray(unit_ids),
        run.test_mask,
        run.rem_mask,
        nrem_mask=run.nrem_mask,
        movement_var_deg2=run.movement_var_deg2,
        acausal=acausal,
        tolerance_deg=tolerance_deg,
        n_shuffles=n_shuffles,
        min_shift_s=min_shift_s,
        seed=seed,
        keep_posterior=keep_posterior,
        label=label,
        verbose=verbose,
        n_jobs=n_jobs,
    )


def _read_out(
    data,
    model,
    partner_ids,
    wake_mask,
    rem_mask,
    movement_var_deg2,
    acausal,
    tolerance_deg,
    n_shuffles,
    min_shift_s,
    seed,
    keep_posterior,
    label,
    verbose,
    n_jobs=None,
    nrem_mask=None,
) -> TransferRun:
    """Decode wake and sleep with a model and judge each against its null.

    Parameters
    ----------
    data : DecoderData
        Binned data, already holding the partner units' counts.
    model : EncodingModel
        Model restricted to the matched units.
    partner_ids : sequence
        Units whose spikes stand in for the model's.
    wake_mask, rem_mask : numpy.ndarray of bool
        Bins to decode; `rem_mask` None skips REM.
    movement_var_deg2, acausal, tolerance_deg, n_shuffles, min_shift_s, seed
        As for :func:`apply_decoder`.
    keep_posterior : bool
        Keep the full posterior.
    label : str
        Name of the run.
    verbose : bool
        Print progress.
    n_jobs : int, optional
        As for :func:`shuffle_test`.
    nrem_mask : numpy.ndarray of bool, optional
        NREM bins to decode; None skips NREM.

    Returns
    -------
    TransferRun
        The read-out.
    """
    shared = {
        "movement_var_deg2": movement_var_deg2,
        "acausal": acausal,
        "tolerance_deg": tolerance_deg,
    }
    wake = decode(
        data,
        model,
        wake_mask,
        keep_posterior=keep_posterior,
        label=f"{label} wake",
        **shared,
    )
    if verbose:
        print(f"  {wake}")
    wake_shuffles = {}
    if n_shuffles:
        wake_shuffles = shuffle_test(
            data,
            model,
            wake_mask,
            kind="shift",
            metric=WAKE_SHUFFLE_METRICS,
            n_shuffles=n_shuffles,
            min_shift_s=min_shift_s,
            seed=seed,
            n_jobs=n_jobs,
            **shared,
        )
        if verbose:
            for test in wake_shuffles.values():
                print(f"    {test}")

    sleep = {
        state: _decode_state(
            data,
            model,
            mask,
            state,
            label=f"{label} {state}",
            shared=shared,
            n_shuffles=n_shuffles,
            seed=seed,
            n_jobs=n_jobs,
            keep_posterior=keep_posterior,
            verbose=verbose,
            warn_prefix=f"{label}: ",
            stacklevel=4,
        )
        for state, mask in (("REM", rem_mask), ("NREM", nrem_mask))
    }
    rem, rem_shuffle, rem_mask = sleep["REM"]
    nrem, nrem_shuffle, nrem_mask = sleep["NREM"]

    return TransferRun(
        label=label,
        unit_ids=np.asarray(model.unit_ids),
        partner_ids=np.asarray(partner_ids),
        data=data,
        wake=wake,
        wake_mask=wake_mask,
        wake_shuffles=wake_shuffles,
        rem=rem,
        rem_mask=rem_mask,
        rem_shuffle=rem_shuffle,
        nrem=nrem,
        nrem_mask=nrem_mask,
        nrem_shuffle=nrem_shuffle,
    )


# ── looking at it ────────────────────────────────────────────────────────
def plot_encoding_model(
    model: EncodingModel, sort_by_preferred: bool = True, ax=None, cmap="magma"
):
    """Plot the tuning curves the decoder is using, as a units x heading heatmap.

    Each unit's curve is scaled to its own peak, so the picture is about where
    a unit fires rather than how hard. Sorted by preferred direction, a healthy
    head-direction population is a clean diagonal band.

    Parameters
    ----------
    model : EncodingModel
        Model to draw.
    sort_by_preferred : bool, default True
        Sort units by preferred direction.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    cmap : colormap, default magma

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    import matplotlib.pyplot as plt

    order = (
        np.argsort(model.preferred_deg)
        if sort_by_preferred
        else np.arange(len(model.unit_ids))
    )
    peak = model.rate_hz.max(axis=1, keepdims=True)
    normalized = model.rate_hz / np.where(peak > 0, peak, 1.0)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(
        normalized[order],
        aspect="auto",
        origin="lower",
        extent=(0, 360, -0.5, len(order) - 0.5),
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("head direction (deg)")
    ax.set_ylabel(
        "unit (sorted by preferred direction)" if sort_by_preferred else "unit"
    )
    ax.set_xticks(np.arange(0, 361, 90))
    ax.set_title(
        f"encoding model: {len(model.unit_ids)} units, "
        f"{model.train_time_s:.0f}s of training wake"
    )
    ax.figure.colorbar(image, ax=ax, label="rate / peak rate")
    return ax


def plot_decoded(
    decoded: Decoded,
    t0: float | None = None,
    window_s: float = 60.0,
    show_posterior: bool = True,
    ax=None,
):
    """Plot decoded vs. actual heading over a window, on top of the posterior.

    The static version of :func:`spikeshpc.decoder_widget.show_decoded`, for
    saving a figure. Note it plots against the *recording* clock, so a blocked
    test set leaves the axis mostly empty; the widget lays the decoded bins end
    to end instead.

    Lines are cut rather than joined wherever a join would be a lie: across the
    0/360 seam, where one step on the ring is a full-height plunge on a linear
    axis, and across a splice between non-adjacent stretches. The posterior
    underneath is the honest version of the decode -- the MAP is only its
    darkest pixel, and a broad or split posterior means the population was not
    committing.

    ``t0`` defaults to the *first decoded bin*, so with the default
    ``window_s`` equal to ``split_train_test``'s ``block_s`` the figure shows
    exactly the first held-out block. Changing ``test_fraction`` often leaves
    that block unchanged -- the split draws blocks in a fixed order and takes a
    longer prefix, so a larger fraction adds blocks rather than reshuffling
    them -- and the window then contains identical ground truth and a decoded
    trace differing by an angle bin or so. Pass ``t0`` explicitly, or compare
    ``run.test.metrics``, rather than reading two such figures as evidence that
    nothing changed.

    Parameters
    ----------
    decoded : Decoded
        The decode.
    t0 : float, optional
        Start of the window, in seconds; defaults to the first decoded bin.
    window_s : float, default 60.0
        Window length, in seconds.
    show_posterior : bool, default True
        Shade the posterior behind the lines.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.

    Raises
    ------
    ValueError
        If no decoded bins fall in the window.
    """
    import matplotlib.pyplot as plt

    if t0 is None:
        t0 = float(decoded.time_s[0])
    window = (decoded.time_s >= t0) & (decoded.time_s < t0 + window_s)
    if not window.any():
        raise ValueError(f"no decoded bins in [{t0}, {t0 + window_s}]s")

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))

    from .decoder_widget import ACTUAL_COLOR, DECODED_COLOR, break_at

    time = decoded.time_s[window]
    if show_posterior and decoded.posterior is not None and len(time) > 1:
        step = np.median(np.diff(time))
        angle_width = 360.0 / len(decoded.bin_centers_deg)
        posterior = decoded.posterior[window]
        ax.pcolormesh(
            np.r_[time - step / 2, time[-1] + step / 2],
            np.r_[
                decoded.bin_centers_deg - angle_width / 2,
                decoded.bin_centers_deg[-1] + angle_width / 2,
            ],
            posterior.T,
            cmap="Blues",
            shading="flat",
            vmin=0.0,
            vmax=max(float(np.percentile(posterior, 99.5)), 1e-6),
        )

    # Lines, not points, but cut wherever a join would be a lie: across the
    # 0/360 seam, and across a splice between non-adjacent stretches.
    breaks = np.flatnonzero(np.diff(decoded.run_index[window]) != 0) + 1
    ax.plot(
        *break_at(time, decoded.actual_deg[window], breaks),
        color=ACTUAL_COLOR,
        lw=1.3,
        label="actual",
        zorder=3,
    )
    ax.plot(
        *break_at(time, decoded.decoded_deg[window], breaks),
        color=DECODED_COLOR,
        lw=1.3,
        label="decoded",
        zorder=4,
    )
    ax.set_facecolor("white")
    ax.set_xlim(t0, t0 + window_s)
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 90))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("head direction (deg)")

    # Which run this is, on the figure. Two decodes from different train/test
    # splits routinely produce a window that looks identical -- the split
    # assigns whole blocks, so the first test block is often the same one, and
    # inside it the ground truth is the same by construction while the decoded
    # trace moves by an angle bin or two. Without the window and the score
    # written down, those figures are indistinguishable by eye and invite the
    # conclusion that a parameter did nothing.
    subtitle = (
        f"{window.sum()} of {decoded.n_decoded} decoded bins, "
        f"{t0:.0f}-{t0 + window_s:.0f}s"
    )
    if "median_abs_error_deg" in decoded.metrics:
        subtitle += (
            f" | whole set: median {decoded.metrics['median_abs_error_deg']:.1f} deg, "
            f"{decoded.duration_s.sum():.0f}s"
        )
    ax.set_title(f"{decoded.label}\n{subtitle}", fontsize=10)
    ax.legend(loc="upper right", markerscale=4, framealpha=0.85)
    return ax


def plot_error(decoded: Decoded, axes=None, cmap="magma"):
    """Plot an error histogram and an actual-vs-decoded confusion map.

    The confusion map is the one that shows *how* a decoder fails. A bright
    diagonal is success; a bright anti-diagonal or a constant offset from the
    diagonal is a systematic mapping error rather than noise, and a bright
    horizontal band is the decoder falling back on one favourite heading
    whenever the evidence is thin.

    Parameters
    ----------
    decoded : Decoded
        The decode.
    axes : sequence of matplotlib.axes.Axes, optional
        The two axes to draw on.
    cmap : colormap, default magma

    Returns
    -------
    sequence of matplotlib.axes.Axes
        The two axes.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    left, right = axes

    left.hist(decoded.error_deg, bins=np.arange(-180, 181, 5), color="#377eb8")
    left.axvline(0, color="k", lw=0.8)
    left.set_xlabel("decoded - actual (deg)")
    left.set_ylabel("bins")
    left.set_xlim(-180, 180)
    left.set_xticks(np.arange(-180, 181, 90))
    if "median_abs_error_deg" in decoded.metrics:
        left.set_title(
            f"median |error| = {decoded.metrics['median_abs_error_deg']:.1f} deg"
        )

    edges = np.linspace(0, 360, 73)
    counts, _, _ = np.histogram2d(
        decoded.actual_deg, decoded.decoded_deg, bins=(edges, edges)
    )
    with np.errstate(invalid="ignore"):
        counts = counts / counts.sum(axis=1, keepdims=True)
    image = right.imshow(
        counts.T,
        origin="lower",
        extent=(0, 360, 0, 360),
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )
    right.plot([0, 360], [0, 360], color="w", lw=0.6, alpha=0.5)
    right.set_xlabel("actual (deg)")
    right.set_ylabel("decoded (deg)")
    right.set_xticks(np.arange(0, 361, 90))
    right.set_yticks(np.arange(0, 361, 90))
    right.figure.colorbar(image, ax=right, label="p(decoded | actual)")
    right.set_title(decoded.label)
    return axes


def plot_shuffle(shuffle: ShuffleTest, ax=None):
    """Plot the null distribution with the observed value marked on it.

    Parameters
    ----------
    shuffle : ShuffleTest
        Test to draw.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(shuffle.null, bins=25, color="#999999", label=f"shuffled ({shuffle.kind})")
    ax.axvline(shuffle.observed, color="#e41a1c", lw=2, label="observed")
    ax.set_xlabel(shuffle.metric.replace("_", " "))
    ax.set_ylabel("shuffles")
    ax.set_title(f"p = {shuffle.p_value:.4f}, z = {shuffle.z_score:+.1f}")
    ax.legend()
    return ax


_METRIC_LABELS = {
    "median_abs_error_deg": "median |error| (deg)",
    "mean_abs_error_deg": "mean |error| (deg)",
    "rmse_deg": "RMSE (deg)",
    "frac_within_deg": "fraction within tolerance",
    "circular_correlation": "circular correlation",
    "mean_posterior_max": "posterior max (confidence)",
    "population_rate_hz": "population rate (Hz)",
}


def plot_metrics_by_group(
    tables,
    metrics=("median_abs_error_deg", "mean_posterior_max", "population_rate_hz"),
    axes=None,
):
    """Plot :func:`metrics_by_group` as bars, one panel per metric.

    Groups run in the order the tables list them. Each tick says how much time
    its group holds, because a striking bar over twenty seconds of data is not
    the finding one over an hour would be. The error panel has interquartile
    whiskers and the 90 degree line that chance would give.

    Parameters
    ----------
    tables : dict
        One table, or ``{series: table}`` to compare several decodes over the
        same groups -- two encoding models, say -- as grouped bars.
    metrics : tuple of str
        Metrics to draw, one panel each.
    axes : sequence of matplotlib.axes.Axes, optional
        One axes per metric.

    Returns
    -------
    sequence of matplotlib.axes.Axes
        The axes.

    Raises
    ------
    ValueError
        If every table is empty, or `axes` does not match `metrics`.
    """
    import matplotlib.pyplot as plt

    # one table is {label: metrics}; several are {series: {label: metrics}}
    first = next(iter(tables.values()), None)
    if isinstance(first, dict) and "time_s" in first:
        tables = {"": tables}
    series = list(tables)
    groups = []
    for table in tables.values():
        groups.extend(g for g in table if g not in groups)
    if not groups:
        raise ValueError("nothing to plot: every table is empty")

    if axes is None:
        _, axes = plt.subplots(
            1, len(metrics), figsize=(4.2 * len(metrics), 3.8), squeeze=False
        )
    axes = np.atleast_1d(np.asarray(axes, dtype=object).ravel())
    if len(axes) != len(metrics):
        raise ValueError(f"{len(metrics)} metrics but {len(axes)} axes")

    x = np.arange(len(groups))
    width = 0.8 / len(series)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def column(table, key):
        return np.array([table.get(g, {}).get(key, np.nan) for g in groups], float)

    reference = tables[series[0]]
    ticks = [
        f"{g}\n{reference[g]['time_s']:.0f}s" if g in reference else str(g)
        for g in groups
    ]
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        for k, name in enumerate(series):
            table = tables[name]
            values = column(table, metric)
            yerr = None
            if metric == "median_abs_error_deg":
                low = column(table, "q25_abs_error_deg")
                high = column(table, "q75_abs_error_deg")
                if np.isfinite(low).any():
                    yerr = np.vstack([values - low, high - values])
            ax.bar(
                x + (k - (len(series) - 1) / 2) * width,
                values,
                width,
                yerr=yerr,
                capsize=2,
                color=colors[k % len(colors)],
                label=name or None,
            )
        if metric == "median_abs_error_deg":
            ax.axhline(90.0, color="0.4", ls="--", lw=1, label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_ylabel(_METRIC_LABELS.get(metric, metric.replace("_", " ")))
        ax.spines[["top", "right"]].set_visible(False)
        # the series once, on the first panel, plus wherever chance is drawn
        if (i == 0 or metric == "median_abs_error_deg") and (
            ax.get_legend_handles_labels()[0]
        ):
            ax.legend(fontsize=8, frameon=False)
    axes[0].figure.tight_layout()
    return axes


def plot_transfer_summary(
    runs: dict, colors: dict | None = None, references: dict | None = None, axes=None
):
    """Plot how one decoder did on each recording, each against its own nulls.

    Three panels, one column per recording, each tick saying how many units it
    was decoded with:

    - wake median |error|: filled is raw, open is with the constant offset
      removed; gray bar is 5-95% of the time-shift null; dashed is the 90 deg
      of chance
    - wake offset: how far the population reads turned
    - REM posterior max: against 5-95% of its unit-permutation null

    and a fourth, NREM posterior max drawn the same way, when any run decoded
    NREM.

    Parameters
    ----------
    runs : dict
        ``{recording: TransferRun}`` in the order to draw them, the baseline's
        own :func:`reference_decode` included if it should have a column.
    colors : dict, optional
        Each recording's color (default: the property cycle).
    references : dict, optional
        Per recording, the baseline's held-out data decoded with that
        recording's units, drawn as an open black diamond in its column -- the
        number that recording has to be compared with.
    axes : sequence of matplotlib.axes.Axes, optional
        The axes to draw on: three, or four when any run decoded NREM.

    Returns
    -------
    sequence of matplotlib.axes.Axes
        The three (or four) axes.

    Raises
    ------
    ValueError
        If there are no runs to plot, or `axes` is the wrong length.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = list(runs)
    if not names:
        raise ValueError("no runs to plot")
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = {name: cycle[i % len(cycle)] for i, name in enumerate(names)}
    references = references or {}
    # the sleep states some run decoded, each a panel of its own
    states = ["rem"]
    if any(run.nrem is not None for run in runs.values()):
        states.append("nrem")
    n_panels = 2 + len(states)
    if axes is None:
        width = max(3.0 * n_panels, 2.0 * n_panels + 1.3 * len(names))
        _, axes = plt.subplots(1, n_panels, figsize=(width, 3.8))
    if len(axes) != n_panels:
        raise ValueError(f"{n_panels} panels to draw but {len(axes)} axes")
    error_ax, offset_ax = axes[:2]
    sleep_axes = dict(zip(states, axes[2:]))
    null_style = dict(color="0.8", lw=7, zorder=1)
    reference_style = dict(
        marker="D", ls="none", mfc="none", mec="k", mew=1.2, ms=7, zorder=4
    )

    for i, name in enumerate(names):
        run, color = runs[name], colors[name]
        wake = run.wake.metrics
        raw_null = run.wake_shuffles.get("median_abs_error_deg")
        if raw_null is not None:
            error_ax.vlines(i, *np.percentile(raw_null.null, [5, 95]), **null_style)
        error_ax.plot(
            i - 0.12, wake["median_abs_error_deg"], "o", color=color, ms=7, zorder=3
        )
        error_ax.plot(
            i + 0.12,
            wake["median_abs_error_corrected_deg"],
            "o",
            mfc="none",
            mec=color,
            mew=1.5,
            ms=7,
            zorder=3,
        )
        offset_ax.plot(i, run.offset_deg, "o", color=color, ms=7, zorder=3)
        for state, ax in sleep_axes.items():
            decoded = getattr(run, state)
            if decoded is None:
                continue
            shuffle = getattr(run, f"{state}_shuffle")
            if shuffle is not None:
                ax.vlines(i, *np.percentile(shuffle.null, [5, 95]), **null_style)
            ax.plot(
                i,
                decoded.metrics["mean_posterior_max"],
                "o",
                color=color,
                ms=7,
                zorder=3,
            )

        reference = references.get(name)
        if reference is not None:
            error_ax.plot(
                i, reference.wake.metrics["median_abs_error_deg"], **reference_style
            )
            offset_ax.plot(i, reference.offset_deg, **reference_style)
            for state, ax in sleep_axes.items():
                decoded = getattr(reference, state)
                if decoded is not None:
                    ax.plot(i, decoded.metrics["mean_posterior_max"], **reference_style)

    # the legend in black, whatever color each recording's markers are
    handles = [
        Line2D([], [], marker="o", ls="none", color="k", ms=7, label="raw"),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            mfc="none",
            mec="k",
            mew=1.5,
            ms=7,
            label="offset removed",
        ),
    ]
    if any(run.wake_shuffles for run in runs.values()):
        handles.insert(0, Line2D([], [], color="0.8", lw=7, label="shuffled"))
    if references:
        handles.append(Line2D([], [], label="baseline, same units", **reference_style))
    error_ax.axhline(90.0, color="0.4", ls="--", lw=1)
    error_ax.set_ylim(0, None)
    error_ax.set_ylabel("wake median |error| (deg)")
    error_ax.legend(handles=handles, fontsize=7, frameon=False)
    offset_ax.axhline(0.0, color="0.6", lw=0.8)
    offset_ax.set_ylim(-180, 180)
    offset_ax.set_yticks(np.arange(-180, 181, 90))
    offset_ax.set_ylabel("wake offset (deg)")
    for state, ax in sleep_axes.items():
        ax.set_ylabel(f"{state.upper()} posterior max")

    ticks = [f"{name}\n{len(runs[name].unit_ids)} units" for name in names]
    for ax in axes:
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].figure.tight_layout()
    return axes
