"""Bayesian decoding of head direction from head-direction-tuned units.

The method is the one in Moritz's ``run_decoder.py``: a sorted-spikes point
process decoder on a ring state space. Per-unit tuning curves are the encoding
model, the likelihood of a time bin is Poisson given those curves, and a
Gaussian random walk on the ring supplies the dynamics that carry belief from
one bin to the next. The readout is the MAP of the posterior.

It is reimplemented here rather than delegated to
``replay_trajectory_classification``, which supplied the decoder there. RTC is
unmaintained, does not install on this environment's Python, and its ring
"track graph" is scaffolding for a general linearized-track machinery that head
direction does not need -- a circle is already one-dimensional and periodic.
The maths below is the same; what it drops is a dependency and the
interpolation layer RTC's fixed-rate time grid demanded.

Everything is indexed by *inter-frame interval*, exactly as
:mod:`spikeshpc.optitrack.tuning` is. That is what makes the decoder's encoding
model literally the tuning curves the units were selected on, and it means the
ground truth is measured rather than resampled: a decoder bin is a whole number
of camera frames, so its edges are real shutter-closure timestamps.

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

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .optitrack.tuning import (
    _bin_headings,
    compute_hd_tuning_curve,
    compute_mean_vector_length,
)
from .states import frames_in_states

__all__ = [
    "DecoderData",
    "Decoded",
    "EncodingModel",
    "DecoderRun",
    "ShuffleTest",
    "prepare_decoder_data",
    "state_interval_mask",
    "split_train_test",
    "fit_encoding_model",
    "ring_transition",
    "decode",
    "shuffle_test",
    "run_decoder",
    "plot_encoding_model",
    "plot_decoded",
    "plot_error",
    "plot_shuffle",
]


# ── angles ───────────────────────────────────────────────────────────────
def circular_difference(a_deg, b_deg) -> np.ndarray:
    """Signed a - b, wrapped into (-180, 180]."""
    return (np.asarray(a_deg, float) - np.asarray(b_deg, float) + 180.0) % 360.0 - 180.0


def circular_mean(angles_deg, axis=None) -> np.ndarray:
    """Mean direction in [0, 360), from the resultant vector."""
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
        """Nominal bin width -- the median, since frame intervals wobble."""
        return float(np.median(self.duration_s))

    def __repr__(self) -> str:
        return (
            f"DecoderData({self.n_bins} bins of {1e3 * self.bin_s:.1f} ms, "
            f"{self.n_units} units, {self.duration_s.sum():.0f}s)"
        )


def _as_sorting(obj):
    """Accept either a sorting or a sorting analyzer."""
    return obj.sorting if hasattr(obj, "sorting") else obj


def _frames_per_bin(bin_s: float, frame_s: float, tolerance: float = 0.01) -> int:
    """How many camera frames fit in ``bin_s``, forgiving a near-miss.

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
    """
    ratio = bin_s / frame_s
    nearest = int(round(ratio))
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

    ``bin_s`` is a *target*: the grid is the largest whole number of camera
    frames not exceeding it, so bin edges stay on measured timestamps. At 120
    fps the 50 ms default is 6 frames.
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
    """Which decoder bins lie wholly inside `states`.

    A bin survives only if both of its edges do, which is the same rule
    :func:`spikeshpc.states.intervals_between_frames` applies to inter-frame
    intervals: a bin straddling a state boundary belongs to neither state.
    """
    _, keep = frames_in_states(data.edges, intervals, states)
    return keep


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

    Returns two boolean arrays over all bins; both are subsets of ``mask`` and
    they do not overlap.
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
        n_test = max(1, int(round(test_fraction * n_blocks)))
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
    """Tuning curves for every unit, from the bins under `mask` only.

    ``min_rate_hz`` floors the curves. Without it an angle a unit happened
    never to fire at has rate zero, its log-likelihood is -inf, and a single
    spike there vetoes that heading outright however much the rest of the
    population likes it -- one unit's sampling gap becoming a hard constraint.
    The floor makes a surprising spike merely expensive.

    Angle bins the animal barely visited during training are reported in
    ``occupancy_s``; a warning names them, because a curve there is an estimate
    from almost nothing and the decoder has no way to know that.
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


# ── dynamics ─────────────────────────────────────────────────────────────
def movement_variance(
    data: DecoderData,
    mask: np.ndarray | None = None,
    per_100hz: float = 2.0,
) -> float:
    """Random-walk variance in deg^2 per decoder bin.

    With ``mask`` given, it is measured: the variance of the frame-to-frame
    change in heading, scaled to the bin width. With ``mask`` None it is the
    ``per_100hz`` convention carried over from ``run_decoder.py`` -- deg^2 per
    10 ms bin, scaled linearly with bin duration as diffusion requires.
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
    """p(angle now | angle one bin ago) as a wrapped Gaussian random walk.

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
    """log p(spikes in bin | head direction), up to a constant, per (bin, angle).

    Units are taken to be conditionally independent given head direction and
    Poisson within a bin, so the population log-likelihood is the sum over
    units of ``n log(lambda dt) - lambda dt``. Terms that do not depend on the
    angle -- ``log(dt)`` and ``log(n!)`` -- are dropped: they shift every
    column of a row by the same amount and vanish in the normalization.
    """
    counts = np.asarray(counts, dtype=float)
    rate_hz = np.asarray(rate_hz, dtype=float)
    expected = np.asarray(duration_s, dtype=float)[:, None] * rate_hz.sum(axis=0)
    return counts @ np.log(rate_hz) - expected


def _runs(mask: np.ndarray, duration_s: np.ndarray, max_gap_s: float, min_run_s: float):
    """Contiguous stretches of `mask`, as (start, stop) slices into all bins.

    Belief propagates along a run and is reset at each new one: the prior says
    the head has not moved far since the previous bin, which is a claim about
    an adjacent bin and nothing else. Carrying it across the hour between two
    wake epochs would assert the animal ended where it began.
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


def _forward_backward(log_likelihood, transition, acausal=True):
    """Causal and acausal posteriors over one run, by the HMM forward-backward.

    The causal posterior at bin t uses spikes up to t only, so it is what a
    decoder running live would report. The acausal one also uses everything
    after t; it is better, and it is the right choice for offline analysis, but
    it cannot be read as a prediction.
    """
    n_t, n_x = log_likelihood.shape
    # Subtracting each row's max before exponentiating keeps the likelihood
    # away from underflow; it is a per-row constant, so the normalized
    # posterior is unchanged.
    likelihood = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
    uniform = np.full(n_x, 1.0 / n_x)

    def normalize(vector):
        total = vector.sum()
        return vector / total if total > 0 else uniform.copy()

    causal = np.empty((n_t, n_x))
    posterior = uniform
    for t in range(n_t):
        prior = posterior if t == 0 else posterior @ transition
        posterior = normalize(prior * likelihood[t])
        causal[t] = posterior

    if not acausal:
        return causal, None

    smoothed = np.empty((n_t, n_x))
    smoothed[-1] = causal[-1]
    backward = np.ones(n_x)
    for t in range(n_t - 2, -1, -1):
        backward = normalize(transition @ (likelihood[t + 1] * backward))
        smoothed[t] = normalize(causal[t] * backward)
    return causal, smoothed


@dataclass
class Decoded:
    """One decoded stretch: the MAP angle per bin and how sure it was.

    ``actual_deg`` is the measured heading. During REM it is whatever the
    OptiTrack rigid body was pointing at while the animal slept, so it is not
    ground truth for anything and ``metrics`` is left empty.
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
    """How close the decode came, by four measures that fail differently.

    The median absolute error is the headline because it is what a reader
    pictures and because it survives the occasional bin where the population
    falls silent and the posterior wanders. The mean and the RMSE are given
    alongside precisely because they do not: the gap between median and RMSE is
    the size of the tail. ``frac_within_deg`` is the same tail question asked
    directly, and the circular correlation is the only one of the four that a
    constant offset -- a miscalibrated head frame -- would not punish.
    """
    error = circular_difference(decoded_deg, actual_deg)
    error = error[np.isfinite(error)]
    if error.size == 0:
        return {}
    absolute = np.abs(error)
    return {
        "median_abs_error_deg": float(np.median(absolute)),
        "mean_abs_error_deg": float(absolute.mean()),
        "rmse_deg": float(np.sqrt(np.mean(error**2))),
        "frac_within_deg": float(np.mean(absolute <= tolerance_deg)),
        "tolerance_deg": float(tolerance_deg),
        "circular_correlation": circular_correlation(decoded_deg, actual_deg),
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

    log_likelihood = poisson_log_likelihood(data.counts, data.duration_s, model.rate_hz)

    posteriors, indices, run_ids = [], [], []
    for run_id, (a, b) in enumerate(runs):
        causal, smoothed = _forward_backward(
            log_likelihood[a:b], transition, acausal=acausal
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

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(posterior > 0, posterior * np.log2(posterior), 0.0)
    entropy = -terms.sum(axis=1)

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
    if with_metrics:
        result.metrics = decoding_metrics(decoded_deg, actual_deg, tolerance_deg)
    result.metrics["mean_posterior_max"] = float(np.mean(result.posterior_max))
    result.metrics["mean_entropy_bits"] = float(np.mean(entropy))
    return result


# ── the control ──────────────────────────────────────────────────────────
@dataclass
class ShuffleTest:
    """An observed statistic against the null distribution it is judged by."""

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
    **decode_kwargs,
) -> ShuffleTest:
    """Re-decode `n_shuffles` times with the coding destroyed, for comparison.

    ``kind="shift"`` circularly shifts the spike data against the heading by at
    least ``min_shift_s``. Every unit moves together, so firing rates, burst
    structure, population synchrony and the animal's occupancy all survive
    intact and only their alignment in time is broken. That is the null for
    "does this decoder track the animal": use it wherever there is a measured
    heading to be wrong about, i.e. on the wake test set.

    ``kind="units"`` instead permutes which tuning curve belongs to which unit.
    Rates and synchrony again survive; what breaks is the population code
    itself. This is the null to use on REM, where there is no heading to
    misalign against and the question is whether the posterior is sharper and
    more coherent than a scrambled population would make it -- so pair it with
    a metric like ``mean_posterior_max``.
    """
    if kind not in ("shift", "units"):
        raise ValueError(f"kind must be 'shift' or 'units', got {kind!r}")

    rng = np.random.default_rng(seed)
    observed = decode(
        data, model, mask, keep_posterior=False, label="observed", **decode_kwargs
    )
    if metric not in observed.metrics:
        raise ValueError(f"metric {metric!r} is not one of {sorted(observed.metrics)}")

    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        if kind == "shift":
            min_shift = max(1, int(round(min_shift_s / data.bin_s)))
            if 2 * min_shift >= data.n_bins:
                raise ValueError(
                    f"min_shift_s={min_shift_s} leaves no room to shift "
                    f"{data.duration_s.sum():.0f}s of data"
                )
            shift = int(rng.integers(min_shift, data.n_bins - min_shift))
            surrogate = DecoderData(
                counts=np.roll(data.counts, shift, axis=0),
                heading_deg=data.heading_deg,
                duration_s=data.duration_s,
                time_s=data.time_s,
                edges=data.edges,
                unit_ids=data.unit_ids,
                bin_frames=data.bin_frames,
            )
            shuffled_model = model
        else:
            surrogate = data
            order = rng.permutation(len(model.unit_ids))
            shuffled_model = EncodingModel(
                bin_centers_deg=model.bin_centers_deg,
                rate_hz=model.rate_hz[order],
                unit_ids=model.unit_ids,
                occupancy_s=model.occupancy_s,
                train_time_s=model.train_time_s,
                min_rate_hz=model.min_rate_hz,
            )

        run = decode(
            surrogate,
            shuffled_model,
            mask,
            keep_posterior=False,
            label=f"shuffle {i}",
            **decode_kwargs,
        )
        null[i] = run.metrics[metric]
        if progress and (i + 1) % 10 == 0:
            print(f"      shuffle {i + 1}/{n_shuffles}", end="\r")

    # error-like metrics are better when small; confidence-like ones when large
    better = "lower" if "error" in metric or "rmse" in metric else "higher"
    hits = (
        np.count_nonzero(null <= observed.metrics[metric])
        if better == "lower"
        else np.count_nonzero(null >= observed.metrics[metric])
    )
    return ShuffleTest(
        metric=metric,
        observed=float(observed.metrics[metric]),
        null=null,
        p_value=float((1 + hits) / (n_shuffles + 1)),
        better=better,
        kind=kind,
    )


# ── the whole thing ──────────────────────────────────────────────────────
@dataclass
class DecoderRun:
    """Everything one call to :func:`run_decoder` produced."""

    data: DecoderData
    model: EncodingModel
    train_mask: np.ndarray
    test_mask: np.ndarray
    test: Decoded
    test_shuffle: ShuffleTest | None = None
    rem: Decoded | None = None
    rem_shuffle: ShuffleTest | None = None
    rem_mask: np.ndarray | None = None
    movement_var_deg2: float = float("nan")

    def summary(self) -> str:
        lines = [
            f"{self.model}",
            f"  train {self.data.duration_s[self.train_mask].sum():.0f}s / "
            f"test {self.data.duration_s[self.test_mask].sum():.0f}s",
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
        if self.rem is not None:
            lines.append(f"  REM: {self.rem}")
            lines.append(
                f"    mean posterior max = "
                f"{self.rem.metrics['mean_posterior_max']:.3f} "
                f"(wake test {self.test.metrics['mean_posterior_max']:.3f})"
            )
            if self.rem_shuffle is not None:
                lines.append(f"    vs shuffle: {self.rem_shuffle}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


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
    keep_posterior: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> DecoderRun:
    """Train on wake, test on held-out wake against a shuffle, then decode REM.

    ``sorting``: may be a sorting or a sorting analyzer

    ``unit_ids``: the head-direction-tuned units you selected

    ``heading_deg`` and ``frame_times`` are the shutter-aligned pair

    ``intervals``: ``scoring.intervals`` from :func:`spikeshpc.load_states`

    ``bin_s``: bin size in seconds. To set to camera frame rate, use (1/frame rate). Default 1/60

    ``n_angle_bins``: number of bins to devide heading space into. Default 180

    ``smooth_sigma_deg``: how much to smooth tuning curves. Default 10 degrees

    ``min_rate_hz``: applies this floor to tuning curves to avoid overweighting random spikes. Default 0.1 hz

    ``test_fraction``: test/train split. Default 0.3

    ``split_mode``: how to split the wake data into train/test chunks. Can be 'blocks' or 'contiguous'. Default 'blocks'

    ``block_s``: if 'blocks' mode is used, how long in seconds the blocks should be. Default 60

    ``movement_var_deg2``: maximum degrees/bin to allow for heading changes/bin.
      ``None`` = (2 deg^2 per 10 ms bin, scaled to ``bin_s``) per convention. Default
      ``'estimate'`` = estimate from the actual heading
      ``np.inf`` = remove limit completely
      ``number`` = fix at this value

    ``acausal``: whether to use causal or acausal decoder. Default True (acausal)

    ``tolerance_deg``: for evaluating model. Acceptable variance from true heading. Default 10 degrees

    ``n_shuffles``: number of shuffles for control analysis. Default 100

    ``min_shift_s``: amount to shift spike trains vs heading for shuffle analysis. Default 30 s

    ``decode_rem``: whether or not to run decoder on REM data. Default True. Set False for fine-tuning model

    ``keep_posterior``: whether to save posteriors for REM analysis. Default True

    ``seed``: seed for generating the random train/test split. Default 0

    ``verbose``: print progress on model generation. Default True

    Tuning guide
    ------------
    The knob that matters most is not in this list: it is which units you pass
    in. A decoder is a weighted vote of tuning curves, so adding a unit with a
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

    **seed** (0) -- fixes both the train/test split and the shuffles. Vary it
      to check a result is not an artefact of one particular split; if the
      metrics move a lot across seeds, the test set is too small.
    """
    data = prepare_decoder_data(sorting, unit_ids, heading_deg, frame_times, bin_s)
    if verbose:
        print(f"  {data}")

    wake_mask = state_interval_mask(data, intervals, "WAKE")
    if not wake_mask.any():
        raise ValueError("no decoder bin falls inside a WAKE interval")

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

    shared = dict(
        movement_var_deg2=movement_var_deg2,
        acausal=acausal,
        tolerance_deg=tolerance_deg,
    )
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
            **shared,
        )
        if verbose:
            print(f"    {test_shuffle}")

    rem = rem_shuffle = rem_mask = None
    if decode_rem:
        rem_mask = state_interval_mask(data, intervals, "REM")
        if not rem_mask.any():
            warnings.warn("no decoder bin falls inside a REM interval", stacklevel=2)
            rem_mask = None
        else:
            rem = decode(
                data,
                model,
                rem_mask,
                keep_posterior=keep_posterior,
                with_metrics=False,
                label="REM",
                **shared,
            )
            if verbose:
                print(f"  {rem}")
            if n_shuffles:
                # no heading to be wrong about in REM, so the null has to break
                # the population code rather than its alignment in time
                rem_shuffle = shuffle_test(
                    data,
                    model,
                    rem_mask,
                    kind="units",
                    metric="mean_posterior_max",
                    n_shuffles=n_shuffles,
                    seed=seed,
                    with_metrics=False,
                    **shared,
                )
                if verbose:
                    print(f"    {rem_shuffle}")

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
        movement_var_deg2=movement_var_deg2,
    )


# ── looking at it ────────────────────────────────────────────────────────
def plot_encoding_model(model: EncodingModel, sort_by_preferred: bool = True, ax=None):
    """The tuning curves the decoder is using, as a units x heading heatmap.

    Each unit's curve is scaled to its own peak, so the picture is about where
    a unit fires rather than how hard. Sorted by preferred direction, a healthy
    head-direction population is a clean diagonal band.
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
        cmap="viridis",
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
    """Decoded vs. actual heading over a window, on top of the posterior.

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
        color=ACTUAL_COLOR, lw=1.3, label="actual", zorder=3,
    )
    ax.plot(
        *break_at(time, decoded.decoded_deg[window], breaks),
        color=DECODED_COLOR, lw=1.3, label="decoded", zorder=4,
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


def plot_error(decoded: Decoded, axes=None):
    """Error histogram and an actual-vs-decoded confusion map.

    The confusion map is the one that shows *how* a decoder fails. A bright
    diagonal is success; a bright anti-diagonal or a constant offset from the
    diagonal is a systematic mapping error rather than noise, and a bright
    horizontal band is the decoder falling back on one favourite heading
    whenever the evidence is thin.
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
        cmap="magma",
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
    """The null distribution with the observed value marked on it."""
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
