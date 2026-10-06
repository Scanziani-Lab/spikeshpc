"""Turns in a heading trace: how often it rotates, which way, and how long it holds still.

Written for the question whether a head-direction population, read out by the
decoder, reports the head turning while it is still -- after a lesion, say. The
same measures are taken of the decoded heading and of the measured (OptiTrack)
one, so the two can be set side by side. Everything runs on the decoder's bins,
one contiguous stretch at a time:

  * Angular velocity is the rotation over ``smooth_s``, centred on each bin,
    divided by the time it spans. The averaging is what makes the decoded trace
    usable: it moves on the decoder's 4-degree grid, so a bin-to-bin difference
    is either 0 or at least 40 deg/s.
  * A jump -- a single-bin step larger than ``max_step_deg`` -- is not a
    rotation. The decoded heading is the posterior's peak, and it can hop from
    one peak to another; the measured head never moves 45 degrees in a 100 ms
    bin. A jump splits the stretch, is counted on its own, and is never
    averaged over. A piece too short for half of ``smooth_s`` (a flicker
    between two jumps) has no velocity and is left out of every measure.
  * The heading is **constant** in a bin where |velocity| is below
    ``turn_threshold``, and **turning** otherwise. A **turn** is a run of
    turning bins in one direction.
  * **Net drift** is the net rotation per minute, over the same pieces, jumps
    left out. It needs no threshold: turning back and forth cancels in it,
    while a steady spin too slow to count as turning adds up.
  * Clockwise is as seen from above; see :func:`clockwise_sign`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decoder import circular_difference

__all__ = [
    "Turns",
    "clockwise_sign",
    "find_turns",
    "plot_turn_summary",
    "plot_turn_sweep",
    "plot_turn_trace",
]

STATES = ("wake", "REM")
REM_SHADE = "0.93"  # behind the REM columns of the summary


def clockwise_sign(flip_direction: bool) -> int:
    """Get the sign of clockwise rotation for a heading convention.

    ``optitrack.compute_heading`` measures from +Z toward +X, the right-handed
    sense about +Y (up): counterclockwise seen from above. ``flip_direction``
    negates it, so the tuning notebook's heading (``flip_direction=True``) grows
    clockwise.

    Parameters
    ----------
    flip_direction : bool
        The ``flip_direction`` the heading was computed with.

    Returns
    -------
    int
        +1 if heading grows clockwise (seen from above), -1 if it shrinks.
    """
    return 1 if flip_direction else -1


@dataclass
class Turns:
    """One heading trace's turns, from :func:`find_turns`. Clockwise is positive.

    Attributes
    ----------
    label : str
        Name of the trace.
    turn_threshold : float
        Angular velocity above which a bin is turning, in deg/s.
    smooth_s : float
        Velocity smoothing window, in seconds.
    max_step_deg : float
        Largest step between bins that is not a jump.
    bin_s : float
        Bin width, in seconds.
    duration_s : float
        Length of every bin together, in seconds.
    analysed_s : float
        Time in bins with a velocity: not in a flicker between jumps.
    constant_s : float
        Of that, the time below `turn_threshold`.
    net_rotation_deg : float
        Net rotation over the analysed pieces, jumps left out.
    velocity_deg_s : numpy.ndarray
        Angular velocity per bin, NaN where not analysed.
    jump_time_s : numpy.ndarray
        The bin each jump landed in.
    turn_start_s, turn_stop_s : numpy.ndarray
        First and last turning bin of each turn.
    turn_direction : numpy.ndarray
        +1 clockwise, -1 counterclockwise, per turn.
    turn_rotation_deg : numpy.ndarray
        Signed rotation over each turn and half a window either side.
    """

    label: str
    turn_threshold: float  # deg/s
    smooth_s: float
    max_step_deg: float
    bin_s: float
    duration_s: float  # every bin
    analysed_s: float  # bins with a velocity: not in a flicker between jumps
    constant_s: float  # of those, the ones below turn_threshold
    net_rotation_deg: float  # over the analysed pieces, jumps left out
    velocity_deg_s: np.ndarray  # per bin, NaN where not analysed
    jump_time_s: np.ndarray  # the bin each jump landed in
    turn_start_s: np.ndarray  # first turning bin
    turn_stop_s: np.ndarray  # last turning bin
    turn_direction: np.ndarray  # +1 clockwise, -1 counterclockwise
    turn_rotation_deg: np.ndarray  # signed, over the turn and half a window either side

    @property
    def n_turns(self) -> int:
        return len(self.turn_direction)

    @property
    def n_clockwise(self) -> int:
        return int(np.sum(self.turn_direction > 0))

    @property
    def n_counterclockwise(self) -> int:
        return int(np.sum(self.turn_direction < 0))

    @property
    def n_jumps(self) -> int:
        return len(self.jump_time_s)

    @property
    def constant_fraction(self) -> float:
        return self.constant_s / self.analysed_s if self.analysed_s else np.nan

    @property
    def turns_per_min(self) -> float:
        return 60.0 * self.n_turns / self.analysed_s if self.analysed_s else np.nan

    @property
    def drift_deg_per_min(self) -> float:
        """Net rotation per analysed minute.

        A steady spin shows here, back-and-forth turning does not.

        Returns
        -------
        float
            Degrees per minute; NaN if nothing was analysed.
        """
        return 60.0 * self.net_rotation_deg / self.analysed_s if self.analysed_s else np.nan

    @property
    def jumps_per_min(self) -> float:
        return 60.0 * self.n_jumps / self.duration_s if self.duration_s else np.nan

    @property
    def clockwise_ratio(self) -> float:
        """Get the number of clockwise turns over counterclockwise ones.

        Returns
        -------
        float
            The ratio; inf with no counterclockwise turn, NaN with no turns.
        """
        cw, ccw = self.n_clockwise, self.n_counterclockwise
        if ccw:
            return cw / ccw
        return np.inf if cw else np.nan

    def as_row(self) -> dict:
        rotation = np.abs(self.turn_rotation_deg)
        return {
            "minutes": self.duration_s / 60.0,
            "analysed_min": self.analysed_s / 60.0,
            "constant_min": self.constant_s / 60.0,
            "constant_pct": 100.0 * self.constant_fraction,
            "n_turns": self.n_turns,
            "turns_per_min": self.turns_per_min,
            "n_cw": self.n_clockwise,
            "n_ccw": self.n_counterclockwise,
            "cw_ccw_ratio": self.clockwise_ratio,
            "median_turn_deg": float(np.median(rotation)) if rotation.size else np.nan,
            "net_rotation_deg": self.net_rotation_deg,
            "drift_deg_per_min": self.drift_deg_per_min,
            "n_jumps": self.n_jumps,
            "jumps_per_min": self.jumps_per_min,
            "turn_threshold": self.turn_threshold,
            "smooth_s": self.smooth_s,
            "max_step_deg": self.max_step_deg,
        }

    def __repr__(self) -> str:
        return (
            f"Turns({self.label}: {self.duration_s / 60:.1f} min, "
            f"{100 * self.constant_fraction:.0f}% constant, {self.n_turns} turns "
            f"({self.n_clockwise} CW / {self.n_counterclockwise} CCW), "
            f"drift {self.drift_deg_per_min:+.0f} deg/min, {self.n_jumps} jumps)"
        )


def find_turns(
    time_s,
    heading_deg,
    run_index=None,
    *,
    turn_threshold: float,
    smooth_s: float = 1.0,
    max_step_deg: float = 45.0,
    clockwise: int = 1,
    bin_s: float | None = None,
    label: str = "",
) -> Turns:
    """Find the turns in one heading trace; see the module docstring for the rules.

    The velocity window is the odd number of bins closest to `smooth_s`.

    Parameters
    ----------
    time_s, heading_deg : array-like
        Time and heading per bin, as a decode saves them.
    run_index : array-like, optional
        Marks the trace's contiguous stretches; without it, a gap of more than
        one and a half bins in `time_s` starts a new one.
    turn_threshold : float
        Angular velocity above which a bin is turning, in deg/s.
    smooth_s : float, default 1.0
        Velocity smoothing window, in seconds.
    max_step_deg : float, default 45.0
        Largest step between bins that is not a jump.
    clockwise : int, default 1
        :func:`clockwise_sign` of the heading's convention.
    bin_s : float, optional
        Bin width; defaults to the typical step of `time_s`.
    label : str, default ""
        Name of the trace.

    Returns
    -------
    Turns
        The turns found.

    Raises
    ------
    ValueError
        If the trace is empty, the inputs disagree in length, a parameter is
        invalid, or `bin_s` cannot be inferred.
    """
    time_s = np.asarray(time_s, dtype=float)
    heading = np.asarray(heading_deg, dtype=float)
    n = len(time_s)
    if n == 0:
        raise ValueError("the trace is empty")
    if heading.shape != (n,):
        raise ValueError(f"heading_deg has {heading.shape} entries, time_s has {n}")
    if clockwise not in (1, -1):
        raise ValueError(f"clockwise must be +1 or -1, not {clockwise!r}")
    for name, value in (("turn_threshold", turn_threshold), ("smooth_s", smooth_s),
                        ("max_step_deg", max_step_deg)):
        if not value > 0:
            raise ValueError(f"{name} must be positive, not {value!r}")
    if bin_s is None:
        steps = np.diff(time_s)
        steps = steps[steps > 0]
        if steps.size == 0:
            raise ValueError("bin_s cannot be inferred from fewer than two distinct times")
        bin_s = float(np.median(steps))

    runs = _runs(time_s, run_index, bin_s)
    step = circular_difference(heading[1:], heading[:-1])
    same_run = runs[1:] == runs[:-1]
    finite = np.isfinite(step)
    jump = same_run & finite & (np.abs(step) > max_step_deg)
    boundary = ~same_run | ~finite | jump

    # pieces between boundaries, and the heading unwrapped inside each one
    piece = np.concatenate([[0], np.cumsum(boundary)]).astype(np.int64)
    unwrapped = np.concatenate([[0.0], np.cumsum(np.where(boundary, 0.0, step))])
    starts = np.flatnonzero(np.concatenate([[True], boundary]))
    stops = np.concatenate([starts[1:], [n]]) - 1
    first, last = starts[piece], stops[piece]

    half = max(1, int(np.floor(smooth_s / (2.0 * bin_s) + 0.5)))
    index = np.arange(n)
    lo = np.maximum(index - half, first)
    hi = np.minimum(index + half, last)
    span = time_s[hi] - time_s[lo]
    analysed = np.isfinite(heading) & (span >= (half - 0.25) * bin_s)
    velocity = np.full(n, np.nan)
    velocity[analysed] = clockwise * (unwrapped[hi] - unwrapped[lo])[analysed] / span[analysed]

    speed = np.abs(np.where(analysed, velocity, 0.0))
    turning = analysed & (speed >= turn_threshold)
    direction = np.where(turning, np.sign(np.where(analysed, velocity, 0.0)), 0).astype(np.int64)
    new_group = np.ones(n, dtype=bool)
    new_group[1:] = (direction[1:] != direction[:-1]) | (piece[1:] != piece[:-1])
    group_start = np.flatnonzero(new_group)
    group_stop = np.concatenate([group_start[1:], [n]]) - 1
    keep = direction[group_start] != 0
    turn_start, turn_stop = group_start[keep], group_stop[keep]
    around_start = np.maximum(turn_start - half, first[turn_start])
    around_stop = np.minimum(turn_stop + half, last[turn_stop])
    # a piece is analysed in every bin or in none: it either spans half a window or not
    kept = analysed[starts]
    net_rotation = clockwise * float(np.sum((unwrapped[stops] - unwrapped[starts])[kept]))

    return Turns(
        label=label,
        turn_threshold=float(turn_threshold),
        smooth_s=float(smooth_s),
        max_step_deg=float(max_step_deg),
        bin_s=float(bin_s),
        duration_s=n * bin_s,
        analysed_s=float(analysed.sum()) * bin_s,
        constant_s=float((analysed & ~turning).sum()) * bin_s,
        net_rotation_deg=net_rotation,
        velocity_deg_s=velocity,
        jump_time_s=time_s[1:][jump],
        turn_start_s=time_s[turn_start],
        turn_stop_s=time_s[turn_stop],
        turn_direction=direction[turn_start],
        turn_rotation_deg=clockwise * (unwrapped[around_stop] - unwrapped[around_start]),
    )


def _runs(time_s, run_index, bin_s) -> np.ndarray:
    """Number the contiguous stretches of a trace.

    A new stretch starts at each change of `run_index` and at each time gap.

    Parameters
    ----------
    time_s : numpy.ndarray
        Time per bin.
    run_index : numpy.ndarray or None
        Run index per bin.
    bin_s : float
        Bin width, in seconds.

    Returns
    -------
    numpy.ndarray
        Stretch index per bin.

    Raises
    ------
    ValueError
        If `run_index` and `time_s` differ in length.
    """
    gap = np.diff(time_s) > 1.5 * bin_s
    if run_index is not None:
        run_index = np.asarray(run_index)
        if run_index.shape != time_s.shape:
            raise ValueError(f"run_index has {run_index.shape} entries, time_s has {time_s.shape}")
        gap |= run_index[1:] != run_index[:-1]
    return np.concatenate([[0], np.cumsum(gap)])


# ── plots ────────────────────────────────────────────────────────────────
def plot_turn_summary(table: pd.DataFrame, recordings, colors: dict, units: dict | None = None,
                      types: dict | None = None, axes=None):
    """Plot constant heading, turns, their direction and net drift, per recording.

    Columns run in `recordings` order, wake then REM, the REM ones shaded gray.
    In a wake column the decode and the measured heading sit side by side, and
    the diamond stands apart to their left.

    Four panels: percent of time at a constant heading; turns per minute, with
    each trace's total count beside its marker (a pair's pointing away from
    each other, so they never overprint); clockwise over counterclockwise turns
    on a log axis, where 1 is no preference, with the two counts placed the
    same way; and net drift in deg/min, clockwise up, which does not depend on
    the threshold.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per trace: ``recording``, ``state`` ("wake" or "REM"),
        ``source`` and the :meth:`Turns.as_row` measures. ``source`` is
        "decoded" (a filled circle in the recording's color), "optitrack" (the
        measured heading, an open square; drawn for wake only), "reference"
        (the baseline decoded with that recording's units, an open black
        diamond, as in :func:`~spikeshpc.decoder.plot_transfer_summary`) or
        "ring" (the recording's own ring from :mod:`spikeshpc.ring`, decoded
        from its units' co-firing: an open triangle in its color).
    recordings : sequence of str
        Recordings to draw, in order.
    colors : dict
        Colour of each recording.
    units : dict, optional
        Unit label of each recording.
    types : dict, optional
        Type label of each recording.
    axes : sequence of matplotlib.axes.Axes, optional
        The four axes to draw on.

    Returns
    -------
    sequence of matplotlib.axes.Axes
        The four axes.

    Raises
    ------
    ValueError
        If no recording of the table is in `recordings`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    recordings = [r for r in recordings if r in set(table["recording"])]
    if not recordings:
        raise ValueError("no recording of the table is in `recordings`")
    units, types = units or {}, types or {}
    if axes is None:
        _, axes = plt.subplots(4, 1, figsize=(max(8.0, 3.0 * len(recordings)), 11.5),
                               sharex=True)
    constant_ax, rate_ax, ratio_ax, drift_ax = axes

    rows = {(row.recording, row.state, row.source): row for row in table.itertuples()}
    # x offset within a column: a wake decode and the measured heading as a pair,
    # the baseline with the same units set apart to the left and the ring to the
    # right; each source keeps its offset in both columns. OptiTrack is drawn for
    # wake only.
    placement = {
        ("wake", "reference"): -0.36, ("wake", "decoded"): -0.06, ("wake", "optitrack"): 0.1,
        ("wake", "ring"): 0.3,
        ("REM", "reference"): -0.36, ("REM", "decoded"): -0.06, ("REM", "ring"): 0.14,
    }
    styles = {
        "decoded": lambda c: dict(marker="o", ls="none", color=c, ms=7, zorder=3),
        "optitrack": lambda c: dict(marker="s", ls="none", mfc="none", mec=c, mew=1.5, ms=7,
                                    zorder=3),
        "reference": lambda c: dict(marker="D", ls="none", mfc="none", mec="k", mew=1.2, ms=7,
                                    zorder=4),
        "ring": lambda c: dict(marker="^", ls="none", mfc="none", mec=c, mew=1.5, ms=7,
                               zorder=3),
    }
    # counts beside a decode's and the measured heading's markers. Side by side,
    # two labels at one height would print over each other, so a pair's point
    # outward: the higher marker's above, the lower one's below (the decode's
    # above on a tie).
    above = dict(xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                 fontsize=6, color="0.3")
    below = dict(above, xytext=(0, -7), va="top")
    counted = (  # panel, the measure it draws, its label
        (rate_ax, "turns_per_min", lambda row: f"{row.n_turns}"),
        (ratio_ax, "cw_ccw_ratio", lambda row: f"{row.n_cw}:{row.n_ccw}"),
    )

    ticks, labels, constants, rates, ratios, drifts = [], [], [], [], [], []
    for j, recording in enumerate(recordings):
        for k, state in enumerate(STATES):
            x = 2 * j + k
            ticks.append(x)
            labels.append(state)
            labelled = {}  # source -> (x, row) of the markers that carry counts
            for source, style in styles.items():
                row = rows.get((recording, state, source))
                offset = placement.get((state, source))
                if row is None or offset is None:
                    continue
                marker = style(colors[recording])
                constants.append(row.constant_pct)
                rates.append(row.turns_per_min)
                drifts.append(row.drift_deg_per_min)
                constant_ax.plot(x + offset, row.constant_pct, **marker)
                rate_ax.plot(x + offset, row.turns_per_min, **marker)
                drift_ax.plot(x + offset, row.drift_deg_per_min, **marker)
                if np.isfinite(row.cw_ccw_ratio) and row.cw_ccw_ratio > 0:
                    ratios.append(row.cw_ccw_ratio)
                    ratio_ax.plot(x + offset, row.cw_ccw_ratio, **marker)
                if source in ("decoded", "optitrack"):
                    labelled[source] = (x + offset, row)
            for ax, measure, text in counted:
                ys = {
                    source: getattr(row, measure) for source, (_, row) in labelled.items()
                    if np.isfinite(getattr(row, measure))
                    and (measure != "cw_ccw_ratio" or getattr(row, measure) > 0)  # as drawn
                }
                up = {source: True for source in ys}
                if len(ys) == 2:
                    up = {"decoded": ys["decoded"] >= ys["optitrack"],
                          "optitrack": ys["optitrack"] > ys["decoded"]}
                for source, y in ys.items():
                    ax.annotate(text(labelled[source][1]), (labelled[source][0], y),
                                **(above if up[source] else below))
        for ax in axes:
            ax.axvspan(2 * j + 0.5, 2 * j + 1.5, color=REM_SHADE, lw=0, zorder=0)
            if j:
                ax.axvline(2 * j - 0.5, color="0.85", lw=0.8, zorder=0)

    # markers, not bars, so the percent axis can start near the lowest value
    lowest = np.nanmin(constants) if np.isfinite(constants).any() else 0.0
    constant_ax.set_ylim(max(0.0, 10.0 * np.floor((lowest - 5.0) / 10.0)), 100.5)
    constant_ax.set_ylabel("constant heading\n(% of time)")
    highest = np.nanmax(rates) if np.isfinite(rates).any() else 1.0
    rate_ax.set_ylim(0, 1.2 * highest if highest > 0 else 1.0)  # room for the counts
    rate_ax.set_ylabel("turns per minute\n(total beside each)")
    ratio_ax.set_yscale("log", base=2)
    ratio_ax.axhline(1.0, color="0.6", lw=0.8, zorder=0)
    widest = max((abs(np.log2(r)) for r in ratios), default=0.0)
    reach = 2.0 ** max(1, int(np.ceil(widest + 0.25)))
    ratio_ax.set_ylim(1 / reach, reach)
    powers = 2.0 ** np.arange(-np.log2(reach), np.log2(reach) + 1)
    ratio_ax.set_yticks(powers)
    ratio_ax.set_yticklabels([f"{p:g}" for p in powers])
    ratio_ax.yaxis.set_minor_locator(NullLocator())
    ratio_ax.set_ylabel("CW : CCW turns\n(1 = no preference)")
    drift_ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    span = np.nanmax(np.abs(drifts)) if np.isfinite(drifts).any() else 0.0
    span = 1.15 * span if span > 0 else 1.0
    drift_ax.set_ylim(-span, span)
    drift_ax.set_ylabel("net drift\n(deg/min, CW up)")

    # two rows of labels: wake / REM under each column, the recording under each pair
    names = []
    for recording in recordings:
        name = recording + (f" ({types[recording]})" if recording in types else "")
        names.append(name + (f", {units[recording]} units" if recording in units else ""))
    bottom = axes[-1]
    bottom.set_xticks(ticks)
    bottom.set_xticklabels(labels, fontsize=8)
    bottom.set_xticks([2 * j + 0.5 for j in range(len(recordings))], minor=True)
    bottom.set_xticklabels(names, minor=True, fontsize=8)
    bottom.tick_params(axis="x", which="minor", length=0, pad=16)
    bottom.set_xlim(-0.6, 2 * len(recordings) - 0.4)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([], [], label="decoded", **styles["decoded"]("k")),
        Line2D([], [], label="OptiTrack (measured)", **styles["optitrack"]("k")),
    ]
    if "reference" in set(table["source"]):
        handles.append(Line2D([], [], label="baseline, same units", **styles["reference"]("k")))
    if "ring" in set(table["source"]):
        handles.append(Line2D([], [], label="own ring (correlations)", **styles["ring"]("k")))
    constant_ax.legend(handles=handles, fontsize=7, frameon=False, loc="best", ncol=4)
    axes[0].figure.tight_layout()
    return axes


def plot_turn_sweep(table: pd.DataFrame, recordings, colors: dict, current: float | None = None,
                    min_turns: int = 20, axes=None):
    """Plot constant heading, turn rate and turn direction against the turn threshold.

    Rows of panels: the three measures; columns: wake, then REM. One line per
    recording and source, in the recording's color: solid decoded, dashed the
    measured heading (wake only; in REM the head does not turn), dotted the
    baseline decoded with that recording's units, dash-dot the recording's own
    ring. Net drift has no threshold, so it is not here.

    Parameters
    ----------
    table : pandas.DataFrame
        As for :func:`plot_turn_summary`, with one set of rows per
        ``turn_threshold``.
    recordings : sequence of str
        Recordings to draw, in order.
    colors : dict
        Colour of each recording.
    current : float, optional
        Marks the threshold in use with a gray band.
    min_turns : int, default 20
        A ratio from fewer turns than this is not drawn.
    axes : numpy.ndarray of matplotlib.axes.Axes, optional
        The 3 x 2 axes to draw on.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The 3 x 2 axes.

    Raises
    ------
    ValueError
        If no recording of the table is in `recordings`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    recordings = [r for r in recordings if r in set(table["recording"])]
    if not recordings:
        raise ValueError("no recording of the table is in `recordings`")
    thresholds = np.sort(table["turn_threshold"].unique())
    if axes is None:
        _, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex=True, sharey="row")
    axes = np.asarray(axes)
    styles = {
        "decoded": dict(ls="-", marker="o"),
        "optitrack": dict(ls="--", marker="s", mfc="none"),
        "reference": dict(ls=":", marker="D", mfc="none"),
        "ring": dict(ls="-.", marker="^", mfc="none"),
    }

    ratios = []
    for column, state in enumerate(STATES):
        constant_ax, rate_ax, ratio_ax = axes[:, column]
        for recording in recordings:
            for source, style in styles.items():
                if state == "REM" and source == "optitrack":
                    continue
                rows = table[
                    (table["recording"] == recording)
                    & (table["state"] == state)
                    & (table["source"] == source)
                ].sort_values("turn_threshold")
                if rows.empty:
                    continue
                line = dict(color=colors[recording], lw=1.2, ms=4, **style)
                x = rows["turn_threshold"].to_numpy(dtype=float)
                constant_ax.plot(x, rows["constant_pct"].to_numpy(dtype=float), **line)
                rate_ax.plot(x, rows["turns_per_min"].to_numpy(dtype=float), **line)
                ratio = rows["cw_ccw_ratio"].to_numpy(dtype=float)
                enough = (rows["n_turns"].to_numpy() >= min_turns) & np.isfinite(ratio) & (ratio > 0)
                ratios.extend(ratio[enough])
                ratio_ax.plot(x, np.where(enough, ratio, np.nan), **line)
        constant_ax.set_title(state, fontsize=10)

    for ax in axes.flat:
        if current is not None:
            ax.axvline(current, color="0.85", lw=4, zorder=0)
        ax.set_xscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xticks(thresholds)
        ax.set_xticklabels([f"{t:g}" for t in thresholds], fontsize=8)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("turn_threshold (deg/s)")
    axes[0, 0].set_ylabel("constant heading\n(% of time)")
    axes[1, 0].set_ylabel("turns per minute")
    axes[1, 0].set_ylim(bottom=0)
    axes[2, 0].set_ylabel(f"CW : CCW turns\n(from {min_turns} turns)")
    widest = max((abs(np.log2(r)) for r in ratios), default=0.0)
    reach = 2.0 ** max(1, int(np.ceil(widest + 0.25)))
    powers = 2.0 ** np.arange(-np.log2(reach), np.log2(reach) + 1)
    for ax in axes[2]:
        ax.set_yscale("log", base=2)
        ax.axhline(1.0, color="0.6", lw=0.8, zorder=0)
        ax.set_ylim(1 / reach, reach)
        ax.set_yticks(powers)
        ax.set_yticklabels([f"{p:g}" for p in powers])
        ax.yaxis.set_minor_locator(NullLocator())

    handles = [Line2D([], [], color=colors[r], lw=2, label=r) for r in recordings]
    handles += [
        Line2D([], [], color="k", lw=1.2, ms=4, label="decoded", **styles["decoded"]),
        Line2D([], [], color="k", lw=1.2, ms=4, label="OptiTrack (measured)",
               **styles["optitrack"]),
    ]
    if "reference" in set(table["source"]):
        handles.append(Line2D([], [], color="k", lw=1.2, ms=4, label="baseline, same units",
                              **styles["reference"]))
    if "ring" in set(table["source"]):
        handles.append(Line2D([], [], color="k", lw=1.2, ms=4, label="own ring (correlations)",
                              **styles["ring"]))
    if current is not None:
        handles.append(Line2D([], [], color="0.85", lw=4, label=f"in use: {current:g} deg/s"))
    axes[0, 1].legend(handles=handles, fontsize=7, frameon=False, loc="best")
    axes[0, 0].figure.tight_layout()
    return axes


def plot_turn_trace(time_s, heading_deg, turns: Turns, measured_deg=None, start_s=None,
                    length_s: float = 60.0, color="k", axes=None):
    """Plot a stretch of one trace, to check the turn detection by eye.

    Top: the heading (dots) and optionally the measured one (a gray line), with
    each turn shaded -- blue clockwise, orange counterclockwise -- and each
    jump marked by a dashed line. Bottom: the angular velocity, clockwise up,
    against +/- ``turns.turn_threshold``.

    Parameters
    ----------
    time_s, heading_deg : array-like
        The trace `turns` was found in.
    turns : Turns
        Turns found in the trace.
    measured_deg : array-like, optional
        Measured heading in the same bins.
    start_s : float, optional
        Start of the stretch; defaults to the trace's first time.
    length_s : float, default 60.0
        Length of the stretch, in seconds.
    color : str, default "k"
        Colour of the heading dots.
    axes : sequence of matplotlib.axes.Axes, optional
        The two axes to draw on.

    Returns
    -------
    sequence of matplotlib.axes.Axes
        The two axes.

    Raises
    ------
    ValueError
        If no bin falls in the stretch.
    """
    import matplotlib.pyplot as plt

    time_s = np.asarray(time_s, dtype=float)
    if start_s is None:
        start_s = float(time_s[0])
    window = (time_s >= start_s) & (time_s < start_s + length_s)
    if not window.any():
        raise ValueError(f"no bin between {start_s} and {start_s + length_s} s")
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    heading_ax, velocity_ax = axes
    t = time_s[window] - start_s

    shade = {1: ("tab:blue", "clockwise turn"), -1: ("tab:orange", "counterclockwise turn")}
    named = set()
    for a, b, d in zip(turns.turn_start_s, turns.turn_stop_s, turns.turn_direction):
        if b < start_s or a >= start_s + length_s:
            continue
        c, name = shade[int(d)]
        heading_ax.axvspan(a - start_s, b - start_s + turns.bin_s, color=c, alpha=0.2, lw=0,
                           label=None if d in named else name)
        velocity_ax.axvspan(a - start_s, b - start_s + turns.bin_s, color=c, alpha=0.2, lw=0)
        named.add(d)
    jumps = turns.jump_time_s[(turns.jump_time_s >= start_s)
                              & (turns.jump_time_s < start_s + length_s)]
    for i, jump in enumerate(jumps):
        heading_ax.axvline(jump - start_s, color="0.5", ls="--", lw=0.6,
                           label="jump" if i == 0 else None)

    if measured_deg is not None:
        measured = np.asarray(measured_deg, dtype=float)[window]
        heading_ax.plot(t, measured, color="0.6", lw=1.2, label="measured")
    heading_ax.plot(t, np.asarray(heading_deg, dtype=float)[window], ".", color=color, ms=3,
                    label=turns.label or "heading")
    heading_ax.set_ylim(0, 360)
    heading_ax.set_yticks([0, 90, 180, 270, 360])
    heading_ax.set_ylabel("heading (deg)")
    heading_ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=5)

    velocity_ax.plot(t, turns.velocity_deg_s[window], color=color, lw=1)
    for sign in (1, -1):
        velocity_ax.axhline(sign * turns.turn_threshold, color="0.5", ls=":", lw=0.8)
    velocity_ax.axhline(0.0, color="0.8", lw=0.6)
    velocity_ax.set_ylabel("deg/s, CW up")
    velocity_ax.set_xlabel(f"time from {start_s:.0f} s (s)")
    velocity_ax.set_xlim(0, length_s)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].figure.tight_layout()
    return axes
