"""Sync an OptiTrack take to an ephys recording via a camera-shutter TTL.

The OptiTrack camera system drives a TTL line high while a frame is exposing
and lets it fall back to 0V the instant the shutter closes and the frame is
acquired, so the falling edges on that channel are the frame-acquisition
timestamps in the ephys recording's own clock.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import OptitrackTake


def ttl_threshold(trace, max_iter: int = 50):
    """Crossing level for a two-state signal, whatever its duty cycle.

    Isodata: start at the midpoint of the range and iterate to the midpoint
    between the means of the two classes it separates.

    A percentile threshold cannot do this job. A camera exposure pulse can be
    a couple of samples out of a 250-sample frame period, and once the high
    state occupies under 1% of the trace the 99th percentile sits in the
    *baseline* -- the threshold collapses onto the noise floor and every
    baseline wiggle reads as an edge. Isodata depends on the two levels, not
    on how much time is spent at each, and recovers the right rate down to
    one-sample pulses.

    Returns None for a trace with no range at all.
    """
    trace = np.asarray(trace, dtype=float)
    lo, hi = float(np.min(trace)), float(np.max(trace))
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return None
    t = (lo + hi) / 2
    for _ in range(max_iter):
        below, above = trace <= t, trace > t
        if not below.any() or not above.any():
            return None
        new = (trace[below].mean() + trace[above].mean()) / 2
        if np.isclose(new, t):
            break
        t = new
    return float(t)


def rail_fraction(trace, threshold=None) -> float:
    """Fraction of samples sitting at either level: ~1 for a TTL, ~0.5 for noise.

    Measured against the two class means either side of `threshold`, so a
    short exposure pulse still scores ~1. This is what separates a real TTL
    from a floating input, where the threshold is meaningless and the detector
    would otherwise return millions of noise crossings.
    """
    trace = np.asarray(trace, dtype=float)
    if threshold is None:
        threshold = ttl_threshold(trace)
    if threshold is None:
        return 0.0
    below, above = trace <= threshold, trace > threshold
    if not below.any() or not above.any():
        return 0.0
    low_mean, high_mean = trace[below].mean(), trace[above].mean()
    span = high_mean - low_mean
    if span <= 0:
        return 0.0
    near_low = np.mean(trace <= low_mean + 0.1 * span)
    near_high = np.mean(trace >= high_mean - 0.1 * span)
    return float(near_low + near_high)


def extract_shutter_close_times(
    events_raw, channel_id: str = "ADC0", min_rail_fraction: float = 0.8, force: bool = False
) -> np.ndarray:
    """Falling-edge (shutter-close) timestamps on ``channel_id``, in recording time.

    The threshold is found by :func:`ttl_threshold`, which works whether
    ``get_traces`` returns raw ADC counts or scaled volts, and whatever the
    exposure pulse's duty cycle.

    Channels whose :func:`rail_fraction` falls below ``min_rail_fraction``
    carry no TTL -- an unconnected input would otherwise yield millions of
    noise crossings -- and are rejected. Pass ``force=True`` to override, or
    call :func:`describe_analog_channels` to find the right channel.
    """
    shutter_close_times = []
    for seg_idx in range(events_raw.get_num_segments()):
        trace = events_raw.get_traces(
            segment_index=seg_idx, channel_ids=[channel_id]
        ).flatten()
        times = events_raw.get_times(segment_index=seg_idx)

        threshold = ttl_threshold(trace)
        rails = rail_fraction(trace, threshold)
        if (threshold is None or rails < min_rail_fraction) and not force:
            raise ValueError(
                f"Channel {channel_id!r} does not look like a TTL: only "
                f"{rails:.1%} of samples sit at either level (a square wave is "
                f">{min_rail_fraction:.0%}, at any duty cycle). It is probably "
                "an unconnected input, or the shutter signal is on a different "
                "channel. Run optitrack.describe_analog_channels(events_raw) "
                "to see every channel, or pass force=True to threshold it anyway."
            )
        if threshold is None:  # force=True on a flat trace
            continue

        above = trace > threshold
        falling_sample = np.flatnonzero(above[:-1] & ~above[1:]) + 1
        shutter_close_times.append(times[falling_sample])

    return np.concatenate(shutter_close_times)


def describe_analog_channels(
    events_raw,
    segment_index: int = 0,
    n_chunks: int = 20,
    chunk_samples: int = 30000,
    min_rail_fraction: float = 0.8,
    verbose: bool = True,
):
    """Which channels of an analog stream actually carry a TTL?

    Samples `n_chunks` windows spread through the recording rather than
    reading the whole thing, and reports each channel's rail fraction and
    crossing rate. A shutter TTL shows a rail fraction near 1 and a crossing
    rate matching the camera frame rate.
    """
    fs = events_raw.get_sampling_frequency()
    n_total = events_raw.get_num_frames(segment_index=segment_index)
    starts = np.linspace(0, max(n_total - chunk_samples, 0), n_chunks).astype(int)
    starts = np.unique(starts)

    rows = []
    for channel_id in events_raw.channel_ids:
        pieces = [
            events_raw.get_traces(
                segment_index=segment_index,
                channel_ids=[channel_id],
                start_frame=int(s),
                end_frame=int(min(s + chunk_samples, n_total)),
            ).flatten()
            for s in starts
        ]
        trace = np.concatenate(pieces)
        threshold = ttl_threshold(trace)
        rails = rail_fraction(trace, threshold)
        if threshold is None:
            crossings, duty = 0, 0.0
        else:
            above = trace > threshold
            crossings = int(np.count_nonzero(above[:-1] & ~above[1:]))
            duty = float(above.mean())
        rows.append(
            {
                "channel_id": str(channel_id),
                "min": float(trace.min()),
                "max": float(trace.max()),
                "threshold": threshold,
                "duty_cycle": duty,
                "rail_fraction": rails,
                "crossings_per_s": crossings / (len(trace) / fs),
                "looks_like_ttl": rails >= min_rail_fraction,
            }
        )

    if verbose:
        print(f"{'channel':<14}{'min':>11}{'max':>11}{'duty':>8}"
              f"{'rail frac':>11}{'edges/s':>10}   verdict")
        print("-" * 76)
        for r in rows:
            print(f"{r['channel_id']:<14}{r['min']:>11.4g}{r['max']:>11.4g}"
                  f"{r['duty_cycle']:>8.2%}{r['rail_fraction']:>11.3f}"
                  f"{r['crossings_per_s']:>10.1f}"
                  f"   {'TTL' if r['looks_like_ttl'] else 'not a TTL'}")
        if not any(r["looks_like_ttl"] for r in rows):
            print("\n  No channel on this stream looks like a TTL. Check that the "
                  "shutter\n  output was connected for this session, and that this "
                  "is the right stream.")
    return rows


def plot_shutter_close_sanity_check(
    events_raw,
    shutter_close_times: np.ndarray,
    channel_id: str = "ADC0",
    n_events: int = 2,
    window: float = 0.02,
    segment_index: int = 0,
):
    """Zoom in on the first/last ``n_events`` detected events side by side.

    Rather than eyeballing one arbitrary time window, this shows the falling-
    edge detection holds up at both ends of the recording. Assumes a single
    segment (``segment_index``), which is what OneBox recordings are here.
    """
    times = events_raw.get_times(segment_index=segment_index)
    trace = events_raw.get_traces(
        segment_index=segment_index, channel_ids=[channel_id]
    ).flatten()

    n = len(shutter_close_times)
    check_indices = list(range(min(n_events, n))) + list(
        range(max(n - n_events, 0), n)
    )
    check_indices = sorted(set(check_indices))

    fig, axes = plt.subplots(
        1, len(check_indices), figsize=(4 * len(check_indices), 3), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, event_idx in zip(axes, check_indices):
        t_event = shutter_close_times[event_idx]
        in_window = (times >= t_event - window) & (times <= t_event + window)
        ax.plot(times[in_window], trace[in_window])
        ax.axvline(t_event, color="r", linestyle="--", linewidth=1)
        ax.set_title(f"event {event_idx} / {n - 1}")
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel(f"{channel_id} (raw)")
    fig.tight_layout()
    return fig


def save_shutter_close_times(shutter_close_times: np.ndarray, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    np.save(out_path, shutter_close_times)
    return out_path


def cross_check_with_optitrack_csv(
    shutter_close_times: np.ndarray, optitrack_csv_path: str | Path
) -> dict:
    """Compare detected event count/timing against the CSV's own frame count/rate."""
    with open(optitrack_csv_path, "r") as f:
        header_fields = f.readline().strip().split(",")
    header = dict(zip(header_fields[0::2], header_fields[1::2]))

    optitrack_total_frames = int(header["Total Exported Frames"])
    optitrack_frame_rate = float(header["Capture Frame Rate"])

    n_detected = len(shutter_close_times)
    expected_period = 1 / optitrack_frame_rate
    isis = np.diff(shutter_close_times)
    bad_isis = np.flatnonzero(np.abs(isis - expected_period) > 0.2 * expected_period)

    n_regular = len(isis) - len(bad_isis)
    # A genuine shutter train is regular to within a few frames. Anything else
    # means the wrong channel was thresholded, so say so rather than returning
    # a dict of nonsense for the caller to squint at.
    plausible = (
        abs(optitrack_total_frames - n_detected) < 0.01 * optitrack_total_frames
        and len(isis) > 0
        and n_regular > 0.99 * len(isis)
    )
    if not plausible:
        print(
            f"  WARNING: detected {n_detected} events against "
            f"{optitrack_total_frames} CSV frames, and only {n_regular}/"
            f"{len(isis)} intervals are within 20% of "
            f"{1 / optitrack_frame_rate * 1000:.2f} ms. This is not a "
            f"{optitrack_frame_rate:g} Hz shutter train -- check the channel "
            "with optitrack.describe_analog_channels(events_raw)."
        )

    return {
        "optitrack_total_frames": optitrack_total_frames,
        "optitrack_frame_rate": optitrack_frame_rate,
        "n_detected": n_detected,
        "frame_count_difference": optitrack_total_frames - n_detected,
        "n_regular_intervals": n_regular,
        "n_intervals": len(isis),
        "irregular_interval_indices": bad_isis,
        "plausible": plausible,
    }


def align_frames_to_shutter_events(
    shutter_close_times: np.ndarray,
    take: OptitrackTake,
    assume_missing: str = "start",
) -> np.ndarray:
    """CSV frame indices that line up 1:1 with ``shutter_close_times``.

    The detected event count and the CSV's frame count can differ by a
    frame or two at a recording boundary (a camera exposure that started or
    ended outside the ephys recording's own time window). Which end the
    missing frame(s) are on isn't recoverable from the counts alone --
    verify with :func:`plot_shutter_close_sanity_check` and pass
    ``assume_missing="end"`` if the default looks wrong.
    """
    n_detected = len(shutter_close_times)
    n_csv = len(take.frame_numbers)
    diff = n_csv - n_detected
    if diff < 0:
        raise ValueError(
            f"More detected shutter-closure events ({n_detected}) than OptiTrack "
            f"CSV frames ({n_csv}); check `channel_id` / detection threshold."
        )

    if assume_missing == "start":
        return np.arange(diff, n_csv)
    elif assume_missing == "end":
        return np.arange(0, n_csv - diff)
    raise ValueError(f"assume_missing must be 'start' or 'end', got {assume_missing!r}")
