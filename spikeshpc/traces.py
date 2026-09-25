"""Reading and plotting a short stretch of raw traces from a long recording.

spikeinterface's plot_traces and get_traces only ever mean to read the frames
asked for, but on a long Open Ephys recording they can still take a whole
machine down, for two reasons that have nothing to do with the traces:

* the synchronized time vector. ``timestamps.npy`` holds one float64 per
  sample -- eleven gigabytes for 12.75 hours at 30 kHz -- and
  ``si.read_openephys(load_sync_timestamps=True)`` reads all of it into RAM.
  :func:`spikeshpc.io.read_openephys_synced` keeps it memory-mapped instead,
  and nothing here asks for more of it than the requested range.
* the lookup from time to frame. That is a binary search on the time vector,
  which is only meaningful if the vector is increasing. Concatenated sessions
  whose clocks restart, or a partly unsynced stream, are not; the search then
  lands anywhere, a 200 ms request becomes a span of hours, and get_traces
  tries to allocate all of it.

Here the time vector is only ever indexed -- a binary search on a memmap reads
a few dozen pages -- and a range is refused, rather than read, when the frames
it resolved to cannot possibly fit in the time asked for, or when the traces
would be larger than ``max_gb``.
"""

import warnings

import numpy as np


def _segment_index(recording, segment_index):
    if segment_index is None:
        if recording.get_num_segments() != 1:
            raise ValueError("Pass segment_index for a multi-segment recording.")
        return 0
    return int(segment_index)


def _time_vector(recording, segment_index):
    """The segment's time vector as stored -- a memmap stays a memmap.

    Never ``get_times()``: with no time vector that builds one, np.arange over
    every sample in the segment.
    """
    return recording.get_time_info(segment_index=segment_index)["time_vector"]


def time_range_to_frames(recording, time_range, segment_index=None,
                         relative: bool = False):
    """``(start_frame, end_frame)`` covering ``time_range`` seconds.

    ``time_range`` is on the recording's own clock -- the synchronized
    acquisition clock for anything read through
    :func:`spikeshpc.io.read_openephys_synced`, which rarely starts at zero.
    ``relative=True`` measures it from the segment's first sample instead.

    Raises rather than returning a span that cannot be right: an empty one
    (usually a range given on the wrong clock) or one holding more samples
    than ``time_range`` has room for at the sampling rate, which only a time
    vector that is not increasing can produce.
    """
    segment_index = _segment_index(recording, segment_index)
    t0, t1 = (float(t) for t in time_range)
    if not t1 > t0:
        raise ValueError(f"time_range must be increasing, got ({t0}, {t1}).")

    fs = recording.get_sampling_frequency()
    n = recording.get_num_samples(segment_index=segment_index)
    seg_start = float(recording.get_start_time(segment_index=segment_index))
    seg_end = float(recording.get_end_time(segment_index=segment_index))
    if relative:
        t0, t1 = t0 + seg_start, t1 + seg_start

    if t1 > seg_end + 1.0 / fs:
        warnings.warn(
            f"time_range ends after the segment ({seg_end:.6f} s); clipping to it.",
            stacklevel=2,
        )

    tv = _time_vector(recording, segment_index)
    if tv is None:
        frames = np.round((np.array([t0, t1]) - seg_start) * fs).astype(np.int64)
    else:
        frames = np.searchsorted(tv, [t0, t1], side="left")
    start, end = (int(f) for f in np.clip(frames, 0, n))

    if end <= start:
        clock = (" (its synchronized clock -- pass relative=True to count "
                 "from the first sample)") if tv is not None and not relative else ""
        raise ValueError(
            f"time_range ({t0:.6f}, {t1:.6f}) s holds no samples; segment "
            f"{segment_index} runs from {seg_start:.6f} to {seg_end:.6f} s{clock}."
        )

    most = int(np.ceil((t1 - t0) * fs * 1.01)) + 2
    consistent = end - start <= most
    if tv is not None and consistent:
        # Two element reads: the ends of the span must actually lie in range.
        consistent = tv[start] >= t0 and tv[end - 1] < t1
    if not consistent:
        raise ValueError(
            f"time_range ({t0:.6f}, {t1:.6f}) s resolved to frames {start}-{end} "
            f"({(end - start) / fs:.1f} s of data), which cannot be right: the "
            "segment's time vector is not increasing around here. That happens "
            "when concatenated sessions' clocks restart, or a stream is partly "
            "unsynced. Refusing to read it."
        )
    return start, end


def _frame_times(recording, segment_index, start, end):
    tv = _time_vector(recording, segment_index)
    if tv is None:
        fs = recording.get_sampling_frequency()
        seg_start = float(recording.get_start_time(segment_index=segment_index))
        return np.arange(start, end) / fs + seg_start
    return np.array(tv[start:end], dtype="float64")


def _read(recording, segment_index, start, end, channel_ids, return_in_uV,
          max_gb):
    n_channels = (recording.get_num_channels() if channel_ids is None
                  else len(channel_ids))
    itemsize = 4 if return_in_uV else recording.get_dtype().itemsize
    gb = (end - start) * n_channels * itemsize / 1e9
    if gb > max_gb:
        raise ValueError(
            f"{end - start} samples x {n_channels} channels is {gb:.2f} GB, over "
            f"max_gb={max_gb}. Ask for less, or raise max_gb if that is intended."
        )
    return recording.get_traces(
        segment_index=segment_index,
        start_frame=start,
        end_frame=end,
        channel_ids=channel_ids,
        return_in_uV=return_in_uV,
    )


def get_traces(recording, time_range, *, segment_index=None, channel_ids=None,
               return_in_uV: bool = False, relative: bool = False,
               return_times: bool = False, max_gb: float = 1.0):
    """Traces for ``time_range`` seconds only, ``(samples, channels)``.

    ``recording.get_traces`` addressed by time rather than frame, reading
    nothing outside the range. See :func:`time_range_to_frames` for how
    ``time_range`` and ``relative`` are read. Channels come back in the order
    of ``channel_ids``. ``return_times=True`` also returns each sample's time,
    on the same clock as ``time_range``.

    Refuses anything over ``max_gb`` before reading it.
    """
    segment_index = _segment_index(recording, segment_index)
    start, end = time_range_to_frames(recording, time_range, segment_index,
                                      relative=relative)
    traces = _read(recording, segment_index, start, end, channel_ids,
                   return_in_uV, max_gb)
    if not return_times:
        return traces
    times = _frame_times(recording, segment_index, start, end)
    if relative:
        times -= float(recording.get_start_time(segment_index=segment_index))
    return traces, times


def _as_layers(recording):
    if isinstance(recording, dict):
        return dict(recording)
    if isinstance(recording, (list, tuple)):
        return {f"rec{i}": rec for i, rec in enumerate(recording)}
    return {"rec": recording}


def _events_in_segment(events, segment_index):
    if isinstance(events, (list, tuple)) and events and isinstance(
        events[0], np.ndarray
    ):
        events = events[segment_index]
    events = np.asarray(events)
    if events.dtype.names is None:
        return events.astype("float64"), None
    duration = events["duration"] if "duration" in events.dtype.names else None
    return np.asarray(events["time"], dtype="float64"), duration


def plot_traces(recording, time_range=None, *, segment_index=None,
                channel_ids=None, order_channel_by_depth: bool = False,
                mode: str = "auto", return_in_uV: bool = False,
                relative: bool = False, cmap="RdBu_r", clim=None,
                show_channel_ids: bool = False, events=None,
                events_color="gray", events_alpha: float = 0.5, color=None,
                scale: float = 1.0, vspacing_factor: float = 1.5,
                with_colorbar: bool = True, add_legend: bool = True,
                max_gb: float = 1.0, ax=None):
    """``si.plot_traces`` for long recordings: reads ``time_range`` and nothing else.

    Takes spikeinterface's keywords, so an existing call can switch over
    unchanged (matplotlib only). ``recording`` may be one recording or a
    dict/list of them drawn as overlaid layers; frames are resolved once, from
    the first. ``time_range`` defaults to the first second, and is on the
    recording's own clock unless ``relative=True`` (see
    :func:`time_range_to_frames`); the x axis uses the same clock, and so do
    ``events`` -- float times or a structured array with ``time`` and optional
    ``duration``.

    ``mode="auto"`` draws lines up to 64 channels and a heat map above that.
    ``order_channel_by_depth`` sorts the channels by probe position without
    wrapping the recording in another preprocessing step. Returns the axes.
    """
    import matplotlib.pyplot as plt

    layers = _as_layers(recording)
    keys = list(layers)
    rec0 = layers[keys[0]]
    segment_index = _segment_index(rec0, segment_index)
    seg_start = float(rec0.get_start_time(segment_index=segment_index))

    if time_range is None:
        time_range = (0.0, 1.0) if relative else (seg_start, seg_start + 1.0)
    channel_ids = list(rec0.channel_ids if channel_ids is None else channel_ids)
    locations = (rec0.get_channel_locations(channel_ids=channel_ids)
                 if rec0.has_channel_location() else None)
    if order_channel_by_depth and locations is not None:
        order = np.lexsort((locations[:, 0], locations[:, 1]))
        channel_ids = [channel_ids[i] for i in order]
        locations = locations[order]

    start, end = time_range_to_frames(rec0, time_range, segment_index,
                                      relative=relative)
    times = _frame_times(rec0, segment_index, start, end)
    if relative:
        times -= seg_start
    list_traces = [
        scale * _read(layers[k], segment_index, start, end, channel_ids,
                      return_in_uV, max_gb)
        for k in keys
    ]

    n = len(channel_ids)
    if mode == "auto":
        mode = "line" if n <= 64 else "map"
    if mode not in ("line", "map"):
        raise ValueError(f'mode must be "auto", "line" or "map", got {mode!r}.')

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    dt = 1.0 / rec0.get_sampling_frequency()
    t_lo, t_hi = times[0], times[-1] + dt

    if mode == "line":
        vspacing = float(np.max(np.abs(list_traces[0]))) * vspacing_factor or 1.0
        offsets = np.arange(n) * vspacing
        if color is not None:
            colors = [color] + ["k"] * (len(keys) - 1)
        elif len(keys) > 1:
            colors = [f"C{i % 10}" for i in range(len(keys))]
        else:
            colors = ["k"]
        for key, traces, c in zip(keys, list_traces, colors):
            lines = ax.plot(times, traces + offsets, color=c, lw=0.8)
            lines[0].set_label(key)
        if show_channel_ids:
            ax.set_yticks(offsets)
            ax.set_yticklabels([str(c) for c in channel_ids])
        else:
            ax.get_yaxis().set_visible(False)
        ax.set_ylim(-vspacing, vspacing * n)
        if add_legend and len(keys) > 1:
            ax.legend(loc="upper right")
    else:
        if len(keys) != 1:
            raise ValueError('mode="map" draws one recording; pass just one.')
        if isinstance(clim, dict):
            clim = clim[keys[0]]
        y = locations[:, 1] if locations is not None else np.arange(n)
        y_lo, y_hi = float(np.min(y)), float(np.max(y))
        if y_hi == y_lo:
            y_lo, y_hi = 0.0, float(n)
        im = ax.imshow(
            list_traces[0].T, interpolation="nearest", origin="lower",
            aspect="auto", extent=(t_lo, t_hi, y_lo, y_hi), cmap=cmap,
        )
        im.set_clim(*(clim if clim is not None else (-200, 200)))
        if with_colorbar:
            ax.figure.colorbar(im, ax=ax)
        if show_channel_ids:
            row = (y_hi - y_lo) / n
            ax.set_yticks(y_lo + row * (np.arange(n) + 0.5))
            ax.set_yticklabels([str(c) for c in channel_ids])
        else:
            ax.set_yticks([y_lo, y_hi])

    if events is not None:
        event_times, durations = _events_in_segment(events, segment_index)
        inside = (event_times >= t_lo) & (event_times < t_hi)
        for i in np.flatnonzero(inside):
            if durations is not None:
                ax.axvspan(event_times[i], event_times[i] + durations[i],
                           alpha=events_alpha, color=events_color)
            else:
                ax.axvline(event_times[i], alpha=events_alpha,
                           color=events_color)

    ax.set_xlim(t_lo, t_hi)
    ax.set_xlabel("time from start (s)" if relative else "time (s)")
    return ax
