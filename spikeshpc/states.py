"""Automatic WAKE/NREM/REM scoring, after Watson et al. (2016).

Watson, Levenstein, Greene, Gelinas, Buzsáki & Rinzel, "Network Homeostasis
and State Dynamics of Neocortical Sleep", Neuron 90(4):839-852.
https://pmc.ncbi.nlm.nih.gov/articles/PMC4873379/

Three signals are computed from LFP, in 1 s steps over a sliding window
(`window_s`, 5 s by default; the paper uses 10 s):

  broadband   first principal component of the z-scored log spectrogram
              (1-100 Hz, log-spaced), sign-fixed to increase with slow-wave
              power. High during NREM.
  theta       ratio of 5-10 Hz power to 2-16 Hz power. High during REM.
  emg         mean zero-lag correlation between 300-600 Hz filtered signals
              at spatially separated sites. High during waking movement.

Each is thresholded at the trough between its two modes, and states follow
the buzcode rules: NREM where broadband is high; of the rest, WAKE where EMG
is high and REM where EMG is low and theta is high.

THIS IS A REIMPLEMENTATION from the published description, not a port of
buzcode's SleepScoreMaster (https://github.com/buzsakilab/buzcode/tree/master/detectors),
and it has not been validated against hand-scored data. Known differences:

  * buzcode picks single SW and theta channels by searching for the most
    bimodal one; here a handful of channels spread along the probe are
    averaged, or you name them yourself. On a probe spanning several
    structures you should name them -- theta is a hippocampal signal.
  * buzcode filters the EMG band on 1250 Hz LFP, putting 600 Hz at 0.96
    Nyquist. We resample to 2500 Hz first (`emg_rate`).
  * buzcode applies per-state minimum durations and transition rules; here a
    single `min_state_duration_s` is enforced for every state.
  * The paper curates the automatic scoring by hand afterwards. Treat this
    output as a starting point -- every metric is saved alongside the states
    so you can re-threshold without recomputing.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spikeinterface.full as si
from scipy.signal import butter, sosfiltfilt

from .config import STATES_DIRNAME
from .io import channel_positions, read_recording
from .optitrack.io import read_rigid_body_track

# buzcode's SleepState convention.
STATE_CODES = {"WAKE": 1, "NREM": 3, "REM": 5}
CODE_NAMES = {v: k for k, v in STATE_CODES.items()}


# ── channel selection ────────────────────────────────────────────────────
def pick_channels(rec, n, explicit=None, exclude=()):
    """`n` channels spread evenly along the probe's long axis.

    Averaging a few sites is more robust than buzcode's single-channel pick
    while staying cheap: only these channels are ever read from disk.
    """
    excluded = {str(c) for c in exclude}
    available = [c for c in rec.channel_ids if str(c) not in excluded]
    if not available:
        raise ValueError("No channels left after excluding bad channels.")

    if explicit:
        wanted = {str(c) for c in explicit}
        chosen = [c for c in rec.channel_ids if str(c) in wanted]
        missing = wanted - {str(c) for c in chosen}
        if missing:
            raise ValueError(f"Channels not in this recording: {sorted(missing)}")
        return chosen

    depth = dict(zip(map(str, rec.channel_ids), channel_positions(rec)[:, 1]))
    ordered = sorted(available, key=lambda c: depth[str(c)])
    if n >= len(ordered):
        return ordered
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in np.unique(idx)]


def _mean_trace(rec, channel_ids, chunk_s=120.0):
    """Mean trace (uV) across `channel_ids`, accumulated chunk by chunk."""
    sub = rec.select_channels(channel_ids)
    n = sub.get_num_frames()
    fs = sub.get_sampling_frequency()
    out = np.empty(n, dtype=np.float32)
    step = max(int(chunk_s * fs), 1)
    for start in range(0, n, step):
        stop = min(start + step, n)
        chunk = sub.get_traces(start_frame=start, end_frame=stop, return_in_uV=True)
        out[start:stop] = np.asarray(chunk, dtype=np.float32).mean(axis=1)
    return out


# ── spectral metrics ─────────────────────────────────────────────────────
def log_spectrogram(sig, fs, window_s, step_s, freq_range, n_freqs):
    """Power at log-spaced frequencies, on a sliding window.

    MATLAB's spectrogram() evaluates arbitrary frequency vectors directly; the
    equivalent here is a dense rFFT interpolated onto the log-spaced grid.

    Returns (times, freqs, spec) with spec shaped (n_freqs, n_windows).
    """
    nwin = round(window_s * fs)
    nstep = round(step_s * fs)
    if len(sig) < nwin:
        raise ValueError(
            f"Recording is {len(sig) / fs:.1f} s, shorter than the "
            f"{window_s} s spectrogram window."
        )
    n_windows = 1 + (len(sig) - nwin) // nstep

    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), int(n_freqs))
    taper = np.hanning(nwin).astype(np.float32)
    fft_freqs = np.fft.rfftfreq(nwin, 1.0 / fs)
    spec = np.empty((len(freqs), n_windows), dtype=np.float32)

    windows = np.lib.stride_tricks.sliding_window_view(sig, nwin)
    block = 256  # bounds the temporary to block x nwin floats
    for start in range(0, n_windows, block):
        stop = min(start + block, n_windows)
        seg = windows[np.arange(start, stop) * nstep] * taper
        power = np.abs(np.fft.rfft(seg, axis=-1)) ** 2
        for j in range(power.shape[0]):
            spec[:, start + j] = np.interp(freqs, fft_freqs, power[j])

    times = (np.arange(n_windows) * nstep + nwin / 2.0) / fs
    return times, freqs, spec


def broadband_slow_wave(spec, freqs, slow_wave_max_hz=32.0):
    """PC1 of the z-scored log spectrogram, signed so NREM is high."""
    from sklearn.decomposition import PCA

    log_spec = np.log10(spec + np.finfo(np.float32).tiny)
    z = (log_spec - log_spec.mean(axis=1, keepdims=True)) / (
        log_spec.std(axis=1, keepdims=True) + 1e-12
    )
    pc1 = PCA(n_components=1).fit_transform(z.T).ravel()

    # The sign of a principal component is arbitrary. The paper anchors it to
    # low-frequency power, which is what rises in NREM.
    low = z[freqs <= slow_wave_max_hz].mean(axis=0)
    if np.corrcoef(pc1, low)[0, 1] < 0:
        pc1 = -pc1
    return ((pc1 - pc1.mean()) / (pc1.std() + 1e-12)).astype(np.float32)


def theta_ratio(spec, freqs, theta_band, ref_band):
    """Power in `theta_band` over power in `ref_band`."""
    num = spec[(freqs >= theta_band[0]) & (freqs <= theta_band[1])].sum(axis=0)
    den = spec[(freqs >= ref_band[0]) & (freqs <= ref_band[1])].sum(axis=0)
    return (num / np.maximum(den, np.finfo(np.float32).tiny)).astype(np.float32)


def emg_from_lfp(
    rec,
    channel_ids,
    band,
    times,
    window_s,
    min_distance_um=100.0,
    chunk_windows=256,
):
    """Mean zero-lag correlation between high-frequency signals at distant sites.

    Volume conduction correlates neighbouring contacts whatever the animal is
    doing, so only pairs at least `min_distance_um` apart are averaged.
    Evaluated on `window_s` windows centred on `times`.
    """
    sub = rec.select_channels(channel_ids)
    fs = sub.get_sampling_frequency()
    nyquist = fs / 2.0
    if band[1] >= nyquist:
        raise ValueError(
            f"EMG band {band} reaches the Nyquist frequency of {nyquist} Hz. "
            "Raise state_scoring.emg_rate."
        )

    pos = channel_positions(sub)
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    iu = np.triu_indices(len(channel_ids), k=1)
    far = dist[iu] >= min_distance_um
    if not far.any():
        raise ValueError(
            f"No EMG channel pairs at least {min_distance_um} um apart. "
            "Lower state_scoring.emg_min_distance_um or pick more channels."
        )

    sos = butter(
        4, [band[0] / nyquist, band[1] / nyquist], btype="bandpass", output="sos"
    )
    nwin = round(window_s * fs)
    half = nwin // 2
    n_total = sub.get_num_frames()
    centres = np.clip((times * fs).round().astype(int), half, n_total - (nwin - half))

    emg = np.empty(len(times), dtype=np.float32)
    for start in range(0, len(centres), chunk_windows):
        stop = min(start + chunk_windows, len(centres))
        lo = centres[start] - half
        hi = centres[stop - 1] + (nwin - half)
        # Filter with padding either side so chunk edges are not transients.
        pad = min(int(fs), lo, n_total - hi)
        raw = sub.get_traces(
            start_frame=lo - pad, end_frame=hi + pad, return_in_uV=True
        )
        filtered = sosfiltfilt(sos, np.asarray(raw, dtype=np.float64), axis=0)
        filtered = filtered[pad : filtered.shape[0] - pad] if pad else filtered

        for j in range(start, stop):
            a = centres[j] - half - lo
            seg = filtered[a : a + nwin]
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(seg, rowvar=False)
            emg[j] = np.nanmean(corr[iu][far])
    return emg


# ── thresholding and state assignment ────────────────────────────────────
def bimodal_threshold(x, n_grid=1000, seed=0):
    """Split a bimodal distribution at the trough between its two modes.

    A two-component Gaussian mixture stands in for buzcode's histogram dip
    search; the threshold is where the components' posteriors cross. Falls
    back to the median when the fit is degenerate (i.e. the distribution is
    not actually bimodal), which is the honest answer for a recording that
    never left one state.
    """
    from sklearn.mixture import GaussianMixture

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    finite = x[np.isfinite(x[:, 0])]
    if len(finite) < 10:
        return float(np.median(finite)) if len(finite) else 0.0

    gm = GaussianMixture(n_components=2, random_state=seed).fit(finite)
    lo, hi = np.sort(gm.means_.ravel())
    if not np.isfinite([lo, hi]).all() or np.isclose(lo, hi):
        return float(np.median(finite))

    grid = np.linspace(lo, hi, n_grid).reshape(-1, 1)
    order = np.argsort(gm.means_.ravel())
    post = gm.predict_proba(grid)[:, order]
    crossings = np.flatnonzero(np.diff(np.sign(post[:, 0] - post[:, 1])))
    if crossings.size == 0:
        return float(np.median(finite))
    return float(grid[crossings[0], 0])


def enforce_min_duration(codes, step_s, min_duration_s):
    """Absorb runs shorter than `min_duration_s` into the longer neighbour.

    Repeated until stable, shortest run first, so a brief flicker cannot
    survive by sitting between two other brief flickers.
    """
    codes = np.asarray(codes).copy()
    min_len = round(min_duration_s / step_s)
    if min_len <= 1:
        return codes

    while True:
        edges = np.flatnonzero(np.diff(codes)) + 1
        starts = np.r_[0, edges]
        stops = np.r_[edges, len(codes)]
        lengths = stops - starts
        short = np.flatnonzero(lengths < min_len)
        if short.size == 0 or len(starts) == 1:
            return codes

        i = short[np.argmin(lengths[short])]
        if i == 0:
            codes[starts[i] : stops[i]] = codes[starts[i + 1]]
        elif i == len(starts) - 1:
            codes[starts[i] : stops[i]] = codes[starts[i - 1]]
        else:
            before, after = i - 1, i + 1
            winner = before if lengths[before] >= lengths[after] else after
            codes[starts[i] : stops[i]] = codes[starts[winner]]


def classify_states(broadband, theta, emg, thresholds):
    """buzcode's decision rules over the three thresholded signals."""
    nrem = broadband > thresholds["broadband"]
    quiet = emg <= thresholds["emg"]
    rem = (~nrem) & quiet & (theta > thresholds["theta"])

    codes = np.full(len(broadband), STATE_CODES["WAKE"], dtype=np.int16)
    codes[nrem] = STATE_CODES["NREM"]
    codes[rem] = STATE_CODES["REM"]
    return codes


def intervals_from_states(codes, times, step_s):
    """Contiguous runs as {'WAKE': [[start, stop], ...], ...} in seconds."""
    intervals = {name: [] for name in STATE_CODES}
    if len(codes) == 0:
        return intervals
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(codes)]
    for a, b in zip(starts, stops):
        name = CODE_NAMES.get(int(codes[a]))
        if name is None:
            continue
        # Bin centres bound the run; extend by half a step to its real edges.
        intervals[name].append(
            [float(times[a] - step_s / 2), float(times[b - 1] + step_s / 2)]
        )
    return intervals


# ── movement ─────────────────────────────────────────────────────────────
def held_frames(position) -> np.ndarray:
    """Frames whose position is bit-identical to the one before.

    When Motive loses the rigid body it does not write a gap -- it repeats the
    last known position, frame after frame, until it reacquires. Those frames
    are missing data wearing the costume of perfect stillness, and they are
    exactly the samples a movement threshold must not see: a run of them drags
    a bin's mean speed to zero, and the frame that finally moves carries the
    whole accumulated displacement in one frame interval, which reads as a
    violent burst.
    """
    position = np.asarray(position, dtype=float)
    held = np.zeros(len(position), dtype=bool)
    if len(position) > 1:
        held[1:] = np.all(np.diff(position, axis=0) == 0, axis=1)
    return held


def binned_speed(
    frame_times,
    position,
    times,
    step_s,
    max_gap_s=0.5,
    interpolate_gaps_s=2.0,
    verbose=True,
):
    """Mean speed per state bin, from tracked position sampled at frame_times.

    Frames that are non-finite, or held over from a dropout
    (:func:`held_frames`), are treated as missing. Speed is then taken between
    consecutive surviving frames over the real elapsed interval -- but only
    where that interval is at most `max_gap_s`, since a displacement measured
    across a six-second dropout is not a speed at any instant within it.

    Bins left with no usable frames come back NaN. Runs of NaN shorter than
    `interpolate_gaps_s` are filled by interpolating between the neighbouring
    bins; longer ones are left NaN, which the veto skips rather than guessing
    at. Set `interpolate_gaps_s=0` to fill nothing.
    """
    frame_times = np.asarray(frame_times, dtype=float)
    position = np.asarray(position, dtype=float)
    if len(frame_times) != len(position):
        raise ValueError(
            f"{len(frame_times)} frame times but {len(position)} position "
            "samples; align them before calling (see "
            "optitrack.align_frames_to_shutter_events)."
        )

    finite = np.isfinite(position).all(axis=1)
    held = held_frames(position)
    usable = finite & ~held

    idx = np.flatnonzero(usable)
    speed = np.full(len(position), np.nan)
    if idx.size > 1:
        step = np.linalg.norm(np.diff(position[idx], axis=0), axis=1)
        elapsed = np.diff(frame_times[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(elapsed > 0, step / elapsed, np.nan)
        # a step bridging a long gap is an average over the gap, not a speed;
        # leaving it out is what stops the post-dropout burst
        values[elapsed > max_gap_s] = np.nan
        speed[idx[1:]] = values

    edges = np.r_[times - step_s / 2, times[-1] + step_s / 2]
    which = np.digitize(frame_times, edges) - 1
    ok = (which >= 0) & (which < len(times)) & np.isfinite(speed)

    totals = np.bincount(which[ok], weights=speed[ok], minlength=len(times))
    counts = np.bincount(which[ok], minlength=len(times))
    out = np.full(len(times), np.nan)
    np.divide(totals, counts, out=out, where=counts > 0)

    dropped = int(held.sum())
    empty = int(np.isnan(out).sum())
    filled = interpolate_gaps(out, step_s, interpolate_gaps_s)
    if verbose and (dropped or empty):
        print(
            f"      tracking: {dropped} held frame(s) "
            f"({dropped / max(len(position), 1):.2%}) treated as dropouts; "
            f"{empty} bin(s) left empty, {filled} interpolated"
        )
    return out


def interpolate_gaps(values, step_s, max_gap_s=2.0) -> int:
    """Fill short runs of NaN in place by interpolating across them.

    Only runs shorter than `max_gap_s`, and only those with real values on
    both sides -- a gap at either end has nothing to interpolate between, and
    a long one would be invention rather than repair. Returns how many bins
    were filled.
    """
    if max_gap_s <= 0:
        return 0
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    if not missing.any() or missing.all():
        return 0

    max_bins = round(max_gap_s / step_s)
    edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(int), 0]))
    starts, stops = edges[::2], edges[1::2]

    index = np.arange(len(values))
    known = ~missing
    filled = 0
    for a, b in zip(starts, stops):
        if b - a > max_bins or a == 0 or b == len(values):
            continue  # too long, or open at one end
        values[a:b] = np.interp(index[a:b], index[known], values[known])
        filled += b - a
    return filled


def movement_threshold(speed, floor=1e-3, seed=0):
    """Split immobility from locomotion on log10 speed.

    Log scale on purpose. Movement never reaches zero (breathing, postural
    sway and tracking jitter create floor) so the question is which of two
    (non-zero) modes each bin is in. Two modes are roughly log-normal and
    well separated; the same bimodal split used for the LFP metrics finds
    the trough between them.
    """
    log_speed = np.log10(np.maximum(np.asarray(speed, dtype=float), floor))
    finite = log_speed[np.isfinite(log_speed)]
    if finite.size < 10:
        return None
    return bimodal_threshold(finite, seed=seed)


def apply_movement_veto(
    codes,
    speed,
    threshold=None,
    step_s=1.0,
    min_duration_s=6.0,
    veto=("NREM", "REM"),
    floor=1e-3,
    verbose=True,
):
    """Reassign to WAKE any bin scored asleep while the animal was moving
    (based on optitrack data)

    Deliberately asymmetric. Gross movement proves the animal is awake, but
    stillness proves nothing. Used as a veto, the tracker adds information
    the LFP cannot: it is the only signal here that can catch running, which
    drives hippocampal theta and is otherwise indistinguishable from REM theta.

    Bins with no tracking data (NaN) are left untouched. Minimum-duration
    smoothing is re-applied afterwards, since vetoing punches holes in
    otherwise good bouts.

    Returns (codes, info).
    """
    codes = np.asarray(codes).copy()
    speed = np.asarray(speed, dtype=float)
    if threshold is None:
        threshold = movement_threshold(speed, floor=floor)
    if threshold is None:
        return codes, {"applied": False, "reason": "no usable movement data"}

    log_speed = np.log10(np.maximum(speed, floor))
    moving = np.isfinite(log_speed) & (log_speed > threshold)
    targets = np.isin(codes, [STATE_CODES[name] for name in veto])

    before = codes.copy()
    codes[moving & targets] = STATE_CODES["WAKE"]
    codes = enforce_min_duration(codes, step_s, min_duration_s)

    changed = codes != before
    info_before = before
    info = {
        "applied": True,
        "threshold_log10": float(threshold),
        "threshold_speed": float(10**threshold),
        "vetoed_states": list(veto),
        "coverage": float(np.isfinite(speed).mean()),
        "fraction_moving": float(moving.mean()),
        "n_reassigned": int(changed.sum()),
        "fraction_reassigned": float(changed.mean()),
    }
    info["codes_before"] = info_before
    if verbose:
        print(
            f"      movement veto: threshold {10**threshold:.1f} units/s, "
            f"{moving.mean():.1%} of bins moving, "
            f"{changed.sum()} bins reassigned ({changed.mean():.1%})"
        )
    return codes, info


def load_movement(
    config, session, times, step_s, phys_path=None, output_dir=None, phys_type=None
):
    """Per-bin speed for `session`, or None when tracking is unavailable.

    `optitrack_csv` and `frame_times` are format strings taking `{session}`,
    which is how the same config covers every session in a run.

    `frame_times` may be left unset: the shutter TTL is then extracted from
    the recording's own ADC stream and cached, so a session can be scored
    straight off the rig without a notebook step first.
    """
    csv_template = config.get("optitrack_csv")
    if not csv_template:
        print("      movement: no optitrack_csv configured, skipping")
        return None

    csv_path = Path(str(csv_template).format(session=session))
    if not csv_path.exists():
        print(f"      movement: no OptiTrack CSV for {session}, skipping")
        return None

    times_template = config.get("frame_times")
    times_path = (
        Path(str(times_template).format(session=session)) if times_template else None
    )
    if times_path is None or not times_path.exists():
        if phys_path is None or output_dir is None:
            print(
                "      movement: no frame_times and nothing to derive them "
                "from, skipping"
            )
            return None
        from .shutter import derive_shutter_times

        times_path = derive_shutter_times(
            phys_path, output_dir, session, config, phys_type, csv_path
        )
        if times_path is None:
            return None

    frame_times = np.load(times_path)
    try:
        track = read_rigid_body_track(csv_path, config.get("rigid_body"))
    except ValueError as e:
        # A malformed or ambiguous export is worth reporting, but not worth
        # losing a sorting job over an optional signal.
        print(f"      movement: {e} -- skipping the veto")
        return None

    position = track.position
    if len(frame_times) != len(position):
        print(
            f"      movement: {len(frame_times)} frame times but "
            f"{len(position)} tracked frames for {session}; skipping rather "
            "than guessing the alignment"
        )
        return None

    print(
        f"      movement: {track.name!r}, {len(position)} frames from "
        f"{csv_path.name} @ {track.frame_rate:g} Hz"
    )
    return binned_speed(frame_times, position, times, step_s)


# ── top level ────────────────────────────────────────────────────────────
def _smooth(x, step_s, smooth_s):
    width = max(round(smooth_s / step_s), 1)
    if width <= 1:
        return x
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def score_recording(rec_lfp, rec_emg, config, exclude_channels=(), speed=None):
    """Compute the three signals and the state sequence for one recording.

    `speed` is an optional per-bin movement trace on the same time base; see
    :func:`apply_movement_veto` for how it is used.
    """
    step_s = float(config["step_s"])

    sw_ids = pick_channels(
        rec_lfp, config["n_sw_channels"], config["sw_channels"], exclude_channels
    )
    theta_ids = pick_channels(
        rec_lfp, config["n_theta_channels"], config["theta_channels"], exclude_channels
    )
    emg_ids = pick_channels(
        rec_emg, config["n_emg_channels"], config["emg_channels"], exclude_channels
    )
    print(f"      slow-wave channels: {[str(c) for c in sw_ids]}")
    print(f"      theta channels    : {[str(c) for c in theta_ids]}")
    print(f"      emg channels      : {[str(c) for c in emg_ids]}")

    fs = rec_lfp.get_sampling_frequency()
    sw_sig = _mean_trace(rec_lfp, sw_ids)
    times, freqs, spec = log_spectrogram(
        sw_sig,
        fs,
        config["window_s"],
        step_s,
        config["freq_range"],
        config["n_freqs"],
    )
    broadband = broadband_slow_wave(spec, freqs, config["slow_wave_max_hz"])

    if list(map(str, theta_ids)) == list(map(str, sw_ids)):
        theta_spec, theta_freqs = spec, freqs
    else:
        _, theta_freqs, theta_spec = log_spectrogram(
            _mean_trace(rec_lfp, theta_ids),
            fs,
            config["window_s"],
            step_s,
            config["freq_range"],
            config["n_freqs"],
        )
    theta = theta_ratio(
        theta_spec, theta_freqs, config["theta_band"], config["theta_ref_band"]
    )

    emg = emg_from_lfp(
        rec_emg,
        emg_ids,
        config["emg_band"],
        times,
        window_s=config["window_s"],
        min_distance_um=config["emg_min_distance_um"],
    )

    smooth_s = config["smooth_s"]
    broadband_s = _smooth(broadband, step_s, smooth_s)
    theta_s = _smooth(theta, step_s, smooth_s)
    emg_s = _smooth(emg, step_s, smooth_s)

    thresholds = {
        "broadband": bimodal_threshold(broadband_s),
        "theta": bimodal_threshold(theta_s),
        "emg": bimodal_threshold(emg_s),
    }
    codes = classify_states(broadband_s, theta_s, emg_s, thresholds)
    codes = enforce_min_duration(codes, step_s, config["min_state_duration_s"])

    movement_info = {"applied": False}
    if speed is not None:
        mv = config.get("movement") or {}
        codes, movement_info = apply_movement_veto(
            codes,
            speed,
            threshold=mv.get("threshold"),
            step_s=step_s,
            min_duration_s=config["min_state_duration_s"],
            veto=tuple(mv.get("veto", ("NREM", "REM"))),
        )

    fractions = {
        name: float(np.mean(codes == code)) for name, code in STATE_CODES.items()
    }
    print(
        "      thresholds: " + ", ".join(f"{k}={v:.3f}" for k, v in thresholds.items())
    )
    print(
        "      state fractions: "
        + ", ".join(f"{k}={v:.1%}" for k, v in fractions.items())
    )

    return {
        "times": times,
        "broadband": broadband_s,
        "theta": theta_s,
        "emg": emg_s,
        "speed": speed,
        "codes": codes,
        "codes_before_veto": movement_info.pop("codes_before", None),
        "thresholds": thresholds,
        "movement": movement_info,
        "fractions": fractions,
        "intervals": intervals_from_states(codes, times, step_s),
        "channels": {
            "slow_wave": [str(c) for c in sw_ids],
            "theta": [str(c) for c in theta_ids],
            "emg": [str(c) for c in emg_ids],
        },
        "lfp_rate": float(fs),
        "emg_rate": float(rec_emg.get_sampling_frequency()),
    }


def _resample_to(rec, rate):
    """Resample only if we are actually going down; never upsample."""
    current = rec.get_sampling_frequency()
    if abs(current - rate) < 1e-6:
        return rec
    if current < rate:
        print(
            f"      source is {current:.0f} Hz, below the requested {rate:.0f} Hz; "
            "using it as is"
        )
        return rec
    return si.resample(rec, round(rate))


def score_session(
    phys_path,
    output_dir: Path,
    config: dict,
    phys_type=None,
    stream_name=None,
    exclude_channels=(),
):
    """Score one recording and write states/<session>_states.json + _metrics.npz.

    Uses the acquisition system's LF band when it has one, otherwise resamples
    the AP band down. Returns the result dict, with 'session' and 'duration_s'
    added.
    """
    from .channels import drop_sync_channels

    phys_path = Path(phys_path)
    session = phys_path.stem or phys_path.name

    rec_lf, _, lf_stream = read_recording(phys_path, phys_type, None, band="lf")
    if rec_lf is not None:
        print(
            f"      LFP source: {lf_stream} @ {rec_lf.get_sampling_frequency():.0f} Hz"
        )
        source = rec_lf
    else:
        source, _, ap_stream = read_recording(phys_path, phys_type, stream_name, "ap")
        print(
            f"      no LF stream; deriving LFP from {ap_stream} @ "
            f"{source.get_sampling_frequency():.0f} Hz"
        )
    source, _ = drop_sync_channels(source)

    rec_lfp = _resample_to(source, config["lfp_rate"])
    rec_emg = _resample_to(source, config["emg_rate"])

    # The bin grid is fixed by the spectrogram, so derive it the same way here
    # rather than duplicating the arithmetic inside score_recording.
    speed = None
    movement_cfg = config.get("movement") or {}
    if movement_cfg.get("enabled"):
        n = rec_lfp.get_num_frames()
        fs = rec_lfp.get_sampling_frequency()
        nwin = round(config["window_s"] * fs)
        nstep = round(config["step_s"] * fs)
        n_windows = 1 + (n - nwin) // nstep
        grid = (np.arange(n_windows) * nstep + nwin / 2.0) / fs
        speed = load_movement(
            movement_cfg,
            session,
            grid,
            config["step_s"],
            phys_path=phys_path,
            output_dir=output_dir,
            phys_type=phys_type,
        )

    result = score_recording(rec_lfp, rec_emg, config, exclude_channels, speed=speed)
    result["session"] = session
    result["phys_path"] = str(phys_path.resolve())
    result["duration_s"] = float(
        source.get_num_frames() / source.get_sampling_frequency()
    )

    states_dir = output_dir / STATES_DIRNAME
    states_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "times": result["times"],
        "broadband": result["broadband"],
        "theta": result["theta"],
        "emg": result["emg"],
        "codes": result["codes"],
    }
    if result.get("speed") is not None:
        arrays["speed"] = result["speed"]
    # what the LFP alone called, before movement overruled it -- the only way
    # to review which calls the veto actually changed
    if result.get("codes_before_veto") is not None:
        arrays["codes_before_veto"] = result["codes_before_veto"]
    np.savez_compressed(states_dir / f"{session}_metrics.npz", **arrays)
    summary = {
        k: v
        for k, v in result.items()
        if k
        not in (
            "times",
            "broadband",
            "theta",
            "emg",
            "codes",
            "speed",
            "codes_before_veto",
        )
    }
    summary["state_codes"] = STATE_CODES
    summary["step_s"] = config["step_s"]
    with open(states_dir / f"{session}_states.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      -> {states_dir / (session + '_states.json')}")
    return result


def merge_to_concatenated_time(results, info, output_dir: Path):
    """Shift each session's intervals into the concatenated recording's clock.

    State scoring runs per session, but the sorting is concatenated, so this
    is the file you actually join spikes against.
    """
    from .config import STATES_CONCAT_NAME

    fs = info["sampling_frequency"]
    offsets = info["sample_offsets"]
    merged = {name: [] for name in STATE_CODES}
    for result, offset in zip(results, offsets):
        shift = offset / fs
        for name, spans in result["intervals"].items():
            merged[name].extend([[a + shift, b + shift] for a, b in spans])

    record = {
        "sampling_frequency": fs,
        "sample_offsets": offsets,
        "sessions": [r["session"] for r in results],
        "state_codes": STATE_CODES,
        "intervals": {k: sorted(v) for k, v in merged.items()},
    }
    path = output_dir / STATES_DIRNAME / STATES_CONCAT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"    concatenated-time intervals -> {path}")
    return record


# ── using the scoring downstream ─────────────────────────────────────────
@dataclass
class StateEpoch:
    """One contiguous run of a single state."""

    state: str
    start: float  # seconds, leading edge of the first bin
    stop: float  # seconds, trailing edge of the last bin
    first_bin: int
    last_bin: int  # inclusive

    @property
    def duration(self) -> float:
        return self.stop - self.start

    def __repr__(self):
        return (
            f"StateEpoch({self.state}, {self.start:.1f}-{self.stop:.1f}s, "
            f"{self.duration:.1f}s)"
        )


def state_epochs(codes, times, state_codes=None, step_s=1.0, states=None):
    """Contiguous runs of one state, as StateEpoch objects in time order."""
    codes = np.asarray(codes)
    times = np.asarray(times, dtype=float)
    names = {v: k for k, v in (state_codes or STATE_CODES).items()}
    wanted = (
        None
        if states is None
        else {s.upper() for s in ([states] if isinstance(states, str) else states)}
    )

    if codes.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(codes)]

    epochs = []
    for a, b in zip(starts, stops):
        name = names.get(int(codes[a]), "?")
        if wanted is not None and name not in wanted:
            continue
        epochs.append(
            StateEpoch(
                state=name,
                start=float(times[a] - step_s / 2),
                stop=float(times[b - 1] + step_s / 2),
                first_bin=int(a),
                last_bin=int(b - 1),
            )
        )
    return epochs


def times_in_states(query_times, intervals, states=("WAKE",)) -> np.ndarray:
    """Boolean mask: which of `query_times` fall inside any of `states`.

    `intervals` is the {state: [[start, stop], ...]} mapping from the scoring
    (either StateScoring.intervals or states_concatenated.json). Times are
    expected on the same clock as the intervals.
    """
    query_times = np.asarray(query_times, dtype=float)
    wanted = [states] if isinstance(states, str) else list(states)

    spans = []
    for name in wanted:
        spans.extend(intervals.get(name.upper(), []))
    if not spans:
        return np.zeros(len(query_times), dtype=bool)

    spans = np.asarray(sorted(spans), dtype=float)
    # searchsorted against the flattened edges: an odd insertion point means
    # the time landed inside a span
    starts, stops = spans[:, 0], spans[:, 1]
    idx = np.searchsorted(starts, query_times, side="right") - 1
    inside = np.zeros(len(query_times), dtype=bool)
    valid = idx >= 0
    inside[valid] = query_times[valid] < stops[idx[valid]]
    return inside


def intervals_between_frames(frame_mask) -> np.ndarray:
    """Which inter-frame intervals lie wholly inside the kept frames.

    Analyses that work per inter-frame interval -- head-direction tuning, for
    instance -- cannot simply be handed a filtered `frame_times` array. Doing
    that silently splices the gap where a sleep epoch was removed into one
    enormous "interval" that absorbs every spike fired during it and credits
    them to a single heading. The fix is to drop those intervals rather than
    the frames: interval i survives only when frames i and i+1 both do.
    """
    frame_mask = np.asarray(frame_mask, dtype=bool)
    if frame_mask.size < 2:
        return np.zeros(max(frame_mask.size - 1, 0), dtype=bool)
    return frame_mask[:-1] & frame_mask[1:]


def frames_in_states(frame_times, intervals, states=("WAKE",)):
    """(frame_mask, interval_mask) for frames sampled at `frame_times`.

    Pass `interval_mask` to the tuning functions; see
    :func:`intervals_between_frames` for why the frame mask alone is not
    enough.
    """
    frame_mask = times_in_states(frame_times, intervals, states)
    return frame_mask, intervals_between_frames(frame_mask)


def slice_recording_to_states(
    recording, intervals, states=("WAKE",), min_duration_s=0.0
):
    """The recording restricted to `states`, as one concatenated segment.

    Sample indices no longer correspond to the original recording's clock, so
    this is for analyses that only need the samples themselves. Anything that
    has to line up with spike times or tracking should use
    :func:`times_in_states` and keep the original clock.
    """
    import spikeinterface.full as si

    fs = recording.get_sampling_frequency()
    n = recording.get_num_frames()
    wanted = [states] if isinstance(states, str) else list(states)

    spans = []
    for name in wanted:
        spans.extend(intervals.get(name.upper(), []))
    if not spans:
        raise ValueError(f"No {wanted} intervals to slice to.")

    pieces = []
    for start, stop in sorted(spans):
        if stop - start < min_duration_s:
            continue
        a = max(round(start * fs), 0)
        b = min(round(stop * fs), n)
        if b > a:
            pieces.append(recording.frame_slice(a, b))
    if not pieces:
        raise ValueError(
            f"No {wanted} intervals survived the {min_duration_s}s minimum."
        )

    kept = sum(p.get_num_frames() for p in pieces)
    print(
        f"    {len(pieces)} {'/'.join(wanted)} epochs, "
        f"{kept / fs:.1f}s of {n / fs:.1f}s ({kept / n:.1%})"
    )
    return si.concatenate_recordings(pieces) if len(pieces) > 1 else pieces[0]


def rescore_movement(
    states_path,
    session=None,
    optitrack_csv=None,
    frame_times=None,
    rigid_body=None,
    min_duration_s=None,
    veto=None,
    threshold=None,
    write=True,
):
    """Re-apply the movement veto to an already-scored session, in place.

    For when the tracking was right but its timing was not. The veto is the
    only part of the scoring that reads the camera: the broadband, theta and
    EMG traces come from LFP alone, and so does ``codes_before_veto``. So a
    session whose shutter timestamps were on the wrong clock does not need its
    spectrograms recomputing -- it needs the veto applied again to the states
    the LFP already decided, with the movement trace in the right place. That
    is minutes of work against hours, and it touches no signal that was not
    already wrong.

    Requires ``codes_before_veto`` in the saved metrics, which is what makes
    the veto undoable. Sessions scored before it was recorded, or with the veto
    disabled, have to be scored again properly -- their `codes` cannot be
    separated back into a decision and an override.

    ``min_duration_s``, ``veto`` and ``threshold`` default to whatever the
    original run used, read back from the saved summary, so the only thing that
    changes is the movement trace. Returns the updated
    :class:`~spikeshpc.io.StateScoring`.
    """
    import shutil

    from .io import load_states

    scoring = load_states(states_path, session)
    if scoring.codes_before_veto is None:
        raise ValueError(
            f"{scoring.session!r} has no 'codes_before_veto' saved, so the "
            "movement veto cannot be undone and re-applied -- the states it "
            "produced are not separable from the ones the LFP decided. Re-run "
            "score_session for this recording instead."
        )

    before = np.asarray(scoring.codes).copy()
    attach_movement(scoring, optitrack_csv, frame_times, rigid_body)

    previous = dict(scoring.metadata.get("movement") or {})
    if min_duration_s is None:
        min_duration_s = scoring.metadata.get("min_state_duration_s", 6.0)
    if veto is None:
        veto = tuple(previous.get("vetoed_states") or ("NREM", "REM"))

    codes, info = apply_movement_veto(
        scoring.codes_before_veto,
        scoring.speed,
        threshold=threshold,
        step_s=scoring.step_s,
        min_duration_s=min_duration_s,
        veto=veto,
    )
    info.pop("codes_before", None)
    scoring.codes = codes
    scoring.intervals = intervals_from_states(codes, scoring.times, scoring.step_s)
    scoring.fractions = {
        name: float(np.mean(codes == code)) for name, code in STATE_CODES.items()
    }
    scoring.metadata["movement"] = info

    moved = int(np.count_nonzero(codes != before))
    print(
        f"    re-vetoed {scoring.session!r}: {info['n_reassigned']} bins "
        f"reassigned (was {previous.get('n_reassigned', '?')}), "
        f"{moved} bins differ from the scoring on disk "
        f"({moved / len(codes):.1%})"
    )
    print(
        "    fractions now "
        + ", ".join(f"{n} {scoring.fractions[n]:.1%}" for n in ("WAKE", "NREM", "REM"))
    )

    if not write:
        return scoring
    if scoring.source is None:
        raise ValueError("This scoring has no source path; pass write=False.")

    json_path = Path(scoring.source)
    npz_path = json_path.with_name(f"{scoring.session}_metrics.npz")
    for original in (json_path, npz_path):
        backup = original.with_suffix(".orig" + original.suffix)
        if original.exists() and not backup.exists():
            shutil.copy2(original, backup)
            print(f"    kept the original scoring at {backup.name}")

    summary = dict(scoring.metadata)
    summary["intervals"] = scoring.intervals
    summary["fractions"] = scoring.fractions
    summary["state_codes"] = scoring.state_codes or STATE_CODES
    summary["step_s"] = scoring.step_s
    summary["rescored_movement"] = True
    json_path.write_text(json.dumps(summary, indent=2))

    arrays = {
        "times": scoring.times,
        "codes": scoring.codes,
        "codes_before_veto": scoring.codes_before_veto,
        "speed": scoring.speed,
    }
    for name in ("broadband", "theta", "emg"):
        if getattr(scoring, name) is not None:
            arrays[name] = getattr(scoring, name)
    np.savez_compressed(npz_path, **arrays)
    print(f"    wrote {json_path.name} and {npz_path.name}")
    return scoring


def attach_movement(scoring, optitrack_csv=None, frame_times=None, rigid_body=None):
    """Compute per-bin speed from the tracking files and attach it to `scoring`.

    The movement trace is only saved with the scoring when the veto ran, so a
    session scored without it -- or before it existed -- has no movement panel
    to look at. This recomputes the trace from the OptiTrack export so it can
    always be plotted alongside the LFP signals, whether or not it was used to
    decide anything.

    `frame_times` may be an array or a path; left None it is looked for in the
    states directory (where the pipeline caches it) and then next to the CSV.
    Returns `scoring`, modified in place.
    """
    from .optitrack.io import read_rigid_body_track

    if optitrack_csv is None:
        raise ValueError("optitrack_csv is required to compute movement.")
    csv_path = Path(optitrack_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if frame_times is None:
        candidates = []
        if getattr(scoring, "source", None) is not None:
            candidates.append(
                Path(scoring.source).parent
                / f"{scoring.session}_shutter_close_times.npy"
            )
        candidates.append(csv_path.parent / "optitrack_shutter_close_times.npy")
        candidates.append(
            csv_path.parent / f"{scoring.session}_shutter_close_times.npy"
        )
        for candidate in candidates:
            if candidate.exists():
                frame_times = candidate
                break
        else:
            raise FileNotFoundError(
                "No shutter-close times found for "
                f"{scoring.session!r}; looked in "
                f"{[str(c) for c in candidates]}. Pass frame_times= explicitly."
            )

    times = (
        np.load(frame_times)
        if isinstance(frame_times, (str, Path))
        else np.asarray(frame_times, dtype=float)
    )
    track = read_rigid_body_track(csv_path, rigid_body)
    if len(times) != len(track.position):
        raise ValueError(
            f"{len(times)} shutter times but {len(track.position)} tracked "
            f"frames for {scoring.session!r}; these are not the same take."
        )

    scoring.speed = binned_speed(times, track.position, scoring.times, scoring.step_s)
    covered = float(np.isfinite(scoring.speed).mean())
    print(
        f"    movement attached: {track.name!r}, {len(track.position)} frames, "
        f"{covered:.1%} of bins covered"
    )
    return scoring
