"""Automatic WAKE/NREM/REM scoring, after Watson et al. (2016).

Watson, Levenstein, Greene, Gelinas, Buzsáki & Rinzel, "Network Homeostasis
and State Dynamics of Neocortical Sleep", Neuron 90(4):839-852.
https://pmc.ncbi.nlm.nih.gov/articles/PMC4873379/

Three signals are computed from LFP, in 1 s steps over a 10 s window:

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
from pathlib import Path

import numpy as np
import spikeinterface.full as si
from scipy.signal import butter, sosfiltfilt

from .config import STATES_DIRNAME
from .io import channel_positions, read_recording

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
    nwin = int(round(window_s * fs))
    nstep = int(round(step_s * fs))
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
    nwin = int(round(window_s * fs))
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
    min_len = int(round(min_duration_s / step_s))
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


# ── top level ────────────────────────────────────────────────────────────
def _smooth(x, step_s, smooth_s):
    width = max(int(round(smooth_s / step_s)), 1)
    if width <= 1:
        return x
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def score_recording(rec_lfp, rec_emg, config, exclude_channels=()):
    """Compute the three signals and the state sequence for one recording."""
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
        "codes": codes,
        "thresholds": thresholds,
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
    return si.resample(rec, int(round(rate)))


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

    result = score_recording(rec_lfp, rec_emg, config, exclude_channels)
    result["session"] = session
    result["phys_path"] = str(phys_path.resolve())
    result["duration_s"] = float(
        source.get_num_frames() / source.get_sampling_frequency()
    )

    states_dir = output_dir / STATES_DIRNAME
    states_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        states_dir / f"{session}_metrics.npz",
        times=result["times"],
        broadband=result["broadband"],
        theta=result["theta"],
        emg=result["emg"],
        codes=result["codes"],
    )
    summary = {
        k: v
        for k, v in result.items()
        if k not in ("times", "broadband", "theta", "emg", "codes")
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
