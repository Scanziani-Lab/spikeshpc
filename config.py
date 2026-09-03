"""Pipeline defaults and the output-file names every stage agrees on."""

import copy

# ── Files written into output_dir ────────────────────────────────────────
CONCAT_BIN_NAME = "concatenated.bin"
CONCAT_INFO_NAME = "concat_info.json"
CHANMAP_NAME = "chanMap.mat"
PROBE_NAME = "probe.json"
BAD_CHANNELS_NAME = "bad_channels.json"
STATES_DIRNAME = "states"
STATES_CONCAT_NAME = "states_concatenated.json"
SORTER_DIRNAME = "kilosort4"
ANALYZER_NAME = "analyzer.zarr"

DEFAULT_PIPELINE = {
    "job_kwargs": dict(n_jobs=16, chunk_duration="1s", progress_bar=True),
    # Optional spikeinterface steps applied *before* the concatenated binary
    # is written. Empty by default: kilosort4 does its own highpass filtering,
    # CAR and whitening, so filtering twice is usually not what you want.
    # Override with e.g. {"preprocessing": {"bandpass_filter": {}}}.
    "preprocessing": {},
    "binary": {
        # dtype of the concatenated binary. None keeps the recording's own
        # dtype (int16 for raw SpikeGLX/OpenEphys).
        "dtype": None,
        # Hand kilosort the acquisition system's own binary instead of writing
        # a copy, when that is possible: one recording, no preprocessing, no
        # dtype change, and the file's layout verified against the loaded
        # traces. Rewriting a 9 h session is ~an hour of pure copying and a
        # second full-size file on disk. Falls back to writing, with a reason
        # logged, whenever any of those conditions fails.
        "reuse_source": True,
    },
    "concatenation": {
        # How far apart two contacts may be and still count as the same
        # electrode site when aligning recordings. Neuropixels site pitch is
        # >= 15 um, so 1 um is loose enough for float noise and tight enough
        # to never match neighbours.
        "align_tolerance_um": 1.0,
        # Max allowed spread in sampling rate across recordings. 0 is strict.
        # SpikeGLX reports the true clock rate (e.g. 30000.048828125) while
        # OpenEphys reports a nominal 30000.0, so mixing systems needs a
        # tolerance here -- and that clock difference is real, so long
        # concatenations across systems will drift.
        "sampling_frequency_max_diff": 0.0,
    },
    # Brain-state scoring, after Watson et al. 2016. Runs per recording,
    # BEFORE concatenation, so the 10 s spectrogram window never straddles a
    # junction between sessions. See spikesphc/states.py for the method and
    # its deviations from the buzcode original.
    "state_scoring": {
        "enabled": True,
        # LFP is taken from the acquisition system's own LF band when there is
        # one (SpikeGLX imec0.lf, OpenEphys ...-LFP) and otherwise resampled
        # down from the AP band.
        "lfp_rate": 1250.0,
        # The EMG band (300-600 Hz) sits at 0.96 Nyquist for 1250 Hz data,
        # which is a poor place for a Butterworth. buzcode filters there
        # anyway; we resample to 2500 Hz first instead.
        "emg_rate": 2500.0,
        "window_s": 10.0,
        "step_s": 1.0,
        "freq_range": [1.0, 100.0],
        "n_freqs": 100,
        # Slow-wave PC is sign-fixed to correlate positively with power below
        # this cutoff, as in the paper ("power in the low (<32 Hz) frequencies").
        "slow_wave_max_hz": 32.0,
        "theta_band": [5.0, 10.0],
        "theta_ref_band": [2.0, 16.0],
        "emg_band": [300.0, 600.0],
        # Channels averaged for the broadband/theta signals and correlated for
        # the EMG. null = pick this many spread evenly along the probe.
        # Theta is strongest at the hippocampal fissure, so on a probe that
        # spans several structures you will usually want to name channels
        # explicitly rather than average over everything.
        "sw_channels": None,
        "theta_channels": None,
        "emg_channels": None,
        "n_sw_channels": 8,
        "n_theta_channels": 8,
        "n_emg_channels": 8,
        # Volume conduction correlates nearby sites regardless of muscle tone,
        # so EMG pairs closer than this are dropped.
        "emg_min_distance_um": 100.0,
        "smooth_s": 10.0,
        "min_state_duration_s": 6.0,
    },
    # Channels to exclude from sorting -- dead/broken sites, or anything out of
    # the brain. Channel ids as listed in concat_info.json (preferred) or ints
    # read as 0-based rows of concatenated.bin. Read fresh on every run, so a
    # revised list can be applied with --skip_preprocessing, without rewriting
    # the binary.
    "bad_channels": [],
    # Optional automatic detection, unioned with the manual list above. Any key
    # other than "enabled" is passed through to si.detect_bad_channels, e.g.
    # {"method": "std"} or {"channel_filters": ["dead", "noise"]} to keep
    # out-of-brain channels. seed is fixed so a re-run reproduces the same set.
    "detect_bad_channels": {
        "enabled": False,
        "method": "coherence+psd",
        "seed": 0,
    },
    # Overrides on kilosort.DEFAULT_SETTINGS. n_chan_bin/fs are filled in from
    # the recording and should not be set here. Sorting parameters can also be
    # set from pipeline_config.json
    "sorting": {},
    "postprocessing": {
        "random_spikes": {},
        "noise_levels": {},
        "templates": {},
        "unit_locations": {"method": "center_of_mass"},
        "amplitude_scalings": {},
        "spike_amplitudes": {},
        "spike_locations": {},
        "waveforms": {},
        "template_similarity": {},
        "correlograms": {},
        "auto_correlograms": {},
        "isi_histograms": {},
        "principal_components": {},
        "valid_unit_periods": {},
        "quality_metrics": {},
        "template_metrics": {"include_multi_channel_metrics": True},
    },
}


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` onto a copy of `base`."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
