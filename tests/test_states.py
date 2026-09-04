"""Tests for spikeshpc.states -- the pieces that are ours, not spikeinterface's."""

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from spikeshpc import states as S

FS = 2500.0
NCH = 8
BLOCK = 120.0
PLAN = ["WAKE", "NREM", "REM", "NREM", "WAKE"]
# slow, theta, broadband, shared-HF (muscle) weights per state
WEIGHTS = {
    "WAKE": (1.0, 0.5, 1.0, 1.0),
    "NREM": (6.0, 0.3, 1.0, 0.0),
    "REM": (0.8, 5.0, 1.0, 0.0),
}


@pytest.fixture(scope="module")
def synthetic():
    """LFP with the spectral signature of each state, plus the ground truth.

    Band-limited power rather than pure tones: PCA on a spectrogram looks for a
    broadband low-frequency mode, and a single sinusoid gives it a narrow spike
    that PC1 latches onto instead.
    """
    import probeinterface
    import spikeinterface.full as si

    rng = np.random.default_rng(0)
    n = int(BLOCK * FS)
    slow_sos = butter(2, [0.5, 4.0], btype="band", fs=FS, output="sos")
    theta_sos = butter(2, [6.0, 9.0], btype="band", fs=FS, output="sos")

    def band(sos):
        x = sosfiltfilt(sos, rng.normal(0, 1, size=(n, NCH)), axis=0)
        return x / x.std()

    traces, truth = [], []
    for state in PLAN:
        w_slow, w_theta, w_broad, w_emg = WEIGHTS[state]
        sig = (
            w_slow * band(slow_sos)
            + w_theta * band(theta_sos)
            + w_broad * rng.normal(0, 1, size=(n, NCH))
        )
        if w_emg:
            sig += w_emg * rng.normal(0, 1, size=n)[:, None]
        traces.append(sig * 50.0)
        truth += [state] * int(BLOCK)

    rec = si.NumpyRecording(
        np.concatenate(traces).astype("float32"), sampling_frequency=FS
    )
    probe = probeinterface.Probe(ndim=2)
    probe.set_contacts(
        positions=np.c_[np.zeros(NCH), np.arange(NCH) * 200.0],
        shapes="circle",
        shape_params={"radius": 5},
    )
    probe.set_device_channel_indices(np.arange(NCH))
    rec = rec.set_probe(probe)
    rec.set_property("gain_to_uV", np.ones(NCH))
    rec.set_property("offset_to_uV", np.zeros(NCH))
    return rec, np.array(truth)


def test_spectrogram_shape_and_time_base():
    sig = np.random.default_rng(0).normal(size=int(60 * FS)).astype("float32")
    times, freqs, spec = S.log_spectrogram(sig, FS, 10.0, 1.0, [1.0, 100.0], 100)

    assert spec.shape == (100, len(times))
    assert freqs[0] == pytest.approx(1.0)
    assert freqs[-1] == pytest.approx(100.0)
    # 1 s step, first window centred half a window in
    assert times[1] - times[0] == pytest.approx(1.0)
    assert times[0] == pytest.approx(5.0)
    assert len(times) == 1 + (len(sig) - int(10 * FS)) // int(FS)


def test_spectrogram_finds_a_tone():
    t = np.arange(int(60 * FS)) / FS
    tone = np.sin(2 * np.pi * 7.0 * t).astype("float32")
    _, freqs, spec = S.log_spectrogram(tone, FS, 10.0, 1.0, [1.0, 100.0], 100)
    assert freqs[np.argmax(spec.mean(axis=1))] == pytest.approx(7.0, abs=0.5)


def test_spectrogram_rejects_short_recording():
    with pytest.raises(ValueError, match="shorter than"):
        S.log_spectrogram(np.zeros(int(5 * FS), dtype="float32"), FS,
                          10.0, 1.0, [1.0, 100.0], 100)


def test_theta_ratio_is_a_fraction():
    freqs = np.logspace(0, 2, 100)
    spec = np.random.default_rng(0).random((100, 50)).astype("float32")
    ratio = S.theta_ratio(spec, freqs, [5.0, 10.0], [2.0, 16.0])
    # 5-10 is inside 2-16, so the ratio cannot exceed 1
    assert np.all(ratio >= 0) and np.all(ratio <= 1.0 + 1e-6)


def test_bimodal_threshold_splits_two_modes():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 5000), rng.normal(10, 1, 5000)]
    assert 3.0 < S.bimodal_threshold(x) < 7.0


def test_bimodal_threshold_falls_back_on_unimodal():
    x = np.random.default_rng(0).normal(0, 1, 5000)
    assert S.bimodal_threshold(x) == pytest.approx(np.median(x), abs=1.0)


def test_min_duration_absorbs_short_runs():
    codes = np.array([3] * 20 + [1] * 2 + [3] * 20)
    assert np.all(S.enforce_min_duration(codes, 1.0, 6.0) == 3)


def test_min_duration_keeps_long_runs():
    codes = np.array([3] * 20 + [1] * 20)
    assert np.array_equal(S.enforce_min_duration(codes, 1.0, 6.0), codes)


def test_intervals_bound_runs_by_half_a_step():
    iv = S.intervals_from_states(np.array([1, 1, 3, 3, 5]), np.arange(5.0), 1.0)
    assert iv["WAKE"] == [[-0.5, 1.5]]
    assert iv["NREM"] == [[1.5, 3.5]]
    assert iv["REM"] == [[3.5, 4.5]]


def test_classify_follows_buzcode_rules():
    thr = {"broadband": 0.0, "theta": 0.5, "emg": 0.1}
    #                    NREM   REM    WAKE(emg)  WAKE(quiet, no theta)
    broadband = np.array([1.0, -1.0, -1.0, -1.0])
    theta = np.array([0.0, 0.9, 0.9, 0.0])
    emg = np.array([0.0, 0.0, 0.5, 0.0])
    codes = S.classify_states(broadband, theta, emg, thr)
    assert [S.CODE_NAMES[c] for c in codes] == ["NREM", "REM", "WAKE", "WAKE"]


def test_pick_channels_spreads_and_excludes():
    import probeinterface
    import spikeinterface.full as si

    rec = si.NumpyRecording(np.zeros((100, 8), "float32"), sampling_frequency=FS)
    probe = probeinterface.Probe(ndim=2)
    probe.set_contacts(
        positions=np.c_[np.zeros(8), np.arange(8) * 20.0],
        shapes="circle", shape_params={"radius": 5},
    )
    probe.set_device_channel_indices(np.arange(8))
    rec = rec.set_probe(probe)

    picked = [str(c) for c in S.pick_channels(rec, 4)]
    assert len(picked) == 4
    assert picked[0] == "0" and picked[-1] == "7"  # spans the probe

    kept = [str(c) for c in S.pick_channels(rec, 8, exclude=["0", "1"])]
    assert "0" not in kept and "1" not in kept

    with pytest.raises(ValueError, match="not in this recording"):
        S.pick_channels(rec, 2, explicit=["nope"])


def test_scoring_recovers_planted_states(synthetic):
    """End to end: the three signals must separate the planted states."""
    from spikeshpc.config import DEFAULT_PIPELINE

    rec, truth = synthetic
    cfg = dict(DEFAULT_PIPELINE["state_scoring"])
    cfg.update(lfp_rate=1250.0, emg_rate=FS, emg_min_distance_um=100.0)

    result = S.score_recording(S._resample_to(rec, cfg["lfp_rate"]), rec, cfg)
    times = result["times"]
    true = truth[np.clip(times.round().astype(int), 0, len(truth) - 1)]
    pred = np.array([S.CODE_NAMES[int(c)] for c in result["codes"]])

    # The 10 s window straddles block boundaries, so those bins are ambiguous.
    interior = np.array(
        [min(x % BLOCK, BLOCK - (x % BLOCK)) >= 6 for x in times]
    )
    assert (pred[interior] == true[interior]).mean() > 0.9

    # and each metric must be highest in the state it is meant to mark
    for metric, state in (("broadband", "NREM"), ("theta", "REM"), ("emg", "WAKE")):
        means = {s: result[metric][true == s].mean() for s in ("WAKE", "NREM", "REM")}
        assert max(means, key=means.get) == state, (metric, means)


def test_merge_shifts_sessions_onto_concatenated_clock(tmp_path):
    result = {
        "session": "a",
        "intervals": {"WAKE": [[0.0, 10.0]], "NREM": [[10.0, 20.0]], "REM": []},
    }
    info = {"sampling_frequency": 30000.0, "sample_offsets": [0, 30000 * 100]}
    merged = S.merge_to_concatenated_time([result, result], info, tmp_path)

    assert merged["intervals"]["WAKE"] == [[0.0, 10.0], [100.0, 110.0]]
    assert merged["intervals"]["NREM"] == [[10.0, 20.0], [110.0, 120.0]]
    assert (tmp_path / "states" / "states_concatenated.json").exists()
