"""Tests for spikeshpc.drift -- the junction report."""

import json

import numpy as np
import pytest

from spikeshpc import drift as D
from spikeshpc.config import CONCAT_INFO_NAME, SORTER_DIRNAME

FS = 30000.0
BATCH = 60000              # 2 s per batch
N1 = N2 = 60 * BATCH       # 120 s each -> 60 batches each


def make_run(tmp_path, dshift, sample_offsets=None, pitch=20.0, sig_interp=20.0):
    """A minimal output_dir: concat_info.json + kilosort4/ops.npy."""
    offsets = sample_offsets or [0, N1, N1 + N2]
    n_chan = 8
    info = {
        "sampling_frequency": FS,
        "sample_offsets": offsets,
        "num_samples": [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)],
        "phys_paths": [f"/data/sess{i}" for i in range(len(offsets) - 1)],
        "channel_locations": np.c_[
            np.zeros(n_chan), np.arange(n_chan) * pitch
        ].tolist(),
    }
    (tmp_path / CONCAT_INFO_NAME).write_text(json.dumps(info))
    (tmp_path / SORTER_DIRNAME).mkdir(exist_ok=True)
    ops = {"dshift": np.asarray(dshift), "batch_size": BATCH,
           "sig_interp": sig_interp, "max_channel_distance": 32.0}
    np.save(tmp_path / SORTER_DIRNAME / "ops.npy", ops, allow_pickle=True)
    return tmp_path


def step_dshift(step_um, n_blocks=9, noise=0.0, seed=0):
    """Flat at 0 for session 1, flat at -step for session 2."""
    rng = np.random.default_rng(seed)
    n = 120
    d = np.zeros((n, n_blocks))
    d[n // 2 :] = -step_um
    if noise:
        d += rng.normal(0, noise, d.shape)
    return d


def test_row_pitch_from_channel_locations():
    loc = np.c_[np.zeros(8), np.arange(8) * 20.0]
    assert D.probe_row_pitch(loc) == pytest.approx(20.0)
    # staggered columns at the same depths must not read as zero pitch
    loc2 = np.c_[np.tile([0, 16.0], 4), np.repeat(np.arange(4) * 20.0, 2)]
    assert D.probe_row_pitch(loc2) == pytest.approx(20.0)


def test_row_pitch_is_none_for_a_single_row():
    assert D.probe_row_pitch(np.c_[np.arange(4) * 16.0, np.zeros(4)]) is None


def test_reports_the_step(tmp_path):
    make_run(tmp_path, step_dshift(54.0))
    report = D.drift_at_junction(tmp_path, verbose=False)

    assert len(report["junctions"]) == 1
    j = report["junctions"][0]
    assert j["step_um"] == pytest.approx(54.0, abs=0.5)
    assert j["step_in_row_pitch"] == pytest.approx(2.7, abs=0.1)
    assert j["step_in_sig_interp"] == pytest.approx(2.7, abs=0.1)
    assert j["boundary_time_s"] == pytest.approx(N1 / FS)
    assert j["boundary_batch"] == N1 // BATCH


@pytest.mark.parametrize(
    "step,pitch,expected",
    [
        (5.0, 20.0, "negligible"),          # below one row, whatever sig_interp says
        (18.0, 20.0, "negligible"),         # still under a row pitch
        (18.0, 15.0, "correctable"),        # over a row, within sig_interp
        (30.0, 15.0, "marginal"),           # 1.5x sig_interp
        (54.0, 20.0, "beyond correction"),  # 2.7x sig_interp -- the real case
    ],
)
def test_verdict_thresholds(tmp_path, step, pitch, expected):
    make_run(tmp_path, step_dshift(step), pitch=pitch)
    report = D.drift_at_junction(tmp_path, verbose=False)
    assert report["junctions"][0]["verdict"] == expected


def test_verdict_scales_with_sig_interp(tmp_path):
    """A 54 um step is fine if the interpolation kernel is wide enough."""
    make_run(tmp_path, step_dshift(54.0), sig_interp=60.0)
    assert D.drift_at_junction(tmp_path, verbose=False)["junctions"][0][
        "verdict"
    ] == "correctable"


def test_junction_batch_is_excluded_from_both_sides(tmp_path):
    """The straddling batch is a blend; it must not bias either mean."""
    d = step_dshift(54.0)
    b = N1 // BATCH
    d[b] = -27.0                      # the blended batch
    make_run(tmp_path, d)
    j = D.drift_at_junction(tmp_path, verbose=False)["junctions"][0]
    assert j["pre_mean_um"] == pytest.approx(0.0, abs=1e-6)
    assert j["post_mean_um"] == pytest.approx(-54.0, abs=1e-6)


def test_three_sessions_report_two_junctions(tmp_path):
    n = 180
    d = np.zeros((n, 9))
    d[60:120] = -54.0
    d[120:] = -20.0
    make_run(tmp_path, d, sample_offsets=[0, N1, 2 * N1, 3 * N1])
    js = D.drift_at_junction(tmp_path, verbose=False)["junctions"]
    assert len(js) == 2
    assert js[0]["step_um"] == pytest.approx(54.0, abs=0.5)
    assert js[1]["step_um"] == pytest.approx(34.0, abs=0.5)


def test_nonrigid_spread_is_measured(tmp_path):
    d = step_dshift(54.0)
    d[60:] = -54.0 + np.linspace(-3, 3, 9)   # a 6 um tilt across the probe
    make_run(tmp_path, d)
    j = D.drift_at_junction(tmp_path, verbose=False)["junctions"][0]
    assert j["nonrigid_spread_um"] == pytest.approx(6.0, abs=0.1)


def test_single_session_has_no_junction(tmp_path):
    make_run(tmp_path, step_dshift(0.0), sample_offsets=[0, N1])
    assert D.drift_at_junction(tmp_path, verbose=False) is None


def test_missing_dshift_is_handled(tmp_path):
    make_run(tmp_path, step_dshift(0.0))
    ops = {"dshift": None, "batch_size": BATCH}
    np.save(tmp_path / SORTER_DIRNAME / "ops.npy", ops, allow_pickle=True)
    assert D.drift_at_junction(tmp_path, verbose=False) is None


def test_missing_inputs_raise_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="run pre-processing first"):
        D.drift_at_junction(tmp_path, verbose=False)

    make_run(tmp_path, step_dshift(0.0))
    (tmp_path / SORTER_DIRNAME / "ops.npy").unlink()
    with pytest.raises(FileNotFoundError, match="run the sorting stage first"):
        D.drift_at_junction(tmp_path, verbose=False)


def test_verbose_output_names_the_sessions(tmp_path, capsys):
    make_run(tmp_path, step_dshift(54.0))
    D.drift_at_junction(tmp_path)
    out = capsys.readouterr().out
    assert "sess0 -> sess1" in out
    assert "STEP" in out and "54." in out
    assert "BEYOND CORRECTION" in out
    assert "row pitch" in out


def test_plot_writes_a_png(tmp_path):
    make_run(tmp_path, step_dshift(54.0))
    path = D.plot_drift(tmp_path)
    assert path.exists() and path.stat().st_size > 0
