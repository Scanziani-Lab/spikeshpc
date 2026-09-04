"""Tests for the OptiTrack movement veto in spikeshpc.states."""

import numpy as np
import pytest

from spikeshpc import states as S

STEP = 1.0
FRAME_RATE = 120.0


def frames_and_times(bin_speeds, step=STEP, rate=FRAME_RATE):
    """Synthesise a position track whose per-bin speed is `bin_speeds`."""
    per_bin = int(rate * step)
    times, pos = [], [np.zeros(3)]
    t = 0.0
    for v in bin_speeds:
        for _ in range(per_bin):
            t += 1 / rate
            times.append(t)
            pos.append(pos[-1] + np.array([v / rate, 0.0, 0.0]))
    return np.array(times), np.array(pos[1:])


def bin_times(n, step=STEP):
    # matches the spectrogram grid: centres at window/2 + k*step
    return np.arange(n) * step + step / 2


def test_binned_speed_recovers_known_speeds():
    speeds = [1.0, 50.0, 3.0, 200.0]
    ft, pos = frames_and_times(speeds)
    times = bin_times(len(speeds))
    got = S.binned_speed(ft, pos, times, STEP)
    np.testing.assert_allclose(got, speeds, rtol=0.05)


def test_binned_speed_marks_empty_bins_nan():
    ft, pos = frames_and_times([5.0, 5.0])
    times = bin_times(5)          # more bins than tracking covers
    got = S.binned_speed(ft, pos, times, STEP)
    assert np.isfinite(got[:2]).all()
    assert np.isnan(got[2:]).all()


def test_binned_speed_ignores_tracking_dropouts():
    """A NaN gap must not read as stillness."""
    ft, pos = frames_and_times([10.0, 10.0, 10.0])
    pos[130:180] = np.nan          # drop out mid-recording
    times = bin_times(3)
    got = S.binned_speed(ft, pos, times, STEP)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, 10.0, rtol=0.15)


def test_binned_speed_rejects_mismatched_lengths():
    ft, pos = frames_and_times([1.0])
    with pytest.raises(ValueError, match="align them before calling"):
        S.binned_speed(ft[:-5], pos, bin_times(1), STEP)


def test_movement_threshold_splits_immobility_from_locomotion():
    """Breathing keeps speed off zero, so the split must be between two modes."""
    rng = np.random.default_rng(0)
    still = 10 ** rng.normal(np.log10(3.0), 0.15, 4000)   # breathing floor
    moving = 10 ** rng.normal(np.log10(40.0), 0.25, 4000)
    thr = S.movement_threshold(np.r_[still, moving])
    assert 3.0 < 10**thr < 40.0
    # and it must not sit at zero, which is what a naive threshold would do
    assert 10**thr > 5.0


def test_movement_threshold_is_none_without_data():
    assert S.movement_threshold(np.full(5, np.nan)) is None


def test_veto_reassigns_moving_sleep_to_wake():
    codes = np.full(60, S.STATE_CODES["REM"], dtype=np.int16)
    speed = np.full(60, 2.0)
    speed[:30] = 50.0                      # first half the animal is running
    out, info = S.apply_movement_veto(codes, speed, threshold=1.0, step_s=STEP)

    assert info["applied"] is True
    assert np.all(out[:30] == S.STATE_CODES["WAKE"])
    assert np.all(out[30:] == S.STATE_CODES["REM"])


def test_veto_is_asymmetric_and_never_creates_sleep():
    """Stillness must not push a WAKE bin towards sleep."""
    codes = np.full(60, S.STATE_CODES["WAKE"], dtype=np.int16)
    speed = np.full(60, 0.5)               # perfectly still, but awake
    out, _ = S.apply_movement_veto(codes, speed, threshold=1.0, step_s=STEP)
    assert np.all(out == S.STATE_CODES["WAKE"])


def test_veto_leaves_untracked_bins_alone():
    codes = np.full(60, S.STATE_CODES["NREM"], dtype=np.int16)
    speed = np.full(60, np.nan)
    speed[:30] = 50.0
    out, info = S.apply_movement_veto(codes, speed, threshold=1.0, step_s=STEP)
    assert np.all(out[:30] == S.STATE_CODES["WAKE"])
    assert np.all(out[30:] == S.STATE_CODES["NREM"])
    assert info["coverage"] == pytest.approx(0.5)


def test_veto_respects_the_state_list():
    codes = np.full(60, S.STATE_CODES["NREM"], dtype=np.int16)
    speed = np.full(60, 50.0)
    out, _ = S.apply_movement_veto(
        codes, speed, threshold=1.0, step_s=STEP, veto=("REM",)
    )
    assert np.all(out == S.STATE_CODES["NREM"])   # NREM not in the veto list


def test_veto_reapplies_min_duration_smoothing():
    """Vetoing punches holes; the result must not be left fragmented."""
    codes = np.full(120, S.STATE_CODES["NREM"], dtype=np.int16)
    speed = np.full(120, 2.0)
    speed[60:62] = 50.0                    # a 2 s twitch inside a long bout
    out, _ = S.apply_movement_veto(
        codes, speed, threshold=1.0, step_s=STEP, min_duration_s=6.0
    )
    assert np.all(out == S.STATE_CODES["NREM"]), "2 s hole should be absorbed"


def test_veto_reports_nothing_to_do_without_movement():
    codes = np.full(10, S.STATE_CODES["REM"], dtype=np.int16)
    out, info = S.apply_movement_veto(codes, np.full(10, np.nan))
    assert info["applied"] is False
    np.testing.assert_array_equal(out, codes)


def test_load_movement_skips_when_unconfigured(capsys):
    assert S.load_movement({}, "s1", bin_times(10), STEP) is None
    assert "no optitrack_csv" in capsys.readouterr().out


def test_load_movement_skips_when_files_are_absent(tmp_path, capsys):
    cfg = {
        "optitrack_csv": str(tmp_path / "{session}.csv"),
        "frame_times": str(tmp_path / "{session}.npy"),
    }
    assert S.load_movement(cfg, "s1", bin_times(10), STEP) is None
    assert "no tracking files" in capsys.readouterr().out


def test_running_theta_is_removed_but_immobile_theta_survives():
    """The case the veto exists for: locomotion theta looks exactly like REM.

    Both halves are scored REM by the LFP. Only the tracker can tell them
    apart, and only the moving half should lose the label.
    """
    codes = np.full(120, S.STATE_CODES["REM"], dtype=np.int16)
    rng = np.random.default_rng(0)
    speed = np.r_[
        10 ** rng.normal(np.log10(40.0), 0.2, 60),   # running with theta
        10 ** rng.normal(np.log10(3.0), 0.15, 60),   # true REM: breathing only
    ]
    out, info = S.apply_movement_veto(codes, speed, step_s=STEP)

    assert (out[:60] == S.STATE_CODES["WAKE"]).mean() > 0.9
    assert (out[60:] == S.STATE_CODES["REM"]).mean() > 0.9
    assert 3.0 < info["threshold_speed"] < 40.0
