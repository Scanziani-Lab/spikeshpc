"""Loading the scoring back, and using it to restrict downstream analysis."""

import json

import numpy as np
import pytest

from spikeshpc.io import StateScoring, load_states
from spikeshpc.states import (
    frames_in_states,
    intervals_between_frames,
    slice_recording_to_states,
    state_epochs,
    times_in_states,
)

CODES = {"WAKE": 1, "NREM": 3, "REM": 5}


def write_scoring(dirpath, session="s1", codes=None, step_s=1.0, with_speed=True):
    """A states/ directory as stage 1 writes it."""
    dirpath.mkdir(parents=True, exist_ok=True)
    codes = np.asarray(
        codes if codes is not None else [1] * 10 + [3] * 10 + [5] * 5 + [1] * 5,
        dtype=np.int16,
    )
    n = len(codes)
    times = np.arange(n) * step_s + step_s / 2
    rng = np.random.default_rng(0)

    arrays = dict(
        times=times, codes=codes,
        broadband=rng.normal(size=n).astype("float32"),
        theta=rng.random(n).astype("float32"),
        emg=rng.random(n).astype("float32"),
    )
    if with_speed:
        arrays["speed"] = rng.random(n).astype("float32") * 50
    np.savez_compressed(dirpath / f"{session}_metrics.npz", **arrays)

    intervals = {name: [] for name in CODES}
    lookup = {v: k for k, v in CODES.items()}
    edges = np.flatnonzero(np.diff(codes)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, n]):
        intervals[lookup[int(codes[a])]].append(
            [float(times[a] - step_s / 2), float(times[b - 1] + step_s / 2)]
        )
    (dirpath / f"{session}_states.json").write_text(json.dumps({
        "session": session, "step_s": step_s, "state_codes": CODES,
        "intervals": intervals,
        "thresholds": {"broadband": 0.1, "theta": 0.2, "emg": 0.8},
        "fractions": {k: float(np.mean(codes == v)) for k, v in CODES.items()},
    }))
    return codes, times, intervals


# ── loading ─────────────────────────────────────────────────────────────
def test_load_states_reads_summary_and_signals(tmp_path):
    codes, times, _ = write_scoring(tmp_path / "states")
    s = load_states(tmp_path / "states")

    assert isinstance(s, StateScoring)
    assert s.session == "s1"
    np.testing.assert_array_equal(s.codes, codes)
    np.testing.assert_allclose(s.times, times)
    assert s.thresholds["emg"] == 0.8
    assert set(s.signals()) == {
        "broadband LFP (PC1)", "theta ratio",
        "EMG (LFP correlation)", "movement (mm/s)",
    }
    assert s.names[0] == "WAKE" and s.names[-1] == "WAKE"


def test_load_states_accepts_the_json_directly(tmp_path):
    write_scoring(tmp_path / "states")
    s = load_states(tmp_path / "states" / "s1_states.json")
    assert s.session == "s1"


def test_signals_omits_movement_when_it_was_not_recorded(tmp_path):
    write_scoring(tmp_path / "states", with_speed=False)
    assert "movement (mm/s)" not in load_states(tmp_path / "states").signals()


def test_several_sessions_must_be_named(tmp_path):
    write_scoring(tmp_path / "states", session="a")
    write_scoring(tmp_path / "states", session="b")

    with pytest.raises(ValueError, match="holds 2 sessions"):
        load_states(tmp_path / "states")
    assert load_states(tmp_path / "states", session="b").session == "b"


def test_missing_session_lists_what_is_there(tmp_path):
    write_scoring(tmp_path / "states", session="a")
    with pytest.raises(FileNotFoundError, match=r"found: \['a'\]"):
        load_states(tmp_path / "states", session="nope")


def test_missing_metrics_npz_is_reported(tmp_path):
    write_scoring(tmp_path / "states")
    (tmp_path / "states" / "s1_metrics.npz").unlink()
    with pytest.raises(FileNotFoundError, match="only holds the summary"):
        load_states(tmp_path / "states")


# ── epochs ──────────────────────────────────────────────────────────────
def test_epochs_are_contiguous_runs_with_bin_indices(tmp_path):
    write_scoring(tmp_path / "states")
    epochs = load_states(tmp_path / "states").epochs()

    assert [e.state for e in epochs] == ["WAKE", "NREM", "REM", "WAKE"]
    assert [e.duration for e in epochs] == [10.0, 10.0, 5.0, 5.0]
    assert epochs[0].start == 0.0 and epochs[0].stop == 10.0
    assert epochs[1].first_bin == 10 and epochs[1].last_bin == 19
    assert epochs[-1].stop == 30.0


def test_epochs_can_be_restricted_to_one_state(tmp_path):
    write_scoring(tmp_path / "states")
    rem = load_states(tmp_path / "states").epochs("REM")
    assert len(rem) == 1 and rem[0].state == "REM"


def test_state_epochs_handles_an_empty_recording():
    assert state_epochs(np.array([], dtype=int), np.array([])) == []


# ── masking ─────────────────────────────────────────────────────────────
def test_times_in_states_selects_the_right_span():
    intervals = {"WAKE": [[0.0, 10.0], [25.0, 30.0]], "NREM": [[10.0, 25.0]]}
    q = np.array([-1.0, 0.0, 5.0, 9.99, 10.0, 20.0, 26.0, 30.0, 31.0])
    mask = times_in_states(q, intervals, "WAKE")
    # half-open [start, stop): a time exactly on a boundary belongs to the
    # epoch that starts there, never to both
    assert list(mask) == [False, True, True, True, False, False, True, False, False]


def test_times_in_states_with_several_states():
    intervals = {"WAKE": [[0.0, 10.0]], "NREM": [[10.0, 20.0]], "REM": [[20.0, 25.0]]}
    q = np.array([5.0, 15.0, 22.0])
    assert list(times_in_states(q, intervals, ("WAKE", "REM"))) == [True, False, True]


def test_times_in_states_with_no_such_state():
    assert not times_in_states(np.arange(5.0), {"WAKE": [[0, 5]]}, "REM").any()


# ── the interval hazard ─────────────────────────────────────────────────
def test_interval_mask_drops_intervals_that_span_a_removed_gap():
    """The whole reason this is not just a frame mask.

    Frames 2 and 5 bracket a removed span; the interval between them would
    otherwise be spliced together and absorb everything that happened in it.
    """
    frame_mask = np.array([True, True, True, False, False, True, True])
    keep = intervals_between_frames(frame_mask)
    assert list(keep) == [True, True, False, False, False, True]


def test_frames_in_states_returns_both_masks():
    intervals = {"WAKE": [[0.0, 3.0], [6.0, 9.0]]}
    frame_times = np.arange(10.0)
    frames, ivals = frames_in_states(frame_times, intervals, "WAKE")
    assert list(frames) == [True, True, True, False, False, False,
                            True, True, True, False]
    # no interval bridges the 3-6 s gap
    assert list(ivals) == [True, True, False, False, False, False, True, True, False]
    assert len(ivals) == len(frame_times) - 1


def test_intervals_between_frames_on_degenerate_input():
    assert len(intervals_between_frames(np.array([True]))) == 0
    assert len(intervals_between_frames(np.array([], dtype=bool))) == 0


# ── recording slicing ───────────────────────────────────────────────────
def test_slice_recording_keeps_only_the_wanted_samples():
    import spikeinterface.full as si

    fs = 100.0
    rec = si.NumpyRecording(
        np.arange(1000, dtype="float32")[:, None], sampling_frequency=fs
    )
    intervals = {"WAKE": [[0.0, 2.0], [5.0, 7.0]], "NREM": [[2.0, 5.0]]}
    sliced = slice_recording_to_states(rec, intervals, "WAKE")

    assert sliced.get_num_frames() == 400
    got = sliced.get_traces().ravel()
    np.testing.assert_array_equal(got[:200], np.arange(0, 200))
    np.testing.assert_array_equal(got[200:], np.arange(500, 700))


def test_slice_recording_can_skip_brief_epochs():
    import spikeinterface.full as si

    rec = si.NumpyRecording(
        np.zeros((1000, 1), dtype="float32"), sampling_frequency=100.0
    )
    intervals = {"WAKE": [[0.0, 0.5], [5.0, 8.0]]}
    sliced = slice_recording_to_states(rec, intervals, "WAKE", min_duration_s=1.0)
    assert sliced.get_num_frames() == 300


def test_slice_recording_without_the_state_raises():
    import spikeinterface.full as si

    rec = si.NumpyRecording(
        np.zeros((100, 1), dtype="float32"), sampling_frequency=100.0
    )
    with pytest.raises(ValueError, match="No \\['REM'\\] intervals"):
        slice_recording_to_states(rec, {"WAKE": [[0.0, 1.0]]}, "REM")


# ── reviewing what the movement veto changed ────────────────────────────
def _write_with_veto(dirpath, session="v1"):
    """A scoring where the veto turned some REM bins into WAKE."""
    dirpath.mkdir(parents=True, exist_ok=True)
    before = np.array([5] * 10 + [3] * 10 + [5] * 10, dtype=np.int16)
    after = before.copy()
    after[:10] = 1                       # the first REM block was vetoed
    n = len(after)
    times = np.arange(n) + 0.5

    np.savez_compressed(
        dirpath / f"{session}_metrics.npz",
        times=times, codes=after, codes_before_veto=before,
        broadband=np.zeros(n, "float32"), theta=np.zeros(n, "float32"),
        emg=np.zeros(n, "float32"), speed=np.r_[np.full(10, 50.0), np.full(20, 2.0)],
    )
    (dirpath / f"{session}_states.json").write_text(json.dumps({
        "session": session, "step_s": 1.0, "state_codes": CODES,
        "intervals": {}, "thresholds": {}, "fractions": {},
    }))


def test_vetoed_bins_are_recoverable(tmp_path):
    _write_with_veto(tmp_path / "states")
    s = load_states(tmp_path / "states")

    assert s.codes_before_veto is not None
    assert s.vetoed.sum() == 10
    assert list(s.vetoed[:10]) == [True] * 10


def test_vetoed_epochs_keep_their_original_label(tmp_path):
    """A rejected REM call is WAKE now, so browsing REM can never find it."""
    _write_with_veto(tmp_path / "states")
    s = load_states(tmp_path / "states")

    assert [e.state for e in s.epochs()] == ["WAKE", "NREM", "REM"]
    vetoed = s.vetoed_epochs()
    assert len(vetoed) == 1
    assert vetoed[0].state == "REM"          # what the LFP alone called it
    assert (vetoed[0].start, vetoed[0].stop) == (0.0, 10.0)


def test_vetoed_epochs_needs_the_pre_veto_codes(tmp_path):
    write_scoring(tmp_path / "states")       # written without them
    s = load_states(tmp_path / "states")
    assert not s.vetoed.any()
    with pytest.raises(ValueError, match="no codes_before_veto saved"):
        s.vetoed_epochs()


# ── attaching movement to a scoring that never saved it ─────────────────
def _motive_csv(path, n_frames, frame_rate=120.0, seed=0):
    header = [["Format Version", "1.23", "Capture Frame Rate", f"{frame_rate:.6f}",
               "Total Exported Frames", str(n_frames)], [],
              ["", "", "Rigid Body", "Rigid Body", "Rigid Body"],
              ["", "", "Headset", "Headset", "Headset"],
              ["", "", "1", "1", "1"], ["", "", "", "", ""],
              ["", "", "Position", "Position", "Position"],
              ["Frame", "Time (Seconds)", "X", "Y", "Z"]]
    lines = [",".join(map(str, r)) for r in header]
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(0, 2.0, (n_frames, 3)), axis=0)
    for i in range(n_frames):
        lines.append(f"{i},{i / frame_rate:.6f},"
                     + ",".join(f"{v:.4f}" for v in pos[i]))
    path.write_text("\n".join(lines) + "\n")


def test_movement_can_be_attached_after_the_fact(tmp_path):
    """Scored without the veto, so no speed was saved -- attach it anyway."""
    from spikeshpc.states import attach_movement

    write_scoring(tmp_path / "states", with_speed=False)
    s = load_states(tmp_path / "states")
    assert s.speed is None
    assert "movement (mm/s)" not in s.signals()

    n_frames = 30 * 120
    _motive_csv(tmp_path / "take.csv", n_frames)
    np.save(tmp_path / "frames.npy", np.arange(n_frames) / 120.0)

    attach_movement(s, tmp_path / "take.csv", tmp_path / "frames.npy")
    assert s.speed is not None and len(s.speed) == len(s.times)
    assert "movement (mm/s)" in s.signals()
    assert np.isfinite(s.speed).all()


def test_load_states_can_attach_movement_directly(tmp_path):
    write_scoring(tmp_path / "states", with_speed=False)
    n_frames = 30 * 120
    _motive_csv(tmp_path / "take.csv", n_frames)
    np.save(tmp_path / "frames.npy", np.arange(n_frames) / 120.0)

    s = load_states(tmp_path / "states", optitrack_csv=tmp_path / "take.csv",
                    frame_times=tmp_path / "frames.npy")
    assert "movement (mm/s)" in s.signals()


def test_frame_times_are_found_next_to_the_states_dir(tmp_path):
    """The pipeline caches them there, so they should not need naming."""
    from spikeshpc.states import attach_movement

    states = tmp_path / "states"
    write_scoring(states, with_speed=False)
    n_frames = 30 * 120
    _motive_csv(tmp_path / "take.csv", n_frames)
    np.save(states / "s1_shutter_close_times.npy", np.arange(n_frames) / 120.0)

    s = load_states(states)
    attach_movement(s, tmp_path / "take.csv")
    assert s.speed is not None


def test_frame_times_are_found_next_to_the_csv(tmp_path):
    """The notebook's own convention."""
    from spikeshpc.states import attach_movement

    write_scoring(tmp_path / "states", with_speed=False)
    n_frames = 30 * 120
    _motive_csv(tmp_path / "take.csv", n_frames)
    np.save(tmp_path / "optitrack_shutter_close_times.npy",
            np.arange(n_frames) / 120.0)

    s = load_states(tmp_path / "states")
    attach_movement(s, tmp_path / "take.csv")
    assert s.speed is not None


def test_missing_frame_times_lists_where_it_looked(tmp_path):
    from spikeshpc.states import attach_movement

    write_scoring(tmp_path / "states", with_speed=False)
    _motive_csv(tmp_path / "take.csv", 100)
    s = load_states(tmp_path / "states")
    with pytest.raises(FileNotFoundError, match="No shutter-close times found"):
        attach_movement(s, tmp_path / "take.csv")


def test_a_mismatched_take_is_refused(tmp_path):
    """Different counts mean a different take; aligning them would be invented."""
    from spikeshpc.states import attach_movement

    write_scoring(tmp_path / "states", with_speed=False)
    _motive_csv(tmp_path / "take.csv", 500)
    np.save(tmp_path / "frames.npy", np.arange(400) / 120.0)

    s = load_states(tmp_path / "states")
    with pytest.raises(ValueError, match="not the same take"):
        attach_movement(s, tmp_path / "take.csv", tmp_path / "frames.npy")
