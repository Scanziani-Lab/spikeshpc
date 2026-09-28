"""Loading the scoring back, and using it to restrict downstream analysis."""

import json

import numpy as np
import pytest

from spikeshpc.io import StateScoring, load_states
from spikeshpc.states import (
    behavior_interval_mask,
    frames_in_states,
    interval_mask_to_spans,
    intervals_between_frames,
    seconds_since,
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


# ── behaviour as an interval mask, for downstream analysis ──────────────
def _track(n_frames, rate=60.0, seed=0):
    """Shutter-aligned per-frame arrays for a head that behaves itself.

    Heading random-walks, elevation stays within a few tens of degrees of
    level, and speed is lognormal around a trot -- a bulk with no tail in it
    for an outlier cut to find.
    """
    rng = np.random.default_rng(seed)
    frame_times = np.arange(n_frames) / rate
    heading = np.cumsum(rng.normal(0, 0.5, n_frames)) % 360.0
    elevation = rng.normal(0, 10, n_frames)
    speed = 10 ** rng.normal(1.6, 0.25, n_frames)
    return frame_times, heading, {"speed": speed}, elevation


def test_with_no_thresholds_the_starting_mask_comes_back():
    frame_times, heading, kinematics, elevation = _track(600)
    _, wake = frames_in_states(frame_times, {"WAKE": [[0.0, 4.0], [6.0, 10.0]]}, "WAKE")

    keep, thresholds = behavior_interval_mask(
        heading, frame_times, kinematics, elevation, wake, verbose=False
    )
    np.testing.assert_array_equal(keep, wake)
    assert thresholds["max_elevation_deg"] is None
    assert thresholds["min_speed_mm_s"] is None and thresholds["automatic"] == []

    # and with no starting mask, every interval is a candidate
    everything, _ = behavior_interval_mask(None, frame_times, verbose=False)
    assert everything.shape == (len(frame_times) - 1,) and everything.all()


def test_an_interval_needs_both_of_its_frames_to_pass():
    elevation = np.array([0.0, 0.0, 70.0, 0.0, 0.0, 0.0])
    keep, _ = behavior_interval_mask(
        None, np.arange(6.0), elevation_deg=elevation, max_elevation_deg=60,
        verbose=False,
    )
    # frame 2 fails, so both intervals it bounds go with it
    assert list(keep) == [True, False, False, True, True]


def test_elevation_is_cut_on_its_magnitude():
    """Nose straight down leaves heading as ill-defined as nose straight up."""
    elevation = np.array([0.0, -70.0, 0.0, 0.0, 0.0, 70.0, 0.0])
    keep, _ = behavior_interval_mask(
        None, np.arange(7.0), elevation_deg=elevation, max_elevation_deg=60,
        verbose=False,
    )
    assert list(keep) == [False, False, True, True, False, False]


def test_minimum_and_maximum_speed_select_a_band():
    frame_times = np.arange(8.0)
    kinematics = {"speed": np.array([5.0, 5.0, 50.0, 50.0, 50.0, 500.0, 500.0, 50.0])}

    band, _ = behavior_interval_mask(
        None, frame_times, kinematics, min_speed_mm_s=10, max_speed_mm_s=100,
        verbose=False,
    )
    assert list(band) == [False, False, True, True, False, False, False]

    # one threshold splits moving from still, and a frame exactly on it is
    # still -- so no interval is both
    moving, _ = behavior_interval_mask(
        None, frame_times, kinematics, min_speed_mm_s=50, verbose=False
    )
    still, _ = behavior_interval_mask(
        None, frame_times, kinematics, max_speed_mm_s=50, verbose=False
    )
    assert list(moving) == [False, False, False, False, False, True, False]
    assert list(still) == [True, True, True, True, False, False, False]
    assert not (moving & still).any()


def test_angular_velocity_is_read_off_the_heading():
    """A two-turn spin in one second is dropped; the still head around it is not."""
    frame_times = np.arange(600) / 60.0
    heading = np.full(600, 90.0)
    heading[300:360] = (90.0 + 12.0 * np.arange(1, 61)) % 360.0  # 720 deg/s

    keep, _ = behavior_interval_mask(
        heading, frame_times, max_angular_velocity_deg_s=360, verbose=False
    )
    assert not keep[305:354].any()
    assert keep[:285].all() and keep[375:].all()


def test_automatic_maxima_cut_a_planted_tail_and_spare_the_bulk():
    frame_times, heading, kinematics, elevation = _track(20_000)
    glitches = np.arange(500, 20_000, 1000)  # 20 isolated frames
    elevation[glitches] = 88.0
    kinematics["speed"][glitches] = 20_000.0
    # a 90-degree step: the rigid body re-identified the wrong way round
    jumps = np.zeros(len(heading))
    jumps[glitches] = 90.0
    heading = (heading + np.cumsum(jumps)) % 360.0

    for name in ("max_elevation_deg", "max_angular_velocity_deg_s", "max_speed_mm_s"):
        keep, thresholds = behavior_interval_mask(
            heading, frame_times, kinematics, elevation, verbose=False,
            **{name: "automatic"},
        )
        assert thresholds["automatic"] == [name]
        # every glitch is cut, along with the interval leading into it...
        assert not keep[glitches - 1].any(), name
        # ...and very little of the rest. Elevation's bulk is half-normal, so
        # its fence (2.4 sd) trims a percent or two of honest frames with it
        assert keep.mean() > 0.95, (name, keep.mean())


def test_automatic_minimum_speed_splits_still_from_moving():
    """The breathing floor on one side of the trough, locomotion on the other."""
    rng = np.random.default_rng(0)
    still = 10 ** rng.normal(np.log10(3.0), 0.15, 3000)
    moving = 10 ** rng.normal(np.log10(40.0), 0.25, 3000)
    frame_times = np.arange(6000) / 60.0

    keep, thresholds = behavior_interval_mask(
        None, frame_times, {"speed": np.r_[still, moving]},
        min_speed_mm_s="automatic", verbose=False,
    )
    assert 3.0 < thresholds["min_speed_mm_s"] < 40.0
    assert keep[:2999].mean() < 0.02 and keep[3000:].mean() > 0.95


def test_automatic_thresholds_read_only_the_starting_intervals():
    """Whatever happens outside wake cannot move a cut made for wake."""
    frame_times, heading, kinematics, elevation = _track(12_000)
    _, wake = frames_in_states(frame_times, {"WAKE": [[0.0, 100.0]]}, "WAKE")
    wild_kinematics = {"speed": kinematics["speed"].copy()}
    wild_kinematics["speed"][6500:] = 1e6
    wild_elevation = elevation.copy()
    wild_elevation[6500:] = 89.0

    settings = dict(
        max_elevation_deg="automatic",
        max_speed_mm_s="automatic",
        min_speed_mm_s="automatic",
        verbose=False,
    )
    keep, thresholds = behavior_interval_mask(
        heading, frame_times, kinematics, elevation, wake, **settings
    )
    wild_keep, wild_thresholds = behavior_interval_mask(
        heading, frame_times, wild_kinematics, wild_elevation, wake, **settings
    )
    assert wild_thresholds == thresholds
    np.testing.assert_array_equal(wild_keep, keep)
    assert not keep[wake.sum():].any()


def test_nan_fails_only_the_thresholds_that_read_it():
    frame_times = np.arange(5.0)
    kinematics = {"speed": np.full(5, 50.0)}
    elevation = np.array([0.0, np.nan, 0.0, 0.0, 0.0])

    speed_only, _ = behavior_interval_mask(
        None, frame_times, kinematics, elevation, max_speed_mm_s=100, verbose=False
    )
    assert speed_only.all()
    with_elevation, _ = behavior_interval_mask(
        None, frame_times, kinematics, elevation, max_elevation_deg=60, verbose=False
    )
    assert list(with_elevation) == [False, False, True, True]


def test_min_duration_drops_short_stretches():
    frame_times = np.arange(21) * 0.1
    kinematics = {"speed": np.full(21, 50.0)}
    kinematics["speed"][[3, 10]] = 0.0  # leaves stretches of 0.2, 0.5 and 0.9 s
    start = np.ones(20, dtype=bool)

    keep, thresholds = behavior_interval_mask(
        None, frame_times, kinematics, interval_mask=start, min_speed_mm_s=10,
        min_duration_s=0.4, verbose=False,
    )
    assert not keep[:2].any()
    assert keep[4:9].all() and keep[11:].all()
    assert thresholds["min_duration_s"] == 0.4
    assert start.all()  # the mask passed in is left alone


def test_the_thresholds_are_plain_json():
    """They go into the parameters a tuning is saved with."""
    frame_times, heading, kinematics, elevation = _track(3000)
    _, thresholds = behavior_interval_mask(
        heading, frame_times, kinematics, elevation,
        max_elevation_deg="automatic", max_speed_mm_s=500, verbose=False,
    )
    assert json.loads(json.dumps(thresholds)) == thresholds
    assert thresholds["max_speed_mm_s"] == 500.0
    assert thresholds["automatic"] == ["max_elevation_deg"]
    assert thresholds["min_speed_mm_s"] is None


def test_masks_and_inputs_of_the_wrong_shape_are_refused():
    frame_times, heading, _, elevation = _track(100)
    frames, ivals = frames_in_states(frame_times, {"WAKE": [[0.0, 1.0]]}, "WAKE")

    with pytest.raises(ValueError, match="masks intervals, not"):
        behavior_interval_mask(heading, frame_times, interval_mask=frames, verbose=False)
    with pytest.raises(ValueError, match="is a tuple"):
        behavior_interval_mask(
            heading, frame_times, interval_mask=(frames, ivals), verbose=False
        )
    with pytest.raises(ValueError, match="one entry per frame"):
        behavior_interval_mask(
            heading, frame_times, elevation_deg=elevation[:-1],
            max_elevation_deg=60, verbose=False,
        )


def test_a_threshold_needs_its_input_and_a_word_it_knows():
    frame_times, heading, kinematics, _ = _track(100)
    with pytest.raises(ValueError, match="elevation_deg is needed"):
        behavior_interval_mask(
            heading, frame_times, kinematics, max_elevation_deg=60, verbose=False
        )
    with pytest.raises(ValueError, match="'automatic' or None"):
        behavior_interval_mask(
            heading, frame_times, kinematics, min_speed_mm_s="auto", verbose=False
        )


def test_spans_of_a_mask_are_its_runs_for_seconds_since():
    frame_times = np.arange(10.0)
    mask = np.array([True, True, False, False, True, False, False, True, True])

    spans = interval_mask_to_spans(frame_times, mask)
    assert spans == [[0.0, 2.0], [4.0, 5.0], [7.0, 9.0]]
    np.testing.assert_allclose(
        seconds_since(np.array([1.0, 3.0, 6.5, 9.0]), spans), [0.0, 1.0, 1.5, 0.0]
    )
    assert interval_mask_to_spans(frame_times, np.zeros(9, dtype=bool)) == []


def test_seconds_since_is_zero_inside_and_counts_up_after():
    spans = [[10.0, 20.0], [30.0, 40.0]]
    q = np.array([15.0, 20.0, 25.0, 30.0, 45.0])
    np.testing.assert_allclose(seconds_since(q, spans), [0.0, 0.0, 5.0, 0.0, 5.0])


def test_seconds_since_is_infinite_before_the_first_span():
    got = seconds_since(np.array([0.0, 5.0, 12.0]), [[10.0, 20.0]])
    assert np.isinf(got[:2]).all() and got[2] == 0.0
    assert np.isinf(seconds_since(np.array([1.0]), [])).all()


def test_seconds_since_counts_from_the_latest_end_whatever_the_order():
    """A short span nested in a long one must not end the long one early."""
    nested = [[0.0, 50.0], [10.0, 12.0]]
    np.testing.assert_allclose(seconds_since(np.array([30.0, 60.0]), nested), [0.0, 10.0])
    unsorted = [[30.0, 40.0], [0.0, 5.0]]
    np.testing.assert_allclose(seconds_since(np.array([45.0]), unsorted), [5.0])


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
