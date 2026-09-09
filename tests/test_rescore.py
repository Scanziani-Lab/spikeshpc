"""Re-applying the movement veto after the shutter timing has been corrected.

The veto is the only part of the scoring that reads the camera, so when the
shutter timestamps turn out to have been on the wrong clock, the LFP work does
not need repeating -- only the override it produced. `codes_before_veto` is
what makes that possible: it is the scoring the LFP decided, before movement
was allowed to overrule it.
"""

import json

import numpy as np
import pytest

from spikeshpc.states import STATE_CODES, rescore_movement

RATE = 120.0
STEP_S = 1.0


def motive_csv(path, position, frame_rate=RATE):
    """A Motive export whose rigid body follows `position`."""
    header = [
        ["Format Version", "1.23", "Capture Frame Rate", f"{frame_rate:.6f}",
         "Total Exported Frames", str(len(position))], [],
        ["", "", "Rigid Body", "Rigid Body", "Rigid Body"],
        ["", "", "Headset", "Headset", "Headset"],
        ["", "", "1", "1", "1"], ["", "", "", "", ""],
        ["", "", "Position", "Position", "Position"],
        ["Frame", "Time (Seconds)", "X", "Y", "Z"],
    ]
    lines = [",".join(map(str, r)) for r in header]
    for i, (x, y, z) in enumerate(position):
        lines.append(f"{i},{i / frame_rate:.6f},{x:.4f},{y:.4f},{z:.4f}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def session(tmp_path):
    """A session scored asleep throughout, with the animal running in the middle.

    The LFP called the whole thing NREM. The animal was in fact walking from
    100 s to 200 s, so a correctly timed movement trace vetoes that span to
    WAKE -- and a trace shifted by ten seconds vetoes the wrong span.
    """
    n_bins = 300
    times = np.arange(n_bins) * STEP_S + STEP_S / 2
    codes_before = np.full(n_bins, STATE_CODES["NREM"], dtype=np.int16)

    n_frames = int(RATE * n_bins)
    frame_times = np.arange(n_frames) / RATE
    moving = (frame_times >= 100.0) & (frame_times < 200.0)
    # 200 mm/s while walking, a millimetre-scale tremor while asleep
    step = np.where(moving, 200.0 / RATE, 0.05 / RATE)
    position = np.column_stack(
        [np.cumsum(step), np.zeros(n_frames), np.zeros(n_frames)]
    )

    states = tmp_path / "states"
    states.mkdir()
    csv = tmp_path / "take.csv"
    motive_csv(csv, position)
    np.save(states / "s1_shutter_close_times.npy", frame_times)

    rng = np.random.default_rng(0)
    np.savez_compressed(
        states / "s1_metrics.npz",
        times=times,
        codes=codes_before.copy(),
        codes_before_veto=codes_before,
        broadband=rng.normal(size=n_bins).astype("float32"),
        theta=rng.random(n_bins).astype("float32"),
        emg=rng.random(n_bins).astype("float32"),
        speed=np.zeros(n_bins, dtype="float32"),
    )
    (states / "s1_states.json").write_text(json.dumps({
        "session": "s1", "step_s": STEP_S, "state_codes": STATE_CODES,
        "intervals": {"NREM": [[0.0, float(n_bins)]], "WAKE": [], "REM": []},
        "thresholds": {"broadband": 0.1, "theta": 0.2, "emg": 0.8},
        "fractions": {"WAKE": 0.0, "NREM": 1.0, "REM": 0.0},
        "movement": {"applied": True, "n_reassigned": 0,
                     "vetoed_states": ["NREM", "REM"]},
        "min_state_duration_s": 6.0,
    }))
    return {"states": states, "csv": csv, "frame_times": frame_times,
            "n_bins": n_bins}


def walking(scoring):
    """Which bins came out WAKE."""
    return np.flatnonzero(np.asarray(scoring.codes) == STATE_CODES["WAKE"])


# ── the correction ──────────────────────────────────────────────────────
def test_the_veto_lands_on_the_bins_the_animal_was_moving(session):
    scoring = rescore_movement(
        session["states"], optitrack_csv=session["csv"],
        frame_times=session["frame_times"],
    )
    awake = walking(scoring)
    assert awake.size > 0
    assert 95 <= awake.min() <= 105, awake.min()
    assert 195 <= awake.max() <= 205, awake.max()


def test_a_shifted_clock_vetoes_the_wrong_span(session):
    """What the bug did: the same movement, credited ten seconds late."""
    correct = rescore_movement(
        session["states"], optitrack_csv=session["csv"],
        frame_times=session["frame_times"], write=False,
    )
    good = walking(correct).copy()

    shifted = rescore_movement(
        session["states"], optitrack_csv=session["csv"],
        frame_times=session["frame_times"] + 10.0, write=False,
    )
    bad = walking(shifted)

    assert bad.min() == pytest.approx(good.min() + 10, abs=2)
    assert np.setdiff1d(bad, good).size >= 8


def test_the_lfp_signals_are_left_exactly_as_they_were(session):
    """Nothing here recomputes a spectrogram; that is the entire point."""
    before = np.load(session["states"] / "s1_metrics.npz")
    keep = {k: before[k].copy() for k in ("broadband", "theta", "emg", "times")}

    rescore_movement(
        session["states"], optitrack_csv=session["csv"],
        frame_times=session["frame_times"],
    )
    after = np.load(session["states"] / "s1_metrics.npz")
    for name, original in keep.items():
        np.testing.assert_array_equal(after[name], original)


def test_codes_before_veto_survives_so_it_can_be_done_again(session):
    scoring = rescore_movement(
        session["states"], optitrack_csv=session["csv"],
        frame_times=session["frame_times"],
    )
    saved = np.load(session["states"] / "s1_metrics.npz")
    assert "codes_before_veto" in saved
    np.testing.assert_array_equal(
        saved["codes_before_veto"], scoring.codes_before_veto
    )
    assert (saved["codes_before_veto"] == STATE_CODES["NREM"]).all()


def test_running_it_twice_gives_the_same_answer(session):
    first = rescore_movement(
        session["states"], optitrack_csv=session["csv"],
        frame_times=session["frame_times"],
    )
    second = rescore_movement(
        session["states"], optitrack_csv=session["csv"],
        frame_times=session["frame_times"],
    )
    np.testing.assert_array_equal(first.codes, second.codes)


# ── writing back ────────────────────────────────────────────────────────
def test_the_original_is_kept_once(session):
    states = session["states"]
    rescore_movement(states, optitrack_csv=session["csv"],
                     frame_times=session["frame_times"])
    assert (states / "s1_states.orig.json").exists()
    assert (states / "s1_metrics.orig.npz").exists()
    first = (states / "s1_states.orig.json").read_text()

    rescore_movement(states, optitrack_csv=session["csv"],
                     frame_times=session["frame_times"] + 5.0)
    # the backup is still the pipeline's own output, not the first correction
    assert (states / "s1_states.orig.json").read_text() == first


def test_the_summary_is_rewritten_consistently(session):
    scoring = rescore_movement(
        session["states"], optitrack_csv=session["csv"],
        frame_times=session["frame_times"],
    )
    summary = json.loads((session["states"] / "s1_states.json").read_text())

    assert summary["rescored_movement"] is True
    assert summary["movement"]["n_reassigned"] > 0
    assert summary["fractions"]["WAKE"] == pytest.approx(
        float(np.mean(np.asarray(scoring.codes) == STATE_CODES["WAKE"]))
    )
    # intervals and codes must not disagree
    spans = summary["intervals"]["WAKE"]
    assert spans and sum(b - a for a, b in spans) == pytest.approx(
        summary["fractions"]["WAKE"] * session["n_bins"] * STEP_S, abs=STEP_S
    )


def test_write_false_leaves_the_files_alone(session):
    before = (session["states"] / "s1_states.json").read_text()
    rescore_movement(
        session["states"], optitrack_csv=session["csv"],
        frame_times=session["frame_times"], write=False,
    )
    assert (session["states"] / "s1_states.json").read_text() == before
    assert not (session["states"] / "s1_states.orig.json").exists()


def test_the_shutter_cache_is_found_without_being_named(session):
    """The pipeline caches it beside the states, which is where to look."""
    scoring = rescore_movement(session["states"], optitrack_csv=session["csv"])
    assert walking(scoring).size > 0


# ── what it refuses ─────────────────────────────────────────────────────
def test_a_session_without_codes_before_veto_is_refused(session):
    """Its states cannot be separated into a decision and an override."""
    path = session["states"] / "s1_metrics.npz"
    arrays = {k: v for k, v in np.load(path).items() if k != "codes_before_veto"}
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="no 'codes_before_veto' saved"):
        rescore_movement(session["states"], optitrack_csv=session["csv"],
                         frame_times=session["frame_times"])


def test_a_mismatched_take_is_refused(session):
    with pytest.raises(ValueError, match="not the same take"):
        rescore_movement(
            session["states"], optitrack_csv=session["csv"],
            frame_times=session["frame_times"][:-50],
        )


def test_a_missing_csv_is_refused(session):
    with pytest.raises(FileNotFoundError):
        rescore_movement(
            session["states"], optitrack_csv=session["states"] / "nope.csv",
            frame_times=session["frame_times"],
        )
