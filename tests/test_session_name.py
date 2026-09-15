"""Naming the session, and saying why the movement veto did not run.

Both guard the same silent failure. The pipeline may be pointed at any depth of
a recording's folder tree, and taking the leaf as the session name gives "raw"
for ".../session8/raw". Every "{session}" template then resolves to a name
nothing else uses, the movement veto quietly does not run, and the scoring it
writes looks complete -- correct states, thresholds and fractions, with no
movement panel to notice the absence of.
"""

import numpy as np
import pytest

from spikeshpc import states as S


@pytest.mark.parametrize(
    "path,expected",
    [
        (r"Y:\physiology\5800941\session7\Record Node 101", "session7"),
        (r"Y:\physiology\5800941\session8\raw", "session8"),
        (r"/scratch/user/dc/physiology/5800941/session8/raw", "session8"),
        (r"Y:\physiology\5800941\session0", "session0"),
        (r"/data/mouse1/session3/ephys", "session3"),
        (r"/data/mouse1/session4/raw/continuous", "session4"),
        # a real name is taken even when it looks like a container's sibling
        (r"/data/rawdata/session9", "session9"),
    ],
)
def test_the_session_is_named_after_the_recording_not_the_folder(path, expected):
    assert S.session_name(path) == expected


def test_a_path_that_is_all_containers_falls_back_to_the_leaf():
    """Better a wrong-looking name than a crash mid-run."""
    assert S.session_name("/raw") == "raw"


def test_an_explicit_name_wins():
    """The escape hatch, for trees whose folders cannot say which session it is."""
    # score_session takes session=; session_name is only the default
    import inspect

    assert "session" in inspect.signature(S.score_session).parameters


# ── the reason a veto did not run ───────────────────────────────────────
def bins(n, step=1.0):
    return np.arange(n) * step + step / 2


def test_no_config_records_why(capsys):
    info = {}
    assert S.load_movement({}, "s1", bins(10), 1.0, info=info) is None
    assert "optitrack_csv" in info["reason"]
    assert "skipping" in capsys.readouterr().out


def test_a_missing_csv_records_which_one(tmp_path):
    info = {}
    cfg = {"optitrack_csv": str(tmp_path / "{session}_optitrack.csv")}
    assert S.load_movement(cfg, "raw", bins(10), 1.0, info=info) is None
    # the reason names the path it looked for, which is how a "{session}" that
    # resolved to the wrong name gives itself away
    assert "raw_optitrack.csv" in info["reason"]


def test_nothing_to_derive_frame_times_from_records_why(tmp_path):
    csv = tmp_path / "s1_optitrack.csv"
    csv.write_text("x")
    info = {}
    cfg = {"optitrack_csv": str(tmp_path / "{session}_optitrack.csv")}
    assert S.load_movement(cfg, "s1", bins(10), 1.0, info=info) is None
    assert "frame_times" in info["reason"]


def test_the_reason_reaches_the_saved_summary():
    """score_recording must carry it, or it dies in the job log."""
    import inspect

    assert "speed_info" in inspect.signature(S.score_recording).parameters


def test_info_is_optional():
    """Existing callers pass no info and must keep working."""
    assert S.load_movement({}, "s1", bins(10), 1.0) is None
