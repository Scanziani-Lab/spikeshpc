"""Tests for spikeshpc's vendored OptiTrack CSV reader."""

import numpy as np
import pytest

from spikeshpc.tracking import (
    HEADER_ROWS,
    read_rigid_body_track,
    rigid_body_names,
)


def write_csv(path, bodies=("Headset",), n_frames=10, frame_rate=120.0,
              include_rotation=True, drop_axis=None, markers=True):
    """A Motive-shaped export: 8 header rows, then Frame,Time,<body columns>."""
    header = [[] for _ in range(HEADER_ROWS)]
    header[0] = ["Format Version", "1.23",
                 "Capture Frame Rate", f"{frame_rate:.6f}",
                 "Total Exported Frames", str(n_frames)]
    header[1] = []
    for row, first in ((2, "Type"), (3, "Name"), (4, "ID"), (5, "Parent"),
                       (6, ""), (7, "")):
        header[row] = ["", ""]
    header[7][0], header[7][1] = "Frame", "Time (Seconds)"

    col_of = {}
    for body in bodies:
        components = (["Rotation"] if include_rotation else []) + ["Position"]
        for comp in components:
            axes = "XYZW" if comp == "Rotation" else "XYZ"
            for axis in axes:
                if comp == "Position" and axis == drop_axis and body == bodies[0]:
                    continue
                header[2].append("Rigid Body")
                header[3].append(body)
                header[4].append("1")
                header[5].append("")
                header[6].append(comp)
                header[7].append(axis)
                if comp == "Position":
                    col_of[(body, axis)] = len(header[2]) - 1
        if markers:  # marker columns must be ignored
            for axis in "XYZ":
                header[2].append("Rigid Body Marker")
                header[3].append(f"{body}:Marker001")
                header[4].append("2")
                header[5].append(body)
                header[6].append("Position")
                header[7].append(axis)

    n_cols = len(header[2])
    rng = np.random.default_rng(0)
    values = rng.normal(0, 100, (n_frames, n_cols))
    lines = [",".join(map(str, r)) for r in header]
    for i in range(n_frames):
        lines.append(",".join([str(i), f"{i / frame_rate:.6f}",
                               *[f"{v:.6f}" for v in values[i]]]))
    path.write_text("\n".join(lines) + "\n")
    return values, col_of


def test_reads_position_for_a_single_body(tmp_path):
    p = tmp_path / "take.csv"
    values, col_of = write_csv(p, n_frames=25)
    track = read_rigid_body_track(p)

    assert track.name == "Headset"
    assert track.frame_rate == pytest.approx(120.0)
    assert track.position.shape == (25, 3)
    assert len(track) == 25
    np.testing.assert_array_equal(track.frame_numbers, np.arange(25))
    for j, axis in enumerate("XYZ"):
        np.testing.assert_allclose(
            track.position[:, j], values[:, col_of[("Headset", axis)] - 2], rtol=1e-5
        )


def test_marker_and_rotation_columns_are_ignored(tmp_path):
    """Position only -- picking up a marker or a quaternion would be silent."""
    p = tmp_path / "take.csv"
    values, col_of = write_csv(p, include_rotation=True, markers=True)
    track = read_rigid_body_track(p)
    assert track.position.shape[1] == 3
    # the position columns sit at 6,7,8, after the quaternion -- a reader that
    # takes the leading columns instead would silently return rotation
    for j, axis in enumerate("XYZ"):
        np.testing.assert_allclose(
            track.position[:, j], values[:, col_of[("Headset", axis)] - 2], rtol=1e-5
        )


def test_second_body_reads_its_own_columns_not_the_leading_ones(tmp_path):
    """Regression: pandas usecols+integer names reads the first N columns.

    The second body's position sits far along the row, so this is where a
    positional mix-up shows up as plausible-but-wrong tracking data.
    """
    p = tmp_path / "take.csv"
    values, col_of = write_csv(p, bodies=("Headset", "Platform"), n_frames=15)
    track = read_rigid_body_track(p, rigid_body="Platform")
    assert col_of[("Platform", "X")] > 8, "fixture should place it late in the row"
    for j, axis in enumerate("XYZ"):
        np.testing.assert_allclose(
            track.position[:, j], values[:, col_of[("Platform", axis)] - 2], rtol=1e-5
        )


def test_rigid_body_names_lists_bodies(tmp_path):
    p = tmp_path / "take.csv"
    write_csv(p, bodies=("Headset", "Platform"))
    assert rigid_body_names(p) == ["Headset", "Platform"]


def test_several_bodies_must_be_named(tmp_path):
    p = tmp_path / "take.csv"
    write_csv(p, bodies=("Headset", "Platform"))
    with pytest.raises(ValueError, match="has 2 rigid bodies"):
        read_rigid_body_track(p)


def test_named_body_is_selected(tmp_path):
    p = tmp_path / "take.csv"
    values, col_of = write_csv(p, bodies=("Headset", "Platform"), n_frames=12)
    track = read_rigid_body_track(p, rigid_body="Platform")
    assert track.name == "Platform"
    np.testing.assert_allclose(
        track.position[:, 0], values[:, col_of[("Platform", "X")] - 2], rtol=1e-5
    )


def test_unknown_body_lists_the_available_ones(tmp_path):
    p = tmp_path / "take.csv"
    write_csv(p, bodies=("Headset", "Platform"))
    with pytest.raises(ValueError, match="available: \\['Headset', 'Platform'\\]"):
        read_rigid_body_track(p, rigid_body="Nope")


def test_missing_axis_is_reported(tmp_path):
    p = tmp_path / "take.csv"
    write_csv(p, drop_axis="Z")
    with pytest.raises(ValueError, match="missing position axes \\['Z'\\]"):
        read_rigid_body_track(p)


def test_a_file_that_is_not_a_motive_export_is_rejected(tmp_path):
    p = tmp_path / "nope.csv"
    p.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="fewer than 8 header rows"):
        read_rigid_body_track(p)


def test_header_without_a_frame_rate_is_rejected(tmp_path):
    p = tmp_path / "take.csv"
    write_csv(p)
    lines = p.read_text().splitlines()
    lines[0] = "Format Version,1.23"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="no 'Capture Frame Rate'"):
        read_rigid_body_track(p)


def test_no_rigid_bodies_at_all_is_rejected(tmp_path):
    p = tmp_path / "take.csv"
    write_csv(p)
    lines = p.read_text().splitlines()
    lines[2] = lines[2].replace("Rigid Body,", "Marker,")
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="No rigid-body Position columns"):
        read_rigid_body_track(p)


def test_load_movement_uses_the_vendored_reader(tmp_path, capsys):
    """End to end through the config path, with no optitrack anywhere."""
    from spikeshpc.states import load_movement

    csv_path = tmp_path / "s1.csv"
    write_csv(csv_path, n_frames=240, frame_rate=120.0)
    np.save(tmp_path / "s1.npy", np.arange(240) / 120.0)

    cfg = {
        "optitrack_csv": str(tmp_path / "{session}.csv"),
        "frame_times": str(tmp_path / "{session}.npy"),
    }
    times = np.arange(2) * 1.0 + 0.5
    speed = load_movement(cfg, "s1", times, 1.0)

    assert speed is not None and speed.shape == (2,)
    assert np.isfinite(speed).all()
    out = capsys.readouterr().out
    assert "'Headset'" in out and "240 frames" in out and "120 Hz" in out


def test_load_movement_reports_a_bad_csv_without_raising(tmp_path, capsys):
    from spikeshpc.states import load_movement

    (tmp_path / "s1.csv").write_text("not,a,motive,export\n")
    np.save(tmp_path / "s1.npy", np.arange(10.0))
    cfg = {
        "optitrack_csv": str(tmp_path / "{session}.csv"),
        "frame_times": str(tmp_path / "{session}.npy"),
    }
    assert load_movement(cfg, "s1", np.arange(10.0), 1.0) is None
    assert "skipping the veto" in capsys.readouterr().out


def test_spikeshpc_does_not_import_optitrack():
    """The point of vendoring: no cross-repo import anywhere in the package.

    Resolved from this file rather than from `spikeshpc.__file__`, which is
    None whenever the package is shadowed by a same-named namespace directory.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "spikeshpc"
    modules = sorted(root.glob("*.py"))
    assert modules, f"no package modules found at {root}"

    offenders = [
        f"{p.name}:{i}"
        for p in modules
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "import optitrack" in line or "from optitrack" in line
    ]
    assert not offenders, f"optitrack imported at {offenders}"
