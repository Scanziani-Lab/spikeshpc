"""Parsing for OptiTrack Motive CSV exports.

The export has an 8-row header before the data starts:

1. take metadata (``Format Version``, ``Total Exported Frames``, ``Capture
   Frame Rate``, ...) as flat key/value pairs
2. blank
3. column ``Type`` (``Rigid Body`` / ``Rigid Body Marker`` / ``Marker``)
4. column ``Name`` (e.g. ``Headset``, ``Headset:Marker 001``)
5. column ``ID``
6. column ``Parent``
7. column component (``Rotation`` / ``Position``)
8. column axis (``X`` / ``Y`` / ``Z`` / ``W``); columns 0-1 are instead
   ``Frame`` and ``Time (Seconds)``

Only rigid-body ``Rotation``/``Position`` columns are loaded — marker columns
are skipped, since nothing here needs the raw marker cloud.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_HEADER_ROWS = 8


@dataclass
class RigidBodyTrack:
    """One rigid body's pose across all frames of a take."""

    rotation_xyzw: np.ndarray  # (n_frames, 4), quaternion
    position: np.ndarray  # (n_frames, 3), millimeters


@dataclass
class PositionTrack:
    """One rigid body's position, read without parsing the rest of the take.

    What the brain-state movement veto needs. :func:`load_optitrack_csv`
    returns the full take instead, rotation and all, for the head-direction
    analysis.
    """

    name: str
    position: np.ndarray  # (n_frames, 3), millimetres
    frame_numbers: np.ndarray  # (n_frames,)
    frame_rate: float
    metadata: dict[str, str] = field(default_factory=dict)

    def __len__(self):
        return len(self.position)


@dataclass
class OptitrackTake:
    """A parsed OptiTrack Motive CSV export."""

    frame_numbers: np.ndarray  # (n_frames,)
    times: np.ndarray  # (n_frames,) seconds, take-relative clock
    frame_rate: float
    rigid_bodies: dict[str, RigidBodyTrack] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


def load_optitrack_csv(csv_path: str | Path) -> OptitrackTake:
    """Load an OptiTrack Motive CSV export's rigid-body rotation/position data."""
    csv_path = Path(csv_path)

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header_rows = [next(reader) for _ in range(_HEADER_ROWS)]

    metadata_fields = header_rows[0]
    metadata = dict(zip(metadata_fields[0::2], metadata_fields[1::2]))
    frame_rate = float(metadata["Capture Frame Rate"])

    type_row, name_row = header_rows[2], header_rows[3]
    component_row, axis_row = header_rows[6], header_rows[7]

    # Map each rigid body name -> {(component, axis): column_index}, e.g.
    # {"Headset": {("Rotation", "X"): 2, ..., ("Position", "Z"): 8}}
    rigid_body_cols: dict[str, dict[tuple[str, str], int]] = {}
    for col_idx in range(2, len(type_row)):
        if type_row[col_idx] != "Rigid Body":
            continue
        name = name_row[col_idx]
        key = (component_row[col_idx], axis_row[col_idx])
        rigid_body_cols.setdefault(name, {})[key] = col_idx

    needed_cols = [0, 1]
    for cols in rigid_body_cols.values():
        needed_cols.extend(cols.values())
    needed_cols = sorted(set(needed_cols))

    # usecols only, no `names`. Passing integer `names` alongside integer
    # `usecols` makes pandas treat usecols as matching the names rather than
    # column positions: it then reads the FIRST len(names) columns and labels
    # them with the wanted indices, silently returning the wrong data whenever
    # the wanted columns are not the leading ones. That is invisible for a
    # single rigid body -- its columns are the leading ones -- and returns
    # another body's track as soon as there are two.
    data = pd.read_csv(
        csv_path, skiprows=_HEADER_ROWS, header=None, usecols=needed_cols
    )

    rigid_bodies = {}
    for name, cols in rigid_body_cols.items():
        rotation_xyzw = np.column_stack(
            [data[cols[("Rotation", axis)]].to_numpy() for axis in "XYZW"]
        )
        position = np.column_stack(
            [data[cols[("Position", axis)]].to_numpy() for axis in "XYZ"]
        )
        rigid_bodies[name] = RigidBodyTrack(rotation_xyzw=rotation_xyzw, position=position)

    return OptitrackTake(
        frame_numbers=data[0].to_numpy(),
        times=data[1].to_numpy(),
        frame_rate=frame_rate,
        rigid_bodies=rigid_bodies,
        metadata=metadata,
    )


def _read_header(csv_path: Path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            return [next(reader) for _ in range(_HEADER_ROWS)]
        except StopIteration as e:
            raise ValueError(
                f"{csv_path} has fewer than {_HEADER_ROWS} header rows; this "
                "does not look like a Motive CSV export."
            ) from e


def rigid_body_names(csv_path) -> list[str]:
    """Names of the rigid bodies in a Motive export, in column order."""
    rows = _read_header(Path(csv_path))
    type_row, name_row = rows[2], rows[3]
    names = []
    for col in range(2, min(len(type_row), len(name_row))):
        if type_row[col] == "Rigid Body" and name_row[col] not in names:
            names.append(name_row[col])
    return names


def read_rigid_body_track(csv_path, rigid_body: str | None = None) -> PositionTrack:
    """Load one rigid body's per-frame position from a Motive CSV export.

    `rigid_body` may be left None when the take has exactly one; with several
    it must be named, since picking one arbitrarily would silently track the
    wrong object.
    """
    csv_path = Path(csv_path)
    rows = _read_header(csv_path)

    metadata_fields = rows[0]
    metadata = dict(zip(metadata_fields[0::2], metadata_fields[1::2]))
    if "Capture Frame Rate" not in metadata:
        raise ValueError(
            f"{csv_path.name} header has no 'Capture Frame Rate'; this does "
            "not look like a Motive CSV export."
        )
    frame_rate = float(metadata["Capture Frame Rate"])

    type_row, name_row = rows[2], rows[3]
    component_row, axis_row = rows[6], rows[7]

    # {body name: {axis: column index}} for Position columns only
    columns: dict[str, dict[str, int]] = {}
    width = min(len(type_row), len(name_row), len(component_row), len(axis_row))
    for col in range(2, width):
        if type_row[col] != "Rigid Body" or component_row[col] != "Position":
            continue
        columns.setdefault(name_row[col], {})[axis_row[col]] = col

    if not columns:
        raise ValueError(f"No rigid-body Position columns found in {csv_path.name}.")

    if rigid_body is None:
        if len(columns) > 1:
            raise ValueError(
                f"{csv_path.name} has {len(columns)} rigid bodies "
                f"({sorted(columns)}); name one via state_scoring.movement."
                "rigid_body."
            )
        rigid_body = next(iter(columns))
    elif rigid_body not in columns:
        raise ValueError(
            f"Rigid body {rigid_body!r} not in {csv_path.name}; "
            f"available: {sorted(columns)}"
        )

    axes = columns[rigid_body]
    missing = {"X", "Y", "Z"} - set(axes)
    if missing:
        raise ValueError(
            f"Rigid body {rigid_body!r} is missing position axes "
            f"{sorted(missing)} in {csv_path.name}."
        )

    # usecols only, no `names`. Passing integer `names` alongside integer
    # `usecols` makes pandas treat usecols as matching the names rather than
    # column positions: it then reads the FIRST len(names) columns and labels
    # them with the wanted indices, silently returning the wrong data whenever
    # the wanted columns are not the leading ones. With usecols alone the
    # frame keeps the original column positions as its labels.
    wanted = sorted({0, *(axes[a] for a in "XYZ")})
    data = pd.read_csv(csv_path, skiprows=_HEADER_ROWS, header=None, usecols=wanted)
    return PositionTrack(
        name=rigid_body,
        position=np.column_stack([data[axes[a]].to_numpy(dtype=float) for a in "XYZ"]),
        frame_numbers=data[0].to_numpy(),
        frame_rate=frame_rate,
        metadata=metadata,
    )
