"""Reading rigid-body position from an OptiTrack Motive CSV export.

Only what the brain-state movement veto needs: one rigid body's position per
frame. Rotation, marker clouds and the take's own clock are all skipped -- if
you want those, the `optitrack` package parses the format in full. Vendored
here so that spikeshpc has no dependency on it.

The export has an 8-row header before the data:

  1. take metadata (``Format Version``, ``Total Exported Frames``,
     ``Capture Frame Rate``, ...) as flat key/value pairs
  2. blank
  3. column ``Type``      (``Rigid Body`` / ``Rigid Body Marker`` / ``Marker``)
  4. column ``Name``      (e.g. ``Headset``, ``Headset:Marker 001``)
  5. column ``ID``
  6. column ``Parent``
  7. column component    (``Rotation`` / ``Position``)
  8. column axis         (``X`` / ``Y`` / ``Z`` / ``W``); columns 0-1 are
     instead ``Frame`` and ``Time (Seconds)``
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEADER_ROWS = 8


@dataclass
class RigidBodyTrack:
    """One rigid body's position across a take."""

    name: str
    position: np.ndarray  # (n_frames, 3), millimetres
    frame_numbers: np.ndarray  # (n_frames,)
    frame_rate: float
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.position)


def _read_header(csv_path: Path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            return [next(reader) for _ in range(HEADER_ROWS)]
        except StopIteration as e:
            raise ValueError(
                f"{csv_path} has fewer than {HEADER_ROWS} header rows; this "
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


def read_rigid_body_track(csv_path, rigid_body: str | None = None) -> RigidBodyTrack:
    """Load one rigid body's per-frame position from a Motive CSV export.

    `rigid_body` may be left None when the take has exactly one; with several
    it must be named, since picking one arbitrarily would silently track the
    wrong object.
    """
    import pandas as pd  # ships with spikeinterface[full]

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
    data = pd.read_csv(csv_path, skiprows=HEADER_ROWS, header=None, usecols=wanted)
    return RigidBodyTrack(
        name=rigid_body,
        position=np.column_stack([data[axes[a]].to_numpy(dtype=float) for a in "XYZ"]),
        frame_numbers=data[0].to_numpy(),
        frame_rate=frame_rate,
        metadata=metadata,
    )
