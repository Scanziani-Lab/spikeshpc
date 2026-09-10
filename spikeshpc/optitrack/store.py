"""Saving head-direction tuning so the next notebook does not recompute it.

The shuffle test is the expensive part -- hundreds of re-binned curves per
unit -- and its result is a handful of numbers per unit plus one curve. Nothing
about it should have to be redone to run a decoder, so this writes the whole
lot out and reads it back.

Two files, because the payload is two different kinds of thing:

  <name>.npz    the arrays: heading per camera frame, one curve per unit, and
                the per-unit statistics as parallel columns. Compressed
                binary, because heading_deg alone is a million-odd floats and
                JSON would store it as several megabytes of decimal text,
                slowly, and with rounding nobody asked for.
  <name>.json   the same statistics as a readable table, plus the parameters
                the run used. Nothing reads this back -- it is there so the
                choices behind a saved result can be checked months later
                without loading anything.

``load_hd_tuning`` reads the npz alone, so the JSON can be edited or deleted
without breaking anything downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .tuning import HDTuningStats

__all__ = ["HDTuning", "save_hd_tuning", "load_hd_tuning"]

# the per-unit statistic columns, in the order they are stored
_STAT_FIELDS = (
    "mean_vector_length",
    "preferred_direction_deg",
    "peak_rate_hz",
    "mean_rate_hz",
    "n_spikes",
    "p_value",
    "mvl_threshold",
    "significant",
    "too_quiet",
)


@dataclass
class HDTuning:
    """One session's head-direction tuning, as saved by the tuning notebook.

    ``unit_ids`` orders every per-unit array here. ``tuned_ids`` is the subset
    that passed the significance test, which is what a decoder should be given.
    """

    session: str
    heading_deg: np.ndarray  # one per shutter-closure event
    bin_centers_deg: np.ndarray
    unit_ids: np.ndarray
    curves: np.ndarray  # (n_units, n_bins), rate in Hz
    depths: np.ndarray  # (n_units,), probe depth
    stats: dict = field(default_factory=dict)  # unit_id -> HDTuningStats
    parameters: dict = field(default_factory=dict)

    @property
    def tuned_ids(self) -> np.ndarray:
        """The significantly tuned units, in `unit_ids` order."""
        return np.array(
            [u for u in self.unit_ids if self.stats[u].significant],
            dtype=self.unit_ids.dtype,
        )

    def curve(self, unit_id) -> tuple[np.ndarray, np.ndarray]:
        """``(bin_centers_deg, rate_hz)`` for one unit, as the widgets want it."""
        index = int(np.flatnonzero(self.unit_ids == unit_id)[0])
        return self.bin_centers_deg, self.curves[index]

    def curve_dict(self, unit_ids=None) -> dict:
        """``{unit_id: (bin_centers, rate)}``, for the tuning-curve widget."""
        wanted = self.unit_ids if unit_ids is None else unit_ids
        return {u: self.curve(u) for u in wanted}

    def depth_dict(self, unit_ids=None) -> dict:
        wanted = self.unit_ids if unit_ids is None else unit_ids
        lookup = dict(zip(self.unit_ids, self.depths))
        return {u: lookup[u] for u in wanted}

    def __repr__(self) -> str:
        return (
            f"HDTuning({self.session!r}, {len(self.unit_ids)} units, "
            f"{len(self.tuned_ids)} tuned, {len(self.heading_deg)} frames)"
        )


def save_hd_tuning(
    path,
    session: str,
    heading_deg,
    tuning_curves: dict,
    stats: dict,
    unit_depths: dict,
    parameters: dict | None = None,
) -> Path:
    """Write tuning curves, statistics and heading to ``<path>.npz`` + ``.json``.

    ``tuning_curves``, ``stats`` and ``unit_depths`` are the dicts the tuning
    functions return, keyed by unit id. Units missing from any of them are
    dropped rather than padded, so what comes back is exactly the set that has
    a curve, a statistic and a depth.
    """
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    unit_ids = [u for u in tuning_curves if u in stats and u in unit_depths]
    if not unit_ids:
        raise ValueError(
            "No unit has a curve, a statistic and a depth all three; nothing "
            "to save. Check that the same unit_ids were used throughout."
        )
    dropped = len(tuning_curves) - len(unit_ids)

    bin_centers = np.asarray(tuning_curves[unit_ids[0]][0], dtype=float)
    curves = np.vstack([np.asarray(tuning_curves[u][1], dtype=float) for u in unit_ids])

    columns = {
        name: np.array([getattr(stats[u], name) for u in unit_ids]) for name in _STAT_FIELDS
    }

    # the shuffled envelope, if the significance test kept it. Stored only when
    # every unit has one of the same shape -- a ragged stack would have to be
    # an object array, and that means allow_pickle on the way back in.
    bands = [getattr(stats[u], "null_band", None) for u in unit_ids]
    extra = {}
    if all(b is not None for b in bands) and len({np.shape(b) for b in bands}) == 1:
        first = stats[unit_ids[0]]
        extra = {
            "null_band": np.stack([np.asarray(b, dtype=float) for b in bands]),
            "null_bin_centers_deg": np.asarray(
                first.null_bin_centers_deg, dtype=float
            ),
            "null_percentiles": np.asarray(first.null_percentiles, dtype=float),
        }

    np.savez_compressed(
        path,
        **extra,
        session=np.array(session),
        heading_deg=np.asarray(heading_deg, dtype=np.float32),
        bin_centers_deg=bin_centers,
        # in their own dtype, not stringified: these have to come back able to
        # index a sorting, and `sorting.get_unit_spike_train("7")` is not
        # `sorting.get_unit_spike_train(7)`
        unit_ids=np.array(unit_ids),
        curves=curves,
        depths=np.array([float(unit_depths[u]) for u in unit_ids]),
        parameters=np.array(json.dumps(parameters or {})),
        **columns,
    )

    readable = {
        "session": session,
        "n_units": len(unit_ids),
        "n_tuned": int(columns["significant"].sum()),
        "n_frames": int(len(heading_deg)),
        "parameters": parameters or {},
        "units": {
            str(u): {k: _plain(getattr(stats[u], k)) for k in _STAT_FIELDS}
            for u in unit_ids
        },
    }
    path.with_suffix(".json").write_text(json.dumps(readable, indent=2))

    print(
        f"    saved {len(unit_ids)} units ({readable['n_tuned']} tuned) to "
        f"{path.name} and {path.with_suffix('.json').name}"
        + (f"; dropped {dropped} without a full set" if dropped else "")
    )
    return path


def _plain(value):
    """numpy scalars are not JSON-serializable; their Python twins are."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def load_hd_tuning(path) -> HDTuning:
    """Read back what :func:`save_hd_tuning` wrote."""
    path = Path(path).with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}

    unit_ids = arrays["unit_ids"]
    has_band = "null_band" in arrays
    stats = {}
    for i, unit in enumerate(unit_ids):
        fields = {
            name: _plain(arrays[name][i]) for name in _STAT_FIELDS if name in arrays
        }
        if has_band:
            fields.update(
                null_band=arrays["null_band"][i],
                null_bin_centers_deg=arrays["null_bin_centers_deg"],
                null_percentiles=tuple(arrays["null_percentiles"].tolist()),
            )
        stats[unit] = HDTuningStats(**fields)
    return HDTuning(
        session=str(arrays["session"]),
        heading_deg=arrays["heading_deg"],
        bin_centers_deg=arrays["bin_centers_deg"],
        unit_ids=unit_ids,
        curves=arrays["curves"],
        depths=arrays["depths"],
        stats=stats,
        parameters=json.loads(str(arrays["parameters"])),
    )
