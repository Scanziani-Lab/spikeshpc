"""Reading raw recordings, and reading the concatenated binary back."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import spikeinterface.full as si

from .config import CHANMAP_NAME, CONCAT_INFO_NAME, PROBE_NAME


def detect_phys_type(phys_path: Path) -> str:
    """Return 'spikeglx' or 'openephysbinary' by sniffing acquisition sidecars.

    `phys_path` may be a session/run directory or a single binary file --
    in the latter case its parent directory is searched.
    """
    folder = phys_path.parent if phys_path.is_file() else phys_path
    # SpikeGLX writes a .meta ini file alongside every .bin.
    if any(folder.rglob("*.meta")):
        return "spikeglx"
    # OpenEphys binary format writes structure.oebin per recording, and every
    # Record Node writes settings.xml.
    if any(folder.rglob("structure.oebin")) or any(folder.rglob("settings.xml")):
        return "openephysbinary"
    raise ValueError(
        f"Could not tell whether {folder} is SpikeGLX or OpenEphys "
        "(found no *.meta, structure.oebin or settings.xml). "
        "Pass --phys_type explicitly."
    )


def _stream_candidates(stream_names, phys_type: str, band: str):
    """Streams matching the requested band, per acquisition system."""
    if phys_type == "spikeglx":
        # 'imec0.ap' alongside 'imec0.lf', 'imec0.ap-SYNC' and 'nidq'. Analog
        # inputs land on the NI card rather than the probe.
        if band == "adc":
            return [s for s in stream_names if s.lower().startswith("nidq")]
        suffix = ".ap" if band == "ap" else ".lf"
        return [s for s in stream_names if s.lower().endswith(suffix)]

    if band == "adc":
        # OpenEphys: '...ProbeA-ADC' on a OneBox, or a separate DAQ stream.
        return [
            s
            for s in stream_names
            if re.search(r"[-_.]adc$", s, flags=re.IGNORECASE)
        ]

    # OpenEphys names the AP band after the probe -- e.g.
    # 'Record Node 101#Neuropix-PXI-100.ProbeA' -- with the ADC and LFP bands
    # as sibling streams ('...ProbeA-ADC', '...ProbeA-LFP') and the DAQ on an
    # unrelated name ('...#NI-DAQmx-103.PXIe-6341').
    probe_streams = [s for s in stream_names if "probe" in s.lower()]
    band_suffix = re.compile(r"[-_.](adc|lfp|lf)$", flags=re.IGNORECASE)
    if band == "ap":
        return [s for s in probe_streams if not band_suffix.search(s)]
    return [
        s
        for s in probe_streams
        if re.search(r"[-_.](lfp|lf)$", s, flags=re.IGNORECASE)
    ]


def infer_stream_name(
    phys_path: Path, folder: Path, phys_type: str, band: str = "ap"
) -> str | None:
    """Pick the raw phys stream to load. Returns None if the band is absent.

    `band` is 'ap' (wideband/spikes) or 'lf' (LFP). Only the AP band is
    guaranteed to exist -- Neuropixels 2.0 has no separate LF stream.
    """
    # A SpikeGLX binary names its own stream: <run>_g0_t0.imec0.ap.bin
    if phys_path.is_file() and phys_type == "spikeglx" and band == "ap":
        parts = phys_path.name.split(".")
        if len(parts) >= 3:
            return ".".join(parts[-3:-1])  # e.g. 'imec0.ap'

    from spikeinterface.extractors import get_neo_streams

    stream_names, _ = get_neo_streams(phys_type, folder)
    matches = _stream_candidates(stream_names, phys_type, band)

    if not matches:
        if band != "ap":
            return None  # caller falls back to deriving LFP from the AP band
        raise ValueError(
            f"Could not pick an AP stream for {phys_type} from {stream_names} "
            f"in {folder}. Pass --stream_name explicitly."
        )
    if len(matches) > 1:
        # e.g. a dual-probe recording -- which one to sort is the user's call.
        raise ValueError(
            f"Multiple {band.upper()} streams found in {folder}: {matches}. "
            "Pass --stream_name."
        )
    return matches[0]


def read_recording(
    phys_path,
    phys_type: str | None = None,
    stream_name: str | None = None,
    band: str = "ap",
):
    """Load one recording, auto-detecting acquisition system and stream.

    Accepts either a run/session directory or a path to a raw binary file.
    Returns (recording, phys_type, stream_name); recording is None when
    `band` is absent from this dataset.
    """
    phys_path = Path(phys_path).resolve()
    if not phys_path.exists():
        raise FileNotFoundError(phys_path)

    phys_type = phys_type or detect_phys_type(phys_path)
    folder = phys_path.parent if phys_path.is_file() else phys_path
    stream_name = stream_name or infer_stream_name(phys_path, folder, phys_type, band)
    if stream_name is None:
        return None, phys_type, None

    if phys_type == "spikeglx":
        rec = si.read_spikeglx(folder_path=folder, stream_name=stream_name)
    elif phys_type == "openephysbinary":
        rec = si.read_openephys(folder_path=folder, stream_name=stream_name)
    else:
        raise ValueError(f"Unsupported phys_type: {phys_type!r}")

    return rec, phys_type, stream_name


def write_channel_map(rec, output_dir: Path, channel_rows=None) -> Path:
    """Write chanMap.mat in the format kilosort expects.

    With `channel_rows=None` this reuses spikeinterface's own writer (the one
    si.run_sorter uses for the kilosort family), which assumes the binary holds
    exactly this recording's channels in order.

    When sorting a source binary in place, the file can hold rows we do not
    want -- SpikeGLX keeps SY0 as a 385th row -- so `channel_rows` gives each
    channel's row index in the file. kilosort reads `n_chan_bin` rows per
    sample and then keeps `chanMap`, so this is how the extra rows get dropped.
    """
    if channel_rows is None:
        from spikeinterface.sorters.external.kilosortbase import KilosortBase

        KilosortBase._generate_channel_map_file(rec, output_dir)
        return output_dir / CHANMAP_NAME

    import scipy.io

    rows = np.asarray(channel_rows, dtype=np.int64)
    positions = channel_positions(rec)
    if len(rows) != len(positions):
        raise ValueError(
            f"channel_rows has {len(rows)} entries for "
            f"{len(positions)} channels."
        )
    scipy.io.savemat(
        str(output_dir / CHANMAP_NAME),
        {
            "Nchannels": len(rows),
            "connected": np.full((len(rows), 1), True),
            "chanMap0ind": rows,
            "chanMap": rows + 1,  # kilosort.io.load_probe subtracts 1
            "xcoords": positions[:, 0].astype(float),
            "ycoords": positions[:, 1].astype(float),
            "kcoords": np.ones(len(rows), dtype=float),
            "fs": float(rec.get_sampling_frequency()),
        },
    )
    return output_dir / CHANMAP_NAME


def _spikeglx_source_binary(folder: Path, stream_name: str):
    matches = sorted(folder.rglob(f"*.{stream_name}.bin"))
    return matches[0] if len(matches) == 1 else None


def _openephys_source_binary(folder: Path, stream_name: str):
    # 'Record Node 101#Neuropix-PXI-100.ProbeA' lives in
    # .../continuous/Neuropix-PXI-100.ProbeA/continuous.dat
    leaf = stream_name.split("#")[-1]
    matches = [
        p for p in folder.rglob("continuous.dat") if p.parent.name == leaf
    ]
    return matches[0] if len(matches) == 1 else None


def locate_source_binary(phys_path: Path, phys_type: str, stream_name: str):
    """Path to the flat binary behind a stream, or None if it is not obvious."""
    phys_path = Path(phys_path)
    if phys_path.is_file() and phys_path.suffix in (".bin", ".dat", ".raw"):
        return phys_path
    folder = phys_path.parent if phys_path.is_file() else phys_path
    if phys_type == "spikeglx":
        return _spikeglx_source_binary(folder, stream_name)
    if phys_type == "openephysbinary":
        return _openephys_source_binary(folder, stream_name)
    return None


def check_source_binary(rec, path: Path, dtype="int16", n_check_samples=30000):
    """Can `rec` be sorted straight out of `path`? Returns (n_file_channels, rows).

    Returns None when it cannot, which is not a failure -- the caller just
    writes a fresh binary instead. Correctness here matters more than the time
    saved, so the layout is not assumed: traces read directly from the file are
    compared against spikeinterface's own for the same samples, at the start,
    middle and end. Anything less than an exact match declines the shortcut.
    """
    path = Path(path)
    if not path.exists():
        return None

    itemsize = np.dtype(dtype).itemsize
    n_samples = rec.get_num_frames()
    n_rec = rec.get_num_channels()
    total = path.stat().st_size
    if total % (itemsize * n_samples) != 0:
        return None
    n_file = total // (itemsize * n_samples)
    if n_file < n_rec:
        return None

    # SpikeGLX and OpenEphys both store the stream's channels first and in
    # order; the only extra is SpikeGLX's trailing sync row.
    rows = np.arange(n_rec, dtype=np.int64)
    mm = np.memmap(path, dtype=dtype, mode="r", shape=(int(n_samples), int(n_file)))
    try:
        starts = [0, max(n_samples // 2, 0), max(n_samples - n_check_samples, 0)]
        for start in starts:
            stop = min(start + n_check_samples, n_samples)
            if stop <= start:
                continue
            from_file = np.asarray(mm[start:stop][:, rows])
            from_si = rec.get_traces(
                start_frame=start, end_frame=stop, return_in_uV=False
            )
            if not np.array_equal(from_file, np.asarray(from_si)):
                return None
    finally:
        del mm

    return int(n_file), rows


def load_concatenated(output_dir: Path):
    """Re-open the binary written by preprocess(), probe and gains restored.

    This is what makes --skip_preprocessing work: the sorting and
    post-processing stages read the recording back from disk rather than
    re-deriving it from the raw session folders.
    """
    import probeinterface

    info_path = output_dir / CONCAT_INFO_NAME
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found -- run the pre-processing stage first "
            "(drop --skip_preprocessing) or point --output_dir at a completed run."
        )
    info = json.loads(info_path.read_text())

    # binary_path is absolute when the source recording is being sorted in
    # place; binary_file is the name inside output_dir otherwise.
    path = Path(info.get("binary_path") or (output_dir / info["binary_file"]))
    rows = info.get("channel_rows")
    n_file = info.get("file_num_channels", info["num_channels"])

    if rows is None:
        rec = si.read_binary(
            file_paths=path,
            sampling_frequency=info["sampling_frequency"],
            dtype=info["dtype"],
            num_channels=info["num_channels"],
            channel_ids=info["channel_ids"],
            gain_to_uV=info["gain_to_uV"],
            offset_to_uV=info["offset_to_uV"],
            # .get for concat_info.json written before this key existed.
            is_filtered=info.get("is_filtered"),
        )
    else:
        # The file carries rows we do not sort (SpikeGLX's SY0), so open it at
        # its true width and slice. Gains are attached after the slice, since
        # they are per kept channel rather than per file row.
        full = si.read_binary(
            file_paths=path,
            sampling_frequency=info["sampling_frequency"],
            dtype=info["dtype"],
            num_channels=n_file,
            channel_ids=[str(i) for i in range(n_file)],
            is_filtered=info.get("is_filtered"),
        )
        rec = si.ChannelSliceRecording(
            full,
            channel_ids=[str(r) for r in rows],
            renamed_channel_ids=info["channel_ids"],
        )
        if info.get("gain_to_uV") is not None:
            rec.set_property("gain_to_uV", np.asarray(info["gain_to_uV"]))
        if info.get("offset_to_uV") is not None:
            rec.set_property("offset_to_uV", np.asarray(info["offset_to_uV"]))

    rec = rec.set_probegroup(probeinterface.read_probeinterface(output_dir / PROBE_NAME))
    return rec, info


def channel_positions(rec):
    """(num_channels, 2) array of contact positions, in probe coordinates."""
    loc = np.asarray(rec.get_channel_locations(), dtype=float)
    if loc.ndim != 2 or loc.shape[1] != 2:
        raise ValueError(
            f"Expected 2D channel locations, got shape {loc.shape}. "
            "3D probe geometries are not supported."
        )
    return loc


# ─────────────────────────────────────────────────────────────────────────
# Brain-state scoring results
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class StateScoring:
    """One session's scored brain states, as written by stage 1.

    `codes` and the per-bin signals share the `times` grid (bin centres, in
    seconds on that recording's own clock). `intervals` holds the same
    information as contiguous [start, stop] spans per state name.
    """

    session: str
    times: np.ndarray
    codes: np.ndarray
    broadband: np.ndarray | None = None
    theta: np.ndarray | None = None
    emg: np.ndarray | None = None
    speed: np.ndarray | None = None
    codes_before_veto: np.ndarray | None = None
    intervals: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    fractions: dict = field(default_factory=dict)
    state_codes: dict = field(default_factory=dict)
    step_s: float = 1.0
    metadata: dict = field(default_factory=dict)
    source: Path | None = None

    @property
    def names(self) -> np.ndarray:
        """State name per bin, e.g. array(['WAKE', 'WAKE', 'NREM', ...])."""
        lookup = {v: k for k, v in self.state_codes.items()}
        return np.array([lookup.get(int(c), "?") for c in self.codes])

    def signals(self) -> dict:
        """The per-bin traces that are actually present, in display order."""
        wanted = (
            ("broadband", "broadband LFP (PC1)"),
            ("theta", "theta ratio"),
            ("emg", "EMG (LFP correlation)"),
            ("speed", "movement (mm/s)"),
        )
        return {
            label: getattr(self, name)
            for name, label in wanted
            if getattr(self, name) is not None
        }

    def epochs(self, states=None) -> list:
        """Contiguous runs of one state: (state, start_s, stop_s, i0, i1).

        Built from the per-bin codes rather than from `intervals`, so each
        epoch carries the bin indices a plot needs.
        """
        # imported here: states.py reads recordings through this module,
        # so a module-level import would be circular
        from .states import state_epochs

        return state_epochs(
            self.codes, self.times, self.state_codes, self.step_s, states
        )

    @property
    def vetoed(self) -> np.ndarray:
        """Bins the movement veto reassigned, if the pre-veto codes were saved."""
        if self.codes_before_veto is None:
            return np.zeros(len(self.codes), dtype=bool)
        return np.asarray(self.codes_before_veto) != np.asarray(self.codes)

    def vetoed_epochs(self) -> list:
        """Spans the veto overruled, labelled by what the LFP alone called.

        These no longer carry their original label -- a REM call rejected for
        movement is WAKE now -- so stepping through the scored epochs will
        never show them. This is how to review the veto's own decisions.
        """
        from .states import state_epochs

        if self.codes_before_veto is None:
            raise ValueError(
                f"{self.session!r} has no codes_before_veto saved, so the "
                "veto's changes cannot be recovered. Re-run state scoring to "
                "record them."
            )
        changed = self.vetoed
        if not changed.any():
            return []
        # label each span by its ORIGINAL state, and keep only changed bins
        original = np.where(changed, self.codes_before_veto, -1)
        return [
            e
            for e in state_epochs(
                original, self.times, self.state_codes, self.step_s
            )
            if e.state != "?"
        ]

    def __repr__(self):
        parts = ", ".join(f"{k}={v:.1%}" for k, v in sorted(self.fractions.items()))
        return (
            f"StateScoring({self.session!r}, {len(self.times)} bins of "
            f"{self.step_s:g}s, {parts})"
        )


def load_states(states_path, session: str | None = None,
                optitrack_csv=None, frame_times=None,
                rigid_body=None) -> StateScoring:
    """Load one session's scoring from a states/ directory or a _states.json.

    With several sessions in the directory, name one -- returning an arbitrary
    session's states would be a quiet way to analyse the wrong recording.

    Pass `optitrack_csv` to attach the movement trace even when the scoring
    was run without the veto, so it can always be plotted alongside the LFP
    signals. See :func:`spikeshpc.states.attach_movement`.
    """
    states_path = Path(states_path)

    if states_path.is_file():
        json_path = states_path
    else:
        candidates = sorted(states_path.glob("*_states.json"))
        if session is not None:
            json_path = states_path / f"{session}_states.json"
            if not json_path.exists():
                found = [p.name[: -len("_states.json")] for p in candidates]
                raise FileNotFoundError(
                    f"No scoring for session {session!r} in {states_path}; "
                    f"found: {found}"
                )
        elif not candidates:
            raise FileNotFoundError(f"No *_states.json in {states_path}")
        elif len(candidates) > 1:
            found = [p.name[: -len("_states.json")] for p in candidates]
            raise ValueError(
                f"{states_path} holds {len(candidates)} sessions ({found}); "
                "pass session= to choose one."
            )
        else:
            json_path = candidates[0]

    summary = json.loads(json_path.read_text())
    name = summary.get("session") or json_path.name[: -len("_states.json")]

    metrics_path = json_path.with_name(f"{name}_metrics.npz")
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found; the per-bin signals live there and "
            f"{json_path.name} only holds the summary."
        )
    with np.load(metrics_path) as m:
        arrays = {k: m[k] for k in m.files}

    scoring = StateScoring(
        session=name,
        times=arrays["times"],
        codes=arrays["codes"],
        codes_before_veto=arrays.get("codes_before_veto"),
        broadband=arrays.get("broadband"),
        theta=arrays.get("theta"),
        emg=arrays.get("emg"),
        speed=arrays.get("speed"),
        intervals=summary.get("intervals", {}),
        thresholds=summary.get("thresholds", {}),
        fractions=summary.get("fractions", {}),
        state_codes=summary.get("state_codes", {"WAKE": 1, "NREM": 3, "REM": 5}),
        step_s=float(summary.get("step_s", 1.0)),
        metadata=summary,
        source=json_path,
    )

    if optitrack_csv is not None:
        from .states import attach_movement

        attach_movement(scoring, optitrack_csv, frame_times, rigid_body)
    return scoring
