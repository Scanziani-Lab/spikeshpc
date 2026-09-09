"""Reading raw recordings, and reading the concatenated binary back."""

import json
import re
import warnings
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
            s for s in stream_names if re.search(r"[-_.]adc$", s, flags=re.IGNORECASE)
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
        s for s in probe_streams if re.search(r"[-_.](lfp|lf)$", s, flags=re.IGNORECASE)
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


def sync_clock_problem(rec, sample: int = 100_000) -> str | None:
    """Why this recording's time vector cannot be trusted, or None if it can.

    Open Ephys writes ``timestamps.npy`` per stream, but a stream that was
    never synchronized to the software clock gets a file of ``-1.0`` rather
    than no file at all. spikeinterface hands those straight back, so
    ``load_sync_timestamps=True`` on an unsynced stream silently timestamps
    every sample at -1 second. Checked here rather than trusted, because the
    failure is invisible downstream: shutter edges all land at -1, no analysis
    errors, and every spike is credited to the wrong heading.

    The endpoints and a stride through the middle are enough -- a time vector
    can be tens of gigabytes, and a partial monotonicity check that costs
    nothing is worth more than a total one nobody runs.
    """
    for segment in range(rec.get_num_segments()):
        times = rec.get_times(segment_index=segment)
        if len(times) == 0:
            continue
        first, last = float(times[0]), float(times[-1])
        if not (np.isfinite(first) and np.isfinite(last)):
            return f"segment {segment} has non-finite timestamps"
        if first < 0:
            return (
                f"segment {segment} starts at t={first:g}s -- this stream was "
                "never synchronized, so Open Ephys filled timestamps.npy with -1"
            )
        if last <= first:
            return f"segment {segment} does not advance ({first:g}s to {last:g}s)"
        step = max(1, len(times) // sample)
        if not np.all(np.diff(times[::step]) > 0):
            return f"segment {segment} timestamps are not increasing"
    return None


def read_openephys_synced(folder, stream_name, require_sync: bool = False):
    """Open Ephys on the acquisition clock, not on one inferred from the rate.

    Every Open Ephys read in this package goes through here, because the two
    clocks are not interchangeable and the difference is not small. A OneBox
    ADC stream declares 30300.5 Hz in structure.oebin and actually runs at
    about 30303 Hz: 83 ppm, which is two full seconds over a six-hour session,
    or a hundred-odd camera frames of error by the end of it. Timestamps
    inferred from the declared rate also ignore that each stream starts at its
    own offset on the shared clock, so the ADC and the probe drift apart from
    the first sample.

    Where the stream carries no sync (see :func:`sync_clock_problem`) this
    falls back to the inferred clock and says so loudly -- an unsynced session
    is still worth scoring, but nobody should discover afterwards that its
    camera alignment was the version with the drift in it. Pass
    ``require_sync=True`` to refuse instead.
    """
    rec = si.read_openephys(
        folder_path=folder, stream_name=stream_name, load_sync_timestamps=True
    )
    problem = sync_clock_problem(rec)
    if problem is None:
        return rec

    message = (
        f"{stream_name!r} has no usable synchronized clock: {problem}. "
        "Falling back to timestamps inferred from the declared sampling rate, "
        "which drift against the other streams -- times from this stream "
        "should not be compared with times from a synced one."
    )
    if require_sync:
        raise ValueError(message.replace("Falling back to", "Refusing to use"))
    warnings.warn(message, stacklevel=2)
    return si.read_openephys(
        folder_path=folder, stream_name=stream_name, load_sync_timestamps=False
    )


def open_stream(folder, phys_type: str, stream_name: str):
    """Open one named stream, on the acquisition clock where there is one.

    The single place either acquisition system is handed to spikeinterface, so
    that "always use the synchronized timestamps" is a property of the package
    rather than a habit each call site has to remember.
    """
    if phys_type == "spikeglx":
        return si.read_spikeglx(folder_path=folder, stream_name=stream_name)
    if phys_type == "openephysbinary":
        return read_openephys_synced(folder, stream_name)
    raise ValueError(f"Unsupported phys_type: {phys_type!r}")


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

    return open_stream(folder, phys_type, stream_name), phys_type, stream_name


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
            f"channel_rows has {len(rows)} entries for {len(positions)} channels."
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
    matches = [p for p in folder.rglob("continuous.dat") if p.parent.name == leaf]
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


def _first_timestamp(sidecar: Path):
    """The first entry of an Open Ephys timestamps.npy, if it is a real time.

    Memory-mapped: these files run to gigabytes and one element is wanted.
    """
    if not sidecar.exists():
        return None
    first = float(np.load(sidecar, mmap_mode="r")[0])
    return None if not np.isfinite(first) or first < 0 else first


def probe_start_time(phys_path, phys_type: str):
    """When the sorted stream's first sample happened, on the acquisition clock.

    This is the number that converts between the two clocks the pipeline uses.
    Spike times out of kilosort, and the state-scoring bin grid, both count
    from the sorted stream's first sample -- zero. Anything read through
    :func:`read_openephys_synced` instead counts from the acquisition system's
    own epoch, which is several seconds earlier. Subtract this from one to get
    the other.

    Found by walking the recording's own layout rather than by asking
    spikeinterface to enumerate streams: that opens the whole dataset through
    neo, which is a great deal of work to read one float, and it fails on a
    tree that has the continuous data but not every sidecar. Returns None when
    there is no synchronized clock to speak of, so callers can leave times
    where they are instead of shifting them by a guess.
    """
    if phys_type != "openephysbinary":
        return None

    phys_path = Path(phys_path)
    folder = phys_path.parent if phys_path.is_file() else phys_path
    streams = {
        p.parent.name: p.parent
        for p in folder.rglob("continuous.dat")
        if p.parent.parent.name == "continuous"
    }
    probes = _stream_candidates(list(streams), phys_type, band="ap")
    if len(probes) != 1:
        return None
    return _first_timestamp(streams[probes[0]] / "timestamps.npy")


def stream_start_time(phys_path, phys_type: str, stream_name: str):
    """When one named stream's first sample happened, on the acquisition clock.

    See :func:`probe_start_time`, which is the same question asked of whichever
    stream was sorted.
    """
    binary = locate_source_binary(Path(phys_path), phys_type, stream_name)
    if binary is None or phys_type != "openephysbinary":
        return None
    return _first_timestamp(binary.parent / "timestamps.npy")


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

    rec = rec.set_probegroup(
        probeinterface.read_probeinterface(output_dir / PROBE_NAME)
    )
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
            for e in state_epochs(original, self.times, self.state_codes, self.step_s)
            if e.state != "?"
        ]

    def __repr__(self):
        parts = ", ".join(f"{k}={v:.1%}" for k, v in sorted(self.fractions.items()))
        return (
            f"StateScoring({self.session!r}, {len(self.times)} bins of "
            f"{self.step_s:g}s, {parts})"
        )


def load_states(
    states_path,
    session: str | None = None,
    optitrack_csv=None,
    frame_times=None,
    rigid_body=None,
) -> StateScoring:
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
