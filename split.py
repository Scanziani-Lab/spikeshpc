"""Split a concatenated run back into its constituent recordings.

Sorting runs on one concatenated binary so that units are tracked across
sessions, but analysis usually wants the sessions apart again. This takes the
concatenated recording, sorting, analyzer and brain-state intervals plus the
sample offsets recorded in concat_info.json, and hands back one object of each
per original recording, all on a common session-local clock:

  * frame 0 of a split recording is frame 0 of that original recording
  * spike frames in a split sorting are session-local (frame_slice shifts them)
  * state intervals are session-local seconds, clipped to the session
  * `timestamps()` gives seconds for every sample, matching all of the above

Unit ids are preserved across every session, so unit 57 is the same unit
everywhere -- that is the point of sorting the concatenation in the first
place. A unit that never fires in a given session survives there as an empty
spike train rather than disappearing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import spikeinterface.full as si

from .config import (
    ANALYZER_NAME,
    CONCAT_INFO_NAME,
    SORTER_DIRNAME,
    STATES_CONCAT_NAME,
    STATES_DIRNAME,
)

SESSIONS_DIRNAME = "sessions"


@dataclass
class SessionSplit:
    """One original recording, carved back out of the concatenated run."""

    index: int
    name: str
    sample_offset: int
    num_samples: int
    sampling_frequency: float
    phys_path: str | None = None
    recording: object | None = None
    sorting: object | None = None
    analyzer: object | None = None
    states: dict | None = None
    metrics: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.num_samples / self.sampling_frequency

    @property
    def t_start(self) -> float:
        """Where this session begins on the concatenated clock, in seconds."""
        return self.sample_offset / self.sampling_frequency

    def timestamps(self, relative_to: str = "session", dtype=np.float64):
        """Seconds for every sample in this session.

        'session' starts at 0 and lines up with the split recording, sorting
        and states. 'concatenated' keeps the original run's clock, for joining
        back against the unsplit objects.

        This materialises num_samples * 8 bytes -- ~2.6 GB for 3 h at 30 kHz.
        Prefer arithmetic on frame indices where you can. float32 is not
        offered on purpose: past a few minutes its 24-bit mantissa cannot
        resolve a sample.
        """
        if relative_to not in ("session", "concatenated"):
            raise ValueError(
                f"relative_to must be 'session' or 'concatenated', got {relative_to!r}"
            )
        dtype = np.dtype(dtype)
        if dtype != np.float64:
            raise ValueError(
                f"timestamps need float64; {dtype} cannot resolve single samples "
                "over a long recording."
            )
        start = self.sample_offset if relative_to == "concatenated" else 0
        return (np.arange(self.num_samples, dtype=dtype) + start) / self.sampling_frequency

    def __repr__(self):
        have = [
            n
            for n in ("recording", "sorting", "analyzer", "states")
            if getattr(self, n) is not None
        ]
        return (
            f"SessionSplit({self.name!r}, {self.duration_s:.1f}s, "
            f"t_start={self.t_start:.1f}s, has={have})"
        )


def session_bounds(info: dict):
    """[(start_frame, stop_frame), ...] for each original recording."""
    offsets = info["sample_offsets"]
    return [(offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]


def session_names(info: dict):
    """Session names taken from the source paths, as state scoring names them."""
    paths = info.get("phys_paths") or []
    names = [Path(p).stem or Path(p).name for p in paths]
    if len(names) != len(info["sample_offsets"]) - 1:
        names = [f"session{i}" for i in range(len(info["sample_offsets"]) - 1)]
    return names


def split_recording(recording, info: dict):
    """Lazy frame_slice views, one per original recording."""
    total = sum(info["num_samples"])
    if recording.get_num_frames() != total:
        raise ValueError(
            f"Recording has {recording.get_num_frames()} frames but "
            f"{CONCAT_INFO_NAME} accounts for {total}. Are they from the same run?"
        )
    return [recording.frame_slice(a, b) for a, b in session_bounds(info)]


def split_sorting(sorting, info: dict):
    """Per-session sortings; frame_slice re-bases spike frames onto each session."""
    return [sorting.frame_slice(a, b) for a, b in session_bounds(info)]


def split_states(states: dict, info: dict):
    """Concatenated-clock state intervals -> per-session, session-local seconds.

    Intervals are clipped to each session's span. Scoring runs per session
    before concatenation, so nothing should straddle a junction; anything that
    does (e.g. states scored on the concatenation itself) is split across the
    sessions it covers rather than being dropped.
    """
    fs = info["sampling_frequency"]
    per_session = []
    for start, stop in session_bounds(info):
        t0, t1 = start / fs, stop / fs
        clipped = {}
        for name, spans in states["intervals"].items():
            kept = []
            for a, b in spans:
                lo, hi = max(float(a), t0), min(float(b), t1)
                if hi > lo:
                    kept.append([lo - t0, hi - t0])
            clipped[name] = kept
        per_session.append(clipped)
    return per_session


def parent_extension_names(analyzer):
    """Extensions an analyzer has, whichever format it is in.

    get_saved_extension_names() only works for zarr/binary_folder analyzers
    and raises on an in-memory one, which is what you get from a notebook.
    """
    names = []
    try:
        names = list(analyzer.get_saved_extension_names())
    except ValueError:
        pass
    for name in analyzer.get_loaded_extension_names():
        if name not in names:
            names.append(name)
    return names


def split_analyzer(
    analyzer,
    recordings,
    sortings,
    extensions=None,
    folder=None,
    format="memory",
    inherit_sparsity=True,
    **job_kwargs,
):
    """Build one analyzer per session from the split recording/sorting pairs.

    SortingAnalyzer has no time slicing, so the extensions are genuinely
    recomputed on each session's own data -- that is the point (a template
    averaged over the whole concatenation tells you nothing about drift
    between sessions), but it costs about as much as the original
    post-processing run.

    `extensions` defaults to the parent's computed extensions with the same
    parameters. Sparsity is inherited so templates are directly comparable
    across sessions and with the parent.
    """
    if extensions is None:
        extensions = {}
        for name in parent_extension_names(analyzer):
            ext = analyzer.get_extension(name)
            extensions[name] = dict(ext.params) if ext is not None else {}

    sparsity = analyzer.sparsity if inherit_sparsity else None
    analyzers = []
    for i, (rec, sorting) in enumerate(zip(recordings, sortings)):
        target = None
        if folder is not None:
            target = Path(folder) / f"{i}" / ANALYZER_NAME
            target.parent.mkdir(parents=True, exist_ok=True)
        sub = si.create_sorting_analyzer(
            recording=rec,
            sorting=sorting,
            folder=target,
            format=format if target is None else "zarr",
            sparsity=sparsity,
            overwrite=target is not None,
        )
        if extensions:
            sub.compute(extensions, **job_kwargs)
        analyzers.append(sub)
    return analyzers


def split_run(
    output_dir,
    recording=None,
    sorting=None,
    analyzer=None,
    states=None,
    info=None,
    with_analyzer: bool = False,
    extensions=None,
    analyzer_format: str = "memory",
    **job_kwargs,
):
    """Split a completed run into one SessionSplit per original recording.

    Anything not passed in is loaded from `output_dir`; pass objects directly
    to split ones you already have in memory. `with_analyzer` is off by
    default because it recomputes every extension per session.

    Returns a list of SessionSplit, in the order the recordings were
    concatenated.
    """
    from .io import load_concatenated

    output_dir = Path(output_dir)

    if info is None or recording is None:
        loaded_rec, loaded_info = load_concatenated(output_dir)
        recording = recording if recording is not None else loaded_rec
        info = info if info is not None else loaded_info

    if sorting is None:
        results_dir = output_dir / SORTER_DIRNAME
        if (results_dir / "spike_times.npy").exists():
            sorting = si.read_kilosort(folder_path=results_dir)
        else:
            print(f"    no sorting in {results_dir}, skipping")

    if analyzer is None and with_analyzer:
        analyzer_path = output_dir / ANALYZER_NAME
        if analyzer_path.exists():
            analyzer = si.load_sorting_analyzer(analyzer_path)
        else:
            print(f"    no analyzer at {analyzer_path}, skipping")

    if states is None:
        states_path = output_dir / STATES_DIRNAME / STATES_CONCAT_NAME
        if states_path.exists():
            states = json.loads(states_path.read_text())
        else:
            print(f"    no {STATES_CONCAT_NAME}, skipping states")

    bounds = session_bounds(info)
    names = session_names(info)
    paths = info.get("phys_paths") or [None] * len(bounds)

    recordings = split_recording(recording, info) if recording is not None else None
    sortings = split_sorting(sorting, info) if sorting is not None else None
    per_states = split_states(states, info) if states is not None else None

    analyzers = None
    if with_analyzer and analyzer is not None:
        if recordings is None or sortings is None:
            raise ValueError("Splitting the analyzer needs both a recording and a sorting.")
        print(f"    recomputing analyzer extensions for {len(bounds)} session(s)...")
        analyzers = split_analyzer(
            analyzer,
            recordings,
            sortings,
            extensions=extensions,
            folder=(output_dir / SESSIONS_DIRNAME if analyzer_format != "memory" else None),
            format=analyzer_format,
            **job_kwargs,
        )

    splits = []
    for i, (start, stop) in enumerate(bounds):
        splits.append(
            SessionSplit(
                index=i,
                name=names[i],
                sample_offset=int(start),
                num_samples=int(stop - start),
                sampling_frequency=float(info["sampling_frequency"]),
                phys_path=paths[i],
                recording=recordings[i] if recordings else None,
                sorting=sortings[i] if sortings else None,
                analyzer=analyzers[i] if analyzers else None,
                states=per_states[i] if per_states else None,
            )
        )
    for s in splits:
        print(f"    {s}")
    return splits


def save_splits(splits, output_dir, save_timestamps: bool = False):
    """Write each session to <output_dir>/sessions/<name>/.

    `save_timestamps` writes the full seconds-per-sample array as
    timestamps.npy. It is off by default because it is 8 bytes per sample --
    everything needed to regenerate it is in split_info.json.
    """
    output_dir = Path(output_dir)
    written = []
    for s in splits:
        session_dir = output_dir / SESSIONS_DIRNAME / s.name
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "index": s.index,
            "name": s.name,
            "phys_path": s.phys_path,
            "sample_offset": s.sample_offset,
            "num_samples": s.num_samples,
            "sampling_frequency": s.sampling_frequency,
            "t_start": s.t_start,
            "duration_s": s.duration_s,
        }
        with open(session_dir / "split_info.json", "w") as f:
            json.dump(meta, f, indent=2)

        if s.states is not None:
            with open(session_dir / "states.json", "w") as f:
                json.dump({"intervals": s.states}, f, indent=2)

        if s.sorting is not None:
            spikes = {
                str(u): s.sorting.get_unit_spike_train(u).astype(np.int64)
                for u in s.sorting.unit_ids
            }
            np.savez_compressed(session_dir / "spike_trains.npz", **spikes)

        if s.analyzer is not None:
            s.analyzer.save_as(folder=session_dir / ANALYZER_NAME, format="zarr")

        if save_timestamps:
            np.save(session_dir / "timestamps.npy", s.timestamps())

        written.append(session_dir)
        print(f"    -> {session_dir}")
    return written


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="spikeshpc-split",
        description="Split a concatenated run back into its constituent recordings.",
    )
    parser.add_argument("output_dir", type=Path, help="A completed run's --output_dir")
    parser.add_argument(
        "--with_analyzer",
        action="store_true",
        help="Also build a per-session analyzer (recomputes every extension).",
    )
    parser.add_argument(
        "--save_timestamps",
        action="store_true",
        help="Write timestamps.npy per session (8 bytes per sample).",
    )
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args(argv)

    si.set_global_job_kwargs(n_jobs=args.n_jobs)
    splits = split_run(
        args.output_dir,
        with_analyzer=args.with_analyzer,
        analyzer_format="zarr" if args.with_analyzer else "memory",
    )
    save_splits(splits, args.output_dir, save_timestamps=args.save_timestamps)
    print("\n✓ Split complete.")
    return splits


if __name__ == "__main__":
    main()
