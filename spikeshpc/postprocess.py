"""Stage 4: read kilosort4's output back and build the sorting analyzer."""

import json
import pickle
from pathlib import Path

import numpy as np
import spikeinterface.full as si

from .config import ANALYZER_NAME, SORTER_DIRNAME

# Spike band. kilosort filters internally and the binary it sorted is raw, so
# without this every waveform the analyzer extracts is dominated by the LFP and
# the DC offset rather than by the spike.
DEFAULT_BANDPASS = {"freq_min": 300.0, "freq_max": 6000.0}

SPIKE_LOCATIONS = "spike_locations"


def _filtered(rec, bandpass):
    """The recording the analyzer should measure waveforms from.

    kilosort's own input must not be filtered twice -- it highpasses, common-
    references and whitens internally -- but the analyzer is a different
    consumer of the same binary and has no such step. Handing it the raw
    recording makes every waveform-derived extension meaningless: the mean
    waveform becomes the DC offset common to every channel, so templates come
    out near-identical across units, sparsity picks the same channels for
    everyone, and centre-of-mass puts every unit at the same depth.

    Pass ``bandpass=False`` to opt out (the binary is already filtered, say);
    a dict is merged over :data:`DEFAULT_BANDPASS`.
    """
    if bandpass is False:
        print("    analyzer: bandpass disabled, using the recording as given")
        return rec

    kwargs = {**DEFAULT_BANDPASS, **(bandpass or {})}
    print(
        f"    analyzer: bandpass {kwargs['freq_min']:.0f}-{kwargs['freq_max']:.0f} Hz "
        "before extracting waveforms"
    )
    return si.bandpass_filter(rec, **kwargs)


# The filters an analyzer's recording chain may carry that analyzer_recording
# knows how to put back, by the class name its saved provenance records.
REPLAYABLE_FILTERS = {
    "BandpassFilterRecording": si.bandpass_filter,
    "HighpassFilterRecording": si.highpass_filter,
}


def analyzer_recording(rec, analyzer):
    """`rec` as `analyzer` measured it: its channels, through its own filters.

    An analyzer loaded on another machine usually cannot reopen its recording
    -- the saved path is relative (``../concatenated.bin``) or points at the
    cluster -- so it gets a temporary one, and ``set_temporary_recording``
    checks channels and dtype but not filtering. Handed the binary as loaded,
    everything that reads traces afterwards -- SLAy's spike snippets, the
    GUI's trace view, any recompute -- measures a different signal from the
    one the extensions were computed on: the median-subtracted but unfiltered
    binary, LFP and all, where the analyzer saw a 300-6000 Hz band.

    Rather than assume that band, the analyzer's own provenance is replayed:
    each filter layer its recording passed through, with the parameters it
    was built with, and none if it had none -- as for an analyzer built on a
    binary that is already high-passed, which a guessed bandpass would filter
    twice. Two checks make a mismatch loud rather than silent:

      * `rec` must be filtered exactly when the base of the analyzer's chain
        was (so an old analyzer is not paired with a rewritten binary, and a
        recording filtered by hand is not filtered again), and
      * the result must be filtered exactly when the analyzer's recording
        was (``rec_attributes["is_filtered"]``), which is all that can be
        checked when the provenance cannot be read at all.

    A layer that is neither a channel slice nor a filter raises: dropping it
    quietly is the failure this function exists to prevent.
    """
    out = rec.select_channels(list(analyzer.channel_ids))
    layers = _saved_recording_layers(analyzer)
    if layers is not None:
        *above, (_, _, base_annotations) = layers
        unknown = [
            name
            for name, _, _ in above
            if name not in REPLAYABLE_FILTERS and name != "ChannelSliceRecording"
        ]
        if unknown:
            raise ValueError(
                f"The analyzer's recording passes through {', '.join(unknown)}, which "
                "analyzer_recording cannot replay. Rebuild that recording yourself and "
                "pass it to analyzer.set_temporary_recording()."
            )
        base_filtered = base_annotations.get("is_filtered")
        if base_filtered is not None and bool(rec.is_filtered()) != bool(base_filtered):
            raise ValueError(
                "The recording given is "
                + ("already filtered, but the analyzer was built from an unfiltered one "
                   "(and filtered it itself): passing it would filter it twice."
                   if rec.is_filtered() else
                   "not filtered, but the analyzer was built from a filtered one: it is not "
                   "the recording this analyzer was made from (a rewritten binary?).")
                + " Pass the recording as loaded for this analyzer (load_concatenated)."
            )
        for name, kwargs, _ in reversed(above):
            if name in REPLAYABLE_FILTERS:
                out = REPLAYABLE_FILTERS[name](out, **kwargs)

    expected = analyzer.rec_attributes.get("is_filtered")
    if expected is not None and bool(out.is_filtered()) != bool(expected):
        raise ValueError(
            f"The analyzer measured {'a filtered' if expected else 'an unfiltered'} recording, "
            f"but the one rebuilt for it is {'filtered' if out.is_filtered() else 'not'}"
            + ("" if layers is not None else ", and it kept no readable record of its filters")
            + ". Rebuild the analyzer's recording yourself and pass it to "
            "analyzer.set_temporary_recording()."
        )
    return out


def _saved_recording_layers(analyzer):
    """``(class name, kwargs, annotations)`` per layer of the analyzer's saved recording.

    Outermost first; the last entry is the reader at the bottom of the chain.
    Read from the provenance the analyzer wrote when it was created -- a dict
    of class names and parameters, never instantiated, so it works although
    the files it names are gone. None for an in-memory analyzer, or one that
    kept no provenance.
    """
    folder = getattr(analyzer, "folder", None)
    node = None
    if analyzer.format == "zarr":
        import zarr

        root = zarr.open(str(folder), mode="r")
        if "recording" in root:
            node = root["recording"][0]
    elif analyzer.format == "binary_folder":
        folder = Path(folder)
        if (folder / "recording.json").is_file():
            node = json.loads((folder / "recording.json").read_text())
        elif (folder / "recording.pickle").is_file():
            with open(folder / "recording.pickle", "rb") as f:
                node = pickle.load(f)
    if not isinstance(node, dict):
        return None

    layers = []
    while isinstance(node, dict) and "class" in node:
        kwargs = dict(node.get("kwargs", {}))
        parent = kwargs.pop("recording", None) or kwargs.pop("parent_recording", None)
        layers.append((node["class"].rsplit(".", 1)[-1], kwargs, node.get("annotations", {})))
        node = parent
    return layers or None


def kilosort_spike_locations(analyzer, results_dir: Path) -> np.ndarray:
    """kilosort's per-spike positions, reordered to the analyzer's spike vector.

    kilosort already estimates an (x, y) for every spike and writes it to
    spike_positions.npy, which is the same quantity ``spike_locations`` spends
    a full pass over the recording to compute. Reusing it is the one
    substitution among kilosort's outputs that is like for like: same units,
    same definition, one row per spike.

    The reordering is the part that has to be right. kilosort writes spikes in
    detection order -- sorted by sample, but arbitrary among spikes sharing
    one -- while spikeinterface sorts those ties by unit. Copying index for
    index therefore lines up the sample indices exactly and still hands ~7% of
    spikes (on this rig's data) the position belonging to a different unit,
    with nothing to show for it afterwards. Sorting kilosort's spikes by
    (sample, cluster) reproduces spikeinterface's order, and the result is
    checked against both fields before it is used.
    """
    results_dir = Path(results_dir)
    positions = np.load(results_dir / "spike_positions.npy")
    times = np.load(results_dir / "spike_times.npy")
    clusters = np.load(results_dir / "spike_clusters.npy")

    spikes = analyzer.sorting.to_spike_vector()
    if len(positions) != len(spikes):
        raise ValueError(
            f"kilosort wrote {len(positions)} spike positions but the analyzer "
            f"holds {len(spikes)} spikes; these are not the same sorting."
        )

    order = np.lexsort((clusters, times))
    unit_ids = np.asarray(analyzer.sorting.unit_ids)
    if not np.array_equal(times[order], spikes["sample_index"]) or not np.array_equal(
        clusters[order], unit_ids[spikes["unit_index"]]
    ):
        raise ValueError(
            "kilosort's spikes could not be matched to the analyzer's one for "
            "one, so its positions cannot be trusted to belong to the right "
            "spikes. Recompute spike_locations instead (use_KS_positions=False)."
        )

    located = np.zeros(len(spikes), dtype=[("x", "float64"), ("y", "float64")])
    located["x"] = positions[order, 0]
    located["y"] = positions[order, 1]
    return located


def attach_spike_locations(analyzer, located: np.ndarray):
    """Register `located` as the analyzer's spike_locations extension.

    Built and saved directly rather than computed. There is no public route
    for this -- ``compute`` is the only supported way in -- so the extension
    object is assembled the way ``compute`` would leave it: parameters set
    (which creates the folder), data attached, run marked complete, saved.
    """
    from spikeinterface.core.sortinganalyzer import get_extension_class

    extension = get_extension_class(SPIKE_LOCATIONS)(analyzer)
    extension.set_params(save=True, method="kilosort_spike_positions")
    extension.data[SPIKE_LOCATIONS] = located
    extension.run_info = {"run_completed": True, "runtime_s": 0.0}
    analyzer.extensions[SPIKE_LOCATIONS] = extension
    extension.save()
    return extension


def postprocess(
    rec,
    output_dir: Path,
    extensions: dict,
    bad_channel_ids=None,
    bandpass=None,
    use_KS_positions: bool = True,
):
    """Read kilosort4's output back into spikeinterface and save the analyzer.

    `bad_channel_ids` are dropped from the recording first so the analyzer is
    built on the same channels kilosort4 sorted -- otherwise templates and
    center-of-mass unit locations would be computed over dead channels that
    the sorter never saw.

    `bandpass` filters the recording before any waveform is extracted; see
    :func:`_filtered` for why that is not optional in practice.

    `use_KS_positions` takes the ``spike_locations`` extension from
    kilosort's own spike_positions.npy instead of recomputing it. On a 6.6 h
    session that extension alone costs about two hours, and kilosort has
    already done the work. Falls back to computing it, with the reason
    printed, if the two sortings cannot be matched spike for spike.
    """
    results_dir = output_dir / SORTER_DIRNAME
    if not (results_dir / "spike_times.npy").exists():
        raise FileNotFoundError(
            f"No kilosort4 output in {results_dir} -- run the sorting stage first."
        )
    if bad_channel_ids:
        keep = [c for c in rec.channel_ids if str(c) not in set(bad_channel_ids)]
        print(f"    excluding {len(bad_channel_ids)} bad channel(s) from the analyzer")
        rec = rec.select_channels(keep)

    rec = _filtered(rec, bandpass)

    sorting = si.read_kilosort(folder_path=results_dir)
    print(f"    loaded {len(sorting.unit_ids)} units")

    analyzer = si.create_sorting_analyzer(
        recording=rec,
        sorting=sorting,
        folder=(output_dir / ANALYZER_NAME),
        format="zarr",
        overwrite=True,
    )

    extensions = dict(extensions)
    borrowed = None
    if use_KS_positions and SPIKE_LOCATIONS in extensions:
        if not (results_dir / "spike_positions.npy").exists():
            print("    spike_locations: no spike_positions.npy, computing it instead")
        else:
            try:
                borrowed = kilosort_spike_locations(analyzer, results_dir)
                extensions.pop(SPIKE_LOCATIONS)
            except ValueError as e:
                print(f"    spike_locations: {e}")

    analyzer.compute(extensions)

    if borrowed is not None:
        attach_spike_locations(analyzer, borrowed)
        print(f"    spike_locations: took {len(borrowed):,} positions from kilosort")
    return analyzer
