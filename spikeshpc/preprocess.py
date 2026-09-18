"""Stage 1: load, align, concatenate, and write the binary kilosort4 sorts."""

import json
from pathlib import Path

import numpy as np
import spikeinterface.full as si
from spikeinterface.core.baserecording import BaseRecording, BaseRecordingSegment

from .channels import (
    align_channels_by_location,
    check_gain_consistency,
    drop_sync_channels,
    resolve_bad_channels,
)
from .config import CONCAT_BIN_NAME, CONCAT_INFO_NAME, PROBE_NAME
from .io import (
    channel_positions,
    check_source_binary,
    locate_source_binary,
    read_recording,
    write_channel_map,
)


class _InterleavedSegment(BaseRecordingSegment):
    """Traces assembled from two parent segments, in an arbitrary order.

    Exists because `si.aggregate_channels` cannot be trusted with this:
    its own segment (`ChannelsAggregationRecordingSegment.get_traces`, as of
    spikeinterface 0.104.8) groups a requested `channel_indices` by which
    source recording each one came from and concatenates the groups in
    first-encountered order -- silently discarding the caller's requested
    order whenever that order interleaves the two sources, which a request
    for the original channel order always does unless every bad channel
    happens to be a trailing block. Confirmed directly: asking a
    good-then-bad aggregate for its channels back in original order returns
    each channel's neighbour's data, not its own.

    This does the same per-source batching -- one get_traces call per
    parent, not one per channel, so it costs nothing extra -- but scatters
    each source's columns back into the positions actually requested.
    """

    def __init__(self, segments, source, source_index, times_kwargs):
        BaseRecordingSegment.__init__(self, **times_kwargs)
        self._segments = segments
        self._source = np.asarray(source)
        self._source_index = np.asarray(source_index)

    def get_num_samples(self) -> int:
        return self._segments[0].get_num_samples()

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        n_total = len(self._source)
        if channel_indices is None:
            wanted = np.arange(n_total)
        elif isinstance(channel_indices, slice):
            wanted = np.arange(n_total)[channel_indices]
        else:
            wanted = np.asarray(channel_indices)

        out = None
        for which, segment in enumerate(self._segments):
            mask = self._source[wanted] == which
            if not np.any(mask):
                continue
            parent_indices = self._source_index[wanted[mask]]
            traces = segment.get_traces(start_frame, end_frame, parent_indices)
            if out is None:
                out = np.empty((traces.shape[0], len(wanted)), dtype=traces.dtype)
            out[:, mask] = traces
        return out


def _interleave_channels(processed, raw, channel_order):
    """`processed` and `raw`'s channels, recombined in `channel_order`.

    `channel_order` is a permutation of `processed.channel_ids +
    raw.channel_ids` combined (every id from both, each exactly once) -- the
    original, pre-split channel order, normally.

    Metadata (probe geometry, gains, `is_filtered`, ...) is taken from
    `si.aggregate_channels`, which gets that part right -- it is only the
    *trace* reordering that cannot be trusted for an interleaved request (see
    `_InterleavedSegment`). `copy_metadata` maps every property across by id,
    so it does not matter that the aggregate's own channel order differs
    from `channel_order`.
    """
    combined = si.aggregate_channels([processed, raw])
    result = BaseRecording(
        processed.get_sampling_frequency(), channel_order, processed.get_dtype()
    )
    combined.copy_metadata(result, only_main=False, ids=list(channel_order))

    processed_ids = list(map(str, processed.channel_ids))
    raw_ids = list(map(str, raw.channel_ids))
    source = np.array(
        [0 if str(c) in processed_ids else 1 for c in channel_order]
    )
    source_index = np.array(
        [
            processed_ids.index(str(c)) if s == 0 else raw_ids.index(str(c))
            for c, s in zip(channel_order, source)
        ]
    )
    for seg_p, seg_r in zip(
        processed._recording_segments, raw._recording_segments
    ):
        result.add_recording_segment(
            _InterleavedSegment(
                (seg_p, seg_r), source, source_index, seg_p.get_times_kwargs()
            )
        )
    return result


def _apply_preprocessing(rec, preprocessing: dict, bad_ids: list):
    """Run `preprocessing` on `rec`, without letting `bad_ids` take part in it.

    Filters that pool information across channels are only as robust as the
    minority of contamination they can tolerate: a global median (common
    reference) shrugs off a handful of outliers, but a per-channel filter
    like phase_shift has no such protection, since it never sees the other
    channels at all. A channel pinned at a rail value can come back from a
    fractional-sample-delay filter swinging far outside its original range at
    every processing boundary -- measured on real data, a channel that was a
    constant -30690 for an entire 976s recording came back from phase_shift
    swinging from -30928 to +32277.

    kilosort already excludes bad_channels from every one of its own
    computations (it slices to `chanMap` before any filtering, CAR, or
    whitening -- see BinaryFiltered.filter() in kilosort's own source), so
    leaving them out here changes nothing about the sort. What it buys is a
    written binary that does not carry manufactured nonsense on rows nothing
    is *supposed* to read the wrong way, and it means a QC plot of a "bad"
    channel still shows the real signal that made it bad, not an artifact of
    correcting it as though it were good.

    Their rows are put back afterward, raw and in their original position, so
    the channel count/order/chanMap the rest of this function relies on are
    unaffected -- only the SAMPLES on those specific rows differ from what an
    unguarded `si.apply_preprocessing_pipeline` would have produced.
    """
    if not bad_ids:
        return si.apply_preprocessing_pipeline(rec, preprocessing)

    bad_set = set(bad_ids)
    good_ids = [c for c in rec.channel_ids if str(c) not in bad_set]
    print(f"    excluding {len(bad_ids)} bad channel(s) from preprocessing: {bad_ids}")

    processed = si.apply_preprocessing_pipeline(rec.select_channels(good_ids), preprocessing)
    raw = rec.select_channels(bad_ids)
    if raw.get_dtype() != processed.get_dtype():
        raw = si.astype(raw, processed.get_dtype())

    result = _interleave_channels(processed, raw, list(rec.channel_ids))
    # aggregate_channels (inside _interleave_channels) does not carry this
    # annotation over from its inputs the way copy_metadata carries the rest.
    result.annotate(is_filtered=processed.is_filtered())
    return result


def preprocess(
    phys_paths,
    output_dir: Path,
    phys_type: str | None = None,
    stream_name: str | None = None,
    preprocessing: dict | None = None,
    dtype=None,
    align_tolerance_um: float = 1.0,
    sampling_frequency_max_diff: float = 0.0,
    reuse_source: bool = True,
    bad_channels=None,
) -> dict:
    """Load, sync-strip, align, concatenate and write the binary + metadata.

    `bad_channels` are excluded from `preprocessing` itself (see
    :func:`_apply_preprocessing`) -- they are resolved against this
    recording's own channel ids, which is the same set `concat_info.json`
    ends up recording, so entries may be given exactly as they will be passed
    to the sorting stage.

    Returns the concat_info dict, which is also written to concat_info.json
    and is everything the sorting/post-processing stages need.
    """
    import probeinterface

    recs, removed_sync, num_samples = [], [], []
    for path in phys_paths:
        rec, phys_type, stream_name = read_recording(path, phys_type, stream_name)
        print(
            f"    {path}  [{phys_type}/{stream_name}]  "
            f"{rec.get_num_channels()} ch, {rec.get_num_frames()} samples"
        )
        rec, removed = drop_sync_channels(rec)
        recs.append(rec)
        removed_sync.append(removed)
        num_samples.append(int(rec.get_num_frames()))

    alignment = None
    if len(recs) > 1:
        recs, alignment = align_channels_by_location(recs, align_tolerance_um)
        gains_consistent = check_gain_consistency(recs)
        print(f"    concatenating {len(recs)} recordings...")
        rec = si.concatenate_recordings(
            recs, sampling_frequency_max_diff=sampling_frequency_max_diff
        )
    else:
        gains_consistent = True
        rec = recs[0]

    if preprocessing:
        print(f"    applying preprocessing: {list(preprocessing)}")
        _, bad_ids = resolve_bad_channels(
            bad_channels, {"channel_ids": [str(c) for c in rec.channel_ids]}
        )
        rec = _apply_preprocessing(rec, preprocessing, bad_ids)

    try:
        probegroup = rec.get_probegroup()
    except ValueError as e:
        raise ValueError(
            "Recording has no probe attached; cannot write a channel map. "
            "Check that the .meta/settings.xml sidecars are next to the binary."
        ) from e

    # ── Sort the source binary in place when nothing has to change ───────
    # Rewriting is pure copying: hours of I/O for a long single-session
    # recording, and a second full-size copy on disk. It is only avoidable
    # when there is one recording, no preprocessing, and the file's own layout
    # matches what kilosort will read.
    source_path, channel_rows, n_file_channels = None, None, None
    if reuse_source:
        why = None
        if len(phys_paths) > 1:
            why = "multiple recordings must be concatenated"
        elif preprocessing:
            why = "preprocessing changes the samples"
        elif dtype is not None and np.dtype(dtype) != rec.get_dtype():
            why = f"a dtype change to {dtype} was requested"
        elif rec.get_dtype() != np.dtype("int16"):
            why = f"recording dtype is {rec.get_dtype()}, not int16"
        else:
            candidate = locate_source_binary(phys_paths[0], phys_type, stream_name)
            if candidate is None:
                why = "could not identify a single source binary"
            else:
                checked = check_source_binary(rec, candidate, str(rec.get_dtype()))
                if checked is None:
                    why = f"{candidate.name} does not match the loaded traces"
                else:
                    n_file_channels, channel_rows = checked
                    source_path = candidate
        if source_path is None:
            print(f"    writing a new binary ({why})")

    if source_path is not None:
        bin_path = source_path
        extra = n_file_channels - len(channel_rows)
        print(f"    sorting the source binary in place -> {bin_path}")
        print(f"      verified {len(channel_rows)} channels against "
              f"{n_file_channels} rows in the file"
              + (f" ({extra} extra row(s) skipped)" if extra else ""))
    else:
        bin_path = output_dir / CONCAT_BIN_NAME
        print(f"    writing combined binary -> {bin_path}")
        si.write_binary_recording(
            rec, file_paths=bin_path, dtype=dtype, add_file_extension=False, verbose=True
        )

    write_channel_map(rec, output_dir, channel_rows)
    probeinterface.write_probeinterface(output_dir / PROBE_NAME, probegroup)

    # Cumulative sample offsets let you split the concatenated sorting back
    # into per-recording pieces: recording i spans [offsets[i], offsets[i+1]).
    offsets, running = [0], 0
    for n in num_samples:
        running += n
        offsets.append(running)

    gains = rec.get_property("gain_to_uV")
    offs = rec.get_property("offset_to_uV")
    concat_info = {
        "phys_paths": [str(Path(p).resolve()) for p in phys_paths],
        "phys_type": phys_type,
        "stream_name": stream_name,
        "binary_file": bin_path.name,
        "binary_path": str(bin_path.resolve()),
        "sorted_in_place": source_path is not None,
        # Rows per sample in the binary, and which of them are our channels.
        # These differ from num_channels only when sorting a source file that
        # carries extra rows.
        "file_num_channels": int(n_file_channels or rec.get_num_channels()),
        "channel_rows": None if channel_rows is None else [int(r) for r in channel_rows],
        "num_samples": num_samples,
        "sample_offsets": offsets,
        "total_samples": running,
        "sampling_frequency": float(rec.get_sampling_frequency()),
        "num_channels": int(rec.get_num_channels()),
        "dtype": str(rec.get_dtype() if dtype is None else dtype),
        # Lets load_concatenated() restore the flag, so detect_bad_channels
        # does not highpass a second time when preprocessing already filtered.
        "is_filtered": bool(rec.is_filtered()),
        "channel_ids": [str(c) for c in rec.channel_ids],
        "gain_to_uV": gains.tolist() if gains is not None else None,
        "offset_to_uV": offs.tolist() if offs is not None else None,
        "gains_consistent": gains_consistent,
        "removed_sync_channels": removed_sync,
        "channel_alignment": (
            None
            if alignment is None
            else {"tolerance_um": align_tolerance_um, "per_recording": alignment}
        ),
        "channel_locations": channel_positions(rec).tolist(),
    }
    with open(output_dir / CONCAT_INFO_NAME, "w") as f:
        json.dump(concat_info, f, indent=2)
    return concat_info
