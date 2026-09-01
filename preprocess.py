"""Stage 1: load, align, concatenate, and write the binary kilosort4 sorts."""

import json
from pathlib import Path

import spikeinterface.full as si

from .channels import (
    align_channels_by_location,
    check_gain_consistency,
    drop_sync_channels,
)
from .config import CONCAT_BIN_NAME, CONCAT_INFO_NAME, PROBE_NAME
from .io import channel_positions, read_recording, write_channel_map


def preprocess(
    phys_paths,
    output_dir: Path,
    phys_type: str | None = None,
    stream_name: str | None = None,
    preprocessing: dict | None = None,
    dtype=None,
    align_tolerance_um: float = 1.0,
    sampling_frequency_max_diff: float = 0.0,
) -> dict:
    """Load, sync-strip, align, concatenate and write the binary + metadata.

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
        rec = si.apply_preprocessing_pipeline(rec, preprocessing)

    try:
        probegroup = rec.get_probegroup()
    except ValueError as e:
        raise ValueError(
            "Recording has no probe attached; cannot write a channel map. "
            "Check that the .meta/settings.xml sidecars are next to the binary."
        ) from e

    bin_path = output_dir / CONCAT_BIN_NAME
    print(f"    writing combined binary -> {bin_path}")
    si.write_binary_recording(
        rec, file_paths=bin_path, dtype=dtype, add_file_extension=False, verbose=True
    )

    write_channel_map(rec, output_dir)
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
        "binary_file": CONCAT_BIN_NAME,
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
