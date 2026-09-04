"""Channel bookkeeping: sync stripping, geometry alignment, bad channels."""

import json
import re
from pathlib import Path

import numpy as np
import spikeinterface.full as si

from .config import BAD_CHANNELS_NAME, CONCAT_INFO_NAME
from .io import channel_positions


def _is_sync_label(label) -> bool:
    """True for SpikeGLX sync channel labels: 'SY0', 'SY0;768:768', 'imec0.ap#SY0'."""
    return any(
        re.fullmatch(r"sy\d*", part.strip(), flags=re.IGNORECASE)
        for part in re.split(r"[;#]", str(label))
    )


def drop_sync_channels(rec):
    """Remove the SpikeGLX sync channel (SY0) so it is not concatenated/sorted.

    Recent neo/spikeinterface expose sync as its own '<stream>-SYNC' stream, so
    the AP stream is usually already clean -- this is a no-op then. Older
    versions carry SY0 as the last channel of the AP stream.

    Returns (recording, removed_labels).
    """
    names = rec.get_property("channel_names")
    labels = (
        [str(n) for n in names]
        if names is not None
        else [str(c) for c in rec.channel_ids]
    )

    keep = [cid for cid, lab in zip(rec.channel_ids, labels) if not _is_sync_label(lab)]
    removed = [lab for lab in labels if _is_sync_label(lab)]
    if not removed:
        return rec, []
    print(f"    dropping {len(removed)} sync channel(s): {removed}")
    return rec.select_channels(keep), removed


def align_channels_by_location(recs, tolerance_um: float = 1.0):
    """Reorder recordings so channel i is the same electrode site in all of them.

    SpikeGLX and OpenEphys name and order channels completely differently, and
    even two SpikeGLX runs will disagree if the imro table changed between
    them -- channel 'AP100' is a slot, not a site. spikeinterface's
    concatenate_recordings only checks that the channel *id arrays* are equal,
    so it will happily stack mismatched sites (or refuse outright across
    systems). Matching on the probe geometry instead is what actually makes
    channel i mean one thing for the whole concatenated recording.

    Each recording is matched to the first one by nearest contact position,
    accepting pairs within `tolerance_um`. Sites missing from any recording
    are dropped from all of them; the surviving channels keep the first
    recording's order and ids.

    Returns (aligned_recs, report).
    """
    ref = recs[0]
    ref_loc = channel_positions(ref)

    keep = np.ones(len(ref_loc), dtype=bool)
    nearest_per_rec, residual_per_rec = [], []
    for rec in recs:
        loc = channel_positions(rec)
        d = np.linalg.norm(ref_loc[:, None, :] - loc[None, :, :], axis=-1)
        nearest = d.argmin(axis=1)
        residual = d[np.arange(len(ref_loc)), nearest]
        keep &= residual <= tolerance_um
        nearest_per_rec.append(nearest)
        residual_per_rec.append(residual)

    keep_idx = np.flatnonzero(keep)  # in the first recording's channel order
    if keep_idx.size == 0:
        raise ValueError(
            "No electrode sites are common to all recordings (within "
            f"{tolerance_um} um). Are these the same probe/insertion? "
            "Check the probe geometries before concatenating."
        )

    canonical_ids = ref.channel_ids[keep_idx]
    aligned, report = [], []
    for j, (rec, nearest, residual) in enumerate(
        zip(recs, nearest_per_rec, residual_per_rec)
    ):
        # Nearest-neighbour matching is only a site correspondence if it is
        # one-to-one; two sites collapsing onto one channel means tolerance_um
        # is wider than the contact pitch.
        picked = nearest[keep_idx]
        if np.unique(picked).size != picked.size:
            raise ValueError(
                f"Recording {j}: tolerance_um={tolerance_um} matched two "
                "electrode sites to the same channel. Lower it below the "
                "contact pitch of your probe."
            )
        parent_ids = rec.channel_ids[picked]
        report.append(
            {
                "num_channels_in": int(rec.get_num_channels()),
                "num_matched": int(keep_idx.size),
                "num_dropped": int(rec.get_num_channels() - keep_idx.size),
                "max_residual_um": float(residual[keep_idx].max()),
                "reordered": not np.array_equal(parent_ids, rec.channel_ids),
            }
        )
        if not report[-1]["reordered"]:
            aligned.append(rec)
            continue
        # renamed_channel_ids gives every recording the first one's id set, which
        # is what concatenate_recordings compares. ChannelSliceRecording keeps
        # the order we pass and rewires the probe's contact_vector to match.
        aligned.append(
            si.ChannelSliceRecording(
                rec, channel_ids=parent_ids, renamed_channel_ids=canonical_ids
            )
        )

    dropped = int(len(ref_loc) - keep_idx.size)
    print(
        f"    aligned channels by probe position: {keep_idx.size} sites common "
        f"to all {len(recs)} recording(s)" + (f", {dropped} dropped" if dropped else "")
    )
    for j, r in enumerate(report):
        if r["reordered"] or r["num_dropped"]:
            print(
                f"      rec {j}: {r['num_channels_in']} -> {r['num_matched']} ch, "
                f"reordered={r['reordered']}, "
                f"max residual {r['max_residual_um']:.3f} um"
            )
    return aligned, report


def check_gain_consistency(recs):
    """Warn if recordings disagree on gain/offset -- µV scaling would be mixed.

    Not fatal: the concatenated int16 samples are still valid, but any
    amplitude-based metric spans recordings on different scales.
    """
    gains = [rec.get_property("gain_to_uV") for rec in recs]
    offsets = [rec.get_property("offset_to_uV") for rec in recs]
    consistent = True
    for values, label in ((gains, "gain_to_uV"), (offsets, "offset_to_uV")):
        if any(v is None for v in values):
            continue
        if not all(np.allclose(values[0], v) for v in values[1:]):
            consistent = False
            print(
                f"    WARNING: {label} differs across recordings "
                f"(e.g. {np.ravel(values[0])[:3]} vs {np.ravel(values[1])[:3]}). "
                "Amplitudes will not be comparable across the concatenated span."
            )
    return consistent


def detect_bad_channels_auto(rec, output_dir: Path, config: dict, manual=None):
    """Flag bad channels with si.detect_bad_channels and record the labels.

    Writes bad_channels.json (every channel's label, plus what was detected,
    what was listed manually, and the union actually applied) so the call can
    be reviewed -- and so a later post-processing-only re-run can reuse exactly
    the set that was sorted rather than re-detecting.

    Returns the detected channel ids as strings.
    """
    kwargs = {k: v for k, v in config.items() if k != "enabled"}
    # JSON has no set literal, so accept a list for channel_filters.
    if isinstance(kwargs.get("channel_filters"), list):
        kwargs["channel_filters"] = set(kwargs["channel_filters"])

    if not rec.is_filtered():
        print("    recording is unfiltered; detect_bad_channels highpasses on the fly")
    bad_ids, labels = si.detect_bad_channels(rec, **kwargs)

    detected = [str(c) for c in bad_ids]
    by_label = {}
    for cid, label in zip(rec.channel_ids, labels):
        by_label.setdefault(str(label), []).append(str(cid))
    summary = ", ".join(f"{n}={len(v)}" for n, v in sorted(by_label.items()))
    print(f"    detected {len(detected)} bad channel(s) [{summary}]")
    for label, chans in sorted(by_label.items()):
        if label != "good":
            print(f"      {label}: {chans}")

    manual = [str(c) for c in (manual or [])]
    record = {
        "method": config.get("method", "coherence+psd"),
        "kwargs": {
            k: sorted(v) if isinstance(v, set) else v for k, v in kwargs.items()
        },
        "labels": {str(c): str(l) for c, l in zip(rec.channel_ids, labels)},
        "detected": detected,
        "manual": manual,
        "applied": sorted(set(detected) | set(manual)),
    }
    with open(output_dir / BAD_CHANNELS_NAME, "w") as f:
        json.dump(record, f, indent=2)
    print(f"    labels written to {output_dir / BAD_CHANNELS_NAME}")
    return detected


def resolve_bad_channels(bad_channels, info: dict):
    """Map a user-supplied bad-channel list to rows of the concatenated binary.

    Entries may be channel ids as written in concat_info.json (preferred --
    stable across re-runs and readable) or plain ints, which are taken as
    0-based row indices into concatenated.bin. Unrecognised entries raise
    rather than being skipped: a typo'd channel name should not cost a GPU job.

    Returns (indices, ids) for the resolved channels, both in row order.
    """
    if not bad_channels:
        return [], []

    channel_ids = [str(c) for c in info["channel_ids"]]
    indices, unresolved = set(), []
    for entry in bad_channels:
        # bool is an int subclass; nobody means row 0/1 by True/False.
        if isinstance(entry, int) and not isinstance(entry, bool):
            if 0 <= entry < len(channel_ids):
                indices.add(entry)
            else:
                unresolved.append(entry)
        elif str(entry) in channel_ids:
            indices.add(channel_ids.index(str(entry)))
        else:
            unresolved.append(entry)

    if unresolved:
        raise ValueError(
            f"bad_channels entries not found in this recording: {unresolved}. "
            f"Expected channel ids from {CONCAT_INFO_NAME} (e.g. "
            f"{channel_ids[:3]}...) or ints in [0, {len(channel_ids)})."
        )

    indices = sorted(indices)
    return indices, [channel_ids[i] for i in indices]
