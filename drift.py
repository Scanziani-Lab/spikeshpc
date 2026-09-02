"""Report kilosort4's estimated drift across each concatenation junction.

Sorting a concatenation only works if the probe sits in nearly the same place
in every session. kilosort estimates the shift per batch (ops['dshift']) and
corrects for it by interpolating between channels, but that interpolation has
a Gaussian kernel of width `sig_interp` -- correcting a shift much larger than
that means synthesising each channel from contacts several rows away, and the
waveform you get out is not the waveform that went in.

This reads ops.npy and concat_info.json and reports the step at each junction
in micrometres, in units of the probe's own row pitch, and against the
settings that decide whether the correction is trustworthy.
"""

import json
from pathlib import Path

import numpy as np

from .config import CONCAT_INFO_NAME, SORTER_DIRNAME

# kilosort4 defaults these are judged against; the report reads the run's own
# ops where it can and falls back to these.
DEFAULT_SIG_INTERP = 20.0
DEFAULT_MAX_CHANNEL_DISTANCE = 32.0


def probe_row_pitch(channel_locations):
    """Smallest non-zero spacing between contact rows, in micrometres."""
    y = np.unique(np.round(np.asarray(channel_locations, dtype=float)[:, 1], 3))
    if y.size < 2:
        return None
    return float(np.min(np.diff(y)))


def _verdict(step_um, sig_interp, pitch):
    """How much to trust a correction of this size."""
    if pitch is not None and step_um < pitch:
        return "negligible", "Below one row pitch. Concatenation is fine."
    if step_um <= sig_interp:
        return (
            "correctable",
            "Within sig_interp, so the interpolation is well posed.",
        )
    if step_um <= 2 * sig_interp:
        return (
            "marginal",
            "Past sig_interp. Check that units are tracked across the junction "
            "before trusting cross-session comparisons.",
        )
    return (
        "beyond correction",
        f"More than 2x sig_interp ({sig_interp:.0f} um). Correcting this "
        "synthesises each channel from contacts several rows away, so the "
        "same neuron yields different templates either side of the junction. "
        "Sort the sessions separately and match units post hoc instead.",
    )


def drift_at_junction(output_dir, window_batches: int = 3, verbose: bool = True):
    """Summarise the drift step at every junction in a concatenated run.

    `window_batches` batches either side of each junction are averaged. Returns
    a dict with one entry per junction, or None when there is nothing to report
    (a single session, or drift correction disabled with nblocks=0).
    """
    output_dir = Path(output_dir)
    info_path = output_dir / CONCAT_INFO_NAME
    ops_path = output_dir / SORTER_DIRNAME / "ops.npy"

    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found -- run pre-processing first.")
    if not ops_path.exists():
        raise FileNotFoundError(f"{ops_path} not found -- run the sorting stage first.")

    info = json.loads(info_path.read_text())
    ops = np.load(ops_path, allow_pickle=True).item()

    offsets = info["sample_offsets"]
    if len(offsets) < 3:
        if verbose:
            print("    only one recording; no junction to report")
        return None

    dshift = ops.get("dshift")
    if dshift is None:
        if verbose:
            print("    ops['dshift'] is None (nblocks=0?); no drift estimate to report")
        return None
    dshift = np.atleast_2d(np.asarray(dshift, dtype=float))

    batch_size = int(ops.get("batch_size") or ops.get("NT"))
    fs = float(info["sampling_frequency"])
    sig_interp = float(ops.get("sig_interp") or DEFAULT_SIG_INTERP)
    max_chan_dist = float(
        ops.get("max_channel_distance") or DEFAULT_MAX_CHANNEL_DISTANCE
    )
    pitch = probe_row_pitch(info["channel_locations"])
    per_batch = dshift.mean(axis=1)  # collapse probe sections to one trace

    if verbose:
        print(f"    drift estimate: {dshift.shape[0]} batches x "
              f"{dshift.shape[1]} probe sections "
              f"(nblocks={(dshift.shape[1] + 1) // 2})")
        if pitch is not None:
            print(f"    probe row pitch: {pitch:.1f} um | "
                  f"sig_interp: {sig_interp:.0f} um | "
                  f"max_channel_distance: {max_chan_dist:.0f} um")

    junctions = []
    for i in range(1, len(offsets) - 1):
        boundary = offsets[i]
        b = int(boundary // batch_size)
        lo = max(b - window_batches, 0)
        hi = min(b + window_batches, dshift.shape[0])
        # The batch containing the junction straddles both sessions, so it is
        # excluded from both sides rather than being assigned to one.
        pre = dshift[lo:b]
        post = dshift[b + 1 : hi + 1]
        if pre.size == 0 or post.size == 0:
            continue

        step = float(abs(post.mean() - pre.mean()))
        level, advice = _verdict(step, sig_interp, pitch)
        entry = {
            "junction_index": i,
            "boundary_sample": int(boundary),
            "boundary_time_s": boundary / fs,
            "boundary_batch": b,
            "pre_mean_um": float(pre.mean()),
            "post_mean_um": float(post.mean()),
            "step_um": step,
            "step_in_row_pitch": (step / pitch) if pitch else None,
            "step_in_sig_interp": step / sig_interp,
            "nonrigid_spread_um": float(np.ptp(post, axis=1).mean()),
            "verdict": level,
            "advice": advice,
        }
        junctions.append(entry)

        if verbose:
            name_a = Path(info["phys_paths"][i - 1]).stem
            name_b = Path(info["phys_paths"][i]).stem
            print()
            print(f"    junction {i}: {name_a} -> {name_b} "
                  f"at {entry['boundary_time_s']:.0f} s (batch {b})")
            print(f"      shift before : {entry['pre_mean_um']:+.1f} um")
            print(f"      shift after  : {entry['post_mean_um']:+.1f} um")
            line = f"      STEP         : {step:.1f} um"
            if pitch:
                line += f"  ({step / pitch:.1f} x row pitch)"
            line += f"  ({step / sig_interp:.1f} x sig_interp)"
            print(line)
            tilt = f"      non-rigid tilt across probe: {entry['nonrigid_spread_um']:.1f} um"
            if step > 0:
                tilt += f" ({entry['nonrigid_spread_um'] / step:.0%} of the step)"
            print(tilt)
            print(f"      -> {level.upper()}: {advice}")

    # Within-session wobble puts the junction step in context.
    if verbose and junctions:
        print()
        for i in range(len(offsets) - 1):
            a = int(offsets[i] // batch_size)
            b = int(offsets[i + 1] // batch_size)
            seg = per_batch[a:b]
            if seg.size:
                name = Path(info["phys_paths"][i]).stem
                print(f"    within {name}: drift ranges {np.ptp(seg):.1f} um "
                      f"({seg.min():+.1f} to {seg.max():+.1f})")

    return {
        "batch_size": batch_size,
        "sampling_frequency": fs,
        "sig_interp": sig_interp,
        "max_channel_distance": max_chan_dist,
        "row_pitch_um": pitch,
        "junctions": junctions,
    }


def plot_drift(output_dir, ax=None, save: bool = True):
    """Plot the per-batch drift with the session junctions marked."""
    import matplotlib

    if save and ax is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    info = json.loads((output_dir / CONCAT_INFO_NAME).read_text())
    ops = np.load(output_dir / SORTER_DIRNAME / "ops.npy", allow_pickle=True).item()
    dshift = np.atleast_2d(np.asarray(ops["dshift"], dtype=float))
    batch_size = int(ops.get("batch_size") or ops.get("NT"))
    fs = float(info["sampling_frequency"])
    t = np.arange(dshift.shape[0]) * batch_size / fs

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, dshift, lw=0.5, alpha=0.6)
    ax.plot(t, dshift.mean(axis=1), color="k", lw=1.5, label="mean")
    for offset in info["sample_offsets"][1:-1]:
        ax.axvline(offset / fs, color="crimson", ls="--", lw=1.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("depth shift (um)")
    ax.set_title("kilosort drift estimate; dashed = session junction")
    ax.legend(loc="best", fontsize=8)

    if save:
        path = output_dir / SORTER_DIRNAME / "drift_at_junction.png"
        ax.figure.savefig(path, dpi=150, bbox_inches="tight")
        print(f"    -> {path}")
        return path
    return ax


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="spikeshpc-drift",
        description="Report kilosort's drift step across concatenation junctions.",
    )
    parser.add_argument("output_dir", type=Path, help="A completed run's --output_dir")
    parser.add_argument("--window_batches", type=int, default=3)
    parser.add_argument("--plot", action="store_true", help="Also save a PNG.")
    args = parser.parse_args(argv)

    report = drift_at_junction(args.output_dir, args.window_batches)
    if report and args.plot:
        plot_drift(args.output_dir)
    return report


if __name__ == "__main__":
    main()
