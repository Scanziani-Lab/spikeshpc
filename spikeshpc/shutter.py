"""Camera shutter timestamps from the acquisition system's own ADC.

The movement veto needs each OptiTrack frame placed on the ephys clock. That
mapping comes from a TTL the camera emits on an ADC input: one pulse per
frame, and its falling edge is the shutter closing.

Doing this in the pipeline rather than by hand means a recording can be sorted
the day it comes off the rig, with no notebook step in between. The
cross-check against the take's own frame count is what makes that safe to
automate: if the edge count does not match the CSV, the result is rejected
rather than silently used to mis-time every spike.
"""

from pathlib import Path

import numpy as np

from .config import STATES_DIRNAME
from .optitrack.sync import (
    cross_check_with_optitrack_csv,
    describe_analog_channels,
    extract_shutter_close_times,
    plot_shutter_close_sanity_check,
    rail_fraction,
    ttl_threshold,
)


def find_adc_stream(phys_path, phys_type: str, stream_name=None):
    """Name of the analog/ADC stream carrying the camera TTL, or None."""
    if stream_name:
        return stream_name
    from .io import infer_stream_name

    phys_path = Path(phys_path)
    folder = phys_path.parent if phys_path.is_file() else phys_path
    return infer_stream_name(phys_path, folder, phys_type, band="adc")


def pick_ttl_channel(events, channel_id=None, min_rail_fraction: float = 0.8):
    """The channel carrying the shutter TTL.

    With `channel_id` given it is only checked; otherwise every channel on the
    stream is scanned and the one that actually looks like a square wave is
    used. Picking by name alone is how a floating input ends up being
    thresholded into millions of noise crossings.
    """
    if channel_id is not None:
        trace = events.get_traces(channel_ids=[channel_id], end_frame=None).flatten()
        rails = rail_fraction(trace, ttl_threshold(trace))
        if rails < min_rail_fraction:
            raise ValueError(
                f"Channel {channel_id!r} does not look like a TTL "
                f"({rails:.1%} of samples at either level). Leave "
                "movement.adc_channel unset to auto-detect it."
            )
        return channel_id

    rows = describe_analog_channels(
        events, min_rail_fraction=min_rail_fraction, verbose=True
    )
    ttls = [r for r in rows if r["looks_like_ttl"]]
    if not ttls:
        raise ValueError(
            "No channel on this stream looks like a TTL. Was the camera "
            "shutter output connected for this session?"
        )
    if len(ttls) > 1:
        # Several square waves: which is the camera is not ours to guess.
        raise ValueError(
            f"{len(ttls)} channels look like TTLs "
            f"({[r['channel_id'] for r in ttls]}); set movement.adc_channel."
        )
    return ttls[0]["channel_id"]


def derive_shutter_times(
    phys_path,
    output_dir: Path,
    session: str,
    config: dict,
    phys_type=None,
    optitrack_csv=None,
):
    """Extract, check, plot and cache shutter-close times for one recording.

    Returns the path to the saved .npy, or None if the TTL could not be used.
    Re-running is cheap: an existing cache is returned untouched.
    """
    import spikeinterface.full as si

    states_dir = Path(output_dir) / STATES_DIRNAME
    cached = states_dir / f"{session}_shutter_close_times.npy"
    if cached.exists():
        print(f"      shutter: reusing {cached.name}")
        return cached

    stream = find_adc_stream(phys_path, phys_type, config.get("adc_stream_name"))
    if stream is None:
        print("      shutter: no ADC stream on this recording, skipping")
        return None

    phys_path = Path(phys_path)
    folder = phys_path.parent if phys_path.is_file() else phys_path
    if phys_type == "spikeglx":
        events = si.read_spikeglx(folder_path=folder, stream_name=stream)
    else:
        events = si.read_openephys(folder_path=folder, stream_name=stream)
    print(f"      shutter: ADC stream {stream!r}, "
          f"{events.get_num_channels()} channels")

    try:
        channel = pick_ttl_channel(events, config.get("adc_channel"))
        times = extract_shutter_close_times(events, channel_id=channel)
    except ValueError as e:
        print(f"      shutter: {e} -- skipping")
        return None

    print(f"      shutter: {len(times)} falling edges on {channel!r}")
    if len(times) > 1:
        isis = np.diff(times)
        print(f"        interval {isis.mean() * 1000:.3f} +/- "
              f"{isis.std() * 1000:.3f} ms ({1 / isis.mean():.2f} Hz)")

    # The take's own frame count is the only independent check available, and
    # a mismatch means every frame would be mistimed. Refuse rather than cache.
    if optitrack_csv is not None and Path(optitrack_csv).exists():
        try:
            stats = cross_check_with_optitrack_csv(times, optitrack_csv)
        except Exception as e:
            print(f"        cross-check failed: {e}")
            return None
        print(f"        CSV frames {stats['optitrack_total_frames']}, "
              f"detected {stats['n_detected']}, "
              f"difference {stats['frame_count_difference']}")
        if not stats.get("plausible", True):
            print("        -> not a plausible shutter train, refusing to cache it")
            return None

    states_dir.mkdir(parents=True, exist_ok=True)
    np.save(cached, times)

    if config.get("save_sanity_plot", True):
        try:
            import matplotlib

            matplotlib.use("Agg")
            fig = plot_shutter_close_sanity_check(events, times, channel_id=channel)
            png = states_dir / f"{session}_shutter_check.png"
            fig.savefig(png, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(fig)
            print(f"        sanity plot -> {png.name}")
        except Exception as e:  # a missing plot must not fail a sorting job
            print(f"        (sanity plot skipped: {e})")

    print(f"      shutter: cached -> {cached.name}")
    return cached
