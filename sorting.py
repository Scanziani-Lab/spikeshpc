"""Stage 3: kilosort4, driven directly rather than through si.run_sorter."""

from pathlib import Path

from .channels import resolve_bad_channels
from .config import CHANMAP_NAME, SORTER_DIRNAME


def run_kilosort4(
    output_dir: Path,
    info: dict,
    settings_overrides: dict | None = None,
    bad_channels=None,
):
    """Sort the concatenated binary with kilosort4 default settings."""
    from kilosort import DEFAULT_SETTINGS, run_kilosort
    from kilosort.io import load_probe

    settings = dict(DEFAULT_SETTINGS)
    # n_chan_bin counts rows in the binary, so it stays at the full channel
    # count even when bad channels are excluded -- kilosort drops those from
    # the probe, not from the file it reads.
    settings["n_chan_bin"] = info["num_channels"]
    settings["fs"] = info["sampling_frequency"]
    settings.update(settings_overrides or {})

    bad_idx, bad_ids = resolve_bad_channels(bad_channels, info)
    if bad_idx:
        print(f"    excluding {len(bad_idx)} bad channel(s) from the probe:")
        print(f"      ids     : {bad_ids}")
        print(f"      rows    : {bad_idx}")

    results_dir = output_dir / SORTER_DIRNAME
    print(f"    kilosort4 results -> {results_dir}")
    run_kilosort(
        settings=settings,
        probe=load_probe(output_dir / CHANMAP_NAME),
        filename=output_dir / info["binary_file"],
        results_dir=results_dir,
        data_dtype=info["dtype"],
        bad_channels=bad_idx or None,
    )
    return results_dir
