"""Stage 3: kilosort4, driven directly rather than through si.run_sorter."""

import inspect
from pathlib import Path

from .channels import resolve_bad_channels
from .config import CHANMAP_NAME, SORTER_DIRNAME

# Arguments run_kilosort4 already supplies from the pipeline's own state.
# settings_overrides must not touch these
_RESERVED_KILOSORT_ARGS = frozenset({
    "settings", "probe", "probe_name", "filename", "data_dir", "file_object",
    "results_dir", "data_dtype", "bad_channels",
})


def split_settings_overrides(overrides: dict, default_settings: dict, run_kilosort):
    """Route `overrides` to kilosort's `settings` dict or to `run_kilosort`'s
    own keyword arguments, by which one actually defines the key.

    `run_kilosort4` used to dump every override straight into `settings`, so
    there was no way to reach flags like `do_CAR` or `save_preprocessed_copy`
    that `run_kilosort` takes directly rather than through `settings`. Keys
    are classified against `default_settings` (kilosort's own
    `DEFAULT_SETTINGS`) and `run_kilosort`'s signature, not hand-maintained
    lists, so a future kilosort adding or renaming a flag is picked up for
    free rather than needing this file edited too.

    Returns `(settings_updates, kwargs)`. Raises `ValueError` for a key that
    is neither a settings key nor a `run_kilosort` parameter (almost always a
    typo -- silently dropping it would be a worse failure than refusing it),
    and for a key that names an argument the pipeline itself already supplies
    (see `_RESERVED_KILOSORT_ARGS`).
    """
    valid_kwargs = set(inspect.signature(run_kilosort).parameters) - {"settings"}

    reserved = sorted(set(overrides) & _RESERVED_KILOSORT_ARGS)
    if reserved:
        raise ValueError(
            f"settings_overrides cannot set {reserved}: the pipeline supplies "
            "these itself from the concatenated recording. Bad channels go "
            "through run_kilosort4's own bad_channels= argument."
        )

    settings_updates, kwargs, unknown = {}, {}, []
    for key, value in overrides.items():
        if key in default_settings:
            settings_updates[key] = value
        elif key in valid_kwargs:
            kwargs[key] = value
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(
            f"settings_overrides has unrecognized key(s) {sorted(unknown)}: "
            "not in kilosort.DEFAULT_SETTINGS and not a run_kilosort() "
            "argument. Check for a typo."
        )
    return settings_updates, kwargs


def run_kilosort4(
    output_dir: Path,
    info: dict,
    settings_overrides: dict | None = None,
    bad_channels=None,
):
    """Sort the concatenated binary with kilosort4 default settings.

    `settings_overrides` may mix keys from kilosort's `settings` dict (e.g.
    `nblocks`) with keyword arguments `run_kilosort` takes directly (e.g.
    `do_CAR`, `save_preprocessed_copy`, `shank_idx`); see
    :func:`split_settings_overrides` for how they are told apart.
    """
    from kilosort import DEFAULT_SETTINGS, run_kilosort
    from kilosort.io import load_probe

    settings_updates, kilosort_kwargs = split_settings_overrides(
        settings_overrides or {}, DEFAULT_SETTINGS, run_kilosort
    )

    settings = dict(DEFAULT_SETTINGS)
    # n_chan_bin counts rows in the binary, so it stays at the full channel
    # count even when bad channels are excluded -- kilosort drops those from
    # the probe, not from the file it reads. When the source binary is sorted
    # in place it can hold rows we never loaded (SpikeGLX's SY0), and chanMap
    # is what selects ours back out.
    settings["n_chan_bin"] = info.get("file_num_channels", info["num_channels"])
    settings["fs"] = info["sampling_frequency"]
    settings.update(settings_updates)

    bad_idx, bad_ids = resolve_bad_channels(bad_channels, info)
    if bad_idx:
        print(f"    excluding {len(bad_idx)} bad channel(s) from the probe:")
        print(f"      ids     : {bad_ids}")
        print(f"      rows    : {bad_idx}")
    if kilosort_kwargs:
        print(f"    run_kilosort() overrides: {kilosort_kwargs}")

    results_dir = output_dir / SORTER_DIRNAME
    print(f"    kilosort4 results -> {results_dir}")
    run_kilosort(
        settings=settings,
        probe=load_probe(output_dir / CHANMAP_NAME),
        filename=Path(info.get("binary_path") or (output_dir / info["binary_file"])),
        results_dir=results_dir,
        data_dtype=info["dtype"],
        bad_channels=bad_idx or None,
        **kilosort_kwargs,
    )
    return results_dir
