"""The four-stage runner."""

import json
import os
import tempfile
from pathlib import Path

import spikeinterface.full as si

from .channels import detect_bad_channels_auto, resolve_bad_channels
from .config import (
    ANALYZER_NAME,
    BAD_CHANNELS_NAME,
    DEFAULT_PIPELINE,
    STATES_DIRNAME,
    deep_merge,
)
from .io import load_concatenated
from .postprocess import postprocess
from .preprocess import preprocess
from .sorting import run_kilosort4
from .states import merge_to_concatenated_time, score_session


def run_pipeline(
    phys_path,
    phys_type: str | None = None,
    stream_name: str | None = None,
    output_dir: Path | None = None,
    skip_statescoring: bool = False,
    skip_preprocessing: bool = False,
    skip_sorting: bool = False,
    skip_postprocessing: bool = False,
    tmp_dir: Path | None = None,
    **pipeline_kwargs,
):
    """Score states, concatenate, sort with kilosort4, and post-process.

    Parameters
    ----------
    phys_path : Path or list of Path
        One recording directory or raw binary file, or a list of them to
        concatenate before sorting.
    phys_type : str, optional
        'spikeglx' or 'openephysbinary'. Auto-detected from the sidecar files
        (.meta vs structure.oebin/settings.xml) when None.
    stream_name : str, optional
        Name of the raw phys stream saved by SpikeGLX/OpenEphys -- e.g.
        'imec0.ap' or 'Record Node 101#Neuropix-PXI-100.ProbeA'. Inferred
        from the filename/available streams when None.
    output_dir : Path, optional
        Where the concatenated binary, chanMap.mat, states, kilosort4 output
        and analyzer are written. Defaults to the first phys_path.
    skip_statescoring : bool
        Skip brain-state scoring and reuse whatever is in output_dir/states.
    skip_preprocessing : bool
        Skip loading/concatenating/writing the binary and reuse the one
        already in output_dir.
    skip_sorting : bool
        Skip kilosort4 and reuse the existing output_dir/kilosort4 results.
    skip_postprocessing : bool
        Stop after sorting; do not build the sorting analyzer.
    tmp_dir : Path, optional
        Scratch space for intermediate files. Defaults to output_dir/"tmp".
        Must be on a filesystem with room for kilosort4's intermediates.
    **pipeline_kwargs
        Overrides deep-merged onto DEFAULT_PIPELINE, e.g.
        run_pipeline(..., sorting={"nblocks": 5}).
    """
    pipeline = deep_merge(DEFAULT_PIPELINE, pipeline_kwargs)
    si.set_global_job_kwargs(**pipeline["job_kwargs"])

    # Resolve to absolute before any chdir below, so relative CLI paths still work.
    phys_paths = [phys_path] if isinstance(phys_path, (str, Path)) else list(phys_path)
    phys_paths = [Path(p).resolve() for p in phys_paths]
    output_dir = Path(output_dir).resolve() if output_dir is not None else phys_paths[0]
    if output_dir.is_file():
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Keep every intermediate write off the home partition ─────────────
    # kilosort4's intermediates are ~the size of the raw recording (hundreds
    # of GB), and home partition quotas are too small.
    # Three separate things have to be redirected:
    #   1. spikeinterface's cache (defaults to tempfile.gettempdir())
    #   2. TMPDIR/TMP/TEMP, which numpy memmap, torch and kilosort4 consult
    #   3. the cwd, since sorters and friends write relative paths there
    tmp_dir = Path(tmp_dir).resolve() if tmp_dir is not None else output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for var in ("TMPDIR", "TMP", "TEMP"):
        os.environ[var] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)
    si.set_global_tmp_folder(tmp_dir)
    os.chdir(output_dir)

    print(f"output_dir : {output_dir}")
    print(f"tmp_dir    : {tmp_dir}")

    # ── 1/4 Brain-state scoring ──────────────────────────────────────────
    # Deliberately before concatenation: the 10 s spectrogram window and the
    # EMG correlation window would otherwise straddle session junctions and
    # smear one recording's LFP into its neighbour's states.
    state_cfg = pipeline.get("state_scoring") or {}
    state_results = []
    if skip_statescoring or not state_cfg.get("enabled", True):
        why = "skip_statescoring=True" if skip_statescoring else "disabled in config"
        print(f"[1/4] {why}, skipping brain-state scoring...")
    else:
        print(f"[1/4] scoring brain states for {len(phys_paths)} recording(s)...")
        for path in phys_paths:
            print(f"    {path}")
            state_results.append(
                score_session(
                    path,
                    output_dir,
                    state_cfg,
                    phys_type=phys_type,
                    stream_name=stream_name,
                    exclude_channels=pipeline.get("bad_channels") or [],
                )
            )
        print("[1/4] state scoring complete.")

    # ── 2/4 Pre-processing ───────────────────────────────────────────────
    if skip_preprocessing:
        print("[2/4] skip_preprocessing=True, reusing existing concatenated binary...")
    else:
        print("[2/4] loading and concatenating recording(s)...")
        preprocess(
            phys_paths,
            output_dir,
            phys_type=phys_type,
            stream_name=stream_name,
            preprocessing=pipeline.get("preprocessing"),
            dtype=pipeline["binary"]["dtype"],
            align_tolerance_um=pipeline["concatenation"]["align_tolerance_um"],
            sampling_frequency_max_diff=(
                pipeline["concatenation"]["sampling_frequency_max_diff"]
            ),
        )
        print("[2/4] pre-processing complete.")

    rec, info = load_concatenated(output_dir)

    # Only now are the sample offsets known, so this is where per-session
    # state intervals can be mapped onto the concatenated clock.
    if state_results:
        merge_to_concatenated_time(state_results, info, output_dir)

    bad_channels = list(pipeline.get("bad_channels") or [])
    detect_cfg = pipeline.get("detect_bad_channels") or {}
    if detect_cfg.get("enabled"):
        record_path = output_dir / BAD_CHANNELS_NAME
        if skip_sorting and record_path.exists():
            # Re-detecting here could disagree with what was actually sorted
            # (different config, or a change to spikeinterface), which would
            # silently build the analyzer on the wrong channel set.
            applied = json.loads(record_path.read_text())["applied"]
            print(f"    reusing {len(applied)} bad channel(s) from {BAD_CHANNELS_NAME}")
            bad_channels = applied
        else:
            print("    detecting bad channels...")
            bad_channels += detect_bad_channels_auto(
                rec, output_dir, detect_cfg, manual=bad_channels
            )

    # Resolve up front so a bad entry fails now rather than after sorting.
    _, bad_channel_ids = resolve_bad_channels(bad_channels, info)

    # ── 3/4 Spike sorting ────────────────────────────────────────────────
    if skip_sorting:
        print("[3/4] skip_sorting=True, reusing existing kilosort4 output...")
    else:
        print("[3/4] starting kilosort4...")
        run_kilosort4(output_dir, info, pipeline.get("sorting"), bad_channels)
        print("[3/4] sorting complete.")

    # ── 4/4 Post-processing ──────────────────────────────────────────────
    if skip_postprocessing:
        print("[4/4] skip_postprocessing=True, stopping after sorting.")
        print("\n✓ Pipeline complete.")
        return None

    print("[4/4] creating sorting analyzer...")
    # Reload rather than reusing the object above: the analyzer holds a
    # reference to the recording, and re-opening keeps that pointed at the
    # on-disk binary regardless of which stages ran.
    rec, _ = load_concatenated(output_dir)
    analyzer = postprocess(rec, output_dir, pipeline["postprocessing"], bad_channel_ids)
    print("[4/4] analyzer saved.")

    print("\n✓ Pipeline complete.")
    return analyzer
