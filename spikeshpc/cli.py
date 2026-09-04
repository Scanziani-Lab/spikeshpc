"""Command line entry point."""

import argparse
import json
from pathlib import Path

from .config import BAD_CHANNELS_NAME, DEFAULT_PIPELINE, deep_merge
from .pipeline import run_pipeline


def build_parser():
    parser = argparse.ArgumentParser(
        prog="spikeshpc",
        description="Spike sorting pipeline for use with HPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # single SpikeGLX run, format and stream auto-detected
  spikeshpc /data/mouse1_g0 --output_dir /scratch/mouse1

  # concatenate three runs (sync channel dropped, sample counts recorded)
  spikeshpc /data/m1_g0 /data/m1_g1 /data/m1_g2 --output_dir /scratch/m1

  # re-run only post-processing against a finished sort
  spikeshpc /data/m1_g0 --output_dir /scratch/m1 \\
      --skip_statescoring --skip_preprocessing --skip_sorting

  # re-sort with dead channels excluded, reusing the concatenated binary
  spikeshpc /data/m1_g0 --output_dir /scratch/m1 \\
      --skip_statescoring --skip_preprocessing --bad_channels AP191 AP192 287

  # brain-state scoring only
  spikeshpc /data/m1_g0 /data/m1_g1 --output_dir /scratch/m1 \\
      --skip_preprocessing --skip_sorting --skip_postprocessing

outputs (in --output_dir):
  states/            per-session WAKE/NREM/REM intervals + metrics, and
                     states_concatenated.json on the concatenated clock
  concatenated.bin   combined binary handed to kilosort4
  concat_info.json   per-session sample counts + offsets, channel alignment
  chanMap.mat        channel map in kilosort's format
  probe.json         probe geometry, used to reload the binary
  bad_channels.json  per-channel labels + the set applied (autodetect only)
  kilosort4/         kilosort4 results (phy format)
  analyzer.zarr      spikeinterface sorting analyzer
            """,
    )
    parser.add_argument(
        "phys_path",
        type=Path,
        nargs="*",
        help=(
            "Recording session directory or raw binary file. "
            "Pass several to concatenate them before sorting. "
            "May instead be set as run.phys_paths in --pipeline_config."
        ),
    )
    parser.add_argument(
        "--phys_type",
        type=str,
        default=None,
        choices=["spikeglx", "openephysbinary"],
        help="Acquisition system (default: auto-detect from sidecar files)",
    )
    parser.add_argument(
        "--stream_name",
        type=str,
        default=None,
        help=(
            "Name of raw phys stream saved by OE/SpikeGLX -- e.g. 'imec0.ap', or "
            "'Record Node 101#Neuropix-PXI-100.ProbeA' (default: auto-detect the "
            "AP-band stream; required if the session has more than one probe)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write states/binary/chanMap/sorting/analyzer (default: first phys_path)",
    )
    parser.add_argument(
        "--tmp_dir",
        type=Path,
        default=None,
        help=(
            "Scratch space for intermediate files (default: <output_dir>/tmp). "
            "Keep this on a partition with room for kilosort4's intermediates "
            "-- roughly the size of the raw recording."
        ),
    )
    parser.add_argument(
        "--skip_statescoring",
        action="store_true",
        help="Reuse the brain-state scoring already in <output_dir>/states.",
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Reuse the concatenated binary already in --output_dir.",
    )
    parser.add_argument(
        "--skip_sorting",
        action="store_true",
        help="Reuse the kilosort4 output already in --output_dir.",
    )
    parser.add_argument(
        "--skip_postprocessing",
        action="store_true",
        help="Stop after sorting; do not build the sorting analyzer.",
    )
    parser.add_argument(
        "--bad_channels",
        nargs="+",
        default=None,
        metavar="CH",
        help=(
            "Channels to exclude from sorting, as channel ids listed in "
            "concat_info.json (e.g. --bad_channels AP191 AP192) or 0-based "
            "rows of concatenated.bin. Overrides bad_channels in "
            "--pipeline_config."
        ),
    )
    parser.add_argument(
        "--detect_bad_channels",
        action="store_true",
        help=(
            "Also flag bad channels automatically with si.detect_bad_channels, "
            "unioned with --bad_channels. Per-channel labels are written to "
            f"{BAD_CHANNELS_NAME} for review."
        ),
    )
    parser.add_argument(
        "--pipeline_config",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file with pipeline overrides "
            '(e.g. {"sorting": {"nblocks": 5}}), deep-merged onto DEFAULT_PIPELINE.'
        ),
    )
    return parser


def load_pipeline_config(path):
    """Read a --pipeline_config JSON file, reporting where a bad one breaks.

    json.load's own error names neither the file nor the offending text, which
    is a poor way to lose a queued job.
    """
    path = Path(path)
    text = path.read_text()
    try:
        overrides = json.loads(text)
    except json.JSONDecodeError as e:
        lines = text.splitlines()
        context = []
        for n in range(max(e.lineno - 3, 1), min(e.lineno + 1, len(lines)) + 1):
            marker = ">>" if n == e.lineno else "  "
            context.append(f"  {marker} {n:>3} | {lines[n - 1]}")
            if n == e.lineno:
                context.append(f"       {' ' * len(str(n))} | {' ' * (e.colno - 1)}^")
        raise SystemExit(
            f"ERROR: {path} is not valid JSON.\n"
            f"  {e.msg} at line {e.lineno}, column {e.colno}\n"
            + "\n".join(context)
            + "\n\n  'Expecting ',' delimiter' almost always means the line above "
            "is missing\n  a trailing comma. A comma after the LAST entry in an "
            "object is also\n  invalid JSON -- unlike Python.\n"
            f"  Check the whole file with: python -m json.tool {path}"
        ) from e

    if not isinstance(overrides, dict):
        raise SystemExit(
            f"ERROR: {path} must contain a JSON object, got "
            f"{type(overrides).__name__}."
        )

    # A misspelled top-level key would otherwise be accepted in silence and
    # the run would quietly use defaults for whatever it was meant to set.
    unknown = sorted(set(overrides) - set(DEFAULT_PIPELINE))
    if unknown:
        raise SystemExit(
            f"ERROR: {path} has unrecognised top-level key(s): {unknown}\n"
            f"  Valid keys are: {sorted(DEFAULT_PIPELINE)}"
        )
    return overrides


def main(argv=None):
    args = build_parser().parse_args(argv)

    pipeline_overrides = {}
    if args.pipeline_config is not None:
        pipeline_overrides = load_pipeline_config(args.pipeline_config)

    if args.bad_channels is not None:
        # argparse hands back strings; digit-only tokens are row indices, and
        # anything else is a channel id.
        pipeline_overrides["bad_channels"] = [
            int(c) if c.lstrip("-").isdigit() else c for c in args.bad_channels
        ]

    if args.detect_bad_channels:
        # deep_merge so the flag turns detection on without dropping any
        # detect_bad_channels tuning already set in --pipeline_config.
        pipeline_overrides = deep_merge(
            pipeline_overrides, {"detect_bad_channels": {"enabled": True}}
        )

    # The "run" block lets one config drive a laptop and a cluster job alike.
    # Anything given on the command line wins over it, so a config can be
    # overridden for a one-off without editing the file.
    run_cfg = pipeline_overrides.pop("run", None) or {}

    phys_path = args.phys_path or run_cfg.get("phys_paths") or []
    if not phys_path:
        build_parser().error(
            "no recordings given: pass them as arguments, or set "
            'run.phys_paths in --pipeline_config'
        )

    def pick(cli_value, key):
        return cli_value if cli_value is not None else run_cfg.get(key)

    # store_true flags are False rather than None when absent, so OR them:
    # the config can turn a stage off, the flag can too, neither turns it on.
    def skip(cli_flag, key):
        return bool(cli_flag) or bool(run_cfg.get(key, False))

    return run_pipeline(
        phys_path=[Path(p) for p in phys_path],
        phys_type=pick(args.phys_type, "phys_type"),
        stream_name=pick(args.stream_name, "stream_name"),
        output_dir=pick(args.output_dir, "output_dir"),
        tmp_dir=pick(args.tmp_dir, "tmp_dir"),
        skip_statescoring=skip(args.skip_statescoring, "skip_statescoring"),
        skip_preprocessing=skip(args.skip_preprocessing, "skip_preprocessing"),
        skip_sorting=skip(args.skip_sorting, "skip_sorting"),
        skip_postprocessing=skip(args.skip_postprocessing, "skip_postprocessing"),
        **pipeline_overrides,
    )


if __name__ == "__main__":
    main()
