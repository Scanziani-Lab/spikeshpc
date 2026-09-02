"""Spike sorting pipeline for use with HPC.

Four independently re-runnable stages, each gated by a --skip_* flag:

  1. state scoring    per recording and BEFORE concatenation, LFP is scored
                      into WAKE/NREM/REM after Watson et al. 2016.
  2. pre-processing   spikeinterface loads the recording(s), auto-detecting
                      SpikeGLX vs OpenEphys, drops the SpikeGLX sync channel
                      (SY0), aligns channels by probe geometry, concatenates,
                      and writes a single flat binary plus chanMap.mat /
                      probe.json / concat_info.json.
  3. sorting          kilosort4 is driven directly (not through
                      si.run_sorter) on that binary with DEFAULT_SETTINGS.
  4. post-processing  spikeinterface reads the kilosort4 output back and
                      builds/saves the sorting analyzer.

Everything the later stages need is written to disk by the earlier ones, so
any stage can be re-run on its own against an existing output_dir.

Pipeline parameters live in spikeshpc.config.DEFAULT_PIPELINE and can be
overridden per-run either by passing kwargs to run_pipeline() directly (e.g.
from a notebook) or, from the CLI, via a --pipeline_config JSON file that is
deep-merged on top of the defaults.
"""

from .config import DEFAULT_PIPELINE, deep_merge
from .drift import drift_at_junction, plot_drift
from .io import detect_phys_type, load_concatenated, read_recording
from .pipeline import run_pipeline
from .split import SessionSplit, save_splits, split_run
from .states import score_recording, score_session

__all__ = [
    "DEFAULT_PIPELINE",
    "SessionSplit",
    "deep_merge",
    "detect_phys_type",
    "drift_at_junction",
    "load_concatenated",
    "plot_drift",
    "read_recording",
    "run_pipeline",
    "save_splits",
    "score_recording",
    "score_session",
    "split_run",
]
