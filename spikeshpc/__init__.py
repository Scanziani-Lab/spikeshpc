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
from .decoder import (
    Decoded,
    DecoderData,
    DecoderRun,
    EncodingModel,
    ShuffleTest,
    decode,
    fit_encoding_model,
    plot_decoded,
    plot_encoding_model,
    plot_error,
    plot_shuffle,
    prepare_decoder_data,
    run_decoder,
    shuffle_test,
    split_train_test,
    state_interval_mask,
)
from .drift import drift_at_junction, plot_drift
from .io import (
    StateScoring,
    detect_phys_type,
    load_concatenated,
    load_states,
    open_stream,
    probe_start_time,
    read_openephys_synced,
    read_recording,
    sync_clock_problem,
)
from .pipeline import run_pipeline
from .split import SessionSplit, save_splits, split_run
from .states import (
    frames_in_states,
    score_recording,
    score_session,
    slice_recording_to_states,
    times_in_states,
)
from .editor import StateEditor, show_state_editor
from .widgets import show_state_epochs

__all__ = [
    "DEFAULT_PIPELINE",
    "Decoded",
    "DecoderData",
    "DecoderRun",
    "EncodingModel",
    "SessionSplit",
    "ShuffleTest",
    "StateScoring",
    "decode",
    "deep_merge",
    "detect_phys_type",
    "drift_at_junction",
    "fit_encoding_model",
    "frames_in_states",
    "load_concatenated",
    "load_states",
    "open_stream",
    "probe_start_time",
    "read_openephys_synced",
    "sync_clock_problem",
    "plot_decoded",
    "plot_drift",
    "plot_encoding_model",
    "plot_error",
    "plot_shuffle",
    "prepare_decoder_data",
    "read_recording",
    "run_decoder",
    "run_pipeline",
    "save_splits",
    "score_recording",
    "score_session",
    "shuffle_test",
    "split_train_test",
    "state_interval_mask",
    "StateEditor",
    "show_state_editor",
    "show_state_epochs",
    "slice_recording_to_states",
    "split_run",
    "times_in_states",
]
