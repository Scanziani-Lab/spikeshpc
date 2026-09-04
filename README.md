# spikeshpc

Neuropixels spike-sorting pipeline for HPC: brain-state scoring, channel-aligned
concatenation, kilosort4, and spikeinterface post-processing.

Four independently re-runnable stages, each gated by a `--skip_*` flag:

| # | Stage | Writes |
|---|-------|--------|
| 1 | **state scoring** — per session, *before* concatenation | `states/` |
| 2 | **pre-processing** — load, align channels by probe geometry, concatenate | `concatenated.bin`, `concat_info.json`, `chanMap.mat`, `probe.json` |
| 3 | **sorting** — kilosort4, driven directly | `kilosort4/` |
| 4 | **post-processing** — sorting analyzer | `analyzer.zarr` |

Everything a later stage needs is written to disk by the earlier ones, so any stage
can be re-run on its own against an existing `--output_dir`.

## Layout

```
pyproject.toml
hpc_load_sort_post.py     entry point that needs no install
spikeshpc/                the package
slurm/                    multisession_sorting.slurm, pipeline_config.example.json
containers/               si_kilosort4.def (apptainer image definition)
tests/
```

## Install

```bash
pip install -e .              # numpy, scipy, spikeinterface[full], probeinterface
pip install -e ".[sorting]"   # ...plus kilosort4
```

kilosort4 and torch are large and CUDA-specific; on the cluster they come from the
container image rather than pip, which is why they sit in an optional extra.

No install is needed to run from a checkout — `hpc_load_sort_post.py` puts the repo
root on `sys.path` itself, which is how the slurm jobs invoke it inside the container.

## Use

```bash
# whole pipeline; acquisition system and stream auto-detected
spikeshpc /data/m1_g0 /data/m1_g1 --output_dir /scratch/m1

# re-sort with dead channels excluded, reusing everything before sorting
spikeshpc /data/m1_g0 --output_dir /scratch/m1 \
    --skip_statescoring --skip_preprocessing --bad_channels 191 192

# brain-state scoring only
spikeshpc /data/m1_g0 /data/m1_g1 --output_dir /scratch/m1 \
    --skip_preprocessing --skip_sorting --skip_postprocessing
```

`python -m spikeshpc` and `python hpc_load_sort_post.py` are equivalent entry points.

Parameters live in `spikeshpc.config.DEFAULT_PIPELINE` and are overridden per run with
`--pipeline_config` (see `slurm/pipeline_config.example.json`) or as kwargs to
`run_pipeline()` from a notebook.

On the cluster, see `slurm/multisession_sorting.slurm`.

## Brain-state scoring

WAKE/NREM/REM from LFP, after [Watson et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4873379/).
Runs per session *before* concatenation, so the 10 s spectrogram window never straddles
a junction between sessions.

Three signals in 1 s steps: the first principal component of the z-scored log
spectrogram (high in NREM), the 5–10 Hz / 2–16 Hz power ratio (high in REM), and a
pseudo-EMG from zero-lag correlations between 300–600 Hz signals at separated sites
(high in waking movement). Each is split at the trough between its two modes.

This is a reimplementation from the published description, **not** a port of buzcode's
`SleepScoreMaster`, and it has not been validated against hand-scored data. See the
module docstring in `spikeshpc/states.py` for the specific deviations. Theta is a
hippocampal signal — on a probe spanning several structures, set
`state_scoring.theta_channels` explicitly rather than averaging over everything.

### OptiTrack movement veto

Optional. Gross movement proves the animal is awake, so it overrules an NREM/REM call;
stillness proves nothing and is ignored, since a mouse can sit motionless and wide
awake. This is the only signal here that catches running, whose hippocampal theta is
otherwise indistinguishable from REM's.

The immobility threshold is a bimodal split on log10 speed, so breathing and postural
sway — which keep movement well off zero — are handled without a hand-tuned floor.

`spikeshpc/tracking.py` reads the Motive CSV directly; the `optitrack` package is not
a dependency.

## Utilities

- `spikeshpc-drift <output_dir>` — kilosort's drift step across each concatenation
  junction, in µm and in units of the probe's own row pitch. Runs automatically after
  sorting when there is more than one session.
- `spikeshpc-split <output_dir>` — carve the concatenated recording, sorting, analyzer
  and state intervals back into one object per original recording, on a common
  session-local clock.
