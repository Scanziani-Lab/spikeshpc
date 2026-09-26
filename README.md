# spikeshpc

Neuropixels spike-sorting pipeline for HPC: brain-state scoring, channel-aligned
concatenation, kilosort4, and spikeinterface post-processing.

Four independently re-runnable stages, each gated by a `--skip_*` flag:

| # | Stage | Writes |
|---|-------|--------|
| 1 | **state scoring** — per session, *before* concatenation | `states/` |
| 2 | **pre-processing** — load, align channels by probe geometry, concatenate | `concatenated.bin`, `concat_info.json`, `chanMap.mat`, `probe.json` |
| 3 | **sorting** — kilosort4, driven directly, in a freshly started process | `kilosort4/` |
| 4 | **post-processing** — sorting analyzer | `analyzer.zarr` |

Everything a later stage needs is written to disk by the earlier ones, so any stage
can be re-run on its own against an existing `--output_dir`.

## Layout

```
pyproject.toml
pipeline_config.example.json   example run config -- works locally and on the cluster
hpc_load_sort_post.py          entry point that needs no install
spikeshpc/                     the package
slurm/                         multisession_sorting.slurm (cluster job wrapper)
containers/                    si_kilosort4.def (apptainer image definition)
tests/
```

## The run config

One JSON file per run. The same file can drive a local run or a slurm job.
Start from [`pipeline_config.example.json`](pipeline_config.example.json):

- **`run`** — *which* data: `phys_paths` (one or more recording directories or raw
  binaries; several are concatenated), `output_dir`, `tmp_dir`, the four
  `skip_*` stage flags, and `bind_paths` (cluster only, see below). `phys_type` and
  `stream_name` may also go here; both are auto-detected when left out.
- **everything else** — *how* to process it: `job_kwargs`, `preprocessing`,
  `state_scoring`, `bad_channels`, `detect_bad_channels`, `sorting` (kilosort4
  settings), and so on. These are deep-merged onto `spikeshpc.config.DEFAULT_PIPELINE`,
  so a config only needs the keys it changes. A misspelled top-level key is an error
  rather than silently ignored.

Paths are written for whichever machine will run the job, so in practice you keep one
config per run per machine. Copy the example rather than editing it —
`pipeline_config.json` at the repo root is git-ignored for exactly this, or keep run
configs next to the data. Before a first run, check at least:

- `run.phys_paths`, `run.output_dir`, `run.tmp_dir`
- `job_kwargs.n_jobs` — no more than the cores you have (on slurm, `--cpus-per-task`)
- `state_scoring.movement` — the example enables the OptiTrack movement veto with a
  cluster path; point `optitrack_csv` at your tracking files or set `"enabled": false`

Anything given on the command line overrides the `run` block, so a config can be
reused for a one-off (e.g. `--skip_sorting`) without editing it.

## Running locally

### Install

```bash
pip install -e .              # numpy, scipy, spikeinterface[full], probeinterface
pip install -e ".[sorting]"   # ...plus kilosort4
```

kilosort4 and torch are large and CUDA-specific, which is why they sit in an optional
extra. If `pip` gives you a CPU-only torch, install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/)
first. State scoring and pre-processing alone need neither.

### Run

```bash
spikeshpc --pipeline_config pipeline_config.json
```

or, without a config, give the recordings and flags directly:

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

`python -m spikeshpc` and `python hpc_load_sort_post.py` are equivalent entry points;
the latter needs no install, since it puts the repo root on `sys.path` itself.
`spikeshpc --help` lists every flag.

On Windows, write paths in the JSON with forward slashes (`"D:/data/m1_g0"`) or
doubled backslashes (`"D:\\data\\m1_g0"`) — a single backslash is invalid JSON.

From a notebook, call `run_pipeline()` with the same pieces as keyword arguments:

```python
from spikeshpc.pipeline import run_pipeline

run_pipeline(
    ["/data/m1_g0", "/data/m1_g1"],
    output_dir="/scratch/m1",
    skip_statescoring=True,
    sorting={"nblocks": 5},          # any top-level config key other than "run"
)
```

## Running on an HPC (slurm + apptainer)

On the cluster nothing is pip-installed: the job runs `hpc_load_sort_post.py` from a
clone of this repo inside a container that provides spikeinterface, kilosort4 and CUDA.

**1. Clone the repo** somewhere the compute nodes can see (home is fine — it only
holds code):

```bash
git clone https://github.com/Scanziani-Lab/spikeshpc.git ~/spikeshpc
```

**2. Build the container** once, on a Linux host with fakeroot such as the login node,
and keep the `.sif` on scratch:

```bash
apptainer build --fakeroot /scratch/user/$USER/containers/si_kilosort4.sif containers/si_kilosort4.def
```

The header of [`containers/si_kilosort4.def`](containers/si_kilosort4.def) has a
check to run on a GPU node afterwards; it matters for newer (Blackwell) cards.

**3. Write the run config** — copy [`pipeline_config.example.json`](pipeline_config.example.json)
to scratch (e.g. `/scratch/user/$USER/runs/m1.json`) and fill in the cluster paths as
[above](#the-run-config). Keep `run.tmp_dir` and `run.output_dir` off `/home`:
kilosort's intermediates are roughly the size of the recording, and overrunning the
home quota kills the job with no traceback. For a multi-session run, `output_dir` also
needs room for `concatenated.bin`, a copy of every session combined.

The container can only see directories that are bind-mounted into it. The job derives
those from the config — recordings, output and scratch directories, and the OptiTrack
path templates — so `run.bind_paths` can normally stay empty. Set it only if something
lives outside those, e.g. behind a symlink.

**4. Edit [`slurm/multisession_sorting.slurm`](slurm/multisession_sorting.slurm)** —
only the deployment details live there:

- `REPO_DIR` — the clone from step 1
- `SI_SIF` — the image from step 2
- `PIPELINE_CONFIG` — the config from step 3
- the `#SBATCH` header: partition, `--cpus-per-task` (keep `job_kwargs.n_jobs` at or
  below it), memory, time, and the `--output`/`--error` log directory, which must
  already exist
- `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR`, which also need to be off `/home`

**5. Submit:**

```bash
sbatch slurm/multisession_sorting.slurm
```

The script checks that the config, the image and every recording exist before
launching, so a typo fails immediately instead of after a GPU has been allocated.

To re-run part of the pipeline, set the matching `run.skip_*` flags (or change
`bad_channels`, sorting settings, …) in the config and submit again; each stage reads
what the earlier ones left in `output_dir`. The same config can also be run by hand
inside the container from an interactive GPU session:

```bash
apptainer exec --nv --bind /scratch/user/$USER /scratch/user/$USER/containers/si_kilosort4.sif \
    python -u ~/spikeshpc/hpc_load_sort_post.py --pipeline_config /scratch/user/$USER/runs/m1.json
```

## Brain-state scoring

WAKE/NREM/REM from LFP, after [Watson et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4873379/).
Runs per session *before* concatenation, so the 10 s spectrogram window never straddles
a junction between sessions.

Three signals in 1 s steps over a 5 s window: the first principal component of the z-scored log
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

## Head-direction decoding

`spikeshpc/decoder.py` decodes head direction from head-direction-tuned units: a
sorted-spikes point-process decoder on a ring, with the units' own tuning curves as
the encoding model, a Poisson likelihood per time bin and a Gaussian random walk for
the dynamics. `run_decoder()` trains on wake, tests on held-out wake against a
shuffle control, and then decodes REM, where there is no camera heading to recover.

```python
run = run_decoder(analyzer, hd_unit_ids, heading_deg, shutter_close_times,
                  scoring.intervals, test_fraction=0.3)
print(run.summary())
```

Time bins are whole numbers of camera frames, so bin edges are measured
shutter-closure timestamps and nothing is interpolated between the camera and the
decoder. The two shuffle controls answer different questions: shifting the spike
train against the heading tests whether the decoder tracks the animal (wake), while
permuting tuning curves across units tests whether the population code is real at
all (REM, where nothing can be misaligned). See `2_train_test_decoder.ipynb`.

The method follows Moritz's `run_decoder.py`, which used
`replay_trajectory_classification`; that package is unmaintained and does not install
here, and its linearized-track machinery is scaffolding a circle does not need.

## Looking at raw traces

`spikeshpc.plot_traces` and `spikeshpc.get_traces` take the same arguments as
spikeinterface's functions of the same name. Use them instead on long recordings:

```python
from spikeshpc import open_stream, plot_traces, get_traces

rec = open_stream(phys_path, "openephysbinary", stream_name)
plot_traces(rec, time_range=(1000.0, 1000.2), relative=True, mode="map")
traces, times = get_traces(rec, (1000.0, 1000.2), relative=True, return_times=True)
```

Only the requested frames are read, and the time vector is never loaded whole. A
range is refused, not read, in two cases. The first is when it resolves to more
samples than it can hold, which happens when a clock restarts mid-recording
(spikeinterface would read hours of data there). The second is when the traces
would be larger than `max_gb` (default 1). `time_range` is on the recording's own
clock. For Open Ephys that is the synchronized acquisition clock, which does not
start at 0; `relative=True` counts from the first sample instead.

Open Ephys `timestamps.npy` is also memory-mapped now rather than read into RAM
(~11 GB for 12.75 h at 30 kHz). When several sessions are concatenated, their
timestamps are joined once into `output_dir/sync_timestamps.npy`.

## Utilities

- `spikeshpc-drift <output_dir>` — kilosort's drift step across each concatenation
  junction, in µm and in units of the probe's own row pitch. Runs automatically after
  sorting when there is more than one session.
- `spikeshpc-split <output_dir>` — carve the concatenated recording, sorting, analyzer
  and state intervals back into one object per original recording, on a common
  session-local clock.
