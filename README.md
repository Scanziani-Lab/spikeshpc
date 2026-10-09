# spikeshpc

Processing pipeline for analyzing Neuropixels recordings along with OptiTrack head tracking.
Can be run on a local desktop or HPC. Also includes wrapper functions for SpikeInterface,
Bombcell, UnitMatch and more for post-processing analysis

Initial pipeline contains four stages, each skippable with a flag:

| # | Stage | Writes |
| - | ----- | ------ |
| 1 | **state scoring** | `states/` |
| 2 | **pre-processing** | `preprocessed.bin`, `concat_info.json`, `chanMap.mat`, `probe.json` |
| 3 | **sorting** | `kilosort4/` |
| 4 | **post-processing** | `analyzer.zarr` |

Each stage writes outputs to disk so later stages can be run independently
using existing `--output_dir`.

## Layout

```text
pyproject.toml
pipeline_config.example.json   example run config - works locally and on the cluster
hpc_load_sort_post.py          entry point that needs no install
spikeshpc/                     
slurm/                         multisession_sorting.slurm (HPC job wrapper)
containers/                    apptainer image definition for use on HPC
envs/unitmatch/                Python 3.12 environment for UnitMatch (see Install)
tests/
```

## Run config file (for local and HPC runs)

Use one JSON file per run. The same file can drive a local run or a slurm job.
Start from template [`pipeline_config.example.json`](pipeline_config.example.json):

- **`run`** specifies which data: `phys_paths` (one or more recording directories or raw
  binaries; multiple are concatenated), `output_dir`, `tmp_dir`, the four
  `skip_*` stage flags, and `bind_paths` (cluster only, see below). `phys_type` and
  `stream_name` may also go here; both are auto-detected when left out.
- **everything else** specifies how to process it: `job_kwargs`, `preprocessing`,
  `state_scoring`, `bad_channels`, `detect_bad_channels`, `sorting` (kilosort4
  settings), etc. These are merged onto `spikeshpc.config.DEFAULT_PIPELINE`,
  overwriting defaults when there are conflicts

Paths are written for whichever machine runs the pipeline, so keep one config
per run per machine. Keep run configs separate from repo.
Before a first run, check the following:

- `run.phys_paths`, `run.output_dir`, `run.tmp_dir`
- `job_kwargs.n_jobs` — <= the number of available cores (on slurm, `--cpus-per-task`)
- `state_scoring.movement` — the example enables an OptiTrack movement veto with a
  cluster path; point `optitrack_csv` at your tracking files or set `"enabled": false`

Input to the command line overrides the `run` block in the config file, so a config file
can be reused for a one-off run (e.g. `--skip_sorting`) without editing.

## Running locally

### Install

With [uv](https://docs.astral.sh/uv/), from the repo root:

```bash
uv sync         # .venv on Python 3.14: the package, kilosort4, CUDA torch, notebook/GUI tools, pytest
uv run pytest
```

To use the environment in Jupyter notebooks, register it once as a kernel:

```bash
uv run python -m ipykernel install --user --name spikeshpc --display-name "spikeshpc (Python 3.14)"
```

UnitMatch has its own environment, [`envs/unitmatch`](envs/unitmatch/pyproject.toml) with
Python 3.12: UnitMatchPy pins numpy<2, and spikeinterface 0.105 onwards needs numpy 2. It
installs this repo and UnitMatchPy from a clone of my [UnitMatch fork](https://github.com/Scanziani-Lab/UnitMatch)
next to the repo, both editable.

```bash
git clone --filter=blob:none -b spikeshpc https://github.com/Scanziani-Lab/UnitMatch.git ../UnitMatch
uv sync --directory envs/unitmatch
uv run --directory envs/unitmatch python -m ipykernel install --user --name unitmatch --display-name "unitmatch (Python 3.12)"
```

Or with pip:

```bash
pip install -e .              # numpy, scipy, spikeinterface[full], probeinterface
pip install -e ".[sorting]"   # ...plus kilosort4
pip install -e ".[video]"     # ...plus OpenCV, for the OptiTrack heading video widget
```

If `pip` gives you a CPU-only torch, install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/)
first. State scoring and pre-processing don't need kilosort or PyTorch.

### Run

```bash
spikeshpc --pipeline_config pipeline_config.json
```

or, without a config, give the recordings and flags directly:

```bash
# whole pipeline; acquisition system and stream auto-detected
spikeshpc /data/session0 /data/session1 --output_dir /scratch/m1

# re-sort with dead channels excluded, reusing everything before sorting
spikeshpc /data/session0 --output_dir /scratch/m1 \
    --skip_statescoring --skip_preprocessing --bad_channels 191 192

# brain-state scoring only
spikeshpc /data/session0 /data/session1 --output_dir /scratch/m1 \
    --skip_preprocessing --skip_sorting --skip_postprocessing
```

`python -m spikeshpc` and `python hpc_load_sort_post.py` are equivalent;
the latter doesn't need installation since it puts the repo root on `sys.path`.
`spikeshpc --help` lists every flag.

On Windows, write paths in the JSON with forward slashes (`"D:/data/m1_g0"`) or
doubled backslashes (`"D:\\data\\m1_g0"`) to avoid invalid JSON.

From a Jupyter notebook, call `run_pipeline()` with the same pieces as keyword arguments:

```python
from spikeshpc.pipeline import run_pipeline

run_pipeline(
    ["/data/session0", "/data/session1"],
    output_dir="/scratch/m1",
    skip_statescoring=True,
    sorting={"nblocks": 5},          # any top-level config key other than "run"
)
```

## Running on an HPC (slurm + apptainer)

On the cluster the job runs `hpc_load_sort_post.py` from a clone of this repo inside
a container that provides spikeinterface, kilosort4 and CUDA.

**1. Clone the repo** somewhere the compute nodes can see (should be small enough to
easily fit on home mount):

```bash
git clone https://github.com/Scanziani-Lab/spikeshpc.git ~/spikeshpc
```

**2. Build the container** do this once on a Linux host with fakeroot such as the login node,
on a mount such as /scratch with a large file size limit (the rsulting `.sif` is ~10 GB):

```bash
apptainer build --fakeroot /scratch/user/$USER/containers/si_kilosort4.sif containers/si_kilosort4.def
```

The header of [`containers/si_kilosort4.def`](containers/si_kilosort4.def) has a
check to run on a GPU node afterwards; only matters for newer (Blackwell) cards.

**3. Write the run config** — copy [`pipeline_config.example.json`](pipeline_config.example.json)
to `/scratch` or `/home` and fill in the cluster paths as [above](#run-config-file-for-local-and-hpc-runs).
Keep `run.tmp_dir` and `run.output_dir` off the `/home` mount due to storage restrictions.

The `.sif` container can only see directories that are bind-mounted into it. Each job derives
them from the config (recordings, output and scratch directories, and the OptiTrack
path templates) so `run.bind_paths` can normally stay empty. Set it only if something
lives outside those, e.g. behind a symlink.

**4. Edit [`slurm/multisession_sorting.slurm`](slurm/multisession_sorting.slurm)** with the
following parameters:

- `#SBATCH` header: set the partition, `--cpus-per-task` (keep `job_kwargs.n_jobs` at or
  below it), memory, time, and the `--output`/`--error` log directory, which must already exist
- `REPO_DIR` — repository clone from step 1
- `SI_SIF` — `.sif` image from step 2
- `PIPELINE_CONFIG` — edited config JSON from step 3
- `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR`, need to be off `/home`

**5. Submit:**

```bash
sbatch slurm/multisession_sorting.slurm
```

A config can also be run inside the container from an interactive GPU session:

```bash
apptainer exec --nv --bind /scratch/user/$USER /scratch/user/$USER/containers/si_kilosort4.sif \
    python -u ~/spikeshpc/hpc_load_sort_post.py --pipeline_config /scratch/user/$USER/runs/m1.json
```

## Looking at raw traces

`spikeshpc.plot_traces` and `spikeshpc.get_traces` take the same arguments as
spikeinterface's functions of the same name. Use them instead on long recordings:

```python
from spikeshpc import open_stream, plot_traces, get_traces

rec = open_stream(phys_path, "openephysbinary", stream_name)
plot_traces(rec, time_range=(1000.0, 1000.2), relative=True, mode="map")
traces, times = get_traces(rec, (1000.0, 1000.2), relative=True, return_times=True)
```

Only the requested frames are read, and the time vector is never loaded whole.
`time_range` is on the recording's clock. For Open Ephys that is the synchronized
acquisition clock, which doesn't necessarily start at 0.
`relative=True` counts from the first sample instead.

Open Ephys `timestamps.npy` is also memory-mapped rather than read into RAM
(~11 GB for 12.75 h at 30 kHz). When several sessions are concatenated, their
timestamps are joined once into `output_dir/sync_timestamps.npy`.

## Brain-state scoring

WAKE/NREM/REM from LFP, after [Watson et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4873379/).
Runs per session before concatenation so the 10 s spectrogram window never straddles a junction between
sessions.

Three signals in 1 s steps over a 5 s window: the first principal component of the z-scored log
spectrogram (high in NREM), the 5–10 Hz / 2–16 Hz power ratio (high in REM), and a
pseudo-EMG from zero-lag correlations between 300–600 Hz signals at separated sites
(high in waking movement). Each is split at the trough between its two peaks.

This is a reimplementation from the published description, **not** an exact port of buzcode's
`SleepScoreMaster`. It has not been validated against hand-scored data. See the
module docstring in `spikeshpc/states.py` for the specific deviations. Theta is a
hippocampal signal — on a probe spanning several structures, set `state_scoring.theta_channels`
explicitly rather than averaging over everything.

With no dedicated LFP stream (Neuropixels 2.0) the LFP and the pseudo-EMG are each resampled
from the 30 kHz AP band, and each is a full read of the raw binary: twice over a ~1 TB file
for a 12 h recording. My runs on the cluster's scratch have taken about 20 m per hour of recording.
Each pass shows a progress bar in stderr (the job's `.err` file), switched on/off by `job_kwargs.progress_bar`.

### OptiTrack movement veto

Optional. Gross movement proves the animal is awake, so it overrules a NREM/REM call.
The immobility threshold is a bimodal split on log10 speed, so breathing and postural
sway (which keep movement non-zero) are handled without a hand-tuned floor.

`spikeshpc/optitrack/io.py` reads the Motive CSV directly.

## Curating results

`spikeshpc/curation.py` contains functions to view and curate the results of the analysis pipeline

## Head-direction decoding

`spikeshpc/decoder.py` has functions to train a decoder on head-direction-tuned units.
The decoder is a sorted-spikes point-process decoder on a ring, with the units' tuning curves
as the encoding model, a Poisson likelihood per time bin and a Gaussian random walk for
the dynamics. `run_decoder()` trains on wake, tests on held-out wake against a
shuffle control, and then decodes REM/NREM.

```python
run = run_decoder(analyzer, hd_unit_ids, heading_deg, shutter_close_times,
                  scoring.intervals, test_fraction=0.3)
print(run.summary())
```

`decode_nrem=True` decodes NREM as well, judged against the same unit-permutation null as REM.
`apply_decoder` and `reference_decode` carry it to other recordings, and `plot_transfer_summary`
adds an NREM panel when there is one. NREM is long, so it costs the run's longest decode and
shuffles to match, and the random walk was fitted to waking head movement while the internal
heading moves faster in NREM: decode it again with `movement_var_deg2=np.inf` as the control.

### Ring attractor from correlations

`spikeshpc/ring.py` reads heading without tuning curves. It is influenced by ideas from SPUD
(Chaudhuri et al. 2019):

1. Pairs of units are scored by zero-lag correlation minus the mean correlation 5–10 s out.
2. Units are placed on a ring by Isomap of those scores.
3. Heading is decoded with a population vector around the ring.

A unit's place on the ring belongs to the network rather than to its tuning, so it can check
cross-day unit matching where tuning is disrupted: `compare_rings` asks whether matched units sit
at the same places on two rings, and are correlated alike in both.

```python
# band_unit_ids: GOOD + MUA units in the depth band of the head-direction structure
ring_run = run_ring(analyzer, band_unit_ids, heading_deg, shutter_close_times,
                    scoring.intervals, interval_mask=hd.interval_mask)
print(ring_run.summary())
print(ring_vs_tuning(ring_run.ring, hd).summary())   # did it find the tuned units?
```

`run_ring` masks and splits wake exactly as `run_decoder` does, so with the same settings both
hold out the same bins. Its decodes are `Decoded` objects, so `show_decoded`, `plot_error` and
`metrics_by_group` work on them.

A ring's decode is read as heading through its own alignment, or, with `RingAlignment.then`,
through matched units onto another recording's ring and that ring's alignment. The second way
doesn't need the new recording's tuning. `plot_turn_summary` and `plot_turn_sweep` take a
`"ring"` source beside the decoded one.

## Utilities

- `spikeshpc-drift <output_dir>` — kilosort's drift step across each concatenation
  junction, in µm and in units of the probe's own row pitch. Runs automatically after
  sorting when there is more than one session.
- `spikeshpc-split <output_dir>` — carve the concatenated recording, sorting, analyzer
  and state intervals back into one object per original recording, on a common
  session-local clock.
