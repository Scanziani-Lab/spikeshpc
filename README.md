# spikeshpc

Processing pipeline for Neuropixels recordings along with OptiTrack head tracking on HPC: 
brain-state scoring, channel-aligned concatenation, pre-processing, kilosort4 
and spikeinterface post-processing.

Four independent re-runnable stages, each with a `--skip_*` flag:

| # | Stage | Writes |
|---|-------|--------|
| 1 | **state scoring** — per session, before concatenation. Uses an adapted Buzsaki Lab algorithm (from Buzcode) | `states/` |
| 2 | **pre-processing** — load, align channels, concatenate, pre-process | `preprocessed.bin`, `concat_info.json`, `chanMap.mat`, `probe.json` |
| 3 | **sorting** — kilosort4 in a freshly started process | `kilosort4/` |
| 4 | **post-processing** — builds a spike interface sorting analyzer | `analyzer.zarr` |

Everything a later stage needs is written to disk by the earlier ones, so any stage
can be re-run on its own using existing `--output_dir`.

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

## Run config file (for local and HPC runs)

Use one JSON file per run. The same file can drive a local run or a slurm job.
Start from [`pipeline_config.example.json`](pipeline_config.example.json):

- **`run`** — *which* data: `phys_paths` (one or more recording directories or raw
  binaries; multiple are concatenated), `output_dir`, `tmp_dir`, the four
  `skip_*` stage flags, and `bind_paths` (cluster only, see below). `phys_type` and
  `stream_name` may also go here; both are auto-detected when left out.
- **everything else** — *how* to process it: `job_kwargs`, `preprocessing`,
  `state_scoring`, `bad_channels`, `detect_bad_channels`, `sorting` (kilosort4
  settings), and so on. These are merged onto `spikeshpc.config.DEFAULT_PIPELINE`,
  overwriting defaults when there are conflicts. A misspelled top-level key raises
   an error

Paths are written for whichever machine will run the job, so in practice you keep one
config per run per machine. Copy the example rather than editing it 
(`pipeline_config.json` at the repo root is git-ignored), or keep run
configs next to the data. Before a first run, check the following:

- `run.phys_paths`, `run.output_dir`, `run.tmp_dir`
- `job_kwargs.n_jobs` — no more than the cores you have (on slurm, `--cpus-per-task`)
- `state_scoring.movement` — the example enables an OptiTrack movement veto with a
  cluster path; point `optitrack_csv` at your tracking files or set `"enabled": false`

Anything given on the command line overrides the `run` block in the config file, 
so a config can be reused for a one-off (e.g. `--skip_sorting`) without editing.

## Running locally

### Install

```bash
pip install -e .              # numpy, scipy, spikeinterface[full], probeinterface
pip install -e ".[sorting]"   # ...plus kilosort4
```

If `pip` gives you a CPU-only torch, install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/)
first. State scoring and pre-processing alone don't need kilosort or PyTorch.

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

`python -m spikeshpc` and `python hpc_load_sort_post.py` are equivalent;
the latter needs no install since it puts the repo root on `sys.path`.
`spikeshpc --help` lists every flag.

On Windows, write paths in the JSON with forward slashes (`"D:/data/m1_g0"`) or
doubled backslashes (`"D:\\data\\m1_g0"`) to avoid invalid JSON.

From a jupyter notebook, call `run_pipeline()` with the same pieces as keyword arguments:

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
check to run on a GPU node afterwards; it matters for newer (Blackwell) cards.

**3. Write the run config** — copy [`pipeline_config.example.json`](pipeline_config.example.json)
to `/scratch` or `/home` and fill in the cluster paths as [above](#the-run-config). 
Keep `run.tmp_dir` and `run.output_dir` off `/home` due to file size restrictions.

The `.sif` container can only see directories that are bind-mounted into it. The job derives
them from the config — recordings, output and scratch directories, and the OptiTrack
path templates — so `run.bind_paths` can normally stay empty. Set it only if something
lives outside those, e.g. behind a symlink.

**4. Edit [`slurm/multisession_sorting.slurm`](slurm/multisession_sorting.slurm)** —
only the deployment details live there:

- `REPO_DIR` — repository clone from step 1
- `SI_SIF` — `.sif` image from step 2
- `PIPELINE_CONFIG` — edited config JSON from step 3
- the `#SBATCH` header: set the partition, `--cpus-per-task` (keep `job_kwargs.n_jobs` at or
  below it), memory, time, and the `--output`/`--error` log directory, which must already exist
- `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR`, need to be off `/home`

**5. Submit:**

```bash
sbatch slurm/multisession_sorting.slurm
```

The script checks that the config, the image and every recording exist before
launching.

To re-run part of the pipeline, set the matching `run.skip_*` flags (or change
`bad_channels`, sorting settings, etc.) in the config and submit again; each stage reads
what the earlier ones left in `output_dir`. The same config can also be run by hand
inside the container from an interactive GPU session:

```bash
apptainer exec --nv --bind /scratch/user/$USER /scratch/user/$USER/containers/si_kilosort4.sif \
    python -u ~/spikeshpc/hpc_load_sort_post.py --pipeline_config /scratch/user/$USER/runs/m1.json
```

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
Each pass shows a progress bar on stderr (the job's `.err` file), switched on/off by `job_kwargs.progress_bar`.

### OptiTrack movement veto

Optional. Gross movement proves the animal is awake, so it overrules an NREM/REM call;
stillness proves nothing and is ignored. This is the only signal here that catches running, 
whose hippocampal theta is otherwise indistinguishable from REM.

The immobility threshold is a bimodal split on log10 speed, so breathing and postural
sway (which keep movement non-zero) are handled without a hand-tuned floor.

`spikeshpc/tracking.py` reads the Motive CSV directly; the `optitrack` package is not
a dependency.

## Curating results

`0_load_inspect.ipynb` runs bombcell, UnitRefine and SLAy on a session's sorting, then
opens spikeinterface-gui with outputs displayed (`spikeshpc/curation.py`):

- each tool's call is shown in a sortable column in the unit table
- the quality label starts filled in wherever bombcell and UnitRefine agree and blank
  where they disagree
- SLAy's merge proposals are listed in the Merge tab with their scores, to accept
  (ctrl+a) or ignore.

GUI reads everything from the session's `curation/` folder:

| file | contents |
|---|---|
| `sorting.json` | the sorting the folder belongs to: unit ids, spikes per unit, a templates checksum |
| `unitrefine_labels.csv`, `slay_merges.npz` | UnitRefine's and SLAy's results, next to bombcell's own files |
| `automated_labels.csv` | every tool's call, one row per unit: what the GUI shows |
| `sigui_curation.json` | written by the GUI's "Save curation"; the GUI resumes from it (delete it to start over) |
| `curated_units.csv` | the automated calls plus your label, removal, merge group and split, per unit |
| `sigui.log` | the GUI process's output, for when the window does not appear |

```bash
python -m spikeshpc.curation <output_dir> [--curation-dir DIR]   # started by launch_gui()
```

Everything in `curation/` is keyed by unit ID and each re-sort will have unique unit IDs. 
Because of this, each curation folder is tied to the sorting it was made from. After a pipeline re-run,
make sure to move the old folder aside rather than loading the last sorting's labels.

Things to note on Windows:

- spikeinterface recomputes `valid_unit_periods` whenever units are split or merged, in
  a process pool it has to send the whole sorting to. At 10^8 spikes that fails
  (`OSError 22`), so SLAy's parameter search and `si.apply_curation` run inside
  `detached_extensions(analyzer, "valid_unit_periods")`.
- An analyzer opened here cannot reopen its own recording. `analyzer_recording()`
  rebuilds it from the binary, re-applying the analyzer's own filters, or none if it had
  none.

The Merge tab is filled through spikeinterface-gui 0.13.1's internals, and
`tests/test_curation.py` fails if an upgrade moves them.

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

The method follows Moritz's `run_decoder.py`, which used `replay_trajectory_classification`; 
that package is unmaintained and does not install here

`decode_nrem=True` decodes NREM as well, judged against the same unit-permutation null as REM.
`apply_decoder` and `reference_decode` carry it to other recordings, and `plot_transfer_summary`
adds an NREM panel when there is one. NREM is long, so it costs the run's longest decode and
shuffles to match, and the random walk was fitted to waking head movement while the internal
heading moves faster in NREM: decode it again with `movement_var_deg2=np.inf` as the control.

### Ring attractor from correlations

`spikeshpc/ring.py` reads heading without tuning curves. It is a port of the lab's
`HD_CCH_to_ring.m`, with ideas from SPUD (Chaudhuri et al. 2019):

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
`metrics_by_group` work on them. Where it departs from the MATLAB, and why, is in the module
docstring. The biggest change: units with no partner correlated beyond noise are screened off
before the embedding, because left in they pull the ring apart. Units of other structures have
partners of their own, though, so give the ring one structure's units by depth: on session7 the
whole probe's 285 units decoded wake at 70°, the 51 in the head-direction band at 18°.

A ring's decode is read as heading through its own alignment, or, with `RingAlignment.then`,
through matched units onto another recording's ring and that ring's alignment. The second way
needs none of the recording's own tuning. `plot_turn_summary` and `plot_turn_sweep` take a
`"ring"` source beside the decoded one.

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
