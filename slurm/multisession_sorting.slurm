#!/bin/bash
#SBATCH --job-name=ks4
#SBATCH --partition=gpu                     # e.g. your cluster's GPU partition
#SBATCH --gres=gpu:1                        # kilosort4 needs exactly 1 GPU
#SBATCH --cpus-per-task=23                  # keep job_kwargs.n_jobs <= this
#SBATCH --mem=368G
#SBATCH --time=12:00:00
# %x_%j -> <jobname>_<jobid>.out. The underscore before the job ID is required
# by slurmtail, whose find_log_files() only matches *_<JOBID>.{out,err}.
# NOTE: this directory must already exist -- SLURM will not create it, and the
# job fails to start if it is missing.
#SBATCH --output=/home/remote/davcollins/logs/%x_%j.out
#SBATCH --error=/home/remote/davcollins/logs/%x_%j.err

set -euo pipefail

# module load apptainer   # adjust to your cluster's module name if different

# ── The only three things that live here ─────────────────────────────────
# Everything else -- recordings, output and scratch directories, skip flags,
# bad channels, brain-state and sorting parameters -- comes out of
# PIPELINE_CONFIG, so the same file drives this job and a local run:
#
#     spikeshpc --pipeline_config my_run.json
#
# These three are deployment facts rather than analysis parameters, and two of
# them are needed before the container can be started at all.

# Clone of github.com/Scanziani-Lab/spikeshpc. This script lives in its slurm/
# subdirectory; the entry point sits at the repo root next to the package, so
# nothing here needs the package pip-installed.
REPO_DIR=/home/remote/davcollins/spikeshpc

# Single combined image (spikeinterface + kilosort4 + CUDA).
SI_SIF=/scratch/user/davcollins/containers/si_kilosort4.sif

# See slurm/pipeline_config.example.json.
PIPELINE_CONFIG=/scratch/user/davcollins/runs/session6a.json

# Apptainer's own cache/tmp default to ~/.apptainer, which is on the home
# quota. Redirect those too.
export APPTAINER_CACHEDIR="/scratch/user/davcollins/apptainer/cache"
export APPTAINER_TMPDIR="/scratch/user/davcollins/apptainer/tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# ── Everything below is derived; you should not need to edit it ──────────
if [[ ! -f "${PIPELINE_CONFIG}" ]]; then
    echo "ERROR: pipeline config does not exist: ${PIPELINE_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${SI_SIF}" ]]; then
    echo "ERROR: container image does not exist: ${SI_SIF}" >&2
    exit 1
fi

# Read the paths out of the config, using the container's python so the host
# needs no python of its own. The repo and the config's own directory have to
# be bound for this first call; the rest are discovered from what it prints.
CONFIG_DIR=$(dirname "${PIPELINE_CONFIG}")
eval "$(apptainer exec \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    --bind "${CONFIG_DIR}:${CONFIG_DIR}" \
    --env "PYTHONPATH=${REPO_DIR}" \
    "${SI_SIF}" \
    python -m spikeshpc.slurm_env "${PIPELINE_CONFIG}")"

if [[ ${#SPIKESHPC_PHYS[@]} -eq 0 ]]; then
    echo "ERROR: ${PIPELINE_CONFIG} sets no run.phys_paths" >&2
    exit 1
fi

# Fail on a typo now rather than after the GPU has been allocated.
for d in "${SPIKESHPC_PHYS[@]}"; do
    if [[ ! -e "${d}" ]]; then
        echo "ERROR: recording does not exist: ${d}" >&2
        exit 1
    fi
done

# Stage 2 writes concatenated.bin into the output directory, which therefore
# needs room for every session combined -- unless a single recording is being
# sorted in place, in which case nothing is copied.
[[ -n "${SPIKESHPC_OUTPUT_DIR}" ]] && mkdir -p "${SPIKESHPC_OUTPUT_DIR}"
[[ -n "${SPIKESHPC_TMP_DIR}" ]] && mkdir -p "${SPIKESHPC_TMP_DIR}"

BIND_ARGS=(--bind "${REPO_DIR}:${REPO_DIR}")
for d in "${SPIKESHPC_BIND[@]}"; do
    [[ -e "${d}" ]] && BIND_ARGS+=(--bind "${d}:${d}")
done

echo "config    : ${PIPELINE_CONFIG}"
echo "recordings: ${#SPIKESHPC_PHYS[@]}"
echo "output    : ${SPIKESHPC_OUTPUT_DIR:-<from config>}"
echo "binding   : ${SPIKESHPC_BIND[*]}"

apptainer exec --nv \
    "${BIND_ARGS[@]}" \
    "${SI_SIF}" \
    python -u "${REPO_DIR}/hpc_load_sort_post.py" \
        --pipeline_config "${PIPELINE_CONFIG}"

# Outputs in run.output_dir:
#   states/            per-session WAKE/NREM/REM intervals + metrics, the
#                      derived shutter times and their sanity plot, and
#                      states_concatenated.json on the concatenated clock
#   concatenated.bin   combined binary handed to kilosort4 (multi-session only)
#   concat_info.json   per-session sample counts + offsets, channel alignment
#   chanMap.mat        channel map in kilosort's format
#   probe.json         probe geometry, used to reload the binary
#   bad_channels.json  per-channel labels + the set applied (autodetect only)
#   kilosort4/         kilosort4 results (phy format)
#   analyzer.zarr      spikeinterface sorting analyzer
