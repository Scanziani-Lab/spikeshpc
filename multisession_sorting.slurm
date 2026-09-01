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

# module load apptainer   # adjust to your cluster's module name if different (e.g. singularity)

# ── Paths -- edit these for your run ─────────────────────────────────────
REPO_DIR=/home/remote/davcollins/spikes

# Single combined image (spikeinterface + kilosort4 + CUDA). Read by apptainer
# on the host, so it does not need its own bind-mount.
SI_SIF=/scratch/user/davcollins/containers/si_kilosort4.sif

# Parent of your recording/output directories. Bind-mounted below at the SAME
# path inside the container, so paths on the command line resolve identically
# inside and out.
DATA_DIR=/scratch/user/davcollins/physiology/5800941

PHYS_DIRS=(
    "${DATA_DIR}/session0"
    "${DATA_DIR}/session3"
)   # one or more; multiple dirs are aligned by probe geometry and concatenated

# Stage 1 writes concatenated.bin here, so OUTPUT_DIR needs room for a full
# copy of all the sessions above combined -- not just one of them.
OUTPUT_DIR="${DATA_DIR}/kilosort"

# Scratch for spikeinterface's cache and torch/numpy memmaps. MUST NOT be on
# /home -- the quota there is far too small, and overrunning it kills the job
# with no traceback.
TMP_DIR="${DATA_DIR}/tmp"

# Acquisition system and stream are auto-detected (SpikeGLX .meta vs OpenEphys
# structure.oebin/settings.xml; the AP-band stream is picked per format).
# Uncomment to force them -- required if a session has more than one probe,
# since the pipeline refuses to guess between them.
# PHYS_ARGS=(--phys_type openephysbinary --stream_name "Record Node 101#Neuropix-PXI-100.ProbeA")
PHYS_ARGS=()

# Apptainer's own cache/tmp default to ~/.apptainer, which is on the same
# home quota. Redirect those too.
export APPTAINER_CACHEDIR="/scratch/user/davcollins/apptainer/cache"
export APPTAINER_TMPDIR="/scratch/user/davcollins/apptainer/tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# See slurm/pipeline_config.example.json. NOTE: the "sorting" key now holds
# kilosort.DEFAULT_SETTINGS overrides, not si.run_sorter kwargs -- kilosort
# raises "Unrecognized settings" on leftovers like sorter_name or
# save_preprocessed_copy, so strip those from an old config.
PIPELINE_CONFIG="${REPO_DIR}/pipeline_config.json"

# ── Stage skipping -- re-run one part of the pipeline ────────────────────
# All four stages run by default. To reuse earlier output, set the matching
# variable at submit time, e.g. redo only post-processing with:
#
#   sbatch --export=ALL,SKIP_STATESCORING=1,SKIP_PREPROCESSING=1,SKIP_SORTING=1 \
#       slurm/multisession_sorting.slurm
#
# Stage 1 is brain-state scoring (WAKE/NREM/REM), which runs per session
# BEFORE concatenation so a 10 s spectrogram window never straddles a junction
# between sessions. It is independent of sorting -- skip it on re-sorts.
#
# Skipping a stage requires its inputs to already be in OUTPUT_DIR:
#   --skip_preprocessing  needs concatenated.bin + concat_info.json + probe.json
#   --skip_sorting        needs kilosort4/
SKIP_ARGS=()
if [[ "${SKIP_STATESCORING:-0}" == "1" ]]; then
    SKIP_ARGS+=(--skip_statescoring)
fi
if [[ "${SKIP_PREPROCESSING:-0}" == "1" ]]; then
    SKIP_ARGS+=(--skip_preprocessing)
fi
if [[ "${SKIP_SORTING:-0}" == "1" ]]; then
    SKIP_ARGS+=(--skip_sorting)
fi
if [[ "${SKIP_POSTPROCESSING:-0}" == "1" ]]; then
    SKIP_ARGS+=(--skip_postprocessing)
fi

# ── Bad channels -- dead sites, or anything out of the brain ─────────────
# Space-separated channel ids as listed in concat_info.json, or 0-based rows
# of concatenated.bin. Overrides the "bad_channels" list in PIPELINE_CONFIG.
# Excluded from both the kilosort probe and the analyzer; the binary itself is
# untouched, so a revised list only needs the sorting stage re-run:
#
#   sbatch --export=ALL,SKIP_PREPROCESSING=1,BAD_CHANNELS="191 192 193" \
#       slurm/multisession_sorting.slurm
BAD_ARGS=()
if [[ -n "${BAD_CHANNELS:-}" ]]; then
    # Unquoted on purpose: split the list into one argument per channel.
    BAD_ARGS=(--bad_channels ${BAD_CHANNELS})
fi
# Also flag bad channels automatically (si.detect_bad_channels), unioned with
# BAD_CHANNELS above. Per-channel good/dead/noise/out labels land in
# bad_channels.json in OUTPUT_DIR for review. Tune the method there or in
# PIPELINE_CONFIG under "detect_bad_channels".
#
#   sbatch --export=ALL,DETECT_BAD_CHANNELS=1 slurm/multisession_sorting.slurm
if [[ "${DETECT_BAD_CHANNELS:-0}" == "1" ]]; then
    BAD_ARGS+=(--detect_bad_channels)
fi

# ── Fail early on typos in the paths above, before burning a GPU slot ────
for d in "${PHYS_DIRS[@]}"; do
    if [[ ! -d "${d}" ]]; then
        echo "ERROR: recording directory does not exist: ${d}" >&2
        exit 1
    fi
done
if [[ ! -f "${PIPELINE_CONFIG}" ]]; then
    echo "ERROR: pipeline config does not exist: ${PIPELINE_CONFIG}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}"

echo "sessions : ${#PHYS_DIRS[@]}"
echo "output   : ${OUTPUT_DIR}"
# ${arr[@]+...} keeps `set -u` from tripping over an empty array on bash < 4.4.
echo "skipping : ${SKIP_ARGS[@]+${SKIP_ARGS[@]}}"
echo "bad chans: ${BAD_CHANNELS:-none} (autodetect=${DETECT_BAD_CHANNELS:-0})"

apptainer exec --nv \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${SI_SIF}" \
    python -u "${REPO_DIR}/hpc_load_sort_post.py" \
        "${PHYS_DIRS[@]}" \
        ${PHYS_ARGS[@]+"${PHYS_ARGS[@]}"} \
        --output_dir "${OUTPUT_DIR}" \
        --tmp_dir "${TMP_DIR}" \
        --pipeline_config "${PIPELINE_CONFIG}" \
        ${BAD_ARGS[@]+"${BAD_ARGS[@]}"} \
        ${SKIP_ARGS[@]+"${SKIP_ARGS[@]}"}

# Outputs in ${OUTPUT_DIR}:
#   states/            per-session WAKE/NREM/REM intervals + metrics, plus
#                      states_concatenated.json on the concatenated clock
#   concatenated.bin   combined binary handed to kilosort4
#   concat_info.json   per-session sample counts + offsets, channel alignment
#   chanMap.mat        channel map in kilosort's format
#   probe.json         probe geometry, used to reload the binary
#   bad_channels.json  per-channel labels + the set applied (autodetect only)
#   kilosort4/         kilosort4 results (phy format)
#   analyzer.zarr      spikeinterface sorting analyzer
