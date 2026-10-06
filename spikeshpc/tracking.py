"""Running UnitMatch across recordings of one probe, and the bookkeeping around it.

UnitMatchPy is not a dependency of spikeshpc, so it is imported inside
:func:`run_unitmatch` only; everything else here runs wherever spikeshpc does.
:mod:`spikeshpc.matching` turns the probabilities this returns into partners.
"""

from __future__ import annotations

import subprocess

import numpy as np

__all__ = [
    "git_commit",
    "recording_paths",
    "run_unitmatch",
    "waveform_status",
]


def recording_paths(processed_root, raw_root, name):
    """The files and folders of one recording, by name."""
    root = processed_root / name
    return {
        "ks": root / "kilosort4",
        "analyzer": root / "analyzer.zarr",
        "states": root / "states",
        "shutter": root / "states" / f"{name}_shutter_close_times.npy",
        "tuning": root / "tuning" / "hd_tuning.npz",
        "curation": root / "curation",
        "data": raw_root / name / "preprocessed.bin",
        "meta": raw_root
        / name
        / "raw"
        / "experiment1"
        / "recording1"
        / "structure.oebin",
    }


def git_commit(folder):
    """The commit a repository is at, marked if it has uncommitted changes."""
    try:
        git = ["git", "-C", str(folder)]
        head = subprocess.run(
            git + ["rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            git + ["status", "--porcelain"], capture_output=True, text=True, check=True
        )
        return head.stdout.strip() + (
            " +uncommitted changes" if dirty.stdout.strip() else ""
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def waveform_status(paths, r, spike_counts):
    """Whether recording `r`'s extracted raw waveforms match its current sort."""
    folder = paths[r]["ks"] / "RawWaveforms"
    files = {
        int(p.name[4:].split("_")[0]): p for p in folder.glob("Unit*_RawSpikes.npy")
    }
    if not files:
        return "missing", folder, []
    units = np.flatnonzero(spike_counts[r])
    sorted_at = (paths[r]["ks"] / "spike_clusters.npy").stat().st_mtime
    absent = [u for u in units if u not in files]
    older = [u for u in units if u in files and files[u].stat().st_mtime < sorted_at]
    if absent or older:
        raise RuntimeError(
            f"{r}: {folder} does not match the current sort ({len(absent)} units have no "
            f"file, {len(older)} files are older than spike_clusters.npy). Move it aside "
            "and re-run this cell to extract them afresh."
        )
    return "ok", folder, sorted(set(files) - set(units.tolist()))


def run_unitmatch(paths, names, units_of, spike_counts, keep_waveforms=False):
    """Score every pair of units across recordings `names` with UnitMatch."""
    from UnitMatchPy import bayes_functions as umb
    from UnitMatchPy import default_params as umd
    from UnitMatchPy import overlord as umo
    from UnitMatchPy import utils as umu

    ks_dirs = [str(paths[r]["ks"]) for r in names]
    param = umd.get_default_param()
    param["KS_dirs"] = ks_dirs  # the GUI and assign_unique_id look for this key
    # param= syncs spike_width / peak_loc / waveidx to the files (61 for KS4)
    wave_paths, _, channel_pos = umu.paths_from_KS(ks_dirs, param=param)
    assert param["peak_loc"] in param["waveidx"], "peak_loc must lie inside waveidx"
    param = umu.get_probe_geometry(channel_pos[0], param)

    good_units = [units_of[r] for r in names]
    waveform, session_id, session_switch, within_session, param = umu.load_good_units(
        good_units, wave_paths, param
    )
    param["n_units_per_session"] = [len(spike_counts[r]) for r in names]
    assert waveform.shape[1] == param["spike_width"]
    clus_info = {
        "good_units": good_units,
        "session_switch": session_switch,
        "session_id": session_id,
        "original_ids": np.concatenate(good_units),
    }

    props = umo.extract_parameters(waveform, channel_pos, clus_info, param)
    total_score, candidate_pairs, scores_to_include, predictors = (
        umo.extract_metric_scores(props, session_switch, within_session, param, niter=2)
    )
    prior_match = 1 - param["n_expected_matches"] / param["n_units"] ** 2
    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    kernels = umb.get_parameter_kernels(
        scores_to_include, labels, cond, param, add_one=1
    )
    probability = umb.apply_naive_bayes(
        kernels, np.array((prior_match, 1 - prior_match)), predictors, param, cond
    )
    return {
        "names": names,
        "prob": probability[:, 1].reshape(param["n_units"], param["n_units"]),
        "clus_info": clus_info,
        "param": param,
        "props": props,
        "total_score": total_score,
        "scores_to_include": scores_to_include,
        "within_session": within_session,
        "channel_pos": channel_pos,
        "waveform": waveform if keep_waveforms else None,  # only the GUI needs it
    }
