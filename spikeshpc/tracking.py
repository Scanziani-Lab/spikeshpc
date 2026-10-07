"""The steps of notebook 3_tracking: following a baseline's units across recordings.

UnitMatch scores every pair of units across recordings of one probe; the
baseline's units are then given one partner in each other recording, and the
partners' tuning, a decoder trained on the baseline, and each recording's own
ring show what happened to the head-direction cells from one recording to the
next.

UnitMatchPy is not a dependency of spikeshpc, so it is imported inside the
functions that call it only (:func:`extract_waveforms`, :func:`run_unitmatch`,
:func:`save_unitmatch`, :func:`open_unitmatch_gui`); everything else here runs
wherever spikeshpc does. :mod:`spikeshpc.matching` turns UnitMatch's
probabilities into partners.

The functions share the notebook's names for things:

  * ``rec_type``: ``{recording: "baseline" | "pre" | "post"}``, in
    chronological order, exactly one baseline (:func:`check_settings`)
  * ``paths``: ``{recording: recording_paths(...)}``
  * ``spike_counts``: ``{recording: spikes per unit id}`` of each sort
  * ``tuning``: ``{recording: HDTuning}``
  * ``units_of``: ``{recording: unit ids UnitMatch sees}``
  * ``partner``: ``{(recording, baseline unit): its partner there}``
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .curation import final_unit_labels
from .decoder import apply_decoder, plot_transfer_summary, reference_decode, run_decoder
from .io import load_states
from .matching import (
    estimate_rotation,
    plot_tracked_tuning,
    putative_matches,
    recording_colors,
    select_partners,
    tuning_similarity,
)
from .optitrack import load_hd_tuning
from .ring import (
    compare_rings,
    plot_ring,
    plot_ring_comparison,
    plot_ring_vs_tuning,
    ring_vs_tuning,
    run_ring,
)

__all__ = [
    "check_settings",
    "compare_matched_rings",
    "decoder_units",
    "extract_waveforms",
    "git_commit",
    "load_recordings_for_tracking",
    "match_counts",
    "open_unitmatch_gui",
    "plot_decoder_summary",
    "plot_matched_tuning",
    "plot_ring_comparisons",
    "plot_rings",
    "plot_rings_against_tuning",
    "prepare_output_dir",
    "probe_drift",
    "recording_paths",
    "return_matches",
    "ring_band",
    "ring_summary",
    "rings_against_tuning",
    "run_rings",
    "run_unitmatch",
    "save_decoder",
    "save_figures",
    "save_matches",
    "save_rings",
    "save_transfer",
    "save_unitmatch",
    "select_units",
    "software_versions",
    "train_decoder",
    "transfer_decoder",
    "unitmatch_groups",
    "waveform_status",
]

BASELINE_UNITS = ("tuned", "good", "all")
UNITMATCH_MODES = ("pairwise", "joint")
OUTPUT_FOLDERS = ("unitmatch", "matches", "tuning", "decoder", "ring")

# The columns of each summary table worth reading in the notebook; the saved
# CSVs keep them all.
DECODER_COLUMNS = [
    "n_units",
    "wake_median_abs_error_deg",
    "wake_median_abs_error_corrected_deg",
    "wake_offset_deg",
    "wake_circular_correlation",
    "rem_mean_posterior_max",
    "nrem_mean_posterior_max",
]
RING_COLUMNS = [
    "type",
    "drift_um",
    "n_candidates",
    "n_units",
    "n_members",
    "eigenvalue_ratio",
    "consistency_r",
    "split_half_r",
    "alignment_margin",
    "test_median_abs_error_deg",
    "test_p",
    "rem_mean_length",
    "rem_p",
]
RING_COMPARISON_COLUMNS = [
    "type",
    "matched",
    "n_pairs",
    "angle_circular_r",
    "angle_median_abs_error_deg",
    "angle_null_mean",
    "angle_p",
    "structure_r",
    "structure_null_mean",
    "structure_p",
    "split_half_r_a",
    "split_half_r_b",
]


def _check_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _split(rec_type):
    """The baseline, and the other recordings in order."""
    baseline = next(r for r, kind in rec_type.items() if kind == "baseline")
    return baseline, [r for r in rec_type if r != baseline]


def _curve(r, unit, tuning, tuning_ids):
    return tuning[r].curve(unit)[1] if unit in tuning_ids[r] else None


def _tuning_stat(r, unit, name, tuning):
    stats = tuning[r].stats.get(unit)
    return getattr(stats, name) if stats is not None else np.nan


def _load_analyzer(folder):
    from spikeinterface import load_sorting_analyzer

    return load_sorting_analyzer(folder, load_extensions=False)


# ── the recordings ───────────────────────────────────────────────────────────


def check_settings(recordings, recording_types, baseline_units, unitmatch_mode):
    """Check the notebook's settings before anything runs.

    Returns
    -------
    baseline : str
        The baseline recording.
    others : list of str
        Every other recording, in order.
    rec_type : dict
        ``{recording: type}``, in order.
    """
    if len(recordings) != len(recording_types):
        raise ValueError("give one type per recording")
    if len(set(recordings)) != len(recordings):
        raise ValueError("a recording is listed twice")
    unknown = set(recording_types) - {"baseline", "pre", "post"}
    if unknown:
        raise ValueError(f"unknown recording type(s) {sorted(unknown)}")
    if list(recording_types).count("baseline") != 1:
        raise ValueError("exactly one recording must be the baseline")
    if len(recordings) < 2:
        raise ValueError("nothing to match the baseline against")
    _check_choice("baseline_units", baseline_units, BASELINE_UNITS)
    _check_choice("unitmatch_mode", unitmatch_mode, UNITMATCH_MODES)
    if "post" not in recording_types:
        warnings.warn("no post-lesion recording in the list")
    rec_type = dict(zip(recordings, recording_types))
    baseline, others = _split(rec_type)
    return baseline, others, rec_type


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


def load_recordings_for_tracking(paths, rec_type):
    """Load every recording's sort and tuning, and check that they belong together.

    Everything here is keyed by unit id, and a re-sort renumbers units 0..N-1,
    so a file from an older sort loads onto the new one without complaint.
    spike_clusters.npy is the sort itself; the analyzer that was curated and
    the tuning are checked against it.

    Returns
    -------
    spike_counts : dict
        ``{recording: spikes per unit id}``.
    tuning : dict
        ``{recording: HDTuning}``.
    overview : pandas.DataFrame
        Units and tuned units per recording.

    Raises
    ------
    FileNotFoundError
        If a recording's files, or the baseline's curated labels, are missing.
    ValueError
        If a recording's files come from different sorts.
    """
    baseline, others = _split(rec_type)
    missing = [
        f"{r}: {paths[r][key]}"
        for r in rec_type
        for key in ("ks", "analyzer", "states", "shutter", "tuning")
        if not paths[r][key].exists()
    ]
    labels_file = paths[baseline]["curation"] / "manual_vs_bombcell_classifications.csv"
    if not labels_file.is_file():
        missing.append(f"{baseline}: {labels_file} (the baseline's curated labels)")
    if missing:
        raise FileNotFoundError("missing:\n  " + "\n  ".join(missing))

    spike_counts, tuning, overview = {}, {}, []
    for r in rec_type:
        counts = np.bincount(np.load(paths[r]["ks"] / "spike_clusters.npy").ravel())
        spike_counts[r] = counts
        # The analyzer's record of its sort, written when it was curated. A spike
        # or two may differ -- the analyzer drops spikes past the end of the binary
        # -- where a different sort differs everywhere.
        fingerprint = paths[r]["curation"] / "sorting.json"
        if fingerprint.is_file():
            saved = json.loads(fingerprint.read_text())
            same_units = saved["unit_ids"] == list(range(len(counts)))
            if not (
                same_units
                and (
                    np.abs(np.asarray(saved["num_spikes"]) - counts)
                    <= np.maximum(2, 0.001 * counts)
                ).all()
            ):
                raise ValueError(
                    f"{r}: kilosort4/ and the analyzer curated in curation/ "
                    f"({saved.get('created', '?')}) are different sorts"
                )
        else:
            warnings.warn(
                f"{r}: no curation/sorting.json, so the analyzer is not checked against kilosort4/"
            )

        hd = load_hd_tuning(paths[r]["tuning"])
        if len(hd.heading_deg) != len(np.load(paths[r]["shutter"], mmap_mode="r")):
            raise ValueError(
                f"{r}: its tuning was computed against a different shutter file"
            )
        if not set(hd.unit_ids.tolist()) <= set(range(len(counts))):
            raise ValueError(f"{r}: its tuning names units this sort does not have")
        tuned_on = hd.parameters.get("ks_path")
        if tuned_on and Path(tuned_on).resolve() != paths[r]["ks"].resolve():
            warnings.warn(f"{r}: its tuning says it was computed on {tuned_on}")
        tuning[r] = hd
        overview.append(
            {
                "recording": r,
                "type": rec_type[r],
                "units": len(counts),
                "with tuning": len(hd.unit_ids),
                "HD tuned": len(hd.tuned_ids),
            }
        )
    for r in others:
        if not np.allclose(tuning[r].bin_centers_deg, tuning[baseline].bin_centers_deg):
            raise ValueError(
                f"{r}: its tuning curves are on a different ring of bins from the baseline's"
            )
    return spike_counts, tuning, pd.DataFrame(overview).set_index("recording")


def prepare_output_dir(output_dir, overwrite=False):
    """Make `output_dir` and its subfolders, refusing old results unless `overwrite`."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already holds results: choose another output_dir or set overwrite = True"
        )
    for folder in OUTPUT_FOLDERS:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)


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


def software_versions():
    """The versions behind a run, for its config.json; None for a package not installed."""
    from importlib.metadata import PackageNotFoundError, version

    def installed(package):
        try:
            return version(package)
        except PackageNotFoundError:
            return None

    return {
        "spikeshpc": git_commit(Path(__file__).resolve().parents[1]),
        "UnitMatchPy": installed("UnitMatchPy"),
        "spikeinterface": installed("spikeinterface"),
        "numpy": np.__version__,
    }


# ── UnitMatch ────────────────────────────────────────────────────────────────


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


def extract_waveforms(
    paths,
    spike_counts,
    sample_amount=1000,
    spike_width=61,
    samples_before=20,
    n_channels=384,
):
    """Extract raw waveforms for every recording that has none.

    UnitMatch compares units by the average waveform of each half of their
    spikes, extracted from the binary into kilosort4/RawWaveforms/. All units
    are extracted, since every unit of the other recordings is searched. A
    folder that disagrees with the sort is never overwritten:
    :func:`waveform_status` raises, and it must be moved aside by hand.

    Returns
    -------
    dict
        ``{recording: waveform_status(...)}``, after extraction.
    """
    status = {r: waveform_status(paths, r, spike_counts) for r in spike_counts}
    for r, (_, folder, extra) in status.items():
        if extra:
            print(
                f"{r}: {len(extra)} file(s) in {folder.name} for units this sort does not have "
                f"(Unit{extra[0]}..Unit{extra[-1]}) -- never loaded, safe to move aside"
            )

    to_extract = [r for r in status if status[r][0] == "missing"]
    if to_extract:
        from UnitMatchPy import extract_raw_data as ume

        absent = [
            str(paths[r][key])
            for r in to_extract
            for key in ("data", "meta")
            if not paths[r][key].exists()
        ]
        if absent:
            raise FileNotFoundError("extracting waveforms needs:\n  " + "\n  ".join(absent))
        print(f"extracting waveforms for {to_extract}: tens of minutes per recording")
        ks_dirs = [str(paths[r]["ks"]) for r in to_extract]
        spike_ids, spike_times, no_good_units, all_unit_ids = ume.extract_KS_data(
            ks_dirs, extract_good_units_only=False
        )
        samples_after = spike_width - samples_before
        ume.extraction_pipeline(
            data_paths=[str(paths[r]["data"]) for r in to_extract],
            meta_paths=[str(paths[r]["meta"]) for r in to_extract],
            ks_paths=ks_dirs,
            n_sessions=len(to_extract),
            spike_ids=spike_ids,
            all_unit_ids=all_unit_ids,
            spike_times=spike_times,
            sample_amount=sample_amount,
            samples_before=samples_before,
            samples_after=samples_after,
            half_width=spike_width // 2,
            spike_width=spike_width,
            max_width=samples_after,
            n_channels=n_channels,
            KS4_data=True,
            extract_good_units_only=False,
            good_units=no_good_units,
        )
        del spike_ids, spike_times
        status.update({r: waveform_status(paths, r, spike_counts) for r in to_extract})
    print("waveforms:", {r: s[0] for r, s in status.items()})
    return status


def select_units(paths, spike_counts, tuning, rec_type, baseline_units="tuned"):
    """Choose the units UnitMatch sees.

    The baseline contributes the units named by `baseline_units`: "tuned"
    (head-direction tuned and curated GOOD or MUA), "good" (curated GOOD) or
    "all" (GOOD and MUA). Every other recording contributes every unit. The
    labels are read by unit id: bombcell's table can skip a unit (session9's
    161, session10's 29), and reading it by row hands every later unit the
    label of the one after it.

    Returns
    -------
    labels_base : pandas.Series
        The baseline's curated label of every unit in its sort.
    tuned_base : list
        The baseline's tuned, curated units.
    units_of : dict
        ``{recording: unit ids}``.
    """
    _check_choice("baseline_units", baseline_units, BASELINE_UNITS)
    baseline, others = _split(rec_type)
    labels_base = final_unit_labels(
        paths[baseline]["curation"], unit_ids=np.arange(len(spike_counts[baseline]))
    )
    curated = set(labels_base.index[labels_base.isin(["GOOD", "MUA"])].tolist())
    tuned_ids = tuning[baseline].tuned_ids.tolist()
    tuned_base = [u for u in tuned_ids if u in curated]
    uncurated = sorted(set(tuned_ids) - curated)
    if uncurated:
        warnings.warn(
            f"{baseline}: tuned units that are not curated GOOD/MUA, left out: {uncurated}"
        )

    if baseline_units == "tuned":
        base_units = tuned_base
    elif baseline_units == "good":
        base_units = labels_base.index[labels_base == "GOOD"].tolist()
    else:
        base_units = sorted(curated)

    units_of = {r: np.flatnonzero(spike_counts[r]) for r in others}
    units_of[baseline] = np.array(sorted(base_units), dtype=int)
    print(
        f"{baseline} (baseline): {len(units_of[baseline])} '{baseline_units}' units "
        f"({len(tuned_base)} tuned, {len(curated)} curated GOOD/MUA, "
        f"{len(spike_counts[baseline])} in the sort)"
    )
    for r in others:
        print(f"{r} ({rec_type[r]}): all {len(units_of[r])} units")
    return labels_base, tuned_base, units_of


def unitmatch_groups(rec_type, unitmatch_mode):
    """The recordings of each UnitMatch run, by run name.

    pairwise  one run per recording, against the baseline alone
    joint     one run over every recording, drift aligned in a chain through
              neighbours in the list
    """
    _check_choice("unitmatch_mode", unitmatch_mode, UNITMATCH_MODES)
    baseline, others = _split(rec_type)
    if unitmatch_mode == "pairwise":
        return {f"{baseline}_vs_{r}": [baseline, r] for r in others}
    return {"joint": list(rec_type)}


def run_unitmatch(paths, names, units_of, spike_counts, keep_waveforms=False, verbose=True):
    """Score every pair of units across recordings `names` with UnitMatch.

    Waveforms, metrics, drift correction and naive Bayes. The probability
    matrix is indexed by ROW in the stacked unit list (the first recording's
    units first), never by cluster id; ``clus_info["original_ids"]`` maps a row
    back to its cluster. With `verbose`, UnitMatch's own evaluation of the run
    is printed too.
    """
    from UnitMatchPy import bayes_functions as umb
    from UnitMatchPy import default_params as umd
    from UnitMatchPy import overlord as umo
    from UnitMatchPy import utils as umu

    if verbose:
        print("\n=== UnitMatch: " + " + ".join(f"{r} ({len(units_of[r])})" for r in names))
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
    prob = probability[:, 1].reshape(param["n_units"], param["n_units"])
    if verbose:
        umu.evaluate_output(
            prob, param, within_session, session_switch, match_threshold=0.75
        )
    return {
        "names": names,
        "prob": prob,
        "clus_info": clus_info,
        "param": param,
        "props": props,
        "total_score": total_score,
        "scores_to_include": scores_to_include,
        "within_session": within_session,
        "channel_pos": channel_pos,
        "waveform": waveform if keep_waveforms else None,  # only the GUI needs it
    }


def save_unitmatch(run, folder, match_threshold=0.5):
    """Save a run's outputs as UnitMatch does, and its matrix with the ids that index it.

    In UnitMatch's MatchTable.csv the row ``ID1=a, ID2=b`` holds ``P[b, a]``.
    um_probability.npz holds the matrix with each row's unit and session ids.
    """
    from UnitMatchPy import assign_unique_id as uma
    from UnitMatchPy import save_utils as ums

    folder = Path(folder)
    above = (run["prob"] > match_threshold).astype(float)
    props = run["props"]
    ums.save_to_output(
        str(folder),
        run["scores_to_include"],
        np.argwhere(above == 1),
        run["prob"],
        props["avg_centroid"],
        props["avg_waveform"],
        props["avg_waveform_per_tp"],
        props["max_site"],
        run["total_score"],
        above,
        run["clus_info"],
        run["param"],
        UIDs=uma.assign_unique_id(run["prob"], run["param"], run["clus_info"]),
        matches_curated=None,
        save_match_table=True,
    )
    np.savez_compressed(
        folder / "um_probability.npz",
        prob=run["prob"],
        unit_ids=run["clus_info"]["original_ids"],
        session_ids=run["clus_info"]["session_id"],
        recordings=np.array(run["names"]),
    )


def open_unitmatch_gui(run, match_threshold=0.5):
    """Inspect a run in UnitMatch's GUI.

    Its Unit A/B lists are row numbers; the "Original Unit IDs" box names the
    clusters. What is marked there is not fed back: the matches come from
    :func:`return_matches`. The run must have kept its waveforms
    (``run_unitmatch(..., keep_waveforms=True)``).

    Returns
    -------
    is_match, not_match, matches_GUI
        What :func:`UnitMatchPy.GUI.run_GUI` returns.
    """
    from UnitMatchPy import GUI as umg

    if run["waveform"] is None:
        raise ValueError("the GUI needs the waveforms: run UnitMatch with open_gui = True")
    props = run["props"]
    umg.process_info_for_GUI(
        run["prob"],
        match_threshold,
        run["scores_to_include"],
        run["total_score"],
        props["amplitude"],
        props["spatial_decay"],
        props["avg_centroid"],
        props["avg_waveform"],
        props["avg_waveform_per_tp"],
        props["good_wave_idxs"],
        props["max_site"],
        props["max_site_mean"],
        run["waveform"],
        run["within_session"],
        run["channel_pos"],
        run["clus_info"],
        run["param"],
    )
    return umg.run_GUI()


# ── matches ──────────────────────────────────────────────────────────────────


def return_matches(
    um_runs,
    unitmatch_mode,
    rec_type,
    tuning,
    units_of,
    tuned_base,
    labels_base,
    match_threshold=0.5,
    min_rotation_pairs=3,
):
    """Give each baseline unit at most one partner in each other recording.

    candidate   P > match_threshold in EITHER direction: P[a, b] and P[b, a]
                come from opposite halves of each unit's spikes and often
                disagree, one near 1 and the other near 0
    pre         a tuned baseline unit's candidates are split by how alike the
                tuning curves are once the population's common rotation
                between the recordings is undone (measured on pairs that need
                no split). An untuned unit's curve is noise, so its candidates
                are split by average P, and those pairs rank after the ones
                tuning decided.
    post        split by average P alone: a lesion that may disrupt tuning
                must not decide which unit is which
    one-to-one  the better-ranked of two baseline units keeps a contested
                partner; the other falls back to its next candidate, if any

    Returns
    -------
    candidates : pandas.DataFrame
        Every candidate pair, with why it was or was not chosen.
    matched : pandas.DataFrame
        The chosen pairs.
    wide : pandas.DataFrame
        One row per baseline unit, its partner (if any) in each recording.
    partner : dict
        ``{(recording, baseline unit): partner unit}``.
    rotations : dict
        ``{recording: {"rotation_deg", "resultant_length", "n_pairs"}}``, for
        the pre recordings.
    """
    baseline, others = _split(rec_type)
    tuning_ids = {r: set(tuning[r].unit_ids.tolist()) for r in rec_type}
    tuned_set = set(tuned_base)
    tables, rotations = [], {}
    for r in others:
        run = um_runs[f"{baseline}_vs_{r}" if unitmatch_mode == "pairwise" else "joint"]
        session = {name: i for i, name in enumerate(run["names"])}
        candidates = putative_matches(
            run["prob"],
            run["clus_info"]["original_ids"],
            run["clus_info"]["session_id"],
            ref=session[baseline],
            other=session[r],
            threshold=match_threshold,
        )
        if rec_type[r] == "pre":
            rotation, resultant, n_pairs = estimate_rotation(
                candidates, tuning[baseline], tuning[r], min_pairs=min_rotation_pairs
            )
            rotations[r] = {
                "rotation_deg": rotation,
                "resultant_length": resultant,
                "n_pairs": n_pairs,
            }
            similarity = []
            for a, b in zip(candidates["ref_unit"], candidates["other_unit"]):
                curve_a = _curve(baseline, a, tuning, tuning_ids)
                curve_b = _curve(r, b, tuning, tuning_ids)
                similarity.append(
                    np.nan
                    if a not in tuned_set or curve_a is None or curve_b is None
                    else tuning_similarity(curve_a, curve_b, rotation)
                )
            chosen = select_partners(candidates, "tuning", similarity)
        else:
            chosen = select_partners(candidates, "probability")
        chosen.insert(0, "recording", r)
        chosen.insert(1, "type", rec_type[r])
        tables.append(chosen)

    candidates = pd.concat(tables, ignore_index=True)
    candidates["ref_tuned"] = [
        _tuning_stat(baseline, u, "significant", tuning) for u in candidates["ref_unit"]
    ]
    candidates["other_tuned"] = [
        _tuning_stat(r, u, "significant", tuning)
        for r, u in zip(candidates["recording"], candidates["other_unit"])
    ]
    candidates["ref_pd_deg"] = [
        _tuning_stat(baseline, u, "preferred_direction_deg", tuning)
        for u in candidates["ref_unit"]
    ]
    candidates["other_pd_deg"] = [
        _tuning_stat(r, u, "preferred_direction_deg", tuning)
        for r, u in zip(candidates["recording"], candidates["other_unit"])
    ]
    matched = candidates[candidates["chosen"]].reset_index(drop=True)
    partner = {
        (row.recording, row.ref_unit): row.other_unit for row in matched.itertuples()
    }

    wide = pd.DataFrame(index=pd.Index(units_of[baseline], name=f"{baseline}_unit"))
    wide["label"] = labels_base.reindex(wide.index)
    wide["tuned"] = wide.index.isin(list(tuned_set))
    wide["pd_deg"] = [
        _tuning_stat(baseline, u, "preferred_direction_deg", tuning) for u in wide.index
    ]
    for r in others:
        found = matched[matched["recording"] == r].set_index("ref_unit")
        wide[r] = found["other_unit"].reindex(wide.index).astype("Int64")
        wide[f"{r}_p_avg"] = found["p_avg"].reindex(wide.index)
    return candidates, matched, wide, partner, rotations


def match_counts(candidates, rotations, rec_type, n_baseline_units, tuned_base):
    """Count, for each recording, how the baseline's units fared in the matching."""
    _, others = _split(rec_type)
    counts = []
    for r in others:
        c = candidates[candidates["recording"] == r]
        with_candidate = c["ref_unit"].nunique()
        counts.append(
            {
                "recording": r,
                "type": rec_type[r],
                "baseline units": n_baseline_units,
                "with a candidate": with_candidate,
                "with several": int((c.groupby("ref_unit").size() > 1).sum()),
                "contested partners": int((c.groupby("other_unit").size() > 1).sum()),
                "fell back": int((c["reason"] == "fallback").sum()),
                "lost their partner": int(with_candidate - c["chosen"].sum()),
                "matched": int(c["chosen"].sum()),
                "matched & tuned": int(
                    c.loc[c["chosen"], "ref_unit"].isin(tuned_base).sum()
                ),
                "rotation (deg)": rotations.get(r, {}).get("rotation_deg", np.nan),
                "rotation pairs": rotations.get(r, {}).get("n_pairs", np.nan),
            }
        )
    return pd.DataFrame(counts).set_index("recording")


def save_matches(folder, candidates, matched, wide, rotations):
    """Save what :func:`return_matches` found."""
    folder = Path(folder)
    candidates.to_csv(folder / "candidates.csv", index=False)
    matched.to_csv(folder / "matches.csv", index=False)
    wide.to_csv(folder / "matches_wide.csv")
    (folder / "rotation.json").write_text(json.dumps(rotations, indent=2))


def plot_matched_tuning(partner, tuning, rec_type, tuned_base, rotations, unitmatch_mode):
    """Plot each matched baseline unit's tuning in every recording it was found in.

    Black the baseline, gray other pre-lesion recordings, red after the lesion
    (darker = later). In Hz, as recorded: a pre recording's rotation is in the
    legend, not undone. A panel's title lists the unit's partner in each other
    recording, "-" where none was found and "*" where its partner has no
    curve. In joint mode only the units found in every recording are drawn.

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    baseline, others = _split(rec_type)
    recordings = list(rec_type)
    tuning_ids = {r: set(tuning[r].unit_ids.tolist()) for r in recordings}
    # "9:347" rather than "session9:347"; only a shared prefix of letters is
    # dropped, so session1 and session10 stay apart
    prefix = re.match(r"\D*", os.path.commonprefix(recordings)).group(0)
    short = {r: r[len(prefix) :] or r for r in recordings}
    legend = [
        f"{r} ({rec_type[r]}"
        + (f", turned {rotations[r]['rotation_deg']:+.0f} deg" if r in rotations else "")
        + ")"
        for r in recordings
    ]

    plotted = sorted({unit for (_, unit) in partner})
    if unitmatch_mode == "joint":
        plotted = [u for u in plotted if all((r, u) in partner for r in others)]
    print(
        f"drawing {len(plotted)} baseline units, matched in "
        f"{'every' if unitmatch_mode == 'joint' else 'at least one'} recording"
    )

    curves, titles = {}, {}
    for u in plotted:
        found = {}
        base_curve = _curve(baseline, u, tuning, tuning_ids)
        if base_curve is not None:
            found[baseline] = (tuning[baseline].bin_centers_deg, base_curve)
        names = []
        for r in others:
            v = partner.get((r, u))
            if v is None:
                names.append(f"{short[r]}:-")
                continue
            other_curve = _curve(r, v, tuning, tuning_ids)
            names.append(f"{short[r]}:{v}" + ("" if other_curve is not None else "*"))
            if other_curve is not None:
                found[r] = (tuning[r].bin_centers_deg, other_curve)
        curves[u] = found
        tuned = " (tuned)" if u in tuned_base else ""
        titles[u] = f"{u}{tuned} | " + " ".join(names)

    colors = recording_colors(rec_type.values())
    return plot_tracked_tuning(curves, recordings, colors, labels=legend, titles=titles)


def save_figures(figures, pdf_path, png_dpi=None):
    """Save figures as the pages of one PDF and, with `png_dpi`, each as a numbered PNG too."""
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = Path(pdf_path)
    with PdfPages(pdf_path) as pdf:
        for page, fig in enumerate(figures, start=1):
            pdf.savefig(fig)
            if png_dpi:
                fig.savefig(pdf_path.with_name(f"{pdf_path.stem}_{page}.png"), dpi=png_dpi)


# ── the decoder ──────────────────────────────────────────────────────────────


def decoder_units(partner, tuned_base, rec_type, unitmatch_mode, min_decoder_units=5):
    """Choose the tuned baseline units the decoder is trained on and read out with.

    pairwise  a recording is decoded with the tuned baseline units found in it,
              and the decoder is trained on every unit found anywhere
    joint     every recording is decoded with the tuned baseline units found
              in all of them, the units the decoder is trained on

    A recording with fewer than `min_decoder_units` is left out of
    `decoder_sets`, with a warning.

    Returns
    -------
    decoder_sets : dict
        ``{recording: baseline units}`` to decode it with.
    train_units : list
        Baseline units to train on.

    Raises
    ------
    ValueError
        If fewer than `min_decoder_units` units are left to train on.
    """
    _, others = _split(rec_type)
    found_in = {r: {u for (rec, u) in partner if rec == r} for r in others}
    if unitmatch_mode == "pairwise":
        decoder_sets = {r: [u for u in tuned_base if u in found_in[r]] for r in others}
        train_units = [u for u in tuned_base if any(u in s for s in decoder_sets.values())]
    else:
        common = [u for u in tuned_base if all(u in found_in[r] for r in others)]
        decoder_sets = {r: common for r in others}
        train_units = common
    for r in others:
        print(f"{r} ({rec_type[r]}): {len(decoder_sets[r])} tuned baseline units matched")
    if len(train_units) < min_decoder_units:
        raise ValueError(
            f"only {len(train_units)} tuned baseline units were matched, fewer than "
            f"min_decoder_units = {min_decoder_units}: too few to decode. The matches and "
            "tuning plots are saved; baseline_units = 'tuned' searches every tuned unit"
            + (
                ", and pairwise mode needs them found in just one recording"
                if unitmatch_mode == "joint"
                else ""
            )
        )
    for r in others:
        if len(decoder_sets[r]) < min_decoder_units:
            warnings.warn(
                f"{r}: only {len(decoder_sets[r])} tuned baseline units were matched there, "
                "so it is not decoded"
            )
            del decoder_sets[r]
    return decoder_sets, train_units


def train_decoder(paths, tuning, baseline, train_units, decoder_settings):
    """Train the decoder on the baseline exactly as notebook 2 trains it."""
    hd = tuning[baseline]
    print(f"\n=== training on {baseline}: {len(train_units)} units")
    return run_decoder(
        sorting=_load_analyzer(paths[baseline]["analyzer"]),
        unit_ids=train_units,
        heading_deg=hd.heading_deg,
        frame_times=np.load(paths[baseline]["shutter"]),
        intervals=load_states(paths[baseline]["states"]).intervals,
        interval_mask=hd.interval_mask,
        **decoder_settings,
    )


def transfer_decoder(
    decoder_run,
    paths,
    tuning,
    partner,
    decoder_sets,
    rec_type,
    unitmatch_mode,
    decoder_settings,
    transfer_shuffles=0,
    rotations=None,
):
    """Read the baseline's decoder out on every other recording.

    Each baseline unit's partner stands in for it. Each wake decode is scored
    raw and after removing its constant offset: a population whose preferred
    directions all turned by +x is decoded at -x. The number to compare a
    recording's decode with is its reference, the baseline's own held-out wake
    and REM decoded with the same units: in joint mode that is the baseline's
    column itself, in pairwise mode one reference per recording.

    Returns
    -------
    baseline_column : TransferRun
        The baseline's held-out data decoded with every unit the model holds.
    references : dict
        ``{recording: TransferRun}``, the baseline with its units.
    transfers : dict
        ``{recording: TransferRun}``, the recording itself.
    """
    baseline, _ = _split(rec_type)
    rotations = rotations or {}
    shared = {
        key: decoder_settings[key]
        for key in ("acausal", "tolerance_deg", "min_shift_s", "seed")
    }
    shared["n_shuffles"] = transfer_shuffles
    verbose = decoder_settings["verbose"]
    train_units = np.asarray(decoder_run.model.unit_ids).tolist()

    if unitmatch_mode == "joint":
        baseline_column = reference_decode(
            decoder_run, train_units, label=baseline, verbose=verbose, **shared
        )
    else:
        # every unit the model holds; run_decoder has already tested it against its shuffle
        baseline_column = reference_decode(
            decoder_run,
            train_units,
            label=baseline,
            verbose=False,
            **{**shared, "n_shuffles": 0},
        )

    references, transfers = {}, {}
    for r, units in decoder_sets.items():
        print(f"\n=== {r} ({rec_type[r]}): {len(units)} units")
        references[r] = (
            baseline_column
            if unitmatch_mode == "joint"
            else reference_decode(
                decoder_run,
                units,
                label=f"{baseline} with {r}'s units",
                verbose=verbose,
                **shared,
            )
        )
        transfers[r] = apply_decoder(
            decoder_run.model,
            _load_analyzer(paths[r]["analyzer"]),
            {u: partner[(r, u)] for u in units},
            tuning[r].heading_deg,
            np.load(paths[r]["shutter"]),
            load_states(paths[r]["states"]).intervals,
            interval_mask=tuning[r].interval_mask,
            movement_var_deg2=decoder_run.movement_var_deg2,
            bin_s=decoder_settings["bin_s"],
            decode_rem=decoder_settings["decode_rem"],
            decode_nrem=decoder_settings.get("decode_nrem", False),
            keep_posterior=decoder_settings["keep_posterior"],
            label=r,
            verbose=verbose,
            **shared,
        )
        if r in rotations:
            print(
                f"  its tuning turned {rotations[r]['rotation_deg']:+.0f} deg from the baseline's"
            )
    return baseline_column, references, transfers


def save_transfer(transfer, path):
    """A TransferRun's decodes and nulls, as arrays."""
    arrays = {"unit_ids": transfer.unit_ids, "partner_ids": transfer.partner_ids}
    for state in ("wake", "rem", "nrem"):
        decoded = getattr(transfer, state)
        if decoded is None:
            continue
        for field in (
            "time_s",
            "decoded_deg",
            "actual_deg",
            "posterior_max",
            "entropy_bits",
            "run_index",
        ):
            arrays[f"{state}_{field}"] = getattr(decoded, field)
    for metric, test in transfer.wake_shuffles.items():
        arrays[f"wake_null_{metric}"] = test.null
    if transfer.rem_shuffle is not None:
        arrays["rem_null_mean_posterior_max"] = transfer.rem_shuffle.null
    if transfer.nrem_shuffle is not None:
        arrays["nrem_null_mean_posterior_max"] = transfer.nrem_shuffle.null
    np.savez_compressed(path, **arrays)


def save_decoder(
    folder, decoder_run, baseline_column, references, transfers, rec_type, unitmatch_mode
):
    """Save the baseline's run, every decode, and a summary table of them all.

    baseline_run.npz holds the baseline's run as notebook 4 reads it,
    ``<recording>_decode.npz`` each transfer (:func:`save_transfer`), and in
    pairwise mode ``<recording>_reference_decode.npz`` each reference.

    Returns
    -------
    pandas.DataFrame
        One row per decode: the baseline, then each recording's reference and
        transfer.
    """
    folder = Path(folder)
    baseline, _ = _split(rec_type)
    empty = np.array([])
    rem, nrem = decoder_run.rem, decoder_run.nrem
    np.savez_compressed(
        folder / "baseline_run.npz",
        unit_ids=decoder_run.model.unit_ids,
        tuning_rate_hz=decoder_run.model.rate_hz,
        tuning_bin_centers_deg=decoder_run.model.bin_centers_deg,
        occupancy_s=decoder_run.model.occupancy_s,
        movement_var_deg2=decoder_run.movement_var_deg2,
        test_time_s=decoder_run.test.time_s,
        test_decoded_deg=decoder_run.test.decoded_deg,
        test_actual_deg=decoder_run.test.actual_deg,
        test_posterior_max=decoder_run.test.posterior_max,
        test_entropy_bits=decoder_run.test.entropy_bits,
        test_null=(
            decoder_run.test_shuffle.null if decoder_run.test_shuffle is not None else empty
        ),
        rem_time_s=rem.time_s if rem is not None else empty,
        rem_decoded_deg=rem.decoded_deg if rem is not None else empty,
        rem_posterior_max=rem.posterior_max if rem is not None else empty,
        nrem_time_s=nrem.time_s if nrem is not None else empty,
        nrem_decoded_deg=nrem.decoded_deg if nrem is not None else empty,
        nrem_posterior_max=nrem.posterior_max if nrem is not None else empty,
    )

    def row(r, role, run):
        return {"recording": r, "type": rec_type[r], "role": role, **run.as_row()}

    rows = [row(baseline, "baseline", baseline_column)]
    texts = [decoder_run.summary()]
    for r in transfers:
        save_transfer(transfers[r], folder / f"{r}_decode.npz")
        if unitmatch_mode == "pairwise":
            save_transfer(references[r], folder / f"{r}_reference_decode.npz")
        rows += [row(r, "reference", references[r]), row(r, "transfer", transfers[r])]
        texts += [references[r].summary(), transfers[r].summary()]
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "decoder_summary.csv", index=False)
    (folder / "decoder_summary.txt").write_text("\n\n".join(texts))
    return summary


def plot_decoder_summary(baseline_column, transfers, rec_type, unitmatch_mode):
    """Plot how the baseline's decoder did on each recording, in recording order.

    Filled: raw wake error; open: with the constant offset removed; gray: the
    shuffled null.

    Returns
    -------
    matplotlib.figure.Figure
    """
    baseline, _ = _split(rec_type)
    color_of = dict(zip(rec_type, recording_colors(rec_type.values())))
    columns = {  # in recording order, so the columns read as a time course
        r: baseline_column if r == baseline else transfers[r]
        for r in rec_type
        if r == baseline or r in transfers
    }
    axes = plot_transfer_summary(
        columns, colors={r: color_of[r] for r in columns}, references=None
    )
    fig = axes[0].figure
    fig.suptitle(
        f"decoder trained on {baseline}, {len(baseline_column.unit_ids)} units "
        f"({unitmatch_mode})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ── rings ────────────────────────────────────────────────────────────────────


def ring_band(hd_base, ring_depth_um=None):
    """The depth band of the head-direction structure on the baseline's probe, in um.

    `ring_depth_um` if given, else the 10th-90th percentiles of the baseline's
    tuned units' depths, widened by 100 um each way.
    """
    tuned_depth = hd_base.depths[np.isin(hd_base.unit_ids, hd_base.tuned_ids)]
    return ring_depth_um or (
        float(np.percentile(tuned_depth, 10)) - 100.0,
        float(np.percentile(tuned_depth, 90)) + 100.0,
    )


def probe_drift(partner, tuning, rec_type):
    """The probe's drift from the baseline to each recording, in um.

    The median depth change of a recording's matched units, so which units
    fall in a ring's band is anatomy: only this shift, tens of um, leans on
    the matches.

    Returns
    -------
    dict
        ``{recording: drift_um}``, 0 for the baseline.
    """
    baseline, others = _split(rec_type)
    base_depth = tuning[baseline].depth_dict()
    drift_um = {baseline: 0.0}
    for r in others:
        depth_r = tuning[r].depth_dict()
        shifts = [
            depth_r[v] - base_depth[u]
            for (rec, u), v in partner.items()
            if rec == r and u in base_depth and v in depth_r
        ]
        if not shifts:
            warnings.warn(
                f"{r}: no matched unit has a depth in both recordings, so its band is not shifted"
            )
        drift_um[r] = float(np.median(shifts)) if shifts else 0.0
    return drift_um


def run_rings(paths, tuning, rec_type, band_um, drift_um, decoder_settings, ring_settings):
    """Fit each recording's ring from its own spikes alone (:func:`spikeshpc.ring.run_ring`).

    Each recording's units in the band, shifted by its drift, are placed on a
    ring by how they co-fire. Neither tuning nor the matching goes into a ring,
    so the rings can test the matching. Heading is used once per recording, on
    its own training wake, to say where 0 deg is on its ring and which way
    round it runs. The decoder's split and bin_s are used, so a ring's decodes
    sit on the decoder's grid. A recording whose ring cannot be fitted is
    skipped with a warning.

    Returns
    -------
    ring_runs : dict
        ``{recording: RingRun}``.
    ring_window : dict
        ``{recording: (low_um, high_um)}``, the band each ring was fitted in.
    """
    ring_runs, ring_window = {}, {}
    for r in rec_type:
        low, high = band_um[0] + drift_um[r], band_um[1] + drift_um[r]
        depth_r = tuning[r].depth_dict()
        units = [u for u in tuning[r].unit_ids.tolist() if low <= depth_r[u] <= high]
        n_tuned = int(np.isin(units, tuning[r].tuned_ids).sum())
        print(
            f"\n=== {r} ({rec_type[r]}): {len(units)} GOOD + MUA units in "
            f"{low:.0f}-{high:.0f} um (drift {drift_um[r]:+.0f} um), {n_tuned} tuned"
        )
        try:
            ring_runs[r] = run_ring(
                sorting=_load_analyzer(paths[r]["analyzer"]),
                unit_ids=units,
                heading_deg=tuning[r].heading_deg,
                frame_times=np.load(paths[r]["shutter"]),
                intervals=load_states(paths[r]["states"]).intervals,
                interval_mask=tuning[r].interval_mask,
                bin_s=decoder_settings["bin_s"],
                test_fraction=decoder_settings["test_fraction"],
                split_mode=decoder_settings["split_mode"],
                block_s=decoder_settings["block_s"],
                seed=decoder_settings["seed"],
                tolerance_deg=decoder_settings["tolerance_deg"],
                min_shift_s=decoder_settings["min_shift_s"],
                **ring_settings,
            )
        except ValueError as error:
            warnings.warn(f"{r}: no ring ({error})")
            continue
        ring_window[r] = (low, high)
        print(ring_runs[r].summary())
    return ring_runs, ring_window


def ring_summary(ring_runs, rec_type, drift_um):
    """One row of headline numbers per ring."""
    return pd.DataFrame(
        {
            r: {"type": rec_type[r], "drift_um": drift_um[r], **run.as_row()}
            for r, run in ring_runs.items()
        }
    ).T


def plot_rings(ring_runs, rec_type):
    """Plot each ring (:func:`spikeshpc.ring.plot_ring`), one figure per recording."""
    figures = []
    for r, run in ring_runs.items():
        axes = plot_ring(run.ring)
        axes[0].figure.suptitle(f"{r} ({rec_type[r]})", fontsize=10)
        figures.append(axes[0].figure)
    return figures


def compare_matched_rings(ring_runs, partner, rec_type, n_shuffles=1000):
    """Ask whether the matched units keep their places on the ring.

    The baseline's ring against each other recording's, on the matched pairs
    both rings hold (:func:`spikeshpc.ring.compare_rings`), each against the
    partners re-paired at random:

    positions  where a partner sits on its ring against where its baseline
               unit sits on the baseline's, after the best reflection and
               rotation. Read it on the members of both rings too: a
               non-member's place is noise
    structure  every two matched units correlated alike in both recordings (a
               Mantel test of the pair metric). It needs no embedding, so it
               answers even where a ring is too degraded to read

    Agreement says the matches are the same cells AND that their network kept
    its structure. Each ring's split-half r is about as high as the structure r
    can get. Fewer than 4 pairs are not compared.

    Returns
    -------
    comparisons : dict
        ``{(recording, "on both rings" | "members of both"): RingComparison}``.
    table : pandas.DataFrame
        One row per recording and set of pairs.
    """
    baseline, others = _split(rec_type)
    comparisons, rows = {}, []
    for r in others:
        if baseline not in ring_runs or r not in ring_runs:
            continue
        ring_a, ring_b = ring_runs[baseline].ring, ring_runs[r].ring
        on_a, on_b = set(ring_a.unit_ids.tolist()), set(ring_b.unit_ids.tolist())
        members_a = set(ring_a.members().tolist())
        members_b = set(ring_b.members().tolist())
        unit_map = {u: v for (rec, u), v in partner.items() if rec == r}
        subsets = {
            "on both rings": unit_map,
            "members of both": {
                u: v for u, v in unit_map.items() if u in members_a and v in members_b
            },
        }
        for subset, pairs in subsets.items():
            n_pairs = sum(u in on_a and v in on_b for u, v in pairs.items())
            row = {
                "recording": r,
                "type": rec_type[r],
                "pairs": subset,
                "matched": len(unit_map),
                "n_pairs": n_pairs,
            }
            if n_pairs >= 4:
                comparison = compare_rings(
                    ring_a, ring_b, unit_map=pairs, n_shuffles=n_shuffles
                )
                comparisons[r, subset] = comparison
                row.update(comparison.as_row())
                print(f"\n{baseline} vs {r}, {subset}: {comparison.summary()}")
            else:
                print(f"\n{baseline} vs {r}, {subset}: {n_pairs} pairs, too few to compare")
            rows.append(row)
    return comparisons, pd.DataFrame(rows).set_index(["recording", "pairs"])


def plot_ring_comparisons(comparisons, rec_type):
    """Plot each comparison (:func:`spikeshpc.ring.plot_ring_comparison`)."""
    baseline, _ = _split(rec_type)
    figures = []
    for (r, subset), comparison in comparisons.items():
        axes = plot_ring_comparison(comparison, labels=(baseline, r))
        axes[0].figure.suptitle(
            f"{baseline} vs {r} ({rec_type[r]}), matched units {subset}", fontsize=10
        )
        figures.append(axes[0].figure)
    return figures


def rings_against_tuning(ring_runs, tuning, rec_type, n_shuffles=1000):
    """Hold each ring against its recording's tuning (:func:`spikeshpc.ring.ring_vs_tuning`).

    Before the lesion the ring's members are the tuned units, sitting at their
    preferred directions. A ring that keeps its matched units' places
    (:func:`compare_matched_rings`) while its tuned units stop sitting at their
    preferred directions is a network intact but no longer anchored to the
    head; a ring that loses both is a network that changed.

    Returns
    -------
    results : dict
        ``{recording: RingTuning}``.
    table : pandas.DataFrame
        One row per recording.
    """
    results, rows = {}, []
    for r, run in ring_runs.items():
        result = ring_vs_tuning(run.ring, tuning[r], n_shuffles=n_shuffles)
        print(f"\n{r} ({rec_type[r]}): {result.summary()}")
        results[r] = result
        table, angles = result.table, result.angles
        rows.append(
            {
                "recording": r,
                "type": rec_type[r],
                "tuned": int(table["tuned"].sum()),
                "tuned members": int((table["tuned"] & table["member"]).sum()),
                "untuned members": int((~table["tuned"] & table["member"]).sum()),
                "auc_partner_snr": result.auc_partner_snr,
                "auc_coupling": result.auc_coupling,
                "position_circular_r": angles.circular_r if angles else np.nan,
                "position_error_deg": angles.median_abs_error_deg if angles else np.nan,
                "position_p": (
                    angles.shuffle.p_value if angles and angles.shuffle else np.nan
                ),
                "structure_r": result.structure_r,
                "structure_p": result.structure.p_value if result.structure else np.nan,
            }
        )
    return results, pd.DataFrame(rows).set_index("recording")


def plot_rings_against_tuning(results, rec_type):
    """Plot each ring against its tuning (:func:`spikeshpc.ring.plot_ring_vs_tuning`)."""
    figures = []
    for r, result in results.items():
        axes = plot_ring_vs_tuning(result)
        axes[0].figure.suptitle(f"{r} ({rec_type[r]})", fontsize=10)
        figures.append(axes[0].figure)
    return figures


def save_rings(
    folder, ring_runs, comparisons, rec_type, band_um, drift_um, ring_window, ring_settings
):
    """Save each ring and its decodes, for notebook 4.

    Notebook 4 reads a ring's decode as heading one of two ways:

    own       through the recording's own alignment, fitted to its own heading
    baseline  through its matched units onto the baseline's ring (the members
              of both where there are 4 or more, else every pair on both), then
              through the baseline's alignment. It leans on the matching tested
              by :func:`compare_matched_rings` and not at all on the
              recording's own tuning

    Decodes are saved in the ring's own frame (``*_ring_deg``), so either
    applies.
    """
    folder = Path(folder)
    baseline, _ = _split(rec_type)
    for r, run in ring_runs.items():
        own = run.alignment
        via, via_p, via_pairs = None, np.nan, None
        if r == baseline:
            via, via_pairs = own, "itself"
        else:
            via_pairs = next(
                (s for s in ("members of both", "on both rings") if (r, s) in comparisons),
                None,
            )
            if via_pairs is not None:
                comparison = comparisons[r, via_pairs]
                via = comparison.angles.alignment.then(ring_runs[baseline].alignment)
                if comparison.angles.shuffle is not None:
                    via_p = comparison.angles.shuffle.p_value
        ring = run.ring
        arrays = dict(
            unit_ids=ring.unit_ids,
            angle_deg=ring.angle_deg,
            coupling=ring.coupling,
            member=np.isin(ring.unit_ids, ring.members()),
            depth_window_um=np.array(ring_window[r]),
            drift_um=drift_um[r],
            own_flip=own.flip,
            own_offset_deg=own.offset_deg,
            own_resultant=own.resultant,
            own_margin=own.margin,
        )
        if via is not None:
            arrays.update(
                baseline_flip=via.flip,
                baseline_offset_deg=via.offset_deg,
                baseline_p=via_p,
                baseline_pairs=np.array(via_pairs),
            )
        for key, decoded in (("wake", run.test), ("rem", run.rem), ("nrem", run.nrem)):
            if decoded is None:
                continue
            arrays.update(
                {
                    f"{key}_time_s": decoded.time_s,
                    f"{key}_ring_deg": own.invert(decoded.decoded_deg),
                    f"{key}_actual_deg": decoded.actual_deg,
                    f"{key}_run_index": decoded.run_index,
                    f"{key}_length": decoded.posterior_max,
                }
            )
        np.savez_compressed(folder / f"{r}_ring.npz", **arrays)

    ring_summary(ring_runs, rec_type, drift_um).to_csv(folder / "rings.csv")
    (folder / "ring_settings.json").write_text(
        json.dumps(
            {
                "band_um": band_um,
                "drift_um": drift_um,
                "windows_um": ring_window,
                "ring_settings": ring_settings,
            },
            indent=2,
        )
    )
    (folder / "rings.txt").write_text(
        "\n\n".join(f"{r} ({rec_type[r]})\n{run.summary()}" for r, run in ring_runs.items())
    )
    print(f"saved {len(ring_runs)} rings to {folder}")
