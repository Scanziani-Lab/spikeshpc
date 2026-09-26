"""Hand bombcell, UnitRefine and SLAy's calls to spikeinterface-gui, and read back yours.

Three automated tools each have an opinion about every unit. bombcell and
UnitRefine label it (good / MUA / noise, each in its own words); SLAy proposes
which units kilosort split in two. This module puts all three in front of
spikeinterface-gui at once -- each tool's call as a sortable column in the unit
table, a quality label pre-filled wherever bombcell and UnitRefine agree, SLAy's
pairs listed in the Merge tab to accept or ignore -- and turns what you save
there into a per-unit table.

Everything passes through files in the session's curation folder, not through
objects, because the GUI runs in a process of its own (``python -m
spikeshpc.curation``, started by :func:`launch_gui`). Two things force that:

  * Qt bindings. ``%matplotlib qt`` in the notebook loads PyQt6 (matplotlib's
    first choice). spikeinterface-gui imports PySide6, and pyqtgraph, finding
    both, takes PyQt6: widgets of two bindings in one window, which fails. A
    fresh interpreter has only PySide6.
  * Size. On a 13 h session the GUI's controller builds several GB of spike
    arrays before the window appears; if that dies, it should not take the
    notebook's kernel -- and an afternoon of SLAy -- with it.

Two more things this module exists to get right:

  * valid_unit_periods. Splitting or merging units makes spikeinterface
    recompute it for the new units in a process pool whose initializer is
    handed the whole sorting. Windows starts workers by pickling that through
    a pipe, and at 10^8 spikes the write fails (OSError 22) -- which is what
    stopped SLAy's automatic parameter search, and what would stop
    ``apply_curation``. :func:`detached_extensions` hides it from both.
  * Staleness. Every cache here is keyed by unit id, and a re-sort numbers
    its units 0..N-1 all over again, so a label for unit 12 from last week's
    sorting reloads without complaint onto a different unit 12.
    :func:`check_curation_dir` ties the folder to one sorting and refuses to
    mix it with another.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANALYZER_NAME

# ── the hand-off files, all in the session's curation folder ─────────────
CURATION_DIRNAME = "curation"  # the default folder, next to the analyzer
SORTING_FILE = "sorting.json"  # which sorting every file here belongs to
UNITREFINE_FILE = "unitrefine_labels.csv"
SLAY_FILE = "slay_merges.npz"
AUTOMATED_FILE = "automated_labels.csv"  # every tool's call, one row per unit
CURATION_FILE = "sigui_curation.json"  # what the GUI's "Save curation" writes
LOG_FILE = "sigui.log"
CURATED_FILE = "curated_units.csv"  # the automated calls plus your decisions
BOMBCELL_MANUAL_FILE = "manual_unit_classifications.csv"  # bombcell's GUI writes it

# SLAy's pairwise scores, in the order the Merge tab shows them
SLAY_METRICS = ("final_metric", "similarity", "ccg_metric", "refractory_penalty")

# spikeinterface-gui's own default. Its g/m/n/c shortcuts are switched on only
# when the quality options are exactly these, so the set must not change.
QUALITY = {"quality": {"label_options": ["good", "noise", "MUA"], "exclusive": True}}

# every tool's label that has a quality equivalent: bombcell's own calls, the
# names of your calls in bombcell's GUI, and UnitRefine's. Non-somatic units
# and missing calls have none, so they never count as agreement.
TO_QUALITY = {
    "GOOD": "good", "MUA": "MUA", "NOISE": "noise",
    "good": "good", "noise": "noise",
    "sua": "good", "mua": "MUA",
}
BOMBCELL_MANUAL_NAMES = {0: "noise", 1: "good", 2: "MUA", 3: "non-somatic"}  # -1 = not yet classified

TEXT_COLUMNS = ("auto_label", "bc_label", "bc_manual", "ur_label", "slay_partners")
NUMERIC_COLUMNS = ("ur_prob", "slay_group", "slay_score")

# quality metrics and sorting properties shown next to the automated calls,
# which the GUI appends after these; any missing from this analyzer are skipped
DISPLAYED_COLUMNS = [
    "KSLabel", "num_spikes", "firing_rate", "snr", "amplitude_median",
    "rp_contamination", "presence_ratio", "y",
]

# the GUI process's Qt: PySide6 throughout, pyqtgraph included
QT_ENV = {"QT_API": "pyside6", "PYQTGRAPH_QT_LIB": "PySide6"}

_running = {}  # curation folder -> the GUI process launched for it


# ── staleness ─────────────────────────────────────────────────────────────
def sorting_fingerprint(analyzer) -> dict:
    """What identifies the sorting an analyzer holds, and the waveforms it measured.

    Unit ids alone identify nothing -- every sorting has a unit 0 -- but the
    spike count of every unit does, and a checksum of the average templates
    also catches an analyzer rebuilt on a different binary around the same
    sorting.
    """
    counts = analyzer.sorting.count_num_spikes_per_unit(outputs="array")
    templates = analyzer.get_extension("templates")
    digest = None
    if templates is not None:
        average = np.ascontiguousarray(templates.get_data(operator="average"))
        digest = hashlib.sha1(average.tobytes()).hexdigest()
    return {
        "n_units": int(len(counts)),
        "n_spikes": int(np.sum(counts)),
        "templates_sha1": digest,
        "unit_ids": [_py(u) for u in analyzer.unit_ids],
        "num_spikes": [int(c) for c in counts],
    }


def check_curation_dir(curation_dir, analyzer) -> None:
    """Refuse to use a curation folder made for a different sorting.

    The first call writes ``sorting.json``, tying the folder -- and whatever is
    in it already -- to this analyzer's sorting. Every later call compares, and
    raises on any difference rather than let a cache from the previous sort
    load onto this one. Nothing is moved or deleted: that is left to you.
    """
    curation_dir = Path(curation_dir)
    curation_dir.mkdir(parents=True, exist_ok=True)
    path = curation_dir / SORTING_FILE
    current = sorting_fingerprint(analyzer)

    if not path.is_file():
        record = {"analyzer": str(getattr(analyzer, "folder", None)),
                  "created": datetime.now().isoformat(timespec="seconds"), **current}
        path.write_text(json.dumps(record))
        others = [p.name for p in curation_dir.iterdir() if p.name != SORTING_FILE]
        note = (f" The {len(others)} file(s) already there are taken to be from it too."
                if others else "")
        print(f"{curation_dir} now belongs to this sorting "
              f"({current['n_units']} units, {current['n_spikes']:,} spikes).{note}")
        return

    saved = json.loads(path.read_text())
    changes = []
    if saved["unit_ids"] != current["unit_ids"]:
        changes.append(f"units {saved['n_units']} -> {current['n_units']}")
    elif saved["num_spikes"] != current["num_spikes"]:
        changed = sum(a != b for a, b in zip(saved["num_spikes"], current["num_spikes"]))
        changes.append(f"spike counts differ for {changed} units "
                       f"({saved['n_spikes']:,} -> {current['n_spikes']:,} spikes)")
    if None not in (saved.get("templates_sha1"), current["templates_sha1"]) and \
            saved["templates_sha1"] != current["templates_sha1"]:
        changes.append("the templates differ (analyzer rebuilt?)")
    if changes:
        raise RuntimeError(
            f"{curation_dir} holds results for a different sorting than this analyzer "
            f"({'; '.join(changes)}; recorded {saved.get('created', '?')}). Move or delete "
            "that folder, or point curation_path somewhere new, and run again."
        )


# ── valid_unit_periods on Windows ─────────────────────────────────────────
@contextmanager
def detached_extensions(analyzer, *names):
    """Hide extensions from `analyzer` for the length of a with-block.

    Splits and merges carry over only the extensions the analyzer has loaded,
    so an extension popped from ``analyzer.extensions`` is simply not
    recomputed for the new units. Memory only: the saved analyzer keeps it,
    and it is put back when the block ends, however it ends.
    """
    held = {name: analyzer.extensions.pop(name) for name in names if name in analyzer.extensions}
    try:
        yield analyzer
    finally:
        analyzer.extensions.update(held)


# ── SLAy ──────────────────────────────────────────────────────────────────
def save_slay(curation_dir, merges, metrics) -> dict:
    """Keep what ``compute_slay_merges`` returned; returns it as :func:`load_slay` would.

    The merge groups are ragged, so they are stored flat: every member, and
    the index of the group it belongs to.
    """
    merges = [[_py(u) for u in group] for group in merges]
    np.savez(
        Path(curation_dir) / SLAY_FILE,
        group_members=np.array([u for group in merges for u in group], dtype=np.int64),
        group_index=np.array([g for g, group in enumerate(merges) for _ in group], dtype=np.int64),
        **{name: np.asarray(metrics[name], dtype=float) for name in SLAY_METRICS},
    )
    return {"merges": merges, **{name: np.asarray(metrics[name], dtype=float) for name in SLAY_METRICS}}


def load_slay(curation_dir, n_units: int) -> dict:
    """``{"merges": [[unit ids], ...], metric: n_units x n_units}`` from :func:`save_slay`."""
    with np.load(Path(curation_dir) / SLAY_FILE) as f:
        members, index = f["group_members"], f["group_index"]
        n_groups = int(index.max()) + 1 if len(index) else 0
        slay = {"merges": [members[index == g].tolist() for g in range(n_groups)]}
        for name in SLAY_METRICS:
            if f[name].shape != (n_units, n_units):
                raise ValueError(
                    f"{SLAY_FILE} has {name} for {f[name].shape[0]} units, not {n_units}: "
                    "it was computed for another sorting. Delete it and run SLAy again."
                )
            slay[name] = f[name]
    return slay


def slay_pairs(slay, unit_ids) -> list:
    """Every pair within each SLAy group, most confident first.

    The Merge tab takes pairs: its table has a column per pairwise score only
    when every group has two units (larger groups make it fail), and pairs let
    each one be judged on its own. Accepting two pairs that share a unit
    merges all three, so nothing is lost by splitting a group up.
    """
    position = {_py(u): i for i, u in enumerate(unit_ids)}
    score = slay["final_metric"]
    pairs = [pair for group in slay["merges"] for pair in itertools.combinations(group, 2)]
    pairs.sort(key=lambda p: -score[position[p[0]], position[p[1]]])
    return [list(p) for p in pairs]


# ── the per-unit table ────────────────────────────────────────────────────
def automated_labels(unit_ids, bc_qm, bc_type_string, ur_labels, slay=None,
                     bc_manual_file=None) -> pd.DataFrame:
    """Every tool's call for every unit, indexed by unit id in the analyzer's order.

    Columns:
      auto_label     good / MUA / noise where bombcell and UnitRefine agree,
                     else "" -- the GUI's starting label, so the blanks are
                     what needs looking at
      bc_label       bombcell's type, matched to units by phy_clusterID (its
                     rows are not assumed to be in unit order)
      bc_manual      your call in bombcell's GUI, if `bc_manual_file` exists;
                     where set, it stands in for bc_label in auto_label
      ur_label       UnitRefine's call (noise / sua / mua), and ur_prob its
                     probability
      slay_group     the SLAy merge group the unit is in, -1 if none, with
                     slay_partners the other members and slay_score the best
                     final_metric to one of them
    """
    index = pd.Index([_py(u) for u in unit_ids], name="unit_id")
    table = pd.DataFrame(index=index)

    bc_ids = np.asarray(bc_qm["phy_clusterID"]).astype(np.int64)
    bombcell = pd.Series(np.asarray(bc_type_string, dtype=object), index=bc_ids)
    _require_overlap(index, bombcell.index, "bombcell's phy_clusterID")
    table["bc_label"] = bombcell.reindex(index).fillna("").astype(str)

    if bc_manual_file is not None and Path(bc_manual_file).is_file():
        manual = pd.read_csv(bc_manual_file)
        codes = pd.Series(manual["manual_classification"].to_numpy(),
                          index=manual["unit_id"].to_numpy().astype(np.int64))
        table["bc_manual"] = codes.reindex(index).map(BOMBCELL_MANUAL_NAMES).fillna("").astype(str)

    _require_overlap(index, ur_labels.index, "UnitRefine's labels")
    unitrefine = ur_labels.reindex(index)
    table["ur_label"] = unitrefine["unitrefine_label"].fillna("").astype(str)
    table["ur_prob"] = unitrefine["unitrefine_probability"].astype(float)

    table["slay_group"] = -1
    table["slay_partners"] = ""
    table["slay_score"] = np.nan
    if slay is not None:
        position = {u: i for i, u in enumerate(index)}
        # .loc would quietly add a row for a unit the analyzer does not have
        unknown = sorted({u for group in slay["merges"] for u in group} - set(position))
        if unknown:
            raise ValueError(f"SLAy's merge groups name units this analyzer does not have: {unknown[:10]}")
        score = slay["final_metric"]
        for g, group in enumerate(slay["merges"]):
            for u in group:
                partners = [p for p in group if p != u]
                table.loc[u, "slay_group"] = g
                table.loc[u, "slay_partners"] = ",".join(str(p) for p in partners)
                table.loc[u, "slay_score"] = max(score[position[u], position[p]] for p in partners)

    bombcell_call = table["bc_label"]
    if "bc_manual" in table:
        bombcell_call = table["bc_manual"].where(table["bc_manual"] != "", table["bc_label"])
    ours, theirs = bombcell_call.map(TO_QUALITY), table["ur_label"].map(TO_QUALITY)
    table["auto_label"] = ours.where(ours.notna() & (ours == theirs), "").astype(str)
    return table[["auto_label", *[c for c in table.columns if c != "auto_label"]]]


def read_automated_labels(curation_dir) -> pd.DataFrame:
    """The table :func:`automated_labels` wrote, with its types exactly as they were.

    Read as text and converted column by column: left to pandas, an empty
    label becomes NaN (then the string "nan"), and a partner list that is a
    single unit id becomes a number.
    """
    table = pd.read_csv(Path(curation_dir) / AUTOMATED_FILE, dtype=str, keep_default_na=False)
    table = table.set_index("unit_id")
    try:
        table.index = table.index.astype(np.int64)
    except ValueError:
        pass  # string unit ids
    for col in NUMERIC_COLUMNS:
        if col in table:
            table[col] = pd.to_numeric(table[col].where(table[col] != ""))
    return table


def sigui_properties(table) -> dict:
    """The table as the GUI's ``extra_unit_properties``: one 1-D array per column.

    Text must be a numpy string array. The GUI's unit table drops any column
    of object dtype -- the dtype pandas gives text -- with no more than a
    warning, so every tool's column would otherwise silently not be there.
    """
    properties = {}
    for col in table.columns:
        values = table[col]
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            properties[col] = values.to_numpy()
        else:
            properties[col] = np.asarray(values.fillna("").astype(str), dtype=str)
    return properties


def initial_curation(table) -> dict:
    """A spikeinterface curation (format 2) with ``auto_label`` as the starting labels."""
    return {
        "format_version": "2",
        "unit_ids": [_py(u) for u in table.index],
        "label_definitions": copy.deepcopy(QUALITY),
        "manual_labels": [
            {"unit_id": _py(u), "labels": {"quality": [label]}}
            for u, label in table["auto_label"].items()
            if isinstance(label, str) and label
        ],
        "merges": [],
        "splits": [],
        "removed": [],
    }


def save_curation(curation_data, path) -> None:
    """Write the GUI's curation where :func:`open_gui` and :func:`load_curation_result` find it.

    Validated first, so a curation that would not load again is refused while
    it can still be fixed, and written through a temporary file, so a crash
    mid-write cannot leave half of one.
    """
    from spikeinterface.curation.curation_model import Curation

    path = Path(path)
    text = Curation(**curation_data).model_dump_json(indent=2)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    print(f"curation saved to {path}", flush=True)


def load_curation_result(curation_dir):
    """``(curation, table)``: what you saved in the GUI, unit by unit.

    `table` is :func:`automated_labels`'s table with your decisions added --
    ``quality`` (your label, "" if none), ``removed``, ``merge_group`` (-1 if
    not merged) and ``split`` -- and is also written to ``curated_units.csv``.
    `curation` is the spikeinterface model, for ``si.apply_curation``.
    """
    from spikeinterface.curation import curation_label_to_dataframe, load_curation

    curation_dir = Path(curation_dir)
    path = curation_dir / CURATION_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {path}: press 'Save curation' in the GUI first")
    curation = load_curation(path)

    table = read_automated_labels(curation_dir)
    labels = curation_label_to_dataframe(curation)
    labels.index.name = "unit_id"
    table = table.join(labels)
    table["removed"] = table.index.isin(curation.removed)
    table["merge_group"] = -1
    for g, merge in enumerate(curation.merges):
        table.loc[merge.unit_ids, "merge_group"] = g
    table["split"] = table.index.isin([split.unit_id for split in curation.splits])
    table.to_csv(curation_dir / CURATED_FILE)
    return curation, table


# ── the GUI ───────────────────────────────────────────────────────────────
def open_gui(analyzer, table, slay, curation_dir, recording=None, title=None):
    """Build the spikeinterface-gui window; the caller runs the Qt event loop.

    The curation starts from ``sigui_curation.json`` if you saved one, and
    from :func:`initial_curation` otherwise. SLAy's pairs are put in the Merge
    tab by setting the two attributes its table is drawn from -- the tab has
    no input for proposals made elsewhere -- which ties this to
    spikeinterface-gui's internals (0.13.1); tests/test_curation.py fails if
    they move. Its "Calculate merges" button replaces the list with a
    spikeinterface preset's; reopen the GUI to get SLAy's back.
    """
    from spikeinterface_gui import run_mainwindow

    curation_dir = Path(curation_dir)
    unit_index = pd.Index([_py(u) for u in analyzer.unit_ids], name="unit_id")
    if set(table.index) != set(unit_index):
        raise ValueError(f"{AUTOMATED_FILE} does not list this analyzer's units: rebuild it (cell 2.5)")
    table = table.reindex(unit_index)  # the GUI takes each column as an array in unit order

    saved = curation_dir / CURATION_FILE
    if saved.is_file():
        curation = json.loads(saved.read_text())
        print(f"resuming the curation in {saved}", flush=True)
    else:
        curation = initial_curation(table)

    def on_save(curation_data, path):
        try:
            save_curation(curation_data, path)
        except Exception as e:
            from spikeinterface_gui.myqt import QT

            QT.QMessageBox.critical(win, "Curation not saved", f"{type(e).__name__}: {e}")
            raise

    win = run_mainwindow(
        analyzer,
        mode="desktop",
        curation=True,
        curation_dict=curation,
        label_definitions=copy.deepcopy(QUALITY),
        recording=recording,
        extra_unit_properties=sigui_properties(table),
        displayed_unit_properties=list(DISPLAYED_COLUMNS),
        curation_callback=on_save,
        curation_callback_kwargs={"path": saved},
        start_app=False,
        verbose=True,
    )
    view = win.views.get("merge")
    if slay is not None and view is not None:
        view.proposed_merge_unit_groups_all = slay_pairs(slay, analyzer.unit_ids)
        view.merge_info = {name: slay[name] for name in SLAY_METRICS}
        view._refresh()  # refresh() skips a tab that is not in front
    n_pairs = view.table.rowCount() if view is not None and view.table is not None else 0
    n_labelled = len(win.controller.curation_data["manual_labels"])
    print(f"GUI ready: {len(unit_index)} units, {n_labelled} labelled, {n_pairs} SLAy pairs "
          f"in the Merge tab", flush=True)
    if title:
        win.setWindowTitle(title)
    return win


def launch_gui(processed_dir, curation_dir, analyzer=None, wait_s: float = 5.0):
    """Start the GUI in its own process; returns it (a ``subprocess.Popen``).

    Everything it shows is read from `curation_dir`, so :func:`automated_labels`'
    table must be saved there first. Pass the notebook's `analyzer` to check
    the folder belongs to it here, where an error is seen; the GUI process
    checks again, but can only say so in ``sigui.log``. Returns once the
    process has survived `wait_s` seconds, which catches failures to start;
    anything later -- the analyzer takes minutes to load -- goes to the log.
    """
    processed_dir, curation_dir = Path(processed_dir), Path(curation_dir)
    if analyzer is not None:
        check_curation_dir(curation_dir, analyzer)
    if not (curation_dir / AUTOMATED_FILE).is_file():
        raise FileNotFoundError(f"no {curation_dir / AUTOMATED_FILE}: save automated_labels() there first")
    previous = _running.get(curation_dir)
    if previous is not None and previous.poll() is None:
        raise RuntimeError(
            f"a GUI for {curation_dir} is already open (pid {previous.pid}); close it first, "
            "or two windows will overwrite each other's saves"
        )

    log_path = curation_dir / LOG_FILE
    # utf-8: the progress bars would otherwise crash a child writing to a
    # file in Windows' ANSI code page; unbuffered, so the log is live
    env = {**os.environ, **QT_ENV, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-m", "spikeshpc.curation", str(processed_dir),
           "--curation-dir", str(curation_dir)]
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _running[curation_dir] = proc
    try:
        code = proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        print(f"SpikeInterface GUI starting in its own window (pid {proc.pid}); on a long "
              f"session it takes a few minutes to appear. Progress and errors: {log_path}")
        return proc
    raise RuntimeError(f"the GUI process exited at once (code {code}):\n\n{_tail(log_path)}")


def main(argv=None):
    """``python -m spikeshpc.curation <processed_dir> [--curation-dir DIR]``."""
    for key, value in QT_ENV.items():
        os.environ.setdefault(key, value)
    parser = argparse.ArgumentParser(
        prog="python -m spikeshpc.curation",
        description="Open spikeinterface-gui on a session's analyzer with bombcell, "
                    "UnitRefine and SLAy's calls from its curation folder.",
    )
    parser.add_argument("processed_dir", type=Path, help="the pipeline's output_dir for the session")
    parser.add_argument("--curation-dir", type=Path, default=None,
                        help=f"default: <processed_dir>/{CURATION_DIRNAME}")
    args = parser.parse_args(argv)
    processed_dir = args.processed_dir
    curation_dir = args.curation_dir or processed_dir / CURATION_DIRNAME

    import spikeinterface.full as si

    from .io import load_concatenated
    from .postprocess import analyzer_recording

    t0 = time.perf_counter()
    print(f"loading {processed_dir / ANALYZER_NAME}", flush=True)
    analyzer = si.load_sorting_analyzer(processed_dir / ANALYZER_NAME)
    check_curation_dir(curation_dir, analyzer)
    recording, _ = load_concatenated(processed_dir)
    recording = analyzer_recording(recording, analyzer)
    table = read_automated_labels(curation_dir)
    slay = load_slay(curation_dir, len(analyzer.unit_ids)) if (curation_dir / SLAY_FILE).is_file() else None
    print(f"loaded in {time.perf_counter() - t0:.0f} s; building the GUI", flush=True)

    from spikeinterface_gui.myqt import mkQApp
    import pyqtgraph.Qt

    if pyqtgraph.Qt.QT_LIB != "PySide6":
        raise RuntimeError(f"pyqtgraph is on {pyqtgraph.Qt.QT_LIB}, the GUI on PySide6: set "
                           "PYQTGRAPH_QT_LIB=PySide6 before starting this process")
    app = mkQApp()
    win = open_gui(analyzer, table, slay, curation_dir, recording=recording,
                   title=f"SpikeInterface GUI - {processed_dir.name}")
    print(f"window open after {time.perf_counter() - t0:.0f} s", flush=True)
    app.exec()
    return win


# ── helpers ───────────────────────────────────────────────────────────────
def _py(unit_id):
    """A unit id as a plain int or str, which JSON and pydantic both take."""
    return unit_id.item() if isinstance(unit_id, np.generic) else unit_id


def _require_overlap(unit_index, other_index, what):
    if len(unit_index.intersection(other_index)) == 0:
        raise ValueError(
            f"none of {what} match the analyzer's unit ids "
            f"({list(other_index[:3])}... vs {list(unit_index[:3])}...)"
        )


def _tail(path, n_lines=40):
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(lines[-n_lines:])


if __name__ == "__main__":
    main()
