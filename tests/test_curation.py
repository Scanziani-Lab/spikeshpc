"""Handing bombcell, UnitRefine and SLAy to spikeinterface-gui, and reading it back.

Every failure guarded here is silent:

  * bombcell's rows matched to units by position rather than by phy_clusterID
    put each label on whichever unit happens to share the row;
  * a text column of object dtype -- what pandas gives text -- is dropped from
    the GUI's unit table with no more than a warning;
  * a cache from the previous sorting reloads onto a re-sort without
    complaint, because both number their units 0..N-1;
  * quality options that differ from the GUI's default switch off its
    g/m/n/c shortcuts;
  * the Merge tab is filled through two attributes of spikeinterface-gui's
    MergeView, which a new version may rename (the Qt test below).
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import spikeinterface.full as si

from spikeshpc.curation import (
    AUTOMATED_FILE,
    BOMBCELL_MANUAL_FILE,
    CURATED_FILE,
    CURATION_FILE,
    QUALITY,
    SLAY_METRICS,
    SORTING_FILE,
    automated_labels,
    check_curation_dir,
    detached_extensions,
    initial_curation,
    load_curation_result,
    load_slay,
    read_automated_labels,
    save_curation,
    save_slay,
    sigui_properties,
    slay_pairs,
)

N_UNITS = 6
BC_TYPES = ["GOOD", "MUA", "NOISE", "GOOD", "NON-SOMA", ""]
UR_TYPES = ["sua", "mua", "noise", "mua", "sua", "noise"]
AGREED = ["good", "MUA", "noise", "", "", ""]  # what the two above agree on


def make_analyzer(seed=0, **templates_params):
    recording, sorting = si.generate_ground_truth_recording(
        durations=[10.0], num_units=N_UNITS, num_channels=16, seed=seed
    )
    sorting = sorting.rename_units(np.arange(N_UNITS))  # integer ids, as kilosort's are
    analyzer = si.create_sorting_analyzer(sorting, recording, format="memory", sparse=False)
    analyzer.compute(["random_spikes", "noise_levels"])
    analyzer.compute("templates", **templates_params)
    analyzer.compute("unit_locations")
    return analyzer


@pytest.fixture(scope="module")
def analyzer():
    return make_analyzer()


def unitrefine(labels, probabilities=None):
    """``si.unitrefine_label_units``'s output: indexed by unit id."""
    probabilities = probabilities or [0.9] * len(labels)
    return pd.DataFrame(
        {"unitrefine_label": labels, "unitrefine_probability": probabilities},
        index=pd.Index(range(len(labels))),
    )


def bombcell(types, ids=None):
    """bombcell's quality metrics and type strings, row-aligned with each other."""
    ids = np.arange(len(types)) if ids is None else np.asarray(ids)
    return {"phy_clusterID": ids}, np.array(types, dtype=object)


@pytest.fixture
def slay():
    """Two groups: 0/2/5 (three units, so three pairs) and 1/4."""
    rng = np.random.default_rng(0)
    metrics = {}
    for k, name in enumerate(SLAY_METRICS):
        m = rng.uniform(0, 0.3, (N_UNITS, N_UNITS))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 0)
        metrics[name] = m
    final = metrics["final_metric"]
    for (a, b), value in {(0, 2): 0.9, (2, 5): 0.8, (0, 5): 0.7, (1, 4): 0.6}.items():
        final[a, b] = final[b, a] = value
    return {"merges": [[0, 2, 5], [1, 4]], **metrics}


def table_for(slay=None, bc_manual_file=None):
    qm, types = bombcell(BC_TYPES)
    return automated_labels(range(N_UNITS), qm, types, unitrefine(UR_TYPES), slay=slay,
                            bc_manual_file=bc_manual_file)


# ── staleness ─────────────────────────────────────────────────────────────
def test_a_folder_is_tied_to_the_first_sorting_it_sees(tmp_path, analyzer):
    check_curation_dir(tmp_path, analyzer)
    assert (tmp_path / SORTING_FILE).is_file()
    check_curation_dir(tmp_path, analyzer)  # the same analyzer again: fine


def test_a_resort_numbering_its_units_the_same_way_is_refused(tmp_path, analyzer):
    other = make_analyzer(seed=1)
    assert list(other.unit_ids) == list(analyzer.unit_ids), "the test needs colliding unit ids"
    check_curation_dir(tmp_path, analyzer)
    with pytest.raises(RuntimeError, match="spike counts differ"):
        check_curation_dir(tmp_path, other)


def test_an_analyzer_rebuilt_around_the_same_sorting_is_refused(tmp_path, analyzer):
    rebuilt = make_analyzer(seed=0, ms_before=0.5)  # same spikes, different waveforms
    check_curation_dir(tmp_path, analyzer)
    with pytest.raises(RuntimeError, match="templates differ"):
        check_curation_dir(tmp_path, rebuilt)


# ── the per-unit table ────────────────────────────────────────────────────
def test_bombcell_rows_are_matched_to_units_by_id_not_position():
    rows = [3, 0, 5, 1, 4, 2]  # bombcell's own row order
    qm, types = bombcell([BC_TYPES[u] for u in rows], ids=rows)
    table = automated_labels(range(N_UNITS), qm, types, unitrefine(UR_TYPES))
    assert table["bc_label"].tolist() == BC_TYPES


def test_a_label_is_prefilled_only_where_bombcell_and_unitrefine_agree():
    table = table_for()
    assert table["auto_label"].tolist() == AGREED  # disagreement, NON-SOMA, no call: blank


def test_your_call_in_bombcells_gui_stands_in_for_bombcells_own(tmp_path):
    manual = tmp_path / BOMBCELL_MANUAL_FILE
    pd.DataFrame({"unit_id": [0, 3, 1], "manual_classification": [3, 2, -1]}).to_csv(manual, index=False)
    table = table_for(bc_manual_file=manual)
    assert table["bc_manual"].tolist() == ["non-somatic", "", "", "MUA", "", ""]
    assert table.loc[0, "auto_label"] == ""  # you overruled GOOD: no longer agreement
    assert table.loc[3, "auto_label"] == "MUA"  # you settled the disagreement UnitRefine's way
    assert table.loc[1, "auto_label"] == "MUA"  # unclassified: bombcell's own call counts


def test_units_in_a_slay_group_know_their_partners_and_best_score(slay):
    table = table_for(slay)
    assert table["slay_group"].tolist() == [0, 1, 0, -1, 1, 0]
    assert table.loc[0, "slay_partners"] == "2,5"
    assert table.loc[5, "slay_score"] == pytest.approx(0.8)  # its best partner is 2
    assert np.isnan(table.loc[3, "slay_score"])


def test_a_slay_group_naming_a_unit_the_analyzer_lacks_is_refused(slay):
    slay["merges"].append([2, 99])
    with pytest.raises(ValueError, match="99"):
        table_for(slay)


def test_every_column_reaches_the_gui_after_a_round_trip_through_csv(tmp_path, analyzer, slay):
    from spikeinterface.widgets.utils import make_units_table_from_analyzer

    slay["merges"] = [[0, 2]]  # single partners: "2" must stay text, not become 2.0
    table = table_for(slay)
    table.to_csv(tmp_path / AUTOMATED_FILE)
    back = read_automated_labels(tmp_path)
    pd.testing.assert_frame_equal(back, table)

    properties = sigui_properties(back)
    for name, values in properties.items():
        assert values.ndim == 1 and values.dtype.kind in "iuUSfb", name
    assert properties["auto_label"].tolist() == AGREED  # blanks stay "", not "nan"
    units_table = make_units_table_from_analyzer(analyzer, extra_properties=properties)
    assert set(table.columns) <= set(units_table.columns)


def test_the_starting_curation_is_valid_and_keeps_the_guis_shortcuts():
    from spikeinterface.curation.curation_model import Curation
    from spikeinterface_gui.curation_tools import default_label_definitions

    model = Curation(**initial_curation(table_for()))
    labelled = {label.unit_id: label.labels["quality"] for label in model.manual_labels}
    assert labelled == {0: ["good"], 1: ["MUA"], 2: ["noise"]}
    assert set(QUALITY["quality"]["label_options"]) == set(default_label_definitions["quality"]["label_options"])


# ── SLAy ──────────────────────────────────────────────────────────────────
def test_slay_results_survive_a_round_trip(tmp_path, slay):
    saved = save_slay(tmp_path, slay["merges"], slay)
    loaded = load_slay(tmp_path, N_UNITS)
    assert loaded["merges"] == saved["merges"] == [[0, 2, 5], [1, 4]]
    for name in SLAY_METRICS:
        np.testing.assert_array_equal(loaded[name], slay[name])


def test_a_slay_file_for_another_sorting_is_refused(tmp_path, slay):
    save_slay(tmp_path, slay["merges"], slay)
    with pytest.raises(ValueError, match="another sorting"):
        load_slay(tmp_path, N_UNITS + 1)


def test_a_three_unit_group_becomes_its_three_pairs_best_first(slay):
    assert slay_pairs(slay, np.arange(N_UNITS)) == [[0, 2], [2, 5], [0, 5], [1, 4]]


def test_detached_extensions_come_back_even_when_the_block_fails():
    holder = SimpleNamespace(extensions={"templates": "t", "valid_unit_periods": "v"})
    with pytest.raises(RuntimeError):
        with detached_extensions(holder, "valid_unit_periods", "never_computed"):
            assert "valid_unit_periods" not in holder.extensions
            raise RuntimeError
    assert holder.extensions == {"templates": "t", "valid_unit_periods": "v"}


# ── reading back ──────────────────────────────────────────────────────────
def test_what_the_gui_saves_comes_back_unit_by_unit(tmp_path):
    from spikeinterface.curation.curation_model import Curation

    table = table_for()
    table.to_csv(tmp_path / AUTOMATED_FILE)
    curation = initial_curation(table)
    curation["manual_labels"].append({"unit_id": 3, "labels": {"quality": ["good"]}})
    curation["merges"] = [{"unit_ids": [0, 2]}]
    curation["removed"] = [5]
    # the GUI hands its callback a model dump, not the dict it started from
    save_curation(Curation(**curation).model_dump(), tmp_path / CURATION_FILE)

    _, result = load_curation_result(tmp_path)
    assert result["quality"].tolist() == ["good", "MUA", "noise", "good", "", ""]
    assert result["merge_group"].tolist() == [0, -1, 0, -1, -1, -1]
    assert result["removed"].tolist() == [False] * 5 + [True]
    assert not result["split"].any()
    assert (tmp_path / CURATED_FILE).is_file()


def test_an_invalid_curation_is_refused_before_it_is_written(tmp_path):
    curation = initial_curation(table_for())
    curation["removed"] = [0]
    curation["merges"] = [{"unit_ids": [0, 2]}]  # merged and deleted at once
    with pytest.raises(Exception):
        save_curation(curation, tmp_path / CURATION_FILE)
    assert not (tmp_path / CURATION_FILE).exists()


# ── the GUI itself ────────────────────────────────────────────────────────
def test_the_gui_shows_slays_pairs_every_column_and_the_prefilled_labels(tmp_path, analyzer, slay):
    pytest.importorskip("PySide6")
    if "PyQt6.QtCore" in sys.modules:
        pytest.skip("PyQt6 is loaded in this process, and pyqtgraph would take it")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    from spikeinterface_gui.myqt import mkQApp

    from spikeshpc.curation import open_gui

    app = mkQApp()
    table = table_for(slay)
    win = open_gui(analyzer, table, slay, tmp_path)
    try:
        merge = win.views["merge"]
        assert merge.table.rowCount() == len(slay_pairs(slay, analyzer.unit_ids))
        headers = [merge.table.horizontalHeaderItem(c).text() for c in range(merge.table.columnCount())]
        assert set(SLAY_METRICS) <= set(headers)

        controller = win.controller
        assert {"auto_label", "bc_label", "ur_label", "ur_prob", "slay_group", "slay_score"} <= set(
            controller.units_table.columns
        )
        labelled = {label["unit_id"] for label in controller.curation_data["manual_labels"]}
        assert labelled == {0, 1, 2}
        assert controller.has_default_quality_labels  # the g/m/n/c shortcuts are on
    finally:
        win.close()
        app.processEvents()
