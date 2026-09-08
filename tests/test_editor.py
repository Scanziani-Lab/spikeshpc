"""The interactive state editor: live thresholds, manual painting, saving."""

import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from spikeshpc.editor import StateEditor, show_state_editor  # noqa: E402
from spikeshpc.io import load_states  # noqa: E402
from spikeshpc.states import STATE_CODES  # noqa: E402

from test_state_use import write_scoring  # noqa: E402


class Key:
    def __init__(self, key):
        self.key = key


class Mouse:
    def __init__(self, inaxes=None, xdata=None, ydata=None):
        self.inaxes, self.xdata, self.ydata = inaxes, xdata, ydata


@pytest.fixture
def editor(tmp_path):
    codes = np.tile([1] * 30 + [3] * 30, 4).astype(np.int16)
    write_scoring(tmp_path / "states", codes=codes)
    s = load_states(tmp_path / "states")
    e = StateEditor(s, window_s=120.0)
    yield e
    matplotlib.pyplot.close(e.fig)


def drag_threshold(editor, index, to):
    ax = editor.axes[index]
    line = editor.threshold_lines[index]
    editor._on_press(Mouse(ax, 0.0, line.get_ydata()[0]))
    editor._on_motion(Mouse(ax, 0.0, to))
    editor._on_release(Mouse(ax, 0.0, to))


def select(editor, lo, hi):
    """Drag out a time span, well clear of the threshold line's grab radius."""
    ax = editor.axes[0]
    y0, y1 = ax.get_ylim()
    y = y0 + 0.01 * (y1 - y0)
    editor._on_press(Mouse(ax, lo, y))
    assert editor._select is not None, "press was taken as a threshold drag"
    editor._on_motion(Mouse(ax, hi, y))
    editor._on_release(Mouse(ax, hi, y))


# ── layout ──────────────────────────────────────────────────────────────
def test_editor_shows_a_panel_and_histogram_per_signal(editor):
    assert len(editor.axes) == 4
    assert len(editor.hists) == 4
    # broadband/theta/emg carry thresholds; movement does not
    assert [t is not None for t in editor.threshold_lines] == [True, True, True, False]


def test_overview_and_hypnogram_exist(editor):
    assert editor.ax_overview is not None and editor.ax_hypno is not None
    assert editor.ax_overview.get_xlim() == pytest.approx(
        (editor.times[0], editor.times[-1])
    )


# ── live thresholds ─────────────────────────────────────────────────────
def test_dragging_a_threshold_reclassifies(editor):
    before = editor.codes.copy()
    # push the broadband threshold far above the data: nothing is NREM now
    drag_threshold(editor, 0, 1e3)
    assert editor.thresholds["broadband"] == pytest.approx(1e3)
    assert not (editor.codes == STATE_CODES["NREM"]).any()
    assert not np.array_equal(editor.codes, before)


def test_dragging_the_other_way_makes_everything_nrem(editor):
    drag_threshold(editor, 0, -1e3)
    assert (editor.codes == STATE_CODES["NREM"]).all()


def test_a_threshold_drag_updates_the_histogram_marker_too(editor):
    drag_threshold(editor, 1, 0.42)
    assert editor.hist_lines[1].get_ydata()[0] == pytest.approx(0.42)


def test_clicking_a_trace_does_not_move_a_distant_threshold(editor):
    before = dict(editor.thresholds)
    ax = editor.axes[0]
    lo, hi = ax.get_ylim()
    editor._on_press(Mouse(ax, 5.0, lo))     # nowhere near the line
    editor._on_release(Mouse(ax, 5.0, lo))
    assert editor.thresholds == before


# ── manual painting ─────────────────────────────────────────────────────
def test_painting_a_span_overrides_the_automatic_call(editor):
    select(editor, 10.0, 30.0)
    editor._on_key(Key("5"))                 # REM

    span = (editor.times >= 10.0) & (editor.times <= 30.0)
    assert (editor.codes[span] == STATE_CODES["REM"]).all()
    assert (editor.manual[span] == STATE_CODES["REM"]).all()
    assert (editor.manual[~span] == -1).all()


def test_manual_edits_survive_a_later_threshold_change(editor):
    """The reason manual labels are kept separate from the automatic ones."""
    select(editor, 10.0, 30.0)
    editor._on_key(Key("r"))                 # REM
    span = (editor.times >= 10.0) & (editor.times <= 30.0)

    drag_threshold(editor, 0, -1e3)          # everything else becomes NREM
    assert (editor.codes[span] == STATE_CODES["REM"]).all()
    assert (editor.codes[~span] == STATE_CODES["NREM"]).all()


def test_painting_can_be_handed_back_to_the_rules(editor):
    select(editor, 10.0, 30.0)
    editor._on_key(Key("5"))
    span = (editor.times >= 10.0) & (editor.times <= 30.0)
    assert (editor.manual[span] >= 0).all()

    select(editor, 10.0, 30.0)
    editor._on_key(Key("0"))
    assert (editor.manual == -1).all()


def test_painting_without_a_selection_does_nothing(editor):
    before = editor.codes.copy()
    editor._on_key(Key("3"))
    np.testing.assert_array_equal(editor.codes, before)


@pytest.mark.parametrize("key,state", [("1", "WAKE"), ("w", "WAKE"),
                                       ("3", "NREM"), ("n", "NREM"),
                                       ("5", "REM"), ("r", "REM")])
def test_every_paint_key(editor, key, state):
    select(editor, 10.0, 30.0)
    editor._on_key(Key(key))
    span = (editor.times >= 10.0) & (editor.times <= 30.0)
    assert (editor.codes[span] == STATE_CODES[state]).all()


# ── navigation ──────────────────────────────────────────────────────────
def test_pan_and_zoom(editor):
    start = editor.t0
    editor._on_key(Key("d"))
    assert editor.t0 > start
    editor._on_key(Key("a"))
    assert editor.t0 == pytest.approx(start)

    wide = editor.window_s
    editor._on_key(Key("z"))
    assert editor.window_s > wide
    editor._on_key(Key("Z"))
    assert editor.window_s == pytest.approx(wide)


def test_panning_stops_at_the_ends(editor):
    for _ in range(8):
        editor._on_key(Key("a"))
    assert editor.t0 >= editor.times[0] - 1e-9
    for _ in range(16):
        editor._on_key(Key("d"))
    assert editor.t0 <= editor.times[-1]


def test_clicking_the_overview_jumps(editor):
    target = 150.0
    editor._on_press(Mouse(editor.ax_overview, target, 0.5))
    assert editor.t0 == pytest.approx(target - editor.window_s / 2, abs=1.0)


# ── undo / reset ────────────────────────────────────────────────────────
def test_undo_restores_thresholds_and_edits(editor):
    original = dict(editor.thresholds)
    select(editor, 10.0, 30.0)
    editor._on_key(Key("5"))
    drag_threshold(editor, 0, 1e3)

    editor._on_key(Key("u"))                 # undo the drag
    assert editor.thresholds["broadband"] == pytest.approx(original["broadband"])
    span = (editor.times >= 10.0) & (editor.times <= 30.0)
    assert (editor.manual[span] >= 0).all()

    editor._on_key(Key("u"))                 # undo the paint
    assert (editor.manual == -1).all()


def test_reset_returns_to_the_pipeline_output(editor):
    original_codes = np.asarray(editor.scoring.codes).copy()
    select(editor, 10.0, 30.0)
    editor._on_key(Key("5"))
    drag_threshold(editor, 0, 1e3)

    editor._on_key(Key("R"))
    assert (editor.manual == -1).all()
    assert editor.thresholds == editor.scoring.thresholds
    np.testing.assert_array_equal(editor.codes, original_codes)


# ── saving ──────────────────────────────────────────────────────────────
def test_save_writes_edits_and_keeps_the_original(editor, tmp_path, capsys):
    select(editor, 10.0, 30.0)
    editor._on_key(Key("5"))
    drag_threshold(editor, 1, 0.05)

    path = editor.save()
    out = capsys.readouterr().out
    assert "kept the automatic scoring" in out

    states_dir = tmp_path / "states"
    assert (states_dir / "s1_states.orig.json").exists()
    assert (states_dir / "s1_metrics.orig.npz").exists()

    summary = json.loads(path.read_text())
    assert summary["edited"] is True
    assert summary["n_manual_bins"] > 0
    assert summary["thresholds"]["theta"] == pytest.approx(0.05)

    reloaded = load_states(states_dir)
    np.testing.assert_array_equal(reloaded.codes, editor.codes)
    assert reloaded.thresholds["theta"] == pytest.approx(0.05)


def test_save_does_not_clobber_an_existing_backup(editor, tmp_path):
    states_dir = tmp_path / "states"
    editor.save()
    first = (states_dir / "s1_states.orig.json").read_text()

    select(editor, 10.0, 30.0)
    editor._on_key(Key("5"))
    editor.save()
    # the backup is still the pipeline's own output, not the first edit
    assert (states_dir / "s1_states.orig.json").read_text() == first


def test_save_keeps_the_movement_trace(editor, tmp_path):
    editor.save()
    reloaded = load_states(tmp_path / "states")
    assert reloaded.speed is not None
    assert "movement (mm/s)" in reloaded.signals()


def test_save_without_a_source_path_is_refused(tmp_path):
    write_scoring(tmp_path / "states")
    s = load_states(tmp_path / "states")
    s.source = None
    e = StateEditor(s)
    with pytest.raises(ValueError, match="no source path"):
        e.save()
    matplotlib.pyplot.close(e.fig)


def test_show_state_editor_returns_the_editor(tmp_path):
    write_scoring(tmp_path / "states")
    e = show_state_editor(load_states(tmp_path / "states"))
    assert isinstance(e, StateEditor)
    matplotlib.pyplot.close(e.fig)


def test_missing_signals_is_reported(tmp_path):
    write_scoring(tmp_path / "states")
    s = load_states(tmp_path / "states")
    s.broadband = s.theta = s.emg = s.speed = None
    with pytest.raises(ValueError, match="no per-bin signals saved"):
        StateEditor(s)


# ── the selection must survive the button coming up ─────────────────────
def test_selection_persists_after_release(editor):
    """You label a span after letting go, not while still holding the button."""
    select(editor, 10.0, 30.0)
    assert editor._select is not None
    assert sorted(editor._select) == [10.0, 30.0]
    assert editor._selecting is False


def test_moving_the_mouse_after_release_does_not_move_the_selection(editor):
    """The bug: the span kept following the cursor once the button was up."""
    select(editor, 10.0, 30.0)
    frozen = sorted(editor._select)

    ax = editor.axes[0]
    y0, y1 = ax.get_ylim()
    y = y0 + 0.01 * (y1 - y0)
    for x in (60.0, 90.0, 5.0):
        editor._on_motion(Mouse(ax, x, y))
    assert sorted(editor._select) == frozen


def test_a_span_can_still_be_labelled_long_after_release(editor):
    select(editor, 10.0, 30.0)
    ax = editor.axes[0]
    editor._on_motion(Mouse(ax, 200.0, 0.0))     # wander off
    editor._on_key(Key("r"))

    span = (editor.times >= 10.0) & (editor.times <= 30.0)
    assert (editor.codes[span] == STATE_CODES["REM"]).all()


def test_a_click_without_a_drag_clears_the_selection(editor):
    select(editor, 10.0, 30.0)
    assert editor._select is not None

    ax = editor.axes[0]
    y0, y1 = ax.get_ylim()
    y = y0 + 0.01 * (y1 - y0)
    editor._on_press(Mouse(ax, 50.0, y))
    editor._on_release(Mouse(ax, 50.0, y))
    assert editor._select is None


def test_starting_a_new_drag_replaces_the_old_selection(editor):
    select(editor, 10.0, 30.0)
    select(editor, 100.0, 140.0)
    assert sorted(editor._select) == [100.0, 140.0]


def test_dragging_a_threshold_does_not_disturb_the_selection(editor):
    select(editor, 10.0, 30.0)
    drag_threshold(editor, 1, 0.3)
    assert sorted(editor._select) == [10.0, 30.0]
    assert editor._selecting is False
