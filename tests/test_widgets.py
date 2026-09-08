"""The state-epoch review widget."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from spikeshpc.io import load_states  # noqa: E402
from spikeshpc.widgets import StateEpochWidget, show_state_epochs  # noqa: E402

from test_state_use import write_scoring  # noqa: E402


class Key:
    def __init__(self, key):
        self.key = key


@pytest.fixture
def scoring(tmp_path):
    write_scoring(tmp_path / "states")
    return load_states(tmp_path / "states")


def test_widget_draws_one_axis_per_signal(scoring, recwarn):
    w = StateEpochWidget(scoring)
    assert len(w.axes) == len(scoring.signals()) == 4
    assert len(w.epochs) == 4
    assert w.index == 0
    matplotlib.pyplot.close(w.fig)


def test_arrow_keys_step_and_wrap(scoring):
    w = StateEpochWidget(scoring)
    labels = []
    for _ in range(len(w.epochs) + 1):
        labels.append(w.label.get_text())
        w._on_key(Key("right"))
    # wrapped back to the first epoch
    assert labels == ["WAKE", "NREM", "REM", "WAKE", "WAKE"]
    assert w.index == 1

    w.index = 0
    w._draw()
    w._on_key(Key("left"))
    assert w.index == len(w.epochs) - 1
    matplotlib.pyplot.close(w.fig)


def test_other_keys_do_nothing(scoring):
    w = StateEpochWidget(scoring)
    w._on_key(Key("up"))
    w._on_key(Key("q"))
    assert w.index == 0
    matplotlib.pyplot.close(w.fig)


def test_the_state_is_written_on_the_figure(scoring):
    w = StateEpochWidget(scoring)
    assert w.label.get_text() == "WAKE"
    assert "epoch 1/4" in w.title.get_text()
    assert "WAKE" in w.title.get_text()
    w._on_key(Key("right"))
    assert w.label.get_text() == "NREM"
    matplotlib.pyplot.close(w.fig)


def test_epoch_is_shaded_and_context_shown(scoring):
    w = StateEpochWidget(scoring, context_s=5.0)
    lo, hi = w.axes[0].get_xlim()
    epoch = w.epochs[0]
    assert lo == pytest.approx(epoch.start - 5.0)
    assert hi == pytest.approx(epoch.stop + 5.0)
    assert all(s is not None for s in w.spans)
    matplotlib.pyplot.close(w.fig)


def test_can_review_a_single_state(scoring):
    w = show_state_epochs(scoring, states="REM")
    assert len(w.epochs) == 1 and w.epochs[0].state == "REM"
    matplotlib.pyplot.close(w.fig)


def test_brief_epochs_can_be_skipped(scoring):
    w = show_state_epochs(scoring, min_duration_s=8.0)
    assert [e.duration for e in w.epochs] == [10.0, 10.0]
    matplotlib.pyplot.close(w.fig)


def test_no_matching_epochs_is_an_error(scoring):
    with pytest.raises(ValueError, match="No epochs to show"):
        show_state_epochs(scoring, min_duration_s=1e6)


def test_missing_signals_is_reported(tmp_path):
    write_scoring(tmp_path / "states", session="bare")
    s = load_states(tmp_path / "states", session="bare")
    s.broadband = s.theta = s.emg = s.speed = None
    with pytest.raises(ValueError, match="no per-bin signals saved"):
        StateEpochWidget(s)


def test_thresholds_are_drawn_where_they_exist(scoring):
    w = StateEpochWidget(scoring)
    # broadband/theta/emg have thresholds; movement does not
    assert w.thresholds[:3] == [0.1, 0.2, 0.8]
    assert w.thresholds[3] is None
    matplotlib.pyplot.close(w.fig)


def test_warns_on_a_non_interactive_backend(scoring):
    """Agg silently drops key events, which looks like a frozen widget."""
    with pytest.warns(UserWarning, match="does not deliver key-press events"):
        w = StateEpochWidget(scoring)
    matplotlib.pyplot.close(w.fig)


def test_vetoed_mode_shows_the_overruled_calls(tmp_path):
    """These spans are WAKE in the final scoring; this is the only way to see them."""
    from test_state_use import _write_with_veto

    _write_with_veto(tmp_path / "states")
    s = load_states(tmp_path / "states")

    w = show_state_epochs(s, vetoed=True)
    assert len(w.epochs) == 1
    assert w.epochs[0].state == "REM"
    assert "REM" in w.label.get_text() and "WAKE" in w.label.get_text()
    assert "vetoed 1/1" in w.title.get_text()
    matplotlib.pyplot.close(w.fig)


def test_vetoed_mode_without_recorded_codes_says_so(scoring):
    with pytest.raises(ValueError, match="no codes_before_veto saved"):
        show_state_epochs(scoring, vetoed=True)


# ── fixed y-axes ────────────────────────────────────────────────────────
def test_y_limits_are_the_same_on_every_epoch(scoring):
    """The point: a trace at the same height means the same value everywhere."""
    w = StateEpochWidget(scoring)
    first = [tuple(ax.get_ylim()) for ax in w.axes]

    for _ in range(len(w.epochs)):
        w._on_key(Key("right"))
        assert [tuple(ax.get_ylim()) for ax in w.axes] == first
    matplotlib.pyplot.close(w.fig)


def test_limits_come_from_the_whole_session_not_the_visible_window(scoring):
    w = StateEpochWidget(scoring, context_s=0.0)
    for label, (lo, hi) in w.ylim.items():
        values = np.asarray(scoring.signals()[label], dtype=float)
        # the robust range covers the bulk of the session, not just one epoch
        inside = np.mean((values >= lo) & (values <= hi))
        assert inside > 0.9, (label, inside)
    matplotlib.pyplot.close(w.fig)


def test_an_outlier_does_not_stretch_the_axis(tmp_path):
    """min/max would let one artefact squash every real trace flat.

    Needs a realistic number of bins: the 99.5th percentile of thirty values
    is the maximum, so percentile robustness only bites on a real session.
    """
    codes = np.tile([1] * 100 + [3] * 100, 10).astype(np.int16)
    write_scoring(tmp_path / "states", codes=codes)
    s = load_states(tmp_path / "states")
    s.emg = s.emg.copy()
    s.emg[0] = 1e6

    w = StateEpochWidget(s)
    lo, hi = w.ylim["EMG (LFP correlation)"]
    assert hi < 100, hi
    matplotlib.pyplot.close(w.fig)


def test_the_threshold_stays_inside_the_axis(scoring):
    """It is the reference the state was decided against; hiding it is useless."""
    scoring.thresholds = dict(scoring.thresholds, emg=5.0)   # far above the data
    w = StateEpochWidget(scoring)
    lo, hi = w.ylim["EMG (LFP correlation)"]
    assert lo <= 5.0 <= hi
    matplotlib.pyplot.close(w.fig)


def test_limits_can_be_overridden_per_signal(scoring):
    w = StateEpochWidget(scoring, ylim={"movement (mm/s)": (0, 200)})
    assert w.ylim["movement (mm/s)"] == (0.0, 200.0)
    assert tuple(w.axes[3].get_ylim()) == (0.0, 200.0)
    matplotlib.pyplot.close(w.fig)


def test_padding_never_opens_a_negative_region_on_a_positive_signal(scoring):
    """Movement cannot be negative; on a log axis that would waste the panel."""
    w = StateEpochWidget(scoring)
    assert w.ylim["movement (mm/s)"][0] >= 0.0
    matplotlib.pyplot.close(w.fig)


def test_a_signal_can_be_put_on_a_log_axis(scoring):
    w = StateEpochWidget(scoring, log_signals=("movement (mm/s)",))
    assert w.axes[3].get_yscale() == "symlog"
    assert w.axes[0].get_yscale() == "linear"
    matplotlib.pyplot.close(w.fig)
