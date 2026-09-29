"""Scrolling through a trace's turns: navigation, what is drawn, and lines that do not lie."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from spikeshpc.raster import UnitRaster  # noqa: E402
from spikeshpc.turns import find_turns  # noqa: E402
from spikeshpc.turns_widget import (  # noqa: E402
    JUMP_COLOR,
    SPLICE_COLOR,
    TurnWidget,
    show_turns,
)

BIN = 0.1


class Key:
    def __init__(self, key):
        self.key = key


def heading_from(*pieces, start_deg=10.0):
    """Heading per 100 ms bin from (seconds, deg/s) pieces, wrapped to [0, 360)."""
    rates = np.concatenate([np.full(int(round(s / BIN)), float(rate)) for s, rate in pieces])
    return (start_deg + np.concatenate([[0.0], np.cumsum(rates[:-1] * BIN)])) % 360.0


@pytest.fixture(scope="module")
def trace():
    """Two decoded stretches 100 s apart: a CW and a CCW turn, then a CW turn and a jump."""
    first = heading_from((20, 0), (1, 90), (20, 0), (1, -90), (20, 0))  # 62 s
    second = heading_from((20, 0), (1, 90), (20, 0), start_deg=200.0)  # 41 s
    second[300:] = (second[300:] + 150.0) % 360.0  # hops 150 degrees 30 s in
    heading = np.concatenate([first, second])
    time = np.concatenate([np.arange(first.size) * BIN, 162.0 + np.arange(second.size) * BIN])
    run_index = np.repeat([0, 1], [first.size, second.size])
    turns = find_turns(time, heading, run_index, turn_threshold=30.0, bin_s=BIN)
    assert (turns.n_turns, turns.n_clockwise, turns.n_jumps) == (3, 2, 1)
    return time, heading, run_index, turns


@pytest.fixture
def widget(trace):
    time, heading, run_index, turns = trace
    w = TurnWidget(time, heading, turns, run_index=run_index, measured_deg=heading,
                   window_s=60.0)
    yield w
    plt.close(w.fig)


def drawn(widget, kind, ax=None):
    """How many spans or lines of one kind the view holds."""
    return sum(len(a.get_paths()) if isinstance(a, PolyCollection) else len(a.get_segments())
               for a in widget._drawn
               if isinstance(a, kind) and (ax is None or a.axes is ax))


# ── the axis ────────────────────────────────────────────────────────────
def test_it_opens_on_the_first_decoded_bin(widget):
    assert widget.t0 == 0.0 and widget.window_s == 60.0
    assert widget.heading_ax.get_xlim() == pytest.approx((0.0, 60.0))
    assert widget.velocity_ax.get_xlim() == pytest.approx((0.0, 60.0))


def test_the_axis_is_decoded_time_not_recording_time(widget, trace):
    time = trace[0]
    assert widget.total_s == pytest.approx(len(time) * BIN)
    assert widget.total_s < time[-1] - time[0]
    assert widget.break_s == pytest.approx([62.0])  # where the second stretch starts


# ── navigation ──────────────────────────────────────────────────────────
def test_arrows_pan_by_half_a_window(widget):
    widget._on_key(Key("right"))
    assert widget.t0 == pytest.approx(30.0)
    widget._on_key(Key("left"))
    assert widget.t0 == pytest.approx(0.0)


def test_panning_stops_at_the_ends(widget):
    for _ in range(4):
        widget._on_key(Key("left"))
    assert widget.t0 == 0.0
    for _ in range(6):
        widget._on_key(Key("right"))
    assert widget.t0 == pytest.approx(widget.total_s - widget.window_s)


def test_up_and_down_change_how_much_is_shown(widget):
    widget._on_key(Key("up"))
    assert widget.window_s == pytest.approx(90.0)
    widget._on_key(Key("down"))
    assert widget.window_s == pytest.approx(60.0)
    for _ in range(10):
        widget._on_key(Key("up"))
    assert widget.window_s == pytest.approx(widget.total_s)  # shorter than 15 minutes


def test_home_and_end_jump(widget):
    widget._on_key(Key("end"))
    assert widget.t0 == pytest.approx(widget.total_s - widget.window_s)
    widget._on_key(Key("home"))
    assert widget.t0 == 0.0


def test_an_unrelated_key_does_nothing(widget):
    before = (widget.t0, widget.window_s)
    widget._on_key(Key("q"))
    assert (widget.t0, widget.window_s) == before


def test_the_slider_and_the_arrows_move_together(widget):
    widget.slider.set_val(20.0)
    assert widget.t0 == pytest.approx(20.0)
    assert widget.heading_ax.get_xlim()[0] == pytest.approx(20.0)
    widget._on_key(Key("right"))
    assert widget.slider.val == pytest.approx(widget.t0)


# ── what is drawn ───────────────────────────────────────────────────────
def test_the_turns_in_view_are_shaded_on_both_axes(widget):
    # the first minute holds the first stretch's CW and CCW turns
    assert drawn(widget, PolyCollection, widget.heading_ax) == 2
    assert drawn(widget, PolyCollection, widget.velocity_ax) == 2
    widget._on_key(Key("end"))  # 43-103 s: only the second stretch's CW turn
    assert drawn(widget, PolyCollection, widget.heading_ax) == 1


def test_jumps_and_splices_are_marked_where_they_are(widget):
    assert drawn(widget, LineCollection) == 0  # neither in the first minute
    widget._on_key(Key("end"))  # the splice at 62 s and the jump at 92 s
    jumps = [a for a in widget._drawn if isinstance(a, LineCollection)
             and a.axes is widget.heading_ax and a.get_linewidth()[0] == pytest.approx(0.8)]
    assert sum(len(a.get_segments()) for a in jumps) == 1
    splices = [a for a in widget._drawn if isinstance(a, LineCollection)
               and a.get_linewidth()[0] == pytest.approx(1.0)]
    assert len(splices) == 2  # one per axis
    assert JUMP_COLOR != SPLICE_COLOR


def test_what_is_drawn_is_replaced_not_stacked(widget):
    widget._on_key(Key("end"))
    at_end = len(widget._drawn)
    for key in ("home", "right", "end", "left", "end"):
        widget._on_key(Key(key))
    assert len(widget._drawn) == at_end


def test_no_line_joins_across_a_splice_or_a_jump(widget):
    widget.window_s = widget.total_s
    widget.t0 = 0.0
    widget._draw()
    velocity = widget.velocity_line.get_ydata()
    assert np.isnan(velocity).sum() >= 2  # one cut at the splice, one at the jump
    assert widget.decoded_dots.get_linestyle() == "None"  # dots: they never join


def test_the_title_gives_the_window_and_the_whole_trace(widget):
    title = widget.heading_ax.get_title()
    assert "here:" in title and "whole trace:" in title
    assert "2 turns (1 CW / 1 CCW)" in title
    assert "recording" in title and "decoded" in title


# ── options and checks ──────────────────────────────────────────────────
def test_it_works_without_a_measured_heading(trace):
    time, heading, run_index, turns = trace
    w = show_turns(time, heading, turns, run_index=run_index, window_s=30.0)
    try:
        assert isinstance(w, TurnWidget) and w.window_s == 30.0
        labels = [t.get_text() for t in w.heading_ax.get_legend().get_texts()]
        assert "measured" not in labels and "clockwise turn" in labels
    finally:
        plt.close(w.fig)


def test_the_stretches_come_from_time_gaps_without_a_run_index(trace):
    time, heading, _, turns = trace
    w = TurnWidget(time, heading, turns)
    try:
        assert w.break_s == pytest.approx([62.0])
    finally:
        plt.close(w.fig)


def test_without_a_raster_there_are_two_panels(widget):
    assert widget.raster_ax is None and widget.raster_panel is None
    assert widget.axes == [widget.heading_ax, widget.velocity_ax]


# ── the raster ──────────────────────────────────────────────────────────
@pytest.fixture
def raster():
    """Three units: spikes in both decoded stretches, and one in the gap between them."""
    return UnitRaster(
        spike_times=[[5.02, 20.51, 100.0], [30.0, 170.0], [61.8]],
        preferred_deg=[240.0, 0.0, 120.0],
        labels=["a", "b", "c"],
    )


@pytest.fixture
def raster_widget(trace, raster):
    time, heading, run_index, turns = trace
    w = show_turns(time, heading, turns, run_index=run_index, window_s=60.0, raster=raster)
    yield w
    plt.close(w.fig)


def test_the_raster_goes_on_top_with_the_title(raster_widget):
    w = raster_widget
    assert w.axes == [w.raster_ax, w.heading_ax, w.velocity_ax]
    assert w.raster_ax.get_position().y0 > w.heading_ax.get_position().y1
    assert "whole trace:" in w.raster_ax.get_title() and w.heading_ax.get_title() == ""


def test_the_raster_draws_the_window_on_the_decoded_axis(raster_widget):
    w = raster_widget
    assert w.raster_panel.n_drawn == 3  # 5.02, 20.51 and 30.0 s; 61.8 s is at 61.85 on the axis
    w._on_key(Key("end"))  # 43-103 s: 61.8 s, and 170 s from the second stretch
    assert w.raster_panel.n_drawn == 2  # 100 s fell between the stretches: never drawn


def test_splices_cross_the_raster_too(raster_widget):
    raster_widget._on_key(Key("end"))  # the splice at 62 s
    splices = [a for a in raster_widget._drawn if isinstance(a, LineCollection)
               and a.get_linewidth()[0] == pytest.approx(1.0)]
    assert {a.axes for a in splices} == set(raster_widget.axes)


# ── the offset ──────────────────────────────────────────────────────────
def test_apply_offset_moves_the_decoded_heading_and_nothing_else(trace):
    time, heading, run_index, turns = trace
    plain = TurnWidget(time, heading, turns, run_index=run_index)
    moved = TurnWidget(time, heading, turns, run_index=run_index, apply_offset=True,
                       offset_deg=-44.0)
    try:
        expected = (plain.decoded_dots.get_ydata() + 44.0) % 360.0
        np.testing.assert_allclose(moved.decoded_dots.get_ydata(), expected)
        np.testing.assert_allclose(moved.velocity_line.get_ydata(), plain.velocity_line.get_ydata())
        assert moved.heading_ax.get_title() == plain.heading_ax.get_title()  # same turns, drift
        labels = [t.get_text() for t in moved.heading_ax.get_legend().get_texts()]
        assert "decoded, -44 deg offset removed" in labels
    finally:
        plt.close(plain.fig)
        plt.close(moved.fig)


def test_an_offset_is_only_applied_when_asked(trace):
    time, heading, run_index, turns = trace
    w = TurnWidget(time, heading, turns, offset_deg=-44.0)  # apply_offset left False
    try:
        np.testing.assert_allclose(w.shown_deg, heading)
    finally:
        plt.close(w.fig)


def test_apply_offset_without_an_offset_is_refused(trace):
    time, heading, _, turns = trace
    with pytest.raises(ValueError, match="apply_offset needs offset_deg"):
        TurnWidget(time, heading, turns, apply_offset=True)


def test_a_trace_that_does_not_match_its_turns_is_refused(trace):
    time, heading, run_index, turns = trace
    with pytest.raises(ValueError, match="one value per bin"):
        TurnWidget(time[:-1], heading[:-1], turns)
    with pytest.raises(ValueError, match="measured_deg"):
        TurnWidget(time, heading, turns, measured_deg=heading[:5])
    with pytest.raises(ValueError, match="holds 1 bins"):
        TurnWidget(time[:1], heading[:1], turns)
