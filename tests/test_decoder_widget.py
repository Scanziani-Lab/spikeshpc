"""Scrolling through a decode: navigation, splices, and lines that do not lie."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spikeshpc.decoder import (  # noqa: E402
    decode,
    fit_encoding_model,
    prepare_decoder_data,
    split_train_test,
    state_interval_mask,
)
from spikeshpc.decoder_widget import (  # noqa: E402
    ACTUAL_COLOR,
    DECODED_COLOR,
    DecodedWidget,
    break_at,
    show_decoded,
)

from test_decoder import (  # noqa: E402
    N_UNITS, RATE, FakeSorting, poisson_spike_times, random_walk_heading,
    von_mises_rates,
)


class Key:
    def __init__(self, key):
        self.key = key


@pytest.fixture(scope="module")
def decoded():
    """A blocked split, so the test set really is spliced together."""
    rng = np.random.default_rng(0)
    n = int(240 * RATE)
    frame_times = np.arange(n) / RATE
    heading = random_walk_heading(n, rng)
    preferred = np.linspace(0, 360, N_UNITS, endpoint=False)
    sorting = FakeSorting(
        poisson_spike_times(von_mises_rates(heading, preferred), frame_times, rng)
    )
    intervals = {"WAKE": [[0.0, float(frame_times[-1])]], "NREM": [], "REM": []}

    # coarse bins and few angle bins: this file tests navigation, and a fine
    # posterior only makes every redraw slower
    data = prepare_decoder_data(sorting, sorting.unit_ids, heading, frame_times, 0.1)
    wake = state_interval_mask(data, intervals, "WAKE")
    train, test = split_train_test(data, wake, 0.4, "blocks", 20.0, seed=1)
    model = fit_encoding_model(data, train, n_angle_bins=60)
    return decode(data, model, test, label="wake test")


@pytest.fixture
def widget(decoded):
    w = DecodedWidget(decoded, window_s=60.0)
    yield w
    plt.close(w.fig)


# ── the broken line ─────────────────────────────────────────────────────
def test_a_line_is_cut_where_it_would_cross_the_seam():
    x = np.arange(5.0)
    y = np.array([350.0, 355.0, 5.0, 10.0, 15.0])
    bx, by = break_at(x, y, [])
    assert np.isnan(by).sum() == 1
    assert np.isnan(by[2])


def test_a_line_is_cut_at_a_splice():
    x = np.arange(4.0)
    y = np.array([10.0, 12.0, 14.0, 16.0])   # no wrap anywhere
    bx, by = break_at(x, y, [2])
    assert np.isnan(by).sum() == 1
    assert np.isnan(by[2])


def test_a_clean_line_is_left_alone():
    x, y = np.arange(4.0), np.array([10.0, 12.0, 14.0, 16.0])
    bx, by = break_at(x, y, [])
    np.testing.assert_array_equal(bx, x)
    np.testing.assert_array_equal(by, y)


def test_a_single_point_does_not_crash_the_cutter():
    bx, by = break_at([1.0], [10.0], [])
    assert len(bx) == 1


# ── layout and colours ──────────────────────────────────────────────────
def test_the_scheme_is_white_black_and_green(widget):
    assert widget.ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    assert widget.actual_line.get_color() == ACTUAL_COLOR
    assert widget.decoded_line.get_color() == DECODED_COLOR
    # curves, not dots
    assert widget.actual_line.get_linestyle() != "None"
    assert widget.actual_line.get_marker() in ("", "None", None)


def test_it_opens_on_the_first_decoded_bin(widget):
    assert widget.t0 == 0.0
    assert widget.window_s == 60.0
    assert widget.ax.get_xlim() == pytest.approx((0.0, 60.0))


def test_the_axis_is_decoded_time_not_recording_time(widget, decoded):
    """Laid end to end, so no pixel is spent on data that was trained on."""
    assert widget.total_s == pytest.approx(decoded.duration_s.sum())
    assert widget.total_s < decoded.time_s[-1] - decoded.time_s[0]


# ── navigation ──────────────────────────────────────────────────────────
def test_arrows_pan_by_half_a_window(widget):
    widget._on_key(Key("right"))
    assert widget.t0 == pytest.approx(30.0)
    widget._on_key(Key("left"))
    assert widget.t0 == pytest.approx(0.0)


def test_panning_stops_at_the_ends(widget):
    widget.t0 = 1.0
    for _ in range(4):
        widget._on_key(Key("left"))
    assert widget.t0 == 0.0

    widget.t0 = widget.total_s - widget.window_s - 1.0
    for _ in range(4):
        widget._on_key(Key("right"))
    assert widget.t0 == pytest.approx(widget.total_s - widget.window_s)


def test_up_and_down_change_how_much_is_shown(widget):
    widget._on_key(Key("up"))
    assert widget.window_s > 60.0
    widget._on_key(Key("down"))
    assert widget.window_s == pytest.approx(60.0)


def test_the_window_cannot_shrink_or_grow_past_its_limits(widget):
    for _ in range(12):          # 1.5**12 is a factor of 130 either way
        widget._on_key(Key("down"))
    assert widget.window_s == pytest.approx(widget.min_window_s)
    for _ in range(20):
        widget._on_key(Key("up"))
    assert widget.window_s == pytest.approx(min(widget.max_window_s, widget.total_s))


def test_home_and_end_jump(widget):
    widget._on_key(Key("end"))
    assert widget.t0 == pytest.approx(widget.total_s - widget.window_s)
    widget._on_key(Key("home"))
    assert widget.t0 == 0.0


def test_an_unrelated_key_does_nothing(widget):
    before = (widget.t0, widget.window_s)
    widget._on_key(Key("q"))
    assert (widget.t0, widget.window_s) == before


# ── the slider ──────────────────────────────────────────────────────────
def test_the_slider_moves_the_view(widget):
    target = widget.slider.valmax / 2.0
    widget.slider.set_val(target)
    assert widget.t0 == pytest.approx(target)
    assert widget.ax.get_xlim()[0] == pytest.approx(target)


def test_the_arrows_move_the_slider_too(widget):
    widget._on_key(Key("right"))
    assert widget.slider.val == pytest.approx(widget.t0)


def test_syncing_the_slider_does_not_recurse(widget):
    """The slider drives the view and the view drives the slider.

    Two half-window steps, unless that runs off the end -- this asserts they
    stay in step, not any particular position.
    """
    widget.window_s = min(60.0, widget.total_s / 4)
    widget.t0 = 0.0
    widget._draw()

    widget._on_key(Key("right"))
    widget._on_key(Key("right"))
    assert widget.t0 == pytest.approx(widget.window_s)
    assert widget.slider.val == pytest.approx(widget.t0)


def test_the_slider_range_follows_the_window(widget):
    widget._on_key(Key("up"))
    assert widget.slider.valmax == pytest.approx(
        max(widget.total_s - widget.window_s, 1e-9)
    )


# ── splices ─────────────────────────────────────────────────────────────
def test_splices_are_marked(decoded, widget):
    assert widget.breaks.size > 0, "the blocked split should have spliced"
    visible = ((widget.break_s > widget.t0)
               & (widget.break_s < widget.t0 + widget.window_s)).sum()
    assert len(widget._break_lines) == visible


def test_a_splice_is_where_the_run_changes(decoded, widget):
    at = widget.breaks[0]
    assert decoded.run_index[at] != decoded.run_index[at - 1]


def test_the_title_counts_the_splices_on_screen(widget):
    widget.window_s = widget.total_s
    widget.t0 = 0.0
    widget._draw()
    assert "splice" in widget.ax.get_title()


def test_the_lines_do_not_join_across_a_splice(widget):
    """A splice is not a movement, and a line drawn over one says it was."""
    widget.window_s = widget.total_s
    widget.t0 = 0.0
    widget._draw()
    y = widget.actual_line.get_ydata()
    assert np.isnan(y).sum() >= widget.breaks.size


# ── redrawing ───────────────────────────────────────────────────────────
def test_the_posterior_is_redrawn_not_stacked(widget):
    for _ in range(5):
        widget._on_key(Key("right"))
    assert len([c for c in widget.ax.collections]) <= 2


def test_the_splice_lines_are_redrawn_not_stacked(widget):
    counts = []
    for _ in range(6):
        widget._on_key(Key("right"))
        counts.append(len(widget._break_lines))
    visible = ((widget.break_s > widget.t0)
               & (widget.break_s < widget.t0 + widget.window_s)).sum()
    assert counts[-1] == visible


def test_the_title_reports_both_clocks(widget):
    title = widget.ax.get_title()
    assert "decoded" in title and "recording" in title
    assert "median |error| here" in title


def test_it_works_without_a_posterior(decoded):
    """keep_posterior=False is a real option, and the lines still carry it."""
    saved, decoded.posterior = decoded.posterior, None
    try:
        w = DecodedWidget(decoded, window_s=30.0)
        assert w._mesh is None
        assert w.actual_line.get_xdata().size > 0
        plt.close(w.fig)
    finally:
        decoded.posterior = saved


def test_show_decoded_returns_the_widget(decoded):
    w = show_decoded(decoded, window_s=20.0)
    try:
        assert isinstance(w, DecodedWidget)
        assert w.window_s == 20.0
    finally:
        plt.close(w.fig)


def test_too_short_a_decode_is_refused(decoded):
    from dataclasses import replace

    tiny = replace(decoded, time_s=decoded.time_s[:1], duration_s=decoded.duration_s[:1])
    with pytest.raises(ValueError, match="holds 1 bins"):
        DecodedWidget(tiny)
