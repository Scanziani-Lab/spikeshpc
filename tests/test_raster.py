"""Spike rasters on a decoded-time axis: where each spike lands, row order and color.

The failures guarded here all draw a plausible raster:

  * a spike placed by recording time on an axis of decoded time, so it lands
    in the wrong bin -- or drawn at all when it fell in no decoded bin;
  * rows in unit order rather than preferred direction, so a sweep of activity
    across the population does not read as one;
  * a colormap applied to the wrong unit after sorting.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from spikeshpc.raster import RasterPanel, UnitRaster, on_decoded_axis, unit_raster  # noqa: E402

from test_decoder import FakeAnalyzer, FakeSorting  # noqa: E402

# two decoded stretches on the recording clock: 10-12 s and 50-51 s, 0.5 s bins
BIN_START = np.array([10.0, 10.5, 11.0, 11.5, 50.0, 50.5])
BIN_STOP = BIN_START + 0.5
AXIS_EDGES = np.arange(7) * 0.5  # laid end to end: 0-2 s, then 2-3 s


# ── placing spikes ──────────────────────────────────────────────────────
def test_a_spike_keeps_its_offset_into_its_bin():
    x = on_decoded_axis([10.1, 11.9, 50.2], BIN_START, BIN_STOP, AXIS_EDGES)
    np.testing.assert_allclose(x, [0.1, 1.9, 2.2])


def test_a_spike_in_no_decoded_bin_is_dropped():
    x = on_decoded_axis([5.0, 12.0, 30.0, 49.99, 51.0, 60.0], BIN_START, BIN_STOP, AXIS_EDGES)
    assert x.size == 0


def test_sorted_spikes_come_back_sorted():
    times = np.sort(np.random.default_rng(0).uniform(0, 60, 500))
    x = on_decoded_axis(times, BIN_START, BIN_STOP, AXIS_EDGES)
    assert np.all(np.diff(x) >= 0)


# ── the raster ──────────────────────────────────────────────────────────
def test_unit_raster_reads_a_sorting_or_an_analyzer():
    sorting = FakeSorting({3: np.array([11.0, 10.2]), 7: np.array([50.4])})
    for source in (sorting, FakeAnalyzer(sorting)):
        raster = unit_raster(source, [7, 3], [200.0, 400.0])
        assert raster.labels == ["7", "3"]
        np.testing.assert_allclose(raster.spike_times[1], [10.2, 11.0])  # sorted
        np.testing.assert_allclose(raster.preferred_deg, [200.0, 40.0])  # wrapped


def test_a_raster_that_does_not_add_up_is_refused():
    with pytest.raises(ValueError, match="2 units of spikes but 1 preferred"):
        UnitRaster([[1.0], [2.0]], [10.0], ["a", "b"])
    with pytest.raises(ValueError, match="at least one unit"):
        UnitRaster([], [], [])


@pytest.fixture
def panel():
    # units listed out of order: preferred 270, 10, 355 and 90 degrees
    raster = UnitRaster(
        spike_times=[[10.1, 10.6], [10.2, 50.1, 50.6], [11.2], [30.0, 11.7]],
        preferred_deg=[270.0, 10.0, 355.0, 90.0],
        labels=["a", "b", "c", "d"],
    )
    fig, ax = plt.subplots()
    ax.set_xlim(0, 3)
    p = RasterPanel(ax, raster, BIN_START, BIN_STOP, AXIS_EDGES)
    yield p
    plt.close(fig)


def test_rows_run_up_by_preferred_direction(panel):
    assert list(panel.order) == [1, 3, 0, 2]  # 10, 90, 270, 355 degrees
    labels = [t.get_text() for t in panel.ax.get_yticklabels()]
    assert labels == ["b", "d", "a", "c"]
    assert panel.ax.get_ylim() == (0, 4)


def test_each_row_takes_its_preferred_directions_color(panel):
    hsv = matplotlib.colormaps["hsv"]
    np.testing.assert_allclose(panel.row_colors[0], hsv(10.0 / 360.0))
    np.testing.assert_allclose(panel.row_colors[3], hsv(355.0 / 360.0))
    # hsv wraps round: a unit near 360 is colored nearly like one near 0
    assert np.allclose(hsv(0.0), hsv(1.0), atol=0.1)


def test_only_the_spikes_in_the_window_are_drawn(panel):
    assert panel.draw(0.0, 3.0) == 7  # the spike at 30 s is in no decoded bin
    assert panel.draw(0.0, 1.0) == 3  # 10.1, 10.2 and 10.6 s
    segments = panel._collection.get_segments()
    rows = sorted(round(s[0][1] - 0.12) for s in segments)
    assert rows == [0, 2, 2]  # unit b (row 0) once, unit a (row 2) twice


def test_the_raster_is_redrawn_not_stacked(panel):
    for t0 in (0.0, 1.0, 2.0, 0.0):
        panel.draw(t0, t0 + 1.0)
    assert sum(isinstance(c, LineCollection) for c in panel.ax.collections) == 1


def test_too_many_spikes_leave_a_note_instead(panel):
    panel.max_spikes = 5
    assert panel.draw(0.0, 3.0) == 0
    assert panel._collection is None
    assert "7 spikes in view" in panel._note.get_text()
    panel.draw(0.0, 1.0)  # zoomed in, drawn again
    assert panel._note is None and panel.n_drawn == 3


def test_the_colorbar_reads_back_preferred_direction(panel):
    bar = panel.add_colorbar(panel.ax.figure)
    assert list(bar.get_ticks()) == [0, 90, 180, 270, 360]
