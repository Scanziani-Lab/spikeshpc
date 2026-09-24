"""The fast bombcell GUI draws what bombcell's own GUI draws, without the rescans.

Synthetic units, notebook mode under Agg; the Qt window needs a display.
"""

import numpy as np
import pytest

bc_gui = pytest.importorskip("bombcell.unit_quality_gui")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from spikeshpc.bombcell_gui import (  # noqa: E402
    HISTOGRAM_METRICS,
    FastUnitQualityGUI,
    _scatter_to_markers,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive", "ignore::RuntimeWarning"
)
FS = 30000.0


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(0)
    n_units, n_ch, n_t, duration = 6, 8, 61, 600.0
    rates = np.array([2.0, 8.0, 15.0, 4.0, 30.0, 1.0])
    times, clusters = [], []
    for unit, rate in enumerate(rates):
        t = rng.uniform(0, duration, rng.poisson(rate * duration))
        times.append(t)
        clusters.append(np.full(len(t), unit))
    times, clusters = np.concatenate(times), np.concatenate(clusters)
    order = np.argsort(times)
    times, clusters = times[order], clusters[order]

    shape = -np.exp(-((np.arange(n_t) - 20) ** 2) / 8.0) + 0.3 * np.exp(-((np.arange(n_t) - 30) ** 2) / 30.0)
    templates = np.zeros((n_units, n_t, n_ch))
    for unit in range(n_units):
        peak = unit % n_ch
        for ch in range(n_ch):
            templates[unit, :, ch] = shape * np.exp(-abs(ch - peak) / 1.5)

    ephys = {
        "spike_times": times,
        "spike_clusters": clusters,
        "template_waveforms": templates,
        "template_amplitudes": rng.normal(20, 3, len(times)).clip(5),
        "channel_positions": np.column_stack([np.zeros(n_ch), 20.0 * np.arange(n_ch)]),
    }
    qm = pd.DataFrame({"phy_clusterID": np.arange(n_units),
                       "nSpikes": np.bincount(clusters, minlength=n_units)})
    for metric in HISTOGRAM_METRICS.values():
        if metric not in qm:
            qm[metric] = rng.normal(1.0, 0.3, n_units)
    param = {"ephys_sample_rate": FS, "minNumSpikes": 50, "tauR_valuesMin": 0.002,
             "tauR_valuesMax": 0.002, "tauR_valuesStep": 0.0005, "computeTimeChunks": False}
    unit_types = np.array([1, 2, 0, 1, 3, 2])
    return ephys, qm, param, unit_types


@pytest.fixture
def gui(inputs, tmp_path):
    ephys, qm, param, unit_types = inputs
    g = FastUnitQualityGUI(ephys, qm.copy(), param=dict(param), unit_types=unit_types.copy(),
                           save_path=str(tmp_path), window="notebook", prefetch=False)
    yield g
    g.close()
    plt.close("all")


def test_notebook_mode_draws_bombcells_layout_and_keeps_one_figure(gui):
    assert not gui._qt
    n_axes = len(gui.fig.axes)
    gui.unit_slider.value = 3
    assert gui._shown_idx == 3
    assert len(gui.fig.axes) == n_axes
    assert plt.get_fignums() == [gui.fig.number]


def test_units_by_depth_matches_a_direct_computation(gui, inputs):
    ephys = inputs[0]
    loc = gui._location_data()
    for i, log_rate, depth in zip(loc["unit_idx"], loc["log_rate"], loc["depth"]):
        unit = gui.unique_units[i]
        t = ephys["spike_times"][ephys["spike_clusters"] == unit]
        assert log_rate == pytest.approx(np.log10(max(len(t) / (t.max() - t.min()), 0.01)))
        max_ch = int(gui.all_max_channels[unit])
        assert depth == ephys["channel_positions"][max_ch, 1]


def test_panels_see_only_the_current_units_spikes_and_then_everything_again(gui, inputs):
    ephys = inputs[0]
    with gui._scoped_to(4):
        assert np.all(gui.ephys_data["spike_clusters"] == 4)
        np.testing.assert_array_equal(
            gui.ephys_data["spike_times"], ephys["spike_times"][ephys["spike_clusters"] == 4])
    assert gui.ephys_data is ephys


def test_the_amplitude_fit_panel_is_bombcells(gui):
    for idx in range(gui.n_units):
        data = gui.get_unit_data(idx)
        fig, (theirs, ours) = plt.subplots(1, 2)
        with gui._scoped_to(data["unit_id"]):
            bc_gui.InteractiveUnitQualityGUI.plot_amplitude_fit(gui, theirs, data)
            gui.plot_amplitude_fit(ours, data)
        assert sorted(t.get_text() for t in theirs.texts) == sorted(t.get_text() for t in ours.texts)
        assert [p.get_width() for p in theirs.patches] == [p.get_width() for p in ours.patches]
        for a, b in zip(theirs.lines, ours.lines, strict=True):
            np.testing.assert_allclose(a.get_xydata(), b.get_xydata())
        plt.close(fig)


def test_fits_are_cached_per_unit(gui):
    unit = gui.unique_units[gui.current_unit_idx]
    assert unit in gui._fits
    before = gui._fits[unit]
    gui.unit_slider.value = 1
    gui.unit_slider.value = 0
    assert gui._fits[unit] is before


def test_big_scatters_become_markers_with_the_same_points_and_limits():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 100, 20000), rng.normal(0, 1, 20000)
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=3, alpha=0.6, c="darkorange", edgecolors="none")
    ax.scatter([1, 2], [0, 0], s=500)  # small: left alone
    limits = ax.get_xlim(), ax.get_ylim()
    _scatter_to_markers(ax)
    assert len(ax.collections) == 1 and len(ax.lines) == 1
    np.testing.assert_array_equal(ax.lines[0].get_xydata(), np.column_stack([x, y]))
    assert ax.lines[0].get_markersize() == pytest.approx(np.sqrt(3))
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    plt.close(fig)


def test_qt_is_refused_without_the_qt_backend(inputs, tmp_path):
    ephys, qm, param, unit_types = inputs
    with pytest.raises(ValueError, match="%matplotlib qt"):
        FastUnitQualityGUI(ephys, qm.copy(), param=dict(param), unit_types=unit_types.copy(),
                           save_path=str(tmp_path), window="qt")
