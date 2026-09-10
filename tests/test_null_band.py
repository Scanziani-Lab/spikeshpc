"""The shuffled band: what a chance tuning curve looks like for this unit.

A p-value compresses the whole shuffle distribution into one number. For a
sparse unit that number is least intuitive precisely where it matters most, so
the distribution is kept as a per-bin envelope and drawn under the real curve.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spikeshpc.optitrack.store import load_hd_tuning, save_hd_tuning  # noqa: E402
from spikeshpc.optitrack.tuning import (  # noqa: E402
    compute_all_units_tuning_curves,
    compute_hd_tuning_significance,
)
from spikeshpc.optitrack.widgets import show_hd_tuning_widget  # noqa: E402

from test_tuning_store import FakeAnalyzer, FakeSorting, RATE  # noqa: E402


@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(1)
    n = int(RATE * 400)
    frame_times = np.arange(n) / RATE
    heading = np.cumsum(rng.normal(0, 3.0, n)) % 360.0

    offset = np.abs(((heading - 200 + 180) % 360) - 180)
    tuned = frame_times[rng.random(n) < np.where(offset < 30, 0.4, 0.004)]
    flat = frame_times[rng.random(n) < 0.05]           # fires regardless of heading

    analyzer = FakeAnalyzer(FakeSorting({1: tuned, 2: flat}))
    curves = compute_all_units_tuning_curves(analyzer, heading, frame_times, n_bins=36)
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, n_bins=36, n_shuffles=200, min_shift_s=5.0
    )
    return analyzer, heading, frame_times, curves, stats


# ── the band itself ─────────────────────────────────────────────────────
def test_the_band_is_kept_for_every_unit(scored):
    *_, stats = scored
    for stat in stats.values():
        assert stat.null_band.shape == (3, 36)
        assert stat.null_percentiles == (2.5, 50.0, 97.5)
        assert len(stat.null_bin_centers_deg) == 36


def test_the_band_is_ordered_by_percentile(scored):
    *_, stats = scored
    for stat in stats.values():
        low, median, high = stat.null_band
        assert np.all(low <= median + 1e-9)
        assert np.all(median <= high + 1e-9)


def test_a_tuned_unit_rises_clear_of_its_own_shuffles(scored):
    """Compared bin by bin: the band is a function of heading, not one number.

    A shifted train keeps every spike, so the null sits at the unit's mean
    rate across the whole circle. A tuned unit beats it near its preferred
    direction and falls below it everywhere else -- which is why the
    interesting comparison is per bin, and why max-against-max would be
    comparing two different headings.
    """
    _, _, _, curves, stats = scored
    _, rate = curves[1]
    low, _, high = stats[1].null_band

    peak = int(np.argmax(rate))
    assert rate[peak] > 2 * high[peak], (rate[peak], high[peak])
    # and it is outside the band over a lobe, not at a single lucky bin
    assert np.count_nonzero(rate > high) >= 3
    # while the flanks sit below where chance would put them
    assert np.count_nonzero(rate < low) >= 3
    assert stats[1].significant


def test_an_untuned_unit_stays_inside_its_band(scored):
    """The comparison has to be able to come out the other way."""
    _, _, _, curves, stats = scored
    _, rate = curves[2]
    low, _, high = stats[2].null_band
    inside = (rate >= low - 1e-9) & (rate <= high + 1e-9)
    assert inside.mean() > 0.8, inside.mean()
    assert not stats[2].significant


def test_the_band_sits_at_the_unit_s_own_rate(scored):
    """A shift moves spikes about; it does not create or destroy them."""
    _, _, _, curves, stats = scored
    for unit in (1, 2):
        median = stats[unit].null_band[1]
        assert median.mean() == pytest.approx(stats[unit].mean_rate_hz, rel=0.35)


def test_the_band_can_be_declined(scored):
    analyzer, heading, frame_times, _, _ = scored
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, n_bins=36, n_shuffles=20,
        min_shift_s=5.0, null_percentiles=(),
    )
    assert all(s.null_band is None for s in stats.values())
    assert all(s.null_percentiles == () for s in stats.values())


def test_keeping_the_band_does_not_change_the_verdict(scored):
    analyzer, heading, frame_times, _, with_band = scored
    without = compute_hd_tuning_significance(
        analyzer, heading, frame_times, n_bins=36, n_shuffles=200,
        min_shift_s=5.0, null_percentiles=(),
    )
    for unit in with_band:
        assert with_band[unit].p_value == without[unit].p_value
        assert with_band[unit].significant == without[unit].significant


# ── through the store ───────────────────────────────────────────────────
def test_the_band_survives_a_save_and_reload(tmp_path, scored):
    _, heading, _, curves, stats = scored
    path = save_hd_tuning(
        tmp_path / "hd", "s1", heading, curves, stats, {1: 0.0, 2: 10.0}
    )
    loaded = load_hd_tuning(path)
    for unit in (1, 2):
        np.testing.assert_allclose(loaded.stats[unit].null_band, stats[unit].null_band)
        assert loaded.stats[unit].null_percentiles == (2.5, 50.0, 97.5)
        np.testing.assert_allclose(
            loaded.stats[unit].null_bin_centers_deg, stats[unit].null_bin_centers_deg
        )


def test_a_file_without_bands_still_loads(tmp_path, scored):
    """Older saves, and runs that declined the band."""
    analyzer, heading, frame_times, curves, _ = scored
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, n_bins=36, n_shuffles=20,
        min_shift_s=5.0, null_percentiles=(),
    )
    path = save_hd_tuning(
        tmp_path / "nb", "s1", heading, curves, stats, {1: 0.0, 2: 10.0}
    )
    loaded = load_hd_tuning(path)
    assert loaded.stats[1].null_band is None
    assert loaded.stats[1].p_value == stats[1].p_value


# ── the widget ──────────────────────────────────────────────────────────
def test_the_widget_shades_the_band_and_reports_the_verdict(scored):
    _, _, _, curves, stats = scored
    w = show_hd_tuning_widget(curves, unit_depths={1: -20.0, 2: 5.0}, stats=stats)
    try:
        assert w._null_artists, "nothing was shaded"
        title = w.ax.get_title()
        assert "MVL" in title and "p =" in title
        assert "tuned" in title
        assert w.ax.get_legend() is not None
    finally:
        plt.close(w.fig)


def test_the_shading_closes_across_the_zero_seam(scored):
    """A band stopping at the first and last bin centre leaves a gap at north."""
    _, _, _, curves, stats = scored
    w = show_hd_tuning_widget(curves, stats=stats)
    try:
        band = w._null_artists[0].get_paths()[0].vertices[:, 0]
        assert band.min() < 0.0 and band.max() > 360.0
    finally:
        plt.close(w.fig)


def test_the_axis_fits_the_band_as_well_as_the_curve(scored):
    """An untuned unit's band can out-reach its curve; clipping it would flatter."""
    _, _, _, curves, stats = scored
    w = show_hd_tuning_widget(curves, stats=stats)
    try:
        w.index = list(curves).index(2)
        w._draw()
        assert w.ax.get_ylim()[1] >= stats[2].null_band[-1].max()
    finally:
        plt.close(w.fig)


def test_switching_units_redraws_the_band_rather_than_stacking_them(scored):
    _, _, _, curves, stats = scored
    w = show_hd_tuning_widget(curves, stats=stats)
    try:
        before = len(w._null_artists)
        for _ in range(4):
            w._on_key(type("E", (), {"key": "right"})())
        assert len(w._null_artists) == before
    finally:
        plt.close(w.fig)


def test_the_widget_still_works_with_no_stats_at_all(scored):
    _, _, _, curves, _ = scored
    w = show_hd_tuning_widget(curves)
    try:
        assert not w._null_artists
        assert "MVL" not in w.ax.get_title()
    finally:
        plt.close(w.fig)


def test_the_band_can_be_hidden(scored):
    _, _, _, curves, stats = scored
    w = show_hd_tuning_widget(curves, stats=stats, show_null=False)
    try:
        assert not w._null_artists
        assert "MVL" in w.ax.get_title()   # the verdict is still worth having
    finally:
        plt.close(w.fig)
