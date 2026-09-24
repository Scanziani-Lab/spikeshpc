"""The population MVL cut, and the population tuning plot.

The shuffle test passes any curve that beats chance, which with enough spikes
includes curves that are barely directional. Among tuned units the MVLs split
into that weak group and a strong one; the cut keeps the strong one.
"""

import copy

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spikeshpc.optitrack.store import load_hd_tuning, save_hd_tuning  # noqa: E402
from spikeshpc.optitrack.tuning import (  # noqa: E402
    apply_mvl_cutoff,
    compute_all_units_tuning_curves,
    compute_hd_tuning_significance,
    find_bimodal_threshold,
    plot_hd_tuning_population,
)

from test_tuning_store import FakeAnalyzer, FakeSorting, RATE  # noqa: E402

STRONG = {1: 30.0, 2: 120.0, 3: 200.0, 4: 290.0}  # unit -> preferred heading
WEAK = {5: 60.0, 6: 160.0, 7: 250.0, 8: 330.0}
FLAT = 9


@pytest.fixture(scope="module")
def population():
    rng = np.random.default_rng(4)
    n = int(RATE * 600)
    frame_times = np.arange(n) / RATE
    heading = np.cumsum(rng.normal(0, 3.0, n)) % 360.0

    def near(pref, width):
        return np.abs(((heading - pref + 180) % 360) - 180) < width

    spikes = {}
    for unit, pref in STRONG.items():  # sharp: nearly all firing within 30 deg
        p = np.where(near(pref, 30), 0.3, 0.003)
        spikes[unit] = frame_times[rng.random(n) < p]
    for unit, pref in WEAK.items():  # broad, shallow bump on a big baseline
        p = np.where(near(pref, 90), 0.06, 0.04)
        spikes[unit] = frame_times[rng.random(n) < p]
    spikes[FLAT] = frame_times[rng.random(n) < 0.05]

    analyzer = FakeAnalyzer(FakeSorting(spikes))
    common = dict(n_bins=36, n_shuffles=200, min_shift_s=5.0, min_peak_rate_hz=1.0)
    return analyzer, heading, frame_times, common


def _tuned(stats):
    return sorted(u for u, s in stats.items() if s.significant)


# ── the threshold itself ────────────────────────────────────────────────
def test_the_threshold_falls_in_the_gap():
    values = [0.10, 0.12, 0.15, 0.11, 0.55, 0.62, 0.70]
    cut = find_bimodal_threshold(values)
    assert 0.15 < cut < 0.55


def test_the_threshold_ignores_order_and_nans():
    a = find_bimodal_threshold([0.7, 0.1, np.nan, 0.12, 0.6])
    b = find_bimodal_threshold([0.1, 0.12, 0.6, 0.7])
    assert a == b


def test_nothing_to_split_is_refused():
    with pytest.raises(ValueError):
        find_bimodal_threshold([0.3])
    with pytest.raises(ValueError):
        find_bimodal_threshold([0.3, 0.3, 0.3])


# ── the cut, applied by the significance test ──────────────────────────
def test_without_a_cut_the_weak_units_count_as_tuned(population):
    analyzer, heading, frame_times, common = population
    stats = compute_hd_tuning_significance(analyzer, heading, frame_times, **common)
    assert _tuned(stats) == sorted([*STRONG, *WEAK])
    assert not any(s.weakly_tuned for s in stats.values())
    assert all(np.isnan(s.mvl_cutoff) for s in stats.values())


def test_bimodal_keeps_the_strong_group(population):
    analyzer, heading, frame_times, common = population
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, min_mvl="bimodal", **common
    )
    assert _tuned(stats) == sorted(STRONG)
    assert all(stats[u].weakly_tuned for u in WEAK)
    assert not stats[FLAT].weakly_tuned  # never passed the shuffle to begin with
    cut = stats[1].mvl_cutoff
    assert max(stats[u].mean_vector_length for u in WEAK) < cut
    assert min(stats[u].mean_vector_length for u in STRONG) > cut
    assert "weakly tuned" in str(stats[5])


def test_a_float_is_used_as_the_cut(population):
    analyzer, heading, frame_times, common = population
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, min_mvl=1.0, **common
    )
    assert _tuned(stats) == []
    assert all(s.mvl_cutoff == 1.0 for s in stats.values())


def test_the_cut_does_not_touch_the_statistics_themselves(population):
    analyzer, heading, frame_times, common = population
    a = compute_hd_tuning_significance(analyzer, heading, frame_times, **common)
    b = compute_hd_tuning_significance(
        analyzer, heading, frame_times, min_mvl="bimodal", **common
    )
    for u in a:
        assert a[u].p_value == b[u].p_value
        assert a[u].mean_vector_length == b[u].mean_vector_length


def test_a_misspelled_mode_fails_before_the_shuffles(population):
    analyzer, heading, frame_times, common = population
    with pytest.raises(ValueError, match="bimodal"):
        compute_hd_tuning_significance(
            analyzer, heading, frame_times, min_mvl="bimodel", **common
        )


def test_the_cut_can_be_redone_and_undone(population):
    analyzer, heading, frame_times, common = population
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, min_mvl="bimodal", **common
    )
    apply_mvl_cutoff(stats, None)
    assert _tuned(stats) == sorted([*STRONG, *WEAK])
    apply_mvl_cutoff(stats, "bimodal")
    assert _tuned(stats) == sorted(STRONG)


def test_bimodal_with_one_tuned_unit_warns_and_cuts_nothing(population):
    analyzer, heading, frame_times, common = population
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, unit_ids=[1, FLAT], **common
    )
    with pytest.warns(UserWarning, match="two tuned units"):
        cut = apply_mvl_cutoff(stats, "bimodal")
    assert np.isnan(cut)
    assert _tuned(stats) == [1]


def test_the_cut_survives_a_round_trip(tmp_path, population):
    analyzer, heading, frame_times, common = population
    curves = compute_all_units_tuning_curves(analyzer, heading, frame_times, n_bins=36)
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, min_mvl="bimodal", **common
    )
    depths = {u: 0.0 for u in curves}
    path = save_hd_tuning(tmp_path / "hd", "s1", heading, curves, stats, depths)
    loaded = load_hd_tuning(path)

    assert sorted(loaded.tuned_ids) == sorted(STRONG)
    assert loaded.stats[5].weakly_tuned
    assert loaded.stats[5].mvl_cutoff == pytest.approx(stats[5].mvl_cutoff)

    apply_mvl_cutoff(loaded.stats, None)  # re-cut a loaded file, no shuffles
    assert sorted(loaded.tuned_ids) == sorted([*STRONG, *WEAK])


# ── the population plot ─────────────────────────────────────────────────
@pytest.fixture(scope="module")
def loaded(tmp_path_factory, population):
    analyzer, heading, frame_times, common = population
    curves = compute_all_units_tuning_curves(analyzer, heading, frame_times, n_bins=360)
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, min_mvl="bimodal", **common
    )
    depths = {u: 0.0 for u in curves}
    path = save_hd_tuning(
        tmp_path_factory.mktemp("pop") / "hd", "s1", heading, curves, stats, depths
    )
    return load_hd_tuning(path)


def test_the_histogram_counts_each_tuned_unit_once_at_its_peak(loaded):
    fig, (ax_hist, ax_polar) = plot_hd_tuning_population(loaded, n_hist_bins=12)
    bars = ax_hist.patches
    assert len(bars) == len(STRONG)
    for bar, pref in zip(bars, sorted(STRONG.values())):  # drawn in peak order
        assert bar.get_x() <= pref + 15 and pref - 15 <= bar.get_x() + bar.get_width()
    assert len(ax_polar.lines) == len(STRONG)
    plt.close(fig)


def test_every_polar_curve_is_normalized_and_closed(loaded):
    fig, (_, ax_polar) = plot_hd_tuning_population(loaded, significant_only=False)
    assert len(ax_polar.lines) == len(loaded.unit_ids)
    for line in ax_polar.lines:
        theta, r = line.get_data()
        assert r.max() == pytest.approx(1.0)
        assert r[0] == r[-1] and theta[0] == theta[-1]
    plt.close(fig)


def test_units_get_distinct_colors_from_the_given_map(loaded):
    fig, (ax_hist, ax_polar) = plot_hd_tuning_population(loaded, cmap="viridis")
    line_colors = [tuple(line.get_color()) for line in ax_polar.lines]
    bar_colors = [tuple(bar.get_facecolor()) for bar in ax_hist.patches]
    assert len(set(line_colors)) == len(line_colors)
    assert line_colors == bar_colors  # same unit, same color in both panels
    cmap = matplotlib.colormaps["viridis"]
    assert np.allclose(line_colors[0], cmap(0.5 / len(line_colors)))
    plt.close(fig)


def test_nothing_tuned_says_so(loaded):
    loaded = copy.deepcopy(loaded)
    apply_mvl_cutoff(loaded.stats, 1.0)
    with pytest.raises(ValueError, match="none are significantly tuned"):
        plot_hd_tuning_population(loaded)
