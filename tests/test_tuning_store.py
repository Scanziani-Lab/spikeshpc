"""The quiet-unit floor, and saving tuning so the decoder need not recompute it."""

import json

import numpy as np
import pytest

from spikeshpc.optitrack.store import load_hd_tuning, save_hd_tuning
from spikeshpc.optitrack.tuning import (
    compute_all_units_tuning_curves,
    compute_hd_tuning_significance,
)

RATE = 120.0


class FakeSorting:
    def __init__(self, spikes):
        self.unit_ids = np.array(sorted(spikes))
        self._spikes = {k: np.sort(np.asarray(v, dtype=float)) for k, v in spikes.items()}

    def get_unit_spike_train(self, unit_id, return_times=False):
        return self._spikes[int(unit_id)]


class FakeAnalyzer:
    def __init__(self, sorting):
        self.sorting = sorting


@pytest.fixture
def session():
    """One brisk head-direction cell and one that fires the same way, but rarely.

    Both are tuned to 90 degrees and both concentrate their spikes there, so
    the shuffle test has no reason to separate them -- the sparse unit's null
    is just as sparse as it is. Only the peak rate tells them apart.
    """
    rng = np.random.default_rng(0)
    n = int(RATE * 600)
    frame_times = np.arange(n) / RATE
    heading = np.cumsum(rng.normal(0, 3.0, n)) % 360.0

    offset = np.abs(((heading - 90 + 180) % 360) - 180)
    near = offset < 30

    loud = frame_times[rng.random(n) < np.where(near, 0.5, 0.005)]
    quiet = frame_times[rng.random(n) < np.where(near, 0.004, 0.00004)]

    sorting = FakeSorting({1: loud, 2: quiet})
    return FakeAnalyzer(sorting), heading, frame_times


# ── the quiet-unit floor ────────────────────────────────────────────────
def test_both_units_look_tuned_without_a_rate_floor(session):
    analyzer, heading, frame_times = session
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, n_bins=36, n_shuffles=200, min_shift_s=5.0
    )
    assert stats[1].significant and stats[2].significant
    assert not stats[1].too_quiet and not stats[2].too_quiet


def test_the_rate_floor_drops_the_unit_firing_almost_nothing(session):
    analyzer, heading, frame_times = session
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, n_bins=36, n_shuffles=200,
        min_shift_s=5.0, min_peak_rate_hz=1.0,
    )
    assert stats[1].peak_rate_hz >= 1.0
    assert stats[2].peak_rate_hz < 1.0

    assert stats[1].significant and not stats[1].too_quiet
    assert not stats[2].significant and stats[2].too_quiet


def test_the_floor_does_not_touch_the_statistics_themselves(session):
    """It gates the verdict; it must not quietly change the p-value or the MVL."""
    analyzer, heading, frame_times = session
    common = dict(n_bins=36, n_shuffles=200, min_shift_s=5.0, seed=3)
    without = compute_hd_tuning_significance(analyzer, heading, frame_times, **common)
    with_floor = compute_hd_tuning_significance(
        analyzer, heading, frame_times, min_peak_rate_hz=1.0, **common
    )
    for unit in (1, 2):
        assert without[unit].p_value == with_floor[unit].p_value
        assert without[unit].mean_vector_length == with_floor[unit].mean_vector_length
        assert without[unit].peak_rate_hz == with_floor[unit].peak_rate_hz


def test_a_floor_of_zero_changes_nothing(session):
    analyzer, heading, frame_times = session
    common = dict(n_bins=36, n_shuffles=100, min_shift_s=5.0, seed=1)
    a = compute_hd_tuning_significance(analyzer, heading, frame_times, **common)
    b = compute_hd_tuning_significance(
        analyzer, heading, frame_times, min_peak_rate_hz=0.0, **common
    )
    assert [a[u].significant for u in a] == [b[u].significant for u in b]
    assert not any(s.too_quiet for s in b.values())


def test_the_verdict_appears_in_the_summary_line(session):
    analyzer, heading, frame_times = session
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, n_bins=36, n_shuffles=100,
        min_shift_s=5.0, min_peak_rate_hz=1.0,
    )
    assert "too quiet" in str(stats[2])
    assert "too quiet" not in str(stats[1])


# ── saving and reloading ────────────────────────────────────────────────
@pytest.fixture
def saved(tmp_path, session):
    analyzer, heading, frame_times = session
    curves = compute_all_units_tuning_curves(
        analyzer, heading, frame_times, n_bins=36
    )
    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, n_bins=36, n_shuffles=100,
        min_shift_s=5.0, min_peak_rate_hz=1.0,
    )
    depths = {1: -120.5, 2: 340.0}
    path = save_hd_tuning(
        tmp_path / "tuning" / "hd", "s1", heading, curves, stats, depths,
        parameters={"n_bins": 36, "min_peak_rate_hz": 1.0, "states": ["WAKE"]},
    )
    return path, heading, curves, stats, depths


def test_a_round_trip_preserves_everything_the_decoder_needs(saved):
    path, heading, curves, stats, depths = saved
    loaded = load_hd_tuning(path)

    assert loaded.session == "s1"
    np.testing.assert_allclose(loaded.heading_deg, heading, rtol=1e-6)
    np.testing.assert_array_equal(loaded.unit_ids, [1, 2])
    for unit in (1, 2):
        np.testing.assert_allclose(loaded.curve(unit)[1], curves[unit][1])
        np.testing.assert_allclose(loaded.curve(unit)[0], curves[unit][0])
        assert loaded.depths[list(loaded.unit_ids).index(unit)] == depths[unit]


def test_unit_ids_come_back_able_to_index_a_sorting(saved, session):
    """Stringified ids would look fine and then fail against the sorting."""
    path, *_ = saved
    analyzer, _, _ = session
    loaded = load_hd_tuning(path)

    assert loaded.unit_ids.dtype.kind in "iu", loaded.unit_ids.dtype
    for unit in loaded.tuned_ids:
        assert analyzer.sorting.get_unit_spike_train(unit).size > 0


def test_the_tuned_set_is_what_the_floor_left(saved):
    path, _, _, stats, _ = saved
    loaded = load_hd_tuning(path)
    assert list(loaded.tuned_ids) == [1]
    assert loaded.stats[2].too_quiet
    assert not loaded.stats[2].significant


def test_statistics_survive_the_round_trip(saved):
    path, _, _, stats, _ = saved
    loaded = load_hd_tuning(path)
    for unit in (1, 2):
        for name in ("mean_vector_length", "preferred_direction_deg",
                     "peak_rate_hz", "mean_rate_hz", "p_value"):
            assert getattr(loaded.stats[unit], name) == pytest.approx(
                getattr(stats[unit], name)
            )
        assert loaded.stats[unit].n_spikes == stats[unit].n_spikes


def test_the_parameters_are_kept_so_the_choices_can_be_checked_later(saved):
    path, *_ = saved
    loaded = load_hd_tuning(path)
    assert loaded.parameters["min_peak_rate_hz"] == 1.0
    assert loaded.parameters["states"] == ["WAKE"]


def test_the_json_sidecar_is_readable_and_not_needed(saved):
    path, *_ = saved
    sidecar = path.with_suffix(".json")
    summary = json.loads(sidecar.read_text())
    assert summary["n_units"] == 2 and summary["n_tuned"] == 1
    assert summary["units"]["2"]["too_quiet"] is True

    sidecar.unlink()
    assert load_hd_tuning(path).session == "s1"


def test_the_widget_dicts_come_back_in_the_shape_the_widgets_want(saved):
    path, _, curves, _, _ = saved
    loaded = load_hd_tuning(path)

    as_dict = loaded.curve_dict(loaded.tuned_ids)
    assert list(as_dict) == [1]
    bins, rate = as_dict[1]
    np.testing.assert_allclose(rate, curves[1][1])
    assert loaded.depth_dict(loaded.tuned_ids)[1] == pytest.approx(-120.5)


def test_a_unit_missing_a_depth_is_dropped_rather_than_padded(tmp_path, saved):
    path, heading, curves, stats, depths = saved
    out = save_hd_tuning(
        tmp_path / "partial", "s1", heading, curves, stats, {1: -120.5}
    )
    assert list(load_hd_tuning(out).unit_ids) == [1]


def test_saving_nothing_is_refused(tmp_path, saved):
    _, heading, curves, stats, _ = saved
    with pytest.raises(ValueError, match="nothing"):
        save_hd_tuning(tmp_path / "empty", "s1", heading, curves, stats, {})


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hd_tuning(tmp_path / "nope")
