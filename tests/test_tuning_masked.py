"""Restricting head-direction tuning to wake epochs."""

import numpy as np
import pytest

from spikeshpc.optitrack.tuning import (
    compute_all_units_tuning_curves,
    compute_hd_tuning_significance,
    compute_mean_vector_length,
)
from spikeshpc.states import frames_in_states

RATE = 120.0


class FakeSorting:
    def __init__(self, spike_times):
        self.unit_ids = ["u1"]
        self._spikes = np.asarray(spike_times, dtype=float)

    def get_unit_spike_train(self, unit_id, return_times=False):
        return self._spikes


class FakeAnalyzer:
    def __init__(self, sorting):
        self.sorting = sorting


@pytest.fixture
def wake_then_sleep():
    """Tuned during wake, then a long immobile sleep block at one heading.

    Wake heading is a random walk rather than a periodic sweep: a sweep
    repeats, so circularly shifting the spike train lands back in register and
    the shuffle null comes out as tuned as the data.

    Sleep: the head is parked at 270 degrees and the unit fires steadily --
    the contamination that filtering is meant to remove.
    """
    rng = np.random.default_rng(0)
    # Sleep sits BETWEEN two wake blocks on purpose. With it at the end, a
    # prefiltered frame_times simply stops before the sleep spikes and
    # np.histogram drops them; the splice only bites when the removed span has
    # kept frames either side of it, which is the usual case.
    n_wake = int(RATE * 150)
    n_sleep = int(RATE * 240)
    n_total = 2 * n_wake + n_sleep
    frame_times = np.arange(n_total) / RATE

    wake_a = np.cumsum(rng.normal(0, 4.0, n_wake)) % 360.0
    wake_b = np.cumsum(rng.normal(0, 4.0, n_wake)) % 360.0
    # the head is already parked at the sleep heading on the last wake frame,
    # so the spliced interval's heading is 270 and far from the tuned peak --
    # otherwise the contamination hides under the real 90-degree lobe
    wake_a[-1] = 270.0
    heading = np.r_[wake_a, np.full(n_sleep, 270.0), wake_b]

    # fires near 90 degrees, with a low background elsewhere
    wake_idx = np.r_[np.arange(n_wake), np.arange(n_wake + n_sleep, n_total)]
    offset = np.abs(((heading[wake_idx] - 90 + 180) % 360) - 180)
    fires = rng.random(len(wake_idx)) < np.where(offset < 30, 0.8, 0.02)
    wake_spikes = frame_times[wake_idx][fires] + rng.uniform(
        0, 1 / RATE, int(fires.sum())
    )

    sleep_start = float(frame_times[n_wake])
    sleep_stop = float(frame_times[n_wake + n_sleep])
    sleep_spikes = np.arange(sleep_start, sleep_stop, 0.01)   # 100 Hz

    sorting = FakeSorting(np.sort(np.r_[wake_spikes, sleep_spikes]))
    intervals = {
        "WAKE": [[0.0, sleep_start], [sleep_stop, float(frame_times[-1] + 1 / RATE)]],
        "NREM": [[sleep_start, sleep_stop]],
    }
    return FakeAnalyzer(sorting), heading, frame_times, intervals


def preferred(curve):
    bin_centers, rate = curve
    return compute_mean_vector_length(bin_centers, rate)


def test_unfiltered_tuning_carries_a_spurious_peak_at_the_sleep_heading(
    wake_then_sleep,
):
    """Sleep firing at a parked heading shows up as a second peak.

    Occupancy normalisation stops the long dwell from dominating outright --
    it divides the sleep spikes by the sleep duration -- so the failure is a
    spurious lobe at 270 degrees and a less selective curve, not a flipped
    preferred direction.
    """
    analyzer, heading, frame_times, intervals = wake_then_sleep
    _, keep = frames_in_states(frame_times, intervals, "WAKE")

    bins, unfiltered = compute_all_units_tuning_curves(
        analyzer, heading, frame_times, n_bins=36
    )["u1"]
    _, filtered = compute_all_units_tuning_curves(
        analyzer, heading, frame_times, n_bins=36, interval_mask=keep
    )["u1"]

    bin_270 = int(np.argmin(np.abs(bins - 270)))
    assert unfiltered[bin_270] > 5 * filtered[bin_270], (
        unfiltered[bin_270], filtered[bin_270]
    )
    # and the curve is less directional with the sleep block left in
    assert compute_mean_vector_length(bins, unfiltered)[0] < compute_mean_vector_length(
        bins, filtered
    )[0]


def test_masking_to_wake_recovers_the_real_preferred_direction(wake_then_sleep):
    analyzer, heading, frame_times, intervals = wake_then_sleep
    _, keep = frames_in_states(frame_times, intervals, "WAKE")

    curves = compute_all_units_tuning_curves(
        analyzer, heading, frame_times, n_bins=36, interval_mask=keep
    )
    mvl, direction = preferred(curves["u1"])
    assert abs(((direction - 90 + 180) % 360) - 180) < 20, direction
    assert mvl > 0.5


def test_prefiltering_the_arrays_instead_gives_the_wrong_answer(wake_then_sleep):
    """Why interval_mask exists rather than a filtered frame_times.

    Handing in only the wake frames splices the removed sleep block into one
    long interval. Its duration dominates the occupancy and every sleep spike
    lands in it, so the sleep heading contaminates the curve anyway.
    """
    analyzer, heading, frame_times, intervals = wake_then_sleep
    frame_mask, keep = frames_in_states(frame_times, intervals, "WAKE")

    bins, naive = compute_all_units_tuning_curves(
        analyzer, heading[frame_mask], frame_times[frame_mask], n_bins=36
    )["u1"]
    _, correct = compute_all_units_tuning_curves(
        analyzer, heading, frame_times, n_bins=36, interval_mask=keep
    )["u1"]

    # every sleep spike is credited to whatever heading the last kept frame
    # happened to have, and the 240 s gap becomes that bin's occupancy
    boundary = int(np.flatnonzero(np.diff(frame_mask.astype(int)) == -1)[0])
    spliced_heading = heading[boundary]
    assert spliced_heading == 270.0
    spliced = int(np.argmin(np.abs(bins - spliced_heading)))
    assert naive[spliced] > correct[spliced] + 10, (
        spliced_heading, naive[spliced], correct[spliced]
    )


def test_significance_accepts_the_same_mask(wake_then_sleep):
    analyzer, heading, frame_times, intervals = wake_then_sleep
    _, keep = frames_in_states(frame_times, intervals, "WAKE")

    stats = compute_hd_tuning_significance(
        analyzer, heading, frame_times, n_bins=36, n_shuffles=200,
        min_shift_s=5.0, interval_mask=keep,
    )
    s = stats["u1"]
    assert s.significant
    assert abs(((s.preferred_direction_deg - 90 + 180) % 360) - 180) < 20


def test_a_frame_length_mask_is_rejected(wake_then_sleep):
    """Off-by-one here would silently shift every heading by one frame."""
    analyzer, heading, frame_times, intervals = wake_then_sleep
    frame_mask, _ = frames_in_states(frame_times, intervals, "WAKE")

    with pytest.raises(ValueError, match="masks intervals, not"):
        compute_all_units_tuning_curves(
            analyzer, heading, frame_times, interval_mask=frame_mask
        )


def test_an_empty_mask_is_rejected(wake_then_sleep):
    analyzer, heading, frame_times, _ = wake_then_sleep
    empty = np.zeros(len(frame_times) - 1, dtype=bool)
    with pytest.raises(ValueError, match="keeps no intervals"):
        compute_all_units_tuning_curves(
            analyzer, heading, frame_times, interval_mask=empty
        )


def test_a_full_mask_matches_no_mask(wake_then_sleep):
    analyzer, heading, frame_times, _ = wake_then_sleep
    everything = np.ones(len(frame_times) - 1, dtype=bool)

    a = compute_all_units_tuning_curves(analyzer, heading, frame_times, n_bins=36)
    b = compute_all_units_tuning_curves(
        analyzer, heading, frame_times, n_bins=36, interval_mask=everything
    )
    np.testing.assert_allclose(a["u1"][1], b["u1"][1])
