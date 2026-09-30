"""Turns, constant heading and turn direction in a heading trace.

The failures guarded here all give plausible-looking counts:

  * reading the decoder's 4-degree jitter as turning, which a bin-to-bin
    velocity does (one grid step is 40 deg/s);
  * reading a jump between two posterior peaks as a fast turn;
  * a turn across north unwrapped the long way round, or split in two;
  * velocity averaged across the edge of two decoded stretches;
  * clockwise and counterclockwise swapped by the heading's sign convention;
  * a slow steady spin read as a still head, which only the net drift shows.
"""

import numpy as np
import pandas as pd
import pytest

from spikeshpc.turns import (
    Turns,
    clockwise_sign,
    find_turns,
    plot_turn_summary,
    plot_turn_sweep,
    plot_turn_trace,
)

BIN = 0.1  # the decoder's bins


def trace(*pieces, start_deg=10.0):
    """(time, heading) per 100 ms bin, from (seconds, deg/s) pieces, wrapped to [0, 360)."""
    rates = np.concatenate([np.full(int(round(s / BIN)), float(rate)) for s, rate in pieces])
    heading = start_deg + np.concatenate([[0.0], np.cumsum(rates[:-1] * BIN)])
    return np.arange(len(heading)) * BIN, heading % 360.0


def turns_of(time, heading, **kwargs):
    kwargs.setdefault("turn_threshold", 30.0)
    kwargs.setdefault("bin_s", BIN)
    return find_turns(time, heading, **kwargs)


# ── constant heading ─────────────────────────────────────────────────────
def test_the_decoders_grid_jitter_is_not_turning():
    rng = np.random.default_rng(0)
    time = np.arange(600) * BIN
    heading = 100.0 + 4.0 * rng.integers(-1, 2, size=time.size)  # +/- one 4-degree bin
    turns = turns_of(time, heading)
    assert turns.n_turns == 0
    assert turns.constant_fraction == 1.0
    assert turns.analysed_s == pytest.approx(turns.duration_s)


def test_a_higher_threshold_never_shortens_constant_time():
    rng = np.random.default_rng(1)
    time = np.arange(3000) * BIN
    heading = np.cumsum(rng.normal(0.0, 6.0, time.size)) % 360.0
    constant = [turns_of(time, heading, turn_threshold=t).constant_s for t in (10, 20, 40, 80)]
    assert constant == sorted(constant)
    assert constant[0] < constant[-1]


# ── turns ────────────────────────────────────────────────────────────────
def test_one_clockwise_turn():
    time, heading = trace((10, 0), (2, 90), (10, 0))  # 180 degrees, heading growing
    turns = turns_of(time, heading, clockwise=1)
    assert turns.n_turns == 1 and turns.n_clockwise == 1 and turns.n_counterclockwise == 0
    assert turns.turn_rotation_deg[0] == pytest.approx(180.0, abs=1.0)
    assert turns.turn_start_s[0] == pytest.approx(10.0, abs=0.3)
    assert turns.turn_stop_s[0] == pytest.approx(12.0, abs=0.3)
    turning = turns.turn_stop_s[0] - turns.turn_start_s[0] + BIN
    assert turns.constant_s == pytest.approx(turns.duration_s - turning)
    assert turns.clockwise_ratio == np.inf


def test_the_heading_convention_decides_which_way_is_clockwise():
    time, heading = trace((10, 0), (2, 90), (10, 0))
    turns = turns_of(time, heading, clockwise=-1)  # heading grows counterclockwise
    assert turns.n_counterclockwise == 1 and turns.n_clockwise == 0
    assert turns.turn_rotation_deg[0] == pytest.approx(-180.0, abs=1.0)
    assert np.nanmax(turns.velocity_deg_s) <= 0.0
    assert turns.clockwise_ratio == 0.0


def test_clockwise_sign_follows_the_tuning_notebooks_flip():
    # compute_heading is counterclockwise seen from above; flip_direction reverses it
    assert clockwise_sign(True) == 1
    assert clockwise_sign(False) == -1


def test_a_turn_across_north_is_one_turn_the_short_way_round():
    time, heading = trace((5, 0), (1.2, 100), (5, 0), start_deg=300.0)  # 300 -> 60
    turns = turns_of(time, heading)
    assert turns.n_turns == 1 and turns.n_jumps == 0
    assert turns.turn_rotation_deg[0] == pytest.approx(120.0, abs=1.0)


def test_turns_both_ways_and_their_ratio():
    time, heading = trace((5, 0), (1, 90), (5, 0), (1, 90), (5, 0), (1, -90), (5, 0),
                          (1, 90), (5, 0))
    turns = turns_of(time, heading)
    assert list(turns.turn_direction) == [1, 1, -1, 1]
    assert turns.clockwise_ratio == 3.0
    assert turns.turns_per_min == pytest.approx(4 / (turns.analysed_s / 60.0))
    row = turns.as_row()
    assert (row["n_turns"], row["n_cw"], row["n_ccw"]) == (4, 3, 1)
    assert row["median_turn_deg"] == pytest.approx(90.0, abs=1.0)
    assert row["net_rotation_deg"] == pytest.approx(180.0)  # 90 + 90 - 90 + 90
    assert row["drift_deg_per_min"] == pytest.approx(60.0 * 180.0 / turns.analysed_s)


# ── net drift ────────────────────────────────────────────────────────────
def test_net_drift_catches_a_spin_too_slow_to_count_as_turning():
    time, heading = trace((5, 0), (60, -10), (5, 0))  # 600 degrees counterclockwise
    turns = turns_of(time, heading)  # turning starts at 30 deg/s
    assert turns.n_turns == 0 and turns.constant_fraction == 1.0
    assert turns.net_rotation_deg == pytest.approx(-600.0)
    assert turns.drift_deg_per_min == pytest.approx(60.0 * -600.0 / turns.analysed_s)
    assert turns_of(time, heading, clockwise=-1).net_rotation_deg == pytest.approx(600.0)


def test_turning_back_and_forth_nets_to_no_drift():
    time, heading = trace((5, 0), (1, 90), (5, 0), (1, -90), (5, 0))
    turns = turns_of(time, heading)
    assert turns.n_turns == 2
    assert turns.net_rotation_deg == pytest.approx(0.0, abs=1e-9)


def test_a_slow_drift_is_constant_and_a_fast_one_is_a_turn():
    time, heading = trace((5, 0), (10, 10), (5, 0))  # 100 degrees at 10 deg/s
    assert turns_of(time, heading, turn_threshold=30.0).n_turns == 0
    assert turns_of(time, heading, turn_threshold=5.0).n_turns == 1


# ── jumps ────────────────────────────────────────────────────────────────
def test_a_jump_between_posterior_peaks_is_not_a_turn():
    time, heading = trace((10, 0), (10, 0))
    heading[100:] = 200.0  # hops 190 degrees in one bin, then holds
    turns = turns_of(time, heading)
    assert turns.n_jumps == 1 and turns.n_turns == 0
    assert turns.jump_time_s[0] == pytest.approx(time[100])
    assert turns.constant_fraction == 1.0
    assert turns.net_rotation_deg == 0.0  # nor does it add to the drift
    # with no jump rule the same hop reads as a turn
    assert turns_of(time, heading, max_step_deg=359.0).n_turns == 1


def test_a_flicker_between_two_jumps_is_left_out():
    time, heading = trace((20, 0), start_deg=20.0)
    heading[100:102] = 200.0  # two bins on the other peak
    turns = turns_of(time, heading)
    assert turns.n_jumps == 2 and turns.n_turns == 0
    assert np.isnan(turns.velocity_deg_s[100:102]).all()
    assert turns.analysed_s == pytest.approx(turns.duration_s - 2 * BIN)
    assert turns.net_rotation_deg == 0.0


def test_a_missing_heading_splits_the_stretch():
    time, heading = trace((20, 0))
    heading[100] = np.nan
    turns = turns_of(time, heading)
    assert turns.n_turns == 0 and turns.n_jumps == 0
    assert np.isnan(turns.velocity_deg_s[100])
    assert turns.analysed_s == pytest.approx(turns.duration_s - BIN)


# ── stretches ────────────────────────────────────────────────────────────
def test_velocity_is_never_taken_across_two_stretches():
    time, heading = trace((10, 0), (10, 0))
    heading[100:] = 50.0  # 40 degrees apart: not a jump, and not a turn either
    run_index = np.repeat([0, 1], 100)
    turns = turns_of(time, heading, run_index=run_index)
    assert turns.n_turns == 0 and turns.n_jumps == 0
    # the same trace read as one stretch turns at the join
    assert turns_of(time, heading).n_turns == 1


def test_a_time_gap_starts_a_new_stretch():
    time, heading = trace((10, 0), (10, 0))
    heading[100:] = 50.0
    time[100:] += 30.0  # the decoder skipped half a minute
    turns = find_turns(time, heading, turn_threshold=30.0)  # bin_s inferred from time
    assert turns.bin_s == pytest.approx(BIN)
    assert turns.n_turns == 0 and turns.duration_s == pytest.approx(200 * BIN)


# ── checks ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(clockwise=0), "clockwise"),
        (dict(turn_threshold=0.0), "turn_threshold"),
        (dict(smooth_s=-1.0), "smooth_s"),
        (dict(max_step_deg=0.0), "max_step_deg"),
    ],
)
def test_bad_settings_are_refused(kwargs, message):
    time, heading = trace((5, 0))
    with pytest.raises(ValueError, match=message):
        turns_of(time, heading, **kwargs)


def test_mismatched_lengths_are_refused():
    time, heading = trace((5, 0))
    with pytest.raises(ValueError, match="entries"):
        turns_of(time, heading[:-1])
    with pytest.raises(ValueError, match="run_index"):
        turns_of(time, heading, run_index=np.zeros(3))
    with pytest.raises(ValueError, match="empty"):
        turns_of([], [])


# ── plots ────────────────────────────────────────────────────────────────
def summary_table(thresholds=(30.0,)):
    rows = []
    time, heading = trace((5, 0), (1, 90), (5, 0), (1, -90), (5, 0), (1, 45), (5, 0))
    for threshold in thresholds:
        for recording in ("s1", "s2"):
            for state in ("wake", "REM"):
                sources = ["decoded", "optitrack"] + (["reference"] if recording == "s2" else [])
                for source in sources:
                    row = turns_of(time, heading, turn_threshold=threshold).as_row()
                    rows.append({"recording": recording, "state": state, "source": source, **row})
    return pd.DataFrame(rows)


def test_plot_turn_summary_draws_wake_then_rem_per_recording():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = plot_turn_summary(summary_table(), ["s1", "s2"], {"s1": "k", "s2": "r"},
                             units={"s1": 27, "s2": 20}, types={"s1": "baseline", "s2": "post"})
    assert len(axes) == 4
    constant_ax, rate_ax, ratio_ax, drift_ax = axes
    assert ratio_ax.get_yscale() == "log"
    assert [t.get_text() for t in drift_ax.get_xticklabels()] == ["wake", "REM", "wake", "REM"]
    minor = [t.get_text() for t in drift_ax.get_xticklabels(minor=True)]
    assert minor == ["s1 (baseline), 27 units", "s2 (post), 20 units"]
    # OptiTrack is drawn for wake only: s1 wake has decoded + optitrack, s1 REM decoded alone
    drawn = [line for line in drift_ax.get_lines() if line.get_marker() in ("o", "s", "D")]
    assert len(drawn) == 2 + 1 + 3 + 2  # s1 wake, s1 REM, s2 wake, s2 REM
    low, high = drift_ax.get_ylim()
    assert low == pytest.approx(-high)  # symmetric about no drift
    plt.close(axes[0].figure)


def test_the_wake_pair_sits_together_and_rem_is_shaded():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = plot_turn_summary(summary_table(), ["s1", "s2"], {"s1": "k", "s2": "r"})
    try:
        rate_ax = axes[1]
        # s2's wake column, at x = 2: the decode, the measured heading and the diamond
        x = {line.get_marker(): line.get_xdata()[0] for line in rate_ax.get_lines()
             if line.get_marker() in ("o", "s", "D") and 1.5 < line.get_xdata()[0] < 2.5}
        pair = abs(x["o"] - x["s"])
        assert pair < abs(x["o"] - x["D"]) and pair < abs(x["s"] - x["D"])
        for ax in axes:
            shaded = sorted((p.get_x(), p.get_x() + p.get_width()) for p in ax.patches)
            assert shaded == [(0.5, 1.5), (2.5, 3.5)]  # behind each REM column
        # the pair's counts on either side of their markers, where they cannot collide
        offsets = sorted(t.xyann[1] for t in rate_ax.texts if 1.5 < t.xy[0] < 2.5)
        assert offsets == [-7, 7]
    finally:
        plt.close(axes[0].figure)


@pytest.mark.parametrize("measured_rate, decoded_label, measured_label", [
    (1.0, 7, -7),  # the measured heading turned less: its count goes below
    (99.0, -7, 7),  # turned more: its count goes above, the decode's below
])
def test_a_pairs_counts_point_away_from_each_other(measured_rate, decoded_label, measured_label):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = summary_table()
    measured = (table["recording"] == "s1") & (table["state"] == "wake") & (table["source"] == "optitrack")
    table.loc[measured, "turns_per_min"] = measured_rate
    axes = plot_turn_summary(table, ["s1", "s2"], {"s1": "k", "s2": "r"})
    try:
        offset = {round(t.xy[0], 2): t.xyann[1] for t in axes[1].texts if t.xy[0] < 0.5}
        assert offset == {-0.06: decoded_label, 0.1: measured_label}  # s1 wake: decode, measured
    finally:
        plt.close(axes[0].figure)


def test_plot_turn_sweep_draws_each_measure_against_the_threshold():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = summary_table(thresholds=(20.0, 40.0, 80.0))
    axes = plot_turn_sweep(table, ["s1", "s2"], {"s1": "k", "s2": "r"}, current=40.0,
                           min_turns=3)
    assert axes.shape == (3, 2)
    wake_lines = [line for line in axes[1, 0].get_lines() if line.get_marker() != "None"]
    rem_lines = [line for line in axes[1, 1].get_lines() if line.get_marker() != "None"]
    assert len(wake_lines) == 5  # s1 decoded + optitrack, s2 decoded + optitrack + reference
    assert len(rem_lines) == 3  # no OptiTrack line in REM
    assert list(wake_lines[0].get_xdata()) == [20.0, 40.0, 80.0]
    # 3 turns at 20 and 40 deg/s; at 80 the 45 deg/s one no longer counts, and 2 < min_turns
    s1_wake_ratio = axes[2, 0].get_lines()[0].get_ydata()
    assert np.isfinite(s1_wake_ratio[:2]).all() and np.isnan(s1_wake_ratio[2])
    plt.close(axes[0, 0].figure)


def test_plot_turn_trace_shades_the_turns():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time, heading = trace((5, 0), (1, 90), (5, 0), (1, -90), (5, 0))
    turns = turns_of(time, heading, label="decoded")
    axes = plot_turn_trace(time, heading, turns, measured_deg=heading, length_s=20.0)
    assert len(axes) == 2
    assert len(axes[0].patches) == 2  # one clockwise, one counterclockwise
    plt.close(axes[0].figure)
    with pytest.raises(ValueError, match="no bin"):
        plot_turn_trace(time, heading, turns, start_s=1e6)


def test_turns_repr_and_row():
    time, heading = trace((5, 0), (1, 90), (5, 0))
    turns = turns_of(time, heading, label="s1 wake decoded")
    assert isinstance(turns, Turns)
    assert "s1 wake decoded" in repr(turns) and "1 turns" in repr(turns)
    row = turns.as_row()
    assert row["turn_threshold"] == 30.0 and row["constant_pct"] <= 100.0
