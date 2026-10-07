"""The steps of notebook 3_tracking that need no recording: spikeshpc.tracking's bookkeeping.

Following units across recordings fails quietly when one step hands the next
the wrong recordings or units: a baseline taken for another recording, a
partner chosen by the wrong rule, or a decoder trained on units one recording
lacks. These run the steps on a few made-up units.
"""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spikeshpc.optitrack import HDTuning
from spikeshpc.tracking import (
    check_settings,
    decoder_units,
    match_counts,
    plot_matched_tuning,
    probe_drift,
    return_matches,
    ring_band,
    save_figures,
    unitmatch_groups,
)

BINS = np.arange(360) + 0.5
REC_TYPE = {"session1": "baseline", "session2": "pre", "session3": "post"}


def von_mises(preferred_deg):
    return 20.0 * np.exp(4.0 * (np.cos(np.deg2rad(BINS - preferred_deg)) - 1.0))


def tuning_of(name, preferred, tuned=(), depths=None):
    """An HDTuning with a von Mises curve per unit; `tuned` units are significant."""
    units = sorted(preferred)
    return HDTuning(
        session=name,
        heading_deg=np.zeros(10),
        bin_centers_deg=BINS,
        unit_ids=np.array(units),
        curves=np.array([von_mises(preferred[u]) for u in units]),
        depths=np.array([(depths or {}).get(u, 0.0) for u in units], dtype=float),
        stats={
            u: SimpleNamespace(
                significant=u in tuned, preferred_direction_deg=float(preferred[u])
            )
            for u in units
        },
    )


# ── settings ─────────────────────────────────────────────────────────────────


def test_check_settings_splits_off_the_baseline():
    baseline, others, rec_type = check_settings(
        ["s1", "s2", "s3"], ["pre", "baseline", "post"], "tuned", "joint"
    )
    assert baseline == "s2"
    assert others == ["s1", "s3"]
    assert list(rec_type) == ["s1", "s2", "s3"]


@pytest.mark.parametrize(
    "types, units, mode, message",
    [
        (["baseline", "baseline"], "tuned", "joint", "exactly one"),
        (["baseline", "later"], "tuned", "joint", "unknown recording type"),
        (["baseline", "post"], "some", "joint", "baseline_units"),
        (["baseline", "post"], "tuned", "both", "unitmatch_mode"),
    ],
)
def test_check_settings_refuses_bad_settings(types, units, mode, message):
    with pytest.raises(ValueError, match=message):
        check_settings(["s1", "s2"], types, units, mode)


def test_check_settings_warns_without_a_post_lesion_recording():
    with pytest.warns(UserWarning, match="no post-lesion"):
        check_settings(["s1", "s2"], ["baseline", "pre"], "tuned", "joint")


def test_unitmatch_groups():
    assert unitmatch_groups(REC_TYPE, "joint") == {"joint": list(REC_TYPE)}
    assert unitmatch_groups(REC_TYPE, "pairwise") == {
        "session1_vs_session2": ["session1", "session2"],
        "session1_vs_session3": ["session1", "session3"],
    }


# ── matches ──────────────────────────────────────────────────────────────────

# One joint run: the baseline's units 10-12, session2's 0-2, session3's 5-6
UNITS = [10, 11, 12, 0, 1, 2, 5, 6]
SESSIONS = [0, 0, 0, 1, 1, 1, 2, 2]


def joint_run():
    prob = np.zeros((len(UNITS), len(UNITS)))
    row = {(s, u): i for i, (s, u) in enumerate(zip(SESSIONS, UNITS))}

    def p(a, b, value):  # P[a's row, b's column]
        prob[row[a], row[b]] = value

    p((0, 10), (1, 1), 0.9)  # in one direction only, still a candidate
    p((0, 11), (1, 0), 0.8)  # 11 has two candidates in session2: by P, unit 0 ...
    p((1, 0), (0, 11), 0.7)
    p((0, 11), (1, 2), 0.6)  # ... by tuning, unit 2
    p((0, 10), (2, 6), 0.9)  # 10 and 11 both want session3's unit 6; 11 is likelier
    p((0, 11), (2, 6), 0.95)
    return {
        "names": list(REC_TYPE),
        "prob": prob,
        "clus_info": {
            "original_ids": np.array(UNITS),
            "session_id": np.array(SESSIONS),
        },
    }


def matching_inputs():
    tuning = {
        "session1": tuning_of("session1", {10: 0, 11: 90, 12: 180}, tuned=(10, 11)),
        "session2": tuning_of("session2", {0: 270, 1: 0, 2: 90}, tuned=(0, 1, 2)),
        "session3": tuning_of("session3", {5: 0, 6: 90}, tuned=(5, 6)),
    }
    units_of = {"session1": np.array([10, 11, 12])}
    labels = pd.Series({10: "GOOD", 11: "GOOD", 12: "MUA"})
    return tuning, units_of, labels


@pytest.mark.filterwarnings("ignore:only 1 unambiguous")
def test_return_matches_follows_each_recordings_rule():
    tuning, units_of, labels = matching_inputs()
    candidates, matched, wide, partner, rotations = return_matches(
        {"joint": joint_run()},
        "joint",
        REC_TYPE,
        tuning,
        units_of,
        tuned_base=[10, 11],
        labels_base=labels,
    )
    assert partner == {
        ("session2", 10): 1,
        ("session2", 11): 2,  # pre: the tuning curve decides, not P
        ("session3", 11): 6,  # post: P decides, and 10 has nothing to fall back on
    }
    assert set(rotations) == {"session2"}
    assert wide.loc[11, "session2"] == 2 and pd.isna(wide.loc[10, "session3"])
    assert wide["tuned"].tolist() == [True, True, False]
    assert wide["label"].tolist() == ["GOOD", "GOOD", "MUA"]

    counts = match_counts(candidates, rotations, REC_TYPE, 3, tuned_base=[10, 11])
    assert counts.loc["session2", "matched"] == 2
    assert counts.loc["session3", "contested partners"] == 1
    assert counts.loc["session3", "lost their partner"] == 1


@pytest.mark.filterwarnings("ignore:only 1 unambiguous")
def test_plot_matched_tuning_draws_the_units_found_everywhere():
    tuning, units_of, labels = matching_inputs()
    _, _, _, partner, rotations = return_matches(
        {"joint": joint_run()}, "joint", REC_TYPE, tuning, units_of, [10, 11], labels
    )
    figures = plot_matched_tuning(partner, tuning, REC_TYPE, [10, 11], rotations, "joint")
    assert len(figures) == 1
    titles = [ax.get_title() for ax in figures[0].axes if ax.get_visible()]
    assert titles == ["11 (tuned) | 2:2 3:6"]
    plt.close("all")


# ── the decoder's units ──────────────────────────────────────────────────────

PARTNER = {
    ("session2", 10): 1,
    ("session2", 11): 0,
    ("session3", 11): 6,
    ("session3", 12): 5,
}


def test_joint_decoder_units_are_the_ones_found_everywhere():
    sets, train = decoder_units(PARTNER, [10, 11, 12], REC_TYPE, "joint", 1)
    assert train == [11]
    assert sets == {"session2": [11], "session3": [11]}
    with pytest.raises(ValueError, match="only 1 tuned baseline units"):
        decoder_units(PARTNER, [10, 11, 12], REC_TYPE, "joint", 2)


def test_pairwise_decoder_units_leave_out_a_recording_with_too_few():
    partner = {**PARTNER, ("session2", 12): 2}
    with pytest.warns(UserWarning, match="session3: only 2"):
        sets, train = decoder_units(partner, [10, 11, 12], REC_TYPE, "pairwise", 3)
    assert train == [10, 11, 12]
    assert sets == {"session2": [10, 11, 12]}


# ── rings ────────────────────────────────────────────────────────────────────


def test_probe_drift_is_the_median_depth_change_of_the_matched_units():
    tuning = {
        "session1": tuning_of(
            "s1", {10: 0, 11: 0, 12: 0}, depths={10: 100, 11: 200, 12: 300}
        ),
        "session2": tuning_of("s2", {0: 0, 1: 0}, depths={0: 210, 1: 130}),
        "session3": tuning_of("s3", {7: 0}, depths={7: 0}),
    }
    with pytest.warns(UserWarning, match="session3: no matched unit"):
        drift = probe_drift(PARTNER, tuning, REC_TYPE)
    assert drift == {"session1": 0.0, "session2": 20.0, "session3": 0.0}


def test_ring_band_widens_the_tuned_units_depths():
    depths = {u: 100.0 * u for u in range(1, 11)}
    hd = tuning_of("s1", {u: 0 for u in depths}, tuned=range(1, 11), depths=depths)
    low, high = ring_band(hd)
    assert low == pytest.approx(np.percentile(list(depths.values()), 10) - 100)
    assert high == pytest.approx(np.percentile(list(depths.values()), 90) + 100)
    assert ring_band(hd, (0, 50)) == (0, 50)


def test_save_figures_writes_one_pdf_and_a_png_per_page(tmp_path):
    figures = [plt.figure() for _ in range(2)]
    save_figures(figures, tmp_path / "pages.pdf", png_dpi=50)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "pages.pdf",
        "pages_1.png",
        "pages_2.png",
    ]
    plt.close("all")
