"""Choosing each baseline unit's partner in another recording from UnitMatch's matrix.

The failures guarded here all produce a plausible-looking match table:

  * reading only one direction of UnitMatch's asymmetric matrix drops pairs
    whose two cross-validation halves disagree -- which is most of them;
  * comparing tuning curves without undoing the population's common rotation
    prefers whichever candidate happens to point where the baseline unit did;
  * letting two baseline units keep the same partner, or letting the loser go
    unmatched when it had another candidate.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spikeshpc.matching import (
    estimate_rotation,
    plot_tracked_tuning,
    putative_matches,
    recording_colors,
    select_partners,
    tuning_similarity,
)

BINS = np.arange(360) + 0.5  # the 1-degree ring hd_tuning curves live on


def von_mises(preferred_deg, kappa=4.0, peak_hz=20.0):
    return peak_hz * np.exp(kappa * (np.cos(np.deg2rad(BINS - preferred_deg)) - 1.0))


def hd(preferred: dict, significant=None):
    """The part of an HDTuning the matching reads: per-unit significance and PD."""
    significant = significant or {}
    return SimpleNamespace(
        stats={
            u: SimpleNamespace(
                preferred_direction_deg=float(pd_deg), significant=significant.get(u, True)
            )
            for u, pd_deg in preferred.items()
        }
    )


def candidates_from(pairs):
    """A candidates table as putative_matches returns it, from (ref, other, p_avg)."""
    table = pd.DataFrame(pairs, columns=["ref_unit", "other_unit", "p_avg"])
    table["p_ref_row"] = table["p_avg"]
    table["p_other_row"] = table["p_avg"]
    table["n_ref_candidates"] = table.groupby("ref_unit")["other_unit"].transform("size")
    table["n_other_candidates"] = table.groupby("other_unit")["ref_unit"].transform("size")
    return table


# ── candidates ───────────────────────────────────────────────────────────
@pytest.fixture
def matrix():
    """Two ref units (10, 11) and three other units (20, 21, 22)."""
    unit_ids = np.array([[10], [11], [20], [21], [22]])  # UnitMatch hands them over (n, 1)
    session_ids = np.array([0, 0, 1, 1, 1])
    prob = np.zeros((5, 5))
    prob[0, 2] = 0.9  # 10 -> 20, forward only
    prob[3, 1] = 0.7  # 21 -> 11, backward only
    prob[0, 4] = prob[4, 0] = 0.5  # exactly at threshold: not a candidate
    prob[1, 4], prob[4, 1] = 0.6, 0.2  # 11 <-> 22, one direction
    return prob, unit_ids, session_ids


def test_a_pair_counts_if_either_direction_passes(matrix):
    table = putative_matches(*matrix, ref=0, other=1, threshold=0.5)
    pairs = set(zip(table["ref_unit"], table["other_unit"]))
    assert pairs == {(10, 20), (11, 21), (11, 22)}

    row = table.set_index(["ref_unit", "other_unit"]).loc[(11, 21)]
    assert row["p_ref_row"] == 0.0 and row["p_other_row"] == pytest.approx(0.7)
    assert row["p_avg"] == pytest.approx(0.35)


def test_candidate_counts_are_per_unit(matrix):
    table = putative_matches(*matrix, ref=0, other=1).set_index(["ref_unit", "other_unit"])
    assert table.loc[(11, 21), "n_ref_candidates"] == 2
    assert table.loc[(10, 20), "n_ref_candidates"] == 1
    assert table.loc[(11, 22), "n_other_candidates"] == 1


def test_a_session_with_no_units_is_refused(matrix):
    with pytest.raises(ValueError, match="has no units"):
        putative_matches(*matrix, ref=0, other=7)


def test_a_mismatched_matrix_is_refused(matrix):
    prob, unit_ids, session_ids = matrix
    with pytest.raises(ValueError, match="unit ids"):
        putative_matches(prob[:4, :4], unit_ids, session_ids, ref=0, other=1)


# ── rotation ─────────────────────────────────────────────────────────────
def test_the_population_rotation_is_read_off_the_unambiguous_pairs():
    rng = np.random.default_rng(0)
    ref_pd = {u: pd_deg for u, pd_deg in zip(range(8), np.linspace(0, 315, 8))}
    # every unambiguous pair turned +50 deg, through north for some of them
    other_pd = {100 + u: (p + 50 + rng.normal(0, 4)) % 360 for u, p in ref_pd.items()}
    pairs = [(u, 100 + u, 0.9) for u in ref_pd]
    # an ambiguous ref unit whose candidates point anywhere must not count
    ref_pd[50] = 10.0
    other_pd.update({150: 250.0, 151: 200.0})
    pairs += [(50, 150, 0.8), (50, 151, 0.7)]
    # nor a pair where one side is untuned
    ref_pd[60], other_pd[160] = 90.0, 300.0
    pairs.append((60, 160, 0.9))

    rotation, resultant, n_pairs = estimate_rotation(
        candidates_from(pairs), hd(ref_pd), hd(other_pd, significant={160: False})
    )
    assert rotation == pytest.approx(50.0, abs=3.0)
    assert resultant > 0.95
    assert n_pairs == 8


def test_too_few_pairs_means_no_rotation_and_a_warning():
    pairs = candidates_from([(1, 101, 0.9), (2, 102, 0.9)])
    with pytest.warns(UserWarning, match="assuming the population did not turn"):
        rotation, resultant, n_pairs = estimate_rotation(
            pairs, hd({1: 0.0, 2: 90.0}), hd({101: 40.0, 102: 130.0}), min_pairs=3
        )
    assert rotation == 0.0 and np.isnan(resultant) and n_pairs == 2


def test_a_rotation_is_signed_other_minus_ref():
    pairs = candidates_from([(u, 100 + u, 0.9) for u in range(4)])
    ref_pd = {u: 90.0 * u for u in range(4)}
    other_pd = {100 + u: (90.0 * u - 30.0) % 360 for u in range(4)}
    rotation, _, _ = estimate_rotation(pairs, hd(ref_pd), hd(other_pd))
    assert rotation == pytest.approx(-30.0)


# ── tuning similarity ────────────────────────────────────────────────────
def test_similarity_undoes_the_rotation():
    ref, turned = von_mises(100.0), von_mises(150.0)
    assert tuning_similarity(ref, turned, rotation_deg=50.0) == pytest.approx(1.0)
    assert tuning_similarity(ref, turned) < 0.5


def test_a_flat_curve_has_no_similarity():
    assert np.isnan(tuning_similarity(von_mises(10.0), np.zeros(360)))


def test_curves_on_different_rings_are_refused():
    with pytest.raises(ValueError, match="different shapes"):
        tuning_similarity(np.ones(360), np.ones(72))


# ── choosing partners ────────────────────────────────────────────────────
def test_the_rotated_true_partner_beats_a_distractor_pointing_the_old_way():
    """Before the lesion the whole population turned +50 deg.

    Unit 1's true partner (11) points at 150 deg now; the distractor (12) still
    points where unit 1 used to, at 100, but is broader. Plain correlation
    prefers the distractor; correlation after the rotation does not.
    """
    table = candidates_from([(1, 11, 0.6), (1, 12, 0.8)])
    curves = {1: von_mises(100.0), 11: von_mises(150.0), 12: von_mises(100.0, kappa=1.0)}

    def pick(rotation):
        similarity = [
            tuning_similarity(curves[a], curves[b], rotation)
            for a, b in zip(table["ref_unit"], table["other_unit"])
        ]
        chosen = select_partners(table, "tuning", similarity)
        return chosen.loc[chosen["chosen"], ["other_unit", "reason"]].iloc[0].tolist()

    assert pick(50.0) == [11, "best tuning similarity"]
    assert pick(0.0)[0] == 12  # what ignoring the rotation would have done


def test_after_the_lesion_the_highest_average_p_wins():
    table = candidates_from([(1, 11, 0.6), (1, 12, 0.8)])
    chosen = select_partners(table, "probability")
    row = chosen[chosen["chosen"]].iloc[0]
    assert row["other_unit"] == 12 and row["reason"] == "highest average P"


def test_a_contested_partner_goes_to_the_better_pair_and_the_loser_falls_back():
    # 2 wants 11 more than 1 does; 1 still has 12
    table = candidates_from([(1, 11, 0.9), (1, 12, 0.6), (2, 11, 0.95)])
    chosen = select_partners(table, "probability").set_index(["ref_unit", "other_unit"])

    assert chosen.loc[(2, 11), "chosen"] and chosen.loc[(2, 11), "reason"] == "only candidate"
    assert chosen.loc[(1, 12), "chosen"] and chosen.loc[(1, 12), "reason"] == "fallback"
    assert not chosen.loc[(1, 11), "chosen"]
    assert chosen.loc[(1, 11), "reason"] == "lost to unit 2"


def test_no_partner_is_used_twice_and_no_ref_unit_gets_two():
    rng = np.random.default_rng(3)
    pairs = [(a, b, rng.uniform(0.5, 1.0)) for a in range(8) for b in range(20, 26)
             if rng.uniform() < 0.5]
    chosen = select_partners(candidates_from(pairs), "probability")
    kept = chosen[chosen["chosen"]]
    assert kept["ref_unit"].is_unique and kept["other_unit"].is_unique
    # and every pass-over is explained
    assert set(chosen.loc[~chosen["chosen"], "reason"].str.split().str[0]) <= {"lost", "not"}


def test_a_better_candidate_leaves_the_rest_not_needed():
    chosen = select_partners(candidates_from([(1, 11, 0.9), (1, 12, 0.7)]), "probability")
    assert chosen.set_index("other_unit").loc[12, "reason"] == "not needed"


def test_pairs_without_a_curve_rank_after_every_pair_with_one():
    table = candidates_from([(1, 11, 0.95), (1, 12, 0.55)])
    chosen = select_partners(table, "tuning", similarity=[np.nan, 0.1])
    assert chosen.loc[chosen["chosen"], "other_unit"].tolist() == [12]


def test_a_unit_with_no_curve_at_all_is_matched_on_p():
    table = candidates_from([(1, 11, 0.6), (1, 12, 0.8)])
    chosen = select_partners(table, "tuning", similarity=[np.nan, np.nan])
    row = chosen[chosen["chosen"]].iloc[0]
    assert row["other_unit"] == 12 and row["reason"] == "highest average P"


def test_bad_rules_are_refused():
    table = candidates_from([(1, 11, 0.6)])
    with pytest.raises(ValueError, match="needs a similarity"):
        select_partners(table, "tuning")
    with pytest.raises(ValueError, match="rule must be"):
        select_partners(table, "vibes")
    with pytest.raises(ValueError, match="similarities for"):
        select_partners(table, "tuning", similarity=[0.1, 0.2])


# ── drawing ──────────────────────────────────────────────────────────────
def test_colors_follow_the_recording_types():
    colors = recording_colors(["pre", "baseline", "post", "post"])
    assert colors[1] == (0.0, 0.0, 0.0, 1.0)
    r, g, b, _ = colors[0]
    assert r == pytest.approx(g) == pytest.approx(b)  # a gray
    early, late = colors[2], colors[3]
    assert early[0] > early[1] and late[0] > late[1]  # reds
    assert sum(late[:3]) < sum(early[:3])  # the later post-lesion one is darker


def test_an_unknown_recording_type_is_refused():
    with pytest.raises(ValueError, match="unknown recording type"):
        recording_colors(["baseline", "lesion"])


def test_tracked_tuning_draws_units_missing_from_some_recordings():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recordings = ["s7", "s9", "s10"]
    curves = {
        u: {r: (BINS, von_mises(40.0 * u + 10 * k)) for k, r in enumerate(recordings)
            if not (u == 2 and r == "s9")}
        for u in range(5)
    }
    figures = plot_tracked_tuning(
        curves, recordings, recording_colors(["baseline", "pre", "post"]),
        titles={2: "unit 2 (no s9)"}, per_page=2,
    )
    assert len(figures) == 3  # 5 units, 2 per page
    first = figures[0]
    assert [t.get_text() for t in first.legends[0].get_texts()] == recordings
    second_page_axes = [ax for ax in figures[1].axes if ax.get_visible()]
    assert second_page_axes[0].get_title() == "unit 2 (no s9)"
    assert len(second_page_axes[0].lines) == 2
    for fig in figures:
        plt.close(fig)
