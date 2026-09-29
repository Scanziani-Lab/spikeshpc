"""Following a baseline recording's units into other recordings of the same probe.

UnitMatch scores every pair of units across two recordings with a match
probability from their waveforms. This module turns that matrix into one
partner per baseline unit per recording, and draws what the tracked units'
head-direction tuning did from one recording to the next. It deliberately does
not import UnitMatchPy: it takes UnitMatch's arrays, so it runs wherever
spikeshpc does.

Three things decide a partner:

  * A candidate is any pair above the threshold in *either* direction.
    ``P[a, b]`` and ``P[b, a]`` come from opposite cross-validation halves of
    each unit's spikes and often disagree -- routinely one near 1 and the other
    near 0 for the same pair.
  * Several candidates for one baseline unit are split by a rule that depends
    on the recording. Before the lesion, by how alike the two tuning curves are
    once the population's common rotation is taken out (``rule="tuning"``).
    After it, by average probability alone (``rule="probability"``), since a
    lesion that may disrupt tuning must not decide which unit is which.
  * One partner per baseline unit and one baseline unit per partner, found
    greedily: the best-ranked pair takes both units, and a baseline unit that
    lost its partner falls back to its next candidate, if it has one.

Head-direction cells turn together. Between recordings the whole population's
preferred directions can shift by one common angle -- +49 degrees between
session7 and session9 -- so tuning is compared after undoing that angle,
measured from the pairs that need no tie-break at all.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .decoder import circular_difference

__all__ = [
    "estimate_rotation",
    "plot_tracked_tuning",
    "putative_matches",
    "recording_colors",
    "select_partners",
    "tuning_similarity",
]

RECORDING_TYPES = ("baseline", "pre", "post")


def putative_matches(prob, unit_ids, session_ids, ref, other, threshold: float = 0.5) -> pd.DataFrame:
    """Every (ref unit, other unit) pair UnitMatch puts above `threshold` in either direction.

    ``prob`` is UnitMatch's ``(n, n)`` match probability, with rows and columns
    in the order of ``unit_ids`` (``clus_info["original_ids"]``) and
    ``session_ids`` (``clus_info["session_id"]``). ``ref`` and ``other`` are
    session ids as ``session_ids`` holds them.

    Columns:
      ref_unit, other_unit   cluster ids in their own recordings
      p_ref_row              P[ref, other]: the ref unit's row
      p_other_row            P[other, ref]: the other unit's row
      p_avg                  their mean
      n_ref_candidates       how many candidates this ref unit has
      n_other_candidates     how many ref units this other unit is a candidate for

    UnitMatch's MatchTable.csv is oriented the other way round: its row with
    ``ID1 = a, ID2 = b`` holds ``P[b, a]``.
    """
    prob = np.asarray(prob, dtype=float)
    unit_ids = np.asarray(unit_ids).ravel()
    session_ids = np.asarray(session_ids).ravel()
    if prob.shape != (len(unit_ids), len(unit_ids)) or len(session_ids) != len(unit_ids):
        raise ValueError(
            f"prob is {prob.shape} but there are {len(unit_ids)} unit ids and "
            f"{len(session_ids)} session ids"
        )
    rows_ref = np.flatnonzero(session_ids == ref)
    rows_other = np.flatnonzero(session_ids == other)
    if not rows_ref.size or not rows_other.size:
        raise ValueError(f"session {ref!r} or {other!r} has no units in this matrix")

    forward = prob[np.ix_(rows_ref, rows_other)]
    backward = prob[np.ix_(rows_other, rows_ref)].T
    i, j = np.nonzero(np.maximum(forward, backward) > threshold)
    table = pd.DataFrame(
        {
            "ref_unit": unit_ids[rows_ref[i]],
            "other_unit": unit_ids[rows_other[j]],
            "p_ref_row": forward[i, j],
            "p_other_row": backward[i, j],
        }
    )
    table["p_avg"] = (table["p_ref_row"] + table["p_other_row"]) / 2
    table["n_ref_candidates"] = table.groupby("ref_unit")["other_unit"].transform("size")
    table["n_other_candidates"] = table.groupby("other_unit")["ref_unit"].transform("size")
    return table.sort_values(["ref_unit", "p_avg"], ascending=[True, False], ignore_index=True)


def estimate_rotation(candidates: pd.DataFrame, hd_ref, hd_other, min_pairs: int = 3):
    """How far the whole head-direction population turned between two recordings.

    Measured on the pairs least likely to be wrong -- a ref unit with exactly
    one candidate that is nobody else's candidate -- where both units are
    significantly tuned, as the circular mean of their preferred directions'
    difference (other - ref). ``hd_ref`` and ``hd_other`` are the two
    recordings' :class:`spikeshpc.optitrack.HDTuning`; a unit missing from one
    is skipped.

    Returns ``(rotation_deg, R, n_pairs)``: the angle in (-180, 180], the
    resultant length of the differences (near 1 when the population turned as
    one), and how many pairs it rests on. With fewer than `min_pairs` there is
    nothing to measure, so the rotation is 0, R is NaN, and a warning says so.
    """
    unique = candidates[
        (candidates["n_ref_candidates"] == 1) & (candidates["n_other_candidates"] == 1)
    ]
    differences = []
    for ref_unit, other_unit in zip(unique["ref_unit"], unique["other_unit"]):
        ref_stats = hd_ref.stats.get(ref_unit)
        other_stats = hd_other.stats.get(other_unit)
        if ref_stats is None or other_stats is None:
            continue
        if not (ref_stats.significant and other_stats.significant):
            continue
        differences.append(
            circular_difference(
                other_stats.preferred_direction_deg, ref_stats.preferred_direction_deg
            )
        )

    n_pairs = len(differences)
    if n_pairs < min_pairs:
        warnings.warn(
            f"only {n_pairs} unambiguous pair(s) with both units tuned (need "
            f"{min_pairs}): assuming the population did not turn",
            stacklevel=2,
        )
        return 0.0, float("nan"), n_pairs
    resultant = np.exp(1j * np.deg2rad(np.asarray(differences, dtype=float))).mean()
    rotation = float(circular_difference(np.rad2deg(np.angle(resultant)), 0.0))
    return rotation, float(np.abs(resultant)), n_pairs


def tuning_similarity(curve_ref, curve_other, rotation_deg: float = 0.0) -> float:
    """Pearson r between two tuning curves, after turning `curve_other` back by `rotation_deg`.

    Both are rates over the same evenly spaced ring of heading bins starting at
    0 degrees -- the curves :func:`spikeshpc.optitrack.load_hd_tuning` returns --
    so the rotation is a circular shift by the nearest whole number of bins.
    NaN if either curve is flat, as a unit that never fired in the tuning
    intervals is.
    """
    ref = np.asarray(curve_ref, dtype=float)
    other = np.asarray(curve_other, dtype=float)
    if ref.shape != other.shape or ref.ndim != 1:
        raise ValueError(f"curves of different shapes: {ref.shape} vs {other.shape}")
    shift = int(round(rotation_deg / (360.0 / len(other))))
    other = np.roll(other, -shift)
    if not (ref.std() > 0 and other.std() > 0):
        return float("nan")
    return float(np.corrcoef(ref, other)[0, 1])


def select_partners(candidates: pd.DataFrame, rule: str, similarity=None) -> pd.DataFrame:
    """One partner per ref unit and one ref unit per partner, from `candidates`.

    ``rule="probability"`` ranks pairs by ``p_avg``. ``rule="tuning"`` ranks
    them by ``similarity``, highest first -- one value per row of
    `candidates`, from :func:`tuning_similarity` -- and puts pairs without one
    after every pair that has one, in ``p_avg`` order. Leave it NaN wherever
    tuning cannot tell candidates apart: a curve missing or flat, or a ref
    unit that is not head-direction tuned, whose curve is noise.

    Walking down the ranking, a pair is accepted when neither of its units is
    already taken. So the better-ranked of two ref units competing for one
    partner keeps it, and the other falls back to its next candidate, if it has
    one -- even a ref unit whose only candidate that partner was.

    Returns `candidates` with columns added:
      similarity   (rule "tuning" only)
      rank         0 for the best-ranked pair
      chosen       the pairs that stand
      reason       "only candidate", "best tuning similarity", "highest
                   average P" or "fallback" (its better candidates went to
                   other ref units) for chosen pairs; "lost to unit N" or "not
                   needed" (its ref unit took a better candidate) for the rest
    """
    table = candidates.reset_index(drop=True).copy()
    p_avg = table["p_avg"].to_numpy(dtype=float)
    ref_units = table["ref_unit"].to_numpy()
    if rule == "tuning":
        if similarity is None:
            raise ValueError('rule="tuning" needs a similarity for every candidate')
        scores = np.asarray(similarity, dtype=float)
        if scores.shape != (len(table),):
            raise ValueError(f"{len(scores)} similarities for {len(table)} candidates")
        table["similarity"] = scores
        scored = np.isfinite(scores)
        # np.lexsort sorts by its last key first: scored pairs, then by
        # similarity, then by p_avg, then by unit so ties are deterministic
        order = np.lexsort((ref_units, -p_avg, -np.where(scored, scores, 0.0), ~scored))
    elif rule == "probability":
        scored = np.zeros(len(table), dtype=bool)
        order = np.lexsort((ref_units, -p_avg))
    else:
        raise ValueError(f"rule must be 'tuning' or 'probability', got {rule!r}")

    rank = np.empty(len(table), dtype=int)
    rank[order] = np.arange(len(table))
    table["rank"] = rank

    owner_of = {}  # partner -> the ref unit that took it
    partner_of = {}  # ref unit -> the row it took
    chosen = np.zeros(len(table), dtype=bool)
    blocked_by = {}  # row -> why it was passed over
    for row in order:
        ref_unit, other_unit = table.at[row, "ref_unit"], table.at[row, "other_unit"]
        if ref_unit in partner_of:
            blocked_by[row] = "not needed"
        elif other_unit in owner_of:
            blocked_by[row] = f"lost to unit {owner_of[other_unit]}"
        else:
            chosen[row] = True
            owner_of[other_unit] = ref_unit
            partner_of[ref_unit] = row
    table["chosen"] = chosen

    n_candidates = table.groupby("ref_unit")["other_unit"].transform("size").to_numpy()
    best_rank = table.groupby("ref_unit")["rank"].transform("min").to_numpy()
    reasons = []
    for row in range(len(table)):
        if not chosen[row]:
            reasons.append(blocked_by[row])
        elif n_candidates[row] == 1:
            reasons.append("only candidate")
        elif rank[row] != best_rank[row]:
            reasons.append("fallback")
        elif scored[row]:
            reasons.append("best tuning similarity")
        else:
            reasons.append("highest average P")
    table["reason"] = reasons
    return table


def recording_colors(types) -> list:
    """A line color per recording from its type: "baseline", "pre" or "post".

    The baseline is black. Other pre-lesion recordings are gray and post-lesion
    ones red, in shades from light to dark in list order -- so with the list in
    chronological order the latest post-lesion recording is the darkest red. A
    single recording of a type gets that type's middle shade.
    """
    import matplotlib

    types = list(types)
    unknown = sorted(set(types) - set(RECORDING_TYPES))
    if unknown:
        raise ValueError(f"unknown recording type(s) {unknown}; use {RECORDING_TYPES}")

    def shades(cmap_name, n, lightest, darkest):
        cmap = matplotlib.colormaps[cmap_name]
        levels = [(lightest + darkest) / 2] if n == 1 else np.linspace(lightest, darkest, n)
        return [cmap(level) for level in levels]

    colors = [None] * len(types)
    for kind, cmap_name, lightest, darkest in (
        ("pre", "Greys", 0.35, 0.65),
        ("post", "Reds", 0.40, 0.95),
    ):
        rows = [i for i, t in enumerate(types) if t == kind]
        for i, color in zip(rows, shades(cmap_name, len(rows), lightest, darkest)):
            colors[i] = color
    for i, t in enumerate(types):
        if t == "baseline":
            colors[i] = (0.0, 0.0, 0.0, 1.0)
    return colors


def plot_tracked_tuning(
    curves: dict,
    recordings,
    colors,
    labels=None,
    titles: dict | None = None,
    ncols: int = 6,
    per_page: int = 36,
    panel_size=(2.4, 1.9),
) -> list:
    """One panel per tracked unit: its tuning curve in every recording it was found in.

    ``curves`` is ``{unit: {recording: (bin_centers_deg, rate_hz)}}``, keyed by
    the baseline unit; a recording in which it had no partner, or its partner
    no curve, is simply absent. ``recordings`` fixes the drawing and legend
    order, ``colors`` gives one color per recording (see
    :func:`recording_colors`) and ``labels`` the legend text (default: the
    recording names). ``titles`` optionally overrides a panel's title.

    Curves are in Hz, each panel scaled to its own unit, so a rate that fell
    after the lesion shows as a lower curve and not just a flatter one. At most
    `per_page` panels go on a figure, each with the legend across its top.

    Returns the list of figures.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    recordings = list(recordings)
    colors = dict(zip(recordings, colors))
    labels = dict(zip(recordings, labels if labels is not None else recordings))
    titles = titles or {}
    units = [u for u in curves if curves[u]]
    if not units:
        raise ValueError("no unit has a curve in any recording")

    handles = [Line2D([], [], color=colors[r], lw=1.6, label=labels[r]) for r in recordings]
    figures = []
    for start in range(0, len(units), per_page):
        page = units[start : start + per_page]
        columns = max(1, min(ncols, len(page)))
        rows = int(np.ceil(len(page) / columns))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(panel_size[0] * columns, panel_size[1] * rows + 0.6),
            squeeze=False,
        )
        for ax, unit in zip(axes.flat, page):
            for recording in recordings:
                if recording not in curves[unit]:
                    continue
                centers, rate = (np.asarray(a, dtype=float) for a in curves[unit][recording])
                ax.plot(centers, rate, color=colors[recording], lw=1.2)
            ax.set_title(titles.get(unit, f"unit {unit}"), fontsize=7)
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 90))
            ax.set_ylim(0, None)
            ax.tick_params(labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes.flat[len(page) :]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("Hz", fontsize=7)
        fig.supxlabel("heading (deg)", fontsize=8)
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7,
                   frameon=False)
        top = 1.0 - 0.45 / (panel_size[1] * rows + 0.6)
        fig.tight_layout(rect=(0, 0, 1, top))
        figures.append(fig)
    return figures
