"""Interactive scroll-through-units view of head-direction tuning curves."""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from ._backend import warn_if_noninteractive_backend


class HDTuningCurveWidget:
    """One unit's firing-rate-vs-heading curve at a time; Left/Right to switch units.

    Pass ``stats`` and each curve is drawn over the shuffled distribution it
    was tested against: the shaded band is where a chance curve for this unit
    lives, given its own spike count and the animal's own occupancy. A p-value
    says how far outside that band the real curve fell; seeing the band says
    what "outside" amounted to. For a sparse unit the two can be very
    different impressions, which is the point.

    The band is pointwise, not simultaneous -- across 36 bins, one excursion
    above the 97.5th percentile is unremarkable. The verdict in the title is
    still the MVL shuffle test.

    Requires an interactive matplotlib backend (``%matplotlib widget`` or
    ``%matplotlib qt`` in Jupyter) and the figure to have keyboard focus
    (click on it once) before arrow keys will do anything.
    """

    def __init__(
        self,
        tuning_curves: dict,
        unit_depths: dict | None = None,
        stats: dict | None = None,
        show_null: bool = True,
    ):
        warn_if_noninteractive_backend()
        self.unit_ids = list(tuning_curves.keys())
        self.tuning_curves = tuning_curves
        self.unit_depths = unit_depths or {}
        self.stats = stats or {}
        self.show_null = show_null
        self.index = 0

        self.fig, self.ax = plt.subplots()
        self._null_artists = []
        (self.line,) = self.ax.plot([], [], color="#1f77b4", lw=1.8, zorder=3)
        self.ax.set_xlabel("Heading (degrees)")
        self.ax.set_ylabel("Firing rate (spikes/s)")
        self.ax.set_xlim(0, 360)
        self.ax.set_xticks(np.arange(0, 361, 90))
        self.fig.text(
            0.99, 0.01, "← → to switch units", ha="right", fontsize=8, color="gray"
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    def _draw_null(self, unit_id) -> float:
        """Shade the shuffled distribution. Returns its highest value, or 0."""
        for artist in self._null_artists:
            artist.remove()
        self._null_artists = []

        stat = self.stats.get(unit_id)
        band = getattr(stat, "null_band", None)
        if not self.show_null or band is None:
            return 0.0

        band = np.asarray(band, dtype=float)
        centers = np.asarray(stat.null_bin_centers_deg, dtype=float)
        percentiles = list(stat.null_percentiles)

        # wrap one bin round on each side so the shading closes at 0/360
        # rather than stopping short of the seam
        x = np.r_[centers[-1] - 360.0, centers, centers[0] + 360.0]
        band = np.column_stack([band[:, -1], band, band[:, 0]])

        if len(percentiles) >= 2:
            self._null_artists.append(
                self.ax.fill_between(
                    x, band[0], band[-1], color="#bbbbbb", alpha=0.55,
                    lw=0, zorder=1,
                    label=f"shuffled {percentiles[0]:g}-{percentiles[-1]:g}%",
                )
            )
        median = len(percentiles) // 2
        if len(percentiles) % 2 == 1:
            self._null_artists.extend(
                self.ax.plot(
                    x, band[median], color="#777777", lw=1.0, ls="--", zorder=2,
                    label="shuffled median",
                )
            )
        return float(band.max())

    def _draw(self):
        unit_id = self.unit_ids[self.index]
        bin_centers, rate = self.tuning_curves[unit_id]
        self.line.set_data(bin_centers, rate)

        null_max = self._draw_null(unit_id)
        top = max(float(np.max(rate)), null_max) * 1.1
        self.ax.set_ylim(0, max(top, 1.0))

        title = f"unit {unit_id} ({self.index + 1}/{len(self.unit_ids)})"
        depth = self.unit_depths.get(unit_id)
        if depth is not None:
            title += f", depth {depth:.0f} µm"

        stat = self.stats.get(unit_id)
        if stat is not None:
            verdict = "tuned" if stat.significant else "not tuned"
            if getattr(stat, "too_quiet", False):
                verdict = "too quiet"
            title += (
                f"\nMVL {stat.mean_vector_length:.3f} "
                f"(chance {stat.mvl_threshold:.3f}), p = {stat.p_value:.4f}, "
                f"peak {stat.peak_rate_hz:.1f} Hz — {verdict}"
            )
        self.ax.set_title(title, fontsize=10)

        if self._null_artists:
            self.ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        elif self.ax.get_legend() is not None:
            self.ax.get_legend().remove()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "right":
            self.index = (self.index + 1) % len(self.unit_ids)
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.unit_ids)
        else:
            return
        self._draw()


def show_hd_tuning_widget(
    tuning_curves: dict,
    unit_depths: dict | None = None,
    stats: dict | None = None,
    show_null: bool = True,
) -> HDTuningCurveWidget:
    """Open an interactive figure; Left/Right arrow keys step through units.

    ``tuning_curves`` is the output of
    :func:`optitrack.tuning.compute_all_units_tuning_curves`. ``unit_depths``
    (unit_id -> probe depth) is the output of
    :func:`optitrack.tuning.get_unit_depths`; if given, each unit's depth is
    shown alongside its ID in the plot title.

    ``stats`` is the output of
    :func:`optitrack.tuning.compute_hd_tuning_significance`. Given it, each
    curve is drawn over the shuffled band it was tested against and the title
    carries the verdict. Note that the band comes from the significance test's
    binning, which is coarser than the plotted curve's by default -- both are
    smoothed in degrees, so they are comparable, but the band will look
    blockier.
    """
    return HDTuningCurveWidget(
        tuning_curves, unit_depths=unit_depths, stats=stats, show_null=show_null
    )
