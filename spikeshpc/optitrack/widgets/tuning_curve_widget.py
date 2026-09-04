"""Interactive scroll-through-units view of head-direction tuning curves."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ._backend import warn_if_noninteractive_backend


class HDTuningCurveWidget:
    """One unit's firing-rate-vs-heading curve at a time; Left/Right to switch units.

    Requires an interactive matplotlib backend (``%matplotlib widget`` or
    ``%matplotlib qt`` in Jupyter) and the figure to have keyboard focus
    (click on it once) before arrow keys will do anything.
    """

    def __init__(self, tuning_curves: dict, unit_depths: dict | None = None):
        warn_if_noninteractive_backend()
        self.unit_ids = list(tuning_curves.keys())
        self.tuning_curves = tuning_curves
        self.unit_depths = unit_depths or {}
        self.index = 0

        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [])
        self.ax.set_xlabel("Heading (degrees)")
        self.ax.set_ylabel("Firing rate (spikes/s)")
        self.ax.set_xlim(0, 360)
        self.fig.text(0.99, 0.01, "← → to switch units", ha="right", fontsize=8, color="gray")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    def _draw(self):
        unit_id = self.unit_ids[self.index]
        bin_centers, rate = self.tuning_curves[unit_id]
        self.line.set_data(bin_centers, rate)
        self.ax.set_ylim(0, max(float(rate.max()) * 1.1, 1.0))
        title = f"unit {unit_id} ({self.index + 1}/{len(self.unit_ids)})"
        depth = self.unit_depths.get(unit_id)
        if depth is not None:
            title += f", depth {depth:.0f} µm"
        self.ax.set_title(title)
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
    tuning_curves: dict, unit_depths: dict | None = None
) -> HDTuningCurveWidget:
    """Open an interactive figure; Left/Right arrow keys step through units.

    ``tuning_curves`` is the output of
    :func:`optitrack.tuning.compute_all_units_tuning_curves`. ``unit_depths``
    (unit_id -> probe depth) is the output of
    :func:`optitrack.tuning.get_unit_depths`; if given, each unit's depth is
    shown alongside its ID in the plot title.
    """
    return HDTuningCurveWidget(tuning_curves, unit_depths=unit_depths)
