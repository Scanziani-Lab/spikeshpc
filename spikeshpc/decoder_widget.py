"""Scrolling view of a decoded stretch, against the heading it should match.

The x-axis is *decoded* time, not recording time: the held-out bins laid end to
end. A blocked train/test split scatters the test set across the session, so
plotting against the recording clock would spend most of the axis on training
data that is not drawn. Laying the decoded bins end to end instead means every
pixel is data, and the price is that the axis is no longer the recording's own
clock -- so each splice is marked, and the real time of the visible span is in
the title.

Those splice marks earn their keep beyond bookkeeping. Belief does not cross
them: each run starts from a uniform prior (see :func:`spikeshpc.decoder.decode`),
so the first bins after a mark are decoded from their own spikes alone and are
the least certain in the window. A wobble there means something different from
a wobble in the middle of a run.
"""

from __future__ import annotations

import numpy as np

from .optitrack.widgets._backend import warn_if_noninteractive_backend

__all__ = ["DecodedWidget", "show_decoded"]

ACTUAL_COLOR = "black"
DECODED_COLOR = "#00a000"


def break_at(x, y, breaks, wrap_threshold: float = 180.0):
    """Insert NaN into a line so it does not draw joins that never happened.

    Two kinds of false join. A heading crossing 0/360 is one step on the ring
    and a full-height plunge on a linear axis, so a plain line reports a
    violent turn where the animal turned a degree. And a splice between two
    non-adjacent stretches of the recording is not a movement at all.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return x, y

    cuts = set(int(i) for i in breaks)
    cuts.update((np.flatnonzero(np.abs(np.diff(y)) > wrap_threshold) + 1).tolist())
    if not cuts:
        return x, y

    at = np.array(sorted(cuts), dtype=int)
    return np.insert(x, at, np.nan), np.insert(y, at, np.nan)


class DecodedWidget:
    """Scroll through a decoded stretch; arrow keys and a slider.

    ``left`` / ``right`` move by half a window, so consecutive views overlap
    and nothing falls between them. ``up`` / ``down`` lengthen and shorten the
    window. ``home`` / ``end`` jump to the first and last decoded bin, and the
    slider at the bottom goes anywhere in between.

    Requires an interactive matplotlib backend (``%matplotlib widget`` or
    ``%matplotlib qt``) and the figure to have keyboard focus: click it once.
    """

    def __init__(
        self,
        decoded,
        window_s: float = 60.0,
        show_posterior: bool = True,
        cmap: str = "Blues",
        min_window_s: float = 1.0,
        max_window_s: float | None = None,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        warn_if_noninteractive_backend()
        if decoded.n_decoded < 2:
            raise ValueError(f"{decoded.label!r} holds {decoded.n_decoded} bins")

        self.decoded = decoded
        self.show_posterior = show_posterior and decoded.posterior is not None
        self.cmap = cmap

        # decoded time: the bins laid end to end, so no pixel is spent on the
        # data that was held back for training
        self.edges = np.r_[0.0, np.cumsum(decoded.duration_s)]
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.total_s = float(self.edges[-1])

        # a splice is where the decode restarted: belief does not cross it
        self.breaks = np.flatnonzero(np.diff(decoded.run_index) != 0) + 1
        self.break_s = self.edges[self.breaks]

        self.min_window_s = float(min_window_s)
        self.max_window_s = float(max_window_s or self.total_s)
        self.window_s = float(np.clip(window_s, self.min_window_s, self.max_window_s))
        self.t0 = 0.0

        self.fig, self.ax = plt.subplots(figsize=(12, 4.5))
        self.fig.subplots_adjust(bottom=0.24, top=0.86)
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")

        (self.actual_line,) = self.ax.plot(
            [], [], color=ACTUAL_COLOR, lw=1.4, label="actual", zorder=3
        )
        (self.decoded_line,) = self.ax.plot(
            [], [], color=DECODED_COLOR, lw=1.4, label="decoded", zorder=4
        )
        self._mesh = None
        self._break_lines = []

        self.ax.set_ylim(0, 360)
        self.ax.set_yticks(np.arange(0, 361, 90))
        self.ax.set_ylabel("head direction (deg)")
        self.ax.set_xlabel("decoded time (s)")

        slider_ax = self.fig.add_axes([0.125, 0.07, 0.775, 0.035])
        self.slider = Slider(
            slider_ax,
            "position",
            0.0,
            max(self.total_s - self.window_s, 1e-9),
            valinit=0.0,
            color="#b0c4de",
        )
        self._syncing = False
        self.slider.on_changed(self._on_slider)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.fig.text(
            0.99, 0.005, "← → pan   ↑ ↓ window   home/end",
            ha="right", fontsize=8, color="gray",
        )
        self._draw()

    # ── state ───────────────────────────────────────────────────────────
    def _clamp(self):
        self.window_s = float(
            np.clip(self.window_s, self.min_window_s, min(self.max_window_s, self.total_s))
        )
        self.t0 = float(np.clip(self.t0, 0.0, max(self.total_s - self.window_s, 0.0)))

    def _sync_slider(self):
        """Keep the slider with the view without it firing back at us."""
        self._syncing = True
        try:
            self.slider.valmax = max(self.total_s - self.window_s, 1e-9)
            self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
            self.slider.set_val(min(self.t0, self.slider.valmax))
        finally:
            self._syncing = False

    def _on_slider(self, value):
        if self._syncing:
            return
        self.t0 = float(value)
        self._clamp()
        self._draw()

    def _on_key(self, event):
        step = self.window_s / 2.0
        if event.key == "right":
            self.t0 += step
        elif event.key == "left":
            self.t0 -= step
        elif event.key == "up":
            self.window_s *= 1.5
        elif event.key == "down":
            self.window_s /= 1.5
        elif event.key == "home":
            self.t0 = 0.0
        elif event.key == "end":
            self.t0 = self.total_s
        else:
            return
        self._clamp()
        self._sync_slider()
        self._draw()

    # ── drawing ─────────────────────────────────────────────────────────
    def _draw(self):
        self._clamp()
        stop = self.t0 + self.window_s
        window = (self.centers >= self.t0) & (self.centers < stop)

        if self._mesh is not None:
            self._mesh.remove()
            self._mesh = None
        for line in self._break_lines:
            line.remove()
        self._break_lines = []

        if not window.any():
            self.ax.set_xlim(self.t0, stop)
            self.fig.canvas.draw_idle()
            return

        index = np.flatnonzero(window)
        x = self.centers[index]

        if self.show_posterior and len(index) > 1:
            posterior = self.decoded.posterior[index]
            width = 360.0 / len(self.decoded.bin_centers_deg)
            self._mesh = self.ax.pcolormesh(
                self.edges[index[0] : index[-1] + 2],
                np.r_[
                    self.decoded.bin_centers_deg - width / 2,
                    self.decoded.bin_centers_deg[-1] + width / 2,
                ],
                posterior.T,
                cmap=self.cmap,
                shading="flat",
                vmin=0.0,
                # scaled to the window, not the run: a confident stretch would
                # otherwise wash out every uncertain one on the same axis
                vmax=max(float(np.percentile(posterior, 99.5)), 1e-6),
                zorder=1,
            )

        # the line must not join across a splice, so cut where the run changes
        local_breaks = np.flatnonzero(np.diff(self.decoded.run_index[index]) != 0) + 1
        self.actual_line.set_data(
            *break_at(x, self.decoded.actual_deg[index], local_breaks)
        )
        self.decoded_line.set_data(
            *break_at(x, self.decoded.decoded_deg[index], local_breaks)
        )

        for at in self.break_s[(self.break_s > self.t0) & (self.break_s < stop)]:
            self._break_lines.append(
                self.ax.axvline(
                    at, color="#c03030", lw=1.0, ls="--", alpha=0.9, zorder=5
                )
            )

        self.ax.set_xlim(self.t0, stop)
        self._set_title(index)
        self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
        self.fig.canvas.draw_idle()

    def _set_title(self, index):
        real = self.decoded.time_s[index]
        title = (
            f"{self.decoded.label} — decoded {self.t0:.0f}-"
            f"{self.t0 + self.window_s:.0f}s of {self.total_s:.0f}s "
            f"(recording {real[0]:.0f}-{real[-1]:.0f}s)"
        )
        error = self.decoded.error_deg[index]
        error = error[np.isfinite(error)]
        if error.size:
            title += f"\nmedian |error| here {np.median(np.abs(error)):.1f} deg"
            if "median_abs_error_deg" in self.decoded.metrics:
                title += (
                    f", whole set {self.decoded.metrics['median_abs_error_deg']:.1f} deg"
                )
        n_breaks = int(
            ((self.break_s > self.t0) & (self.break_s < self.t0 + self.window_s)).sum()
        )
        if n_breaks:
            title += f" | {n_breaks} splice{'s' if n_breaks > 1 else ''} (dashed)"
        self.ax.set_title(title, fontsize=10)


def show_decoded(decoded, window_s: float = 60.0, **kwargs) -> DecodedWidget:
    """Open the scrolling decode view.

    ``left``/``right`` pan by half a window, ``up``/``down`` change how much
    time is shown, ``home``/``end`` jump to either end, and the slider goes
    anywhere. Dashed red lines mark splices between non-adjacent stretches of
    the recording, where the decoder restarted from a uniform prior.
    """
    return DecodedWidget(decoded, window_s=window_s, **kwargs)
