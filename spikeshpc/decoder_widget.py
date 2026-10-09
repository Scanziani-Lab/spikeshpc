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

from .optitrack.widgets._backend import use_backend
from .raster import RasterPanel

__all__ = ["DecodedWidget", "show_decoded"]

RASTER_MODES = ("text", "colorbar", "ridgeline")
ACTUAL_COLOR = "black"
DECODED_COLOR = "#00a000"
MARK_COLOR = "0.45"
MARK_ALPHA = 0.18


def break_at(x, y, breaks, wrap_threshold: float = 180.0):
    """Insert NaN into a line so it does not draw joins that never happened.

    Two kinds of false join. A heading crossing 0/360 is one step on the ring
    and a full-height plunge on a linear axis, so a plain line reports a
    violent turn where the animal turned a degree. And a splice between two
    non-adjacent stretches of the recording is not a movement at all.

    Parameters
    ----------
    x, y : array-like
        Line coordinates.
    breaks : array-like
        Indices after which the line is cut outright.
    wrap_threshold : float, default 180.0
        A step in `y` larger than this is treated as a wrap-around and cut.

    Returns
    -------
    x, y : numpy.ndarray
        Copies with NaN inserted at the cuts.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return x, y

    cuts = {int(i) for i in breaks}
    cuts.update((np.flatnonzero(np.abs(np.diff(y)) > wrap_threshold) + 1).tolist())
    if not cuts:
        return x, y

    at = np.array(sorted(cuts), dtype=int)
    return np.insert(x, at, np.nan), np.insert(y, at, np.nan)


def wrap_through(x, y, breaks=(), top: float = 360.0):
    """Draw a heading line that crosses north the short way.

    The line leaves one edge of the axis and comes back in at the other.

    Where two neighbouring points are more than half the ring apart on the
    axis, the short way between them crosses 0/360: the line runs to the top
    (or bottom) edge at the time it would reach it, breaks, and comes back in
    from the opposite edge -- as the heading itself does. :func:`break_at`
    cuts there instead, which leaves a gap wherever the heading crosses north.
    `breaks` are cut outright (a splice, say), as in :func:`break_at`.

    Parameters
    ----------
    x, y : array-like
        Line coordinates; `y` is a heading.
    breaks : array-like, default ()
        Indices after which the line is cut outright.
    top : float, default 360.0
        Top of the heading axis.

    Returns
    -------
    x, y : numpy.ndarray
        Copies with points inserted where the line crosses an edge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return x, y

    cut = np.array(sorted({int(i) for i in breaks if 0 < int(i) < len(y)}), dtype=int)
    joined = np.ones(len(y) - 1, dtype=bool)
    joined[cut - 1] = False
    step = np.diff(y)
    i = np.flatnonzero(joined & (np.abs(step) > top / 2))  # between points i and i + 1
    short = step[i] - top * np.sign(step[i])
    edge = np.where(short > 0, top, 0.0)
    at_edge = x[i] + (edge - y[i]) / short * (x[i + 1] - x[i])
    gap = np.full(len(i), np.nan)

    position = np.concatenate([cut, np.repeat(i + 1, 3)])
    new_x = np.concatenate(
        [np.full(len(cut), np.nan), np.column_stack([at_edge, gap, at_edge]).ravel()]
    )
    new_y = np.concatenate(
        [np.full(len(cut), np.nan), np.column_stack([edge, gap, top - edge]).ravel()]
    )
    order = np.argsort(position, kind="stable")
    return np.insert(x, position[order], new_x[order]), np.insert(
        y, position[order], new_y[order]
    )


class DecodedWidget:
    """Scroll through a decoded stretch; arrow keys and a slider.

    ``left`` / ``right`` move by half a window, so consecutive views overlap
    and nothing falls between them. ``up`` / ``down`` lengthen and shorten the
    window. ``home`` / ``end`` jump to the first and last decoded bin, and the
    slider at the bottom goes anywhere in between.

    Needs an interactive matplotlib backend, which `backend` switches to, and
    the figure to have keyboard focus: click it once.

    Parameters
    ----------
    decoded : Decoded
        The decode to show.
    window_s : float, default 60.0
        Initial window length, in seconds.
    show_posterior : bool, default True
        Draw the posterior behind the heading.
    cmap : str, default "Blues"
        Colormap of the posterior.
    min_window_s : float, default 1.0
        Shortest window.
    max_window_s : float, optional
        Longest window.
    mark : array-like of bool, optional
        One value per decoded bin, shaded gray behind the traces -- the bins
        where the animal was still, say, so a failure can be seen against what
        the animal was doing rather than inferred from the heading trace going
        flat.
    mark_label : str, default "still"
        Legend name of `mark`.
    raster : spikeshpc.raster.UnitRaster, optional
        Spikes of the decoder's units, from
        :func:`~spikeshpc.raster.unit_raster`, drawn above the heading: a row
        per unit, sorted by preferred direction and colored by it.
    raster_color : str or Colormap, default "hsv"
        Colormap for the raster.
    max_raster_spikes : int, default 150000
        A window holding more spikes shows a note instead of the ticks.
    raster_mode : {"text", "colorbar", "ridgeline"}, default "text"
        What goes left of the raster. ``"text"``: the unit labels, with the
        preferred-direction colorbar on the right. ``"colorbar"``: that
        colorbar, in place of the labels. ``"ridgeline"``: each unit's tuning
        curve on its row, filled with the colormap, over a small heading
        colorbar -- the raster must carry tuning curves
        (``unit_raster(..., tuning=model)``).
    backend : str or None, default "qt"
        Matplotlib backend to switch to first, as ``%matplotlib`` names it:
        ``"qt"`` opens a window, ``"widget"`` draws in the notebook. Only
        applied in IPython; None keeps the current one.

    Raises
    ------
    ValueError
        If fewer than two bins were decoded, `raster_mode` is unknown, or it
        is ``"ridgeline"`` for a raster without tuning curves.
    """

    def __init__(
        self,
        decoded,
        window_s: float = 60.0,
        show_posterior: bool = True,
        cmap: str = "Blues",
        min_window_s: float = 1.0,
        max_window_s: float | None = None,
        mark=None,
        mark_label: str = "still",
        raster=None,
        raster_color="hsv",
        max_raster_spikes: int = 150_000,
        raster_mode: str = "text",
        backend: str | None = "qt",
    ):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.widgets import Slider

        use_backend(backend)
        if decoded.n_decoded < 2:
            raise ValueError(f"{decoded.label!r} holds {decoded.n_decoded} bins")
        if raster_mode not in RASTER_MODES:
            raise ValueError(f"raster_mode must be one of {RASTER_MODES}, not {raster_mode!r}")
        if raster_mode == "ridgeline" and raster is not None and raster.tuning_hz is None:
            raise ValueError(
                "raster_mode='ridgeline' needs tuning curves on the raster: build it "
                "with unit_raster(..., tuning=run.model) or tuning=hd"
            )
        if mark is not None:
            mark = np.asarray(mark, dtype=bool)
            if mark.shape != (decoded.n_decoded,):
                raise ValueError(
                    f"mark has {mark.shape} entries but {decoded.label!r} has "
                    f"{decoded.n_decoded} decoded bins"
                )

        self.decoded = decoded
        self.show_posterior = show_posterior and decoded.posterior is not None
        self.cmap = cmap
        self.mark = mark
        self.mark_label = mark_label
        self._mark_patches = []
        self._mark_handle = Patch(
            facecolor=MARK_COLOR, alpha=MARK_ALPHA, label=mark_label
        )

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

        if raster is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 4.5))
            self.fig.subplots_adjust(bottom=0.24, top=0.86)
            self.raster_ax = None
            slider_box = [0.125, 0.07, 0.775, 0.035]
        else:
            self.fig, (self.raster_ax, self.ax) = plt.subplots(
                2,
                1,
                figsize=(12, 8.0),
                sharex=True,
                gridspec_kw={"height_ratios": [1.25, 1.0]},
            )
            # the ridgeline's heading axis hangs below the raster: make room
            hspace = 0.15 if raster_mode == "ridgeline" else 0.06
            self.fig.subplots_adjust(bottom=0.14, top=0.9, hspace=hspace)
            self.raster_ax.set_facecolor("white")
            slider_box = [0.125, 0.045, 0.775, 0.025]
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

        # the spikes each bin was decoded from, on the same decoded-time axis
        self.raster_panel = None
        if raster is not None:
            start = decoded.time_s - decoded.duration_s / 2
            self.raster_panel = RasterPanel(
                self.raster_ax,
                raster,
                start,
                start + decoded.duration_s,
                self.edges,
                cmap=raster_color,
                max_spikes=max_raster_spikes,
            )
            if raster_mode == "text":
                self.raster_panel.add_colorbar(self.fig)
            else:
                self.raster_panel.hide_unit_labels()
                if raster_mode == "colorbar":
                    self.raster_panel.add_colorbar(self.fig, width=0.01, side="left")
                else:
                    self.raster_panel.add_ridgeline(self.fig)

        slider_ax = self.fig.add_axes(slider_box)
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
            0.99,
            0.005,
            "← → pan   ↑ ↓ window   home/end",
            ha="right",
            fontsize=8,
            color="gray",
        )
        self._draw()

    # ── state ───────────────────────────────────────────────────────────
    def _clamp(self):
        self.window_s = float(
            np.clip(
                self.window_s, self.min_window_s, min(self.max_window_s, self.total_s)
            )
        )
        self.t0 = float(np.clip(self.t0, 0.0, max(self.total_s - self.window_s, 0.0)))

    def _sync_slider(self):
        """Move the slider with the view without it firing back."""
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
        for patch in self._mark_patches:
            patch.remove()
        self._mark_patches = []
        if self.raster_panel is not None:
            self.raster_panel.draw(self.t0, stop)

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
            for ax in (self.ax, self.raster_ax):
                if ax is not None:
                    self._break_lines.append(
                        ax.axvline(
                            at, color="#c03030", lw=1.0, ls="--", alpha=0.9, zorder=5
                        )
                    )

        # above the posterior, below the traces; one span per marked run, and
        # the window is a contiguous range of bins so a run is too
        handles = [self.actual_line, self.decoded_line]
        if self.mark is not None:
            handles.append(self._mark_handle)
            edges = np.diff(np.r_[0, self.mark[index].astype(int), 0])
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                self._mark_patches.append(
                    self.ax.axvspan(
                        self.edges[index[a]],
                        self.edges[index[b - 1] + 1],
                        color=MARK_COLOR,
                        alpha=MARK_ALPHA,
                        lw=0,
                        zorder=2,
                    )
                )

        self.ax.set_xlim(self.t0, stop)
        self._set_title(index)
        self.ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
            ncol=len(handles),
        )
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
                title += f", whole set {self.decoded.metrics['median_abs_error_deg']:.1f} deg"
        n_breaks = int(
            ((self.break_s > self.t0) & (self.break_s < self.t0 + self.window_s)).sum()
        )
        if n_breaks:
            title += f" | {n_breaks} splice{'s' if n_breaks > 1 else ''} (dashed)"
        if self.mark is not None:
            title += f" | {self.mark_label} {self.mark[index].mean():.0%} of window"
        top = self.raster_ax if self.raster_ax is not None else self.ax
        top.set_title(title, fontsize=10)


def show_decoded(decoded, window_s: float = 60.0, **kwargs) -> DecodedWidget:
    """Open the scrolling decode view.

    ``left``/``right`` pan by half a window, ``up``/``down`` change how much
    time is shown, ``home``/``end`` jump to either end, and the slider goes
    anywhere. Dashed red lines mark splices between non-adjacent stretches of
    the recording, where the decoder restarted from a uniform prior.

    Parameters
    ----------
    decoded : Decoded
        The decode to show.
    window_s : float, default 60.0
        Initial window length, in seconds.
    **kwargs
        Passed to :class:`DecodedWidget`: for example ``mark`` (a boolean per
        decoded bin, to shade bins gray, e.g. stillness) and ``raster`` (from
        :func:`~spikeshpc.raster.unit_raster`) for the decoder's units' spikes
        above the heading, colored by preferred direction through
        ``raster_color``; ``raster_mode`` (``"text"``, ``"colorbar"`` or
        ``"ridgeline"``) for what sits left of the raster; and ``backend``
        (default ``"qt"``; ``"widget"`` to draw in the notebook).

    Returns
    -------
    DecodedWidget
        The open widget.
    """
    return DecodedWidget(decoded, window_s=window_s, **kwargs)
