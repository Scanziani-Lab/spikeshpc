"""Scrolling view of one heading trace's turns, to see what the detection calls a turn.

The x-axis is decoded time, as in :mod:`spikeshpc.decoder_widget`: the trace's
bins laid end to end, so the stretches the decoder skipped take no room, and
each splice between two of them is marked. Above, the decoded heading, the
measured one, and the turns and jumps :func:`spikeshpc.turns.find_turns` found;
below, the angular velocity they came from, against the threshold. The title
gives the visible window's own share of constant heading, its turns and its
drift, beside the whole trace's, so a stretch can be judged against the rest.
Optionally, on top, the spikes of the units the decoder read
(:mod:`spikeshpc.raster`).
"""

from __future__ import annotations

import numpy as np

from .decoder_widget import break_at
from .optitrack.widgets._backend import warn_if_noninteractive_backend
from .raster import RasterPanel
from .turns import Turns, _runs

__all__ = ["TurnWidget", "show_turns"]

MEASURED_COLOR = "0.55"
SPLICE_COLOR = "#c03030"
JUMP_COLOR = "0.35"
TURN_COLORS = {1: "tab:blue", -1: "tab:orange"}  # clockwise, counterclockwise
TURN_ALPHA = 0.22


class TurnWidget:
    """Scroll through one trace's turns; arrow keys and a slider.

    ``left`` / ``right`` move by half a window, so consecutive views overlap.
    ``up`` / ``down`` lengthen and shorten the window, ``home`` / ``end`` jump
    to either end, and the slider goes anywhere in between -- the keys of
    :class:`~spikeshpc.decoder_widget.DecodedWidget`.

    ``time_s``, ``heading_deg`` and ``run_index`` are the trace ``turns`` was
    found in; ``measured_deg`` optionally the measured heading in the same bins.
    The window grows to ``max_window_s``, by default the whole trace or 15
    minutes, whichever is shorter.

    ``raster`` (a :class:`~spikeshpc.raster.UnitRaster`) adds the spikes of the
    units the decoder read, on top: a row per unit, sorted by preferred
    direction and colored by it through ``raster_color``. A window holding more
    than ``max_raster_spikes`` spikes shows a note instead of the ticks.

    ``apply_offset=True`` subtracts ``offset_deg`` -- notebook 3's
    ``wake_offset_deg`` for the decode, the constant rotation between what the
    decoder reads and the measured heading -- from the decoded heading before
    it is drawn. Only the drawing moves: turns, velocity and drift do not
    depend on a constant, and the raster stays in the decoder's frame.

    Requires an interactive matplotlib backend (``%matplotlib qt`` or
    ``%matplotlib widget``) and the figure to have keyboard focus: click it once.
    """

    def __init__(
        self,
        time_s,
        heading_deg,
        turns: Turns,
        run_index=None,
        measured_deg=None,
        window_s: float = 60.0,
        color="k",
        min_window_s: float = 5.0,
        max_window_s: float | None = None,
        raster=None,
        raster_color="hsv",
        max_raster_spikes: int = 150_000,
        apply_offset: bool = False,
        offset_deg: float | None = None,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.widgets import Slider

        warn_if_noninteractive_backend()
        self.time_s = np.asarray(time_s, dtype=float)
        self.heading_deg = np.asarray(heading_deg, dtype=float)
        n = len(self.time_s)
        if n < 2:
            raise ValueError(f"{turns.label!r} holds {n} bins")
        if self.heading_deg.shape != (n,) or turns.velocity_deg_s.shape != (n,):
            raise ValueError(
                f"heading_deg and the turns' velocity must have one value per bin ({n}): "
                "pass the trace the turns were found in"
            )
        self.measured_deg = None
        if measured_deg is not None:
            self.measured_deg = np.asarray(measured_deg, dtype=float)
            if self.measured_deg.shape != (n,):
                raise ValueError(f"measured_deg has {self.measured_deg.shape} entries, not {n}")
        if apply_offset and (offset_deg is None or not np.isfinite(offset_deg)):
            raise ValueError(
                "apply_offset needs offset_deg: the decode's wake_offset_deg from notebook 3"
            )
        self.turns = turns
        self.color = color
        self.offset_deg = float(offset_deg) if apply_offset else 0.0
        self.shown_deg = (self.heading_deg - self.offset_deg) % 360.0  # what is drawn

        # decoded time: the bins laid end to end, splices marked where they meet
        self.edges = np.arange(n + 1) * turns.bin_s
        self.centers = self.edges[:-1] + turns.bin_s / 2
        self.total_s = float(self.edges[-1])
        self.breaks = np.flatnonzero(np.diff(_runs(self.time_s, run_index, turns.bin_s))) + 1
        self.break_s = self.edges[self.breaks]
        # the turns and jumps as bin indices along that axis
        self.turn_first = np.searchsorted(self.time_s, turns.turn_start_s)
        self.turn_last = np.searchsorted(self.time_s, turns.turn_stop_s)
        self.jump_index = np.searchsorted(self.time_s, turns.jump_time_s)
        self._cuts = np.union1d(self.breaks, self.jump_index)  # where no line may join

        self.min_window_s = float(min_window_s)
        self.max_window_s = float(max_window_s or min(self.total_s, 900.0))
        self.window_s = float(np.clip(window_s, self.min_window_s, self.max_window_s))
        self.t0 = 0.0

        if raster is None:
            self.fig, (self.heading_ax, self.velocity_ax) = plt.subplots(
                2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
            )
            self.fig.subplots_adjust(bottom=0.17, top=0.87, hspace=0.08)
            self.raster_ax = None
            slider_box = [0.125, 0.05, 0.775, 0.03]
        else:
            self.fig, (self.raster_ax, self.heading_ax, self.velocity_ax) = plt.subplots(
                3, 1, figsize=(12, 9.0), sharex=True,
                gridspec_kw={"height_ratios": [1.5, 2.2, 1.0]},
            )
            self.fig.subplots_adjust(bottom=0.12, top=0.9, hspace=0.07)
            slider_box = [0.125, 0.035, 0.775, 0.022]
        self.fig.patch.set_facecolor("white")

        (self.measured_line,) = self.heading_ax.plot(
            [], [], color=MEASURED_COLOR, lw=1.4, label="measured", zorder=3
        )
        label = turns.label or "decoded"
        if apply_offset:
            label += f", {self.offset_deg:+.0f} deg offset removed"
        (self.decoded_dots,) = self.heading_ax.plot(
            [], [], ".", color=color, ms=3.5, label=label, zorder=4
        )
        (self.velocity_line,) = self.velocity_ax.plot([], [], color=color, lw=1.1, zorder=3)
        self._drawn = []  # turn spans, jump and splice lines: redrawn with the window

        self.heading_ax.set_ylim(0, 360)
        self.heading_ax.set_yticks(np.arange(0, 361, 90))
        self.heading_ax.set_ylabel("heading (deg)")
        speed = np.abs(turns.velocity_deg_s[np.isfinite(turns.velocity_deg_s)])
        reach = 2.5 * turns.turn_threshold
        if speed.size:
            reach = max(reach, 1.1 * float(np.percentile(speed, 99.5)))
        self.velocity_ax.set_ylim(-reach, reach)
        for sign in (1, -1):
            self.velocity_ax.axhline(sign * turns.turn_threshold, color="0.5", ls=":", lw=0.9)
        self.velocity_ax.axhline(0.0, color="0.85", lw=0.8)
        self.velocity_ax.set_ylabel("deg/s, CW up")
        self.velocity_ax.set_xlabel("decoded time (s)")
        self.axes = [
            ax for ax in (self.raster_ax, self.heading_ax, self.velocity_ax) if ax is not None
        ]
        for ax in self.axes:
            ax.set_facecolor("white")
            ax.spines[["top", "right"]].set_visible(False)

        # the spikes the decode was read from, on the same decoded-time axis
        self.raster_panel = None
        if raster is not None:
            start = self.time_s - turns.bin_s / 2
            self.raster_panel = RasterPanel(
                self.raster_ax, raster, start, start + turns.bin_s, self.edges,
                cmap=raster_color, max_spikes=max_raster_spikes,
            )
            self.raster_panel.add_colorbar(self.fig)

        handles = [self.decoded_dots]
        if self.measured_deg is not None:
            handles.append(self.measured_line)
        handles += [
            Patch(facecolor=TURN_COLORS[1], alpha=TURN_ALPHA, label="clockwise turn"),
            Patch(facecolor=TURN_COLORS[-1], alpha=TURN_ALPHA, label="counterclockwise turn"),
            Line2D([], [], color=JUMP_COLOR, ls="--", lw=0.8, label="jump"),
            Line2D([], [], color=SPLICE_COLOR, ls="--", lw=1.0, label="splice"),
        ]
        self.heading_ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9,
                               ncol=len(handles))

        slider_ax = self.fig.add_axes(slider_box)
        self.slider = Slider(
            slider_ax, "position", 0.0, max(self.total_s - self.window_s, 1e-9), valinit=0.0,
            color="#b0c4de",
        )
        self._syncing = False
        self.slider.on_changed(self._on_slider)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.text(0.99, 0.005, "← → pan   ↑ ↓ window   home/end", ha="right", fontsize=8,
                      color="gray")
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
        for artist in self._drawn:
            artist.remove()
        self._drawn = []

        index = np.flatnonzero((self.centers >= self.t0) & (self.centers < stop))
        for ax in self.axes:
            ax.set_xlim(self.t0, stop)
        if self.raster_panel is not None:
            self.raster_panel.draw(self.t0, stop)
        if index.size == 0:
            self.fig.canvas.draw_idle()
            return
        first, last = index[0], index[-1]
        x = self.centers[index]
        cuts = self._cuts[(self._cuts > first) & (self._cuts <= last)] - first

        self.decoded_dots.set_data(x, self.shown_deg[index])
        if self.measured_deg is not None:
            self.measured_line.set_data(*break_at(x, self.measured_deg[index], cuts))
        self.velocity_line.set_data(
            *break_at(x, self.turns.velocity_deg_s[index], cuts, wrap_threshold=np.inf)
        )

        # one collection per kind and axis, however many turns and jumps a long
        # window holds -- a post-lesion wake trace has thousands of each
        axes = (self.heading_ax, self.velocity_ax)
        visible = (self.turn_last >= first) & (self.turn_first <= last)
        for sign, color in TURN_COLORS.items():
            chosen = visible & (self.turns.turn_direction == sign)
            if not chosen.any():
                continue
            starts = self.edges[self.turn_first[chosen]]
            spans = list(zip(starts, self.edges[self.turn_last[chosen] + 1] - starts))
            for ax in axes:
                low, high = ax.get_ylim()
                self._drawn.append(ax.broken_barh(spans, (low, high - low), facecolors=color,
                                                  alpha=TURN_ALPHA, lw=0, zorder=1))
        jumps = self.jump_index[(self.jump_index >= first) & (self.jump_index <= last)]
        if jumps.size:
            self._drawn.append(self.heading_ax.vlines(self.edges[jumps], 0, 360, colors=JUMP_COLOR,
                                                      linestyles="--", lw=0.8, zorder=2))
        splices = self.break_s[(self.break_s > self.t0) & (self.break_s < stop)]
        if splices.size:
            for ax in self.axes:
                low, high = ax.get_ylim()
                self._drawn.append(ax.vlines(splices, low, high, colors=SPLICE_COLOR,
                                             linestyles="--", lw=1.0, zorder=5))

        self._set_title(index)
        self.fig.canvas.draw_idle()

    def _set_title(self, index):
        turns = self.turns
        real = self.time_s[index]
        title = (
            f"{turns.label} — decoded {self.t0:.0f}-{self.t0 + self.window_s:.0f}s of "
            f"{self.total_s:.0f}s (recording {real[0]:.0f}-{real[-1]:.0f}s)"
        )
        velocity = turns.velocity_deg_s[index]
        velocity = velocity[np.isfinite(velocity)]
        if velocity.size:
            constant = 100.0 * np.mean(np.abs(velocity) < turns.turn_threshold)
            starting = (self.turn_first >= index[0]) & (self.turn_first <= index[-1])
            direction = turns.turn_direction[starting]
            title += (
                f"\nhere: {constant:.0f}% constant, {starting.sum()} turns "
                f"({np.sum(direction > 0)} CW / {np.sum(direction < 0)} CCW), "
                f"drift {60.0 * velocity.mean():+.0f} deg/min"
            )
        title += (
            f"   |   whole trace: {100.0 * turns.constant_fraction:.0f}% constant, "
            f"{turns.turns_per_min:.1f} turns/min, drift {turns.drift_deg_per_min:+.0f} deg/min"
        )
        self.axes[0].set_title(title, fontsize=10)


def show_turns(time_s, heading_deg, turns: Turns, window_s: float = 60.0, **kwargs) -> TurnWidget:
    """Open the scrolling view of one trace's turns.

    ``left``/``right`` pan by half a window, ``up``/``down`` change how much
    time is shown, ``home``/``end`` jump to either end, and the slider goes
    anywhere. Pass the trace ``turns`` was found in; ``run_index``,
    ``measured_deg`` and ``color`` go to :class:`TurnWidget`, as do ``raster``
    and ``raster_color`` (the decoder's units' spikes on top, colored by
    preferred direction, default ``"hsv"``) and ``apply_offset`` with
    ``offset_deg`` (notebook 3's wake offset, subtracted from the decoded
    heading before it is drawn).
    """
    return TurnWidget(time_s, heading_deg, turns, window_s=window_s, **kwargs)
