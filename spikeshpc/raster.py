"""Spike rasters of the units a decoder read, for the scrolling decode views.

One row per unit, sorted by preferred direction, lowest at the bottom: the rows
run up the page the way the heading axis beneath them does, so where the
population's activity sits in the raster, the decoded heading should sit at the
same height below. Each unit's ticks take its preferred direction's color from
a colormap -- hsv by default, which wraps round, so 0 and 360 degrees get
nearly the same red.

Spikes are drawn on the views' decoded-time axis: each is placed in the decoded
bin it fell in, at its offset into that bin, and a spike in no decoded bin --
in the data the decoder trained on, or in another state -- is not drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decoder import _as_sorting

__all__ = ["RasterPanel", "UnitRaster", "on_decoded_axis", "unit_raster"]


@dataclass
class UnitRaster:
    """The spikes of the units a decode used, and each unit's preferred direction.

    ``spike_times`` holds one array per unit, in seconds on the recording clock
    the decoder binned them on; ``preferred_deg`` and ``labels`` one entry per
    unit, in the same order.
    """

    spike_times: list
    preferred_deg: np.ndarray
    labels: list

    def __post_init__(self):
        self.spike_times = [np.sort(np.asarray(t, dtype=float)) for t in self.spike_times]
        self.preferred_deg = np.asarray(self.preferred_deg, dtype=float) % 360.0
        self.labels = [str(label) for label in self.labels]
        n = len(self.spike_times)
        if n == 0:
            raise ValueError("a raster needs at least one unit")
        if self.preferred_deg.shape != (n,) or len(self.labels) != n:
            raise ValueError(
                f"{n} units of spikes but {self.preferred_deg.size} preferred directions "
                f"and {len(self.labels)} labels"
            )

    @property
    def n_units(self) -> int:
        return len(self.spike_times)


def unit_raster(sorting, unit_ids, preferred_deg, labels=None) -> UnitRaster:
    """The raster of `unit_ids` in `sorting` (or a sorting analyzer), timed as the decoder timed them.

    Spike times come from ``get_unit_spike_train(unit, return_times=True)``, the
    call :func:`~spikeshpc.decoder.prepare_decoder_data` bins, so each tick sits
    in the bin it was decoded from. ``preferred_deg`` is each unit's preferred
    direction in the decoder's model -- ``model.preferred_deg`` of an
    :class:`~spikeshpc.decoder.EncodingModel`. For a decode read through matched
    units, it is the preferred direction of the baseline unit each one stands
    in for. ``labels`` default to the unit ids.
    """
    sorting = _as_sorting(sorting)
    unit_ids = list(unit_ids)
    return UnitRaster(
        spike_times=[sorting.get_unit_spike_train(u, return_times=True) for u in unit_ids],
        preferred_deg=preferred_deg,
        labels=[str(u) for u in unit_ids] if labels is None else list(labels),
    )


def on_decoded_axis(times, bin_start_s, bin_stop_s, axis_edges) -> np.ndarray:
    """Where each of `times` falls on a decoded-time axis; a time in no decoded bin is dropped.

    ``bin_start_s`` and ``bin_stop_s`` are the decoded bins on the recording
    clock, in time order and not overlapping; ``axis_edges`` is where each bin
    starts on the decoded-time axis. A time keeps its offset into its bin, so
    sorted times come back sorted.
    """
    times = np.asarray(times, dtype=float)
    bin_start_s = np.asarray(bin_start_s, dtype=float)
    index = np.searchsorted(bin_start_s, times, side="right") - 1
    inside = index >= 0
    inside[inside] = times[inside] < np.asarray(bin_stop_s, dtype=float)[index[inside]]
    index = index[inside]
    return np.asarray(axis_edges, dtype=float)[index] + (times[inside] - bin_start_s[index])


class RasterPanel:
    """A :class:`UnitRaster` drawn on one axes of a scrolling view, a window at a time.

    ``bin_start_s`` / ``bin_stop_s`` are the view's decoded bins on the
    recording clock and ``axis_edges`` where each starts on its decoded-time
    axis. A window holding more than ``max_spikes`` spikes is not drawn -- the
    axes says so instead -- since a view zoomed out over hours would otherwise
    stall on hundreds of thousands of ticks.
    """

    def __init__(self, ax, raster: UnitRaster, bin_start_s, bin_stop_s, axis_edges,
                 cmap="hsv", max_spikes: int = 150_000):
        import matplotlib

        self.ax = ax
        self.raster = raster
        self.max_spikes = int(max_spikes)
        self.colormap = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        self.order = np.argsort(raster.preferred_deg, kind="stable")  # row -> unit
        self.row_colors = self.colormap(raster.preferred_deg[self.order] / 360.0)
        self.positions = [
            on_decoded_axis(raster.spike_times[unit], bin_start_s, bin_stop_s, axis_edges)
            for unit in self.order
        ]
        self.n_drawn = 0
        self._collection = None
        self._note = None

        n = raster.n_units
        ax.set_ylim(0, n)
        every = max(1, int(np.ceil(n / 40)))  # at most 40 labels
        rows = np.arange(0, n, every)
        ax.set_yticks(rows + 0.5)
        ax.set_yticklabels([raster.labels[self.order[r]] for r in rows], fontsize=6)
        ax.tick_params(axis="y", length=0)
        ax.set_ylabel("units by preferred\ndirection", fontsize=9)

    def draw(self, t0: float, stop: float) -> int:
        """Draw the spikes between `t0` and `stop` on the decoded-time axis; returns how many."""
        from matplotlib.collections import LineCollection

        for artist in (self._collection, self._note):
            if artist is not None:
                artist.remove()
        self._collection = self._note = None

        spans = [np.searchsorted(x, [t0, stop]) for x in self.positions]
        total = int(sum(b - a for a, b in spans))
        if total > self.max_spikes:
            self._note = self.ax.text(
                0.5, 0.5, f"{total:,} spikes in view -- press down to draw them",
                transform=self.ax.transAxes, ha="center", va="center", fontsize=9, color="0.4",
            )
            self.n_drawn = 0
            return 0
        x = np.concatenate([x[a:b] for x, (a, b) in zip(self.positions, spans)])
        row = np.repeat(np.arange(len(spans)), [b - a for a, b in spans])
        segments = np.empty((x.size, 2, 2))
        segments[:, :, 0] = x[:, None]
        segments[:, 0, 1] = row + 0.12
        segments[:, 1, 1] = row + 0.88
        self._collection = LineCollection(segments, colors=self.row_colors[row], linewidths=0.9,
                                          zorder=3)
        self.ax.add_collection(self._collection)
        self.n_drawn = total
        return total

    def add_colorbar(self, fig, width: float = 0.008, pad: float = 0.006):
        """A strip right of the raster reading color back to preferred direction."""
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        box = self.ax.get_position()
        cax = fig.add_axes([box.x1 + pad, box.y0, width, box.height])
        bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 360), cmap=self.colormap), cax=cax)
        bar.set_ticks([0, 90, 180, 270, 360])
        bar.ax.tick_params(labelsize=7)
        bar.set_label("preferred (deg)", fontsize=7)
        return bar
