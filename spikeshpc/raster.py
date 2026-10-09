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

    Attributes
    ----------
    spike_times : list of numpy.ndarray
        One array per unit, in seconds on the recording clock the decoder
        binned them on.
    preferred_deg : numpy.ndarray
        Preferred direction of each unit, in the same order.
    labels : list
        Label of each unit, in the same order.
    tuning_deg : numpy.ndarray, optional
        Heading bin centres of `tuning_hz`, for the ridgeline beside the raster.
    tuning_hz : numpy.ndarray, optional
        Tuning curve of each unit, shape (n_units, n_bins), in the same order;
        a row of NaN for a unit without one.
    """

    spike_times: list
    preferred_deg: np.ndarray
    labels: list
    tuning_deg: np.ndarray | None = None
    tuning_hz: np.ndarray | None = None

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
        if (self.tuning_deg is None) != (self.tuning_hz is None):
            raise ValueError("tuning_deg and tuning_hz come together")
        if self.tuning_deg is not None:
            self.tuning_deg = np.asarray(self.tuning_deg, dtype=float)
            self.tuning_hz = np.asarray(self.tuning_hz, dtype=float)
            if self.tuning_hz.shape != (n, self.tuning_deg.size):
                raise ValueError(
                    f"tuning curves of shape {self.tuning_hz.shape} for {n} units "
                    f"and {self.tuning_deg.size} heading bins"
                )

    @property
    def n_units(self) -> int:
        return len(self.spike_times)


def unit_raster(sorting, unit_ids, preferred_deg, labels=None, tuning=None) -> UnitRaster:
    """Build the raster of `unit_ids`, timed as the decoder timed them.

    Spike times come from ``get_unit_spike_train(unit, return_times=True)``, the
    call :func:`~spikeshpc.decoder.prepare_decoder_data` bins, so each tick sits
    in the bin it was decoded from.

    Parameters
    ----------
    sorting : spikeinterface BaseSorting or SortingAnalyzer
        Source of the spike trains.
    unit_ids : sequence
        Units to include.
    preferred_deg : array-like
        Each unit's preferred direction in the decoder's model
        (``model.preferred_deg`` of an
        :class:`~spikeshpc.decoder.EncodingModel`). For a decode read through
        matched units, the preferred direction of the baseline unit each one
        stands in for.
    labels : sequence, optional
        Unit labels; defaults to the unit ids.
    tuning : EncodingModel or HDTuning, optional
        Where to read each unit's tuning curve, for the ridgeline beside the
        raster (``raster_mode="ridgeline"`` in
        :func:`~spikeshpc.decoder_widget.show_decoded`): the decoder's model,
        or :func:`~spikeshpc.optitrack.load_hd_tuning`'s curves, which cover
        units the model does not. Curves are matched by unit id; a unit
        without one gets an empty row.

    Returns
    -------
    UnitRaster
        The raster.

    Raises
    ------
    ValueError
        If `tuning` holds none of `unit_ids`.
    """
    sorting = _as_sorting(sorting)
    unit_ids = list(unit_ids)
    tuning_deg = tuning_hz = None
    if tuning is not None:
        tuning_deg, tuning_hz = _tuning_rows(tuning, unit_ids)
    return UnitRaster(
        spike_times=[sorting.get_unit_spike_train(u, return_times=True) for u in unit_ids],
        preferred_deg=preferred_deg,
        labels=[str(u) for u in unit_ids] if labels is None else list(labels),
        tuning_deg=tuning_deg,
        tuning_hz=tuning_hz,
    )


def _tuning_rows(tuning, unit_ids):
    """Pull the tuning curve of each of `unit_ids` out of `tuning`.

    Parameters
    ----------
    tuning : EncodingModel or HDTuning
        Anything with ``bin_centers_deg``, ``unit_ids`` and the curves in
        ``rate_hz`` (the model) or ``curves`` (the stored tuning).
    unit_ids : list
        Units wanted, in raster order.

    Returns
    -------
    tuning_deg : numpy.ndarray
        Heading bin centres.
    tuning_hz : numpy.ndarray
        One curve per unit, shape (len(unit_ids), n_bins); NaN where `tuning`
        has no curve for the unit.
    """
    curves = getattr(tuning, "rate_hz", None)
    if curves is None:
        curves = tuning.curves
    curves = np.asarray(curves, dtype=float)
    # by string, so 7, np.int64(7) and "7" are one unit
    row_of = {str(u): i for i, u in enumerate(tuning.unit_ids)}
    rows = [row_of.get(str(u)) for u in unit_ids]
    if all(row is None for row in rows):
        raise ValueError(
            f"the tuning curves cover none of the raster's {len(unit_ids)} units"
        )
    tuning_hz = np.full((len(unit_ids), curves.shape[1]), np.nan)
    for j, row in enumerate(rows):
        if row is not None:
            tuning_hz[j] = curves[row]
    return np.asarray(tuning.bin_centers_deg, dtype=float), tuning_hz


def _closed_curves(bin_centers_deg, curves):
    """Extend tuning curves to 0 and 360 degrees, so they span the whole axis.

    Bin centres stop half a bin short of either end; the value at the seam is
    interpolated between the last bin and the first, which are neighbours on
    the circle.

    Parameters
    ----------
    bin_centers_deg : numpy.ndarray
        Heading bin centres, ascending, within [0, 360).
    curves : numpy.ndarray
        Curves, shape (n_units, n_bins).

    Returns
    -------
    x : numpy.ndarray
        0, the bin centres, then 360.
    curves : numpy.ndarray
        The curves at `x`, shape (n_units, n_bins + 2).
    """
    centers = np.asarray(bin_centers_deg, dtype=float)
    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    gap = centers[0] + 360.0 - centers[-1]  # last bin to first, across the seam
    weight = (360.0 - centers[-1]) / gap
    seam = curves[:, -1] + weight * (curves[:, 0] - curves[:, -1])
    return np.r_[0.0, centers, 360.0], np.column_stack([seam, curves, seam])


def on_decoded_axis(times, bin_start_s, bin_stop_s, axis_edges) -> np.ndarray:
    """Map recording times onto the decoded-time axis.

    A time keeps its offset into its bin, so sorted times come back sorted.

    Parameters
    ----------
    times : array-like
        Times on the recording clock, in seconds.
    bin_start_s, bin_stop_s : array-like
        Decoded bins on the recording clock, in time order and not
        overlapping.
    axis_edges : array-like
        Where each bin starts on the decoded-time axis.

    Returns
    -------
    numpy.ndarray
        Positions on the decoded-time axis; a time in no decoded bin is
        dropped.
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

    A window holding more than `max_spikes` spikes is not drawn -- the axes
    says so instead -- since a view zoomed out over hours would otherwise stall
    on hundreds of thousands of ticks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    raster : UnitRaster
        Spikes to draw.
    bin_start_s, bin_stop_s : array-like
        The view's decoded bins on the recording clock.
    axis_edges : array-like
        Where each bin starts on the decoded-time axis.
    cmap : str or matplotlib.colors.Colormap, default "hsv"
        Colormap mapping preferred direction to color.
    max_spikes : int, default 150000
        Largest window, in spikes, that is drawn.
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

    def hide_unit_labels(self):
        """Drop the unit labels and the axis label, to make room on the left."""
        self.ax.set_yticks([])
        self.ax.set_ylabel("")

    def draw(self, t0: float, stop: float) -> int:
        """Draw the spikes between `t0` and `stop` on the decoded-time axis.

        Parameters
        ----------
        t0, stop : float
            Window edges on the decoded-time axis.

        Returns
        -------
        int
            Number of spikes drawn.
        """
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

    def add_colorbar(self, fig, width: float = 0.008, pad: float = 0.006,
                     side: str = "right"):
        """Add a strip beside the raster reading color back to preferred direction.

        The strip is linear in degrees while the rows are spaced by rank, so
        it gives a row's color, not its height.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure holding the raster axes.
        width : float, default 0.008
            Strip width, in figure fractions.
        pad : float, default 0.006
            Gap between the raster and the strip, in figure fractions.
        side : {"right", "left"}, default "right"
            Side of the raster. On the left, the strip stands in for the unit
            labels (see :meth:`hide_unit_labels`), its ticks and label facing
            out.

        Returns
        -------
        matplotlib.colorbar.Colorbar
            The colorbar.
        """
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', not {side!r}")
        box = self.ax.get_position()
        x0 = box.x1 + pad if side == "right" else box.x0 - pad - width
        cax = fig.add_axes([x0, box.y0, width, box.height])
        bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 360), cmap=self.colormap), cax=cax)
        bar.set_ticks([0, 90, 180, 270, 360])
        if side == "left":
            cax.yaxis.set_ticks_position("left")
            cax.yaxis.set_label_position("left")
            bar.ax.tick_params(labelsize=8)
            bar.set_label("preferred (deg)", fontsize=9)
        else:
            bar.ax.tick_params(labelsize=7)
            bar.set_label("preferred (deg)", fontsize=7)
        return bar

    def add_ridgeline(self, fig, width: float = 0.06, pad: float = 0.006,
                      height: float = 2.0, bar_height: float = 0.007):
        """Draw each row's tuning curve left of the raster, as a ridgeline.

        Each unit's curve sits on its own raster row and rises `height` rows
        above it, scaled to its own peak, so a sharp unit and a broad one are
        compared by shape rather than rate. The area under each curve is
        filled with the color of that unit's ticks -- its preferred direction
        -- and a short colorbar beneath is the ridgeline's heading axis, so a
        curve peaking where the colorbar has its own color is a unit whose
        preferred direction matches its curve. Lower rows are drawn over
        higher ones.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure holding the raster axes.
        width : float, default 0.06
            Width of the ridgeline, in figure fractions.
        pad : float, default 0.006
            Gap between the ridgeline and the raster, in figure fractions.
        height : float, default 2.0
            Height of each curve's peak above its row, in rows.
        bar_height : float, default 0.007
            Height of the heading colorbar, in figure fractions.

        Returns
        -------
        ridge_ax : matplotlib.axes.Axes
            The ridgeline.
        bar : matplotlib.colorbar.Colorbar
            Its heading axis.

        Raises
        ------
        ValueError
            If the raster carries no tuning curves.
        """
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        if self.raster.tuning_hz is None:
            raise ValueError(
                "the raster has no tuning curves to draw: build it with "
                "unit_raster(..., tuning=model) or tuning=hd"
            )
        box = self.ax.get_position()
        ridge_ax = fig.add_axes([box.x0 - pad - width, box.y0, width, box.height])
        ridge_ax.set_xlim(0, 360)
        ridge_ax.set_ylim(self.ax.get_ylim())
        ridge_ax.axis("off")

        x, curves = _closed_curves(self.raster.tuning_deg, self.raster.tuning_hz[self.order])
        n = len(self.order)
        for row, curve in enumerate(curves):
            peak = np.nanmax(curve) if np.isfinite(curve).any() else 0.0
            if not peak > 0:
                continue  # no curve, or a silent unit: an empty row
            base = row + 0.1
            y = base + height * curve / peak
            ridge_ax.fill_between(x, base, y, facecolor=self.row_colors[row], lw=0,
                                  zorder=2 * (n - row), clip_on=False)
            ridge_ax.plot(x, y, color="0.15", lw=0.5, zorder=2 * (n - row) + 1, clip_on=False)

        cax = fig.add_axes([box.x0 - pad - width, box.y0 - pad - bar_height, width, bar_height])
        bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 360), cmap=self.colormap),
                           cax=cax, orientation="horizontal")
        bar.set_ticks([0, 180, 360])
        bar.ax.tick_params(labelsize=6, length=2, pad=1)
        bar.set_label("heading (deg)", fontsize=6, labelpad=1)
        return ridge_ax, bar
