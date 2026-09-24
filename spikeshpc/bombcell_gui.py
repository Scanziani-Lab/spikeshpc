"""bombcell's unit-quality GUI, fast enough to curate with, and in one window.

``bombcell.unit_quality_gui()`` takes about a minute to move between units on
a long recording. Nearly all of that is the "Units by depth" panel, which
re-derives every unit's firing rate from the full spike train on every redraw:
one pass over all spikes per unit, ~500 passes per click, for numbers that
never change. The other panels each rescan the whole spike train again to
find the current unit's spikes, and its loader reads ``pc_features.npy`` --
tens of GB on a long session -- into memory for a GUI that never looks at it.

:class:`FastUnitQualityGUI` subclasses bombcell's GUI and keeps its figure:
same panels, same buttons, same saved classification files. What changes:

  * the units-by-depth data is computed once, and the panel is one scatter
    instead of one per unit
  * every panel sees only the current unit's spikes, indexed once per unit
    rather than once per panel
  * the scaling-factor panel's cutoff-Gaussian fit -- bombcell's, unchanged,
    but up to 4 s a unit, since a quarter of units run it to its 5000-
    evaluation limit -- is cached, and the next unit's is fitted in a
    background process while you look at the current one
  * the PC features are not loaded
  * under ``%matplotlib qt`` the controls are Qt widgets in the figure's own
    window (with Left/Right for prev/next unit), and the figure is redrawn in
    place rather than rebuilt as a new window; otherwise the ipywidgets
    controls are as before, above an inline figure rendered at a lower dpi
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from bombcell.unit_quality_gui import InteractiveUnitQualityGUI

__all__ = ["FastUnitQualityGUI", "load_gui_inputs", "unit_quality_gui"]

FIG_SIZE = (30, 25)  # inches; bombcell's layout and font sizes assume it
WINDOW_TITLE = "bombcell unit quality"
CONTROLS_PX = 230  # title, three rows of buttons, toolbar and status bar
# the per-spike arrays: sliced down to the current unit before any panel sees them
PER_SPIKE_KEYS = ("spike_times", "spike_clusters", "template_amplitudes")
TYPE_COLORS = {  # bombcell's own, by unit type code
    0: [1, 0, 0],  # noise
    1: [0, 0.7, 0],  # good
    2: [1, 0.7, 0.2],  # MUA
    3: [0, 0, 1],  # non-somatic
    4: [0, 0, 1],  # non-somatic MUA
}
# the metric behind each of bombcell's histograms, by the x label it gives it
HISTOGRAM_METRICS = {
    "# peaks": "nPeaks",
    "# troughs": "nTroughs",
    "baseline flatness": "waveformBaselineFlatness",
    "waveform duration": "waveformDuration_peakTrough",
    "peak_2/trough": "scndPeakToTroughRatio",
    "spatial decay": "spatialDecaySlope",
    "peak_1/peak_2": "peak1ToPeak2Ratio",
    "peak_{main}/trough": "mainPeakToTroughRatio",
    "amplitude": "rawAmplitude",
    "signal/noise (SNR)": "signalToNoiseRatio",
    "refractory period viol. (RPV)": "fractionRPVs_estimatedTauR",
    "# spikes": "nSpikes",
    "presence ratio": "presenceRatio",
    "% spikes missing": "percentageSpikesMissing_gaussian",
    "maximum drift": "maxDriftEstimate",
    "isolation dist.": "isolationDistance",
    "L-ratio": "Lratio",
}
BUTTON_COLORS = {  # the ipywidgets button_style palette
    "info": "#5bc0de",
    "success": "#5cb85c",
    "warning": "#f0ad4e",
    "danger": "#d9534f",
    "primary": "#337ab7",
}


def load_gui_inputs(ks_dir, param: dict, save_path=None):
    """bombcell's ``load_metrics_for_gui``, minus the PC features.

    Returns ``(ephys_data, raw_waveforms, param)``. ``param`` is a copy, with
    ``ephysKilosortPath`` filled in if it was missing so bombcell can find its
    precomputed ``for_GUI`` data.
    """
    from bombcell.loading_utils import handle_manual_curation

    ks_dir = Path(ks_dir)
    bc_dir = Path(save_path) if save_path is not None else ks_dir / "bombcell"
    param = dict(param)
    param.setdefault("ephysKilosortPath", str(ks_dir))

    spike_templates = np.load(ks_dir / "spike_templates.npy").squeeze()
    times_file = ks_dir / "spike_times_corrected.npy"
    if not times_file.exists():
        times_file = ks_dir / "spike_times.npy"
    spike_samples = np.load(times_file).squeeze()
    amplitudes = np.load(ks_dir / "amplitudes.npy").squeeze().astype(np.float64)

    whitened = np.load(ks_dir / "templates.npy")
    templates = (whitened @ np.load(ks_dir / "whitening_mat_inv.npy")).astype(whitened.dtype)
    pc_ind_file = ks_dir / "pc_feature_ind.npy"
    # only the (small) channel index is needed, for merged/split units
    pc_ind = np.load(pc_ind_file).squeeze() if pc_ind_file.exists() else np.nan
    spike_clusters, templates, _ = handle_manual_curation(
        ks_dir, spike_templates, templates, pc_ind
    )

    ephys_data = {
        "spike_times": spike_samples / param["ephys_sample_rate"],
        "spike_clusters": spike_clusters,
        "template_waveforms": templates,
        "template_amplitudes": amplitudes,
        "channel_positions": np.load(ks_dir / "channel_positions.npy").squeeze(),
    }

    raw_waveforms = None
    peak_channels = bc_dir / "templates._bc_rawWaveformPeakChannels.npy"
    for name, by_unit_id in (
        ("_bc_rawWaveforms_kilosort_format.npy", True),
        ("templates._bc_rawWaveforms.npy", False),
    ):
        if (bc_dir / name).exists() and peak_channels.exists():
            raw_waveforms = {
                "average": np.load(bc_dir / name, allow_pickle=True),
                "peak_channels": np.load(peak_channels, allow_pickle=True),
                "indexed_by_unit_id": by_unit_id,
            }
            break
    return ephys_data, raw_waveforms, param


def _gaussian_cut(x, a, x0, sigma, xcut):
    """bombcell's cutoff Gaussian: a Gaussian, zeroed below ``xcut``."""
    g = a * np.exp(-(x - x0) ** 2 / (2 * sigma**2))
    g[x < xcut] = 0
    return g


def _fit_cut_gaussian(bin_centers, hist_counts, p0, bounds):
    """bombcell's GUI amplitude fit, exactly; ``None`` where it gives up.

    Module-level so a worker process can run it.
    """
    from scipy.optimize import curve_fit

    try:
        popt, _ = curve_fit(_gaussian_cut, bin_centers, hist_counts, p0=p0,
                            bounds=bounds, maxfev=5000)
    except Exception:
        return None
    return popt


class _Value:
    """An ipywidget's ``.value``, backed by a getter and a setter.

    bombcell's navigation methods all end in ``self.unit_slider.value = i``;
    standing one of these in for the slider lets them drive Qt widgets as-is.
    """

    def __init__(self, get, set_):
        self._get, self._set = get, set_

    @property
    def value(self):
        return self._get()

    @value.setter
    def value(self, v):
        self._set(v)


class FastUnitQualityGUI(InteractiveUnitQualityGUI):
    """bombcell's ``InteractiveUnitQualityGUI``, without the per-click rescans.

    ``window`` is ``"qt"`` (controls and figure in one Qt window; needs
    ``%matplotlib qt``), ``"notebook"`` (bombcell's ipywidgets controls and an
    inline figure), or ``"auto"`` (Qt if that is the active backend). ``dpi``
    overrides the Qt figure's resolution, which is otherwise picked so the
    whole figure fits the screen; ``notebook_dpi`` is the inline one, which
    bombcell leaves at the default and so renders a 3000 x 2500 image per unit.
    ``prefetch=False`` fits each unit's amplitudes when it is shown rather
    than in a background process beforehand.

    Everything else is passed to bombcell's GUI unchanged.
    """

    def __init__(
        self,
        ephys_data,
        quality_metrics,
        *,
        window: str = "auto",
        dpi: float | None = None,
        notebook_dpi: float = 50,
        prefetch: bool = True,
        **kwargs,
    ):
        self._full_ephys = ephys_data
        self._fits = {}  # unit_id -> popt, None (fit failed), or a Future
        self._prefetch_on = prefetch
        self._pool = None
        self._index_cache = OrderedDict()
        self._view = (None, None)
        self._locations = None
        self._location_ax = None
        self._click_canvas = None
        self._shown_idx = None
        self._hist = None  # the persistent histogram panel, in the Qt window
        self._qt = _resolve_window(window)
        self._dpi = dpi
        self._notebook_dpi = notebook_dpi
        self.fig = None
        super().__init__(ephys_data, quality_metrics, **kwargs)  # draws the first unit

    # ── one unit's spikes, found once ────────────────────────────────────
    def _spike_index(self, unit_id) -> np.ndarray:
        cache = self._index_cache
        if unit_id in cache:
            cache.move_to_end(unit_id)
            return cache[unit_id]
        idx = np.flatnonzero(self._full_ephys["spike_clusters"] == unit_id)
        cache[unit_id] = idx
        if len(cache) > 32:  # a few hundred MB at most, for a long session
            cache.popitem(last=False)
        return idx

    @contextmanager
    def _scoped_to(self, unit_id):
        """``self.ephys_data`` restricted to one unit's spikes, for the duration.

        bombcell's panels each find the unit's spikes with
        ``ephys_data["spike_clusters"] == unit_id``; handed only that unit's
        spikes, that mask is all-true and costs nothing.
        """
        if self._view[0] != unit_id:
            idx = self._spike_index(unit_id)
            view = dict(self._full_ephys)
            for key in PER_SPIKE_KEYS:
                if key in view:
                    view[key] = view[key][idx]
            self._view = (unit_id, view)
        previous = self.ephys_data
        self.ephys_data = self._view[1]
        try:
            yield
        finally:
            self.ephys_data = previous

    def get_unit_data(self, unit_idx):
        if unit_idx >= self.n_units:
            return None
        with self._scoped_to(self.unique_units[unit_idx]):
            return super().get_unit_data(unit_idx)

    def update_display(self):
        busy = self._qt
        if busy:
            from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets

            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        try:
            with self._scoped_to(self.unique_units[self.current_unit_idx]):
                super().update_display()
            self._shown_idx = self.current_unit_idx
        finally:
            if busy:
                QtWidgets.QApplication.restoreOverrideCursor()
        # "next" is the usual move, and auto-advance's; start on its fit now
        nxt = self.current_unit_idx + 1
        if self._qt:  # after the canvas has painted, not before
            from matplotlib.backends.qt_compat import QtCore

            QtCore.QTimer.singleShot(50, lambda: self._prefetch(nxt))
        else:
            self._prefetch(nxt)

    # ── the amplitude fit: bombcell's, cached and fitted ahead ───────────
    def _amplitude_fit_inputs(self, unit_data):
        """The amplitudes bombcell's panel fits, and ``(bin_centers, counts, p0, bounds, bin_width)``.

        The second is ``None`` when there are too few spikes to fit.
        """
        spike_times = unit_data["spike_times"]
        metrics = unit_data["metrics"]
        amplitudes = self._full_ephys["template_amplitudes"][self._spike_index(unit_data["unit_id"])]
        if self.param and self.param.get("computeTimeChunks", False):
            starts = metrics.get("useTheseTimesStart", None)
            stops = metrics.get("useTheseTimesStop", None)
            if starts is not None and stops is not None:
                keep = np.zeros(len(spike_times), dtype=bool)
                for start, stop in zip(np.atleast_1d(starts), np.atleast_1d(stops)):
                    if not (np.isnan(start) or np.isnan(stop)):
                        keep |= (spike_times >= start) & (spike_times <= stop)
                amplitudes = amplitudes[keep]
        if len(amplitudes) <= 10:
            return amplitudes, None

        hist_counts, bin_edges = np.histogram(amplitudes, bins=min(50, int(len(amplitudes) / 10)))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        lo, hi, median = np.min(amplitudes), np.max(amplitudes), np.median(amplitudes)
        p0 = [np.max(hist_counts), median, np.std(amplitudes), lo]
        bounds = ([0, lo, 0, lo], [np.inf, hi, np.ptp(amplitudes), median])
        return amplitudes, (bin_centers, hist_counts, p0, bounds, bin_edges[1] - bin_edges[0])

    def _cut_gaussian_fit(self, unit_id, inputs):
        """The fit for ``unit_id``: from the cache, the background worker, or here."""
        entry = self._fits.get(unit_id)
        if isinstance(entry, Future):
            # finished, or already running (so its answer comes soonest); a
            # queued one is cancelled and fitted here instead
            if entry.done() or not entry.cancel():
                try:
                    self._fits[unit_id] = entry.result()
                    return self._fits[unit_id]
                except Exception:  # the worker died; fit it here
                    pass
        elif unit_id in self._fits:
            return entry  # fitted already (None if it failed)
        self._fits[unit_id] = _fit_cut_gaussian(*inputs[:4])
        return self._fits[unit_id]

    def _prefetch(self, idx):
        """Start fitting unit ``idx``'s amplitudes in a worker process."""
        if not self._prefetch_on or idx >= self.n_units:
            return
        unit_id = self.unique_units[idx]
        if unit_id in self._fits:
            return
        _, inputs = self._amplitude_fit_inputs(self.get_unit_data(idx))
        if inputs is None:
            return
        try:
            if self._pool is None:
                from concurrent.futures import ProcessPoolExecutor

                self._pool = ProcessPoolExecutor(max_workers=1)
            self._fits[unit_id] = self._pool.submit(_fit_cut_gaussian, *inputs[:4])
        except Exception:  # no worker processes here: fit on demand instead
            self._prefetch_on = False

    def close(self):
        """Stop the background fitting process (the window closing does this too)."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def plot_amplitude_fit(self, ax, unit_data):
        """bombcell's scaling-factor distribution panel, with the fit from :meth:`_cut_gaussian_fit`."""
        from scipy.stats import norm

        font = dict(fontsize=13, fontfamily="DejaVu Sans")
        amp_ylim = getattr(self, "_amplitude_ylim", None)

        def note(text, **kw):
            ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, **kw)

        if not len(unit_data["spike_times"]):
            note("No spike data\navailable", fontfamily="DejaVu Sans")
        elif "template_amplitudes" not in self._full_ephys:
            note("No scaling factor data\navailable")
        else:
            amplitudes, inputs = self._amplitude_fit_inputs(unit_data)
            if inputs is None:
                note("Insufficient data\nfor scaling factor fit")
            else:
                bin_centers, hist_counts, _, _, bin_width = inputs
                ax.barh(bin_centers, hist_counts, height=bin_width * 0.8,
                        facecolor="grey", edgecolor="black")
                popt = self._cut_gaussian_fit(unit_data["unit_id"], inputs)
                if popt is None:
                    note("Fit failed")
                else:
                    y_smooth = np.linspace(np.min(amplitudes), np.max(amplitudes), 200)
                    ax.plot(_gaussian_cut(y_smooth, *popt), y_smooth, "r-", linewidth=2)
                    percent_missing = 100 * (1 - norm.cdf((popt[1] - popt[3]) / popt[2]))
                    ax.text(0.5, 0.98, f"{percent_missing:.1f}", transform=ax.transAxes,
                            va="top", ha="center", color=[0.7, 0.7, 0.7], fontsize=13,
                            weight="bold")
                ax.set_xlabel("count", **font)
                ax.set_ylabel("Scaling factor", **font)
                ax.tick_params(labelsize=13)
                if amp_ylim is not None:
                    ax.set_ylim(amp_ylim)

        ax.set_title("Scaling factor \n distribution", fontsize=15, fontweight="bold",
                     fontfamily="DejaVu Sans")
        self.add_metrics_text(ax, unit_data, "amplitude_fit")

    # ── the units-by-depth panel, from numbers computed once ─────────────
    def _location_data(self):
        """Depth, log firing rate and color of every unit, as bombcell computes them."""
        if self._locations is not None:
            return self._locations

        clusters = self._full_ephys["spike_clusters"]
        times = self._full_ephys["spike_times"]
        positions = self._full_ephys["channel_positions"]
        n = int(max(clusters.max(), self.unique_units.max())) + 1
        counts = np.bincount(clusters, minlength=n)
        first = np.full(n, np.inf)
        last = np.full(n, -np.inf)
        np.minimum.at(first, clusters, times)
        np.maximum.at(last, clusters, times)

        rows = []
        for i, unit_id in enumerate(self.unique_units):
            if unit_id >= len(self.all_max_channels):
                continue
            max_ch = int(self.all_max_channels[unit_id])
            duration = last[unit_id] - first[unit_id]
            if max_ch >= len(positions) or not counts[unit_id] or not duration > 0:
                continue
            rate = max(counts[unit_id] / duration, 0.01)  # bombcell's floor, for the log
            code = None if self.bombcell_unit_types is None else self.bombcell_unit_types[i]
            rows.append((i, np.log10(rate), positions[max_ch, 1], TYPE_COLORS.get(code, TYPE_COLORS[1])))

        idx, log_rate, depth, color = zip(*rows) if rows else ((), (), (), ())
        self._locations = {
            "unit_idx": np.array(idx, dtype=int),
            "log_rate": np.array(log_rate),
            "depth": np.array(depth),
            "color": np.array(color).reshape(-1, 3),
        }
        return self._locations

    def plot_unit_location(self, ax, unit_data):
        """bombcell's panel, drawn from :meth:`_location_data`: same look, one scatter."""
        import matplotlib.pyplot as plt

        loc = self._location_data()
        ax.set_title("Units by depth", fontsize=15, fontweight="bold", fontfamily="DejaVu Sans")
        if not len(loc["unit_idx"]):
            ax.text(0.5, 0.5, "No units with\nvalid locations", ha="center", va="center",
                    transform=ax.transAxes)
            return

        current = loc["unit_idx"] == self.current_unit_idx
        ax.scatter(loc["log_rate"][~current], loc["depth"][~current],
                   c=loc["color"][~current], s=30, alpha=0.7, zorder=5)
        ax.scatter(loc["log_rate"][current], loc["depth"][current],
                   c=loc["color"][current], s=80, edgecolors="black", linewidths=2, zorder=10)
        ax.set_xlabel("Log₁₀ firing rate (sp/s)", fontsize=13, fontfamily="DejaVu Sans")
        ax.set_ylabel("Depth from tip of probe (μm)", fontsize=13, fontfamily="DejaVu Sans")
        ax.tick_params(labelsize=13)

        ylim, xlim = ax.get_ylim(), ax.get_xlim()
        x_range, y_range = xlim[1] - xlim[0], ylim[1] - ylim[0]
        arrow_x = xlim[0] - x_range * 1.2
        arrow_start_y = ylim[0] + y_range * 0.05
        arrow_end_y = ylim[1] - y_range * 0.05
        ax.annotate("", xy=(arrow_x, arrow_end_y), xytext=(arrow_x, arrow_start_y),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=2), annotation_clip=False)
        label_x = arrow_x - x_range * 0.02
        ax.text(label_x, arrow_start_y - y_range * 0.01, "deepest = tip \n of the probe",
                ha="center", va="top", fontsize=16, fontfamily="DejaVu Sans", clip_on=False,
                fontweight="bold")
        ax.text(label_x, arrow_end_y + y_range * 0.02, "most superficial", ha="center",
                va="bottom", fontsize=16, fontfamily="DejaVu Sans", clip_on=False,
                fontweight="bold")
        legend = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=TYPE_COLORS[code],
                       markersize=8, label=name)
            for code, name in ((1, "good"), (2, "mua"), (0, "noise"), (3, "non-somatic"))
        ]
        ax.legend(handles=legend, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=13)

        # one handler per canvas: in Qt the figure is redrawn, not replaced,
        # and bombcell's connect-per-draw would pile up a handler per unit
        self._location_ax = ax
        if self._click_canvas is not ax.figure.canvas:
            ax.figure.canvas.mpl_connect("button_press_event", self._on_location_click)
            self._click_canvas = ax.figure.canvas

    def _on_location_click(self, event):
        """Jump to the unit nearest a click on the units-by-depth panel."""
        ax, loc = self._location_ax, self._locations
        if event.inaxes is not ax or event.xdata is None or not len(loc["unit_idx"]):
            return
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        dist = np.hypot((loc["log_rate"] - event.xdata) / (x1 - x0),
                        (loc["depth"] - event.ydata) / (y1 - y0))
        nearest = int(np.argmin(dist))
        if dist[nearest] < 0.1:
            target = int(loc["unit_idx"][nearest])
            # not from inside the canvas's own event: the redraw clears the
            # axes this click is still being delivered to
            self._defer(lambda: setattr(self.unit_slider, "value", target))

    def _defer(self, fn):
        if self._qt:
            from matplotlib.backends.qt_compat import QtCore

            QtCore.QTimer.singleShot(0, fn)
        else:
            fn()

    # ── drawing: bombcell's landscape layout, into a figure it is handed ─
    def plot_unit(self, unit_idx):
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        unit_data = self.get_unit_data(unit_idx)
        if unit_data is None:
            return
        if self._qt:
            self._draw_landscape(self.fig, unit_data)
            self.fig.canvas.draw()
            return
        with self.plot_output:
            clear_output(wait=True)
            if self.fig is not None:
                plt.close(self.fig)  # only ours -- bombcell closes every figure
            self.fig = plt.figure(figsize=FIG_SIZE, dpi=self._notebook_dpi)
            self._draw_landscape(self.fig, unit_data)
            plt.show()

    def _draw_landscape(self, fig, unit_data):
        """``InteractiveUnitQualityGUI._plot_unit_landscape``, drawing into ``fig``.

        In the Qt window the figure outlives the unit, and the histogram panel
        -- the same fourteen histograms for every unit, bar a marker -- is
        drawn once; later units clear and redraw only their own panels and
        move the markers.
        """
        import matplotlib.pyplot as plt

        reuse = self._hist is not None and self._hist["fig"] is fig
        if reuse:
            for ax in [a for a in fig.axes if a not in self._hist["axes"]]:
                ax.remove()
        else:
            fig.clear()
            fig.patch.set_facecolor("white")
        plt.figure(fig)  # the histogram panel draws into the current figure

        def cell(loc, rowspan, colspan, **kw):
            return plt.subplot2grid((100, 30), loc, rowspan=rowspan, colspan=colspan, fig=fig, **kw)

        self.plot_unit_location(cell((0, 0), 100, 1), unit_data)
        self.plot_template_waveform(cell((0, 2), 20, 6), unit_data)
        self.plot_raw_waveforms(cell((0, 9), 20, 6), unit_data)
        self.plot_spatial_decay(cell((30, 2), 20, 6), unit_data)
        self.plot_autocorrelogram(cell((30, 9), 20, 6), unit_data)
        ax_amplitude = cell((60, 2), 20, 10)
        self.plot_amplitudes_over_time(ax_amplitude, unit_data)
        _scatter_to_markers(ax_amplitude)
        self.plot_time_bin_metrics(cell((85, 2), 10, 10, sharex=ax_amplitude), unit_data)
        self.plot_amplitude_fit(cell((60, 13), 20, 2), unit_data)
        unit_axes = list(fig.axes)
        if reuse:
            unit_axes = [a for a in unit_axes if a not in self._hist["axes"]]
            self._move_histogram_markers()
        else:
            self.plot_histograms_panel(fig, unit_data)
            if self._qt:
                self._adopt_histograms(fig, [a for a in fig.axes if a not in unit_axes])
        fig.subplots_adjust(left=0.03, right=0.98, top=0.99, bottom=0.08, hspace=0.4, wspace=0.4)
        self._format_axes(unit_axes if reuse else fig.get_axes())

    def _adopt_histograms(self, fig, axes):
        """Keep the histogram panel, and swap bombcell's markers for ones that can move."""
        markers = []
        for ax in axes:
            metric = HISTOGRAM_METRICS.get(ax.get_xlabel())
            bars = list(ax.patches)
            if metric is None or not bars:
                continue
            for collection in list(ax.collections):  # bombcell's one marker
                collection.remove()
            edges = np.array([b.get_x() for b in bars] + [bars[-1].get_x() + bars[-1].get_width()])
            heights = np.array([b.get_height() for b in bars])
            marker = ax.scatter([np.nan], [np.nan], marker="v", s=500, color="black", alpha=1.0,
                                zorder=15, edgecolors="white", linewidths=4)
            markers.append((metric, edges, heights, marker))
        self._hist = {"fig": fig, "axes": set(axes), "markers": markers}
        self._move_histogram_markers()

    def _move_histogram_markers(self):
        """Put each histogram's marker over the current unit, as bombcell places it."""
        idx = self.current_unit_idx
        for metric, edges, heights, marker in self._hist["markers"]:
            values = np.asarray(self.quality_metrics[metric], dtype=float)
            value = values[idx] if idx < len(values) else np.nan
            if np.isnan(value):
                marker.set_visible(False)
                continue
            b = np.digitize(value, edges) - 1
            height = heights[b] if 0 <= b < len(heights) else 0.5
            marker.set_offsets([[value, height + 0.15]])
            marker.set_visible(True)

    @staticmethod
    def _format_axes(axes):
        """bombcell's font and tick pass, by axes creation order."""
        AXIS_LABEL_FONTSIZE = 20
        TICK_LABEL_FONTSIZE = 14
        LEGEND_FONTSIZE = 16
        PLOT_TITLE_FONTSIZE = 22

        for i, ax in enumerate(axes):
            if ax.get_position().height < 0.05:
                continue
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=PLOT_TITLE_FONTSIZE, fontweight="bold")
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=AXIS_LABEL_FONTSIZE, labelpad=1)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=AXIS_LABEL_FONTSIZE, labelpad=1)
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
            if i in (1, 2, 3, 5, 6):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
            else:
                for lim, set_ticks, set_labels in (
                    (ax.get_xlim(), ax.set_xticks, ax.set_xticklabels),
                    (ax.get_ylim(), ax.set_yticks, ax.set_yticklabels),
                ):
                    set_ticks([lim[0], lim[1]])
                    set_labels(
                        [f"{v:.2f}" if 0.01 < abs(v) < 1000 else f"{v:.0f}" for v in lim],
                        fontsize=TICK_LABEL_FONTSIZE,
                    )
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(LEGEND_FONTSIZE)

    # ── navigation ───────────────────────────────────────────────────────
    def goto_unit_number(self, b=None):
        """bombcell's, without its second redraw of the same unit."""
        requested = int(self.unit_input.value)
        ids = self.unique_units
        if requested in ids:
            idx = int(np.flatnonzero(ids == requested)[0])
        else:
            above = np.flatnonzero(ids >= requested)
            idx = int(above[0]) if len(above) else self.n_units - 1
            self._say(f"Unit {requested} doesn't exist (no spikes); showing unit {ids[idx]} instead.")
        self.current_unit_idx = idx
        self.unit_slider.value = idx

    def classify_unit(self, classification):
        unit_id = self.unique_units[self.current_unit_idx]
        super().classify_unit(classification)  # prints its own summary to the notebook
        if self._qt:
            names = {0: "noise", 1: "good", 2: "MUA", 3: "non-somatic"}
            done = int(np.sum(self.manual_unit_types != -1))
            self._say(f"Unit {unit_id} marked {names.get(classification, classification)} "
                      f"-- {done}/{self.n_units} classified")

    def _say(self, message):
        if self._qt:
            self._window.statusBar().showMessage(message, 8000)
        else:
            print(message)

    def _go_to(self, idx):
        """Qt: show unit ``idx`` and bring the controls into line with it."""
        idx = int(min(max(int(idx), 0), self.n_units - 1))
        self.current_unit_idx = idx
        for widget, value in ((self._slider, idx), (self._spin, int(self.unique_units[idx]))):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        if idx != self._shown_idx:
            self.update_display()

    # ── the controls ─────────────────────────────────────────────────────
    def setup_widgets(self):
        if not self._qt:
            super().setup_widgets()
        # Qt: built with the window, in display_gui

    def display_gui(self):
        if not self._qt:
            return super().display_gui()
        self._build_qt_window()
        self.update_display()

    def _build_qt_window(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is None:  # %matplotlib qt makes one; a plain script might not have
            from matplotlib.backends.backend_qt import _create_qApp

            app = _create_qApp()
        screen = app.primaryScreen().availableGeometry()
        dpi = self._dpi or max(30.0, min(0.97 * screen.width() / FIG_SIZE[0],
                                         (0.95 * screen.height() - CONTROLS_PX) / FIG_SIZE[1]))

        plt.close(WINDOW_TITLE)  # re-running the cell replaces the window
        self.fig = plt.figure(WINDOW_TITLE, figsize=FIG_SIZE, dpi=dpi)
        canvas = self.fig.canvas
        canvas.mpl_connect("close_event", lambda event: self.close())
        window = self._window = canvas.manager.window

        def button(text, style, slot, width=None):
            b = QtWidgets.QPushButton(text)
            color = QtGui.QColor(BUTTON_COLORS[style])
            b.setStyleSheet(
                f"QPushButton {{background: {color.name()}; color: white; font-weight: bold;"
                f" border-radius: 3px; padding: 5px 10px;}}"
                f"QPushButton:pressed {{background: {color.darker(130).name()};}}"
            )
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # arrow keys stay with the window
            if width:
                b.setMinimumWidth(width)
            b.clicked.connect(lambda *_: slot())
            return b

        def row(*items):
            layout = QtWidgets.QHBoxLayout()
            layout.addStretch(1)
            for item in items:
                if isinstance(item, str):
                    label = QtWidgets.QLabel(item)
                    label.setStyleSheet("font-weight: bold; padding: 0 6px;")
                    layout.addWidget(label)
                else:
                    layout.addWidget(item)
            layout.addStretch(1)
            return layout

        self._title = QtWidgets.QLabel()
        self._title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._title.setTextFormat(QtCore.Qt.TextFormat.RichText)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(0, self.n_units - 1)
        self._slider.setTracking(False)  # one redraw on release, not one per pixel dragged
        self._slider.setMinimumWidth(400)
        self._slider.valueChanged.connect(self._go_to)
        self._spin = QtWidgets.QSpinBox()
        self._spin.setRange(0, int(self.unique_units.max()))
        self._spin.lineEdit().returnPressed.connect(self.goto_unit_number)

        controls = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(controls)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(3)
        layout.addWidget(self._title)
        layout.addLayout(row(
            button("◀", "info", self.prev_unit, 60), button("▶", "info", self.next_unit, 60),
            "unit", self._slider, "go to ID:", self._spin,
            button("Go", "primary", self.goto_unit_number),
        ))
        layout.addLayout(row(
            "bombcell type:",
            button("◀ good", "success", self.goto_prev_good),
            button("good ▶", "success", self.goto_next_good),
            button("◀ MUA", "warning", self.goto_prev_mua),
            button("MUA ▶", "warning", self.goto_next_mua),
            button("◀ non-somatic", "primary", self.goto_prev_nonsomatic),
            button("non-somatic ▶", "primary", self.goto_next_nonsomatic),
            button("◀ noise", "danger", self.goto_prev_noise),
            button("noise ▶", "danger", self.goto_next_noise),
        ))
        layout.addLayout(row(
            button("▶ next unclassified", "info", self.goto_next_unclassified),
            "manual classification:",
            button("mark as good", "success", lambda: self.classify_unit(1)),
            button("mark as MUA", "warning", lambda: self.classify_unit(2)),
            button("mark as non-somatic", "primary", lambda: self.classify_unit(3)),
            button("mark as noise", "danger", lambda: self.classify_unit(0)),
        ))

        # the figure's own window: controls above the canvas, toolbar kept
        window.takeCentralWidget()  # detaches the canvas without deleting it
        central = QtWidgets.QWidget()
        stack = QtWidgets.QVBoxLayout(central)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)
        stack.addWidget(controls)
        stack.addWidget(canvas, 1)
        window.setCentralWidget(central)
        window.statusBar().showMessage("← → previous/next unit;  click a unit in 'Units by depth' to jump to it")

        self._shortcuts = []
        for key, slot in (("Left", self.prev_unit), ("Right", self.next_unit)):
            shortcut = _qshortcut(QtGui, QtWidgets)(QtGui.QKeySequence(key), window)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

        # bombcell's methods set .value on these; route that to the Qt widgets
        self.unit_slider = _Value(self._slider.value, self._go_to)
        self.unit_input = _Value(self._spin.value, self._spin.setValue)
        self.unit_info = _Value(self._title.text, self._title.setText)

        width = int(FIG_SIZE[0] * dpi) + 20
        height = int(FIG_SIZE[1] * dpi) + CONTROLS_PX
        window.resize(min(width, screen.width()), min(height, screen.height()))
        window.show()
        window.raise_()
        window.activateWindow()


def _scatter_to_markers(ax, min_points: int = 5000):
    """Redraw big one-color scatters as marker-only lines: same look, a fraction of the draw.

    A scatter draws every point as its own path; a line's markers are stamped.
    For the amplitude-over-time panel -- one dot per spike, hundreds of
    thousands of them -- that is most of the figure's draw time.
    """
    from matplotlib.collections import PathCollection

    for coll in list(ax.collections):
        if not isinstance(coll, PathCollection) or len(coll.get_offsets()) < min_points:
            continue
        sizes, faces = coll.get_sizes(), coll.get_facecolor()
        if len(sizes) != 1 or len(faces) != 1:  # per-point sizes or colors: leave it be
            continue
        xy = np.asarray(coll.get_offsets())
        ax.plot(xy[:, 0], xy[:, 1], linestyle="none", marker="o",
                markersize=np.sqrt(sizes[0]), markerfacecolor=faces[0],
                markeredgecolor="none", zorder=coll.get_zorder())
        coll.remove()


def _qshortcut(QtGui, QtWidgets):
    """QShortcut moved from QtWidgets (Qt5) to QtGui (Qt6)."""
    return getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut


def _resolve_window(window: str) -> bool:
    """True for the Qt window, False for the notebook one."""
    import matplotlib

    on_qt = "qt" in matplotlib.get_backend().lower()
    if window == "auto":
        return on_qt
    if window == "qt":
        if not on_qt:
            raise ValueError(
                f"window='qt' needs the Qt backend, but it is {matplotlib.get_backend()!r}. "
                "Run `%matplotlib qt` in its own cell first."
            )
        return True
    if window == "notebook":
        return False
    raise ValueError(f"window must be 'auto', 'qt' or 'notebook', not {window!r}")


def unit_quality_gui(
    ks_dir,
    quality_metrics,
    unit_types=None,
    param=None,
    save_path=None,
    ephys_properties=None,
    auto_advance: bool = True,
    window: str = "auto",
    dpi: float | None = None,
    prefetch: bool = True,
) -> FastUnitQualityGUI:
    """Drop-in for ``bombcell.unit_quality_gui(ks_dir=..., ...)``; see :class:`FastUnitQualityGUI`.

    Run ``%matplotlib qt`` in a cell before this one to get the whole GUI in
    one Qt window; under the inline backend it keeps bombcell's notebook
    controls. Manual classifications are read from and saved to
    ``save_path`` exactly as bombcell's GUI does.
    """
    if param is None:
        raise ValueError("param is required: it holds the sample rate and the thresholds")
    if save_path is None:
        save_path = ks_dir  # bombcell's default
    ephys_data, raw_waveforms, param = load_gui_inputs(ks_dir, param, save_path)
    return FastUnitQualityGUI(
        ephys_data,
        quality_metrics,
        ephys_properties=ephys_properties,
        raw_waveforms=raw_waveforms,
        param=param,
        unit_types=unit_types,
        save_path=str(save_path),
        auto_advance=auto_advance,
        window=window,
        dpi=dpi,
        prefetch=prefetch,
    )
