"""Interactive brain-state editor, after buzcode's StateEditor.

The automatic scoring is a starting point (see spikeshpc/states.py), and the
paper it follows curates its output by hand. This is where that happens:
drag a threshold and every bin re-classifies live, paint a span with a state
when the signals disagree with you, and save the result back.

Two things it deliberately does that the automatic pass cannot:

  * shows all four signals against a *shared* hypnogram, so a call and the
    evidence for it are on screen together
  * keeps manual edits separate from the automatic classification, so moving
    a threshold afterwards does not silently discard them

Needs an interactive matplotlib backend (``%matplotlib widget`` or
``%matplotlib qt``) and keyboard focus on the figure -- click it once.
"""

import json
import shutil
from pathlib import Path

import numpy as np

from .optitrack.widgets._backend import warn_if_noninteractive_backend
from .states import (
    STATE_CODES,
    apply_movement_veto,
    classify_states,
    enforce_min_duration,
)

STATE_COLORS = {"WAKE": "#d95f02", "NREM": "#1b9e77", "REM": "#7570b3"}
# which signal each threshold belongs to, and which way "above" points
THRESHOLD_OF = {
    "broadband LFP (PC1)": "broadband",
    "theta ratio": "theta",
    "EMG (LFP correlation)": "emg",
}
PAINT_KEYS = {"1": "WAKE", "w": "WAKE", "3": "NREM", "n": "NREM",
              "5": "REM", "r": "REM", "0": None, "x": None}

HELP = (
    "drag threshold lines to re-classify  |  drag on a trace to select  |  "
    "1/3/5 or w/n/r paint  0 clear  |  a/d pan  z/Z zoom  |  "
    "u undo  R reset  S save  h help"
)


class StateEditor:
    """Full-session state editor with live thresholds and manual painting."""

    def __init__(self, scoring, window_s: float = 600.0, log_signals=("movement (mm/s)",),
                 robust=(0.5, 99.5)):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

        warn_if_noninteractive_backend()
        self.scoring = scoring
        self.times = np.asarray(scoring.times, dtype=float)
        self.step_s = float(scoring.step_s)
        self.signals = scoring.signals()
        if not self.signals:
            raise ValueError(
                f"{scoring.session!r} has no per-bin signals saved; the "
                "_metrics.npz next to the states JSON is what holds them."
            )

        self.thresholds = dict(scoring.thresholds)
        movement = (scoring.metadata or {}).get("movement") or {}
        self._veto = movement if movement.get("applied") else None
        self.min_duration_s = float(
            (scoring.metadata or {}).get("min_state_duration_s", 6.0)
        )
        # manual labels override the automatic ones; -1 means "not set"
        self.manual = np.full(len(self.times), -1, dtype=np.int16)
        self.codes = np.asarray(scoring.codes, dtype=np.int16).copy()
        self._history = []
        self._select = None
        self._selecting = False
        self._drag = None

        self.window_s = float(window_s)
        self.t0 = float(self.times[0])

        n = len(self.signals)
        self.fig = plt.figure(figsize=(13, 1.9 * n + 2.2))
        gs = GridSpec(n + 2, 2, figure=self.fig, width_ratios=[6, 1],
                      height_ratios=[0.7, 0.35] + [1] * n, hspace=0.12,
                      wspace=0.03, left=0.08, right=0.98, top=0.94, bottom=0.07)

        # column 0 only: spanning both columns would make these wider than
        # the traces and put their time axis out of register with them
        self.ax_overview = self.fig.add_subplot(gs[0, 0])
        self.ax_hypno = self.fig.add_subplot(gs[1, 0])
        self.axes, self.hists = [], []
        for i in range(n):
            # every time axis shares the hypnogram's, so they cannot drift apart
            ax = self.fig.add_subplot(gs[i + 2, 0], sharex=self.ax_hypno)
            self.axes.append(ax)
            self.hists.append(self.fig.add_subplot(gs[i + 2, 1], sharey=ax))

        self.lines, self.threshold_lines, self.hist_lines = [], [], []
        self.ylim = {}
        for ax, hx, label in zip(self.axes, self.hists, self.signals):
            (line,) = ax.plot([], [], lw=0.9, color="black")
            self.lines.append(line)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=8)

            values = np.asarray(self.signals[label], dtype=float)
            finite = values[np.isfinite(values)]
            key = THRESHOLD_OF.get(label)
            lo, hi = np.percentile(finite, robust) if finite.size else (0.0, 1.0)
            if key and self.thresholds.get(key) is not None:
                lo, hi = min(lo, self.thresholds[key]), max(hi, self.thresholds[key])
            pad = 0.06 * ((hi - lo) or abs(hi) or 1.0)
            lo, hi = lo - pad, hi + pad
            if finite.size and finite.min() >= 0:
                lo = max(lo, 0.0)
            if label in log_signals:
                ax.set_yscale("symlog", linthresh=1.0)
                hx.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(lo, hi)
            self.ylim[label] = (lo, hi)

            # the distribution the threshold is meant to split. Linear bins
            # under a log axis pile everything into the bottom bar.
            if finite.size:
                if label in log_signals:
                    positive = finite[finite > 0]
                    bins = np.logspace(
                        np.log10(max(positive.min(), 1e-3)),
                        np.log10(max(positive.max(), 1e-2)), 60,
                    ) if positive.size else 60
                else:
                    bins = np.linspace(lo, hi, 80)
                hx.hist(finite, bins=bins, orientation="horizontal", color="0.7")
            hx.set_xticks([])
            hx.tick_params(labelleft=False, labelsize=7)

            value = self.thresholds.get(key) if key else None
            if value is None:
                self.threshold_lines.append(None)
                self.hist_lines.append(None)
            else:
                self.threshold_lines.append(
                    ax.axhline(value, ls="--", lw=1.2, color="tab:red",
                               alpha=0.9, picker=True)
                )
                self.hist_lines.append(
                    hx.axhline(value, ls="--", lw=1.2, color="tab:red", alpha=0.9)
                )

        for ax in self.axes[:-1]:
            ax.tick_params(labelbottom=False)
        self.axes[-1].set_xlabel("time (s)")
        self.viewport = self.ax_overview.axvspan(
            0, 0, color="tab:blue", alpha=0.20, zorder=3
        )
        self.selection = [
            ax.axvspan(0, 0, color="tab:blue", alpha=0.0, zorder=0) for ax in self.axes
        ]
        # matplotlib has no rich text, so the title is assembled from
        # separate pieces: the state names have to carry the hypnogram's
        # colours to be readable against it at a glance
        self._title_parts = {}
        pieces = [self._title_part("head", "")]
        for name in STATE_CODES:
            pieces.append(self._title_part(name, "", color=STATE_COLORS[name],
                                           weight="bold"))
        pieces.append(self._title_part("tail", ""))
        self.title = AnchoredOffsetbox(
            loc="upper center", frameon=False, pad=0.0, borderpad=0.15,
            bbox_to_anchor=(0.5, 1.0), bbox_transform=self.fig.transFigure,
            child=HPacker(children=pieces, pad=0, sep=10, align="baseline"),
        )
        self.fig.add_artist(self.title)
        self.help = self.fig.text(0.5, 0.005, HELP, ha="center", fontsize=7,
                                  color="gray")

        self._draw_overview()
        for name, cid in (("button_press_event", self._on_press),
                          ("button_release_event", self._on_release),
                          ("motion_notify_event", self._on_motion),
                          ("key_press_event", self._on_key)):
            self.fig.canvas.mpl_connect(name, cid)
        self._reclassify()

    def _title_part(self, key, text, color="black", weight="normal"):
        from matplotlib.offsetbox import TextArea

        area = TextArea(text, textprops=dict(color=color, fontsize=11,
                                             fontweight=weight))
        self._title_parts[key] = area
        return area

    # ── classification ───────────────────────────────────────────────────
    def _auto_codes(self):
        """Re-run the decision rules at the current thresholds.

        Including the movement veto, when the scoring was produced with one:
        re-classifying without it would silently undo every reassignment the
        moment a threshold is nudged.
        """
        get = self.signals.get
        broadband = get("broadband LFP (PC1)")
        theta = get("theta ratio")
        emg = get("EMG (LFP correlation)")
        if broadband is None or theta is None or emg is None:
            return np.asarray(self.scoring.codes, dtype=np.int16).copy()

        codes = classify_states(broadband, theta, emg, self.thresholds)
        codes = enforce_min_duration(codes, self.step_s, self.min_duration_s)
        if self._veto is not None:
            speed = self.signals.get("movement (mm/s)")
            if speed is not None:
                codes, _ = apply_movement_veto(
                    codes, speed,
                    threshold=self._veto["threshold_log10"],
                    step_s=self.step_s,
                    min_duration_s=self.min_duration_s,
                    veto=tuple(self._veto.get("vetoed_states", ("NREM", "REM"))),
                    verbose=False,
                )
        return np.asarray(codes, dtype=np.int16)

    def _push(self):
        """Snapshot the current state so it can be undone.

        Called before a change, never after: pushing afterwards records the
        new value and makes undo a no-op.
        """
        self._history.append((dict(self.thresholds), self.manual.copy()))
        del self._history[:-50]

    def _reclassify(self):
        auto = self._auto_codes()
        self.codes = np.where(self.manual >= 0, self.manual, auto).astype(np.int16)
        self._draw()

    # ── drawing ──────────────────────────────────────────────────────────
    def _hypnogram(self, ax, times, codes):
        ax.clear()
        for name, code in STATE_CODES.items():
            mask = codes == code
            if not mask.any():
                continue
            edges = np.flatnonzero(np.diff(mask.astype(int)))
            starts = np.r_[0, edges + 1][mask[np.r_[0, edges + 1]]]
            for a in starts:
                b = a
                while b + 1 < len(mask) and mask[b + 1]:
                    b += 1
                ax.axvspan(times[a] - self.step_s / 2, times[b] + self.step_s / 2,
                           color=STATE_COLORS[name], lw=0)
        ax.set_yticks([])
        ax.set_ylim(0, 1)

    def _draw_overview(self):
        self._hypnogram(self.ax_overview, self.times, self.codes)
        self.ax_overview.set_xlim(self.times[0], self.times[-1])
        self.ax_overview.set_ylabel("all", fontsize=8)
        # ticks above, or they collide with the hypnogram strip below
        self.ax_overview.xaxis.set_ticks_position("top")
        self.ax_overview.tick_params(labelsize=7, pad=1)
        self.viewport = self.ax_overview.axvspan(
            self.t0, self.t0 + self.window_s, color="tab:blue", alpha=0.25, zorder=3
        )

    def _draw(self):
        lo, hi = self.t0, self.t0 + self.window_s
        window = (self.times >= lo) & (self.times <= hi)
        for line, label in zip(self.lines, self.signals):
            line.set_data(self.times[window], self.signals[label][window])

        self._hypnogram(self.ax_hypno, self.times, self.codes)
        self.ax_hypno.set_ylabel("state", fontsize=8)
        self.ax_hypno.tick_params(labelbottom=False)
        self.ax_hypno.set_xlim(lo, hi)   # shared, so the traces follow

        for ax, label in zip(self.axes, self.signals):
            ax.set_ylim(*self.ylim[label])

        try:
            self.viewport.remove()
        except (ValueError, AttributeError):
            pass
        self.viewport = self.ax_overview.axvspan(
            lo, hi, color="tab:blue", alpha=0.25, zorder=3
        )

        edited = int((self.manual >= 0).sum())
        self._title_parts["head"].set_text(
            f"{self.scoring.session}   {lo:.0f}-{hi:.0f}s"
        )
        for name, code in STATE_CODES.items():
            self._title_parts[name].set_text(
                f"{name} {np.mean(self.codes == code):.1%}"
            )
        thresholds = "  ".join(
            f"{k}={v:.3f}" for k, v in sorted(self.thresholds.items()) if v is not None
        )
        self._title_parts["tail"].set_text(
            f"|   {thresholds}" + (f"   |   {edited} bins edited" if edited else "")
        )
        self.fig.canvas.draw_idle()

    # ── interaction ──────────────────────────────────────────────────────
    def _threshold_at(self, event):
        """The threshold line under the cursor, if any."""
        for i, (ax, line) in enumerate(zip(self.axes, self.threshold_lines)):
            if line is None or event.inaxes is not ax:
                continue
            lo, hi = ax.get_ylim()
            if abs(event.ydata - line.get_ydata()[0]) < 0.04 * abs(hi - lo):
                return i
        return None

    def _on_press(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        if event.inaxes is self.ax_overview:
            self.t0 = float(event.xdata) - self.window_s / 2
            self._clamp()
            self._draw()
            return
        hit = self._threshold_at(event)
        if hit is not None:
            self._drag = hit
            return
        if event.inaxes in self.axes:
            self._select = [float(event.xdata), float(event.xdata)]
            self._selecting = True

    def _on_motion(self, event):
        if event.ydata is None and self._drag is not None:
            return
        if self._drag is not None and event.inaxes is self.axes[self._drag]:
            self.threshold_lines[self._drag].set_ydata([event.ydata] * 2)
            self.hist_lines[self._drag].set_ydata([event.ydata] * 2)
            self.fig.canvas.draw_idle()
        elif self._selecting and event.xdata is not None:
            self._select[1] = float(event.xdata)
            self._show_selection()

    def _on_release(self, event):
        if self._drag is not None:
            label = list(self.signals)[self._drag]
            key = THRESHOLD_OF.get(label)
            if key:
                self._push()
                self.thresholds[key] = float(
                    self.threshold_lines[self._drag].get_ydata()[0]
                )
                self._reclassify()
                self._draw_overview()
                self._draw()
            self._drag = None
        elif self._selecting:
            # the span stays put once the button is up, so it can be labelled;
            # a click without a drag just clears whatever was selected
            self._selecting = False
            if abs(self._select[1] - self._select[0]) < self.step_s:
                self._select = None
            self._show_selection()

    def _show_selection(self):
        for ax, span in zip(self.axes, self.selection):
            try:
                span.remove()
            except (ValueError, AttributeError):
                pass
        if self._select is None:
            self.selection = [
                ax.axvspan(0, 0, alpha=0.0) for ax in self.axes
            ]
        else:
            lo, hi = sorted(self._select)
            self.selection = [
                ax.axvspan(lo, hi, color="tab:blue", alpha=0.15, zorder=0)
                for ax in self.axes
            ]
        self.fig.canvas.draw_idle()

    def _clamp(self):
        span = self.times[-1] - self.times[0]
        self.window_s = float(np.clip(self.window_s, 10 * self.step_s, span))
        self.t0 = float(
            np.clip(self.t0, self.times[0], max(self.times[-1] - self.window_s,
                                                self.times[0]))
        )

    def paint(self, state):
        """Assign `state` (or None to clear) to the selected span."""
        if self._select is None:
            return
        lo, hi = sorted(self._select)
        span = (self.times >= lo) & (self.times <= hi)
        if not span.any():
            return
        self._push()
        self.manual[span] = -1 if state is None else STATE_CODES[state]
        self._reclassify()
        self._draw_overview()
        self._draw()

    def _on_key(self, event):
        key = event.key
        if key in PAINT_KEYS:
            self.paint(PAINT_KEYS[key])
        elif key in ("a", "left"):
            self.t0 -= self.window_s / 2
            self._clamp(); self._draw()
        elif key in ("d", "right"):
            self.t0 += self.window_s / 2
            self._clamp(); self._draw()
        elif key == "z":
            self.window_s *= 1.5
            self._clamp(); self._draw()
        elif key == "Z":
            self.window_s /= 1.5
            self._clamp(); self._draw()
        elif key == "u":
            self.undo()
        elif key == "R":
            self.reset()
        elif key == "S":
            self.save()
        elif key == "h":
            self.help.set_visible(not self.help.get_visible())
            self.fig.canvas.draw_idle()

    def undo(self):
        if not self._history:
            return
        self.thresholds, self.manual = self._history.pop()
        for i, (line, hline, label) in enumerate(
            zip(self.threshold_lines, self.hist_lines, self.signals)
        ):
            key = THRESHOLD_OF.get(label)
            if line is not None and key and self.thresholds.get(key) is not None:
                line.set_ydata([self.thresholds[key]] * 2)
                hline.set_ydata([self.thresholds[key]] * 2)
        self._reclassify()
        self._draw_overview()
        self._draw()

    def reset(self):
        """Back to the thresholds and states the pipeline produced."""
        self._push()
        self.thresholds = dict(self.scoring.thresholds)
        self.manual[:] = -1
        for line, hline, label in zip(
            self.threshold_lines, self.hist_lines, self.signals
        ):
            key = THRESHOLD_OF.get(label)
            if line is not None and key and self.thresholds.get(key) is not None:
                line.set_ydata([self.thresholds[key]] * 2)
                hline.set_ydata([self.thresholds[key]] * 2)
        self.codes = np.asarray(self.scoring.codes, dtype=np.int16).copy()
        self._draw_overview()
        self._draw()

    # ── saving ───────────────────────────────────────────────────────────
    def save(self, path=None):
        """Write the edited scoring back, keeping a copy of the original.

        Overwrites the pipeline's own files by default so that everything
        downstream picks the edits up, but only after copying them aside once
        as ``*.orig.*`` -- automatic scoring is cheap to regenerate, an hour
        of hand curation is not.
        """
        from .states import intervals_from_states

        source = path or getattr(self.scoring, "source", None)
        if source is None:
            raise ValueError(
                "This scoring has no source path; pass save(path=...) with the "
                "_states.json to write."
            )
        json_path = Path(source)
        npz_path = json_path.with_name(f"{self.scoring.session}_metrics.npz")

        for original in (json_path, npz_path):
            backup = original.with_suffix(".orig" + original.suffix)
            if original.exists() and not backup.exists():
                shutil.copy2(original, backup)
                print(f"    kept the automatic scoring at {backup.name}")

        summary = dict(self.scoring.metadata)
        summary["thresholds"] = {
            k: (float(v) if v is not None else None)
            for k, v in self.thresholds.items()
        }
        summary["fractions"] = {
            name: float(np.mean(self.codes == code))
            for name, code in STATE_CODES.items()
        }
        summary["intervals"] = intervals_from_states(
            self.codes, self.times, self.step_s
        )
        summary["edited"] = True
        summary["n_manual_bins"] = int((self.manual >= 0).sum())
        json_path.write_text(json.dumps(summary, indent=2))

        arrays = {"times": self.times, "codes": self.codes,
                  "manual": self.manual}
        for name, key in (("broadband", "broadband LFP (PC1)"),
                          ("theta", "theta ratio"),
                          ("emg", "EMG (LFP correlation)"),
                          ("speed", "movement (mm/s)")):
            if key in self.signals:
                arrays[name] = self.signals[key]
        if self.scoring.codes_before_veto is not None:
            arrays["codes_before_veto"] = self.scoring.codes_before_veto
        np.savez_compressed(npz_path, **arrays)

        print(f"    saved {summary['n_manual_bins']} manual bins and "
              f"thresholds to {json_path.name}")
        return json_path


def show_state_editor(scoring, window_s: float = 600.0,
                      log_signals=("movement (mm/s)",),
                      robust=(0.5, 99.5)) -> StateEditor:
    """Open the interactive state editor.

    Drag a threshold line and every bin re-classifies against it live. Drag on
    a trace to select a span, then 1/3/5 (or w/n/r) to label it by hand and 0
    to hand it back to the automatic rules. ``a``/``d`` pan, ``z``/``Z`` zoom,
    ``u`` undo, ``R`` reset to the pipeline's output, ``S`` save.

    Manual labels are kept separately from the automatic classification, so
    moving a threshold afterwards will not discard them.
    """
    return StateEditor(scoring, window_s=window_s, log_signals=log_signals,
                       robust=robust)
