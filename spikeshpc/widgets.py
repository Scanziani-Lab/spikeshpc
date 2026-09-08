"""Interactive review of the automatic brain-state scoring.

The scoring is a starting point, not an answer (see spikeshpc/states.py), so
there has to be a fast way to look at what it decided and why. This steps
through one scored epoch at a time with the arrow keys, showing every signal
the decision was made from against the thresholds that were applied.
"""

import numpy as np

from .optitrack.widgets._backend import warn_if_noninteractive_backend

STATE_COLORS = {"WAKE": "#d95f02", "NREM": "#1b9e77", "REM": "#7570b3"}


class StateEpochWidget:
    """One scored epoch at a time; Left/Right to step through them.

    Each signal is drawn over the epoch plus `context_s` either side, so the
    transitions into and out of it are visible -- a REM call is only credible
    if it emerges from NREM, and that cannot be judged from the epoch alone.
    The epoch itself is shaded, its threshold drawn as a dashed line.

    Y-axes are fixed across every epoch, so a trace at the same height means
    the same value wherever you are in the session. They are set from the
    `robust` percentiles of the whole recording rather than its min and max,
    which a single artefact would otherwise stretch beyond use; the threshold
    is always kept inside them, since it is the reference the state was
    decided against. Override any of them with
    ``ylim={"movement (mm/s)": (0, 200)}``.

    Requires an interactive matplotlib backend (``%matplotlib widget`` or
    ``%matplotlib qt``) and the figure to have keyboard focus: click it once.
    """

    def __init__(self, scoring, states=None, context_s: float = 60.0,
                 min_duration_s: float = 0.0, vetoed: bool = False,
                 ylim=None, robust=(0.5, 99.5), log_signals=()):
        import matplotlib.pyplot as plt

        warn_if_noninteractive_backend()

        self.scoring = scoring
        self.context_s = float(context_s)
        self.vetoed = vetoed
        source = scoring.vetoed_epochs() if vetoed else scoring.epochs(states)
        if vetoed and states is not None:
            wanted = {states} if isinstance(states, str) else set(states)
            source = [e for e in source if e.state in {s.upper() for s in wanted}]
        self.epochs = [e for e in source if e.duration >= min_duration_s]
        if not self.epochs:
            what = "veto-reassigned spans" if vetoed else "epochs"
            raise ValueError(
                f"No {what} to show (states={states}, "
                f"min_duration_s={min_duration_s})."
            )
        self.index = 0

        self.signals = scoring.signals()
        if not self.signals:
            raise ValueError(
                f"{scoring.session!r} has no per-bin signals saved; the "
                "_metrics.npz next to the states JSON is what holds them."
            )

        n = len(self.signals)
        self.fig, self.axes = plt.subplots(
            n, 1, sharex=True, figsize=(10, 2.0 * n), constrained_layout=True
        )
        self.axes = np.atleast_1d(self.axes)

        self.lines, self.spans, self.thresholds = [], [], []
        threshold_for = {
            "broadband LFP (PC1)": "broadband",
            "theta ratio": "theta",
            "EMG (LFP correlation)": "emg",
        }
        for ax, label in zip(self.axes, self.signals):
            (line,) = ax.plot([], [], lw=0.9, color="black")
            self.lines.append(line)
            self.spans.append(None)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=8)

            key = threshold_for.get(label)
            value = scoring.thresholds.get(key) if key else None
            if value is not None:
                ax.axhline(value, ls="--", lw=0.8, color="tab:red", alpha=0.7)
            self.thresholds.append(value)

        self._set_fixed_limits(ylim or {}, robust, log_signals)

        self.axes[-1].set_xlabel("time (s)")
        self.title = self.fig.suptitle("", fontsize=11)
        self.label = self.axes[0].text(
            0.5, 0.92, "", transform=self.axes[0].transAxes,
            ha="center", va="top", fontsize=15, fontweight="bold",
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    def _set_fixed_limits(self, overrides, robust, log_signals):
        """One y-range per signal for the whole session, set once."""
        self.ylim = {}
        for ax, label, threshold in zip(self.axes, self.signals, self.thresholds):
            if label in log_signals:
                ax.set_yscale("symlog", linthresh=1.0)
            if label in overrides:
                lo, hi = overrides[label]
            else:
                values = np.asarray(self.signals[label], dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    continue
                lo, hi = np.percentile(finite, robust)
                if threshold is not None:  # the line the call was made against
                    lo, hi = min(lo, threshold), max(hi, threshold)
                pad = 0.06 * ((hi - lo) or abs(hi) or 1.0)
                lo, hi = lo - pad, hi + pad
                # padding must not open a negative region on a signal that
                # cannot go negative -- on a log axis that wastes half the panel
                if finite.min() >= 0:
                    lo = max(lo, 0.0)
            ax.set_ylim(lo, hi)
            self.ylim[label] = (float(lo), float(hi))

    # ── drawing ──────────────────────────────────────────────────────────
    def _draw(self):
        epoch = self.epochs[self.index]
        times = self.scoring.times
        lo = epoch.start - self.context_s
        hi = epoch.stop + self.context_s
        window = (times >= lo) & (times <= hi)

        for ax, line, span, label in zip(
            self.axes, self.lines, range(len(self.spans)), self.signals
        ):
            values = self.signals[label][window]
            line.set_data(times[window], values)
            if self.spans[span] is not None:
                self.spans[span].remove()
            self.spans[span] = ax.axvspan(
                epoch.start, epoch.stop,
                color=STATE_COLORS.get(epoch.state, "gray"), alpha=0.18, zorder=0,
            )
        self.axes[0].set_xlim(lo, hi)
        self.label.set_text(
            f"{epoch.state} → WAKE" if self.vetoed else epoch.state
        )
        self.label.set_color(STATE_COLORS.get(epoch.state, "black"))
        kind = "vetoed" if self.vetoed else "epoch"
        self.title.set_text(
            f"{self.scoring.session} — {kind} {self.index + 1}/{len(self.epochs)}"
            f"   {epoch.state}   {epoch.duration:.1f}s"
            f"   ({epoch.start:.1f}–{epoch.stop:.1f} s)"
            f"      ← → to step"
        )
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "right":
            self.index = (self.index + 1) % len(self.epochs)
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.epochs)
        else:
            return
        self._draw()


def show_state_epochs(scoring, states=None, context_s: float = 60.0,
                      min_duration_s: float = 0.0, vetoed: bool = False,
                      ylim=None, robust=(0.5, 99.5),
                      log_signals=()) -> StateEpochWidget:
    """Open an interactive figure; Left/Right step through scored epochs.

    ``scoring`` is what :func:`spikeshpc.io.load_states` returns. Pass
    ``states="REM"`` to review only the calls most worth checking, and
    ``min_duration_s`` to skip the briefest ones.

    ``vetoed=True`` steps through the spans the movement veto overruled
    instead, each labelled by what the LFP alone had called it. Those spans
    are WAKE in the final scoring, so they cannot be found any other way --
    and they are the calls whose rejection is worth checking.

    Y-axes are fixed across epochs so heights are comparable; see
    :class:`StateEpochWidget` for ``ylim``, ``robust`` and ``log_signals``.
    Movement spans orders of magnitude between immobility and locomotion, so
    ``log_signals=("movement (mm/s)",)`` is often easier to read.
    """
    return StateEpochWidget(
        scoring, states=states, context_s=context_s,
        min_duration_s=min_duration_s, vetoed=vetoed,
        ylim=ylim, robust=robust, log_signals=log_signals,
    )
