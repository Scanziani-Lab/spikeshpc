"""Shared check for the one thing that silently breaks both widgets: a
non-interactive matplotlib backend."""

from __future__ import annotations

import warnings

import matplotlib

# Backends that render to a static image/file and never deliver GUI events.
# Matched by exact name (so e.g. "qtagg" isn't caught by a stray "agg" substring
# check) plus a separate "inline" substring match for matplotlib_inline's
# backend, whose name is a module path.
_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "pgf", "template"}


def warn_if_noninteractive_backend() -> None:
    """Warn if key presses on the widget's figure would be silently dropped.

    Jupyter defaults to the ``inline`` backend, which renders each figure as
    a static image at cell-execution time -- ``fig.canvas.mpl_connect`` still
    "succeeds", but no event ever fires, so the widget looks like it drew once
    and stopped responding. Run ``%matplotlib widget`` (or ``%matplotlib qt``)
    in its own cell before creating either widget.
    """
    backend = matplotlib.get_backend().lower()
    if "inline" in backend or backend in _NON_INTERACTIVE_BACKENDS:
        warnings.warn(
            f"matplotlib backend is {matplotlib.get_backend()!r}, which does not "
            "deliver key-press events -- this widget will draw once and then "
            "ignore arrow keys. Run `%matplotlib widget` (or `%matplotlib qt`) "
            "in its own cell before creating it.",
            stacklevel=3,
        )
