"""Shared handling of the one thing that silently breaks every widget: a
non-interactive matplotlib backend."""

from __future__ import annotations

import warnings

import matplotlib

# Backends that render to a static image/file and never deliver GUI events.
# Matched by exact name (so e.g. "qtagg" isn't caught by a stray "agg" substring
# check) plus a separate "inline" substring match for matplotlib_inline's
# backend, whose name is a module path.
_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "pgf", "template"}


def use_backend(backend: str | None) -> None:
    """Switch matplotlib to `backend` the way ``%matplotlib <backend>`` does.

    Each widget calls this before it creates its figure, so the notebook
    needs no ``%matplotlib`` cell of its own. In IPython this goes through
    ``shell.enable_matplotlib`` -- what the magic itself runs -- because a GUI
    backend also needs the kernel to run its event loop, and
    ``plt.switch_backend`` alone leaves a window that never redraws. Nothing
    is done when `backend` is already in use.

    Outside IPython (a script, pytest) there is no ``%matplotlib`` to mirror,
    so the backend chosen with ``matplotlib.use`` or ``MPLBACKEND`` is kept.

    The switch is not undone afterwards: figures drawn later in the notebook
    use `backend` too, until the next ``%matplotlib``.

    Parameters
    ----------
    backend : str or None
        Anything ``%matplotlib`` accepts: ``"qt"``, ``"widget"``, ``"tk"``...
        None keeps whatever backend is active.

    Warns
    -----
    UserWarning
        If the backend in use afterwards does not deliver key presses.
    """
    shell = _ipython_shell()
    if backend is not None and shell is not None and not _is_active(backend):
        shell.enable_matplotlib(backend)
    warn_if_noninteractive_backend(stacklevel=4)


def _ipython_shell():
    """The running IPython shell, or None outside IPython."""
    # Imported here, not at the top: `import spikeshpc` loads every widget,
    # and the cluster image that runs the pipeline has no IPython.
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def _is_active(backend: str) -> bool:
    """Whether `backend` is the one pyplot is already using.

    Compared by name only. Resolving a name through matplotlib's backend
    registry imports the backend module, and importing ipympl ("widget")
    switches the backend all by itself (its ``__init__`` calls
    ``matplotlib.use``) -- so asking would change the answer.

    Parameters
    ----------
    backend : str
        Backend or GUI name, as ``%matplotlib`` takes it.

    Returns
    -------
    bool
        True if `backend` names the active backend, or a GUI framework
        ("qt", "tk") whose default backend is active. An alias it does not
        recognise ("ipympl" for "widget") gives False, which only costs a
        redundant switch.
    """
    from matplotlib.backends import backend_registry

    name = backend.lower()
    current = matplotlib.get_backend().lower()
    return current in {name, backend_registry.backend_for_gui_framework(name)}


def warn_if_noninteractive_backend(stacklevel: int = 3) -> None:
    """Warn if key presses on the widget's figure would be silently dropped.

    Jupyter defaults to the ``inline`` backend, which renders each figure as
    a static image at cell-execution time -- ``fig.canvas.mpl_connect`` still
    "succeeds", but no event ever fires, so the widget looks like it drew once
    and stopped responding.

    Parameters
    ----------
    stacklevel : int, default 3
        Passed to :func:`warnings.warn`; the default points at whoever
        constructed the widget.

    Warns
    -----
    UserWarning
        If the active backend is not interactive.
    """
    backend = matplotlib.get_backend().lower()
    if "inline" in backend or backend in _NON_INTERACTIVE_BACKENDS:
        warnings.warn(
            f"matplotlib backend is {matplotlib.get_backend()!r}, which does not "
            "deliver key-press events -- this widget will draw once and then "
            "ignore arrow keys. Pass `backend='qt'` (or 'widget') to the widget, "
            "or run `%matplotlib qt` in its own cell before creating it.",
            stacklevel=stacklevel,
        )
