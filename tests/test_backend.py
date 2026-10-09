"""Each widget switches matplotlib to its `backend` itself, as %matplotlib would."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from spikeshpc.io import load_states  # noqa: E402
from spikeshpc.optitrack.widgets import _backend  # noqa: E402
from spikeshpc.widgets import show_state_epochs  # noqa: E402

from test_state_use import write_scoring  # noqa: E402


class Shell:
    """Stands in for IPython; records what %matplotlib would have been asked."""

    def __init__(self):
        self.calls = []

    def enable_matplotlib(self, gui=None):
        self.calls.append(gui)


@pytest.fixture
def shell(monkeypatch):
    shell = Shell()
    monkeypatch.setattr(_backend, "_ipython_shell", lambda: shell)
    return shell


@pytest.fixture
def scoring(tmp_path):
    write_scoring(tmp_path / "states")
    return load_states(tmp_path / "states")


def test_switches_through_ipython(shell, recwarn):
    _backend.use_backend("qt")
    assert shell.calls == ["qt"]


def test_none_keeps_the_backend(shell, recwarn):
    _backend.use_backend(None)
    assert shell.calls == []


def test_an_active_backend_is_not_switched_again(shell, recwarn):
    """Agg is active, so asking for it (by any alias) is a no-op."""
    _backend.use_backend("agg")
    _backend.use_backend("Agg")
    assert shell.calls == []


def test_a_gui_name_matches_its_backend(shell, monkeypatch):
    """%matplotlib qt leaves "qtagg" active, which is what "qt" asks for."""
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "qtagg")
    _backend.use_backend("qt")
    assert shell.calls == []


def test_asking_for_widget_does_not_switch_on_its_own(shell, recwarn):
    """Importing ipympl calls matplotlib.use; the check must not import it."""
    _backend.use_backend("widget")
    assert shell.calls == ["widget"]
    assert matplotlib.get_backend().lower() == "agg"


def test_an_unknown_name_is_left_to_ipython(shell, recwarn):
    """%matplotlib's own error is the clearer one."""
    _backend.use_backend("nonsense")
    assert shell.calls == ["nonsense"]


def test_outside_ipython_the_chosen_backend_stays(monkeypatch):
    """A script or pytest picks its backend with matplotlib.use."""
    monkeypatch.setattr(_backend, "_ipython_shell", lambda: None)
    with pytest.warns(UserWarning, match="does not deliver key-press events"):
        _backend.use_backend("qt")
    assert matplotlib.get_backend().lower() == "agg"


def test_no_ipython_installed_means_no_shell(monkeypatch):
    """The cluster image has no IPython."""
    import sys

    monkeypatch.setitem(sys.modules, "IPython", None)
    assert _backend._ipython_shell() is None


def test_widgets_default_to_qt(shell, scoring, recwarn):
    w = show_state_epochs(scoring)
    matplotlib.pyplot.close(w.fig)
    assert shell.calls == ["qt"]


def test_widgets_pass_their_backend_on(shell, scoring, recwarn):
    w = show_state_epochs(scoring, backend="widget")
    matplotlib.pyplot.close(w.fig)
    assert shell.calls == ["widget"]


def test_the_warning_points_at_the_caller(monkeypatch, scoring):
    """Not at _backend.py: the line to fix is the one that made the widget."""
    monkeypatch.setattr(_backend, "_ipython_shell", lambda: None)
    with pytest.warns(UserWarning, match="does not deliver") as record:
        w = show_state_epochs(scoring)
    matplotlib.pyplot.close(w.fig)
    assert record[0].filename.endswith("widgets.py")
