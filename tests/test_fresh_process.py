"""_run_in_fresh_process: how stage 3 keeps kilosort4 out of this process.

Every test here starts a real interpreter, so each costs a few seconds --
most of it the child importing spikeshpc to find _report_back.
"""

import os

import pytest

from spikeshpc.pipeline import _run_in_fresh_process


def test_the_call_happens_in_another_process_and_its_result_comes_back():
    assert _run_in_fresh_process(os.getpid) != os.getpid()


def test_a_failure_raises_here_carrying_the_childs_traceback():
    with pytest.raises(RuntimeError, match="ValueError: invalid literal"):
        _run_in_fresh_process(int, "not a number")


def test_a_child_that_dies_without_reporting_is_an_error_not_a_hang():
    # os._exit skips everything, including the report back -- the same as
    # the out-of-memory killer taking kilosort down mid-sort.
    with pytest.raises(RuntimeError, match="exit code 3"):
        _run_in_fresh_process(os._exit, 3)
