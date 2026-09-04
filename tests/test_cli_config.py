"""Tests for --pipeline_config loading: a bad config must fail legibly."""

import json

import pytest

from spikeshpc.cli import load_pipeline_config
from spikeshpc.config import DEFAULT_PIPELINE


def test_valid_config_round_trips(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"sorting": {"nblocks": 5}, "bad_channels": [191]}))
    assert load_pipeline_config(p) == {"sorting": {"nblocks": 5}, "bad_channels": [191]}


def test_empty_object_is_fine(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text("{}")
    assert load_pipeline_config(p) == {}


def test_missing_comma_names_the_file_and_shows_the_line(tmp_path):
    """The failure seen on the cluster: no comma after the previous entry."""
    p = tmp_path / "cfg.json"
    p.write_text('{\n  "preprocessing": {}\n  "bad_channels": []\n}\n')

    with pytest.raises(SystemExit) as excinfo:
        load_pipeline_config(p)
    message = str(excinfo.value)

    assert "cfg.json" in message                       # names the file
    assert "line 3" in message                         # points at the line
    assert '"bad_channels": []' in message             # shows the source
    assert "missing" in message and "trailing comma" in message
    assert "python -m json.tool" in message            # how to check the rest


def test_trailing_comma_is_reported(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{\n  "bad_channels": [],\n}\n')
    with pytest.raises(SystemExit, match="not valid JSON"):
        load_pipeline_config(p)


def test_context_window_survives_an_error_on_the_first_line(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text("not json at all")
    with pytest.raises(SystemExit, match="not valid JSON"):
        load_pipeline_config(p)


def test_error_on_the_last_line_does_not_run_off_the_end(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{\n  "sorting": {\n')
    with pytest.raises(SystemExit, match="not valid JSON"):
        load_pipeline_config(p)


def test_misspelled_top_level_key_is_rejected(tmp_path):
    """A silent default is worse than a failed submit."""
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"state_scorring": {"enabled": False}}))

    with pytest.raises(SystemExit) as excinfo:
        load_pipeline_config(p)
    message = str(excinfo.value)
    assert "state_scorring" in message
    assert "state_scoring" in message  # the valid-keys list suggests the fix


def test_every_default_key_is_accepted(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({k: DEFAULT_PIPELINE[k] for k in DEFAULT_PIPELINE}))
    assert set(load_pipeline_config(p)) == set(DEFAULT_PIPELINE)


def test_non_object_config_is_rejected(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(SystemExit, match="must contain a JSON object"):
        load_pipeline_config(p)


def test_missing_file_still_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline_config(tmp_path / "nope.json")
