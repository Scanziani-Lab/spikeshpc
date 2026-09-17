"""Routing settings_overrides to kilosort's settings dict vs. run_kilosort()'s
own keyword arguments.

kilosort isn't a hard dependency of this package (it ships via the cluster's
container image, not pip -- see pyproject.toml), so it's not installed here.
split_settings_overrides takes `run_kilosort` as a plain callable, so it's
tested against a fake with a realistic signature; run_kilosort4 itself is
tested by injecting fake `kilosort` / `kilosort.io` modules into sys.modules.
"""

import sys
import types
from pathlib import Path

import pytest

from spikeshpc.sorting import run_kilosort4, split_settings_overrides

DEFAULT_SETTINGS = {"nblocks": 6, "Th_universal": 9.0, "fs": 30000.0, "n_chan_bin": 385}


def fake_run_kilosort(
    settings,
    probe=None,
    probe_name=None,
    filename=None,
    data_dir=None,
    file_object=None,
    results_dir=None,
    data_dtype=None,
    do_CAR=True,
    invert_sign=False,
    device=None,
    progress_bar=None,
    save_extra_vars=False,
    clear_cache=False,
    save_preprocessed_copy=False,
    bad_channels=None,
    shank_idx=None,
    verbose_console=False,
    verbose_log=False,
    torch_thread_lim=None,
):
    """Mirrors kilosort.run_kilosort's real signature, without depending on it."""


# ── split_settings_overrides ─────────────────────────────────────────────
def test_settings_keys_go_to_settings_updates():
    updates, kwargs = split_settings_overrides(
        {"nblocks": 3}, DEFAULT_SETTINGS, fake_run_kilosort
    )
    assert updates == {"nblocks": 3}
    assert kwargs == {}


def test_run_kilosort_keys_go_to_kwargs():
    updates, kwargs = split_settings_overrides(
        {"do_CAR": False, "shank_idx": [0, 1]}, DEFAULT_SETTINGS, fake_run_kilosort
    )
    assert updates == {}
    assert kwargs == {"do_CAR": False, "shank_idx": [0, 1]}


def test_a_mix_is_split_correctly():
    updates, kwargs = split_settings_overrides(
        {"nblocks": 3, "do_CAR": False, "Th_universal": 12.0,
         "save_preprocessed_copy": True},
        DEFAULT_SETTINGS, fake_run_kilosort,
    )
    assert updates == {"nblocks": 3, "Th_universal": 12.0}
    assert kwargs == {"do_CAR": False, "save_preprocessed_copy": True}


def test_empty_overrides_is_fine():
    assert split_settings_overrides({}, DEFAULT_SETTINGS, fake_run_kilosort) == ({}, {})


def test_an_unknown_key_is_refused():
    with pytest.raises(ValueError, match=r"nblcoks"):
        split_settings_overrides(
            {"nblcoks": 3}, DEFAULT_SETTINGS, fake_run_kilosort  # typo
        )


def test_a_reserved_key_is_refused():
    with pytest.raises(ValueError, match="filename"):
        split_settings_overrides(
            {"filename": "/tmp/other.bin"}, DEFAULT_SETTINGS, fake_run_kilosort
        )


@pytest.mark.parametrize(
    "key", ["settings", "probe", "probe_name", "filename", "data_dir",
            "file_object", "results_dir", "data_dtype", "bad_channels"]
)
def test_every_reserved_arg_is_refused(key):
    with pytest.raises(ValueError, match="settings_overrides cannot set"):
        split_settings_overrides({key: object()}, DEFAULT_SETTINGS, fake_run_kilosort)


def test_settings_itself_is_not_treated_as_a_settings_key():
    """'settings' is run_kilosort's first positional arg, not a tuning value."""
    with pytest.raises(ValueError):
        split_settings_overrides({"settings": {}}, DEFAULT_SETTINGS, fake_run_kilosort)


def test_reserved_and_unknown_together_reports_the_reserved_one_first():
    """Whichever error fires, it must not be the misleading one."""
    with pytest.raises(ValueError, match="settings_overrides cannot set"):
        split_settings_overrides(
            {"filename": "x", "nblcoks": 3}, DEFAULT_SETTINGS, fake_run_kilosort
        )


def test_kilosort_gaining_a_new_flag_is_picked_up_automatically():
    """The classification reads the live signature, not a hand-written list."""

    def future_run_kilosort(settings, brand_new_flag=False, **kw):
        pass

    updates, kwargs = split_settings_overrides(
        {"brand_new_flag": True}, DEFAULT_SETTINGS, future_run_kilosort
    )
    assert kwargs == {"brand_new_flag": True}


# ── run_kilosort4, with kilosort faked out ───────────────────────────────
@pytest.fixture
def fake_kilosort(monkeypatch):
    calls = []

    def run_kilosort(
        settings,
        probe=None,
        probe_name=None,
        filename=None,
        data_dir=None,
        file_object=None,
        results_dir=None,
        data_dtype=None,
        do_CAR=True,
        invert_sign=False,
        device=None,
        progress_bar=None,
        save_extra_vars=False,
        clear_cache=False,
        save_preprocessed_copy=False,
        bad_channels=None,
        shank_idx=None,
        verbose_console=False,
        verbose_log=False,
        torch_thread_lim=None,
    ):
        calls.append({
            "settings": dict(settings),
            "kwargs": dict(
                probe=probe, probe_name=probe_name, filename=filename,
                data_dir=data_dir, file_object=file_object,
                results_dir=results_dir, data_dtype=data_dtype, do_CAR=do_CAR,
                invert_sign=invert_sign, device=device,
                progress_bar=progress_bar, save_extra_vars=save_extra_vars,
                clear_cache=clear_cache,
                save_preprocessed_copy=save_preprocessed_copy,
                bad_channels=bad_channels, shank_idx=shank_idx,
                verbose_console=verbose_console, verbose_log=verbose_log,
                torch_thread_lim=torch_thread_lim,
            ),
        })

    kilosort_mod = types.ModuleType("kilosort")
    kilosort_mod.DEFAULT_SETTINGS = dict(DEFAULT_SETTINGS)
    kilosort_mod.run_kilosort = run_kilosort

    kilosort_io_mod = types.ModuleType("kilosort.io")
    kilosort_io_mod.load_probe = lambda path: f"probe:{path}"
    kilosort_mod.io = kilosort_io_mod

    monkeypatch.setitem(sys.modules, "kilosort", kilosort_mod)
    monkeypatch.setitem(sys.modules, "kilosort.io", kilosort_io_mod)
    return calls


def make_info(**extra):
    info = {
        "num_channels": 384,
        "sampling_frequency": 30000.0,
        "dtype": "int16",
        "binary_file": "concatenated.bin",
        "channel_ids": [str(i) for i in range(384)],
    }
    info.update(extra)
    return info


def test_settings_overrides_update_the_settings_dict(tmp_path, fake_kilosort):
    run_kilosort4(tmp_path, make_info(), settings_overrides={"nblocks": 2})
    assert fake_kilosort[0]["settings"]["nblocks"] == 2
    # untouched defaults survive
    assert fake_kilosort[0]["settings"]["Th_universal"] == 9.0


def test_run_kilosort_kwargs_reach_the_call_not_the_settings(tmp_path, fake_kilosort):
    run_kilosort4(
        tmp_path, make_info(),
        settings_overrides={"do_CAR": False, "shank_idx": [0, 1]},
    )
    call = fake_kilosort[0]
    assert call["kwargs"]["do_CAR"] is False
    assert call["kwargs"]["shank_idx"] == [0, 1]
    assert "do_CAR" not in call["settings"]
    assert "shank_idx" not in call["settings"]


def test_n_chan_bin_and_fs_are_still_derived_from_info(tmp_path, fake_kilosort):
    run_kilosort4(tmp_path, make_info(num_channels=384, sampling_frequency=30000.0))
    settings = fake_kilosort[0]["settings"]
    assert settings["n_chan_bin"] == 384
    assert settings["fs"] == 30000.0


def test_file_num_channels_wins_over_num_channels_for_n_chan_bin(tmp_path, fake_kilosort):
    """SpikeGLX's SY0 row: the file has one more row than kilosort's probe."""
    run_kilosort4(tmp_path, make_info(num_channels=384, file_num_channels=385))
    assert fake_kilosort[0]["settings"]["n_chan_bin"] == 385


def test_bad_channels_still_reach_run_kilosort(tmp_path, fake_kilosort):
    info = make_info(channel_ids=["10", "11", "12"], num_channels=3)
    run_kilosort4(tmp_path, info, bad_channels=["11"])
    assert fake_kilosort[0]["kwargs"]["bad_channels"] == [1]


def test_a_reserved_override_key_stops_before_calling_run_kilosort(
    tmp_path, fake_kilosort
):
    with pytest.raises(ValueError, match="settings_overrides cannot set"):
        run_kilosort4(
            tmp_path, make_info(), settings_overrides={"filename": "/tmp/x.bin"}
        )
    assert fake_kilosort == []


def test_no_overrides_behaves_as_before(tmp_path, fake_kilosort):
    run_kilosort4(tmp_path, make_info())
    call = fake_kilosort[0]
    assert call["kwargs"]["bad_channels"] is None
    assert call["settings"]["nblocks"] == DEFAULT_SETTINGS["nblocks"]
