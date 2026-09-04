"""Sorting the acquisition system's own binary instead of rewriting it."""

import json

import numpy as np
import pytest

from spikeshpc import io as IO
from spikeshpc.config import CONCAT_INFO_NAME
from spikeshpc.preprocess import preprocess

FS = 30000.0
NCH = 8
NSAMP = 6000


def make_binary(path, n_extra_rows=0, seed=0):
    """A flat int16 file with NCH channels plus `n_extra_rows` trailing rows.

    Mimics SpikeGLX, whose .ap.bin keeps SY0 as a 385th row that the AP stream
    does not expose.
    """
    rng = np.random.default_rng(seed)
    data = rng.integers(-3000, 3000, (NSAMP, NCH + n_extra_rows), dtype="int16")
    data.tofile(path)
    return data


def recording_from(data, n_channels=NCH):
    import probeinterface
    import spikeinterface.full as si

    rec = si.NumpyRecording(
        np.ascontiguousarray(data[:, :n_channels]), sampling_frequency=FS
    )
    probe = probeinterface.Probe(ndim=2)
    probe.set_contacts(
        positions=np.c_[np.zeros(n_channels), np.arange(n_channels) * 20.0],
        shapes="circle", shape_params={"radius": 5},
    )
    probe.set_device_channel_indices(np.arange(n_channels))
    rec = rec.set_probe(probe)
    rec.set_property("gain_to_uV", np.full(n_channels, 2.34))
    rec.set_property("offset_to_uV", np.zeros(n_channels))
    return rec


def test_check_accepts_an_exact_file(tmp_path):
    p = tmp_path / "a.bin"
    rec = recording_from(make_binary(p))
    n_file, rows = IO.check_source_binary(rec, p)
    assert n_file == NCH
    assert np.array_equal(rows, np.arange(NCH))


def test_check_accepts_a_file_with_a_trailing_sync_row(tmp_path):
    """SpikeGLX: 385 rows on disk, 384 in the AP stream."""
    p = tmp_path / "a.ap.bin"
    rec = recording_from(make_binary(p, n_extra_rows=1))
    n_file, rows = IO.check_source_binary(rec, p)
    assert n_file == NCH + 1
    assert np.array_equal(rows, np.arange(NCH))


def test_check_rejects_a_file_whose_traces_differ(tmp_path):
    """The guard that stops us silently sorting the wrong channels."""
    p = tmp_path / "a.bin"
    data = make_binary(p)
    scrambled = data[:, ::-1].copy()      # same shape, different column order
    assert IO.check_source_binary(recording_from(scrambled), p) is None


def test_check_rejects_a_size_mismatch(tmp_path):
    p = tmp_path / "a.bin"
    data = make_binary(p)
    p.write_bytes(p.read_bytes() + b"\x00" * 7)   # not a whole number of samples
    assert IO.check_source_binary(recording_from(data), p) is None


def test_check_rejects_a_missing_file(tmp_path):
    p = tmp_path / "a.bin"
    rec = recording_from(make_binary(p))
    assert IO.check_source_binary(rec, tmp_path / "nope.bin") is None


def test_locate_spikeglx_binary(tmp_path):
    run = tmp_path / "m1_g0" / "m1_g0_imec0"
    run.mkdir(parents=True)
    (run / "m1_g0_t0.imec0.ap.bin").touch()
    (run / "m1_g0_t0.imec0.ap.meta").touch()
    (run / "m1_g0_t0.imec0.lf.bin").touch()
    found = IO.locate_source_binary(tmp_path / "m1_g0", "spikeglx", "imec0.ap")
    assert found is not None and found.name == "m1_g0_t0.imec0.ap.bin"


def test_locate_openephys_binary(tmp_path):
    base = tmp_path / "rec1" / "continuous"
    (base / "Neuropix-PXI-100.ProbeA").mkdir(parents=True)
    (base / "Neuropix-PXI-100.ProbeA" / "continuous.dat").touch()
    (base / "NI-DAQmx-103.PXIe-6341").mkdir(parents=True)
    (base / "NI-DAQmx-103.PXIe-6341" / "continuous.dat").touch()
    found = IO.locate_source_binary(
        tmp_path, "openephysbinary", "Record Node 101#Neuropix-PXI-100.ProbeA"
    )
    assert found is not None
    assert found.parent.name == "Neuropix-PXI-100.ProbeA"


def test_locate_is_none_when_ambiguous(tmp_path):
    (tmp_path / "a.imec0.ap.bin").touch()
    (tmp_path / "b.imec0.ap.bin").touch()
    assert IO.locate_source_binary(tmp_path, "spikeglx", "imec0.ap") is None


def test_chanmap_carries_file_row_indices(tmp_path):
    import scipy.io

    rec = recording_from(make_binary(tmp_path / "a.bin", n_extra_rows=1))
    rows = np.arange(NCH)
    IO.write_channel_map(rec, tmp_path, channel_rows=rows)
    m = scipy.io.loadmat(tmp_path / "chanMap.mat")

    assert np.array_equal(m["chanMap"].ravel(), rows + 1)   # kilosort is 1-indexed
    assert np.array_equal(m["chanMap0ind"].ravel(), rows)
    assert m["connected"].sum() == NCH

    # and kilosort must be able to read it back
    from kilosort.io import load_probe

    probe = load_probe(tmp_path / "chanMap.mat")
    assert np.array_equal(probe["chanMap"], rows)
    assert probe["n_chan"] == NCH


def _run_preprocess(tmp_path, rec, source, **kwargs):
    import sys
    from unittest import mock

    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    fake = lambda p, t=None, s=None, band="ap": (rec, "spikeglx", "imec0.ap")
    with mock.patch.object(
        sys.modules["spikeshpc.preprocess"], "read_recording", fake
    ):
        info = preprocess([source], out, preprocessing={}, dtype=None, **kwargs)
    return out, info


def test_preprocess_reuses_the_source_and_writes_no_copy(tmp_path):
    import spikeinterface.full as si

    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)
    src = tmp_path / "m1_g0_t0.imec0.ap.bin"
    rec = recording_from(make_binary(src, n_extra_rows=1))

    out, info = _run_preprocess(tmp_path, rec, src)

    assert info["sorted_in_place"] is True
    assert info["binary_path"] == str(src.resolve())
    assert info["file_num_channels"] == NCH + 1     # what kilosort reads per sample
    assert info["num_channels"] == NCH              # what we actually sort
    assert info["channel_rows"] == list(range(NCH))
    assert not (out / "concatenated.bin").exists(), "a copy was written anyway"


def test_reloaded_recording_matches_the_source(tmp_path):
    import spikeinterface.full as si

    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)
    src = tmp_path / "m1_g0_t0.imec0.ap.bin"
    data = make_binary(src, n_extra_rows=1)
    rec = recording_from(data)

    out, _ = _run_preprocess(tmp_path, rec, src)
    reloaded, info = IO.load_concatenated(out)

    assert reloaded.get_num_channels() == NCH
    assert reloaded.get_num_frames() == NSAMP
    np.testing.assert_array_equal(
        reloaded.get_traces(return_in_uV=False), data[:, :NCH]
    )
    # the sync row must not have leaked in
    assert list(map(str, reloaded.channel_ids)) == info["channel_ids"]
    np.testing.assert_allclose(
        reloaded.get_property("gain_to_uV"), np.full(NCH, 2.34)
    )
    np.testing.assert_allclose(
        reloaded.get_channel_locations(), rec.get_channel_locations()
    )


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"reuse_source": False}, "disabled"),
        ({"preprocessing": {"bandpass_filter": {}}}, "preprocessing"),
        ({"dtype": "float32"}, "dtype change"),
    ],
)
def test_falls_back_to_writing_when_it_must(tmp_path, kwargs, reason):
    import sys
    from unittest import mock

    import spikeinterface.full as si

    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)
    src = tmp_path / "m1_g0_t0.imec0.ap.bin"
    rec = recording_from(make_binary(src, n_extra_rows=1))

    out = tmp_path / "out"
    out.mkdir()
    call = {"preprocessing": {}, "dtype": None}
    call.update(kwargs)
    fake = lambda p, t=None, s=None, band="ap": (rec, "spikeglx", "imec0.ap")
    with mock.patch.object(
        sys.modules["spikeshpc.preprocess"], "read_recording", fake
    ):
        info = preprocess([src], out, **call)

    assert info["sorted_in_place"] is False, reason
    assert (out / "concatenated.bin").exists()
    assert info["channel_rows"] is None
    # and it still reloads correctly
    reloaded, _ = IO.load_concatenated(out)
    assert reloaded.get_num_channels() == NCH


def test_kilosort_gets_the_file_width_not_the_channel_count(tmp_path):
    """n_chan_bin must be rows-per-sample, or every sample is misaligned."""
    src = tmp_path / "m1_g0_t0.imec0.ap.bin"
    rec = recording_from(make_binary(src, n_extra_rows=1))
    import spikeinterface.full as si

    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)
    _, info = _run_preprocess(tmp_path, rec, src)
    assert info["file_num_channels"] == NCH + 1
    assert info["file_num_channels"] != info["num_channels"]


def test_old_concat_info_without_the_new_keys_still_loads(tmp_path):
    """concat_info.json written before this feature must keep working."""
    import spikeinterface.full as si

    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)
    src = tmp_path / "m1_g0_t0.imec0.ap.bin"
    rec = recording_from(make_binary(src))
    out, info = _run_preprocess(tmp_path, rec, src, reuse_source=False)

    stripped = {k: v for k, v in info.items()
                if k not in ("binary_path", "channel_rows",
                             "file_num_channels", "sorted_in_place")}
    (out / CONCAT_INFO_NAME).write_text(json.dumps(stripped))
    reloaded, _ = IO.load_concatenated(out)
    assert reloaded.get_num_channels() == NCH
