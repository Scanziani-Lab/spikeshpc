"""Reading a short stretch of a long recording must touch only that stretch.

spikeinterface's plot_traces finds the frames for a time range by binary
search on the time vector, reads them, and plots them. Here the same thing is
done without ever materialising the time vector, and a range that resolved
to more frames than it can hold -- which only a clock that goes backwards can
produce -- is refused instead of read.
"""

import numpy as np
import probeinterface
import pytest

import spikeinterface.full as si

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from spikeshpc import io as IO  # noqa: E402
from spikeshpc.traces import (  # noqa: E402
    get_traces,
    plot_traces,
    time_range_to_frames,
)

FS = 1000.0
NCH = 6
NSAMP = 20_000
T0 = 2345.5  # a sync clock rarely starts at zero


def make_recording(times=None, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(-500, 500, (NSAMP, NCH)).astype("int16")
    rec = si.NumpyRecording(data, sampling_frequency=FS)
    probe = probeinterface.Probe(ndim=2)
    # deliberately out of depth order
    depths = np.array([100.0, 0.0, 60.0, 20.0, 80.0, 40.0])
    probe.set_contacts(positions=np.c_[np.zeros(NCH), depths],
                       shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(NCH))
    rec = rec.set_probe(probe)
    rec.set_property("gain_to_uV", np.full(NCH, 0.195))
    rec.set_property("offset_to_uV", np.zeros(NCH))
    if times is not None:
        rec.set_times(times, with_warning=False)
    return rec, data


def memmapped_clock(tmp_path, times):
    path = tmp_path / "timestamps.npy"
    np.save(path, np.asarray(times, dtype="float64"))
    return np.load(path, mmap_mode="r")


@pytest.fixture
def synced(tmp_path):
    return make_recording(memmapped_clock(tmp_path, T0 + np.arange(NSAMP) / FS))


def forbid_get_times(rec, monkeypatch):
    for segment in rec._recording_segments:
        def boom():
            raise AssertionError("the whole time vector was asked for")
        monkeypatch.setattr(segment, "get_times", boom)


# ── time range to frames ────────────────────────────────────────────────
def test_frames_without_a_time_vector_match_spikeinterface():
    rec, _ = make_recording()
    start, end = time_range_to_frames(rec, (3.0, 3.2))
    assert (start, end) == (3000, 3200)
    np.testing.assert_array_equal(
        [start, end], rec.time_to_sample_index(np.array([3.0, 3.2]))
    )


def test_frames_on_a_sync_clock(synced):
    rec, _ = synced
    assert time_range_to_frames(rec, (T0 + 3.0, T0 + 3.2)) == (3000, 3200)


def test_relative_counts_from_the_first_sample(synced):
    rec, _ = synced
    assert time_range_to_frames(rec, (3.0, 3.2), relative=True) == (3000, 3200)


def test_a_range_on_the_wrong_clock_says_so(synced):
    rec, _ = synced
    with pytest.raises(ValueError, match="relative=True"):
        time_range_to_frames(rec, (3.0, 3.2))


def test_a_range_past_the_end_is_clipped_with_a_warning(synced):
    rec, _ = synced
    with pytest.warns(UserWarning, match="clipping"):
        start, end = time_range_to_frames(rec, (T0 + 19.9, T0 + 25.0))
    assert (start, end) == (19_900, NSAMP)


def test_a_clock_that_restarts_is_refused_not_read(tmp_path):
    """Two sessions end to end, the second clock starting over.

    A binary search on that lands on garbage: here a 0.2 s request resolves to
    over 5 s of frames. spikeinterface reads them; this must not.
    """
    half = NSAMP // 2
    times = np.r_[T0 + np.arange(half) / FS, T0 + 5.0 + np.arange(half) / FS]
    rec, _ = make_recording(memmapped_clock(tmp_path, times))
    frames = np.searchsorted(times, [T0 + 5.0, T0 + 5.2])
    assert frames[1] - frames[0] > 200 * 10, "the scenario needs a runaway span"
    with pytest.raises(ValueError, match="not increasing"):
        time_range_to_frames(rec, (T0 + 5.0, T0 + 5.2))


# ── get_traces ──────────────────────────────────────────────────────────
def test_traces_are_exactly_the_requested_frames(synced, monkeypatch):
    rec, data = synced
    forbid_get_times(rec, monkeypatch)
    traces, times = get_traces(rec, (T0 + 3.0, T0 + 3.2), return_times=True)
    np.testing.assert_array_equal(traces, data[3000:3200])
    np.testing.assert_allclose(times, T0 + np.arange(3000, 3200) / FS)


def test_channel_order_and_scaling(synced):
    rec, data = synced
    ids = rec.channel_ids
    traces = get_traces(rec, (T0 + 1.0, T0 + 1.1), channel_ids=[ids[3], ids[0]],
                        return_in_uV=True)
    np.testing.assert_allclose(traces, data[1000:1100][:, [3, 0]] * 0.195,
                               rtol=1e-6)


def test_relative_times_come_back_relative(synced):
    rec, _ = synced
    _, times = get_traces(rec, (1.0, 1.01), relative=True, return_times=True)
    np.testing.assert_allclose(times, np.arange(1000, 1010) / FS, atol=1e-9)


def test_too_much_is_refused_before_it_is_read(synced, monkeypatch):
    rec, _ = synced
    monkeypatch.setattr(rec, "get_traces", lambda **kw: pytest.fail("read"))
    with pytest.raises(ValueError, match="max_gb"):
        get_traces(rec, (0.0, 10.0), relative=True, max_gb=1e-6)


# ── plot_traces ─────────────────────────────────────────────────────────
def test_line_plot_draws_only_the_range(synced, monkeypatch):
    rec, data = synced
    forbid_get_times(rec, monkeypatch)
    ax = plot_traces(rec, (T0 + 3.0, T0 + 3.2), mode="line")
    assert len(ax.lines) == NCH
    assert len(ax.lines[0].get_xdata()) == 200
    np.testing.assert_allclose(ax.get_xlim(), (T0 + 3.0, T0 + 3.2), atol=1e-9)
    matplotlib.pyplot.close(ax.figure)


def test_layers_are_overlaid(synced):
    rec, _ = synced
    ax = plot_traces({"raw": rec, "cmr": si.common_reference(rec)},
                     (T0 + 3.0, T0 + 3.1), mode="line")
    assert len(ax.lines) == 2 * NCH
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["raw", "cmr"]
    matplotlib.pyplot.close(ax.figure)


def test_depth_order_sorts_the_rows(synced):
    rec, data = synced
    ax = plot_traces(rec, (T0 + 3.0, T0 + 3.01), mode="line",
                     order_channel_by_depth=True, show_channel_ids=True)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [str(rec.channel_ids[i]) for i in (1, 3, 5, 2, 4, 0)]
    matplotlib.pyplot.close(ax.figure)


def test_map_mode_is_one_image(synced):
    rec, _ = synced
    ax = plot_traces(rec, (1.0, 1.5), relative=True, mode="map")
    (im,) = ax.get_images()
    assert im.get_array().shape == (NCH, 500)
    assert ax.get_xlabel() == "time from start (s)"
    matplotlib.pyplot.close(ax.figure)


def test_only_events_in_range_are_drawn(synced):
    rec, _ = synced
    events = T0 + np.array([0.5, 3.05, 3.1, 9.0])
    ax = plot_traces(rec, (T0 + 3.0, T0 + 3.2), mode="map", events=events)
    assert len(ax.lines) == 2
    matplotlib.pyplot.close(ax.figure)


# ── io: the sync clock stays on disk ────────────────────────────────────
def test_open_ephys_timestamps_are_memory_mapped(tmp_path):
    folder = tmp_path / "continuous" / "ProbeA"
    folder.mkdir(parents=True)
    np.save(folder / "sample_numbers.npy", np.arange(NSAMP))
    np.save(folder / "timestamps.npy", T0 + np.arange(NSAMP) / FS)

    rec, _ = make_recording()
    rec._stream_folders = [folder]
    assert IO._attach_sync_times(rec)
    tv = rec.get_time_info()["time_vector"]
    assert isinstance(tv, np.memmap)
    assert rec.get_start_time() == T0


def test_without_stream_folders_nothing_is_attached():
    rec, _ = make_recording()
    assert not IO._attach_sync_times(rec)
    assert not rec.has_time_vector()


def test_concatenated_sessions_are_joined_on_disk(tmp_path, monkeypatch):
    a = T0 + np.arange(100) / FS
    b = T0 + 50.0 + np.arange(150) / FS
    monkeypatch.setattr(IO, "_stream_timestamps",
                        lambda path, *_: {"s1": a, "s2": b}[path])
    info = {"phys_type": "openephysbinary", "phys_paths": ["s1", "s2"],
            "num_samples": [100, 150], "stream_name": "ProbeA"}

    times = IO._concatenated_sync_times(info, tmp_path)
    assert isinstance(times, np.memmap)
    np.testing.assert_array_equal(times, np.r_[a, b])

    written = (tmp_path / IO.SYNC_TIMES_NAME).stat().st_mtime_ns
    del times
    again = IO._concatenated_sync_times(info, tmp_path)
    np.testing.assert_array_equal(again, np.r_[a, b])
    assert (tmp_path / IO.SYNC_TIMES_NAME).stat().st_mtime_ns == written
