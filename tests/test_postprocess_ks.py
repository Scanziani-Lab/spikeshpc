"""Filtering the analyzer's recording, and borrowing kilosort's spike positions.

The two failures these guard against are both silent. An unfiltered recording
produces extensions that exist, load, and are wrong -- every unit at the same
depth, because the mean waveform is the DC offset. And copying kilosort's
per-spike positions index for index lines the sample indices up exactly while
handing a few percent of spikes the position of a different unit, because the
two disagree about how to order spikes that share a sample.
"""

import shutil

import numpy as np
import pytest

import spikeinterface.full as si

from spikeshpc.postprocess import (
    DEFAULT_BANDPASS,
    attach_spike_locations,
    kilosort_spike_locations,
    postprocess,
)

FS = 30000.0
DURATION = 4.0
N_CHANNELS = 16


def write_kilosort_folder(folder, times, clusters, positions, n_channels=N_CHANNELS):
    """A folder si.read_kilosort will open, with the arrays in KS's own order."""
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "spike_times.npy", np.asarray(times, dtype=np.int64))
    np.save(folder / "spike_clusters.npy", np.asarray(clusters, dtype=np.int32))
    np.save(folder / "spike_templates.npy", np.asarray(clusters, dtype=np.int32))
    np.save(folder / "spike_positions.npy", np.asarray(positions, dtype=np.float32))
    np.save(folder / "channel_map.npy", np.arange(n_channels, dtype=np.int32))
    np.save(
        folder / "channel_positions.npy",
        np.column_stack(
            [np.zeros(n_channels), np.arange(n_channels, dtype=float) * 20.0]
        ),
    )
    n_units = int(np.max(clusters)) + 1
    np.save(folder / "templates.npy", np.zeros((n_units, 20, n_channels), "float32"))
    np.save(folder / "amplitudes.npy", np.ones(len(times), "float32"))
    (folder / "params.py").write_text(
        f"n_channels_dat = {n_channels}\ndtype = 'int16'\noffset = 0\n"
        f"sample_rate = {FS}\nhp_filtered = False\n"
    )
    return folder


@pytest.fixture
def session(tmp_path):
    """A sorting with deliberate ties: spikes sharing a sample, out of unit order.

    Ties are the whole point. kilosort writes them in detection order, so the
    file has (sample 100, unit 3) before (sample 100, unit 1); spikeinterface
    sorts them the other way. Without ties the naive copy would pass.
    """
    rng = np.random.default_rng(0)
    n = 600
    times = np.sort(rng.integers(50, int(DURATION * FS) - 50, n))
    # force a healthy fraction of shared samples
    times[1::3] = times[0::3][: len(times[1::3])]
    times = np.sort(times)
    clusters = rng.integers(0, 4, n).astype(np.int32)

    # positions carry the spike's identity, so a misalignment is detectable
    positions = np.column_stack([clusters * 10.0 + 1.0, times * 1.0])

    ks = write_kilosort_folder(tmp_path / "kilosort4", times, clusters, positions)
    rec = si.generate_recording(
        num_channels=N_CHANNELS, sampling_frequency=FS, durations=[DURATION]
    )
    rec = rec.save(folder=tmp_path / "rec")
    return rec, ks, times, clusters, positions


def build_analyzer(rec, ks):
    sorting = si.read_kilosort(folder_path=ks)
    return si.create_sorting_analyzer(
        recording=rec, sorting=sorting, format="memory", sparse=False
    )


# ── the reordering ──────────────────────────────────────────────────────
def test_the_fixture_actually_contains_ties(session):
    _, _, times, _, _ = session
    assert np.count_nonzero(np.diff(times) == 0) > 50, "no ties: the test is vacuous"


def test_positions_follow_the_spike_not_the_row(session):
    """Each position encodes its own unit, so a swap shows up as a mismatch."""
    rec, ks, times, clusters, positions = session
    analyzer = build_analyzer(rec, ks)
    located = kilosort_spike_locations(analyzer, ks)

    spikes = analyzer.sorting.to_spike_vector()
    unit_ids = np.asarray(analyzer.sorting.unit_ids)
    expected_unit = unit_ids[spikes["unit_index"]]
    np.testing.assert_allclose(located["x"], expected_unit * 10.0 + 1.0)
    np.testing.assert_allclose(located["y"], spikes["sample_index"])


def test_the_naive_index_copy_would_have_been_wrong(session):
    """The failure this guards against, demonstrated rather than asserted."""
    rec, ks, times, clusters, positions = session
    analyzer = build_analyzer(rec, ks)
    spikes = analyzer.sorting.to_spike_vector()
    unit_ids = np.asarray(analyzer.sorting.unit_ids)

    naive_x = positions[:, 0]                       # row i -> row i
    correct_x = kilosort_spike_locations(analyzer, ks)["x"]
    wrong = naive_x != correct_x
    assert wrong.any(), "the fixture failed to produce any misordering"
    # and the naive version disagrees with the units it claims to describe
    assert not np.allclose(naive_x, unit_ids[spikes["unit_index"]] * 10.0 + 1.0)


def test_a_different_sorting_is_refused(session, tmp_path):
    rec, ks, times, clusters, positions = session
    analyzer = build_analyzer(rec, ks)
    np.save(ks / "spike_positions.npy", positions[:-5])
    with pytest.raises(ValueError, match="not the same sorting"):
        kilosort_spike_locations(analyzer, ks)


def test_mismatched_spikes_are_refused(session):
    """Same count, different spikes: it must not quietly use them."""
    rec, ks, times, clusters, positions = session
    analyzer = build_analyzer(rec, ks)
    np.save(ks / "spike_clusters.npy", (clusters + 1) % 4)
    with pytest.raises(ValueError, match="could not be matched"):
        kilosort_spike_locations(analyzer, ks)


# ── attaching it ────────────────────────────────────────────────────────
def test_the_attached_extension_reads_back(session):
    rec, ks, times, clusters, positions = session
    analyzer = build_analyzer(rec, ks)
    analyzer.compute(["random_spikes", "templates"])

    located = kilosort_spike_locations(analyzer, ks)
    attach_spike_locations(analyzer, located)

    data = analyzer.get_extension("spike_locations").get_data()
    np.testing.assert_array_equal(data, located)
    assert analyzer.has_extension("spike_locations")


def test_the_provenance_says_where_it_came_from(session):
    rec, ks, *_ = session
    analyzer = build_analyzer(rec, ks)
    analyzer.compute(["random_spikes", "templates"])
    attach_spike_locations(analyzer, kilosort_spike_locations(analyzer, ks))
    params = analyzer.get_extension("spike_locations").params
    assert params["method"] == "kilosort_spike_positions"


# ── the filter ──────────────────────────────────────────────────────────
def test_postprocess_filters_before_extracting(session, tmp_path, capsys):
    rec, ks, *_ = session
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    shutil.copytree(ks, out / "kilosort4", dirs_exist_ok=True)

    analyzer = postprocess(
        rec, out, {"random_spikes": {}, "templates": {}}, use_KS_positions=False
    )
    printed = capsys.readouterr().out
    assert "bandpass 300-6000 Hz" in printed
    # the analyzer's recording must be the filtered one, not the raw
    assert "filter" in type(analyzer.recording).__name__.lower()


def test_the_filter_can_be_turned_off(session, tmp_path, capsys):
    rec, ks, *_ = session
    out = tmp_path / "out2"
    out.mkdir()
    shutil.copytree(ks, out / "kilosort4", dirs_exist_ok=True)

    analyzer = postprocess(
        rec, out, {"random_spikes": {}}, bandpass=False, use_KS_positions=False
    )
    assert "bandpass disabled" in capsys.readouterr().out
    assert "filter" not in type(analyzer.recording).__name__.lower()


def test_the_band_can_be_overridden(session, tmp_path, capsys):
    rec, ks, *_ = session
    out = tmp_path / "out3"
    out.mkdir()
    shutil.copytree(ks, out / "kilosort4", dirs_exist_ok=True)

    postprocess(
        rec, out, {"random_spikes": {}},
        bandpass={"freq_min": 500.0}, use_KS_positions=False,
    )
    printed = capsys.readouterr().out
    assert "bandpass 500-6000 Hz" in printed, printed
    assert DEFAULT_BANDPASS["freq_min"] == 300.0, "defaults must not be mutated"


# ── end to end ──────────────────────────────────────────────────────────
def test_postprocess_borrows_the_positions_and_skips_computing_them(
    session, tmp_path, capsys
):
    rec, ks, *_ = session
    out = tmp_path / "out4"
    out.mkdir()
    shutil.copytree(ks, out / "kilosort4", dirs_exist_ok=True)

    analyzer = postprocess(
        rec, out,
        {"random_spikes": {}, "noise_levels": {}, "templates": {},
         "spike_locations": {}},
        use_KS_positions=True,
    )
    printed = capsys.readouterr().out
    assert "took" in printed and "positions from kilosort" in printed

    located = analyzer.get_extension("spike_locations").get_data()
    spikes = analyzer.sorting.to_spike_vector()
    unit_ids = np.asarray(analyzer.sorting.unit_ids)
    np.testing.assert_allclose(located["x"], unit_ids[spikes["unit_index"]] * 10.0 + 1.0)


def test_it_falls_back_to_computing_when_kilosort_cannot_be_matched(
    session, tmp_path, capsys
):
    rec, ks, times, clusters, positions = session
    out = tmp_path / "out5"
    out.mkdir()
    shutil.copytree(ks, out / "kilosort4", dirs_exist_ok=True)
    np.save(out / "kilosort4" / "spike_positions.npy", positions[:-5])

    analyzer = postprocess(
        rec, out,
        {"random_spikes": {}, "noise_levels": {}, "templates": {},
         "spike_locations": {}},
        use_KS_positions=True,
    )
    assert "not the same sorting" in capsys.readouterr().out
    # computed rather than skipped, so the extension is still there
    assert analyzer.has_extension("spike_locations")
    assert analyzer.get_extension("spike_locations").params["method"] != (
        "kilosort_spike_positions"
    )
