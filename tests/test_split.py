"""Tests for spikeshpc.split -- carving a concatenated run back apart."""

import json

import numpy as np
import pytest

from spikeshpc import split as SP

FS = 30000.0
# two sessions: 10 s then 6 s
NUM_SAMPLES = [int(10 * FS), int(6 * FS)]
OFFSETS = [0, NUM_SAMPLES[0], NUM_SAMPLES[0] + NUM_SAMPLES[1]]
INFO = {
    "sampling_frequency": FS,
    "num_samples": NUM_SAMPLES,
    "sample_offsets": OFFSETS,
    "phys_paths": ["/data/sess_a", "/data/sess_b"],
}


@pytest.fixture
def concatenated():
    """A concatenated recording + sorting with spikes at known frames."""
    import probeinterface
    import spikeinterface.full as si

    total = OFFSETS[-1]
    rng = np.random.default_rng(0)
    rec = si.NumpyRecording(
        rng.normal(0, 10, size=(total, 4)).astype("int16"), sampling_frequency=FS
    )
    probe = probeinterface.Probe(ndim=2)
    probe.set_contacts(
        positions=np.c_[np.zeros(4), np.arange(4) * 20.0],
        shapes="circle", shape_params={"radius": 5},
    )
    probe.set_device_channel_indices(np.arange(4))
    rec = rec.set_probe(probe)

    spikes = {
        # in session a (frames < 300000) and session b
        "u1": np.array([1000, 50000, 310000, 400000]),
        "u2": np.array([2000, 299999]),          # session a only
        "u3": np.array([305000]),                # session b only
    }
    sorting = si.NumpySorting.from_unit_dict([spikes], sampling_frequency=FS)
    return rec, sorting, spikes


def test_session_bounds_and_names():
    assert SP.session_bounds(INFO) == [(0, 300000), (300000, 480000)]
    assert SP.session_names(INFO) == ["sess_a", "sess_b"]


def test_session_names_fall_back_when_paths_are_missing():
    info = dict(INFO)
    info["phys_paths"] = ["only-one"]
    assert SP.session_names(info) == ["session0", "session1"]


def test_split_recording_lengths(concatenated):
    rec, _, _ = concatenated
    parts = SP.split_recording(rec, INFO)
    assert [p.get_num_frames() for p in parts] == NUM_SAMPLES


def test_split_recording_rejects_a_mismatched_run(concatenated):
    rec, _, _ = concatenated
    bad = dict(INFO)
    bad["num_samples"] = [1, 2]
    with pytest.raises(ValueError, match="Are they from the same run"):
        SP.split_recording(rec, bad)


def test_split_recording_traces_match_the_source(concatenated):
    rec, _, _ = concatenated
    parts = SP.split_recording(rec, INFO)
    np.testing.assert_array_equal(
        parts[1].get_traces(start_frame=0, end_frame=100),
        rec.get_traces(start_frame=OFFSETS[1], end_frame=OFFSETS[1] + 100),
    )


def test_split_sorting_rebases_spike_frames(concatenated):
    _, sorting, spikes = concatenated
    parts = SP.split_sorting(sorting, INFO)

    # every spike lands in exactly one session, shifted to session-local frames
    for unit, frames in spikes.items():
        recovered = np.concatenate(
            [
                parts[i].get_unit_spike_train(unit) + OFFSETS[i]
                for i in range(len(parts))
            ]
        )
        np.testing.assert_array_equal(np.sort(recovered), np.sort(frames))


def test_split_sorting_keeps_all_units_including_silent_ones(concatenated):
    _, sorting, _ = concatenated
    parts = SP.split_sorting(sorting, INFO)
    for p in parts:
        assert set(map(str, p.unit_ids)) == {"u1", "u2", "u3"}
    # u3 fires only in session b
    assert parts[0].get_unit_spike_train("u3").size == 0
    assert parts[1].get_unit_spike_train("u3").size == 1


def test_split_sorting_frames_stay_inside_their_session(concatenated):
    _, sorting, _ = concatenated
    for i, part in enumerate(SP.split_sorting(sorting, INFO)):
        for unit in part.unit_ids:
            train = part.get_unit_spike_train(unit)
            assert np.all(train >= 0)
            assert np.all(train < NUM_SAMPLES[i])


def test_split_states_clips_and_rebases():
    states = {
        "intervals": {
            "WAKE": [[0.0, 4.0], [10.0, 12.0]],   # second one is in session b
            "NREM": [[4.0, 10.0]],
            "REM": [],
        }
    }
    a, b = SP.split_states(states, INFO)
    assert a["WAKE"] == [[0.0, 4.0]]
    assert a["NREM"] == [[4.0, 10.0]]
    assert b["WAKE"] == [[0.0, 2.0]]              # 10-12 s -> 0-2 s local
    assert b["NREM"] == []


def test_split_states_divides_an_interval_that_straddles_a_junction():
    states = {"intervals": {"WAKE": [[8.0, 12.0]], "NREM": [], "REM": []}}
    a, b = SP.split_states(states, INFO)
    assert a["WAKE"] == [[8.0, 10.0]]
    assert b["WAKE"] == [[0.0, 2.0]]


def test_split_states_clips_the_half_step_overhang():
    # intervals_from_states extends half a step past the first/last bin centre
    states = {"intervals": {"WAKE": [[-0.5, 3.0]], "NREM": [], "REM": []}}
    a, _ = SP.split_states(states, INFO)
    assert a["WAKE"] == [[0.0, 3.0]]


def test_timestamps_line_up_with_the_other_objects():
    s = SessionSplit = SP.SessionSplit(
        index=1, name="b", sample_offset=OFFSETS[1],
        num_samples=NUM_SAMPLES[1], sampling_frequency=FS,
    )
    local = s.timestamps()
    assert len(local) == NUM_SAMPLES[1]
    assert local[0] == 0.0
    assert local[1] == pytest.approx(1 / FS)
    assert local[-1] == pytest.approx((NUM_SAMPLES[1] - 1) / FS)

    absolute = s.timestamps(relative_to="concatenated")
    assert absolute[0] == pytest.approx(OFFSETS[1] / FS)
    np.testing.assert_allclose(absolute - local, s.t_start)

    # a session-local spike frame indexes straight into the timestamp array
    assert local[15000] == pytest.approx(0.5)


def test_timestamps_reject_lossy_dtypes_and_bad_reference():
    s = SP.SessionSplit(0, "a", 0, 100, FS)
    with pytest.raises(ValueError, match="cannot resolve single samples"):
        s.timestamps(dtype=np.float32)
    with pytest.raises(ValueError, match="must be 'session' or 'concatenated'"):
        s.timestamps(relative_to="elsewhere")


def test_split_run_end_to_end(tmp_path, concatenated):
    rec, sorting, _ = concatenated
    states = {"intervals": {"WAKE": [[0.0, 10.0]], "NREM": [[10.0, 16.0]], "REM": []}}

    splits = SP.split_run(
        tmp_path, recording=rec, sorting=sorting, states=states, info=INFO
    )
    assert len(splits) == 2
    a, b = splits

    assert (a.name, b.name) == ("sess_a", "sess_b")
    assert a.duration_s == pytest.approx(10.0)
    assert b.t_start == pytest.approx(10.0)
    assert a.recording.get_num_frames() == NUM_SAMPLES[0]
    assert b.sorting.get_unit_spike_train("u3").size == 1
    assert a.states["WAKE"] == [[0.0, 10.0]]
    assert b.states["NREM"] == [[0.0, 6.0]]
    assert a.analyzer is None       # with_analyzer defaults off


def test_save_splits_writes_each_session(tmp_path, concatenated):
    rec, sorting, _ = concatenated
    states = {"intervals": {"WAKE": [[0.0, 16.0]], "NREM": [], "REM": []}}
    splits = SP.split_run(
        tmp_path, recording=rec, sorting=sorting, states=states, info=INFO
    )
    SP.save_splits(splits, tmp_path, save_timestamps=True)

    for s in splits:
        d = tmp_path / SP.SESSIONS_DIRNAME / s.name
        meta = json.loads((d / "split_info.json").read_text())
        assert meta["num_samples"] == s.num_samples
        assert meta["t_start"] == pytest.approx(s.t_start)

        trains = np.load(d / "spike_trains.npz")
        assert set(trains.files) == {"u1", "u2", "u3"}

        ts = np.load(d / "timestamps.npy")
        assert len(ts) == s.num_samples
        assert ts[0] == 0.0

        assert (d / "states.json").exists()


def test_split_analyzer_rebuilds_per_session(tmp_path, concatenated):
    import spikeinterface.full as si

    rec, sorting, _ = concatenated
    parent = si.create_sorting_analyzer(
        recording=rec, sorting=sorting, format="memory"
    )
    parent.compute({"random_spikes": {}, "noise_levels": {}, "templates": {}})

    recs = SP.split_recording(rec, INFO)
    sorts = SP.split_sorting(sorting, INFO)
    subs = SP.split_analyzer(parent, recs, sorts, n_jobs=1)

    assert len(subs) == 2
    for sub, n in zip(subs, NUM_SAMPLES):
        # same units and channels as the parent, but only this session's data
        assert list(map(str, sub.unit_ids)) == list(map(str, parent.unit_ids))
        assert sub.get_num_channels() == parent.get_num_channels()
        assert sub.recording.get_num_frames() == n
        # the parent's extensions were carried over, not silently dropped
        assert set(SP.parent_extension_names(parent)) <= set(
            SP.parent_extension_names(sub)
        )


def test_parent_extension_names_works_on_an_in_memory_analyzer(concatenated):
    """get_saved_extension_names() raises for format='memory'; we must not."""
    import spikeinterface.full as si

    rec, sorting, _ = concatenated
    analyzer = si.create_sorting_analyzer(
        recording=rec, sorting=sorting, format="memory"
    )
    analyzer.compute({"random_spikes": {}, "noise_levels": {}})
    assert set(SP.parent_extension_names(analyzer)) == {"random_spikes", "noise_levels"}
