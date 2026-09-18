"""Bad channels must not take part in preprocessing, only be tolerated by it.

The bug this guards against was found on real data: five channels pinned at a
constant rail value (std = 0.0 across 976s) were fed through `phase_shift`
along with everything else, and came back swinging from -30928 to +32277 --
beyond the valid int16 range -- at processing boundaries. kilosort itself
already excludes bad_channels from every one of its own computations (it
slices to chanMap before CAR/filtering/whitening, on every batch), so this
never affected sorting. But `preprocess()` had no such exclusion, so the
*written binary* carried that manufactured nonsense on rows nothing besides
kilosort's own bookkeeping should ever read.

A mean-based common reference is used in several tests below rather than the
median the real config used, because mean has no robustness to outliers at
all -- a single contaminated channel skews it for every channel -- which
makes "was the bad channel excluded" an unambiguous, dramatic question rather
than one median's robustness could paper over with only a handful of bad
channels.
"""

import numpy as np
import pytest

import spikeinterface.full as si

from spikeshpc.preprocess import _apply_preprocessing, _interleave_channels, preprocess

FS = 30000.0
N_CHANNELS = 6
N_SAMPLES = 3000
RAIL_VALUE = 30690.0


def make_recording(rail_channel=None, rail_value=RAIL_VALUE, seed=0):
    """`N_CHANNELS` channels of modest noise; optionally one pinned at a rail."""
    rng = np.random.default_rng(seed)
    traces = rng.normal(0, 50.0, (N_SAMPLES, N_CHANNELS)).astype("float32")
    if rail_channel is not None:
        traces[:, rail_channel] = rail_value

    rec = si.NumpyRecording(traces, sampling_frequency=FS)
    rec = rec.rename_channels([f"CH{i}" for i in range(N_CHANNELS)])
    probe = __import__("probeinterface").Probe(ndim=2)
    probe.set_contacts(
        positions=np.c_[np.zeros(N_CHANNELS), np.arange(N_CHANNELS) * 20.0],
        shapes="circle", shape_params={"radius": 5},
    )
    probe.set_device_channel_indices(np.arange(N_CHANNELS))
    return rec.set_probe(probe)


# ── _apply_preprocessing directly ────────────────────────────────────────
def test_with_no_bad_ids_it_is_exactly_the_plain_pipeline():
    """center estimates its per-channel mean from a random chunk sample, so
    two calls only agree exactly when both are seeded the same way -- fixing
    the seed here isolates that the early-return path (no bad_ids) really is
    an unmodified call to apply_preprocessing_pipeline, not a near-miss."""
    rec = make_recording()
    pre = {"center": {"mode": "mean", "seed": 0}}
    a = _apply_preprocessing(rec, pre, [])
    b = si.apply_preprocessing_pipeline(rec, pre)
    np.testing.assert_array_equal(
        a.get_traces(return_in_uV=False), b.get_traces(return_in_uV=False)
    )


# ── _interleave_channels: the aggregate_channels footgun, directly ──────
def test_interleaving_survives_when_bad_channels_are_not_a_trailing_block():
    """The exact bug: si.aggregate_channels(...).select_channels(original_order)
    silently returns each channel's NEIGHBOUR's data whenever the reassembled
    order interleaves the two sources, because its own segment groups a
    requested channel_indices by source recording and concatenates the groups
    in first-encountered order, discarding the order actually asked for.
    Verified directly against that library behaviour before writing the fix.
    """
    rec = make_recording(rail_channel=2)  # bad channel is neither first nor last
    good_ids = [c for c in rec.channel_ids if c != "CH2"]
    processed = si.apply_preprocessing_pipeline(rec.select_channels(good_ids), {})
    raw = rec.select_channels(["CH2"])

    result = _interleave_channels(processed, raw, list(rec.channel_ids))
    got = result.get_traces(return_in_uV=False)
    raw_traces = rec.get_traces(return_in_uV=False)

    np.testing.assert_allclose(got[:, 2], raw_traces[:, 2], atol=1e-3)
    np.testing.assert_allclose(got[:, [0, 1, 3, 4, 5]], raw_traces[:, [0, 1, 3, 4, 5]])


@pytest.mark.parametrize("bad_positions", [[0], [5], [0, 5], [1, 3], [0, 1, 2, 3, 4]])
def test_interleaving_is_correct_for_every_bad_channel_arrangement(bad_positions):
    """Leading block, trailing block, scattered, and nearly-all-bad."""
    rec = make_recording()
    traces = rec.get_traces(return_in_uV=False).copy()
    for i, p in enumerate(bad_positions):
        traces[:, p] = RAIL_VALUE * (1 if i % 2 == 0 else -1)
    rec = si.NumpyRecording(traces, sampling_frequency=FS).rename_channels(
        [f"CH{i}" for i in range(N_CHANNELS)]
    )

    bad_ids = [f"CH{p}" for p in bad_positions]
    good_ids = [c for c in rec.channel_ids if c not in bad_ids]
    processed = si.apply_preprocessing_pipeline(rec.select_channels(good_ids), {})
    raw = rec.select_channels(bad_ids)

    result = _interleave_channels(processed, raw, list(rec.channel_ids))
    np.testing.assert_allclose(
        result.get_traces(return_in_uV=False), traces, atol=1e-3
    )


def test_interleaving_works_with_a_windowed_get_traces_call():
    """start_frame/end_frame must reach the right parent segment correctly."""
    rec = make_recording(rail_channel=1)
    good_ids = [c for c in rec.channel_ids if c != "CH1"]
    processed = si.apply_preprocessing_pipeline(rec.select_channels(good_ids), {})
    raw = rec.select_channels(["CH1"])

    result = _interleave_channels(processed, raw, list(rec.channel_ids))
    got = result.get_traces(start_frame=100, end_frame=200, return_in_uV=False)
    expected = rec.get_traces(start_frame=100, end_frame=200, return_in_uV=False)
    np.testing.assert_allclose(got, expected, atol=1e-3)


def test_interleaving_a_single_requested_channel_at_a_time():
    """channel_indices as a length-1 request -- an edge the grouping loop must
    still handle, since one source's mask can be all-False."""
    rec = make_recording(rail_channel=3)
    good_ids = [c for c in rec.channel_ids if c != "CH3"]
    processed = si.apply_preprocessing_pipeline(rec.select_channels(good_ids), {})
    raw = rec.select_channels(["CH3"])

    result = _interleave_channels(processed, raw, list(rec.channel_ids))
    only_good = result.select_channels(["CH0"])
    only_bad = result.select_channels(["CH3"])
    raw_traces = rec.get_traces(return_in_uV=False)
    np.testing.assert_allclose(
        only_good.get_traces(return_in_uV=False)[:, 0], raw_traces[:, 0]
    )
    np.testing.assert_allclose(
        only_bad.get_traces(return_in_uV=False)[:, 0], raw_traces[:, 3]
    )


def test_a_rail_channel_does_not_skew_the_mean_reference_for_others():
    """The dramatic case: mean has no protection against even one outlier."""
    rec = make_recording(rail_channel=0)
    pre = {"common_reference": {"reference": "global", "operator": "average"}}

    unguarded = si.apply_preprocessing_pipeline(rec, pre)
    guarded = _apply_preprocessing(rec, pre, ["CH0"])

    raw = rec.get_traces(return_in_uV=False)
    good = [1, 2, 3, 4, 5]
    # unguarded: every good channel's reference is dragged toward the rail
    unguarded_traces = unguarded.get_traces(return_in_uV=False)
    assert np.abs(unguarded_traces[:, good]).mean() > RAIL_VALUE / N_CHANNELS / 2

    # guarded: good channels look like a sane CAR over the good channels alone
    guarded_traces = guarded.get_traces(return_in_uV=False)
    expected_ref = raw[:, good].mean(axis=1, keepdims=True)
    np.testing.assert_allclose(
        guarded_traces[:, good], raw[:, good] - expected_ref, atol=1e-3
    )


def test_the_bad_channel_comes_back_raw_not_referenced():
    rec = make_recording(rail_channel=2)
    pre = {"common_reference": {"reference": "global", "operator": "average"}}
    out = _apply_preprocessing(rec, pre, ["CH2"])

    raw = rec.get_traces(return_in_uV=False)[:, 2]
    got = out.get_traces(return_in_uV=False)[:, 2]
    np.testing.assert_allclose(got, raw, atol=1e-3)


def test_channel_order_and_ids_are_unchanged():
    rec = make_recording(rail_channel=1)
    pre = {"center": {"mode": "mean"}}
    out = _apply_preprocessing(rec, pre, ["CH1"])
    assert list(map(str, out.channel_ids)) == list(map(str, rec.channel_ids))


def test_probe_geometry_follows_each_channel_through_the_reshuffle():
    rec = make_recording(rail_channel=3)
    pre = {"center": {"mode": "mean"}}
    out = _apply_preprocessing(rec, pre, ["CH3"])
    np.testing.assert_allclose(
        out.get_channel_locations(), rec.get_channel_locations()
    )


def test_is_filtered_survives_the_recombination():
    """aggregate_channels does not carry this annotation on its own."""
    rec = make_recording(rail_channel=0)
    pre = {"bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0}}
    out = _apply_preprocessing(rec, pre, ["CH0"])
    assert out.is_filtered() is True


def test_dtype_promotion_is_matched_before_aggregating():
    """Preprocessing promotes to float32; the excluded raw channel must too."""
    rec = make_recording(rail_channel=0)
    assert rec.get_dtype() == np.dtype("float32")  # NumpyRecording is already float
    pre = {"center": {"mode": "mean"}}
    out = _apply_preprocessing(rec, pre, ["CH0"])
    assert out.get_dtype() == np.dtype("float32")


def test_multiple_bad_channels_are_all_excluded():
    rec = make_recording()
    traces = rec.get_traces(return_in_uV=False).copy()
    traces[:, 0] = RAIL_VALUE
    traces[:, 4] = -RAIL_VALUE
    rec = si.NumpyRecording(traces, sampling_frequency=FS)
    rec = rec.rename_channels([f"CH{i}" for i in range(N_CHANNELS)])
    probe = __import__("probeinterface").Probe(ndim=2)
    probe.set_contacts(
        positions=np.c_[np.zeros(N_CHANNELS), np.arange(N_CHANNELS) * 20.0],
        shapes="circle", shape_params={"radius": 5},
    )
    probe.set_device_channel_indices(np.arange(N_CHANNELS))
    rec = rec.set_probe(probe)

    pre = {"common_reference": {"reference": "global", "operator": "average"}}
    out = _apply_preprocessing(rec, pre, ["CH0", "CH4"])
    got = out.get_traces(return_in_uV=False)
    np.testing.assert_allclose(got[:, 0], traces[:, 0], atol=1e-3)
    np.testing.assert_allclose(got[:, 4], traces[:, 4], atol=1e-3)

    good = [1, 2, 3, 5]
    expected_ref = traces[:, good].mean(axis=1, keepdims=True)
    np.testing.assert_allclose(got[:, good], traces[:, good] - expected_ref, atol=1e-3)


# ── through preprocess() itself ──────────────────────────────────────────
def _run_preprocess(tmp_path, rec, source, **kwargs):
    import sys
    from unittest import mock

    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    fake = lambda p, t=None, s=None, band="ap": (rec, "openephysbinary", "ProbeA")
    with mock.patch.object(
        sys.modules["spikeshpc.preprocess"], "read_recording", fake
    ):
        info = preprocess([source], out, dtype=None, **kwargs)
    return out, info


def test_preprocess_end_to_end_excludes_the_listed_bad_channel(tmp_path):
    from spikeshpc.io import load_concatenated

    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)
    rec = make_recording(rail_channel=0)
    raw = rec.get_traces(return_in_uV=False).copy()
    src = tmp_path / "rec.bin"

    out, info = _run_preprocess(
        tmp_path, rec, src,
        preprocessing={"common_reference": {"reference": "global", "operator": "average"}},
        bad_channels=["CH0"],
    )

    assert info["channel_ids"] == [f"CH{i}" for i in range(N_CHANNELS)]
    reloaded, _ = load_concatenated(out)
    got = reloaded.get_traces(return_in_uV=False)
    np.testing.assert_allclose(got[:, 0], raw[:, 0], atol=1.0)


def test_preprocess_with_no_bad_channels_is_unaffected(tmp_path):
    from spikeshpc.io import load_concatenated

    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)
    rec = make_recording()
    src = tmp_path / "rec.bin"

    out, info = _run_preprocess(
        tmp_path, rec, src, preprocessing={"center": {"mode": "mean"}}, bad_channels=[]
    )
    assert info["channel_ids"] == [f"CH{i}" for i in range(N_CHANNELS)]
    reloaded, _ = load_concatenated(out)
    assert reloaded.get_num_channels() == N_CHANNELS


def test_preprocess_defaults_bad_channels_to_none_gracefully(tmp_path):
    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)
    rec = make_recording()
    src = tmp_path / "rec.bin"
    # bad_channels omitted entirely
    _run_preprocess(tmp_path, rec, src, preprocessing={"center": {"mode": "mean"}})


def test_an_unresolvable_bad_channel_raises_a_clear_error(tmp_path):
    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)
    rec = make_recording()
    src = tmp_path / "rec.bin"
    with pytest.raises(ValueError, match="not found in this recording"):
        _run_preprocess(
            tmp_path, rec, src,
            preprocessing={"center": {"mode": "mean"}},
            bad_channels=["CH99"],
        )


def test_bad_channels_without_preprocessing_is_a_no_op(tmp_path):
    """No preprocessing means nothing to protect against; must not error."""
    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)
    rec = make_recording(rail_channel=0)
    src = tmp_path / "rec.bin"
    make_binary_bytes = rec.get_traces(return_in_uV=False).astype("int16")
    # preprocess() with reuse_source needs int16 to attempt sort-in-place,
    # but that is not the point here -- just confirm bad_channels is inert.
    out, info = _run_preprocess(
        tmp_path, rec, src, preprocessing=None, bad_channels=["CH0"],
        reuse_source=False,
    )
    assert info["channel_ids"] == [f"CH{i}" for i in range(N_CHANNELS)]
