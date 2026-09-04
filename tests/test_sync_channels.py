"""Tests for the TTL-vs-noise guard on the shutter ADC channel."""

import numpy as np
import pytest

from spikeshpc import optitrack
from spikeshpc.optitrack import sync

FS = 30000.0
DUR = 20.0
FRAME_RATE = 120.0


def _ttl(n, fs=FS, rate=FRAME_RATE, noise=0.01, seed=0):
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    return (np.sin(2 * np.pi * rate * t) > 0).astype(float) * 5.0 + rng.normal(
        0, noise, n
    )


def _narrow_ttl(n, width_samples, fs=FS, rate=FRAME_RATE, seed=0):
    """A short exposure pulse: `width_samples` high out of each frame period.

    This is the case that broke the original percentile threshold -- below ~1%
    duty the 99th percentile falls in the baseline.
    """
    rng = np.random.default_rng(seed)
    trace = rng.integers(-1, 2, n).astype(float)   # quantised +/- 1 LSB baseline
    period = int(fs / rate)
    for k in range(0, n - width_samples, period):
        trace[k : k + width_samples] = 1000.0
    return trace


def _floating(n, seed=1):
    """An unconnected ADC input: white noise on a slow random-walk drift."""
    rng = np.random.default_rng(seed)
    drift = np.cumsum(rng.normal(0, 0.002, n))
    return rng.normal(0, 0.05, n) + (drift - drift.mean())


def make_stream(traces_by_channel):
    import spikeinterface.full as si

    ids = list(traces_by_channel)
    data = np.column_stack([traces_by_channel[c] for c in ids]).astype("float32")
    return si.NumpyRecording(data, sampling_frequency=FS, channel_ids=ids)


@pytest.mark.parametrize("noise", [0.01, 0.15])
def test_rail_fraction_is_high_for_a_ttl(noise):
    assert sync.rail_fraction(_ttl(int(FS * DUR), noise=noise)) > 0.9


DEFAULT_GUARD = 0.8   # extract_shutter_close_times' min_rail_fraction


@pytest.mark.parametrize("seed", range(5))
def test_rail_fraction_rejects_a_floating_input(seed):
    """Must land below the guard, with margin, on every noise realisation."""
    rails = sync.rail_fraction(_floating(int(FS * DUR), seed=seed))
    assert rails < DEFAULT_GUARD - 0.1, rails


def test_ttl_and_noise_sit_either_side_of_the_guard():
    n = int(FS * DUR)
    ttl_rails = min(
        sync.rail_fraction(_narrow_ttl(n, w)) for w in (125, 25, 8, 4, 2, 1)
    )
    noise_rails = max(
        sync.rail_fraction(_floating(n, seed=s)) for s in range(5)
    )
    assert noise_rails < DEFAULT_GUARD < ttl_rails
    assert ttl_rails - noise_rails > 0.2, "the two classes are not well separated"


def test_rail_fraction_handles_a_flat_trace():
    assert sync.rail_fraction(np.zeros(1000)) == 0.0


@pytest.mark.parametrize("width", [125, 25, 8, 4, 2, 1])
def test_narrow_pulses_still_read_as_a_ttl(width):
    """Down to a one-sample pulse -- 0.4% duty -- this is still a TTL."""
    trace = _narrow_ttl(int(FS * DUR), width)
    assert sync.rail_fraction(trace) > 0.9, f"{width} samples rejected"


@pytest.mark.parametrize("width", [125, 25, 8, 4, 2, 1])
def test_threshold_lands_between_the_levels_at_any_duty_cycle(width):
    """The regression: a percentile threshold collapses onto the noise floor.

    Below ~1% duty the 99th percentile sits in the baseline, so the threshold
    slices the noise and every wiggle becomes an edge.
    """
    trace = _narrow_ttl(int(FS * DUR), width)
    t = sync.ttl_threshold(trace)
    assert 1.0 < t < 1000.0, f"{width}-sample pulse: threshold {t} is not between levels"

    percentile_threshold = np.percentile(trace, [1, 99]).mean()
    if width <= 2:  # the regime that used to fail
        assert percentile_threshold <= 1.0


@pytest.mark.parametrize("width", [125, 25, 8, 4, 2, 1])
def test_extraction_recovers_the_frame_rate_at_any_duty_cycle(width):
    n = int(FS * DUR)
    stream = make_stream({"ADC0": _narrow_ttl(n, width)})
    times = sync.extract_shutter_close_times(stream, channel_id="ADC0")

    assert len(times) == pytest.approx(FRAME_RATE * DUR, rel=0.02)
    assert np.diff(times).mean() == pytest.approx(1 / FRAME_RATE, rel=0.01)


def test_threshold_is_none_for_a_flat_trace():
    assert sync.ttl_threshold(np.zeros(1000)) is None


def test_extraction_works_on_a_real_ttl():
    n = int(FS * DUR)
    stream = make_stream({"ADC0": _ttl(n)})
    times = sync.extract_shutter_close_times(stream, channel_id="ADC0")

    assert len(times) == pytest.approx(FRAME_RATE * DUR, rel=0.02)
    isis = np.diff(times)
    assert isis.mean() == pytest.approx(1 / FRAME_RATE, rel=0.01)
    assert isis.std() < 1e-4       # a real train is regular


def test_extraction_refuses_a_noise_channel():
    """The bug: a relative threshold slices noise into millions of edges."""
    n = int(FS * DUR)
    stream = make_stream({"ADC0": _floating(n)})

    with pytest.raises(ValueError, match="does not look like a TTL"):
        sync.extract_shutter_close_times(stream, channel_id="ADC0")


def test_the_refusal_names_the_way_out():
    stream = make_stream({"ADC0": _floating(int(FS * DUR))})
    with pytest.raises(ValueError) as excinfo:
        sync.extract_shutter_close_times(stream, channel_id="ADC0")
    message = str(excinfo.value)
    assert "describe_analog_channels" in message
    assert "force=True" in message


def test_force_overrides_the_guard():
    """force=True restores the old behaviour: threshold it anyway, no error."""
    n = int(FS * DUR)
    stream = make_stream({"ADC0": _floating(n)})
    times = sync.extract_shutter_close_times(stream, channel_id="ADC0", force=True)
    assert len(times) > 0
    # and the result is irregular, unlike a real train
    assert np.diff(times).std() > 1e-3


def test_describe_finds_a_narrow_pulse_ttl_among_noise_channels():
    """The real situation: a 2-sample exposure pulse on one of several ADCs."""
    n = int(FS * DUR)
    stream = make_stream(
        {
            "ADC0": _floating(n, seed=1),
            "ADC1": _floating(n, seed=2),
            "ADC2": _narrow_ttl(n, 2),
            "ADC3": np.zeros(n),
        }
    )
    rows = sync.describe_analog_channels(stream, n_chunks=5, verbose=False)
    by_id = {r["channel_id"]: r for r in rows}

    assert by_id["ADC2"]["looks_like_ttl"] is True
    assert by_id["ADC2"]["crossings_per_s"] == pytest.approx(FRAME_RATE, rel=0.1)
    assert by_id["ADC2"]["duty_cycle"] < 0.01
    for other in ("ADC0", "ADC1", "ADC3"):
        assert by_id[other]["looks_like_ttl"] is False


def test_describe_finds_the_ttl_among_noise_channels():
    n = int(FS * DUR)
    stream = make_stream(
        {
            "ADC0": _floating(n, seed=1),
            "ADC1": _floating(n, seed=2),
            "ADC2": _ttl(n),
            "ADC3": np.zeros(n),
        }
    )
    rows = sync.describe_analog_channels(stream, n_chunks=5, verbose=False)

    by_id = {r["channel_id"]: r for r in rows}
    assert len(rows) == 4
    assert by_id["ADC2"]["looks_like_ttl"] is True
    assert by_id["ADC2"]["crossings_per_s"] == pytest.approx(FRAME_RATE, rel=0.1)
    for other in ("ADC0", "ADC1", "ADC3"):
        assert by_id[other]["looks_like_ttl"] is False


def test_describe_reports_when_nothing_looks_like_a_ttl(capsys):
    n = int(FS * DUR)
    stream = make_stream({"ADC0": _floating(n), "ADC1": _floating(n, seed=3)})
    sync.describe_analog_channels(stream, n_chunks=5)
    out = capsys.readouterr().out
    assert "No channel on this stream looks like a TTL" in out


def test_cross_check_flags_an_implausible_train(tmp_path, capsys):
    csv = tmp_path / "take.csv"
    csv.write_text(
        "Format Version,1.23,Capture Frame Rate,120.000000,"
        "Total Exported Frames,2400\n"
    )
    # noise-derived "events": far too many, wildly irregular
    rng = np.random.default_rng(0)
    bogus = np.sort(rng.uniform(0, 20, 20000))
    stats = sync.cross_check_with_optitrack_csv(bogus, csv)

    assert stats["plausible"] is False
    assert "not a 120 Hz shutter train" in capsys.readouterr().out


def test_cross_check_accepts_a_real_train(tmp_path):
    csv = tmp_path / "take.csv"
    csv.write_text(
        "Format Version,1.23,Capture Frame Rate,120.000000,"
        "Total Exported Frames,2400\n"
    )
    good = np.arange(2400) / FRAME_RATE
    stats = sync.cross_check_with_optitrack_csv(good, csv)
    assert stats["plausible"] is True
    assert stats["frame_count_difference"] == 0


def test_exports_are_reachable_from_the_package():
    assert optitrack.describe_analog_channels is sync.describe_analog_channels
    assert optitrack.rail_fraction is sync.rail_fraction
