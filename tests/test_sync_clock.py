"""Open Ephys must be read on the acquisition clock, never on an inferred one.

Two clocks exist for every Open Ephys stream and they are not interchangeable.
`timestamps.npy` is what the acquisition system actually measured; the other is
sample index divided by the rate `structure.oebin` declares. On a real OneBox
the ADC declares 30300.5 Hz and runs at about 30303 -- 83 ppm, which is nearly
two seconds of error by the end of a six-hour session, on the one array whose
whole job is to say when each camera frame happened.

The trap is that Open Ephys does not omit `timestamps.npy` for a stream it
never synchronized: it writes a file of -1.0. spikeinterface hands that back
without complaint, so asking for the good clock and not checking is its own way
of getting a bad one.
"""

import numpy as np
import pytest

import spikeinterface.full as si

from spikeshpc import shutter
from spikeshpc.io import (
    open_stream,
    probe_start_time,
    read_openephys_synced,
    sync_clock_problem,
)

FS = 30000.0
DECLARED_FS = 30300.5
TRUE_FS = 30303.0  # what the hardware really does: 83 ppm fast
RATE = 120.0
DUR = 20.0


def ttl(n, width=4, rate=RATE, fs=FS, seed=0):
    rng = np.random.default_rng(seed)
    trace = rng.integers(-1, 2, n).astype(float)
    period = int(fs / rate)
    for k in range(0, n - width, period):
        trace[k : k + width] = 1000.0
    return trace


def recording(traces, fs=FS, times=None):
    ids = list(traces)
    data = np.column_stack([traces[c] for c in ids]).astype("float32")
    rec = si.NumpyRecording(data, sampling_frequency=fs, channel_ids=ids)
    if times is not None:
        rec.set_times(np.asarray(times, dtype=float))
    return rec


# ── recognising an unusable clock ───────────────────────────────────────
def test_a_real_clock_is_accepted():
    n = 10_000
    rec = recording({"ADC0": np.zeros(n)}, times=8.2 + np.arange(n) / FS)
    assert sync_clock_problem(rec) is None


def test_an_unsynced_stream_is_caught():
    """Open Ephys fills timestamps.npy with -1 rather than leaving it out."""
    n = 10_000
    rec = recording({"ADC0": np.zeros(n)}, times=np.full(n, -1.0))
    problem = sync_clock_problem(rec)
    assert problem is not None
    assert "never synchronized" in problem


def test_a_stalled_clock_is_caught():
    n = 10_000
    rec = recording({"ADC0": np.zeros(n)}, times=np.full(n, 5.0))
    assert "does not advance" in sync_clock_problem(rec)


def test_a_clock_that_goes_backwards_is_caught():
    """A local dip, so the endpoints still advance and only the middle is wrong."""
    n = 10_000
    times = 8.0 + np.arange(n) / FS
    times[n // 2 : n // 2 + 1000] -= 0.5
    rec = recording({"ADC0": np.zeros(n)}, times=times)

    assert times[-1] > times[0], "the fixture must not fail the cruder check"
    assert "not increasing" in sync_clock_problem(rec)


def test_the_check_is_cheap_on_a_long_recording():
    """Time vectors run to tens of gigabytes; the check must not walk them."""
    n = 5_000_000
    rec = recording({"ADC0": np.zeros(n, dtype="float32")}, times=np.arange(n) / FS)
    assert sync_clock_problem(rec, sample=1000) is None


# ── the loader ──────────────────────────────────────────────────────────
def synced(monkeypatch, good, bad):
    """Patch si.read_openephys to return `good` with sync and `bad` without."""
    def fake(folder_path=None, stream_name=None, load_sync_timestamps=False, **kw):
        return good if load_sync_timestamps else bad

    monkeypatch.setattr(si, "read_openephys", fake)


def test_the_synced_clock_is_used_when_there_is_one(monkeypatch):
    n = 1000
    good = recording({"ADC0": np.zeros(n)}, times=9.8 + np.arange(n) / TRUE_FS)
    bad = recording({"ADC0": np.zeros(n)}, times=np.arange(n) / DECLARED_FS)
    synced(monkeypatch, good, bad)

    rec = read_openephys_synced("folder", "ADC")
    assert rec is good
    assert rec.get_times()[0] == pytest.approx(9.8)


def test_an_unsynced_stream_falls_back_and_says_so(monkeypatch):
    """Better a warned-about inferred clock than every timestamp at -1."""
    n = 1000
    unusable = recording({"ADC0": np.zeros(n)}, times=np.full(n, -1.0))
    inferred = recording({"ADC0": np.zeros(n)}, times=np.arange(n) / DECLARED_FS)
    synced(monkeypatch, unusable, inferred)

    with pytest.warns(UserWarning, match="no usable synchronized clock"):
        rec = read_openephys_synced("folder", "ADC")
    assert rec is inferred


def test_the_fallback_can_be_refused(monkeypatch):
    n = 1000
    unusable = recording({"ADC0": np.zeros(n)}, times=np.full(n, -1.0))
    synced(monkeypatch, unusable, unusable)

    with pytest.raises(ValueError, match="Refusing to use"):
        read_openephys_synced("folder", "ADC", require_sync=True)


def test_open_stream_routes_open_ephys_through_the_check(monkeypatch):
    n = 1000
    unusable = recording({"ADC0": np.zeros(n)}, times=np.full(n, -1.0))
    inferred = recording({"ADC0": np.zeros(n)}, times=np.arange(n) / DECLARED_FS)
    synced(monkeypatch, unusable, inferred)

    with pytest.warns(UserWarning, match="no usable synchronized clock"):
        open_stream("folder", "openephysbinary", "ADC")


def test_open_stream_rejects_an_unknown_system():
    with pytest.raises(ValueError, match="Unsupported phys_type"):
        open_stream("folder", "neuralynx", "ADC")


# ── the two clocks have different origins ───────────────────────────────
def open_ephys_tree(root, streams):
    """A minimal Open Ephys binary layout: {stream: first timestamp or None}.

    A negative first timestamp writes the all -1 file Open Ephys leaves behind
    for a stream it never synchronized; None writes no sidecar at all.
    """
    base = root / "Record Node 101" / "experiment1" / "recording1" / "continuous"
    for name, t0 in streams.items():
        folder = base / name
        folder.mkdir(parents=True)
        (folder / "continuous.dat").write_bytes(b"\0" * 16)
        if t0 is None:
            continue
        times = np.full(100, -1.0) if t0 < 0 else t0 + np.arange(100) / FS
        np.save(folder / "timestamps.npy", times)
    return root


def test_the_probe_start_time_is_read_without_opening_the_stream(tmp_path):
    """The ADC sits beside the probe and must not be mistaken for it."""
    open_ephys_tree(
        tmp_path,
        {"OneBox-109.ProbeA": 9.808033, "OneBox-109.OneBox-ADC": 9.825045},
    )
    assert probe_start_time(tmp_path, "openephysbinary") == pytest.approx(9.808033)


def test_an_unsynced_probe_has_no_start_time(tmp_path):
    """-1 is Open Ephys saying it never synced, not a time nine seconds early."""
    open_ephys_tree(tmp_path, {"OneBox-100.ProbeA": -1.0})
    assert probe_start_time(tmp_path, "openephysbinary") is None


def test_a_missing_sidecar_gives_no_start_time(tmp_path):
    open_ephys_tree(tmp_path, {"OneBox-109.ProbeA": None})
    assert probe_start_time(tmp_path, "openephysbinary") is None


def test_spikeglx_has_no_start_time_to_read(tmp_path):
    open_ephys_tree(tmp_path, {"OneBox-109.ProbeA": 9.8})
    assert probe_start_time(tmp_path, "spikeglx") is None


def test_shutter_times_land_on_the_spike_clock(tmp_path, monkeypatch):
    """Kilosort counts from the probe's first sample; the ADC does not.

    Leaving that gap in place puts every camera frame ~10 s late against every
    spike, which smears tuning curves toward flat rather than failing loudly.
    """
    probe_t0 = 9.808033
    root = open_ephys_tree(tmp_path / "phys", {"OneBox-109.ProbeA": probe_t0})

    n = int(TRUE_FS * DUR)
    measured = recording(
        {"ADC0": ttl(n, fs=TRUE_FS)},
        fs=DECLARED_FS,
        times=9.825 + np.arange(n) / TRUE_FS,
    )
    synced(monkeypatch, measured, measured)
    monkeypatch.setattr(shutter, "find_adc_stream", lambda *a, **k: "ADC")

    out = shutter.derive_shutter_times(
        root, tmp_path, "s1", {"save_sanity_plot": False}, "openephysbinary"
    )
    times = np.load(out)

    # the first pulse is 9.825 - 9.808 = 17 ms after the probe's first sample,
    # plus however far into the ADC trace the first exposure falls
    assert times[0] == pytest.approx(9.825 - probe_t0, abs=0.01)
    assert times[0] < 1.0, "still on the acquisition clock"


def test_an_unsynced_probe_leaves_the_times_alone(tmp_path, monkeypatch, capsys):
    """Both clocks are then inferred and share an origin; shifting would break it."""
    root = open_ephys_tree(tmp_path / "phys", {"OneBox-100.ProbeA": -1.0})

    n = int(TRUE_FS * DUR)
    events = recording(
        {"ADC0": ttl(n, fs=TRUE_FS)}, fs=DECLARED_FS, times=np.arange(n) / DECLARED_FS
    )
    synced(monkeypatch, events, events)
    monkeypatch.setattr(shutter, "find_adc_stream", lambda *a, **k: "ADC")

    out = shutter.derive_shutter_times(
        root, tmp_path, "s1", {"save_sanity_plot": False}, "openephysbinary"
    )
    assert np.load(out)[0] > 0
    assert "no synchronized clock" in capsys.readouterr().out


# ── the acquisition type has to survive the trip ────────────────────────
def test_derive_detects_the_acquisition_type_when_it_is_not_given(
    tmp_path, monkeypatch
):
    """phys_type=None used to be swallowed by an `else: read_openephys`.

    That made an unresolved acquisition type look harmless, right up until a
    SpikeGLX recording was read as Open Ephys. open_stream refuses it instead,
    so anything that can be handed a None has to resolve it first.
    """
    root = open_ephys_tree(tmp_path / "phys", {"OneBox-109.ProbeA": 9.808033})
    (root / "Record Node 101" / "experiment1" / "recording1"
     / "structure.oebin").write_text("{}")

    n = int(TRUE_FS * DUR)
    events = recording(
        {"ADC0": ttl(n, fs=TRUE_FS)}, fs=DECLARED_FS,
        times=9.825 + np.arange(n) / TRUE_FS,
    )
    synced(monkeypatch, events, events)
    monkeypatch.setattr(shutter, "find_adc_stream", lambda *a, **k: "ADC")

    out = shutter.derive_shutter_times(
        root, tmp_path, "s1", {"save_sanity_plot": False}, phys_type=None
    )
    assert out is not None and out.exists()
    # and the origin correction still ran, which needs the type resolved too
    assert np.load(out)[0] < 1.0


def test_score_session_resolves_the_type_before_using_it(monkeypatch, tmp_path):
    """The regression itself.

    score_session took phys_type=None, let read_recording detect it privately,
    and then passed the original None on to the movement veto -- which reached
    open_stream and failed there, four calls from where the type was known.
    """
    from spikeshpc import states

    seen = {}

    def fake_read_recording(phys_path, phys_type=None, stream_name=None, band="ap"):
        seen["phys_type"] = phys_type
        raise RuntimeError("far enough")

    monkeypatch.setattr(states, "detect_phys_type", lambda p: "openephysbinary")
    monkeypatch.setattr(states, "read_recording", fake_read_recording)

    with pytest.raises(RuntimeError, match="far enough"):
        states.score_session(tmp_path / "rec", tmp_path, {"lfp_rate": 1250.0})

    assert seen["phys_type"] == "openephysbinary", (
        "score_session must resolve the acquisition type before anything "
        "downstream has to guess it"
    )


# ── the sanity plot must be on one clock ────────────────────────────────
def ttl_with_a_late_start(n, start_s, fs, rate=RATE, width=4):
    """A pulse train that begins partway in, as OptiTrack's does."""
    trace = np.zeros(n)
    period = int(fs / rate)
    for k in range(int(start_s * fs), n - width, period):
        trace[k : k + width] = 1000.0
    return trace


def test_the_sanity_plot_puts_the_trace_on_the_marker_s_clock():
    """The figure that lied.

    Shutter times shifted onto the spike clock, drawn against a trace still on
    the acquisition clock. A pulse train is periodic, so an offset close to a
    whole number of frame periods lands the marker on the wrong pulse and the
    figure looks right everywhere the train is dense -- it only breaks at the
    start, before the camera began.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spikeshpc.optitrack.sync import (
        extract_shutter_close_times,
        plot_shutter_close_sanity_check,
    )

    fs, offset = 30000.0, 9.808033
    n = int(fs * 60)
    events = recording(
        {"ADC0": ttl_with_a_late_start(n, start_s=20.0, fs=fs)},
        fs=fs,
        times=offset + np.arange(n) / fs,
    )
    acquisition_clock = extract_shutter_close_times(events, channel_id="ADC0")
    spike_clock = acquisition_clock - offset

    told = plot_shutter_close_sanity_check(
        events, spike_clock, channel_id="ADC0", time_offset=offset
    )
    titles = [ax.get_title() for ax in told.axes]
    assert not any("NO EDGE" in t for t in titles), titles
    for title in titles:
        residual = float(title.split("\n")[1].split()[0])
        assert abs(residual) < 0.05, title
    plt.close(told)

    # and without being told, the first events fall where the camera had not
    # started yet -- which is exactly how the real figure gave itself away
    fooled = plot_shutter_close_sanity_check(
        events, spike_clock, channel_id="ADC0", time_offset=0.0
    )
    assert any("NO EDGE" in ax.get_title() for ax in fooled.axes)
    plt.close(fooled)


def test_derive_draws_the_sanity_plot_on_the_shifted_clock(tmp_path, monkeypatch):
    """End to end: the cached times and the plotted trace must agree."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    seen = {}

    def spy(events_raw, times, channel_id=None, time_offset=0.0, **kw):
        seen["time_offset"] = time_offset
        seen["first"] = float(times[0])
        return matplotlib.pyplot.figure()

    monkeypatch.setattr(shutter, "plot_shutter_close_sanity_check", spy)

    probe_t0 = 9.808033
    root = open_ephys_tree(tmp_path / "phys", {"OneBox-109.ProbeA": probe_t0})
    n = int(TRUE_FS * DUR)
    events = recording(
        {"ADC0": ttl(n, fs=TRUE_FS)}, fs=DECLARED_FS,
        times=9.825 + np.arange(n) / TRUE_FS,
    )
    synced(monkeypatch, events, events)
    monkeypatch.setattr(shutter, "find_adc_stream", lambda *a, **k: "ADC")

    shutter.derive_shutter_times(
        root, tmp_path, "s1", {"save_sanity_plot": True}, "openephysbinary"
    )
    assert seen["time_offset"] == pytest.approx(probe_t0)
    assert seen["first"] < 1.0, "times were shifted but the offset was not passed on"


# ── the bug this was found through ──────────────────────────────────────
def test_shutter_times_come_from_the_measured_clock(tmp_path, monkeypatch):
    """The regression test for the drift.

    The ADC's declared rate is 83 ppm slow, so timing the same samples from it
    stretches the shutter train. Over a real six-hour session that is two
    seconds -- a hundred-odd frames by the end. Here the run is short, so the
    test asserts the exact stretch factor rather than a wall-clock error.
    """
    n = int(TRUE_FS * DUR)
    traces = {"ADC0": ttl(n, fs=TRUE_FS)}
    measured = recording(traces, fs=DECLARED_FS, times=9.8 + np.arange(n) / TRUE_FS)
    declared = recording(traces, fs=DECLARED_FS, times=np.arange(n) / DECLARED_FS)
    synced(monkeypatch, measured, declared)
    monkeypatch.setattr(shutter, "find_adc_stream", lambda *a, **k: "ADC")

    out = shutter.derive_shutter_times(
        tmp_path / "rec", tmp_path, "s1", {"save_sanity_plot": False},
        "openephysbinary",
    )
    times = np.load(out)

    # it starts on the acquisition clock, not at zero
    assert times[0] > 9.8

    # The same samples, divided by two different rates. The pulse period is a
    # whole number of samples, so each clock gives an exact interval and the
    # only question is which divisor was used.
    period = int(TRUE_FS / RATE)
    assert np.diff(times).mean() == pytest.approx(period / TRUE_FS, rel=1e-6)
    assert np.diff(times).mean() < period / DECLARED_FS
    assert declared.get_times()[-1] > measured.get_times()[-1] - 9.8


def test_the_clock_is_reported_in_the_log(tmp_path, monkeypatch, capsys):
    """A silent regression here would be invisible until a tuning curve smeared."""
    n = int(TRUE_FS * DUR)
    measured = recording(
        {"ADC0": ttl(n, fs=TRUE_FS)},
        fs=DECLARED_FS,
        times=9.8 + np.arange(n) / TRUE_FS,
    )
    synced(monkeypatch, measured, measured)
    monkeypatch.setattr(shutter, "find_adc_stream", lambda *a, **k: "ADC")

    shutter.derive_shutter_times(
        tmp_path / "rec", tmp_path, "s1", {"save_sanity_plot": False},
        "openephysbinary",
    )
    out = capsys.readouterr().out
    assert "clock: starts at 9.800s" in out
    assert f"declared {DECLARED_FS:.1f} Hz" in out
    assert "realised 30303" in out
