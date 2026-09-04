"""Deriving camera-shutter times from the recording's own ADC stream."""

import numpy as np
import pytest

from spikeshpc import shutter
from spikeshpc.io import infer_stream_name
from spikeshpc.optitrack import sync

FS = 30000.0
RATE = 120.0
DUR = 20.0


def ttl(n, width=4, rate=RATE, seed=0):
    """A narrow exposure pulse train, like a real camera strobe."""
    rng = np.random.default_rng(seed)
    trace = rng.integers(-1, 2, n).astype(float)
    period = int(FS / rate)
    for k in range(0, n - width, period):
        trace[k: k + width] = 1000.0
    return trace


def floating(n, seed=1):
    rng = np.random.default_rng(seed)
    drift = np.cumsum(rng.normal(0, 0.002, n))
    return rng.normal(0, 0.05, n) + (drift - drift.mean())


def adc_stream(traces):
    import spikeinterface.full as si

    ids = list(traces)
    data = np.column_stack([traces[c] for c in ids]).astype("float32")
    return si.NumpyRecording(data, sampling_frequency=FS, channel_ids=ids)


def motive_csv(path, n_frames, frame_rate=RATE):
    header = [["Format Version", "1.23", "Capture Frame Rate", f"{frame_rate:.6f}",
               "Total Exported Frames", str(n_frames)], [],
              ["", "", "Rigid Body", "Rigid Body", "Rigid Body"],
              ["", "", "Headset", "Headset", "Headset"],
              ["", "", "1", "1", "1"], ["", "", "", "", ""],
              ["", "", "Position", "Position", "Position"],
              ["Frame", "Time (Seconds)", "X", "Y", "Z"]]
    lines = [",".join(map(str, r)) for r in header]
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        x, y, z = rng.normal(0, 50, 3)
        lines.append(f"{i},{i / frame_rate:.6f},{x:.4f},{y:.4f},{z:.4f}")
    path.write_text("\n".join(lines) + "\n")


# ── ADC stream discovery ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "phys_type,streams,expected",
    [
        ("openephysbinary",
         ["Record Node 101#OneBox-108.ProbeA",
          "Record Node 101#OneBox-108.OneBox-ADC"],
         "Record Node 101#OneBox-108.OneBox-ADC"),
        ("openephysbinary",
         ["Record Node 101#Neuropix-PXI-100.ProbeA",
          "Record Node 101#Neuropix-PXI-100.ProbeA-ADC"],
         "Record Node 101#Neuropix-PXI-100.ProbeA-ADC"),
        ("spikeglx", ["imec0.ap", "imec0.lf", "nidq"], "nidq"),
        # no analog inputs recorded
        ("openephysbinary", ["Record Node 101#Neuropix-PXI-100.ProbeA"], None),
    ],
)
def test_adc_stream_is_found(phys_type, streams, expected):
    from pathlib import Path
    from unittest import mock

    with mock.patch("spikeinterface.extractors.get_neo_streams",
                    return_value=(streams, list(range(len(streams))))):
        got = infer_stream_name(Path("/fake"), Path("/fake"), phys_type, "adc")
    assert got == expected


# ── channel selection ───────────────────────────────────────────────────
def test_ttl_channel_is_auto_detected_among_dead_inputs():
    n = int(FS * DUR)
    events = adc_stream({"ADC0": floating(n, 1), "ADC1": ttl(n),
                         "ADC2": floating(n, 2)})
    assert shutter.pick_ttl_channel(events) == "ADC1"


def test_named_channel_is_verified_not_trusted():
    """Naming a dead input must fail loudly, not silently threshold noise."""
    n = int(FS * DUR)
    events = adc_stream({"ADC0": floating(n), "ADC1": ttl(n)})
    assert shutter.pick_ttl_channel(events, "ADC1") == "ADC1"
    with pytest.raises(ValueError, match="does not look like a TTL"):
        shutter.pick_ttl_channel(events, "ADC0")


def test_no_ttl_anywhere_is_reported():
    n = int(FS * DUR)
    events = adc_stream({"ADC0": floating(n, 1), "ADC1": floating(n, 2)})
    with pytest.raises(ValueError, match="No channel on this stream looks like a TTL"):
        shutter.pick_ttl_channel(events)


def test_two_ttls_must_be_disambiguated():
    n = int(FS * DUR)
    events = adc_stream({"ADC0": ttl(n), "ADC1": ttl(n, rate=60.0)})
    with pytest.raises(ValueError, match="set movement.adc_channel"):
        shutter.pick_ttl_channel(events)


# ── end to end ──────────────────────────────────────────────────────────
def _patch_reader(monkeypatch, events):
    import spikeinterface.full as si

    monkeypatch.setattr(si, "read_openephys", lambda **kw: events)
    monkeypatch.setattr(shutter, "find_adc_stream", lambda *a, **k: "ADC")


def test_derive_extracts_checks_caches_and_plots(tmp_path, monkeypatch):
    n = int(FS * DUR)
    events = adc_stream({"ADC0": floating(n), "ADC1": ttl(n)})
    _patch_reader(monkeypatch, events)

    csv = tmp_path / "take.csv"
    motive_csv(csv, n_frames=int(RATE * DUR))

    out = shutter.derive_shutter_times(
        tmp_path / "rec", tmp_path, "s1", {}, "openephysbinary", csv
    )
    assert out is not None and out.exists()
    times = np.load(out)
    assert len(times) == pytest.approx(RATE * DUR, rel=0.02)
    assert np.diff(times).mean() == pytest.approx(1 / RATE, rel=0.01)
    assert (tmp_path / "states" / "s1_shutter_check.png").exists()


def test_derive_reuses_its_cache(tmp_path, monkeypatch, capsys):
    states = tmp_path / "states"
    states.mkdir()
    sentinel = np.arange(5.0)
    np.save(states / "s1_shutter_close_times.npy", sentinel)

    # would raise if it tried to read the recording at all
    monkeypatch.setattr(shutter, "find_adc_stream",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    out = shutter.derive_shutter_times(tmp_path / "rec", tmp_path, "s1", {})
    np.testing.assert_array_equal(np.load(out), sentinel)
    assert "reusing" in capsys.readouterr().out


def test_a_frame_count_mismatch_is_not_cached(tmp_path, monkeypatch, capsys):
    """Mistimed frames would mis-place every spike, so refuse the result."""
    n = int(FS * DUR)
    events = adc_stream({"ADC0": ttl(n)})
    _patch_reader(monkeypatch, events)

    csv = tmp_path / "take.csv"
    motive_csv(csv, n_frames=99999)          # nothing like the detected count

    out = shutter.derive_shutter_times(
        tmp_path / "rec", tmp_path, "s1", {}, "openephysbinary", csv
    )
    assert out is None
    assert not (tmp_path / "states" / "s1_shutter_close_times.npy").exists()
    assert "refusing to cache" in capsys.readouterr().out


def test_no_adc_stream_skips_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(shutter, "find_adc_stream", lambda *a, **k: None)
    assert shutter.derive_shutter_times(tmp_path / "rec", tmp_path, "s1", {}) is None
    assert "no ADC stream" in capsys.readouterr().out


def test_dead_adc_channel_skips_instead_of_returning_noise(tmp_path, monkeypatch, capsys):
    n = int(FS * DUR)
    _patch_reader(monkeypatch, adc_stream({"ADC0": floating(n)}))
    out = shutter.derive_shutter_times(
        tmp_path / "rec", tmp_path, "s1", {}, "openephysbinary", None
    )
    assert out is None
    assert "skipping" in capsys.readouterr().out


def test_load_movement_derives_frame_times_when_absent(tmp_path, monkeypatch, capsys):
    """The point of all this: no notebook step needed before scoring."""
    from spikeshpc.states import load_movement

    n = int(FS * DUR)
    _patch_reader(monkeypatch, adc_stream({"ADC0": ttl(n)}))
    csv = tmp_path / "s1.csv"
    motive_csv(csv, n_frames=int(RATE * DUR))

    cfg = {"optitrack_csv": str(tmp_path / "{session}.csv"), "frame_times": None}
    speed = load_movement(
        cfg, "s1", np.arange(5) + 0.5, 1.0,
        phys_path=tmp_path / "rec", output_dir=tmp_path,
        phys_type="openephysbinary",
    )
    assert speed is not None and np.isfinite(speed).any()
    out = capsys.readouterr().out
    assert "shutter:" in out and "cached" in out
