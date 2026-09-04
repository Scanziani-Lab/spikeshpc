"""Tests for spikeshpc channel handling and stream inference."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from spikeshpc import channels as C
from spikeshpc import io as IO

INFO = {"channel_ids": [f"AP{i}" for i in range(8)], "num_channels": 8}


@pytest.mark.parametrize(
    "label,expected",
    [
        ("SY0", True),
        ("SY0;768:768", True),
        ("imec0.ap#SY0", True),
        ("AP0;0:0", False),
        ("LF3", False),
        ("0", False),
    ],
)
def test_sync_label_detection(label, expected):
    assert C._is_sync_label(label) is expected


@pytest.mark.parametrize(
    "bad,expected",
    [
        ([], ([], [])),
        (["AP3", "AP1"], ([1, 3], ["AP1", "AP3"])),
        ([5, 2], ([2, 5], ["AP2", "AP5"])),
        (["AP3", 3, 0], ([0, 3], ["AP0", "AP3"])),  # duplicates collapse
    ],
)
def test_resolve_bad_channels(bad, expected):
    assert C.resolve_bad_channels(bad, INFO) == expected


@pytest.mark.parametrize("bad", [["AP99"], [8], [-1], ["typo"], [True]])
def test_resolve_bad_channels_rejects_unknown(bad):
    with pytest.raises(ValueError, match="not found in this recording"):
        C.resolve_bad_channels(bad, INFO)


SGLX = ["imec0.ap", "imec0.ap-SYNC", "imec0.lf", "nidq"]
OE_NP2 = [
    "Record Node 101#Neuropix-PXI-100.ProbeA",
    "Record Node 101#Neuropix-PXI-100.ProbeA-ADC",
    "Record Node 101#NI-DAQmx-103.PXIe-6341",
]
OE_NP1 = [
    "Record Node 102#Neuropix-PXI-100.ProbeA-AP",
    "Record Node 102#Neuropix-PXI-100.ProbeA-LFP",
]


@pytest.mark.parametrize(
    "phys_type,streams,band,expected",
    [
        ("spikeglx", SGLX, "ap", "imec0.ap"),
        ("spikeglx", SGLX, "lf", "imec0.lf"),
        ("openephysbinary", OE_NP2, "ap", OE_NP2[0]),
        # Neuropixels 2.0 has no LF band; the caller decimates the AP band
        ("openephysbinary", OE_NP2, "lf", None),
        ("openephysbinary", OE_NP1, "ap", OE_NP1[0]),
        ("openephysbinary", OE_NP1, "lf", OE_NP1[1]),
    ],
)
def test_infer_stream_name(phys_type, streams, band, expected):
    with mock.patch(
        "spikeinterface.extractors.get_neo_streams",
        return_value=(streams, list(range(len(streams)))),
    ):
        got = IO.infer_stream_name(Path("/fake"), Path("/fake"), phys_type, band)
    assert got == expected


@pytest.mark.parametrize(
    "phys_type,streams",
    [
        ("spikeglx", ["imec0.ap", "imec1.ap"]),
        ("openephysbinary", OE_NP2 + ["Record Node 101#Neuropix-PXI-100.ProbeB"]),
    ],
)
def test_multiple_probes_must_be_disambiguated(phys_type, streams):
    with mock.patch(
        "spikeinterface.extractors.get_neo_streams",
        return_value=(streams, list(range(len(streams)))),
    ):
        with pytest.raises(ValueError, match="Multiple AP streams"):
            IO.infer_stream_name(Path("/fake"), Path("/fake"), phys_type, "ap")


def test_missing_ap_stream_raises():
    with mock.patch(
        "spikeinterface.extractors.get_neo_streams", return_value=(["nidq"], [0])
    ):
        with pytest.raises(ValueError, match="Could not pick an AP stream"):
            IO.infer_stream_name(Path("/fake"), Path("/fake"), "spikeglx", "ap")


def test_spikeglx_binary_filename_names_its_stream(tmp_path):
    binfile = tmp_path / "m1_g0_t0.imec0.ap.bin"
    binfile.touch()
    assert IO.infer_stream_name(binfile, tmp_path, "spikeglx", "ap") == "imec0.ap"


def test_gain_mismatch_is_reported():
    import probeinterface
    import spikeinterface.full as si

    def make(gain):
        rec = si.NumpyRecording(np.zeros((100, 4), "int16"), sampling_frequency=30000.0)
        probe = probeinterface.Probe(ndim=2)
        probe.set_contacts(
            positions=np.c_[np.zeros(4), np.arange(4) * 20.0],
            shapes="circle", shape_params={"radius": 5},
        )
        probe.set_device_channel_indices(np.arange(4))
        rec = rec.set_probe(probe)
        rec.set_property("gain_to_uV", np.full(4, gain))
        return rec

    assert C.check_gain_consistency([make(2.34), make(2.34)]) is True
    assert C.check_gain_consistency([make(2.34), make(4.68)]) is False
