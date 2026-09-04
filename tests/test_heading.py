from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spikeshpc.optitrack.heading import (
    IDENTITY_HEAD_FRAME,
    calibrate_head_frame,
    compute_head_elevation,
    compute_heading,
)


def _single_axis(axis: str, angles, degrees: bool = False) -> Rotation:
    """Rotation about one axis, for a scalar or a sequence of angles.

    scipy 1.18 requires the angles' last dimension to match the number of
    sequence axes, so a 1-D array of N angles for a single axis has to be
    passed as (N, 1). Older scipy accepted both shapes.
    """
    angles = np.asarray(angles, dtype=float)
    if angles.ndim == 1:
        angles = angles[:, None]
    return Rotation.from_euler(axis, angles, degrees=degrees)


def _quat_for_yaw(yaw_deg: float | np.ndarray) -> np.ndarray:
    """Quaternion for a rotation of ``yaw_deg`` about the Y (up) axis."""
    return _single_axis("y", yaw_deg, degrees=True).as_quat()


def test_zero_yaw_is_zero():
    rotation_xyzw = _quat_for_yaw(0.0)[None, :]
    heading = compute_heading(rotation_xyzw, IDENTITY_HEAD_FRAME)
    assert np.isclose(heading[0], 0.0)


def test_yaw_is_recovered_exactly_beyond_ninety_degrees():
    """Regression: an Euler-angle "Y" component would fold this into [-90, 90]
    instead of tracking the full circle."""
    rotation_xyzw = np.stack([_quat_for_yaw(y) for y in (90.0, 180.0, 260.0)])
    heading = compute_heading(rotation_xyzw, IDENTITY_HEAD_FRAME)
    assert np.allclose(heading, [90.0, 180.0, 260.0], atol=1e-6)


def test_yaw_wraps_to_zero_to_360():
    heading = compute_heading(_quat_for_yaw(-45.0)[None, :], IDENTITY_HEAD_FRAME)
    assert np.isclose(heading[0], 315.0)


def test_pitch_and_roll_about_the_nose_leave_heading_alone():
    """Heading must track the nose's compass azimuth only. Rolling about the
    nose axis can't move it at all, and pitching about the lateral axis only
    changes the nose's elevation, so neither may perturb the reported yaw."""
    yaw = Rotation.from_euler("y", 40.0, degrees=True)
    quats = [
        (yaw * Rotation.from_euler(axis, angle, degrees=True)).as_quat()
        for axis in "zx"
        for angle in (-60.0, -20.0, 0.0, 20.0, 60.0)
    ]
    heading = compute_heading(np.stack(quats), IDENTITY_HEAD_FRAME)
    assert np.allclose(heading, 40.0, atol=1e-6)


def test_heading_follows_the_calibrated_forward_axis_not_a_body_axis():
    """The nose generally isn't along a body axis, and heading must follow the
    nose. A forward axis 30 degrees off body +Z has to read 30 degrees off."""
    forward = Rotation.from_euler("y", 30.0, degrees=True).apply([0.0, 0.0, 1.0])
    heading = compute_heading(_quat_for_yaw(10.0)[None, :], forward)
    assert np.isclose(heading[0], 40.0, atol=1e-6)


def test_flip_direction_reverses_the_turn_but_not_the_zero_reference():
    rotation_xyzw = np.stack([_quat_for_yaw(0.0), _quat_for_yaw(90.0)])
    heading = compute_heading(rotation_xyzw, IDENTITY_HEAD_FRAME, flip_direction=True)
    assert np.allclose(heading, [0.0, 270.0], atol=1e-6)


def test_offset_rotates_the_zero_reference_without_reversing_the_turn():
    rotation_xyzw = np.stack([_quat_for_yaw(0.0), _quat_for_yaw(90.0)])
    heading = compute_heading(rotation_xyzw, IDENTITY_HEAD_FRAME, offset_deg=45.0)
    assert np.allclose(heading, [45.0, 135.0], atol=1e-6)


def test_elevation_is_signed_and_zero_for_a_level_head():
    level = compute_head_elevation(_quat_for_yaw(37.0)[None, :], IDENTITY_HEAD_FRAME)
    assert np.isclose(level[0], 0.0, atol=1e-9)

    # Nose up: an intrinsic -25 degree pitch about X lifts body +Z off the floor.
    nose_up = Rotation.from_euler("x", -25.0, degrees=True).as_quat()[None, :]
    assert np.isclose(compute_head_elevation(nose_up, IDENTITY_HEAD_FRAME)[0], 25.0)


_ARBITRARY_BODY = Rotation.from_euler("yxz", [40.0, 15.0, -20.0], degrees=True)
_N_TURNS = 4.0


def _synthetic_run(
    forward_body: np.ndarray,
    up_body: np.ndarray,
    constant_tilt_deg: float = 0.0,
    jitter_deg: float = 0.0,
    n: int = 6000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An animal circling the arena with its nose exactly on the tangent.

    The true world orientation of the head is built first, then re-expressed in
    a rigid-body frame whose axes are ``forward_body`` / ``up_body`` -- exactly
    the situation Motive leaves you in when the body was created at an arbitrary
    head orientation. ``jitter_deg`` adds zero-mean pitch and roll on top of
    ``constant_tilt_deg``.
    """
    rng = np.random.default_rng(seed)
    times = np.arange(n) / 120.0
    yaw = np.linspace(0.0, _N_TURNS * 2 * np.pi, n)

    radius = 300.0
    position = np.column_stack(
        [radius * -np.cos(yaw), np.zeros(n), radius * np.sin(yaw)]
    )

    # Anatomical frame is the arena's own: +Z nose, +Y dorsal, +X left.
    left_body = np.cross(up_body, forward_body)
    body_from_anatomical = np.column_stack([left_body, up_body, forward_body])

    pitch = np.radians(constant_tilt_deg + jitter_deg * rng.standard_normal(n))
    roll = np.radians(jitter_deg * rng.standard_normal(n))
    world_from_anatomical = (
        _single_axis("y", yaw)
        * _single_axis("x", pitch)
        * _single_axis("z", roll)
    ).as_matrix()

    matrices = world_from_anatomical @ body_from_anatomical.T
    return Rotation.from_matrix(matrices).as_quat(), position, times


def test_calibration_recovers_an_arbitrary_rigid_body_orientation():
    """The whole point: the local axes carry no anatomical meaning, so the nose
    and dorsal axes have to come back out of the locomotion itself."""
    forward_body = _ARBITRARY_BODY.apply([0.0, 0.0, 1.0])
    up_body = _ARBITRARY_BODY.apply([0.0, 1.0, 0.0])
    rotation_xyzw, position, times = _synthetic_run(
        forward_body, up_body, jitter_deg=20.0
    )

    frame = calibrate_head_frame(
        rotation_xyzw, position, times, speed_threshold_mm_s=10.0
    )

    assert np.degrees(np.arccos(np.clip(frame.forward @ forward_body, -1, 1))) < 2.0
    assert np.degrees(np.arccos(np.clip(frame.up @ up_body, -1, 1))) < 2.0
    assert frame.concentration > 0.9
    assert abs(frame.residual_deg) < 2.0

    # Orthonormal by construction.
    axes = np.column_stack([frame.forward, frame.left, frame.up])
    assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-9)


def test_calibrated_heading_tracks_true_yaw_through_a_constant_tilt():
    """A head held at a fixed 40 degree pitch the whole take. The recovered
    ``up`` necessarily absorbs that tilt -- a permanently tilted head is
    indistinguishable from a tilted marker plate, and neither the CSV nor the
    behaviour can separate them -- but the *heading* must still be exact,
    because ``forward`` is fixed against travel rather than against ``up``."""
    forward_body = _ARBITRARY_BODY.apply([0.0, 0.0, 1.0])
    up_body = _ARBITRARY_BODY.apply([0.0, 1.0, 0.0])
    rotation_xyzw, position, times = _synthetic_run(
        forward_body, up_body, constant_tilt_deg=40.0
    )

    frame = calibrate_head_frame(
        rotation_xyzw, position, times, speed_threshold_mm_s=10.0
    )
    heading = compute_heading(rotation_xyzw, frame)

    expected = np.degrees(np.linspace(0.0, _N_TURNS * 2 * np.pi, len(heading))) % 360.0
    error = np.abs(np.degrees(np.angle(np.exp(1j * np.radians(heading - expected)))))
    assert error.max() < 1.0


def test_projecting_forward_perpendicular_to_up_survives_a_running_lean():
    """A mouse pitches into the direction it is running, so the raw alignment
    with travel tilts out of the horizontal plane. Left unprojected it would
    score *better* against travel while being a worse heading estimate, so the
    projection has to hold the nose axis near horizontal anyway."""
    forward_body = _ARBITRARY_BODY.apply([0.0, 0.0, 1.0])
    up_body = _ARBITRARY_BODY.apply([0.0, 1.0, 0.0])
    rotation_xyzw, position, times = _synthetic_run(
        forward_body, up_body, constant_tilt_deg=20.0, jitter_deg=15.0
    )

    frame = calibrate_head_frame(
        rotation_xyzw, position, times, speed_threshold_mm_s=10.0
    )
    assert abs(frame.forward @ frame.up) < 1e-9

    elevation = compute_head_elevation(rotation_xyzw, frame)
    assert abs(np.median(elevation)) < 5.0


def test_calibration_refuses_a_take_with_no_locomotion():
    rotation_xyzw = np.tile(_quat_for_yaw(0.0), (1000, 1))
    position = np.zeros((1000, 3))
    times = np.arange(1000) / 120.0
    with pytest.raises(ValueError, match="to calibrate"):
        calibrate_head_frame(rotation_xyzw, position, times)
