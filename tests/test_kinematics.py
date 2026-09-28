from __future__ import annotations

import numpy as np

from spikeshpc.optitrack.kinematics import compute_angular_velocity, compute_kinematics


def test_constant_velocity_is_recovered():
    times = np.linspace(0, 10, 200)
    velocity_true = np.array([2.0, -1.0, 0.5])
    position = np.outer(times, velocity_true)

    kin = compute_kinematics(position, times)

    assert np.allclose(kin["velocity"][10:-10], velocity_true, atol=1e-6)
    assert np.allclose(kin["acceleration"][10:-10], 0.0, atol=1e-6)


def test_constant_acceleration_is_recovered():
    times = np.linspace(0, 10, 200)
    accel_true = np.array([1.0, 0.0, -0.5])
    velocity0 = np.array([0.5, 1.0, 0.0])
    position = velocity0 * times[:, None] + 0.5 * accel_true * times[:, None] ** 2

    kin = compute_kinematics(position, times)

    expected_velocity = velocity0 + accel_true * times[:, None]
    assert np.allclose(kin["velocity"][10:-10], expected_velocity[10:-10], atol=1e-3)
    assert np.allclose(kin["acceleration"][10:-10], accel_true, atol=1e-3)
    assert np.allclose(kin["speed"], np.linalg.norm(kin["velocity"], axis=1))


def test_a_steady_turn_across_north_has_a_steady_angular_velocity():
    """Wrapping from 359 to 0 degrees is a small step, not a 360-degree lurch."""
    times = np.arange(600) / 60.0
    left = (350.0 + 30.0 * times) % 360.0
    right = (10.0 - 30.0 * times) % 360.0

    assert np.allclose(compute_angular_velocity(left, times), 30.0, atol=1e-6)
    assert np.allclose(compute_angular_velocity(right, times), -30.0, atol=1e-6)
