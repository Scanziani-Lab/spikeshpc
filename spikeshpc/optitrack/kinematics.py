"""Position, velocity, acceleration, and head turning from a rigid body's OptiTrack track."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def compute_kinematics(
    position: np.ndarray,
    times: np.ndarray,
    smooth_window: int = 11,
    polyorder: int = 3,
) -> dict[str, np.ndarray]:
    """Smoothed position, velocity, and acceleration for an (n, 3) position track.

    Velocity and acceleration are taken as the Savitzky-Golay filter's 1st/2nd
    derivatives directly (``deriv=1``/``2``), rather than by differentiating
    raw frame-to-frame position deltas, since finite-differencing unsmoothed
    marker positions amplifies tracking jitter into large spurious values.

    ``times`` is assumed uniformly sampled (true for a fixed-frame-rate
    OptiTrack take); the sample spacing used for the derivatives is its
    median step.
    """
    position = np.asarray(position, dtype=float)
    dt = float(np.median(np.diff(times)))

    position_smoothed = savgol_filter(position, smooth_window, polyorder, axis=0)
    velocity = savgol_filter(
        position, smooth_window, polyorder, deriv=1, delta=dt, axis=0
    )
    acceleration = savgol_filter(
        position, smooth_window, polyorder, deriv=2, delta=dt, axis=0
    )

    return {
        "position_smoothed": position_smoothed,
        "velocity": velocity,
        "speed": np.linalg.norm(velocity, axis=1),
        "acceleration": acceleration,
        "accel_magnitude": np.linalg.norm(acceleration, axis=1),
    }


def compute_angular_velocity(
    heading_deg: np.ndarray,
    times: np.ndarray,
    smooth_window: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Angular head velocity in deg/s: how fast the heading turns, per frame.

    Signed, positive in the direction heading increases. The heading is
    unwrapped first, so a turn across 0/360 is a small step rather than a
    360-degree jump, and then differentiated with the same Savitzky-Golay
    filter :func:`compute_kinematics` uses, for the same reason: a raw
    frame-to-frame difference turns tracking jitter into spurious spikes.

    ``times`` is assumed uniformly sampled, as there; the sample spacing is its
    median step.
    """
    unwrapped = np.unwrap(np.deg2rad(np.asarray(heading_deg, dtype=float)))
    dt = float(np.median(np.diff(times)))
    turning = savgol_filter(unwrapped, smooth_window, polyorder, deriv=1, delta=dt)
    return np.rad2deg(turning)
