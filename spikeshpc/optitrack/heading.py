"""Heading (yaw) angle from OptiTrack rigid-body quaternions.

Arena convention: Y points up toward the overhead cameras, X runs along the
short edge and Z along the long edge, so the floor is the X-Z plane and
heading is a compass azimuth within it::

    heading(t) = atan2(v_x, v_z),   v = R(t) @ forward

measured from +Z toward +X, which is the right-handed sense about +Y. This is
the quantity a head-direction tuning curve wants: the allocentric azimuth of
the animal's nose, unaffected by how much the head is pitched or rolled at the
time.

Everything hinges on ``forward`` -- the nose direction *in the rigid body's own
local frame* -- and the trap is that it cannot be guessed. Motive defines a
rigid body's local axes by freezing the world axes at the moment the body is
created, so local +X/+Y/+Z carry no anatomical meaning at all: they record
whatever orientation the head happened to have on the rig when you clicked
"create". In the 2026-08-20 take the nose sits 26 degrees off local -Z, and
local +Y lands within 6 degrees of the head's dorsal axis only because the
animal happened to be upright at creation time. Guess a forward axis and the
heading comes out rotated by an arbitrary constant -- which is exactly the
"plotting it over the video makes no sense" failure mode.

:func:`calibrate_head_frame` recovers the axes from the data instead, using
the fact that a running mouse travels roughly where its head points. On this
take the recovered heading sits within 1.4 degrees of the travel direction on
average across every speed band from 100 to 300 mm/s, and the closed-form
solution agrees with a brute-force sweep over all candidate forward axes to
0.09 degrees.

Two other approaches were tried, and it's worth recording why both are wrong:

1. A twist-swing decomposition about world Y, reporting the twist
   (``2 * atan2(qy, qw)``). This is exactly invariant to head tilt, which
   sounds like what's wanted, but the twist about a *world* axis is not the
   azimuth of a *body* axis -- the two coincide only when the head is level.
   Measured against the calibrated heading on this take the twist is off by a
   constant -153.7 degrees, plus a tilt-dependent residual: a median of 0.5
   degrees while the head is within 15 degrees of upright, but 10.9 degrees at
   45-60 degrees of tilt and 19.9 degrees beyond that. This animal spends 21%
   of frames tilted more than 45 degrees, so that residual is not a corner case.
2. Reading an angle out of a quaternion-to-Euler decomposition. The Y
   component of Motive's own intrinsic "XYZ" convention (confirmed to match
   ``Rotation.as_euler("XYZ")`` to ~0.001 degrees over 2000 frames of a
   from-Motive Eulerian export) is the *middle* angle of that sequence, which
   is mathematically confined to [-90, 90] and is not a global azimuth at all.
   An outer-axis-first sequence does span the full circle, and the choice of
   second axis is not arbitrary the way it first appears -- it selects *which
   body axis* you get the azimuth of. ``as_euler("YXZ")[:, 0]`` is identically
   the azimuth of body +Z, and ``as_euler("YZX")[:, 0]`` is identically the
   azimuth of body +X minus 90 degrees (both verified to 1e-13 over 20000
   random rotations). So an Euler route can give the right answer, but only
   once you already know the nose axis -- and if the nose isn't along a body
   axis, as here, none of the six sequences is it. Projecting a calibrated
   forward vector is the same computation without the special cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import compute_kinematics


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _format_vector(v: np.ndarray) -> str:
    return "[" + " ".join(f"{c:+.4f}" for c in v) + "]"


@dataclass
class HeadFrame:
    """The animal's anatomical axes, expressed in the rigid body's local frame.

    ``forward`` (nose), ``left``, and ``up`` (dorsal) are orthonormal. The
    remaining fields describe how well the calibration that produced them was
    constrained -- see :func:`calibrate_head_frame`.
    """

    forward: np.ndarray  # (3,)
    left: np.ndarray  # (3,)
    up: np.ndarray  # (3,)
    n_calibration_frames: int
    concentration: float  # circular resultant length of heading - travel, in [0, 1]
    residual_deg: float  # mean of heading - travel; ~0 by construction

    def __str__(self) -> str:
        return (
            f"HeadFrame(forward={_format_vector(self.forward)}, "
            f"left={_format_vector(self.left)}, up={_format_vector(self.up)})\n"
            f"  calibrated on {self.n_calibration_frames} running frames; "
            f"heading-vs-travel concentration R={self.concentration:.3f}, "
            f"residual {self.residual_deg:+.2f} deg"
        )


#: A rigid body whose local axes already are the anatomical ones -- nose along
#: +Z, dorsal along +Y. Only correct if the body was created with the animal
#: level and facing +Z; use :func:`calibrate_head_frame` otherwise.
IDENTITY_HEAD_FRAME = HeadFrame(
    forward=np.array([0.0, 0.0, 1.0]),
    left=np.array([1.0, 0.0, 0.0]),
    up=np.array([0.0, 1.0, 0.0]),
    n_calibration_frames=0,
    concentration=float("nan"),
    residual_deg=float("nan"),
)


def calibrate_head_frame(
    rotation_xyzw: np.ndarray,
    position: np.ndarray,
    times: np.ndarray,
    speed_threshold_mm_s: float = 150.0,
    smooth_window: int = 31,
    polyorder: int = 3,
    min_calibration_frames: int = 500,
) -> HeadFrame:
    """Recover the animal's anatomical axes in the rigid body's local frame.

    Two cues, each solved in closed form:

    ``up`` is the body-frame direction that points at world +Y on average over
    the whole take. Row 1 of each world-from-body rotation matrix is world up
    written in body coordinates, so averaging those rows and normalizing gives
    it directly. This works because the animal is upright far more often than
    not; it does not require any frame in particular to be upright.

    ``forward`` is the body-frame direction that points along travel during
    locomotion. Maximizing ``sum_t (R_t @ f) . d_t`` over unit ``f``, with
    ``d_t`` the ground-plane travel direction, has the solution
    ``f = normalize(sum_t R_t.T @ d_t)`` -- a Wahba-style alignment with no
    search involved.

    That raw solution is then projected perpendicular to ``up``, which matters
    more than it looks: a running mouse pitches into the direction it is
    heading, so the unprojected fit tilts out of the horizontal plane (by 21
    degrees on this take) chasing that lean. Left unprojected it scores
    *better* against travel while being a much worse heading estimate, because
    a near-vertical axis has an azimuth that spins meaninglessly whenever the
    animal is not running.

    ``speed_threshold_mm_s`` selects the locomotion bouts. Head and travel
    direction decouple at low speed -- on this take the circular concentration
    between them falls from 0.75 above 300 mm/s to 0.04 in the 50-100 mm/s
    band -- so a threshold well into running is what makes the fit sharp.
    """
    rotation_xyzw = np.asarray(rotation_xyzw, dtype=float)
    matrices = Rotation.from_quat(rotation_xyzw).as_matrix()

    up = _normalize(matrices[:, 1, :].mean(axis=0))

    velocity = compute_kinematics(position, times, smooth_window, polyorder)["velocity"]
    ground_speed = np.hypot(velocity[:, 0], velocity[:, 2])
    running = ground_speed > speed_threshold_mm_s
    n_running = int(running.sum())
    if n_running < min_calibration_frames:
        raise ValueError(
            f"Only {n_running} frames exceed {speed_threshold_mm_s} mm/s, need at "
            f"least {min_calibration_frames} to calibrate. Lower "
            f"`speed_threshold_mm_s`, or pass a `HeadFrame` determined some other "
            f"way if the animal never really locomotes in this take."
        )

    travel = np.zeros((n_running, 3))
    travel[:, 0] = velocity[running, 0] / ground_speed[running]
    travel[:, 2] = velocity[running, 2] / ground_speed[running]

    forward = _normalize(np.einsum("nji,nj->i", matrices[running], travel))
    forward = _normalize(forward - (forward @ up) * up)
    left = np.cross(up, forward)

    heading = _azimuth(matrices[running], forward)
    resultant = np.exp(1j * (heading - np.arctan2(travel[:, 0], travel[:, 2]))).mean()

    return HeadFrame(
        forward=forward,
        left=left,
        up=up,
        n_calibration_frames=n_running,
        concentration=float(np.abs(resultant)),
        residual_deg=float(np.degrees(np.angle(resultant))),
    )


def _azimuth(matrices: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Radian azimuth of ``forward`` carried into the world by each matrix."""
    v = matrices @ forward
    return np.arctan2(v[:, 0], v[:, 2])


def _forward_vector(head_frame: HeadFrame | np.ndarray) -> np.ndarray:
    forward = head_frame.forward if isinstance(head_frame, HeadFrame) else head_frame
    return _normalize(np.asarray(forward, dtype=float))


def compute_heading(
    rotation_xyzw: np.ndarray,
    head_frame: HeadFrame | np.ndarray,
    flip_direction: bool = False,
    offset_deg: float = 0.0,
) -> np.ndarray:
    """Heading in degrees, wrapped to [0, 360).

    ``head_frame`` is a :class:`HeadFrame` from :func:`calibrate_head_frame`,
    or a bare (3,) forward vector in the rigid body's local frame. It has no
    default on purpose: the previous signature defaulted to a guessed axis and
    silently returned a heading rotated by a constant.

    ``offset_deg`` rotates the zero reference and ``flip_direction`` reverses
    the turn sense without moving it. Neither is needed to make the heading
    self-consistent -- calibration already fixes both against the animal's own
    travel direction -- but the overhead camera's orientation relative to the
    arena axes isn't in the CSV, so they're the knobs for lining the compass up
    with a top-down video.
    """
    rotation_xyzw = np.asarray(rotation_xyzw, dtype=float)
    matrices = Rotation.from_quat(rotation_xyzw).as_matrix()
    heading = np.degrees(_azimuth(matrices, _forward_vector(head_frame)))
    if flip_direction:
        heading = -heading
    return (heading + offset_deg) % 360.0


def compute_head_elevation(
    rotation_xyzw: np.ndarray,
    head_frame: HeadFrame | np.ndarray,
) -> np.ndarray:
    """Degrees the nose sits above (+) or below (-) horizontal, per frame.

    Heading is the azimuth of a projection onto the floor, so it degrades as
    the nose approaches vertical and is undefined at exactly +/-90 degrees.
    Mask on this to drop frames where the animal is looking straight up or
    down; on the 2026-08-20 take only 0.9% of frames exceed 80 degrees, so it's
    a rare-event guard rather than a routine correction.
    """
    rotation_xyzw = np.asarray(rotation_xyzw, dtype=float)
    matrices = Rotation.from_quat(rotation_xyzw).as_matrix()
    v = matrices @ _forward_vector(head_frame)
    return np.degrees(np.arcsin(np.clip(v[:, 1], -1.0, 1.0)))
