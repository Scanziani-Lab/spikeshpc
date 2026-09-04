"""optitrack — sync OptiTrack motion capture with electrophysiology.

Workflow, in order:

1. Extract camera shutter-closure timestamps from an ephys TTL channel and
   sync them to the OptiTrack take (:mod:`optitrack.sync`).
2. Load the OptiTrack Motive CSV export (:mod:`optitrack.io`).
3. Calibrate the head frame and compute heading (:mod:`optitrack.heading`)
   and/or kinematics (:mod:`optitrack.kinematics`) from a rigid body's track.
4. Compute per-unit head-direction tuning curves (:mod:`optitrack.tuning`).
5. Inspect interactively (:mod:`optitrack.widgets`).
"""

from .heading import (
    IDENTITY_HEAD_FRAME,
    HeadFrame,
    calibrate_head_frame,
    compute_head_elevation,
    compute_heading,
)
from .io import (
    OptitrackTake,
    PositionTrack,
    RigidBodyTrack,
    load_optitrack_csv,
    read_rigid_body_track,
    rigid_body_names,
)
from .kinematics import compute_kinematics
from .sync import (
    align_frames_to_shutter_events,
    cross_check_with_optitrack_csv,
    describe_analog_channels,
    extract_shutter_close_times,
    rail_fraction,
    plot_shutter_close_sanity_check,
    save_shutter_close_times,
)
from .tuning import (
    HDTuningStats,
    compute_all_units_tuning_curves,
    compute_frame_firing_rates,
    compute_hd_tuning_curve,
    compute_hd_tuning_significance,
    compute_mean_vector_length,
    get_unit_depths,
)

__version__ = "0.1.0"

__all__ = [
    "IDENTITY_HEAD_FRAME",
    "HDTuningStats",
    "HeadFrame",
    "OptitrackTake",
    "PositionTrack",
    "RigidBodyTrack",
    "align_frames_to_shutter_events",
    "calibrate_head_frame",
    "compute_all_units_tuning_curves",
    "compute_frame_firing_rates",
    "compute_hd_tuning_curve",
    "compute_hd_tuning_significance",
    "compute_head_elevation",
    "compute_heading",
    "compute_kinematics",
    "compute_mean_vector_length",
    "cross_check_with_optitrack_csv",
    "describe_analog_channels",
    "extract_shutter_close_times",
    "rail_fraction",
    "get_unit_depths",
    "load_optitrack_csv",
    "read_rigid_body_track",
    "rigid_body_names",
    "plot_shutter_close_sanity_check",
    "save_shutter_close_times",
]
