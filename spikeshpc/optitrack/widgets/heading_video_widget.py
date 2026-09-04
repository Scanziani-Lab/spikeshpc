"""Frame-by-frame video viewer with a computed-heading compass overlay.

This checks :func:`optitrack.heading.compute_heading` against the animal.
The heading itself is calibrated from the data by
:func:`optitrack.heading.calibrate_head_frame`, but how the overhead camera is
mounted relative to the arena axes isn't in the CSV, so the arrow may still
need rotating (``compass_rotation_offset_deg``) or mirroring
(``flip_direction`` on ``compute_heading``) before it lines up on screen. Step
through frames here and adjust those two until it does.
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..io import OptitrackTake
from ._backend import warn_if_noninteractive_backend


class HeadingVideoWidget:
    """Steps through video frames with Left/Right, drawing a compass arrow.

    Assumes video frame ``i`` corresponds to OptiTrack CSV frame
    ``i + frame_offset`` -- Motive's raw per-camera export is normally 1:1
    with capture frames, but that isn't independently verifiable from the
    files alone, which is exactly what stepping through this widget checks.

    Requires an interactive matplotlib backend (``%matplotlib widget`` or
    ``%matplotlib qt`` in Jupyter) and the figure to have keyboard focus
    (click on it once) before arrow keys will do anything.
    """

    def __init__(
        self,
        video_path,
        heading_deg: np.ndarray,
        take: OptitrackTake | None = None,
        frame_offset: int = 0,
        start_frame: int = 0,
        compass_rotation_offset_deg: float = 0.0,
    ):
        warn_if_noninteractive_backend()
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        self.n_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if take is not None and self.n_video_frames != len(take.frame_numbers):
            print(
                f"Warning: video has {self.n_video_frames} frames but the "
                f"OptiTrack CSV has {len(take.frame_numbers)} -- check `frame_offset`."
            )

        self.heading_deg = heading_deg
        self.frame_offset = frame_offset
        self.compass_rotation_offset_deg = compass_rotation_offset_deg
        self.current_frame = -1

        self.fig, self.ax = plt.subplots()
        self.ax.axis("off")
        self.im = None

        # Fixed compass in the upper-left corner, in axes-fraction coordinates
        # so it stays put regardless of image content; only the arrow moves.
        self._center = (0.12, 0.85)
        self._radius = 0.08
        self.ax.add_patch(
            plt.Circle(
                self._center,
                self._radius,
                transform=self.ax.transAxes,
                facecolor="none",
                edgecolor="white",
                linewidth=1.5,
                zorder=10,
            )
        )
        (self.arrow_line,) = self.ax.plot(
            [], [], color="red", linewidth=2, transform=self.ax.transAxes, zorder=11
        )
        self.fig.text(0.99, 0.01, "← → to step frames", ha="right", fontsize=8, color="gray")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._show_frame(start_frame)

    def _heading_for_video_frame(self, frame_idx: int) -> float | None:
        optitrack_frame = frame_idx + self.frame_offset
        if 0 <= optitrack_frame < len(self.heading_deg):
            return float(self.heading_deg[optitrack_frame])
        return None

    def _show_frame(self, frame_idx: int):
        frame_idx = int(np.clip(frame_idx, 0, self.n_video_frames - 1))
        if frame_idx != self.current_frame + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_frame = frame_idx

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.im is None:
            self.im = self.ax.imshow(frame_rgb)
        else:
            self.im.set_data(frame_rgb)

        heading = self._heading_for_video_frame(frame_idx)
        if heading is not None:
            theta = np.radians(heading + self.compass_rotation_offset_deg)
            cx, cy = self._center
            tip = (cx + self._radius * np.sin(theta), cy + self._radius * np.cos(theta))
            self.arrow_line.set_data([cx, tip[0]], [cy, tip[1]])
        else:
            self.arrow_line.set_data([], [])

        self.ax.set_title(f"video frame {frame_idx}/{self.n_video_frames - 1}")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "right":
            self._show_frame(self.current_frame + 1)
        elif event.key == "left":
            self._show_frame(self.current_frame - 1)

    def close(self):
        self.cap.release()


def show_heading_video_widget(
    video_path,
    heading_deg: np.ndarray,
    take: OptitrackTake | None = None,
    frame_offset: int = 0,
    start_frame: int = 0,
    compass_rotation_offset_deg: float = 0.0,
) -> HeadingVideoWidget:
    """Open an interactive figure; Left/Right arrow keys step ±1 video frame.

    Draws a compass arrow (upper-left) for that frame's ``heading_deg``,
    e.g. from :func:`optitrack.heading.compute_heading`.
    """
    return HeadingVideoWidget(
        video_path,
        heading_deg,
        take=take,
        frame_offset=frame_offset,
        start_frame=start_frame,
        compass_rotation_offset_deg=compass_rotation_offset_deg,
    )
