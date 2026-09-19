"""Turret aim point in the camera frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class AimTarget:
    """
    Where in the image the turret barrel is pointed.

    Coordinates are pixels in the camera preview. Defaults to frame center
    when ``x`` / ``y`` are left as None and resolved against a frame.
    """

    x: Optional[float] = None
    y: Optional[float] = None
    # Normalized fallbacks used when pixel x/y are unset (0..1, origin top-left).
    norm_x: float = 0.5
    norm_y: float = 0.5
    # Pixel radius counted as "on target" / lined up with the barrel.
    radius: int = 16

    def point(self, frame_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """Resolve the aim point in pixel coordinates for a frame (h, w, ...)."""
        h, w = frame_shape[:2]
        px = self.x if self.x is not None else self.norm_x * (w - 1)
        py = self.y if self.y is not None else self.norm_y * (h - 1)
        return int(round(px)), int(round(py))

    def error_to(
        self, target_xy: Tuple[int, int], frame_shape: Tuple[int, ...]
    ) -> Tuple[int, int, float]:
        """
        Pixel error from the aim point to ``target_xy``.

        Returns ``(dx, dy, distance)`` where positive ``dx`` means the
        detection is to the right of the aim point, positive ``dy`` below.
        """
        ax, ay = self.point(frame_shape)
        dx = int(target_xy[0] - ax)
        dy = int(target_xy[1] - ay)
        return dx, dy, float(np.hypot(dx, dy))

    def is_on_target(
        self, target_xy: Tuple[int, int], frame_shape: Tuple[int, ...]
    ) -> bool:
        _, _, dist = self.error_to(target_xy, frame_shape)
        return dist <= self.radius

    def nearest(
        self,
        centers: Sequence[Tuple[int, int]],
        frame_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, Tuple[int, int], float]]:
        """
        Pick the detection center closest to the aim point.

        Returns ``(index, (x, y), distance)`` or None if ``centers`` is empty.
        """
        if not centers:
            return None
        best_i = 0
        best_xy = centers[0]
        best_dist = self.error_to(best_xy, frame_shape)[2]
        for i, xy in enumerate(centers[1:], start=1):
            dist = self.error_to(xy, frame_shape)[2]
            if dist < best_dist:
                best_i, best_xy, best_dist = i, xy, dist
        return best_i, best_xy, best_dist


def draw_target(
    frame_bgr: np.ndarray,
    aim: AimTarget,
    selected_xy: Optional[Tuple[int, int]] = None,
    color: Tuple[int, int, int] = (0, 220, 255),
    selected_color: Tuple[int, int, int] = (0, 140, 255),
) -> np.ndarray:
    """
    Draw the turret aim crosshair (and optional line to a selected detection).
    """
    out = frame_bgr
    ax, ay = aim.point(out.shape)
    arm = max(12, aim.radius + 6)

    cv2.circle(out, (ax, ay), aim.radius, color, 1, cv2.LINE_AA)
    cv2.line(out, (ax - arm, ay), (ax + arm, ay), color, 1, cv2.LINE_AA)
    cv2.line(out, (ax, ay - arm), (ax, ay + arm), color, 1, cv2.LINE_AA)
    cv2.circle(out, (ax, ay), 2, color, -1, cv2.LINE_AA)
    cv2.putText(
        out,
        "aim",
        (ax + aim.radius + 4, ay - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )

    if selected_xy is not None:
        sx, sy = selected_xy
        dx, dy, dist = aim.error_to(selected_xy, out.shape)
        on_target = dist <= aim.radius
        line_color = (0, 255, 0) if on_target else selected_color
        cv2.line(out, (ax, ay), (sx, sy), line_color, 1, cv2.LINE_AA)
        cv2.circle(out, (sx, sy), 5, line_color, 1, cv2.LINE_AA)
        label = (
            "ON TARGET"
            if on_target
            else f"err dx={dx} dy={dy} d={dist:.0f}px"
        )
        cv2.putText(
            out,
            label,
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            line_color,
            2,
            cv2.LINE_AA,
        )

    return out


def aim_target_from_args(
    target_x: Optional[float],
    target_y: Optional[float],
    target_norm_x: float,
    target_norm_y: float,
    target_radius: int,
) -> AimTarget:
    """Build an AimTarget from CLI-style arguments."""
    return AimTarget(
        x=target_x,
        y=target_y,
        norm_x=target_norm_x,
        norm_y=target_norm_y,
        radius=target_radius,
    )
