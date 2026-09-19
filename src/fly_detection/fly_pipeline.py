"""Static fly detection via dark-blob filtering + temporal confirmation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FlyBox:
    """Axis-aligned fly bounding box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h


class StaticFlyDetector:
    """
    Detect stationary dark fly-like blobs in a BGR frame or still image.

    Pipeline:
      1. Blur + grayscale
      2. Threshold dark pixels (flies are typically dark on lighter walls)
      3. Morphology clean-up + contour filter by area / aspect ratio
      4. Optional multi-frame confirmation so only stable (static) blobs pass
    """

    def __init__(
        self,
        min_area: int = 15,
        max_area: int = 900,
        dark_threshold: int = 70,
        blur_ksize: int = 5,
        min_aspect: float = 0.35,
        max_aspect: float = 2.8,
        confirm_frames: int = 3,
        match_distance: float = 28.0,
    ) -> None:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        self.min_area = min_area
        self.max_area = max_area
        self.dark_threshold = dark_threshold
        self.blur_ksize = blur_ksize
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.confirm_frames = max(1, confirm_frames)
        self.match_distance = match_distance
        self._history: Deque[List[FlyBox]] = deque(maxlen=self.confirm_frames)

    def reset(self) -> None:
        self._history.clear()

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def detect_raw(self, frame_bgr: np.ndarray) -> List[FlyBox]:
        """Single-frame dark-blob detections (no temporal filter)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        # Flies = dark; invert so they become bright blobs for contouring.
        _, mask = cv2.threshold(
            blurred, self.dark_threshold, 255, cv2.THRESH_BINARY_INV
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        flies: List[FlyBox] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            aspect = w / float(h)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            peri = cv2.arcLength(contour, True)
            circularity = 0.0 if peri == 0 else (4.0 * np.pi * area) / (peri * peri)
            # Prefer compact blobs; score in [0, 1].
            score = float(np.clip(0.35 * circularity + 0.65 * (1.0 - abs(1.0 - aspect)), 0, 1))
            flies.append(FlyBox(int(x), int(y), int(w), int(h), score=score))

        flies.sort(key=lambda f: f.score, reverse=True)
        return flies

    def detect(self, frame_bgr: np.ndarray) -> List[FlyBox]:
        """
        Detect static flies.

        Requires a blob to appear near the same location across
        ``confirm_frames`` consecutive calls (set to 1 to disable).
        """
        current = self.detect_raw(frame_bgr)
        if self.confirm_frames <= 1:
            return current

        self._history.append(current)
        if len(self._history) < self.confirm_frames:
            return []

        confirmed: List[FlyBox] = []
        for fly in current:
            if self._is_stable(fly):
                confirmed.append(fly)
        return confirmed

    def _is_stable(self, fly: FlyBox) -> bool:
        cx, cy = fly.center
        for past in list(self._history)[:-1]:
            if not any(
                np.hypot(cx - other.center[0], cy - other.center[1])
                <= self.match_distance
                for other in past
            ):
                return False
        return True

    def detect_images(
        self, paths: Sequence[str | Path]
    ) -> List[Tuple[str, List[FlyBox]]]:
        """Run detection on still images (temporal filter disabled)."""
        saved = self.confirm_frames
        self.confirm_frames = 1
        self.reset()
        results: List[Tuple[str, List[FlyBox]]] = []
        try:
            for path in paths:
                frame = self.load_image(path)
                results.append((str(path), self.detect(frame)))
        finally:
            self.confirm_frames = saved
            self.reset()
        return results


def draw_flies(
    frame_bgr: np.ndarray,
    flies: List[FlyBox],
    color: Tuple[int, int, int] = (40, 40, 255),
    selected_index: Optional[int] = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    for i, fly in enumerate(flies):
        is_selected = selected_index is not None and i == selected_index
        box_color = (0, 255, 255) if is_selected else color
        thickness = 2 if is_selected else 1
        cv2.rectangle(
            out,
            (fly.x, fly.y),
            (fly.x + fly.w, fly.y + fly.h),
            box_color,
            thickness,
        )
        cx, cy = fly.center
        cv2.circle(out, (cx, cy), 3, box_color, -1)
        label = f"fly {fly.score:.2f}"
        if is_selected:
            label = "TARGET " + label
        cv2.putText(
            out,
            label,
            (fly.x, max(0, fly.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        f"flies: {len(flies)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out
