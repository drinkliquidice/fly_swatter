"""Printed-fly detection: dark blob on a large white paper background."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PaperBox:
    """White paper region behind a printed fly."""

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass(frozen=True)
class FlyBox:
    """Axis-aligned fly bounding box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    paper: Optional[PaperBox] = None

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
    Detect printed flies on white paper.

    A valid target requires:
      - a white paper region at least ``min_paper_w`` x ``min_paper_h`` (default 300x300)
      - a dark fly blob on that paper at least ``min_fly_w`` x ``min_fly_h`` (default 20x20)
    """

    def __init__(
        self,
        min_fly_w: int = 20,
        min_fly_h: int = 20,
        max_fly_w: int = 120,
        max_fly_h: int = 120,
        min_paper_w: int = 300,
        min_paper_h: int = 300,
        white_threshold: int = 200,
        dark_threshold: int = 110,
        blur_ksize: int = 5,
        min_aspect: float = 0.35,
        max_aspect: float = 2.8,
        confirm_frames: int = 3,
        match_distance: float = 28.0,
        # Back-compat aliases used by older CLI flags:
        min_area: int | None = None,
        max_area: int | None = None,
    ) -> None:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        self.min_fly_w = min_fly_w
        self.min_fly_h = min_fly_h
        self.max_fly_w = max_fly_w
        self.max_fly_h = max_fly_h
        self.min_paper_w = min_paper_w
        self.min_paper_h = min_paper_h
        self.white_threshold = white_threshold
        self.dark_threshold = dark_threshold
        self.blur_ksize = blur_ksize
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.confirm_frames = max(1, confirm_frames)
        self.match_distance = match_distance
        # Optional area clamps if callers still pass them.
        self.min_area = min_area if min_area is not None else min_fly_w * min_fly_h
        self.max_area = max_area if max_area is not None else max_fly_w * max_fly_h
        self._history: Deque[List[FlyBox]] = deque(maxlen=self.confirm_frames)
        self.last_papers: List[PaperBox] = []

    def reset(self) -> None:
        self._history.clear()
        self.last_papers = []

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def _find_papers(self, gray: np.ndarray) -> List[PaperBox]:
        # Bright regions = paper. Slight blur reduces print-edge noise.
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        _, white = cv2.threshold(
            blurred, self.white_threshold, 255, cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=2)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        papers: List[PaperBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_paper_w or h < self.min_paper_h:
                continue
            # Paper should be reasonably rectangular / large filled area.
            area = cv2.contourArea(contour)
            if area < 0.45 * w * h:
                continue
            papers.append(PaperBox(int(x), int(y), int(w), int(h)))
        papers.sort(key=lambda p: p.w * p.h, reverse=True)
        return papers

    def _flies_on_paper(
        self, gray: np.ndarray, paper: PaperBox
    ) -> List[FlyBox]:
        x0, y0, pw, ph = paper.x, paper.y, paper.w, paper.h
        roi = gray[y0 : y0 + ph, x0 : x0 + pw]
        if roi.size == 0:
            return []

        blurred = cv2.GaussianBlur(roi, (self.blur_ksize, self.blur_ksize), 0)
        # Dark print on white paper.
        _, dark = cv2.threshold(
            blurred, self.dark_threshold, 255, cv2.THRESH_BINARY_INV
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        flies: List[FlyBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_fly_w or h < self.min_fly_h:
                continue
            if w > self.max_fly_w or h > self.max_fly_h:
                continue
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            # Reject blobs that are basically the whole paper (shadows / frames).
            if w * h > 0.25 * pw * ph:
                continue
            aspect = w / float(h) if h else 0.0
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            peri = cv2.arcLength(contour, True)
            circularity = (
                0.0 if peri == 0 else (4.0 * np.pi * area) / (peri * peri)
            )
            score = float(
                np.clip(
                    0.35 * circularity + 0.65 * (1.0 - abs(1.0 - aspect)),
                    0,
                    1,
                )
            )
            flies.append(
                FlyBox(
                    x=int(x0 + x),
                    y=int(y0 + y),
                    w=int(w),
                    h=int(h),
                    score=score,
                    paper=paper,
                )
            )
        return flies

    def detect_raw(self, frame_bgr: np.ndarray) -> List[FlyBox]:
        """Single-frame printed-fly detections (no temporal filter)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        papers = self._find_papers(gray)
        self.last_papers = papers

        flies: List[FlyBox] = []
        for paper in papers:
            flies.extend(self._flies_on_paper(gray, paper))

        flies.sort(key=lambda f: f.score, reverse=True)
        return flies

    def detect(self, frame_bgr: np.ndarray) -> List[FlyBox]:
        """
        Detect printed flies on white paper.

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
    papers: Optional[List[PaperBox]] = None,
) -> np.ndarray:
    out = frame_bgr.copy()

    paper_list = papers
    if paper_list is None:
        seen = []
        for fly in flies:
            if fly.paper is not None:
                key = fly.paper.as_xyxy
                if key not in seen:
                    seen.append(key)
                    if paper_list is None:
                        paper_list = []
                    paper_list.append(fly.paper)

    if paper_list:
        for paper in paper_list:
            cv2.rectangle(
                out,
                (paper.x, paper.y),
                (paper.x + paper.w, paper.y + paper.h),
                (220, 220, 220),
                1,
            )
            cv2.putText(
                out,
                f"paper {paper.w}x{paper.h}",
                (paper.x, max(0, paper.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

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
        label = f"fly {fly.w}x{fly.h}"
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
