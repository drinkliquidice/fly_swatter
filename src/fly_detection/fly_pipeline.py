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
    Detect printed flies on white letter/A4 paper held in portrait.

    A valid target prefers:
      - a bright portrait paper region (taller than wide, ~letter/A4 aspect)
      - a darker printed fly on that paper at least ``min_fly_w`` x ``min_fly_h``
      - the fly centered on the paper (middle 50% of the paper box)
    """

    def __init__(
        self,
        min_fly_w: int = 10,
        min_fly_h: int = 10,
        max_fly_w: int = 280,
        max_fly_h: int = 280,
        min_paper_w: int = 160,
        min_paper_h: int = 220,
        white_threshold: int = 145,
        dark_threshold: int = 175,
        blur_ksize: int = 3,
        min_aspect: float = 0.15,
        max_aspect: float = 6.0,
        # Portrait letter/A4: height/width ≈ 1.29 (US Letter) to 1.41 (A4).
        require_portrait: bool = True,
        min_paper_aspect: float = 1.10,
        max_paper_aspect: float = 1.85,
        # Fly must be within this fraction of paper half-size from center (0.35 ≈ middle 70%).
        center_frac: float = 0.35,
        # Pixel must be at least this much darker than paper median.
        min_contrast: float = 4.0,
        confirm_frames: int = 1,
        match_distance: float = 48.0,
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
        self.require_portrait = require_portrait
        self.min_paper_aspect = min_paper_aspect
        self.max_paper_aspect = max_paper_aspect
        self.center_frac = center_frac
        self.min_contrast = min_contrast
        self.confirm_frames = max(1, confirm_frames)
        self.match_distance = match_distance
        self.min_area = min_area if min_area is not None else max(20, (min_fly_w * min_fly_h) // 3)
        self.max_area = max_area if max_area is not None else max_fly_w * max_fly_h
        self._history: Deque[List[FlyBox]] = deque(maxlen=max(self.confirm_frames, 1))
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

    def _paper_candidates_from_mask(self, mask: np.ndarray) -> List[PaperBox]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        papers: List[PaperBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_paper_w or h < self.min_paper_h:
                continue
            # Letter / A4 held portrait: taller than wide.
            paper_aspect = h / float(w) if w else 0.0
            if self.require_portrait:
                if h <= w:
                    continue
                if paper_aspect < self.min_paper_aspect or paper_aspect > self.max_paper_aspect:
                    continue
            area = cv2.contourArea(contour)
            # Lenient fill: wrinkled / angled paper still counts.
            if area < 0.15 * w * h:
                continue
            papers.append(PaperBox(int(x), int(y), int(w), int(h)))
        papers.sort(key=lambda p: p.w * p.h, reverse=True)
        return papers

    def _find_papers(self, gray: np.ndarray) -> List[PaperBox]:
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        # Pass 1: fixed bright threshold (works in even lighting).
        _, white = cv2.threshold(
            blurred, self.white_threshold, 255, cv2.THRESH_BINARY
        )
        papers = self._paper_candidates_from_mask(white)
        if papers:
            return papers

        # Pass 2: adaptive threshold for uneven / dim lighting.
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            -5,
        )
        papers = self._paper_candidates_from_mask(adaptive)
        if papers:
            return papers

        # Pass 3: take the brightest large connected region even if under size.
        # Helps when paper is partially out of frame but still mostly visible.
        _, soft = cv2.threshold(
            blurred, max(120, self.white_threshold - 40), 255, cv2.THRESH_BINARY
        )
        soft_papers = self._paper_candidates_from_mask(soft)
        if soft_papers:
            return soft_papers

        # Last resort: whole-frame "paper" only if the frame itself is portrait-ish.
        if float(np.mean(blurred)) >= self.white_threshold - 25:
            h, w = gray.shape[:2]
            if w >= self.min_paper_w // 2 and h >= self.min_paper_h // 2:
                if (not self.require_portrait) or (h > w):
                    return [PaperBox(0, 0, w, h)]
        return []

    def _flies_on_paper(
        self, gray: np.ndarray, paper: PaperBox
    ) -> List[FlyBox]:
        x0, y0, pw, ph = paper.x, paper.y, paper.w, paper.h
        roi = gray[y0 : y0 + ph, x0 : x0 + pw]
        if roi.size == 0:
            return []

        blurred = cv2.GaussianBlur(roi, (self.blur_ksize, self.blur_ksize), 0)
        paper_med = float(np.median(blurred))
        paper_mean = float(np.mean(blurred))

        # Relative-to-paper masks only. An absolute threshold above the paper
        # brightness would mark the entire sheet as "dark".
        rel_thr = paper_med - self.min_contrast
        soft_thr = paper_med - max(3.0, self.min_contrast * 0.5)
        dyn_abs = min(float(self.dark_threshold), rel_thr)
        _, fixed = cv2.threshold(
            blurred, dyn_abs, 255, cv2.THRESH_BINARY_INV
        )
        relative = np.where(blurred < rel_thr, 255, 0).astype(np.uint8)
        soft = np.where(blurred < soft_thr, 255, 0).astype(np.uint8)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            6,
        )
        # Prefer relative evidence; adaptive only where paper is also darker.
        dark = cv2.bitwise_or(fixed, relative)
        dark = cv2.bitwise_or(dark, soft)
        dark = cv2.bitwise_or(dark, cv2.bitwise_and(adaptive, soft))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)

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
            if w * h > 0.45 * pw * ph:
                continue
            aspect = w / float(h) if h else 0.0
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            fly_cx = x + w / 2.0
            fly_cy = y + h / 2.0
            if abs(fly_cx - pw / 2.0) > self.center_frac * pw:
                continue
            if abs(fly_cy - ph / 2.0) > self.center_frac * ph:
                continue

            patch = blurred[y : y + h, x : x + w]
            if patch.size == 0:
                continue
            local_mean = float(np.mean(patch))
            if local_mean > paper_mean - max(2.0, self.min_contrast * 0.35):
                continue

            peri = cv2.arcLength(contour, True)
            circularity = (
                0.0 if peri == 0 else (4.0 * np.pi * area) / (peri * peri)
            )
            contrast = max(0.0, (paper_mean - local_mean) / 50.0)
            score = float(
                np.clip(
                    0.20 * circularity
                    + 0.30 * (1.0 - abs(1.0 - aspect))
                    + 0.50 * contrast,
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
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out
