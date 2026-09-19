"""Static fly detection pipeline (placeholder for upcoming models)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


class StaticFlyDetector:
    """
    Detect static flies in a frame or still image.

    Not wired to a trained model yet — returns no detections until a model
    path is provided. Use ``load_image`` / ``detect`` as the future API.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self._ready = False
        if model_path:
            # Hook for loading ONNX / blob / OpenCV DNN weights later.
            raise NotImplementedError(
                "Static fly model loading is not implemented yet. "
                "Pass model_path=None and use still-image helpers for now."
            )

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def detect(self, frame_bgr: np.ndarray) -> List[FlyBox]:
        """Run static fly detection. Empty until a model is hooked up."""
        _ = frame_bgr
        if not self._ready:
            return []
        return []

    def detect_images(self, paths: Sequence[str | Path]) -> List[Tuple[str, List[FlyBox]]]:
        results: List[Tuple[str, List[FlyBox]]] = []
        for path in paths:
            frame = self.load_image(path)
            results.append((str(path), self.detect(frame)))
        return results


def draw_flies(
    frame_bgr: np.ndarray,
    flies: List[FlyBox],
    color: Tuple[int, int, int] = (40, 40, 255),
) -> np.ndarray:
    out = frame_bgr.copy()
    for fly in flies:
        cv2.rectangle(
            out,
            (fly.x, fly.y),
            (fly.x + fly.w, fly.y + fly.h),
            color,
            2,
        )
        cv2.putText(
            out,
            f"fly {fly.score:.2f}",
            (fly.x, max(0, fly.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out
