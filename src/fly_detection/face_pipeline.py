"""Moving-face detection on OAK preview frames."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Official YuNet weights from opencv_zoo (used on OpenCV 5+).
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_NAME = "face_detection_yunet_2023mar.onnx"
_MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class FaceBox:
    """Axis-aligned face bounding box in pixel coordinates."""

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


def _ensure_yunet_model(model_path: Path | None = None) -> Path:
    path = model_path or (_MODELS_DIR / _YUNET_NAME)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet face model to {path} ...")
    urllib.request.urlretrieve(_YUNET_URL, path)
    return path


class FaceDetector:
    """
    Detect faces in BGR frames.

    Uses OpenCV Haar cascades when available (OpenCV 4.x). On OpenCV 5+,
    falls back to YuNet (``FaceDetectorYN``), downloading weights on first use.
    """

    def __init__(
        self,
        cascade_path: str | None = None,
        yunet_path: str | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (60, 60),
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> None:
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._backend: str
        self._cascade = None
        self._yunet = None
        self._yunet_input_size: Tuple[int, int] | None = None

        if hasattr(cv2, "CascadeClassifier"):
            path = cascade_path or (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._cascade = cv2.CascadeClassifier(path)
            if self._cascade.empty():
                raise RuntimeError(f"Failed to load face cascade: {path}")
            self._backend = "haar"
            return

        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise RuntimeError(
                "No face detector backend found. Install opencv-python 4.x "
                "(Haar) or 5.x (YuNet)."
            )

        model = _ensure_yunet_model(Path(yunet_path) if yunet_path else None)
        # Input size is updated per-frame in detect().
        self._yunet = cv2.FaceDetectorYN_create(
            str(model),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            5000,
        )
        self._backend = "yunet"

    @property
    def backend(self) -> str:
        return self._backend

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        if self._backend == "haar":
            return self._detect_haar(frame_bgr)
        return self._detect_yunet(frame_bgr)

    def _detect_haar(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]

    def _detect_yunet(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        h, w = frame_bgr.shape[:2]
        size = (w, h)
        if self._yunet_input_size != size:
            self._yunet.setInputSize(size)
            self._yunet_input_size = size

        _, faces = self._yunet.detect(frame_bgr)
        if faces is None:
            return []

        boxes: List[FaceBox] = []
        for row in faces:
            x, y, bw, bh = row[:4]
            boxes.append(FaceBox(int(x), int(y), int(bw), int(bh)))
        return boxes


def draw_faces(
    frame_bgr: np.ndarray,
    faces: List[FaceBox],
    color: Tuple[int, int, int] = (0, 220, 80),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with face boxes and centers drawn."""
    out = frame_bgr.copy()
    for face in faces:
        x1, y1, x2, y2 = face.as_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cx, cy = face.center
        cv2.circle(out, (cx, cy), 4, color, -1)
        cv2.putText(
            out,
            "face",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        f"faces: {len(faces)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out
