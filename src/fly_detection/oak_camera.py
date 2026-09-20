"""OAK-1 (DepthAI) color camera capture for the Raspberry Pi 5."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import cv2
import depthai as dai
import numpy as np


class OakCamera:
    """
    Streaming RGB frames from an OAK-1 via a DepthAI pipeline.

    ``mount_rotate_ccw_deg`` describes how the camera body is rotated on the
    turret relative to upright. Frames are rotated the opposite way so
    detection / display see a level image.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        interleaved: bool = False,
        mount_rotate_ccw_deg: int = 90,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.interleaved = interleaved
        self.mount_rotate_ccw_deg = ((mount_rotate_ccw_deg % 360) + 360) % 360
        # Request capture size so that after correcting the mount rotation
        # the frame is ``width`` x ``height``.
        if self.mount_rotate_ccw_deg in (90, 270):
            self._capture_w = height
            self._capture_h = width
        else:
            self._capture_w = width
            self._capture_h = height
        self._device: Optional[dai.Device] = None
        self._queue: Optional[dai.DataOutputQueue] = None

    def _build_pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setVideoSize(self._capture_w, self._capture_h)
        cam.setPreviewSize(self._capture_w, self._capture_h)
        cam.setInterleaved(self.interleaved)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(self.fps)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("preview")
        cam.preview.link(xout.input)

        return pipeline

    def start(self) -> None:
        if self._device is not None:
            return
        pipeline = self._build_pipeline()
        self._device = dai.Device(pipeline)
        self._queue = self._device.getOutputQueue(
            name="preview", maxSize=4, blocking=False
        )

    def stop(self) -> None:
        if self._device is not None:
            self._device.close()
        self._device = None
        self._queue = None

    def _correct_orientation(self, frame: np.ndarray) -> np.ndarray:
        """Undo the physical mount rotation so the image is upright."""
        if self.mount_rotate_ccw_deg == 0:
            return frame
        if self.mount_rotate_ccw_deg == 90:
            # Mounted 90° CCW → rotate frame 90° CW to upright.
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.mount_rotate_ccw_deg == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.mount_rotate_ccw_deg == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab the latest upright preview frame. Returns (ok, frame_bgr)."""
        if self._queue is None:
            raise RuntimeError("OakCamera is not started. Call start() first.")

        packet = self._queue.tryGet()
        if packet is None:
            return False, None

        frame = self._correct_orientation(packet.getCvFrame())
        return True, frame

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the camera is stopped."""
        self.start()
        try:
            while self._device is not None:
                ok, frame = self.read()
                if ok and frame is not None:
                    yield frame
        finally:
            self.stop()

    def __enter__(self) -> "OakCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def show_frame(window: str, frame: np.ndarray) -> bool:
    """
    Display a frame. Returns False if the user pressed 'q' or Escape.
    """
    cv2.imshow(window, frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def destroy_windows() -> None:
    cv2.destroyAllWindows()
