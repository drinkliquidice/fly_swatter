"""OAK-1 (DepthAI) color camera capture for the Raspberry Pi 5."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import cv2
import depthai as dai
import numpy as np


class OakCamera:
    """Streaming RGB frames from an OAK-1 via a DepthAI pipeline."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        interleaved: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.interleaved = interleaved
        self._device: Optional[dai.Device] = None
        self._queue: Optional[dai.DataOutputQueue] = None

    def _build_pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setVideoSize(self.width, self.height)
        cam.setPreviewSize(self.width, self.height)
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

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab the latest preview frame. Returns (ok, frame_bgr)."""
        if self._queue is None:
            raise RuntimeError("OakCamera is not started. Call start() first.")

        packet = self._queue.tryGet()
        if packet is None:
            return False, None

        frame = packet.getCvFrame()
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
