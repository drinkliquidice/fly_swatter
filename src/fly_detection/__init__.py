"""Vision pipeline: OAK-1 camera, moving faces, and static fly detection."""

from .face_pipeline import FaceBox, FaceDetector, draw_faces
from .fly_pipeline import FlyBox, StaticFlyDetector, draw_flies

__all__ = [
    "OakCamera",
    "FaceBox",
    "FaceDetector",
    "draw_faces",
    "FlyBox",
    "StaticFlyDetector",
    "draw_flies",
]


def __getattr__(name: str):
    if name == "OakCamera":
        from .oak_camera import OakCamera

        return OakCamera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
