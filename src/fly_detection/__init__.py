"""Vision pipeline: OAK-1 camera, moving faces, static flies, and aim target."""

from .face_pipeline import FaceBox, FaceDetector, draw_faces
from .fly_pipeline import FlyBox, StaticFlyDetector, draw_flies
from .target import AimTarget, draw_target

__all__ = [
    "OakCamera",
    "AimTarget",
    "draw_target",
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
