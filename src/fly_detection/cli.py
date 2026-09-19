"""Runnable vision commands: preview, faces, and face-triggered motor move."""

from __future__ import annotations

import argparse
import time

from .face_pipeline import FaceDetector, draw_faces

# 28BYJ-48 @ 2-phase full step: 2048 steps = 360°, so 1024 = 180°.
STEPS_180_DEG = 1024


def cmd_preview(args: argparse.Namespace) -> int:
    """Show the raw OAK-1 camera feed."""
    from .oak_camera import OakCamera, destroy_windows, show_frame

    print("Starting OAK-1 preview. Press 'q' or Esc to quit.")
    with OakCamera(width=args.width, height=args.height, fps=args.fps) as cam:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            if not show_frame("OAK-1 preview", frame):
                break
    destroy_windows()
    return 0


def cmd_faces(args: argparse.Namespace) -> int:
    """Run face detection and overlay boxes on the live view."""
    from .oak_camera import OakCamera, destroy_windows, show_frame

    detector = FaceDetector()
    print("Starting face detection. Press 'q' or Esc to quit.")
    with OakCamera(width=args.width, height=args.height, fps=args.fps) as cam:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            faces = detector.detect(frame)
            overlay = draw_faces(frame, faces)
            if not show_frame("Face detection", overlay):
                break
    destroy_windows()
    return 0


def cmd_aim(args: argparse.Namespace) -> int:
    """
    Detect faces; when one appears, rotate the stepper 180° once per trigger.

    Debounced so a continuous face does not keep spinning the motor.
    """
    from .oak_camera import OakCamera, destroy_windows, show_frame
    from turret_control import MaxSpeed5VStepper

    detector = FaceDetector()
    motor = MaxSpeed5VStepper(
        in1=args.in1,
        in2=args.in2,
        in3=args.in3,
        in4=args.in4,
    )
    cooldown_s = args.cooldown
    last_fire = 0.0
    face_was_present = False

    print(
        "Aim mode: face detection + 180° motor move. "
        "Press 'q' or Esc to quit."
    )
    try:
        with OakCamera(width=args.width, height=args.height, fps=args.fps) as cam:
            while True:
                ok, frame = cam.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                faces = detector.detect(frame)
                overlay = draw_faces(frame, faces)
                now = time.monotonic()
                face_present = len(faces) > 0

                # Rising edge + cooldown: fire once when a face newly appears.
                if (
                    face_present
                    and not face_was_present
                    and (now - last_fire) >= cooldown_s
                ):
                    _put_status(overlay, "MOVING 180 deg")
                    if not show_frame("Aim (face + motor)", overlay):
                        break
                    print(f"Face detected ({len(faces)}) — moving motor 180°...")
                    motor.move_max_5v(steps=STEPS_180_DEG, clockwise=args.clockwise)
                    last_fire = time.monotonic()
                    print("Motor move complete.")
                else:
                    status = "tracking" if face_present else "searching"
                    _put_status(overlay, status)
                    if not show_frame("Aim (face + motor)", overlay):
                        break

                face_was_present = face_present
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        motor.close()
        destroy_windows()
    return 0


def _put_status(frame, text: str) -> None:
    import cv2

    cv2.putText(
        frame,
        text,
        (10, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fly-detect",
        description="Fly swatter vision pipeline (OAK-1 on Raspberry Pi 5)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_camera_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--width", type=int, default=640, help="Preview width")
        p.add_argument("--height", type=int, default=480, help="Preview height")
        p.add_argument("--fps", type=int, default=30, help="Camera FPS")

    p_preview = sub.add_parser("preview", help="Show the raw OAK-1 camera feed")
    add_camera_args(p_preview)
    p_preview.set_defaults(func=cmd_preview)

    p_faces = sub.add_parser(
        "faces",
        help="Run face detection and draw boxes on the live view",
    )
    add_camera_args(p_faces)
    p_faces.set_defaults(func=cmd_faces)

    p_aim = sub.add_parser(
        "aim",
        help="On face detection, rotate the stepper motor 180 degrees",
    )
    add_camera_args(p_aim)
    p_aim.add_argument("--in1", type=int, default=17, help="ULN2003 IN1 GPIO")
    p_aim.add_argument("--in2", type=int, default=27, help="ULN2003 IN2 GPIO")
    p_aim.add_argument("--in3", type=int, default=22, help="ULN2003 IN3 GPIO")
    p_aim.add_argument("--in4", type=int, default=23, help="ULN2003 IN4 GPIO")
    p_aim.add_argument(
        "--cooldown",
        type=float,
        default=3.0,
        help="Seconds to wait before another motor move after a trigger",
    )
    p_aim.add_argument(
        "--clockwise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Motor direction for the 180° move (default: clockwise)",
    )
    p_aim.set_defaults(func=cmd_aim)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
