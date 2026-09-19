"""Aim CLI: track faces or flies, show aim crosshair, drive four motors 180°."""

from __future__ import annotations

import argparse
import time
from typing import Callable, List, Optional, Tuple

from .face_pipeline import FaceDetector, draw_faces
from .fly_pipeline import StaticFlyDetector, draw_flies
from .target import AimTarget, aim_target_from_args, draw_target


def _aim_from_namespace(args: argparse.Namespace) -> AimTarget:
    return aim_target_from_args(
        target_x=args.target_x,
        target_y=args.target_y,
        target_norm_x=args.target_norm_x,
        target_norm_y=args.target_norm_y,
        target_radius=args.target_radius,
    )


def _selected_center(
    aim: AimTarget,
    centers: List[Tuple[int, int]],
    frame_shape,
) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    nearest = aim.nearest(centers, frame_shape)
    if nearest is None:
        return None, None
    index, xy, _ = nearest
    return index, xy


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


def _build_turret(args: argparse.Namespace):
    from turret_control import (
        DEFAULT_MOTOR_PINS,
        FourMotorTurret,
        parse_motor_pins,
    )

    pins = parse_motor_pins(args.motor_pins) if args.motor_pins else DEFAULT_MOTOR_PINS
    return FourMotorTurret(motor_pins=pins)


def _run_aim_loop(
    args: argparse.Namespace,
    *,
    mode_name: str,
    window_title: str,
    detect_fn: Callable,
    draw_fn: Callable,
) -> int:
    """
    Shared aim loop: detect → crosshair on nearest target → move all motors 180°.
    """
    from .oak_camera import OakCamera, destroy_windows, show_frame
    from turret_control import STEPS_180_DEG

    aim = _aim_from_namespace(args)
    turret = _build_turret(args)
    cooldown_s = args.cooldown
    last_fire = 0.0
    was_present = False

    print(
        f"Aim ({mode_name}): crosshair + 180° on all 4 motors when detected. "
        "Press 'q' or Esc to quit."
    )
    try:
        with OakCamera(width=args.width, height=args.height, fps=args.fps) as cam:
            while True:
                ok, frame = cam.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                detections = detect_fn(frame)
                centers = [d.center for d in detections]
                sel_i, sel_xy = _selected_center(aim, centers, frame.shape)
                overlay = draw_fn(frame, detections, selected_index=sel_i)
                overlay = draw_target(overlay, aim, selected_xy=sel_xy)

                now = time.monotonic()
                present = len(detections) > 0

                if present and not was_present and (now - last_fire) >= cooldown_s:
                    _put_status(overlay, "MOVING 180 deg (4 motors)")
                    if not show_frame(window_title, overlay):
                        break
                    print(
                        f"{mode_name.capitalize()} detected ({len(detections)}) "
                        "— moving all 4 motors 180°..."
                    )
                    turret.move_max_5v(steps=STEPS_180_DEG, clockwise=args.clockwise)
                    last_fire = time.monotonic()
                    print("Motor move complete.")
                else:
                    if present and sel_xy is not None:
                        status = (
                            "on target"
                            if aim.is_on_target(sel_xy, frame.shape)
                            else "tracking"
                        )
                    else:
                        status = "searching"
                    _put_status(overlay, status)
                    if not show_frame(window_title, overlay):
                        break

                was_present = present
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        turret.close()
        destroy_windows()
    return 0


def cmd_aim_faces(args: argparse.Namespace) -> int:
    detector = FaceDetector()
    return _run_aim_loop(
        args,
        mode_name="faces",
        window_title="Aim faces",
        detect_fn=detector.detect,
        draw_fn=draw_faces,
    )


def cmd_aim_flies(args: argparse.Namespace) -> int:
    detector = StaticFlyDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        dark_threshold=args.dark_threshold,
        confirm_frames=args.confirm_frames,
    )
    return _run_aim_loop(
        args,
        mode_name="flies",
        window_title="Aim flies",
        detect_fn=detector.detect,
        draw_fn=draw_flies,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fly-detect",
        description=(
            "Fly swatter aim modes: track faces or flies, show aim crosshair, "
            "and rotate all four steppers 180° on detection"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_camera_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--width", type=int, default=640, help="Preview width")
        p.add_argument("--height", type=int, default=480, help="Preview height")
        p.add_argument("--fps", type=int, default=30, help="Camera FPS")

    def add_target_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--target-x",
            type=float,
            default=None,
            help="Aim point X in pixels (overrides --target-norm-x)",
        )
        p.add_argument(
            "--target-y",
            type=float,
            default=None,
            help="Aim point Y in pixels (overrides --target-norm-y)",
        )
        p.add_argument(
            "--target-norm-x",
            type=float,
            default=0.5,
            help="Aim point X as fraction of frame width (default: 0.5)",
        )
        p.add_argument(
            "--target-norm-y",
            type=float,
            default=0.5,
            help="Aim point Y as fraction of frame height (default: 0.5)",
        )
        p.add_argument(
            "--target-radius",
            type=int,
            default=16,
            help="Pixel radius counted as lined up with the barrel",
        )

    def add_motor_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--motor-pins",
            type=str,
            default=None,
            help=(
                "Four motors as IN1,IN2,IN3,IN4 groups separated by ';'. "
                "Default: 17,27,22,23;5,6,13,19;12,16,20,21;18,24,25,8"
            ),
        )
        p.add_argument(
            "--cooldown",
            type=float,
            default=3.0,
            help="Seconds to wait before another motor move after a trigger",
        )
        p.add_argument(
            "--clockwise",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Motor direction for the 180° move (default: clockwise)",
        )

    p_faces = sub.add_parser(
        "aim-faces",
        help="Aim at faces: crosshair + move all 4 motors 180° on detection",
    )
    add_camera_args(p_faces)
    add_target_args(p_faces)
    add_motor_args(p_faces)
    p_faces.set_defaults(func=cmd_aim_faces)

    p_flies = sub.add_parser(
        "aim-flies",
        help="Aim at static flies: crosshair + move all 4 motors 180° on detection",
    )
    add_camera_args(p_flies)
    add_target_args(p_flies)
    add_motor_args(p_flies)
    p_flies.add_argument("--min-area", type=int, default=15, help="Min blob area")
    p_flies.add_argument("--max-area", type=int, default=900, help="Max blob area")
    p_flies.add_argument(
        "--dark-threshold",
        type=int,
        default=70,
        help="Grayscale cutoff: pixels darker than this are candidate flies",
    )
    p_flies.add_argument(
        "--confirm-frames",
        type=int,
        default=3,
        help="Frames a blob must persist to count as static",
    )
    p_flies.set_defaults(func=cmd_aim_flies)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
