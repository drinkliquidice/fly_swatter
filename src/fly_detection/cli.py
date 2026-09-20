"""CLI: scan + center-track faces or flies with motor1 pan."""

from __future__ import annotations

import argparse
import time
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

from .face_pipeline import FaceDetector, draw_faces
from .fly_pipeline import StaticFlyDetector, draw_flies
from .target import AimTarget, aim_target_from_args, draw_target


class TrackState(Enum):
    SEARCHING = auto()
    TRACKING = auto()


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


def _build_motors(args: argparse.Namespace):
    from turret_control import DEFAULT_MOTOR_PINS, TurretMotors, parse_motor_pins

    pins = parse_motor_pins(args.motor_pins) if args.motor_pins else DEFAULT_MOTOR_PINS
    return TurretMotors(motor_pins=pins)


def _run_target_loop(
    args: argparse.Namespace,
    *,
    mode_name: str,
    window_title: str,
    detect_fn: Callable,
    draw_fn: Callable,
) -> int:
    """
    Motor1 pans back and forth across a scan arc while searching.

    On target lock: stop the scan and keep the target centered in the
    upright frame (camera mount rotation is corrected in OakCamera).
    """
    from .oak_camera import OakCamera, destroy_windows, show_frame
    from turret_control import STEPS_PER_REV, steps_per_pixel

    aim = _aim_from_namespace(args)
    motors = _build_motors(args)
    state = TrackState.SEARCHING
    lost_frames = 0
    spp: Optional[float] = None

    scan_range = max(1, int(round(args.scan_degrees / 360.0 * STEPS_PER_REV)))
    scan_pos = 0  # steps from the CCW end of the sweep
    scan_dir = 1  # +1 toward CW end, -1 toward CCW end

    print(
        f"target-{mode_name}: motor1 sweeps ±{args.scan_degrees:.0f}° until a "
        f"target is found (camera mount {args.mount_rotate_ccw}° CCW corrected), "
        "then holds it centered. Press 'q' or Esc to quit."
    )
    try:
        with OakCamera(
            width=args.width,
            height=args.height,
            fps=args.fps,
            mount_rotate_ccw_deg=args.mount_rotate_ccw,
        ) as cam:
            while True:
                ok, frame = cam.read()
                if not ok or frame is None:
                    time.sleep(0.002)
                    continue

                if spp is None:
                    spp = steps_per_pixel(frame.shape[1], fov_deg=args.fov)

                detections = detect_fn(frame)
                centers = [d.center for d in detections]
                sel_i, sel_xy = _selected_center(aim, centers, frame.shape)
                overlay = draw_fn(frame, detections, selected_index=sel_i)
                overlay = draw_target(overlay, aim, selected_xy=sel_xy)

                if sel_xy is not None:
                    lost_frames = 0
                    dx, _dy, _dist = aim.error_to(sel_xy, frame.shape)

                    if state is TrackState.SEARCHING:
                        motors.pan_stop()
                        state = TrackState.TRACKING
                        print(f"Target locked ({mode_name}) — centering with motor1.")

                    if abs(dx) <= args.target_radius:
                        motors.pan_stop()
                        _put_status(overlay, "TRACKING on-center")
                    else:
                        raw_steps = int(round(abs(dx) * spp))
                        steps = max(1, min(raw_steps, args.max_track_steps))
                        move_cw = (dx > 0) ^ args.invert_pan
                        # Logical sweep position (independent of invert-pan wiring).
                        scan_pos = int(
                            max(
                                0,
                                min(
                                    scan_range,
                                    scan_pos + (steps if dx > 0 else -steps),
                                ),
                            )
                        )
                        motors.pan_step(
                            steps,
                            clockwise=move_cw,
                            delay=args.step_delay,
                        )
                        _put_status(
                            overlay,
                            f"TRACKING dx={dx} steps={steps}",
                        )
                else:
                    lost_frames += 1
                    if (
                        state is TrackState.TRACKING
                        and lost_frames >= args.lost_frames
                    ):
                        state = TrackState.SEARCHING
                        print("Target lost — resuming 90° back-and-forth scan.")

                    if state is TrackState.SEARCHING:
                        if scan_dir > 0 and scan_pos >= scan_range:
                            scan_dir = -1
                        elif scan_dir < 0 and scan_pos <= 0:
                            scan_dir = 1

                        if scan_dir > 0:
                            remaining = scan_range - scan_pos
                        else:
                            remaining = scan_pos
                        steps = min(args.scan_steps, max(remaining, 0))
                        if steps <= 0:
                            scan_dir *= -1
                            steps = min(args.scan_steps, scan_range)

                        move_cw = (scan_dir > 0) ^ args.invert_pan
                        motors.pan_step(
                            steps,
                            clockwise=move_cw,
                            delay=args.step_delay,
                        )
                        scan_pos = int(
                            max(0, min(scan_range, scan_pos + scan_dir * steps))
                        )
                        deg = scan_pos / STEPS_PER_REV * 360.0
                        _put_status(
                            overlay,
                            f"SEARCHING sweep {deg:.0f}/{args.scan_degrees:.0f} deg",
                        )
                    else:
                        motors.pan_stop()
                        _put_status(overlay, "TRACKING (target briefly lost)")

                if not show_frame(window_title, overlay):
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        motors.stop_all()
        motors.close()
        destroy_windows()
    return 0


def cmd_target_face(args: argparse.Namespace) -> int:
    detector = FaceDetector()
    return _run_target_loop(
        args,
        mode_name="face",
        window_title="target-face",
        detect_fn=detector.detect,
        draw_fn=draw_faces,
    )


def cmd_target_flies(args: argparse.Namespace) -> int:
    detector = StaticFlyDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        dark_threshold=args.dark_threshold,
        confirm_frames=args.confirm_frames,
    )
    return _run_target_loop(
        args,
        mode_name="flies",
        window_title="target-flies",
        detect_fn=detector.detect,
        draw_fn=draw_flies,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fly_swatter",
        description=(
            "Scan with motor1 until a face or fly is found, then keep it "
            "centered in the camera frame"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--width", type=int, default=640, help="Preview width")
        p.add_argument("--height", type=int, default=480, help="Preview height")
        p.add_argument("--fps", type=int, default=30, help="Camera FPS")
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
            help="Aim X as fraction of width (default: center)",
        )
        p.add_argument(
            "--target-norm-y",
            type=float,
            default=0.5,
            help="Aim Y as fraction of height (default: center)",
        )
        p.add_argument(
            "--target-radius",
            type=int,
            default=20,
            help="Deadzone pixels counted as centered",
        )
        p.add_argument(
            "--mount-rotate-ccw",
            type=int,
            default=90,
            choices=(0, 90, 180, 270),
            help="Physical camera mount rotation CCW degrees (default: 90)",
        )
        p.add_argument(
            "--fov",
            type=float,
            default=55.0,
            help=(
                "Upright horizontal FOV in degrees after mount correction "
                "(OAK-1 ~55° once rotated 90°; was ~69° before rotation)"
            ),
        )
        p.add_argument(
            "--scan-degrees",
            type=float,
            default=90.0,
            help="Search sweep arc in degrees (back and forth, default: 90)",
        )
        p.add_argument(
            "--scan-steps",
            type=int,
            default=8,
            help="Motor1 steps between frames while searching",
        )
        p.add_argument(
            "--max-track-steps",
            type=int,
            default=64,
            help="Max motor1 steps per frame while correcting aim",
        )
        p.add_argument(
            "--step-delay",
            type=float,
            default=0.0012,
            help="Delay between stepper phases (seconds)",
        )
        p.add_argument(
            "--invert-pan",
            action="store_true",
            help="Invert motor1 direction relative to pixel error / scan",
        )
        p.add_argument(
            "--lost-frames",
            type=int,
            default=15,
            help="Frames without a target before resuming search",
        )
        p.add_argument(
            "--motor-pins",
            type=str,
            default=None,
            help=(
                "Override pins as m1;m2;m3;m4 each IN1,IN2,IN3,IN4. "
                "Default: 17,27,22,23;10,9,11,25;5,6,13,12;19,16,26,20"
            ),
        )

    p_face = sub.add_parser(
        "target-face",
        help="Motor1 scan/track human faces",
    )
    add_shared(p_face)
    p_face.set_defaults(func=cmd_target_face)

    p_flies = sub.add_parser(
        "target-flies",
        help="Motor1 scan/track static flies",
    )
    add_shared(p_flies)
    p_flies.add_argument("--min-area", type=int, default=15, help="Min blob area")
    p_flies.add_argument("--max-area", type=int, default=900, help="Max blob area")
    p_flies.add_argument(
        "--dark-threshold",
        type=int,
        default=70,
        help="Grayscale cutoff for fly blobs",
    )
    p_flies.add_argument(
        "--confirm-frames",
        type=int,
        default=3,
        help="Frames a blob must persist to count as static",
    )
    p_flies.set_defaults(func=cmd_target_flies)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
