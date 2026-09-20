"""CLI: scan + XY center-track faces or flies (pan M1, tilt M2)."""

from __future__ import annotations

import argparse
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

from .face_pipeline import FaceDetector, draw_faces
from .fly_pipeline import StaticFlyDetector, draw_flies
from .pid import PID
from .target import AimTarget, aim_target_from_args, draw_target


class TrackState(Enum):
    SEARCHING = auto()
    TRACKING = auto()


class FirePhase(Enum):
    IDLE = auto()
    SHOOTING = auto()
    RELOADING = auto()


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
    Full target pipeline:
      1. Motor1 sweeps while searching
      2. On lock: proportional X (M1) + Y (M2) centering
      3. When within ``target_radius`` px: M3/M4 shoot (0→5086) then reload
    """
    from .oak_camera import OakCamera, destroy_windows, show_frame
    from turret_control import STEPS_PER_REV, steps_per_pixel

    aim = _aim_from_namespace(args)
    motors = _build_motors(args)
    state = TrackState.SEARCHING
    fire_phase = FirePhase.IDLE
    lost_frames = 0
    spp_x: Optional[float] = None
    spp_y: Optional[float] = None
    last_shot = 0.0
    # Require leaving tolerance (or cooldown) before the next shot.
    shot_this_lock = False
    shoot_thread: Optional[threading.Thread] = None
    shoot_error: list[BaseException] = []
    # P-only PID lock (ki=kd=0) on pixel error for pan (X) and tilt (Y).
    pid_x = PID(kp=args.kp, ki=0.0, kd=0.0)
    pid_y = PID(kp=args.kp, ki=0.0, kd=0.0)
    last_pid_t = time.monotonic()

    scan_range = max(1, int(round(args.scan_degrees / 360.0 * STEPS_PER_REV)))
    scan_pos = 0
    scan_dir = 1

    def _start_shoot(dx: int, dy: int) -> None:
        nonlocal shoot_thread, fire_phase, shot_this_lock

        def _worker() -> None:
            nonlocal fire_phase
            try:
                fire_phase = FirePhase.SHOOTING
                motors.shoot(
                    steps=args.shoot_steps,
                    invert_m4=args.invert_m4,
                    delay=args.wind_delay,
                )
                fire_phase = FirePhase.RELOADING
                motors.reload(
                    steps=args.shoot_steps,
                    invert_m4=args.invert_m4,
                    delay=args.wind_delay,
                )
            except BaseException as exc:  # noqa: BLE001 — report back to loop
                shoot_error.append(exc)
            finally:
                fire_phase = FirePhase.IDLE

        print(
            f"On target (dx={dx}, dy={dy}) — "
            f"shoot while aiming, then reload while still..."
        )
        shot_this_lock = True
        shoot_error.clear()
        fire_phase = FirePhase.SHOOTING
        shoot_thread = threading.Thread(target=_worker, daemon=True)
        shoot_thread.start()

    def _finish_shoot_if_done() -> None:
        nonlocal shoot_thread, fire_phase, last_shot
        if shoot_thread is None:
            return
        if shoot_thread.is_alive():
            return
        shoot_thread.join(timeout=0.0)
        shoot_thread = None
        if shoot_error:
            print(f"Shoot/reload failed: {shoot_error[0]}")
            shoot_error.clear()
        else:
            print("Ready (loaded at 0). Resuming full track.")
        last_shot = time.monotonic()
        fire_phase = FirePhase.IDLE

    print(
        f"target-{mode_name}: scan → track X/Y → shoot+reload when within "
        f"{args.target_radius}px (shoot={args.shoot_steps} full-steps). "
        "Press 'q' or Esc to quit."
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

                if spp_x is None or spp_y is None:
                    h, w = frame.shape[:2]
                    spp_x = steps_per_pixel(w, fov_deg=args.fov)
                    spp_y = steps_per_pixel(h, fov_deg=args.fov_v)

                _finish_shoot_if_done()
                reloading = fire_phase is FirePhase.RELOADING
                shooting = fire_phase is FirePhase.SHOOTING

                detections = detect_fn(frame)
                centers = [d.center for d in detections]
                sel_i, sel_xy = _selected_center(aim, centers, frame.shape)
                overlay = draw_fn(frame, detections, selected_index=sel_i)
                overlay = draw_target(overlay, aim, selected_xy=sel_xy)

                # While reloading: keep the feed live, but hold pan/tilt still.
                if reloading:
                    motors.pan_stop()
                    motors.tilt_stop()
                    _put_status(overlay, "RELOADING (aim frozen)")
                    if not show_frame(window_title, overlay):
                        break
                    continue

                if sel_xy is not None:
                    lost_frames = 0
                    dx, dy, _dist = aim.error_to(sel_xy, frame.shape)

                    if state is TrackState.SEARCHING:
                        motors.pan_stop()
                        motors.tilt_stop()
                        state = TrackState.TRACKING
                        shot_this_lock = False
                        pid_x.reset()
                        pid_y.reset()
                        last_pid_t = time.monotonic()
                        print(
                            f"Target locked ({mode_name}) — "
                            f"PID lock (kp={args.kp}) on motor1 (X) + motor2 (Y)."
                        )

                    on_x = abs(dx) <= args.target_radius
                    on_y = abs(dy) <= args.target_radius

                    if on_x and on_y:
                        motors.pan_stop()
                        motors.tilt_stop()
                        now = time.monotonic()
                        can_shoot = (
                            not shot_this_lock
                            and not shooting
                            and (now - last_shot) >= args.shoot_cooldown
                            and (shoot_thread is None)
                        )
                        if can_shoot:
                            _put_status(overlay, "SHOOTING (still aiming)")
                            _start_shoot(dx, dy)
                        elif shooting:
                            _put_status(overlay, "SHOOTING + ON TARGET")
                        else:
                            _put_status(overlay, "ON TARGET (armed/cooldown)")
                    else:
                        # Left the deadzone — allow another shot on re-center.
                        if shot_this_lock and not shooting and (
                            abs(dx) > args.target_radius * 2
                            or abs(dy) > args.target_radius * 2
                        ):
                            shot_this_lock = False

                        now_pid = time.monotonic()
                        dt = max(1e-3, now_pid - last_pid_t)
                        last_pid_t = now_pid

                        pan_steps = 0
                        tilt_steps = 0
                        pan_cw = True
                        tilt_cw = True

                        if not on_x:
                            # P-only PID on pixel error → motor half-steps.
                            u_x = pid_x.update(float(dx), dt=dt)
                            raw = int(round(abs(u_x) * spp_x))
                            pan_steps = max(1, min(raw, args.max_track_steps))
                            pan_cw = (u_x < 0) ^ args.invert_pan
                            scan_pos = int(
                                max(
                                    0,
                                    min(
                                        scan_range,
                                        scan_pos
                                        + (pan_steps if dx > 0 else -pan_steps),
                                    ),
                                )
                            )
                        else:
                            pid_x.reset()

                        if not on_y:
                            u_y = pid_y.update(float(dy), dt=dt)
                            raw = int(round(abs(u_y) * spp_y))
                            tilt_steps = max(1, min(raw, args.max_track_steps))
                            tilt_cw = (u_y < 0) ^ args.invert_tilt
                        else:
                            pid_y.reset()

                        motors.correct_aim(
                            pan_steps=pan_steps,
                            pan_cw=pan_cw,
                            tilt_steps=tilt_steps,
                            tilt_cw=tilt_cw,
                            delay=args.step_delay,
                        )
                        status = (
                            f"SHOOTING + PID dx={dx} dy={dy}"
                            if shooting
                            else f"PID dx={dx} dy={dy} "
                            f"pan={pan_steps} tilt={tilt_steps}"
                        )
                        _put_status(overlay, status)
                else:
                    lost_frames += 1
                    if (
                        state is TrackState.TRACKING
                        and lost_frames >= args.lost_frames
                        and not shooting
                    ):
                        state = TrackState.SEARCHING
                        shot_this_lock = False
                        pid_x.reset()
                        pid_y.reset()
                        motors.tilt_stop()
                        print("Target lost — resuming pan sweep.")

                    if state is TrackState.SEARCHING and not shooting:
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

                        move_cw = (scan_dir < 0) ^ args.invert_pan
                        motors.pan_step(
                            steps,
                            clockwise=move_cw,
                            delay=args.scan_step_delay,
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
                        motors.tilt_stop()
                        label = (
                            "SHOOTING (no target)"
                            if shooting
                            else "TRACKING (target briefly lost)"
                        )
                        _put_status(overlay, label)

                if not show_frame(window_title, overlay):
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        motors.cancel_wind()
        if shoot_thread is not None and shoot_thread.is_alive():
            shoot_thread.join(timeout=2.0)
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
        min_fly_w=args.min_fly_w,
        min_fly_h=args.min_fly_h,
        max_fly_w=args.max_fly_w,
        max_fly_h=args.max_fly_h,
        min_paper_w=args.min_paper_w,
        min_paper_h=args.min_paper_h,
        white_threshold=args.white_threshold,
        dark_threshold=args.dark_threshold,
        confirm_frames=args.confirm_frames,
        require_portrait=not args.allow_landscape_paper,
        min_paper_aspect=args.min_paper_aspect,
        max_paper_aspect=args.max_paper_aspect,
    )

    def _draw(frame, flies, selected_index=None):
        return draw_flies(
            frame,
            flies,
            selected_index=selected_index,
            papers=detector.last_papers,
        )

    return _run_target_loop(
        args,
        mode_name="flies",
        window_title="target-flies",
        detect_fn=detector.detect,
        draw_fn=_draw,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fly_swatter",
        description=(
            "Scan, center on a face/fly in X+Y, then shoot+reload when "
            "within tolerance"
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
            help="Deadzone pixels counted as centered / allowed to shoot (default: 20)",
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
                "(used for pan / X)"
            ),
        )
        p.add_argument(
            "--fov-v",
            type=float,
            default=69.0,
            help=(
                "Upright vertical FOV in degrees after mount correction "
                "(used for tilt / Y)"
            ),
        )
        p.add_argument(
            "--kp",
            "--track-gain",
            type=float,
            default=0.1,
            help="PID proportional gain for pan/tilt lock (P-only; default: 0.1)",
        )
        p.add_argument(
            "--scan-degrees",
            type=float,
            default=100.0,
            help="Search sweep arc in degrees (back and forth, default: 145)",
        )
        p.add_argument(
            "--scan-steps",
            type=int,
            default=5,
            help="Motor1 half-steps between frames while searching (lower = slower)",
        )
        p.add_argument(
            "--scan-step-delay",
            type=float,
            default=0.007,
            help="Delay between search pan steps in seconds (higher = slower; default: 0.007)",
        )
        p.add_argument(
            "--max-track-steps",
            type=int,
            default=128,
            help="Max pan or tilt half-steps per frame while correcting aim",
        )
        p.add_argument(
            "--step-delay",
            type=float,
            default=0.004,
            help="Delay between track stepper phases in seconds (higher = more torque; default: 0.004)",
        )
        p.add_argument(
            "--invert-pan",
            action="store_true",
            help="Invert motor1 (pan/X) direction",
        )
        p.add_argument(
            "--invert-tilt",
            action="store_true",
            help="Invert motor2 (tilt/Y) direction",
        )
        p.add_argument(
            "--lost-frames",
            type=int,
            default=15,
            help="Frames without a target before resuming search",
        )
        p.add_argument(
            "--shoot-steps",
            type=int,
            default=5086,
            help="Full-steps from loaded (0) to shot (default: 5086)",
        )
        p.add_argument(
            "--wind-delay",
            type=float,
            default=0.002,
            help="M3/M4 step delay while shooting/reloading (default: 0.002s)",
        )
        p.add_argument(
            "--shoot-cooldown",
            type=float,
            default=2.0,
            help="Minimum seconds between shots",
        )
        p.add_argument(
            "--invert-m4",
            action="store_true",
            help="Invert motor4 relative to motor3 while winding (default: not inverted)",
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
        help="Scan/track printed flies on portrait letter/A4 paper",
    )
    add_shared(p_flies)
    p_flies.add_argument(
        "--min-fly-w", type=int, default=10, help="Min fly width in pixels"
    )
    p_flies.add_argument(
        "--min-fly-h", type=int, default=10, help="Min fly height in pixels"
    )
    p_flies.add_argument(
        "--max-fly-w", type=int, default=280, help="Max fly width in pixels"
    )
    p_flies.add_argument(
        "--max-fly-h", type=int, default=280, help="Max fly height in pixels"
    )
    p_flies.add_argument(
        "--min-paper-w",
        type=int,
        default=160,
        help="Min paper width in pixels (portrait letter/A4)",
    )
    p_flies.add_argument(
        "--min-paper-h",
        type=int,
        default=220,
        help="Min paper height in pixels (portrait letter/A4)",
    )
    p_flies.add_argument(
        "--min-paper-aspect",
        type=float,
        default=1.15,
        help="Min paper height/width (letter≈1.29, A4≈1.41)",
    )
    p_flies.add_argument(
        "--max-paper-aspect",
        type=float,
        default=1.75,
        help="Max paper height/width (allows camera perspective)",
    )
    p_flies.add_argument(
        "--allow-landscape-paper",
        action="store_true",
        help="Disable portrait-only paper filter",
    )
    p_flies.add_argument(
        "--white-threshold",
        type=int,
        default=145,
        help="Grayscale cutoff for paper (brighter = paper)",
    )
    p_flies.add_argument(
        "--dark-threshold",
        type=int,
        default=175,
        help="Grayscale cutoff for printed fly on the paper (higher = more sensitive)",
    )
    p_flies.add_argument(
        "--confirm-frames",
        type=int,
        default=1,
        help="Frames a fly must persist (1 = immediate)",
    )
    p_flies.set_defaults(func=cmd_target_flies)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
