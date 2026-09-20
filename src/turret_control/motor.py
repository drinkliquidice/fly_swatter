"""Four ULN2003 / 28BYJ-48 motors for the fly swatter turret."""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

from gpiozero import OutputDevice

# BCM pin map from the Pi 5 header wiring (IN1, IN2, IN3, IN4).
MOTOR1_PINS = (17, 27, 22, 23)  # pan / side-to-side (yaw)
MOTOR2_PINS = (10, 9, 11, 25)   # turret elevation / up-down (pitch)
MOTOR3_PINS = (5, 6, 13, 12)    # spring / rubber-band wind
MOTOR4_PINS = (19, 16, 26, 20)  # opposing wind (mirrors motor 3)

DEFAULT_MOTOR_PINS: Tuple[Tuple[int, int, int, int], ...] = (
    MOTOR1_PINS,
    MOTOR2_PINS,
    MOTOR3_PINS,
    MOTOR4_PINS,
)

# 28BYJ-48 @ 2-phase full step: 2048 steps = 360°.
STEPS_PER_REV = 2048
STEPS_180_DEG = STEPS_PER_REV // 2

# Slow step delay for max torque on 28BYJ-48 @ 5V (pan / tilt).
DEFAULT_STEP_DELAY = 0.004


class MaxSpeed5VStepper:
    """Single 28BYJ-48 on a ULN2003 driver."""

    def __init__(self, in1: int, in2: int, in3: int, in4: int):
        self.pins = [
            OutputDevice(in1),
            OutputDevice(in2),
            OutputDevice(in3),
            OutputDevice(in4),
        ]
        self.sequence = [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ]
        self._phase = 0

    def _apply(self, pattern: Sequence[int]) -> None:
        for pin, state in zip(self.pins, pattern):
            if state:
                pin.on()
            else:
                pin.off()

    def step(self, steps: int = 1, clockwise: bool = True, delay: float = DEFAULT_STEP_DELAY) -> None:
        """Take ``steps`` full-steps (no accel curve — good for scan/track)."""
        if steps <= 0:
            return
        seq = self.sequence if clockwise else list(reversed(self.sequence))
        seq_len = len(seq)
        for _ in range(steps):
            self._phase = (self._phase + 1) % seq_len
            self._apply(seq[self._phase])
            time.sleep(delay)

    def move_max_5v(self, steps: int = STEPS_PER_REV, clockwise: bool = True) -> None:
        """Blocking move with acceleration (large wind / homing moves)."""
        # Torque-first accel profile (slower than peak 5V speed).
        start_delay = 0.005
        min_delay = 0.003
        accel_steps = min(200, max(1, steps // 4))

        seq = self.sequence if clockwise else list(reversed(self.sequence))
        seq_len = len(seq)
        current_delay = start_delay
        delay_step = (start_delay - min_delay) / accel_steps

        for step_i in range(steps):
            self._phase = (self._phase + 1) % seq_len
            self._apply(seq[self._phase])
            time.sleep(current_delay)

            if step_i < accel_steps and current_delay > min_delay:
                current_delay -= delay_step
            elif step_i >= (steps - accel_steps) and current_delay < start_delay:
                current_delay += delay_step

        self.stop()

    def stop(self) -> None:
        for pin in self.pins:
            pin.off()

    def close(self) -> None:
        self.stop()
        for pin in self.pins:
            pin.close()


class TurretMotors:
    """
    Named access to all four steppers.

    - motor1: pan (side-to-side) — search sweep + X tracking
    - motor2: turret elevation (up-down) — Y tracking
    - motor3 + motor4: opposing spring/rubber-band winders
    """

    def __init__(
        self,
        motor_pins: Sequence[Sequence[int]] | None = None,
    ) -> None:
        pins = list(motor_pins or DEFAULT_MOTOR_PINS)
        if len(pins) != 4:
            raise ValueError(f"Expected 4 motors, got {len(pins)}")
        self.motor1 = MaxSpeed5VStepper(*pins[0])
        self.motor2 = MaxSpeed5VStepper(*pins[1])
        self.motor3 = MaxSpeed5VStepper(*pins[2])
        self.motor4 = MaxSpeed5VStepper(*pins[3])
        self.motors = [self.motor1, self.motor2, self.motor3, self.motor4]

    # --- Motor 1: pan (X) ----------------------------------------------------

    def pan_step(
        self,
        steps: int = 1,
        clockwise: bool = True,
        delay: float = DEFAULT_STEP_DELAY,
    ) -> None:
        self.motor1.step(steps, clockwise=clockwise, delay=delay)

    def pan_stop(self) -> None:
        self.motor1.stop()

    # --- Motor 2: elevation / tilt (Y) ---------------------------------------

    def tilt_step(
        self,
        steps: int = 1,
        clockwise: bool = True,
        delay: float = DEFAULT_STEP_DELAY,
    ) -> None:
        self.motor2.step(steps, clockwise=clockwise, delay=delay)

    def tilt_stop(self) -> None:
        self.motor2.stop()

    # Back-compat aliases
    def turret_step(
        self,
        steps: int = 1,
        clockwise: bool = True,
        delay: float = DEFAULT_STEP_DELAY,
    ) -> None:
        self.tilt_step(steps, clockwise=clockwise, delay=delay)

    def turret_stop(self) -> None:
        self.tilt_stop()

    def correct_aim(
        self,
        pan_steps: int = 0,
        pan_cw: bool = True,
        tilt_steps: int = 0,
        tilt_cw: bool = True,
        delay: float = DEFAULT_STEP_DELAY,
    ) -> None:
        """Interleave pan (M1) and tilt (M2) steps for XY centering."""
        pan_steps = max(0, int(pan_steps))
        tilt_steps = max(0, int(tilt_steps))
        if pan_steps == 0 and tilt_steps == 0:
            return
        pan_seq = (
            self.motor1.sequence
            if pan_cw
            else list(reversed(self.motor1.sequence))
        )
        tilt_seq = (
            self.motor2.sequence
            if tilt_cw
            else list(reversed(self.motor2.sequence))
        )
        for i in range(max(pan_steps, tilt_steps)):
            if i < pan_steps:
                self.motor1._phase = (self.motor1._phase + 1) % 4
                self.motor1._apply(pan_seq[self.motor1._phase])
            if i < tilt_steps:
                self.motor2._phase = (self.motor2._phase + 1) % 4
                self.motor2._apply(tilt_seq[self.motor2._phase])
            time.sleep(delay)
        if pan_steps:
            self.motor1.stop()
        if tilt_steps:
            self.motor2.stop()

    # --- Motors 3 + 4: opposing winders --------------------------------------

    def wind(
        self,
        steps: int,
        *,
        tighten: bool = True,
        delay: float = DEFAULT_STEP_DELAY,
    ) -> None:
        """
        Wind the spring/rubber band.

        Motors face each other, so motor4 always runs opposite motor3.
        ``tighten=True`` winds in; ``False`` releases.
        """
        if steps <= 0:
            return
        m3_cw = tighten
        m4_cw = not tighten
        seq3 = (
            self.motor3.sequence
            if m3_cw
            else list(reversed(self.motor3.sequence))
        )
        seq4 = (
            self.motor4.sequence
            if m4_cw
            else list(reversed(self.motor4.sequence))
        )
        for _ in range(steps):
            self.motor3._phase = (self.motor3._phase + 1) % 4
            self.motor4._phase = (self.motor4._phase + 1) % 4
            self.motor3._apply(seq3[self.motor3._phase])
            self.motor4._apply(seq4[self.motor4._phase])
            time.sleep(delay)
        self.motor3.stop()
        self.motor4.stop()

    def wind_degrees(self, degrees: float, *, tighten: bool = True) -> None:
        steps = int(round(abs(degrees) / 360.0 * STEPS_PER_REV))
        self.wind(steps, tighten=tighten)

    def stop_all(self) -> None:
        for motor in self.motors:
            motor.stop()

    def close(self) -> None:
        for motor in self.motors:
            motor.close()


# Back-compat alias used by older scripts.
FourMotorTurret = TurretMotors


def steps_per_pixel(frame_span: int, fov_deg: float = 69.0) -> float:
    """
    Convert pixel error along one axis → motor steps (1:1 mount).

    Use frame width + horizontal FOV for pan (X), frame height + vertical
    FOV for tilt (Y).
    """
    if frame_span <= 0:
        raise ValueError("frame_span must be positive")
    return (STEPS_PER_REV / 360.0) * (fov_deg / float(frame_span))


def parse_motor_pins(spec: str) -> Tuple[Tuple[int, int, int, int], ...]:
    """Parse ``m1;m2;m3;m4`` where each group is ``IN1,IN2,IN3,IN4``."""
    groups = [g.strip() for g in spec.split(";") if g.strip()]
    if len(groups) != 4:
        raise ValueError(
            "Expected 4 motor pin groups separated by ';', "
            f"got {len(groups)}: {spec!r}"
        )
    parsed: List[Tuple[int, int, int, int]] = []
    for group in groups:
        parts = [int(p.strip()) for p in group.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Each motor needs 4 pins, got {group!r}")
        parsed.append((parts[0], parts[1], parts[2], parts[3]))
    return tuple(parsed)


if __name__ == "__main__":
    turret = TurretMotors()
    try:
        print("Smoke: pan motor1 90°, then wind m3/m4 45° opposing...")
        turret.pan_step(STEPS_PER_REV // 4, clockwise=True)
        turret.pan_stop()
        turret.wind_degrees(45, tighten=True)
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        turret.close()
