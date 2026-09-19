import time
from typing import List, Sequence, Tuple

from gpiozero import OutputDevice

# Default BCM pins for four ULN2003 boards on a Pi 5.
DEFAULT_MOTOR_PINS: Tuple[Tuple[int, int, int, int], ...] = (
    (17, 27, 22, 23),  # motor 0
    (5, 6, 13, 19),    # motor 1
    (12, 16, 20, 21),  # motor 2
    (18, 24, 25, 8),   # motor 3
)

# 28BYJ-48 @ 2-phase full step: 2048 steps = 360°.
STEPS_PER_REV = 2048
STEPS_180_DEG = STEPS_PER_REV // 2


class MaxSpeed5VStepper:
    def __init__(self, in1=17, in2=27, in3=22, in4=23):
        """Initializes GPIO pins for ULN2003 driver on Pi 5."""
        self.pins = [
            OutputDevice(in1),
            OutputDevice(in2),
            OutputDevice(in3),
            OutputDevice(in4),
        ]

        # 2-Phase Full-Stepping: Delivers maximum torque at 5V
        self.sequence = [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ]

    def move_max_5v(self, steps=2048, clockwise=True):
        """
        Drives the motor at peak 5V speed using a calibrated acceleration curve.

        :param steps: Total steps to execute (2048 full-steps = 1 revolution)
        :param clockwise: Direction of rotation
        """
        start_delay = 0.0025
        min_delay = 0.00085
        accel_steps = 300

        seq = self.sequence if clockwise else list(reversed(self.sequence))
        seq_len = len(seq)

        current_delay = start_delay
        delay_step = (start_delay - min_delay) / accel_steps

        for step in range(steps):
            pattern = seq[step % seq_len]

            for pin, state in zip(self.pins, pattern):
                if state:
                    pin.on()
                else:
                    pin.off()

            time.sleep(current_delay)

            if step < accel_steps and current_delay > min_delay:
                current_delay -= delay_step
            elif step >= (steps - accel_steps) and current_delay < start_delay:
                current_delay += delay_step

        self.stop()

    def stop(self):
        """Disables all pins to avoid drawing continuous current at 5V when stationary."""
        for pin in self.pins:
            pin.off()

    def close(self):
        self.stop()
        for pin in self.pins:
            pin.close()


class FourMotorTurret:
    """
    Four 28BYJ-48 steppers on ULN2003 boards, stepped in lockstep.

    All motors share the same motion command so a detection trigger turns
    every axis together (placeholder until per-axis kinematics exist).
    """

    def __init__(
        self,
        motor_pins: Sequence[Sequence[int]] | None = None,
    ) -> None:
        pins = motor_pins or DEFAULT_MOTOR_PINS
        if len(pins) != 4:
            raise ValueError(f"Expected 4 motors, got {len(pins)}")
        self.motors: List[MaxSpeed5VStepper] = [
            MaxSpeed5VStepper(in1=p[0], in2=p[1], in3=p[2], in4=p[3]) for p in pins
        ]
        self.sequence = self.motors[0].sequence

    def move_max_5v(self, steps: int = STEPS_180_DEG, clockwise: bool = True) -> None:
        """Move all four motors the same number of steps, in sync."""
        start_delay = 0.0025
        min_delay = 0.00085
        accel_steps = 300

        seq = self.sequence if clockwise else list(reversed(self.sequence))
        seq_len = len(seq)
        current_delay = start_delay
        delay_step = (start_delay - min_delay) / accel_steps

        for step in range(steps):
            pattern = seq[step % seq_len]
            for motor in self.motors:
                for pin, state in zip(motor.pins, pattern):
                    if state:
                        pin.on()
                    else:
                        pin.off()

            time.sleep(current_delay)

            if step < accel_steps and current_delay > min_delay:
                current_delay -= delay_step
            elif step >= (steps - accel_steps) and current_delay < start_delay:
                current_delay += delay_step

        self.stop()

    def stop(self) -> None:
        for motor in self.motors:
            motor.stop()

    def close(self) -> None:
        for motor in self.motors:
            motor.close()


def parse_motor_pins(spec: str) -> Tuple[Tuple[int, int, int, int], ...]:
    """
    Parse ``m0in1,m0in2,m0in3,m0in4;m1...;m2...;m3...`` into four pin tuples.
    """
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
    turret = FourMotorTurret()
    try:
        print("Running four 28BYJ-48 motors 180° at maximum 5V speed...")
        turret.move_max_5v(steps=STEPS_180_DEG, clockwise=True)
        print("Motion complete.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        turret.close()
