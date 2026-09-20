"""Turret motor control for the fly swatter."""

from .motor import (
    DEFAULT_MOTOR_PINS,
    FULL_STEP_SEQ,
    HALF_STEP_SEQ,
    MOTOR1_PINS,
    MOTOR2_PINS,
    MOTOR3_PINS,
    MOTOR4_PINS,
    SHOOT_STEPS,
    STEPS_180_DEG,
    STEPS_PER_REV,
    WIND_STEP_DELAY,
    FourMotorTurret,
    MaxSpeed5VStepper,
    TurretMotors,
    parse_motor_pins,
    steps_per_pixel,
)

__all__ = [
    "DEFAULT_MOTOR_PINS",
    "FULL_STEP_SEQ",
    "HALF_STEP_SEQ",
    "MOTOR1_PINS",
    "MOTOR2_PINS",
    "MOTOR3_PINS",
    "MOTOR4_PINS",
    "SHOOT_STEPS",
    "STEPS_180_DEG",
    "STEPS_PER_REV",
    "WIND_STEP_DELAY",
    "FourMotorTurret",
    "MaxSpeed5VStepper",
    "TurretMotors",
    "parse_motor_pins",
    "steps_per_pixel",
]
