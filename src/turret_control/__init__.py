"""Turret motor control for the fly swatter."""

from .motor import (
    DEFAULT_MOTOR_PINS,
    STEPS_180_DEG,
    STEPS_PER_REV,
    FourMotorTurret,
    MaxSpeed5VStepper,
    parse_motor_pins,
)

__all__ = [
    "DEFAULT_MOTOR_PINS",
    "STEPS_180_DEG",
    "STEPS_PER_REV",
    "FourMotorTurret",
    "MaxSpeed5VStepper",
    "parse_motor_pins",
]
