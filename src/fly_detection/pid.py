"""Simple PID controller for turret aiming."""

from __future__ import annotations


class PID:
    """
    Standard PID on a single error signal.

    Defaults are P-only (kp=0.1, ki=0, kd=0) for stable hackathon aiming.
    """

    def __init__(self, kp: float = 0.1, ki: float = 0.0, kd: float = 0.0) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0
        self._has_prev = False

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._has_prev = False

    def update(self, error: float, dt: float = 1.0) -> float:
        """Return control effort for ``error`` (same units as the error)."""
        if dt <= 0:
            dt = 1.0
        self._integral += error * dt
        if self._has_prev:
            derivative = (error - self._prev_error) / dt
        else:
            derivative = 0.0
            self._has_prev = True
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative
