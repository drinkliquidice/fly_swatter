import argparse
import time
from gpiozero import OutputDevice

class StepperController:
    def __init__(self, step_pin=17, dir_pin=27):
        """Initializes GPIO pins for STEP/DIR driver control on Pi 5."""
        self.step_pin = OutputDevice(step_pin)
        self.dir_pin = OutputDevice(dir_pin)

    def move(self, steps: int, clockwise: bool = True, start_delay: float = 0.004, min_delay: float = 0.0002, accel_steps: int = 400):
        """
        Drives the stepper motor with trapezoidal acceleration to prevent stalling.
        
        :param steps: Total pulse steps to execute.
        :param clockwise: True for CW rotation, False for CCW.
        :param start_delay: Initial pulse delay (seconds) at start/stop.
        :param min_delay: Target pulse delay (seconds) at top speed.
        :param accel_steps: Number of steps used for ramping up and down.
        """
        # Set direction line
        if clockwise:
            self.dir_pin.on()
        else:
            self.dir_pin.off()

        current_delay = start_delay
        delay_step = (start_delay - min_delay) / accel_steps if accel_steps > 0 else 0

        for step in range(steps):
            # Generate pulse
            self.step_pin.on()
            time.sleep(current_delay / 2)
            self.step_pin.off()
            time.sleep(current_delay / 2)

            # Acceleration phase (ramping up speed)
            if step < accel_steps and current_delay > min_delay:
                current_delay -= delay_step
            # Deceleration phase (ramping down speed)
            elif step >= (steps - accel_steps) and current_delay < start_delay:
                current_delay += delay_step

    def close(self):
        self.step_pin.close()
        self.dir_pin.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drive a stepper motor via GPIO on Raspberry Pi 5.")
    parser.add_argument("--steps", type=int, default=3200, help="Total steps to run")
    parser.add_argument("--ccw", action="store_true", help="Rotate counter-clockwise")
    args = parser.parse_args()

    motor = StepperController(step_pin=17, dir_pin=27)
    try:
        print(f"Running motor for {args.steps} steps...")
        motor.move(steps=args.steps, clockwise=not args.ccw)
        print("Finished successfully.")
    except KeyboardInterrupt:
        print("\nMotion interrupted by user.")
    finally:
        motor.close()