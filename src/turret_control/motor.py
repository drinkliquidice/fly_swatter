import time
from gpiozero import OutputDevice

class MaxSpeed5VStepper:
    def __init__(self, in1=17, in2=27, in3=22, in4=23):
        """Initializes GPIO pins for ULN2003 driver on Pi 5."""
        self.pins = [
            OutputDevice(in1),
            OutputDevice(in2),
            OutputDevice(in3),
            OutputDevice(in4)
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
        # Tuned parameters for standard 5V operation
        start_delay = 0.0025  # 2.5ms start delay (prevents initial rotor slip)
        min_delay = 0.00085   # 0.85ms peak delay (absolute max speed threshold at 5V)
        accel_steps = 300     # Ramp steps required to hit top speed safely
        
        seq = self.sequence if clockwise else list(reversed(self.sequence))
        seq_len = len(seq)
        
        current_delay = start_delay
        delay_step = (start_delay - min_delay) / accel_steps

        for step in range(steps):
            pattern = seq[step % seq_len]
            
            # Apply pin outputs
            for pin, state in zip(self.pins, pattern):
                if state:
                    pin.on()
                else:
                    pin.off()

            time.sleep(current_delay)

            # Acceleration phase
            if step < accel_steps and current_delay > min_delay:
                current_delay -= delay_step
            # Deceleration phase
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

if __name__ == "__main__":
    motor = MaxSpeed5VStepper(in1=17, in2=27, in3=22, in4=23)
    try:
        print("Running 28BYJ-48 at maximum 5V speed...")
        # 2048 steps = 1 full revolution in 2-phase full-step mode
        motor.move_max_5v(steps=2048, clockwise=True)
        print("Motion complete.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        motor.close()