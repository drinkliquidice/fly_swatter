# fly_swatter

Hack The North 2026 Automatic Fly Detection Swatter

## Hardware

- Raspberry Pi 5
- Luxonis OAK-1 camera
- Four 28BYJ-48 + ULN2003 drivers

### GPIO map (BCM)

| Motor | Role                         | IN1 | IN2 | IN3 | IN4 |
|-------|------------------------------|-----|-----|-----|-----|
| M1    | Pan (side-to-side / X)       | 17  | 27  | 22  | 23  |
| M2    | Tilt (up-down / Y)           | 10  | 9   | 11  | 25  |
| M3    | Spring / rubber-band wind    | 5   | 6   | 13  | 12  |
| M4    | Opposing wind (mirrors M3)   | 19  | 16  | 26  | 20  |

## Setup (conda)

```bash
cd fly_swatter
conda activate flyswatter   # or stepper
pip install -e .
```

## Commands

```bash
fly_swatter target-face
fly_swatter target-flies
```

**Tracking behavior**
1. Motor1 sweeps back and forth across a **145°** arc while searching
2. On lock: proportional control centers the target on the crosshair in **X and Y**
   - Motor1 → pan (dx)
   - Motor2 → tilt (dy)
3. When within **20 px**, Motors 3+4 **shoot** (0 → 5086 full-steps) then
   **reload** (5086 → 0). Camera is mounted above the turret output.

The OAK-1 is mounted **90° counter-clockwise**; frames are rotated upright
automatically before detection/display.

Quit with `q` / `Esc`.

### Useful flags

```bash
fly_swatter target-face --track-gain 0.1 --target-radius 20
fly_swatter target-face --shoot-steps 5086 --wind-delay 0.002
fly_swatter target-face --invert-pan --invert-tilt
fly_swatter target-flies --fov 55 --fov-v 69 --scan-degrees 145
```

## Layout

```
src/
  fly_detection/     # OAK camera, face/fly detectors, aim target, CLI
  turret_control/    # pan (M1), tilt (M2), opposing wind (M3/M4)
```
