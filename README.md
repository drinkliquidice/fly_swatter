# fly_swatter

Hack The North 2026 Automatic Fly Detection Swatter

## Hardware

- Raspberry Pi 5
- Luxonis OAK-1 camera
- 28BYJ-48 steppers + ULN2003 drivers (see `src/turret_control/motor.py`)

## Setup (conda)

This project uses the `stepper` conda environment.

```bash
cd flyswatter
conda activate 
pip install -e .
```

To create the env from scratch (or recreate it):

```bash
conda env create -f environment.yml
conda activate stepper
```

To update an existing `flyswatter` env after dependency changes:

```bash
conda activate flyswatter
pip install -e .
```

On the Pi you need a display (local HDMI or VNC) for the preview windows.
If `cv2.imshow` fails with a Qt/GTK error, install the GUI OpenCV build inside the env:

```bash
conda activate flyswatter
pip uninstall -y opencv-python-headless
pip install "opencv-python>=4.8"
```

## Vision commands

Activate the env first, then run:

```bash
conda activate flyswatter

# 1. Raw OAK-1 camera feed
fly-detect preview

# 2. Face detection with boxes drawn on the live view
fly-detect faces

# 3. On face detection, rotate the stepper 180° (uses motor.py)
fly-detect aim
```

Equivalent module form:

```bash
python -m fly_detection preview
python -m fly_detection faces
python -m fly_detection aim
```

Quit any window with `q` or `Esc`.

### Aim options

```bash
fly-detect aim --cooldown 3 --in1 17 --in2 27 --in3 22 --in4 23
fly-detect aim --no-clockwise   # opposite direction
```

`aim` triggers on a newly appearing face (rising edge) and waits `--cooldown`
seconds before it will fire again.

## Layout

```
src/
  fly_detection/     # OAK camera + face pipeline (+ fly placeholder)
  turret_control/    # stepper control
```
