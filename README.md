
# 👀 AI Eye-Contact Monitor

A real-time computer vision app that checks whether you are **looking at the camera**, tracking your **eyes and head separately**. It also detects speaking without eye contact, leaving your seat, and hand gestures (👍 and ❤️).

Built with **Python, OpenCV and Google MediaPipe** (Face Mesh with iris tracking + Hands).

## Features
- 🟢 **Green face box** only when both your eyes and your head face the camera, 🔴 red otherwise
- **Eye direction** (LEFT / RIGHT / UP / DOWN) and **head direction** shown separately
- Yellow arrows showing where your eyeballs are pointing
- ⚠️ Warning when you **speak without eye contact**
- ⚠️ Error when you **leave the camera**, **move from your seat**, or **cover your face with your hand**
- 👍 **Thumbs up** and ❤️ **heart** gestures trigger animated pop-up symbols
- **5-step personal calibration**, saved locally

## Installation
Requires Python 3.9–3.12 and a webcam.

    git clone https://github.com/YOUR-USERNAME/eye-contact-monitor.git
    cd eye-contact-monitor
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Usage

    python eye_contact_monitor.py

On first run, follow the blue calibration bar and keep your head still:
1. Look at the camera, then press **SPACE**
2. Move your eyes to the left edge of the screen, then press **SPACE**
3. Move your eyes to the right edge, then press **SPACE**
4. Move your eyes to the top, then press **SPACE**
5. Move your eyes to the bottom, then press **SPACE**

| Key | Action |
|-----|--------|
| R | Redo calibration |
| Q | Quit |

> **macOS:** allow camera access in *System Settings → Privacy & Security → Camera*.

## Tuning
Settings are at the top of `eye_contact_monitor.py`:
- `SENSITIVITY`: eye strictness (lower = stricter)
- `HEAD_YAW_LIMIT` / `HEAD_PITCH_LIMIT`: head-turn allowance
- `POSITION_SHIFT`: how far you can move from your seat
- `ABSENT_SECONDS`: delay before the "not in front of camera" error
- `GESTURE_FRAMES`: how long a gesture must be held

## How it works
- **Eye gaze:** the iris position inside each eye, compared with your calibrated limits
- **Head pose:** the nose position relative to the face edges, forehead and chin
- **Speaking:** how much the mouth opening varies over time
- **Gestures:** MediaPipe Hands landmarks plus rules based on finger geometry

## Author
Kolapalli Lakshmi Nrusimha Kalyan.# eye-contact-monitor
