# PoseDetector – Real-Time Squat Counter

PoseDetector is a lightweight computer vision project that tracks human pose landmarks with [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose) and counts squat repetitions in real time. The project provides two complementary scripts:

- `pose_tracker.py` — minimal, highly visual squat counter with an on-screen progress bar.
- `camera_test.py` — terminal-driven counter with configurable thresholds, angle smoothing and optional live preview.

Together, they are useful for prototyping fitness applications, testing camera setups, or experimenting with pose-estimation based feedback loops.

---

## Features

- Real-time pose tracking:Streams from the default webcam (`cv2.VideoCapture(0)`) and uses MediaPipe Pose for landmark detection.
- Squat detection & counting: Measures knee angles and updates a repetition counter when a full down-up cycle is detected.
- User feedback: Provides HUD overlays (`pose_tracker.py`) or terminal guidance (`camera_test.py`) to help maintain proper framing.
- Robustness helpers: Includes visibility checks, inactivity resets, angle smoothing, and configurable thresholds to reduce false positives.


---

## Requirements

- Python 3.9+ (Mac, Linux, or Windows)
- A webcam or connected video capture device
- Install dependencies:

  ```bash
  python3 -m venv .venv         # optional but recommended
  source .venv/bin/activate     # Windows: .venv\Scripts\activate
  pip install --upgrade pip
  pip install opencv-python mediapipe numpy
  ```

> macOS tip: If the camera feed fails with `CAP_AVFOUNDATION` errors, grant Terminal (or your IDE) camera access under System Settings → Privacy & Security → Camera.

---

## Project Structure

```
PoseDetector/
├── pose_tracker.py                # GUI-based squat counter with progress bar
├── camera_test.py                 # Terminal-guided counter with state machine
├── run_metrics_epoch0-299.csv     # Sample training metrics (epochs 0-299)
├── run_accuracy_curve.png         # Training accuracy visualization
├── run_loss_curve.png             # Training loss visualization
└── run_combined_accuracy_loss.png # Combined accuracy/loss plot
```

---

## Usage

 1. Visual Squat Counter (`pose_tracker.py`)

Launch the script to open a window titled "Squat Counter". The program tracks the right leg, renders skeletal landmarks, displays a vertical progress bar, and increments the counter when a full squat is detected.

```bash
python pose_tracker.py
```

Controls & behaviour:

- `q` — quit the application.
- If no skeleton is detected for >5 seconds, the counter resets (prevents stale states).
- On-screen text prompts suggest repositioning when the body leaves the frame.

### 2. Configurable Counter (`camera_test.py`)

This variant favors terminal feedback with more control over detection parameters. Adjustable constants near the top of the file influence behaviour:

```python
SHOW_VIDEO = True     # Set False to disable the OpenCV window
IMG_W, IMG_H = 640, 480
VIS_THRESH = 0.7      # Landmark visibility filter
DOWN_ANGLE = 70       # Degrees defining "bottom" of squat
UP_ANGLE = 160        # Degrees defining "top" of squat
BOTTOM_HOLD_FRAMES = 3
SMOOTH_WIN = 5        # Moving-average window for knee angle
```

Run the script:

```bash
python camera_test.py
```

What to expect:

- Terminal prints the smoothed knee angle, current stage (`idle`, `going_down`, `bottom`, `going_up`), and total reps (≈4 updates/sec).
- Additional guidance appears if hips/knees/ankles are not fully visible or if no pose is found.
- With `SHOW_VIDEO=True`, an OpenCV window overlays landmarks and a simple HUD; press `q` to quit.

---

## Customization Ideas

- Change the tracked leg: Modify indices in `pose_tracker.py` to use the left leg or averaged angles.
- Add audio cues: Integrate `playsound` or `simpleaudio` to announce completed reps or incorrect form.
- Persist metrics: Extend the scripts to log sessions (timestamps, rep counts) to CSV/JSON.
- Model experiments: Use the provided training curves and metrics as a starting point for training custom classifiers on captured pose data.

---

## Troubleshooting

- No camera feed: Ensure no other app is using the webcam and confirm permissions. On macOS, test with `QuickTime Player` or `FaceTime`.
- Frequent false positives: Increase `BOTTOM_HOLD_FRAMES`, tighten `DOWN_ANGLE`/`UP_ANGLE`, or require both legs (see logic in `camera_test.py`).
- Laggy rendering: Lower the frame size (`IMG_W`, `IMG_H`) or disable `SHOW_VIDEO`.

---

## Contributing

1. Fork the repo on GitHub.
2. Create a feature branch: `git checkout -b feature/my-idea`.
3. Commit your changes with clear messages.
4. Push and open a pull request describing the improvements.

---

## License

This project is provided as-is under the MIT License. Feel free to adapt it for personal or commercial use. See `LICENSE` (add your preferred license file if not present) for details.


