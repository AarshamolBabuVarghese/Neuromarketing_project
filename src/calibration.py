"""
calibration.py
----------------
Without calibration, eye_tracking.py's gaze estimate is just a rough
geometric guess (iris position relative to eye corners) - it varies a
lot between people depending on eye shape, camera angle, and distance
from the screen. This is the #1 accuracy fix for a real gaze tracker.

This module shows 5 dots at known screen positions, one at a time,
and records the RAW eye-ratio while the participant looks at each dot.
It then fits a simple linear mapping: raw ratio -> actual screen pixel.

Run once per participant, before their first trial:
    python src/calibration.py --user_id 5

Saves: data/processed_sessions/calibration_<user_id>.json
---------------------------------------------------------------
FIX (this version): added the same horizontal mirror flip that
capture_pipeline.py applies to real trial frames. Without this,
calibration was fit in the UNFLIPPED coordinate space while actual
trials ran in the FLIPPED space -- the two didn't agree, so gaze
mapping stayed wrong even after the capture_pipeline fix.

IMPORTANT: any calibration file saved before this fix is now stale.
Delete old files in data/processed_sessions/calibration_*.json and
re-run calibration for every participant after applying this.
---------------------------------------------------------------
"""
import argparse
import json
import os
import time
import cv2
import numpy as np
import yaml

from eye_tracking import raw_gaze_ratio

with open("config/config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

_SCREEN_W = _CFG["screen"]["width"]
_SCREEN_H = _CFG["screen"]["height"]
_CAM_INDEX = _CFG["capture"]["camera_index"]
_CALIB_DIR = "data/processed_sessions"

# 5-point calibration: four corners + center (as fractions of screen size)
_POINTS = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]
_SAMPLE_SECONDS = 1.5


def _calibration_path(user_id):
    return os.path.join(_CALIB_DIR, f"calibration_{user_id}.json")


def run_calibration(user_id):
    os.makedirs(_CALIB_DIR, exist_ok=True)
    cap = cv2.VideoCapture(_CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for calibration.")

    # Warm-up, same reasoning as capture_pipeline.py: a freshly opened
    # camera's first frames are often under-exposed/unfocused.
    for _ in range(15):
        cap.read()
        time.sleep(0.03)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    cv2.moveWindow("Calibration", 0, 0)

    print("If you don't see a red dot in a second, check your taskbar — "
          "the calibration window may have opened behind this terminal. "
          "Alt-Tab to it.\n")

    screen_pts, ratio_pts = [], []

    for point_num, (fx, fy) in enumerate(_POINTS, 1):
        sx, sy = int(fx * _SCREEN_W), int(fy * _SCREEN_H)
        print(f"Point {point_num}/{len(_POINTS)}: showing dot at ({sx}, {sy})...")
        samples = []
        start = time.time()

        while time.time() - start < _SAMPLE_SECONDS:
            ok, frame = cap.read()
            if not ok:
                continue

            # MIRROR FIX: must match capture_pipeline.py exactly, or the
            # calibration mapping and real trials are in different
            # coordinate spaces.
            frame = cv2.flip(frame, 1)

            canvas = np.zeros((_SCREEN_H, _SCREEN_W, 3), dtype="uint8")
            cv2.circle(canvas, (sx, sy), 15, (0, 0, 255), -1)
            cv2.putText(canvas, "Look at the red dot", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Calibration", canvas)
            cv2.waitKey(1)

            ratio_x, ratio_y = raw_gaze_ratio(frame)
            if ratio_x is not None:
                samples.append((ratio_x, ratio_y))

        if samples:
            avg_ratio = tuple(np.mean(samples, axis=0))
            ratio_pts.append(avg_ratio)
            screen_pts.append((sx, sy))
            print(f"  -> captured {len(samples)} samples.")
        else:
            print(f"  Warning: no eyes detected for calibration point ({sx},{sy}) - skipping.")

    cap.release()
    cv2.destroyAllWindows()

    if len(ratio_pts) < 3:
        raise RuntimeError("Not enough calibration points captured. Check lighting/webcam and retry.")

    # Fit: [screen_x, screen_y] = A . [ratio_x, ratio_y, 1]  (linear least squares)
    R = np.array([[rx, ry, 1.0] for rx, ry in ratio_pts])
    Sx = np.array([sx for sx, sy in screen_pts])
    Sy = np.array([sy for sx, sy in screen_pts])

    coef_x, *_ = np.linalg.lstsq(R, Sx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(R, Sy, rcond=None)

    calibration = {"coef_x": coef_x.tolist(), "coef_y": coef_y.tolist()}
    with open(_calibration_path(user_id), "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"Calibration saved for user {user_id} -> {_calibration_path(user_id)}")
    return calibration


def load_calibration(user_id):
    """Returns calibration dict, or None if this user hasn't calibrated yet."""
    path = _calibration_path(user_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def apply_calibration(ratio_x, ratio_y, calibration):
    """Maps a raw eye ratio to actual screen pixel coords using saved calibration."""
    cx, cy = calibration["coef_x"], calibration["coef_y"]
    screen_x = cx[0] * ratio_x + cx[1] * ratio_y + cx[2]
    screen_y = cy[0] * ratio_x + cy[1] * ratio_y + cy[2]
    return float(screen_x), float(screen_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=int, required=True)
    args = parser.parse_args()
    run_calibration(args.user_id)