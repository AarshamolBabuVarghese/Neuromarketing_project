"""
capture_pipeline.py
---------------------
Runs one trial: shows a stimulus FULLSCREEN for N seconds while sampling
the webcam, and returns the aggregated result for that trial (dominant
emotion, average gaze point / most-looked-at AOI, attention_time).

Uses per-participant calibration (see calibration.py) if it exists, for
accurate gaze mapping. Falls back to an uncalibrated estimate with a
printed warning if the participant hasn't calibrated yet.

This module only knows about ONE trial. experiment_controller.py loops
this over a whole session (multiple stimuli, one participant).
"""
import time
import cv2
import numpy as np
import yaml
from collections import Counter

from face_detection import detect_face, crop_face
from emotion_detection import analyze_emotion
from eye_tracking import estimate_gaze, gaze_to_aoi
from calibration import load_calibration

with open("config/config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

_TRIAL_SEC = _CFG["capture"]["trial_duration_sec"]
_SAMPLE_SEC = _CFG["capture"]["sample_interval_sec"]
_CAM_INDEX = _CFG["capture"]["camera_index"]
_SCREEN_W = _CFG["screen"]["width"]
_SCREEN_H = _CFG["screen"]["height"]


def run_trial(user_id, stimulus_path=None, trial_seconds=None):
    """
    Opens the webcam, displays a stimulus image fullscreen, and samples
    emotion + gaze at a fixed interval for `trial_seconds`.

    Returns a dict:
      {
        "emotion": str, "emotion_score": float,
        "gaze_x": float, "gaze_y": float, "aoi": str,
        "attention_time": float   # seconds the participant's face was detected
      }
    """
    trial_seconds = trial_seconds or _TRIAL_SEC
    calibration = load_calibration(user_id)
    if calibration is None:
        print(f"  [warning] No calibration found for user {user_id} - "
              f"gaze coordinates will be a rough uncalibrated estimate. "
              f"Run: python -m src.calibration --user_id {user_id}")

    cap = cv2.VideoCapture(_CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera_index in config.yaml")

    stimulus_img = None
    if stimulus_path:
        stimulus_img = cv2.imread(stimulus_path)
        if stimulus_img is None:
            raise RuntimeError(f"Could not load stimulus image: {stimulus_path}")
        stimulus_img = cv2.resize(stimulus_img, (_SCREEN_W, _SCREEN_H))
        cv2.namedWindow("Stimulus", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Stimulus", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    emotions_seen, emotion_scores = [], []
    gaze_points, aois_seen = [], []
    face_detected_time = 0.0

    start = time.time()
    last_sample = 0.0

    while time.time() - start < trial_seconds:
        ok, frame = cap.read()
        if not ok:
            continue

        if stimulus_img is not None:
            cv2.imshow("Stimulus", stimulus_img)
            cv2.waitKey(1)

        now = time.time() - start
        if now - last_sample < _SAMPLE_SEC:
            continue
        last_sample = now

        box = detect_face(frame)
        if box is not None:
            face_detected_time += _SAMPLE_SEC
            crop = crop_face(frame, box)
            emotion, score = analyze_emotion(crop)
            if emotion:
                emotions_seen.append(emotion)
                emotion_scores.append(score)

        gx, gy = estimate_gaze(frame, calibration=calibration)
        if gx is not None:
            gaze_points.append((gx, gy))
            aoi = gaze_to_aoi(gx, gy)
            if aoi:
                aois_seen.append(aoi)

    cap.release()
    cv2.destroyAllWindows()

    if not emotions_seen:
        raise RuntimeError("No face detected during trial - check lighting/camera position.")

    dominant_emotion = Counter(emotions_seen).most_common(1)[0][0]
    avg_emotion_score = sum(emotion_scores) / len(emotion_scores)

    if gaze_points:
        avg_gx = float(np.mean([p[0] for p in gaze_points]))
        avg_gy = float(np.mean([p[1] for p in gaze_points]))
    else:
        avg_gx = avg_gy = None

    dominant_aoi = Counter(aois_seen).most_common(1)[0][0] if aois_seen else None

    return {
        "emotion": dominant_emotion,
        "emotion_score": avg_emotion_score,
        "gaze_x": avg_gx,
        "gaze_y": avg_gy,
        "aoi": dominant_aoi,
        "attention_time": round(face_detected_time, 6),
    }


if __name__ == "__main__":
    print(run_trial(user_id=0))