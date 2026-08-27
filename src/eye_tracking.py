"""
eye_tracking.py
----------------
Estimates where on the (screen-space) stimulus the participant is
looking, using MediaPipe FaceMesh's iris landmarks (refine_landmarks=True
gives 478 points including irises).

This is a lightweight geometric gaze estimator, not a calibrated
eye-tracker: it maps iris position *relative to the eye corners* onto
screen coordinates. It's good enough to tell which AOI (Area of
Interest - e.g. "product A" vs "product B") someone is looking at,
which is exactly what the neuromarketing analysis needs.
"""
import yaml
import numpy as np
import mediapipe as mp

with open("config/config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

_SCREEN_W = _CFG["screen"]["width"]
_SCREEN_H = _CFG["screen"]["height"]
_AOIS = _CFG["screen"]["aoi"]

_mp_face_mesh = mp.solutions.face_mesh
_face_mesh = _mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark indices (MediaPipe FaceMesh w/ iris refinement)
_LEFT_EYE = [33, 133]      # left eye corner landmarks
_RIGHT_EYE = [362, 263]    # right eye corner landmarks
_LEFT_IRIS = 468
_RIGHT_IRIS = 473


def raw_gaze_ratio(frame_bgr):
    """
    Returns (ratio_x, ratio_y): normalized eye position, independent of
    screen size. ratio_x/ratio_y are in [0, 1]. Returns (None, None) if
    no face/eyes detected. This is what calibration.py samples.
    """
    h, w = frame_bgr.shape[:2]
    rgb = frame_bgr[:, :, ::-1]
    results = _face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    lm = results.multi_face_landmarks[0].landmark

    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    left_corner_l, left_corner_r = pt(_LEFT_EYE[0]), pt(_LEFT_EYE[1])
    right_corner_l, right_corner_r = pt(_RIGHT_EYE[0]), pt(_RIGHT_EYE[1])
    left_iris, right_iris = pt(_LEFT_IRIS), pt(_RIGHT_IRIS)

    def norm_pos(iris, c1, c2):
        eye_vec = c2 - c1
        denom = np.dot(eye_vec, eye_vec)
        if denom == 0:
            return 0.5
        t = np.dot(iris - c1, eye_vec) / denom
        return float(np.clip(t, 0, 1))

    ratio_x = (norm_pos(left_iris, left_corner_l, left_corner_r) +
               norm_pos(right_iris, right_corner_l, right_corner_r)) / 2

    eye_mid_y = (left_corner_l[1] + left_corner_r[1] + right_corner_l[1] + right_corner_r[1]) / 4
    iris_mid_y = (left_iris[1] + right_iris[1]) / 2
    ratio_y = float(np.clip(0.5 + (iris_mid_y - eye_mid_y) / (h * 0.05), 0, 1))

    return ratio_x, ratio_y


def estimate_gaze(frame_bgr, calibration=None):
    """
    Returns (gaze_x, gaze_y) in screen pixel coordinates, or (None, None)
    if no face/eyes are detected.

    If `calibration` is provided (see calibration.py), uses the fitted
    per-participant mapping for real accuracy. Otherwise falls back to
    a rough, uncalibrated linear guess across the full screen size -
    fine for smoke-testing the pipeline, NOT accurate enough for real
    client-facing analysis.
    """
    ratio_x, ratio_y = raw_gaze_ratio(frame_bgr)
    if ratio_x is None:
        return None, None

    if calibration is not None:
        from calibration import apply_calibration
        return apply_calibration(ratio_x, ratio_y, calibration)

    # Uncalibrated fallback
    return ratio_x * _SCREEN_W, ratio_y * _SCREEN_H


def gaze_to_aoi(gaze_x, gaze_y):
    """Maps a screen-space gaze point to one of the configured AOI labels."""
    if gaze_x is None or gaze_y is None:
        return None
    for label, (x1, y1, x2, y2) in _AOIS.items():
        if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
            return label
    return None


if __name__ == "__main__":
    import cv2
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if ok:
        gx, gy = estimate_gaze(frame)
        print("Gaze:", gx, gy, "-> AOI:", gaze_to_aoi(gx, gy))