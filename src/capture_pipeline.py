"""
capture_pipeline.py
---------------------
Runs trials showing a stimulus FULLSCREEN for N seconds while sampling
the webcam, and returns the aggregated result for that trial (dominant
emotion, average gaze point / most-looked-at AOI, attention_time).

Uses per-participant calibration (see calibration.py) if it exists, for
accurate gaze mapping. Falls back to an uncalibrated estimate with a
printed warning if the participant hasn't calibrated yet.

experiment_controller.py should call open_session() ONCE at the start
of a participant's session, pass the returned cap into run_trial() for
every stimulus, then call close_session() ONCE at the very end.

FIX (this version): previously the camera AND the fullscreen window
were opened fresh and fully torn down inside run_trial() for EVERY
single stimulus. That meant the window visually disappeared between
every trial, handing focus back to whatever was behind it (e.g. VS
Code) -- which is also why on-screen alert banners were never
actually seen, and why the "topmost" flag kept failing (it was being
re-applied to a freshly recreated window each time instead of held
once). Fixed by opening the camera + window once per SESSION and
reusing them across all trials.

Also restores two earlier fixes that were lost in a prior rewrite:
  - mirror flip (a raw webcam frame is left/right-flipped relative to
    the participant's real left/right -- without this, gaze AOI
    detection silently reports the wrong side)
  - camera warm-up (a freshly opened camera's first frames are often
    under-exposed/unfocused, causing false "no face" misses)

And adds a real Windows-API foreground-force (via ctypes), since
cv2's own WND_PROP_TOPMOST flag is known to be unreliable across
OpenCV builds -- this actually pulls the window to the front instead
of just requesting it.

Two real-time data-quality alerts, rendered directly on the fullscreen
stimulus window and recorded in the returned trial dict:
  1. LOOKED AWAY -- red banner + beep if no face detected for
     _AWAY_ALERT_SEC seconds in a row.
  2. MULTIPLE FACES -- red banner + beep if 2+ people are ever seen
     in frame at once.
"""
import sys
import time
import cv2
import numpy as np
import yaml
import mediapipe as mp
from collections import Counter, defaultdict

try:
    import winsound
    def _beep():
        winsound.Beep(1000, 150)
except ImportError:
    def _beep():
        pass  # non-Windows: skip the audible alert, banner still shows

if sys.platform == "win32":
    import ctypes

    def _force_foreground(window_title="Stimulus"):
        """Actually pulls the window to the front and gives it focus,
        instead of just requesting it like cv2's topmost flag does."""
        hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.BringWindowToTop(hwnd)
else:
    def _force_foreground(window_title="Stimulus"):
        pass  # only implemented for Windows; safe no-op elsewhere

from face_detection import detect_face, crop_face
from emotion_detection import analyze_emotion_verbose
from eye_tracking import estimate_gaze, gaze_to_aoi
from calibration import load_calibration

with open("config/config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

_TRIAL_SEC = _CFG["capture"]["trial_duration_sec"]
_SAMPLE_SEC = _CFG["capture"]["sample_interval_sec"]
_CAM_INDEX = _CFG["capture"]["camera_index"]
_SCREEN_W = _CFG["screen"]["width"]
_SCREEN_H = _CFG["screen"]["height"]

_SETTLE_SEC = 0.5
_AWAY_ALERT_SEC = 0.6
_ALERT_COOLDOWN_SEC = 1.5
_WINDOW_NAME = "Stimulus"

_mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)


def _count_faces(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _mp_face_detection.process(rgb)
    return len(result.detections) if result.detections else 0


def open_session():
    """
    Call ONCE at the start of a participant's session. Opens the
    camera, warms it up, and creates one persistent fullscreen window
    reused for every stimulus -- it never closes between trials, so
    focus never drops back to another app.

    Returns a `cap` object to pass into every run_trial() call.
    """
    cap = cv2.VideoCapture(_CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera_index in config.yaml")

    for _ in range(15):
        cap.read()
        time.sleep(0.03)

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    cv2.moveWindow(_WINDOW_NAME, 0, 0)

    cv2.imshow(_WINDOW_NAME, np.zeros((_SCREEN_H, _SCREEN_W, 3), dtype="uint8"))
    cv2.waitKey(1)
    _force_foreground(_WINDOW_NAME)

    return cap


def close_session(cap):
    """Call ONCE at the very end of the session, after the last trial."""
    cap.release()
    cv2.destroyAllWindows()


def run_trial(user_id, cap, stimulus_path=None, trial_seconds=None):
    """
    Runs one trial using an ALREADY-OPEN camera + window from
    open_session(). Does NOT open or close the camera/window itself.

    Returns a dict:
      {
        "emotion": str, "emotion_score": float,
        "gaze_x": float, "gaze_y": float, "aoi": str,
        "attention_time": float,
        "looked_away_sec": float,
        "multi_face_detected": bool,
      }
    """
    trial_seconds = trial_seconds or _TRIAL_SEC
    calibration = load_calibration(user_id)
    if calibration is None:
        print(f"  [warning] No calibration found for user {user_id} - "
              f"gaze coordinates will be a rough uncalibrated estimate. "
              f"Run: python src/calibration.py --user_id {user_id}")

    stimulus_img = None
    if stimulus_path:
        stimulus_img = cv2.imread(stimulus_path)
        if stimulus_img is None:
            raise RuntimeError(f"Could not load stimulus image: {stimulus_path}")
        stimulus_img = cv2.resize(stimulus_img, (_SCREEN_W, _SCREEN_H))
        # Re-assert foreground each trial as a safety net -- cheap,
        # and covers the rare case something else steals focus
        # mid-session (e.g. a Windows notification popup).
        _force_foreground(_WINDOW_NAME)

    emotion_confidence_totals = defaultdict(float)
    emotion_scores = []
    gaze_points, aois_seen = [], []
    face_detected_time = 0.0
    looked_away_sec = 0.0
    multi_face_detected = False

    away_since = None
    last_alert_time = -999.0
    last_debug_print = -999.0

    start = time.time()
    last_sample = 0.0

    while time.time() - start < trial_seconds:
        ok, frame = cap.read()
        if not ok:
            continue

        # MIRROR FIX: a raw webcam frame is mirrored relative to the
        # participant's real left/right -- must flip before any
        # face/gaze processing, or AOI direction is silently wrong.
        frame = cv2.flip(frame, 1)

        now = time.time() - start

        box = detect_face(frame)
        active_alert = None

        if box is None:
            if away_since is None:
                away_since = now
            away_duration = now - away_since
            if away_duration >= _AWAY_ALERT_SEC:
                active_alert = "LOOK AT THE SCREEN"
        else:
            if away_since is not None:
                looked_away_sec += now - away_since
                away_since = None

        n_faces = _count_faces(frame)
        if n_faces > 1:
            multi_face_detected = True
            active_alert = "ONLY ONE PERSON SHOULD BE IN FRAME"

        if active_alert and (now - last_alert_time) >= _ALERT_COOLDOWN_SEC:
            _beep()
            last_alert_time = now

        if now - last_debug_print >= 0.5:
            status = "face OK" if box is not None else "NO FACE (away?)"
            print(f"    [{now:4.1f}s] {status}  faces_in_frame={n_faces}", end="\r")
            last_debug_print = now

        if stimulus_img is not None:
            display = stimulus_img.copy() if active_alert else stimulus_img
            if active_alert:
                cv2.rectangle(display, (0, 0), (_SCREEN_W, 60), (0, 0, 220), -1)
                cv2.putText(display, active_alert, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow(_WINDOW_NAME, display)
            cv2.waitKey(1)

        if now - last_sample < _SAMPLE_SEC:
            continue
        last_sample = now

        if box is not None:
            face_detected_time += _SAMPLE_SEC
            if now >= _SETTLE_SEC:
                crop = crop_face(frame, box)
                emotion, score, breakdown = analyze_emotion_verbose(crop)
                if emotion and breakdown:
                    for label, confidence in breakdown.items():
                        emotion_confidence_totals[label] += confidence
                    emotion_scores.append(score)

        gx, gy = estimate_gaze(frame, calibration=calibration)
        if gx is not None:
            gaze_points.append((gx, gy))
            aoi = gaze_to_aoi(gx, gy)
            if aoi:
                aois_seen.append(aoi)

    if away_since is not None:
        looked_away_sec += (time.time() - start) - away_since

    if not emotion_confidence_totals:
        raise RuntimeError("No face detected during trial - check lighting/camera position.")

    dominant_emotion = max(emotion_confidence_totals, key=emotion_confidence_totals.get)
    avg_emotion_score = sum(emotion_scores) / len(emotion_scores)

    if gaze_points:
        avg_gx = float(np.mean([p[0] for p in gaze_points]))
        avg_gy = float(np.mean([p[1] for p in gaze_points]))
    else:
        avg_gx = avg_gy = None

    dominant_aoi = Counter(aois_seen).most_common(1)[0][0] if aois_seen else None

    if multi_face_detected:
        print("  [alert] Multiple faces were detected at some point during this trial — "
              "consider re-running it with only one participant in frame.")
    if looked_away_sec > 0:
        print(f"  [alert] Participant looked away for {looked_away_sec:.1f}s during this trial.")

    return {
        "emotion": dominant_emotion,
        "emotion_score": avg_emotion_score,
        "gaze_x": avg_gx,
        "gaze_y": avg_gy,
        "aoi": dominant_aoi,
        "attention_time": round(face_detected_time, 6),
        "looked_away_sec": round(looked_away_sec, 2),
        "multi_face_detected": multi_face_detected,
    }


if __name__ == "__main__":
    session_cap = open_session()
    try:
        print(run_trial(user_id=0, cap=session_cap))
    finally:
        close_session(session_cap)