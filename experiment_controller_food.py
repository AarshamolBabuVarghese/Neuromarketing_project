"""
Food Category Neuromarketing Experiment Controller (v2)
==========================================================
CHANGES FROM v1:
  - Stimuli are no longer hardcoded by filename (no more "pepsi.png" /
    "coca_cola.png" placeholders). Each category folder is scanned
    automatically for whatever images you've actually put there.
  - Feature tag per stimulus is derived from its filename automatically
    (you control the tag simply by naming the file, e.g. "spice_claim.png").
  - Each stimulus is shown for a FIXED 6 second window and auto-advances
    with zero manual action needed.
  - Live on-screen overlay shows real-time concentration % and how many
    faces are currently in frame (so 2-person-in-frame is visible instantly,
    not just after the session ends).
  - Camera opens exactly ONCE for the whole app run (single permission
    prompt), reused across every stimulus and every category.

Flow:
  1. Participant picks a food category via on-screen buttons (Tkinter).
  2. Script scans stimuli/<category_folder>/ for every image file there.
  3. Each stimulus displays for a fixed 6 sec window, auto-advancing.
  4. Live overlay + logged data: concentration ratio, face count,
     multi-face alerts, rating (1-5, optional), reaction time.
  5. Results saved to a CSV per session under results/.

Dependencies:
  pip install opencv-python==4.10.0.84 deepface mediapipe pyyaml

Folder structure expected — put YOUR OWN images here, any filenames:
  stimuli/
    chips_namkeen/    -> 5 to 7 images
    beverages/        -> 5 to 7 images
    chocolates/       -> 5 to 7 images
    biscuits/          -> 5 to 7 images
    instant_food/      -> 5 to 7 images

Run:
  python experiment_controller_food.py
"""

import cv2
import os
import csv
import time
import random
import glob
import uuid
import threading
import numpy as np
import tkinter as tk
from collections import deque
from datetime import datetime

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

STIMULI_ROOT = "stimuli"           # base folder holding category subfolders
RESULTS_DIR = "results"            # where per-session/master CSV logs are saved
DATA_DIR = "data"                  # where the project's real dataset lives

# IMPORTANT: anchor every path to THIS SCRIPT'S OWN FOLDER, not to whatever
# directory you happened to launch python from. This is what was causing a
# brand-new "data/unified_dataset_clean.csv" to get created somewhere unexpected
# instead of appending to your real, existing one.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STIMULI_ROOT = os.path.join(BASE_DIR, STIMULI_ROOT)
RESULTS_DIR = os.path.join(BASE_DIR, RESULTS_DIR)
DATA_DIR = os.path.join(BASE_DIR, DATA_DIR)
UNIFIED_DATASET_PATH = os.path.join(DATA_DIR, "unified_dataset_clean.csv")

# Same screen.width/screen.height your eye_tracking.py reads, used here to
# decide whether a CALIBRATED gaze point actually lands on the screen —
# this is what makes "looking at screen" mean something real, instead of
# just "eyes are technically visible."
SCREEN_W, SCREEN_H = 1920, 1080  # fallback if config.yaml can't be read
try:
    import yaml as _yaml
    with open(os.path.join(BASE_DIR, "config", "config.yaml"), "r") as _f:
        _cfg = _yaml.safe_load(_f)
    SCREEN_W = _cfg["screen"]["width"]
    SCREEN_H = _cfg["screen"]["height"]
except Exception as _e:
    print(f"[WARN] Could not read screen dimensions from config.yaml "
          f"({_e}) — using fallback {SCREEN_W}x{SCREEN_H}.")
SCREEN_LOOKING_MARGIN = 0.15  # allow gaze up to 15% of screen size beyond
                               # the edge before calling it "off-screen" —
                               # accounts for calibration noise near edges

# Exact column order your dashboard.py / report_generator.py already expect.
UNIFIED_COLUMNS = [
    "user_id", "emotion", "trial_id", "gaze_x", "gaze_y", "attention_time",
    "emotion_score", "engagement_score", "Product Features", "Eye direction",
]

DISPLAY_SEC = 6.0                  # fixed time per stimulus (set MIN/MAX below
                                    # instead if you want slight randomization)
USE_RANDOM_RANGE = False           # True -> use MIN/MAX_DISPLAY_SEC instead
MIN_DISPLAY_SEC = 5.0
MAX_DISPLAY_SEC = 6.0

CONCENTRATION_THRESHOLD = 0.6      # min fraction of frames w/ face detected
ALERT_COOLDOWN_SEC = 1.5           # don't spam the alert every frame
WEBCAM_INDEX = 0

MAX_ALLOWED_FACES = 1              # study is single-participant; >1 = flag it
MULTI_FACE_INVALID_THRESHOLD = 0.15  # if >15% of a trial's frames show 2+ faces,
                                      # the trial is marked unreliable

MIN_STIMULI_RECOMMENDED = 5
MAX_STIMULI_RECOMMENDED = 7
VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

DEBUG_ATTENTION = True  # prints live face-count + recent ratio to console
                         # roughly twice a second, so you can SEE what the
                         # detector is picking up instead of guessing why
                         # an alert did/didn't fire. Set False once satisfied.

ATTENTION_GRABBER_MAX_WAIT_SEC = 6.0     # cap on how long the "get ready" screen waits
ATTENTION_LOCK_FRAMES_NEEDED = 12        # consecutive frames w/ face needed to "lock in"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------------------------------------------------
# EMOTION / GAZE PLUG-IN HOOKS
# ----------------------------------------------------------------------
# unified_dataset_clean.csv needs "emotion", "gaze_x", "gaze_y", "emotion_score",
# and "Eye direction" — this script doesn't compute those itself, it calls
# your existing emotion_detection.py / eye_tracking.py modules.
#
# SPEED FIX: importing those modules pulls in tensorflow + deepface +
# mediapipe, which can take 10-20+ seconds. Doing that at the top of the
# file (like before) freezes the WHOLE script before anything appears on
# screen. Instead, these are loaded lazily in a background thread kicked
# off at program start (see _load_heavy_modules / MODULES_READY below),
# overlapping with human-paced steps (typing your ID, picking a category,
# the "Get Ready" screen) so there's no dead frozen time.
import sys

EMOTION_MODULE_AVAILABLE = False
GAZE_MODULE_AVAILABLE = False
FACE_DETECTOR_AVAILABLE = False
CALIBRATION_MODULE_AVAILABLE = False
_analyze_emotion = None
_estimate_gaze = None
_gaze_to_aoi = None
_mp_face_detector = None
_run_calibration = None
_load_calibration = None
CURRENT_CALIBRATION = None  # set once per participant via ensure_calibration()
MODULES_READY = threading.Event()


def _load_heavy_modules():
    """Runs in a background thread. Imports the heavy analysis modules and
    warms them up with a dummy frame, then signals MODULES_READY."""
    global EMOTION_MODULE_AVAILABLE, GAZE_MODULE_AVAILABLE, FACE_DETECTOR_AVAILABLE
    global CALIBRATION_MODULE_AVAILABLE
    global _analyze_emotion, _estimate_gaze, _gaze_to_aoi, _mp_face_detector
    global _run_calibration, _load_calibration

    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    sys.path.insert(0, BASE_DIR)

    try:
        from emotion_detection import analyze_emotion as _ae
        _analyze_emotion = _ae
        EMOTION_MODULE_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] emotion_detection module not available: {e}")

    try:
        from eye_tracking import estimate_gaze as _eg, gaze_to_aoi as _gta
        _estimate_gaze = _eg
        _gaze_to_aoi = _gta
        GAZE_MODULE_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] eye_tracking module not available: {e}")

    try:
        from calibration import run_calibration as _rc, load_calibration as _lc
        _run_calibration = _rc
        _load_calibration = _lc
        CALIBRATION_MODULE_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] calibration module not available: {e}")

    # ACCURACY FIX: Haar cascade is prone to false positives (background
    # patterns getting mistaken for faces) and misses angled/turned faces
    # too easily. MediaPipe's Face Detection model is far more reliable
    # for BOTH "is anyone actually here" and "are there 2+ people" — and
    # it natively supports counting multiple faces, unlike the eye_tracking
    # module's FaceMesh (which is locked to max_num_faces=1 for gaze work).
    try:
        import mediapipe as mp
        _mp_face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.6
        )
        FACE_DETECTOR_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] mediapipe face detector not available: {e}")

    # Warm up (forces model weights to load now, in the background, instead
    # of during whichever trial happens to be first).
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    if EMOTION_MODULE_AVAILABLE:
        try:
            get_emotion_reading(dummy)
        except Exception:
            pass
    if GAZE_MODULE_AVAILABLE:
        try:
            get_gaze_reading(dummy)
        except Exception:
            pass
    if FACE_DETECTOR_AVAILABLE:
        try:
            count_faces_accurate(dummy)
        except Exception:
            pass

    print("\n===== MODULE DIAGNOSTIC (background load finished) =====")
    print(f"Emotion module (DeepFace) loaded:        {EMOTION_MODULE_AVAILABLE}")
    print(f"Gaze module (MediaPipe FaceMesh) loaded: {GAZE_MODULE_AVAILABLE}")
    print(f"Face counter (MediaPipe) loaded:         {FACE_DETECTOR_AVAILABLE}")
    if not (EMOTION_MODULE_AVAILABLE and GAZE_MODULE_AVAILABLE and FACE_DETECTOR_AVAILABLE):
        print("Scroll up to the [WARN] line above for the exact import error.")
    print("==========================================================\n")

    MODULES_READY.set()


def count_faces_accurate(frame):
    """Returns the number of faces in frame using MediaPipe's Face
    Detection model — much lower false-positive rate than Haar cascade,
    and correctly counts multiple people. Falls back to the Haar cascade
    only if MediaPipe's detector isn't loaded yet."""
    if FACE_DETECTOR_AVAILABLE and _mp_face_detector is not None:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = _mp_face_detector.process(rgb)
            return len(results.detections) if results.detections else 0
        except Exception:
            pass
    return len(detect_faces(frame))  # Haar fallback (only during load window)


def is_looking_at_screen(frame):
    """Returns True only if the CALIBRATED gaze point actually falls
    within the screen's bounds (with a small margin for noise).

    IMPORTANT: this is deliberately NOT the same as "are eyes detected."
    MediaPipe can track your irises perfectly well even while you're
    looking far to the side — that's the whole point of gaze tracking.
    So checking mere trackability was the bug: it stayed True regardless
    of where you were actually looking. This version uses the real
    calibrated (gx, gy) vs. actual screen size to answer the right
    question — is the estimated gaze point on the screen or not."""
    if not GAZE_MODULE_AVAILABLE:
        return True  # can't judge yet (still loading) — don't false-alarm
    try:
        gx, gy = _estimate_gaze(frame, calibration=CURRENT_CALIBRATION)
        if gx is None:
            return False  # eyes not trackable at all = definitely not looking
        margin_x = SCREEN_W * SCREEN_LOOKING_MARGIN
        margin_y = SCREEN_H * SCREEN_LOOKING_MARGIN
        on_screen = (-margin_x <= gx <= SCREEN_W + margin_x and
                     -margin_y <= gy <= SCREEN_H + margin_y)
        return on_screen
    except Exception:
        return True  # analysis hiccup shouldn't itself trigger an alert


# ----------------------------------------------------------------------
# PRECISE DISTURBANCE CLASSIFIER
# ----------------------------------------------------------------------
# Instead of one generic "low concentration" alert, this tells you the
# EXACT cause each time, in priority order. Thresholds are tunable below.
LOW_LIGHT_THRESHOLD = 60        # mean grayscale brightness (0-255); below = too dark
MIN_FACE_WIDTH_RATIO = 0.12     # face narrower than 12% of frame width = too far
EDGE_MARGIN_RATIO = 0.04        # face within 4% of any edge = not centered

DISTURBANCE_MESSAGES = {
    "MULTIPLE_FACES": "Multiple people detected — only one participant allowed.",
    "NO_FACE": "No one detected — please sit in front of the camera.",
    "POOR_LIGHTING": "Lighting is too dark for reliable detection — please add light.",
    "TOO_FAR": "You're too far from the camera — please move closer.",
    "OFF_CENTER": "Please center yourself in the camera frame.",
    "NOT_LOOKING": "Please look at the screen.",
    "OK": None,
}


def classify_disturbance(frame, num_faces, face_boxes):
    """Returns one specific disturbance code (see DISTURBANCE_MESSAGES)
    instead of a single vague 'low concentration' flag, checked in
    priority order — highest-priority real problem wins."""
    if num_faces > MAX_ALLOWED_FACES:
        return "MULTIPLE_FACES"
    if num_faces == 0:
        return "NO_FACE"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < LOW_LIGHT_THRESHOLD:
        return "POOR_LIGHTING"

    if face_boxes:
        x, y, fw, fh = face_boxes[0]
        h, w = frame.shape[:2]
        if fw / w < MIN_FACE_WIDTH_RATIO:
            return "TOO_FAR"
        margin_x, margin_y = w * EDGE_MARGIN_RATIO, h * EDGE_MARGIN_RATIO
        if x < margin_x or (x + fw) > (w - margin_x) or y < margin_y or (y + fh) > (h - margin_y):
            return "OFF_CENTER"

    if not is_looking_at_screen(frame):
        return "NOT_LOOKING"

    return "OK"


def get_face_boxes_accurate(frame):
    """Returns face bounding boxes as (x, y, w, h) in pixel coordinates,
    using the same accurate MediaPipe detector as count_faces_accurate —
    not the old Haar cascade, which was silently causing missed crops."""
    if FACE_DETECTOR_AVAILABLE and _mp_face_detector is not None:
        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = _mp_face_detector.process(rgb)
            boxes = []
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    bw = min(int(bbox.width * w), w - x)
                    bh = min(int(bbox.height * h), h - y)
                    if bw > 0 and bh > 0:
                        boxes.append((x, y, bw, bh))
            return boxes
        except Exception as e:
            print(f"[WARN] accurate face box detection failed: {e}")
    return list(detect_faces(frame))  # Haar fallback only if MediaPipe unavailable


def get_emotion_reading(frame):
    """Detects a face, crops it, and runs your real DeepFace-based
    analyze_emotion() from emotion_detection.py. Every failure path now
    prints WHY, instead of silently returning NA."""
    if not EMOTION_MODULE_AVAILABLE:
        print("[WARN] emotion skipped: EMOTION_MODULE_AVAILABLE is False")
        return "NA", "NA"
    faces = get_face_boxes_accurate(frame)
    if len(faces) == 0:
        print("[WARN] emotion skipped: no face detected in this frame")
        return "NA", "NA"
    x, y, w, h = faces[0]  # first/largest detected face in this frame
    face_crop = frame[y:y + h, x:x + w]
    if face_crop.size == 0:
        print("[WARN] emotion skipped: face crop was empty")
        return "NA", "NA"
    try:
        emotion, score = _analyze_emotion(face_crop)
        if emotion is None:
            print("[WARN] emotion skipped: analyze_emotion returned None "
                  "(check emotion_detection.py's own error print above)")
            return "NA", "NA"
        return emotion, score
    except Exception as e:
        print(f"[WARN] emotion analysis failed: {e}")
        return "NA", "NA"


def get_gaze_reading(frame):
    """Uses your real estimate_gaze()/gaze_to_aoi() from eye_tracking.py,
    now WITH the per-participant calibration profile (CURRENT_CALIBRATION,
    set once via ensure_calibration() at session start) when available —
    this is what actually makes gaze_x/gaze_y/Eye direction meaningful,
    instead of the rough uncalibrated geometric guess."""
    if not GAZE_MODULE_AVAILABLE:
        print("[WARN] gaze skipped: GAZE_MODULE_AVAILABLE is False")
        return "NA", "NA", "NA"
    try:
        gx, gy = _estimate_gaze(frame, calibration=CURRENT_CALIBRATION)
        if gx is None:
            print("[WARN] gaze skipped: estimate_gaze found no trackable "
                  "eyes in this frame (face turned away, poor lighting, or "
                  "out of frame)")
            return "NA", "NA", "NA"
        aoi = _gaze_to_aoi(gx, gy)
        return round(gx, 2), round(gy, 2), (aoi if aoi else "NA")
    except Exception as e:
        print(f"[WARN] gaze estimation failed: {e}")
        return "NA", "NA", "NA"


def get_next_trial_id():
    """Continues trial_id numbering from wherever unified_dataset_clean.csv left
    off, instead of restarting at 1 and colliding with existing rows."""
    if not os.path.isfile(UNIFIED_DATASET_PATH):
        return 1
    max_id = 0
    try:
        with open(UNIFIED_DATASET_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    max_id = max(max_id, int(row.get("trial_id", 0)))
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return max_id + 1


def prompt_user_id():
    """Asks once at startup, matching your original main.py behavior."""
    root = tk.Tk()
    root.withdraw()
    entry_result = {"value": None}

    dialog = tk.Toplevel(root)
    dialog.title("Participant ID")
    dialog.geometry("320x140")
    tk.Label(dialog, text="Enter participant / user ID:", font=("Segoe UI", 11)).pack(pady=10)
    entry = tk.Entry(dialog, font=("Segoe UI", 11), width=20)
    entry.pack(pady=5)
    entry.focus()

    def submit():
        entry_result["value"] = entry.get().strip()
        dialog.destroy()
        root.destroy()

    tk.Button(dialog, text="Start", command=submit, width=12).pack(pady=10)
    dialog.protocol("WM_DELETE_WINDOW", submit)
    root.wait_window(dialog)

    return entry_result["value"] or f"anon_{uuid.uuid4().hex[:6]}"


def ensure_calibration(user_id, cap):
    """Loads this participant's saved calibration if it exists; otherwise
    runs the interactive 5-point calibration from calibration.py.

    IMPORTANT: calibration.py opens its OWN cv2.VideoCapture handle, so we
    must release our persistent `cap` first (or two processes would fight
    over the same camera device), then reopen a fresh one afterward.
    Returns the (possibly new) cap object — always use its return value.
    """
    global CURRENT_CALIBRATION

    if not CALIBRATION_MODULE_AVAILABLE:
        print("[NOTE] Calibration module not loaded yet/unavailable — "
              "gaze will use the uncalibrated fallback for this session.")
        return cap

    existing = _load_calibration(user_id)
    if existing:
        CURRENT_CALIBRATION = existing
        print(f"[INFO] Loaded existing calibration for user {user_id}.")
        return cap

    print(f"[INFO] No calibration on file for user {user_id} — running "
          f"5-point calibration now. Look at each red dot as it appears.")
    cap.release()
    cv2.destroyAllWindows()
    try:
        CURRENT_CALIBRATION = _run_calibration(user_id)
    except Exception as e:
        print(f"[WARN] Calibration failed ({e}) — continuing with "
              f"uncalibrated gaze for this session.")
        CURRENT_CALIBRATION = None

    new_cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not new_cap.isOpened():
        print("[ERROR] Could not reopen webcam after calibration.")
    return new_cap


# category_key -> (display_name, subfolder). No stimuli list here anymore —
# images are auto-discovered from the folder at run time.
CATEGORY_CONFIG = {
    "1": {"name": "Chips & Namkeen", "folder": "chips_namkeen"},
    "2": {"name": "Beverages", "folder": "beverages"},
    "3": {"name": "Chocolates & Confectionery", "folder": "chocolates"},
    "4": {"name": "Biscuits & Cookies", "folder": "biscuits"},
    "5": {"name": "Instant / Ready-to-Eat", "folder": "instant_food"},
}



# ----------------------------------------------------------------------

def discover_stimuli(category_key):
    """Scans stimuli/<folder>/ and returns [(filepath, feature_tag), ...]
    for every image found. feature_tag = filename without extension."""
    cfg = CATEGORY_CONFIG[category_key]
    folder = os.path.join(STIMULI_ROOT, cfg["folder"])

    if not os.path.isdir(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return []

    files = []
    for ext in VALID_IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    files.sort()

    if not files:
        print(f"[WARN] No images found in {folder}. Add some images there first.")
        return []

    if len(files) < MIN_STIMULI_RECOMMENDED:
        print(f"[NOTE] Only {len(files)} images found in {cfg['name']} "
              f"(recommended {MIN_STIMULI_RECOMMENDED}-{MAX_STIMULI_RECOMMENDED}). Proceeding anyway.")
    elif len(files) > MAX_STIMULI_RECOMMENDED:
        print(f"[NOTE] {len(files)} images found in {cfg['name']} "
              f"(recommended {MIN_STIMULI_RECOMMENDED}-{MAX_STIMULI_RECOMMENDED}). Using all of them.")

    stimuli = []
    for filepath in files:
        feature_tag = os.path.splitext(os.path.basename(filepath))[0]
        stimuli.append((filepath, feature_tag))
    return stimuli


# ----------------------------------------------------------------------
# CATEGORY SELECTION UI (Tkinter buttons)
# ----------------------------------------------------------------------

def selection_gui():
    """Shows category choice buttons. Returns the chosen category_key or None."""
    chosen = {"key": None}

    root = tk.Tk()
    root.title("Neuromarketing Study — Choose a Food Category")
    root.geometry("420x360")

    tk.Label(
        root, text="Which category would you like to be tested on?",
        font=("Segoe UI", 13, "bold"), wraplength=380, pady=15
    ).pack()

    def pick(key):
        chosen["key"] = key
        root.destroy()

    for key, cfg in CATEGORY_CONFIG.items():
        n_images = len(discover_stimuli(key))
        label = f"{cfg['name']}  ({n_images} stimuli)"
        tk.Button(
            root, text=label, font=("Segoe UI", 11),
            width=32, height=2, command=lambda k=key: pick(k)
        ).pack(pady=6)

    root.mainloop()
    return chosen["key"]


# ----------------------------------------------------------------------
# ATTENTION / CONCENTRATION CHECK
# ----------------------------------------------------------------------

def detect_faces(frame):
    """Returns the list of detected face bounding boxes in this frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
    return faces  # list of (x, y, w, h); len() tells you how many people


def _beep():
    """Best-effort attention beep. Silently does nothing on non-Windows
    systems or if audio isn't available — never blocks the study."""
    try:
        import winsound
        winsound.Beep(1000, 200)
    except Exception:
        pass


def show_attention_grabber(cap, category_name):
    """Pre-stimulus screen shown ONCE at the start of a category run.
    Flashes/beeps to get the participant's attention and waits until their
    face is reliably detected before the actual stimuli begin. Gives up and
    proceeds anyway after ATTENTION_GRABBER_MAX_WAIT_SEC so it never hangs
    the study if the camera angle is imperfect."""
    window_name = "Get Ready"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 700)

    start = time.time()
    consecutive_face_frames = 0
    attention_locked = False
    _beep()

    while time.time() - start < ATTENTION_GRABBER_MAX_WAIT_SEC:
        canvas = np.full((700, 900, 3), 25, dtype=np.uint8)  # dark background

        ok, cam_frame = cap.read()
        if ok:
            num_faces = count_faces_accurate(cam_frame)
            if num_faces >= 1:
                consecutive_face_frames += 1
            else:
                consecutive_face_frames = 0

            if consecutive_face_frames >= ATTENTION_LOCK_FRAMES_NEEDED:
                attention_locked = True
                break

        # Flashing "GET READY" text (blinks every ~0.4s)
        blink_on = int((time.time() - start) * 2.5) % 2 == 0
        if blink_on:
            cv2.putText(canvas, "GET READY", (230, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 140, 255), 4)
        cv2.putText(canvas, f"Starting {category_name} study", (150, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "Please look directly at the camera", (140, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("Session aborted by participant.")

    # Brief confirmation flash before stimuli start
    canvas = np.full((700, 900, 3), 25, dtype=np.uint8)
    msg = "Attention confirmed - starting now" if attention_locked else "Starting now"
    color = (0, 200, 0) if attention_locked else (0, 165, 255)
    cv2.putText(canvas, msg, (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(500)
    cv2.destroyWindow(window_name)

    if not attention_locked:
        print(f"[NOTE] Attention lock not confirmed within "
              f"{ATTENTION_GRABBER_MAX_WAIT_SEC}s — starting stimuli anyway.")


# ----------------------------------------------------------------------
# SINGLE STIMULUS TRIAL
# ----------------------------------------------------------------------

def run_trial(cap, stim_path, feature_tag, category_name, user_id, trial_id):
    """
    Displays one stimulus, runs the timed window, tracks concentration and
    face count live, captures rating + reaction time. Returns a results dict
    that includes both the detailed QA metrics AND the fields needed to
    build a unified_dataset_clean.csv row.
    """
    img = cv2.imread(stim_path)
    if img is None:
        print(f"[WARN] Could not read image, skipping: {stim_path}")
        return None

    display_duration = (
        random.uniform(MIN_DISPLAY_SEC, MAX_DISPLAY_SEC)
        if USE_RANDOM_RANGE else DISPLAY_SEC
    )
    window_name = "Stimulus"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 700)

    start_time = time.time()
    rating = None
    reaction_time_ms = None
    frame_count = 0
    attentive_count = 0          # frames with exactly 1 face AND looking
    multi_face_count = 0         # frames with 2+ faces
    last_alert_time = {}         # per-disturbance-code cooldown tracker
    disturbance_counts = {code: 0 for code in DISTURBANCE_MESSAGES}
    show_alert_until = 0
    current_alert_message = None
    current_num_faces = 0
    is_looking = True
    last_cam_frame = None
    last_detect_time = 0.0
    DETECT_INTERVAL_SEC = 0.15  # ~6-7 checks/sec — plenty for a concentration
                                 # ratio over a 6s window, far cheaper than
                                 # running face detection on every single frame

    # RECENT-WINDOW tracker for the alert (separate from the cumulative
    # concentration_ratio used in the CSV/summary, which is fine as a
    # whole-trial average). Using only the last ~2 seconds means an issue
    # is caught quickly, instead of getting diluted by a trial-long average.
    RECENT_WINDOW_SEC = 2.0
    RECENT_WINDOW_SAMPLES = max(3, int(RECENT_WINDOW_SEC / DETECT_INTERVAL_SEC))
    MIN_SAMPLES_FOR_ALERT = 3
    recent_attentive = deque(maxlen=RECENT_WINDOW_SAMPLES)
    last_debug_print = 0.0

    while True:
        elapsed = time.time() - start_time
        remaining = display_duration - elapsed
        if remaining <= 0:
            break

        ok, cam_frame = cap.read()
        if ok:
            last_cam_frame = cam_frame
            now = time.time()
            if now - last_detect_time >= DETECT_INTERVAL_SEC:
                last_detect_time = now
                frame_count += 1

                # ACCURATE face count + boxes (MediaPipe)
                current_num_faces = count_faces_accurate(cam_frame)
                face_boxes = get_face_boxes_accurate(cam_frame) if current_num_faces else []

                is_looking = (is_looking_at_screen(cam_frame)
                              if current_num_faces == MAX_ALLOWED_FACES else False)

                attentive_this_check = (current_num_faces == MAX_ALLOWED_FACES
                                         and is_looking)
                if attentive_this_check:
                    attentive_count += 1
                if current_num_faces > MAX_ALLOWED_FACES:
                    multi_face_count += 1

                recent_attentive.append(1 if attentive_this_check else 0)

                # ---- PRECISE disturbance classification ----
                code = classify_disturbance(cam_frame, current_num_faces, face_boxes)
                disturbance_counts[code] += 1

                if DEBUG_ATTENTION and time.time() - last_debug_print > 0.5:
                    last_debug_print = time.time()
                    dbg_gx, dbg_gy = "NA", "NA"
                    if GAZE_MODULE_AVAILABLE and current_num_faces == MAX_ALLOWED_FACES:
                        try:
                            _gx, _gy = _estimate_gaze(cam_frame, calibration=CURRENT_CALIBRATION)
                            if _gx is not None:
                                dbg_gx, dbg_gy = round(_gx, 0), round(_gy, 0)
                        except Exception:
                            pass
                    print(f"[DEBUG] faces={current_num_faces}  looking={is_looking}  "
                          f"gaze=({dbg_gx},{dbg_gy})  screen={SCREEN_W}x{SCREEN_H}  "
                          f"disturbance={code}  samples={len(recent_attentive)}")

                recent_ratio = (sum(recent_attentive) / len(recent_attentive)
                                 if recent_attentive else 1.0)

                # Only alert once a disturbance has been consistent for a
                # few checks (avoids flicker from one bad frame) — except
                # MULTIPLE_FACES, which alerts immediately since it's
                # unambiguous the instant it's seen.
                should_alert = (
                    code != "OK" and
                    (code == "MULTIPLE_FACES" or
                     (len(recent_attentive) >= MIN_SAMPLES_FOR_ALERT and recent_ratio < CONCENTRATION_THRESHOLD))
                )
                cooldown_ok = time.time() - last_alert_time.get(code, 0) > ALERT_COOLDOWN_SEC

                if should_alert and cooldown_ok:
                    last_alert_time[code] = time.time()
                    current_alert_message = DISTURBANCE_MESSAGES[code]
                    show_alert_until = time.time() + 1.5
                    print(f"[ALERT:{code}] '{feature_tag}' — {current_alert_message}")
                elif code == "OK":
                    current_alert_message = None

        # ---- Build display frame with LIVE overlay ----
        display_frame = img.copy()
        h, w = display_frame.shape[:2]
        live_ratio = (sum(recent_attentive) / len(recent_attentive)) if recent_attentive else 1.0

        cv2.putText(display_frame, f"{category_name}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
        cv2.putText(display_frame, f"Time left: {remaining:0.1f}s", (20, h - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Concentration: {live_ratio*100:0.0f}%   Faces: {current_num_faces}   Looking: {'Yes' if is_looking else 'No'}",
                    (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        cv2.putText(display_frame, "Rate 1(low)-5(high) purchase interest",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if current_alert_message and time.time() < show_alert_until:
            cv2.putText(display_frame, current_alert_message.upper(),
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if rating is None and key in [ord(str(n)) for n in range(1, 6)]:
            rating = int(chr(key))
            reaction_time_ms = round((time.time() - start_time) * 1000, 1)

        if key == 27:  # ESC to abort whole session
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("Session aborted by participant.")

    cv2.destroyWindow(window_name)

    concentration_ratio = round(attentive_count / frame_count, 3) if frame_count else 0.0
    multi_face_ratio = round(multi_face_count / frame_count, 3) if frame_count else 0.0
    valid_response = rating is not None
    reliable_trial = multi_face_ratio <= MULTI_FACE_INVALID_THRESHOLD
    attention_time = round(concentration_ratio * display_duration, 2)
    engagement_score = round(concentration_ratio * 10, 2)  # simple 0-10 proxy;
    # replace with your project's real engagement formula if it differs.

    # ---- SPEED FIX ----
    # DeepFace/MediaPipe analysis takes 1-3+ seconds. Running it here would
    # block the NEXT stimulus from appearing, adding a visible dead pause
    # after every image. Instead, kick it off in a background thread and
    # let the loop move on immediately — results get filled in and joined
    # later, right before saving, so nothing is lost, it's just no longer
    # blocking the on-screen pacing.
    analysis_result = {"emotion": "NA", "emotion_score": "NA",
                        "gaze_x": "NA", "gaze_y": "NA", "eye_direction": "NA"}

    def _analyze_in_background(frame, out):
        if frame is None:
            return
        # If the heavy modules are still loading (e.g. this is the very
        # first trial and the app just started), wait for them here — this
        # is a background thread, so waiting doesn't freeze the display.
        MODULES_READY.wait(timeout=30)
        e, es = get_emotion_reading(frame)
        gx, gy, ed = get_gaze_reading(frame)
        out["emotion"], out["emotion_score"] = e, es
        out["gaze_x"], out["gaze_y"], out["eye_direction"] = gx, gy, ed

    analysis_thread = threading.Thread(
        target=_analyze_in_background, args=(last_cam_frame, analysis_result), daemon=True
    )
    analysis_thread.start()

    return {
        # --- detailed QA metrics (kept in results/ CSVs for your own review) ---
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "category": category_name,
        "stimulus": os.path.basename(stim_path),
        "feature_tag": feature_tag,
        "rating_1to5": rating if rating is not None else "",
        "reaction_time_ms": reaction_time_ms if reaction_time_ms is not None else "",
        "concentration_ratio": concentration_ratio,
        "multi_face_ratio": multi_face_ratio,
        "checks_no_face": disturbance_counts["NO_FACE"],
        "checks_multiple_faces": disturbance_counts["MULTIPLE_FACES"],
        "checks_poor_lighting": disturbance_counts["POOR_LIGHTING"],
        "checks_too_far": disturbance_counts["TOO_FAR"],
        "checks_off_center": disturbance_counts["OFF_CENTER"],
        "checks_not_looking": disturbance_counts["NOT_LOOKING"],
        "checks_ok": disturbance_counts["OK"],
        "alert_triggered": frame_count > 0 and disturbance_counts["OK"] < frame_count,
        "multi_face_alert_triggered": disturbance_counts["MULTIPLE_FACES"] > 0,
        "reliable_trial": reliable_trial,
        "valid_response": valid_response,
        "display_duration_sec": round(display_duration, 2),
        # --- fields mapped directly into unified_dataset_clean.csv ---
        "user_id": user_id,
        "trial_id": trial_id,
        "attention_time": attention_time,
        "engagement_score": engagement_score,
        "Product Features": feature_tag,
        # --- these get filled in once the background thread finishes ---
        "_analysis_thread": analysis_thread,
        "_analysis_result": analysis_result,
    }


# ----------------------------------------------------------------------
# RUN FULL EXPERIMENT FOR A CATEGORY
# ----------------------------------------------------------------------

def run_experiment(category_key, cap, user_id):
    """cap: an already-open cv2.VideoCapture, reused across the whole app run
    so the OS camera-permission prompt only ever appears once."""
    cfg = CATEGORY_CONFIG[category_key]
    category_name = cfg["name"]

    stimuli_order = discover_stimuli(category_key)
    if not stimuli_order:
        print(f"[ERROR] No stimuli available for {category_name}. Add images to "
              f"stimuli/{cfg['folder']}/ and try again.")
        return
    random.shuffle(stimuli_order)  # randomize presentation order per participant

    # Attention-grabber screen runs ONCE, before the first stimulus of this
    # category run — not before every individual stimulus.
    show_attention_grabber(cap, category_name)

    next_trial_id = get_next_trial_id()  # continues from unified_dataset_clean.csv's last id

    results = []
    try:
        for stim_path, feature_tag in stimuli_order:
            trial_result = run_trial(cap, stim_path, feature_tag, category_name,
                                      user_id, next_trial_id)
            if trial_result:
                results.append(trial_result)
                next_trial_id += 1
                flag = "" if trial_result["reliable_trial"] else "  [UNRELIABLE: multiple faces]"
                print(f"  -> {trial_result['stimulus']} | rating={trial_result['rating_1to5']} "
                      f"| RT={trial_result['reaction_time_ms']}ms "
                      f"| concentration={trial_result['concentration_ratio']}{flag}")
    except KeyboardInterrupt as e:
        print(str(e))
        raise

    if results:
        # All stimuli have finished displaying by now — join each trial's
        # background emotion/gaze thread (should already be done or very
        # close to it, since they've had the whole rest of the category
        # run to finish) and merge the results in before saving.
        print("Finalizing emotion/gaze analysis...")
        for trial_result in results:
            thread = trial_result.pop("_analysis_thread", None)
            analysis = trial_result.pop("_analysis_result", {})
            if thread is not None:
                thread.join(timeout=15)  # safety cap, shouldn't normally wait this long
            trial_result["emotion"] = analysis.get("emotion", "NA")
            trial_result["emotion_score"] = analysis.get("emotion_score", "NA")
            trial_result["gaze_x"] = analysis.get("gaze_x", "NA")
            trial_result["gaze_y"] = analysis.get("gaze_y", "NA")
            trial_result["Eye direction"] = analysis.get("eye_direction", "NA")

        save_results(category_name, results)
        append_to_unified_dataset(results)
        print_summary(results)
    else:
        print("No trials completed — nothing saved.")


# ----------------------------------------------------------------------
# STORAGE
# ----------------------------------------------------------------------

def save_results(category_name, results):
    fname = f"{category_name.replace(' ', '_').replace('&', 'and')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join(RESULTS_DIR, fname)
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[SAVED] Session file: {path}")


def append_to_unified_dataset(results):
    """Appends every trial from this run DIRECTLY into data/unified_dataset_clean.csv
    using the exact column schema your dashboard.py / report_generator.py
    already expect. This is what makes the dashboard and report generation
    pick up new sessions automatically — no manual step needed."""
    file_exists = os.path.isfile(UNIFIED_DATASET_PATH)

    rows = [{col: r.get(col, "NA") for col in UNIFIED_COLUMNS} for r in results]

    with open(UNIFIED_DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVED] Appended {len(rows)} rows directly to {UNIFIED_DATASET_PATH}")


# ----------------------------------------------------------------------
# QUICK SUMMARY (feeds into your existing analysis.py / dashboard.py)
# ----------------------------------------------------------------------

def print_summary(results):
    valid = [r for r in results if r["valid_response"]]
    n = len(results)
    n_valid = len(valid)
    avg_rt = (sum(r["reaction_time_ms"] for r in valid) / n_valid) if n_valid else None
    avg_concentration = sum(r["concentration_ratio"] for r in results) / n
    n_unreliable = sum(1 for r in results if not r["reliable_trial"])

    total_no_face = sum(r["checks_no_face"] for r in results)
    total_multiface = sum(r["checks_multiple_faces"] for r in results)
    total_lighting = sum(r["checks_poor_lighting"] for r in results)
    total_too_far = sum(r["checks_too_far"] for r in results)
    total_off_center = sum(r["checks_off_center"] for r in results)
    total_not_looking = sum(r["checks_not_looking"] for r in results)
    total_ok = sum(r["checks_ok"] for r in results)
    total_checks = total_no_face + total_multiface + total_lighting + \
        total_too_far + total_off_center + total_not_looking + total_ok

    print("\n----- SESSION SUMMARY -----")
    print(f"Trials completed:        {n}")
    print(f"Valid responses:         {n_valid} ({n_valid/n*100:.0f}% response accuracy)")
    print(f"Avg reaction time:       {avg_rt:.0f} ms" if avg_rt else "Avg reaction time: N/A")
    print(f"Avg concentration ratio: {avg_concentration:.2f}")
    print(f"Unreliable trials (2+ people in frame too often): {n_unreliable}/{n}")
    print("\nDisturbance breakdown (out of {} checks):".format(total_checks))
    print(f"  OK (attentive):        {total_ok}")
    print(f"  No one detected:       {total_no_face}")
    print(f"  Multiple faces:        {total_multiface}")
    print(f"  Poor lighting:         {total_lighting}")
    print(f"  Too far from camera:   {total_too_far}")
    print(f"  Off-center in frame:   {total_off_center}")
    print(f"  Not looking at screen: {total_not_looking}")
    if n_unreliable:
        print("  -> Re-run this participant alone; those trials' scores are not trustworthy.")
    print("----------------------------\n")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[INFO] Writing to unified dataset at: {UNIFIED_DATASET_PATH}")
    if not os.path.isfile(UNIFIED_DATASET_PATH):
        print("[WARN] That file doesn't exist yet at this exact path — "
              "double check this matches your real data/unified_dataset_clean.csv "
              "before proceeding, or existing rows won't be found/continued.")

    # SPEED FIX: kick off the heavy tensorflow/deepface/mediapipe import +
    # warm-up RIGHT NOW, in the background — before the webcam even opens,
    # before the participant ID prompt, before category selection. All of
    # those human-paced steps happen in parallel with this load instead of
    # waiting for it up front. Its own diagnostic block prints whenever it
    # finishes, which will likely be well before the first real trial.
    print("[INFO] Loading AI models in the background (this can take 10-20s "
          "the first time) — you can go ahead and enter your ID / pick a "
          "category while this finishes.\n")
    threading.Thread(target=_load_heavy_modules, daemon=True).start()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Check camera permissions/index.")
    else:
        user_id = prompt_user_id()
        print(f"Participant ID: {user_id}")

        if not MODULES_READY.is_set():
            print("[INFO] Waiting for AI models to finish loading before calibration...")
            MODULES_READY.wait(timeout=30)

        cap = ensure_calibration(user_id, cap)

        try:
            while True:
                category_key = selection_gui()
                if category_key is None:
                    print("No category selected. Exiting.")
                    break
                print(f"Starting experiment for: {CATEGORY_CONFIG[category_key]['name']}")
                run_experiment(category_key, cap, user_id)

                again = input("\nRun another category? (y/n): ").strip().lower()
                if again != "y":
                    break
        except KeyboardInterrupt:
            print("Session ended by participant.")
        finally:
            cap.release()
            cv2.destroyAllWindows()