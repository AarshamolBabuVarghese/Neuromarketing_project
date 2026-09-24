"""
webapp_backend/live_capture.py
---------------------------------------------------------------
Browser-based version of the data-collection session normally run
via `python src/experiment_controller.py`. Reuses your existing
face_detection.py, eye_tracking.py, emotion_detection.py,
calibration.py and storage.py from src/ AS-IS (imported, never
edited) -- only the frame source changes: the browser's webcam via
getUserMedia + canvas, posted here as JPEG frames, instead of
cv2.VideoCapture reading a camera directly. Results are written into
unified_dataset_clean.csv via the same storage.append_trial() call
experiment_controller.py already uses, so the output is identical in
shape and downstream scripts (rec.py, dashboard.py) don't need to
know the data came from the browser.

Must be run with PROJECT_ROOT as the working directory (same
requirement your existing scripts already have, since config.yaml
and data/ paths are relative).
---------------------------------------------------------------
"""
import base64
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from functools import wraps

import cv2
import jwt
import numpy as np
import yaml
from flask import Blueprint, request, jsonify, send_from_directory

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# These are your existing, unmodified modules from src/ — imported,
# never rewritten. Only this file's frame source differs from theirs.
from face_detection import detect_face, crop_face
from emotion_detection import analyze_emotion_verbose
from eye_tracking import estimate_gaze, gaze_to_aoi, raw_gaze_ratio
from calibration import load_calibration
import storage as project_storage

import mediapipe as mp

with open(os.path.join(PROJECT_ROOT, "config", "config.yaml"), "r") as f:
    _CFG = yaml.safe_load(f)

_TRIAL_SEC = _CFG["capture"]["trial_duration_sec"]
_SAMPLE_SEC = _CFG["capture"]["sample_interval_sec"]
_SCREEN_W = _CFG["screen"]["width"]
_SCREEN_H = _CFG["screen"]["height"]
_SETTLE_SEC = 0.5
_AWAY_ALERT_SEC = 0.6

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "stimulus_manifest.csv")
STIMULI_DIR = os.path.join(PROJECT_ROOT, "stimuli", "ab_composites")

# 5-point calibration, identical positions to calibration.py's _POINTS
_CALIB_POINTS = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]
_CALIB_SAMPLE_SECONDS = 1.5

SECRET_KEY = os.environ.get("WEBAPP_SECRET_KEY", "dev-secret-change-me")

live_bp = Blueprint("live", __name__)

_mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)


def _count_faces(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _mp_face_detection.process(rgb)
    return len(result.detections) if result.detections else 0


def token_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Session expired, please log in again"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return wrapper


def _decode_frame(data_url):
    """Turns a 'data:image/jpeg;base64,....' string into a BGR numpy
    frame, with the same mirror flip capture_pipeline.py applies."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode frame")
    return cv2.flip(frame, 1)


# ------------------------------------------------------------------
# Config / manifest / stimulus images
# ------------------------------------------------------------------
@live_bp.route("/api/live/config", methods=["GET"])
@token_required
def live_config():
    return jsonify({
        "trial_seconds": _TRIAL_SEC,
        "sample_interval_sec": _SAMPLE_SEC,
        "screen_width": _SCREEN_W,
        "screen_height": _SCREEN_H,
        "aoi": _CFG["screen"]["aoi"],
        "calibration_points": _CALIB_POINTS,
        "calibration_sample_seconds": _CALIB_SAMPLE_SECONDS,
    })


@live_bp.route("/api/live/manifest", methods=["GET"])
@token_required
def live_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return jsonify({"error": f"Manifest not found at {MANIFEST_PATH}"}), 404
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    categories = sorted({r["category"] for r in rows if r.get("category")})
    return jsonify({"rows": rows, "categories": categories})


@live_bp.route("/api/live/stimulus-image/<path:filename>", methods=["GET"])
def stimulus_image(filename):
    # Not behind token_required: <img> tags can't send Authorization
    # headers, and these are just local stimulus images, not data.
    return send_from_directory(STIMULI_DIR, filename)


# ------------------------------------------------------------------
# Calibration -- mirrors calibration.py's run_calibration(), but
# driven frame-by-frame from the browser instead of cv2.VideoCapture.
# ------------------------------------------------------------------
CALIB_STATE = {}  # user_id -> {point_index: [(ratio_x, ratio_y), ...]}


@live_bp.route("/api/live/calibration/start", methods=["POST"])
@token_required
def calibration_start():
    body = request.get_json(force=True) or {}
    user_id = str(body.get("user_id"))
    CALIB_STATE[user_id] = {i: [] for i in range(len(_CALIB_POINTS))}
    return jsonify({"points": _CALIB_POINTS, "screen_width": _SCREEN_W, "screen_height": _SCREEN_H})


@live_bp.route("/api/live/calibration/sample", methods=["POST"])
@token_required
def calibration_sample():
    body = request.get_json(force=True) or {}
    user_id = str(body.get("user_id"))
    point_index = int(body.get("point_index"))
    frame = _decode_frame(body.get("frame"))

    ratio_x, ratio_y = raw_gaze_ratio(frame)
    if ratio_x is not None:
        CALIB_STATE.setdefault(user_id, {}).setdefault(point_index, []).append((ratio_x, ratio_y))
        return jsonify({"detected": True, "samples_so_far": len(CALIB_STATE[user_id][point_index])})
    return jsonify({"detected": False})


@live_bp.route("/api/live/calibration/finish", methods=["POST"])
@token_required
def calibration_finish():
    body = request.get_json(force=True) or {}
    user_id = str(body.get("user_id"))
    state = CALIB_STATE.get(user_id)
    if not state:
        return jsonify({"error": "No calibration session in progress for this user"}), 400

    screen_pts, ratio_pts = [], []
    for i, (fx, fy) in enumerate(_CALIB_POINTS):
        samples = state.get(i, [])
        if not samples:
            continue
        avg_ratio = tuple(np.mean(samples, axis=0))
        ratio_pts.append(avg_ratio)
        screen_pts.append((int(fx * _SCREEN_W), int(fy * _SCREEN_H)))

    if len(ratio_pts) < 3:
        return jsonify({"error": "Not enough calibration points captured — check lighting/webcam and retry."}), 400

    R = np.array([[rx, ry, 1.0] for rx, ry in ratio_pts])
    Sx = np.array([sx for sx, sy in screen_pts])
    Sy = np.array([sy for sx, sy in screen_pts])
    coef_x, *_ = np.linalg.lstsq(R, Sx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(R, Sy, rcond=None)

    calibration = {"coef_x": coef_x.tolist(), "coef_y": coef_y.tolist()}

    calib_dir = os.path.join(PROJECT_ROOT, "data", "processed_sessions")
    os.makedirs(calib_dir, exist_ok=True)
    calib_path = os.path.join(calib_dir, f"calibration_{user_id}.json")
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)

    CALIB_STATE.pop(user_id, None)
    return jsonify({"status": "saved", "path": calib_path, "points_used": len(ratio_pts)})


# ------------------------------------------------------------------
# Trial capture -- mirrors capture_pipeline.py's run_trial(), driven
# frame-by-frame from the browser instead of a tight cv2 while loop.
# One active trial at a time, same as the original one-session design.
# ------------------------------------------------------------------
TRIAL_STATE = {}


@live_bp.route("/api/live/trial/start", methods=["POST"])
@token_required
def trial_start():
    global TRIAL_STATE
    body = request.get_json(force=True) or {}
    user_id = str(body.get("user_id"))
    product_feature_label = body.get("product_feature_label")

    TRIAL_STATE = {
        "user_id": user_id,
        "product_feature_label": product_feature_label,
        "calibration": load_calibration(user_id),
        "start_time": time.time(),
        "emotion_confidence_totals": defaultdict(float),
        "emotion_scores": [],
        "gaze_points": [],
        "aois_seen": [],
        "face_detected_time": 0.0,
        "looked_away_sec": 0.0,
        "multi_face_detected": False,
        "away_since": None,
    }
    return jsonify({
        "status": "started",
        "calibrated": TRIAL_STATE["calibration"] is not None,
        "trial_seconds": _TRIAL_SEC,
    })


@live_bp.route("/api/live/trial/frame", methods=["POST"])
@token_required
def trial_frame():
    if not TRIAL_STATE:
        return jsonify({"error": "No trial in progress"}), 400

    body = request.get_json(force=True) or {}
    frame = _decode_frame(body.get("frame"))
    now = time.time() - TRIAL_STATE["start_time"]

    box = detect_face(frame)
    alert = None

    if box is None:
        if TRIAL_STATE["away_since"] is None:
            TRIAL_STATE["away_since"] = now
        if now - TRIAL_STATE["away_since"] >= _AWAY_ALERT_SEC:
            alert = "LOOK AT THE SCREEN"
    else:
        if TRIAL_STATE["away_since"] is not None:
            TRIAL_STATE["looked_away_sec"] += now - TRIAL_STATE["away_since"]
            TRIAL_STATE["away_since"] = None

    n_faces = _count_faces(frame)
    if n_faces > 1:
        TRIAL_STATE["multi_face_detected"] = True
        alert = "ONLY ONE PERSON SHOULD BE IN FRAME"

    if box is not None:
        TRIAL_STATE["face_detected_time"] += _SAMPLE_SEC
        if now >= _SETTLE_SEC:
            crop = crop_face(frame, box)
            emotion, score, breakdown = analyze_emotion_verbose(crop)
            if emotion and breakdown:
                for label, confidence in breakdown.items():
                    TRIAL_STATE["emotion_confidence_totals"][label] += confidence
                TRIAL_STATE["emotion_scores"].append(score)

    gx, gy = estimate_gaze(frame, calibration=TRIAL_STATE["calibration"])
    if gx is not None:
        TRIAL_STATE["gaze_points"].append((gx, gy))
        aoi = gaze_to_aoi(gx, gy)
        if aoi:
            TRIAL_STATE["aois_seen"].append(aoi)

    return jsonify({"alert": alert, "elapsed": round(now, 2), "faces_in_frame": n_faces})


@live_bp.route("/api/live/trial/finish", methods=["POST"])
@token_required
def trial_finish():
    global TRIAL_STATE
    if not TRIAL_STATE:
        return jsonify({"error": "No trial in progress"}), 400

    state = TRIAL_STATE
    if state["away_since"] is not None:
        state["looked_away_sec"] += (time.time() - state["start_time"]) - state["away_since"]

    if not state["emotion_confidence_totals"]:
        TRIAL_STATE = {}
        return jsonify({"error": "No face detected during this trial — check lighting/camera position, then retry."}), 400

    dominant_emotion = max(state["emotion_confidence_totals"], key=state["emotion_confidence_totals"].get)
    avg_emotion_score = sum(state["emotion_scores"]) / len(state["emotion_scores"])

    if state["gaze_points"]:
        avg_gx = float(np.mean([p[0] for p in state["gaze_points"]]))
        avg_gy = float(np.mean([p[1] for p in state["gaze_points"]]))
    else:
        avg_gx = avg_gy = None

    dominant_aoi = Counter(state["aois_seen"]).most_common(1)[0][0] if state["aois_seen"] else None
    aoi_variant = dominant_aoi.split(" ")[0] if dominant_aoi else "NA"

    saved_row = project_storage.append_trial(
        user_id=state["user_id"],
        emotion=dominant_emotion,
        gaze_x=avg_gx,
        gaze_y=avg_gy,
        attention_time=round(state["face_detected_time"], 6),
        emotion_score=avg_emotion_score,
        product_order=state["product_feature_label"],
        aoi_label=aoi_variant,
    )

    result = {
        "saved_row": saved_row,
        "multi_face_detected": state["multi_face_detected"],
        "looked_away_sec": round(state["looked_away_sec"], 2),
    }
    TRIAL_STATE = {}
    return jsonify(result)


@live_bp.route("/api/live/session/log", methods=["POST"])
@token_required
def session_log():
    body = request.get_json(force=True) or {}
    project_storage.log_session_meta(str(body.get("user_id")), note=body.get("note", ""))
    return jsonify({"status": "logged"})
