"""
storage.py
-----------
Single source of truth for the unified dataset's schema. Every trial
captured in real time is appended here so unified_dataset_clean.csv
stays consistent with the public/baseline rows already in it.

Columns (matching your existing file exactly):
  user_id, emotion, trial_id, gaze_x, gaze_y, attention_time,
  emotion_score, engagement_score, Product Features, Eye direction
"""
import os
import csv
import yaml
from datetime import datetime

with open("config/config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

UNIFIED_PATH = _CFG["paths"]["unified_dataset"]

COLUMNS = [
    "user_id", "emotion", "trial_id", "gaze_x", "gaze_y",
    "attention_time", "emotion_score", "engagement_score",
    "Product Features", "Eye direction",
]


def _ensure_file():
    os.makedirs(os.path.dirname(UNIFIED_PATH), exist_ok=True)
    if not os.path.exists(UNIFIED_PATH):
        with open(UNIFIED_PATH, "w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)


def next_trial_id():
    """Reads the current max trial id in the file and returns the next one."""
    _ensure_file()
    max_id = 0
    with open(UNIFIED_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_id = max(max_id, int(row["trial_id"]))
            except (ValueError, TypeError):
                continue
    return max_id + 1


def append_trial(user_id, emotion, gaze_x, gaze_y, attention_time,
                  emotion_score, product_order, aoi_label):
    """
    Appends one real-time captured trial as a new row.
    engagement_score is computed the same way as in the existing
    dataset: emotion_score + attention_time.
    """
    _ensure_file()
    trial_id = next_trial_id()
    engagement_score = round(float(emotion_score) + float(attention_time), 6)

    row = {
        "user_id": user_id,
        "emotion": emotion,
        "trial_id": trial_id,
        "gaze_x": round(gaze_x, 2) if gaze_x is not None else "",
        "gaze_y": round(gaze_y, 2) if gaze_y is not None else "",
        "attention_time": round(attention_time, 6),
        "emotion_score": emotion_score,
        "engagement_score": engagement_score,
        "Product Features": product_order,
        "Eye direction": aoi_label if aoi_label else "NA",
    }

    with open(UNIFIED_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)

    return row


def log_session_meta(user_id, note=""):
    """Optional: append a line to a session log for auditing captures."""
    os.makedirs("data/processed_sessions", exist_ok=True)
    log_path = "data/processed_sessions/session_log.csv"
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "user_id", "note"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), user_id, note])