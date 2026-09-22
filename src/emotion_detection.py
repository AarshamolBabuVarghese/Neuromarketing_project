"""
emotion_detection.py
---------------------
Wraps DeepFace to turn a face crop into:
  - dominant emotion label (happy/sad/angry/neutral/surprise/fear/disgust)
  - a numeric emotion_score on the same 0-1 scale already used in
    unified_dataset_clean.csv (see config.yaml -> emotion.score_map)

DEBUG ADDITION: analyze_emotion_verbose() also returns the full
per-emotion confidence breakdown (not just the winning label), so
you can see WHY it picked what it picked -- e.g. if 'happy' wins
with only 28% confidence while 'neutral' is at 26%, that's a
near-coin-flip the single-frame approach is prone to, not a real
misread.
"""
import yaml
from deepface import DeepFace

with open("config/config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

_SCORE_MAP = _CFG["emotion"]["score_map"]
_BACKEND = _CFG["emotion"]["detector_backend"]


def analyze_emotion(face_crop_bgr):
    """
    Runs DeepFace emotion analysis on a face crop.
    Returns (emotion_label:str, emotion_score:float) or (None, None)
    if no face could be analyzed.
    """
    dominant, score, _ = analyze_emotion_verbose(face_crop_bgr)
    return dominant, score


def analyze_emotion_verbose(face_crop_bgr):
    """
    Same as analyze_emotion, but also returns the full per-emotion
    confidence breakdown (a dict like {'happy': 62.3, 'neutral': 21.1,
    'sad': 8.4, ...}) so you can see how confident/ambiguous the call
    actually was, instead of only the winning label.

    Returns (emotion_label, emotion_score, emotion_breakdown) or
    (None, None, None) if no face could be analyzed.
    """
    try:
        result = DeepFace.analyze(
            face_crop_bgr,
            actions=["emotion"],
            detector_backend=_BACKEND,
            enforce_detection=False,
            silent=True,
        )
        # DeepFace returns a list when given a batch; normalize to dict
        if isinstance(result, list):
            result = result[0]

        dominant = result["dominant_emotion"].lower()
        breakdown = {k.lower(): round(v, 1) for k, v in result["emotion"].items()}
        score = _SCORE_MAP.get(dominant, 0.5)  # default to neutral if unmapped
        return dominant, score, breakdown
    except Exception as e:
        print(f"[emotion_detection] failed to analyze frame: {e}")
        return None, None, None


if __name__ == "__main__":
    import cv2
    from face_detection import detect_face, crop_face

    cap = cv2.VideoCapture(0)
    print("Capturing 10 frames, 0.3s apart — make different expressions "
          "(smile, neutral, frown) and watch how the breakdown shifts.\n")
    import time
    for i in range(10):
        ok, frame = cap.read()
        if not ok:
            continue
        box = detect_face(frame)
        crop = crop_face(frame, box)
        dominant, score, breakdown = analyze_emotion_verbose(crop)
        print(f"[{i+1}] dominant={dominant}  score={score}  breakdown={breakdown}")
        time.sleep(0.3)
    cap.release()