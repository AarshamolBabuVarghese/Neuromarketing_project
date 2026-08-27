"""
emotion_detection.py
---------------------
Wraps DeepFace to turn a face crop into:
  - dominant emotion label (happy/sad/angry/neutral/surprise/fear/disgust)
  - a numeric emotion_score on the same 0-1 scale already used in
    unified_dataset_clean.csv (see config.yaml -> emotion.score_map)
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
        score = _SCORE_MAP.get(dominant, 0.5)  # default to neutral if unmapped
        return dominant, score
    except Exception as e:
        print(f"[emotion_detection] failed to analyze frame: {e}")
        return None, None


if __name__ == "__main__":
    import cv2
    from face_detection import detect_face, crop_face

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if ok:
        box = detect_face(frame)
        crop = crop_face(frame, box)
        print(analyze_emotion(crop))