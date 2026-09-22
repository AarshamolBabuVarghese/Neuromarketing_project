"""
face_detection.py
------------------
Fast face bounding-box detector used to crop frames before they are
handed to DeepFace (emotion) and MediaPipe FaceMesh (gaze).

Using a Haar cascade here (bundled with opencv-python) is deliberate:
it's cheap, so it can run every frame without slowing down the
capture loop. DeepFace / MediaPipe do their own, more accurate,
face localization internally anyway - this module just avoids wasting
time running heavy models on frames with no face at all.

FIX (this version): Haar cascades are notoriously sensitive to
lighting and face size. Two changes to reduce false "no face
detected" misses:
  1. Histogram equalization on the grayscale frame before detection --
     evens out under/over-exposed lighting, which was likely
     contributing to the skipped trials.
  2. Smaller minSize + slightly relaxed minNeighbors -- accepts
     smaller/farther faces and is a bit less strict about confirming
     detections, trading a small amount of false-positive risk for
     far fewer missed real faces (the participant is the only person
     in frame, so stray false positives are low-risk here).
---------------------------------------------------------------
"""
import cv2

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def detect_face(frame_bgr):
    """
    Returns the largest detected face as (x, y, w, h), or None if no
    face is found in the frame.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # normalize lighting/contrast

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    # pick the largest face (closest to camera / most likely the participant)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return tuple(faces[0])  # (x, y, w, h)


def crop_face(frame_bgr, box, margin=0.2):
    """Crop the frame to the face box with a small margin, for DeepFace."""
    if box is None:
        return frame_bgr
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(frame_bgr.shape[1], x + w + mx), min(frame_bgr.shape[0], y + h + my)
    return frame_bgr[y0:y1, x0:x1]


if __name__ == "__main__":
    # Quick manual test: python -m src.face_detection
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if ok:
        box = detect_face(frame)
        print("Face box:", box)
    else:
        print("Could not read from webcam.")