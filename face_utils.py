"""
Shared helpers: face detection (OpenCV YuNet) and cropping.

YuNet is a small, accurate deep-learning face detector that ships with
OpenCV (cv2.FaceDetectorYN). Its model file lives in
models/face_detection_yunet_2023mar.onnx.
"""
import os

import cv2

DETECTOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models", "face_detection_yunet_2023mar.onnx")


class FaceDetector:
    def __init__(self, model_path=DETECTOR_PATH, score_threshold=0.7):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face detector model not found: {model_path}")
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320),
                                             score_threshold, 0.3, 5000)

    def detect(self, img):
        """Return a list of faces as (x, y, w, h, score), biggest first."""
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        if faces is None:
            return []
        out = [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[14])) for f in faces]
        return sorted(out, key=lambda f: f[2] * f[3], reverse=True)


def square_crop(img, x, y, w, h, margin=0.3):
    """Crop a square around the face box (x, y, w, h) with a margin."""
    H, W = img.shape[:2]
    cx, cy = x + w / 2, y + h / 2
    side = min(max(w, h) * (1 + margin), W, H)
    x1 = int(max(0, min(W - side, cx - side / 2)))
    y1 = int(max(0, min(H - side, cy - side / 2)))
    s = int(side)
    return img[y1:y1 + s, x1:x1 + s]


def center_crop(img):
    H, W = img.shape[:2]
    s = min(H, W)
    return img[(H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]


def face_crop(img, detector, size=224):
    """Detect the biggest face, crop it and resize to size x size.
    Returns (crop, found_face: bool)."""
    faces = detector.detect(img)
    if faces:
        x, y, w, h, _ = faces[0]
        crop, found = square_crop(img, x, y, w, h), True
    else:
        crop, found = center_crop(img), False
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), found
