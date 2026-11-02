"""
Step 3 - Recognise the face in a single image.

Usage:
    python predict.py path/to/photo.jpg
"""
import argparse
import json

import cv2
import numpy as np
import tensorflow as tf

from face_utils import FaceDetector, face_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="models/face_model.keras")
    ap.add_argument("--classes", default="models/class_names.json")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="below this confidence the face is reported as 'Unknown'")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    with open(args.classes) as f:
        class_names = json.load(f)
    detector = FaceDetector()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    crop, found = face_crop(img, detector, args.size)
    if not found:
        print("No face detected - using the center of the image.")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32")
    probs = model.predict(rgb[None, ...], verbose=0)[0]

    for name, p in sorted(zip(class_names, probs), key=lambda t: -t[1]):
        print(f"  {name:<12} {p * 100:6.2f}%")
    best = int(np.argmax(probs))
    label = class_names[best] if probs[best] >= args.threshold else "Unknown"
    print(f"\nPrediction: {label} ({probs[best] * 100:.1f}%)")


if __name__ == "__main__":
    main()
