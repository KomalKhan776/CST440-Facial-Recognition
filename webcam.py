"""
Step 4 - Live face recognition from your webcam.

Usage:
    python webcam.py            (press  q  to quit)
    python webcam.py --camera 1 (if you have more than one camera)
"""
import argparse
import json

import cv2
import numpy as np
import tensorflow as tf

from face_utils import FaceDetector, square_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/face_model.keras")
    ap.add_argument("--classes", default="models/class_names.json")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    with open(args.classes) as f:
        class_names = json.load(f)
    detector = FaceDetector()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for (x, y, w, h, _) in detector.detect(frame):
            crop = square_crop(frame, x, y, w, h)
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32")
            probs = model.predict(rgb[None, ...], verbose=0)[0]
            best = int(np.argmax(probs))
            conf = probs[best]
            name = class_names[best] if conf >= args.threshold else "Unknown"
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} {conf * 100:.0f}%", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
