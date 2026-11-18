"""
Laptop-webcam demo of the BOARD model (iter5, int8 TFLite) - same model and Unknown rule as the Nano.

Preprocessing matches training: face square crop -> grayscale -> 224 -> central 75 % -> 96x96 -> int8.
Text on screen + in the terminal, and the name is spoken (Windows speech) when the label changes.

Keys:  q quit   s sound on/off   b board mode (centre crop, no face detector - like the Nano)

Usage:
    python webcam_board.py
    python webcam_board.py --camera 1 --threshold 0.55
"""
import argparse
import json
import os
import subprocess
from collections import deque

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import cv2
import numpy as np
import tensorflow as tf

from face_utils import FaceDetector, square_crop


def speak(text):
    cmd = ("Add-Type -AssemblyName System.Speech; "
           f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')")
    subprocess.Popen(["powershell", "-NoProfile", "-Command", cmd],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Int8Model:
    def __init__(self, path):
        self.it = tf.lite.Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def __call__(self, img96):  # float32 96x96x1, 0-255
        s, zp = self.inp["quantization"]
        x = np.clip(np.round(img96 / s + zp), -128, 127).astype(np.int8)[None]
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        o = self.it.get_tensor(self.out["index"])[0].astype(np.float32)
        s, zp = self.out["quantization"]
        return (o - zp) * s


def to96(bgr_square):
    g = cv2.cvtColor(bgr_square, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA)
    c = int(224 * (1 - 0.75) / 2)
    g = g[c:224 - c, c:224 - c]
    return cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None]


def label(probs, names, th):
    i = int(probs.argmax())
    name = names[i] if (probs[i] >= th or names[i] == "NoPerson") else "Unknown"
    return name, float(probs[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/iter5/tiny_face_model_int8.tflite")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    model = Int8Model(args.model)
    names = json.load(open(os.path.join(os.path.dirname(args.model), "class_names.json")))
    detector = FaceDetector()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")
    print("Classes:", names, "| threshold", args.threshold)
    print("Keys: q quit, s sound on/off, b board mode")

    sound, board = True, False
    recent, announced = deque(maxlen=6), None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        if board:  # Nano-style: square centre crop, no detector
            side = min(H, W); x0, y0 = (W - side) // 2, (H - side) // 2
            name, conf = label(model(to96(frame[y0:y0 + side, x0:x0 + side])), names, args.threshold)
            box = (x0, y0, side, side)
        else:
            faces = detector.detect(frame)
            if len(faces) == 0:
                name, conf, box = "NoPerson", 1.0, None
            else:
                x, y, w, h, _ = max(faces, key=lambda f: f[2] * f[3])  # biggest face
                crop = square_crop(frame, x, y, w, h)
                name, conf = label(model(to96(crop)), names, args.threshold)
                box = (x, y, w, h)

        recent.append(name)
        stable = name if recent.count(name) == len(recent) == recent.maxlen else None
        if stable and stable != announced:
            announced = stable
            print(f"Detected: {stable} ({conf * 100:.0f}%)")
            if sound and stable != "NoPerson":
                speak(stable if stable != "Unknown" else "Unknown person")

        color = {"Unknown": (0, 0, 255), "NoPerson": (160, 160, 160)}.get(name, (0, 200, 0))
        if box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} {conf * 100:.0f}%", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"iter5 int8 | {'BOARD crop' if board else 'face detector'} | "
                           f"sound {'on' if sound else 'off'} | q s b", (15, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("Board model (iter5 int8) - press q to quit", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            sound = not sound
        if k == ord("b"):
            board = not board
            recent.clear(); announced = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
